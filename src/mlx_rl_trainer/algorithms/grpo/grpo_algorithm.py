#-----------#
# <FILE name="./grpo_algorithm.py">
# Complete File with Dual Gradient (Thinking vs Answer) Support
#
# FIXES APPLIED:
# 1. Optimized gradient computation - reduced from 3 forward passes to 2
# 2. Better error handling and fallback to standard training
# 3. Improved metrics collection efficiency
#
# CONFIGURATION REQUIRED (add to your config):
# trainer:
#   use_dual_gradients: true
#   thinking_layer_start: 22
#   thinking_layer_end: 30
#   answer_layer_start: 31  # Starts AFTER thinking to avoid overlap
#   answer_layer_end: 36
#   answer_gradient_weight: 2.0  # 2x emphasis on fast answer path
#
# DEFAULT BEHAVIOR (non-breaking):
# - If masks not present or use_dual_gradients=false: Falls back to standard training
# - If answer_layer_start not specified: Auto-sets to thinking_layer_end + 1
# - This creates separate pathways: thinking (slow) vs answer (fast)
#
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)

class GRPOAlgorithm:
    def __init__(self, config, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta

    def compute_advantages(self, rewards_flat, samples_per_prompt):
        """Compute advantages with optional baseline normalization."""
        if samples_per_prompt <= 1:
            return (rewards_flat - mx.mean(rewards_flat)) / (mx.std(rewards_flat) + 1e-8)

        batch_size = rewards_flat.shape[0] // samples_per_prompt
        rewards_reshaped = rewards_flat.reshape(batch_size, samples_per_prompt)
        baseline = mx.mean(rewards_reshaped, axis=1, keepdims=True)
        std = mx.std(rewards_reshaped, axis=1, keepdims=True)
        advantages = (rewards_reshaped - baseline) / (std + 1e-8)
        return advantages.flatten()

    def calculate_loss_and_grads(self, rollout_batch, full_config, pad_token_id):
        """Original single gradient computation method."""
        A = rollout_batch
        zero_val = 0.0

        def compute_loss(actor_model):
            tokens_key = 'tokens'
            mask_key = 'response_mask'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[mask_key].shape[1]
            shifted_logits = logits[:, offset-1:-1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)

            log_ratio = gathered_log_probs - A['ref_log_probs']
            kl_term = mx.exp(log_ratio) - 1 - log_ratio
            kl_penalty = kl_term * A[mask_key]

            advantages_expanded = A['advantages'][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * A[mask_key]

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty
            loss = mx.sum(total_loss_per_token) / mx.sum(A[mask_key])

            kl_div = mx.sum(kl_penalty) / mx.sum(A[mask_key])
            policy_loss = mx.sum(policy_loss_term) / mx.sum(A[mask_key])

            return loss, {
                'kl_divergence': kl_div,
                'policy_loss': policy_loss
            }

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(zero_val), {}, {'kl_divergence': zero_val, 'policy_loss': zero_val}

    def calculate_dual_gradient_loss(self, rollout_batch, full_config, pad_token_id):
        """
        Compute separate gradients for thinking and answer tokens.

        This creates two gradient pathways:
        - Thinking gradients: From thinking tokens only
        - Answer gradients: From answer tokens only

        These can be applied with different weights and to different layers
        to create fast vs deep reasoning modes.
        """
        A = rollout_batch

        # Check if we have the necessary masks
        has_thinking_mask = 'thinking_mask' in A
        has_answer_mask = 'answer_mask' in A

        # Fallback to original method if masks not present
        if not has_thinking_mask or not has_answer_mask:
            logger.warning("Thinking/answer masks not found in batch. Falling back to standard gradient computation.")
            loss, grads, metrics = self.calculate_loss_and_grads(A, full_config, pad_token_id)
            return loss, grads, loss, grads, metrics

        def compute_loss_with_mask_and_metrics(actor_model, mask_type, compute_metrics=False):
            """Compute loss using only specific tokens (thinking or answer)."""
            tokens_key = 'tokens'
            response_mask_key = 'response_mask'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            shifted_logits = logits[:, offset-1:-1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)

            log_ratio = gathered_log_probs - A['ref_log_probs']
            kl_term = mx.exp(log_ratio) - 1 - log_ratio

            # Apply the thinking or answer mask
            token_mask = A[mask_type]
            combined_mask = A[response_mask_key] * token_mask

            kl_penalty = kl_term * combined_mask
            advantages_expanded = A['advantages'][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * combined_mask

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty

            # Normalize by actual number of masked tokens
            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss_per_token) / (mask_sum + 1e-8)

            if compute_metrics:
                kl_div = mx.sum(kl_penalty) / (mask_sum + 1e-8)
                policy_loss = mx.sum(policy_loss_term) / (mask_sum + 1e-8)
                return loss, {'kl_divergence': kl_div, 'policy_loss': policy_loss}

            return loss

        # Compute thinking-only gradients
        thinking_loss_fn = lambda model: compute_loss_with_mask_and_metrics(model, 'thinking_mask', False)
        thinking_grads_fn = nn.value_and_grad(self.actor, thinking_loss_fn)
        thinking_loss, thinking_grads = thinking_grads_fn(self.actor)

        # Compute answer-only gradients AND metrics in one pass
        answer_loss_fn = lambda model: compute_loss_with_mask_and_metrics(model, 'answer_mask', True)
        answer_grads_fn = nn.value_and_grad(self.actor, answer_loss_fn)
        (answer_loss, metrics), answer_grads = answer_grads_fn(self.actor)

        # Convert metrics to dict of floats
        metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

        return thinking_loss, thinking_grads, answer_loss, answer_grads, metrics_dict

    def calculate_sft_loss_and_grads(self, rollout_batch, reference_tokens, full_config, pad_token_id):
        """
        Compute SFT (supervised fine-tuning) loss and gradients on answer tokens only.

        This provides a supervised signal to keep answers aligned with reference completions
        while RL optimizes for rewards. Used in hybrid training where:
        - Thinking tokens: RL only
        - Answer tokens: RL + SFT

        Args:
            rollout_batch: Batch containing tokens and masks
            reference_tokens: Ground truth tokens for supervised learning [batch, seq_len]
            full_config: Experiment configuration
            pad_token_id: ID of padding token

        Returns:
            loss: SFT loss value
            grads: Gradients from SFT loss
            metrics_dict: Dictionary with loss breakdown
        """
        A = rollout_batch

        # Check if we have answer mask
        if 'answer_mask' not in A:
            logger.warning("Answer mask not found for SFT. Falling back to response_mask.")
            answer_mask = A.get('response_mask', mx.ones_like(A['tokens'][:, :reference_tokens.shape[1]]))
        else:
            answer_mask = A['answer_mask']

        def compute_sft_loss(actor_model):
            """Compute cross-entropy loss on answer tokens only."""
            tokens_key = 'tokens'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            # Get the response portion
            offset = A[tokens_key].shape[1] - answer_mask.shape[1]
            shifted_logits = logits[:, offset-1:-1, :]  # Shift for next-token prediction
            target_tokens = reference_tokens[:, 1:]  # Target is next token

            # Compute cross-entropy loss
            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)

            # Apply answer mask - only compute loss on answer tokens
            masked_log_probs = -gathered_log_probs * answer_mask

            # Normalize by number of answer tokens
            mask_sum = mx.sum(answer_mask)
            loss = mx.sum(masked_log_probs) / (mask_sum + 1e-8)

            return loss, {'sft_loss': loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_sft_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {'sft_loss': 0.0}

# End of File: ./grpo_algorithm.py
#</FILE>
#-----------#
