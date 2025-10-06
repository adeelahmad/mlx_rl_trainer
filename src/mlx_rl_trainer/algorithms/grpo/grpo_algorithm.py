# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_algorithm.py
# revision_no: 002
# goals_of_writing_code_block: Implement the core GRPO loss and advantage calculation logic, fixing parameter passing.
# type_of_code_response: change existing
"""
GRPO (Group Relative Policy Optimization) core loss implementation.
"""
from typing import Dict, Any, Tuple
import mlx.core as mx
from mlx.nn import log_softmax
import logging
from ..base_algorithm import BaseAlgorithm
from mlx_rl_trainer.core.config import ExperimentConfig # For type hinting

logger = logging.getLogger(__name__)

class GRPOAlgorithm(BaseAlgorithm):
    """
    Implements the core GRPO (Group Relative Policy Optimization) algorithm logic,
    including loss calculation and advantage estimation.
    """
    def calculate_loss_and_grads(self, rollout_batch: Dict[str, mx.array], full_config: ExperimentConfig) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        """
        Calculates the GRPO loss and its gradients.

        Args:
            rollout_batch: Contains 'tokens', 'response_mask', 'advantages', 'ref_log_probs'.
            full_config: The complete ExperimentConfig, from which `grpo_beta` is extracted.

        Returns:
            A tuple: (scalar_loss_tensor, gradients_dict, additional_metrics_dict).
        """
        beta = full_config.trainer.grpo_beta # FIX: Get beta from full_config
        pad_token_id = self.config.tokenizer.pad_token_id # Assuming tokenizer is part of config or directly accessible

        def loss_fn(model):
            tokens, mask, advantages, ref_log_probs = (rollout_batch.get(k) for k in ["tokens", "response_mask", "advantages", "ref_log_probs"])

            # Handle empty rollouts
            if tokens is None or tokens.size == 0 or mask is None or mask.size == 0 or advantages is None or advantages.size == 0 or ref_log_probs is None or ref_log_probs.size == 0:
                logger.warning("Empty rollout_batch in loss_fn, returning 0.0 loss.")
                return mx.array(0.0), mx.array(0.0) # Return loss and dummy kl_mean

            logits = model(tokens)
            gen_len = mask.shape[1]

            # Extract logits and targets corresponding to the generated response part
            logits_resp = logits[:, -gen_len-1:-1, :] # Logits up to the token *before* the current one
            targets_resp = tokens[:, -gen_len:] # Actual generated tokens

            # Calculate current policy's log probabilities for generated tokens
            cur_lp = log_softmax(logits_resp.astype(mx.float32), axis=-1)
            cur_lp = mx.take_along_axis(cur_lp, targets_resp[..., None], axis=-1).squeeze(-1)

            # Calculate log ratio and KL approximation
            log_ratio = mx.clip(cur_lp - ref_log_probs, -8.0, 8.0) # Clip for stability
            ratio = mx.exp(log_ratio)
            kl_approx = ratio - 1.0 - log_ratio # Surrogate for KL divergence

            # Policy gradient term and KL penalty term
            pg_term = -log_ratio * advantages[:, None] * mask
            kl_term = beta * kl_approx * mask

            loss_per_token = pg_term + kl_term

            # Sum over tokens and average over non-masked tokens
            sum_mask = mx.sum(mask)
            if sum_mask.item() == 0: # Avoid division by zero
                logger.warning("Zero sum_mask in loss calculation, returning 0.0 loss.")
                return mx.array(0.0), mx.array(0.0)

            loss = mx.sum(loss_per_token) / sum_mask
            kl_mean = mx.sum(kl_approx * mask) / sum_mask

            # Ensure loss and kl_mean are scalar
            return loss, kl_mean

        # Compute value and gradients
        (loss, kl_mean), grads = mx.nn.value_and_grad(self.actor, loss_fn, has_aux=True)(self.actor)

        return loss, grads, {"kl_divergence": float(kl_mean.item())}

    def compute_advantages(self, rewards_flat: mx.array, samples_per_prompt: int) -> mx.array:
        """
        Computes group-relative advantage estimates.

        Args:
            rewards_flat: 1D MLX array of rewards for all generated samples (num_prompts * samples_per_prompt).
            samples_per_prompt: Number of responses generated per prompt.

        Returns:
            1D MLX array of advantage estimates.
        """
        if samples_per_prompt <= 1:
            # If only one sample, use simple baseline subtraction (mean of all rewards)
            return rewards_flat - mx.mean(rewards_flat)

        num_prompts = rewards_flat.shape[0] // samples_per_prompt

        # Ensure rewards_flat length is a multiple of samples_per_prompt
        if rewards_flat.shape[0] % samples_per_prompt != 0:
            logger.warning("Rewards flat length not multiple of samples per prompt. Truncating for advantage calculation.")
            rewards_flat = rewards_flat[:num_prompts * samples_per_prompt]

        rewards_grouped = rewards_flat.reshape(num_prompts, samples_per_prompt)

        # Calculate mean and standard deviation per group (per prompt)
        mean = mx.mean(rewards_grouped, axis=1, keepdims=True)
        std = mx.std(rewards_grouped, axis=1, ddof=0, keepdims=True) + 1e-8 # Add epsilon for numerical stability

        # Standardize rewards within each group to get advantages
        advantages = (rewards_grouped - mean) / std
        return advantages.flatten()
