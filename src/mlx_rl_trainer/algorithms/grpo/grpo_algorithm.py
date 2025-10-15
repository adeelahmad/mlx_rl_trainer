"""
GRPO Algorithm with Dual Gradient + SFT Layer Control

FEATURES:
1. Dual gradient computation (thinking vs answer paths)
2. SFT loss computation with layer-specific control
3. Three configurable SFT modes:
   - 'all': Apply SFT to all layers (backward compatible)
   - 'answer_only': SFT only on answer layers
   - 'weighted': Different weights for thinking vs answer layers
   - 'exclude_thinking': No SFT on thinking layers (DEFAULT for System 1/2)

CONFIGURATION:
trainer:
  # Dual gradients for System 1/2
  use_dual_gradients: true
  thinking_layer_start: 22
  thinking_layer_end: 30
  answer_layer_start: 31
  answer_layer_end: 36
  answer_gradient_weight: 2.0

  # SFT configuration
  use_sft_on_answer: true
  sft_mode: 'exclude_thinking'  # Recommended for System 1/2
  sft_weight: 0.1
  sft_thinking_weight: 0.0  # For 'weighted' mode
  sft_answer_weight: 1.0    # For 'weighted' mode

DEFAULT BEHAVIOR (non-breaking):
- If sft_mode not specified: 'all' (backward compatible)
- If layer boundaries not specified: Applies to all layers
- 'exclude_thinking' prevents thinking layers from being constrained by SFT
"""
import logging
from typing import Dict, Any, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """Extract layer number from parameter path."""
    import re

    match = re.search(r"\.layers\.(\d+)\.", param_path)
    return int(match.group(1)) if match else None


def _mask_gradients_by_layers(
    grads: Dict,
    config: ExperimentConfig,
    sft_mode: str,
) -> Dict:
    """
    Mask SFT gradients based on layer configuration.

    Modes:
    - 'all': No masking (apply SFT to all layers)
    - 'answer_only': Only answer layers get SFT gradients
    - 'weighted': Different weights for thinking vs answer layers
    - 'exclude_thinking': Zero out thinking layer gradients (DEFAULT)
    """
    # Get layer boundaries
    thinking_start = getattr(config.trainer, "thinking_layer_start", None)
    thinking_end = getattr(config.trainer, "thinking_layer_end", None)
    answer_start = getattr(config.trainer, "answer_layer_start", None)
    answer_end = getattr(config.trainer, "answer_layer_end", None)

    # If layer boundaries not specified or mode is 'all', return gradients as-is
    if sft_mode == "all" or thinking_start is None or answer_start is None:
        return grads

    # Get weights for weighted mode
    thinking_weight = getattr(config.trainer, "sft_thinking_weight", 0.0)
    answer_weight = getattr(config.trainer, "sft_answer_weight", 1.0)

    # Process gradients
    masked_grads = {}
    for key, grad in tree_flatten(grads):
        layer_num = _extract_layer_number(key)

        if layer_num is None:
            # Non-layer parameters (embeddings, lm_head, etc.)
            masked_grads[key] = grad
        else:
            # Layer-specific parameters
            if sft_mode == "answer_only":
                if answer_start <= layer_num <= answer_end:
                    masked_grads[key] = grad
                else:
                    masked_grads[key] = mx.zeros_like(grad)

            elif sft_mode == "weighted":
                if thinking_start <= layer_num <= thinking_end:
                    masked_grads[key] = grad * thinking_weight
                elif answer_start <= layer_num <= answer_end:
                    masked_grads[key] = grad * answer_weight
                else:
                    masked_grads[key] = grad

            elif sft_mode == "exclude_thinking":
                if thinking_start <= layer_num <= thinking_end:
                    masked_grads[key] = mx.zeros_like(grad)
                else:
                    masked_grads[key] = grad

            else:
                masked_grads[key] = grad

    return masked_grads


class GRPOAlgorithm:
    def __init__(self, config, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta

    def compute_advantages(self, rewards_flat, samples_per_prompt):
        """Compute advantages with optional baseline normalization."""
        if samples_per_prompt <= 1:
            return (rewards_flat - mx.mean(rewards_flat)) / (
                mx.std(rewards_flat) + 1e-8
            )

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
            tokens_key = "tokens"
            mask_key = "response_mask"

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[mask_key].shape[1]
            shifted_logits = logits[:, offset - 1 : -1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            log_ratio = gathered_log_probs - A["ref_log_probs"]
            kl_term = mx.exp(log_ratio) - 1 - log_ratio
            kl_penalty = kl_term * A[mask_key]

            advantages_expanded = A["advantages"][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * A[mask_key]

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty
            loss = mx.sum(total_loss_per_token) / mx.sum(A[mask_key])

            kl_div = mx.sum(kl_penalty) / mx.sum(A[mask_key])
            policy_loss = mx.sum(policy_loss_term) / mx.sum(A[mask_key])

            return loss, {"kl_divergence": kl_div, "policy_loss": policy_loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return (
                mx.array(zero_val),
                {},
                {"kl_divergence": zero_val, "policy_loss": zero_val},
            )

    def calculate_dual_gradient_loss(self, rollout_batch, full_config, pad_token_id):
        """
        Compute separate gradients for thinking and answer tokens.

        This creates two gradient pathways:
        - Thinking gradients: From thinking tokens only
        - Answer gradients: From answer tokens only

        These can be applied with different weights and to different layers
        to create fast vs deep reasoning modes (System 1 vs System 2).
        """
        A = rollout_batch

        # Check if we have the necessary masks
        has_thinking_mask = "thinking_mask" in A
        has_answer_mask = "answer_mask" in A

        # Fallback to original method if masks not present
        if not has_thinking_mask or not has_answer_mask:
            logger.warning(
                "Thinking/answer masks not found in batch. Falling back to standard gradient computation."
            )
            loss, grads, metrics = self.calculate_loss_and_grads(
                A, full_config, pad_token_id
            )
            return loss, grads, loss, grads, metrics

        def compute_loss_with_mask_and_metrics(
            actor_model, mask_type, compute_metrics=False
        ):
            """Compute loss using only specific tokens (thinking or answer)."""
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            shifted_logits = logits[:, offset - 1 : -1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            log_ratio = gathered_log_probs - A["ref_log_probs"]
            kl_term = mx.exp(log_ratio) - 1 - log_ratio

            # Apply the thinking or answer mask
            token_mask = A[mask_type]
            combined_mask = A[response_mask_key] * token_mask

            kl_penalty = kl_term * combined_mask
            advantages_expanded = A["advantages"][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * combined_mask

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty

            # Normalize by actual number of masked tokens
            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss_per_token) / (mask_sum + 1e-8)

            if compute_metrics:
                kl_div = mx.sum(kl_penalty) / (mask_sum + 1e-8)
                policy_loss = mx.sum(policy_loss_term) / (mask_sum + 1e-8)
                return loss, {"kl_divergence": kl_div, "policy_loss": policy_loss}

            return loss

        # Compute thinking-only gradients
        thinking_loss_fn = lambda model: compute_loss_with_mask_and_metrics(
            model, "thinking_mask", False
        )
        thinking_grads_fn = nn.value_and_grad(self.actor, thinking_loss_fn)
        thinking_loss, thinking_grads = thinking_grads_fn(self.actor)

        # Compute answer-only gradients AND metrics in one pass
        answer_loss_fn = lambda model: compute_loss_with_mask_and_metrics(
            model, "answer_mask", True
        )
        answer_grads_fn = nn.value_and_grad(self.actor, answer_loss_fn)
        (answer_loss, metrics), answer_grads = answer_grads_fn(self.actor)

        # Convert metrics to dict of floats
        metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

        return thinking_loss, thinking_grads, answer_loss, answer_grads, metrics_dict

    def calculate_sft_loss_and_grads(
        self, rollout_batch, reference_tokens, full_config, pad_token_id
    ):
        """
        Compute SFT (supervised fine-tuning) loss and gradients with layer-specific control.

        SFT provides supervised signal to keep answers aligned with reference completions
        while RL optimizes for rewards.

        NEW: Layer-specific SFT control for System 1/2 architecture:
        - Mode 'exclude_thinking': No SFT on thinking layers (DEFAULT)
        - Mode 'answer_only': SFT only on answer layers
        - Mode 'weighted': Different weights for thinking vs answer layers
        - Mode 'all': Apply to all layers (backward compatible)

        Args:
            rollout_batch: Batch containing tokens and masks
            reference_tokens: Ground truth RESPONSE tokens (not including prompt) [batch, response_len]
            full_config: Experiment configuration
            pad_token_id: ID of padding token

        Returns:
            loss: SFT loss value
            grads: Gradients from SFT loss (layer-masked if configured)
            metrics_dict: Dictionary with loss breakdown
        """
        A = rollout_batch

        # Get SFT mode
        sft_mode = getattr(full_config.trainer, "sft_mode", "all")

        # Log SFT mode on first call
        if not hasattr(self, "_sft_mode_logged"):
            logger.info(f"SFT layer control mode: {sft_mode}")
            if sft_mode == "exclude_thinking":
                logger.info(
                    "System 2 (thinking) layers will NOT receive SFT gradients - only RL signal"
                )
            elif sft_mode == "answer_only":
                logger.info("Only System 1 (answer) layers will receive SFT gradients")
            elif sft_mode == "weighted":
                thinking_w = getattr(full_config.trainer, "sft_thinking_weight", 0.0)
                answer_w = getattr(full_config.trainer, "sft_answer_weight", 1.0)
                logger.info(f"Weighted SFT: thinking={thinking_w}, answer={answer_w}")
            self._sft_mode_logged = True

        # Check if we have answer mask
        if "answer_mask" not in A:
            logger.warning(
                "Answer mask not found for SFT. Falling back to response_mask."
            )
            answer_mask = A.get(
                "response_mask", mx.ones_like(reference_tokens, dtype=mx.float32)
            )
        else:
            answer_mask = A["answer_mask"]

        def compute_sft_loss(actor_model):
            """Compute cross-entropy loss on answer tokens only."""
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            # Get logits for full sequence
            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            # Extract response portion logits
            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]

            # Shift logits for next-token prediction: logits[t] predicts token[t+1]
            response_logits = logits[
                :, offset - 1 : -1, :
            ]  # [batch, response_len, vocab]

            # Reference tokens are already response-only, shift for next-token prediction
            target_tokens = reference_tokens[:, 1:]  # [batch, response_len-1]

            # Answer mask also needs to exclude first token to align with targets
            current_answer_mask = (
                answer_mask[:, 1:] if answer_mask.shape[1] > 1 else answer_mask
            )

            # Verify shapes match and truncate if needed
            min_len = min(
                response_logits.shape[1],
                target_tokens.shape[1],
                current_answer_mask.shape[1],
            )

            if (
                response_logits.shape[1] != target_tokens.shape[1]
                or response_logits.shape[1] != current_answer_mask.shape[1]
            ):
                logger.debug(
                    f"Aligning SFT shapes: logits {response_logits.shape[1]} vs targets {target_tokens.shape[1]} vs mask {current_answer_mask.shape[1]} -> {min_len}"
                )
                response_logits = response_logits[:, :min_len, :]
                target_tokens = target_tokens[:, :min_len]
                current_answer_mask = current_answer_mask[:, :min_len]

            # Compute cross-entropy loss
            log_probs = nn.log_softmax(response_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            # Apply answer mask - only compute loss on answer tokens
            masked_log_probs = -gathered_log_probs * current_answer_mask

            # Normalize by number of answer tokens
            mask_sum = mx.sum(current_answer_mask)
            loss = mx.sum(masked_log_probs) / (mask_sum + 1e-8)

            return loss, {"sft_loss": loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_sft_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)

            # Apply layer-specific masking to gradients
            grads = _mask_gradients_by_layers(grads, full_config, sft_mode)

            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"sft_loss": 0.0}
