# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_algorithm.py
# revision_no: 003
# goals_of_writing_code_block: Fix empty rollout_batch issues with better debugging and validation
# type_of_code_response: replace code
"""
GRPO (Group Relative Policy Optimization) core loss implementation.
"""
from typing import Dict, Any, Tuple
import mlx.core as mx
from mlx.nn import log_softmax, value_and_grad
import logging
from ..base_algorithm import BaseAlgorithm
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


class GRPOAlgorithm(BaseAlgorithm):
    """
    Implements the core GRPO (Group Relative Policy Optimization) algorithm logic,
    including loss calculation and advantage estimation.
    """

    def calculate_loss_and_grads(
        self,
        rollout_batch: Dict[str, mx.array],
        full_config: ExperimentConfig,
        pad_token_id: int,
    ) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        """
        Calculates the GRPO loss and its gradients.

        Args:
            rollout_batch: Contains 'tokens', 'response_mask', 'advantages', 'ref_log_probs'.
            full_config: The complete ExperimentConfig, from which `grpo_beta` is extracted.
            pad_token_id: The pad token ID from the tokenizer.

        Returns:
            A tuple: (scalar_loss_tensor, gradients_dict, additional_metrics_dict).
        """
        beta = full_config.trainer.grpo_beta

        # Validate rollout_batch before proceeding
        required_keys = ["tokens", "response_mask", "advantages", "ref_log_probs"]
        missing_keys = [k for k in required_keys if k not in rollout_batch]

        if missing_keys:
            logger.error(f"Missing required keys in rollout_batch: {missing_keys}")
            logger.error(f"Available keys: {list(rollout_batch.keys())}")
            return mx.array(0.0), {}, {"kl_divergence": 0.0}

        # Debug: Log shapes of all inputs
        for key in required_keys:
            val = rollout_batch[key]
            if val is None:
                logger.error(f"rollout_batch['{key}'] is None")
            else:
                logger.debug(
                    f"rollout_batch['{key}'] shape: {val.shape}, size: {val.size}"
                )

        def loss_fn(model):
            tokens = rollout_batch["tokens"]
            mask = rollout_batch["response_mask"]
            advantages = rollout_batch["advantages"]
            ref_log_probs = rollout_batch["ref_log_probs"]

            # Detailed validation with specific error messages
            if tokens is None or tokens.size == 0:
                logger.error(
                    f"tokens is {'None' if tokens is None else 'empty (size=0)'}"
                )
                return mx.array(0.0), mx.array(0.0)

            if mask is None or mask.size == 0:
                logger.error(f"mask is {'None' if mask is None else 'empty (size=0)'}")
                return mx.array(0.0), mx.array(0.0)

            if advantages is None or advantages.size == 0:
                logger.error(
                    f"advantages is {'None' if advantages is None else 'empty (size=0)'}"
                )
                return mx.array(0.0), mx.array(0.0)

            if ref_log_probs is None or ref_log_probs.size == 0:
                logger.error(
                    f"ref_log_probs is {'None' if ref_log_probs is None else 'empty (size=0)'}"
                )
                return mx.array(0.0), mx.array(0.0)

            # Log actual shapes for debugging
            logger.debug(
                f"Computing loss with tokens.shape={tokens.shape}, mask.shape={mask.shape}"
            )
            logger.debug(
                f"advantages.shape={advantages.shape}, ref_log_probs.shape={ref_log_probs.shape}"
            )

            # Forward pass through the model
            model_output = model(tokens)

            # Handle both tuple and array outputs from the model
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output

            gen_len = mask.shape[1]

            # Validate logits shape
            if logits.shape[1] <= gen_len:
                logger.error(
                    f"Logits sequence length ({logits.shape[1]}) is too short for "
                    f"gen_len ({gen_len}). Cannot compute loss."
                )
                return mx.array(0.0), mx.array(0.0)

            # Extract logits and targets corresponding to the generated response part
            # Logits up to the token *before* the current one
            logits_resp = logits[:, -gen_len - 1 : -1, :]
            # Actual generated tokens
            targets_resp = tokens[:, -gen_len:]

            # Validate shape alignment
            if logits_resp.shape[1] != targets_resp.shape[1]:
                logger.error(
                    f"Shape mismatch: logits_resp.shape[1]={logits_resp.shape[1]} != "
                    f"targets_resp.shape[1]={targets_resp.shape[1]}"
                )
                return mx.array(0.0), mx.array(0.0)

            # Calculate current policy's log probabilities for generated tokens
            cur_lp = log_softmax(logits_resp.astype(mx.float32), axis=-1)
            cur_lp = mx.take_along_axis(
                cur_lp, targets_resp[..., None].astype(mx.int32), axis=-1
            ).squeeze(-1)

            # Validate cur_lp and ref_log_probs alignment
            if cur_lp.shape != ref_log_probs.shape:
                logger.error(
                    f"Shape mismatch: cur_lp.shape={cur_lp.shape} != "
                    f"ref_log_probs.shape={ref_log_probs.shape}"
                )
                return mx.array(0.0), mx.array(0.0)

            # Calculate log ratio and KL approximation
            log_ratio = mx.clip(cur_lp - ref_log_probs, -8.0, 8.0)
            ratio = mx.exp(log_ratio)
            kl_approx = ratio - 1.0 - log_ratio

            # Expand advantages if needed to match sequence length
            if advantages.ndim == 1:
                advantages_expanded = advantages[:, None]
            else:
                advantages_expanded = advantages

            # Ensure advantages broadcast correctly
            if advantages_expanded.shape[0] != mask.shape[0]:
                logger.error(
                    f"Batch size mismatch: advantages.shape[0]={advantages_expanded.shape[0]} != "
                    f"mask.shape[0]={mask.shape[0]}"
                )
                return mx.array(0.0), mx.array(0.0)

            # Policy gradient term and KL penalty term
            pg_term = -log_ratio * advantages_expanded * mask
            kl_term = beta * kl_approx * mask

            loss_per_token = pg_term + kl_term

            # Sum over tokens and average over non-masked tokens
            sum_mask = mx.sum(mask)
            if sum_mask.item() == 0:
                logger.warning("Zero sum_mask in loss calculation, returning 0.0 loss.")
                return mx.array(0.0), mx.array(0.0)

            loss = mx.sum(loss_per_token) / sum_mask
            kl_mean = mx.sum(kl_approx * mask) / sum_mask

            logger.debug(
                f"Computed loss={loss.item():.6f}, kl_mean={kl_mean.item():.6f}"
            )

            return loss, kl_mean

        # Compute value and gradients
        try:
            (loss, kl_mean), grads = value_and_grad(loss_fn, argnums=0)(self.actor)

            # Validate gradients
            if not grads:
                logger.warning(
                    "Empty gradients dictionary returned from value_and_grad"
                )

            return loss, grads, {"kl_divergence": float(kl_mean.item())}

        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"kl_divergence": 0.0}

    def compute_advantages(
        self, rewards_flat: mx.array, samples_per_prompt: int
    ) -> mx.array:
        """
        Computes group-relative advantage estimates.

        Args:
            rewards_flat: 1D MLX array of rewards for all generated samples (num_prompts * samples_per_prompt).
            samples_per_prompt: Number of responses generated per prompt.

        Returns:
            1D MLX array of advantage estimates.
        """
        if rewards_flat is None or rewards_flat.size == 0:
            logger.error("rewards_flat is None or empty in compute_advantages")
            return mx.array([])

        logger.debug(
            f"Computing advantages for rewards_flat.shape={rewards_flat.shape}, samples_per_prompt={samples_per_prompt}"
        )

        if samples_per_prompt <= 1:
            # If only one sample, use simple baseline subtraction (mean of all rewards)
            advantages = rewards_flat - mx.mean(rewards_flat)
            logger.debug(f"Single sample case: advantages.shape={advantages.shape}")
            return advantages

        num_prompts = rewards_flat.shape[0] // samples_per_prompt

        # Ensure rewards_flat length is a multiple of samples_per_prompt
        if rewards_flat.shape[0] % samples_per_prompt != 0:
            logger.warning(
                f"Rewards flat length ({rewards_flat.shape[0]}) not multiple of "
                f"samples_per_prompt ({samples_per_prompt}). Truncating for advantage calculation."
            )
            rewards_flat = rewards_flat[: num_prompts * samples_per_prompt]

        rewards_grouped = rewards_flat.reshape(num_prompts, samples_per_prompt)

        # Calculate mean and standard deviation per group (per prompt)
        mean = mx.mean(rewards_grouped, axis=1, keepdims=True)
        std = mx.std(rewards_grouped, axis=1, ddof=0, keepdims=True) + 1e-8

        # Standardize rewards within each group to get advantages
        advantages = (rewards_grouped - mean) / std
        logger.debug(f"Group-based advantages computed: shape={advantages.shape}")

        return advantages.flatten()
