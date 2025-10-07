"""
GRPO (Group Relative Policy Optimization) Algorithm Implementation.
"""

import logging
from typing import Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


class GRPOAlgorithm:
    """
    GRPO Algorithm: Group Relative Policy Optimization.

    This implementation computes advantages, policy losses, and gradients
    for reinforcement learning from human feedback (RLHF) training.
    """

    def __init__(self, config: Any, actor_model: nn.Module, ref_model: nn.Module):
        """
        Initialize GRPO algorithm.

        Args:
            config: Experiment configuration containing trainer hyperparameters
            actor_model: The policy model being trained
            ref_model: The reference model (frozen)
        """
        self.config = config
        self.actor = actor_model
        self.reference = ref_model

        # Get hyperparameters from config
        self.kl_coef = getattr(config.trainer, "kl_coef", 0.1)
        self.clip_range = getattr(config.trainer, "clip_range", 0.2)

        logger.info(
            f"GRPOAlgorithm initialized with kl_coef={self.kl_coef}, "
            f"clip_range={self.clip_range}"
        )

    def compute_advantages(
        self, rewards_flat: mx.array, samples_per_prompt: int = 1
    ) -> mx.array:
        """
        Compute advantages using group normalization.

        In GRPO, advantages are computed by normalizing rewards within each group
        of responses for the same prompt. This encourages the model to prefer
        better responses relative to other responses for the same prompt.

        Args:
            rewards_flat: Flat array of rewards, shape (batch_size * samples_per_prompt,)
            samples_per_prompt: Number of response samples per prompt

        Returns:
            Advantages array of same shape as rewards_flat
        """
        logger.debug(
            f"Computing advantages for {len(rewards_flat)} rewards, "
            f"{samples_per_prompt} samples per prompt"
        )

        batch_size = len(rewards_flat) // samples_per_prompt

        if samples_per_prompt == 1:
            # Simple case: normalize across the entire batch
            advantages = (rewards_flat - mx.mean(rewards_flat)) / (
                mx.std(rewards_flat) + 1e-8
            )
        else:
            # Reshape to (batch_size, samples_per_prompt)
            rewards_grouped = rewards_flat.reshape(batch_size, samples_per_prompt)

            # Normalize within each group (per prompt)
            mean_per_group = mx.mean(rewards_grouped, axis=1, keepdims=True)
            std_per_group = mx.std(rewards_grouped, axis=1, keepdims=True)

            advantages_grouped = (rewards_grouped - mean_per_group) / (
                std_per_group + 1e-8
            )

            # Flatten back
            advantages = advantages_grouped.reshape(-1)

        logger.debug(
            f"Advantages computed: mean={float(mx.mean(advantages).item()):.4f}, "
            f"std={float(mx.std(advantages).item()):.4f}"
        )

        return advantages

    def calculate_loss_and_grads(
        self, rollout_batch: Dict[str, mx.array], full_config: Any, pad_token_id: int
    ) -> Tuple[mx.array, Dict[str, mx.array], Dict[str, float]]:
        """
        Calculate GRPO loss and gradients.

        Args:
            rollout_batch: Dictionary containing:
                - tokens: Full sequence (prompt + response) of shape (batch, seq_len)
                - response_mask: Mask for response tokens of shape (batch, response_len)
                - advantages: Computed advantages of shape (batch,)
                - ref_log_probs: Reference model log probs of shape (batch, response_len)
                - actor_log_probs: Actor model log probs of shape (batch, response_len) (optional)
            full_config: Full experiment configuration
            pad_token_id: Padding token ID

        Returns:
            Tuple of (loss, gradients_dict, metrics_dict)
        """
        # Extract data from rollout batch
        tokens = rollout_batch["tokens"]
        response_mask = rollout_batch["response_mask"]
        advantages = rollout_batch["advantages"]
        ref_log_probs = rollout_batch["ref_log_probs"]

        logger.debug(
            f"calculate_loss_and_grads: tokens.shape={tokens.shape}, "
            f"response_mask.shape={response_mask.shape}, "
            f"advantages.shape={advantages.shape}"
        )

        # Define loss function for value_and_grad
        def loss_fn(actor_model):
            """
            Compute GRPO loss. Returns (loss, auxiliary_metrics) tuple.
            MLX value_and_grad will differentiate w.r.t. the first return value (loss).
            """
            # Forward pass through actor model
            logits = actor_model(tokens, cache=None)
            if isinstance(logits, tuple):
                logits = logits[0]

            logits = logits.astype(mx.float32)

            # Get log probs for response tokens
            # We need to get log probs for the tokens in the response
            # The logits at position i predict token i+1

            if "actor_log_probs" in rollout_batch:
                # Use pre-computed log probs from generation
                actor_log_probs = rollout_batch["actor_log_probs"]
            else:
                # Compute log probs from logits
                # This is a fallback - ideally we'd have them from generation
                logger.warning("Computing actor_log_probs from logits (fallback)")

                # For now, we'll compute them over the full sequence
                # This requires knowing where responses start
                # Assuming response_mask indicates response positions
                log_probs_all = nn.log_softmax(logits[:, :-1, :], axis=-1)
                response_tokens = tokens[:, 1:]  # Next token targets

                actor_log_probs = mx.take_along_axis(
                    log_probs_all, response_tokens[..., None], axis=-1
                ).squeeze(-1)

                # We only care about response positions
                # response_mask should already be the right shape
                if actor_log_probs.shape != response_mask.shape:
                    # Adjust shapes if needed
                    min_len = min(actor_log_probs.shape[1], response_mask.shape[1])
                    actor_log_probs = actor_log_probs[:, :min_len]
                    response_mask_local = response_mask[:, :min_len]
                else:
                    response_mask_local = response_mask

                actor_log_probs = actor_log_probs * response_mask_local

            # Ensure shapes match
            if actor_log_probs.shape != ref_log_probs.shape:
                logger.error(
                    f"Shape mismatch: actor_log_probs {actor_log_probs.shape} "
                    f"vs ref_log_probs {ref_log_probs.shape}"
                )
                # Try to fix by taking minimum length
                min_len = min(actor_log_probs.shape[1], ref_log_probs.shape[1])
                actor_log_probs = actor_log_probs[:, :min_len]
                ref_log_probs_local = ref_log_probs[:, :min_len]
                response_mask_local = response_mask[:, :min_len]
            else:
                ref_log_probs_local = ref_log_probs
                response_mask_local = response_mask

            # Compute KL divergence (actor vs reference)
            kl_div = actor_log_probs - ref_log_probs_local
            kl_div_masked = kl_div * response_mask_local

            # Mean KL per sequence (sum over tokens, mean over batch)
            kl_per_seq = mx.sum(kl_div_masked, axis=1) / (
                mx.sum(response_mask_local, axis=1) + 1e-8
            )
            kl_mean = mx.mean(kl_per_seq)

            # Compute policy loss
            # Sum log probs over sequence for each sample
            log_probs_sum = mx.sum(actor_log_probs * response_mask_local, axis=1) / (
                mx.sum(response_mask_local, axis=1) + 1e-8
            )

            # GRPO objective: maximize advantages * log_probs - kl_penalty
            # Negative because we minimize loss
            policy_loss = -mx.mean(advantages * log_probs_sum)

            # KL penalty
            kl_penalty = self.kl_coef * kl_mean

            # Total loss
            total_loss = policy_loss + kl_penalty

            # Return (loss, auxiliary_data)
            # Only the first element (total_loss) will be differentiated
            return total_loss, {
                "kl_mean": kl_mean,
                "policy_loss": policy_loss,
                "kl_penalty": kl_penalty,
            }

        # Compute value and gradients using MLX
        try:
            # CRITICAL: Use mx.value_and_grad with proper 2-step pattern
            value_and_grad_fn = mx.value_and_grad(loss_fn)
            (loss, aux_metrics), grads = value_and_grad_fn(self.actor)

            # Validate gradients
            if not grads:
                logger.warning("No gradients computed in calculate_loss_and_grads")
                return (
                    loss,
                    {},
                    {
                        "kl_divergence": 0.0,
                        "policy_loss": 0.0,
                    },
                )

            # Convert gradients to dict format expected by optimizer
            grad_dict = dict(tree_flatten(grads))

            # Prepare metrics
            metrics = {
                "kl_divergence": float(aux_metrics["kl_mean"].item()),
                "policy_loss": float(aux_metrics["policy_loss"].item()),
                "kl_penalty": float(aux_metrics["kl_penalty"].item()),
            }

            logger.debug(
                f"Loss computed: total={float(loss.item()):.4f}, "
                f"policy={metrics['policy_loss']:.4f}, "
                f"kl={metrics['kl_divergence']:.4f}"
            )

            return loss, grad_dict, metrics

        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            # Return zero loss and empty grads on error
            return (
                mx.array(0.0),
                {},
                {
                    "kl_divergence": 0.0,
                    "policy_loss": 0.0,
                },
            )
