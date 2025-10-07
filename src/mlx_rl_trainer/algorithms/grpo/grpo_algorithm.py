"""
GRPO (Group Relative Policy Optimization) Algorithm Implementation.
"""
import logging
from typing import Dict, Any, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


class GRPOAlgorithm:
    """Implements the GRPO loss calculation and advantage estimation."""

    def __init__(
        self, config: ExperimentConfig, actor_model: nn.Module, ref_model: nn.Module
    ):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta

    def compute_advantages(
        self, rewards_flat: mx.array, samples_per_prompt: int
    ) -> mx.array:
        """Computes advantages by normalizing rewards within each prompt's response group."""
        if samples_per_prompt <= 1:
            # Simple whitening across the batch
            return (rewards_flat - mx.mean(rewards_flat)) / (
                mx.std(rewards_flat) + 1e-8
            )

        num_prompts = rewards_flat.shape[0] // samples_per_prompt
        rewards_grouped = rewards_flat.reshape(num_prompts, samples_per_prompt)

        mean_per_group = mx.mean(rewards_grouped, axis=1, keepdims=True)
        std_per_group = mx.std(rewards_grouped, axis=1, keepdims=True)

        advantages_grouped = (rewards_grouped - mean_per_group) / (std_per_group + 1e-8)
        return advantages_grouped.flatten()

    def calculate_loss_and_grads(
        self,
        rollout_batch: Dict[str, mx.array],
        full_config: ExperimentConfig,
        pad_token_id: int,
    ) -> Tuple[mx.array, Dict[str, mx.array], Dict[str, float]]:
        """Calculates GRPO loss and gradients."""

        def loss_fn(actor_model: nn.Module) -> Tuple[mx.array, Dict[str, mx.array]]:
            # 1. Forward pass to get current log probabilities
            logits = actor_model(rollout_batch["tokens"])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            # We need log probs for the response part of the sequence
            prompt_len = (
                rollout_batch["tokens"].shape[1]
                - rollout_batch["response_mask"].shape[1]
            )
            response_logits = logits[:, prompt_len - 1 : -1, :]
            response_tokens = rollout_batch["tokens"][:, prompt_len:]

            log_probs_all = nn.log_softmax(response_logits, axis=-1)
            actor_log_probs = mx.take_along_axis(
                log_probs_all, response_tokens[..., None], axis=-1
            ).squeeze(-1)

            # 2. Compute KL divergence
            log_ratio = actor_log_probs - rollout_batch["ref_log_probs"]
            kl_div = (mx.exp(log_ratio) - 1) - log_ratio
            kl_per_token = kl_div * rollout_batch["response_mask"]

            # 3. Compute Policy Loss
            advantages_expanded = rollout_batch["advantages"][:, None]
            policy_loss_per_token = (
                -log_ratio * advantages_expanded * rollout_batch["response_mask"]
            )

            # 4. Combine for total loss
            total_loss_per_token = policy_loss_per_token + self.beta * kl_per_token

            # Sum over sequence length and mean over batch
            loss = mx.sum(total_loss_per_token) / mx.sum(rollout_batch["response_mask"])

            # Auxiliary metrics
            kl_mean = mx.sum(kl_per_token) / mx.sum(rollout_batch["response_mask"])
            policy_loss_mean = mx.sum(policy_loss_per_token) / mx.sum(
                rollout_batch["response_mask"]
            )

            return loss, {"kl_divergence": kl_mean, "policy_loss": policy_loss_mean}

        try:
            value_and_grad_fn = nn.value_and_grad(self.actor, loss_fn)
            (loss, aux_metrics), grads = value_and_grad_fn(self.actor)

            metrics = {k: float(v.item()) for k, v in aux_metrics.items()}
            return loss, grads, metrics
        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"kl_divergence": 0.0, "policy_loss": 0.0}
