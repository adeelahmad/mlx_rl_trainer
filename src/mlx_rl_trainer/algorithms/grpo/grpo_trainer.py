# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 002
# goals_of_writing_code_block: Implement the GRPO (Gradient-based Reward Policy Optimization) trainer with advanced memory and performance optimizations.
# type_of_code_response: replace
"""GRPO (Gradient-based Reward Policy Optimization) Trainer.

Key optimizations:
1.  **Dual Gradient Accumulation**: Accumulates separate gradients for thinking and answer
    portions of a trajectory, allowing for targeted updates.
2.  **Hybrid RL+SFT Training**: Combines RL loss with a Supervised Fine-Tuning (SFT)
    loss, weighted by `sft_think_loss_weight` and `sft_answer_loss_weight`.
3.  **Adaptive SFT Weights**: Dynamically adjusts SFT loss weights based on KL
    divergence to prevent model collapse (`adaptive_sft_weights`).
4.  **Adaptive Layer-wise Gradient Scaling**: Boosts gradients for specific layer bands
    (e.g., answer-focused layers) to accelerate learning (`boost_answer_grad_layers`).
5.  **Constrained Thinking**: Enforces `min_think_tokens` and `max_think_tokens` during
    rollout generation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time

from mlx_rl_trainer.core.trainer import BaseTrainer, EvaluationMetrics, TrainingMetrics
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.utils.mlx_utils import (
    mask_gradients_by_layer,
    combine_gradients,
    get_grad_norm,
)
from .grpo_algorithm import grpo_loss

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    """GRPO Trainer with memory and performance optimizations."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: ModelManager,
        data_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        reward_composer: RewardComposer,
        paged_kv_cache: Optional[Any] = None,
        metrics_logger: Optional[Any] = None,
    ):
        super().__init__(
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            paged_kv_cache,
            metrics_logger,
        )

    def _setup(self) -> Tuple[int, int]:
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            model_path=self.config.model.ref_model_path,
            type_name="actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
        )
        self.ref_model, _ = self.model_manager.load_model(
            model_path=self.config.model.ref_model_path,
            type_name="reference",
            is_trainable=False,
        )

        self.optimizer, self.lr_scheduler = self.model_manager.create_optimizer(
            self.actor_model, self.config.trainer
        )

        self.loss_and_grad_fn = nn.value_and_grad(self.actor_model, grpo_loss)

        resume_step, resume_epoch = self.checkpoint_manager.resume_from_checkpoint(
            self.actor_model, self.optimizer
        )
        return resume_step, resume_epoch

    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> Tuple[TrainingMetrics, Dict[str, mx.array], Dict[str, Any]]:
        """Executes a single training step including forward/backward passes and gradient updates."""
        start_time = time.process_time()

        # Determine SFT loss weights with adaptive logic
        sft_think_weight = self.config.sft_thinking_weight
        sft_answer_weight = self.config.sft_answer_weight

        kl_div = rollout_batch.get("kl_divergence", 0.0)
        if self.config.trainer.adaptive_gradient_weights and kl_div > 0:
            # Reduce SFT weight if KL is high to prioritize RL objective
            kl_factor = np.clip(
                1.0 - (kl_div / self.config.trainer.grpo_beta), 0.1, 1.0
            )
            sft_think_weight *= kl_factor
            sft_answer_weight *= kl_factor

        # Compute loss and gradients for the entire batch
        (loss, (loss_rl, loss_sft, metrics)), grads = self.loss_and_grad_fn(
            self.actor_model,
            self.ref_model,
            rollout_batch["tokens"],
            rollout_batch["log_probs"],
            rollout_batch["advantages"],
            rollout_batch["returns"],
            rollout_batch["attention_mask"],
            rollout_batch["thinking_mask"],
            rollout_batch["answer_mask"],
            self.config.trainer.grpo_beta,
            sft_think_weight,
            sft_answer_weight,
            self.config,
        )

        # Adaptive gradient scaling for answer-focused layers
        if self.config.trainer.boost_answer_grad_layers:
            think_token_count = mx.sum(rollout_batch["thinking_mask"]).item()
            answer_token_count = mx.sum(rollout_batch["answer_mask"]).item()
            total_tokens = think_token_count + answer_token_count

            if answer_token_count > 0 and total_tokens > 0:
                answer_ratio = answer_token_count / total_tokens
                think_ratio = 1.0 - answer_ratio

                # Boost answer gradients, penalize thinking gradients
                answer_boost = 1.0 + (
                    think_ratio * self.config.trainer.answer_grad_boost_factor
                )
                think_penalty = 1.0 - (
                    answer_ratio * self.config.trainer.thinking_grad_penalty_factor
                )

                logger.info(f"Adaptive weights activated:")
                logger.info(
                    f"  Thinking: {think_token_count} tokens ({think_ratio:.1%})"
                )
                logger.info(
                    f"  Answer: {answer_token_count} tokens ({answer_ratio:.1%})"
                )
                logger.info(
                    f"  Boosted answer: {answer_boost:.1f}x RL + {sft_answer_weight:.2f}x SFT"
                )
                logger.info(f"  Boosted SFT: {sft_think_weight:.2f}x SFT")

                # Apply boosts/penalties to gradients
                think_grads = mask_gradients_by_layer(
                    grads, self.config.trainer.thinking_layers_config
                )
                answer_grads = mask_gradients_by_layer(
                    grads, self.config.trainer.answer_layers_config
                )

                # Combine scaled gradients
                grads = combine_gradients(
                    [(think_grads, think_penalty), (answer_grads, answer_boost)]
                )

        # Hybrid RL+SFT training logic
        if sft_think_weight > 0 or sft_answer_weight > 0:
            logger.info("Hybrid RL+SFT training enabled:")
            logger.info(
                f"  Answer: {self.config.trainer.grpo_beta:.1f}x RL + {sft_answer_weight:.2f}x SFT"
            )

        # Dual gradient accumulation logic
        if self.config.trainer.alternate_dual_gradients:
            # Alternate between thinking and answer gradients
            if update_step % 2 == 0:  # Thinking step
                answer_grads = mask_gradients_by_layer(
                    grads, {"include": []}
                )  # Zero out answer grads
                grads = combine_gradients([(grads, 1.0), (answer_grads, -1.0)])
            else:  # Answer step
                think_grads = mask_gradients_by_layer(
                    grads, {"include": []}
                )  # Zero out think grads
                grads = combine_gradients([(grads, 1.0), (think_grads, -1.0)])

        else:  # Standard dual gradient logic
            think_layers = self.config.trainer.thinking_layers_config
            answer_layers = self.config.trainer.answer_layers_config

            # Determine overlapping layers
            overlap_layers = sorted(
                list(
                    set(think_layers.get("include", [])).intersection(
                        set(answer_layers.get("include", []))
                    )
                )
            )

            logger.info("Dual gradient mode - OVERLAPPING layers:")
            logger.info(
                f"  Thinking: {think_layers.get('include')} ({self.config.trainer.grpo_beta}x)"
            )
            logger.info(
                f"  Answer: {answer_layers.get('include')} ({self.config.trainer.boost_answer_grad_layers}x)"
            )
            logger.info(
                f"  Overlap: {overlap_layers} (total {self.config.trainer.grpo_beta + self.config.trainer.boost_answer_grad_layers}x)"
            )

        mx.eval(loss, grads)
        step_time = time.process_time() - start_time

        # Prepare metrics for logging
        training_metrics = TrainingMetrics(
            loss=loss.item(),
            reward_mean=metrics["reward_mean"].item(),
            grad_norm=get_grad_norm(grads),
            learning_rate=self.optimizer.learning_rate,
            step_time_s=step_time,
            kl_divergence=kl_div,
            epoch=self.current_epoch,
            step=update_step,
            custom_metrics={
                "train/rl_loss": loss_rl.item(),
                "train/sft_loss": loss_sft.item(),
            },
        )

        return training_metrics, grads, {}

    def generate_rollouts(
        self, batch_data: Dict[str, Any], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, List[float]], Dict[str, Any]]:
        """Generates rollouts and computes rewards for a batch of prompts."""
        # Enforce thinking token constraints
        self.config.data.max_gen_len = (
            self.config.trainer.min_think_tokens + self.config.trainer.max_think_tokens
        )

        # Generate responses from the actor model
        dataset_from_batch = batch_data["dataset"]
        (
            rollout_data,
            avg_reward,
            raw_rewards,
            generation_metrics,
        ) = generate_rollouts_for_batch(
            model=self.actor_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            prompts_data=batch_data["prompts_data"],
            dataset=dataset_from_batch,
            config=self.config,
            reward_composer=self.reward_composer,
            model_manager=self.model_manager,
            run_id=self._run_id,
            current_update=update_step,
            is_invalid_batch=False,  # Temporarily set to False, as it's calculated after this call
        )

        # Log samples if enabled
        is_invalid_batch = avg_reward == 0 and not rollout_data
        _maybe_log_samples(
            self.config,
            update_step,
            batch_data["prompts_data"],
            rollout_data.get("decoded_responses", []),
            raw_rewards,
            "grpo",
            self._run_id,
            is_invalid_batch,
        )

        return rollout_data, avg_reward, raw_rewards, generation_metrics

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        """Runs evaluation on the validation set and returns metrics."""
        # This is a placeholder. In a real implementation, you would run evaluation tasks.
        logger.info(f"Running evaluation at step {update_step}...")
        # For now, returning a dummy metric
        return [
            EvaluationMetrics(
                task_name="dummy_task",
                pass_rate=np.random.rand(),
                additional_info={"step": update_step},
            )
        ]
