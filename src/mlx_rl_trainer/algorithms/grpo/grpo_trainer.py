import logging
import time
from typing import Any, Dict

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map
from mlx_lm.tuner.utils import build_schedule

from mlx_rl_trainer.core.trainer import BaseTrainer
from mlx_rl_trainer.core.types import TrainingMetrics
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from mlx_rl_trainer.utils.mlx_utils import (
    mask_grads_to_layer_band,
    _maybe_clip_grad_norm,
)

from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    def _setup(self) -> (int, int):
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            self.config.model.model_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
        )
        self.ref_model, _ = self.model_manager.load_model(
            self.config.model.ref_model_path,
            "reference",
            is_trainable=False,
        )

        self.grpo_algorithm = GRPOAlgorithm(
            self.config, self.actor_model, self.ref_model
        )
        self.optimizer = optim.AdamW(learning_rate=self.config.trainer.learning_rate)
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        step, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )

        return step, metadata.get("epoch", 0)

    def train_step(
        self, rollout_batch: Dict[str, Any], update_step: int
    ) -> (TrainingMetrics, mx.array, Dict[str, float]):
        start_time = time.time()
        custom_metrics = {}

        if (
            self.config.trainer.use_dual_gradients
            and "thinking_mask" in rollout_batch
            and "answer_mask" in rollout_batch
        ):
            (
                loss,
                think_grads,
                answer_loss,
                answer_grads,
                loss_metrics,
            ) = self.grpo_algorithm.calculate_dual_gradient_loss(
                rollout_batch, self.config, self.tokenizer.pad_token_id
            )

            think_layer_start = self.config.trainer.thinking_layer_start
            think_layer_end = self.config.trainer.thinking_layer_end
            answer_layer_start = self.config.trainer.answer_layer_start
            answer_layer_end = self.config.trainer.answer_layer_end

            masked_think_grads = mask_grads_to_layer_band(
                think_grads, start=think_layer_start, end=think_layer_end
            )
            masked_answer_grads = mask_grads_to_layer_band(
                answer_grads, start=answer_layer_start, end=answer_layer_end
            )

            answer_weight = self.config.trainer.answer_gradient_weight
            weighted_answer_grads = tree_map(
                lambda g: g * answer_weight, masked_answer_grads
            )

            grads = tree_map(mx.add, masked_think_grads, weighted_answer_grads)
            total_loss = loss.item() + answer_loss.item()

            _, think_grad_norm = _maybe_clip_grad_norm(masked_think_grads, None)
            _, answer_grad_norm = _maybe_clip_grad_norm(weighted_answer_grads, None)

            custom_metrics["loss/thinking_loss"] = loss.item()
            custom_metrics["loss/answer_rl_loss"] = answer_loss.item()
            custom_metrics["grads/think_layer_norm"] = think_grad_norm
            custom_metrics["grads/answer_layer_norm"] = answer_grad_norm
        else:
            (
                total_loss,
                grads,
                loss_metrics,
            ) = self.grpo_algorithm.calculate_loss_and_grads(
                rollout_batch, self.config, self.tokenizer.pad_token_id
            )
            total_loss = total_loss.item()

        reward_mean = rollout_batch["advantages"].mean().item()
        reward_std = rollout_batch["advantages"].std().item()

        metrics = TrainingMetrics(
            loss=total_loss,
            reward_mean=reward_mean,
            reward_std=reward_std,
            grad_norm=0.0,
            learning_rate=self.lr_scheduler(update_step),
            step_time_s=time.time() - start_time,
            kl_divergence=loss_metrics.get("kl_divergence", 0.0),
            epoch=self.current_epoch,
            step=update_step,
            custom_metrics=custom_metrics,
        )
        return metrics, grads, loss_metrics

    def generate_rollouts(
        self, batch_data: Dict, update_step: int, is_validation: bool = False
    ):
        prompts_data = batch_data.get("prompts_data", [])
        dataset_to_use = (
            self.data_manager._val_dataset
            if is_validation
            else self.data_manager._train_dataset
        )

        return generate_rollouts_for_batch(
            model=self.actor_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            prompts_data=prompts_data,
            dataset=dataset_to_use,
            config=self.config,
            reward_composer=self.reward_composer,
            run_id=self._run_id,
            current_update=update_step,
            is_invalid_batch=False,
        )
