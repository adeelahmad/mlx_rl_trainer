import logging, time, gc, json
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.utils import load_config as mlx_lm_load_config

from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.utils.mlx_utils import (
    _maybe_clip_grad_norm,
    mask_grads_to_layer_band,
    scale_grads_by_band,
)
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    def _setup(self) -> Tuple[int, int]:
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            self.config.model.model_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
        )
        self.ref_model, _ = self.model_manager.load_model(
            self.config.model.ref_model_path, "reference", is_trainable=False
        )
        self.grpo_algorithm = GRPOAlgorithm(
            self.config, self.actor_model, self.ref_model
        )
        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(
                self.config.trainer.optimizer_beta1,
                self.config.trainer.optimizer_beta2,
            ),
            weight_decay=self.config.trainer.optimizer_weight_decay,
        )
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        start_updates, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )
        return metadata.get("num_updates", 0), metadata.get("epoch", 0)

    def generate_rollouts(
        self, batch_data: Dict[str, Any], update_step: int
    ) -> Tuple[Dict, float, Dict]:
        prompts_data = batch_data.get("prompts_data", [])
        is_invalid_batch = any(p.get("is_invalid_sample", False) for p in prompts_data)

        return generate_rollouts_for_batch(
            model=self.actor_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            prompts_data=prompts_data,
            dataset=self.data_manager._train_dataset,
            config=self.config,
            reward_composer=self.reward_composer,
            paged_kv_cache=self.paged_kv_cache,
            run_id=self._run_id,
            current_update=update_step,
            is_invalid_batch=is_invalid_batch,
        )

    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> Tuple[TrainingMetrics, Dict[str, mx.array]]:
        start_time = time.time()

        loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
            rollout_batch, self.config, self.tokenizer.pad_token_id
        )

        # Scale gradients for accumulation
        scaled_grads = tree_map(
            lambda g: g / self.config.trainer.grad_accum_steps, grads
        )

        metrics_obj = TrainingMetrics(
            loss=loss.item(),
            reward_mean=rollout_batch["advantages"].mean().item(),
            reward_std=rollout_batch["advantages"].std().item(),
            grad_norm=0.0,  # Will be calculated in the main run loop after accumulation
            learning_rate=self.lr_scheduler(update_step),
            step_time_s=time.time() - start_time,
            kl_divergence=metrics.get("kl_divergence", 0.0),
            epoch=self.current_epoch,
            step=update_step,
        )
        return metrics_obj, scaled_grads

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        # Placeholder for full evaluation logic
        logger.info(f"Evaluation at step {update_step} is a placeholder.")
        return []
