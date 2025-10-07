import logging, gc
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.utils import load_config as mlx_lm_load_config

from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from .grpo_algorithm import GRPOAlgorithm
from mlx_rl_trainer.utils.mlx_utils import scale_grads_by_band, mask_grads_to_layer_band, _maybe_clip_grad_norm

logger = logging.getLogger(__name__)

class GRPOTrainer(BaseTrainer):
    def _setup(self) -> Tuple[int, int]:
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            self.config.model.model_path, "actor", is_trainable=True, 
            apply_lora=self.config.model.use_lora, lora_config=self.config.model.model_dump()
        )
        self.ref_model, _ = self.model_manager.load_model(self.config.model.ref_model_path, "reference", is_trainable=False)
        
        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)
        self.optimizer = optim.AdamW(learning_rate=self.config.trainer.learning_rate, betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2), weight_decay=self.config.trainer.optimizer_weight_decay)
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)
        
        start_updates, metadata = self.checkpoint_manager.load_latest_state(self.actor_model, self.optimizer)
        return start_updates, metadata.get("epoch", 0)

    def generate_rollouts(self, batch_prompts: List[Dict], update_step: int, is_invalid_batch: bool) -> Tuple[Dict, float, Dict]:
        return generate_rollouts_for_batch(
            model=self.actor_model, ref_model=self.ref_model, tokenizer=self.tokenizer,
            prompts_data=batch_prompts, dataset=self.data_manager._train_dataset,
            config=self.config, reward_composer=self.reward_composer,
            paged_kv_cache=self.paged_kv_cache, run_id=self.run_id, current_update=update_step,
            is_invalid_batch=is_invalid_batch
        )

    def train_step(self, rollout_batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict, Dict]:        
        loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config, self.tokenizer.pad_token_id)
        return loss, grads, metrics

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        logger.info(f"[EVAL] Evaluation at step {update_step} is a placeholder.")
        return []
