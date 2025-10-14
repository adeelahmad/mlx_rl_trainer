# <FILE name="./grpo_trainer.py">
# Complete File with Dual Gradient (Thinking vs Answer) Training Support
#
# FIXES APPLIED:
# 1. Smart default layer ranges - answer layers start AFTER thinking layers (no overlap)
# 2. Configuration validation with helpful error messages
# 3. Clear logging of layer configuration on first dual gradient step
# 4. Detection and warning when layers overlap
#
# LAYER CONFIGURATION:
# Default behavior creates SEPARATE pathways:
#   - Thinking path: layers 22-30 (1x) - learns deep reasoning
#   - Answer path: layers 31-36 (2x) - learns fast responses
#
# To create OVERLAPPING layers (advanced):
#   - Set answer_layer_start: 22 (same as thinking_layer_start)
#   - Layers 22-30 will receive BOTH gradients (3x total)
#
# CONFIGURATION EXAMPLE:
# trainer:
#   use_dual_gradients: true
#   thinking_layer_start: 22
#   thinking_layer_end: 30
#   # answer_layer_start: 31  # Optional - auto-calculated if omitted
#   answer_layer_end: 36
#   answer_gradient_weight: 2.0
#
import logging
import time
import gc
import json
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.utils import load_config as mlx_lm_load_config
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.utils.mlx_utils import _maybe_clip_grad_norm, mask_grads_to_layer_band, scale_grads_by_band
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)

class GRPOTrainer(BaseTrainer):
    def _setup(self):
        """Initialize models, optimizer, and load checkpoints."""
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            self.config.model.model_path,
            'actor',
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump()
        )

        self.ref_model, _ = self.model_manager.load_model(
            self.config.model.ref_model_path,
            'reference',
            is_trainable=False
        )

        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)

        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2),
            weight_decay=self.config.trainer.optimizer_weight_decay
        )

        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        num_updates, state = self.checkpoint_manager.load_latest_state(self.actor_model, self.optimizer)

        # Apply gradient checkpointing if enabled
        if self.config.use_grad_checkpointing:
            logging.info('Applying gradient checkpointing to transformer layers...')
            try:
                model_core = getattr(self.actor_model, 'model', self.actor_model)
                if hasattr(model_core, 'layers') and isinstance(model_core.layers, list):
                    checkpointed_count = 0
                    for layer in model_core.layers:
                        if self.config.grad_checkpoint_layers and self.config.grad_checkpoint_layers > 0 and self.config.grad_checkpoint_layers > checkpointed_count:
                            grad_checkpoint(layer)
                            checkpointed_count += 1
                    logging.info(f"Successfully applied gradient checkpointing to {len(model_core.layers)} layers.")
                else:
                    logging.warning("Could not find a standard '.model.layers' attribute. Gradient checkpointing not applied.")
            except Exception as e:
                logging.error(f"Failed to apply gradient checkpointing: {e}", exc_info=True)

        return state.get('num_updates', 0), state.get('epoch', 0)

    def generate_rollouts(self, batch_data, update_step):
        """Generate rollouts for the given batch."""
        prompts_data = batch_data.get('prompts_data', [])
        is_invalid_batch = any(p.get('is_invalid_sample', False) for p in prompts_data)

        return generate_rollouts_for_batch(
            model=self.actor_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            prompts_data=prompts_data,
            dataset=self.data_manager._train_dataset,
            config=self.config,
            reward_composer=self.reward_composer,
            run_id=self._run_id,
            current_update=update_step,
            is_invalid_batch=is_invalid_batch
        )

    def train_step(self, rollout_batch, update_step):
        """
        Execute single training step with optional dual gradient support.

        If rollout_batch contains 'thinking_mask' and 'answer_mask', applies
        dual gradient approach:
        - Thinking gradients: 1x weight to middle layers
        - Answer gradients: 2x weight to final layers (non-overlapping by default)

        Otherwise falls back to standard gradient computation.
        """
        B = rollout_batch
        start_time = time.time()

        # Check if we should use dual gradient approach
        use_dual_gradients = ('thinking_mask' in B and 'answer_mask' in B)

        if use_dual_gradients and hasattr(self.config.trainer, 'use_dual_gradients') and self.config.trainer.use_dual_gradients:
            # Dual gradient path: thinking vs answer separation
            thinking_loss, thinking_grads, answer_loss, answer_grads, metrics = \
                self.grpo_algorithm.calculate_dual_gradient_loss(B, self.config, self.tokenizer.pad_token_id)

            # Get layer ranges from config or use smart defaults
            thinking_layer_start = getattr(self.config.trainer, 'thinking_layer_start', 22)
            thinking_layer_end = getattr(self.config.trainer, 'thinking_layer_end', 30)

            # Default: answer layers start AFTER thinking layers to avoid overlap
            # This creates separate pathways: thinking (22-30) vs answer (31-36)
            default_answer_start = thinking_layer_end + 1
            answer_layer_start = getattr(self.config.trainer, 'answer_layer_start', default_answer_start)
            answer_layer_end = getattr(self.config.trainer, 'answer_layer_end', 36)
            answer_gradient_weight = getattr(self.config.trainer, 'answer_gradient_weight', 2.0)

            # Validate configuration
            if thinking_layer_start < 0 or thinking_layer_end < thinking_layer_start:
                raise ValueError(f"Invalid thinking layer range: [{thinking_layer_start}, {thinking_layer_end}]")

            if answer_layer_start < 0 or answer_layer_end < answer_layer_start:
                raise ValueError(f"Invalid answer layer range: [{answer_layer_start}, {answer_layer_end}]")

            if answer_layer_end > 100:  # Sanity check for reasonable layer count
                logger.warning(f"Answer layer end ({answer_layer_end}) seems unusually high. Verify your config.")

            # Log layer configuration on first dual gradient step
            if not hasattr(self, '_dual_grad_config_logged'):
                overlap_layers = set(range(thinking_layer_start, thinking_layer_end + 1)) & set(range(answer_layer_start, answer_layer_end + 1))
                if overlap_layers:
                    logger.info(f"Dual gradient mode - OVERLAPPING layers:")
                    logger.info(f"  Thinking: layers {thinking_layer_start}-{thinking_layer_end} (1x weight)")
                    logger.info(f"  Answer: layers {answer_layer_start}-{answer_layer_end} ({answer_gradient_weight}x weight)")
                    logger.info(f"  Overlap: layers {sorted(overlap_layers)} receive BOTH gradients (total {1 + answer_gradient_weight}x)")
                else:
                    logger.info(f"Dual gradient mode - SEPARATE pathways:")
                    logger.info(f"  Thinking path: layers {thinking_layer_start}-{thinking_layer_end} (1x weight)")
                    logger.info(f"  Answer path: layers {answer_layer_start}-{answer_layer_end} ({answer_gradient_weight}x weight)")
                self._dual_grad_config_logged = True

            # Scale thinking gradients (1x weight)
            thinking_grads_scaled = tree_map(
                lambda g: g / self.config.trainer.grad_accum_steps,
                thinking_grads
            )

            # Mask to thinking layers only
            thinking_grads_masked = mask_grads_to_layer_band(
                thinking_grads_scaled,
                start=thinking_layer_start,
                end=thinking_layer_end,
                include_embed=False,
                include_head=False
            )

            # Scale answer gradients (2x weight by default)
            answer_grads_scaled = tree_map(
                lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                answer_grads
            )

            # Mask to answer layers (wider range, includes final layers)
            answer_grads_masked = mask_grads_to_layer_band(
                answer_grads_scaled,
                start=answer_layer_start,
                end=answer_layer_end,
                include_embed=False,
                include_head=True
            )

            # Combine the gradients
            combined_grads = tree_map(
                lambda t, a: t + a,
                thinking_grads_masked,
                answer_grads_masked
            )

            avg_loss = (thinking_loss.item() + answer_loss.item()) / 2

        else:
            # Standard gradient path (original behavior)
            loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
                B,
                self.config,
                self.tokenizer.pad_token_id
            )

            combined_grads = tree_map(
                lambda g: g / self.config.trainer.grad_accum_steps,
                grads
            )

            avg_loss = loss.item()

        # Create training metrics
        training_metrics = TrainingMetrics(
            loss=avg_loss,
            reward_mean=B['advantages'].mean().item(),
            reward_std=B['advantages'].std().item(),
            grad_norm=0.0,
            learning_rate=self.lr_scheduler(update_step),
            step_time_s=time.time() - start_time,
            kl_divergence=metrics.get('kl_divergence', 0.0),
            epoch=self.current_epoch,
            step=update_step
        )

        return training_metrics, combined_grads

    def evaluate(self, update_step):
        """Placeholder for evaluation logic."""
        logger.info(f"Evaluation at step {update_step} is a placeholder.")
        return []

# End of File: ./grpo_trainer.py
#</FILE>
