#-----------#
# <FILE name="./grpo_trainer.py">
# Complete File with Dual Gradient (Thinking vs Answer) Training Support
# Now includes HYBRID RL+SFT mode (Approach 2)
#
# FIXES APPLIED:
# 1. Smart default layer ranges - answer layers start AFTER thinking layers (no overlap)
# 2. Configuration validation with helpful error messages
# 3. Clear logging of layer configuration on first dual gradient step
# 4. Detection and warning when layers overlap
# 5. Hybrid RL+SFT mode for answer tokens to prevent drift
#
# TRAINING MODES:
# 1. Standard GRPO: Single RL gradient to all layers
# 2. Dual Gradient: Thinking (RL) and Answer (RL) to separate layers
# 3. Hybrid (Approach 2): Thinking (RL only) + Answer (RL + SFT combined)
#
# HYBRID MODE (Recommended):
# Prevents answer degradation during RL by mixing supervised learning:
#   - Thinking tokens: RL only → layers 22-30 (learn from rewards)
#   - Answer tokens: RL + SFT → layers 31-36 (learn from rewards + references)
#
# CONFIGURATION EXAMPLES:
#
# Basic Dual Gradient (RL only):
# trainer:
#   use_dual_gradients: true
#   thinking_layer_start: 22
#   thinking_layer_end: 30
#   answer_layer_end: 36
#   answer_gradient_weight: 2.0
#
# Hybrid RL+SFT (Recommended):
# trainer:
#   use_dual_gradients: true
#   use_sft_on_answer: true      # Enable SFT on answer tokens
#   sft_weight: 0.1               # 10% SFT, 90% RL (prevents drift)
#   thinking_layer_start: 22
#   thinking_layer_end: 30
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
            paged_kv_cache=self.paged_kv_cache,
            run_id=self._run_id,
            current_update=update_step,
            is_invalid_batch=is_invalid_batch
        )

    def train_step(self, rollout_batch, update_step):
        """
        Execute single training step with optional dual gradient + SFT support.

        Training modes:
        1. Standard: Single RL gradient to all layers
        2. Dual gradient: Thinking (RL) + Answer (RL) to separate layers
        3. Hybrid (Approach 2): Thinking (RL only) + Answer (RL + SFT combined)

        Hybrid mode creates:
        - Thinking path: RL only → middle layers (learn from rewards)
        - Answer path: RL + SFT → final layers (learn from rewards + references)
        """
        B = rollout_batch
        start_time = time.time()

        # Check if we should use dual gradient approach
        use_dual_gradients = ('thinking_mask' in B and 'answer_mask' in B)
        use_sft_hybrid = (
            use_dual_gradients
            and hasattr(self.config.trainer, 'use_sft_on_answer')
            and self.config.trainer.use_sft_on_answer
            and 'reference_tokens' in B
        )

        if use_dual_gradients and hasattr(self.config.trainer, 'use_dual_gradients') and self.config.trainer.use_dual_gradients:
            # Dual gradient path: thinking vs answer separation
            thinking_loss, thinking_grads, answer_loss, answer_grads, metrics = \
                self.grpo_algorithm.calculate_dual_gradient_loss(B, self.config, self.tokenizer.pad_token_id)

            # Get layer ranges from config or use smart defaults
            thinking_layer_start = getattr(self.config.trainer, 'thinking_layer_start', 22)
            thinking_layer_end = getattr(self.config.trainer, 'thinking_layer_end', 30)

            # Default: answer layers start AFTER thinking layers to avoid overlap
            default_answer_start = thinking_layer_end + 1
            answer_layer_start = getattr(self.config.trainer, 'answer_layer_start', default_answer_start)
            answer_layer_end = getattr(self.config.trainer, 'answer_layer_end', 36)
            answer_gradient_weight = getattr(self.config.trainer, 'answer_gradient_weight', 2.0)

            # Validate configuration
            if thinking_layer_start < 0 or thinking_layer_end < thinking_layer_start:
                raise ValueError(f"Invalid thinking layer range: [{thinking_layer_start}, {thinking_layer_end}]")

            if answer_layer_start < 0 or answer_layer_end < answer_layer_start:
                raise ValueError(f"Invalid answer layer range: [{answer_layer_start}, {answer_layer_end}]")

            if answer_layer_end > 100:
                logger.warning(f"Answer layer end ({answer_layer_end}) seems unusually high. Verify your config.")

            # --- HYBRID SFT+RL MODE ---
            if use_sft_hybrid:
                sft_weight = getattr(self.config.trainer, 'sft_weight', 0.1)  # Default 10% SFT, 90% RL

                # Compute SFT gradients on answer tokens
                sft_loss, sft_grads, sft_metrics = self.grpo_algorithm.calculate_sft_loss_and_grads(
                    B,
                    B['reference_tokens'],
                    self.config,
                    self.tokenizer.pad_token_id
                )

                # Combine RL and SFT gradients for answer tokens
                # answer_grads already scaled by answer_gradient_weight
                answer_grads_scaled = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )

                sft_grads_scaled = tree_map(
                    lambda g: g * sft_weight / self.config.trainer.grad_accum_steps,
                    sft_grads
                )

                # Combine: answer gets both RL and SFT
                combined_answer_grads = tree_map(
                    lambda rl, sft: rl + sft,
                    answer_grads_scaled,
                    sft_grads_scaled
                )

                # Log hybrid mode on first step
                if not hasattr(self, '_hybrid_mode_logged'):
                    logger.info(f"Hybrid RL+SFT training mode enabled:")
                    logger.info(f"  Answer tokens: {answer_gradient_weight:.1f}x RL + {sft_weight:.1f}x SFT")
                    logger.info(f"  SFT helps prevent answer drift during RL")
                    self._hybrid_mode_logged = True

                metrics.update(sft_metrics)
                avg_loss = (thinking_loss.item() + answer_loss.item() + sft_loss.item()) / 3
            else:
                # Standard dual gradient (no SFT)
                combined_answer_grads = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )
                avg_loss = (thinking_loss.item() + answer_loss.item()) / 2

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

            # Mask combined answer gradients (RL or RL+SFT) to answer layers
            answer_grads_masked = mask_grads_to_layer_band(
                combined_answer_grads,
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
#-----------#
