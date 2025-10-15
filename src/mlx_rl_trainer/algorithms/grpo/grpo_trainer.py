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


    def train_step(self, rollout_batch, update_step):
        """
        Execute training step + collect comprehensive metrics.

        Returns:
            (training_metrics, combined_grads, step_metrics_dict)

        NEW: step_metrics_dict contains all metrics for WandB logging
        """
        B = rollout_batch
        start_time = time.time()

        # Initialize metrics dictionary for this step
        step_metrics = {}

        use_dual_gradients = ('thinking_mask' in B and 'answer_mask' in B)
        use_sft_hybrid = (
            use_dual_gradients
            and hasattr(self.config.trainer, 'use_sft_on_answer')
            and self.config.trainer.use_sft_on_answer
            and 'reference_tokens' in B
        )

        if use_dual_gradients and hasattr(self.config.trainer, 'use_dual_gradients') and self.config.trainer.use_dual_gradients:
            # Dual gradient path
            thinking_loss, thinking_grads, answer_loss, answer_grads, metrics = \
                self.grpo_algorithm.calculate_dual_gradient_loss(B, self.config, self.tokenizer.pad_token_id)

            # Layer configuration
            thinking_layer_start = getattr(self.config.trainer, 'thinking_layer_start', 22)
            thinking_layer_end = getattr(self.config.trainer, 'thinking_layer_end', 30)
            default_answer_start = thinking_layer_end + 1
            answer_layer_start = getattr(self.config.trainer, 'answer_layer_start', default_answer_start)
            answer_layer_end = getattr(self.config.trainer, 'answer_layer_end', 36)

            # ADAPTIVE WEIGHT BALANCING with metrics tracking
            thinking_token_count = mx.sum(B['thinking_mask']).item()
            answer_token_count = mx.sum(B['answer_mask']).item()
            total_tokens = thinking_token_count + answer_token_count

            if total_tokens > 0:
                thinking_ratio = thinking_token_count / total_tokens
                answer_ratio = answer_token_count / total_tokens
            else:
                thinking_ratio = 0.5
                answer_ratio = 0.5

            # Record token distribution
            step_metrics['training/thinking_token_count'] = thinking_token_count
            step_metrics['training/answer_token_count'] = answer_token_count
            step_metrics['training/thinking_ratio'] = thinking_ratio
            step_metrics['training/answer_ratio'] = answer_ratio

            base_answer_weight = getattr(self.config.trainer, 'answer_gradient_weight', 2.0)
            base_sft_weight = getattr(self.config.trainer, 'sft_weight', 0.1)
            use_adaptive_weights = getattr(self.config.trainer, 'adaptive_gradient_weights', True)

            # Track base weights
            step_metrics['training/answer_weight_base'] = base_answer_weight
            step_metrics['training/sft_weight_base'] = base_sft_weight

            if use_adaptive_weights and total_tokens < 200:
                if thinking_ratio > 0.7:
                    # Adaptive boosting
                    answer_gradient_weight = base_answer_weight * (1.0 / max(answer_ratio, 0.1))
                    answer_gradient_weight = min(answer_gradient_weight, base_answer_weight * 4.0)

                    sft_weight = base_sft_weight * (1.0 / max(answer_ratio, 0.2))
                    sft_weight = min(sft_weight, base_sft_weight * 3.0)

                    # Record adaptive adjustment
                    step_metrics['training/adaptive_weights_active'] = 1.0
                    step_metrics['training/answer_weight_boost_ratio'] = answer_gradient_weight / base_answer_weight
                    step_metrics['training/sft_weight_boost_ratio'] = sft_weight / base_sft_weight

                    if not hasattr(self, '_adaptive_weights_logged'):
                        logger.info(f"Adaptive weights activated:")
                        logger.info(f"  Thinking: {thinking_token_count:.0f} tokens ({thinking_ratio*100:.1f}%)")
                        logger.info(f"  Answer: {answer_token_count:.0f} tokens ({answer_ratio*100:.1f}%)")
                        logger.info(f"  Boosted answer: {base_answer_weight:.1f} → {answer_gradient_weight:.1f}")
                        logger.info(f"  Boosted SFT: {base_sft_weight:.2f} → {sft_weight:.2f}")
                        self._adaptive_weights_logged = True
                else:
                    answer_gradient_weight = base_answer_weight
                    sft_weight = base_sft_weight
                    step_metrics['training/adaptive_weights_active'] = 0.0
                    step_metrics['training/answer_weight_boost_ratio'] = 1.0
                    step_metrics['training/sft_weight_boost_ratio'] = 1.0
            else:
                answer_gradient_weight = base_answer_weight
                sft_weight = base_sft_weight
                step_metrics['training/adaptive_weights_active'] = 0.0
                step_metrics['training/answer_weight_boost_ratio'] = 1.0
                step_metrics['training/sft_weight_boost_ratio'] = 1.0

            # Record actual weights used
            step_metrics['training/answer_weight_actual'] = answer_gradient_weight
            step_metrics['training/sft_weight_actual'] = sft_weight

            # Validate configuration (unchanged)
            if thinking_layer_start < 0 or thinking_layer_end < thinking_layer_start:
                raise ValueError(f"Invalid thinking layer range")
            if answer_layer_start < 0 or answer_layer_end < answer_layer_start:
                raise ValueError(f"Invalid answer layer range")

            # --- HYBRID SFT+RL MODE ---
            if use_sft_hybrid:
                sft_loss, sft_grads, sft_metrics = self.grpo_algorithm.calculate_sft_loss_and_grads(
                    B, B['reference_tokens'], self.config, self.tokenizer.pad_token_id
                )

                # Scale gradients
                answer_grads_scaled = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )
                sft_grads_scaled = tree_map(
                    lambda g: g * sft_weight / self.config.trainer.grad_accum_steps,
                    sft_grads
                )
                combined_answer_grads = tree_map(
                    lambda rl, sft: rl + sft,
                    answer_grads_scaled,
                    sft_grads_scaled
                )

                # Log hybrid mode
                if not hasattr(self, '_hybrid_mode_logged'):
                    logger.info(f"Hybrid RL+SFT training enabled:")
                    logger.info(f"  Answer: {answer_gradient_weight:.1f}x RL + {sft_weight:.1f}x SFT")
                    self._hybrid_mode_logged = True

                metrics.update(sft_metrics)

                # Record loss components
                step_metrics['loss/thinking_loss'] = thinking_loss.item()
                step_metrics['loss/answer_rl_loss'] = answer_loss.item()
                step_metrics['loss/answer_sft_loss'] = sft_loss.item()
                step_metrics['loss/total'] = (thinking_loss.item() + answer_loss.item() + sft_loss.item()) / 3

                # Calculate contribution percentages
                total_loss = step_metrics['loss/total']
                if total_loss > 0:
                    step_metrics['loss/thinking_contribution_pct'] = (thinking_loss.item() / total_loss) * 100
                    step_metrics['loss/answer_rl_contribution_pct'] = (answer_loss.item() / total_loss) * 100
                    step_metrics['loss/answer_sft_contribution_pct'] = (sft_loss.item() / total_loss) * 100

                avg_loss = step_metrics['loss/total']
            else:
                # Standard dual gradient (no SFT)
                combined_answer_grads = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )

                step_metrics['loss/thinking_loss'] = thinking_loss.item()
                step_metrics['loss/answer_rl_loss'] = answer_loss.item()
                step_metrics['loss/total'] = (thinking_loss.item() + answer_loss.item()) / 2
                avg_loss = step_metrics['loss/total']

            # Log layer configuration once
            if not hasattr(self, '_dual_grad_config_logged'):
                overlap = set(range(thinking_layer_start, thinking_layer_end + 1)) & \
                         set(range(answer_layer_start, answer_layer_end + 1))
                if overlap:
                    logger.info(f"Dual gradient mode - OVERLAPPING layers:")
                    logger.info(f"  Thinking: {thinking_layer_start}-{thinking_layer_end} (1x)")
                    logger.info(f"  Answer: {answer_layer_start}-{answer_layer_end} ({answer_gradient_weight}x)")
                    logger.info(f"  Overlap: {sorted(overlap)} (total {1 + answer_gradient_weight}x)")
                else:
                    logger.info(f"Dual gradient mode - SEPARATE pathways:")
                    logger.info(f"  Thinking: {thinking_layer_start}-{thinking_layer_end} (1x)")
                    logger.info(f"  Answer: {answer_layer_start}-{answer_layer_end} ({answer_gradient_weight}x)")
                self._dual_grad_config_logged = True

            # Mask and combine gradients (unchanged)
            thinking_grads_scaled = tree_map(
                lambda g: g / self.config.trainer.grad_accum_steps,
                thinking_grads
            )
            thinking_grads_masked = mask_grads_to_layer_band(
                thinking_grads_scaled,
                start=thinking_layer_start,
                end=thinking_layer_end,
                include_embed=False,
                include_head=False
            )
            answer_grads_masked = mask_grads_to_layer_band(
                combined_answer_grads,
                start=answer_layer_start,
                end=answer_layer_end,
                include_embed=False,
                include_head=True
            )
            combined_grads = tree_map(
                lambda t, a: t + a,
                thinking_grads_masked,
                answer_grads_masked
            )

        else:
            # Standard gradient path
            loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
                B, self.config, self.tokenizer.pad_token_id
            )
            combined_grads = tree_map(
                lambda g: g / self.config.trainer.grad_accum_steps,
                grads
            )
            avg_loss = loss.item()
            step_metrics['loss/total'] = avg_loss

        # Add common metrics
        step_metrics['training/reward_mean'] = B['advantages'].mean().item()
        step_metrics['training/reward_std'] = B['advantages'].std().item()
        step_metrics['training/learning_rate'] = self.lr_scheduler(update_step)
        step_metrics['training/kl_divergence'] = metrics.get('kl_divergence', 0.0)
        step_metrics['training/step_time_s'] = time.time() - start_time

        # Create training metrics object (for backward compatibility)
        training_metrics = TrainingMetrics(
            loss=avg_loss,
            reward_mean=step_metrics['training/reward_mean'],
            reward_std=step_metrics['training/reward_std'],
            grad_norm=0.0,
            learning_rate=step_metrics['training/learning_rate'],
            step_time_s=step_metrics['training/step_time_s'],
            kl_divergence=step_metrics['training/kl_divergence'],
            epoch=self.current_epoch,
            step=update_step
        )

        return training_metrics, combined_grads, step_metrics


    def generate_rollouts(self, batch_data, update_step):
        """
        Generate rollouts - now returns generation_metrics too!
        """
        prompts_data = batch_data.get('prompts_data', [])
        is_invalid_batch = any(p.get('is_invalid_sample', False) for p in prompts_data)

        # Unpack 4 values now (not 3!)
        rollout_batch, avg_reward, avg_breakdown, generation_metrics = generate_rollouts_for_batch(
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

        return rollout_batch, avg_reward, avg_breakdown, generation_metrics


    def wandb_log(self, step, step_metrics, generation_metrics=None):
        """
        Comprehensive WandB logging for hardware-constrained training.

        Args:
            step: Training step number
            step_metrics: Metrics from train_step
            generation_metrics: Metrics from generate_rollouts (optional)
        """
        if not hasattr(self, 'wandb') or self.wandb is None:
            return

        combined_metrics = {'step': step, **step_metrics}

        # Add generation metrics if available
        if generation_metrics:
            combined_metrics.update(generation_metrics)

        # Log to wandb
        self.wandb.log(combined_metrics)

        # Special alerts for concerning patterns
        if 'generation/thinking_answer_ratio' in combined_metrics:
            ratio = combined_metrics['generation/thinking_answer_ratio']
            if ratio > 5.0:
                logger.warning(f"⚠️ CRITICAL IMBALANCE: Thinking/answer ratio is {ratio:.2f}:1")

        if 'generation/missing_answer_count' in combined_metrics:
            missing = combined_metrics['generation/missing_answer_count']
            if missing > 0:
                logger.warning(f"⚠️ {missing} samples missing answer section")

        if 'training/adaptive_weights_active' in combined_metrics:
            if combined_metrics['training/adaptive_weights_active'] > 0:
                logger.debug(
                    f"✓ Adaptive weights: answer {combined_metrics['training/answer_weight_boost_ratio']:.2f}x, "
                    f"SFT {combined_metrics['training/sft_weight_boost_ratio']:.2f}x"
                )



    def evaluate(self, update_step):
        """Placeholder for evaluation logic."""
        logger.info(f"Evaluation at step {update_step} is a placeholder.")
        return []

# End of File: ./grpo_trainer.py
#</FILE>
#-----------#
