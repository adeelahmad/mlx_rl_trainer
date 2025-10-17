"""
GRPO Trainer - ENHANCED with WandB, Memory Monitoring, Charts, and Error Handling

NEW FEATURES:
1. Comprehensive WandB tracking with custom charts
2. Token tracking (total, dual-layer, layer-wise)
3. Memory monitoring before each iteration with safety checks
4. Automatic chart generation before checkpoints
5. Checkpoint saving with retry/backoff
6. Graceful Metal error handling
7. Terminal alerts for critical events
8. Layer-wise gradient tracking and visualization

SAFETY FEATURES:
- Pre-iteration memory checks with auto-checkpoint if risky
- Graceful recovery from Metal command buffer errors
- Retry logic for failed checkpoint saves
- Memory leak detection and reporting
"""
import logging
import time
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.tuner.trainer import grad_checkpoint

from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics
from mlx_rl_trainer.utils.mlx_utils import _maybe_clip_grad_norm, mask_grads_to_layer_band
from mlx_rl_trainer.generation.generator import generate_rollouts_for_batch
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class TokenTracker:
    """Track tokens processed in training."""
    total_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    dual_gradient_tokens: int = 0
    standard_gradient_tokens: int = 0
    layer_wise_tokens: Dict[int, int] = field(default_factory=dict)

    def update(self, thinking: int, answer: int, is_dual: bool):
        """Update token counts."""
        self.thinking_tokens += thinking
        self.answer_tokens += answer
        total = thinking + answer
        self.total_tokens += total

        if is_dual:
            self.dual_gradient_tokens += total
        else:
            self.standard_gradient_tokens += total

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for logging."""
        return {
            'tokens/total': self.total_tokens,
            'tokens/thinking': self.thinking_tokens,
            'tokens/answer': self.answer_tokens,
            'tokens/dual_gradient': self.dual_gradient_tokens,
            'tokens/standard_gradient': self.standard_gradient_tokens,
            'tokens/thinking_pct': (self.thinking_tokens / max(self.total_tokens, 1)) * 100,
            'tokens/answer_pct': (self.answer_tokens / max(self.total_tokens, 1)) * 100,
        }


class MemoryMonitor:
    """Monitor memory usage and detect issues."""

    def __init__(self, safety_threshold_mb: float = 2048.0):
        self.safety_threshold_mb = safety_threshold_mb
        self.history = []
        self.max_length = 100

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics from both MLX and the system."""
        stats = {}
        try:
            stats['mlx_cache_mb'] = mx.metal.get_cache_memory() / (1024 * 1024)
            stats['mlx_active_mb'] = mx.metal.get_active_memory() / (1024 * 1024)
            stats['mlx_peak_mb'] = mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            pass

        if PSUTIL_AVAILABLE:
            mem_info = psutil.virtual_memory()
            stats['system_available_mb'] = mem_info.available / (1024 * 1024)
            stats['system_total_mb'] = mem_info.total / (1024 * 1024)
            stats['system_used_pct'] = mem_info.percent
        else:
            stats['system_available_mb'] = float('inf')

        return stats

    def check_safety(self, ref_completion_length: int = 0) -> Tuple[bool, str]:
        """
        Check if it's safe to proceed with the next iteration.
        Returns: (is_safe, reason)
        """
        stats = self.get_memory_stats()
        if not stats:
            return True, "Could not get memory stats"

        available_mb = stats.get('system_available_mb', float('inf'))

        if available_mb < self.safety_threshold_mb:
            return False, f"Low system memory: {available_mb:.1f}MB available, threshold is {self.safety_threshold_mb:.1f}MB"

        if ref_completion_length > 2000:
            return False, f"Long reference completion: {ref_completion_length} tokens"

        mlx_active = stats.get('mlx_active_mb', 0)
        if len(self.history) > 10:
            recent_mlx_active = [h.get('mlx_active_mb', 0) for h in self.history[-10:]]
            if mlx_active > np.mean(recent_mlx_active) * 1.5 and mlx_active > 1000:
                return False, f"Possible MLX memory leak: {mlx_active:.1f}MB vs avg {np.mean(recent_mlx_active):.1f}MB"

        return True, "OK"

    def record(self, stats: Dict[str, float]):
        self.history.append({**stats, 'timestamp': time.time()})
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_trend(self) -> Optional[str]:
        if len(self.history) < 20:
            return "INSUFFICIENT_DATA"

        recent = [h.get('mlx_active_mb', 0) for h in self.history[-10:]]
        older = [h.get('mlx_active_mb', 0) for h in self.history[-20:-10]]
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        change_pct = ((recent_avg - older_avg) / max(older_avg, 1)) * 100

        if change_pct > 20:
            return f"INCREASING ({change_pct:+.1f}%)"
        elif change_pct < -20:
            return f"DECREASING ({change_pct:+.1f}%)"
        else:
            return f"STABLE ({change_pct:+.1f}%)"


class MemoryMonitor0:
    """Monitor memory usage and detect issues."""

    def __init__(self, safety_threshold_mb: float = 1000.0):
        self.safety_threshold_mb = safety_threshold_mb
        self.history = []
        self.max_length = 100

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        try:
            cache_mb = mx.metal.get_cache_memory() / (1024 * 1024)
            active_mb = mx.metal.get_active_memory() / (1024 * 1024)
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)

            return {
                'cache_mb': cache_mb,
                'active_mb': active_mb,
                'peak_mb': peak_mb,
                'available_mb': self.safety_threshold_mb - active_mb,
            }
        except:
            return {}

    def check_safety(self, ref_completion_length: int = 0) -> Tuple[bool, str]:
        """
        Check if it's safe to proceed with next iteration.

        Returns:
            (is_safe, reason)
        """
        stats = self.get_memory_stats()
        if not stats:
            return True, "Could not get memory stats"

        active = stats['active_mb']
        available = stats.get('available_mb', float('inf'))

        # Check if we're close to threshold
        if available < 500:
            return False, f"Low memory: {available:.1f}MB available"

        # Check if reference completion is very long
        if ref_completion_length > 2000:
            return False, f"Long reference completion: {ref_completion_length} tokens"

        # Check for memory leak (increasing trend)
        if len(self.history) > 10:
            recent_avg = np.mean([h['active_mb'] for h in self.history[-10:]])
            if active > recent_avg * 1.5:
                return False, f"Possible memory leak: {active:.1f}MB vs avg {recent_avg:.1f}MB"

        return True, "OK"

    def record(self, stats: Dict[str, float]):
        """Record memory statistics."""
        self.history.append({**stats, 'timestamp': time.time()})
        if len(self.history) > self.max_length:
            self.history.pop(0)

    def get_trend(self) -> Optional[str]:
        """Analyze memory usage trend."""
        if len(self.history) < 10:
            return None

        recent = [h['active_mb'] for h in self.history[-10:]]
        older = [h['active_mb'] for h in self.history[-20:-10]] if len(self.history) >= 20 else None

        if older:
            recent_avg = np.mean(recent)
            older_avg = np.mean(older)
            change_pct = ((recent_avg - older_avg) / older_avg) * 100

            if change_pct > 20:
                return f"INCREASING ({change_pct:.1f}%)"
            elif change_pct < -20:
                return f"DECREASING ({change_pct:.1f}%)"
            else:
                return "STABLE"

        return "INSUFFICIENT_DATA"


def terminal_alert(message: str, level: str = "INFO"):
    """
    Try to create a terminal alert (bell/flash).

    Args:
        message: Alert message
        level: INFO, WARNING, ERROR
    """
    try:
        # Terminal bell
        sys.stdout.write('\a')
        sys.stdout.flush()

        # Color codes
        colors = {
            'INFO': '\033[94m',  # Blue
            'WARNING': '\033[93m',  # Yellow
            'ERROR': '\033[91m',  # Red
            'RESET': '\033[0m',
        }

        color = colors.get(level, colors['INFO'])
        reset = colors['RESET']

        # Print with box
        box_width = min(len(message) + 4, 80)
        print(f"\n{color}{'=' * box_width}{reset}")
        print(f"{color}  {message}{reset}")
        print(f"{color}{'=' * box_width}{reset}\n")

    except:
        pass


class GRPOTrainer(BaseTrainer):
    """Enhanced GRPO Trainer with comprehensive monitoring and safety features."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize WandB if configured
        self.wandb = None
        if hasattr(self.config, 'wandb') and self.config.wandb.enabled:
            try:
                import wandb
                self.wandb = wandb
                self._init_wandb()
            except ImportError:
                logger.warning("WandB not installed. Install with: pip install wandb")

        # Initialize trackers
        self.token_tracker = TokenTracker()
        self.memory_monitor = MemoryMonitor(
            safety_threshold_mb=getattr(self.config.trainer, 'memory_safety_threshold_mb', 1000.0)
        )

        # Chart data storage
        self.chart_data = {
            'loss_history': [],
            'reward_history': [],
            'memory_history': [],
            'token_history': [],
            'gradient_norms': {},
        }

        # Error recovery
        self.metal_error_count = 0
        self.max_metal_errors = 3

    def _init_wandb(self):
        """Initialize WandB tracking."""
        if not self.wandb:
            return

        try:
            wandb_config = {
                'model': self.config.model.model_path,
                'learning_rate': self.config.trainer.learning_rate,
                'batch_size': self.config.trainer.ppo_batch_size,
                'grad_accum_steps': self.config.trainer.grad_accum_steps,
                'use_dual_gradients': getattr(self.config.trainer, 'use_dual_gradients', False),
                'use_sft_on_answer': getattr(self.config.trainer, 'use_sft_on_answer', False),
                'thinking_layer_start': getattr(self.config.trainer, 'thinking_layer_start', 22),
                'thinking_layer_end': getattr(self.config.trainer, 'thinking_layer_end', 30),
                'answer_layer_end': getattr(self.config.trainer, 'answer_layer_end', 36),
            }

            self.wandb.init(
                project=self.config.wandb.project,
                name=self.config.wandb.run_name or self._run_id,
                config=wandb_config,
                tags=getattr(self.config.wandb, 'tags', []),
            )

            # Define custom charts
            self._define_wandb_charts()

            logger.info(f"WandB initialized: {self.wandb.run.url}")

        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.wandb = None

    def _define_wandb_charts(self):
        """Define custom WandB charts."""
        if not self.wandb or not self.wandb.run:
            return

        try:
            # Memory usage chart
            self.wandb.define_metric("memory/*", step_metric="step")

            # Token tracking chart
            self.wandb.define_metric("tokens/*", step_metric="step")

            # Loss components chart
            self.wandb.define_metric("loss/*", step_metric="step")

            # Layer-wise gradients
            self.wandb.define_metric("gradients/layer_*", step_metric="step")

            # Training metrics
            self.wandb.define_metric("training/*", step_metric="step")

            # Generation metrics
            self.wandb.define_metric("generation/*", step_metric="step")

            logger.info("WandB custom charts defined")

        except Exception as e:
            logger.warning(f"Could not define WandB charts: {e}")

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
            logger.info('Applying gradient checkpointing to transformer layers...')
            try:
                model_core = getattr(self.actor_model, 'model', self.actor_model)
                if hasattr(model_core, 'layers') and isinstance(model_core.layers, list):
                    checkpointed_count = 0
                    for layer in model_core.layers:
                        if self.config.grad_checkpoint_layers and checkpointed_count < self.config.grad_checkpoint_layers:
                            grad_checkpoint(layer)
                            checkpointed_count += 1
                    logger.info(f"Gradient checkpointing applied to {checkpointed_count} layers")
            except Exception as e:
                logger.error(f"Failed to apply gradient checkpointing: {e}", exc_info=True)

        return state.get('num_updates', 0), state.get('epoch', 0)

    def _track_layer_gradients(self, grads: Dict, step: int) -> Dict[str, float]:
        """Track gradient norms per layer."""
        layer_norms = {}

        for key, grad in tree_flatten(grads):
            if 'layers.' in key:
                # Extract layer number
                import re
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    layer_num = int(match.group(1))
                    if isinstance(grad, mx.array):
                        norm = float(mx.sqrt(mx.sum(grad ** 2)).item())
                        layer_key = f"layer_{layer_num}"
                        if layer_key not in layer_norms:
                            layer_norms[layer_key] = []
                        layer_norms[layer_key].append(norm)

        # Average norms per layer
        layer_avg_norms = {}
        for layer_key, norms in layer_norms.items():
            avg_norm = np.mean(norms)
            layer_avg_norms[f"gradients/{layer_key}_norm"] = avg_norm

            # Store for charting
            if layer_key not in self.chart_data['gradient_norms']:
                self.chart_data['gradient_norms'][layer_key] = []
            self.chart_data['gradient_norms'][layer_key].append({
                'step': step,
                'norm': avg_norm
            })

        return layer_avg_norms

    def _check_pre_iteration_safety(self, batch_data: Dict) -> Tuple[bool, str]:
        """
        Check if it's safe to proceed with next iteration.

        Returns:
            (is_safe, reason)
        """
        # Get memory stats
        mem_stats = self.memory_monitor.get_memory_stats()
        self.memory_monitor.record(mem_stats)

        # Check reference completion length
        max_ref_length = 0
        if 'prompts_data' in batch_data:
            for prompt in batch_data['prompts_data']:
                ref_text = prompt.get('ref_answer_str', '')
                ref_length = len(self.tokenizer.encode(ref_text)) if ref_text else 0
                max_ref_length = max(max_ref_length, ref_length)

        # Safety check
        is_safe, reason = self.memory_monitor.check_safety(max_ref_length)

        # Log memory trend
        trend = self.memory_monitor.get_trend()
        if trend:
            logger.debug(f"Memory trend: {trend}")

        return is_safe, reason

    def _save_checkpoint_with_retry(self, step: int, reason: str = "regular", max_retries: int = 3):
        """
        Save checkpoint with retry logic and backoff.

        Args:
            step: Current training step
            reason: Reason for checkpoint (regular, safety, final)
            max_retries: Maximum number of retry attempts
        """
        for attempt in range(max_retries):
            try:
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    model=self.actor_model,
                    optimizer=self.optimizer,
                    metadata={
                        "num_updates": step,
                        "epoch": self.current_epoch,
                        "reason": reason,
                        "log_id": self._run_id,
                        "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                        "token_stats": self.token_tracker.to_dict(),
                    },
                    current_metric=self.checkpoint_manager.best_metric,
                )

                logger.info(f"✓ Checkpoint saved successfully (reason: {reason})")
                return True

            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.error(f"Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    terminal_alert(
                        f"Checkpoint save failed! Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                        level="WARNING"
                    )
                    time.sleep(wait_time)

                    # Try to free up memory
                    gc.collect()
                    try:
                        mx.metal.clear_cache()
                    except:
                        pass
                else:
                    terminal_alert(
                        f"CRITICAL: Checkpoint save failed after {max_retries} attempts! Check disk space and permissions.",
                        level="ERROR"
                    )
                    return False

        return False

    def _generate_charts(self, step: int):
        """Generate training charts before checkpoint."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            charts_dir = Path(self.config.trainer.output_dir) / "charts"
            charts_dir.mkdir(exist_ok=True, parents=True)

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Step {step}', fontsize=16)

            # Loss history
            if self.chart_data['loss_history']:
                ax = axes[0, 0]
                steps = [d['step'] for d in self.chart_data['loss_history']]
                losses = [d['loss'] for d in self.chart_data['loss_history']]
                ax.plot(steps, losses, 'b-', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                ax.grid(True, alpha=0.3)

            # Reward history
            if self.chart_data['reward_history']:
                ax = axes[0, 1]
                steps = [d['step'] for d in self.chart_data['reward_history']]
                rewards = [d['reward'] for d in self.chart_data['reward_history']]
                ax.plot(steps, rewards, 'g-', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Reward')
                ax.set_title('Average Reward')
                ax.grid(True, alpha=0.3)

            # Memory usage
            if self.chart_data['memory_history']:
                ax = axes[1, 0]
                steps = [d['step'] for d in self.chart_data['memory_history']]
                active = [d['active_mb'] for d in self.chart_data['memory_history']]
                peak = [d['peak_mb'] for d in self.chart_data['memory_history']]
                ax.plot(steps, active, 'r-', label='Active', linewidth=2)
                ax.plot(steps, peak, 'r--', label='Peak', linewidth=1, alpha=0.5)
                ax.set_xlabel('Step')
                ax.set_ylabel('Memory (MB)')
                ax.set_title('Memory Usage')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Token distribution
            if self.chart_data['token_history']:
                ax = axes[1, 1]
                steps = [d['step'] for d in self.chart_data['token_history']]
                thinking = [d['thinking'] for d in self.chart_data['token_history']]
                answer = [d['answer'] for d in self.chart_data['token_history']]
                ax.plot(steps, thinking, 'b-', label='Thinking', linewidth=2)
                ax.plot(steps, answer, 'orange', label='Answer', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Token Count')
                ax.set_title('Token Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save chart
            chart_path = charts_dir / f"training_chart_step_{step}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Chart saved: {chart_path}")

            # Log to WandB if available
            if self.wandb and self.wandb.run:
                self.wandb.log({"training_chart": self.wandb.Image(str(chart_path))}, step=step)

        except Exception as e:
            logger.warning(f"Could not generate charts: {e}")

    def train_step(self, rollout_batch, update_step):
        """Execute training step with comprehensive tracking."""
        B = rollout_batch
        start_time = time.time()
        step_metrics = {}

        use_dual_gradients = ('thinking_mask' in B and 'answer_mask' in B)
        use_sft_hybrid = (
            use_dual_gradients
            and hasattr(self.config.trainer, 'use_sft_on_answer')
            and self.config.trainer.use_sft_on_answer
            and 'reference_tokens' in B
        )

        # Track tokens
        if use_dual_gradients:
            thinking_count = int(mx.sum(B['thinking_mask']).item())
            answer_count = int(mx.sum(B['answer_mask']).item())
            self.token_tracker.update(thinking_count, answer_count, True)
        else:
            # Estimate from response mask
            total = int(mx.sum(B.get('response_mask', mx.array([0]))).item())
            self.token_tracker.update(total // 2, total // 2, False)

        if use_dual_gradients and hasattr(self.config.trainer, 'use_dual_gradients') and self.config.trainer.use_dual_gradients:
            # Dual gradient computation
            thinking_loss, thinking_grads, answer_loss, answer_grads, metrics = \
                self.grpo_algorithm.calculate_dual_gradient_loss(B, self.config, self.tokenizer.pad_token_id)

            # Layer configuration
            thinking_layer_start = getattr(self.config.trainer, 'thinking_layer_start', 22)
            thinking_layer_end = getattr(self.config.trainer, 'thinking_layer_end', 30)
            default_answer_start = thinking_layer_end + 1
            answer_layer_start = getattr(self.config.trainer, 'answer_layer_start', default_answer_start)
            answer_layer_end = getattr(self.config.trainer, 'answer_layer_end', 36)

            # Token distribution
            thinking_token_count = mx.sum(B['thinking_mask']).item()
            answer_token_count = mx.sum(B['answer_mask']).item()
            total_tokens = thinking_token_count + answer_token_count

            if total_tokens > 0:
                thinking_ratio = thinking_token_count / total_tokens
                answer_ratio = answer_token_count / total_tokens
            else:
                thinking_ratio = 0.5
                answer_ratio = 0.5

            step_metrics.update({
                'training/thinking_token_count': thinking_token_count,
                'training/answer_token_count': answer_token_count,
                'training/thinking_ratio': thinking_ratio,
                'training/answer_ratio': answer_ratio,
            })

            # Adaptive weighting
            base_answer_weight = getattr(self.config.trainer, 'answer_gradient_weight', 2.0)
            base_sft_weight = getattr(self.config.trainer, 'sft_weight', 0.1)
            use_adaptive_weights = getattr(self.config.trainer, 'adaptive_gradient_weights', True)

            step_metrics['training/answer_weight_base'] = base_answer_weight
            step_metrics['training/sft_weight_base'] = base_sft_weight

            if use_adaptive_weights and total_tokens < 200:
                if thinking_ratio > 0.7:
                    answer_gradient_weight = base_answer_weight * (1.0 / max(answer_ratio, 0.1))
                    answer_gradient_weight = min(answer_gradient_weight, base_answer_weight * 4.0)
                    sft_weight = base_sft_weight * (1.0 / max(answer_ratio, 0.2))
                    sft_weight = min(sft_weight, base_sft_weight * 3.0)

                    step_metrics.update({
                        'training/adaptive_weights_active': 1.0,
                        'training/answer_weight_boost_ratio': answer_gradient_weight / base_answer_weight,
                        'training/sft_weight_boost_ratio': sft_weight / base_sft_weight,
                    })
                else:
                    answer_gradient_weight = base_answer_weight
                    sft_weight = base_sft_weight
                    step_metrics.update({
                        'training/adaptive_weights_active': 0.0,
                        'training/answer_weight_boost_ratio': 1.0,
                        'training/sft_weight_boost_ratio': 1.0,
                    })
            else:
                answer_gradient_weight = base_answer_weight
                sft_weight = base_sft_weight
                step_metrics.update({
                    'training/adaptive_weights_active': 0.0,
                    'training/answer_weight_boost_ratio': 1.0,
                    'training/sft_weight_boost_ratio': 1.0,
                })

            step_metrics['training/answer_weight_actual'] = answer_gradient_weight
            step_metrics['training/sft_weight_actual'] = sft_weight

            # SFT hybrid mode
            if use_sft_hybrid:
                sft_loss, sft_grads, sft_metrics = self.grpo_algorithm.calculate_sft_loss_and_grads(
                    B, B['reference_tokens'], self.config, self.tokenizer.pad_token_id
                )

                answer_grads_scaled = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )
                sft_grads_scaled = tree_map(
                    lambda g: g * sft_weight / self.config.trainer.grad_accum_steps,
                    sft_grads
                )
                combined_answer_grads = tree_map(lambda rl, sft: rl + sft, answer_grads_scaled, sft_grads_scaled)

                metrics.update(sft_metrics)

                step_metrics.update({
                    'loss/thinking_loss': thinking_loss.item(),
                    'loss/answer_rl_loss': answer_loss.item(),
                    'loss/answer_sft_loss': sft_loss.item(),
                    'loss/total': (thinking_loss.item() + answer_loss.item() + sft_loss.item()) / 3,
                })

                total_loss = step_metrics['loss/total']
                if total_loss > 0:
                    step_metrics.update({
                        'loss/thinking_contribution_pct': (thinking_loss.item() / total_loss) * 100,
                        'loss/answer_rl_contribution_pct': (answer_loss.item() / total_loss) * 100,
                        'loss/answer_sft_contribution_pct': (sft_loss.item() / total_loss) * 100,
                    })

                avg_loss = step_metrics['loss/total']
            else:
                combined_answer_grads = tree_map(
                    lambda g: g * answer_gradient_weight / self.config.trainer.grad_accum_steps,
                    answer_grads
                )

                step_metrics.update({
                    'loss/thinking_loss': thinking_loss.item(),
                    'loss/answer_rl_loss': answer_loss.item(),
                    'loss/total': (thinking_loss.item() + answer_loss.item()) / 2,
                })
                avg_loss = step_metrics['loss/total']

            # Mask and combine gradients
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
            combined_grads = tree_map(lambda t, a: t + a, thinking_grads_masked, answer_grads_masked)

        else:
            # Standard gradient
            loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
                B, self.config, self.tokenizer.pad_token_id
            )
            combined_grads = tree_map(
                lambda g: g / self.config.trainer.grad_accum_steps,
                grads
            )
            avg_loss = loss.item()
            step_metrics['loss/total'] = avg_loss

        # Track layer-wise gradients
        layer_grad_norms = self._track_layer_gradients(combined_grads, update_step)
        step_metrics.update(layer_grad_norms)

        # Common metrics
        step_metrics.update({
            'training/reward_mean': B['advantages'].mean().item(),
            'training/reward_std': B['advantages'].std().item(),
            'training/learning_rate': self.lr_scheduler(update_step),
            'training/kl_divergence': metrics.get('kl_divergence', 0.0),
            'training/step_time_s': time.time() - start_time,
        })

        # Store for charting
        self.chart_data['loss_history'].append({
            'step': update_step,
            'loss': avg_loss
        })
        self.chart_data['reward_history'].append({
            'step': update_step,
            'reward': step_metrics['training/reward_mean']
        })
        if use_dual_gradients:
            self.chart_data['token_history'].append({
                'step': update_step,
                'thinking': step_metrics['training/thinking_token_count'],
                'answer': step_metrics['training/answer_token_count']
            })

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
        """Generate rollouts with error handling."""
        try:
            prompts_data = batch_data.get('prompts_data', [])
            is_invalid_batch = any(p.get('is_invalid_sample', False) for p in prompts_data)

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

        except RuntimeError as e:
            if "METAL" in str(e) or "Command buffer" in str(e):
                self.metal_error_count += 1
                logger.error(f"Metal error in generation ({self.metal_error_count}/{self.max_metal_errors}): {e}")

                # Try to recover
                gc.collect()
                try:
                    mx.metal.clear_cache()
                except:
                    pass

                if self.metal_error_count >= self.max_metal_errors:
                    terminal_alert("CRITICAL: Multiple Metal errors. Saving checkpoint and exiting.", level="ERROR")
                    self._save_checkpoint_with_retry(update_step, reason="metal_error")
                    raise

                # Return empty batch to skip this iteration
                return {}, 0.0, {}, {}
            else:
                raise

    def log_comprehensive_metrics(self, step: int, step_metrics: Dict, generation_metrics: Dict = None):
        """Log all metrics to WandB and chart data."""
        # Memory stats
        mem_stats = self.memory_monitor.get_memory_stats()
        if mem_stats:
            mem_metrics = {
                f'memory/{k}': v for k, v in mem_stats.items()
            }
            step_metrics.update(mem_metrics)

            self.chart_data['memory_history'].append({
                'step': step,
                **mem_stats
            })

        # Token stats
        token_metrics = self.token_tracker.to_dict()
        step_metrics.update(token_metrics)

        # Add generation metrics
        if generation_metrics:
            step_metrics.update(generation_metrics)

        # Log to WandB
        if self.wandb and self.wandb.run:
            self.wandb.log({**step_metrics, 'step': step})

        # Log to metrics logger if available
        if self.metrics_logger:
            self.metrics_logger.log_metrics(step_metrics, step=step)

    async def run(self, should_shutdown):
        """Enhanced training loop with safety checks and error handling."""
        resumed_step, self.current_epoch = self._setup()

        if resumed_step > 0:
            self.global_step = resumed_step + 1
            logger.info(f"Resumed from checkpoint at step {resumed_step}")
        else:
            self.global_step = 0
            logger.info("Starting training from scratch")

        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)

        await self.data_manager.load_datasets()

        from tqdm import trange
        pbar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc="Training",
            unit="step",
        )

        train_data_iterator = iter([])
        grad_accum_steps = self.config.trainer.grad_accum_steps

        with pbar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    logger.info("Shutdown requested")
                    break

                try:
                    # Get batch
                    try:
                        batch_data = next(train_data_iterator)
                    except StopIteration:
                        self.current_epoch += 1
                        logger.info(f"Epoch {self.current_epoch}")
                        train_data_iterator = iter(
                            self.data_manager.get_dataloader("train", self.config.trainer.ppo_batch_size)
                        )
                        batch_data = next(train_data_iterator)

                    # PRE-ITERATION SAFETY CHECK
                    is_safe, reason = self._check_pre_iteration_safety(batch_data)
                    if not is_safe:
                        logger.warning(f"Safety check failed: {reason}")
                        terminal_alert(f"Safety checkpoint triggered: {reason}", level="WARNING")

                        # Save safety checkpoint
                        self._save_checkpoint_with_retry(self.global_step, reason="safety")

                        # Aggressive cleanup
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass

                        # Wait before continuing
                        time.sleep(2)

                    # Gradient accumulation
                    accum_grads = None
                    sum_loss = 0.0
                    sum_reward = 0.0
                    count_microbatches = 0
                    aggregated_raw_rewards = {}

                    for accum_idx in range(grad_accum_steps):
                        # Generate rollouts
                        rollout_batch, avg_reward_mb, raw_reward_components_mb, generation_metrics = \
                            self.generate_rollouts(batch_data, self.global_step)

                        if not rollout_batch or "tokens" not in rollout_batch:
                            logger.warning(f"Empty rollout at step {self.global_step}")
                            continue

                        # Train step
                        metrics_mb, grads_mb, step_metrics = self.train_step(rollout_batch, self.global_step)

                        sum_loss += metrics_mb.loss
                        sum_reward += avg_reward_mb
                        count_microbatches += 1

                        if raw_reward_components_mb:
                            for k, v in raw_reward_components_mb.items():
                                aggregated_raw_rewards[k] = aggregated_raw_rewards.get(k, 0.0) + v

                        # Accumulate gradients
                        if grads_mb:
                            if accum_grads is None:
                                accum_grads = grads_mb
                            else:
                                accum_grads = tree_map(mx.add, accum_grads, grads_mb)

                        # Cleanup
                        del rollout_batch, metrics_mb, grads_mb
                        gc.collect()

                    # Apply gradients
                    if accum_grads and count_microbatches > 0:
                        self.optimizer.learning_rate = self.lr_scheduler(self.global_step)
                        self.optimizer.apply_gradients(accum_grads, self.actor_model.trainable_parameters())
                        mx.eval(self.actor_model.parameters())

                        # Compute averages
                        avg_loss = sum_loss / count_microbatches
                        avg_reward = sum_reward / count_microbatches

                        # Log comprehensive metrics
                        self.log_comprehensive_metrics(
                            self.global_step,
                            step_metrics,
                            generation_metrics
                        )

                        # Update progress
                        pbar.set_postfix({
                            'Loss': f'{avg_loss:.4f}',
                            'Reward': f'{avg_reward:.3f}',
                            'Tokens': f'{self.token_tracker.total_tokens}',
                        })
                        pbar.update(1)

                        # Checkpointing with charts
                        is_checkpoint = (
                            self.config.checkpointing.save_every > 0 and
                            (self.global_step + 1) % self.config.checkpointing.save_every == 0
                        )

                        if is_checkpoint:
                            # Generate charts
                            self._generate_charts(self.global_step)

                            # Save checkpoint
                            self._save_checkpoint_with_retry(self.global_step, reason="regular")

                    self.global_step += 1

                except RuntimeError as e:
                    if "METAL" in str(e) or "Command buffer" in str(e):
                        logger.error(f"Metal command buffer error at step {self.global_step}: {e}")
                        terminal_alert("Metal error encountered! Saving checkpoint...", level="ERROR")

                        # Save emergency checkpoint
                        self._save_checkpoint_with_retry(self.global_step, reason="metal_error")

                        # Aggressive recovery
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass

                        # Reset error counter if we've made progress
                        if self.global_step % 10 == 0:
                            self.metal_error_count = 0

                        # Continue if under error limit
                        if self.metal_error_count < self.max_metal_errors:
                            logger.info("Attempting to continue training...")
                            time.sleep(5)
                            continue
                        else:
                            logger.error("Too many Metal errors. Exiting.")
                            break
                    else:
                        raise

        # Final checkpoint
        self._generate_charts(self.global_step)
        self._save_checkpoint_with_retry(self.global_step, reason="final")

        # Close WandB
        if self.wandb and self.wandb.run:
            self.wandb.finish()

    def evaluate(self, update_step):
        """Placeholder for evaluation."""
        logger.info(f"Evaluation at step {update_step}")
        return []
