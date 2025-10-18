"""GRPO Trainer with Advanced Memory and Performance Optimizations.

This trainer implements Group Relative Policy Optimization with:

1. **MLX Graph Compilation** (shapeless=True):
   - Loss functions are compiled for 2-5x speedup and better memory efficiency
   - First call compiles, subsequent calls use cached compiled version
   - Set config.trainer.use_compile=False to disable for debugging
   
2. **Aggressive Memory Management**:
   - Explicit cleanup with del and gc.collect()
   - Strategic mx.eval() to force execution and free intermediate memory
   - Periodic mx.metal.clear_cache() every 10 steps
   - Limited history buffers (50-100 entries max)
   
3. **Memory Monitoring**:
   - Real-time MLX and system memory tracking
   - Safety checks before expensive operations
   - Automatic trend analysis to detect memory leaks
   
4. **Gradient Efficiency**:
   - In-place operations where safe
   - Immediate deletion of intermediate gradients
   - Efficient tree operations with minimal allocations

Performance Tips:
- First training step will be slow (compilation)
- Subsequent steps will be much faster
- For debugging array values, set use_compile=False
- Monitor memory stats in logs/wandb for optimal batch sizing
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
import re

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
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed. Memory safety checks will be less accurate. "
                   "Run 'pip install psutil'.")


def safe_tree_add(tree1, tree2):
    """Safely add two gradient trees with minimal memory overhead."""
    if not tree1:
        return tree2
    if not tree2:
        return tree1
    
    flat1 = tree_flatten(tree1)
    flat2_dict = dict(tree_flatten(tree2))
    
    result = []
    for path, grad1 in flat1:
        if path in flat2_dict:
            result.append((path, grad1 + flat2_dict.pop(path)))
        else:
            result.append((path, grad1))
    
    result.extend(flat2_dict.items())
    
    del flat1, flat2_dict
    
    return tree_unflatten(result)


@dataclass
class TokenTracker:
    """Memory-efficient token tracking with minimal state."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        total = max(self.total_tokens, 1)
        return {
            'tokens/total': self.total_tokens,
            'tokens/thinking': self.thinking_tokens,
            'tokens/answer': self.answer_tokens,
            'tokens/dual_gradient': self.dual_gradient_tokens,
            'tokens/standard_gradient': self.standard_gradient_tokens,
            'tokens/thinking_pct': self.thinking_tokens / total * 100,
            'tokens/answer_pct': self.answer_tokens / total * 100
        }


class MemoryMonitor:
    """Memory monitoring with efficient history management."""
    
    def __init__(self, safety_threshold_mb: float = 2048.0):
        self.safety_threshold_mb = safety_threshold_mb
        self.history = []
        self.max_length = 50  # Reduced from 100 for memory efficiency
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        try:
            stats['mlx_cache_mb'] = mx.get_cache_memory() / 1_048_576
            stats['mlx_active_mb'] = mx.get_active_memory() / 1_048_576
            stats['mlx_peak_mb'] = mx.get_peak_memory() / 1_048_576
        except Exception as e:
            logger.debug(f"Could not get MLX memory stats: {e}")
        
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            stats['system_available_mb'] = mem.available / 1_048_576
            stats['system_total_mb'] = mem.total / 1_048_576
            stats['system_used_pct'] = mem.percent
        else:
            stats['system_available_mb'] = float('inf')
        
        stats['active_mb'] = stats.get('mlx_active_mb', 0.0)
        
        return stats
    
    def check_safety(self, ref_completion_length: int = 0) -> Tuple[bool, str]:
        """Check if training is safe to continue."""
        stats = self.get_memory_stats()
        self.record(stats)
        
        if not stats:
            return True, 'Could not get memory stats'
        
        available = stats.get('system_available_mb', float('inf'))
        if available < self.safety_threshold_mb:
            return False, (f"Low system memory: {available:.1f}MB available, "
                          f"threshold is {self.safety_threshold_mb:.1f}MB")
        
        if ref_completion_length > 2000:
            return False, f"Long reference completion: {ref_completion_length} tokens"
        
        trend = self.get_trend()
        cache = stats.get('mlx_cache_mb', 0)
        if trend and trend.startswith('INCREASING') and cache > 1024:
            return False, f"Possible MLX memory leak: Cache memory is {trend}"
        
        return True, 'OK'
    
    def record(self, stats: Dict):
        """Record memory stats with automatic cleanup."""
        if stats:
            self.history.append({**stats, 'timestamp': time.time()})
            if len(self.history) > self.max_length:
                del self.history[0]
    
    def get_trend(self) -> str:
        """Analyze memory usage trend."""
        if len(self.history) < 20:
            return 'INSUFFICIENT_DATA'
        
        metric = 'mlx_cache_mb'
        if not all(metric in entry for entry in self.history[-20:]):
            return 'INSUFFICIENT_DATA'
        
        recent = [entry.get(metric, 0) for entry in self.history[-10:]]
        older = [entry.get(metric, 0) for entry in self.history[-20:-10]]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        base = max(older_avg, 1.0)
        change_pct = (recent_avg - older_avg) / base * 100
        
        if change_pct > 20:
            return f"INCREASING ({change_pct:+.1f}%)"
        elif change_pct < -20:
            return f"DECREASING ({change_pct:+.1f}%)"
        else:
            return f"STABLE ({change_pct:+.1f}%)"


def terminal_alert(message: str, level: str = 'INFO'):
    """Print a terminal alert with color."""
    try:
        sys.stdout.write('\a')
        sys.stdout.flush()
        
        colors = {
            'INFO': '\033[94m',
            'WARNING': '\033[93m',
            'ERROR': '\033[91m',
            'RESET': '\033[0m'
        }
        
        color = colors.get(level, colors['INFO'])
        reset = colors['RESET']
        width = min(len(message) + 4, 80)
        
        print(f"\n{color}{'=' * width}{reset}")
        print(f"{color}  {message}{reset}")
        print(f"{color}{'=' * width}{reset}\n")
    except:
        pass


class GRPOTrainer(BaseTrainer):
    """GRPO Trainer with aggressive memory optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.wandb = None
        if hasattr(self.config, 'wandb') and self.config.wandb.enabled:
            try:
                import wandb
                self.wandb = wandb
                self._init_wandb()
            except ImportError:
                logger.warning('WandB not installed. Install with: pip install wandb')
        
        self.token_tracker = TokenTracker()
        self.memory_monitor = MemoryMonitor(
            safety_threshold_mb=getattr(self.config.trainer, 'memory_safety_threshold_mb', 1000)
        )
        
        # Minimal chart data with aggressive limits
        self.chart_data = {
            'loss_history': [],
            'reward_history': [],
            'memory_history': [],
            'token_history': [],
            'gradient_norms': {}
        }
        self._max_history = 100  # Limit all histories
        
        self.metal_error_count = 0
        self.max_metal_errors = 3
    
    def _init_wandb(self):
        """Initialize WandB logging."""
        if not self.wandb:
            return
        
        try:
            config_dict = {
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
                config=config_dict,
                tags=getattr(self.config.wandb, 'tags', [])
            )
            
            self._define_wandb_charts()
            logger.info(f"WandB initialized: {self.wandb.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.wandb = None
    
    def _define_wandb_charts(self):
        """Define WandB custom charts."""
        if not self.wandb or not self.wandb.run:
            return
        
        try:
            self.wandb.define_metric('memory/*', step_metric='step')
            self.wandb.define_metric('tokens/*', step_metric='step')
            self.wandb.define_metric('loss/*', step_metric='step')
            self.wandb.define_metric('gradients/layer_*', step_metric='step')
            self.wandb.define_metric('training/*', step_metric='step')
            self.wandb.define_metric('generation/*', step_metric='step')
            
            logger.info('WandB custom charts defined')
        except Exception as e:
            logger.warning(f"Could not define WandB charts: {e}")
    
    def _setup(self) -> Tuple[int, int]:
        """Setup models and training state."""
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
        
        # Initialize GRPO algorithm with compilation support
        # Note: Set config.trainer.use_compile = False to disable compilation for debugging
        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)
        
        if getattr(self.config.trainer, 'use_compile', True):
            logger.info("MLX graph compilation enabled - first iteration will compile functions")
            logger.info("Compilation provides ~2-5x speedup and better memory efficiency")
            logger.info("Set config.trainer.use_compile=False to disable for debugging")
        
        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2),
            weight_decay=self.config.trainer.optimizer_weight_decay
        )
        
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)
        
        checkpoint_loaded, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )
        
        if self.config.use_grad_checkpointing:
            logger.info('Applying gradient checkpointing to transformer layers...')
            try:
                model = getattr(self.actor_model, 'model', self.actor_model)
                if hasattr(model, 'layers') and isinstance(model.layers, list):
                    count = 0
                    for layer in model.layers:
                        if self.config.grad_checkpoint_layers and count < self.config.grad_checkpoint_layers:
                            grad_checkpoint(layer)
                            count += 1
                    logger.info(f"Gradient checkpointing applied to {count} layers")
            except Exception as e:
                logger.error(f"Failed to apply gradient checkpointing: {e}", exc_info=True)
        
        return metadata.get('num_updates', 0), metadata.get('epoch', 0)
    
    def _track_layer_gradients(self, grads, step: int) -> Dict[str, float]:
        """Track gradient norms by layer with memory-efficient aggregation."""
        layer_grads = {}
        
        for path, grad in tree_flatten(grads):
            if 'layers.' in path:
                match = re.search(r'layers\.(\d+)\.', path)
                if match:
                    layer_num = int(match.group(1))
                    if isinstance(grad, mx.array):
                        norm = float(mx.sqrt(mx.sum(grad ** 2)).item())
                        
                        key = f"layer_{layer_num}"
                        if key not in layer_grads:
                            layer_grads[key] = []
                        layer_grads[key].append(norm)
        
        metrics = {}
        for key, norms in layer_grads.items():
            avg_norm = np.mean(norms)
            metrics[f"gradients/{key}_norm"] = avg_norm
            
            # Store in history with limit
            if key not in self.chart_data['gradient_norms']:
                self.chart_data['gradient_norms'][key] = []
            
            history = self.chart_data['gradient_norms'][key]
            history.append({'step': step, 'norm': avg_norm})
            
            # Trim history
            if len(history) > self._max_history:
                del history[0]
        
        del layer_grads
        return metrics
    
    def _check_pre_iteration_safety(self, batch_data: Dict) -> Tuple[bool, str]:
        """Check safety before iteration with minimal overhead."""
        mem_stats = self.memory_monitor.get_memory_stats()
        self.memory_monitor.record(mem_stats)
        
        max_ref_length = 0
        if 'prompts_data' in batch_data:
            for prompt_data in batch_data['prompts_data']:
                ref_str = prompt_data.get('ref_answer_str', '')
                if ref_str:
                    ref_length = len(self.tokenizer.encode(ref_str))
                    max_ref_length = max(max_ref_length, ref_length)
        
        is_safe, message = self.memory_monitor.check_safety(max_ref_length)
        
        trend = self.memory_monitor.get_trend()
        if trend:
            logger.debug(f"Memory trend: {trend}")
        
        return is_safe, message
    
    def _save_checkpoint_with_retry(self, step: int, reason: str = 'regular', 
                                   is_final: bool = False, max_retries: int = 3) -> bool:
        """Save checkpoint with retry logic and memory cleanup."""
        for attempt in range(max_retries):
            try:
                self.checkpoint_manager.save_checkpoint(
                    step=step,
                    model=self.actor_model,
                    optimizer=self.optimizer,
                    metadata={
                        'num_updates': step,
                        'epoch': self.current_epoch,
                        'reason': reason,
                        'log_id': self._run_id,
                        'save_optimizer_state': self.config.checkpointing.save_optimizer_state,
                        'token_stats': self.token_tracker.to_dict()
                    },
                    current_metric=self.checkpoint_manager.best_metric
                )
                
                logger.info(f"✓ Checkpoint saved successfully (reason: {reason})")
                return True
                
            except Exception as e:
                wait_time = 2 ** attempt
                logger.error(f"Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait_time}s...")
                    terminal_alert(
                        f"Checkpoint save failed! Retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})",
                        level='WARNING'
                    )
                    time.sleep(wait_time)
                    
                    gc.collect()
                    try:
                        mx.metal.clear_cache()
                    except:
                        pass
                else:
                    terminal_alert(
                        f"CRITICAL: Checkpoint save failed after {max_retries} attempts! "
                        "Check disk space and permissions.",
                        level='ERROR'
                    )
                    return False
        
        return False
    
    def _generate_charts(self, step: int):
        """Generate training progress charts with memory efficiency."""
        try:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            
            output_dir = self.config.trainer.output_dir / 'charts'
            output_dir.mkdir(exist_ok=True, parents=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Step {step}', fontsize=16)
            
            # Loss plot
            if self.chart_data['loss_history']:
                ax = axes[0, 0]
                steps = [entry['step'] for entry in self.chart_data['loss_history']]
                losses = [entry['loss'] for entry in self.chart_data['loss_history']]
                ax.plot(steps, losses, 'b-', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                ax.grid(True, alpha=0.3)
            
            # Reward plot
            if self.chart_data['reward_history']:
                ax = axes[0, 1]
                steps = [entry['step'] for entry in self.chart_data['reward_history']]
                rewards = [entry['reward'] for entry in self.chart_data['reward_history']]
                ax.plot(steps, rewards, 'g-', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Reward')
                ax.set_title('Average Reward')
                ax.grid(True, alpha=0.3)
            
            # Memory plot
            if self.chart_data['memory_history']:
                ax = axes[1, 0]
                steps = [entry['step'] for entry in self.chart_data['memory_history']]
                active = [entry.get('mlx_active_mb', 0) for entry in self.chart_data['memory_history']]
                peak = [entry.get('mlx_peak_mb', 0) for entry in self.chart_data['memory_history']]
                ax.plot(steps, active, 'r-', label='Active MLX', linewidth=2)
                ax.plot(steps, peak, 'r--', label='Peak MLX', linewidth=1, alpha=0.5)
                ax.set_xlabel('Step')
                ax.set_ylabel('Memory (MB)')
                ax.set_title('Memory Usage')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Token distribution plot
            if self.chart_data['token_history']:
                ax = axes[1, 1]
                steps = [entry['step'] for entry in self.chart_data['token_history']]
                thinking = [entry['thinking'] for entry in self.chart_data['token_history']]
                answer = [entry['answer'] for entry in self.chart_data['token_history']]
                ax.plot(steps, thinking, 'b-', label='Thinking', linewidth=2)
                ax.plot(steps, answer, 'orange', label='Answer', linewidth=2)
                ax.set_xlabel('Step')
                ax.set_ylabel('Token Count')
                ax.set_title('Token Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            chart_path = output_dir / f'training_chart_step_{step}.png'
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Chart saved: {chart_path}")
            
            if self.wandb and self.wandb.run:
                self.wandb.log({'training_chart': self.wandb.Image(str(chart_path))}, step=step)
            
            del fig, axes
            
        except Exception as e:
            logger.warning(f"Could not generate charts: {e}")
    
    def train_step(self, rollout_batch: Dict, update_step: int):
        """Execute one training step with aggressive memory management."""
        start_time = time.time()
        step_metrics = {}
        
        has_dual_masks = 'thinking_mask' in rollout_batch and 'answer_mask' in rollout_batch
        use_sft = (has_dual_masks and 
                   hasattr(self.config.trainer, 'use_sft_on_answer') and
                   self.config.trainer.use_sft_on_answer and
                   'reference_tokens' in rollout_batch)
        
        # Track tokens
        if has_dual_masks:
            thinking_count = int(mx.sum(rollout_batch['thinking_mask']).item())
            answer_count = int(mx.sum(rollout_batch['answer_mask']).item())
            self.token_tracker.update(thinking_count, answer_count, True)
        else:
            total_tokens = int(mx.sum(rollout_batch.get('response_mask', mx.array([0]))).item())
            self.token_tracker.update(total_tokens // 2, total_tokens // 2, False)
        
        # Compute loss and gradients
        if has_dual_masks and hasattr(self.config.trainer, 'use_dual_gradients') and self.config.trainer.use_dual_gradients:
            thinking_loss, thinking_grads, answer_loss, answer_grads, metrics = \
                self.grpo_algorithm.calculate_dual_gradient_loss(rollout_batch, self.config, self.tokenizer.pad_token_id)
            
            # Get layer ranges
            thinking_start = getattr(self.config.trainer, 'thinking_layer_start', 22)
            thinking_end = getattr(self.config.trainer, 'thinking_layer_end', 30)
            answer_start = getattr(self.config.trainer, 'answer_layer_start', thinking_end + 1)
            answer_end = getattr(self.config.trainer, 'answer_layer_end', 36)
            
            # Token statistics
            thinking_tokens = mx.sum(rollout_batch['thinking_mask']).item()
            answer_tokens = mx.sum(rollout_batch['answer_mask']).item()
            total_mask_tokens = thinking_tokens + answer_tokens
            
            if total_mask_tokens > 0:
                thinking_ratio = thinking_tokens / total_mask_tokens
                answer_ratio = answer_tokens / total_mask_tokens
            else:
                thinking_ratio = 0.5
                answer_ratio = 0.5
            
            step_metrics.update({
                'training/thinking_token_count': thinking_tokens,
                'training/answer_token_count': answer_tokens,
                'training/thinking_ratio': thinking_ratio,
                'training/answer_ratio': answer_ratio
            })
            
            # Adaptive weights
            base_answer_weight = getattr(self.config.trainer, 'answer_gradient_weight', 2.0)
            base_sft_weight = getattr(self.config.trainer, 'sft_weight', 0.1)
            adaptive_enabled = getattr(self.config.trainer, 'adaptive_gradient_weights', True)
            
            step_metrics['training/answer_weight_base'] = base_answer_weight
            step_metrics['training/sft_weight_base'] = base_sft_weight
            
            if adaptive_enabled and total_mask_tokens < 200:
                if thinking_ratio > 0.7:
                    answer_weight = base_answer_weight * (1.0 / max(answer_ratio, 0.1))
                    answer_weight = min(answer_weight, base_answer_weight * 4.0)
                    sft_weight = base_sft_weight * (1.0 / max(answer_ratio, 0.2))
                    sft_weight = min(sft_weight, base_sft_weight * 3.0)
                    step_metrics.update({
                        'training/adaptive_weights_active': 1.0,
                        'training/answer_weight_boost_ratio': answer_weight / base_answer_weight,
                        'training/sft_weight_boost_ratio': sft_weight / base_sft_weight
                    })
                else:
                    answer_weight = base_answer_weight
                    sft_weight = base_sft_weight
                    step_metrics.update({
                        'training/adaptive_weights_active': 0.0,
                        'training/answer_weight_boost_ratio': 1.0,
                        'training/sft_weight_boost_ratio': 1.0
                    })
            else:
                answer_weight = base_answer_weight
                sft_weight = base_sft_weight
                step_metrics.update({
                    'training/adaptive_weights_active': 0.0,
                    'training/answer_weight_boost_ratio': 1.0,
                    'training/sft_weight_boost_ratio': 1.0
                })
            
            step_metrics['training/answer_weight_actual'] = answer_weight
            step_metrics['training/sft_weight_actual'] = sft_weight
            
            # Combine gradients
            if use_sft:
                sft_loss, sft_grads, sft_metrics = self.grpo_algorithm.calculate_sft_loss_and_grads(
                    rollout_batch, rollout_batch['reference_tokens'], self.config, self.tokenizer.pad_token_id
                )
                
                weighted_answer_grads = tree_map(lambda g: g * answer_weight, answer_grads)
                
                if sft_grads:
                    weighted_sft_grads = tree_map(lambda g: g * sft_weight, sft_grads)
                    combined_answer_grads = safe_tree_add(weighted_answer_grads, weighted_sft_grads)
                    del weighted_sft_grads
                else:
                    combined_answer_grads = weighted_answer_grads
                
                del weighted_answer_grads
                
                metrics.update(sft_metrics)
                step_metrics.update({
                    'loss/thinking_loss': thinking_loss.item(),
                    'loss/answer_rl_loss': answer_loss.item(),
                    'loss/answer_sft_loss': sft_loss.item()
                })
                
                total_loss = (thinking_loss.item() + answer_loss.item() + sft_loss.item()) / 3
            else:
                combined_answer_grads = tree_map(lambda g: g * answer_weight, answer_grads)
                step_metrics.update({
                    'loss/thinking_loss': thinking_loss.item(),
                    'loss/answer_rl_loss': answer_loss.item()
                })
                total_loss = (thinking_loss.item() + answer_loss.item()) / 2
            
            step_metrics['loss/total'] = total_loss
            
            # Scale and mask gradients
            grad_accum = self.config.trainer.grad_accum_steps
            scaled_thinking_grads = tree_map(lambda g: g / grad_accum, thinking_grads)
            
            # Mask to layer bands
            from mlx_rl_trainer.algorithms.grpo.grpo_algorithm import _safe_gradient_combine
            combined_grads_unmasked = _safe_gradient_combine(weighted_answer_grads, weighted_sft_grads, operation='add') if use_sft else combined_answer_grads
            
            masked_thinking = mask_grads_to_layer_band(
                scaled_thinking_grads, start=thinking_start, end=thinking_end,
                include_embed=False, include_head=False
            )
            masked_answer = mask_grads_to_layer_band(
                combined_answer_grads, start=answer_start, end=answer_end,
                include_embed=True, include_head=True
            )
            
            final_grads = safe_tree_add(masked_thinking, masked_answer)
            
            del thinking_grads, answer_grads, combined_answer_grads
            del scaled_thinking_grads, masked_thinking, masked_answer
            
        else:
            # Standard gradient computation
            loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
                rollout_batch, self.config, self.tokenizer.pad_token_id
            )
            
            final_grads = tree_map(lambda g: g / self.config.trainer.grad_accum_steps, grads)
            total_loss = loss.item()
            step_metrics['loss/total'] = total_loss
            
            del grads
        
        mx.eval(final_grads)
        
        # Track gradients
        grad_metrics = self._track_layer_gradients(final_grads, update_step)
        step_metrics.update(grad_metrics)
        
        # Compute global grad norm
        grad_norm = self._compute_global_grad_norm(final_grads)
        
        # Update metrics
        step_metrics.update({
            'training/reward_mean': rollout_batch['advantages'].mean().item(),
            'training/reward_std': rollout_batch['advantages'].std().item(),
            'training/learning_rate': self.lr_scheduler(update_step),
            'training/kl_divergence': metrics.get('kl_divergence', 0.0),
            'training/step_time_s': time.time() - start_time
        })
        
        # Update histories with limits
        self.chart_data['loss_history'].append({'step': update_step, 'loss': total_loss})
        if len(self.chart_data['loss_history']) > self._max_history:
            del self.chart_data['loss_history'][0]
        
        self.chart_data['reward_history'].append({'step': update_step, 'reward': step_metrics['training/reward_mean']})
        if len(self.chart_data['reward_history']) > self._max_history:
            del self.chart_data['reward_history'][0]
        
        if has_dual_masks:
            self.chart_data['token_history'].append({
                'step': update_step,
                'thinking': step_metrics['training/thinking_token_count'],
                'answer': step_metrics['training/answer_token_count']
            })
            if len(self.chart_data['token_history']) > self._max_history:
                del self.chart_data['token_history'][0]
        
        training_metrics = TrainingMetrics(
            loss=total_loss,
            reward_mean=step_metrics['training/reward_mean'],
            reward_std=step_metrics['training/reward_std'],
            grad_norm=grad_norm,
            learning_rate=step_metrics['training/learning_rate'],
            step_time_s=step_metrics['training/step_time_s'],
            kl_divergence=step_metrics['training/kl_divergence'],
            epoch=self.current_epoch,
            step=update_step
        )
        
        return training_metrics, final_grads, step_metrics
    
    def _compute_global_grad_norm(self, grads) -> float:
        """Compute global gradient norm efficiently."""
        if not grads:
            return 0.0
        
        total_norm_sq = mx.array(0.0)
        for path, grad in tree_flatten(grads):
            if isinstance(grad, mx.array):
                total_norm_sq += mx.sum(grad ** 2)
        
        result = float(mx.sqrt(total_norm_sq).item())
        del total_norm_sq
        
        return result
    
    def generate_rollouts(self, batch_data: Dict, update_step: int):
        """Generate rollouts with memory-efficient error handling."""
        try:
            prompts_data = batch_data.get('prompts_data', [])
            is_invalid = any(data.get('is_invalid_sample', False) for data in prompts_data)
            
            rollout_batch, avg_reward, generation_metrics, reward_details = generate_rollouts_for_batch(
                model=self.actor_model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                prompts_data=prompts_data,
                dataset=self.data_manager._train_dataset,
                config=self.config,
                reward_composer=self.reward_composer,
                run_id=self._run_id,
                current_update=update_step,
                is_invalid_batch=is_invalid
            )
            
            return rollout_batch, avg_reward, generation_metrics, reward_details
            
        except RuntimeError as e:
            if 'METAL' in str(e) or 'Command buffer' in str(e):
                self.metal_error_count += 1
                logger.error(f"Metal error in generation ({self.metal_error_count}/{self.max_metal_errors}): {e}")
                
                gc.collect()
                try:
                    mx.metal.clear_cache()
                except:
                    pass
                
                if self.metal_error_count >= self.max_metal_errors:
                    terminal_alert('CRITICAL: Multiple Metal errors. Saving checkpoint and exiting.', level='ERROR')
                    self._save_checkpoint_with_retry(update_step, reason='metal_error')
                    raise
                
                return {}, 0.0, {}, {}
            else:
                raise
    
    def log_comprehensive_metrics(self, step: int, step_metrics: Dict, generation_metrics: Optional[Dict] = None):
        """Log comprehensive metrics with memory stats."""
        mem_stats = self.memory_monitor.get_memory_stats()
        
        if mem_stats:
            memory_metrics = {f"memory/{k}": v for k, v in mem_stats.items()}
            step_metrics.update(memory_metrics)
            
            self.chart_data['memory_history'].append({'step': step, **mem_stats})
            if len(self.chart_data['memory_history']) > self._max_history:
                del self.chart_data['memory_history'][0]
        
        token_metrics = self.token_tracker.to_dict()
        step_metrics.update(token_metrics)
        
        if generation_metrics:
            step_metrics.update(generation_metrics)
        
        if self.wandb and self.wandb.run:
            self.wandb.log({**step_metrics, 'step': step})
        
        if self.metrics_logger:
            self.metrics_logger.log_metrics(step_metrics, step=step)
    
    async def run(self, should_shutdown):
        """Main training loop with aggressive memory management."""
        start_step, self.current_epoch = self._setup()
        
        if start_step > 0:
            self.global_step = start_step + 1
            logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            self.global_step = 0
            logger.info('Starting training from scratch')
        
        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)
        
        await self.data_manager.load_datasets()
        
        from tqdm import trange
        progress_bar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc='Training',
            unit='step'
        )
        
        dataloader_iter = iter([])
        grad_accum_steps = self.config.trainer.grad_accum_steps
        
        with progress_bar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    logger.info('Shutdown requested')
                    break
                
                try:
                    # Get batch
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        self.current_epoch += 1
                        logger.info(f"Epoch {self.current_epoch}")
                        
                        dataloader_iter = iter(self.data_manager.get_dataloader(
                            'train', self.config.trainer.ppo_batch_size
                        ))
                        batch = next(dataloader_iter)
                    
                    # Safety check
                    is_safe, safety_msg = self._check_pre_iteration_safety(batch)
                    if not is_safe:
                        logger.warning(f"Safety check failed: {safety_msg}")
                        terminal_alert(f"Safety checkpoint triggered: {safety_msg}", level='WARNING')
                        
                        self._save_checkpoint_with_retry(self.global_step, reason='safety')
                        
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass
                        
                        time.sleep(2)
                    
                    # Gradient accumulation
                    accumulated_grads = None
                    total_loss = 0.0
                    total_reward = 0.0
                    num_valid_steps = 0
                    combined_gen_metrics = {}
                    
                    for accum_idx in range(grad_accum_steps):
                        # Generate rollouts
                        rollout_batch, avg_reward, gen_metrics, reward_details = \
                            self.generate_rollouts(batch, self.global_step)
                        
                        if not rollout_batch or 'tokens' not in rollout_batch:
                            logger.warning(f"Empty rollout at step {self.global_step}")
                            continue
                        
                        # Train step
                        train_metrics, step_grads, detailed_metrics = \
                            self.train_step(rollout_batch, self.global_step)
                        
                        total_loss += train_metrics.loss
                        total_reward += avg_reward
                        num_valid_steps += 1
                        
                        # Accumulate generation metrics
                        if gen_metrics:
                            for key, value in gen_metrics.items():
                                combined_gen_metrics[key] = combined_gen_metrics.get(key, 0.0) + value
                        
                        # Accumulate gradients
                        if step_grads:
                            if accumulated_grads is None:
                                accumulated_grads = step_grads
                            else:
                                accumulated_grads = tree_map(mx.add, accumulated_grads, step_grads)
                        
                        # Cleanup
                        del rollout_batch, train_metrics, step_grads
                        gc.collect()
                    
                    # Apply accumulated gradients
                    if accumulated_grads and num_valid_steps > 0:
                        self.optimizer.learning_rate = self.lr_scheduler(self.global_step)
                        self.optimizer.apply_gradients(accumulated_grads, self.actor_model.trainable_parameters())
                        mx.eval(self.actor_model.parameters())
                        
                        avg_loss = total_loss / num_valid_steps
                        avg_reward = total_reward / num_valid_steps
                        
                        # Log metrics
                        self.log_comprehensive_metrics(self.global_step, detailed_metrics, combined_gen_metrics)
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'Loss': f"{avg_loss:.4f}",
                            'Reward': f"{avg_reward:.3f}",
                            'Tokens': f"{self.token_tracker.total_tokens}"
                        })
                        progress_bar.update(1)
                        
                        # Periodic checkpoint
                        should_save = (self.config.checkpointing.save_every > 0 and
                                      (self.global_step + 1) % self.config.checkpointing.save_every == 0)
                        
                        if should_save:
                            self._generate_charts(self.global_step)
                            self._save_checkpoint_with_retry(self.global_step, reason='regular')
                        
                        del accumulated_grads
                    
                    self.global_step += 1
                    
                    # Periodic memory cleanup
                    if self.global_step % 10 == 0:
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass
                
                except RuntimeError as e:
                    if 'METAL' in str(e) or 'Command buffer' in str(e):
                        logger.error(f"Metal command buffer error at step {self.global_step}: {e}")
                        terminal_alert('Metal error encountered! Saving checkpoint...', level='ERROR')
                        
                        self._save_checkpoint_with_retry(self.global_step, reason='metal_error')
                        
                        gc.collect()
                        try:
                            mx.metal.clear_cache()
                        except:
                            pass
                        
                        # Reset error count periodically
                        if self.global_step % 10 == 0:
                            self.metal_error_count = 0
                        
                        if self.metal_error_count < self.max_metal_errors:
                            logger.info('Attempting to continue training...')
                            time.sleep(5)
                            continue
                        else:
                            logger.error('Too many Metal errors. Exiting.')
                            break
                    else:
                        raise
        
        # Final checkpoint and cleanup
        self._generate_charts(self.global_step)
        self._save_checkpoint_with_retry(self.global_step, reason='final')
        
        if self.wandb and self.wandb.run:
            self.wandb.finish()
    
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        """Evaluate the model."""
        logger.info(f"Evaluation at step {update_step}")
        return []
