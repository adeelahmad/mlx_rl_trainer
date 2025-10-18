import logging
from typing import Dict, Any, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """Extract layer number from parameter path."""
    import re
    match = re.search(r'\.layers\.(\d+)\.', param_path)
    return int(match.group(1)) if match else None


def _mask_gradients_by_layers(grads, config, sft_mode: str):
    """Mask gradients by layer ranges for SFT training."""
    thinking_start = getattr(config.trainer, 'thinking_layer_start', None)
    thinking_end = getattr(config.trainer, 'thinking_layer_end', None)
    answer_start = getattr(config.trainer, 'answer_layer_start', None)
    answer_end = getattr(config.trainer, 'answer_layer_end', None)
    
    if sft_mode == 'all' or thinking_start is None or answer_start is None:
        return grads
    
    thinking_weight = getattr(config.trainer, 'sft_thinking_weight', 0.0)
    answer_weight = getattr(config.trainer, 'sft_answer_weight', 1.0)
    
    is_nested = 'model' in grads and isinstance(grads['model'], dict)
    grad_tree = grads['model'] if is_nested else grads
    
    def apply_mask(grad, path_parts):
        if not isinstance(grad, mx.array):
            return grad
        
        path_str = '.'.join(str(p) for p in path_parts)
        layer_num = _extract_layer_number(path_str)
        
        if layer_num is None:
            return grad
        
        if sft_mode == 'answer_only':
            if answer_start <= layer_num <= answer_end:
                return grad
            return mx.zeros_like(grad)
        elif sft_mode == 'weighted':
            if thinking_start <= layer_num <= thinking_end:
                return grad * thinking_weight if thinking_weight != 0.0 else mx.zeros_like(grad)
            elif answer_start <= layer_num <= answer_end:
                return grad * answer_weight if answer_weight != 1.0 else grad
            return grad
        elif sft_mode == 'exclude_thinking':
            if thinking_start <= layer_num <= thinking_end:
                return mx.zeros_like(grad)
            return grad
        
        return grad
    
    flat_grads = tree_flatten(grad_tree, is_leaf=lambda x: isinstance(x, mx.array))
    masked_grads = [(path, apply_mask(grad, path)) for path, grad in flat_grads]
    result_tree = tree_unflatten(masked_grads)
    
    if is_nested:
        return {'model': result_tree}
    return result_tree


def _robust_tree_combine(tree1, tree2, fn, path: str = ''):
    """Combine two trees robustly with error handling."""
    if isinstance(tree1, mx.array) and isinstance(tree2, mx.array):
        try:
            return fn(tree1, tree2)
        except Exception as e:
            logger.warning(f"Error combining arrays at {path}: {e}. Using tree1 only.")
            return tree1
    
    if isinstance(tree1, mx.array):
        if tree2 is not None and not isinstance(tree2, mx.array):
            logger.debug(f"Type mismatch at {path}: tree1=array, tree2={type(tree2)}. Using tree1.")
        return tree1
    
    if isinstance(tree1, dict):
        if not isinstance(tree2, dict):
            logger.warning(f"Structure mismatch at {path}: tree1=dict, tree2={type(tree2)}. Using tree1 only.")
            return tree1
        
        result = {}
        for key, val1 in tree1.items():
            new_path = f"{path}.{key}" if path else str(key)
            if key in tree2:
                result[key] = _robust_tree_combine(val1, tree2[key], fn, new_path)
            else:
                logger.debug(f"Key '{key}' missing in tree2 at {path}. Using tree1 value only.")
                result[key] = val1
        return result
    
    if isinstance(tree1, (list, tuple)):
        if not isinstance(tree2, (list, tuple)):
            logger.warning(f"Structure mismatch at {path}: tree1=list, tree2={type(tree2)}. Using tree1 only.")
            return tree1
        if len(tree1) != len(tree2):
            logger.warning(f"Length mismatch at {path}: tree1={len(tree1)}, tree2={len(tree2)}. Using tree1 only.")
            return tree1
        result = [_robust_tree_combine(v1, v2, fn, f"{path}[{i}]") 
                  for i, (v1, v2) in enumerate(zip(tree1, tree2))]
        return type(tree1)(result)
    
    return tree1


def _safe_gradient_combine(grad1, grad2, operation: str = 'add'):
    """Safely combine two gradient trees."""
    if not grad1:
        logger.warning('grad1 is empty, returning grad2')
        return grad2 or {}
    if not grad2:
        logger.warning('grad2 is empty, returning grad1')
        return grad1
    
    if operation == 'add':
        combine_fn = lambda a, b: a + b
    elif operation == 'subtract':
        combine_fn = lambda a, b: a - b
    else:
        logger.error(f"Unknown operation: {operation}. Returning grad1.")
        return grad1
    
    try:
        result = _robust_tree_combine(grad1, grad2, combine_fn)
        return result
    except Exception as e:
        logger.error(f"Error in gradient combination: {e}", exc_info=True)
        logger.error('Falling back to grad1 only')
        return grad1


class GRPOAlgorithm:
    """Group Relative Policy Optimization Algorithm."""
    
    def __init__(self, config: ExperimentConfig, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta
        
        # Compile flag - can be disabled for debugging
        self.use_compile = getattr(config.trainer, 'use_compile', True)
        if self.use_compile:
            logger.info("MLX compilation enabled for loss functions")
        else:
            logger.info("MLX compilation disabled (debugging mode)")
    
    def compute_advantages(self, rewards_flat: mx.array, samples_per_prompt: int) -> mx.array:
        """Compute advantages using group normalization."""
        if samples_per_prompt <= 1:
            mean_r = mx.mean(rewards_flat)
            std_r = mx.std(rewards_flat)
            result = (rewards_flat - mean_r) / (std_r + 1e-8)
            mx.eval(result)
            return result
        
        num_prompts = rewards_flat.shape[0] // samples_per_prompt
        rewards_grouped = rewards_flat.reshape(num_prompts, samples_per_prompt)
        
        group_mean = mx.mean(rewards_grouped, axis=1, keepdims=True)
        group_std = mx.std(rewards_grouped, axis=1, keepdims=True)
        
        advantages = (rewards_grouped - group_mean) / (group_std + 1e-8)
        result = advantages.flatten()
        
        mx.eval(result)
        del rewards_grouped, group_mean, group_std, advantages
        
        return result
    
    def calculate_loss_and_grads(self, rollout_batch: Dict, full_config, pad_token_id: int):
        """Calculate GRPO loss and gradients."""
        
        def loss_fn():
            tokens = rollout_batch['tokens']
            response_mask = rollout_batch['response_mask']
            
            logits = self.actor(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)
            
            offset = tokens.shape[1] - response_mask.shape[1]
            logits_for_tokens = logits[:, offset - 1:-1, :]
            target_tokens = tokens[:, offset:]
            
            log_probs = nn.log_softmax(logits_for_tokens, axis=-1)
            action_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)
            
            ratio = action_log_probs - rollout_batch['ref_log_probs']
            kl_term = mx.exp(ratio) - 1 - ratio
            kl_loss = kl_term * response_mask
            
            advantages = rollout_batch['advantages'][:, None]
            policy_loss = -ratio * advantages * response_mask
            
            total_loss = policy_loss + self.beta * kl_loss
            
            mask_sum = mx.sum(response_mask)
            loss = mx.sum(total_loss) / mask_sum
            
            kl_div = mx.sum(kl_loss) / mask_sum
            pol_loss = mx.sum(policy_loss) / mask_sum
            
            mx.eval(loss, kl_div, pol_loss)
            
            return loss, {'kl_divergence': kl_div, 'policy_loss': pol_loss}
        
        try:
            # Compile the gradient function for better performance
            if self.use_compile:
                grad_fn = mx.compile(nn.value_and_grad(self.actor, loss_fn), shapeless=True)
            else:
                grad_fn = nn.value_and_grad(self.actor, loss_fn)
            
            (loss, metrics), grads = grad_fn(self.actor)
            
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            
            mx.eval(grads)
            
            return loss, grads, metrics_dict
            
        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {'kl_divergence': 0.0, 'policy_loss': 0.0}
    
    def calculate_dual_gradient_loss(self, rollout_batch: Dict, full_config, pad_token_id: int):
        """Calculate separate gradients for thinking and answer tokens."""
        has_masks = 'thinking_mask' in rollout_batch and 'answer_mask' in rollout_batch
        
        if not has_masks:
            logger.warning('Thinking/answer masks not found. Falling back to standard gradient computation.')
            loss, grads, metrics = self.calculate_loss_and_grads(rollout_batch, full_config, pad_token_id)
            return loss, grads, loss, grads, metrics
        
        # Create loss functions that work with nn.value_and_grad
        def thinking_loss_fn():
            tokens = rollout_batch['tokens']
            response_mask = rollout_batch['response_mask']
            
            logits = self.actor(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)
            
            offset = tokens.shape[1] - response_mask.shape[1]
            logits_for_tokens = logits[:, offset - 1:-1, :]
            target_tokens = tokens[:, offset:]
            
            log_probs = nn.log_softmax(logits_for_tokens, axis=-1)
            action_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)
            
            ratio = action_log_probs - rollout_batch['ref_log_probs']
            kl_term = mx.exp(ratio) - 1 - ratio
            
            mask = rollout_batch['thinking_mask']
            combined_mask = response_mask * mask
            
            kl_loss = kl_term * combined_mask
            advantages = rollout_batch['advantages'][:, None]
            policy_loss = -ratio * advantages * combined_mask
            
            total_loss = policy_loss + self.beta * kl_loss
            
            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss) / (mask_sum + 1e-8)
            
            mx.eval(loss)
            return loss
        
        def answer_loss_fn():
            tokens = rollout_batch['tokens']
            response_mask = rollout_batch['response_mask']
            
            logits = self.actor(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)
            
            offset = tokens.shape[1] - response_mask.shape[1]
            logits_for_tokens = logits[:, offset - 1:-1, :]
            target_tokens = tokens[:, offset:]
            
            log_probs = nn.log_softmax(logits_for_tokens, axis=-1)
            action_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)
            
            ratio = action_log_probs - rollout_batch['ref_log_probs']
            kl_term = mx.exp(ratio) - 1 - ratio
            
            mask = rollout_batch['answer_mask']
            combined_mask = response_mask * mask
            
            kl_loss = kl_term * combined_mask
            advantages = rollout_batch['advantages'][:, None]
            policy_loss = -ratio * advantages * combined_mask
            
            total_loss = policy_loss + self.beta * kl_loss
            
            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss) / (mask_sum + 1e-8)
            
            # Compute metrics for answer loss
            kl_div = mx.sum(kl_loss) / (mask_sum + 1e-8)
            pol_loss = mx.sum(policy_loss) / (mask_sum + 1e-8)
            
            mx.eval(loss, kl_div, pol_loss)
            return loss, {'kl_divergence': kl_div, 'policy_loss': pol_loss}
        
        # Compile gradient functions for better performance
        if self.use_compile:
            thinking_grad_fn = mx.compile(nn.value_and_grad(self.actor, thinking_loss_fn), shapeless=True)
            answer_grad_fn = mx.compile(nn.value_and_grad(self.actor, answer_loss_fn), shapeless=True)
        else:
            thinking_grad_fn = nn.value_and_grad(self.actor, thinking_loss_fn)
            answer_grad_fn = nn.value_and_grad(self.actor, answer_loss_fn)
        
        thinking_loss, thinking_grads = thinking_grad_fn(self.actor)
        mx.eval(thinking_grads)
        
        (answer_loss, metrics), answer_grads = answer_grad_fn(self.actor)
        mx.eval(answer_grads)
        
        metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
        
        return thinking_loss, thinking_grads, answer_loss, answer_grads, metrics_dict
    
    def calculate_sft_loss_and_grads(self, rollout_batch: Dict, reference_tokens: mx.array, 
                                     full_config, pad_token_id: int):
        """Calculate SFT loss and gradients with layer-wise control."""
        sft_mode = getattr(full_config.trainer, 'sft_mode', 'weighted')
        
        if not hasattr(self, '_sft_mode_logged'):
            logger.info(f"SFT layer control mode: {sft_mode}")
            if sft_mode == 'exclude_thinking':
                logger.info('System 2 (thinking) layers will NOT receive SFT gradients - only RL signal')
            elif sft_mode == 'answer_only':
                logger.info('Only System 1 (answer) layers will receive SFT gradients')
            elif sft_mode == 'weighted':
                thinking_w = getattr(full_config.trainer, 'sft_thinking_weight', 0.0)
                answer_w = getattr(full_config.trainer, 'sft_answer_weight', 1.0)
                logger.info(f"Weighted SFT: thinking={thinking_w}, answer={answer_w}")
            self._sft_mode_logged = True
        
        if 'answer_mask' not in rollout_batch:
            logger.warning('Answer mask not found for SFT. Falling back to response_mask.')
            answer_mask = rollout_batch.get('response_mask', mx.ones_like(reference_tokens, dtype=mx.float32))
        else:
            answer_mask = rollout_batch['answer_mask']
        
        def sft_loss_fn():
            tokens = rollout_batch['tokens']
            response_mask = rollout_batch['response_mask']
            
            logits = self.actor(tokens)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)
            
            offset = tokens.shape[1] - response_mask.shape[1]
            logits_for_tokens = logits[:, offset - 1:-1, :]
            
            seq_len = min(logits_for_tokens.shape[1], reference_tokens.shape[1])
            logits_aligned = logits_for_tokens[:, :seq_len, :]
            targets_aligned = reference_tokens[:, :seq_len]
            mask_aligned = answer_mask[:, :seq_len] if answer_mask.shape[1] >= seq_len else answer_mask
            
            if logits_for_tokens.shape[1] != reference_tokens.shape[1]:
                logger.debug(f"Aligning SFT shapes: logits {logits_for_tokens.shape[1]} vs "
                           f"targets {reference_tokens.shape[1]} vs mask {answer_mask.shape[1]} -> {seq_len}")
            
            log_probs = nn.log_softmax(logits_aligned, axis=-1)
            target_log_probs = mx.take_along_axis(log_probs, targets_aligned[..., None], axis=-1).squeeze(-1)
            
            token_losses = -target_log_probs * mask_aligned
            mask_sum = mx.sum(mask_aligned)
            loss = mx.sum(token_losses) / (mask_sum + 1e-8)
            
            mx.eval(loss)
            
            return loss, {'sft_loss': loss}
        
        try:
            # Compile the gradient function for better performance
            if self.use_compile:
                grad_fn = mx.compile(nn.value_and_grad(self.actor, sft_loss_fn), shapeless=True)
            else:
                grad_fn = nn.value_and_grad(self.actor, sft_loss_fn)
            
            (loss, metrics), grads = grad_fn(self.actor)
            
            grads = _mask_gradients_by_layers(grads, full_config, sft_mode)
            mx.eval(grads)
            
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            
            return loss, grads, metrics_dict
            
        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {'sft_loss': 0.0}
