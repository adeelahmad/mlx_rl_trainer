"""
GRPO Algorithm - BULLETPROOF Gradient Combination

Handles ANY structural mismatch between gradient dictionaries:
- Missing keys (like 'rope')
- Extra keys
- Different nesting depths
- None values
- Non-tensor values

Strategy: Use first gradient dict as template, fill in from second where possible
"""
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


def _mask_gradients_by_layers(
    grads: Dict,
    config: ExperimentConfig,
    sft_mode: str,
) -> Dict:
    """
    Mask SFT gradients based on layer configuration.
    Preserves the 'model' wrapper and full nested structure.
    """
    thinking_start = getattr(config.trainer, 'thinking_layer_start', None)
    thinking_end = getattr(config.trainer, 'thinking_layer_end', None)
    answer_start = getattr(config.trainer, 'answer_layer_start', None)
    answer_end = getattr(config.trainer, 'answer_layer_end', None)

    if sft_mode == 'all' or thinking_start is None or answer_start is None:
        return grads

    thinking_weight = getattr(config.trainer, 'sft_thinking_weight', 0.0)
    answer_weight = getattr(config.trainer, 'sft_answer_weight', 1.0)

    has_model_wrapper = 'model' in grads and isinstance(grads['model'], dict)
    inner_grads = grads['model'] if has_model_wrapper else grads

    def mask_grad(grad, path_parts):
        """Apply mask based on layer number in path."""
        if not isinstance(grad, mx.array):
            return grad

        path_str = '.'.join(str(p) for p in path_parts)
        layer_num = _extract_layer_number(path_str)

        if layer_num is None:
            return grad

        if sft_mode == 'answer_only':
            if answer_start <= layer_num <= answer_end:
                return grad
            else:
                return mx.zeros_like(grad)
        elif sft_mode == 'weighted':
            if thinking_start <= layer_num <= thinking_end:
                return grad * thinking_weight
            elif answer_start <= layer_num <= answer_end:
                return grad * answer_weight
            else:
                return grad
        elif sft_mode == 'exclude_thinking':
            if thinking_start <= layer_num <= thinking_end:
                return mx.zeros_like(grad)
            else:
                return grad
        return grad

    flat_inner = tree_flatten(inner_grads, is_leaf=lambda x: isinstance(x, mx.array))
    masked_flat = [(path, mask_grad(val, path)) for path, val in flat_inner]
    masked_inner = tree_unflatten(masked_flat)

    if has_model_wrapper:
        return {'model': masked_inner}
    return masked_inner


def _robust_tree_combine(tree1: Any, tree2: Any, fn, path="") -> Any:
    """
    Recursively combine two trees, handling structural mismatches.

    Uses tree1 as the template. If tree2 has matching keys, applies fn.
    If tree2 is missing keys, uses tree1 value only.

    Args:
        tree1: First tree (template)
        tree2: Second tree
        fn: Function to combine matching values (e.g., lambda a, b: a + b)
        path: Current path (for logging)

    Returns:
        Combined tree with tree1's structure
    """
    # Base case: both are arrays
    if isinstance(tree1, mx.array) and isinstance(tree2, mx.array):
        try:
            return fn(tree1, tree2)
        except Exception as e:
            logger.warning(f"Error combining arrays at {path}: {e}. Using tree1 only.")
            return tree1

    # tree1 is array but tree2 is not - use tree1
    if isinstance(tree1, mx.array):
        if tree2 is not None and not isinstance(tree2, mx.array):
            logger.debug(f"Type mismatch at {path}: tree1=array, tree2={type(tree2)}. Using tree1.")
        return tree1

    # tree1 is dict
    if isinstance(tree1, dict):
        if not isinstance(tree2, dict):
            logger.warning(f"Structure mismatch at {path}: tree1=dict, tree2={type(tree2)}. Using tree1 only.")
            return tree1

        result = {}
        for key, val1 in tree1.items():
            new_path = f"{path}.{key}" if path else str(key)

            if key in tree2:
                # Both have this key, recurse
                result[key] = _robust_tree_combine(val1, tree2[key], fn, new_path)
            else:
                # tree2 missing this key
                logger.debug(f"Key '{key}' missing in tree2 at {path}. Using tree1 value only.")
                result[key] = val1

        return result

    # tree1 is list/tuple
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

    # Other types (scalars, None, etc.) - use tree1
    return tree1


def _safe_gradient_combine(grad1: Dict, grad2: Dict, operation='add') -> Dict:
    """
    Safely combine two gradient dictionaries with bulletproof error handling.

    Handles:
    - Missing keys (e.g., 'rope' in one but not the other)
    - Structural mismatches
    - Type mismatches
    - None values

    Always returns a valid gradient dict (never crashes).
    """
    if not grad1:
        logger.warning("grad1 is empty, returning grad2")
        return grad2 or {}
    if not grad2:
        logger.warning("grad2 is empty, returning grad1")
        return grad1

    # Define combination function
    if operation == 'add':
        combine_fn = lambda a, b: a + b
    elif operation == 'subtract':
        combine_fn = lambda a, b: a - b
    else:
        logger.error(f"Unknown operation: {operation}. Returning grad1.")
        return grad1

    try:
        # Use robust tree combine
        result = _robust_tree_combine(grad1, grad2, combine_fn)
        return result
    except Exception as e:
        logger.error(f"Error in gradient combination: {e}", exc_info=True)
        logger.error("Falling back to grad1 only")
        return grad1


class GRPOAlgorithm:
    def __init__(self, config, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta

    def compute_advantages(self, rewards_flat, samples_per_prompt):
        """Compute advantages with optional baseline normalization."""
        if samples_per_prompt <= 1:
            return (rewards_flat - mx.mean(rewards_flat)) / (mx.std(rewards_flat) + 1e-8)

        batch_size = rewards_flat.shape[0] // samples_per_prompt
        rewards_reshaped = rewards_flat.reshape(batch_size, samples_per_prompt)
        baseline = mx.mean(rewards_reshaped, axis=1, keepdims=True)
        std = mx.std(rewards_reshaped, axis=1, keepdims=True)
        advantages = (rewards_reshaped - baseline) / (std + 1e-8)
        return advantages.flatten()

    def calculate_loss_and_grads(self, rollout_batch, full_config, pad_token_id):
        """Original single gradient computation method."""
        A = rollout_batch
        zero_val = 0.0

        def compute_loss(actor_model):
            tokens_key = 'tokens'
            mask_key = 'response_mask'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[mask_key].shape[1]
            shifted_logits = logits[:, offset-1:-1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)

            log_ratio = gathered_log_probs - A['ref_log_probs']
            kl_term = mx.exp(log_ratio) - 1 - log_ratio
            kl_penalty = kl_term * A[mask_key]

            advantages_expanded = A['advantages'][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * A[mask_key]

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty
            loss = mx.sum(total_loss_per_token) / mx.sum(A[mask_key])

            kl_div = mx.sum(kl_penalty) / mx.sum(A[mask_key])
            policy_loss = mx.sum(policy_loss_term) / mx.sum(A[mask_key])

            return loss, {
                'kl_divergence': kl_div,
                'policy_loss': policy_loss
            }

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)
            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return mx.array(zero_val), {}, {'kl_divergence': zero_val, 'policy_loss': zero_val}

    def calculate_dual_gradient_loss(self, rollout_batch, full_config, pad_token_id):
        """Compute separate gradients for thinking and answer tokens."""
        A = rollout_batch

        has_thinking_mask = 'thinking_mask' in A
        has_answer_mask = 'answer_mask' in A

        if not has_thinking_mask or not has_answer_mask:
            logger.warning("Thinking/answer masks not found. Falling back to standard gradient computation.")
            loss, grads, metrics = self.calculate_loss_and_grads(A, full_config, pad_token_id)
            return loss, grads, loss, grads, metrics

        def compute_loss_with_mask_and_metrics(actor_model, mask_type, compute_metrics=False):
            tokens_key = 'tokens'
            response_mask_key = 'response_mask'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            shifted_logits = logits[:, offset-1:-1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(log_probs, target_tokens[..., None], axis=-1).squeeze(-1)

            log_ratio = gathered_log_probs - A['ref_log_probs']
            kl_term = mx.exp(log_ratio) - 1 - log_ratio

            token_mask = A[mask_type]
            combined_mask = A[response_mask_key] * token_mask

            kl_penalty = kl_term * combined_mask
            advantages_expanded = A['advantages'][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * combined_mask

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty

            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss_per_token) / (mask_sum + 1e-8)

            if compute_metrics:
                kl_div = mx.sum(kl_penalty) / (mask_sum + 1e-8)
                policy_loss = mx.sum(policy_loss_term) / (mask_sum + 1e-8)
                return loss, {'kl_divergence': kl_div, 'policy_loss': policy_loss}

            return loss

        thinking_loss_fn = lambda model: compute_loss_with_mask_and_metrics(model, 'thinking_mask', False)
        thinking_grads_fn = nn.value_and_grad(self.actor, thinking_loss_fn)
        thinking_loss, thinking_grads = thinking_grads_fn(self.actor)

        answer_loss_fn = lambda model: compute_loss_with_mask_and_metrics(model, 'answer_mask', True)
        answer_grads_fn = nn.value_and_grad(self.actor, answer_loss_fn)
        (answer_loss, metrics), answer_grads = answer_grads_fn(self.actor)

        metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

        return thinking_loss, thinking_grads, answer_loss, answer_grads, metrics_dict

    def calculate_sft_loss_and_grads(self, rollout_batch, reference_tokens, full_config, pad_token_id):
        """Compute SFT loss and gradients with layer-specific control."""
        A = rollout_batch

        sft_mode = getattr(full_config.trainer, 'sft_mode', 'weighted')

        if not hasattr(self, '_sft_mode_logged'):
            logger.info(f"SFT layer control mode: {sft_mode}")
            if sft_mode == 'exclude_thinking':
                logger.info("System 2 (thinking) layers will NOT receive SFT gradients - only RL signal")
            elif sft_mode == 'answer_only':
                logger.info("Only System 1 (answer) layers will receive SFT gradients")
            elif sft_mode == 'weighted':
                thinking_w = getattr(full_config.trainer, 'sft_thinking_weight', 0.0)
                answer_w = getattr(full_config.trainer, 'sft_answer_weight', 1.0)
                logger.info(f"Weighted SFT: thinking={thinking_w}, answer={answer_w}")
            self._sft_mode_logged = True

        if 'answer_mask' not in A:
            logger.warning("Answer mask not found for SFT. Falling back to response_mask.")
            answer_mask = A.get('response_mask', mx.ones_like(reference_tokens, dtype=mx.float32))
        else:
            answer_mask = A['answer_mask']

        def compute_sft_loss(actor_model):
            tokens_key = 'tokens'
            response_mask_key = 'response_mask'

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            response_logits = logits[:, offset-1:-1, :]

            min_len = min(response_logits.shape[1], reference_tokens.shape[1])
            response_logits = response_logits[:, :min_len, :]
            target_tokens = reference_tokens[:, :min_len]
            current_answer_mask = answer_mask[:, :min_len] if answer_mask.shape[1] >= min_len else answer_mask

            if response_logits.shape[1] != target_tokens.shape[1]:
                logger.debug(
                    f"Aligning SFT shapes: "
                    f"logits {response_logits.shape[1]} vs "
                    f"targets {target_tokens.shape[1]} vs "
                    f"mask {current_answer_mask.shape[1]} -> {min_len}"
                )

            log_probs = nn.log_softmax(response_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs,
                target_tokens[..., None],
                axis=-1
            ).squeeze(-1)

            masked_log_probs = -gathered_log_probs * current_answer_mask

            mask_sum = mx.sum(current_answer_mask)
            loss = mx.sum(masked_log_probs) / (mask_sum + 1e-8)

            return loss, {'sft_loss': loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_sft_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)

            # Apply layer-specific masking
            grads = _mask_gradients_by_layers(grads, full_config, sft_mode)

            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict
        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {'sft_loss': 0.0}
