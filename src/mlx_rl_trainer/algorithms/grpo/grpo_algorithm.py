"""
GRPO Algorithm - BULLETPROOF Gradient Combination

Handles ANY structural mismatch between gradient dictionaries:
- Missing keys (like 'rope')
- Extra keys
- Different nesting depths
- None values
- Non-tensor values

Strategy: Use intersection of keys, validate structure, ensure optimizer compatibility
"""
import logging
from typing import Dict, Any, Tuple, Optional, Set
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """Extract layer number from parameter path."""
    import re

    match = re.search(r"\.layers\.(\d+)\.", param_path)
    return int(match.group(1)) if match else None


def _get_gradient_structure(grads: Dict) -> Set[str]:
    """Get flat set of all parameter paths in gradient dict."""
    flat = tree_flatten(grads, is_leaf=lambda x: isinstance(x, mx.array))
    return {".".join(str(p) for p in path) for path, _ in flat}


def _validate_gradient_dict(grads: Dict, context: str = "") -> bool:
    """
    Validate that gradient dict has proper structure for optimizer.

    Returns True if valid, False otherwise (with warnings logged).
    """
    if not grads:
        logger.warning(f"[{context}] Empty gradient dict")
        return False

    try:
        flat = tree_flatten(grads, is_leaf=lambda x: isinstance(x, mx.array))

        array_count = 0
        for path, value in flat:
            if isinstance(value, mx.array):
                array_count += 1
                # Check for NaN/Inf
                if mx.any(mx.isnan(value)) or mx.any(mx.isinf(value)):
                    path_str = ".".join(str(p) for p in path)
                    logger.error(
                        f"[{context}] Invalid gradient at {path_str}: contains NaN/Inf"
                    )
                    return False

        if array_count == 0:
            logger.warning(f"[{context}] No array gradients found")
            return False

        return True

    except Exception as e:
        logger.error(f"[{context}] Error validating gradients: {e}")
        return False


def _create_zero_gradients_like(model) -> Dict:
    """Create zero gradients matching model parameter structure."""
    try:
        # Get model parameters structure
        params = model.parameters()

        def zero_like(x):
            if isinstance(x, mx.array):
                return mx.zeros_like(x)
            return x

        zero_grads = tree_map(
            zero_like, params, is_leaf=lambda x: isinstance(x, mx.array)
        )
        return zero_grads

    except Exception as e:
        logger.error(f"Error creating zero gradients: {e}")
        return {}


def _mask_gradients_by_layers(
    grads: Dict,
    config: ExperimentConfig,
    sft_mode: str,
) -> Dict:
    """
    Mask SFT gradients based on layer configuration.
    Preserves structure and ensures valid output.
    """
    thinking_start = getattr(config.trainer, "thinking_layer_start", None)
    thinking_end = getattr(config.trainer, "thinking_layer_end", None)
    answer_start = getattr(config.trainer, "answer_layer_start", None)
    answer_end = getattr(config.trainer, "answer_layer_end", None)

    if sft_mode == "all" or thinking_start is None or answer_start is None:
        return grads

    thinking_weight = getattr(config.trainer, "sft_thinking_weight", 0.0)
    answer_weight = getattr(config.trainer, "sft_answer_weight", 1.0)

    has_model_wrapper = "model" in grads and isinstance(grads["model"], dict)
    inner_grads = grads["model"] if has_model_wrapper else grads

    def mask_grad(grad, path_parts):
        """Apply mask based on layer number in path."""
        if not isinstance(grad, mx.array):
            return grad

        path_str = ".".join(str(p) for p in path_parts)
        layer_num = _extract_layer_number(path_str)

        if layer_num is None:
            # Non-layer parameters (embeddings, output, etc.) always get full gradient
            return grad

        if sft_mode == "answer_only":
            if answer_start <= layer_num <= answer_end:
                return grad
            else:
                return mx.zeros_like(grad)
        elif sft_mode == "weighted":
            if thinking_start <= layer_num <= thinking_end:
                return grad * thinking_weight
            elif answer_start <= layer_num <= answer_end:
                return grad * answer_weight
            else:
                return grad
        elif sft_mode == "exclude_thinking":
            if thinking_start <= layer_num <= thinking_end:
                return mx.zeros_like(grad)
            else:
                return grad

        return grad

    try:
        flat_inner = tree_flatten(
            inner_grads, is_leaf=lambda x: isinstance(x, mx.array)
        )
        masked_flat = [(path, mask_grad(val, path)) for path, val in flat_inner]
        masked_inner = tree_unflatten(masked_flat)

        result = {"model": masked_inner} if has_model_wrapper else masked_inner

        # Validate output
        if not _validate_gradient_dict(result, "mask_gradients_by_layers"):
            logger.error("Masking produced invalid gradients, returning original")
            return grads

        return result

    except Exception as e:
        logger.error(f"Error masking gradients: {e}", exc_info=True)
        return grads


def _robust_tree_combine(tree1: Any, tree2: Any, fn, path="") -> Tuple[Any, bool]:
    """
    Recursively combine two trees, handling structural mismatches.

    Uses INTERSECTION of keys - only combines where both trees have matching structure.

    Returns:
        (combined_tree, success_flag)
    """
    # Base case: both are arrays
    if isinstance(tree1, mx.array) and isinstance(tree2, mx.array):
        try:
            if tree1.shape != tree2.shape:
                logger.warning(
                    f"Shape mismatch at {path}: "
                    f"tree1={tree1.shape} vs tree2={tree2.shape}. Using tree1."
                )
                return tree1, False

            result = fn(tree1, tree2)

            # Validate result
            if mx.any(mx.isnan(result)) or mx.any(mx.isinf(result)):
                logger.error(
                    f"Invalid result at {path}: NaN/Inf detected. Using tree1."
                )
                return tree1, False

            return result, True

        except Exception as e:
            logger.warning(f"Error combining arrays at {path}: {e}. Using tree1.")
            return tree1, False

    # tree1 is array but tree2 is not - use tree1
    if isinstance(tree1, mx.array):
        if tree2 is not None and not isinstance(tree2, mx.array):
            logger.debug(
                f"Type mismatch at {path}: tree1=array, tree2={type(tree2)}. Using tree1."
            )
        return tree1, False

    # tree1 is dict
    if isinstance(tree1, dict):
        if not isinstance(tree2, dict):
            logger.warning(
                f"Structure mismatch at {path}: tree1=dict, tree2={type(tree2)}. Using tree1."
            )
            return tree1, False

        result = {}
        all_success = True

        # Use INTERSECTION of keys
        common_keys = set(tree1.keys()) & set(tree2.keys())

        if len(common_keys) < len(tree1.keys()):
            missing_in_tree2 = set(tree1.keys()) - common_keys
            logger.debug(
                f"Keys in tree1 but not tree2 at {path}: {missing_in_tree2}. "
                "Using tree1 values for these."
            )

        if len(common_keys) < len(tree2.keys()):
            extra_in_tree2 = set(tree2.keys()) - common_keys
            logger.debug(f"Extra keys in tree2 at {path}: {extra_in_tree2}. Ignoring.")

        # Process all keys from tree1
        for key in tree1.keys():
            new_path = f"{path}.{key}" if path else str(key)

            if key in common_keys:
                # Both have this key, recurse
                result[key], success = _robust_tree_combine(
                    tree1[key], tree2[key], fn, new_path
                )
                all_success = all_success and success
            else:
                # tree2 missing this key, use tree1
                result[key] = tree1[key]
                all_success = False

        return result, all_success

    # tree1 is list/tuple
    if isinstance(tree1, (list, tuple)):
        if not isinstance(tree2, (list, tuple)):
            logger.warning(
                f"Structure mismatch at {path}: tree1=list, tree2={type(tree2)}. Using tree1."
            )
            return tree1, False

        if len(tree1) != len(tree2):
            logger.warning(
                f"Length mismatch at {path}: tree1={len(tree1)}, tree2={len(tree2)}. Using tree1."
            )
            return tree1, False

        result_list = []
        all_success = True

        for i, (v1, v2) in enumerate(zip(tree1, tree2)):
            combined, success = _robust_tree_combine(v1, v2, fn, f"{path}[{i}]")
            result_list.append(combined)
            all_success = all_success and success

        return type(tree1)(result_list), all_success

    # Other types (scalars, None, etc.) - use tree1
    return tree1, False


def _safe_gradient_combine(
    grad1: Dict,
    grad2: Dict,
    operation="add",
    weight1: float = 1.0,
    weight2: float = 1.0,
) -> Tuple[Dict, Dict]:
    """
    Safely combine two gradient dictionaries with bulletproof error handling.

    Args:
        grad1: First gradient dict
        grad2: Second gradient dict
        operation: 'add' or 'subtract'
        weight1: Weight for grad1
        weight2: Weight for grad2

    Returns:
        (combined_grads, metadata_dict) where metadata contains:
            - 'success': bool
            - 'match_rate': float (0-1)
            - 'structure_issues': list of problems
    """
    metadata = {"success": False, "match_rate": 0.0, "structure_issues": []}

    # Validate inputs
    if not grad1:
        logger.warning("grad1 is empty, returning grad2")
        metadata["structure_issues"].append("grad1_empty")
        return grad2 or {}, metadata

    if not grad2:
        logger.warning("grad2 is empty, returning grad1")
        metadata["structure_issues"].append("grad2_empty")
        return grad1, metadata

    # Validate structures
    if not _validate_gradient_dict(grad1, "grad1"):
        metadata["structure_issues"].append("grad1_invalid")
        return grad2 if _validate_gradient_dict(grad2, "grad2") else {}, metadata

    if not _validate_gradient_dict(grad2, "grad2"):
        metadata["structure_issues"].append("grad2_invalid")
        return grad1, metadata

    # Get structures for comparison
    struct1 = _get_gradient_structure(grad1)
    struct2 = _get_gradient_structure(grad2)
    common = struct1 & struct2

    if not common:
        logger.error("No common parameters between grad1 and grad2!")
        metadata["structure_issues"].append("no_common_params")
        metadata["match_rate"] = 0.0
        return grad1, metadata

    match_rate = len(common) / max(len(struct1), len(struct2))
    metadata["match_rate"] = match_rate

    if match_rate < 0.5:
        logger.warning(
            f"Low gradient structure match rate: {match_rate:.1%}. "
            f"Common: {len(common)}, grad1: {len(struct1)}, grad2: {len(struct2)}"
        )
        metadata["structure_issues"].append(f"low_match_rate_{match_rate:.2f}")

    # Define combination function with weights
    if operation == "add":
        combine_fn = lambda a, b: weight1 * a + weight2 * b
    elif operation == "subtract":
        combine_fn = lambda a, b: weight1 * a - weight2 * b
    else:
        logger.error(f"Unknown operation: {operation}. Returning grad1.")
        metadata["structure_issues"].append(f"unknown_operation_{operation}")
        return grad1, metadata

    try:
        # Use robust tree combine
        result, success = _robust_tree_combine(grad1, grad2, combine_fn)

        # Final validation
        if not _validate_gradient_dict(result, "combined"):
            logger.error("Combined gradients are invalid!")
            metadata["structure_issues"].append("invalid_combination")
            return grad1, metadata

        metadata["success"] = success

        if not success:
            logger.warning(
                f"Gradient combination completed with issues (match_rate={match_rate:.1%})"
            )

        return result, metadata

    except Exception as e:
        logger.error(f"Error in gradient combination: {e}", exc_info=True)
        metadata["structure_issues"].append(f"exception_{type(e).__name__}")
        logger.error("Falling back to grad1 only")
        return grad1, metadata


class GRPOAlgorithm:
    def __init__(self, config, actor_model, ref_model):
        self.config = config
        self.actor = actor_model
        self.reference = ref_model
        self.beta = config.trainer.grpo_beta
        self._gradient_stats = {
            "total_combinations": 0,
            "successful_combinations": 0,
            "fallback_count": 0,
        }

    def compute_advantages(self, rewards_flat, samples_per_prompt):
        """Compute advantages with optional baseline normalization."""
        if samples_per_prompt <= 1:
            return (rewards_flat - mx.mean(rewards_flat)) / (
                mx.std(rewards_flat) + 1e-8
            )

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
            tokens_key = "tokens"
            mask_key = "response_mask"

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[mask_key].shape[1]
            shifted_logits = logits[:, offset - 1 : -1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            log_ratio = gathered_log_probs - A["ref_log_probs"]
            kl_term = mx.exp(log_ratio) - 1 - log_ratio
            kl_penalty = kl_term * A[mask_key]

            advantages_expanded = A["advantages"][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * A[mask_key]

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty
            loss = mx.sum(total_loss_per_token) / mx.sum(A[mask_key])

            kl_div = mx.sum(kl_penalty) / mx.sum(A[mask_key])
            policy_loss = mx.sum(policy_loss_term) / mx.sum(A[mask_key])

            return loss, {"kl_divergence": kl_div, "policy_loss": policy_loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)

            # Validate gradients
            if not _validate_gradient_dict(grads, "RL gradients"):
                logger.error("Invalid RL gradients computed!")
                return (
                    mx.array(zero_val),
                    {},
                    {"kl_divergence": zero_val, "policy_loss": zero_val},
                )

            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict

        except Exception as e:
            logger.error(f"Error during loss computation: {e}", exc_info=True)
            return (
                mx.array(zero_val),
                {},
                {"kl_divergence": zero_val, "policy_loss": zero_val},
            )

    def calculate_dual_gradient_loss(self, rollout_batch, full_config, pad_token_id):
        """
        Compute separate gradients for thinking and answer tokens.

        Returns:
            (thinking_loss, thinking_grads, answer_loss, answer_grads, metrics, combination_metadata)
        """
        A = rollout_batch

        has_thinking_mask = "thinking_mask" in A
        has_answer_mask = "answer_mask" in A

        if not has_thinking_mask or not has_answer_mask:
            logger.warning(
                "Thinking/answer masks not found. "
                "Falling back to standard gradient computation."
            )
            loss, grads, metrics = self.calculate_loss_and_grads(
                A, full_config, pad_token_id
            )
            # Return with metadata indicating fallback
            metadata = {
                "mode": "fallback",
                "success": False,
                "structure_issues": ["missing_masks"],
            }
            return loss, grads, loss, grads, metrics, metadata

        def compute_loss_with_mask_and_metrics(
            actor_model, mask_type, compute_metrics=False
        ):
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            shifted_logits = logits[:, offset - 1 : -1, :]
            target_tokens = A[tokens_key][:, offset:]

            log_probs = nn.log_softmax(shifted_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            log_ratio = gathered_log_probs - A["ref_log_probs"]
            kl_term = mx.exp(log_ratio) - 1 - log_ratio

            token_mask = A[mask_type]
            combined_mask = A[response_mask_key] * token_mask

            kl_penalty = kl_term * combined_mask
            advantages_expanded = A["advantages"][:, None]
            policy_loss_term = -log_ratio * advantages_expanded * combined_mask

            total_loss_per_token = policy_loss_term + self.beta * kl_penalty

            mask_sum = mx.sum(combined_mask)
            loss = mx.sum(total_loss_per_token) / (mask_sum + 1e-8)

            if compute_metrics:
                kl_div = mx.sum(kl_penalty) / (mask_sum + 1e-8)
                policy_loss = mx.sum(policy_loss_term) / (mask_sum + 1e-8)
                return loss, {"kl_divergence": kl_div, "policy_loss": policy_loss}

            return loss

        try:
            # Compute thinking gradients
            thinking_loss_fn = lambda model: compute_loss_with_mask_and_metrics(
                model, "thinking_mask", False
            )
            thinking_grads_fn = nn.value_and_grad(self.actor, thinking_loss_fn)
            thinking_loss, thinking_grads = thinking_grads_fn(self.actor)

            # Compute answer gradients
            answer_loss_fn = lambda model: compute_loss_with_mask_and_metrics(
                model, "answer_mask", True
            )
            answer_grads_fn = nn.value_and_grad(self.actor, answer_loss_fn)
            (answer_loss, metrics), answer_grads = answer_grads_fn(self.actor)

            # Validate both gradient sets
            thinking_valid = _validate_gradient_dict(thinking_grads, "thinking_grads")
            answer_valid = _validate_gradient_dict(answer_grads, "answer_grads")

            if not thinking_valid or not answer_valid:
                logger.error("Invalid gradients in dual computation!")
                # Create metadata for failure case
                metadata = {"mode": "dual", "success": False, "structure_issues": []}
                if not thinking_valid:
                    metadata["structure_issues"].append("thinking_invalid")
                if not answer_valid:
                    metadata["structure_issues"].append("answer_invalid")

                # Return fallback
                metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
                return (
                    thinking_loss,
                    thinking_grads,
                    answer_loss,
                    answer_grads,
                    metrics_dict,
                    metadata,
                )

            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}

            # Success metadata
            metadata = {"mode": "dual", "success": True, "structure_issues": []}

            return (
                thinking_loss,
                thinking_grads,
                answer_loss,
                answer_grads,
                metrics_dict,
                metadata,
            )

        except Exception as e:
            logger.error(f"Error in dual gradient computation: {e}", exc_info=True)
            # Return error state
            zero_grads = _create_zero_gradients_like(self.actor)
            metadata = {
                "mode": "dual",
                "success": False,
                "structure_issues": [f"exception_{type(e).__name__}"],
            }
            return (
                mx.array(0.0),
                zero_grads,
                mx.array(0.0),
                zero_grads,
                {"kl_divergence": 0.0, "policy_loss": 0.0},
                metadata,
            )

    def calculate_sft_loss_and_grads(
        self, rollout_batch, reference_tokens, full_config, pad_token_id
    ):
        """Compute SFT loss and gradients with layer-specific control."""
        A = rollout_batch

        sft_mode = getattr(full_config.trainer, "sft_mode", "exclude_thinking")

        if not hasattr(self, "_sft_mode_logged"):
            logger.info(f"SFT layer control mode: {sft_mode}")
            if sft_mode == "exclude_thinking":
                logger.info(
                    "System 2 (thinking) layers will NOT receive SFT gradients - only RL signal"
                )
            elif sft_mode == "answer_only":
                logger.info("Only System 1 (answer) layers will receive SFT gradients")
            elif sft_mode == "weighted":
                thinking_w = getattr(full_config.trainer, "sft_thinking_weight", 0.0)
                answer_w = getattr(full_config.trainer, "sft_answer_weight", 1.0)
                logger.info(f"Weighted SFT: thinking={thinking_w}, answer={answer_w}")
            self._sft_mode_logged = True

        if "answer_mask" not in A:
            logger.warning(
                "Answer mask not found for SFT. Falling back to response_mask."
            )
            answer_mask = A.get(
                "response_mask", mx.ones_like(reference_tokens, dtype=mx.float32)
            )
        else:
            answer_mask = A["answer_mask"]

        def compute_sft_loss(actor_model):
            tokens_key = "tokens"
            response_mask_key = "response_mask"

            logits = actor_model(A[tokens_key])
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.astype(mx.float32)

            offset = A[tokens_key].shape[1] - A[response_mask_key].shape[1]
            response_logits = logits[:, offset - 1 : -1, :]

            min_len = min(response_logits.shape[1], reference_tokens.shape[1])
            response_logits = response_logits[:, :min_len, :]
            target_tokens = reference_tokens[:, :min_len]
            current_answer_mask = (
                answer_mask[:, :min_len]
                if answer_mask.shape[1] >= min_len
                else answer_mask
            )

            if response_logits.shape[1] != target_tokens.shape[1]:
                logger.debug(
                    f"Aligning SFT shapes: "
                    f"logits {response_logits.shape[1]} vs "
                    f"targets {target_tokens.shape[1]} vs "
                    f"mask {current_answer_mask.shape[1]} -> {min_len}"
                )

            log_probs = nn.log_softmax(response_logits, axis=-1)
            gathered_log_probs = mx.take_along_axis(
                log_probs, target_tokens[..., None], axis=-1
            ).squeeze(-1)

            masked_log_probs = -gathered_log_probs * current_answer_mask

            mask_sum = mx.sum(current_answer_mask)
            loss = mx.sum(masked_log_probs) / (mask_sum + 1e-8)

            return loss, {"sft_loss": loss}

        try:
            loss_grad_fn = nn.value_and_grad(self.actor, compute_sft_loss)
            (loss, metrics), grads = loss_grad_fn(self.actor)

            # Validate before masking
            if not _validate_gradient_dict(grads, "SFT gradients (pre-mask)"):
                logger.error("Invalid SFT gradients before masking!")
                return mx.array(0.0), {}, {"sft_loss": 0.0}

            # Apply layer-specific masking
            grads = _mask_gradients_by_layers(grads, full_config, sft_mode)

            # Validate after masking
            if not _validate_gradient_dict(grads, "SFT gradients (post-mask)"):
                logger.error("Invalid SFT gradients after masking!")
                return mx.array(0.0), {}, {"sft_loss": 0.0}

            metrics_dict = {k: float(v.item()) for k, v in metrics.items()}
            return loss, grads, metrics_dict

        except Exception as e:
            logger.error(f"Error during SFT loss computation: {e}", exc_info=True)
            return mx.array(0.0), {}, {"sft_loss": 0.0}

    def get_gradient_statistics(self) -> Dict:
        """Get statistics about gradient combinations."""
        stats = self._gradient_stats.copy()
        if stats["total_combinations"] > 0:
            stats["success_rate"] = (
                stats["successful_combinations"] / stats["total_combinations"]
            )
        else:
            stats["success_rate"] = 0.0
        return stats
