# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/mlx_utils.py
# revision_no: 002
# goals_of_writing_code_block: A collection of MLX-specific utility functions for memory management, gradient manipulation, and dynamic logit processing.
# type_of_code_response: replace
"""MLX-specific utility functions."""

import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import re
import string
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from pathlib import Path

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_rl_trainer.core.config import ExperimentConfig, GenerationConfig
from mlx_rl_trainer.core.exceptions import CheckpointError
import sys

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:

    class MLXLoRALinear:
        pass


logger = logging.getLogger(__name__)

TARGET_FLOAT_DTYPE = mx.bfloat16
LETTER_ALPH = string.ascii_uppercase
_TOOL_LIKE_MARKERS = [
    "<tool_call",
    "</tool_call",
    "<tool>",
    "</tool>",
    "<tool_",
    "<function",
    "</function",
    "<json",
    "</json",
    "<scratchpad",
    "</scratchpad",
]


def limit_memory(max_memory_gb: float) -> Optional[int]:
    """Sets the MLX memory limit using updated API with error handling."""
    try:
        if hasattr(mx, "set_memory_limit"):
            previous_limit = mx.set_memory_limit(int(max_memory_gb * 1024**3))
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB. Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            return previous_limit
        elif hasattr(mx.metal, "set_memory_limit"):  # Check older API location
            previous_limit = mx.metal.set_memory_limit(int(max_memory_gb * 1024**3))
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB (using mx.metal). Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            return previous_limit
        else:
            logging.warning(
                "mx.set_memory_limit() not found in this MLX version. Cannot limit memory."
            )
            return None
    except Exception as e:
        logging.error(f"Failed to set MLX memory limit: {e}", exc_info=True)
        return None


def _is_metal_internal_error(err: BaseException) -> bool:
    s = str(err)
    return (
        ("Command buffer execution failed" in s)
        or ("[METAL]" in s)
        or ("Internal Error" in s)
    )


def metal_recover(stage: str):
    logging.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()


if not hasattr(optim.Optimizer, "_orig_apply_gradients_mlx_metal_patch"):
    optim.Optimizer._orig_apply_gradients_mlx_metal_patch = (
        optim.Optimizer.apply_gradients
    )

    def _apply_gradients_metal_safe(self, grads, params):
        try:
            return optim.Optimizer._orig_apply_gradients_mlx_metal_patch(
                self, grads, params
            )
        except Exception as e:
            if _is_metal_internal_error(e):
                metal_recover("apply_gradients")
                return
            raise
        finally:
            try:
                mx.synchronize()
            except Exception:
                pass
            mx.clear_cache()
            gc.collect()

    optim.Optimizer.apply_gradients = _apply_gradients_metal_safe


def metal_before_update(num_updates: int, data_config: Any, trainer_config: Any):
    if not hasattr(trainer_config, "_orig_max_gen_len"):
        trainer_config._orig_max_gen_len = int(getattr(data_config, "max_gen_len", 160))
        trainer_config._orig_num_samples = int(
            getattr(trainer_config, "num_rollout_samples", 4)
        )

    # max_kv_size is not directly in config, so we'll manage it separately or assume it's handled elsewhere
    # For now, we'll just use the original values if not explicitly set.
    # If it needs to be dynamically adjusted, it should be part of a config object.

    if num_updates < 32:
        data_config.max_gen_len = min(trainer_config._orig_max_gen_len, 160)
        trainer_config.num_rollout_samples = min(trainer_config._orig_num_samples, 4)
    else:
        data_config.max_gen_len = trainer_config._orig_max_gen_len
        trainer_config.num_rollout_samples = trainer_config._orig_num_samples

    if (num_updates % 5) == 0:
        try:
            mx.synchronize()
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()


def make_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    gen_cfg = config.generation
    tag_ids = _resolve_tag_ids(tokenizer, gen_cfg)
    mcq_letter_ids = sorted(set(sum(_letter_token_ids(tokenizer).values(), [])))
    ban_ids = _first_token_ids_for_lexemes(tokenizer, gen_cfg.ban_phrases_for_bias)

    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    B_CLOSE, B_AS, P_REOPEN_THINK, P_EXTRA_TE, P_REOPEN_ANS, B_EOS_ANS = (
        gen_cfg.bias_close_think,
        gen_cfg.bias_answer_start,
        gen_cfg.punish_reopen_think,
        gen_cfg.punish_extra_think_end,
        gen_cfg.punish_reopen_answer,
        gen_cfg.bias_eos_after_answer,
    )
    MIN_ANS, MIN_ANS_MCQ, HARD_MASK, LIFT_MCQ, BAN_MCQ, BAN_NONMCQ = (
        gen_cfg.min_answer_tokens,
        gen_cfg.min_answer_tokens_mcq,
        gen_cfg.hard_mask_mcq_first_token,
        gen_cfg.mcq_letter_lift,
        gen_cfg.mcq_ban_first_bias,
        gen_cfg.nonmcq_ban_first_bias,
    )
    B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK = (
        gen_cfg.mcq_answer_end_bias,
        gen_cfg.min_think_tokens,
        gen_cfg.think_end_early_bias,
        gen_cfg.bias_answer_start_after_min_think,
    )

    def _proc_vectorized(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        if logits.ndim != 2:
            return logits
        B, V = logits.shape
        neg_inf, pad_id = mx.array(-1e9, dtype=logits.dtype), tokenizer.pad_token_id
        max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
        if max_hist_len == 0:
            return logits

        history_mx = mx.array(
            [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list],
            dtype=mx.int32,
        )

        def find_last_pos_mx(tag_id):
            if tag_id is None:
                return mx.full((B,), -1, dtype=mx.int32)
            matches = history_mx == tag_id
            rev_indices = mx.argmax(matches[:, ::-1], axis=1).astype(mx.int32)
            return mx.where(mx.any(matches, axis=1), max_hist_len - 1 - rev_indices, -1)

        last_ts, last_te, last_as, last_ae = (
            find_last_pos_mx(t) for t in (ts, te, as_id, ae)
        )
        history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)

        inside_think = mx.logical_and(
            last_ts != -1, mx.logical_and(last_te < last_ts, last_as < last_ts)
        )
        inside_answer = mx.logical_and(last_as != -1, last_ae < last_as)
        ae_seen = last_ae != -1
        k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
        k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)
        is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)

        if ts is not None and te is not None:
            logits = logits.at[:, ts].add(mx.where(last_te != -1, P_REOPEN_THINK, 0.0))
            if as_id is not None:
                logits = logits.at[:, as_id].add(
                    mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)
                )
            te_count = mx.sum(history_mx == te, axis=1)
            bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
            min_think_penalty_mask = mx.logical_and(inside_think, (k_think < MIN_THINK))
            bias_at_te = mx.where(min_think_penalty_mask, B_END_EARLY, bias_at_te)
            logits = logits.at[:, te].add(bias_at_te)
            can_start_answer = mx.logical_and(
                last_te > last_as, mx.logical_not(inside_answer)
            )
            min_think_ok = mx.logical_not(B_AS_MIN_THINK)
            if B_AS_MIN_THINK:
                min_think_ok = k_think >= MIN_THINK
            can_start_answer = mx.logical_and(can_start_answer, min_think_ok)
            if as_id is not None:
                logits = logits.at[:, as_id].add(mx.where(can_start_answer, B_AS, 0.0))

        if eos_tok is not None:
            logits = logits.at[:, eos_tok].add(mx.where(ae_seen, B_EOS_ANS, 0.0))

        mcq_first_token_mask = mx.logical_and(
            is_mcq_mask, mx.logical_and(inside_answer, (k_answer == 0))
        )
        if mx.any(mcq_first_token_mask).item() and HARD_MASK:
            mcq_allowed_logits = mx.full((V,), neg_inf, dtype=logits.dtype)
            if mcq_letter_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].add(LIFT_MCQ)
            if ban_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].add(BAN_MCQ)
            logits = mx.where(
                mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits
            )

        non_mcq_first_answer = mx.logical_and(
            mx.logical_not(is_mcq_mask), mx.logical_and(inside_answer, (k_answer == 0))
        )
        if ban_ids and BAN_NONMCQ != 0 and mx.any(non_mcq_first_answer).item():
            ban_bias = mx.zeros_like(logits)
            ban_bias = ban_bias.at[:, ban_ids].add(BAN_NONMCQ)
            logits = logits + (ban_bias * non_mcq_first_answer[:, None])

        if ae is not None:
            min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
            min_len_penalty_mask = mx.logical_and(
                inside_answer, (k_answer < min_ans_len)
            )
            logits = logits.at[:, ae].add(mx.where(min_len_penalty_mask, -8.0, 0.0))
            mcq_close_mask = mx.logical_and(
                is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= 1))
            )
            logits = logits.at[:, ae].add(mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0))

        return logits

    return _proc_vectorized


def _resolve_tag_ids(
    tokenizer: TokenizerWrapper, gen_config: GenerationConfig
) -> Dict[str, Optional[int]]:
    def _one_id(tok_str):
        if not tok_str:
            return None
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            return int(ids[0]) if len(ids) == 1 else None
        except Exception:
            return None

    return {
        "think_start": _one_id(gen_config.think_start_tag),
        "think_end": _one_id(gen_config.think_end_tag),
        "answer_start": _one_id(gen_config.answer_start_tag),
        "answer_end": _one_id(gen_config.answer_end_tag),
        "eos": tokenizer.eos_token_id,
    }


def _first_token_ids_for_lexemes(
    tokenizer: TokenizerWrapper, lexemes: Sequence[str]
) -> List[int]:
    ids: List[int] = []
    for lx in lexemes:
        if (
            (t := tokenizer.encode(lx, add_special_tokens=False))
            and t
            and t[0] not in ids
        ):
            ids.append(t[0])
        if (
            (t_space := tokenizer.encode(" " + lx, add_special_tokens=False))
            and t_space
            and t_space[0] not in ids
        ):
            ids.append(t_space[0])
    return ids


def _letter_token_ids(
    tokenizer: TokenizerWrapper, letters: Sequence[str] = LETTER_ALPH
) -> Dict[str, List[int]]:
    out = {}
    for L in letters:
        cand = []
        for suf in ["", " ", ")", ".", " )", " ."]:
            ids = tokenizer.encode(L + suf, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cand:
                cand.append(ids[0])
        out[L] = cand
    return out


def safe_make_sampler(
    config_or_args: Union[ExperimentConfig, GenerationConfig], temp: float
) -> Callable:
    gen_cfg = (
        config_or_args.generation
        if isinstance(config_or_args, ExperimentConfig)
        else config_or_args
    )
    try:
        return make_sampler(
            temp=temp,
            top_p=gen_cfg.sampling_top_p,
            min_p=gen_cfg.sampling_min_p,
            top_k=gen_cfg.sampling_top_k,
        )
    except TypeError:
        return make_sampler(temp=temp, top_p=gen_cfg.sampling_top_p)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """Extract layer number from parameter path."""
    import re

    match = re.search(r"\.layers\.(\d+)\.", param_path)
    return int(match.group(1)) if match else None


def mask_gradients_by_layer(
    grads: Dict[str, mx.array], layer_config: Dict[str, List[str]]
) -> Dict[str, mx.array]:
    masked_grads = {}
    include_layers = [str(x) for x in layer_config.get("include", [])]
    exclude_layers = [str(x) for x in layer_config.get("exclude", [])]

    for key, grad in tree_flatten(grads):
        layer_num = _extract_layer_number(key)

        if layer_num is None:
            # Keep non-layer parameters as is
            masked_grads[key] = grad
        else:
            if (not include_layers or str(layer_num) in include_layers) and (
                str(layer_num) not in exclude_layers
            ):
                masked_grads[key] = grad
            else:
                masked_grads[key] = mx.zeros_like(grad)
    return tree_unflatten(list(masked_grads.items()))


def combine_gradients(
    gradient_tuples: List[Tuple[Dict[str, mx.array], float]]
) -> Dict[str, mx.array]:
    combined_grads = {}
    for grads, weight in gradient_tuples:
        for key, grad in tree_flatten(grads):
            if key not in combined_grads:
                combined_grads[key] = grad * weight
            else:
                combined_grads[key] += grad * weight
    return tree_unflatten(list(combined_grads.items()))


def get_grad_norm(grads: Dict[str, mx.array]) -> float:
    """Calculates the L2 norm of the gradients."""
    total_norm = 0.0
    for _, grad in tree_flatten(grads):
        if isinstance(grad, mx.array):
            total_norm += mx.sum(grad * grad).item()
    return total_norm**0.5
