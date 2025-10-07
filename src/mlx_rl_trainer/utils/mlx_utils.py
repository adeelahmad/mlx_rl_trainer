# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/mlx_utils.py
# revision_no: 004
# goals_of_writing_code_block: Correct the TypeError in the `make_dynamic_tag_bias_processor` by using `mx.logical_*` functions for boolean array operations instead of Python's bitwise operators.
# type_of_code_response: change existing
"""MLX-specific utility functions."""

import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import re
import random
import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from pathlib import Path

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.trainer import CheckpointError
from mlx_rl_trainer.generation.caching import PagedKVCache
from mlx_rl_trainer.core.config import RewardConfig  # For _resolve_tag_ids

logger = logging.getLogger(__name__)

# --- MLX Global Config & Constants ---
TARGET_FLOAT_DTYPE = mx.bfloat16
MIN_REQUIRED_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
SAVE_ON_EXIT_FLAG_PATH = Path(".save_on_exit_request")
LETTER_ALPH = string.ascii_uppercase


# --- MLX-Specific Errors and Recovery (Unchanged) ---
def _is_metal_internal_error(err: BaseException) -> bool:
    s = str(err)
    return (
        "Command buffer execution failed" in s
        or "[METAL]" in s in s
        or "Internal Error" in s
    )


def metal_recover(stage: str):
    logger.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()


def metal_safe_apply_gradients(
    optimizer: optim.Optimizer, grads: Dict[str, mx.array], params: Dict[str, mx.array]
):
    try:
        return optimizer.apply_gradients(grads, params)
    except Exception as e:
        if _is_metal_internal_error(e):
            metal_recover("apply_gradients")
            return None
        raise
    finally:
        mx.clear_cache()
        gc.collect()


# --- Gradient Manipulation (Unchanged) ---
_LAYER_PAT = re.compile(r"(?:^|[^a-zA-Z0-9_])layers\.(\d+)(?:[^0-9_]|$)")
_HEAD_PAT = re.compile(r"\blm_head\b", re.I)


def _find_layer_index(name: str) -> Optional[int]:
    m = _LAYER_PAT.search(name)
    if m:
        return int(m.group(1))
    parts = re.split(r"[\.\/]", name)
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                pass
    return None


def _band_for_name(
    name: str,
    low_band: Tuple[int, int],
    mid_band: Tuple[int, int],
    top_band: Tuple[int, int],
) -> str:
    li = _find_layer_index(name)

    def _in_range(li: int, band_range: Optional[Tuple[int, int]]) -> bool:
        if band_range is None:
            return False
        try:
            s, e = band_range
            return (s is None or li >= int(s)) and (e is None or li <= int(e))
        except Exception:
            return False

    if li is not None:
        if _in_range(li, low_band):
            return "low"
        if _in_range(li, mid_band):
            return "mid"
        if _in_range(li, top_band):
            return "top"
    if _HEAD_PAT.search(name):
        return "head"
    return "other"


def scale_grads_by_band(
    grads_tree: Dict[str, mx.array], config: ExperimentConfig
) -> Dict[str, mx.array]:
    low_mul, mid_mul, top_mul, head_mul = (
        config.trainer.low_mul,
        config.trainer.mid_mul,
        config.trainer.top_mul,
        config.trainer.head_mul,
    )
    g_flat = tree_flatten(grads_tree)
    out = []
    for name, g in g_flat:
        if not isinstance(g, mx.array):
            out.append((name, g))
            continue
        band = _band_for_name(
            name,
            config.trainer.low_band,
            config.trainer.mid_band,
            config.trainer.top_band,
        )
        mul = {"low": low_mul, "mid": mid_mul, "top": top_mul, "head": head_mul}.get(
            band, 1.0
        )
        out.append((name, g * mul))
    return tree_unflatten(out)


# def mask_grads_to_layer_band(grads_tree: Dict[str, mx.array], start: Optional[int], end: Optional[int], *, include_embed: bool = True, include_head: bool = True, include_final_norm: bool = True) -> Dict[str, mx.array]:
#     flat = tree_flatten(grads_tree)
#     kept = []
#     for name, g in flat:
#         if not isinstance(g, mx.array): kept.append((name, g)); continue
#         li = _find_layer_index(name)
#         keep = False
#         if li is not None:
#             keep = (start is None or li >= start) and (end is None or li <= end)
#         else:
#             lname = name.lower()
#             if "embed" in lname or "token_embedding" in lname or "word_embedding" in lname: keep = include_embed
#             elif "final_norm" in lname or "norm_out" in lname or "ln_f" in lname or "final_layer_norm" in lname: keep = include_final_norm
#             elif "lm_head" in lname or ("output" in lname and "head" in lname) or "logits" in lname: keep = include_head
#         kept.append((name, g if keep else mx.zeros_like(g)))
#     return tree_unflatten(kept)

# def mask_grads_to_specific_layers(grads_tree: Dict[str, mx.array], layer_indices: Set[int]) -> Dict[str, mx.array]:
#     flat = tree_flatten(grads_tree)
#     kept = []
#     for name, g in flat:
#         if not isinstance(g, mx.array): kept.append((name, g)); continue
#         if (layer_idx := _find_layer_index(name)) is not None and layer_idx in layer_indices: kept.append((name, g))
#         else: kept.append((name, mx.zeros_like(g)))
#     return tree_unflatten(kept)


def mask_grads_to_layer_band(
    grads_tree: Dict[str, mx.array],
    start: Optional[int],
    end: Optional[int],
    *,
    include_embed: bool = True,
    include_head: bool = True,
    include_final_norm: bool = True,
) -> Dict[str, mx.array]:
    flat = tree_flatten(grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue
        li = _find_layer_index(name)
        keep = False
        if li is not None:
            keep = (start is None or li >= start) and (end is None or li <= end)
        else:
            lname = name.lower()
            if (
                "embed" in lname
                or "token_embedding" in lname
                or "word_embedding" in lname
            ):
                keep = include_embed
            elif (
                "final_norm" in lname
                or "norm_out" in lname
                or "ln_f" in lname
                or "final_layer_norm" in lname
            ):
                keep = include_final_norm
            elif (
                "lm_head" in lname
                or "output" in lname
                and "head" in lname
                or "logits" in lname
            ):
                keep = include_head
        kept.append((name, g if keep else mx.zeros_like(g)))
    return tree_unflatten(kept)


def mask_grads_to_specific_layers(
    grads_tree: Dict[str, mx.array], layer_indices: Set[int]
) -> Dict[str, mx.array]:
    flat = tree_flatten(grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue
        if (
            layer_idx := _find_layer_index(name)
        ) is not None and layer_idx in layer_indices:
            kept.append((name, g))
        else:
            kept.append((name, mx.zeros_like(g)))
    return tree_unflatten(kept)


# --- Attention Masking (Unchanged) ---


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE
) -> mx.array:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")
    B, T = tokens.shape
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T, dtype=dtype)
    padding_mask = (tokens == pad_token_id)[:, None, None, :]
    neg_inf = mx.array(-1e9, dtype=dtype)
    combined = mx.minimum(causal_mask, mx.where(padding_mask, neg_inf, 0.0))
    return combined


# --- Sampling (Unchanged) ---


def safe_make_sampler(config: ExperimentConfig, temp: float) -> Callable:
    top_p = float(config.generation.top_p)
    # Corrected attribute access
    min_p_val = (
        getattr(config.generation, "min_p", 0.0)
        if hasattr(config.generation, "min_p")
        else 0.0
    )
    top_k_val = int(config.generation.top_k)

    try:
        return make_sampler(temp=temp, top_p=top_p, min_p=min_p_val, top_k=top_k_val)
    except TypeError:
        try:
            return make_sampler(temp=temp, top_p=top_p, min_p=min_p_val)
        except TypeError:
            return make_sampler(temp=temp, top_p=top_p)
        except Exception as e:
            logger.error(f"Failed to create sampler: {e}. Falling back to greedy.")
            return make_sampler(temp=0.0, top_p=1.0)


# --- Logit Processors Helper Functions (DEFINITIONS ADDED/CORRECTED HERE) ---

_TOOL_LIKE_MARKERS = [
    "<tool_code>",
    "</tool_code>",
    "<json_output>",
    "</json_output>",
    "<img_base64>",
    "</img_base64>",
    "<ocr_text>",
    "</ocr_text>",
]


def _first_token_ids_for_lexemes(
    tokenizer: TokenizerWrapper, lexemes: Sequence[str]
) -> List[int]:
    ids: List[int] = []
    for lx in lexemes:
        if (t := tokenizer.encode(lx, add_special_tokens=False)) and t[0] not in ids:
            ids.append(t[0])
        if (
            t_space := tokenizer.encode(" " + lx, add_special_tokens=False)
        ) and t_space[0] not in ids:
            ids.append(t_space[0])
    return ids


def _letter_token_ids(
    tokenizer: TokenizerWrapper, letters: Sequence[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) -> Dict[str, List[int]]:
    out = {}
    for L in letters:
        cand = []
        if (ids := tokenizer.encode(L, add_special_tokens=False)) and len(ids) == 1:
            cand.append(ids[0])
        if (
            (ids := tokenizer.encode(" " + L, add_special_tokens=False))
            and len(ids) == 1
            and ids[0] not in cand
        ):
            cand.append(ids[0])
        for suf in [")", ".", " )", " ."]:
            if (
                (ids := tokenizer.encode(L + suf, add_special_tokens=False))
                and len(ids) == 1
                and ids[0] not in cand
            ):
                cand.append(ids[0])
        out[L] = cand
    return out


def _resolve_tag_ids(
    tokenizer: TokenizerWrapper, config: ExperimentConfig
) -> Dict[str, Optional[int]]:
    def _one_id(tok_str):
        if not tok_str:
            return None
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    tags = config.generation
    return {
        "think_start": _one_id(tags.think_start_tag),
        "think_end": _one_id(tags.think_end_tag),
        "answer_start": _one_id(tags.answer_start_tag),
        "answer_end": _one_id(tags.answer_end_tag),
        "eos": tokenizer.eos_token_id,
    }


def make_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    tag_ids = _resolve_tag_ids(tokenizer, config)
    letter_map = _letter_token_ids(tokenizer, letters=LETTER_ALPH)
    mcq_letter_ids = sorted(set(sum(letter_map.values(), [])))
    gen_cfg = config.generation
    ban_ids = _first_token_ids_for_lexemes(tokenizer, gen_cfg.ban_phrases_for_bias)
    encourage_ids = _first_token_ids_for_lexemes(
        tokenizer, gen_cfg.encourage_phrases_for_bias
    )
    tool_ids = _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS)
    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )

    # Extract biases from GenerationConfig
    B_CLOSE, B_AS = gen_cfg.bias_close_think, gen_cfg.bias_answer_start
    P_REOPEN_THINK, P_EXTRA_TE = (
        gen_cfg.punish_reopen_think,
        gen_cfg.punish_extra_think_end,
    )
    P_REOPEN_ANS, B_EOS_ANS = (
        gen_cfg.punish_reopen_answer,
        gen_cfg.bias_eos_after_answer,
    )
    MIN_ANS, MIN_ANS_MCQ = gen_cfg.min_answer_tokens, gen_cfg.min_answer_tokens_mcq
    HARD_MASK, LIFT_MCQ = gen_cfg.hard_mask_mcq_first_token, gen_cfg.mcq_letter_lift
    BAN_MCQ, BAN_NONMCQ = gen_cfg.mcq_ban_first_bias, gen_cfg.nonmcq_ban_first_bias
    MCQ_CLOSE_K, B_MCQ_CLOSE = gen_cfg.mcq_close_after_k, gen_cfg.mcq_answer_end_bias
    MIN_THINK, B_END_EARLY = gen_cfg.min_think_tokens, gen_cfg.think_end_early_bias
    B_AS_MIN_THINK = gen_cfg.bias_answer_start_after_min_think
    B_ENCOURAGE = gen_cfg.encourage_think_bias
    P_TOOL = gen_cfg.tool_call_penalty * -100

    def _proc_vectorized(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        if logits is None or logits.ndim != 2:
            return logits
        B, V = logits.shape
        dtype = logits.dtype
        neg_inf = mx.array(-1e9, dtype=dtype)
        max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
        if max_hist_len == 0:
            return logits

        pad_id = tokenizer.pad_token_id
        history_mx = mx.array(
            [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list],
            dtype=mx.int32,
        )
        if tool_ids:
            logits[:, tool_ids] += P_TOOL

        def find_last_pos_mx(tag_id):
            if tag_id is None:
                return mx.full((B,), -1, dtype=mx.int32)
            matches = history_mx == tag_id
            has_match = mx.any(matches, axis=1)
            indices = mx.arange(max_hist_len)
            masked_indices = mx.where(matches, indices, -1)
            last_indices = mx.max(masked_indices, axis=1)
            return mx.where(has_match, last_indices, -1)

        last_ts, last_te, last_as, last_ae = (
            find_last_pos_mx(ts),
            find_last_pos_mx(te),
            find_last_pos_mx(as_id),
            find_last_pos_mx(ae),
        )
        history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)
        inside_think = (last_ts != -1) & (last_te < last_ts) & (last_as < last_ts)
        inside_answer = (last_as != -1) & (last_ae < last_as)
        ae_seen = last_ae != -1
        k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
        k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)
        is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)

        if ts is not None and te is not None:
            te_count = mx.sum(history_mx == te, axis=1)
            logits[:, ts] += mx.where(last_te != -1, P_REOPEN_THINK, 0.0)
            if as_id is not None:
                logits[:, as_id] += mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)
            bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
            min_think_penalty = mx.logical_and(inside_think, (k_think < MIN_THINK))
            bias_at_te = mx.where(min_think_penalty, B_END_EARLY, bias_at_te)
            logits[:, te] += bias_at_te

            can_start_answer = mx.logical_and(
                (last_te > last_as), mx.logical_not(inside_answer)
            )
            # FIX: Use mx.logical_or and mx.logical_not for boolean array logic.
            if not B_AS_MIN_THINK:
                # If the flag is false, the min_think_ok condition is ignored (always true)
                min_think_ok = mx.full_like(can_start_answer, True)
            else:
                min_think_ok = k_think >= MIN_THINK
            can_start_answer = mx.logical_and(can_start_answer, min_think_ok)

            if as_id is not None:
                logits[:, as_id] += mx.where(can_start_answer, B_AS, 0.0)

        if eos_tok is not None:
            logits[:, eos_tok] += mx.where(ae_seen, B_EOS_ANS, 0.0)
        if encourage_ids and inside_think.any():
            logits[inside_think.tolist(), encourage_ids] += B_ENCOURAGE

        mcq_first_token_mask = mx.logical_and(
            is_mcq_mask, mx.logical_and(inside_answer, (k_answer == 0))
        )
        if mx.any(mcq_first_token_mask).item() and HARD_MASK:
            mcq_allowed_logits = mx.full((V,), neg_inf, dtype=dtype)
            if mcq_letter_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].set(LIFT_MCQ)
            if ban_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].add(BAN_MCQ)
            logits = mx.where(
                mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits
            )

        non_mcq_first_answer = mx.logical_and(
            mx.logical_not(is_mcq_mask), mx.logical_and(inside_answer, (k_answer == 0))
        )
        if mx.any(non_mcq_first_answer).item() and ban_ids:
            logits[non_mcq_first_answer.tolist(), ban_ids] += BAN_NONMCQ

        if ae is not None:
            min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
            min_len_penalty_mask = mx.logical_and(
                inside_answer, (k_answer < min_ans_len)
            )
            logits[:, ae] += mx.where(min_len_penalty_mask, -8.0, 0.0)
            mcq_close_mask = mx.logical_and(
                is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= MCQ_CLOSE_K))
            )
            logits[:, ae] += mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0)

        return logits

    return _proc_vectorized


def _mask_after_answer(
    responses_mx: mx.array,
    initial_mask: mx.array,
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
) -> mx.array:
    if responses_mx.ndim != 2:
        return initial_mask
    B, L_gen = responses_mx.shape
    tag_ids = _resolve_tag_ids(tokenizer, config)
    answer_end_id = tag_ids.get("answer_end")
    if answer_end_id is None:
        return initial_mask
    indices = mx.arange(L_gen)
    is_answer_end = responses_mx == answer_end_id
    first_end_indices = mx.argmin(mx.where(is_answer_end, indices, L_gen + 1), axis=1)
    boundary_index = first_end_indices + 1
    end_mask = (
        mx.broadcast_to(indices[None, :], responses_mx.shape) < boundary_index[:, None]
    )
    return initial_mask.astype(mx.float32) * end_mask.astype(mx.float32)


class ContentAlignBridge(nn.Module):
    pass
