# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/mlx_utils.py
# revision_no: 002
# goals_of_writing_code_block: MLX-specific utility functions, including Metal safety, gradient manipulation, and sampling/logit processing.
# type_of_code_response: change existing
"""MLX-specific utility functions."""

import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gc
import re
import random
import string # For LETTER_ALPH
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from pathlib import Path

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_rl_trainer.core.config import ExperimentConfig # Using ExperimentConfig now
from mlx_rl_trainer.core.trainer import CheckpointError
from mlx_rl_trainer.generation.caching import PagedKVCache # Using PagedKVCache from its new location
# from mlx_rl_trainer.rewards.context import RewardContext # Not directly used here, but in calling functions
from mlx_rl_trainer.core.config import RewardConfig # For RewardConfig type hinting (as StaticRewardConfig)

logger = logging.getLogger(__name__)

# --- MLX Global Config & Constants ---
TARGET_FLOAT_DTYPE = mx.bfloat16
MIN_REQUIRED_BYTES = 2 * 1024 * 1024 * 1024 # 2GB
SAVE_ON_EXIT_FLAG_PATH = Path(".save_on_exit_request")
LETTER_ALPH = string.ascii_uppercase # For MCQ letters

# --- MLX-Specific Errors and Recovery ---

def _is_metal_internal_error(err: BaseException) -> bool:
    """Checks if an exception string indicates a recoverable Metal internal error."""
    s = str(err)
    return 'Command buffer execution failed' in s or '[METAL]' in s in s or 'Internal Error' in s

def metal_recover(stage: str):
    """Attempts to recover the Metal device state after a crash."""
    logger.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()

def metal_safe_apply_gradients(optimizer: optim.Optimizer, grads: Dict[str, mx.array], params: Dict[str, mx.array]):
    """Wrapper for optimizer.apply_gradients to catch and recover from Metal errors."""
    try:
        return optimizer.apply_gradients(grads, params)
    except Exception as e:
        if _is_metal_internal_error(e):
            metal_recover('apply_gradients')
            # Return None or empty grads/params to allow training to continue without crashing
            return None
        raise
    finally:
        # Ensure cache is cleared and GC run, even on success
        mx.clear_cache()
        gc.collect()


# --- Gradient Manipulation ---

_LAYER_PAT = re.compile(r"(?:^|[^a-zA-Z0-9_])layers\.(\d+)(?:[^0-9_]|$)")
_HEAD_PAT = re.compile(r"\blm_head\b", re.I)

def _find_layer_index(name: str) -> Optional[int]:
    """Extracts the layer index from a parameter name (e.g., 'layers.12.attn.q_proj')."""
    m = _LAYER_PAT.search(name)
    if m: return int(m.group(1))

    parts = re.split(r"[\.\/]", name)
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try: return int(parts[i+1])
            except Exception: pass
    return None

def _band_for_name(name: str, low_band: Tuple[int, int], mid_band: Tuple[int, int], top_band: Tuple[int, int]) -> str:
    """Determines which band (low, mid, top, head, other) a parameter belongs to."""
    li = _find_layer_index(name)

    def _in_range(li: int, band_range: Optional[Tuple[int, int]]) -> bool:
        if band_range is None: return False
        try: s, e = band_range; return (s is None or li >= int(s)) and (e is None or li <= int(e))
        except Exception: return False

    if li is not None:
        if _in_range(li, low_band): return 'low'
        if _in_range(li, mid_band): return 'mid'
        if _in_range(li, top_band): return 'top'

    if _HEAD_PAT.search(name): return 'head'

    return 'other'

def scale_grads_by_band(grads_tree: Dict[str, mx.array], params_tree: Dict[str, mx.array], config: ExperimentConfig) -> Dict[str, mx.array]:
    """Applies gradient multipliers based on layer index bands defined in ExperimentConfig."""
    low_band = config.trainer.low_band
    mid_band = config.trainer.mid_band
    top_band = config.trainer.top_band
    low_mul = float(config.trainer.low_mul)
    mid_mul = float(config.trainer.mid_mul)
    top_mul = float(config.trainer.top_mul)
    head_mul = float(config.trainer.head_mul)
    other_mul = 1.0 # Default fallback

    g_flat = tree_flatten(grads_tree)
    out = []

    for name, g in g_flat:
        if not isinstance(g, mx.array): out.append((name, g)); continue

        band = _band_for_name(name, low_band, mid_band, top_band)
        mul = {'low': low_mul, 'mid': mid_mul, 'top': top_mul, 'head': head_mul}.get(band, other_mul)
        out.append((name, g * mul))

    return tree_unflatten(out)

def mask_grads_to_layer_band(grads_tree: Dict[str, mx.array], start: Optional[int], end: Optional[int], *,
                             include_embed: bool = True, include_head: bool = True, include_final_norm: bool = True) -> Dict[str, mx.array]:
    """Masks gradients to zero for layers outside the specified range."""
    flat = tree_flatten(grads_tree)
    kept = []

    for name, g in flat:
        if not isinstance(g, mx.array): kept.append((name, g)); continue

        li = _find_layer_index(name)
        keep = False

        if li is not None:
            in_band = (start is None or li >= start) and (end is None or li <= end)
            keep = in_band
        else: # Non-layer specific params
            lname = name.lower()
            if 'embed' in lname or 'token_embedding' in lname or 'word_embedding' in lname:
                keep = include_embed
            elif 'final_norm' in lname or 'norm_out' in lname or 'ln_f' in lname or 'final_layer_norm' in lname:
                keep = include_final_norm
            elif 'lm_head' in lname or 'output' in lname and 'head' in lname or 'logits' in lname:
                keep = include_head

        kept.append((name, g if keep else mx.zeros_like(g)))

    return tree_unflatten(kept)

def mask_grads_to_specific_layers(grads_tree: Dict[str, mx.array], layer_indices: Set[int]) -> Dict[str, mx.array]:
    """Masks gradients to zero unless they belong to a specific set of layer indices."""
    flat = tree_flatten(grads_tree)
    kept = []

    for name, g in flat:
        if not isinstance(g, mx.array): kept.append((name, g)); continue

        layer_idx = _find_layer_index(name)

        if layer_idx is not None and layer_idx in layer_indices:
            kept.append((name, g))
        else:
            kept.append((name, mx.zeros_like(g)))

    return tree_unflatten(kept)

# --- Attention Masking ---

def _create_4d_attention_mask(tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE) -> mx.array:
    """Creates a combined causal and padding attention mask (4D shape: B, 1, T, T)."""
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")

    batch_size, seq_len = tokens.shape
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=dtype)

    padding_mask_2d = tokens == pad_token_id
    padding_mask_4d_keys = padding_mask_2d[:, None, None, :]

    neg_inf_val = mx.array(-65504. if dtype == mx.bfloat16 else -1e9, dtype=dtype)
    additive_padding_mask = mx.where(padding_mask_4d_keys, neg_inf_val, mx.zeros((batch_size, 1, 1, seq_len), dtype=dtype))

    combined = mx.minimum(causal_mask[None, None, :, :], additive_padding_mask)
    return combined

# --- Sampling ---

def safe_make_sampler(config: ExperimentConfig, temp: float) -> Callable:
    """Safely creates a sampler function, handling various argument inconsistencies."""
    top_p = float(config.generation.top_p)
    min_p = float(config.generation.get("min_p", 0.0)) # Assuming a default here if not explicit
    top_k = int(config.generation.get("top_k", 0)) # 0 means disabled

    # mlx-lm versions often change required arguments for make_sampler
    try:
        return make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
    except TypeError:
        try:
            return make_sampler(temp=temp, top_p=top_p, min_p=min_p)
        except TypeError:
            return make_sampler(temp=temp, top_p=top_p)
        except Exception as e:
            logger.error(f"Failed to create sampler: {e}. Falling back to greedy.")
            return make_sampler(temp=0.0, top_p=1.0) # Absolute fallback to greedy


# --- Logit Processors ---

_TOOL_LIKE_MARKERS = ['<tool_code>', '</tool_code>', '<json_output>', '</json_output>', '<img_base64>', '</img_base64>', '<ocr_text>', '</ocr_text>'] # Updated tags

def _first_token_ids_for_lexemes(tokenizer: TokenizerWrapper, lexemes: Sequence[str]) -> List[int]:
    """Finds the token ID for the first token of a given lexeme (with and without preceding space)."""
    ids: List[int] = []
    for lx in lexemes:
        # Without leading space
        t = tokenizer.encode(lx, add_special_tokens=False)
        if len(t) >= 1 and int(t[0]) not in ids: ids.append(int(t[0]))

        # With leading space
        t = tokenizer.encode(' ' + lx, add_special_tokens=False)
        if len(t) >= 1 and int(t[0]) not in ids: ids.append(int(t[0]))
    return ids

def _letter_token_ids(tokenizer: TokenizerWrapper, letters: Sequence[str] = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') -> Dict[str, List[int]]:
    """Tries to find token IDs for single letter options (A, B, C, D) and common suffixes."""
    out = {}
    for L in letters:
        cand = []
        # Attempt 1: Raw letter
        ids = tokenizer.encode(L, add_special_tokens=False)
        if len(ids) == 1: cand.append(ids[0])

        # Attempt 2: Space + Letter
        ids = tokenizer.encode(' ' + L, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in cand: cand.append(ids[0])

        # Attempt 3: Letter + Punctuation
        for suf in [')', '.', ' )', ' .']:
            ids = tokenizer.encode(L + suf, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in cand: cand.append(ids[0])

        out[L] = cand
    return out

def _resolve_tag_ids(tokenizer: TokenizerWrapper, config: ExperimentConfig) -> Dict[str, Optional[int]]:
    """Helper to convert critical tag strings into their corresponding single token IDs."""
    def _one_id(tok_str):
        if not tok_str: return None
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    # Assuming format_structure is the first reward config
    format_reward_config = next((r for r in config.rewards if r.name == "format_structure"), None)
    if format_reward_config:
        think_start_tag = getattr(format_reward_config, 'think_open_tag', '<think>')
        think_end_tag = getattr(format_reward_config, 'think_close_tag', '</think>')
        answer_start_tag = getattr(format_reward_config, 'answer_open_tag', '<answer>')
        answer_end_tag = getattr(format_reward_config, 'answer_close_tag', '</answer>')
    else:
        think_start_tag = '<think>'
        think_end_tag = '</think>'
        answer_start_tag = '<answer>'
        answer_end_tag = '</answer>'

    return {
        'think_start': _one_id(think_start_tag),
        'think_end': _one_id(think_end_tag),
        'answer_start': _one_id(answer_start_tag),
        'answer_end': _one_id(answer_end_tag),
        'eos': tokenizer.eos_token_id
    }

def make_dynamic_tag_bias_processor(tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]) -> Callable:
    """
    Creates a dynamic logit processor that applies tag-based biases (positive/negative)
    and handles MCQ hard masking based on the current state (inside <think>, inside <answer>, etc.).
    """
    tag_ids = _resolve_tag_ids(tokenizer, config)

    letter_map = _letter_token_ids(tokenizer, letters=LETTER_ALPH)
    mcq_letter_ids = sorted(set(sum(letter_map.values(), [])))

    ban_ids = _first_token_ids_for_lexemes(tokenizer, config.trainer.ban_phrases_for_bias)
    encourage_ids = _first_token_ids_for_lexemes(tokenizer, config.trainer.encourage_phrases_for_bias)
    tool_ids = _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS) # Use predefined markers

    te, ts, as_id, ae, eos_tok = (tag_ids.get(k) for k in ('think_end', 'think_start', 'answer_start', 'answer_end', 'eos'))

    # Store biases locally for cleaner math in the closure
    B_CLOSE = config.trainer.bias_close_think
    B_AS = config.trainer.bias_answer_start
    P_REOPEN_THINK = config.trainer.punish_reopen_think
    P_EXTRA_TE = config.trainer.punish_extra_think_end
    P_REOPEN_ANS = config.trainer.punish_reopen_answer
    B_EOS_ANS = config.trainer.bias_eos_after_answer
    MIN_ANS = config.trainer.min_answer_tokens
    MIN_ANS_MCQ = config.trainer.min_answer_tokens_mcq
    HARD_MASK = config.trainer.hard_mask_mcq_first_token
    LIFT_MCQ = config.trainer.mcq_letter_lift
    BAN_MCQ = config.trainer.mcq_ban_first_bias
    BAN_NONMCQ = config.trainer.nonmcq_ban_first_bias
    MCQ_CLOSE_K = config.trainer.mcq_close_after_k
    B_MCQ_CLOSE = config.trainer.mcq_answer_end_bias # Use configurable bias
    MIN_THINK = config.trainer.min_think_tokens
    B_END_EARLY = config.trainer.think_end_early_bias
    B_AS_MIN_THINK = config.trainer.bias_answer_start_after_min_think
    B_ENCOURAGE = config.trainer.encourage_think_bias
    P_TOOL = config.trainer.tool_call_penalty * -100 # Assuming tool_call_penalty is 0-1 multiplier, make it a strong negative bias

    def _proc_vectorized(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        """
        Vectorized logit processor function applied during generation.
        Applies biases and masks based on current generation state.
        """
        if logits is None or logits.ndim != 2: return logits
        B, V = logits.shape
        dtype = logits.dtype
        neg_inf = mx.array(-1e9, dtype=dtype)

        # Convert python history to a padded MX array for vectorized operations
        max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
        if max_hist_len == 0: return logits

        pad_id = tokenizer.pad_token_id
        history_mx = mx.array(
            [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list], dtype=mx.int32
        )

        # Create masks for special tokens, tools, etc.
        if tool_ids: logits[:, tool_ids] += P_TOOL
        # config.trainer.special_ids is not defined in the provided TrainingArgs; assuming it's meant to be managed externally
        # if config.trainer.get("special_ids"): logits[:, config.trainer.special_ids] -= 20.0

        # Get last positions of tags
        def find_last_pos_mx(tag_id):
            if tag_id is None: return mx.full((B,), -1, dtype=mx.int32)
            matches = history_mx == tag_id
            rev_indices = mx.argmax(mx.flip(matches, axis=1), axis=1)
            return mx.where(mx.any(matches, axis=1), max_hist_len - 1 - rev_indices, -1)

        last_ts, last_te, last_as, last_ae = find_last_pos_mx(ts), find_last_pos_mx(te), find_last_pos_mx(as_id), find_last_pos_mx(ae)

        # Boolean masks
        history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)
        inside_think = (last_ts != -1) & (last_te < last_ts) & (last_as < last_ts)
        inside_answer = (last_as != -1) & (last_ae < last_as)
        ae_seen = last_ae != -1

        # Token counts inside active blocks
        k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
        k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)

        is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)

        # --- Apply biases using vectorized 'where' conditions ---

        # 1. Think Block Management
        if ts is not None and te is not None:
            te_count = mx.array([row.count(te) for row in hist_list], dtype=mx.int32)

            # A. Punishment for Re-opening Tags (Think/Answer)
            logits[:, ts] += mx.where(last_te != -1, P_REOPEN_THINK, 0.0)
            if as_id is not None: logits[:, as_id] += mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)

            # B. Bias for closing tags / Early Think End Penalty
            bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
            min_think_penalty = inside_think & (k_think < MIN_THINK)
            bias_at_te = mx.where(min_think_penalty, B_END_EARLY, bias_at_te)
            if te is not None: logits[:, te] += bias_at_te

            # C. Answer Start Bias
            can_start_answer = (last_te > last_as) & ~inside_answer
            can_start_answer &= ~B_AS_MIN_THINK | (k_think >= MIN_THINK)
            if as_id is not None: logits[:, as_id] += mx.where(can_start_answer, B_AS, 0.0)

        # 2. End-of-Sequence (EOS) Bias
        if eos_tok is not None: logits[:, eos_tok] += mx.where(ae_seen, B_EOS_ANS, 0.0)

        # 3. Shorthand Encouragement (inside think block)
        if encourage_ids and inside_think.any():
            enc_mask_mx = mx.zeros((V,), dtype=dtype)
            enc_mask_mx = enc_mask_mx.at[encourage_ids].set(1.0)
            logits[inside_think.tolist(), :] += enc_mask_mx[None, :] * B_ENCOURAGE

        # 4. MCQ Hard Masking (first token only)
        mcq_first_token_mask = is_mcq_mask & inside_answer & (k_answer == 0)
        if mx.any(mcq_first_token_mask).item() and HARD_MASK:
            mcq_allowed_logits = mx.full((V,), neg_inf, dtype=dtype)
            mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].set(0.0)
            mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].set(mcq_allowed_logits[mcq_letter_ids] + LIFT_MCQ)
            if ban_ids: mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].set(BAN_MCQ)
            if tool_ids: mcq_allowed_logits = mcq_allowed_logits.at[tool_ids].set(P_TOOL) # Heavy penalty
            logits = mx.where(mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits)

        # 5. Non-MCQ First Answer Token Ban
        non_mcq_first_answer = (~is_mcq_mask) & inside_answer & (k_answer == 0)
        if mx.any(non_mcq_first_answer).item() and ban_ids:
            logits[non_mcq_first_answer.tolist(), ban_ids] += BAN_NONMCQ
            if tool_ids: logits[non_mcq_first_answer.tolist(), tool_ids] += P_TOOL

        # 6. Minimum Answer Length Penalty
        if ae is not None:
            min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
            min_len_penalty_mask = inside_answer & (k_answer < min_ans_len)
            logits[:, ae] += mx.where(min_len_penalty_mask, -8.0, 0.0)

        # 7. MCQ Quick Close Bias (after minimal token count)
        if ae is not None:
            mcq_close_mask = is_mcq_mask & inside_answer & (k_answer >= MCQ_CLOSE_K)
            logits[:, ae] += mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0)

        return logits

    return _proc_vectorized

class ContentAlignBridge(nn.Module):
    pass