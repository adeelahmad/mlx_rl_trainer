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
        if hasattr(mx, "get_peak_memory"):
            logging.info(f"Initial peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")
        else:
            logging.warning("mx.get_peak_memory() not found in this MLX version.")

        max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)

        if hasattr(mx, "set_memory_limit"):
            previous_limit = mx.set_memory_limit(max_memory_bytes)
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB. Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            return previous_limit
        elif hasattr(mx.metal, "set_memory_limit"):  # Check older API location
            previous_limit = mx.metal.set_memory_limit(max_memory_bytes)
            logging.info(
                f"MLX memory limit set to {max_memory_gb} GB (using mx.metal). Previous limit: {(previous_limit / (1024**3)):.2f} GB"
            )
            sys.exit(0)
            return previous_limit
        else:
            logging.warning(
                "mx.set_memory_limit() not found in this MLX version. Cannot limit memory."
            )
            return None
    except AttributeError:
        logging.warning(
            "MLX memory management functions (get_peak_memory/set_memory_limit) not found. Check MLX version."
        )
        return None
    except Exception as e:
        logging.error(f"Failed to set MLX memory limit: {e}", exc_info=True)
        return None


def _is_metal_internal_error(err: BaseException) -> bool:
    s = str(err)
    return (
        "Command buffer execution failed" in s
        or "[METAL]" in s
        or "Internal Error" in s
    )


def metal_recover(stage: str):
    logging.warning(f"[METAL] Recovering after error at stage: {stage}")
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
        optimizer.apply_gradients(grads, params)
    except Exception as e:
        if _is_metal_internal_error(e):
            metal_recover("apply_gradients")
            return
        raise
    finally:
        mx.clear_cache()
        gc.collect()


def _find_embedding_layer(root: nn.Module) -> Tuple[str, nn.Module]:
    for name, mod in root.named_modules():
        if isinstance(mod, (nn.Embedding, nn.QuantizedEmbedding)):
            return name, mod
    raise RuntimeError("No nn.Embedding layer found.")


def _freeze_module(module: nn.Module):
    if module:
        for p in module.parameters():
            p.flags.train = False


class ContentAlignBridge(nn.Module):
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        teacher_tokenizer: TokenizerWrapper,
        student_tokenizer: TokenizerWrapper,
        bridge_path: str,
        pool: str = "mean",
        scale: float = 1.0,
        gen_cfg: Optional[GenerationConfig] = None,
    ):
        super().__init__()
        self.tok_t, self.tok_s, self.pool, self.scale = (
            teacher_tokenizer,
            student_tokenizer,
            pool,
            float(scale),
        )
        self.gen_cfg = gen_cfg or GenerationConfig()
        _, self.t_emb = _find_embedding_layer(teacher_model)
        _, self.s_emb = _find_embedding_layer(student_model)
        t_dim, s_dim = int(self.t_emb.weight.shape[1]), int(self.s_emb.weight.shape[1])
        hidden = max(t_dim, s_dim)
        self.bridge = nn.Sequential(
            nn.Linear(t_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, s_dim, bias=False),
        )
        try:
            w = mx.load(str(bridge_path))
            self.bridge.update(tree_unflatten(list(w.items())))
        except Exception as e:
            logger.warning(f"Could not load align bridge weights: {e}")
        self.bridge.eval()
        _freeze_module(self.t_emb)
        _freeze_module(self.s_emb)
        self.bridge.freeze()

    def _batch_tokenize_and_pad(
        self, texts: List[str], tokenizer: TokenizerWrapper
    ) -> Tuple[mx.array, mx.array]:
        """Tokenizes a batch of texts and pads them to the same length."""
        # Tokenize all texts
        token_ids = [tokenizer.encode(s, add_special_tokens=False) or [] for s in texts]
        lengths = [len(ids) for ids in token_ids]

        # If all sequences are empty, return empty tensors
        if not any(lengths):
            return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.bool_)

        max_len = max(lengths)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        # Create padded array and attention mask
        padded_ids = mx.array(
            [ids + [pad_id] * (max_len - len(ids)) for ids in token_ids], dtype=mx.int32
        )
        attention_mask = mx.arange(max_len)[None, :] < mx.array(lengths)[:, None]

        return padded_ids, attention_mask

    def _pool_vec_batch(
        self, tok_emb: mx.array, attention_mask: mx.array, pool: str
    ) -> mx.array:
        """Performs pooling on a batch of token embeddings, respecting padding."""
        if tok_emb.size == 0:
            return mx.zeros((tok_emb.shape[0], tok_emb.shape[-1]), dtype=tok_emb.dtype)

        if pool == "last":
            # Find the index of the last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(axis=1) - 1
            # Gather the embeddings at these indices
            pooled_vec = mx.take_along_axis(
                tok_emb, sequence_lengths[:, None, None], axis=1
            ).squeeze(1)
        else:  # Default to mean pooling
            # Mask out padding tokens by multiplying with the attention mask
            masked_emb = tok_emb * attention_mask[..., None].astype(tok_emb.dtype)
            # Sum embeddings and divide by the number of non-padded tokens
            sum_emb = masked_emb.sum(axis=1)
            num_tokens = attention_mask.sum(axis=1, keepdims=True)
            pooled_vec = sum_emb / mx.maximum(num_tokens, 1)

        return pooled_vec

    def __call__(self, texts: List[str]) -> List[float]:
        """
        Computes alignment scores for a batch of texts in a fully vectorized manner.
        """
        from mlx_rl_trainer.utils.text_utils import extract_answer_region

        if not texts:
            return []

        # Extract answer regions for all texts
        answers = [extract_answer_region(s or "", self.gen_cfg) for s in texts]

        # Batch tokenize and pad for both teacher and student models
        t_ids, t_mask = self._batch_tokenize_and_pad(answers, self.tok_t)
        s_ids, s_mask = self._batch_tokenize_and_pad(answers, self.tok_s)

        # Get embeddings for the entire batch
        t_emb_unpooled = self.t_emb(t_ids)
        s_emb_unpooled = self.s_emb(s_ids)

        # Pool the embeddings for the entire batch
        t_vecs = self._pool_vec_batch(t_emb_unpooled, t_mask, self.pool)
        s_vecs = self._pool_vec_batch(s_emb_unpooled, s_mask, self.pool)

        # Map teacher vectors to student space for the entire batch
        mapped_vecs = self.bridge(t_vecs)

        # Normalize vectors and compute cosine similarity for the entire batch
        a = mapped_vecs / (mx.linalg.norm(mapped_vecs, axis=-1, keepdims=True) + 1e-8)
        b = s_vecs / (mx.linalg.norm(s_vecs, axis=-1, keepdims=True) + 1e-8)
        cos_sim = mx.sum(a * b, axis=-1)

        # Scale and clip scores
        scores = 0.5 * (1.0 + cos_sim)
        final_scores = mx.clip(scores, 0.0, 1.0) * self.scale

        return final_scores.tolist()


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

    def _in_range(layer_idx, band_range):
        if band_range is None or layer_idx is None:
            return False
        s, e = band_range
        return (s is None or layer_idx >= s) and (e is None or layer_idx <= e)

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
    t_cfg = config.trainer
    g_flat = tree_flatten(grads_tree)
    out = []
    for name, g in g_flat:
        if not isinstance(g, mx.array):
            out.append((name, g))
            continue
        band = _band_for_name(name, t_cfg.low_band, t_cfg.mid_band, t_cfg.top_band)
        mul = {
            "low": t_cfg.low_mul,
            "mid": t_cfg.mid_mul,
            "top": t_cfg.top_mul,
            "head": t_cfg.head_mul,
        }.get(band, 1.0)
        out.append((name, g * mul))
    return tree_unflatten(out)


# the common `{'model': ...}` nesting from mlx-lm.
def mask_grads_to_layer_band(
    grads_tree,
    start,
    end,
    *,
    include_embed=True,
    include_head=True,
    include_final_norm=True,
):
    # Check if the gradients are wrapped in a 'model' key and unwrap if necessary.
    # This makes the function robust to different model structures.
    is_wrapped = (
        isinstance(grads_tree, dict)
        and len(grads_tree) == 1
        and "model" in grads_tree
        and isinstance(grads_tree["model"], dict)
    )
    inner_grads_tree = grads_tree["model"] if is_wrapped else grads_tree

    if not isinstance(inner_grads_tree, dict):
        # Return an empty dict if the structure is unexpected to prevent errors
        return {}

    flat = tree_flatten(inner_grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue

        li = _find_layer_index(name)
        keep = False
        if li is not None:
            # Mask layers based on the start and end range
            keep = (start is None or li >= start) and (end is None or li <= end)
        else:
            # Handle non-layer parameters like embeddings and normalization layers
            lname = name.lower()
            if "embed" in lname or "embedding" in lname:
                keep = include_embed
            elif "norm" in lname:
                keep = include_final_norm
            elif "head" in lname:
                keep = include_head

        kept.append((name, g if keep else mx.zeros_like(g)))

    # Reconstruct the inner tree from the filtered list
    unflattened_tree = tree_unflatten(kept)

    # If the original was wrapped, re-wrap the result to maintain structure
    if is_wrapped:
        return {"model": unflattened_tree}
    else:
        return unflattened_tree


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


def _global_grad_norm(grads: Dict[str, mx.array]) -> float:
    try:
        flat = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
        if not flat:
            return 0.0
        sq_sum = sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat)
        total = mx.sqrt(sq_sum)
        mx.eval(total)
        return float(total.item())
    except Exception:
        return 0.0


def _maybe_clip_grad_norm(
    grads_tree: Dict[str, mx.array], max_norm: Optional[float]
) -> Tuple[Dict[str, mx.array], float]:
    if max_norm is None or max_norm <= 0:
        grad_norm = _global_grad_norm(grads_tree)
        return grads_tree, grad_norm
    try:
        clipped_grads, grad_norm_mx = optim.clip_grad_norm(grads_tree, float(max_norm))
        mx.eval(clipped_grads, grad_norm_mx)
        return clipped_grads, float(grad_norm_mx.item())
    except Exception as e:
        logger.warning(
            f"mlx.optim.clip_grad_norm failed: {e}. Falling back to manual clipping."
        )
        grad_norm = _global_grad_norm(grads_tree)
        if grad_norm > max_norm:
            scale = max_norm / (grad_norm + 1e-8)
            clipped_grads = tree_map(lambda g: g.astype(mx.float32) * scale, grads_tree)
            return clipped_grads, grad_norm
        return grads_tree, grad_norm


def metal_before_update(num_updates: int, config: ExperimentConfig):
    if not hasattr(config.generation, "_orig_max_gen_len"):
        setattr(config.generation, "_orig_max_gen_len", config.data.max_gen_len)
        setattr(config, "_orig_max_kv_size", config.max_kv_size)
        setattr(
            config.trainer,
            "_orig_num_rollout_samples",
            config.trainer.num_rollout_samples,
        )
    if num_updates < 32:
        config.data.max_gen_len = min(config.generation._orig_max_gen_len, 160)
        config.max_kv_size = min(config._orig_max_kv_size, 768)
        config.trainer.num_rollout_samples = min(
            config.trainer._orig_num_rollout_samples, 4
        )
    else:
        config.data.max_gen_len = config.generation._orig_max_gen_len
        config.max_kv_size = config._orig_max_kv_size
        config.trainer.num_rollout_samples = config.trainer._orig_num_rollout_samples
    if num_updates % 5 == 0:
        try:
            mx.synchronize()
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE
) -> mx.array:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")
    B, T = tokens.shape
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T, dtype=dtype)
    padding_mask = (tokens == pad_token_id)[:, None, None, :]
    neg_inf = mx.array(-1e9, dtype=dtype)
    return mx.minimum(causal_mask, mx.where(padding_mask, neg_inf, 0.0))


def safe_make_sampler(
    config_or_args: Union[ExperimentConfig, GenerationConfig],
    temp: float,
    tokenizer: TokenizerWrapper,
) -> Callable:
    gen_cfg = (
        config_or_args.generation
        if isinstance(config_or_args, ExperimentConfig)
        else config_or_args
    )
    try:
        # sampler = make_sampler(
        #     temperature,
        #     top_p=1.0,
        #     min_p=0.0,
        #     min_tokens_to_keep=1,
        #     top_k=0,
        #     xtc_probability=0.0,
        #     xtc_threshold=0.0,
        #     xtc_special_tokens=tokenizer.encode("\n")
        #     + list(tokenizer.eos_token_ids),
        # )

        # prompt_cache = cache.make_prompt_cache(model)
        # completion: str | List[int] = generate(
        #     model=model,
        #     tokenizer=tokenizer,
        #     prompt=prompt_text,
        #     max_tokens=max_tokens,
        #     verbose=False,
        #     sampler=sampler,
        #     prompt_cache=prompt_cache,
        # )
        # Determine temperature based on training phase
        # if update_step < 1000:
        #     temperature = 0.9   # High exploration early
        # elif update_step < 5000:
        #     temperature = 0.85  # Balanced
        # else:
        temperature = 0.18   # More conservative later

        return make_sampler(
            temperature=temperature,
            top_p=0.92,              # Nucleus sampling
            min_p=0.04,              # Min probability filter
            min_tokens_to_keep=1,
            top_k=50,                # Top-k filtering
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
        )

    except TypeError:
        return make_sampler(temp=temp, top_p=gen_cfg.sampling_top_p)


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


def make_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    gen_cfg = config.generation
    tag_ids = _resolve_tag_ids(tokenizer, gen_cfg)
    mcq_letter_ids = sorted(set(sum(_letter_token_ids(tokenizer).values(), [])))
    ban_ids = _first_token_ids_for_lexemes(tokenizer, gen_cfg.ban_phrases_for_bias)
    encourage_ids = _first_token_ids_for_lexemes(
        tokenizer, gen_cfg.encourage_phrases_for_bias
    )
    tool_ids = _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS)

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
    MCQ_CLOSE_K, B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK = (
        gen_cfg.mcq_close_after_k,
        gen_cfg.mcq_answer_end_bias,
        gen_cfg.min_think_tokens,
        gen_cfg.think_end_early_bias,
        gen_cfg.bias_answer_start_after_min_think,
    )
    B_ENCOURAGE, P_TOOL = (
        gen_cfg.encourage_think_bias,
        gen_cfg.tool_call_penalty * -10.0,
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
        if tool_ids and P_TOOL < 0:
            logits = logits.at[:, tool_ids].add(P_TOOL)

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

        if encourage_ids and B_ENCOURAGE > 0 and mx.any(inside_think).item():
            encourage_bias = mx.zeros_like(logits)
            encourage_bias = encourage_bias.at[:, encourage_ids].add(B_ENCOURAGE)
            logits = logits + (encourage_bias * inside_think[:, None])

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
                is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= MCQ_CLOSE_K))
            )
            logits = logits.at[:, ae].add(mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0))

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
    initial_mask = initial_mask.astype(mx.float32)
    answer_end_id = _resolve_tag_ids(tokenizer, config.generation).get("answer_end")
    if answer_end_id is None:
        return initial_mask
    indices = mx.arange(L_gen)
    is_answer_end = responses_mx == answer_end_id
    first_end_indices = mx.argmin(mx.where(is_answer_end, indices, L_gen + 1), axis=1)
    boundary_index = first_end_indices + 1
    end_mask = (
        mx.broadcast_to(indices[None, :], responses_mx.shape) < boundary_index[:, None]
    )
    return initial_mask * end_mask.astype(mx.float32)
