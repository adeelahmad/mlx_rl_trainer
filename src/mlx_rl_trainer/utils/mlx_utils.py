_I = "answer_start"
_H = "think_end"
_G = "think_start"
_F = "answer_end"
_E = "head"
_D = True
_C = False
_B = 0.0
_A = None
import logging, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim, gc, re, string, random
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
        0


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


def limit_memory(max_memory_gb):
    D = "set_memory_limit"
    B = max_memory_gb
    try:
        if hasattr(mx, "get_peak_memory"):
            logging.info(f"Initial peak memory: {mx.get_peak_memory()/1e9:.3f} GB")
        else:
            logging.warning("mx.get_peak_memory() not found in this MLX version.")
        C = int(B * 1024 * 1024 * 1024)
        if hasattr(mx, D):
            A = mx.set_memory_limit(C)
            logging.info(
                f"MLX memory limit set to {B} GB. Previous limit: {A/1024**3:.2f} GB"
            )
            return A
        elif hasattr(mx.metal, D):
            A = mx.metal.set_memory_limit(C)
            logging.info(
                f"MLX memory limit set to {B} GB (using mx.metal). Previous limit: {A/1024**3:.2f} GB"
            )
            sys.exit(0)
            return A
        else:
            logging.warning(
                "mx.set_memory_limit() not found in this MLX version. Cannot limit memory."
            )
            return
    except AttributeError:
        logging.warning(
            "MLX memory management functions (get_peak_memory/set_memory_limit) not found. Check MLX version."
        )
        return
    except Exception as E:
        logging.error(f"Failed to set MLX memory limit: {E}", exc_info=_D)
        return


def _is_metal_internal_error(err):
    A = str(err)
    return (
        "Command buffer execution failed" in A
        or "[METAL]" in A
        or "Internal Error" in A
    )


def metal_recover(stage):
    logging.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()


def metal_safe_apply_gradients(optimizer, grads, params):
    try:
        optimizer.apply_gradients(grads, params)
    except Exception as A:
        if _is_metal_internal_error(A):
            metal_recover("apply_gradients")
            return
        raise
    finally:
        mx.clear_cache()
        gc.collect()


def _find_embedding_layer(root):
    for B, A in root.named_modules():
        if isinstance(A, (nn.Embedding, nn.QuantizedEmbedding)):
            return B, A
    raise RuntimeError("No nn.Embedding layer found.")


def _freeze_module(module):
    A = module
    if A:
        for B in A.parameters():
            B.flags.train = _C


class ContentAlignBridge(nn.Module):
    def __init__(
        A,
        teacher_model,
        student_model,
        teacher_tokenizer,
        student_tokenizer,
        bridge_path,
        pool="mean",
        scale=1.0,
        gen_cfg=_A,
    ):
        super().__init__()
        from mlx_rl_trainer.utils.text_utils import extract_answer_region

        A.tok_t, A.tok_s, A.pool, A.scale = (
            teacher_tokenizer,
            student_tokenizer,
            pool,
            float(scale),
        )
        A.gen_cfg = gen_cfg or GenerationConfig()
        E, A.t_emb = _find_embedding_layer(teacher_model)
        E, A.s_emb = _find_embedding_layer(student_model)
        B, C = int(A.t_emb.weight.shape[1]), int(A.s_emb.weight.shape[1])
        D = max(B, C)
        A.bridge = nn.Sequential(
            nn.Linear(B, D, bias=_C), nn.ReLU(), nn.Linear(D, C, bias=_C)
        )
        try:
            F = mx.load(str(bridge_path))
            A.bridge.update(tree_unflatten(list(F.items())))
        except Exception as G:
            logger.warning(f"Could not load align bridge weights: {G}")
        A.bridge.eval()
        _freeze_module(A.t_emb)
        _freeze_module(A.s_emb)
        A.bridge.freeze()

    @staticmethod
    def _pool_vec(tok_emb, pool):
        A = tok_emb
        if A.size == 0:
            return mx.zeros((A.shape[-1],), dtype=A.dtype)
        return A[-1] if pool == "last" else A.mean(axis=0)

    def __call__(A, texts):
        from mlx_rl_trainer.utils.text_utils import extract_answer_region as H

        B = []
        for I in texts:
            C = H(I or "", A.gen_cfg)
            if not C.strip():
                B.append(_B)
                continue
            D, E = (
                A.tok_t.encode(C, add_special_tokens=_C) or [],
                A.tok_s.encode(C, add_special_tokens=_C) or [],
            )
            if not D or not E:
                B.append(_B)
                continue
            J = A._pool_vec(A.t_emb(mx.array(D, dtype=mx.int32)), A.pool)
            F = A._pool_vec(A.s_emb(mx.array(E, dtype=mx.int32)), A.pool)
            G = A.bridge(J)
            K = G / (mx.norm(G) + 1e-08)
            L = F / (mx.norm(F) + 1e-08)
            M = mx.sum(K * L)
            N = 0.5 * (1.0 + M)
            B.append(max(_B, min(1.0, float(mx.clip(N, _B, 1.0).item()) * A.scale)))
        return B


_LAYER_PAT = re.compile("(?:^|[^a-zA-Z0-9_])layers\\.(\\d+)(?:[^0-9_]|$)")
_HEAD_PAT = re.compile("\\blm_head\\b", re.I)


def _find_layer_index(name):
    B = _LAYER_PAT.search(name)
    if B:
        return int(B.group(1))
    A = re.split("[\\.\\/]", name)
    for C, D in enumerate(A):
        if D == "layers" and C + 1 < len(A):
            try:
                return int(A[C + 1])
            except Exception:
                pass


def _band_for_name(name, low_band, mid_band, top_band):
    A = _find_layer_index(name)

    def B(layer_idx, band_range):
        B = band_range
        A = layer_idx
        if B is _A or A is _A:
            return _C
        C, D = B
        return (C is _A or A >= C) and (D is _A or A <= D)

    if A is not _A:
        if B(A, low_band):
            return "low"
        if B(A, mid_band):
            return "mid"
        if B(A, top_band):
            return "top"
    if _HEAD_PAT.search(name):
        return _E
    return "other"


def scale_grads_by_band(grads_tree, config):
    A = config.trainer
    E = tree_flatten(grads_tree)
    B = []
    for C, D in E:
        if not isinstance(D, mx.array):
            B.append((C, D))
            continue
        F = _band_for_name(C, A.low_band, A.mid_band, A.top_band)
        G = {"low": A.low_mul, "mid": A.mid_mul, "top": A.top_mul, _E: A.head_mul}.get(
            F, 1.0
        )
        B.append((C, D * G))
    return tree_unflatten(B)


def mask_grads_to_layer_band(
    grads_tree, start, end, *, include_embed=_D, include_head=_D, include_final_norm=_D
):
    G = start
    H = tree_flatten(grads_tree)
    E = []
    for B, C in H:
        if not isinstance(C, mx.array):
            E.append((B, C))
            continue
        F = _find_layer_index(B)
        A = _C
        if F is not _A:
            A = (G is _A or F >= G) and (end is _A or F <= end)
        else:
            D = B.lower()
            if "embed" in D or "embedding" in D:
                A = include_embed
            elif "norm" in D:
                A = include_final_norm
            elif _E in D:
                A = include_head
        E.append((B, C if A else mx.zeros_like(C)))
    return tree_unflatten(E)


def mask_grads_to_specific_layers(grads_tree, layer_indices):
    D = tree_flatten(grads_tree)
    A = []
    for B, C in D:
        if not isinstance(C, mx.array):
            A.append((B, C))
            continue
        if (E := _find_layer_index(B)) is not _A and E in layer_indices:
            A.append((B, C))
        else:
            A.append((B, mx.zeros_like(C)))
    return tree_unflatten(A)


def _global_grad_norm(grads):
    try:
        A = [A for (B, A) in tree_flatten(grads) if isinstance(A, mx.array)]
        if not A:
            return _B
        C = sum(mx.sum(A.astype(mx.float32) ** 2) for A in A)
        B = mx.sqrt(C)
        mx.eval(B)
        return float(B.item())
    except Exception:
        return _B


def _maybe_clip_grad_norm(grads_tree, max_norm):
    C = max_norm
    A = grads_tree
    if C is _A or C <= 0:
        B = _global_grad_norm(A)
        return A, B
    try:
        D, E = optim.clip_grad_norm(A, float(C))
        mx.eval(D, E)
        return D, float(E.item())
    except Exception as F:
        logger.warning(
            f"mlx.optim.clip_grad_norm failed: {F}. Falling back to manual clipping."
        )
        B = _global_grad_norm(A)
        if B > C:
            G = C / (B + 1e-08)
            D = tree_map(lambda g: g.astype(mx.float32) * G, A)
            return D, B
        return A, B


def metal_before_update(num_updates, config):
    C = "_orig_max_gen_len"
    B = num_updates
    A = config
    if not hasattr(A.generation, C):
        setattr(A.generation, C, A.data.max_gen_len)
        setattr(A, "_orig_max_kv_size", A.max_kv_size)
        setattr(A.trainer, "_orig_num_rollout_samples", A.trainer.num_rollout_samples)
    if B < 32:
        A.data.max_gen_len = min(A.generation._orig_max_gen_len, 160)
        A.max_kv_size = min(A._orig_max_kv_size, 768)
        A.trainer.num_rollout_samples = min(A.trainer._orig_num_rollout_samples, 4)
    else:
        A.data.max_gen_len = A.generation._orig_max_gen_len
        A.max_kv_size = A._orig_max_kv_size
        A.trainer.num_rollout_samples = A.trainer._orig_num_rollout_samples
    if B % 5 == 0:
        try:
            mx.synchronize()
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()


def _create_4d_attention_mask(tokens, pad_token_id, dtype=TARGET_FLOAT_DTYPE):
    B = dtype
    A = tokens
    if A.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {A.shape}")
    G, C = A.shape
    D = nn.MultiHeadAttention.create_additive_causal_mask(C, dtype=B)
    E = (A == pad_token_id)[:, _A, _A, :]
    F = mx.array(-1e9, dtype=B)
    return mx.minimum(D, mx.where(E, F, _B))


def safe_make_sampler(config_or_args, temp):
    B = config_or_args
    A = B.generation if isinstance(B, ExperimentConfig) else B
    try:
        return make_sampler(
            temp=temp,
            top_p=A.sampling_top_p,
            min_p=A.sampling_min_p,
            top_k=A.sampling_top_k,
        )
    except TypeError:
        return make_sampler(temp=temp, top_p=A.sampling_top_p)


def _first_token_ids_for_lexemes(tokenizer, lexemes):
    D = tokenizer
    A = []
    for E in lexemes:
        if (B := D.encode(E, add_special_tokens=_C)) and B and B[0] not in A:
            A.append(B[0])
        if (C := D.encode(" " + E, add_special_tokens=_C)) and C and C[0] not in A:
            A.append(C[0])
    return A


def _letter_token_ids(tokenizer, letters=LETTER_ALPH):
    C = {}
    for D in letters:
        A = []
        for E in ["", " ", ")", ".", " )", " ."]:
            B = tokenizer.encode(D + E, add_special_tokens=_C)
            if len(B) == 1 and B[0] not in A:
                A.append(B[0])
        C[D] = A
    return C


def _resolve_tag_ids(tokenizer, gen_config):
    C = tokenizer
    A = gen_config

    def B(tok_str):
        A = tok_str
        if not A:
            return
        try:
            B = C.encode(A, add_special_tokens=_C)
            return int(B[0]) if len(B) == 1 else _A
        except Exception:
            return

    return {
        _G: B(A.think_start_tag),
        _H: B(A.think_end_tag),
        _I: B(A.answer_start_tag),
        _F: B(A.answer_end_tag),
        "eos": C.eos_token_id,
    }


def make_dynamic_tag_bias_processor(tokenizer, config, mcq_flags):
    B = tokenizer
    A = config.generation
    C = _resolve_tag_ids(B, A)
    W = sorted(set(sum(_letter_token_ids(B).values(), [])))
    G = _first_token_ids_for_lexemes(B, A.ban_phrases_for_bias)
    X = _first_token_ids_for_lexemes(B, A.encourage_phrases_for_bias)
    Y = _first_token_ids_for_lexemes(B, _TOOL_LIKE_MARKERS)
    H, O, E, I, Z = (C.get(A) for A in (_H, _G, _I, _F, "eos"))
    l, m, n, o, p, q = (
        A.bias_close_think,
        A.bias_answer_start,
        A.punish_reopen_think,
        A.punish_extra_think_end,
        A.punish_reopen_answer,
        A.bias_eos_after_answer,
    )
    r, s, t, u, v, a = (
        A.min_answer_tokens,
        A.min_answer_tokens_mcq,
        A.hard_mask_mcq_first_token,
        A.mcq_letter_lift,
        A.mcq_ban_first_bias,
        A.nonmcq_ban_first_bias,
    )
    w, x, b, y, c = (
        A.mcq_close_after_k,
        A.mcq_answer_end_bias,
        A.min_think_tokens,
        A.think_end_early_bias,
        A.bias_answer_start_after_min_think,
    )
    d, e = A.encourage_think_bias, A.tool_call_penalty * -1e1

    def D(hist_list, logits):
        J = hist_list
        A = logits
        if A.ndim != 2:
            return A
        z, A0 = A.shape
        A1, A2 = mx.array(-1e9, dtype=A.dtype), B.pad_token_id
        P = max(len(A) for A in J) if J else 0
        if P == 0:
            return A
        f = mx.array([A + [A2] * (P - len(A)) for A in J], dtype=mx.int32)
        if Y and e < 0:
            A = A.at[:, Y].add(e)

        def A3(tag_id):
            A = tag_id
            if A is _A:
                return mx.full((z,), -1, dtype=mx.int32)
            B = f == A
            C = mx.argmax(B[:, ::-1], axis=1).astype(mx.int32)
            return mx.where(mx.any(B, axis=1), P - 1 - C, -1)

        K, Q, C, R = (A3(A) for A in (O, H, E, I))
        g = mx.array([len(A) for A in J], dtype=mx.int32)
        L = mx.logical_and(K != -1, mx.logical_and(Q < K, C < K))
        D = mx.logical_and(C != -1, R < C)
        A4 = R != -1
        h = mx.where(L, g - (K + 1), 0)
        M = mx.where(D, g - (C + 1), 0)
        N = mx.array(mcq_flags, dtype=mx.bool_)
        if O is not _A and H is not _A:
            A = A.at[:, O].add(mx.where(Q != -1, n, _B))
            if E is not _A:
                A = A.at[:, E].add(mx.where(R > C, p, _B))
            A5 = mx.sum(f == H, axis=1)
            S = mx.where(A5 == 0, l, o)
            A6 = mx.logical_and(L, h < b)
            S = mx.where(A6, y, S)
            A = A.at[:, H].add(S)
            T = mx.logical_and(Q > C, mx.logical_not(D))
            i = mx.logical_not(c)
            if c:
                i = h >= b
            T = mx.logical_and(T, i)
            if E is not _A:
                A = A.at[:, E].add(mx.where(T, m, _B))
        if Z is not _A:
            A = A.at[:, Z].add(mx.where(A4, q, _B))
        if X and d > 0 and mx.any(L).item():
            U = mx.zeros_like(A)
            U = U.at[:, X].add(d)
            A = A + U * L[:, _A]
        j = mx.logical_and(N, mx.logical_and(D, M == 0))
        if mx.any(j).item() and t:
            F = mx.full((A0,), A1, dtype=A.dtype)
            if W:
                F = F.at[W].add(u)
            if G:
                F = F.at[G].add(v)
            A = mx.where(j[:, _A], F[_A, :], A)
        k = mx.logical_and(mx.logical_not(N), mx.logical_and(D, M == 0))
        if G and a != 0 and mx.any(k).item():
            V = mx.zeros_like(A)
            V = V.at[:, G].add(a)
            A = A + V * k[:, _A]
        if I is not _A:
            A7 = mx.where(N, s, r)
            A8 = mx.logical_and(D, M < A7)
            A = A.at[:, I].add(mx.where(A8, -8.0, _B))
            A9 = mx.logical_and(N, mx.logical_and(D, M >= w))
            A = A.at[:, I].add(mx.where(A9, x, _B))
        return A

    return D


def _mask_after_answer(responses_mx, initial_mask, tokenizer, config):
    B = responses_mx
    A = initial_mask
    if B.ndim != 2:
        return A
    J, C = B.shape
    A = A.astype(mx.float32)
    D = _resolve_tag_ids(tokenizer, config.generation).get(_F)
    if D is _A:
        return A
    E = mx.arange(C)
    F = B == D
    G = mx.argmin(mx.where(F, E, C + 1), axis=1)
    H = G + 1
    I = mx.broadcast_to(E[_A, :], B.shape) < H[:, _A]
    return A * I.astype(mx.float32)
