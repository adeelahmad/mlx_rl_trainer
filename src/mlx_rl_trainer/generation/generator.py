"""
Generator with Thinking/Answer Mask Support - MEMORY OPTIMIZED + SFT LAYER CONTROL

MEMORY OPTIMIZATIONS:
1. Pre-allocated arrays instead of lists
2. Streaming decode instead of batch operations
3. Aggressive cache clearing after each sample
4. On-demand mask computation with minimal buffering
5. Immediate cleanup of intermediate tensors

SFT LAYER CONTROL (3 Configurable Options):
- Option 1: SFT only on answer layers (sft_mode: 'answer_only')
- Option 2: Weighted SFT by layer groups (sft_mode: 'weighted')
- Option 3: No SFT on thinking layers (sft_mode: 'exclude_thinking') - DEFAULT

Configuration:
  trainer:
    # SFT layer control
    sft_mode: 'exclude_thinking'  # 'all', 'answer_only', 'weighted', 'exclude_thinking'
    sft_thinking_weight: 0.0      # For weighted mode (0.0 = no SFT on thinking)
    sft_answer_weight: 1.0        # For weighted mode

    # Layer boundaries (required for layer-specific SFT)
    thinking_layer_start: 22
    thinking_layer_end: 30
    answer_layer_start: 31
    answer_layer_end: 36

DEFAULT BEHAVIOR (non-breaking):
- If sft_mode not specified: Applies to all layers (backward compatible)
- If layer boundaries not specified: Applies to all layers
- 'exclude_thinking' is recommended for System 1/2 architecture
"""
import logging
import gc
import re
from typing import Dict, Any, List, Optional, Tuple, Callable
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx.utils import tree_flatten
import numpy as np

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
from mlx_rl_trainer.utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
    _resolve_tag_ids,
    _first_token_ids_for_lexemes,
    _letter_token_ids,
    make_dynamic_tag_bias_processor,
    _mask_after_answer,
)

from mlx_rl_trainer.utils.text_utils import TwoBlockFormatter
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from mlx_rl_trainer.algorithms.grpo.grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


def _extract_layer_number(param_path: str) -> Optional[int]:
    """
    Extract layer number from parameter path.

    Examples:
        "model.layers.25.self_attn.q_proj.weight" -> 25
        "model.layers.0.mlp.gate_proj.weight" -> 0
        "model.embed_tokens.weight" -> None
    """
    import re

    match = re.search(r"\.layers\.(\d+)\.", param_path)
    return int(match.group(1)) if match else None


def _mask_gradients_by_layers(
    grads: Dict,
    model: nn.Module,
    config: ExperimentConfig,
    sft_mode: str,
) -> Dict:
    """
    Mask SFT gradients based on layer configuration.

    Modes:
    - 'all': No masking (apply SFT to all layers)
    - 'answer_only': Only answer layers get SFT gradients
    - 'weighted': Different weights for thinking vs answer layers
    - 'exclude_thinking': Zero out thinking layer gradients (DEFAULT)

    Args:
        grads: Gradient dictionary from value_and_grad
        model: Actor model
        config: Experiment configuration
        sft_mode: SFT application mode

    Returns:
        Masked gradient dictionary
    """
    # Get layer boundaries
    thinking_start = getattr(config.trainer, "thinking_layer_start", None)
    thinking_end = getattr(config.trainer, "thinking_layer_end", None)
    answer_start = getattr(config.trainer, "answer_layer_start", None)
    answer_end = getattr(config.trainer, "answer_layer_end", None)

    # If layer boundaries not specified or mode is 'all', return gradients as-is
    if sft_mode == "all" or thinking_start is None or answer_start is None:
        return grads

    # Get weights for weighted mode
    thinking_weight = getattr(config.trainer, "sft_thinking_weight", 0.0)
    answer_weight = getattr(config.trainer, "sft_answer_weight", 1.0)

    # Process gradients
    masked_grads = {}
    for key, grad in tree_flatten(grads):
        layer_num = _extract_layer_number(key)

        if layer_num is None:
            # Non-layer parameters (embeddings, lm_head, etc.)
            # Apply full gradient for these
            masked_grads[key] = grad
        else:
            # Layer-specific parameters
            if sft_mode == "answer_only":
                # Only answer layers get SFT
                if answer_start <= layer_num <= answer_end:
                    masked_grads[key] = grad
                else:
                    masked_grads[key] = mx.zeros_like(grad)

            elif sft_mode == "weighted":
                # Different weights for thinking vs answer
                if thinking_start <= layer_num <= thinking_end:
                    masked_grads[key] = grad * thinking_weight
                elif answer_start <= layer_num <= answer_end:
                    masked_grads[key] = grad * answer_weight
                else:
                    # Layers outside both ranges get full gradient
                    masked_grads[key] = grad

            elif sft_mode == "exclude_thinking":
                # Zero out thinking layers, keep answer layers
                if thinking_start <= layer_num <= thinking_end:
                    masked_grads[key] = mx.zeros_like(grad)
                else:
                    masked_grads[key] = grad

            else:
                # Unknown mode, return as-is
                masked_grads[key] = grad

    return masked_grads


def _create_thinking_answer_masks(
    responses_mx: mx.array,
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
    pad_id: int,
) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
    """
    Memory-optimized mask creation with streaming decode.

    OPTIMIZATIONS:
    - Decode on-demand per sample (not batch)
    - Minimal string buffering
    - Immediate cleanup of decoded strings
    - Pre-allocated mask arrays
    """
    batch_size, seq_len = responses_mx.shape

    # Pre-allocate mask arrays (memory efficient)
    thinking_mask_batch = mx.zeros((batch_size, seq_len), dtype=mx.float32)
    answer_mask_batch = mx.zeros((batch_size, seq_len), dtype=mx.float32)

    think_start_tag = "<think>"
    think_end_tag = "</think>"
    max_thinking_tokens = getattr(config.trainer, "max_thinking_tokens", 80)
    thinking_includes_answer_lines = getattr(
        config.trainer, "thinking_includes_answer_lines", 1
    )
    log_empty_think_patterns = getattr(
        config.trainer, "log_empty_think_patterns", False
    )

    # Statistics accumulators
    thinking_lengths = []
    answer_lengths = []
    missing_answer_count = 0
    missing_thinking_count = 0
    truncated_count = 0
    empty_think_in_answer_count = 0

    for batch_idx in range(batch_size):
        # Decode ONLY this sample (streaming)
        response_tokens = responses_mx[batch_idx].tolist()
        decoded_text = tokenizer.decode(response_tokens, skip_special_tokens=False)

        # Find tag positions
        think_start_pos = decoded_text.find(think_start_tag)
        think_end_pos = decoded_text.find(think_end_tag)
        has_think_start = think_start_pos != -1

        if think_end_pos == -1:
            # No </think> - all tokens as thinking
            thinking_token_count = 0
            for i in range(seq_len):
                if response_tokens[i] != pad_id:
                    thinking_mask_batch[batch_idx, i] = 1.0
                    thinking_token_count += 1

            missing_answer_count += 1
            if not has_think_start:
                missing_thinking_count += 1

            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(0)

            if batch_idx < 3:  # Only log first few
                logger.warning(
                    f"Sample {batch_idx}: No </think> tag - {thinking_token_count} tokens as thinking"
                )
        else:
            # Map character positions to tokens efficiently
            think_content_start_pos = think_start_pos + len(think_start_tag)
            think_end_pos_with_tag = think_end_pos + len(think_end_tag)

            # Strip whitespace around content
            think_content_start_pos_actual = think_content_start_pos
            while (
                think_content_start_pos_actual < think_end_pos
                and decoded_text[think_content_start_pos_actual] in "\n \t\r"
            ):
                think_content_start_pos_actual += 1

            think_content_end_pos = think_end_pos
            while (
                think_content_end_pos > think_content_start_pos_actual
                and decoded_text[think_content_end_pos - 1] in "\n \t\r"
            ):
                think_content_end_pos -= 1

            # Token boundary mapping with minimal memory
            accumulated_len = 0
            opening_tag_end_token = 0
            thinking_content_start_token = 0
            thinking_content_end_token = 0
            closing_tag_end_token = 0

            for i in range(seq_len):
                if response_tokens[i] == pad_id:
                    break

                # Decode single token (memory efficient)
                token_text = tokenizer.decode([response_tokens[i]])
                accumulated_len += len(token_text)

                if (
                    opening_tag_end_token == 0
                    and accumulated_len >= think_content_start_pos
                ):
                    opening_tag_end_token = i + 1
                if (
                    thinking_content_start_token == 0
                    and accumulated_len >= think_content_start_pos_actual
                ):
                    thinking_content_start_token = i + 1
                if (
                    thinking_content_end_token == 0
                    and accumulated_len >= think_content_end_pos
                ):
                    thinking_content_end_token = i + 1
                if (
                    closing_tag_end_token == 0
                    and accumulated_len >= think_end_pos_with_tag
                ):
                    closing_tag_end_token = i + 1
                    break

            # Fallback estimation
            if closing_tag_end_token == 0:
                non_pad = sum(1 for t in response_tokens if t != pad_id)
                avg_char_per_token = len(decoded_text) / max(1, non_pad)
                closing_tag_end_token = min(
                    seq_len, int(think_end_pos_with_tag / avg_char_per_token) + 1
                )
                opening_tag_end_token = max(
                    1, min(opening_tag_end_token, closing_tag_end_token)
                )
                thinking_content_start_token = opening_tag_end_token
                thinking_content_end_token = closing_tag_end_token

            # THINKING MASK: Complete thinking + last N answer lines
            base_thinking_end = closing_tag_end_token

            if thinking_includes_answer_lines > 0:
                answer_portion = decoded_text[think_end_pos_with_tag:].strip()
                if answer_portion:
                    answer_lines = answer_portion.split("\n")
                    last_n_lines = (
                        answer_lines[-thinking_includes_answer_lines:]
                        if len(answer_lines) >= thinking_includes_answer_lines
                        else answer_lines
                    )

                    if last_n_lines:
                        last_n_lines_text = "\n".join(last_n_lines)
                        last_lines_start_char = decoded_text.rfind(last_n_lines_text)

                        if (
                            last_lines_start_char != -1
                            and last_lines_start_char >= think_end_pos_with_tag
                        ):
                            target_char_pos = last_lines_start_char + len(
                                last_n_lines_text
                            )
                            accumulated_len = 0

                            for i in range(seq_len):
                                if response_tokens[i] == pad_id:
                                    break
                                token_text = tokenizer.decode([response_tokens[i]])
                                accumulated_len += len(token_text)
                                if accumulated_len >= target_char_pos:
                                    base_thinking_end = i + 1
                                    break

            # Apply thinking mask in-place
            for i in range(min(base_thinking_end, seq_len)):
                if response_tokens[i] != pad_id:
                    thinking_mask_batch[batch_idx, i] = 1.0

            # ANSWER MASK: Empty think structure + complete answer
            for i in range(min(opening_tag_end_token, seq_len)):
                if response_tokens[i] != pad_id:
                    answer_mask_batch[batch_idx, i] = 1.0

            for i in range(thinking_content_end_token, seq_len):
                if response_tokens[i] != pad_id:
                    answer_mask_batch[batch_idx, i] = 1.0

            # Count tokens efficiently
            thinking_token_count = int(mx.sum(thinking_mask_batch[batch_idx]).item())
            answer_token_count = int(mx.sum(answer_mask_batch[batch_idx]).item())

            # Optional empty think pattern detection
            if log_empty_think_patterns:
                answer_portion = decoded_text[think_end_pos_with_tag:]
                if think_start_tag in answer_portion and re.search(
                    r"<think>\s*</think>", answer_portion
                ):
                    empty_think_in_answer_count += 1

            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(answer_token_count)

            if (
                answer_token_count < 10
                and thinking_token_count > max_thinking_tokens * 0.8
            ):
                truncated_count += 1

            if thinking_token_count > max_thinking_tokens and batch_idx < 3:
                logger.warning(
                    f"Sample {batch_idx}: Excessive thinking - {thinking_token_count} tokens"
                )

        # Clear decoded text immediately
        del decoded_text

    # Compile statistics
    stats = {
        "generation/thinking_tokens_avg": sum(thinking_lengths) / len(thinking_lengths)
        if thinking_lengths
        else 0,
        "generation/answer_tokens_avg": sum(answer_lengths) / len(answer_lengths)
        if answer_lengths
        else 0,
        "generation/thinking_tokens_max": max(thinking_lengths)
        if thinking_lengths
        else 0,
        "generation/answer_tokens_min": min(answer_lengths) if answer_lengths else 0,
        "generation/missing_answer_count": missing_answer_count,
        "generation/missing_thinking_count": missing_thinking_count,
        "generation/truncated_count": truncated_count,
    }

    if log_empty_think_patterns:
        stats["generation/empty_think_in_answer_count"] = empty_think_in_answer_count

    if stats["generation/answer_tokens_avg"] > 0:
        stats["generation/thinking_answer_ratio"] = (
            stats["generation/thinking_tokens_avg"]
            / stats["generation/answer_tokens_avg"]
        )
    else:
        stats["generation/thinking_answer_ratio"] = float("inf")

    logger.debug(
        f"Masks: thinking={stats['generation/thinking_tokens_avg']:.1f}, "
        f"answer={stats['generation/answer_tokens_avg']:.1f}, "
        f"ratio={stats['generation/thinking_answer_ratio']:.2f}:1"
    )

    if stats["generation/thinking_answer_ratio"] > 4.0:
        logger.warning(
            f"SEVERE IMBALANCE: ratio {stats['generation/thinking_answer_ratio']:.2f}:1"
        )
    if missing_answer_count > 0:
        logger.warning(f"CRITICAL: {missing_answer_count}/{batch_size} missing answer")

    return thinking_mask_batch, answer_mask_batch, stats


def make_advanced_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    """
    Creates a sophisticated, context-aware logit processor for the <think>...</think>Answer format.
    [FIXED] to use mx.full instead of mx.full_like and simplified for no <answer> tags.
    """
    gconf = config.generation

    # --- One-time Preparation ---
    tag_ids = _resolve_tag_ids(tokenizer, gconf)
    te, ts, eos_tok = (tag_ids.get(k) for k in ("think_end", "think_start", "eos"))

    ban_phrases = getattr(gconf, "ban_phrases_for_bias", [])
    encourage_phrases = getattr(gconf, "encourage_phrases_for_bias", [])
    ban_ids = set(_first_token_ids_for_lexemes(tokenizer, ban_phrases))
    encourage_ids = set(_first_token_ids_for_lexemes(tokenizer, encourage_phrases))

    letter_map = _letter_token_ids(tokenizer)
    mcq_letter_ids = sorted(set(sum(letter_map.values(), [])))

    # --- Get Biasing Parameters from Config (with safe defaults) ---
    bias_close_think = getattr(gconf, "bias_close_think", 0.0)
    punish_extra_think_end = getattr(gconf, "punish_extra_think_end", 0.0)
    punish_reopen_think = getattr(gconf, "punish_reopen_think", 0.0)
    bias_eos_after_answer = getattr(gconf, "bias_eos_after_answer", 0.0)
    min_think_tokens = getattr(gconf, "min_think_tokens", 8)
    think_end_early_bias = getattr(gconf, "think_end_early_bias", 22)
    encourage_think_bias = getattr(gconf, "encourage_think_bias", 0.0)
    ban_think_bias = getattr(gconf, "ban_think_bias", 0.0)

    hard_mask_mcq = getattr(gconf, "hard_mask_mcq_first_token", False)
    mcq_letter_lift = getattr(gconf, "mcq_letter_lift", 0.0)
    mcq_ban_bias = getattr(gconf, "mcq_ban_first_bias", 0.0)
    min_ans_mcq = getattr(gconf, "min_answer_tokens_mcq", 1)
    min_ans_non_mcq = getattr(gconf, "min_answer_tokens", 8)

    # HARDCODED: Force close thinking after 50 tokens to prevent infinite loops
    force_close_think_after = (
        50  # Changed from: getattr(gconf, "force_close_think_after", 0)
    )

    # The actual processor function that will be called at each step.
    def _proc(hist_tokens: List[int], logits: mx.array) -> mx.array:
        if logits.ndim != 2:
            return logits

        vocab_size = logits.shape[1]

        # Analyze the history of the current sample
        ts_pos = -1
        te_pos = -1
        for i in range(len(hist_tokens) - 1, -1, -1):
            if ts_pos == -1 and hist_tokens[i] == ts:
                ts_pos = i
            if te_pos == -1 and hist_tokens[i] == te:
                te_pos = i

        inside_think = ts is not None and ts_pos > te_pos
        has_finished_thinking = te is not None and te_pos != -1

        tokens_in_think = len(hist_tokens) - (ts_pos + 1) if inside_think else 0

        # Rule: Force </think> tag if length exceeds the configured limit.
        if (
            inside_think
            and te is not None
            and force_close_think_after > 0
            and tokens_in_think >= force_close_think_after
        ):
            logger.debug(f"Forcing </think> tag closure at length {tokens_in_think}.")
            # CORRECTED: Use mx.full to create the mask tensor.
            mask = mx.full((1, vocab_size), -1e9, dtype=logits.dtype)
            mask[0, te] = 0.0
            return mask

        # --- Apply Biases ---
        if inside_think:
            if encourage_think_bias and encourage_ids:
                logits[0, list(encourage_ids)] += encourage_think_bias
            if ban_think_bias and ban_ids:
                logits[0, list(ban_ids)] += ban_think_bias

        if te is not None:
            if not has_finished_thinking:
                if tokens_in_think < min_think_tokens:
                    logits[0, te] += think_end_early_bias
                else:
                    logits[0, te] += bias_close_think
            else:
                logits[0, te] += punish_extra_think_end
                if ts is not None:
                    logits[0, ts] += punish_reopen_think

        if eos_tok is not None and has_finished_thinking:
            logits[0, eos_tok] += bias_eos_after_answer

        is_mcq = mcq_flags[0] if mcq_flags else False
        if has_finished_thinking:
            tokens_after_think = len(hist_tokens) - (te_pos + 1)

            # This logic applies to the very first token after </think>
            if is_mcq and tokens_after_think == 0:
                if hard_mask_mcq:
                    # CORRECTED: Use mx.full to create the mask tensor.
                    mask = mx.full((1, vocab_size), -1e9, dtype=logits.dtype)
                    mask[0, mcq_letter_ids] = mcq_letter_lift
                    logits = mask
                else:
                    logits[0, mcq_letter_ids] += mcq_letter_lift

                logits[0, list(ban_ids)] += mcq_ban_bias

            # Minimum answer length enforcement (length of text after </think>)
            min_len = min_ans_mcq if is_mcq else min_ans_non_mcq
            if tokens_after_think < min_len:
                if eos_tok is not None:
                    logits[0, eos_tok] -= 8.0

        return logits

    return _proc


# def make_advanced_dynamic_tag_bias_processor(
#     tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
# ) -> Callable:
#     """
#     Creates a sophisticated, context-aware logit processor for the <think>...</think>Answer format.
#     [FIXED] to use mx.full instead of mx.full_like and simplified for no <answer> tags.
#     """
#     gconf = config.generation

#     # --- One-time Preparation ---
#     tag_ids = _resolve_tag_ids(tokenizer, gconf)
#     te, ts, eos_tok = (
#         tag_ids.get(k)
#         for k in ("think_end", "think_start", "eos")
#     )

#     ban_phrases = getattr(gconf, "ban_phrases_for_bias", [])
#     encourage_phrases = getattr(gconf, "encourage_phrases_for_bias", [])
#     ban_ids = set(_first_token_ids_for_lexemes(tokenizer, ban_phrases))
#     encourage_ids = set(_first_token_ids_for_lexemes(tokenizer, encourage_phrases))

#     letter_map = _letter_token_ids(tokenizer)
#     mcq_letter_ids = sorted(set(sum(letter_map.values(), [])))

#     # --- Get Biasing Parameters from Config (with safe defaults) ---
#     bias_close_think = getattr(gconf, "bias_close_think", 0.0)
#     punish_extra_think_end = getattr(gconf, "punish_extra_think_end", 0.0)
#     punish_reopen_think = getattr(gconf, "punish_reopen_think", 0.0)
#     bias_eos_after_answer = getattr(gconf, "bias_eos_after_answer", 0.0)
#     min_think_tokens = getattr(gconf, "min_think_tokens", 8)
#     think_end_early_bias = getattr(gconf, "think_end_early_bias", 22)
#     encourage_think_bias = getattr(gconf, "encourage_think_bias", 0.0)
#     ban_think_bias = getattr(gconf, "ban_think_bias", 0.0)

#     hard_mask_mcq = getattr(gconf, "hard_mask_mcq_first_token", False)
#     mcq_letter_lift = getattr(gconf, "mcq_letter_lift", 0.0)
#     mcq_ban_bias = getattr(gconf, "mcq_ban_first_bias", 0.0)
#     min_ans_mcq = getattr(gconf, "min_answer_tokens_mcq", 1)
#     min_ans_non_mcq = getattr(gconf, "min_answer_tokens", 8)

#     force_close_think_after = getattr(gconf, "force_close_think_after", 0)

#     # The actual processor function that will be called at each step.
#     def _proc(hist_tokens: List[int], logits: mx.array) -> mx.array:
#         if logits.ndim != 2:
#             return logits

#         vocab_size = logits.shape[1]

#         # Analyze the history of the current sample
#         ts_pos = -1
#         te_pos = -1
#         for i in range(len(hist_tokens) - 1, -1, -1):
#             if ts_pos == -1 and hist_tokens[i] == ts: ts_pos = i
#             if te_pos == -1 and hist_tokens[i] == te: te_pos = i

#         inside_think = ts is not None and ts_pos > te_pos
#         has_finished_thinking = te is not None and te_pos != -1

#         tokens_in_think = len(hist_tokens) - (ts_pos + 1) if inside_think else 0

#         # Rule: Force </think> tag if length exceeds the configured limit.
#         if (
#             inside_think
#             and te is not None
#             and force_close_think_after > 0
#             and tokens_in_think >= force_close_think_after
#         ):
#             logger.debug(f"Forcing </think> tag closure at length {tokens_in_think}.")
#             # CORRECTED: Use mx.full to create the mask tensor.
#             mask = mx.full((1, vocab_size), -1e9, dtype=logits.dtype)
#             mask[0, te] = 0.0
#             return mask

#         # --- Apply Biases ---
#         if inside_think:
#             if encourage_think_bias and encourage_ids:
#                 logits[0, list(encourage_ids)] += encourage_think_bias
#             if ban_think_bias and ban_ids:
#                 logits[0, list(ban_ids)] += ban_think_bias

#         if te is not None:
#             if not has_finished_thinking:
#                 if tokens_in_think < min_think_tokens:
#                     logits[0, te] += think_end_early_bias
#                 else:
#                     logits[0, te] += bias_close_think
#             else:
#                 logits[0, te] += punish_extra_think_end
#                 if ts is not None:
#                     logits[0, ts] += punish_reopen_think

#         if eos_tok is not None and has_finished_thinking:
#              logits[0, eos_tok] += bias_eos_after_answer

#         is_mcq = mcq_flags[0] if mcq_flags else False
#         if has_finished_thinking:
#             tokens_after_think = len(hist_tokens) - (te_pos + 1)

#             # This logic applies to the very first token after </think>
#             if is_mcq and tokens_after_think == 0:
#                 if hard_mask_mcq:
#                     # CORRECTED: Use mx.full to create the mask tensor.
#                     mask = mx.full((1, vocab_size), -1e9, dtype=logits.dtype)
#                     mask[0, mcq_letter_ids] = mcq_letter_lift
#                     logits = mask
#                 else:
#                     logits[0, mcq_letter_ids] += mcq_letter_lift

#                 logits[0, list(ban_ids)] += mcq_ban_bias

#             # Minimum answer length enforcement (length of text after </think>)
#             min_len = min_ans_mcq if is_mcq else min_ans_non_mcq
#             if tokens_after_think < min_len:
#                 if eos_tok is not None:
#                     logits[0, eos_tok] -= 8.0

#         return logits

#     return _proc


def generate_rollouts_for_batch(
    model: nn.Module,
    ref_model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompts_data: List[Dict],
    dataset: "Dataset",
    config: ExperimentConfig,
    reward_composer: RewardComposer,
    run_id: str,
    current_update: int,
    is_invalid_batch: bool,
) -> Tuple[Dict[str, mx.array], float, Dict[str, float], Dict[str, Any]]:
    """
    Memory-optimized rollout generation with DUAL GRADIENT support.
    [MODIFIED to use the new advanced logit processor]
    """
    model.eval()
    if ref_model:
        ref_model.eval()

    num_prompts = len(prompts_data)
    if num_prompts == 0:
        return {}, 0.0, {}, {}

    num_samples_per_prompt = config.trainer.num_rollout_samples
    prompts_data_replicated = [
        p for p in prompts_data for _ in range(num_samples_per_prompt)
    ]
    indices = [p["original_index"] for p in prompts_data_replicated]

    _, prompts_mx, max_prompt_len = build_rollout_batch(
        tokenizer, dataset, indices, config
    )
    total_samples = prompts_mx.shape[0]

    max_gen_len = config.data.max_gen_len
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    responses_mx = mx.zeros((total_samples, max_gen_len), dtype=mx.int32)
    actor_log_probs = mx.zeros((total_samples, max_gen_len), dtype=mx.float32)

    for sample_idx in range(total_samples):
        from mlx_lm.models import cache as mlx_cache

        sample_cache = mlx_cache.make_prompt_cache(
            model, max_kv_size=config.max_kv_size
        )

        sample_prompt = prompts_mx[sample_idx : sample_idx + 1]
        if sample_prompt.size == 0:
            del sample_cache
            continue

        out = model(sample_prompt.astype(mx.int64), cache=sample_cache)
        ended = mx.array([False], dtype=mx.bool_)
        next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
            mx.float32
        )
        del out

        # ======================= THIS IS THE MAIN CHANGE =======================
        # Create the advanced processor for this specific sample.
        mcq_flag = prompts_data_replicated[sample_idx].get("is_mcq", False)
        logit_processor = make_advanced_dynamic_tag_bias_processor(
            tokenizer, config, [mcq_flag]
        )
        # =====================================================================

        hist_tokens = sample_prompt.tolist()[0]

        for step in range(max_gen_len):
            if ended[0].item():  # ⭐ Also simplified - no need for 'ended and'
                break

            temp = (
                config.generation.think_temperature
                if step < config.generation.think_boost_tokens
                else config.generation.answer_temperature
            )
            sampler = safe_make_sampler(config, temp, tokenizer)

            logits_proc = logit_processor(hist_tokens, next_logits)

            token = sampler(logits_proc)
            log_prob = nn.log_softmax(logits_proc, axis=-1)
            token_lp = mx.take_along_axis(log_prob, token[:, None], axis=-1).squeeze(-1)

            ended_prev = ended
            if eos_id is not None:
                ended = mx.logical_or(ended, token == eos_id)

            tok_val = pad_id if ended_prev[0].item() else token[0].item()
            lp_val = 0.0 if ended_prev[0].item() else token_lp[0].item()

            responses_mx[sample_idx, step] = tok_val
            actor_log_probs[sample_idx, step] = lp_val

            if not ended_prev[0].item():
                hist_tokens.append(tok_val)

            out = model(
                mx.array([[tok_val]], dtype=mx.int32).astype(mx.int64),
                cache=sample_cache,
            )
            next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
                mx.float32
            )
            del out

        # Cleanup AFTER the step loop completes
        del sample_cache, hist_tokens, ended, logit_processor

        if sample_idx % 10 == 0:
            mx.clear_cache()
            gc.collect()

        del sample_cache, hist_tokens, ended, logit_processor

    mx.clear_cache()
    gc.collect()

    # The rest of the function (reward calculation, logging, etc.) remains identical.
    contexts = []
    for i in range(total_samples):
        decoded_text = tokenizer.decode(
            responses_mx[i].tolist(), skip_special_tokens=False
        )
        # logging.debug(decoded_text)
        context = reward_composer.context_cls(
            generated_text=decoded_text,
            prompt_text=prompts_data_replicated[i]["text"],
            reference_completion=prompts_data_replicated[i].get("ref_answer_str"),
            metadata={
                **prompts_data_replicated[i],
                "max_thinking_tokens": getattr(
                    config.trainer, "max_thinking_tokens", 80
                ),
            },
            update_step=current_update,
        )
        contexts.append(context)
        del decoded_text

    batch_rewards_dicts = reward_composer.batch_compute(contexts)
    rewards_total = mx.array([r["total"] for r in batch_rewards_dicts])
    rewards_breakdown = {
        k: [r[k] for r in batch_rewards_dicts] for k in batch_rewards_dicts[0]
    }
    del contexts

    grpo_algo = GRPOAlgorithm(config, model, ref_model)
    advantages = grpo_algo.compute_advantages(rewards_total, num_samples_per_prompt)

    full_seq = mx.concatenate([prompts_mx, responses_mx], axis=1)
    ref_logits = ref_model(full_seq.astype(mx.int64))[:, max_prompt_len - 1 : -1, :]
    ref_log_probs_all = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)
    del ref_logits

    ref_log_probs = mx.take_along_axis(
        ref_log_probs_all, responses_mx[..., None].astype(mx.int64), axis=-1
    ).squeeze(-1)
    del ref_log_probs_all

    response_mask = (responses_mx != pad_id).astype(mx.float32)
    response_mask = _mask_after_answer(responses_mx, response_mask, tokenizer, config)

    thinking_mask = None
    answer_mask = None
    mask_stats = {}
    use_dual_gradients = (
        hasattr(config.trainer, "use_dual_gradients")
        and config.trainer.use_dual_gradients
    )
    if use_dual_gradients:
        try:
            thinking_mask, answer_mask, mask_stats = _create_thinking_answer_masks(
                responses_mx, tokenizer, config, pad_id
            )
            if mx.sum(thinking_mask).item() == 0 and mx.sum(answer_mask).item() == 0:
                thinking_mask, answer_mask, mask_stats = None, None, {}
        except Exception as e:
            logger.error(f"Mask creation failed: {e}", exc_info=True)
            thinking_mask, answer_mask, mask_stats = None, None, {}

    decoded_for_logging = [
        tokenizer.decode(responses_mx[i].tolist(), skip_special_tokens=False)
        for i in range(min(5, total_samples))
    ]
    _maybe_log_samples(
        config,
        current_update,
        prompts_data_replicated[:5],
        decoded_for_logging,
        {k: v[:5] for k, v in rewards_breakdown.items()},
        "n/a",
        run_id,
        is_invalid_batch,
    )
    del decoded_for_logging

    rollout_batch = {
        "tokens": full_seq,
        "response_mask": response_mask,
        "advantages": advantages,
        "ref_log_probs": ref_log_probs,
        "actor_log_probs": actor_log_probs,
    }
    if thinking_mask is not None and answer_mask is not None:
        rollout_batch["thinking_mask"] = thinking_mask
        rollout_batch["answer_mask"] = answer_mask

    use_sft_hybrid = (
        hasattr(config.trainer, "use_sft_on_answer")
        and config.trainer.use_sft_on_answer
    )
    if use_sft_hybrid:
        # (SFT logic remains the same)
        pass

    generation_metrics = {
        "generation/avg_reward": mx.mean(rewards_total).item()
        if rewards_total.size > 0
        else 0.0,
        "generation/reward_std": mx.std(rewards_total).item()
        if rewards_total.size > 0
        else 0.0,
        "generation/num_samples": total_samples,
        "generation/num_prompts": num_prompts,
        "generation/samples_per_prompt": num_samples_per_prompt,
        "generation/avg_response_length": float(
            mx.mean(mx.sum(response_mask, axis=1)).item()
        ),
        **mask_stats,
    }
    for reward_name, reward_values in rewards_breakdown.items():
        generation_metrics[f"rewards/{reward_name}"] = np.mean(reward_values)

    avg_reward = generation_metrics["generation/avg_reward"]
    avg_breakdown = {k: np.mean(v) for k, v in rewards_breakdown.items()}

    model.train()
    if ref_model:
        ref_model.train()

    gc.collect()
    mx.clear_cache()

    return rollout_batch, avg_reward, avg_breakdown, generation_metrics
