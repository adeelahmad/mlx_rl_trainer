import logging
import gc
import re
from typing import Dict, Any, List, Optional, Tuple
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
    Memory-optimized rollout generation with dual gradient support.

    MEMORY OPTIMIZATIONS:
    1. Pre-allocated response arrays (no intermediate lists)
    2. Per-sample cache with aggressive cleanup
    3. Streaming decode for rewards
    4. Minimal tensor copies
    5. Immediate garbage collection
    6. On-demand mask creation
    """
    model.eval()
    if ref_model:
        ref_model.eval()

    num_prompts = len(prompts_data)
    if num_prompts == 0:
        return {}, 0.0, {}, {}

    num_samples = config.trainer.num_rollout_samples
    prompts_data_replicated = [p for p in prompts_data for _ in range(num_samples)]
    indices = [p["original_index"] for p in prompts_data_replicated]

    _, prompts_mx, max_prompt_len = build_rollout_batch(
        tokenizer, dataset, indices, config
    )
    total_samples = prompts_mx.shape[0]

    max_gen_len = config.data.max_gen_len
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # PRE-ALLOCATE response arrays (memory efficient)
    responses_mx = mx.zeros((total_samples, max_gen_len), dtype=mx.int32)
    actor_log_probs = mx.zeros((total_samples, max_gen_len), dtype=mx.float32)

    # Generate with fresh cache per sample
    for sample_idx in range(total_samples):
        # Create fresh cache
        from mlx_lm.models import cache as mlx_cache

        sample_cache = mlx_cache.make_prompt_cache(
            model, max_kv_size=config.max_kv_size
        )

        sample_prompt = prompts_mx[sample_idx : sample_idx + 1]
        if sample_prompt.size == 0:
            del sample_cache
            continue

        # Initial forward pass
        out = model(sample_prompt.astype(mx.int64), cache=sample_cache)
        next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
            mx.float32
        )
        del out  # Immediate cleanup

        mcq_flag = prompts_data_replicated[sample_idx].get("is_mcq", False)
        logit_processor = make_dynamic_tag_bias_processor(tokenizer, config, [mcq_flag])

        hist_tokens = sample_prompt.tolist()[0]
        ended = mx.array([False], dtype=mx.bool_)

        # Generation loop with in-place array writes
        for step in range(max_gen_len):
            if ended[0].item():
                break

            temp = (
                config.generation.think_temperature
                if step < config.generation.think_boost_tokens
                else config.generation.answer_temperature
            )
            sampler = safe_make_sampler(config, temp=temp)

            logits_proc = logit_processor([hist_tokens], next_logits)
            token = sampler(logits_proc)
            log_prob = nn.log_softmax(logits_proc, axis=-1)
            token_lp = mx.take_along_axis(log_prob, token[:, None], axis=-1).squeeze(-1)

            ended_prev = ended
            if eos_id is not None:
                ended = mx.logical_or(ended, token == eos_id)

            # Write directly to pre-allocated arrays
            tok_val = pad_id if ended_prev[0].item() else token[0].item()
            lp_val = 0.0 if ended_prev[0].item() else token_lp[0].item()

            responses_mx[sample_idx, step] = tok_val
            actor_log_probs[sample_idx, step] = lp_val

            if not ended_prev[0].item():
                hist_tokens.append(tok_val)

            # Continue generation
            out = model(
                mx.array([[tok_val]], dtype=mx.int32).astype(mx.int64),
                cache=sample_cache,
            )
            next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
                mx.float32
            )
            del out

        # Aggressive cleanup
        del sample_cache, hist_tokens, ended, logit_processor
        if sample_idx % 10 == 0:  # Periodic deep cleanup
            mx.clear_cache()
            gc.collect()

    # Final cleanup after generation
    mx.clear_cache()
    gc.collect()

    # Compute rewards (streaming decode)
    contexts = []
    for i in range(total_samples):
        # Decode on-demand
        decoded_text = tokenizer.decode(
            responses_mx[i].tolist(), skip_special_tokens=False
        )

        context = reward_composer.context_cls(
            generated_text=decoded_text,
            prompt_text=prompts_data_replicated[i]["text"],
            reference_completion=prompts_data_replicated[i]["ref_answer_str"],
            metadata={
                **prompts_data_replicated[i],
                "max_thinking_tokens": getattr(
                    config.trainer, "max_thinking_tokens", 80
                ),
            },
            update_step=current_update,
        )
        contexts.append(context)
        del decoded_text  # Immediate cleanup

    batch_rewards_dicts = reward_composer.batch_compute(contexts)
    rewards_total = mx.array([r["total"] for r in batch_rewards_dicts])
    rewards_breakdown = {
        k: [r[k] for r in batch_rewards_dicts] for k in batch_rewards_dicts[0]
    }

    del contexts  # Cleanup

    # Compute advantages
    grpo_algo = GRPOAlgorithm(config, model, ref_model)
    advantages = grpo_algo.compute_advantages(rewards_total, num_samples)

    # Reference log probs
    full_seq = mx.concatenate([prompts_mx, responses_mx], axis=1)
    ref_logits = ref_model(full_seq.astype(mx.int64))[:, max_prompt_len - 1 : -1, :]
    ref_log_probs_all = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)
    del ref_logits  # Cleanup

    ref_log_probs = mx.take_along_axis(
        ref_log_probs_all, responses_mx[..., None].astype(mx.int64), axis=-1
    ).squeeze(-1)
    del ref_log_probs_all  # Cleanup

    # Response mask
    response_mask = (responses_mx != pad_id).astype(mx.float32)
    response_mask = _mask_after_answer(responses_mx, response_mask, tokenizer, config)

    # Create thinking/answer masks with minimal memory
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
                logger.error("Both masks empty!")
                thinking_mask = None
                answer_mask = None
                mask_stats = {}
        except Exception as e:
            logger.error(f"Mask creation failed: {e}", exc_info=True)
            thinking_mask = None
            answer_mask = None
            mask_stats = {}

    # Logging - use _maybe_log_samples which handles config checks internally
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

    # Build rollout batch
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

    # Add reference tokens for SFT
    use_sft_hybrid = (
        hasattr(config.trainer, "use_sft_on_answer")
        and config.trainer.use_sft_on_answer
    )

    if use_sft_hybrid:
        try:
            max_resp_len = responses_mx.shape[1]
            reference_tokens_padded = []

            for prompt_data in prompts_data_replicated:
                ref_text = prompt_data["ref_answer_str"]

                if "prompt_len" in prompt_data:
                    ref_full = tokenizer.encode(ref_text)
                    ref_resp = ref_full[prompt_data["prompt_len"] :]
                elif "text" in prompt_data:
                    prompt_toks = tokenizer.encode(prompt_data["text"])
                    ref_full = tokenizer.encode(ref_text)
                    ref_resp = ref_full[len(prompt_toks) :]
                else:
                    ref_resp = tokenizer.encode(ref_text)

                if len(ref_resp) > max_resp_len:
                    padded = ref_resp[:max_resp_len]
                else:
                    padded = ref_resp + [pad_id] * (max_resp_len - len(ref_resp))
                reference_tokens_padded.append(padded)

            reference_tokens_mx = mx.array(reference_tokens_padded, dtype=mx.int32)
            rollout_batch["reference_tokens"] = reference_tokens_mx
            del reference_tokens_padded  # Cleanup
        except Exception as e:
            logger.error(f"Failed to add reference tokens: {e}", exc_info=True)

    # Compile metrics
    generation_metrics = {
        "generation/avg_reward": mx.mean(rewards_total).item()
        if rewards_total.size > 0
        else 0.0,
        "generation/reward_std": mx.std(rewards_total).item()
        if rewards_total.size > 0
        else 0.0,
        "generation/num_samples": total_samples,
        "generation/num_prompts": num_prompts,
        "generation/samples_per_prompt": num_samples,
        "generation/avg_response_length": float(
            mx.mean(mx.sum(response_mask, axis=1)).item()
        ),
        **mask_stats,
    }

    for reward_name, reward_values in rewards_breakdown.items():
        generation_metrics[f"rewards/{reward_name}"] = np.mean(reward_values)

    avg_reward = generation_metrics["generation/avg_reward"]
    avg_breakdown = {k: np.mean(v) for k, v in rewards_breakdown.items()}

    # Return to training mode
    model.train()
    if ref_model:
        ref_model.train()

    # Final aggressive cleanup
    gc.collect()
    mx.clear_cache()

    return rollout_batch, avg_reward, avg_breakdown, generation_metrics
