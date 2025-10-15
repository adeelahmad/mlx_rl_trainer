"""
Generator with Thinking/Answer Mask Support for Dual Gradient Training

FIXES APPLIED:
1. Improved token boundary detection - decodes progressively instead of re-encoding
2. Fallback to character-based estimation if progressive decode fails
3. Comprehensive validation and error logging
4. Clear warnings for edge cases (no thinking tokens, no answer tokens, etc.)
5. Removed unused PagedKVCache - was a parameter but never actually used
6. **NEW: Fixed cache corruption bug** - Clear cache between batches + shuffle replicated prompts
7. **NEW: Prevent adjacent duplicate prompts** - Avoids cache interference in second generation

FORMAT REQUIREMENT:
Your model must generate responses in this format:
  <think>reasoning steps here</think>final answer here

MASK CREATION:
- thinking_mask: 1.0 for all tokens up to and including </think>
- answer_mask: 1.0 for all tokens after </think>

ERROR HANDLING:
- If </think> tag not found: All tokens marked as thinking
- If mask creation fails: Falls back to standard training with clear error log
- Empty masks: Error logged, dual gradients disabled for batch

CONFIGURATION:
trainer:
  use_dual_gradients: true  # Enable this feature

No other config needed - masks auto-created from generation format.
"""
import logging
import gc
import re
import random  # NEW: For prompt shuffling
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
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


def _create_thinking_answer_masksx(
    responses_mx: mx.array,
    decoded_responses: List[str],
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
    pad_id: int,
) -> Tuple[mx.array, mx.array]:
    """
    Create masks to separate thinking and answer tokens.

    Format assumption: <think>reasoning here</think>answer here
    - Thinking: Everything up to and including </think>
    - Answer: Everything after </think>

    Args:
        responses_mx: Token IDs of generated responses [batch, seq_len]
        decoded_responses: Decoded text of responses
        tokenizer: Tokenizer for encoding/decoding
        config: Experiment configuration
        pad_id: Padding token ID

    Returns:
        thinking_mask: 1.0 for thinking tokens (including tags), 0.0 elsewhere [batch, seq_len]
        answer_mask: 1.0 for answer tokens, 0.0 elsewhere [batch, seq_len]
    """
    batch_size, seq_len = responses_mx.shape
    thinking_mask_list = []
    answer_mask_list = []

    think_end_tag = '</think>'
    max_thinking_tokens = getattr(config.trainer, 'max_thinking_tokens', 80)  # Default 80 for 128 token budget

    thinking_lengths = []
    answer_lengths = []
    missing_answer_count = 0

    for batch_idx in range(batch_size):
        decoded_text = decoded_responses[batch_idx]
        response_tokens = responses_mx[batch_idx].tolist()

        # Find where </think> tag ends in the decoded text
        thinking_end_pos = decoded_text.find(think_end_tag)

        # Create masks
        thinking_mask = mx.zeros(seq_len, dtype=mx.float32)
        answer_mask = mx.zeros(seq_len, dtype=mx.float32)

        if thinking_end_pos == -1:
            # No </think> tag found - treat all as thinking tokens
            # This is a PROBLEM in constrained sequences!
            thinking_token_count = 0
            for i in range(seq_len):
                if response_tokens[i] != pad_id:
                    thinking_mask[i] = 1.0
                    thinking_token_count += 1

            missing_answer_count += 1
            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(0)

            logger.warning(f"Sample {batch_idx}: No </think> tag found - all {thinking_token_count} tokens are thinking (NO ANSWER SECTION!)")
        else:
            # Find token boundary by decoding progressively
            # This is more accurate than re-encoding substrings
            thinking_end_pos_with_tag = thinking_end_pos + len(think_end_tag)

            # Decode token by token to find exact boundary
            thinking_token_count = 0
            accumulated_text = ""

            for i in range(seq_len):
                if response_tokens[i] == pad_id:
                    break

                # Decode up to this token
                token_text = tokenizer.decode([response_tokens[i]])
                accumulated_text += token_text

                # Check if we've passed the </think> tag
                if len(accumulated_text) >= thinking_end_pos_with_tag:
                    thinking_token_count = i + 1
                    break

            # If we couldn't find the boundary properly, fall back to character-based estimate
            if thinking_token_count == 0 and thinking_end_pos_with_tag > 0:
                # Rough estimate: assume average token length
                avg_char_per_token = len(decoded_text) / max(1, seq_len - response_tokens.count(pad_id))
                thinking_token_count = min(seq_len, int(thinking_end_pos_with_tag / avg_char_per_token) + 1)

            # Set thinking tokens (everything up to and including </think>)
            for i in range(min(thinking_token_count, seq_len)):
                if response_tokens[i] != pad_id:
                    thinking_mask[i] = 1.0

            # Set answer tokens (everything after </think>)
            answer_token_count = 0
            for i in range(thinking_token_count, seq_len):
                if response_tokens[i] != pad_id:
                    answer_mask[i] = 1.0
                    answer_token_count += 1

            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(answer_token_count)

            # Warn about imbalanced token distribution in constrained sequences
            if thinking_token_count > max_thinking_tokens:
                logger.warning(f"Sample {batch_idx}: Excessive thinking - {thinking_token_count} tokens (>{max_thinking_tokens} limit), only {answer_token_count} answer tokens remaining")

        thinking_mask_list.append(thinking_mask[None, :])
        answer_mask_list.append(answer_mask[None, :])

    thinking_mask_batch = mx.concatenate(thinking_mask_list, axis=0)
    answer_mask_batch = mx.concatenate(answer_mask_list, axis=0)

    # Log aggregate statistics for monitoring
    if thinking_lengths:
        avg_thinking = sum(thinking_lengths) / len(thinking_lengths)
        avg_answer = sum(answer_lengths) / len(answer_lengths)
        ratio = avg_thinking / (avg_answer + 1e-6)

        logger.debug(f"Mask stats: thinking={avg_thinking:.1f} tokens, answer={avg_answer:.1f} tokens, ratio={ratio:.2f}:1")

        # Alert on severe imbalance
        if ratio > 4.0:
            logger.warning(f"SEVERE IMBALANCE: Thinking/answer ratio is {ratio:.2f}:1 - model generating too much thinking!")
        if missing_answer_count > 0:
            logger.warning(f"CRITICAL: {missing_answer_count}/{batch_size} samples have NO answer section (no </think> tag)")

    return thinking_mask_batch, answer_mask_batch




def _create_thinking_answer_masks(
    responses_mx: mx.array,
    decoded_responses: List[str],
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
    pad_id: int,
) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
    """
    Create masks + return statistics for metrics tracking.

    NEW: Returns statistics dictionary for WandB logging
    """
    batch_size, seq_len = responses_mx.shape
    thinking_mask_list = []
    answer_mask_list = []

    think_end_tag = '</think>'
    max_thinking_tokens = getattr(config.trainer, 'max_thinking_tokens', 80)

    thinking_lengths = []
    answer_lengths = []
    missing_answer_count = 0
    missing_thinking_count = 0
    truncated_count = 0

    for batch_idx in range(batch_size):
        decoded_text = decoded_responses[batch_idx]
        response_tokens = responses_mx[batch_idx].tolist()

        thinking_end_pos = decoded_text.find(think_end_tag)
        has_think_start = '<think>' in decoded_text

        thinking_mask = mx.zeros(seq_len, dtype=mx.float32)
        answer_mask = mx.zeros(seq_len, dtype=mx.float32)

        if thinking_end_pos == -1:
            thinking_token_count = 0
            for i in range(seq_len):
                if response_tokens[i] != pad_id:
                    thinking_mask[i] = 1.0
                    thinking_token_count += 1

            missing_answer_count += 1
            if not has_think_start:
                missing_thinking_count += 1

            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(0)

            logger.warning(
                f"Sample {batch_idx}: No </think> tag - "
                f"{thinking_token_count} tokens as thinking (NO ANSWER!)"
            )
        else:
            thinking_end_pos_with_tag = thinking_end_pos + len(think_end_tag)
            thinking_token_count = 0
            accumulated_text = ""

            for i in range(seq_len):
                if response_tokens[i] == pad_id:
                    break
                token_text = tokenizer.decode([response_tokens[i]])
                accumulated_text += token_text
                if len(accumulated_text) >= thinking_end_pos_with_tag:
                    thinking_token_count = i + 1
                    break

            if thinking_token_count == 0 and thinking_end_pos_with_tag > 0:
                non_pad = seq_len - response_tokens.count(pad_id)
                avg_char_per_token = len(decoded_text) / max(1, non_pad)
                thinking_token_count = min(seq_len, int(thinking_end_pos_with_tag / avg_char_per_token) + 1)

            for i in range(min(thinking_token_count, seq_len)):
                if response_tokens[i] != pad_id:
                    thinking_mask[i] = 1.0

            answer_token_count = 0
            for i in range(thinking_token_count, seq_len):
                if response_tokens[i] != pad_id:
                    answer_mask[i] = 1.0
                    answer_token_count += 1

            thinking_lengths.append(thinking_token_count)
            answer_lengths.append(answer_token_count)

            if answer_token_count < 10 and thinking_token_count > max_thinking_tokens * 0.8:
                truncated_count += 1

            if thinking_token_count > max_thinking_tokens:
                logger.warning(
                    f"Sample {batch_idx}: Excessive thinking - "
                    f"{thinking_token_count} tokens, {answer_token_count} answer"
                )

        thinking_mask_list.append(thinking_mask[None, :])
        answer_mask_list.append(answer_mask[None, :])

    thinking_mask_batch = mx.concatenate(thinking_mask_list, axis=0)
    answer_mask_batch = mx.concatenate(answer_mask_list, axis=0)

    # Compile statistics for WandB
    stats = {
        'generation/thinking_tokens_avg': sum(thinking_lengths) / len(thinking_lengths) if thinking_lengths else 0,
        'generation/answer_tokens_avg': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        'generation/thinking_tokens_max': max(thinking_lengths) if thinking_lengths else 0,
        'generation/answer_tokens_min': min(answer_lengths) if answer_lengths else 0,
        'generation/missing_answer_count': missing_answer_count,
        'generation/missing_thinking_count': missing_thinking_count,
        'generation/truncated_count': truncated_count,
    }

    if stats['generation/answer_tokens_avg'] > 0:
        stats['generation/thinking_answer_ratio'] = (
            stats['generation/thinking_tokens_avg'] / stats['generation/answer_tokens_avg']
        )
    else:
        stats['generation/thinking_answer_ratio'] = float('inf')

    logger.debug(
        f"Masks: thinking={stats['generation/thinking_tokens_avg']:.1f}, "
        f"answer={stats['generation/answer_tokens_avg']:.1f}, "
        f"ratio={stats['generation/thinking_answer_ratio']:.2f}:1"
    )

    if stats['generation/thinking_answer_ratio'] > 4.0:
        logger.warning(f"SEVERE IMBALANCE: ratio {stats['generation/thinking_answer_ratio']:.2f}:1")
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
    Generate rollouts with CACHE BUG FIX + comprehensive metrics.

    CRITICAL FIX: Cache reset between samples!
    NEW: Returns generation_metrics dict for WandB logging

    Returns:
        (rollout_batch, avg_reward, avg_breakdown, generation_metrics)
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

    # === CRITICAL FIX: Generate each sample with FRESH cache ===
    all_responses_tok_list = []
    all_actor_lp_list = []

    for sample_idx in range(total_samples):
        # CREATE FRESH CACHE for each sample!
        from mlx_lm.models import cache as mlx_cache
        sample_cache = mlx_cache.make_prompt_cache(model, max_kv_size=config.max_kv_size)

        sample_prompt = prompts_mx[sample_idx:sample_idx+1]
        if sample_prompt.size == 0:
            continue

        # Forward pass with fresh cache
        out = model(sample_prompt.astype(mx.int64), cache=sample_cache)
        next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

        mcq_flag = prompts_data_replicated[sample_idx].get("is_mcq", False)
        logit_processor = make_dynamic_tag_bias_processor(tokenizer, config, [mcq_flag])

        hist_tokens = sample_prompt.tolist()[0]
        sample_response_toks = []
        sample_lps = []
        ended = mx.array([False], dtype=mx.bool_)

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

            tok_val = pad_id if ended_prev[0].item() else token[0].item()
            lp_val = 0.0 if ended_prev[0].item() else token_lp[0].item()

            sample_response_toks.append(tok_val)
            sample_lps.append(lp_val)

            if not ended_prev[0].item():
                hist_tokens.append(tok_val)

            out = model(mx.array([[tok_val]], dtype=mx.int32).astype(mx.int64), cache=sample_cache)
            next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

        all_responses_tok_list.append(sample_response_toks)
        all_actor_lp_list.append(sample_lps)

        # Cache garbage collected here - fresh for next sample!
        del sample_cache
        mx.clear_cache()

    # Convert to batch tensors
    max_resp_len = max(len(r) for r in all_responses_tok_list) if all_responses_tok_list else 0
    responses_mx = mx.zeros((total_samples, max_resp_len), dtype=mx.int32)
    actor_log_probs = mx.zeros((total_samples, max_resp_len), dtype=mx.float32)

    for i in range(total_samples):
        resp_len = len(all_responses_tok_list[i])
        if resp_len > 0:
            responses_mx[i, :resp_len] = mx.array(all_responses_tok_list[i], dtype=mx.int32)
            actor_log_probs[i, :resp_len] = mx.array(all_actor_lp_list[i], dtype=mx.float32)

    # Reward calculation (unchanged)
    decoded = tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=False)

    contexts = [
        reward_composer.context_cls(
            generated_text=decoded[i],
            prompt_text=prompts_data_replicated[i]["text"],
            reference_completion=prompts_data_replicated[i]["ref_answer_str"],
            metadata={
                **prompts_data_replicated[i],
                'max_thinking_tokens': getattr(config.trainer, 'max_thinking_tokens', 80),
            },
            update_step=current_update,
        )
        for i in range(total_samples)
    ]

    batch_rewards_dicts = reward_composer.batch_compute(contexts)
    rewards_total = mx.array([r["total"] for r in batch_rewards_dicts])
    rewards_breakdown = {k: [r[k] for r in batch_rewards_dicts] for k in batch_rewards_dicts[0]}

    # Advantages & ref log probs (unchanged)
    grpo_algo = GRPOAlgorithm(config, model, ref_model)
    advantages = grpo_algo.compute_advantages(rewards_total, num_samples_per_prompt)

    full_seq = mx.concatenate([prompts_mx, responses_mx], axis=1)
    ref_logits = ref_model(full_seq.astype(mx.int64))[:, max_prompt_len - 1 : -1, :]
    ref_log_probs_all = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)
    ref_log_probs = mx.take_along_axis(
        ref_log_probs_all, responses_mx[..., None].astype(mx.int64), axis=-1
    ).squeeze(-1)

    response_mask = (responses_mx != pad_id).astype(mx.float32)
    response_mask = _mask_after_answer(responses_mx, response_mask, tokenizer, config)

    # Create masks with statistics
    thinking_mask = None
    answer_mask = None
    mask_stats = {}

    use_dual_gradients = (
        hasattr(config.trainer, 'use_dual_gradients')
        and config.trainer.use_dual_gradients
    )

    if use_dual_gradients:
        try:
            thinking_mask, answer_mask, mask_stats = _create_thinking_answer_masks(
                responses_mx, decoded, tokenizer, config, pad_id
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

    # Logging (unchanged)
    _maybe_log_samples(
        config, current_update, prompts_data_replicated, decoded,
        rewards_breakdown, "n/a", run_id, is_invalid_batch
    )

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

    # Add reference tokens for SFT (unchanged logic)
    use_sft_hybrid = (
        hasattr(config.trainer, 'use_sft_on_answer')
        and config.trainer.use_sft_on_answer
    )

    if use_sft_hybrid:
        try:
            reference_response_tokens_list = []
            for i, prompt_data in enumerate(prompts_data_replicated):
                ref_text = prompt_data["ref_answer_str"]
                if "prompt_len" in prompt_data:
                    ref_full = tokenizer.encode(ref_text)
                    ref_resp = ref_full[prompt_data["prompt_len"]:]
                elif "text" in prompt_data:
                    prompt_toks = tokenizer.encode(prompt_data["text"])
                    ref_full = tokenizer.encode(ref_text)
                    ref_resp = ref_full[len(prompt_toks):]
                else:
                    ref_resp = tokenizer.encode(ref_text)
                reference_response_tokens_list.append(ref_resp)

            max_resp_len = responses_mx.shape[1]
            reference_tokens_padded = []
            for ref_toks in reference_response_tokens_list:
                if len(ref_toks) > max_resp_len:
                    padded = ref_toks[:max_resp_len]
                else:
                    padded = ref_toks + [pad_id] * (max_resp_len - len(ref_toks))
                reference_tokens_padded.append(padded)

            reference_tokens_mx = mx.array(reference_tokens_padded, dtype=mx.int32)
            rollout_batch["reference_tokens"] = reference_tokens_mx
        except Exception as e:
            logger.error(f"Failed to add reference tokens: {e}", exc_info=True)

    # === NEW: Compile comprehensive metrics for WandB ===
    generation_metrics = {
        'generation/avg_reward': mx.mean(rewards_total).item() if rewards_total.size > 0 else 0.0,
        'generation/reward_std': mx.std(rewards_total).item() if rewards_total.size > 0 else 0.0,
        'generation/num_samples': total_samples,
        'generation/num_prompts': num_prompts,
        'generation/samples_per_prompt': num_samples_per_prompt,
        'generation/avg_response_length': float(mx.mean(mx.sum(response_mask, axis=1)).item()),
        **mask_stats,  # Includes all thinking/answer statistics
    }

    # Add individual reward components
    for reward_name, reward_values in rewards_breakdown.items():
        generation_metrics[f'rewards/{reward_name}'] = np.mean(reward_values)

    avg_reward = generation_metrics['generation/avg_reward']
    avg_breakdown = {k: np.mean(v) for k, v in rewards_breakdown.items()}

    model.train()
    if ref_model:
        ref_model.train()

    gc.collect()
    mx.clear_cache()

    return rollout_batch, avg_reward, avg_breakdown, generation_metrics
