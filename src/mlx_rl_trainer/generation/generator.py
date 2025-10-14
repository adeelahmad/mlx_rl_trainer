"""
Generator with Thinking/Answer Mask Support for Dual Gradient Training

FIXES APPLIED:
1. Improved token boundary detection - decodes progressively instead of re-encoding
2. Fallback to character-based estimation if progressive decode fails
3. Comprehensive validation and error logging
4. Clear warnings for edge cases (no thinking tokens, no answer tokens, etc.)
5. Removed unused PagedKVCache - was a parameter but never actually used

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


def _create_thinking_answer_masks(
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
            for i in range(seq_len):
                if response_tokens[i] != pad_id:
                    thinking_mask[i] = 1.0
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
            for i in range(thinking_token_count, seq_len):
                if response_tokens[i] != pad_id:
                    answer_mask[i] = 1.0

        thinking_mask_list.append(thinking_mask[None, :])
        answer_mask_list.append(answer_mask[None, :])

    thinking_mask_batch = mx.concatenate(thinking_mask_list, axis=0)
    answer_mask_batch = mx.concatenate(answer_mask_list, axis=0)

    return thinking_mask_batch, answer_mask_batch


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
) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
    """
    Generate rollouts for a batch of prompts with optional thinking/answer mask creation.

    If config.trainer.use_dual_gradients is True, this function will automatically
    create thinking_mask and answer_mask arrays that separate thinking tokens from
    answer tokens, enabling the dual gradient training approach.
    """
    model.eval()
    if ref_model:
        ref_model.eval()

    num_prompts = len(prompts_data)
    if num_prompts == 0:
        return {}, 0.0, {}

    prompts_data_replicated = [
        p for p in prompts_data for _ in range(config.trainer.num_rollout_samples)
    ]
    indices = [p["original_index"] for p in prompts_data_replicated]

    # Use the builder to get tokens correctly
    _, prompts_mx, max_prompt_len = build_rollout_batch(
        tokenizer, dataset, indices, config
    )
    total_samples = prompts_mx.shape[0]

    max_gen_len = config.data.max_gen_len
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # --- Generation Loop ---
    model_caches = cache.make_prompt_cache(model, max_kv_size=config.max_kv_size)
    if prompts_mx.size == 0:
        return {}, 0.0, {}

    out_actor = model(prompts_mx.astype(mx.int64), cache=model_caches)
    next_logits = (out_actor[0] if isinstance(out_actor, tuple) else out_actor)[
        :, -1, :
    ].astype(mx.float32)

    mcq_flags = [p.get("is_mcq", False) for p in prompts_data_replicated]
    logit_processor = make_dynamic_tag_bias_processor(tokenizer, config, mcq_flags)

    hist_tokens_py = prompts_mx.tolist()
    responses_tok_list, actor_lp_cached_list = [], []
    ended = mx.full((total_samples,), False, dtype=mx.bool_)

    for step in range(max_gen_len):
        if mx.all(ended).item():
            break

        temp = (
            config.generation.think_temperature
            if step < config.generation.think_boost_tokens
            else config.generation.answer_temperature
        )
        sampler = safe_make_sampler(config, temp=temp)

        logits_processed = logit_processor(hist_tokens_py, next_logits)

        sampled_tokens = sampler(logits_processed)
        log_probs = nn.log_softmax(logits_processed, axis=-1)
        sampled_log_probs = mx.take_along_axis(
            log_probs, sampled_tokens[:, None], axis=-1
        ).squeeze(-1)

        ended_prev = ended
        if eos_id is not None:
            ended = mx.logical_or(ended, sampled_tokens == eos_id)

        tokens_to_add = mx.where(ended_prev, pad_id, sampled_tokens)
        lp_to_add = mx.where(ended_prev, 0.0, sampled_log_probs)

        responses_tok_list.append(tokens_to_add[:, None])
        actor_lp_cached_list.append(lp_to_add[:, None])

        for i in range(total_samples):
            if not ended_prev[i].item():
                hist_tokens_py[i].append(tokens_to_add[i].item())

        out_next = model(tokens_to_add[:, None].astype(mx.int64), cache=model_caches)
        next_logits = (out_next[0] if isinstance(out_next, tuple) else out_next)[
            :, -1, :
        ].astype(mx.float32)

    mx.eval(responses_tok_list, actor_lp_cached_list)
    responses_mx = (
        mx.concatenate(responses_tok_list, axis=1)
        if responses_tok_list
        else mx.zeros((total_samples, 0), dtype=mx.int32)
    )
    actor_log_probs = (
        mx.concatenate(actor_lp_cached_list, axis=1)
        if actor_lp_cached_list
        else mx.zeros((total_samples, 0), dtype=mx.float32)
    )

    # --- Reward Calculation ---
    decoded = tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=False)

    contexts = [
        reward_composer.context_cls(
            generated_text=decoded[i],
            prompt_text=prompts_data_replicated[i]["text"],
            reference_completion=prompts_data_replicated[i]["ref_answer_str"],
            metadata=prompts_data_replicated[i],
            update_step=current_update,
        )
        for i in range(total_samples)
    ]

    batch_rewards_dicts = reward_composer.batch_compute(contexts)
    rewards_total = mx.array([r["total"] for r in batch_rewards_dicts])
    rewards_breakdown = {
        k: [r[k] for r in batch_rewards_dicts] for k in batch_rewards_dicts[0]
    }

    # --- Advantage & Ref Log Probs ---
    grpo_algo = GRPOAlgorithm(config, model, ref_model)
    advantages = grpo_algo.compute_advantages(
        rewards_total, config.trainer.num_rollout_samples
    )

    full_seq = mx.concatenate([prompts_mx, responses_mx], axis=1)
    ref_logits = ref_model(full_seq.astype(mx.int64))[:, max_prompt_len - 1 : -1, :]
    ref_log_probs_all = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)
    ref_log_probs = mx.take_along_axis(
        ref_log_probs_all, responses_mx[..., None].astype(mx.int64), axis=-1
    ).squeeze(-1)

    response_mask = (responses_mx != pad_id).astype(mx.float32)
    response_mask = _mask_after_answer(responses_mx, response_mask, tokenizer, config)

    # --- Create Thinking/Answer Masks (if dual gradients enabled) ---
    thinking_mask = None
    answer_mask = None

    use_dual_gradients = (
        hasattr(config.trainer, 'use_dual_gradients')
        and config.trainer.use_dual_gradients
    )

    if use_dual_gradients:
        try:
            thinking_mask, answer_mask = _create_thinking_answer_masks(
                responses_mx,
                decoded,
                tokenizer,
                config,
                pad_id
            )

            # Validate masks
            thinking_token_count = mx.sum(thinking_mask).item()
            answer_token_count = mx.sum(answer_mask).item()
            total_token_count = mx.sum(response_mask).item()

            if thinking_token_count == 0 and answer_token_count == 0:
                logger.error(f"Dual gradient mask creation failed: Both masks are empty!")
                thinking_mask = None
                answer_mask = None
            elif thinking_token_count == 0:
                logger.warning(f"No thinking tokens detected in batch. All {int(answer_token_count)} tokens marked as answer.")
            elif answer_token_count == 0:
                logger.warning(f"No answer tokens detected in batch. All {int(thinking_token_count)} tokens marked as thinking.")
            else:
                logger.debug(f"Created masks - Thinking: {int(thinking_token_count)} tokens, Answer: {int(answer_token_count)} tokens, Total: {int(total_token_count)}")

        except Exception as e:
            logger.error(f"Failed to create thinking/answer masks: {e}. Dual gradients will be disabled for this batch.", exc_info=True)
            thinking_mask = None
            answer_mask = None

    # --- Logging ---
    _maybe_log_samples(
        config,
        current_update,
        prompts_data_replicated,
        decoded,
        rewards_breakdown,
        "n/a",
        run_id,
        is_invalid_batch,
    )

    # --- Build Rollout Batch ---
    rollout_batch = {
        "tokens": full_seq,
        "response_mask": response_mask,
        "advantages": advantages,
        "ref_log_probs": ref_log_probs,
        "actor_log_probs": actor_log_probs,
    }

    # Add thinking/answer masks if they were created
    if thinking_mask is not None and answer_mask is not None:
        rollout_batch["thinking_mask"] = thinking_mask
        rollout_batch["answer_mask"] = answer_mask

    avg_reward = mx.mean(rewards_total).item() if rewards_total.size > 0 else 0.0
    avg_breakdown = {k: np.mean(v) for k, v in rewards_breakdown.items()}

    model.train()
    if ref_model:
        ref_model.train()

    gc.collect()
    mx.clear_cache()

    return rollout_batch, avg_reward, avg_breakdown
