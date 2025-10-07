import logging, gc, re
from typing import Dict, Any, List, Optional, Tuple
import mlx.core as mx, mlx.nn as nn
from mlx_lm.models import cache
from mlx_lm.tokenizer_utils import TokenizerWrapper
import numpy as np

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.generation.caching import PagedKVCache
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


def generate_rollouts_for_batch(
    model: nn.Module,
    ref_model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompts_data: List[Dict],
    dataset: "Dataset",
    config: ExperimentConfig,
    reward_composer: RewardComposer,
    paged_kv_cache: Optional[PagedKVCache],
    run_id: str,
    current_update: int,
    is_invalid_batch: bool,
) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
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
    decoded = tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=True)

    contexts = [
        reward_composer.context_cls(
            generated_text=decoded[i],
            prompt_text=prompts_data_replicated[i]["text"],
            reference_completion=prompts_data_replicated[i]["ref_answer_str"],
            metadata=prompts_data_replicated[i],
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

    rollout_batch = {
        "tokens": full_seq,
        "response_mask": response_mask,
        "advantages": advantages,
        "ref_log_probs": ref_log_probs,
        "actor_log_probs": actor_log_probs,
    }

    avg_reward = mx.mean(rewards_total).item() if rewards_total.size > 0 else 0.0
    avg_breakdown = {k: np.mean(v) for k, v in rewards_breakdown.items()}

    model.train()
    if ref_model:
        ref_model.train()  # Set back to train mode if it was changed
    gc.collect()
    mx.clear_cache()

    return rollout_batch, avg_reward, avg_breakdown
