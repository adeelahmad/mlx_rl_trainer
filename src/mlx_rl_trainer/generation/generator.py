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
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
    _mask_after_answer,
)
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from mlx_rl_trainer.algorithms.grpo.grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


def _create_thinking_answer_masks(
    responses_mx: mx.array,
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
) -> Tuple[mx.array, mx.array, Dict[str, float]]:
    batch_size, seq_len = responses_mx.shape
    think_mask = mx.zeros_like(responses_mx, dtype=mx.float32)
    answer_mask = mx.zeros_like(responses_mx, dtype=mx.float32)

    think_start_tag, think_end_tag = (
        config.generation.think_start_tag,
        config.generation.think_end_tag,
    )

    total_think_tokens, total_answer_tokens = 0, 0

    for i in range(batch_size):
        tokens = responses_mx[i].tolist()
        try:
            decoded_text = tokenizer.decode(tokens)
            think_match = re.search(
                f"{re.escape(think_start_tag)}(.*?){re.escape(think_end_tag)}",
                decoded_text,
                re.DOTALL,
            )

            if think_match:
                # Approximate token spans from character spans
                start_char, end_char = think_match.span(1)
                start_tok = len(
                    tokenizer.encode(
                        decoded_text[:start_char], add_special_tokens=False
                    )
                )
                end_tok = len(
                    tokenizer.encode(decoded_text[:end_char], add_special_tokens=False)
                )

                think_mask[i, start_tok:end_tok] = 1.0
                num_think_toks = end_tok - start_tok
                total_think_tokens += num_think_toks

                answer_start_char = think_match.end(0)
                answer_start_tok = len(
                    tokenizer.encode(
                        decoded_text[:answer_start_char], add_special_tokens=False
                    )
                )

                answer_mask[i, answer_start_tok:] = 1.0
                num_answer_toks = seq_len - answer_start_tok
                total_answer_tokens += num_answer_toks
            else:
                answer_mask[i, :] = 1.0
                total_answer_tokens += seq_len
        except Exception as e:
            logger.warning(f"Failed to create masks: {e}")
            answer_mask[i, :] = 1.0
            total_answer_tokens += seq_len

    metrics = {
        "tokens/total_think": total_think_tokens,
        "tokens/total_answer": total_answer_tokens,
    }
    return think_mask, answer_mask, metrics


def generate_rollouts_for_batch(
    model,
    ref_model,
    tokenizer,
    prompts_data,
    dataset,
    config,
    reward_composer,
    run_id,
    current_update,
    is_invalid_batch,
):
    model.eval()
    if ref_model:
        ref_model.eval()

    if not prompts_data:
        return {}, 0.0, {}, {}

    samples_per_prompt = config.trainer.num_rollout_samples
    expanded_prompts = [p for p in prompts_data for _ in range(samples_per_prompt)]

    _, prompts_mx, prompt_len = build_rollout_batch(
        tokenizer, dataset, [p["original_index"] for p in expanded_prompts], config
    )
    batch_size = prompts_mx.shape[0]
    max_gen_len = config.data.max_gen_len

    responses_mx = mx.zeros((batch_size, max_gen_len), dtype=mx.int32)
    logprobs_mx = mx.zeros((batch_size, max_gen_len), dtype=mx.float32)

    kv_cache = cache.Cache(model.n_kv_heads, model.head_dim, config.max_kv_size)
    logits = model(prompts_mx, cache=kv_cache)

    for i in range(max_gen_len):
        logits = logits[:, -1, :]
        sampler = safe_make_sampler(config, temp=config.generation.answer_temperature)
        next_tokens = sampler(logits)
        log_probs = nn.log_softmax(logits, axis=-1)
        next_log_probs = mx.take_along_axis(
            log_probs, next_tokens[:, None], axis=-1
        ).squeeze(-1)
        responses_mx[:, i], logprobs_mx[:, i] = next_tokens, next_log_probs
        logits = model(next_tokens[:, None], cache=kv_cache)

    contexts = [
        reward_composer.context_cls(
            generated_text=tokenizer.decode(responses_mx[i].tolist()),
            prompt_text=p["text"],
            reference_completion=p["ref_answer_str"],
            metadata=p,
            update_step=current_update,
        )
        for i, p in enumerate(expanded_prompts)
    ]
    rewards_list = reward_composer.batch_compute(contexts)
    rewards_flat = mx.array([r["total"] for r in rewards_list])
    raw_reward_metrics = {
        k: [r[k] for r in rewards_list] for k in rewards_list[0] if k != "total"
    }

    grpo_algo = GRPOAlgorithm(config, model, ref_model)
    advantages = grpo_algo.compute_advantages(rewards_flat, samples_per_prompt)

    full_sequence = mx.concatenate([prompts_mx, responses_mx], axis=1)
    ref_logits = ref_model(full_sequence)[:, prompt_len - 1 : -1, :]
    ref_logprobs = mx.take_along_axis(
        nn.log_softmax(ref_logits, -1), responses_mx[..., None], -1
    ).squeeze(-1)

    response_mask = (responses_mx != tokenizer.pad_token_id).astype(mx.float32)

    think_mask, answer_mask, token_metrics = _create_thinking_answer_masks(
        responses_mx, tokenizer, config
    )

    rollout_data = {
        "tokens": full_sequence,
        "response_mask": response_mask,
        "advantages": advantages,
        "ref_log_probs": ref_logprobs,
        "actor_log_probs": logprobs_mx,
        "thinking_mask": think_mask,
        "answer_mask": answer_mask,
    }

    _maybe_log_samples(
        config,
        current_update,
        expanded_prompts,
        [tokenizer.decode(r) for r in responses_mx.tolist()],
        raw_reward_metrics,
        "n/a",
        run_id,
        is_invalid_batch,
    )

    model.train()

    gen_metrics = {
        "avg_response_length": float(mx.sum(response_mask) / batch_size),
        **token_metrics,
    }

    return (
        rollout_data,
        mx.mean(rewards_flat).item(),
        {k: np.mean(v) for k, v in raw_reward_metrics.items()},
        gen_metrics,
    )
