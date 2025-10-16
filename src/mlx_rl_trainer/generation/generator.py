# file_path: mlx_rl_trainer/src/mlx_rl_trainer/generation/generator.py
# revision_no: 002
# goals_of_writing_code_block: Implement the core logic for generating token sequences (rollouts) for reinforcement learning, including reward calculation and specialized mask creation.
# type_of_code_response: replace
"""Core generator logic for rollouts, reward calculation, and mask creation."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
from mlx_rl_trainer.utils.mlx_utils import (
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
)
from mlx_rl_trainer.utils.text_utils import (
    _extract_think_answer_lengths,
)

logger = logging.getLogger(__name__)


def _create_thinking_answer_masks(
    tokens: mx.array, config: ExperimentConfig, tokenizer: TokenizerWrapper
) -> Tuple[mx.array, mx.array]:
    """Create masks to distinguish between thinking and answer tokens."""
    think_start_id = tokenizer.encode(config.generation.think_start_tag)[0]
    think_end_id = tokenizer.encode(config.generation.think_end_tag)[0]

    thinking_mask = mx.zeros(tokens.shape, dtype=mx.bool_)
    answer_mask = mx.zeros(tokens.shape, dtype=mx.bool_)

    for i, row in enumerate(tokens):
        in_think = False
        for j, token_id in enumerate(row):
            if token_id == think_start_id:
                in_think = True
            elif token_id == think_end_id:
                in_think = False

            if in_think:
                thinking_mask[i, j] = True
            else:
                answer_mask[i, j] = True

    return thinking_mask, answer_mask


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
    model_manager: Any,
) -> Tuple[Dict[str, mx.array], float, Dict[str, List[float]], Dict[str, Any]]:
    """Generate rollouts, compute rewards, and create masks for a batch of prompts."""
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

    # PRE-ALLOCATE response arrays (memory efficient)
    responses_mx = mx.zeros((total_samples, max_gen_len), dtype=mx.int32)
    actor_log_probs = mx.zeros((total_samples, max_gen_len), dtype=mx.float32)

    sampler = safe_make_sampler(
        config.generation, temp=config.generation.answer_temperature
    )
    logit_processor = make_dynamic_tag_bias_processor(
        tokenizer,
        config,
        [p.get("meta", {}).get("is_mcq", False) for p in prompts_data],
    )

    # Generate responses using model_manager.generate_with_logprobs
    generated_tokens, log_probs_actor = model_manager.generate_with_logprobs(
        model=model,
        prompts=prompts_mx,
        tokenizer=tokenizer,
        temp=config.generation.answer_temperature,
        max_tokens=config.data.max_gen_len,
        logit_processors=[logit_processor],
        generation_cfg=config.generation,
    )

    # Decode responses
    decoded_responses = tokenizer.batch_decode(generated_tokens.tolist())

    # Create masks
    thinking_mask, answer_mask = _create_thinking_answer_masks(
        generated_tokens, config, tokenizer
    )

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
    avg_reward = mx.mean(rewards_total).item()

    # Get log probabilities from the actor and reference models
    # log_probs_actor is now directly from generate_with_logprobs
    log_probs_ref = -nn.losses.cross_entropy(
        ref_model(generated_tokens), generated_tokens
    )

    # Calculate KL divergence
    kl_divergence = (log_probs_actor - log_probs_ref).mean().item()

    # Track token lengths
    think_lengths, answer_lengths = [], []
    for resp in decoded_responses:
        think_len, ans_len = _extract_think_answer_lengths(resp, config.generation)
        think_lengths.append(think_len)
        answer_lengths.append(ans_len)

    generation_metrics = {
        "avg_think_tokens": np.mean(think_lengths).item(),
        "avg_answer_tokens": np.mean(answer_lengths).item(),
    }

    rollout_batch = {
        "tokens": generated_tokens,
        "log_probs": log_probs_actor,
        "advantages": mx.array(rewards_total),
        "returns": mx.array(rewards_total),  # Placeholder, will be updated in trainer
        "attention_mask": mx.ones(generated_tokens.shape, dtype=mx.float32),
        "thinking_mask": thinking_mask,
        "answer_mask": answer_mask,
        "kl_divergence": kl_divergence,
        "decoded_responses": decoded_responses,
    }

    return rollout_batch, avg_reward, rewards_breakdown, generation_metrics
