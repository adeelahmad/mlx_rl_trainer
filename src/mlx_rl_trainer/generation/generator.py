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
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler, make_dynamic_tag_bias_processor
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

    thinking_mask = mx.zeros_like(tokens, dtype=mx.bool_)
    answer_mask = mx.zeros_like(tokens, dtype=mx.bool_)

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
    prompts_data: List[Dict[str, Any]],
    prompts_mx: mx.array,
    config: ExperimentConfig,
    reward_composer: RewardComposer,
    paged_kv_cache: Optional[Any],
    model_manager: Any,
) -> Tuple[Dict[str, mx.array], float, Dict[str, List[float]], Dict[str, float]]:
    """Generate rollouts, compute rewards, and create masks for a batch of prompts."""
    sampler = safe_make_sampler(config.generation, temp=config.generation.answer_temperature)
    logit_processor = make_dynamic_tag_bias_processor(
        tokenizer, config, [p.get("meta", {}).get("is_mcq", False) for p in prompts_data]
    )

    # Generate responses using model_manager.generate_with_logprobs
    generated_tokens, log_probs_actor = model_manager.generate_with_logprobs(
        model=model,
        prompts=prompts_mx,
        tokenizer=tokenizer,
        temp=config.generation.answer_temperature,
        max_tokens=config.data.max_gen_len,
        cache=paged_kv_cache,
        logit_processors=[logit_processor],
        generation_cfg=config.generation,
    )

    # Decode responses
    decoded_responses = tokenizer.batch_decode(generated_tokens.tolist())

    # Create masks
    thinking_mask, answer_mask = _create_thinking_answer_masks(generated_tokens, config, tokenizer)

    # Compute rewards
    rewards, raw_rewards = reward_composer.batch_compute(
        prompts_data, decoded_responses, config
    )
    avg_reward = np.mean(rewards).item()

    # Get log probabilities from the actor and reference models
    # log_probs_actor is now directly from generate_with_logprobs
    log_probs_ref = -nn.cross_entropy(ref_model(generated_tokens), generated_tokens)

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
        "advantages": mx.array(rewards),
        "returns": mx.array(rewards), # Placeholder, will be updated in trainer
        "attention_mask": mx.ones_like(generated_tokens, dtype=mx.bool_),
        "thinking_mask": thinking_mask,
        "answer_mask": answer_mask,
        "kl_divergence": kl_divergence,
        "decoded_responses": decoded_responses,
    }

    return rollout_batch, avg_reward, raw_rewards, generation_metrics