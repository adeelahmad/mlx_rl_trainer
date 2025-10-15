import logging
import json
import gc
from typing import Dict, Any, List, Tuple, Optional, Union
from datasets import Dataset
import mlx.core as mx
import numpy as np

from mlx_lm.tokenizer_utils import TokenizerWrapper

import re

from mlx_rl_trainer.core.config import ExperimentConfig, DataConfig, GenerationConfig
from mlx_rl_trainer.utils.text_utils import (
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
    clean_completion_string,
)
from mlx_rl_trainer.rewards.format.tag_structure import (
    extract_think_region,
    extract_answer_region,
)

logger = logging.getLogger(__name__)


def _compose_prompt_from_sample(
    sample: Dict[str, Any]
) -> Tuple[str, Optional[str], Optional[str]]:
    ref_ans, ref_think = None, None

    if "prompt" in sample and isinstance(sample["prompt"], str):
        prompt_text = sample["prompt"]
    elif "question" in sample and isinstance(sample["question"], str):
        prompt_text = sample["question"]
    else:
        prompt_text = json.dumps(sample, ensure_ascii=False)

    completion = clean_completion_string(
        sample.get("completion", sample.get("answer", ""))
    )
    if isinstance(completion, str):
        gen_config = GenerationConfig()
        ref_think = extract_think_region(completion, gen_config)
        ref_ans = extract_answer_region(completion, gen_config) or completion.strip()

    return prompt_text, ref_ans, ref_think


def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: Union[ExperimentConfig, DataConfig],
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    """
    Build a batch of rollout prompts with memory-optimized processing.

    Memory optimizations:
    - Pre-allocate output array to avoid list growth
    - Use numpy for intermediate operations
    - Minimize intermediate allocations
    - Clean up temporary variables immediately
    """
    if not indices:
        return [], mx.array([], dtype=mx.int32), 0

    # Extract data_config and system_prompt based on config type
    if isinstance(config, ExperimentConfig):
        data_config = config.data
        system_prompt = getattr(config, "system_prompt", None) or ""
    else:
        data_config = config
        system_prompt = getattr(config, "system_prompt", None) or ""

    pad_id = tokenizer.pad_token_id

    # First pass: collect tokens and find max length
    # Use a more memory-efficient approach by not storing everything at once
    prompts_data: List[Dict[str, Any]] = []
    token_lists: List[List[int]] = []
    max_len_in_batch = 0

    for i in indices:
        try:
            raw = dataset[i]
            prompt_text, ref_ans, ref_think = _compose_prompt_from_sample(raw)

            mcq_meta = _mcq_meta_from_sample(
                {
                    "prompt": prompt_text,
                    "completion": ref_ans,
                    "meta": raw.get("meta", {}),
                }
            )

            # Apply chat template with system prompt
            formatted_prompt = apply_chat_template_wrapper(tokenizer, prompt_text, "")

            # Encode without adding special tokens since they should be in the template
            p_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

            # Truncate if necessary (keeping the end, which is the actual prompt)
            if len(p_tokens) > data_config.max_prompt_len:
                p_tokens = p_tokens[-data_config.max_prompt_len :]

            if not p_tokens:
                logger.warning(f"Skipping empty prompt (idx {i}).")
                # Clean up immediately
                del (
                    raw,
                    prompt_text,
                    ref_ans,
                    ref_think,
                    mcq_meta,
                    formatted_prompt,
                    p_tokens,
                )
                continue

            # Store only essential data in prompts_data (no tokens to avoid duplication)
            entry = {
                "original_index": i,
                "text": formatted_prompt,
                "tokens": p_tokens,  # Keep for now, will be padded version
                "ref_answer_str": ref_ans,
                "ref_think_str": ref_think,
                "ref": raw,
                "is_invalid_sample": raw.get("is_invalid_sample", False),
            }
            entry.update(mcq_meta)

            token_lists.append(p_tokens)
            max_len_in_batch = max(max_len_in_batch, len(p_tokens))
            prompts_data.append(entry)

            # Clean up intermediate variables
            del raw, prompt_text, ref_ans, ref_think, mcq_meta, formatted_prompt

        except Exception as e:
            logger.warning(f"Skipping sample idx {i} due to error: {e}")
            continue

    if not prompts_data:
        # Clean up
        del token_lists
        return [], mx.array([], dtype=mx.int32), 0

    # Pre-allocate numpy array for padded tokens (more memory efficient than list)
    batch_size = len(token_lists)
    padded_array = np.full((batch_size, max_len_in_batch), pad_id, dtype=np.int32)

    # Fill the pre-allocated array with left-padding
    for idx, tok in enumerate(token_lists):
        tok_len = len(tok)
        pad_len = max_len_in_batch - tok_len
        # Left-pad: fill from pad_len onwards
        padded_array[idx, pad_len:] = tok

        # Update prompts_data with padded tokens (keep original tokens for reference)
        # Note: We keep tokens in prompts_data for compatibility
        prompts_data[idx]["tokens"] = [pad_id] * pad_len + tok

    # Convert to mlx array
    prompts_mx = mx.array(padded_array, dtype=mx.int32)

    # Clean up intermediate structures
    del token_lists, padded_array
    gc.collect()

    return prompts_data, prompts_mx, max_len_in_batch
