import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from datasets import Dataset
import mlx.core as mx


from mlx_rl_trainer.core.config import ExperimentConfig, DataConfig, GenerationConfig
from mlx_rl_trainer.utils.text_utils import (
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
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

    completion = sample.get("completion", sample.get("answer", ""))
    if isinstance(completion, str):
        gen_config = GenerationConfig()
        ref_think = extract_think_region(completion, gen_config)
        ref_ans = extract_answer_region(completion, gen_config) or completion.strip()

    return prompt_text, ref_ans, ref_think


def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: Union[ExperimentConfig, DataConfig],  # Can receive either config type
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    prompts_data: List[Dict[str, Any]] = []
    max_len_in_batch = 0
    pad_id = tokenizer.pad_token_id

    # --- FIX START ---
    # Check if we have the full ExperimentConfig or just the DataConfig part
    if hasattr(config, 'data'):
        # It's the full ExperimentConfig
        data_config = config.data
        system_prompt = config.system_prompt
    else:
        # It's just the DataConfig
        data_config = config
        # DataConfig doesn't have a system_prompt, so we use an empty string.
        # This is fine because this path is usually taken by the dataloader,
        # where the final system prompt isn't critical.
        system_prompt = ""
    # --- FIX END ---

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

            formatted_prompt = apply_chat_template_wrapper(
                tokenizer, prompt_text, system_prompt
            )
            p_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

            if len(p_tokens) > data_config.max_prompt_len:
                p_tokens = p_tokens[-data_config.max_prompt_len :]
            if not p_tokens:
                logger.warning(f"Skipping empty prompt (idx {i}).")
                continue

            entry = {
                "original_index": i,
                "text": formatted_prompt,
                "tokens": p_tokens,
                "ref_answer_str": ref_ans,
                "ref_think_str": ref_think,
                "is_invalid_sample": raw.get("is_invalid_sample", False),
            }
            entry.update(mcq_meta)
            prompts_data.append(entry)
            max_len_in_batch = max(max_len_in_batch, len(p_tokens))

        except Exception as e:
            logger.warning(f"Skipping sample idx {i} due to error: {e}")

    if not prompts_data:
        return [], mx.array([], dtype=mx.int32), 0

    padded_tokens = []
    for p in prompts_data:
        tok = p["tokens"]
        pad_len = max_len_in_batch - len(tok)
        padded_tokens.append([pad_id] * pad_len + tok)

    return prompts_data, mx.array(padded_tokens, dtype=mx.int32), max_len_in_batch
