import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from datasets import Dataset
import mlx.core as mx

from mlx_rl_trainer.core.config import ExperimentConfig, GenerationConfig
from mlx_rl_trainer.utils.text_utils import _mcq_meta_from_sample, apply_chat_template_wrapper, extract_think_region, extract_answer_region
from mlx_lm.tokenizer_utils import TokenizerWrapper

logger = logging.getLogger(__name__)

def _compose_prompt_from_sample(sample: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    ref_ans, ref_think = None, None
    gen_config = GenerationConfig()

    prompt_text = sample.get('prompt', sample.get('question', ''))
    completion = sample.get('completion', sample.get('answer', ''))
    
    if isinstance(completion, str):
        ref_think = extract_think_region(completion, gen_config)
        ref_ans = extract_answer_region(completion, gen_config) or completion.strip()
    
    return prompt_text, ref_ans, ref_think

def build_rollout_batch(
    tokenizer: TokenizerWrapper,
    dataset: Dataset,
    indices: List[int],
    config: ExperimentConfig,
) -> Tuple[List[Dict[str, Any]], mx.array, int]:
    
    prompts_data: List[Dict[str, Any]] = []
    max_len_in_batch = 0
    pad_id = tokenizer.pad_token_id

    for i in indices:
        try:
            raw = dataset[i]
            prompt_text, ref_ans, ref_think = _compose_prompt_from_sample(raw)
            
            mcq_meta = _mcq_meta_from_sample({'prompt': prompt_text, 'completion': ref_ans, 'meta': raw.get('meta', {})})
            
            formatted_prompt = apply_chat_template_wrapper(tokenizer, prompt_text, config.system_prompt)
            p_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=True)

            # CORRECTED: Access max_prompt_len via config.data
            if len(p_tokens) > config.data.max_prompt_len:
                p_tokens = p_tokens[-config.data.max_prompt_len:]
            if not p_tokens:
                logger.warning(f"Skipping empty prompt (idx {i}).")
                continue

            entry = {
                'original_index': i,
                'text': formatted_prompt,
                'tokens': p_tokens,
                'ref_answer_str': ref_ans,
                'ref_think_str': ref_think,
                'is_invalid_sample': raw.get('is_invalid_sample', False),
                'meta': raw.get('meta', {}) # Pass original meta along for reward context
            }
            entry['meta'].update(mcq_meta)
            prompts_data.append(entry)
            max_len_in_batch = max(max_len_in_batch, len(p_tokens))

        except Exception as e:
            logger.warning(f"Skipping sample idx {i} due to error during batch building: {e}", exc_info=True)

    if not prompts_data:
        return [], mx.array([], dtype=mx.int32), 0

    padded_tokens = []
    for p in prompts_data:
        tok = p['tokens']
        pad_len = max_len_in_batch - len(tok)
        padded_tokens.append([pad_id] * pad_len + tok)

    return prompts_data, mx.array(padded_tokens, dtype=mx.int32), max_len_in_batch
