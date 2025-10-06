# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/__init__.py
# revision_no: 004
# goals_of_writing_code_block: Finalize the exports in __init__.py to include all utility functions and resolve the ImportError.
# type_of_code_response: change existing
"""Utility module initialization."""
# from .mlx_utils import _is_metal_internal_error, metal_recover, metal_safe_apply_gradients, _create_4d_attention_mask, safe_make_sampler, make_dynamic_tag_bias_processor, _letter_token_ids, _resolve_tag_ids, scale_grads_by_band, mask_grads_to_layer_band, mask_grads_to_specific_layers, ContentAlignBridge, _mask_after_answer, _first_token_ids_for_lexemes

from mlx_rl_trainer.utils.mlx_utils import (
    _is_metal_internal_error,
    metal_recover,
    metal_safe_apply_gradients,
    _create_4d_attention_mask,
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
    _first_token_ids_for_lexemes,
    _letter_token_ids,
    _resolve_tag_ids,
    scale_grads_by_band,
    mask_grads_to_layer_band,
    mask_grads_to_specific_layers,
    ContentAlignBridge,
    _mask_after_answer,
)

from .text_utils import (
    _preview,
    _strip_markup,
    _count_words,
    _tokenize_set,
    _normalize_ans_for_match,
    _contains_keywords,
    _contains_phrase,
    _extract_action_phrases,
    _extract_python_code,
    _get_static_reward_config,
    extract_think_region,
    extract_answer_region,
    _extract_think_answer_lengths,
    _indices_to_letters,
    _letters_to_canonical,
    _match_ref_to_option_index,
    _extract_mcq_options,
    _infer_gold_letters_from_meta,
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
    _tfidf_cosine,
    _extract_final_numeric,
    _extract_predicted_letters,
)
from .math_utils import safe_divide, safe_mean, safe_std, softmax, log_softmax
from .distributed import DistributedUtil

__all__ = [
    "metal_safe_apply_gradients",
    "_is_metal_internal_error",
    "metal_recover",
    "_create_4d_attention_mask",
    "safe_make_sampler",
    "make_dynamic_tag_bias_processor",
    "_first_token_ids_for_lexemes",
    "_letter_token_ids",
    "_resolve_tag_ids",
    "scale_grads_by_band",
    "mask_grads_to_layer_band",
    "mask_grads_to_specific_layers",
    "ContentAlignBridge",
    "_mask_after_answer",  # ADDED to mlx_utils exports
    "_preview",
    "_strip_markup",
    "_count_words",
    "_tokenize_set",
    "_normalize_ans_for_match",
    "_contains_keywords",
    "_contains_phrase",
    "_extract_action_phrases",
    "_extract_python_code",
    "_get_static_reward_config",
    "extract_think_region",
    "extract_answer_region",
    "_extract_think_answer_lengths",
    "_extract_final_numeric",  # ADDED to text_utils exports
    "_indices_to_letters",
    "_letters_to_canonical",
    "_match_ref_to_option_index",
    "_extract_mcq_options",
    "_infer_gold_letters_from_meta",
    "_mcq_meta_from_sample",
    "apply_chat_template_wrapper",
    "_tfidf_cosine",
    "safe_divide",
    "safe_mean",
    "safe_std",
    "softmax",
    "log_softmax",
    "DistributedUtil",
    "_extract_predicted_letters",
]
