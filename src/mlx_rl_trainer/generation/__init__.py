# file_path: mlx_rl_trainer/src/mlx_rl_trainer/generation/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Initialize the generation module.
# type_of_code_response: add new code
"""Text generation utilities."""
from .caching import PagedKVCache # Import PagedKVCache from caching submodule
from .generator import generate_rollouts_for_batch # FIX: Change to direct import for rollout generator
from .async_server import AsyncBatchGenerator, run_async_inference_server # Import sampler specific utilities
from .sampler_utils import make_dynamic_tag_bias_processor, safe_make_sampler, selective_softmax # Import sampler specific utilities

# Add common generation types/helpers here for easier import
__all__ = [
    "PagedKVCache", "generate_rollouts_for_batch", "AsyncBatchGenerator", "run_async_inference_server",
    "make_dynamic_tag_bias_processor", "safe_make_sampler", "selective_softmax"
]
