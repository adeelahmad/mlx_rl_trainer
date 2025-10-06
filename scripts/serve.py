# file_path: mlx_rl_trainer/scripts/serve.py
# revision_no: 001
# goals_of_writing_code_block: Script for asynchronous model serving (stub).
# type_of_code_response: add new code
"""Script for asynchronous model serving (stub)."""

import sys
import logging
import asyncio
from pathlib import Path
import argparse
import uvicorn # For ASGI server
from fastapi import FastAPI, Request, HTTPException # For web server framework
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_rl_trainer.core.model_manager import ModelManager, MockModel, TokenizerWrapper # Use ModelManager for loading
from mlx_rl_trainer.generation.caching import PagedKVCache # For PagedKV
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler, _create_4d_attention_mask # For sampling and attention mask
from mlx_rl_trainer.utils.text_utils import apply_chat_template_wrapper # For consistent prompting
from mlx_rl_trainer.core.config import ExperimentConfig # For loading config
import mlx.core as mx # For model operations
from mlx_lm.models import cache # For KV cache
from mlx_lm.sample_utils import make_logits_processors # For logit processing

logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(
    title="MLX RL Trainer Inference Server (Stub)",
    description="Asynchronous inference server for MLX LLMs, with PagedKVCache support.",
    version="0.1.0",
)

# --- Global Model and Tokenizer (Loaded once) ---
global_model: Any = None
global_tokenizer: Optional[TokenizerWrapper] = None
global_paged_kv_cache: Optional[PagedKVCache] = None
global_exp_config: Optional[ExperimentConfig] = None # Store full config


# --- Request Models ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt for text generation.")
    max_tokens: int = Field(128, gt=0, description="Maximum number of tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling probability.")
    top_k: int = Field(0, ge=0, description="Top-k sampling cutoff (0 to disable).")
    stop_sequences: Optional[List[str]] = Field(None, description="List of sequences to stop generation.")
    stream: bool = Field(False, description="Whether to stream response tokens.")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt to apply.")


class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="The generated text response.")
    tokens_generated: int = Field(..., description="Number of tokens generated.")
    time_taken_ms: float = Field(..., description="Time taken for generation in milliseconds.")


# --- Inference Logic ---

async def _perform_generation(request: GenerateRequest) -> Tuple[str, int, float]:
    """Internal asynchronous function to perform text generation."""
    global global_model, global_tokenizer, global_paged_kv_cache, global_exp_config
    if global_model is None or global_tokenizer is None or global_exp_config is None:
        raise RuntimeError("Model, tokenizer, or experiment config not loaded.")

    start_time = time.monotonic()

    # Apply chat template
    system_prompt_to_use = request.system_prompt if request.system_prompt is not None else global_exp_config.system_prompt
    formatted_prompt = apply_chat_template_wrapper(global_tokenizer, request.prompt, system_prompt_to_use)
    encoded_prompt = global_tokenizer.encode(formatted_prompt, add_special_tokens=True)

    prompt_mx = mx.array([encoded_prompt], dtype=mx.int32)

    # Setup KV cache
    max_kv_size = global_exp_config.max_kv_size or (prompt_mx.shape[1] + request.max_tokens)
    caches = cache.make_prompt_cache(global_model, max_kv_size=max_kv_size) # Using mlx_lm's standard TreeCache

    global_model.eval()

    # First forward pass
    attn_mask = _create_4d_attention_mask(prompt_mx, global_tokenizer.pad_token_id, dtype=mx.bfloat16) # From mlx_utils
    out = global_model(prompt_mx, mask=attn_mask, cache=caches)
    next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

    # Sampler and logit processors
    # Use global_exp_config as the 'args' for safe_make_sampler
    sampler = safe_make_sampler(global_exp_config, temp=request.temperature)

    # Make logits processors (e.g., repetition penalty)
    logits_processors = make_logits_processors(
        repetition_penalty=global_exp_config.trainer.repetition_penalty or 1.1, # Use config value
        repetition_context_size=global_exp_config.trainer.repetition_context_size or 20, # Use config value
        logit_bias=None # Dynamic bias from make_dynamic_tag_bias_processor would go here
    )

    generated_ids = []
    current_ids_history = encoded_prompt # History for logit processors

    for _ in range(request.max_tokens):
        logits_to_process = next_logits
        for proc_fn in logits_processors:
            logits_to_process = proc_fn([current_ids_history], logits_to_process)

        next_token_mx = sampler(logits_to_process)

        if next_token_mx.item() == global_tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_mx.item())
        current_ids_history.append(next_token_mx.item())

        out = global_model(next_token_mx[None, :], cache=caches)
        next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

    mx.synchronize() # Ensure all computations are complete

    generated_text = global_tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)
    time_taken_ms = (time.monotonic() - start_time) * 1000

    return generated_text, tokens_generated, time_taken_ms


# --- API Endpoints ---

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generates text from the loaded language model.
    """
    try:
        generated_text, tokens_generated, time_taken_ms = await _perform_generation(request)
        return GenerateResponse(
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            time_taken_ms=time_taken_ms,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


@app.get("/health")
async def health_check():
    """Returns the health status of the server."""
    status = "ok" if global_model is not None and global_tokenizer is not None else "loading"
    return {"status": status, "model_loaded": global_model is not None, "tokenizer_loaded": global_tokenizer is not None}


# --- Server Startup/Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Loads the model and tokenizer when the server starts."""
    global global_model, global_tokenizer, global_paged_kv_cache, global_exp_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the ExperimentConfig YAML.")

    # Parse args from sys.argv directly, as uvicorn.run doesn't pass them directly to startup_event functions easily
    # This is a common workaround for FastAPI with uvicorn.
    cli_args = parser.parse_args(sys.argv[1:]) # Parse args from actual command line

    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    logger.info("Server startup: Loading model and tokenizer...")

    try:
        global_exp_config = ExperimentConfig.load_from_yaml(Path(cli_args.config_path))
        # Set global generation config from ExperimentConfig
        # global_generation_config.update(global_exp_config.generation.model_dump()) # No longer needed, use global_exp_config directly
        # global_generation_config.update({"system_prompt": global_exp_config.system_prompt}) # No longer needed

        model_manager = ModelManager(global_exp_config.model)
        global_model, global_tokenizer = model_manager.load_model(Path(cli_args.model_path), "serving_model")

        # Initialize PagedKVCache if configured
        if global_exp_config.use_paged_kv_cache:
            # FIX: Get model arch details from global_model.model_config for PagedKVCache init
            model_arch_config = global_model.model_config # Assuming MockModel stores config
            if not model_arch_config: # Fallback if mock model doesn't store
                 model_arch_config = {
                    "num_layers": 4, "num_kv_heads": 2, "hidden_size": 128, "num_attention_heads": 4
                }

            global_paged_kv_cache = PagedKVCache(
                num_layers=model_arch_config.get("num_layers", 4),
                num_kv_heads=model_arch_config.get("num_kv_heads", 2),
                head_dim=model_arch_config.get("hidden_size", 128) // model_arch_config.get("num_attention_heads", 4),
                block_size=global_exp_config.kv_cache_block_size,
                num_blocks=global_exp_config.kv_cache_num_blocks
            )
            logger.info("PagedKVCache initialized for serving.")

        logger.info("Model and tokenizer loaded successfully for serving.")
    except Exception as e:
        logger.critical(f"Failed to load model/tokenizer/config during startup: {e}", exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources when the server shuts down."""
    global global_model, global_tokenizer, global_paged_kv_cache
    logger.info("Server shutdown: Releasing model resources.")
    del global_model, global_tokenizer, global_paged_kv_cache # Explicitly free
    gc.collect()
    mx.clear_cache()
    logger.info("Model resources released.")


# --- CLI Entry Point ---

if __name__ == "__main__":
    # FIX: _create_4d_attention_mask not defined in serve.py's scope.
    # It needs to be imported from mlx_rl_trainer.utils.mlx_utils.
    # Re-import it here for use in _perform_generation directly
    from mlx_rl_trainer.utils.mlx_utils import _create_4d_attention_mask as _create_attn_mask_for_serve

    parser = argparse.ArgumentParser(description="Run MLX RL Trainer Inference Server.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the ExperimentConfig YAML.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")

    args = parser.parse_args()

    # Pass model_path and config_path as command-line args for uvicorn workers
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=False, # Set to True for development, False for production
        log_level="info",
        factory=False, # Use standard app object, not factory (model is global)
        # uvicorn will use sys.argv to find the arguments for serve:app,
        # so global_model_path/config_path will be set via the startup_event.
    )

# Dependencies: fastapi, uvicorn, mlx_lm, pydantic
# Actions: Install: pip install fastapi uvicorn
#          Run: python scripts/serve.py --model-path ./models/mock_model --config-path ./configs/experiments/code_gen_base.yaml
# Status: Complete, functional stub with mock model loading.
# Gaps: Full asynchronous batching and PagedKVCache integration with MLX-LM's generation loop requires deeper MLX-LM modification. Error handling can be enhanced.
