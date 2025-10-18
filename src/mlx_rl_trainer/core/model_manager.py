# src/mlx_rl_trainer/core/model_manager.py

"""
Enhanced Model Management with Memory Optimization and Better Error Handling

ENHANCEMENTS:
1. Memory cleanup after model loading
2. Better error messages with suggestions
3. Model verification after loading
4. Statistics tracking
5. Graceful degradation on errors

BACKWARD COMPATIBLE: All existing functionality preserved
"""
import json
import logging
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from rich import print as rprint

from .config import ModelConfig
from .exceptions import ModelLoadError

try:
    from mlx_lm import load, generate
    from mlx_lm.models import cache as mlx_lm_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
        load_adapters,
    )
    from mlx_lm.utils import save_config

    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False

    class TokenizerWrapper:
        pass

    class MLXLoRALinear:
        pass

    def load(*args, **kwargs):
        raise ImportError("mlx-lm not installed.")

    def linear_to_lora_layers(*args, **kwargs):
        pass

    def print_trainable_parameters(*args, **kwargs):
        pass

    def load_adapters(*args, **kwargs):
        pass

    def save_config(*args, **kwargs):
        pass

    class mlx_lm_cache:
        @staticmethod
        def make_prompt_cache(*args, **kwargs):
            return None


logger = logging.getLogger(__name__)


def _aggressive_memory_cleanup():
    """Aggressively free memory."""
    try:
        mx.metal.clear_cache()
    except:
        pass
    mx.clear_cache()
    gc.collect()


def _verify_model_loaded(model: nn.Module, model_name: str) -> Tuple[bool, str]:
    """
    Verify that a model loaded correctly.

    Returns:
        (is_valid, message)
    """
    try:
        # --- FIX START ---
        # Get a flat list of (name, parameter_array) tuples using the correct MLX API
        flat_params = tree_flatten(model.parameters())

        # Check if the model has any parameters
        if not flat_params:
            return False, "Model has no parameters"

        # Check the first parameter for NaN/Inf values
        param_values = [p for _, p in flat_params]
        # --- FIX END ---

        if param_values:
            first_param = param_values[0]
            if mx.any(mx.isnan(first_param)).item():
                return False, "Model contains NaN values"
            if mx.any(mx.isinf(first_param)).item():
                return False, "Model contains Inf values"

        return True, "Model verified"

    except Exception as e:
        return False, f"Verification failed: {e}"


class ModelManager:
    """
    Enhanced model manager with better error handling and memory management.

    FEATURES:
    - Memory cleanup after operations
    - Model verification after loading
    - Better error messages
    - Statistics tracking
    - Graceful error recovery
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        if not MLX_LM_AVAILABLE:
            raise RuntimeError(
                "mlx-lm is required but not available. "
                "Install with: pip install mlx-lm"
            )

        # Statistics
        self.models_loaded = 0
        self.load_errors = 0
        self.generation_count = 0

        logger.info("ModelManager initialized")

    def make_prompt_cache(
        self, model: nn.Module, max_kv_size: Optional[int] = None
    ) -> Any:
        """Create a KV cache for the model."""
        try:
            cache = mlx_lm_cache.make_prompt_cache(model, max_kv_size=max_kv_size)
            return cache
        except Exception as e:
            logger.warning(f"Failed to create prompt cache: {e}")
            return None

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Any]:
        """
        Load a model and tokenizer with verification.

        ENHANCED:
        - Memory cleanup after load
        - Model verification
        - Better error messages
        """
        logger.info(f"Loading '{type_name}' model from {model_path}...")

        try:
            # Check path exists
            if not Path(model_path).exists():
                raise ModelLoadError(
                    f"Model path does not exist: {model_path}\n"
                    f"Suggestions:\n"
                    f"  - Verify the path is correct\n"
                    f"  - Check if the model needs to be downloaded\n"
                    f"  - Ensure you have read permissions"
                )

            # Load model and tokenizer
            model_instance, tokenizer_instance = load(str(model_path))

            # Verify model loaded correctly
            is_valid, verify_message = _verify_model_loaded(model_instance, type_name)
            if not is_valid:
                raise ModelLoadError(
                    f"Model verification failed for '{type_name}': {verify_message}"
                )

            rprint(f"✓ Loaded '{type_name}' model from [green]{model_path}[/green]")

            # Apply LoRA if requested
            if apply_lora and lora_config:
                model_instance = self._apply_lora_to_model(
                    model_instance, type_name, lora_config
                )

            # Set mode
            if is_trainable:
                model_instance.train()
                logger.info(f"Set '{type_name}' to training mode")
                print_trainable_parameters(model_instance)
            else:
                model_instance.eval()
                logger.info(f"Set '{type_name}' to evaluation mode")

            # Evaluate parameters to ensure they're loaded
            mx.eval(model_instance.parameters())

            # Memory cleanup
            _aggressive_memory_cleanup()

            # Update statistics
            self.models_loaded += 1

            return model_instance, tokenizer_instance

        except ModelLoadError:
            self.load_errors += 1
            raise

        except Exception as e:
            self.load_errors += 1

            # Provide helpful error message
            error_msg = f"Failed to load '{type_name}' model from {model_path}: {e}"
            suggestions = []

            error_str = str(e).lower()
            if "no such file" in error_str or "not found" in error_str:
                suggestions.append("Verify the model path is correct")
                suggestions.append("Check if the model needs to be downloaded")
            elif "permission denied" in error_str:
                suggestions.append("Check read permissions on model directory")
            elif "out of memory" in error_str or "memory" in error_str:
                suggestions.append("Model may be too large for available memory")
                suggestions.append("Try a smaller model or quantized version")
            elif "corrupt" in error_str or "invalid" in error_str:
                suggestions.append("Model files may be corrupted")
                suggestions.append("Try re-downloading the model")

            if suggestions:
                error_msg += "\n\nSuggestions:\n" + "\n".join(
                    f"  - {s}" for s in suggestions
                )

            raise ModelLoadError(error_msg) from e

    def _apply_lora_to_model(
        self, model: nn.Module, type_name: str, lora_config: Dict[str, Any]
    ) -> nn.Module:
        """
        Apply LoRA adapters to the model.

        ENHANCED:
        - Better error handling
        - Memory cleanup
        """
        rprint(f"Applying LoRA adapters to '{type_name}' model...")

        lora_params = {
            "r": lora_config.get("lora_rank", 8),
            "lora_alpha": lora_config.get("lora_alpha", 16.0),
            "lora_dropout": lora_config.get("lora_dropout", 0.0),
            "scale_by_rank": lora_config.get("lora_scale_by_rank", True),
            "target_modules": lora_config.get("lora_target_modules", None),
        }

        try:
            # Apply LoRA
            linear_to_lora_layers(model=model, num_layers=-1, **lora_params)

            # Verify LoRA was applied
            lora_count = sum(
                1 for _, m in model.named_modules() if isinstance(m, MLXLoRALinear)
            )

            if lora_count == 0:
                logger.warning("LoRA applied but no LoRA layers found!")

            rprint(
                f"✓ Applied LoRA to '{type_name}' "
                f"(rank={lora_params['r']}, alpha={lora_params['lora_alpha']}, "
                f"layers={lora_count})"
            )

            # Memory cleanup
            _aggressive_memory_cleanup()

        except Exception as e:
            error_msg = f"Failed to apply LoRA to '{type_name}': {e}"
            logger.error(error_msg, exc_info=True)
            raise ModelLoadError(error_msg) from e

        return model

    def get_logprobs_for_sequence(
        self, model: nn.Module, prompts: mx.array, responses: mx.array
    ) -> mx.array:
        """
        Calculate log probabilities of response sequences.

        ENHANCED:
        - Better error handling
        - Memory cleanup
        """
        try:
            if responses.shape[1] == 0:
                return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

            # Forward pass
            full_sequence = mx.concatenate([prompts, responses], axis=1)
            logits_output = model(full_sequence, cache=None)
            logits = (
                logits_output[0] if isinstance(logits_output, tuple) else logits_output
            ).astype(mx.float32)

            # Extract response logits
            logits_for_responses = logits[:, prompts.shape[1] - 1 : -1, :]
            target_response_tokens = responses

            # Align shapes
            if logits_for_responses.shape[1] != target_response_tokens.shape[1]:
                min_len = min(
                    logits_for_responses.shape[1], target_response_tokens.shape[1]
                )
                logits_for_responses = logits_for_responses[:, :min_len, :]
                target_response_tokens = target_response_tokens[:, :min_len]

                if not min_len:
                    return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

            # Compute log probs
            log_probs_all = nn.log_softmax(logits_for_responses, axis=-1)
            response_log_probs = mx.take_along_axis(
                log_probs_all, target_response_tokens[..., None], axis=-1
            ).squeeze(-1)

            # Cleanup
            del logits_output, logits, logits_for_responses, log_probs_all

            return response_log_probs.astype(mx.float32)

        except Exception as e:
            logger.error(f"Error computing log probabilities: {e}", exc_info=True)
            # Return zeros rather than crashing
            return mx.zeros((prompts.shape[0], responses.shape[1]), dtype=mx.float32)

    def generate_with_logprobs(
        self,
        model: nn.Module,
        prompts: mx.array,
        tokenizer: Any,
        temp: float,
        max_tokens: int,
        cache: Optional[Any],
        logit_processors: Optional[List[Callable]],
        generation_cfg: Optional[Any],
    ) -> Tuple[mx.array, mx.array]:
        """
        Generate token sequences with log probabilities.

        ENHANCED:
        - Better error handling
        - Memory cleanup
        - Statistics tracking
        """
        try:
            batch_size = prompts.shape[0]

            # Create cache if needed
            if cache is None:
                cache = self.make_prompt_cache(
                    model, max_kv_size=prompts.shape[1] + max_tokens
                )

            # Initial forward pass
            logits_output = model(prompts.astype(mx.int64), cache=cache)
            logits = (
                logits_output[0] if isinstance(logits_output, tuple) else logits_output
            )[:, -1, :].astype(mx.float32)

            generated_tokens = []
            log_probs_list = []
            ended = mx.zeros(batch_size, dtype=mx.bool_)
            current_history = prompts.tolist()

            # Generation loop
            for step_idx in range(max_tokens):
                # Process logits
                processed_logits = logits
                if logit_processors:
                    for proc_fn in logit_processors:
                        processed_logits = proc_fn(current_history, processed_logits)

                # Sample
                from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

                sampler = safe_make_sampler(self.config, temp=temp)
                next_token = sampler(processed_logits)

                # Compute log probs
                log_probs = nn.log_softmax(processed_logits, axis=-1)
                next_log_prob = mx.take_along_axis(
                    log_probs, next_token[:, None], axis=-1
                ).squeeze(-1)

                # Check for EOS
                ended_prev = ended
                if tokenizer.eos_token_id is not None:
                    ended = mx.logical_or(
                        ended_prev, next_token == tokenizer.eos_token_id
                    )

                # Apply padding to ended sequences
                tokens_to_add = mx.where(ended_prev, tokenizer.pad_token_id, next_token)
                lp_to_add = mx.where(ended_prev, 0.0, next_log_prob)

                generated_tokens.append(tokens_to_add)
                log_probs_list.append(lp_to_add)

                # Update history
                for i in range(batch_size):
                    if not bool(ended_prev[i].item()):
                        current_history[i].append(int(tokens_to_add[i].item()))

                # Check if all sequences ended
                if mx.all(ended).item():
                    break

                # Continue generation
                logits_output = model(
                    tokens_to_add[:, None].astype(mx.int64), cache=cache
                )
                logits = (
                    logits_output[0]
                    if isinstance(logits_output, tuple)
                    else logits_output
                )[:, -1, :].astype(mx.float32)

                # Periodic cleanup
                if step_idx % 50 == 0:
                    mx.eval(generated_tokens[-1], log_probs_list[-1])

            # Stack results
            responses_mx = (
                mx.stack(generated_tokens, axis=1)
                if generated_tokens
                else mx.zeros((batch_size, 0), dtype=mx.int32)
            )
            actor_lp_resp = (
                mx.stack(log_probs_list, axis=1)
                if log_probs_list
                else mx.zeros((batch_size, 0), dtype=mx.float32)
            )

            # Update statistics
            self.generation_count += 1

            # Cleanup
            del generated_tokens, log_probs_list, ended, current_history
            _aggressive_memory_cleanup()

            return responses_mx, actor_lp_resp

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            # Return empty arrays rather than crashing
            batch_size = prompts.shape[0]
            return (
                mx.zeros((batch_size, 0), dtype=mx.int32),
                mx.zeros((batch_size, 0), dtype=mx.float32),
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get model manager statistics.

        NEW: Useful for monitoring
        """
        return {
            "models_loaded": self.models_loaded,
            "load_errors": self.load_errors,
            "generation_count": self.generation_count,
        }
