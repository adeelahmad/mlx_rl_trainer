"""
Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
from rich import print as rprint

from .config import ModelConfig
from .exceptions import ModelLoadError  # Import from new exceptions module

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
        pass  # Dummy class for type hints

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


class ModelManager:
    """Manages the lifecycle of MLX models, including LoRA application."""

    def __init__(self, config: ModelConfig):
        self.config = config
        if not MLX_LM_AVAILABLE:
            raise RuntimeError(
                "mlx-lm is required but not available. Install with `pip install mlx-lm`."
            )

    def make_prompt_cache(
        self, model: nn.Module, max_kv_size: Optional[int] = None
    ) -> Any:
        """Creates a KV cache for the model."""
        return mlx_lm_cache.make_prompt_cache(model, max_kv_size=max_kv_size)

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Any]:
        """Loads a model and tokenizer, applying LoRA if specified."""
        logging.info(f"Loading '{type_name}' model from {model_path} using mlx-lm...")
        try:
            model_instance, tokenizer_instance = load(str(model_path))
            rprint(f"✓ Loaded '{type_name}' model from [green]{model_path}[/green]")

            if apply_lora and lora_config:
                model_instance = self._apply_lora_to_model(
                    model_instance, type_name, lora_config
                )

            if is_trainable:
                model_instance.train()
                logging.info(f"Set '{type_name}' model to training mode")
                print_trainable_parameters(model_instance)
            else:
                model_instance.eval()
                logging.info(f"Set '{type_name}' model to evaluation mode")
            return model_instance, tokenizer_instance
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load '{type_name}' model from {model_path}: {e}"
            ) from e

    def _apply_lora_to_model(
        self, model: nn.Module, type_name: str, lora_config: Dict[str, Any]
    ) -> nn.Module:
        """Applies LoRA adapters to the model."""
        rprint(f"Applying LoRA adapters to '{type_name}' model...")

        lora_params = {
            "r": lora_config.get("lora_rank", 8),
            "lora_alpha": lora_config.get("lora_alpha", 16.0),
            "lora_dropout": lora_config.get("lora_dropout", 0.0),
            "scale_by_rank": lora_config.get("lora_scale_by_rank", True),
            "target_modules": lora_config.get("lora_target_modules", None),
        }

        try:
            linear_to_lora_layers(model=model, num_layers=-1, **lora_params)
            rprint(
                f"✓ Applied LoRA to '{type_name}' model (rank={lora_params['r']}, alpha={lora_params['lora_alpha']})"
            )
        except Exception as e:
            logging.error(f"Failed to apply LoRA to '{type_name}': {e}", exc_info=True)
            raise
        return model

    def get_logprobs_for_sequence(
        self, model: nn.Module, prompts: mx.array, responses: mx.array
    ) -> mx.array:
        """Calculates the log probabilities of a given response sequence conditioned on a prompt."""
        if responses.shape[1] == 0:
            return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

        full_sequence = mx.concatenate([prompts, responses], axis=1)
        logits_output = model(full_sequence, cache=None)
        logits = (
            logits_output[0] if isinstance(logits_output, tuple) else logits_output
        ).astype(mx.float32)
        logits_for_responses = logits[:, prompts.shape[1] - 1 : -1, :]
        target_response_tokens = responses

        if logits_for_responses.shape[1] != target_response_tokens.shape[1]:
            min_len = min(
                logits_for_responses.shape[1], target_response_tokens.shape[1]
            )
            logits_for_responses = logits_for_responses[:, :min_len, :]
            target_response_tokens = target_response_tokens[:, :min_len]
            if not min_len:
                return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

        log_probs_all = nn.log_softmax(logits_for_responses, axis=-1)
        response_log_probs = mx.take_along_axis(
            log_probs_all, target_response_tokens[..., None], axis=-1
        ).squeeze(-1)

        return response_log_probs.astype(mx.float32)

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
        """Generates token sequences from prompts and returns tokens and their log probabilities."""
        batch_size = prompts.shape[0]
        if cache is None:
            cache = self.make_prompt_cache(
                model, max_kv_size=prompts.shape[1] + max_tokens
            )

        logits_output = model(prompts.astype(mx.int64), cache=cache)
        logits = (
            logits_output[0] if isinstance(logits_output, tuple) else logits_output
        )[:, -1, :].astype(mx.float32)

        generated_tokens = []
        log_probs_list = []
        ended = mx.zeros(batch_size, dtype=mx.bool_)
        current_history = prompts.tolist()

        for _ in range(max_tokens):
            processed_logits = logits
            if logit_processors:
                for proc_fn in logit_processors:
                    processed_logits = proc_fn(current_history, processed_logits)

            from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler

            sampler = safe_make_sampler(
                self.config, temp=temp
            )  # Use self.config as it's an ExperimentConfig

            next_token = sampler(processed_logits)
            log_probs = nn.log_softmax(processed_logits, axis=-1)
            next_log_prob = mx.take_along_axis(
                log_probs, next_token[:, None], axis=-1
            ).squeeze(-1)

            ended_prev = ended
            if tokenizer.eos_token_id is not None:
                ended = mx.logical_or(ended_prev, next_token == tokenizer.eos_token_id)

            tokens_to_add = mx.where(ended_prev, tokenizer.pad_token_id, next_token)
            lp_to_add = mx.where(ended_prev, 0.0, next_log_prob)

            generated_tokens.append(tokens_to_add)
            log_probs_list.append(lp_to_add)

            for i in range(batch_size):
                if not bool(ended_prev[i].item()):
                    current_history[i].append(int(tokens_to_add[i].item()))

            if mx.all(ended).item():
                break

            logits_output = model(tokens_to_add[:, None].astype(mx.int64), cache=cache)
            logits = (
                logits_output[0] if isinstance(logits_output, tuple) else logits_output
            )[:, -1, :].astype(mx.float32)

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
        return responses_mx, actor_lp_resp
