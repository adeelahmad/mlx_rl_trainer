# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/model_manager.py
# revision_no: 003
# goals_of_writing_code_block: Add missing generation and log-probability methods to the ModelManager to make it fully compatible with the GRPOTrainer.
# type_of_code_response: replace code
"""
Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from rich import print as rprint

from .config import ModelConfig
from .trainer import ModelLoadError

# --- MLX-LM Imports (Conditional for Mock) ---
try:
    print("Attempting to import mlx_lm")
    import mlx_lm
    print("mlx_lm imported")
    print("Attempting to import cache")
    from mlx_lm.models import cache
    print("cache imported")
    print("Attempting to import TokenizerWrapper")
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    print("TokenizerWrapper imported")
    print("Attempting to import LoRALinear")
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    print("LoRALinear imported")
    print("Attempting to import tuner utils")
    from mlx_lm.tuner.utils import (
        apply_lora_layers,
        apply_lora_layers_force_qkv_mlp,
        print_trainable_parameters,
    )
    print("tuner utils imported")
    print("Attempting to import mlx_lm_load")
    from mlx_lm.utils import load as mlx_lm_load
    print("mlx_lm_load imported")
    print("Attempting to import mlx_lm_save_config")
    from mlx_lm.utils import save_config as mlx_lm_save_config
    print("mlx_lm_save_config imported")

    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logging.warning(
        "mlx-lm not found. ModelManager will use mock implementations only."
    )
    # Define dummy/mock versions if mlx_lm is not available

    class TokenizerWrapper:
        def __init__(
            self, vocab_size: int = 32000, pad_token_id: int = 0, eos_token_id: int = 2
        ):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id
            self.name_or_path = "mock_tokenizer"

        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            return [1] * min(len(text.split()), 50) + (
                [self.eos_token_id] if add_special_tokens else []
            )

        def batch_decode(
            self, ids: List[List[int]], skip_special_tokens: bool = False
        ) -> List[str]:
            return [f"decoded_response_{i}" for i in range(len(ids))]

        def add_special_tokens(self, special_tokens_dict: Dict[str, Any]) -> int:
            return 0

        def apply_chat_template(
            self,
            messages: List[Dict],
            tokenize: bool = False,
            add_generation_prompt: bool = False,
        ) -> str:
            return "MOCK_CHAT_PROMPT: " + messages[-1]["content"]

        def save_pretrained(self, path: Path):
            pass

    class MLXLoRALinear:
        pass  # Mock class

    class cache:  # Mock cache
        @staticmethod
        def make_prompt_cache(*args, **kwargs):
            return None

    def apply_lora_layers(*args, **kwargs):
        return 0

    def apply_lora_layers_force_qkv_mlp(*args, **kwargs):
        return 0

    def print_trainable_parameters(*args, **kwargs):
        pass

    def mlx_lm_load(path: Path):
        return MockModel(), TokenizerWrapper()

    def mlx_lm_save_config(*args, **kwargs):
        pass


class MockModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_kv_heads: int = 2,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.layers = [nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)]
        self.n_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.n_layers = num_layers
        self.model_config = kwargs  # Store for mock loading

    def __call__(self, x: mx.array, cache: Any = None) -> mx.array:
        # Mock forward pass
        B, L = x.shape
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

    def freeze(self):
        pass

    def train(self, mode: bool = True):
        pass

    def eval(self):
        self.train(False)

    def parameters(self):  # Mock parameters for saving
        return {f"layer.{i}.weight": mx.zeros((1, 1)) for i in range(self.n_layers)}

    def trainable_parameters(self):
        return self.parameters()

    def load_weights(self, weights: List[Tuple[str, mx.array]], strict: bool = False):
        pass

    def update_modules(self, modules: Dict[str, Any]):
        pass


class ModelManager:
    """
    Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Any]:
        """
        Loads a model and its tokenizer, applying LoRA if specified.
        """
        logging.info(f"Attempting to load '{type_name}' model from {model_path}...")
        model_instance: nn.Module
        tokenizer_instance: Any

        if MLX_LM_AVAILABLE:
            try:
                model_instance, tokenizer_instance = mlx_lm_load(model_path.as_posix())
                rprint(
                    f"Successfully loaded '{type_name}' model and tokenizer using [green]mlx-lm[/green]."
                )
            except Exception as e:
                logging.warning(
                    f"mlx-lm model load failed ({e}). Falling back to mock model.",
                    exc_info=True,
                )
                model_instance, tokenizer_instance = self._load_mock_model(model_path)
        else:
            model_instance, tokenizer_instance = self._load_mock_model(model_path)

        if is_trainable:
            model_instance.train()
        else:
            model_instance.freeze()

        if apply_lora and lora_config:
            rprint(f"Applying LoRA adapters to '{type_name}' model...")
            num_wrapped = apply_lora_layers(model_instance, lora_config)
            if num_wrapped == 0:
                logging.warning(
                    f"No LoRA layers matched target_modules. Attempting force-apply to QKV/MLP for '{type_name}'."
                )
                num_wrapped = apply_lora_layers_force_qkv_mlp(
                    model_instance, lora_config
                )
            if num_wrapped == 0:
                logging.warning(
                    f"LoRA application resulted in 0 wrapped layers for '{type_name}'. Check lora_target_modules."
                )
            else:
                print_trainable_parameters(model_instance)

        return model_instance, tokenizer_instance

    def _load_mock_model(self, model_path: Path) -> Tuple[MockModel, TokenizerWrapper]:
        """Loads a mock model and tokenizer for when mlx-lm is not available or fails."""
        try:
            model_config_path = model_path / "config.json"
            if not model_config_path.exists():
                dummy_config = {
                    "config": {
                        "vocab_size": 32000,
                        "embed_dim": 128,
                        "num_layers": 4,
                        "num_kv_heads": 2,
                        "hidden_size": 128,
                        "num_attention_heads": 4,
                    }
                }
                model_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_config_path, "w") as f:
                    json.dump(dummy_config, f)
                logging.info(f"Created dummy mock model config at {model_config_path}.")

            with open(model_config_path, "r") as f:
                cfg_data = json.load(f).get("config", {})

            model = MockModel(**cfg_data)
            tokenizer = TokenizerWrapper(vocab_size=cfg_data.get("vocab_size", 32000))
            rprint(
                f"Successfully loaded [yellow]mock model[/yellow] and tokenizer from {model_path}."
            )
            return model, tokenizer
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load mock model from {model_path}: {e}"
            ) from e

    def save_model(
        self,
        model_name: str,
        save_path: Path,
        model_instance: nn.Module,
        tokenizer_instance: Any,
        model_config_dict: Dict[str, Any],
        training_args_config: Any,
        save_full_model: bool = False,
    ) -> None:
        """Saves the model weights and configuration."""
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            tokenizer_instance.save_pretrained(save_path)

            if MLX_LM_AVAILABLE:
                mlx_lm_save_config(model_config_dict, save_path / "config.json")
            else:
                with open(save_path / "config.json", "w") as f:
                    json.dump(model_config_dict, f, indent=2)

            is_lora_model = any(
                isinstance(m, MLXLoRALinear) for _, m in model_instance.named_modules()
            )

            if is_lora_model:
                if save_full_model:
                    from mlx_lm.tuner.utils import fuse_lora_layers

                    fused_model = fuse_lora_layers(model_instance)
                    mx.save_safetensors(
                        str(save_path / "model.fused.safetensors"),
                        dict(mx.utils.tree_flatten(fused_model.parameters())),
                    )
                    logger.info(
                        f"Saved fused LoRA model weights to {save_path / 'model.fused.safetensors'}"
                    )
                    del fused_model
                else:
                    lora_params = dict(
                        mx.utils.tree_flatten(model_instance.trainable_parameters())
                    )
                    if lora_params:
                        mx.save_safetensors(
                            str(save_path / "adapters.safetensors"), lora_params
                        )
                        logger.info(
                            f"Saved LoRA adapter weights to {save_path / 'adapters.safetensors'}"
                        )
                    else:
                        logger.warning(
                            f"No trainable LoRA parameters found for {model_name}; skipping adapters.safetensors."
                        )
            else:
                mx.save_safetensors(
                    str(save_path / "model.safetensors"),
                    dict(mx.utils.tree_flatten(model_instance.parameters())),
                )
                logger.info(
                    f"Saved full model weights to {save_path / 'model.safetensors'}"
                )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to save model for {model_name} to {save_path}: {e}"
            ) from e

class ModelManager:
    """
    Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Any]:
        """
        Loads a model and its tokenizer, applying LoRA if specified.
        """
        logging.info(f"Attempting to load '{type_name}' model from {model_path}...")
        model_instance: nn.Module
        tokenizer_instance: Any

        if MLX_LM_AVAILABLE:
            try:
                model_instance, tokenizer_instance = mlx_lm_load(model_path.as_posix())
                rprint(
                    f"Successfully loaded '{type_name}' model and tokenizer using [green]mlx-lm[/green]."
                )
            except Exception as e:
                logging.warning(
                    f"mlx-lm model load failed ({e}). Falling back to mock model.",
                    exc_info=True,
                )
                model_instance, tokenizer_instance = self._load_mock_model(model_path)
        else:
            model_instance, tokenizer_instance = self._load_mock_model(model_path)

        if is_trainable:
            model_instance.train()
        else:
            model_instance.freeze()

        if apply_lora and lora_config:
            rprint(f"Applying LoRA adapters to '{type_name}' model...")
            num_wrapped = apply_lora_layers(model_instance, lora_config)
            if num_wrapped == 0:
                logging.warning(
                    f"No LoRA layers matched target_modules. Attempting force-apply to QKV/MLP for '{type_name}'."
                )
                num_wrapped = apply_lora_layers_force_qkv_mlp(
                    model_instance, lora_config
                )
            if num_wrapped == 0:
                logging.warning(
                    f"LoRA application resulted in 0 wrapped layers for '{type_name}'. Check lora_target_modules."
                )
            else:
                print_trainable_parameters(model_instance)

        return model_instance, tokenizer_instance

    def _load_mock_model(self, model_path: Path) -> Tuple[MockModel, TokenizerWrapper]:
        """
        Loads a mock model and tokenizer for when mlx-lm is not available or fails.
        """
        try:
            model_config_path = model_path / "config.json"
            if not model_config_path.exists():
                dummy_config = {
                    "config": {
                        "vocab_size": 32000,
                        "embed_dim": 128,
                        "num_layers": 4,
                        "num_kv_heads": 2,
                        "hidden_size": 128,
                        "num_attention_heads": 4,
                    }
                }
                model_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_config_path, "w") as f:
                    json.dump(dummy_config, f)
                logging.info(f"Created dummy mock model config at {model_config_path}.")

            with open(model_config_path, "r") as f:
                cfg_data = json.load(f).get("config", {})

            model = MockModel(**cfg_data)
            tokenizer = TokenizerWrapper(vocab_size=cfg_data.get("vocab_size", 32000))
            rprint(
                f"Successfully loaded [yellow]mock model[/yellow] and tokenizer from {model_path}."
            )
            return model, tokenizer
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load mock model from {model_path}: {e}"
            ) from e

    def save_model(
        self,
        model_name: str,
        save_path: Path,
        model_instance: nn.Module,
        tokenizer_instance: Any,
        model_config_dict: Dict[str, Any],
        training_args_config: Any,
        save_full_model: bool = False,
    ) -> None:
        """
        Saves the model weights and configuration.
        """
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            tokenizer_instance.save_pretrained(save_path)

            if MLX_LM_AVAILABLE:
                mlx_lm_save_config(model_config_dict, save_path / "config.json")
            else:
                with open(save_path / "config.json", "w") as f:
                    json.dump(model_config_dict, f, indent=2)

            is_lora_model = any(
                isinstance(m, MLXLoRALinear) for _, m in model_instance.named_modules()
            )

            if is_lora_model:
                if save_full_model:
                    from mlx_lm.tuner.utils import fuse_lora_layers

                    fused_model = fuse_lora_layers(model_instance)
                    mx.save_safetensors(
                        str(save_path / "model.fused.safetensors"),
                        dict(mx.utils.tree_flatten(fused_model.parameters())),
                    )
                    logger.info(
                        f"Saved fused LoRA model weights to {save_path / 'model.fused.safetensors'}"
                    )
                    del fused_model
                else:
                    lora_params = dict(
                        mx.utils.tree_flatten(model_instance.trainable_parameters())
                    )
                    if lora_params:
                        mx.save_safetensors(
                            str(save_path / "adapters.safetensors"), lora_params
                        )
                        logger.info(
                            f"Saved LoRA adapter weights to {save_path / 'adapters.safetensors'}"
                        )
                    else:
                        logger.warning(
                            f"No trainable LoRA parameters found for {model_name}; skipping adapters.safetensors."
                        )
            else:
                mx.save_safetensors(
                    str(save_path / "model.safetensors"),
                    dict(mx.utils.tree_flatten(model_instance.parameters())),
                )
                logger.info(
                    f"Saved full model weights to {save_path / 'model.safetensors'}"
                )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to save model for {model_name} to {save_path}: {e}"
            ) from e

    def generate_with_logprobs(
        self,
        model: nn.Module,
        prompts: mx.array,
        tokenizer: Any,
        temp: float = 0.7,
        max_tokens: int = 128,
    ) -> Tuple[mx.array, mx.array]:
        """
        Generates token sequences from prompts and returns the generated tokens
        along with their corresponding log probabilities.
        """
        if MLX_LM_AVAILABLE:
            model_cache = cache.make_prompt_cache(model)
            y = prompts
            logits = model(y, cache=model_cache)[:, -1, :]

            generated_tokens = []
            log_probs = []
            batch_size = prompts.shape[0]
            ended = mx.zeros(batch_size, dtype=mx.bool_)

            for _ in range(max_tokens):
                if temp == 0:
                    next_token = mx.argmax(logits, axis=-1)
                else:
                    next_token = mx.random.categorical(logits * (1 / temp))

                current_log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
                next_log_prob = mx.take_along_axis(
                    current_log_probs, next_token[:, None], axis=-1
                ).squeeze(-1)

                next_token = mx.where(ended, tokenizer.pad_token_id, next_token)
                next_log_prob = mx.where(ended, 0.0, next_log_prob)

                generated_tokens.append(next_token)
                log_probs.append(next_log_prob)

                if tokenizer.eos_token_id is not None:
                    ended = mx.logical_or(ended, next_token == tokenizer.eos_token_id)

                if mx.all(ended).item():
                    break

                logits = model(next_token[:, None], cache=model_cache)[:, -1, :]

            responses_mx = (
                mx.stack(generated_tokens, axis=1)
                if generated_tokens
                else mx.zeros((batch_size, 0), dtype=mx.int32)
            )
            actor_lp_resp = (
                mx.stack(log_probs, axis=1)
                if log_probs
                else mx.zeros((batch_size, 0), dtype=mx.float32)
            )

            return responses_mx, actor_lp_resp
        else:
            logging.warning("mlx-lm not available. Using mock generation.")
            batch_size = prompts.shape[0]
            mock_responses = mx.full(
                (batch_size, max_tokens // 2), tokenizer.eos_token_id, dtype=mx.int32
            )
            mock_log_probs = mx.zeros(
                (batch_size, max_tokens // 2), dtype=mx.float32
            )
            return mock_responses, mock_log_probs

    def get_logprobs_for_sequence(
        self,
        model: nn.Module,
        prompts: mx.array,
        responses: mx.array
    ) -> mx.array:
        """
        Calculates the log probabilities of a given response sequence conditioned on a prompt.
        """
        if MLX_LM_AVAILABLE:
            if responses.shape[1] == 0:
                return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

            full_sequence = mx.concatenate([prompts, responses], axis=1)
            logits = model(full_sequence)

            # We want the logits that *predicted* each token in the response.
            # This means we take logits from the end of the prompt up to the second-to-last token of the full sequence.
            logits_for_responses = logits[:, prompts.shape[1] - 1 : -1, :]

            log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)

            response_log_probs = mx.take_along_axis(
                log_probs, responses[..., None], axis=-1
            ).squeeze(-1)

            return response_log_probs
        else:
            logging.warning("mlx-lm not available. Using mock logprob calculation.")
            return mx.zeros(responses.shape, dtype=mx.float32)
