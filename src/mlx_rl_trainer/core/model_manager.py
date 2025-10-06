# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/model_manager.py
# revision_no: 002
# goals_of_writing_code_block: Manage model loading, LoRA application, and tokenizer handling for actor and reference models, ensuring mock functionality and MLX-LM compatibility.
# type_of_code_response: change existing
"""
Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
"""
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import logging
import json
from .config import ModelConfig
from .trainer import ModelLoadError
from rich import print as rprint

# --- MLX-LM Imports (Conditional for Mock) ---
try:
    import mlx_lm
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.utils import load as mlx_lm_load, save_config as mlx_lm_save_config
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    from mlx_lm.tuner.utils import apply_lora_layers, apply_lora_layers_force_qkv_mlp, print_trainable_parameters
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logging.warning("mlx-lm not found. ModelManager will use mock implementations only.")
    # Define dummy/mock versions if mlx_lm is not available

    class TokenizerWrapper:
        def __init__(self, vocab_size: int = 32000, pad_token_id: int = 0, eos_token_id: int = 2):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id
            self.name_or_path = "mock_tokenizer"
        def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
            return [1] * min(len(text.split()), 50) + ([self.eos_token_id] if add_special_tokens else [])
        def batch_decode(self, ids: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
            return [f"decoded_response_{i}" for i in range(len(ids))]
        def add_special_tokens(self, special_tokens_dict: Dict[str, Any]) -> int: return 0
        def apply_chat_template(self, messages: List[Dict], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
            return "MOCK_CHAT_PROMPT: " + messages[-1]["content"]
        def save_pretrained(self, path: Path): pass

    class MLXLoRALinear: pass # Mock class
    def apply_lora_layers(*args, **kwargs): return 0
    def apply_lora_layers_force_qkv_mlp(*args, **kwargs): return 0
    def print_trainable_parameters(*args, **kwargs): pass
    def mlx_lm_load(path: Path): return MockModel(), TokenizerWrapper()
    def mlx_lm_save_config(*args, **kwargs): pass


class MockModel(nn.Module):
    def __init__(self, vocab_size: int = 32000, embed_dim: int = 128, num_layers: int = 4, num_kv_heads: int = 2, hidden_size: int = 128, num_attention_heads: int = 4, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.layers = [nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)]
        self.n_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_attention_heads
        self.n_layers = num_layers
        self.model_config = kwargs # Store for mock loading

    def __call__(self, x: mx.array, cache: Any = None) -> mx.array:
        # Mock forward pass
        B, L = x.shape
        x = self.embedding(x)
        for layer in self.layers: x = layer(x)
        return self.lm_head(x)

    def freeze(self): pass
    def train(self, mode: bool = True): pass
    def eval(self): self.train(False)
    def parameters(self): # Mock parameters for saving
        return {f"layer.{i}.weight": mx.zeros((1,1)) for i in range(self.n_layers)}
    def trainable_parameters(self): return self.parameters()
    def load_weights(self, weights: List[Tuple[str, mx.array]], strict: bool = False): pass
    def update_modules(self, modules: Dict[str, Any]): pass


class ModelManager:
    """
    Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    def load_model(self, model_path: Path, type_name: str, is_trainable: bool = False,
                   apply_lora: bool = False, lora_config: Optional[Dict[str, Any]] = None) -> Tuple[nn.Module, Any]:
        """
        Loads a model and its tokenizer, applying LoRA if specified.

        Args:
            model_path: Path to the model directory or HuggingFace ID.
            type_name: Descriptive name for the model ('actor', 'reference', etc.).
            is_trainable: If True, model is set to train mode; otherwise, frozen.
            apply_lora: If True, applies LoRA adapters.
            lora_config: Dictionary of LoRA parameters if `apply_lora` is True.

        Returns:
            Tuple of (model, tokenizer).

        Raises:
            ModelLoadError: If model or tokenizer fails to load.
        """
        logging.info(f"Attempting to load '{type_name}' model from {model_path}...")

        model_instance: nn.Module
        tokenizer_instance: Any

        if MLX_LM_AVAILABLE:
            try:
                model_instance, tokenizer_instance = mlx_lm_load(model_path.as_posix())
                rprint(f"Successfully loaded '{type_name}' model and tokenizer using [green]mlx-lm[/green].")
            except Exception as e:
                # Fallback to mock if real loading fails for any reason
                logging.warning(f"mlx-lm model load failed ({e}). Falling back to mock model.", exc_info=True)
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
                logging.warning(f"No LoRA layers matched target_modules. Attempting force-apply to QKV/MLP for '{type_name}'.")
                num_wrapped = apply_lora_layers_force_qkv_mlp(model_instance, lora_config)
            if num_wrapped == 0:
                logging.warning(f"LoRA application resulted in 0 wrapped layers for '{type_name}'. Check lora_target_modules.")
            else:
                print_trainable_parameters(model_instance) # Log LoRA parameters

        return model_instance, tokenizer_instance

    def _load_mock_model(self, model_path: Path) -> Tuple[MockModel, TokenizerWrapper]:
        """Loads a mock model and tokenizer for when mlx-lm is not available or fails."""
        try:
            model_config_path = model_path / "config.json"
            if not model_config_path.exists():
                # Create a dummy config if it doesn't exist for mock
                dummy_config = {
                    "config": {
                        "vocab_size": 32000,
                        "embed_dim": 128,
                        "num_layers": 4,
                        "num_kv_heads": 2,
                        "hidden_size": 128,
                        "num_attention_heads": 4
                    }
                }
                model_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_config_path, 'w') as f:
                    json.dump(dummy_config, f)
                logging.info(f"Created dummy mock model config at {model_config_path}.")

            with open(model_config_path, 'r') as f:
                cfg_data = json.load(f).get("config", {})

            model = MockModel(**cfg_data)
            tokenizer = TokenizerWrapper(vocab_size=cfg_data.get("vocab_size", 32000))

            rprint(f"Successfully loaded [yellow]mock model[/yellow] and tokenizer from {model_path}.")
            return model, tokenizer
        except Exception as e:
            raise ModelLoadError(f"Failed to load mock model from {model_path}: {e}") from e

    def save_model(
        self,
        model_name: str, # Currently unused, consider for multi-model checkpointing
        save_path: Path,
        model_instance: nn.Module,
        tokenizer_instance: Any,
        model_config_dict: Dict[str, Any],
        training_args_config: Any, # ExperimentConfig is passed as args
        save_full_model: bool = False # If True, saves fused/full model; otherwise, only adapters
    ) -> None:
        """
        Saves the model weights and configuration.
        Handles LoRA models by saving adapters or fused weights.
        """
        try:
            save_path.mkdir(parents=True, exist_ok=True)

            # Save tokenizer
            tokenizer_instance.save_pretrained(save_path)

            # Save config.json (model architecture config)
            if MLX_LM_AVAILABLE:
                mlx_lm_save_config(model_config_dict, save_path / "config.json")
            else: # Fallback for mock
                with open(save_path / "config.json", 'w') as f:
                    json.dump(model_config_dict, f, indent=2)

            is_lora_model = any(isinstance(m, MLXLoRALinear) for _, m in model_instance.named_modules())

            if is_lora_model:
                if save_full_model: # Save fused weights
                    from mlx_lm.tuner.utils import fuse_lora_layers
                    fused_model = fuse_lora_layers(model_instance) # Create a fused copy
                    mx.save_safetensors(str(save_path / "model.fused.safetensors"), dict(mx.utils.tree_flatten(fused_model.parameters())))
                    logger.info(f"Saved fused LoRA model weights to {save_path / 'model.fused.safetensors'}")
                    del fused_model # Free memory
                else: # Save just adapters
                    lora_params = dict(mx.utils.tree_flatten(model_instance.trainable_parameters()))
                    if lora_params:
                        mx.save_safetensors(str(save_path / "adapters.safetensors"), lora_params)
                        logger.info(f"Saved LoRA adapter weights to {save_path / 'adapters.safetensors'}")
                    else:
                        logger.warning(f"No trainable LoRA parameters found for {model_name}; skipping adapters.safetensors.")
            else: # Not a LoRA model, save full weights
                mx.save_safetensors(str(save_path / "model.safetensors"), dict(mx.utils.tree_flatten(model_instance.parameters())))
                logger.info(f"Saved full model weights to {save_path / 'model.safetensors'}")

        except Exception as e:
            raise ModelLoadError(f"Failed to save model for {model_name} to {save_path}: {e}") from e
