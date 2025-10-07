# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/model_manager.py
# revision_no: 006
# goals_of_writing_code_block: Use actual mlx_lm utility functions (linear_to_lora_layers, etc.)
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

# --- MLX-LM Imports ---
try:
    import mlx_lm
    from mlx_lm import load, generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    from mlx_lm.tuner.utils import (
        linear_to_lora_layers,
        print_trainable_parameters,
        remove_lora_layers,
        dequantize,
        load_adapters,
    )
    from mlx_lm.utils import save_config

    MLX_LM_AVAILABLE = True
    logging.info("mlx_lm successfully imported")
except ImportError as e:
    MLX_LM_AVAILABLE = False
    error_msg = f"mlx-lm not found: {e}. ModelManager requires mlx-lm to be installed."
    logging.error(error_msg)
    raise ImportError(
        "mlx-lm is required for ModelManager. Install it with: pip install mlx-lm"
    ) from e


class ModelManager:
    """
    Model management lifecycle: loading, LoRA conversion, and multi-model coordination.
    Includes generation and log-probability computation methods for GRPO training.

    This implementation requires mlx-lm to be installed and available.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        if not MLX_LM_AVAILABLE:
            raise RuntimeError("mlx-lm is required but not available")

    def load_model(
        self,
        model_path: Path,
        type_name: str,
        is_trainable: bool = False,
        apply_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[nn.Module, Any]:
        """
        Loads a model and its tokenizer using mlx-lm, applying LoRA if specified.

        Args:
            model_path: Path to the model directory
            type_name: Descriptive name for logging (e.g., "actor", "critic", "reference")
            is_trainable: Whether to set the model to training mode
            apply_lora: Whether to apply LoRA adapters
            lora_config: LoRA configuration dict with keys:
                - 'num_lora_layers': number of layers to convert (default: all)
                - 'rank': LoRA rank
                - 'scale': LoRA alpha / rank
                - 'dropout': dropout probability
                - 'keys': optional list of module keys to target
                - 'use_dora': whether to use DoRA instead of LoRA

        Returns:
            Tuple of (model, tokenizer)
        """
        logging.info(f"Loading '{type_name}' model from {model_path} using mlx-lm...")

        try:
            # Load model and tokenizer using mlx-lm
            model_instance, tokenizer_instance = load(str(model_path))
            rprint(
                f"✓ Successfully loaded '{type_name}' model and tokenizer from [green]{model_path}[/green]"
            )

            # Apply LoRA if requested
            if apply_lora and lora_config:
                model_instance = self._apply_lora_to_model(
                    model_instance, type_name, lora_config
                )

            # Set training mode after LoRA application
            if is_trainable:
                model_instance.train()
                logging.info(f"Set '{type_name}' model to training mode")
            else:
                model_instance.freeze()
                logging.info(f"Froze '{type_name}' model weights")

            return model_instance, tokenizer_instance

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load '{type_name}' model from {model_path}: {e}"
            ) from e

    def _apply_lora_to_model(
        self, model: nn.Module, type_name: str, lora_config: Dict[str, Any]
    ) -> nn.Module:
        """
        Applies LoRA adapters to the model using mlx-lm's linear_to_lora_layers.

        Args:
            model: The model to apply LoRA to
            type_name: Model name for logging
            lora_config: Dict containing LoRA parameters:
                - num_lora_layers: Number of transformer layers to convert (from end)
                - rank: LoRA rank (r)
                - scale: LoRA alpha / rank
                - dropout: Dropout probability
                - keys: Optional list of module names to target
                - use_dora: Whether to use DoRA (default: False)

        Returns:
            Model with LoRA adapters applied
        """
        rprint(f"Applying LoRA adapters to '{type_name}' model...")

        # Extract LoRA parameters with defaults
        num_lora_layers = lora_config.get('num_lora_layers', -1)  # -1 means all layers
        lora_rank = lora_config.get('rank', 8)
        lora_scale = lora_config.get('scale', 16.0)  # alpha / rank
        lora_dropout = lora_config.get('dropout', 0.0)
        lora_keys = lora_config.get('keys', None)
        use_dora = lora_config.get('use_dora', False)

        # Prepare config dict for linear_to_lora_layers
        lora_params = {
            'rank': lora_rank,
            'scale': lora_scale,
            'dropout': lora_dropout,
        }

        if lora_keys is not None:
            lora_params['keys'] = lora_keys

        # Apply LoRA using mlx-lm's utility function
        try:
            linear_to_lora_layers(
                model=model,
                num_layers=num_lora_layers,
                config=lora_params,
                use_dora=use_dora
            )

            rprint(f"✓ Applied LoRA to '{type_name}' model (rank={lora_rank}, scale={lora_scale})")
            print_trainable_parameters(model)

        except Exception as e:
            logging.error(f"Failed to apply LoRA to '{type_name}': {e}")
            raise

        return model

    def load_adapter_weights(
        self,
        model: nn.Module,
        adapter_path: Path,
        type_name: str
    ) -> nn.Module:
        """
        Load pre-trained adapter weights into a model.

        Args:
            model: Model with LoRA layers already applied
            adapter_path: Path to adapter weights directory
            type_name: Model name for logging

        Returns:
            Model with loaded adapter weights
        """
        try:
            rprint(f"Loading adapter weights for '{type_name}' from {adapter_path}")
            model = load_adapters(model, str(adapter_path))
            rprint(f"✓ Loaded adapter weights for '{type_name}'")
            return model
        except Exception as e:
            logging.error(f"Failed to load adapters for '{type_name}': {e}")
            raise

    def fuse_lora_weights(self, model: nn.Module, type_name: str) -> nn.Module:
        """
        Fuse LoRA weights back into the base model weights.

        Args:
            model: Model with LoRA layers
            type_name: Model name for logging

        Returns:
            Model with fused weights (LoRA layers removed)
        """
        rprint(f"Fusing LoRA weights for '{type_name}'...")

        # Check if model has LoRA layers
        has_lora = any(
            isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
        )

        if not has_lora:
            logging.warning(f"No LoRA layers found in '{type_name}' model")
            return model

        # Fuse by converting LoRA weights to base weights, then removing LoRA layers
        # First, we need to get the fused weights from each LoRA layer
        fused_layers = []
        for name, module in model.named_modules():
            if isinstance(module, MLXLoRALinear):
                # Get the fused weight: W + (B @ A) * scale
                base_weight = module.linear.weight
                lora_b = module.lora_b
                lora_a = module.lora_a
                scale = module.scale

                # Fuse: W_fused = W + scale * (B @ A)
                lora_weight = scale * (lora_b @ lora_a)
                fused_weight = base_weight + lora_weight

                # Create new linear layer with fused weights
                bias = hasattr(module.linear, 'bias')
                output_dims, input_dims = fused_weight.shape
                fused_linear = nn.Linear(input_dims, output_dims, bias=bias)
                fused_linear.weight = fused_weight
                if bias:
                    fused_linear.bias = module.linear.bias

                fused_layers.append((name, fused_linear))

        if fused_layers:
            from mlx.utils import tree_unflatten
            model.update_modules(tree_unflatten(fused_layers))
            rprint(f"✓ Fused {len(fused_layers)} LoRA layers in '{type_name}'")

        return model

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
        Saves the model weights, tokenizer, and configuration.

        Args:
            model_name: Name of the model for logging
            save_path: Directory to save the model
            model_instance: The model to save
            tokenizer_instance: The tokenizer to save
            model_config_dict: Model configuration dictionary
            training_args_config: Training arguments (currently unused but kept for compatibility)
            save_full_model: If True and model has LoRA, fuse and save full weights
        """
        try:
            save_path.mkdir(parents=True, exist_ok=True)

            # Save tokenizer
            tokenizer_instance.save_pretrained(str(save_path))
            logging.info(f"Saved tokenizer to {save_path}")

            # Save model config
            save_config(model_config_dict, config_path=save_path / "config.json")
            logging.info(f"Saved config to {save_path / 'config.json'}")

            # Check if model has LoRA adapters
            is_lora_model = any(
                isinstance(m, MLXLoRALinear) for _, m in model_instance.named_modules()
            )

            if is_lora_model:
                if save_full_model:
                    # Fuse LoRA weights into base model and save
                    rprint(f"Fusing LoRA layers for '{model_name}'...")
                    fused_model = self.fuse_lora_weights(model_instance, model_name)

                    weights = dict(mx.utils.tree_flatten(fused_model.parameters()))
                    mx.save_safetensors(
                        str(save_path / "model.safetensors"),
                        weights
                    )
                    logging.info(
                        f"✓ Saved fused model weights to {save_path / 'model.safetensors'}"
                    )
                else:
                    # Save only LoRA adapters
                    lora_params = dict(
                        mx.utils.tree_flatten(model_instance.trainable_parameters())
                    )

                    if lora_params:
                        mx.save_safetensors(
                            str(save_path / "adapters.safetensors"),
                            lora_params
                        )

                        # Also save adapter config for compatibility with mlx-lm
                        adapter_config = {
                            "fine_tune_type": "lora",
                            "num_layers": -1,  # Indicates all layers
                            "lora_parameters": {
                                "rank": getattr(
                                    next(
                                        m for m in model_instance.modules()
                                        if isinstance(m, MLXLoRALinear)
                                    ), 'rank', 8
                                ),
                                "scale": 1.0,  # Will be in the weights
                                "dropout": 0.0,
                            }
                        }

                        with open(save_path / "adapter_config.json", "w") as f:
                            json.dump(adapter_config, f, indent=2)

                        logging.info(
                            f"✓ Saved LoRA adapters ({len(lora_params)} params) to "
                            f"{save_path / 'adapters.safetensors'}"
                        )
                    else:
                        logging.warning(
                            f"No trainable LoRA parameters found for '{model_name}'. "
                            f"Skipping adapters.safetensors."
                        )
            else:
                # Save full model weights
                weights = dict(mx.utils.tree_flatten(model_instance.parameters()))
                mx.save_safetensors(
                    str(save_path / "model.safetensors"),
                    weights
                )
                logging.info(
                    f"✓ Saved full model weights to {save_path / 'model.safetensors'}"
                )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to save '{model_name}' to {save_path}: {e}"
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

        Args:
            model: The model to use for generation
            prompts: Input token IDs of shape (batch_size, prompt_length)
            tokenizer: Tokenizer with eos_token_id and pad_token_id
            temp: Sampling temperature (0 for greedy)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_tokens, log_probs):
                - generated_tokens: Shape (batch_size, generated_length)
                - log_probs: Shape (batch_size, generated_length)
        """
        # Validate inputs
        if prompts.size == 0:
            logging.error(f"Empty prompts array passed to generate_with_logprobs: shape={prompts.shape}")
            raise ValueError(f"Prompts array is empty with shape {prompts.shape}")

        if prompts.ndim != 2:
            logging.error(f"Invalid prompts dimensions: expected 2D, got {prompts.ndim}D with shape {prompts.shape}")
            raise ValueError(f"Prompts must be 2D, got shape {prompts.shape}")

        batch_size, prompt_length = prompts.shape
        if batch_size == 0 or prompt_length == 0:
            logging.error(f"Invalid prompts shape: batch_size={batch_size}, prompt_length={prompt_length}")
            raise ValueError(f"Invalid prompts shape: {prompts.shape}")

        logging.debug(f"generate_with_logprobs: batch_size={batch_size}, prompt_length={prompt_length}, max_tokens={max_tokens}")

        # Create KV cache for efficient generation
        model_cache = make_prompt_cache(model)

        # Initial forward pass with prompts
        y = prompts
        logits = model(y, cache=model_cache)

        # Handle both tuple output (logits, cache) and direct logits
        if isinstance(logits, tuple):
            logits = logits[0]

        # Get logits for next token (last position)
        logits = logits[:, -1, :].astype(mx.float32)

        generated_tokens = []
        log_probs = []
        batch_size = prompts.shape[0]
        ended = mx.zeros(batch_size, dtype=mx.bool_)

        for _ in range(max_tokens):
            # Sample next token
            if temp == 0:
                next_token = mx.argmax(logits, axis=-1)
            else:
                next_token = mx.random.categorical(logits * (1.0 / temp))

            # Calculate log probabilities
            current_log_probs = nn.log_softmax(logits, axis=-1)
            next_log_prob = mx.take_along_axis(
                current_log_probs, next_token[:, None], axis=-1
            ).squeeze(-1)

            # Mask out tokens for sequences that have ended
            next_token = mx.where(ended, tokenizer.pad_token_id, next_token)
            next_log_prob = mx.where(ended, 0.0, next_log_prob)

            generated_tokens.append(next_token)
            log_probs.append(next_log_prob)

            # Check for EOS tokens
            if tokenizer.eos_token_id is not None:
                ended = mx.logical_or(ended, next_token == tokenizer.eos_token_id)

            # Early stopping if all sequences have ended
            if mx.all(ended).item():
                break

            # Forward pass for next iteration
            logits = model(next_token[:, None], cache=model_cache)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits[:, -1, :].astype(mx.float32)

        # Stack results
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

    def get_logprobs_for_sequence(
        self, model: nn.Module, prompts: mx.array, responses: mx.array
    ) -> mx.array:
        """
        Calculates the log probabilities of a given response sequence conditioned on a prompt.
        This is used for computing reference log-probs in GRPO training.

        Args:
            model: The model to use for computing log probs
            prompts: Input token IDs of shape (batch_size, prompt_length)
            responses: Generated token IDs of shape (batch_size, response_length)

        Returns:
            Log probabilities of shape (batch_size, response_length)
        """
        if responses.shape[1] == 0:
            return mx.zeros((prompts.shape[0], 0), dtype=mx.float32)

        # Concatenate prompt and response
        full_sequence = mx.concatenate([prompts, responses], axis=1)

        # Forward pass through model
        out = model(full_sequence, cache=None)

        # Handle both tuple output and direct logits
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        logits = logits.astype(mx.float32)

        # Get logits that predict the response tokens
        # These are the logits from position (prompt_length - 1) to (total_length - 2)
        logits_for_responses = logits[:, prompts.shape[1] - 1 : -1, :]

        # Validate shapes
        if logits_for_responses.shape[1] != responses.shape[1]:
            logging.error(
                f"Shape mismatch in get_logprobs_for_sequence: "
                f"logits shape {logits_for_responses.shape} vs responses shape {responses.shape}. "
                f"Prompt length: {prompts.shape[1]}, Response length: {responses.shape[1]}, "
                f"Full sequence length: {full_sequence.shape[1]}, Logits length: {logits.shape[1]}. "
                f"This may indicate max_tokens is too short. Returning zeros."
            )
            return mx.zeros_like(responses, dtype=mx.float32)

        # Compute log probabilities
        log_probs_all = nn.log_softmax(logits_for_responses, axis=-1)

        # Gather log probs for actual response tokens
        response_log_probs = mx.take_along_axis(
            log_probs_all, responses[..., None], axis=-1
        ).squeeze(-1)

        return response_log_probs.astype(mx.float32)
