"""
Checkpoint management for training persistence.

This module provides a robust CheckpointManager class for MLX training loops,
handling atomic saving, state loading (including LoRA and full models),
and intelligent checkpoint rotation.
"""
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from rich import print as rprint

from .exceptions import CheckpointError

# A try-except block to gracefully handle environments where LoRA might not be installed.
try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:
    # Create a dummy class if LoRA is not available to prevent runtime errors.
    class MLXLoRALinear:
        pass

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages the lifecycle of training checkpoints: saving, loading, and rotation."""

    def __init__(
        self,
        output_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
        base_model_path: Optional[Path] = None, # <-- NON-BREAKING CHANGE
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.base_model_path = base_model_path # <-- Store the path
        self._warned_about_missing_path = False

        self.best_metric: float = -float("inf")
        self._checkpoints: List[Path] = []
        self.resume_from_path: Optional[Path] = None
        self._load_existing_checkpoints()

    def _get_step_from_path(self, path: Path) -> Optional[int]:
        """Utility to extract the training step number from a checkpoint path name."""
        match = re.search(r"update_(\d+)$", path.name)
        if match:
            return int(match.group(1))
        return None

    def _load_existing_checkpoints(self):
        """Loads and sorts existing checkpoints from the output directory."""
        found_dirs_with_steps: List[Tuple[int, Path]] = []

        for p in self.output_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                step = self._get_step_from_path(p)
                if step is not None:
                    found_dirs_with_steps.append((step, p))

        found_dirs_with_steps.sort(key=lambda x: x[0])
        self._checkpoints = [p for _, p in found_dirs_with_steps]

        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                resolved_path = best_symlink.resolve(strict=True)
                if resolved_path.is_dir():
                    with open(resolved_path / "metadata.json", "r") as f:
                        self.best_metric = json.load(f).get(
                            "current_metric", -float("inf")
                        )
            except FileNotFoundError:
                logger.warning("Symlink 'best' is dangling. Removing it.")
                best_symlink.unlink()
            except Exception as e:
                logger.warning(f"Could not load best_metric from 'best' symlink: {e}")

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metadata: Dict[str, Any],
        current_metric: float,
    ):
        """Saves a complete, portable checkpoint atomically."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_update_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp"
        final_path = self.output_dir / checkpoint_name

        if temp_path.exists():
            shutil.rmtree(temp_path)

        try:
            temp_path.mkdir(parents=True)

            # --- Copy essential config and tokenizer files if path provided ---
            if self.base_model_path:
                files_to_copy = [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
                for file_pattern in ["*.model", "*.txt", "*.py"]:
                    for f_path in self.base_model_path.glob(file_pattern):
                        files_to_copy.append(f_path.name)

                for file_name in set(files_to_copy):
                    source_file = self.base_model_path / file_name
                    if source_file.is_file():
                        shutil.copy2(source_file, temp_path / file_name)
            elif not self._warned_about_missing_path:
                rprint(
                    "[yellow]Warning: `base_model_path` not provided to CheckpointManager. "
                    "Checkpoints will not be self-contained.[/yellow]"
                )
                self._warned_about_missing_path = True

            # --- Save model weights (LoRA or full) ---
            is_lora = any(
                isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
            )
            if is_lora:
                adapter_params = dict(tree_flatten(model.trainable_parameters()))
                if adapter_params:
                    mx.save_safetensors(
                        str(temp_path / "adapters.safetensors"), adapter_params
                    )
            else:
                full_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(str(temp_path / "model.safetensors"), full_params)

            if metadata.get("save_optimizer_state", False) and optimizer:
                mx.save_safetensors(
                    str(temp_path / "optimizer.safetensors"),
                    dict(tree_flatten(optimizer.state)),
                )

            metadata["step"] = step
            metadata["current_metric"] = current_metric
            with open(temp_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            os.rename(temp_path, final_path)
            self._checkpoints.append(final_path)
            rprint(
                f"Checkpoint saved to [cyan]{final_path.name}[/cyan] (Metric: {current_metric:.4f})."
            )

            self._update_symlink(final_path, "latest")
            if self.is_best_metric(current_metric):
                self.best_metric = current_metric
                self._update_symlink(final_path, "best")

            self._rotate_checkpoints()
        except Exception as e:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise CheckpointError(f"Atomic save failed for step {step}: {e}") from e

    def load_latest_state(
        self, model: nn.Module, optimizer: optim.Optimizer
    ) -> Tuple[int, Dict[str, Any]]:
        """Loads the latest checkpoint to resume training."""
        chosen_path = self.resume_from_path
        if not chosen_path:
            latest_symlink = self.output_dir / "latest"
            if latest_symlink.is_symlink():
                try:
                    chosen_path = latest_symlink.resolve(strict=True)
                except FileNotFoundError:
                    logger.warning("Symlink 'latest' is dangling. Searching for last checkpoint.")
                    latest_symlink.unlink()
                    if self._checkpoints:
                        chosen_path = self._checkpoints[-1]
            elif self._checkpoints:
                chosen_path = self._checkpoints[-1]

        if not chosen_path or not chosen_path.exists():
            rprint("[yellow]No checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}

        rprint(f"Resuming training from checkpoint: [green]{chosen_path.name}[/green]")
        try:
            with open(chosen_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            is_lora = any(isinstance(m, MLXLoRALinear) for _, m in model.named_modules())
            adapters_file = chosen_path / "adapters.safetensors"
            model_file = chosen_path / "model.safetensors"

            if is_lora and adapters_file.is_file():
                from mlx_lm.tuner.utils import load_adapters
                load_adapters(model, str(chosen_path))
                rprint("Loaded LoRA adapters.")
            elif model_file.is_file():
                model.load_weights(list(mx.load(str(model_file)).items()))
                rprint("Loaded full model weights.")

            optimizer_loaded = False
            optimizer_file = chosen_path / "optimizer.safetensors"
            if metadata.get("save_optimizer_state") and optimizer_file.is_file():
                optimizer.state = tree_unflatten(
                    list(mx.load(str(optimizer_file)).items())
                )
                optimizer_loaded = True
                rprint("Loaded optimizer state.")

            params_to_eval = list(model.parameters())
            if optimizer_loaded:
                params_to_eval.extend(optimizer.state.values())
            mx.eval(params_to_eval)

            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_step = metadata.get("step", 0)
            return resumed_step, metadata
        except Exception as e:
            raise CheckpointError(
                f"Failed to load state from {chosen_path.name}: {e}"
            ) from e

    def _update_symlink(self, target_path: Path, link_name: str):
        """Atomically updates a symlink to point to a new target directory."""
        link_path = self.output_dir / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(
            os.path.relpath(target_path, self.output_dir),
            link_path,
            target_is_directory=True,
        )

    def _rotate_checkpoints(self):
        """Deletes old checkpoints, keeping the last N and the best one."""
        if len(self._checkpoints) <= self.keep_last_n:
            return

        best_path = None
        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                best_path = best_symlink.resolve(strict=True)
            except FileNotFoundError:
                best_symlink.unlink()

        checkpoints_to_keep = set(self._checkpoints[-self.keep_last_n :])
        if best_path:
            checkpoints_to_keep.add(best_path)

        checkpoints_to_delete = []
        for chk in self._checkpoints:
            if chk not in checkpoints_to_keep:
                checkpoints_to_delete.append(chk)

        for chk in checkpoints_to_delete:
            if chk.exists():
                rprint(f"Rotating old checkpoint: [red]{chk.name}[/red]")
                shutil.rmtree(chk, ignore_errors=True)

        self._checkpoints = sorted(
            list(checkpoints_to_keep),
            key=lambda p: self._get_step_from_path(p) or -1,
        )

    def is_best_metric(self, current_metric: float) -> bool:
        """Checks if the current metric is better than the best one seen so far."""
        return self.save_best and current_metric > self.best_metric
