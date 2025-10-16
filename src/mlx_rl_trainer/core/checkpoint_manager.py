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
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set

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
        base_model_path: Optional[Path] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.base_model_path = base_model_path
        self._warned_about_missing_path = False

        self.best_metric: float = -float("inf")
        self._checkpoints: List[Path] = []
        self.resume_from_path: Optional[Path] = None
        self._load_existing_checkpoints()

    def _aggressive_memory_cleanup(self):
        """Aggressively free memory."""
        try:
            mx.metal.clear_cache()
        except:
            pass
        mx.clear_cache()
        gc.collect()

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

        # Clean up temporary variable
        del found_dirs_with_steps

        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                resolved_path = best_symlink.resolve(strict=True)
                if resolved_path.is_dir():
                    metadata_file = resolved_path / "metadata.json"
                    if metadata_file.is_file():
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            # Bug fix: Use get with default value properly
                            self.best_metric = metadata.get(
                                "current_metric", -float("inf")
                            )
                            del metadata
                    else:
                        logger.warning(
                            f"Metadata file missing in best checkpoint: {resolved_path}"
                        )
            except FileNotFoundError:
                logger.warning("Symlink 'best' is dangling. Removing it.")
                best_symlink.unlink()
            except Exception as e:
                logger.warning(f"Could not load best_metric from 'best' symlink: {e}")

    def _copy_base_model_files(self, dest_path: Path):
        """
        Copy essential config and tokenizer files from base model.
        Memory optimized: Use set operations and stream file copies.
        """
        if not self.base_model_path or not self.base_model_path.exists():
            return

        # Build file list efficiently using a set
        files_to_copy: Set[str] = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "added_tokens.json",
            "chat_template.jinja",
        }

        # Add glob patterns efficiently
        for file_pattern in ["*.model", "*.txt", "*.py"]:
            for f_path in self.base_model_path.glob(file_pattern):
                files_to_copy.add(f_path.name)

        # Copy files one by one with error handling
        copied_count = 0
        for file_name in files_to_copy:
            source_file = self.base_model_path / file_name
            if source_file.is_file():
                try:
                    shutil.copy2(source_file, dest_path / file_name)
                    copied_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {file_name}: {e}")

        # Cleanup
        del files_to_copy

        if copied_count > 0:
            logger.debug(f"Copied {copied_count} base model files to checkpoint")

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metadata: Dict[str, Any],
        current_metric: Optional[float] = None,
        retries: int = 3,
        backoff_factor: float = 2.0,
    ):
        """
        Saves a complete, portable checkpoint atomically with retries.
        """
        if current_metric is None:
            current_metric = self.best_metric

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_update_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp"
        final_path = self.output_dir / checkpoint_name

        for attempt in range(retries):
            if temp_path.exists():
                shutil.rmtree(temp_path)

            try:
                temp_path.mkdir(parents=True)

                if self.base_model_path:
                    self._copy_base_model_files(temp_path)
                elif not self._warned_about_missing_path:
                    rprint(
                        "[yellow]Warning: `base_model_path` not provided. Checkpoints may not be self-contained.[/yellow]"
                    )
                    self._warned_about_missing_path = True

                is_lora = any(
                    isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
                )

                if is_lora:
                    adapter_params = dict(tree_flatten(model.trainable_parameters()))
                    if adapter_params:
                        mx.save_safetensors(
                            str(temp_path / "adapters.safetensors"), adapter_params
                        )
                        del adapter_params
                else:
                    full_params = dict(tree_flatten(model.parameters()))
                    mx.save_safetensors(
                        str(temp_path / "model.safetensors"), full_params
                    )
                    del full_params
                self._aggressive_memory_cleanup()

                if metadata.get("save_optimizer_state", False) and optimizer:
                    optimizer_state = dict(tree_flatten(optimizer.state))
                    mx.save_safetensors(
                        str(temp_path / "optimizer.safetensors"), optimizer_state
                    )
                    del optimizer_state
                    self._aggressive_memory_cleanup()

                metadata["step"] = step
                metadata["current_metric"] = current_metric
                metadata["timestamp"] = timestamp

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
                self._aggressive_memory_cleanup()
                return  # Success

            except (IOError, OSError) as e:
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)

                rprint(
                    f"[bold red]CRITICAL: Checkpoint save failed on attempt {attempt + 1}/{retries}. Error: {e}[/bold red]"
                )
                if "No space left on device" in str(e):
                    rprint(
                        "[bold red]Disk may be full. Please check storage.[/bold red]"
                    )
                    break  # No point in retrying if disk is full
                if "Permission denied" in str(e):
                    rprint(
                        "[bold red]Permission denied. Check directory permissions.[/bold red]"
                    )
                    break

                if attempt < retries - 1:
                    sleep_time = backoff_factor**attempt
                    rprint(f"[yellow]Retrying in {sleep_time:.1f} seconds...[/yellow]")
                    time.sleep(sleep_time)
                else:
                    rprint("[bold red]All checkpoint save retries failed.[/bold red]")
                    raise CheckpointError(
                        f"Atomic save failed for step {step}: {e}"
                    ) from e
            except Exception as e:
                if temp_path.exists():
                    shutil.rmtree(temp_path, ignore_errors=True)
                raise CheckpointError(
                    f"An unexpected error occurred during checkpoint save for step {step}: {e}"
                ) from e

    def load_latest_state(
        self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Loads the latest checkpoint to resume training.
        Memory optimized: Stream loading and immediate cleanup.
        Bug fix: Better error handling and None optimizer support.
        """
        chosen_path = self.resume_from_path

        if not chosen_path:
            latest_symlink = self.output_dir / "latest"
            if latest_symlink.is_symlink():
                try:
                    chosen_path = latest_symlink.resolve(strict=True)
                except FileNotFoundError:
                    logger.warning(
                        "Symlink 'latest' is dangling. Searching for last checkpoint."
                    )
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
            # Load metadata
            metadata_file = chosen_path / "metadata.json"
            if not metadata_file.is_file():
                raise CheckpointError(
                    f"Metadata file missing in checkpoint: {chosen_path.name}"
                )

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Determine if this is a LoRA checkpoint
            is_lora = any(
                isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
            )
            adapters_file = chosen_path / "adapters.safetensors"
            model_file = chosen_path / "model.safetensors"

            # Load model weights
            if is_lora and adapters_file.is_file():
                from mlx_lm.tuner.utils import load_adapters

                load_adapters(model, str(chosen_path))
                rprint("Loaded LoRA adapters.")
            elif model_file.is_file():
                # Load weights and immediately cleanup
                weights = list(mx.load(str(model_file)).items())
                model.load_weights(weights)
                del weights
                self._aggressive_memory_cleanup()
                rprint("Loaded full model weights.")
            else:
                raise CheckpointError(
                    f"No model weights found in checkpoint: {chosen_path.name}. "
                    f"Expected {'adapters.safetensors' if is_lora else 'model.safetensors'}"
                )

            # Load optimizer state if requested and available
            optimizer_loaded = False
            if optimizer is not None:  # Bug fix: Check if optimizer is provided
                optimizer_file = chosen_path / "optimizer.safetensors"
                if metadata.get("save_optimizer_state") and optimizer_file.is_file():
                    try:
                        optimizer_state_items = list(
                            mx.load(str(optimizer_file)).items()
                        )
                        optimizer.state = tree_unflatten(optimizer_state_items)
                        del optimizer_state_items
                        optimizer_loaded = True
                        self._aggressive_memory_cleanup()
                        rprint("Loaded optimizer state.")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer state: {e}")

            # Evaluate all loaded parameters
            params_to_eval = list(model.parameters())
            if optimizer_loaded and optimizer is not None:
                params_to_eval.extend(list(optimizer.state.values()))
            mx.eval(params_to_eval)
            del params_to_eval

            # Update best metric
            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_step = metadata.get("step", metadata.get("num_updates", 0))

            # Final cleanup
            self._aggressive_memory_cleanup()

            return resumed_step, metadata

        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(
                f"Failed to load state from {chosen_path.name}: {e}"
            ) from e

    def _update_symlink(self, target_path: Path, link_name: str):
        """Atomically updates a symlink to point to a new target directory."""
        link_path = self.output_dir / link_name

        # Remove existing symlink or file
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()

        # Create new symlink
        os.symlink(
            os.path.relpath(target_path, self.output_dir),
            link_path,
            target_is_directory=True,
        )

    def _rotate_checkpoints(self):
        """
        Deletes old checkpoints, keeping the last N and the best one.
        Memory optimized: Use sets and immediate cleanup.
        """
        if len(self._checkpoints) <= self.keep_last_n:
            return

        # Find the best checkpoint path
        best_path = None
        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                best_path = best_symlink.resolve(strict=True)
            except FileNotFoundError:
                best_symlink.unlink()
                best_path = None

        # Build set of checkpoints to keep
        checkpoints_to_keep: Set[Path] = set(self._checkpoints[-self.keep_last_n :])
        if best_path:
            checkpoints_to_keep.add(best_path)

        # Identify checkpoints to delete
        checkpoints_to_delete = [
            chk for chk in self._checkpoints if chk not in checkpoints_to_keep
        ]

        # Delete old checkpoints
        for chk in checkpoints_to_delete:
            if chk.exists():
                try:
                    rprint(f"Rotating old checkpoint: [red]{chk.name}[/red]")
                    shutil.rmtree(chk, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {chk.name}: {e}")

        # Update the internal list
        self._checkpoints = sorted(
            list(checkpoints_to_keep),
            key=lambda p: self._get_step_from_path(p) or -1,
        )

        # Cleanup
        del checkpoints_to_delete, checkpoints_to_keep

    def is_best_metric(self, current_metric: Optional[float]) -> bool:
        """
        Checks if the current metric is better than the best one seen so far.
        Bug fix: Handle None metric properly.
        """
        if current_metric is None:
            return False
        return self.save_best and current_metric > self.best_metric

    def resume_from_checkpoint(
        self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None
    ) -> Tuple[int, int]:
        """
        Loads the latest checkpoint to resume training.
        """
        resumed_step = 0
        resumed_epoch = 0
        if self._checkpoints:
            try:
                resumed_step, metadata = self.load_latest_state(model, optimizer)
                resumed_epoch = metadata.get("epoch", 0)
            except CheckpointError as e:
                rprint(
                    f"[bold red]Failed to load latest checkpoint: {e}. Starting from scratch.[/bold red]"
                )
            except Exception as e:
                rprint(
                    f"[bold red]An unexpected error occurred while resuming: {e}. Starting from scratch.[/bold red]"
                )
        return resumed_step, resumed_epoch
