"""Checkpoint management for training persistence."""
import json, logging, os, re, shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from .exceptions import CheckpointError
from rich import print as rprint
import random, numpy as np
from mlx_lm.tuner.utils import remove_lora_layers
from mlx_lm.utils import save_config

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:

    class MLXLoRALinear:
        pass


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint lifecycle: saving, loading, and rotation."""

    def __init__(self, output_dir: Path, keep_last_n: int = 3, save_best: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric: float = -float("inf")
        self._checkpoints: List[Path] = []
        self.resume_from_path: Optional[Path] = None
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self):
        checkpoint_regex = re.compile(r"checkpoint_[\d]{8}_[\d]{6}_update_(\d+)$")
        found_dirs_with_steps: List[Tuple[int, Path]] = []

        for p in self.output_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                if match := checkpoint_regex.match(p.name):
                    found_dirs_with_steps.append((int(match.group(1)), p))

        found_dirs_with_steps.sort(key=lambda x: x[0])
        self._checkpoints = [p for _, p in found_dirs_with_steps]

        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink() and best_symlink.resolve().is_dir():
            try:
                with open(best_symlink.resolve() / "metadata.json", "r") as f:
                    self.best_metric = json.load(f).get("current_metric", -float("inf"))
            except Exception as e:
                logging.warning(f"Could not load best_metric from 'best' symlink: {e}")

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        metadata: Dict[str, Any],
        current_metric: float,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timestamp}_update_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp"
        final_path = self.output_dir / checkpoint_name
        shutil.rmtree(temp_path, ignore_errors=True)

        try:
            temp_path.mkdir(parents=True)

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
        chosen_path = self.resume_from_path
        if not chosen_path:
            latest_symlink = self.output_dir / "latest"
            if latest_symlink.is_symlink() and latest_symlink.resolve().is_dir():
                chosen_path = latest_symlink.resolve()
            elif self._checkpoints:
                chosen_path = self._checkpoints[-1]

        if not chosen_path or not chosen_path.exists():
            rprint("[yellow]No checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}

        try:
            with open(chosen_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            if (chosen_path / "model.safetensors").exists():
                model.load_weights(
                    list(mx.load(str(chosen_path / "model.safetensors")).items())
                )

            if (
                any(isinstance(m, MLXLoRALinear) for _, m in model.named_modules())
                and (chosen_path / "adapters.safetensors").exists()
            ):
                from mlx_lm.tuner.utils import load_adapters

                load_adapters(model, str(chosen_path))

            if (
                metadata.get("save_optimizer_state")
                and (chosen_path / "optimizer.safetensors").exists()
            ):
                optimizer.state = tree_unflatten(
                    list(mx.load(str(chosen_path / "optimizer.safetensors")).items())
                )

            mx.eval(model.parameters(), optimizer.state)
            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_updates = metadata.get("num_updates", 0)
            return resumed_updates, metadata
        except Exception as e:
            raise CheckpointError(
                f"Failed to load state from {chosen_path.name}: {e}"
            ) from e

    def _update_symlink(self, target_path: Path, link_name: str):
        link_path = self.output_dir / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(
            os.path.relpath(target_path, self.output_dir),
            link_path,
            target_is_directory=True,
        )

    def _rotate_checkpoints(self):
        self._checkpoints = sorted(
            [p for p in self.output_dir.glob("checkpoint_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        best_path = (
            (self.output_dir / "best").resolve()
            if (self.output_dir / "best").is_symlink()
            else None
        )

        to_delete = [
            chk for chk in self._checkpoints[: -self.keep_last_n] if chk != best_path
        ]
        for chk in to_delete:
            shutil.rmtree(chk, ignore_errors=True)
            if chk in self._checkpoints:
                self._checkpoints.remove(chk)

    def is_best_metric(self, current_metric: float) -> bool:
        return self.save_best and current_metric > self.best_metric
