"""Checkpoint management for training persistence."""
import json, logging, os, re, shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from .exceptions import CheckpointError # CORRECTED IMPORT
from rich import print as rprint
import random, numpy as np

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
    from mlx_lm.tuner.utils import remove_lora_layers, load_adapters
    from mlx_lm.utils import save_config
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    class MLXLoRALinear: pass
    def remove_lora_layers(model: Any): return model
    def load_adapters(model: Any, path: str): return model
    def save_config(config: Dict[str,Any], config_path: str): pass

logger = logging.getLogger(__name__)

def _is_lora_module(m):
    return MLX_LM_AVAILABLE and isinstance(m, MLXLoRALinear)

class CheckpointManager:
    def __init__(self, output_dir: Path, keep_last_n: int = 3, save_best: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric: float = -float("inf")
        self.resume_from_path: Optional[Path] = None
        self._checkpoints: List[Path] = self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> List[Path]:
        found = [p for p in self.output_dir.glob("checkpoint_update_*") if p.is_dir()]
        found.sort(key=lambda p: p.stat().st_mtime)
        rprint(f"Found {len(found)} existing checkpoints.")
        return found
            
    def save_checkpoint(self, step: int, model: nn.Module, optimizer: optim, metadata: Dict[str, Any], current_metric: float):
        checkpoint_name = f"checkpoint_update_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp"
        final_path = self.output_dir / checkpoint_name
        shutil.rmtree(temp_path, ignore_errors=True)
        try:
            temp_path.mkdir(parents=True)
            
            is_lora = any(_is_lora_module(m) for m in model.modules())
            if is_lora:
                adapter_params = dict(tree_flatten(model.trainable_parameters()))
                if adapter_params:
                    mx.save_safetensors(str(temp_path / "adapters.safetensors"), adapter_params)
            else:
                full_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(str(temp_path / "model.safetensors"), full_params)

            if metadata.get("save_optimizer_state", False):
                mx.save_safetensors(str(temp_path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))

            metadata["current_metric"] = current_metric
            with open(temp_path / "metadata.json", "w") as f: json.dump(metadata, f, indent=2, default=str)

            os.rename(temp_path, final_path)
            self._checkpoints.append(final_path)
            rprint(f"Checkpoint saved to [cyan]{final_path.name}[/cyan] (Metric: {current_metric:.4f}).")

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self._update_symlink(final_path, "best")
            
            self._update_symlink(final_path, "latest")
            self._rotate_checkpoints()

        except Exception as e:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise CheckpointError(f"Atomic save failed for step {step}: {e}") from e

    def load_latest_state(self, model: nn.Module, optimizer: optim.Optimizer) -> Tuple[int, Dict[str, Any]]:
        latest_path = self.resume_from_path or self.output_dir / "latest"
        if not (latest_path.exists() or latest_path.is_symlink()):
            rprint("[yellow]No checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}
        
        checkpoint_path = latest_path.resolve()
        rprint(f"Resuming from checkpoint: [cyan]{checkpoint_path.name}[/cyan]...")

        try:
            with open(checkpoint_path / "metadata.json", "r") as f: metadata = json.load(f)

            if (p := checkpoint_path / "model.safetensors").exists():
                model_state = mx.load(str(p))
                model.load_weights(list(model_state.items()))

            if any(_is_lora_module(m) for _, m in model.named_modules()) and (p := checkpoint_path / "adapters.safetensors").exists():
                load_adapters(model, str(checkpoint_path))

            mx.eval(model.parameters())
            
            if metadata.get("save_optimizer_state", False) and (p := checkpoint_path / "optimizer.safetensors").exists():
                opt_state = mx.load(str(p))
                optimizer.state = tree_unflatten(list(opt_state.items()))
                mx.eval(optimizer.state)
            
            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_updates = metadata.get("num_updates", 0)
            
            logging.info(f"Checkpoint loaded. Resuming from step {resumed_updates}.")
            return resumed_updates, metadata
        except Exception as e:
            raise CheckpointError(f"Failed to load state from {checkpoint_path.name}: {e}") from e

    def _update_symlink(self, target_path: Path, link_name: str):
        link_path = self.output_dir / link_name
        if link_path.is_symlink() or link_path.exists(): link_path.unlink()
        os.symlink(os.path.relpath(target_path, self.output_dir), link_path, target_is_directory=True)

    def _rotate_checkpoints(self):
        self._checkpoints = sorted([p for p in self.output_dir.glob("checkpoint_update_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        best_path = (self.output_dir / "best").resolve() if (self.output_dir / "best").is_symlink() else None
        
        to_delete = self._checkpoints[:-self.keep_last_n]
        for chk in to_delete:
            if chk != best_path:
                shutil.rmtree(chk, ignore_errors=True)
                rprint(f"Deleted old checkpoint: {chk.name}")

    def is_best_metric(self, current_metric: float) -> bool:
        return current_metric > self.best_metric
