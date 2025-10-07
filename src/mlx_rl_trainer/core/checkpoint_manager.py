"""Checkpoint management for training persistence."""
import json, logging, os, re, shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from .trainer import CheckpointError
from rich import print as rprint
import random, numpy as np
from mlx_lm.tuner.utils import remove_lora_layers
from mlx_lm.utils import save_config

logger = logging.getLogger(__name__)

def _is_lora_module(m):
    try:
        from mlx_lm.tuner.lora import LoRALinear
        return isinstance(m, LoRALinear)
    except ImportError:
        return False

class CheckpointManager:
    def __init__(self, output_dir: Path, keep_last_n: int = 3):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_metric: float = -float("inf")
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
        shutil.rmtree(final_path, ignore_errors=True)
        
        try:
            temp_path.mkdir(parents=True)
            
            # LoRA-aware saving
            is_lora = any(_is_lora_module(m) for m in model.modules())
            if is_lora:
                adapter_params = dict(tree_flatten(model.trainable_parameters()))
                if adapter_params:
                    mx.save_safetensors(str(temp_path / "adapters.safetensors"), adapter_params)
                    rprint(f"[bold green]Saved LoRA adapters to {temp_path.name}[/bold green]")
                
                # Optionally save fused model
                fused_model = remove_lora_layers(model) # fuse is implicit
                mx.eval(fused_model.parameters())
                full_params = dict(tree_flatten(fused_model.parameters()))
                mx.save_safetensors(str(temp_path / "model.safetensors"), full_params)
                rprint(f"[bold green]Saved fused base model to {temp_path.name}[/bold green]")
            else:
                full_params = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(str(temp_path / "model.safetensors"), full_params)

            if metadata.get("save_optimizer_state", False):
                mx.save_safetensors(str(temp_path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))

            metadata["current_metric"] = current_metric
            with open(temp_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)

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

    def load_latest_state(self, model: nn.Module, optimizer: optim) -> Tuple[int, Dict[str, Any]]:
        latest_path = self.output_dir / "latest"
        if not latest_path.is_symlink():
            rprint("[yellow]No 'latest' checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}
        
        checkpoint_path = latest_path.resolve()
        rprint(f"Resuming from checkpoint: [cyan]{checkpoint_path.name}[/cyan]...")

        try:
            with open(checkpoint_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            model_weights = mx.load(str(checkpoint_path / "model.safetensors"))
            model.load_weights(list(model_weights.items()))

            is_lora = any(_is_lora_module(m) for m in model.modules())
            if is_lora and (checkpoint_path / "adapters.safetensors").exists():
                adapter_weights = mx.load(str(checkpoint_path / "adapters.safetensors"))
                model.update(tree_unflatten(list(adapter_weights.items())))
                rprint("Loaded LoRA adapter weights.")

            if (checkpoint_path / "optimizer.safetensors").exists():
                opt_state = mx.load(str(checkpoint_path / "optimizer.safetensors"))
                optimizer.state = tree_unflatten(list(opt_state.items()))
            
            mx.eval(model.parameters(), optimizer.state)
            self.best_metric = metadata.get("current_metric", -float("inf"))
            resumed_updates = metadata.get("num_updates", 0)
            return resumed_updates, metadata
        except Exception as e:
            raise CheckpointError(f"Failed to load state from {checkpoint_path.name}: {e}") from e

    def _update_symlink(self, target_path: Path, link_name: str):
        link_path = self.output_dir / link_name
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        os.symlink(os.path.relpath(target_path, self.output_dir), link_path, target_is_directory=True)

    def _rotate_checkpoints(self):
        self._checkpoints = sorted([p for p in self.output_dir.glob("checkpoint_update_*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        best_path = (self.output_dir / "best").resolve() if (self.output_dir / "best").is_symlink() else None
        
        to_delete = self._checkpoints[:-self.keep_last_n]
        for chk in to_delete:
            if chk != best_path:
                shutil.rmtree(chk, ignore_errors=True)
                rprint(f"Deleted old checkpoint: {chk.name}")
