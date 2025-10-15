import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from rich import print as rprint

from .exceptions import CheckpointError

try:
    from mlx_lm.tuner.lora import LoRALinear as MLXLoRALinear
except ImportError:

    class MLXLoRALinear:
        pass


logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(
        self,
        output_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
        base_model_path: Optional[Path] = None,
        retries: int = 3,
        backoff_factor: float = 5.0,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.base_model_path = base_model_path
        self._warned_about_missing_path = False
        self.best_metric = -float("inf")
        self._checkpoints = []
        self.resume_from_path = None
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._load_existing_checkpoints()

    def _get_step_from_path(self, path: Path) -> Optional[int]:
        match = re.search(r"update_(\d+)", path.name)
        return int(match.group(1)) if match else None

    def _load_existing_checkpoints(self):
        checkpoints = []
        for p in self.output_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").is_file():
                step = self._get_step_from_path(p)
                if step is not None:
                    checkpoints.append((step, p))

        checkpoints.sort(key=lambda x: x[0])
        self._checkpoints = [p for _, p in checkpoints]

        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink():
            try:
                resolved_path = best_symlink.resolve(strict=True)
                if (resolved_path / "metadata.json").is_file():
                    with open(resolved_path / "metadata.json", "r") as f:
                        meta = json.load(f)
                        self.best_metric = meta.get("current_metric", -float("inf"))
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning(
                    "Best checkpoint symlink is dangling or invalid. Removing."
                )
                best_symlink.unlink()

    def _copy_base_model_files(self, dest_path: Path):
        if not self.base_model_path or not self.base_model_path.is_dir():
            return

        files_to_copy = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        }
        for f in files_to_copy:
            src = self.base_model_path / f
            if src.is_file():
                shutil.copy2(src, dest_path / f)

    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer,
        metadata: Dict[str, Any],
        current_metric: Optional[float] = None,
    ):
        ckpt_name = f"checkpoint_{time.strftime('%Y%m%d_%H%M%S')}_update_{step}"
        tmp_dir = self.output_dir / f".{ckpt_name}.tmp"
        final_dir = self.output_dir / ckpt_name

        for attempt in range(self.retries + 1):
            try:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True)

                self._copy_base_model_files(tmp_dir)

                is_lora = any(
                    isinstance(m, MLXLoRALinear) for _, m in model.named_modules()
                )
                if is_lora:
                    trainable_params = dict(tree_flatten(model.trainable_parameters()))
                    if trainable_params:
                        mx.save_safetensors(
                            str(tmp_dir / "adapters.safetensors"), trainable_params
                        )
                else:
                    mx.save_safetensors(
                        str(tmp_dir / "model.safetensors"),
                        dict(tree_flatten(model.parameters())),
                    )

                if metadata.get("save_optimizer_state", True) and optimizer:
                    mx.save_safetensors(
                        str(tmp_dir / "optimizer.safetensors"),
                        dict(tree_flatten(optimizer.state)),
                    )

                metadata["step"] = step
                metadata["current_metric"] = (
                    current_metric if current_metric is not None else self.best_metric
                )

                with open(tmp_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                os.rename(tmp_dir, final_dir)
                self._checkpoints.append(final_dir)

                rprint(
                    f"Checkpoint saved to [cyan]{final_dir.name}[/cyan] (Metric: {metadata['current_metric']:.4f})."
                )
                self._update_symlink(final_dir, "latest")

                if self.is_best_metric(current_metric):
                    self.best_metric = current_metric
                    rprint(
                        f"[bold green]New best metric: {self.best_metric:.4f}. Updating 'best' checkpoint.[/bold green]"
                    )
                    self._update_symlink(final_dir, "best")

                self._rotate_checkpoints()
                return  # Success

            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1}/{self.retries + 1} failed to save checkpoint: {e}"
                )
                if attempt < self.retries:
                    backoff_time = self.backoff_factor * (2**attempt)
                    rprint(
                        f"[bold yellow]Retrying in {backoff_time:.1f} seconds... (Check disk space or permissions)[/bold yellow]"
                    )
                    time.sleep(backoff_time)
                else:
                    rprint(
                        f"[bold red]FATAL: Failed to save checkpoint after {self.retries + 1} attempts. Training may continue without saving.[/bold red]"
                    )
                    if tmp_dir.exists():
                        shutil.rmtree(tmp_dir)
                    # Do not re-raise to allow training to continue if possible
                    return

    def load_latest_state(self, model, optimizer=None):
        path_to_load = self.resume_from_path
        if not path_to_load:
            latest_symlink = self.output_dir / "latest"
            if latest_symlink.is_symlink():
                try:
                    path_to_load = latest_symlink.resolve(strict=True)
                except FileNotFoundError:
                    pass

        if not path_to_load and self._checkpoints:
            path_to_load = self._checkpoints[-1]

        if not path_to_load or not path_to_load.is_dir():
            rprint("[yellow]No checkpoint found. Starting from scratch.[/yellow]")
            return 0, {}

        rprint(f"Resuming training from checkpoint: [green]{path_to_load.name}[/green]")
        try:
            with open(path_to_load / "metadata.json", "r") as f:
                metadata = json.load(f)

            if (path_to_load / "adapters.safetensors").is_file():
                from mlx_lm.tuner.utils import load_adapters

                load_adapters(model, str(path_to_load))
            elif (path_to_load / "model.safetensors").is_file():
                model.load_weights(
                    list(mx.load(str(path_to_load / "model.safetensors")).items())
                )

            if optimizer and (path_to_load / "optimizer.safetensors").is_file():
                optimizer.state = tree_unflatten(
                    list(mx.load(str(path_to_load / "optimizer.safetensors")).items())
                )

            mx.eval(model.parameters())
            self.best_metric = metadata.get("current_metric", -float("inf"))
            return metadata.get("step", 0), metadata

        except Exception as e:
            raise CheckpointError(
                f"Failed to load state from {path_to_load.name}: {e}"
            ) from e

    def _update_symlink(self, target_path: Path, link_name: str):
        link = self.output_dir / link_name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(
            os.path.relpath(target_path, self.output_dir),
            link,
            target_is_directory=True,
        )

    def _rotate_checkpoints(self):
        if len(self._checkpoints) <= self.keep_last_n:
            return

        best_path = None
        if (self.output_dir / "best").is_symlink():
            try:
                best_path = (self.output_dir / "best").resolve(strict=True)
            except FileNotFoundError:
                pass

        to_keep = set(self._checkpoints[-self.keep_last_n :])
        if best_path:
            to_keep.add(best_path)

        to_remove = [p for p in self._checkpoints if p not in to_keep]

        for p in to_remove:
            if p.exists():
                rprint(f"Rotating old checkpoint: [red]{p.name}[/red]")
                shutil.rmtree(p)

        self._checkpoints = sorted(
            list(to_keep), key=lambda p: self._get_step_from_path(p) or -1
        )

    def is_best_metric(self, current_metric: Optional[float]) -> bool:
        if current_metric is None:
            return False
        return self.save_best and current_metric > self.best_metric
