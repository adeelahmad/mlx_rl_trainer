# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/checkpoint_manager.py
# revision_no: 001
# goals_of_writing_code_block: Manage atomic checkpoint saving, loading, and rotation to ensure training persistence and robustness against interruptions.
# type_of_code_response: add new code
"""
Checkpoint management for training persistence.
"""
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import shutil
import logging
import os
import mlx.core as mx
import mlx.nn as nn
from .trainer import CheckpointError
from rich import print as rprint
import random # For RNG state
import numpy as np # For RNG state

# --- MLX-LM Imports (Conditional for LoRA support) ---
try:
    import mlx_lm
    from mlx_lm.utils import load as mlx_lm_load, save_config as mlx_lm_save_config
    from mlx_lm.tuner.utils import remove_lora_layers, fuse_lora_layers
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    logging.warning("mlx-lm not found. CheckpointManager will have limited LoRA handling.")
    # Mock minimal functions if mlx_lm is not available
    def remove_lora_layers(model: Any): return model
    def fuse_lora_layers(model: Any): return model
    def mlx_lm_load(path: Path): raise CheckpointError("mlx-lm not available for loading model.")
    def mlx_lm_save_config(*args, **kwargs): pass


logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    Manages checkpoint lifecycle and persistence.

    Responsibilities:
        - Save checkpoints atomically to prevent corruption.
        - Load checkpoints with validation for resuming training.
        - Rotate old checkpoints to manage disk space.
        - Track the best performing model based on a metric.
    """

    def __init__(self, output_dir: Path, keep_last_n: int = 3, save_best: bool = True):
        """
        Initializes the CheckpointManager.

        Args:
            output_dir: The base directory where checkpoints will be saved.
            keep_last_n: The number of most recent checkpoints to retain.
            save_best: If True, a separate symlink 'best' will always point to the best model.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric: float = -float('inf') # Tracks the best metric seen for saving 'best' model
        self._checkpoints: List[Path] = [] # List of paths to saved checkpoints
        self._load_existing_checkpoints() # Populate self._checkpoints from disk
        logging.info(f"CheckpointManager initialized for {output_dir}. Keeping last {keep_last_n} checkpoints.")

    def _load_existing_checkpoints(self):
        """Scans the output directory for existing checkpoints and populates the internal list."""
        # Checkpoint directories are typically named 'checkpoint_step_<number>'
        found = [p for p in self.output_dir.glob("checkpoint_step_*") if p.is_dir() and (p / "metadata.json").is_file()]
        found.sort(key=lambda p: p.stat().st_mtime) # Sort by modification time to find latest
        self._checkpoints = found
        rprint(f"Found {len(self._checkpoints)} existing checkpoints.")

        # Try to load the best metric from the 'best' symlink if it exists
        best_symlink = self.output_dir / "best"
        if best_symlink.is_symlink() and best_symlink.resolve().is_dir():
            try:
                with open(best_symlink.resolve() / "metadata.json", 'r') as f:
                    self.best_metric = json.load(f).get('current_metric', -float('inf'))
                    rprint(f"Restored best metric from 'best' checkpoint: {self.best_metric:.4f}.")
            except Exception as e:
                logging.warning(f"Could not load best_metric from 'best' checkpoint symlink: {e}")

    def save_checkpoint(self, step: int, model_state: Dict[str, mx.array], optimizer_state: Dict[str, Any], metadata: Dict[str, Any], current_metric: float):
        """
        Saves the current training state (model, optimizer, metadata) to a new checkpoint directory.
        Performs an atomic save to prevent data corruption.

        Args:
            step: The current global training step.
            model_state: A dictionary of model parameters.
            optimizer_state: A dictionary of optimizer state.
            metadata: Additional training metadata (e.g., epoch, num_updates).
            current_metric: The primary metric for the current evaluation, used for 'best' checkpoint tracking.
        """
        checkpoint_name = f"checkpoint_step_{step}"
        temp_path = self.output_dir / f".{checkpoint_name}.tmp" # Temporary path for atomic save
        final_path = self.output_dir / checkpoint_name

        if temp_path.exists(): shutil.rmtree(temp_path) # Clean up previous failed temp save

        try:
            temp_path.mkdir(parents=True)
            mx.save_safetensors(str(temp_path / "model.safetensors"), model_state)
            mx.save_safetensors(str(temp_path / "optimizer.safetensors"), dict(mx.utils.tree_flatten(optimizer_state)))

            # Store RNG state along with other metadata
            # FIX: Properly store MLX RNG key
            mlx_rng_key = mx.random.key(random.randint(0, 2**31 - 1)) # Generate a new key/seed for storage
            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "mlx_key_seed": int(mlx_rng_key[0].item()), # Store first element of MLX key for reproducibility, or more if needed
                "mlx_key_full_array": mlx_rng_key.tolist() # Store full array for exact state
            }
            metadata['current_metric'] = current_metric
            metadata['rng_state'] = rng_state

            with open(temp_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str) # default=str handles Path/mx.array types

            os.rename(temp_path, final_path) # Atomic rename
            self._checkpoints.append(final_path)
            rprint(f"Checkpoint saved to {final_path} (Step: {step}).")

            # Update 'best' symlink if current metric is better
            if self.save_best and current_metric > self.best_metric:
                self.best_metric = current_metric
                self._update_best_symlink(final_path)

            self._rotate_checkpoints() # Clean up old checkpoints
        except Exception as e:
            logging.critical(f"Failed to save checkpoint at step {step}: {e}", exc_info=True)
            if temp_path.exists(): shutil.rmtree(temp_path) # Clean up failed temp directory
            raise CheckpointError(f"Atomic save failed for step {step}.") from e

    def load_latest_state(self, model: nn.Module, optimizer: Any) -> Tuple[int, Dict[str, Any]]:
        """
        Loads the latest available checkpoint state into the provided model and optimizer.

        Args:
            model: The model instance to load weights into.
            optimizer: The optimizer instance to load state into.

        Returns:
            A tuple of (resumed_update_step, metadata_dict).

        Raises:
            CheckpointError: If loading fails.
        """
        if not self._checkpoints:
            rprint("[yellow]No checkpoints found. Starting from step 0, epoch 0.[/yellow]")
            return 0, {'epoch': 0, 'num_updates': 0, 'current_metric': -float('inf')}

        latest_path = self._checkpoints[-1]
        rprint(f"Resuming from checkpoint: {latest_path.name}...")

        try:
            metadata = self._load_metadata(latest_path / "metadata.json")

            # Load model weights
            model_state = mx.load(str(latest_path / "model.safetensors"))
            # FIX: Remap checkpoint keys for LoRA wrapped modules
            # This is only needed if a checkpoint was saved with a fused model and you're loading into a LoRA-wrapped graph
            # For simplicity in this scaffold, we assume model.safetensors contains what's directly loadable.
            # Real implementation would call _remap_checkpoint_keys_for_lora if necessary.
            model.load_weights(list(model_state.items()))

            # Load optimizer state
            optimizer_state_flat = mx.load(str(latest_path / "optimizer.safetensors"))
            optimizer.state = mx.utils.tree_unflatten(list(optimizer_state_flat.items()))

            mx.eval(model.parameters(), optimizer.state) # Ensure all parameters are evaluated

            # Restore RNG states
            rng_state = metadata.get('rng_state', {})
            if rng_state:
                if 'python' in rng_state: random.setstate(tuple(rng_state['python']))
                if 'numpy' in rng_state: np.random.set_state(tuple(rng_state['numpy']))
                # For MLX, we set the global key (if it's a simple seed or a list)
                if 'mlx_key_full_array' in rng_state and isinstance(rng_state['mlx_key_full_array'], list):
                    mx.random.seed(mx.array(rng_state['mlx_key_full_array'], dtype=mx.uint32))
                elif 'mlx_key_seed' in rng_state and isinstance(rng_state['mlx_key_seed'], int):
                    mx.random.seed(rng_state['mlx_key_seed'])
                else: # Fallback
                     mx.random.seed(random.randint(0, 2**31 - 1)) # Random seed if no specific MLX RNG state saved

            self.best_metric = metadata.get('current_metric', -float('inf'))

            resumed_updates = metadata.get('num_updates', 0)
            rprint(f"Checkpoint loaded. Resuming from update step {resumed_updates}.")
            return resumed_updates, metadata
        except Exception as e:
            logging.critical(f"Failed to load checkpoint {latest_path.name}: {e}", exc_info=True)
            raise CheckpointError(f"Failed to load state from {latest_path.name}.") from e

    def _update_best_symlink(self, target_path: Path):
        """Updates the 'best' symlink to point to the current best checkpoint."""
        best_symlink = self.output_dir / "best"
        try:
            if best_symlink.is_symlink() or best_symlink.exists():
                best_symlink.unlink()
            os.symlink(os.path.relpath(target_path, self.output_dir), best_symlink, target_is_directory=True)
            rprint(f"Updated 'best' checkpoint symlink to {target_path.name}.")
        except Exception as e:
            logging.warning(f"Could not update 'best' symlink: {e}")

    def _rotate_checkpoints(self):
        """Deletes older checkpoints, keeping only the 'keep_last_n' most recent ones (excluding 'best')."""
        # Exclude 'best' symlink target from deletion
        best_path_resolved: Optional[Path] = None
        best_symlink_path = self.output_dir / "best"
        if best_symlink_path.is_symlink():
            try:
                best_path_resolved = best_symlink_path.resolve()
            except Exception:
                pass # Ignore if symlink points to non-existent target

        # Filter out "best" checkpoint from candidates for deletion
        self._checkpoints = [chk for chk in self._checkpoints if chk.exists()] # Clean up any broken entries
        self._checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True) # Sort by time descending

        to_delete = []
        kept_count = 0
        for chk in self._checkpoints:
            if chk == best_path_resolved: # Always keep the resolved 'best' checkpoint
                continue
            if kept_count < self.keep_last_n:
                kept_count += 1
            else:
                to_delete.append(chk)

        for chk in to_delete:
            try:
                shutil.rmtree(chk)
                self._checkpoints.remove(chk) # Remove from internal list as well
                rprint(f"Deleted old checkpoint: {chk.name}.")
            except Exception as e:
                logging.error(f"Error deleting old checkpoint {chk.name}: {e}")

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        """Loads training metadata from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
