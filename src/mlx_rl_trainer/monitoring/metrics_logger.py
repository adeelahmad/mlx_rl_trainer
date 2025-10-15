import csv
import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np

try:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PANDAS_AVAILABLE = MPL_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = MPL_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.utils.text_utils import _preview

logger = logging.getLogger(__name__)
wandb_run = None  # Global variable to hold the wandb run object


def _aggressive_memory_cleanup():
    try:
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


class MetricsLogger:
    def __init__(self, config: ExperimentConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.output_dir = config.trainer.output_dir
        self.file_path = self.output_dir / "training_metrics.csv"
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._headers: List[str] = []
        self._lock = threading.Lock()

        try:
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to open metrics CSV: {e}", exc_info=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        flat_metrics = {"step": step}
        for k, v in metrics.items():
            if isinstance(v, (mx.array, np.ndarray)):
                flat_metrics[k] = v.item()
            elif isinstance(v, (int, float, bool, str)) or v is None:
                flat_metrics[k] = v

        if WANDB_AVAILABLE and wandb_run:
            wandb.log(flat_metrics, step=step)

        with self._lock:
            if self._file is None:
                return
            try:
                current_headers = sorted(flat_metrics.keys())
                if self._writer is None or self._headers != current_headers:
                    self._headers = current_headers
                    self._writer = csv.DictWriter(
                        self._file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    if self.file_path.stat().st_size == 0:
                        self._writer.writeheader()
                self._writer.writerow(flat_metrics)
                self._file.flush()
            except Exception as e:
                logger.error(f"Error writing to CSV: {e}", exc_info=True)

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None
                self._writer = None
        _aggressive_memory_cleanup()


def _emit_plots_from_csv(
    csv_path: Path,
    out_dir: Path,
    config: ExperimentConfig,
    run_id: Optional[str] = None,
):
    if not (PANDAS_AVAILABLE and MPL_AVAILABLE) or not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        if df.empty or "step" not in df.columns:
            logger.warning(
                "Metrics CSV is empty or missing 'step' column. Skipping plot generation."
            )
            return

        df = df.sort_values("step").drop_duplicates("step", keep="last")
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        wandb_images = {}
        for col in df.columns:
            if (
                "step" != col
                and "run_id" != col
                and df[col].dtype in ["float64", "int64"]
                and not df[col].isnull().all()
            ):
                fig, ax = plt.subplots()
                ax.plot(df["step"], df[col])
                ax.set_title(col)
                ax.grid(True)
                plot_path = plot_dir / f"{col.replace('/', '_')}.png"
                fig.savefig(plot_path, dpi=100)
                plt.close(fig)
                if WANDB_AVAILABLE and wandb_run:
                    wandb_images[f"charts/{col}"] = wandb.Image(str(plot_path))

        if wandb_images:
            wandb.log(wandb_images)
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)
    finally:
        plt.close("all")
        _aggressive_memory_cleanup()


def _maybe_log_samples(
    config: ExperimentConfig,
    update_idx: int,
    prompts_data: List,
    decoded_responses: List,
    rewards_data: Dict,
    kl_mode: str,
    run_id: str,
    is_invalid_batch: bool,
):
    if config.monitoring.log_samples_every <= 0 or (
        update_idx % config.monitoring.log_samples_every != 0
    ):
        return

    samples = []
    for i in range(min(config.monitoring.max_logged_samples, len(decoded_responses))):
        p_info = prompts_data[i]
        sample_dict = {
            "step": update_idx,
            "prompt": _preview(p_info.get("text", ""))
            if config.monitoring.log_prompts
            else "[PROMPT REDACTED]",
            "generated": _preview(decoded_responses[i]),
            "reference": _preview(p_info.get("ref_answer_str", "")),
            **{f"reward_{k}": v[i] for k, v in rewards_data.items()},
        }
        samples.append(sample_dict)

    if not samples:
        return

    # --- Log to W&B ---
    if WANDB_AVAILABLE and wandb_run:
        try:
            df = pd.DataFrame(samples)
            table = wandb.Table(dataframe=df)
            wandb.log({"samples/generations": table}, step=update_idx)
        except Exception as e:
            logger.error(f"Failed to log samples to W&B: {e}", exc_info=True)

    # --- Log to local JSONL file ---
    # <-- FIX: THIS BLOCK WAS MISSING -->
    try:
        # Determine the log path. Use the configured path or a default.
        log_path = (
            config.monitoring.sample_log_path
            or config.trainer.output_dir / f"samples_debug_rollouts.jsonl"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write samples to JSONL file: {e}", exc_info=True)
