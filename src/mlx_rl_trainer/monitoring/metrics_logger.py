"""Handles logging to CSV, NDJSON, and Weights & Biases."""
import logging, csv, json, threading, time, gc
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
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

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.utils.text_utils import _preview, _extract_think_answer_lengths

logger = logging.getLogger(__name__)
wandb_run: Any = None


def _aggressive_memory_cleanup():
    """Aggressively free memory."""
    try:
        mx.metal.clear_cache()
    except:
        pass
    mx.clear_cache()
    gc.collect()


def _calculate_mcq_accuracy(
    refs: Optional[List[str]],
    gens: Optional[List[str]],
    is_mcq: Optional[List[bool]],
    k: int,
) -> float:
    if not all((refs, gens, is_mcq)) or k == 0:
        return 0.0
    correct, total = 0, 0
    for i in range(k):
        if is_mcq[i]:
            total += 1
            if refs[i] == gens[i] and refs[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


class MetricsLogger:
    def __init__(self, config: ExperimentConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.output_dir = config.trainer.output_dir
        self.file_path = self.output_dir / f"training_metrics.csv"
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._headers: List[str] = []
        self._lock = threading.Lock()
        self._write_count = 0
        self._cleanup_interval = 100  # Cleanup every N writes

        try:
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to open metrics CSV: {e}", exc_info=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if not self._file or self._file.closed:
            return

        loggable: Dict[str, Any] = {"update_step": step, "run_id": self.run_id}

        # Convert metrics to loggable format
        for k, v in metrics.items():
            if isinstance(v, (mx.array, np.ndarray)):
                loggable[k] = v.item() if v.size == 1 else str(v.tolist())
            elif isinstance(v, (int, float, bool, str)) or v is None:
                loggable[k] = v
            else:
                loggable[k] = str(v)

        with self._lock:
            try:
                current_headers = sorted(loggable.keys())

                # Recreate writer if headers changed
                if self._writer is None or self._headers != current_headers:
                    is_empty = (
                        not self.file_path.exists()
                        or self.file_path.stat().st_size == 0
                    )
                    self._headers = current_headers
                    self._writer = csv.DictWriter(
                        self._file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    if is_empty:
                        self._writer.writeheader()

                # Write the row
                self._writer.writerow(loggable)
                self._file.flush()

                # Increment write counter
                self._write_count += 1

                # Periodic memory cleanup
                if self._write_count % self._cleanup_interval == 0:
                    _aggressive_memory_cleanup()

            except Exception as e:
                logger.error(f"Error writing metrics CSV: {e}", exc_info=True)

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()
                self._file.close()
                self._file = None
                self._writer = None
        # Final cleanup
        _aggressive_memory_cleanup()


def _emit_plots_from_csv(
    csv_path: Path, out_dir: Path, config: ExperimentConfig = None, run_id = None
):
    """
    Generate plots from CSV metrics file.
    Memory optimized: Uses chunked processing and explicit cleanup.
    """
    if (
        not (PANDAS_AVAILABLE and MPL_AVAILABLE)
        or not csv_path.exists()
        or csv_path.stat().st_size < 100
    ):
        return

    try:
        # Read CSV with error handling for corrupted lines
        df = pd.read_csv(csv_path, on_bad_lines='skip')

        if df.empty:
            del df
            return

        # Remove duplicate update_steps, keeping the last entry
        x_col = "update_step"
        if x_col in df.columns:
            # Keep the last entry (most recent log) for each unique 'update_step' value
            df = df.drop_duplicates(subset=[x_col], keep='last')

            # Sort by update_step for proper plotting
            df = df.sort_values(by=x_col).reset_index(drop=True)

        # Define all meaningful plot columns (Y-axes)
        plot_metrics = {
            "train/loss": "loss",
            "train/reward_mean": "reward_mean",
            "train/rewards/raw_total": "reward_total",
            "train/learning_rate": "lr",
            "train/grad_norm": "grad_norm",
            "train/kl_divergence": "kl_divergence",
            "train/rewards/raw_TagStructureReward": "reward_TagStructure",
            "train/rewards/raw_SemanticSimilarityReward": "reward_SemanticSimilarity",
            "train/rewards/raw_CodeExecutionReward": "reward_CodeExecution",
        }

        # Use run_id if available, otherwise use a generic 'plots' directory structure
        plots_dir = out_dir / "plots"
        if run_id:
            plots_dir = plots_dir / run_id

        plots_dir.mkdir(exist_ok=True, parents=True)

        def _plot(y_col: str, fname_suffix: str, x_col: str = "update_step"):
            """Generate a single plot with memory cleanup."""
            if y_col not in df.columns or x_col not in df.columns:
                return

            try:
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))

                # Extract data as numpy arrays for efficient plotting
                x_data = df[x_col].values
                y_data = df[y_col].values

                # Plot
                ax.plot(x_data, y_data)

                # Formatting
                x_label = x_col.replace("_", " ").title()
                y_label = y_col.replace("_", " ").title()

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"{y_label} vs {x_label}")
                ax.grid(True, alpha=0.5)

                fig.tight_layout()

                # Use safe filename
                safe_y_col = y_col.replace('/', '_').replace('.', '_')
                plot_path = plots_dir / f"{safe_y_col}_{fname_suffix}.png"

                # Save and close
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Clean up references
                del fig, ax, x_data, y_data

            except Exception as e:
                logger.warning(f"Failed to plot {y_col}: {e}")
                # Ensure figure is closed even on error
                plt.close('all')

        # Generate each plot
        for col, name in plot_metrics.items():
            _plot(col, name)
            # Periodic cleanup after every few plots
            gc.collect()

        # Final cleanup
        del df
        plt.close('all')
        _aggressive_memory_cleanup()

        logger.info(f"Plots generated in: {plots_dir}")

    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)
        # Ensure all figures are closed on error
        plt.close('all')
        _aggressive_memory_cleanup()


def _maybe_log_samples(
    config: ExperimentConfig,
    update_idx: int,
    prompts_data: List[Dict],
    decoded_responses: List[str],
    rewards_data: Dict,
    kl_mode: str,
    run_id: str,
    is_invalid_batch: bool,
):
    """
    Log sample generations to JSONL file.
    Memory optimized: Process samples one at a time and cleanup.
    """
    if (
        config.monitoring.log_samples_every <= 0
        or update_idx % config.monitoring.log_samples_every != 0
    ):
        return

    try:
        global wandb_run
        out_path = (
            config.monitoring.sample_log_path
            or config.trainer.output_dir / f"samples_debug.jsonl"
        )
        k = min(config.monitoring.max_logged_samples, len(decoded_responses))

        # Use context manager for automatic file closing
        with open(out_path, "a", encoding="utf-8") as f:
            for i in range(k):
                p_idx = i // config.trainer.num_rollout_samples
                if p_idx >= len(prompts_data):
                    continue

                original_sample = prompts_data[p_idx]
                gen_text = decoded_responses[i]

                # Construct reference text
                ref_dict = original_sample.get('ref', {
                    "completion": f"{config.generation.think_start_tag}\n{original_sample.get('ref_think_str','')}{config.generation.think_end_tag}\n{original_sample.get('ref_answer_str','')}"
                })
                ref_text = ref_dict.get("completion", "") if isinstance(ref_dict, dict) else str(ref_dict)

                # Extract lengths
                gen_think_len, gen_ans_len = _extract_think_answer_lengths(
                    gen_text, config.generation
                )
                ref_think_len, ref_ans_len = _extract_think_answer_lengths(
                    ref_text, config.generation
                )

                # Build entry
                entry = {
                    "update": update_idx,
                    "is_invalid_batch": is_invalid_batch,
                    "kl_mode": kl_mode,
                    "prompt": _preview(original_sample.get("text", ""), 1200)
                    if config.monitoring.log_prompts
                    else "[REDACTED]",
                    "generated": _preview(gen_text, 1200),
                    "reference": _preview(ref_text, 1200),
                    "reward_total": rewards_data["total"][i],
                    "gen_think_len": gen_think_len,
                    "gen_ans_len": gen_ans_len,
                    "ref_think_len": ref_think_len,
                    "ref_ans_len": ref_ans_len,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Add individual reward components
                for r_name, r_vals in rewards_data.items():
                    if r_name != "total":
                        entry[f"reward_{r_name}"] = r_vals[i]

                # Write entry
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

                # Clean up entry to free memory
                del entry

        # Periodic cleanup after logging samples
        _aggressive_memory_cleanup()

    except Exception as e:
        logger.error(f"Sample NDJSON logging failed: {e}", exc_info=True)
