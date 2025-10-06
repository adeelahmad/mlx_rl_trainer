# file_path: mlx_rl_trainer/src/mlx_rl_trainer/monitoring/metrics_logger.py
# revision_no: 002
# goals_of_writing_code_block: Handles logging to CSV, NDJSON, and Weights & Biases, with plotting.
# type_of_code_response: change existing
"""Handles logging to CSV, NDJSON, and Weights & Biases."""

import logging
import csv
import json
import os
import re
import time
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import mlx.core as mx
import numpy as np

# Conditional imports
try:
    import pandas as pd
    import matplotlib.pyplot as plt

    PANDAS_AVAILABLE = True
    MPL_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    MPL_AVAILABLE = False

# Import relevant structures from core.config and rewards.context
from mlx_rl_trainer.core.config import ExperimentConfig  # Use ExperimentConfig
from mlx_rl_trainer.rewards.context import RewardContext

# Import text utilities
from mlx_rl_trainer.utils.text_utils import (
    _preview,
    _extract_final_numeric,
    _normalize_ans_for_match,
    extract_think_region,
    extract_answer_region,
    _count_words,
    _letters_to_canonical,
    _indices_to_letters,
)

logger = logging.getLogger(__name__)

def _calculate_mcq_accuracy(
    ref_letters_list: Optional[List[str]],
    gen_letters_list: Optional[List[str]],
    is_mcq_list: Optional[List[bool]],
    k_actual: int,
) -> float:
    """Calculates MCQ accuracy."""
    if not ref_letters_list or not gen_letters_list or not is_mcq_list or k_actual == 0:
        return 0.0

    correct_count = 0
    total_mcq = 0
    for i in range(k_actual):
        if is_mcq_list[i]:
            total_mcq += 1
            if ref_letters_list[i] == gen_letters_list[i]:
                correct_count += 1

    return correct_count / total_mcq if total_mcq > 0 else 0.0

# --- Global W&B Run (Managed externally, but referenced here) ---
wandb_run: Any = None  # Set by main script

# --- Metrics Logger Class (CSV) ---


class MetricsLogger:
    """
    Thread-safe utility for appending training metrics to a CSV file.
    Also provides functionality to generate plots from the logged data.
    """

    def __init__(self, config: ExperimentConfig, run_id: str):
        """
        Initializes the MetricsLogger.

        Args:
            config: The experiment configuration.
            run_id: A unique ID for the current training run.
        """
        self.config = config
        self.run_id = run_id
        self.output_dir = config.trainer.output_dir
        self.file_path = self.output_dir / f"training_metrics_{run_id}.csv"
        self._file: Optional[Any] = None
        self._writer: Optional[csv.DictWriter] = None
        self._headers: List[str] = []
        self._lock = threading.Lock()
        self._logged_file_closed_warning = False

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
            logger.info(f"Metrics CSV logger opened: {self.file_path}")
        except OSError as e:
            logger.error(
                f"Failed to open metrics CSV at {self.file_path}: {e}", exc_info=True
            )
            self._file = None

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Logs a dictionary of metrics for a given training step to the CSV file.

        Args:
            metrics: A dictionary of metrics (e.g., loss, reward, accuracy).
            step: The current training update step.
        """
        if self._file is None or self._file.closed:
            if not self._logged_file_closed_warning:
                logger.warning(
                    f"Metrics file '{self.file_path.name}' is closed. Cannot log metrics."
                )
                self._logged_file_closed_warning = True
            return

        loggable: Dict[str, Union[str, int, float, bool]] = {
            "update_step": step,
            "run_id": self.run_id,
        }

        # Convert MLX arrays/NumPy objects to standard Python types for CSV compatibility
        for key, value in metrics.items():
            if isinstance(value, (mx.array, np.ndarray)):
                try:
                    if value.size == 1:
                        item = value.item()
                        loggable[key] = (
                            item if isinstance(item, (int, float, bool)) else str(item)
                        )
                    else:
                        loggable[key] = str(value.tolist())
                except Exception as e:
                    logger.warning(f"Array conversion error for metric '{key}': {e}")
                    loggable[key] = f"[Array conv error: {e}]"
            elif isinstance(value, (int, float, bool, str)):
                loggable[key] = value
            elif value is None:
                loggable[key] = ""
            else:
                try:
                    loggable[key] = str(value)
                except Exception as e:
                    logger.warning(f"Metric conversion error for '{key}': {e}")
                    loggable[key] = f"[str conv error: {e}]"

        if not loggable:
            return

        with self._lock:
            try:
                # Ensure headers are correct, dynamically add if new metrics appear
                current_headers = ["update_step", "run_id"] + sorted(
                    [k for k in loggable.keys() if k not in ("update_step", "run_id")]
                )

                if self._writer is None or self._headers != current_headers:
                    is_empty = (
                        not self.file_path.exists()
                        or self.file_path.stat().st_size == 0
                    )
                    self._headers = current_headers
                    self._writer = csv.DictWriter(
                        self._file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    if is_empty:  # Write header only if file was empty
                        self._writer.writeheader()

                self._writer.writerow(loggable)
                self._file.flush()  # Ensure data is written to disk immediately
            except Exception as e:
                logger.error(
                    f"Error writing metrics CSV to {self.file_path}: {e}", exc_info=True
                )

    def close(self):
        """Closes the CSV file, ensuring all buffered data is written."""
        with self._lock:
            if self._file and not self._file.closed:
                try:
                    self._file.flush()
                    self._file.close()
                    self._file = None
                    self._writer = None
                    self._headers = []
                    logger.info(f"Metrics CSV logger closed: {self.file_path}")
                except Exception as e:
                    logger.error(
                        f"Error closing metrics CSV file {self.file_path}: {e}"
                    )

    def emit_plots_from_csv(self):
        """Generates basic training plots from the metrics CSV using Pandas/Matplotlib."""
        _emit_plots_from_csv(self.file_path, self.output_dir)


# --- Plotting Utility (moved out of class for clarity) ---


def _emit_plots_from_csv(csv_path: Path, out_dir: Path):
    """
    Generates basic training plots from a CSV metrics file.
    Requires `pandas` and `matplotlib` to be installed.

    Args:
        csv_path: Path to the CSV file containing training metrics.
        out_dir: Directory where the generated plots will be saved.
    """
    if (
        not csv_path.exists() or csv_path.stat().st_size < 100
    ):  # Ensure file exists and has some content
        logger.info("No substantial training_metrics.csv found; skipping plots.")
        return

    if not (PANDAS_AVAILABLE and MPL_AVAILABLE):
        logger.info("pandas/matplotlib not available; skipping plot generation.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(
            f"Plot generation skipped: could not read CSV at {csv_path} ({e}).",
            exc_info=True,
        )
        return

    if df.empty:
        logger.info("Metrics DataFrame is empty; skipping plot generation.")
        return

    # Normalize step column name for consistent x-axis
    if "update_step" not in df.columns:
        if "num_updates" in df.columns:
            df["update_step"] = df["num_updates"]
        elif "step" in df.columns:
            df["update_step"] = df["step"]
        else:
            df["update_step"] = np.arange(len(df), dtype=int)  # Fallback to row index

    try:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating plots in: {plots_dir}")

        def _plot(y_col: str, fname_suffix: str, x_col: str = "update_step"):
            """Helper to create and save a single plot."""
            if y_col in df.columns and x_col in df.columns and not df.empty:
                plt.figure(figsize=(10, 6))  # Standard figure size
                plt.plot(df[x_col].values, df[y_col].values)
                plt.xlabel(x_col.replace("_", " ").title())  # Nicer labels
                plt.ylabel(y_col.replace("_", " ").title())
                plt.title(
                    f"{y_col.replace('_', ' ').title()} over {x_col.replace('_', ' ').title()}"
                )
                plt.grid(True, linestyle="--", alpha=0.7)  # Add grid for readability
                plt.tight_layout()
                plt.savefig(
                    plots_dir
                    / f"{y_col.replace('/', '_').replace('train_', '')}_{fname_suffix}"
                )  # Clean filename
                plt.close()  # Close figure to free memory

        # Plot core training metrics
        _plot("train/loss", "loss_vs_updates.png")
        _plot(
            "train/reward_raw_mean", "reward_mean_vs_updates.png"
        )  # Use raw_mean for consistency
        _plot("train/learning_rate", "lr_vs_updates.png")
        _plot("train/grad_norm", "grad_norm_vs_updates.png")
        _plot(
            "train/reward_rolling_avg", "reward_rolling_avg_vs_updates.png"
        )  # Plot rolling avg

        # Plot evaluation/benchmark metrics if present
        for col in df.columns:
            if col.startswith("eval/") or col.startswith("bench/"):
                _plot(col, f"{col.replace('/', '_')}_vs_updates.png")

        logger.info(f"Plots written successfully to: {plots_dir}")

    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)


# --- Sample Logging (NDJSON) ---


def _maybe_log_samples(
    config: ExperimentConfig,
    update_idx: int,
    prompts_data: List[Dict[str, Any]],
    decoded_responses: List[str],
    rewards_total: List[float],
    rewards_fmt: List[float],
    rewards_cont: List[float],
    prompt_token_lens: List[int],
    response_token_lens: List[int],
    kl_mode: str,
    *,
    run_id: str,
    is_invalid_batch: bool,
    ref_letters_list: Optional[List[str]] = None,
    gen_letters_list: Optional[List[str]] = None,
    is_mcq_list: Optional[List[bool]] = None,
):
    """
    Logs generated samples and summary statistics to an NDJSON file and optionally to Weights & Biases.

    Args:
        config: The ExperimentConfig object.
        update_idx: Current training update step.
        prompts_data: List of dictionaries of original prompt data.
        decoded_responses: List of generated text responses.
        rewards_total: List of total rewards for each generated response.
        rewards_fmt: List of format rewards for each generated response.
        rewards_cont: List of content rewards for each generated response.
        prompt_token_lens: List of token lengths of the input prompts.
        response_token_lens: List of token lengths of the generated responses.
        kl_mode: String indicating the KL divergence calculation mode.
        run_id: Unique ID for the current training run.
        is_invalid_batch: Flag if the current batch contains invalid samples.
        ref_letters_list: List of correct MCQ answer letters (if applicable).
        gen_letters_list: List of generated MCQ answer letters (if applicable).
        is_mcq_list: List of boolean flags indicating if a sample is an MCQ.
    """

    if (
        config.trainer.log_samples_every <= 0
        or update_idx % config.trainer.log_samples_every != 0
    ):
        return

    try:
        global wandb_run
        out_path = (
            Path(config.trainer.sample_log_path)
            if config.trainer.sample_log_path
            else config.trainer.output_dir / f"samples_debug_{run_id}.jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        k = min(config.trainer.max_logged_samples, len(decoded_responses))

        all_gen_think_lens, all_gen_answer_lens = [], []
        all_ref_think_lens, all_ref_answer_lens = [], []
        all_think_length_ratios, all_answer_length_ratios = [], []

        # Dynamically retrieve RewardConfig's tags from ExperimentConfig (assuming format_structure is configured)
        reward_tags_config = next(
            (r for r in config.rewards if r.name == "format_structure"), None
        )
        if reward_tags_config:
            think_start_tag = getattr(reward_tags_config, "think_open_tag", "<think>")
            think_end_tag = getattr(reward_tags_config, "think_close_tag", "</think>")
            answer_start_tag = getattr(
                reward_tags_config, "answer_open_tag", "<answer>"
            )
            answer_end_tag = getattr(
                reward_tags_config, "answer_close_tag", "</answer>"
            )
        else:  # Fallback to hardcoded defaults if format_structure reward is not in config
            think_start_tag, think_end_tag = "<think>", "</think>"
            answer_start_tag, answer_end_tag = "<answer>", "</answer>"

        with open(out_path, "a", encoding="utf-8") as f:
            for i in range(k):
                p_idx = i // max(1, config.trainer.num_rollout_samples)
                if p_idx >= len(prompts_data):
                    continue

                original_sample = prompts_data[p_idx]
                generated_text = decoded_responses[i]

                # Reconstruct full reference text for consistent region extraction
                ref_think_text = original_sample.get("ref_think_str") or ""
                ref_answer_only_text = original_sample.get("ref_answer_str") or ""

                # Use configured tags for reconstructing reference string
                ref_answer_text = f"{think_start_tag}{ref_think_text}{think_end_tag}\n{answer_start_tag}{ref_answer_only_text}{answer_end_tag}"

                # Extract think/answer lengths from generated and reference texts
                gen_think_len, gen_answer_len = _extract_think_answer_lengths(
                    generated_text,
                    think_start_tag,
                    think_end_tag,
                    answer_start_tag,
                    answer_end_tag,
                )
                ref_think_len, ref_answer_len = _extract_think_answer_lengths(
                    ref_answer_text,
                    think_start_tag,
                    think_end_tag,
                    answer_start_tag,
                    answer_end_tag,
                )

                # Calculate ratios, handling potential division by zero
                think_ratio = (
                    float(gen_think_len) / float(max(ref_think_len, 1))
                    if ref_think_len > 0
                    else (1.0 if gen_think_len == 0 else float("inf"))
                )
                answer_ratio = (
                    float(gen_answer_len) / float(max(ref_answer_len, 1))
                    if ref_answer_len > 0
                    else (1.0 if gen_answer_len == 0 else float("inf"))
                )

                all_gen_think_lens.append(gen_think_len)
                all_gen_answer_lens.append(gen_answer_len)
                all_ref_think_lens.append(ref_think_len)
                all_ref_answer_lens.append(ref_answer_len)

                if think_ratio != float("inf"):
                    all_think_length_ratios.append(think_ratio)
                if answer_ratio != float("inf"):
                    all_answer_length_ratios.append(answer_ratio)

                entry = {
                    "run_id": run_id,
                    "update": update_idx,
                    "is_invalid_batch": is_invalid_batch,
                    "invalid_sample_in_source": original_sample.get(
                        "is_invalid_sample", False
                    ),
                    "kl_mode": kl_mode,
                    "prompt_preview": _preview(original_sample.get("text", ""), 600)
                    if config.trainer.log_prompts
                    else "[PROMPT REDACTED]",
                    "generated_preview": _preview(generated_text, 600),
                    "reward_total": float(rewards_total[i]),
                    "reward_format": float(rewards_fmt[i]),
                    "reward_content": float(rewards_cont[i]),
                    "prompt_tokens": int(prompt_token_lens[p_idx])
                    if p_idx < len(prompt_token_lens)
                    else None,
                    "response_tokens": int(response_token_lens[i])
                    if i < len(response_token_lens)
                    else None,
                    "gen_think_length": gen_think_len,
                    "gen_answer_length": gen_answer_len,
                    "ref_think_length": ref_think_len,
                    "ref_answer_length": ref_answer_len,
                    "think_length_ratio": think_ratio
                    if think_ratio != float("inf")
                    else None,
                    "answer_length_ratio": answer_ratio
                    if answer_ratio != float("inf")
                    else None,
                    "ref_answer_preview": _preview(ref_answer_text, 300),
                    "mcq_ref_letter": ref_letters_list[i]
                    if ref_letters_list and i < len(ref_letters_list)
                    else "",
                    "mcq_gen_letter": gen_letters_list[i]
                    if gen_letters_list and i < len(gen_letters_list)
                    else "",
                    "is_mcq": bool(is_mcq_list[i])
                    if is_mcq_list and i < len(is_mcq_list)
                    else original_sample.get("is_mcq", False),
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        # W&B Logging Summary (if enabled)
        if wandb_run is not None and config.monitoring.use_wandb:
            try:
                k_actual = len(rewards_total)
                if k_actual == 0:
                    return

                wandb_metrics = {
                    "samples/reward_total_mean": float(
                        np.mean(rewards_total[:k_actual])
                    ),
                    "samples/reward_total_std": float(np.std(rewards_total[:k_actual])),
                    "samples/reward_format_mean": float(
                        np.mean(rewards_fmt[:k_actual])
                    ),
                    "samples/reward_content_mean": float(
                        np.mean(rewards_cont[:k_actual])
                    ),
                    "samples/gen_think_length_mean": float(np.mean(all_gen_think_lens))
                    if all_gen_think_lens
                    else 0,
                    "samples/gen_think_length_std": float(np.std(all_gen_think_lens))
                    if all_gen_think_lens
                    else 0,
                    "samples/gen_think_length_min": int(np.min(all_gen_think_lens))
                    if all_gen_think_lens
                    else 0,
                    "samples/gen_think_length_max": int(np.max(all_gen_think_lens))
                    if all_gen_think_lens
                    else 0,
                    "samples/ref_think_length_mean": float(np.mean(all_ref_think_lens))
                    if all_ref_think_lens
                    else 0,
                    "samples/ref_answer_length_mean": float(
                        np.mean(all_ref_answer_lens)
                    )
                    if all_ref_answer_lens
                    else 0,
                    "samples/think_length_ratio_mean": float(
                        np.mean(all_think_length_ratios)
                    )
                    if all_think_length_ratios
                    else 0,
                    "samples/answer_length_ratio_mean": float(
                        np.mean(all_answer_length_ratios)
                    )
                    if all_answer_length_ratios
                    else 0,
                    "samples/response_tokens_mean": float(
                        np.mean(response_token_lens[:k_actual])
                    ),
                    "samples/excessive_thinking_rate": float(
                        np.mean(
                            [
                                1.0 if x > config.trainer.think_len_max else 0.0
                                for x in all_gen_think_lens
                            ]
                        )
                    ),
                    "samples/too_brief_thinking_rate": float(
                        np.mean(
                            [
                                1.0 if x < config.trainer.min_think_tokens else 0.0
                                for x in all_gen_think_lens
                            ]
                        )
                    ),
                    "samples/mcq_accuracy": _calculate_mcq_accuracy(
                        ref_letters_list, gen_letters_list, is_mcq_list, k_actual
                    )
                    if ref_letters_list and gen_letters_list
                    else None,
                    "samples/update_idx": update_idx,
                    "samples/kl_mode": kl_mode,
                }

                wandb_metrics = {
                    k: v for (k, v) in wandb_metrics.items() if v is not None
                }
                wandb_run.log(wandb_metrics, step=update_idx)
            except Exception as e:
                logger.warning(
                    f"Failed to log sample summary to wandb: {e}", exc_info=True
                )

    except Exception as e:
        logger.error(f"Sample NDJSON logging failed: {e}", exc_info=True)
