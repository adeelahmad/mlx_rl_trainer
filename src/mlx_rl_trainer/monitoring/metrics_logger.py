"""Handles logging to CSV, NDJSON, and Weights & Biases."""
import logging, csv, json, threading, time
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


def _calculate_mcq_accuracy(refs: Optional[List[str]], gens: Optional[List[str]], is_mcq: Optional[List[bool]], k: int) -> float:
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

        try:
            self._file = open(self.file_path, "a", newline="", encoding="utf-8")
        except OSError as e:
            logger.error(f"Failed to open metrics CSV: {e}", exc_info=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        if not self._file or self._file.closed:
            return
        loggable: Dict[str, Any] = {"update_step": step, "run_id": self.run_id}
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
                if self._writer is None or self._headers != current_headers:
                    is_empty = not self.file_path.exists() or self.file_path.stat().st_size == 0
                    self._headers = current_headers
                    self._writer = csv.DictWriter(self._file, fieldnames=self._headers, extrasaction="ignore")
                    if is_empty:
                        self._writer.writeheader()
                self._writer.writerow(loggable)
                self._file.flush()
            except Exception as e:
                logger.error(f"Error writing metrics CSV: {e}", exc_info=True)

    def close(self):
        with self._lock:
            if self._file and not self._file.closed:
                self._file.close()
                self._file = None


def _emit_plots_from_csv(csv_path: Path, out_dir: Path, config: ExperimentConfig = None):
    if not (PANDAS_AVAILABLE and MPL_AVAILABLE) or not csv_path.exists() or csv_path.stat().st_size < 100:
        return
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        def _plot(y_col: str, fname_suffix: str, x_col: str = "update_step"):
            if y_col in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df[x_col].values, df[y_col].values)
                plt.xlabel(x_col.replace("_", " ").title())
                plt.ylabel(y_col.replace("_", " ").title())
                plt.title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
                plt.grid(True, alpha=0.5)
                plt.tight_layout()
                plt.savefig(plots_dir / f"{y_col.replace('/', '_')}_{fname_suffix}.png")
                plt.close()

        plot_map = {
            "train/loss": "loss",
            "train/reward_raw_mean": "reward",
            "train/learning_rate": "lr",
            "train/grad_norm": "grad_norm",
        }
        for col, name in plot_map.items():
            _plot(col, name)

        logger.info(f"Plots generated in: {plots_dir}")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)


def _maybe_log_samples(config: ExperimentConfig, update_idx: int, prompts_data: List[Dict], decoded_responses: List[str],
                       rewards_data: Dict, kl_mode: str, run_id: str, is_invalid_batch: bool):
    if config.monitoring.log_samples_every <= 0 or update_idx % config.monitoring.log_samples_every != 0:
        return

    try:
        global wandb_run
        out_path = config.monitoring.sample_log_path or config.trainer.output_dir / f"samples_debug_{run_id}.jsonl"
        k = min(config.monitoring.max_logged_samples, len(decoded_responses))

        with open(out_path, "a", encoding="utf-8") as f:
            for i in range(k):
                p_idx = i // config.trainer.num_rollout_samples
                if p_idx >= len(prompts_data):
                    continue

                original_sample = prompts_data[p_idx]
                gen_text = decoded_responses[i]
                ref_text = f"{config.generation.think_start_tag}{original_sample.get('ref_think_str','')}{config.generation.think_end_tag}\n{original_sample.get('ref_answer_str','')}"

                gen_think_len, gen_ans_len = _extract_think_answer_lengths(gen_text, config.generation)
                ref_think_len, ref_ans_len = _extract_think_answer_lengths(ref_text, config.generation)

                entry = {
                    "update": update_idx,
                    "is_invalid_batch": is_invalid_batch,
                    "kl_mode": kl_mode,
                    "prompt": _preview(original_sample.get("text", ""), 600) if config.monitoring.log_prompts else "[REDACTED]",
                    "generated": _preview(gen_text, 600),
                    "reference": _preview(ref_text, 300),
                    "reward_total": rewards_data["total"][i],
                    "gen_think_len": gen_think_len,
                    "gen_ans_len": gen_ans_len,
                    "ref_think_len": ref_think_len,
                    "ref_ans_len": ref_ans_len,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for r_name, r_vals in rewards_data.items():
                    if r_name != "total":
                        entry[f"reward_{r_name}"] = r_vals[i]

                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"Sample NDJSON logging failed: {e}", exc_info=True)
