# /src/mlx_rl_trainer/monitoring/metrics_logger_enhanced.py
# Revision: 001
# Goal: Enhanced metrics logger with comprehensive stats collection
# Type: Enhanced/Replacement Code
# Description: Drop-in replacement for existing metrics_logger.py with more features

import logging
import csv
import json
import threading
import time
import gc
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_rl_trainer.core.config import ExperimentConfig

logger = logging.getLogger(__name__)


class EnhancedMetricsLogger:
    """
    Enhanced metrics logger with comprehensive statistics.

    Features:
    - CSV export
    - JSON export
    - Real-time statistics
    - Memory-efficient storage
    - Automatic cleanup
    - Multiple output formats
    """

    def __init__(self, config: ExperimentConfig, run_id: str):
        """
        Initialize enhanced metrics logger.

        Args:
            config: Experiment configuration
            run_id: Unique run identifier
        """
        self.config = config
        self.run_id = run_id
        self.output_dir = config.trainer.output_dir

        # File paths
        self.csv_path = self.output_dir / "training_metrics.csv"
        self.json_path = self.output_dir / "training_metrics.json"
        self.summary_path = self.output_dir / "metrics_summary.json"

        # CSV file handle
        self._csv_file = None
        self._csv_writer = None
        self._headers = []

        # JSON storage
        self._json_data: List[Dict[str, Any]] = []

        # Threading
        self._lock = threading.Lock()

        # Statistics
        self._write_count = 0
        self._error_count = 0
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 100

        # Initialize
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_csv_file()

        logger.info(f"Initialized EnhancedMetricsLogger: {self.output_dir}")

    def _open_csv_file(self) -> None:
        """Open CSV file for writing."""
        try:
            self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
            logger.info(f"Opened CSV file: {self.csv_path}")
        except OSError as e:
            logger.error(f"Failed to open CSV file: {e}")
            self._csv_file = None

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log metrics to all outputs.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        if not self._csv_file or self._csv_file.closed:
            return

        # Prepare data
        data = {"step": step, "run_id": self.run_id, "timestamp": time.time()}

        # Convert metrics
        for key, value in metrics.items():
            try:
                if (
                    isinstance(value, (mx.array, np.ndarray))
                    if MLX_AVAILABLE
                    else isinstance(value, np.ndarray)
                ):
                    if value.size == 1:
                        data[key] = float(value.item())
                    else:
                        data[key] = float(np.mean(value))
                elif isinstance(value, (int, float, bool, str)) or value is None:
                    data[key] = value
                elif isinstance(value, (list, tuple)):
                    data[f"{key}_count"] = len(value)
                    if value and isinstance(value[0], (int, float)):
                        data[f"{key}_mean"] = np.mean(value)
                else:
                    data[key] = str(value)
            except Exception as e:
                logger.warning(f"Failed to convert metric '{key}': {e}")
                data[key] = "conversion_error"

        # Write to CSV
        with self._lock:
            try:
                # Update CSV writer if needed
                fieldnames = sorted(data.keys())
                if self._csv_writer is None or self._headers != fieldnames:
                    needs_header = (
                        not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                    )
                    self._headers = fieldnames
                    self._csv_writer = csv.DictWriter(
                        self._csv_file, fieldnames=self._headers, extrasaction="ignore"
                    )
                    if needs_header:
                        self._csv_writer.writeheader()

                # Write row
                self._csv_writer.writerow(data)
                self._csv_file.flush()
                self._write_count += 1

                # Store in JSON
                self._json_data.append(data)

                # Periodic cleanup
                current_time = time.time()
                if (
                    self._write_count % self._cleanup_interval == 0
                    or current_time - self._last_cleanup_time > 60
                ):
                    self._cleanup()
                    self._last_cleanup_time = current_time

            except Exception as e:
                self._error_count += 1
                logger.error(f"Error writing metrics: {e}")

                if self._error_count < 3:
                    try:
                        self._csv_writer = None
                        logger.info("Attempting to recreate CSV writer...")
                    except:
                        pass

    def _cleanup(self) -> None:
        """Perform memory cleanup."""
        gc.collect()

        if MLX_AVAILABLE:
            try:
                mx.clear_cache()
            except:
                pass

    def export_json(self, filepath: Optional[Path] = None) -> None:
        """Export metrics to JSON."""
        if filepath is None:
            filepath = self.json_path

        with self._lock:
            with open(filepath, "w") as f:
                json.dump(self._json_data, f, indent=2)

        logger.info(f"Exported metrics to JSON: {filepath}")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self._json_data:
            return {}

        with self._lock:
            # Convert to DataFrame for easy analysis
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(self._json_data)

                summary = {"total_steps": len(df), "metrics": {}}

                # Calculate statistics for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ["step", "timestamp"]:
                        summary["metrics"][col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "final": float(df[col].iloc[-1]),
                        }

                return summary
            else:
                # Basic summary without pandas
                return {
                    "total_steps": len(self._json_data),
                    "write_count": self._write_count,
                    "error_count": self._error_count,
                }

    def export_summary(self, filepath: Optional[Path] = None) -> None:
        """Export summary statistics to JSON."""
        if filepath is None:
            filepath = self.summary_path

        summary = self.generate_summary()

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported summary: {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            "write_count": self._write_count,
            "error_count": self._error_count,
            "file_size_mb": self.csv_path.stat().st_size / 1048576
            if self.csv_path.exists()
            else 0,
            "json_entries": len(self._json_data),
            "last_cleanup_time": self._last_cleanup_time,
        }

    def close(self) -> None:
        """Close logger and save final data."""
        with self._lock:
            # Close CSV file
            if self._csv_file and not self._csv_file.closed:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                    logger.info(f"Closed CSV file. Total writes: {self._write_count}")
                except Exception as e:
                    logger.error(f"Error closing CSV file: {e}")
                finally:
                    self._csv_file = None
                    self._csv_writer = None

            # Export JSON
            if self._json_data:
                self.export_json()
                self.export_summary()

        # Final cleanup
        self._cleanup()


# Dependencies: pandas (optional), numpy, mlx
# Install: pip install pandas numpy mlx
# Usage: logger = EnhancedMetricsLogger(config, run_id="run_001")
#        logger.log_metrics({'loss': 0.5}, step=100)
# Status: Complete and commit-ready
