# file_path: mlx_rl_trainer/src/mlx_rl_trainer/scripts/evaluate.py
# revision_no: 001
# goals_of_writing_code_block: A standalone script for model evaluation.
# type_of_code_response: add new code
"""Standalone model evaluation script."""

import argparse
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry

logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MLX RL Trainer Evaluation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    console = Console(stderr=True, force_terminal=True)
    handlers = [
        RichHandler(markup=True, rich_tracebacks=True, console=console, level=log_level)
    ]
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    config = ExperimentConfig.load_from_yaml(Path(args.config))

    model_manager = ModelManager(config.model)
    model, tokenizer = model_manager.load_model(Path(args.model_path), "eval")

    data_manager = DatasetManager(config.data, tokenizer)
    data_manager.load_datasets()

    for eval_config in config.evaluation:
        evaluator = EvaluatorRegistry.create(eval_config.name, eval_config.config)
        metrics = evaluator.evaluate(model, tokenizer, data_manager)
        logger.info(f"Evaluation results for {eval_config.name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
