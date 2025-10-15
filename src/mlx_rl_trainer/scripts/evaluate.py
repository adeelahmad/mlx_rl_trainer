import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
import asyncio

import mlx.core as mx
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install as rich_install

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.exceptions import ModelLoadError
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry

import mlx_rl_trainer.evaluation

rich_install(show_locals=False)
console = Console(stderr=True, force_terminal=True)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="MLX RL Trainer - Evaluation Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        help="Specific benchmark(s) to run (e.g., 'human_eval').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, handlers=[RichHandler(markup=True)], force=True
    )

    output_dir = Path(args.output_dir) / f"eval_{str(uuid.uuid4())[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting evaluation run. Outputting to: {output_dir}")

    config = ExperimentConfig.load_from_yaml(Path(args.config))
    model_manager = ModelManager(config.model)
    model, tokenizer = model_manager.load_model(
        Path(args.checkpoint), "eval_model", is_trainable=False
    )

    evaluators_to_run = [
        e for e in config.evaluation if not args.benchmarks or e.name in args.benchmarks
    ]
    if not evaluators_to_run:
        logger.critical("No valid evaluators configured or specified to run.")
        sys.exit(1)

    all_results = []
    for eval_config in evaluators_to_run:
        try:
            logger.info(
                f"--- Running evaluator: [bold cyan]{eval_config.name}[/bold cyan] ---"
            )
            evaluator = EvaluatorRegistry.create(eval_config.name, eval_config.config)
            dataset = evaluator.load_dataset(
                eval_config.dataset_path, eval_config.dataset_subset, eval_config.split
            )
            metrics = evaluator.evaluate(model, tokenizer, dataset)
            all_results.append(metrics)

            with open(output_dir / f"{evaluator.name}_results.json", "w") as f:
                json.dump(metrics.to_dict(), f, indent=4)
        except Exception as e:
            logger.error(
                f"Error running evaluator '{eval_config.name}': {e}", exc_info=True
            )

    if all_results:
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=4)

        table = Table(title="Evaluation Summary")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")

        for r in all_results:
            for k, v in r.to_dict().items():
                table.add_row(
                    r.task_name,
                    k.split("/")[-1],
                    f"{v:.4f}" if isinstance(v, float) else str(v),
                )
        console.print(table)


if __name__ == "__main__":
    mx.set_default_device(mx.gpu if mx.gpu_available() else mx.cpu)
    asyncio.run(main())
