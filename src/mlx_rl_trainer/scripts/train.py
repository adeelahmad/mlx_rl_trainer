#!/usr/bin/env python
# scripts/train.py - Rev 001
# Goal: Main training entry point
# Type: New script

import sys
import logging
from pathlib import Path
import argparse
import uuid  # For unique run IDs
import random  # For random seeds
import mlx.core as mx  # For default device setting
import signal  # For signal handling
import asyncio # Import asyncio for async operations

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import core components
from mlx_rl_trainer.core.config import ExperimentConfig  # Use new ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager  # Use new DatasetManager
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.core.trainer import (
    TrainingRuntimeError,
    CustomBaseException,
    ModelLoadError,
    DataLoadError,
)  # Exceptions

# Import algorithm-specific trainer
from mlx_rl_trainer.algorithms.grpo.grpo_trainer import GRPOTrainer

# Import reward implementations to trigger registration
import mlx_rl_trainer.rewards.format.tag_structure  # Registers "format_structure"
import mlx_rl_trainer.rewards.content.semantic_similarity  # Registers "content_similarity"
import mlx_rl_trainer.rewards.programming.code_execution  # Registers "code_execution"

# Add other reward imports here
import mlx_rl_trainer.rewards.reasoning.thinking_quality  # Registers "thinking_quality"
import mlx_rl_trainer.rewards.content.mcq_accuracy  # Registers "mcq_accuracy"
import mlx_rl_trainer.rewards.content.steps_coverage  # Registers "steps_coverage"

# Import evaluation implementations to trigger registration
import mlx_rl_trainer.evaluation.programming.human_eval  # Registers "human_eval"
import mlx_rl_trainer.evaluation.reasoning.gsm8k  # Registers "gsm8k"
import mlx_rl_trainer.evaluation.reasoning.arc  # Registers "arc"
import mlx_rl_trainer.evaluation.general.perplexity  # Registers "perplexity"

# Import monitoring components
from mlx_rl_trainer.monitoring.metrics_logger import MetricsLogger  # Use MetricsLogger
from mlx_rl_trainer.rewards.base_reward import RewardComposer  # For reward composition
from mlx_rl_trainer.rewards.registry import RewardRegistry  # For creating rewards

# Import utils for signal handling and global state (e.g., wandb_run)
from mlx_rl_trainer.utils.mlx_utils import (
    metal_safe_apply_gradients,
)  # For Metal safety
from mlx_rl_trainer.utils.text_utils import (
    _ensure_schedule_dict,
)  # For LR schedule init
from mlx_rl_trainer.generation.caching import (
    PagedKVCache,
)  # For PagedKV (from correct path)
from mlx_rl_trainer.utils.mlx_utils import ContentAlignBridge  # For ContentAlignBridge

# Rich logging setup (from global context)
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as rich_install
from rich import print as rprint

rich_install(show_locals=False)
console = Console(stderr=True, force_terminal=True)  # Global console instance
logger = logging.getLogger(__name__)


# Global shutdown flag (used in base trainer via signal handlers)
shutdown_requested = False


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="MLX RL Trainer")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity level.",
    )
    # Add an argument for --run-server here if that functionality is desired from the main script
    # parser.add_argument(
    #     "--run-server",
    #     action="store_true",
    #     help="Run in async inference server mode instead of training.",
    # )

    args = parser.parse_args()

    # 1. Setup Logging (early, for all subsequent messages)
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [
        RichHandler(markup=True, rich_tracebacks=True, level=log_level, console=console)
    ]
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # 2. Load Configuration
    config_path = Path(args.config)
    try:
        config = ExperimentConfig.load_from_yaml(config_path)
        logger.info(f"Loaded config from {config_path}")
    except (
        FileNotFoundError,
        ValueError,
        Exception,
    ) as e:  # Catch Pydantic ValidationError also
        logger.critical(f"FATAL CONFIGURATION ERROR: {e}")
        sys.exit(1)

    # Ensure checkpoint save_dir is relative to output_dir if not absolute
    if not Path(config.checkpointing.save_dir).is_absolute():
        config.checkpointing.save_dir = str(
            config.trainer.output_dir / config.checkpointing.save_dir
        )
        Path(config.checkpointing.save_dir).mkdir(parents=True, exist_ok=True)

    # 3. Initialize common components
    run_id = str(uuid.uuid4())
    random.seed(config.trainer.seed)
    mx.random.seed(config.trainer.seed)  # FIX: Use mx.random.seed directly
    logger.info(
        f"Starting run with ID: [bold magenta]{run_id}[/bold magenta], seed: {config.trainer.seed}"
    )

    # Set up file logger (after output_dir is confirmed)
    file_handler = logging.FileHandler(
        config.trainer.output_dir / f"training_debug_{run_id}.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)  # Add to root logger

    # Register signal handler for graceful shutdown
    signal.signal(
        signal.SIGINT, lambda s, f: globals().__setitem__("shutdown_requested", True)
    )  # Simple lambda for global flag
    signal.signal(
        signal.SIGTERM, lambda s, f: globals().__setitem__("shutdown_requested", True)
    )

    # 4. Create reward composer
    rewards = []
    for reward_cfg in config.rewards:
        # Dynamically inject tags from ExperimentConfig to individual RewardConfig
        reward_cfg.config["think_open_tag"] = (
            config.rewards[0].think_start_tag if config.rewards else "<think>"
        )
        reward_cfg.config["think_close_tag"] = (
            config.rewards[0].think_end_tag if config.rewards else "</think>"
        )
        reward_cfg.config["answer_open_tag"] = (
            config.rewards[0].answer_start_tag if config.rewards else "<answer>"
        )
        reward_cfg.config["answer_close_tag"] = (
            config.rewards[0].answer_end_tag if config.rewards else "</answer>"
        )

        reward_fn = RewardRegistry.create(reward_cfg.name, reward_cfg.config)
        rewards.append((reward_fn, reward_cfg.weight))
    reward_composer = RewardComposer(rewards)
    logger.info(f"Created reward composer with {len(rewards)} rewards.")

    # 5. Initialize ModelManager, DatasetManager, CheckpointManager
    model_manager = ModelManager(config.model)
    # The tokenizer will be loaded by ModelManager within the trainer's _setup
    data_manager = DatasetManager(
        config.data, tokenizer=None
    )  # Tokenizer is set in trainer setup
    checkpoint_manager = CheckpointManager(
        Path(config.checkpointing.save_dir),
        keep_last_n=config.checkpointing.keep_best_n,
        save_best=True,
    )  # Always save best

    # Initialize PagedKVCache if enabled
    paged_kv_cache = None
    if config.use_paged_kv_cache:
        try:
            # FIX: Attempt to load model config to get architecture details for PagedKVCache
            mock_model_config = {}
            model_config_path = config.model.model_path / "config.json"
            if model_config_path.exists():
                with open(model_config_path, "r") as f:
                    mock_model_config = json.load(f).get("config", {})
            else:  # Fallback to hardcoded mock values
                mock_model_config = {
                    "num_hidden_layers": 4,
                    "num_key_value_heads": 2,
                    "hidden_size": 128,
                    "num_attention_heads": 4,
                }
                logger.warning(
                    f"Could not find model config at {model_config_path}. Using hardcoded defaults for PagedKVCache init."
                )

            paged_kv_cache = PagedKVCache(
                num_layers=mock_model_config.get("num_hidden_layers", 4),
                num_kv_heads=mock_model_config.get("num_key_value_heads", 2),
                head_dim=mock_model_config.get("hidden_size", 128)
                // mock_model_config.get("num_attention_heads", 4),
                block_size=config.kv_cache_block_size,
                num_blocks=config.kv_cache_num_blocks,
            )
            logger.info("PagedKVCache initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize PagedKVCache: {e}", exc_info=True)
            config.use_paged_kv_cache = False  # Disable if init fails

    # 6. Create trainer based on algorithm
    if config.trainer.algorithm == "grpo":
        trainer = GRPOTrainer(
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            paged_kv_cache,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.trainer.algorithm}")

    # 7. Run training loop (delegated to BaseTrainer.run)
    try:
        trainer.run()
        logger.info("Training completed successfully")
    except CustomBaseException as e:
        logger.critical(f"A predictable error halted training: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected system error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application shutdown initiated.")
        # Final plot generation
        MetricsLogger(config, run_id).emit_plots_from_csv()


if __name__ == "__main__":
    mx.set_default_device(
        mx.gpu if mx.gpu_available() else mx.cpu
    )  # Set default device
    main()
