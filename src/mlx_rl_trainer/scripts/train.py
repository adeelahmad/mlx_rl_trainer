#!/usr/bin/env python
import sys, logging, asyncio, uuid, random, signal
from pathlib import Path
import argparse
import mlx.core as mx
from rich.console import Console, rich_traceback
from rich.logging import RichHandler
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.algorithms.grpo.grpo_trainer import GRPOTrainer
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.monitoring.metrics_logger import MetricsLogger, _emit_plots_from_csv
from mlx_rl_trainer.generation.caching import PagedKVCache

# Import rewards and evaluators to register them
import mlx_rl_trainer.rewards
import mlx_rl_trainer.evaluation

rich_traceback.install(show_locals=False)
console = Console(stderr=True, force_terminal=True)
logger = logging.getLogger(__name__)
shutdown_requested = False

def handle_signal(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        rprint("\n[bold yellow]Shutdown requested. Finishing current step and saving checkpoint...[/bold yellow]")
        shutdown_requested = True

async def main():
    parser = argparse.ArgumentParser(description="MLX RL Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, console=console)])
    
    # Load Config
    config = ExperimentConfig.load_from_yaml(Path(args.config))
    run_id = str(uuid.uuid4())[:8]
    config.trainer.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup File Logger
    file_handler = logging.FileHandler(config.trainer.output_dir / f"training_{run_id}.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting run with ID: [bold magenta]{run_id}[/bold magenta]")
    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    mx.random.seed(config.trainer.seed)
    if mx.gpu_available(): mx.set_default_device(mx.gpu)

    # Signal Handling
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    metrics_logger = MetricsLogger(config, run_id)
    trainer = None
    try:
        # Initialize components
        rewards = [(RewardRegistry.create(rc.name, rc.config), rc.weight) for rc in config.rewards]
        reward_composer = RewardComposer(rewards)
        
        model_manager = ModelManager(config.model)
        checkpoint_manager = CheckpointManager(config.trainer.output_dir, config.checkpointing.keep_last_n)
        data_manager = DatasetManager(config.data, tokenizer=None) # Tokenizer is set up in trainer
        
        paged_kv_cache = None
        if config.use_paged_kv_cache:
            # This part still needs model details; we'll assume they are in the config for now
            logger.warning("PagedKVCache initialization requires model config details. Using placeholder values.")
            paged_kv_cache = PagedKVCache(num_layers=32, num_kv_heads=32, head_dim=128, 
                                          block_size=config.kv_cache_block_size, num_blocks=config.kv_cache_num_blocks)

        if config.trainer.algorithm == "grpo":
            trainer = GRPOTrainer(
                config, model_manager, data_manager, checkpoint_manager, 
                reward_composer, paged_kv_cache, metrics_logger
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.trainer.algorithm}")
        
        # Start training
        await trainer.run(lambda: shutdown_requested)

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
    finally:
        if trainer and trainer.global_step > 0:
            rprint("[bold blue]Training finished or interrupted. Saving final state...[/bold blue]")
            trainer.save_final_checkpoint()
        if metrics_logger:
            metrics_logger.close()
            _emit_plots_from_csv(metrics_logger.file_path, config.trainer.output_dir)
            rprint(f"Metrics saved to {metrics_logger.file_path}")
            rprint(f"Plots generated in {config.trainer.output_dir / 'plots'}")
        rprint("[bold green]Shutdown complete.[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
