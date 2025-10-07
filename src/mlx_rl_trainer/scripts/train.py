#!/usr/bin/env python
import sys, logging, asyncio, uuid, random, signal, time, json
from pathlib import Path
import argparse
import mlx.core as mx
from rich.console import Console, rich_traceback
from rich.logging import RichHandler
from rich import print as rprint
import numpy as np

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try: import wandb; WANDB_AVAILABLE=True
except ImportError: WANDB_AVAILABLE=False

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.core.exceptions import CustomBaseException, TrainingRuntimeError
from mlx_rl_trainer.algorithms.grpo.grpo_trainer import GRPOTrainer
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.monitoring.metrics_logger import MetricsLogger, _emit_plots_from_csv

# Import rewards and evaluators to register them
import mlx_rl_trainer.rewards 
import mlx_rl_trainer.evaluation

rich_traceback.install(show_locals=False)
console = Console(stderr=True, force_terminal=True)
logger = logging.getLogger(__name__)

shutdown_requested = False
wandb_run = None

def handle_signal(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        rprint("\n[bold yellow]Shutdown requested. Finishing current step and saving checkpoint...[/bold yellow]")
        shutdown_requested = True

def path_to_str(d):
    if isinstance(d, dict): return {k: path_to_str(v) for k, v in d.items()}
    if isinstance(d, list): return [path_to_str(v) for v in d]
    if isinstance(d, Path): return str(d)
    return d

async def _async_main():
    global shutdown_requested, wandb_run
    parser = argparse.ArgumentParser(description="MLX RL Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (overrides auto-resume)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [RichHandler(markup=True, rich_tracebacks=True, console=console, level=log_level)]
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    try:
        config = ExperimentConfig.load_from_yaml(Path(args.config))
    except Exception as e:
        logger.critical(f"FATAL CONFIGURATION ERROR: {e}"); sys.exit(1)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    config.trainer.output_dir = config.trainer.output_dir / run_id
    config.trainer.output_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(config.trainer.output_dir / f"training_debug_{run_id}.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Starting run with ID: [bold magenta]{run_id}[/bold magenta], output to {config.trainer.output_dir}")

    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    mx.random.seed(config.trainer.seed)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if config.monitoring.use_wandb and WANDB_AVAILABLE:
        try:
            wandb_cfg = path_to_str(config.model_dump())
            wandb_cfg['run_id'] = run_id
            wandb_run = wandb.init(
                project=config.monitoring.wandb_project, entity=config.monitoring.wandb_entity,
                name=config.monitoring.wandb_run_name or run_id, config=wandb_cfg,
                resume='allow', id=run_id
            )
            logger.info(f"W&B logging initialized: {wandb_run.url}")
            from mlx_rl_trainer.monitoring import metrics_logger
            metrics_logger.wandb_run = wandb_run 
        except Exception as e:
            logger.error(f"W&B initialization failed: {e}. Disabling W&B.", exc_info=True)
            config.monitoring.use_wandb = False
            wandb_run = None

    rewards = [(RewardRegistry.create(rc.name, rc.config), rc.weight) for rc in config.rewards]
    reward_composer = RewardComposer(rewards, context_cls=RewardContext)
    
    model_manager = ModelManager(config.model)
    data_manager = DatasetManager(config.data, tokenizer=None) 
    checkpoint_manager = CheckpointManager(
        config.trainer.output_dir / config.checkpointing.save_dir,
        keep_last_n=config.checkpointing.keep_last_n, # CORRECTED from keep_best_n
        save_best=True
    )
    if args.resume: checkpoint_manager.resume_from_path = Path(args.resume)

    metrics_logger = MetricsLogger(config, run_id)
    
    paged_kv_cache = None
    
    if config.trainer.algorithm == "grpo":
        trainer = GRPOTrainer(
            config, model_manager, data_manager, checkpoint_manager, 
            reward_composer, paged_kv_cache, metrics_logger
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.trainer.algorithm}")
    
    try:
        await trainer.run(lambda: shutdown_requested)
        logger.info("[bold green]Training completed successfully![/bold green]")
    except CustomBaseException as e:
        logger.critical(f"A predictable error halted training: {e}", exc_info=True)
        if trainer and trainer.global_step > 0: trainer.save_final_checkpoint(reason="error_halt") 
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}", exc_info=True)
        if trainer and trainer.global_step > 0: trainer.save_final_checkpoint(reason="unexpected_crash") 
        sys.exit(1)
    finally:
        logger.info("Application shutdown sequence initiated.")
        if metrics_logger:
            metrics_logger.close()
            _emit_plots_from_csv(metrics_logger.file_path, config.trainer.output_dir, config)
        if wandb_run: wandb_run.finish()
        logger.info("[bold blue]All resources released. Shutdown complete.[/bold blue]")

def main():
    if mx.gpu_available(): mx.set_default_device(mx.gpu)
    else: mx.set_default_device(mx.cpu)
    rprint(f"MLX using device: [bold cyan]{mx.default_device()}[/bold cyan]")
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()
