# /src/mlx_rl_trainer/monitoring/wandb_logger.py
# Revision: 001
# Goal: Enhanced WandB integration with comprehensive metrics
# Type: New Code
# Description: Advanced Weights & Biases logging with custom charts and tables

import logging
import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedWandBLogger:
    """
    Enhanced Weights & Biases logger with comprehensive metrics.

    Features:
    - Custom charts and visualizations
    - Metric grouping and organization
    - Table logging for samples
    - Artifact logging
    - Real-time alerts
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ):
        """
        Initialize enhanced WandB logger.

        Args:
            project: WandB project name
            entity: WandB entity (username/team)
            name: Run name
            config: Configuration dictionary
            tags: Run tags
            notes: Run notes
        """
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is required for EnhancedWandBLogger")

        self.project = project
        self.entity = entity
        self.run_name = name

        # Initialize WandB
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            reinit=False
        )

        # Define custom charts
        self._define_custom_charts()

        # Metric buffer for batch logging
        self.metric_buffer: Dict[str, List[Any]] = {}
        self.buffer_size = 100

        logger.info(f"Initialized EnhancedWandBLogger: {self.run.url}")

    def _define_custom_charts(self) -> None:
        """Define custom WandB charts."""
        # Define metric step
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

        # Group metrics by category
        categories = [
            'loss', 'reward', 'gradient', 'memory', 'tokens',
            'learning', 'generation', 'thinking', 'answer'
        ]

        for category in categories:
            wandb.define_metric(f"{category}/*", step_metric="step")

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics
            step: Training step
            commit: Whether to commit immediately
        """
        log_dict = {}

        if step is not None:
            log_dict['step'] = step

        # Add metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                log_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    log_dict[key] = float(value.item())

        # Log to WandB
        wandb.log(log_dict, commit=commit)

    def log_training_metrics(
        self,
        loss: float,
        reward: float,
        grad_norm: float,
        learning_rate: float,
        step: int,
        **kwargs
    ) -> None:
        """Log core training metrics."""
        metrics = {
            'loss/total': loss,
            'reward/mean': reward,
            'gradient/norm': grad_norm,
            'learning/rate': learning_rate,
            'step': step
        }

        # Add additional metrics
        metrics.update(kwargs)

        self.log_metrics(metrics, step=step)

    def log_distribution(
        self,
        name: str,
        values: List[float],
        step: Optional[int] = None
    ) -> None:
        """Log distribution of values."""
        if not values:
            return

        values_array = np.array(values)

        # Log histogram
        wandb.log({
            f"distribution/{name}": wandb.Histogram(values_array),
            "step": step
        })

        # Log statistics
        stats = {
            f"distribution/{name}/mean": float(np.mean(values_array)),
            f"distribution/{name}/std": float(np.std(values_array)),
            f"distribution/{name}/min": float(np.min(values_array)),
            f"distribution/{name}/max": float(np.max(values_array)),
            f"distribution/{name}/median": float(np.median(values_array))
        }

        if step is not None:
            stats['step'] = step

        wandb.log(stats)

    def log_sample_table(
        self,
        prompts: List[str],
        generations: List[str],
        references: List[str],
        rewards: List[float],
        step: int,
        max_samples: int = 10
    ) -> None:
        """Log samples as WandB table."""
        if not prompts:
            return

        # Limit number of samples
        n = min(len(prompts), max_samples)

        # Create table
        columns = ['Step', 'Prompt', 'Generation', 'Reference', 'Reward']
        data = []

        for i in range(n):
            data.append([
                step,
                prompts[i][:200],  # Truncate for display
                generations[i][:500],
                references[i][:500],
                rewards[i]
            ])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"samples/step_{step}": table})

    def log_gradient_flow(
        self,
        layer_gradients: Dict[str, float],
        step: int
    ) -> None:
        """Log gradient flow across layers."""
        for layer_name, grad_norm in layer_gradients.items():
            wandb.log({
                f"gradient/layer_{layer_name}": grad_norm,
                "step": step
            })

    def log_memory_stats(
        self,
        allocated_mb: float,
        cached_mb: float,
        peak_mb: float,
        step: int
    ) -> None:
        """Log memory statistics."""
        wandb.log({
            'memory/allocated_mb': allocated_mb,
            'memory/cached_mb': cached_mb,
            'memory/peak_mb': peak_mb,
            'step': step
        })

    def log_token_stats(
        self,
        thinking_tokens: int,
        answer_tokens: int,
        total_tokens: int,
        step: int
    ) -> None:
        """Log token statistics."""
        wandb.log({
            'tokens/thinking': thinking_tokens,
            'tokens/answer': answer_tokens,
            'tokens/total': total_tokens,
            'tokens/ratio': thinking_tokens / max(answer_tokens, 1),
            'step': step
        })

    def log_reward_breakdown(
        self,
        reward_components: Dict[str, float],
        step: int
    ) -> None:
        """Log breakdown of reward components."""
        for component, value in reward_components.items():
            wandb.log({
                f"reward/component/{component}": value,
                "step": step
            })

    def log_image(
        self,
        name: str,
        image_path: Path,
        caption: Optional[str] = None,
        step: Optional[int] = None
    ) -> None:
        """Log image to WandB."""
        log_dict = {
            name: wandb.Image(str(image_path), caption=caption)
        }

        if step is not None:
            log_dict['step'] = step

        wandb.log(log_dict)

    def log_chart(
        self,
        name: str,
        chart_path: Path,
        step: Optional[int] = None
    ) -> None:
        """Log chart image."""
        self.log_image(f"charts/{name}", chart_path, step=step)

    def log_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log artifact to WandB."""
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata
        )

        artifact.add_file(str(artifact_path))
        self.run.log_artifact(artifact)

        logger.info(f"Logged artifact: {artifact_name}")

    def log_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log model checkpoint as artifact."""
        artifact_name = f"checkpoint_step_{step}"

        if metadata is None:
            metadata = {}

        metadata.update({
            'step': step,
            'timestamp': time.time()
        })

        self.log_artifact(
            artifact_name=artifact_name,
            artifact_type='model',
            artifact_path=checkpoint_path,
            metadata=metadata
        )

    def log_config_update(self, config: Dict[str, Any]) -> None:
        """Update run configuration."""
        wandb.config.update(config)

    def alert(
        self,
        title: str,
        text: str,
        level: str = 'INFO',
        wait_duration: int = 300
    ) -> None:
        """Send WandB alert."""
        wandb.alert(
            title=title,
            text=text,
            level=getattr(wandb.AlertLevel, level),
            wait_duration=wait_duration
        )

    def finish(self) -> None:
        """Finish WandB run."""
        if self.run:
            self.run.finish()
            logger.info("Finished WandB run")


# Dependencies: wandb, numpy
# Install: pip install wandb numpy
# Usage: logger = EnhancedWandBLogger(project="my-project")
#        logger.log_training_metrics(loss=0.5, reward=0.8, ...)
# Status: Complete and commit-ready
