# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/trainer.py
# revision_no: 003
# goals_of_writing_code_block: Define abstract base classes and fix evaluation loop logic within the `run` method, ensuring robust metric reporting and checkpointing.
# type_of_code_response: change existing
"""
Base trainer interface and shared training abstractions, designed with OOP and SOLID principles.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
import mlx.core as mx
import mlx.nn as nn
import gc
from mlx.utils import tree_flatten, tree_unflatten
import mlx
from .config import ExperimentConfig  # Use new ExperimentConfig


class CustomBaseException(Exception):
    pass


class ModelLoadError(CustomBaseException):
    pass


class DataLoadError(CustomBaseException):
    pass


class CheckpointError(CustomBaseException):
    pass


class InvalidConfigurationError(CustomBaseException):
    pass


class TrainingRuntimeError(CustomBaseException):
    pass


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    reward_mean: float
    grad_norm: float
    learning_rate: float
    step_time_s: float
    tokens_per_sec: float
    kl_divergence: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "loss": self.loss,
            "reward_mean": self.reward_mean,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "step_time_s": self.step_time_s,
        }
        data.update(self.custom_metrics)
        return data


@dataclass(frozen=True)
class EvaluationMetrics:
    task_name: str
    pass_rate: float = 0.0
    perplexity: Optional[float] = None  # Added perplexity field
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {f"eval/{self.task_name}/pass_rate": self.pass_rate}
        if self.perplexity is not None:
            data[f"eval/{self.task_name}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items():
            data[f"eval/{self.task_name}/{k}"] = v
        return data


class BaseTrainer(ABC):
    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: Any,
        data_manager: Any,
        checkpoint_manager: Any,
    ):
        self.config = config
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager
        self.actor_model: Optional[mx.nn.Module] = None
        self.ref_model: Optional[mx.nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.optimizer: Optional[mx.optimizers.Optimizer] = None
        self.lr_scheduler: Optional[Callable[[int], float]] = None

        # Internal state for training loop
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.optimizer_is_set = False
        self.lr_scheduler_is_set = False
        self.actor_model_is_set = False

        logging.info("BaseTrainer initialized with injected dependencies.")

    @abstractmethod
    def _setup(self) -> Tuple[int, int]:
        """
        Initializes models, tokenizers, optimizers, and loads any existing checkpoints.

        Returns:
            A tuple of (start_update_step, start_epoch) to resume training from.
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> TrainingMetrics:
        """
        Executes a single optimization step of the RL algorithm.

        Args:
            rollout_batch: A dictionary containing data from generated rollouts (e.g., tokens, advantages).
            update_step: The current global update step.

        Returns:
            TrainingMetrics for the current step.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_rollouts(
        self, batch_prompts_data: Dict[str, List[Any]], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """
        Generates a batch of rollouts (trajectories) from the current actor policy.

        Args:
            batch_prompts_data: A dictionary containing input prompts from the dataset.
            update_step: The current global update step.

        Returns:
            A tuple containing:
            - A dictionary of rollout data (e.g., full tokens, response masks, advantages).
            - The average reward obtained from these rollouts.
            - A dictionary of raw reward components for logging.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        """
        Runs evaluation benchmarks on the current model using the validation dataset.

        Args:
            update_step: The current global update step.

        Returns:
            A list of EvaluationMetrics objects, one for each configured evaluator.
        """
        raise NotImplementedError

    async def run(self) -> None:
        """
        Main entry point for the training loop. Orchestrates setup, training steps,
        evaluation, and checkpointing.
        """
        from tqdm import (
            trange,
        )  # Import tqdm here to avoid circular dependency with logging setup

        self.global_step, self.current_epoch = self._setup()
        logging.info(
            f"Training will start from update step {self.global_step}, epoch {self.current_epoch}."
        )

        pbar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc="Training Progress",
            unit="update",
        )

        try:
            # FIX: Ensure data_manager.load_datasets() is called before getting dataloader
            await self.data_manager.load_datasets()
            train_data_iterator = self.data_manager.get_dataloader(
                "train", self.config.trainer.ppo_batch_size
            )

            for update_step in pbar:
                self.global_step = update_step  # Keep self.global_step updated

                # Accumulate gradients over multiple micro-batches
                accumulated_metrics: List[TrainingMetrics] = []
                for _ in range(self.config.trainer.grad_accum_steps):
                    prompts_batch = next(train_data_iterator, None)
                    logging.debug(f"Prompts batch from dataloader: {prompts_batch}")

                    # If an epoch finishes, reset iterator and increment epoch count
                    if prompts_batch is None:
                        self.current_epoch += 1
                        logging.info(
                            f"Epoch finished. Starting new epoch: {self.current_epoch}."
                        )
                        train_data_iterator = self.data_manager.get_dataloader(
                            "train", self.config.trainer.ppo_batch_size
                        )
                        prompts_batch = next(train_data_iterator, None)
                        if prompts_batch is None:
                            raise TrainingRuntimeError(
                                "Data stream exhausted, cannot get next batch."
                            )

                    # Generate rollouts and perform one training step
                    (
                        rollout_data,
                        avg_reward_from_rollouts,
                        raw_rewards_breakdown,
                    ) = self.generate_rollouts([prompts_batch], self.global_step)
                    metrics = self.train_step(
                        [prompts_batch], self.global_step, self.current_epoch
                    )
                    accumulated_metrics.append(metrics)

                # Aggregate and log metrics from accumulation steps
                avg_loss = sum(m.loss for m in accumulated_metrics) / len(
                    accumulated_metrics
                )
                avg_reward_mean = sum(m.reward_mean for m in accumulated_metrics) / len(
                    accumulated_metrics
                )
                avg_grad_norm = sum(m.grad_norm for m in accumulated_metrics) / len(
                    accumulated_metrics
                )
                avg_lr = accumulated_metrics[-1].learning_rate  # Last LR applied

                # Update progress bar with current key metrics
                pbar.set_postfix(
                    {
                        "Loss": f"{avg_loss:.4f}",
                        "Reward": f"{avg_reward_mean:.3f}",
                        "LR": f"{avg_lr:.2e}",
                        "GradN": f"{avg_grad_norm:.3f}",
                    }
                )

                # --- Evaluation and Checkpointing Logic ---
                is_eval_step = (
                    update_step > 0
                    and update_step % self.config.trainer.eval_every == 0
                )
                is_save_step = (
                    update_step > 0
                    and update_step % self.config.trainer.save_every == 0
                )

                eval_metric_for_checkpoint = -float("inf")
                if is_eval_step:
                    logging.info(f"Starting evaluation at update step {update_step}...")
                    eval_results = self.evaluate(
                        update_step
                    )  # This calls configured evaluators

                    # Log evaluation results (assuming metrics_logger is available)
                    from mlx_rl_trainer.monitoring.metrics_logger import (
                        MetricsLogger,
                    )  # Local import for logging

                    metrics_logger_instance = MetricsLogger(
                        self.config, run_id=str(uuid.uuid4())
                    )  # Mock run_id for now
                    for eval_metric in eval_results:
                        metrics_logger_instance.log_metrics(
                            eval_metric.to_dict(), step=update_step
                        )
                        # Identify primary metric for checkpointing (e.g., pass@1 for HumanEval)
                        if (
                            eval_metric.task_name == "human_eval"
                            and "pass@1" in eval_metric.additional_info
                        ):
                            current_pass_rate = eval_metric.additional_info["pass@1"]
                            if current_pass_rate > eval_metric_for_checkpoint:
                                eval_metric_for_checkpoint = current_pass_rate
                        elif (
                            eval_metric.task_name == "gsm8k"
                            and "bench/acc" in eval_metric.additional_info
                        ):
                            current_accuracy = eval_metric.additional_info["bench/acc"]
                            if current_accuracy > eval_metric_for_checkpoint:
                                eval_metric_for_checkpoint = current_accuracy
                        # Add other primary metric selections here

                if is_save_step or (
                    is_eval_step and eval_metric_for_checkpoint > -float("inf")
                ):
                    self.checkpoint_manager.save_checkpoint(
                        step=update_step,
                        model_state=dict(tree_flatten(self.actor_model.parameters())),
                        optimizer_state=self.optimizer.state if self.optimizer else {},
                        metadata={
                            "num_updates": update_step,
                            "epoch": self.current_epoch,
                        },
                        current_metric=eval_metric_for_checkpoint,
                    )

        except (KeyboardInterrupt, TrainingRuntimeError) as e:
            logging.critical(
                f"Training interrupted: {e}. Initiating graceful shutdown.",
                exc_info=True,
            )
        except Exception as e:
            logging.critical(f"An unexpected error halted training: {e}", exc_info=True)
        finally:
            logging.info("Training process finalized. Saving final state.")
            # FIX: Safely access pbar.postfix to prevent crash on early exit
            final_metric = 0.0
            if hasattr(pbar, "postfix") and pbar.postfix is not None:
                try:
                    # Attempt to parse the reward string, e.g., "0.876"
                    reward_str = pbar.postfix.get("Reward")
                    if isinstance(reward_str, str):
                        final_metric = float(reward_str)
                except (ValueError, TypeError):
                    final_metric = 0.0  # Fallback if parsing fails
            try:
                self.checkpoint_manager.save_checkpoint(
                    step=pbar.n,
                    model_state=dict(tree_flatten(self.actor_model.parameters())),
                    optimizer_state=self.optimizer.state if self.optimizer else {},
                    metadata={
                        "num_updates": pbar.n,
                        "epoch": self.current_epoch,
                        "final_run": True,
                    },
                    current_metric=float(final_metric),
                )
            except Exception as e:
                logging.error(f"Failed to save final checkpoint: {e}", exc_info=True)
