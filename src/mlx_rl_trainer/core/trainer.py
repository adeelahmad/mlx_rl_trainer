"""Base trainer interface and shared training abstractions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging, time, gc
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import numpy as np
from tqdm import trange

from .config import ExperimentConfig
from .model_manager import ModelManager
from .dataset_manager import DatasetManager
from .checkpoint_manager import CheckpointManager
from ..monitoring.metrics_logger import MetricsLogger
from .exceptions import (
    TrainingRuntimeError,
    CheckpointError,
)  # Import from the new module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    reward_mean: float
    grad_norm: float
    learning_rate: float
    step_time_s: float
    kl_divergence: float
    epoch: int = 0
    step: int = 0
    reward_std: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "train/loss": self.loss,
            "train/reward_mean": self.reward_mean,
            "train/reward_std": self.reward_std,
            "train/grad_norm": self.grad_norm,
            "train/learning_rate": self.learning_rate,
            "train/step_time_s": self.step_time_s,
            "train/kl_divergence": self.kl_divergence,
            "train/epoch": self.epoch,
            "train/step": self.step,
        }
        data.update(self.custom_metrics)
        return data


@dataclass(frozen=True)
class EvaluationMetrics:
    task_name: str
    pass_rate: float = 0.0
    perplexity: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {f"eval/{self.task_name}/pass_rate": self.pass_rate}
        if self.perplexity is not None:
            data[f"eval/{self.task_name}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items():
            if not k.startswith(f"eval/{self.task_name}/"):
                data[f"eval/{self.task_name}/{k}"] = v
            else:
                data[k] = v
        return data


class BaseTrainer(ABC):
    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: ModelManager,
        data_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        reward_composer: Any,
        paged_kv_cache: Optional[Any],
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        (
            self.config,
            self.model_manager,
            self.data_manager,
            self.checkpoint_manager,
            self.reward_composer,
            self.paged_kv_cache,
            self.metrics_logger,
        ) = (
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            paged_kv_cache,
            metrics_logger,
        )
        (
            self.actor_model,
            self.ref_model,
            self.tokenizer,
            self.optimizer,
            self.lr_scheduler,
        ) = (None, None, None, None, None)
        self.global_step, self.current_epoch = 0, 0
        self._run_id = (
            self.metrics_logger.run_id
            if self.metrics_logger
            else f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        logger.info("BaseTrainer initialized.")

    @abstractmethod
    def _setup(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> Tuple[TrainingMetrics, Dict[str, mx.array]]:
        raise NotImplementedError

    @abstractmethod
    def generate_rollouts(
        self, batch_data: Dict[str, Any], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        raise NotImplementedError

    def save_final_checkpoint(self, reason: str = "final"):
        if self.actor_model:
            self.checkpoint_manager.save_checkpoint(
                step=self.global_step,
                model=self.actor_model,
                optimizer=self.optimizer,
                metadata={
                    "num_updates": self.global_step,
                    "epoch": self.current_epoch,
                    "reason": reason,
                    "log_id": self._run_id,
                    "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                },
                current_metric=self.checkpoint_manager.best_metric,
            )

    async def run(self, should_shutdown: Callable[[], bool]):
        self.global_step, self.current_epoch = self._setup()

        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)
            self.data_manager.set_system_prompt(self.config.system_prompt)

        await self.data_manager.load_datasets()

        pbar = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
            desc="Training Progress",
            unit="update",
            leave=True,
        )
        train_data_iterator = iter(
            self.data_manager.get_dataloader(
                "train", self.config.trainer.ppo_batch_size
            )
        )

        with pbar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    logger.info("Shutdown requested. Breaking training loop.")
                    self.save_final_checkpoint(reason="signal")
                    break

                (
                    accumulated_metrics_list,
                    avg_rewards_list,
                    raw_reward_components_list,
                ) = ([], [], [])
                accum_grads = None

                for _ in range(self.config.trainer.grad_accum_steps):
                    try:
                        batch_data = next(train_data_iterator)
                    except StopIteration:
                        self.current_epoch += 1
                        logger.info(f"Starting Epoch {self.current_epoch}")
                        train_data_iterator = iter(
                            self.data_manager.get_dataloader(
                                "train", self.config.trainer.ppo_batch_size
                            )
                        )
                        try:
                            batch_data = next(train_data_iterator)
                        except StopIteration:
                            raise TrainingRuntimeError("Dataset exhausted.")

                    (
                        rollout_batch,
                        avg_reward_mb,
                        raw_reward_components_mb,
                    ) = self.generate_rollouts(batch_data, self.global_step)

                    if (
                        not rollout_batch
                        or "tokens" not in rollout_batch
                        or not isinstance(rollout_batch["tokens"], mx.array)
                        or rollout_batch["tokens"].size == 0
                    ):
                        logger.warning(
                            f"Micro-batch at step {self.global_step} produced no valid rollouts. Skipping."
                        )
                        continue

                    metrics_mb, grads_mb = self.train_step(
                        rollout_batch, self.global_step
                    )
                    accumulated_metrics_list.append(metrics_mb)
                    avg_rewards_list.append(avg_reward_mb)
                    raw_reward_components_list.append(raw_reward_components_mb)

                    if grads_mb:
                        accum_grads = (
                            mx.utils.tree_map(mx.add, accum_grads, grads_mb)
                            if accum_grads
                            else grads_mb
                        )
                    mx.clear_cache()
                    gc.collect()

                if accum_grads and self.optimizer:
                    grad_norm = np.linalg.norm(
                        [
                            np.linalg.norm(v.flatten())
                            for v in mx.utils.tree_flatten(accum_grads)[1]
                        ]
                    )

                    self.optimizer.set_learning_rate(
                        mx.array(float(self.lr_scheduler(self.global_step)))
                    )
                    self.optimizer.apply_gradients(
                        accum_grads, self.actor_model.trainable_parameters()
                    )
                    mx.eval(self.actor_model.parameters(), self.optimizer.state)

                    avg_loss = np.mean([m.loss for m in accumulated_metrics_list])
                    avg_reward_mean = np.mean(avg_rewards_list)
                    avg_lr = self.lr_scheduler(self.global_step)
                    aggregated_raw_rewards = (
                        {
                            k: np.mean(
                                [
                                    comp.get(k, 0.0)
                                    for comp in raw_reward_components_list
                                ]
                            )
                            for k in raw_reward_components_list[0]
                        }
                        if raw_reward_components_list
                        else {}
                    )

                    if self.metrics_logger:
                        self.metrics_logger.log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/reward_mean": avg_reward_mean,
                                "train/grad_norm": grad_norm,
                                "train/learning_rate": avg_lr,
                                "train/kl_divergence": np.mean(
                                    [m.kl_divergence for m in accumulated_metrics_list]
                                ),
                                "train/epoch": self.current_epoch,
                                "train/step": self.global_step,
                                **{
                                    f"train/rewards/raw_{k}": v
                                    for k, v in aggregated_raw_rewards.items()
                                },
                            },
                            step=self.global_step,
                        )

                    pbar.set_postfix(
                        {
                            "Loss": f"{avg_loss:.4f}",
                            "Rew": f"{avg_reward_mean:.3f}",
                            "LR": f"{avg_lr:.1e}",
                            "GradN": f"{grad_norm:.3f}",
                        }
                    )
                    pbar.update(1)

                    is_eval = (
                        self.config.trainer.eval_every > 0
                        and (self.global_step + 1) % self.config.trainer.eval_every == 0
                    )
                    is_save = (
                        self.config.checkpointing.save_every > 0
                        and (self.global_step + 1)
                        % self.config.checkpointing.save_every
                        == 0
                    )
                    is_final = (
                        self.global_step == self.config.trainer.num_training_steps - 1
                    )

                    primary_metric = -float("inf")
                    if is_eval or is_final:
                        eval_results = self.evaluate(self.global_step)
                        for metric in eval_results:
                            if self.metrics_logger:
                                self.metrics_logger.log_metrics(
                                    metric.to_dict(), step=self.global_step
                                )
                            if metric.pass_rate > primary_metric:
                                primary_metric = metric.pass_rate

                    should_save_best = self.checkpoint_manager.is_best_metric(
                        primary_metric
                    )
                    if is_save or is_final or should_save_best:
                        self.checkpoint_manager.save_checkpoint(
                            step=self.global_step,
                            model=self.actor_model,
                            optimizer=self.optimizer,
                            metadata={
                                "num_updates": self.global_step,
                                "epoch": self.current_epoch,
                                "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                            },
                            current_metric=primary_metric,
                        )

                self.global_step += 1

            self.save_final_checkpoint(
                reason="completed" if not should_shutdown() else "interrupted"
            )
