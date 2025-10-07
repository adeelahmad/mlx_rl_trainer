"""Base trainer interface and shared training abstractions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging, time, gc
import mlx.core as mx, mlx.nn as nn
from tqdm import trange

from .config import ExperimentConfig
from .model_manager import ModelManager
from .dataset_manager import DatasetManager
from .checkpoint_manager import CheckpointManager
from ..monitoring.metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)

class CustomBaseException(Exception): pass
class ModelLoadError(CustomBaseException): pass
class DataLoadError(CustomBaseException): pass
class CheckpointError(CustomBaseException): pass
class InvalidConfigurationError(CustomBaseException): pass
class TrainingRuntimeError(CustomBaseException): pass

@dataclass(frozen=True)
class TrainingMetrics:
    loss: float; reward_mean: float; grad_norm: float; learning_rate: float
    step_time_s: float; kl_divergence: float; epoch: int; step: int

@dataclass(frozen=True)
class EvaluationMetrics:
    task_name: str; pass_rate: float = 0.0; perplexity: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        data = {f"eval/{self.task_name}/pass_rate": self.pass_rate}
        if self.perplexity is not None: data[f"eval/{self.task_name}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items(): data[f"eval/{self.task_name}/{k}"] = v
        return data

class BaseTrainer(ABC):
    def __init__( self, config: ExperimentConfig, model_manager: ModelManager, data_manager: DatasetManager,
                  checkpoint_manager: CheckpointManager, metrics_logger: Optional[MetricsLogger] = None):
        self.config, self.model_manager, self.data_manager, self.checkpoint_manager, self.metrics_logger = \
            config, model_manager, data_manager, checkpoint_manager, metrics_logger
        self.actor_model, self.ref_model, self.tokenizer, self.optimizer, self.lr_scheduler = None, None, None, None, None
        self.global_step, self.current_epoch = 0, 0

    @abstractmethod
    def _setup(self) -> Tuple[int, int]: raise NotImplementedError
    @abstractmethod
    def train_step(self, rollout_batch: Dict[str, mx.array], update_step: int) -> TrainingMetrics: raise NotImplementedError
    @abstractmethod
    def generate_rollouts(self, batch_prompts: List[Dict], update_step: int) -> Tuple[Dict, float, Dict]: raise NotImplementedError
    @abstractmethod
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]: raise NotImplementedError

    def save_final_checkpoint(self):
        if self.actor_model and self.optimizer:
            self.checkpoint_manager.save_checkpoint(
                step=self.global_step, model=self.actor_model, optimizer=self.optimizer,
                metadata={"num_updates": self.global_step, "epoch": self.current_epoch, "save_optimizer_state": self.config.checkpointing.save_optimizer_state},
                current_metric=self.checkpoint_manager.best_metric
            )

    async def run(self, should_shutdown: Callable[[], bool]):
        self.global_step, self.current_epoch = self._setup()
        await self.data_manager.load_datasets()
        best_eval_metric = self.checkpoint_manager.best_metric

        with trange(self.global_step, self.config.trainer.num_training_steps, initial=self.global_step, desc="Training") as pbar:
            train_data_iterator = self.data_manager.get_dataloader("train", self.config.trainer.ppo_batch_size)
            for update_step in pbar:
                if should_shutdown(): break
                self.global_step = update_step
                
                accum_metrics, accum_rewards = [], []
                for _ in range(self.config.trainer.grad_accum_steps):
                    try:
                        prompts_batch = next(train_data_iterator)
                    except StopIteration:
                        self.current_epoch += 1
                        train_data_iterator = self.data_manager.get_dataloader("train", self.config.trainer.ppo_batch_size)
                        prompts_batch = next(train_data_iterator)

                    rollout, avg_rew, raw_rews = self.generate_rollouts(prompts_batch, update_step)
                    if rollout:
                        metrics = self.train_step(rollout, update_step)
                        accum_metrics.append(metrics)
                        accum_rewards.append(avg_rew)
                
                if not accum_metrics: continue

                avg_loss = np.mean([m.loss for m in accum_metrics])
                avg_reward = np.mean(accum_rewards)
                avg_grad_norm = np.mean([m.grad_norm for m in accum_metrics])
                
                pbar.set_postfix({"Loss": f"{avg_loss:.3f}", "Reward": f"{avg_reward:.3f}", "GradN": f"{avg_grad_norm:.3f}"})
                if self.metrics_logger:
                    self.metrics_logger.log_metrics({
                        "train/loss": avg_loss, "train/reward_raw_mean": avg_reward, "train/grad_norm": avg_grad_norm,
                        "train/learning_rate": accum_metrics[-1].learning_rate, "train/kl_divergence": accum_metrics[-1].kl_divergence,
                        "train/epoch": self.current_epoch,
                    }, step=update_step)

                is_eval = self.config.trainer.eval_every > 0 and (update_step + 1) % self.config.trainer.eval_every == 0
                is_save = self.config.checkpointing.save_every > 0 and (update_step + 1) % self.config.checkpointing.save_every == 0

                if is_eval:
                    eval_results = self.evaluate(update_step)
                    primary_metric = -float("inf")
                    for res in eval_results:
                        if self.metrics_logger: self.metrics_logger.log_metrics(res.to_dict(), step=update_step)
                        if res.pass_rate > primary_metric: primary_metric = res.pass_rate
                    
                    if primary_metric > best_eval_metric:
                        best_eval_metric = primary_metric
                        self.save_final_checkpoint() # Save best model

                if is_save and not (is_eval and best_eval_metric == primary_metric):
                    self.save_final_checkpoint()
