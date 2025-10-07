"""
Base trainer interface and shared training abstractions.
"""
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
from .exceptions import TrainingRuntimeError
from ..monitoring.metrics_logger import MetricsLogger

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
        if self.perplexity is not None: data[f"eval/{self.task_name}/perplexity"] = self.perplexity
        for k, v in self.additional_info.items():
            if not k.startswith(f"eval/{self.task_name}/"): data[f"eval/{self.task_name}/{k}"] = v
            else: data[k] = v
        return data

class BaseTrainer(ABC):
    def __init__(self, config, model_manager, data_manager, checkpoint_manager, reward_composer, paged_kv_cache, metrics_logger):
        self.config = config
        self.model_manager = model_manager
        self.data_manager = data_manager
        self.checkpoint_manager = checkpoint_manager
        self.reward_composer = reward_composer
        self.paged_kv_cache = paged_kv_cache
        self.metrics_logger = metrics_logger
        self.actor_model = self.ref_model = self.tokenizer = self.optimizer = self.lr_scheduler = None
        self.global_step, self.current_epoch = 0, 0
        self._run_id = metrics_logger.run_id if metrics_logger else f"run_{time.strftime('%Y%m%d-%H%M%S')}"

    @abstractmethod
    def _setup(self) -> Tuple[int, int]: raise NotImplementedError
    @abstractmethod
    def train_step(self, rollout_batch: Dict[str, mx.array], update_step: int) -> Tuple[TrainingMetrics, Dict[str, mx.array]]: raise NotImplementedError
    @abstractmethod
    def generate_rollouts(self, batch_data: Dict[str, Any], update_step: int) -> Tuple[Dict, float, Dict]: raise NotImplementedError
    @abstractmethod
    def evaluate(self, update_step: int) -> List[EvaluationMetrics]: raise NotImplementedError

    def save_final_checkpoint(self, reason: str = "final"):
        if self.actor_model:
            model_params = dict(mx.utils.tree_flatten(self.actor_model.parameters()))
            optimizer_state = self.optimizer.state if self.optimizer and self.config.checkpointing.save_optimizer_state else {}
            self.checkpoint_manager.save_checkpoint(
                step=self.global_step, model_state=model_params, optimizer_state=optimizer_state,
                metadata={"num_updates": self.global_step, "epoch": self.current_epoch, "reason": reason},
                current_metric=self.checkpoint_manager.best_metric
            )

    async def run(self, should_shutdown: Callable[[], bool]):
        self.global_step, self.current_epoch = self._setup()
        if self.tokenizer:
            self.data_manager.set_tokenizer(self.tokenizer)
            self.data_manager.set_system_prompt(self.config.system_prompt)

        await self.data_manager.load_datasets()
        train_data_iterator = iter(self.data_manager.get_dataloader("train", self.config.trainer.ppo_batch_size))
        
        with trange(self.global_step, self.config.trainer.num_training_steps, initial=self.global_step, desc="Training") as pbar:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    self.save_final_checkpoint(reason="signal"); break

                accum_metrics, avg_rewards, raw_rewards_list, accum_grads = [], [], [], None
                for _ in range(self.config.trainer.grad_accum_steps):
                    try: batch_data = next(train_data_iterator)
                    except StopIteration:
                        self.current_epoch += 1
                        train_data_iterator = iter(self.data_manager.get_dataloader("train", self.config.trainer.ppo_batch_size))
                        batch_data = next(train_data_iterator)

                    rollout, avg_rew, raw_rews = self.generate_rollouts(batch_data, self.global_step)
                    if not rollout or not rollout.get("tokens", []).size: continue

                    metrics, grads = self.train_step(rollout, self.global_step)
                    accum_metrics.append(metrics); avg_rewards.append(avg_rew); raw_rewards_list.append(raw_rews)
                    if grads:
                        accum_grads = mx.utils.tree_map(mx.add, accum_grads, grads) if accum_grads is not None else grads
                    mx.clear_cache(); gc.collect()

                if accum_grads:
                    self.optimizer.set_learning_rate(mx.array(float(self.lr_scheduler(self.global_step))))
                    self.optimizer.apply_gradients(accum_grads, self.actor_model.trainable_parameters())
                    mx.eval(self.actor_model.parameters(), self.optimizer.state)
                    
                    # Aggregate and log
                    avg_loss = np.mean([m.loss for m in accum_metrics])
                    avg_reward_mean = np.mean(avg_rewards)
                    avg_grad_norm = np.mean([m.grad_norm for m in accum_metrics]) # Average of norms
                    pbar.set_postfix({"Loss": f"{avg_loss:.3f}", "Rew": f"{avg_reward_mean:.3f}", "GradN": f"{avg_grad_norm:.3f}"})
                    if self.metrics_logger:
                        log_data = {"train/loss": avg_loss, "train/reward_mean": avg_reward_mean, "train/grad_norm": avg_grad_norm, "train/learning_rate": self.lr_scheduler(self.global_step), "train/epoch": self.current_epoch}
                        self.metrics_logger.log_metrics(log_data, step=self.global_step)

                is_eval = (self.global_step + 1) % self.config.trainer.eval_every == 0
                is_save = (self.global_step + 1) % self.config.checkpointing.save_every == 0
                if is_eval:
                    eval_results = self.evaluate(self.global_step)
                    primary_metric = max([res.pass_rate for res in eval_results], default=-float("inf"))
                    if self.checkpoint_manager.is_best_metric(primary_metric):
                        self.save_final_checkpoint("best_metric")
                
                if is_save and not (is_eval and self.checkpoint_manager.is_best_metric(primary_metric)):
                    self.save_final_checkpoint("periodic")
                
                self.global_step += 1
                pbar.update(1)

        self.save_final_checkpoint("completed" if not should_shutdown() else "interrupted")
