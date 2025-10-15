import gc
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_map
from tqdm import trange

from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.data.dataset_manager import DatasetManager
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.core.types import EvaluationMetrics, TrainingMetrics
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.monitoring.metrics_logger import _emit_plots_from_csv, MetricsLogger
from mlx_rl_trainer.utils.mlx_utils import _maybe_clip_grad_norm

logger = logging.getLogger(__name__)


def _get_memory_usage_mb() -> Dict[str, float]:
    try:
        if hasattr(mx.metal, "is_available") and mx.metal.is_available():
            cache_mb = mx.metal.cache_size() / 1e6
            active_mb = mx.metal.get_active_memory() / 1e6
            peak_mb = mx.metal.get_peak_memory() / 1e6
            return {"cache_mb": cache_mb, "allocated_mb": active_mb, "peak_mb": peak_mb}
    except Exception:
        return {}


class BaseTrainer(ABC):
    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: ModelManager,
        data_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        reward_composer,
        paged_kv_cache,
        metrics_logger: Optional[MetricsLogger] = None,
    ):
        (
            self.config,
            self.model_manager,
            self.data_manager,
            self.checkpoint_manager,
            self.reward_composer,
            self.metrics_logger,
        ) = (
            config,
            model_manager,
            data_manager,
            checkpoint_manager,
            reward_composer,
            metrics_logger,
        )
        self.actor_model = (
            self.ref_model
        ) = self.tokenizer = self.optimizer = self.lr_scheduler = None
        self.global_step = self.current_epoch = 0
        self._run_id = (
            self.metrics_logger.run_id if self.metrics_logger else "local_run"
        )
        self._last_logged_metrics = {}

    @abstractmethod
    def _setup(self) -> (int, int):
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, rollout_batch: Dict, update_step: int
    ) -> (TrainingMetrics, mx.array, Dict):
        raise NotImplementedError

    @abstractmethod
    def generate_rollouts(
        self, batch_data: Dict, update_step: int, is_validation: bool = False
    ):
        raise NotImplementedError

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        if not self.config.evaluation:
            return []
        logger.info(f"Starting evaluation on benchmarks at step {update_step}...")
        self.actor_model.eval()
        all_metrics = []
        for eval_config in self.config.evaluation:
            try:
                evaluator = EvaluatorRegistry.create(
                    eval_config.name, eval_config.config
                )
                dataset = evaluator.load_dataset(
                    eval_config.dataset_path,
                    eval_config.dataset_subset,
                    eval_config.split,
                )
                metrics = evaluator.evaluate(self.actor_model, self.tokenizer, dataset)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(
                    f"Failed to run evaluator '{eval_config.name}': {e}", exc_info=True
                )
        self.actor_model.train()
        return all_metrics

    def _run_validation_loop(self, update_step: int) -> Dict[str, float]:
        if not self.config.data.val_path:
            return {}
        logger.info(f"Starting validation at step {update_step}...")
        self.actor_model.eval()
        val_loader = self.data_manager.get_dataloader(
            "val", self.config.trainer.ppo_batch_size
        )
        total_reward, num_batches = 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= 5:
                break
            _, reward_mean, _, _ = self.generate_rollouts(
                batch, update_step, is_validation=True
            )
            total_reward += reward_mean
            num_batches += 1
        self.actor_model.train()
        if num_batches > 0:
            avg_reward = total_reward / num_batches
            logger.info(f"Validation complete. Average reward: {avg_reward:.4f}")
            return {"validation/reward_mean": avg_reward}
        return {}

    def save_final_checkpoint(self, reason: str = "final"):
        if self.actor_model:
            self.checkpoint_manager.save_checkpoint(
                self.global_step,
                self.actor_model,
                self.optimizer,
                {
                    "step": self.global_step,
                    "epoch": self.current_epoch,
                    "reason": reason,
                    "save_optimizer_state": self.config.checkpointing.save_optimizer_state,
                },
                self.checkpoint_manager.best_metric,
            )

    def _aggressive_memory_cleanup(self):
        mx.clear_cache()
        gc.collect()

    def _log_memory(self, stage: str):
        if self.config.trainer.log_memory_usage:
            mem = _get_memory_usage_mb()
            if mem and self.metrics_logger:
                self.metrics_logger.log_metrics(
                    {f"memory/{stage}/{k}": v for k, v in mem.items()}, self.global_step
                )

    async def run(self, should_shutdown: callable):
        initial_step, self.current_epoch = self._setup()
        self.global_step = initial_step + 1 if initial_step > 0 else 0
        self.data_manager.set_tokenizer(self.tokenizer)
        await self.data_manager.load_datasets()

        t_range = trange(
            self.global_step,
            self.config.trainer.num_training_steps,
            initial=self.global_step,
        )
        data_loader = iter([])

        with t_range:
            while self.global_step < self.config.trainer.num_training_steps:
                if should_shutdown():
                    break
                self._log_memory("loop_start")

                accum_grads, total_metrics, num_micro_batches = None, {}, 0
                self.optimizer.learning_rate = self.lr_scheduler(self.global_step)

                for _ in range(self.config.trainer.grad_accum_steps):
                    try:
                        batch = next(data_loader)
                    except StopIteration:
                        self.current_epoch += 1
                        data_loader = self.data_manager.get_dataloader(
                            "train", self.config.trainer.ppo_batch_size
                        )
                        batch = next(data_loader)

                    self._log_memory("pre_rollout")
                    (
                        rollouts,
                        reward_mean,
                        raw_rewards,
                        gen_metrics,
                    ) = self.generate_rollouts(batch, self.global_step)
                    if not rollouts:
                        continue
                    self._log_memory("post_rollout")

                    train_metrics, grads, loss_metrics = self.train_step(
                        rollouts, self.global_step
                    )
                    self._log_memory("post_train_step")

                    grads = tree_map(
                        lambda g: g / self.config.trainer.grad_accum_steps, grads
                    )
                    accum_grads = (
                        tree_map(mx.add, accum_grads, grads) if accum_grads else grads
                    )

                    num_micro_batches += 1
                    for k, v in train_metrics.to_dict().items():
                        total_metrics[k] = total_metrics.get(k, 0) + v
                    for k, v in loss_metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0) + v
                    for k, v in raw_rewards.items():
                        total_metrics[f"rewards/{k}"] = (
                            total_metrics.get(f"rewards/{k}", 0) + v
                        )
                    for k, v in gen_metrics.items():
                        total_metrics[f"generation/{k}"] = (
                            total_metrics.get(f"generation/{k}", 0) + v
                        )

                if not accum_grads:
                    self.global_step += 1
                    continue

                grads, grad_norm = _maybe_clip_grad_norm(
                    accum_grads, self.config.trainer.grad_clip_norm
                )
                self.optimizer.apply_gradients(
                    grads, self.actor_model.trainable_parameters()
                )
                mx.eval(self.actor_model.parameters(), self.optimizer.state)
                self._log_memory("post_optim_step")

                avg_metrics = {
                    k: v / num_micro_batches for k, v in total_metrics.items()
                }
                avg_metrics["train/grad_norm"] = grad_norm
                self._last_logged_metrics.update(avg_metrics)
                if self.metrics_logger:
                    self.metrics_logger.log_metrics(avg_metrics, self.global_step)
                t_range.set_postfix(
                    {
                        "Reward": f"{avg_metrics.get('train/reward_mean', 0):.3f}",
                        "Loss": f"{avg_metrics.get('train/loss', 0):.4f}",
                    }
                )

                is_last_step = (
                    self.global_step == self.config.trainer.num_training_steps - 1
                )
                if (
                    self.config.trainer.validate_every > 0
                    and (self.global_step + 1) % self.config.trainer.validate_every == 0
                ) or is_last_step:
                    val_metrics = self._run_validation_loop(self.global_step)
                    self._last_logged_metrics.update(val_metrics)
                    if self.metrics_logger:
                        self.metrics_logger.log_metrics(val_metrics, self.global_step)

                if (
                    self.config.trainer.eval_every > 0
                    and (self.global_step + 1) % self.config.trainer.eval_every == 0
                ) or is_last_step:
                    eval_metrics = self.evaluate(self.global_step)
                    for m in eval_metrics:
                        self._last_logged_metrics.update(m.to_dict())
                        self.metrics_logger.log_metrics(m.to_dict(), self.global_step)

                if (
                    self.config.checkpointing.save_every > 0
                    and (self.global_step + 1) % self.config.checkpointing.save_every
                    == 0
                ) or is_last_step:
                    if self.metrics_logger:
                        _emit_plots_from_csv(
                            self.metrics_logger.file_path,
                            self.config.trainer.output_dir,
                            self.config,
                            self._run_id,
                        )
                    metric_for_best = self._last_logged_metrics.get(
                        self.config.checkpointing.best_model_metric
                    )
                    self.checkpoint_manager.save_checkpoint(
                        self.global_step,
                        self.actor_model,
                        self.optimizer,
                        {
                            "save_optimizer_state": self.config.checkpointing.save_optimizer_state
                        },
                        metric_for_best,
                    )

                self.global_step += 1
                t_range.update(1)
                self._aggressive_memory_cleanup()

        self.save_final_checkpoint(reason="completed")
