# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 003
# goals_of_writing_code_block: Merge two GRPOTrainer versions, implementing the missing abstract 'train_step' method and fixing dependencies to create a runnable trainer.
# type_of_code_response: replace code
"""
Concrete GRPO Trainer implementation.
"""
import gc
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm.models import cache  # For KV cache in generation
from mlx_lm.sample_utils import make_logits_processors  # For generation
from mlx_lm.tuner.utils import build_schedule  # For LR scheduling

from ...core.checkpoint_manager import CheckpointManager
from ...core.config import ExperimentConfig, RewardConfig
from ...core.model_manager import ModelManager
from ...core.trainer import (
    BaseTrainer,
    EvaluationMetrics,
    TrainingMetrics,
    TrainingRuntimeError,
)
from ...data.dataset_manager import DatasetManager
from ...evaluation.registry import EvaluatorRegistry
from ...generation.caching import PagedKVCache
from ...monitoring.metrics_logger import _maybe_log_samples, wandb_run
from ...rewards.base_reward import RewardComposer
from ...rewards.context import RewardContext
from ...utils.mlx_utils import (
    _create_4d_attention_mask,
    make_dynamic_tag_bias_processor,
    metal_safe_apply_gradients,
    safe_make_sampler,
)
from .grpo_algorithm import GRPOAlgorithm  # Import GRPOAlgorithm

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    """
    Concrete implementation of the GRPO (Group Relative Policy Optimization) Trainer.

    This class orchestrates the entire GRPO training loop, including model setup,
    rollout generation, loss calculation, optimization, evaluation, and checkpointing.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_manager: ModelManager,
        data_manager: DatasetManager,
        checkpoint_manager: CheckpointManager,
        reward_composer: RewardComposer,
        paged_kv_cache: Optional[PagedKVCache] = None,
    ):
        super().__init__(config, model_manager, data_manager, checkpoint_manager)

        self.reward_composer = reward_composer
        self.grpo_algorithm: Optional[GRPOAlgorithm] = None
        self.paged_kv_cache = paged_kv_cache

        self.tokenizer: Any = None
        self.actor_model: Optional[nn.Module] = None
        self.ref_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.lr_scheduler: Optional[Callable[[int], float]] = None

        self._current_update_step: int = 0
        self._current_epoch: int = 0
        self._run_id: str = str(uuid.uuid4())
        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        """
        Performs all necessary setup: loads models, initializes tokenizer,
        sets up optimizer and LR scheduler, and resumes from a checkpoint if specified.
        """
        from rich import print as rprint

        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")

        actor_path = self.config.model.model_path
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            actor_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config={
                "rank": self.config.model.lora_rank,
                "alpha": self.config.model.lora_alpha,
                "dropout": self.config.model.lora_dropout,
                "scale_by_rank": self.config.model.lora_scale_by_rank,
                "target_modules": self.config.model.lora_target_modules,
            },
        )

        ref_model_path = self.config.model.ref_model_path
        self.ref_model, _ = self.model_manager.load_model(
            ref_model_path, "reference", is_trainable=False
        )
        self.data_manager.tokenizer = self.tokenizer

        rprint(
            f"Loaded actor model: [green]{self.actor_model.__class__.__name__}[/green], ref model: [green]{self.ref_model.__class__.__name__}[/green]."
        )

        self.grpo_algorithm = GRPOAlgorithm(
            self.config, self.actor_model, self.ref_model
        )

        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(
                self.config.trainer.optimizer_beta1,
                self.config.trainer.optimizer_beta2,
            ),
            weight_decay=self.config.trainer.optimizer_weight_decay,
        )
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)
        rprint(
            f"Optimizer: [cyan]{self.optimizer.__class__.__name__}[/cyan], Learning Rate: [cyan]{self.config.trainer.learning_rate:.2e}[/cyan]."
        )

        start_update_step, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )
        self.global_step = start_update_step
        self.current_epoch = metadata.get("epoch", 0)

        rprint(
            f"[bold green]Trainer setup complete.[/bold green] Resuming from update step [bold magenta]{self.global_step}[/bold magenta], epoch [bold magenta]{self.current_epoch}[/bold magenta]."
        )
        return self.global_step, self.current_epoch

    def train_step(
        self, batch: List[Dict[str, Any]], update_step: int, epoch: int
    ) -> TrainingMetrics:
        """
        Performs a single GRPO training step: rollout, loss calculation, and gradient update.
        """
        start_time = time.time()

        if (
            self.actor_model is None
            or self.optimizer is None
            or self.lr_scheduler is None
        ):
            raise TrainingRuntimeError(
                "Trainer is not set up. Call _setup() before training."
            )

        lr = self.lr_scheduler(update_step)
        self.optimizer.learning_rate = lr

        rollout_data, avg_raw_reward, raw_reward_components = self.generate_rollouts(
            batch, update_step
        )

        if not rollout_data:
            logger.warning(
                f"Skipping update step {update_step} due to empty rollout batch."
            )
            return TrainingMetrics(
                loss=0.0,
                mean_reward=0.0,
                learning_rate=lr,
                step_time_s=time.time() - start_time,
                tokens_per_sec=0.0,
            )

        loss, grads, metrics = self._calculate_grpo_loss_and_grads(rollout_data, self.tokenizer.pad_token_id)

        metal_safe_apply_gradients(self.optimizer, self.actor_model, grads)

        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        end_time = time.time()
        step_time = end_time - start_time

        # Calculate prompt_tokens from the original batch (prompts_batch)
        prompt_tokens = sum(sample["input_ids"].shape[0] for sample in batch)
        generated_tokens = int(mx.sum(rollout_data["response_mask"]).item())
        total_tokens = prompt_tokens + generated_tokens
        tokens_per_sec = total_tokens / step_time if step_time > 0 else 0.0

        final_metrics = {
            "loss": loss.item(),
            "reward_mean": avg_raw_reward,
            "grad_norm": avg_grad_norm,
            "learning_rate": lr,
            "step_time_s": step_time,
            "tokens_per_sec": tokens_per_sec,
            **metrics,
            **raw_reward_components,
        }

        return TrainingMetrics(**final_metrics)

    def generate_rollouts(
        self, batch_prompts_data: List[Dict[str, Any]], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """Generates rollouts using the actor model and computes rewards."""
        if self.actor_model is None or self.tokenizer is None or self.ref_model is None:
            raise TrainingRuntimeError(
                "Models or tokenizer not initialized for rollout generation."
            )
        logging.debug(f"generate_rollouts received batch_prompts_data: {batch_prompts_data}")
        # Extract input_ids and original raw data from the processed batch
        prompts_list_of_arrays = []
        for idx, sample in enumerate(batch_prompts_data):
            if not isinstance(sample, dict):
                logging.critical(f"Expected sample to be a dict, but got {type(sample)} at index {idx}. Content: {sample}")
                raise TypeError("Expected sample to be a dict")
            prompts_list_of_arrays.append(sample["input_ids"])
        logging.debug(f"Type of batch_prompts_data: {type(batch_prompts_data)}")
        logging.debug(f"Content of batch_prompts_data (first item): {batch_prompts_data[0] if batch_prompts_data else 'empty'}")
        logging.debug(f"Type of sample (first item): {type(batch_prompts_data[0]) if batch_prompts_data else 'empty'}")
        original_raw_prompts_data = [sample["original_raw_data"] for sample in batch_prompts_data]

        if not prompts_list_of_arrays:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)
        padded_prompts_mx = mx.array(
            [
                mx.concatenate(
                    [
                        arr,
                        mx.full(
                            (max_prompt_len_mb - arr.shape[0],),
                            self.tokenizer.pad_token_id,
                            dtype=mx.int32,
                        ),
                    ],
                    axis=0,
                )
                for arr in prompts_list_of_arrays
            ],
            dtype=mx.int32,
        )

        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        total_samples = len(prompts_list_of_arrays) * num_samples_per_prompt
        prompts_rep = mx.repeat(
            padded_prompts_mx, repeats=num_samples_per_prompt, axis=0
        )

        # Generate rollouts
        self.actor_model.eval()
        # Simplified generation loop for clarity; a real implementation might be more complex
        # We will use the generate function from the model if available, otherwise a manual loop
        responses_mx, actor_lp_resp = self.model_manager.generate_with_logprobs(
            model=self.actor_model,
            prompts=prompts_rep,
            tokenizer=self.tokenizer,
            temp=self.config.trainer.answer_temperature,
            max_tokens=self.config.data.max_gen_len,
        )
        self.actor_model.train()

        decoded_responses = self.tokenizer.batch_decode(
            responses_mx.tolist(), skip_special_tokens=True
        )

        rewards_total = []
        # Use original_raw_prompts_data here
        for i in range(total_samples):
            prompt_idx = i // num_samples_per_prompt
            prompt_data_original = original_raw_prompts_data[prompt_idx] # This is the original dict
            logging.debug(prompt_data_original)
            context = RewardContext(
                generated_text=decoded_responses[i],
                prompt_text=prompt_data_original.get(self.config.data.dataset_prompt_key, ""),
                reference_completion=prompt_data_original.get(self.config.data.dataset_answer_key, ""),
                test_cases=prompt_data_original.get("raw_test_cases", []),
                metadata=prompt_data_original.get("meta", {}),
            )
            rewards_dict = self.reward_composer.compute(context)
            rewards_total.append(rewards_dict.get("total", 0.0))

        rewards_total_mx = mx.array(rewards_total, dtype=mx.float32)
        advantages = self.grpo_algorithm.compute_advantages(
            rewards_total_mx, num_samples_per_prompt
        )
        resp_mask = (responses_mx != self.tokenizer.pad_token_id).astype(mx.float32)

        # Get reference log probs
        ref_lp_resp = self.model_manager.get_logprobs_for_sequence(
            self.ref_model, prompts_rep, responses_mx
        )

        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "actor_log_probs": actor_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx,
        }

        avg_reward = float(np.mean(rewards_total)) if rewards_total else 0.0
        return rollout_data_for_loss, avg_reward, {}

    def _calculate_grpo_loss_and_grads(
        self, rollout_batch: Dict[str, mx.array], pad_token_id: int
    ) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        """Calculates the GRPO loss and gradients using the GRPOAlgorithm."""
        if self.grpo_algorithm is None:
            raise TrainingRuntimeError("GRPOAlgorithm not initialized.")
        return self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config, pad_token_id)

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        """Runs registered evaluators against the validation dataset."""
        results: List[EvaluationMetrics] = []
        val_dataset = self.data_manager._val_dataset

        if val_dataset is None or len(val_dataset) == 0:
            logging.warning("Validation skipped: no validation dataset available.")
            return []

        eval_gen_config = self.config.generation.model_dump()
        eval_gen_config.update({"system_prompt": self.config.system_prompt})

        for eval_cfg in self.config.evaluation:
            try:
                merged_eval_config = eval_cfg.config.copy()
                merged_eval_config.update(eval_gen_config)
                evaluator = EvaluatorRegistry.create(eval_cfg.name, merged_eval_config)
                eval_output = evaluator.evaluate(
                    self.actor_model, self.tokenizer, val_dataset
                )
                results.append(eval_output)
            except Exception as e:
                logging.error(
                    f"Error during evaluation '{eval_cfg.name}': {e}", exc_info=True
                )
        return results
