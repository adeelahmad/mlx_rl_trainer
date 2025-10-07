# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 006
# goals_of_writing_code_block: Fix NameError for PagedKVCache and clean up utility imports/usage to ensure module stability.
# type_of_code_response: replace code
"""
Concrete GRPO Trainer implementation.
"""
import uuid
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from rich import print as rprint
import gc
import mlx.utils  # Ensure utils is imported for tree_flatten

from mlx_lm.tuner.utils import build_schedule
from mlx_lm.models import cache
from mlx_lm.sample_utils import make_logits_processors

from ...core.trainer import (
    BaseTrainer,
    TrainingMetrics,
    EvaluationMetrics,
    TrainingRuntimeError,
)
from ...core.config import ExperimentConfig, RewardConfig
from ...core.model_manager import ModelManager
from ...data.dataset_manager import DatasetManager
from ...core.checkpoint_manager import CheckpointManager
from ...rewards.registry import RewardRegistry
from ...rewards.base_reward import RewardComposer
from ...rewards.context import RewardContext
from ...evaluation.registry import EvaluatorRegistry
from ...generation.caching import PagedKVCache  # FIX: PagedKVCache imported here
from ...utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
    metal_safe_apply_gradients,
    scale_grads_by_band,
    mask_grads_to_layer_band,
)
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


# Helper to calculate global grad norm
def _global_grad_norm_helper(grads: Dict[str, mx.array]) -> float:
    flat = [g for _, g in mx.utils.tree_flatten(grads)]
    if not flat:
        return 0.0
    total = mx.sqrt(mx.add_n(*[mx.sum(g**2) for g in flat]))
    mx.eval(total)
    return float(total.item())


class GRPOTrainer(BaseTrainer):
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
        self._run_id: str = str(uuid.uuid4())
        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        from rich import print as rprint

        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")

        actor_path = self.config.model.model_path
        self.actor_model, self.tokenizer = self.model_manager.load_model(
            actor_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
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
        self, batch_prompts_data: List[Dict[str, Any]], update_step: int, epoch: int
    ) -> TrainingMetrics:
        start_time = time.time()

        if (
            self.actor_model is None
            or self.optimizer is None
            or self.lr_scheduler is None
        ):
            raise TrainingRuntimeError("Trainer is not set up properly.")

        lr = self.lr_scheduler(update_step)
        self.optimizer.learning_rate = lr

        rollout_batch, avg_raw_reward, raw_rewards_breakdown = self.generate_rollouts(
            batch_prompts_data, update_step
        )

        if not rollout_batch or rollout_batch["tokens"].size == 0:
            return TrainingMetrics(
                loss=0.0,
                reward_mean=0.0,
                grad_norm=0.0,
                learning_rate=lr,
                step_time_s=0.0,
                tokens_per_sec=0.0,
                kl_divergence=0.0,
                custom_metrics=raw_rewards_breakdown,
            )

        loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(
            rollout_batch, self.config, self.tokenizer.pad_token_id
        )

        if not mx.isfinite(loss):
            logger.warning(f"Non-finite loss detected: {loss.item()}. Skipping update.")
            return TrainingMetrics(
                loss=loss.item(),
                reward_mean=avg_raw_reward,
                grad_norm=0.0,
                learning_rate=lr,
                step_time_s=0.0,
                tokens_per_sec=0.0,
                kl_divergence=metrics.get("kl_divergence", 0.0),
                custom_metrics=raw_rewards_breakdown,
            )

        # 1. Apply gradient clipping
        if self.config.trainer.grad_clip_norm:
            grads, _ = mx.utils.clip_grad_norm(
                grads, self.config.trainer.grad_clip_norm
            )

        # 2. Apply layer masking
        grads = mask_grads_to_layer_band(
            grads,
            self.config.trainer.train_layer_start,
            self.config.trainer.train_layer_end,
        )

        # 3. Apply gradient scaling (banding)
        grads = scale_grads_by_band(grads, self.config)

        # 4. Calculate final grad norm
        grad_norm_val = _global_grad_norm_helper(grads)

        # 5. Apply gradients
        metal_safe_apply_gradients(self.optimizer, self.actor_model, grads)

        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        end_time = time.time()
        step_time = end_time - start_time

        # Calculate tokens
        prompt_tokens = sum(sample["input_ids"].size for sample in batch_prompts_data)
        generated_tokens = int(mx.sum(rollout_batch["response_mask"]).item())
        total_tokens = prompt_tokens + generated_tokens
        tokens_per_sec = total_tokens / step_time if step_time > 0 else 0.0

        final_metrics = {
            "loss": loss.item(),
            "reward_mean": avg_raw_reward,
            "grad_norm": grad_norm_val,
            "learning_rate": lr,
            "step_time_s": step_time,
            "tokens_per_sec": tokens_per_sec,
            "kl_divergence": metrics.get("kl_divergence", 0.0),
            "custom_metrics": {**metrics, **raw_rewards_breakdown},
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

        prompts_list_of_arrays = [sample["input_ids"] for sample in batch_prompts_data]
        original_raw_prompts_data = [
            sample["original_raw_data"] for sample in batch_prompts_data
        ]

        if not prompts_list_of_arrays:
            return {}, 0.0, {}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)
        padded_prompts_mx = mx.array(
            [
                mx.pad(
                    arr,
                    [(0, max_prompt_len_mb - arr.shape[0])],
                    constant_values=self.tokenizer.pad_token_id,
                )
                for arr in prompts_list_of_arrays
            ],
            dtype=mx.int32,
        )

        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        prompts_rep = mx.repeat(
            padded_prompts_mx, repeats=num_samples_per_prompt, axis=0
        )
        total_samples = prompts_rep.shape[0]

        self.actor_model.eval()

        # Generation
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
        coerced_responses = (
            decoded_responses  # In the advanced flow, coercion happens here.
        )

        rewards_total, rewards_fmt, rewards_cont = [], [], []

        reward_cfg = (
            self.config.rewards[0]
            if self.config.rewards
            else RewardConfig(name="default")
        )

        for i in range(total_samples):
            prompt_idx = i // num_samples_per_prompt
            original_data = original_raw_prompts_data[prompt_idx]

            test_cases_loaded = []
            for tc_str in original_data.get("test_cases", []):
                try:
                    # Test cases are expected to be JSON strings if loaded from DatasetManager
                    test_cases_loaded.append(json.loads(tc_str))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed test case in reward context.")

            context = RewardContext(
                generated_text=coerced_responses[i],
                prompt_text=original_data.get(self.config.data.dataset_prompt_key, ""),
                reference_completion=original_data.get(
                    self.config.data.dataset_answer_key, ""
                ),
                test_cases=test_cases_loaded,
                metadata=original_data.get("meta", {}),
            )

            rewards_dict = self.reward_composer.compute(context)

            rewards_total.append(rewards_dict.get("total", 0.0))
            rewards_fmt.append(rewards_dict.get("format_structure", 0.0))
            rewards_cont.append(
                rewards_dict.get(
                    "code_execution", rewards_dict.get("content_similarity", 0.0)
                )
            )

        rewards_total_mx = mx.array(rewards_total, dtype=mx.float32)
        advantages = self.grpo_algorithm.compute_advantages(
            rewards_total_mx, num_samples_per_prompt
        )
        resp_mask = (responses_mx != self.tokenizer.pad_token_id).astype(mx.float32)

        ref_lp_resp = self.model_manager.get_logprobs_for_sequence(
            self.ref_model, prompts_rep, responses_mx
        )

        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx,
        }

        avg_reward = float(np.mean(rewards_total)) if rewards_total else 0.0

        # del model_caches
        gc.collect()
        mx.clear_cache()

        return (
            rollout_data_for_loss,
            avg_reward,
            {
                "raw_format": np.mean(rewards_fmt) if rewards_fmt else 0.0,
                "raw_content_combined": np.mean(rewards_cont) if rewards_cont else 0.0,
            },
        )

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
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
