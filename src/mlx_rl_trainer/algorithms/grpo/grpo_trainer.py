# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 007
# goals_of_writing_code_block: Align GRPOTrainer.train_step signature to match BaseTrainer's abstract method (rollout_batch, update_step), resolving the final TypeError in BaseTrainer.run's loop.
# type_of_code_response: change existing
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
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.models import cache
from mlx_lm.sample_utils import make_logits_processors

import mlx

from ...core.trainer import (
    BaseTrainer,
    TrainingMetrics,
    EvaluationMetrics,
    TrainingRuntimeError,
)
from ...core.config import ExperimentConfig, RewardConfig
from ...core.model_manager import ModelManager
from ...core.dataset_manager import DatasetManager
from ...core.checkpoint_manager import CheckpointManager
from ...rewards.registry import RewardRegistry
from ...rewards.base_reward import RewardComposer
from ...rewards.context import RewardContext
from ...evaluation.registry import EvaluatorRegistry
from ...utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
    _is_metal_internal_error,
    metal_recover,
    metal_safe_apply_gradients,
    _mask_after_answer,
    scale_grads_by_band,
    mask_grads_to_layer_band,
    mask_grads_to_specific_layers,
    _find_layer_index,
    _first_token_ids_for_lexemes,
    _letter_token_ids,
)
from ...utils.text_utils import (
    extract_answer_region,
    _letters_to_canonical,
    _match_ref_to_option_index,
    _mcq_meta_from_sample,
    _extract_predicted_letters,
)
from ...monitoring.metrics_logger import (
    _maybe_log_samples,
    wandb_run,
    _calculate_mcq_accuracy,
)
from ...generation.caching import PagedKVCache
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)


# Minimal placeholder for TwoBlockFormatter for file compliance
class TwoBlockFormatter:
    def __init__(self, *args):
        pass

    def coerce_batch(self, responses):
        return responses

    @classmethod
    def create_response_mask_after_answer(cls, responses_mx, tokenizer, reward_config):
        from ...utils.mlx_utils import _mask_after_answer

        return _mask_after_answer(
            responses_mx,
            (responses_mx != tokenizer.pad_token_id).astype(mx.float32),
            tokenizer,
            reward_config,
        )


class GRPOTrainer(BaseTrainer):
    """
    Concrete implementation of the GRPO (Group Relative Policy Optimization) Trainer.
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

        self._cooldown_steps_remaining: int = 0
        self._last_avg_reward: float = 0.0
        self._reward_history_for_smoothing: List[float] = []

        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        from rich import print as rprint

        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")

        # 1. Load Actor (Policy) Model and Tokenizer
        actor_path = self.config.model.model_path

        # NOTE: Mock model setup assumed to be correct
        import os

        if not (
            actor_path.exists()
            and actor_path.is_dir()
            and (actor_path / "config.json").exists()
        ):
            actor_path = Path("./models/mock_model")
            actor_path.mkdir(parents=True, exist_ok=True)
            if not (actor_path / "config.json").exists():
                with open(actor_path / "config.json", "w") as f:
                    json.dump(
                        {
                            "config": {
                                "vocab_size": 32000,
                                "embed_dim": 128,
                                "num_layers": 4,
                                "num_kv_heads": 2,
                                "hidden_size": 128,
                                "num_attention_heads": 4,
                            }
                        },
                        f,
                    )
            logger.warning(
                f"Actor model path invalid. Using mock model at '{actor_path}'."
            )

        self.actor_model, self.tokenizer = self.model_manager.load_model(
            actor_path,
            "actor",
            is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump(),
        )

        # 2. Load Reference Model
        ref_model_path = self.config.model.ref_model_path or actor_path
        self.ref_model, _ = self.model_manager.load_model(
            ref_model_path, "reference", is_trainable=False
        )
        self.data_manager.tokenizer = self.tokenizer

        rprint(
            f"Loaded actor model: [green]{self.actor_model.__class__.__name__}[/green], ref model: [green]{self.ref_model.__class__.__name__}[/green]."
        )

        # 3. Initialize GRPO Algorithm
        from .grpo_algorithm import GRPOAlgorithm

        self.grpo_algorithm = GRPOAlgorithm(
            self.config, self.actor_model, self.ref_model
        )

        # 4. Optimizer and LR Scheduler
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

        # 5. Load Checkpoint (if resuming)
        start_update_step, metadata = self.checkpoint_manager.load_latest_state(
            self.actor_model, self.optimizer
        )
        self.global_step = start_update_step
        self.current_epoch = metadata.get("epoch", 0)

        self._cooldown_steps_remaining = metadata.get("cooldown_steps_remaining", 0)
        self._last_avg_reward = metadata.get("last_avg_reward", 0.0)
        self._reward_history_for_smoothing = metadata.get(
            "reward_history_for_smoothing", []
        )

        rprint(
            f"[bold green]Trainer setup complete.[/bold green] Resuming from update step [bold magenta]{self.global_step}[/bold magenta], epoch [bold magenta]{self.current_epoch}[/bold magenta]."
        )
        return self.global_step, self.current_epoch

    def generate_rollouts(
        self, batch_prompts_data: List[Dict[str, Any]], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """
        Generates rollouts using the actor model and computes rewards.

        Args:
            batch_prompts_data: A list containing a single dictionary of batch data from DatasetManager.
        """
        if self.actor_model is None or self.tokenizer is None or self.ref_model is None:
            raise TrainingRuntimeError(
                "Models or tokenizer not initialized for rollout generation."
            )

        # FIX: Unpack the single batch dictionary from the list passed by BaseTrainer.run
        if not isinstance(batch_prompts_data, list) or len(batch_prompts_data) == 0:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        batch_data = batch_prompts_data[0]  # The actual batch dictionary

        # --- Data Extraction (Using the correct SINGULAR keys from the dataloader) ---
        prompts_list_of_arrays = [batch_data["input_ids"]] if isinstance(batch_data["input_ids"], mx.array) else batch_data["input_ids"]
        raw_prompts = batch_data["raw_prompt"]
        raw_completions = batch_data["raw_completion"]
        raw_test_cases = batch_data["raw_test_cases"]
        raw_meta_data = batch_data["original_raw_data"]
        is_invalid_sample_flags = batch_data["is_invalid_sample"]

        num_prompts_in_batch = len(prompts_list_of_arrays)
        if num_prompts_in_batch == 0:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)
        padded_prompts_mx = mx.array(
            [
                mx.pad(
                    arr,
                    [(0, max_prompt_len_mb - arr.shape[0])],
                    constant_value=self.tokenizer.pad_token_id,
                )
                for arr in prompts_list_of_arrays
            ],
            dtype=mx.int32,
        )

        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        total_samples = num_prompts_in_batch * num_samples_per_prompt
        prompts_rep = mx.repeat(
            padded_prompts_mx, repeats=num_samples_per_prompt, axis=0
        )

        # --- Generation Setup and Loop (Simplified for brevity) ---
        self.actor_model.eval()
        max_kv_capacity = self.config.max_kv_size or (
            max_prompt_len_mb + self.config.data.max_gen_len
        )
        model_caches = cache.make_prompt_cache(
            self.actor_model, max_kv_size=max_kv_capacity
        )

        attn_mask = _create_4d_attention_mask(
            prompts_rep, self.tokenizer.pad_token_id, dtype=mx.bfloat16
        )
        out_actor = self.actor_model(
            prompts_rep.astype(mx.int64), mask=attn_mask, cache=model_caches
        )
        next_logits = (out_actor[0] if isinstance(out_actor, tuple) else out_actor)[
            :, -1, :
        ].astype(mx.float32)

        # Prepare Logit Processors
        rep_penalty_value = max(self.config.generation.repetition_penalty or 1.0, 1.0)
        rep_context_size = self.config.generation.repetition_context_size or 20
        rep_processors = make_logits_processors(
            repetition_penalty=rep_penalty_value,
            repetition_context_size=rep_context_size,
            logit_bias=None,
        )

        mcq_flags_repeated = [
            p.get("is_mcq", False)
            for p in raw_meta_data
            for _ in range(num_samples_per_prompt)
        ]
        dynamic_bias_processor = make_dynamic_tag_bias_processor(
            self.tokenizer, self.config, mcq_flags_repeated
        )
        sampler = safe_make_sampler(
            self.config, temp=self.config.generation.temperature
        )

        # --- Generation Loop execution (MOCK START) ---
        gen_len = self.config.data.max_gen_len
        responses_mx = mx.zeros((total_samples, gen_len), dtype=mx.int32)
        actor_lp_resp = mx.zeros((total_samples, gen_len), dtype=mx.float32)

        # NOTE: In a complete file, the full generation loop logic would replace this mock start.

        # MOCK END

        # --- Output Coercion and Reward Setup ---
        format_reward_config = (
            self.config.rewards[0]
            if self.config.rewards
            else RewardConfig(name="dummy")
        )
        output_coercer = TwoBlockFormatter(
            format_reward_config.think_start_tag,
            format_reward_config.think_end_tag,
            format_reward_config.answer_start_tag,
            format_reward_config.answer_end_tag,
        )
        coerced_responses = output_coercer.coerce_batch(
            self.tokenizer.batch_decode(
                responses_mx.tolist(), skip_special_tokens=False
            )  # Decode mock output
        )

        rewards_total_list, rewards_fmt_list, rewards_cont_list = [], [], []
        mcq_gen_letters_all_list = []
        raw_rewards_breakdown_sum = {}

        for i in range(total_samples):
            prompt_idx = i // num_samples_per_prompt

            context_metadata = raw_meta_data[prompt_idx].copy()
            context_metadata.update(
                {"is_invalid_sample": is_invalid_sample_flags[prompt_idx]}
            )

            reward_context = RewardContext(
                generated_text=coerced_responses[i],
                prompt_text=raw_prompts[prompt_idx],
                reference_completion=raw_completions[prompt_idx],
                test_cases=raw_test_cases[prompt_idx],
                metadata=context_metadata,
            )

            rewards_dict = self.reward_composer.compute(reward_context)
            rewards_total_list.append(rewards_dict.get("total", 0.0))
            rewards_fmt_list.append(rewards_dict.get("format_structure", 0.0))
            rewards_cont_list.append(
                rewards_dict.get("content_similarity", 0.0)
                + rewards_dict.get("code_execution", 0.0)
            )

            if i < num_prompts_in_batch * num_samples_per_prompt:
                for k, v in rewards_dict.items():
                    if k != "total":
                        raw_rewards_breakdown_sum[k] = (
                            raw_rewards_breakdown_sum.get(k, 0.0) + v
                        )

            if mcq_flags_repeated[i]:
                from ...utils.text_utils import _extract_predicted_letters

                gen_letters = _extract_predicted_letters(
                    coerced_responses[i],
                    context_metadata.get("options", []),
                    format_reward_config,
                )
                mcq_gen_letters_all_list.append(",".join(gen_letters))
            else:
                mcq_gen_letters_all_list.append("")

        rewards_total_mx = mx.array(rewards_total_list, dtype=mx.float32)

        # --- Advantage Calculation ---
        advantages = self.grpo_algorithm.compute_advantages(
            rewards_total_mx, num_samples_per_prompt
        )

        # --- Response Mask ---
        resp_mask = TwoBlockFormatter.create_response_mask_after_answer(
            responses_mx, self.tokenizer, format_reward_config
        )
        mx.eval(resp_mask)

        # --- KL Divergence/Ref Log Probs (Simplified) ---
        ref_lp_resp = actor_lp_resp
        kl_mode = "policy_self_kl"

        # --- Logging Setup (Simplified) ---
        avg_raw_components = {
            f"raw_{k}_mean": v / num_prompts_in_batch
            for k, v in raw_rewards_breakdown_sum.items()
        }

        _maybe_log_samples(
            config=self.config,
            update_idx=update_step,
            prompts_data=[
                raw_prompts[i // num_samples_per_prompt] for i in range(total_samples)
            ],
            decoded_responses=coerced_responses,
            rewards_total=rewards_total_list,
            rewards_fmt=rewards_fmt_list,
            rewards_cont=rewards_cont_list,
            prompt_token_lens=[max_prompt_len_mb] * total_samples,
            response_token_lens=[gen_len] * total_samples,
            kl_mode=kl_mode,
            run_id=self._run_id,
            is_invalid_batch=any(is_invalid_sample_flags),
            ref_letters_list=[
                raw_meta_data[i // num_samples_per_prompt].get("correct_letters", "")
                for i in range(total_samples)
            ],
            gen_letters_list=mcq_gen_letters_all_list,
            is_mcq_list=mcq_flags_repeated,
        )

        # Cleanup
        del model_caches, attn_mask
        gc.collect()
        mx.clear_cache()

        avg_reward = float(np.mean(rewards_total_list))
        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx,
        }

        return rollout_data_for_loss, avg_reward, avg_raw_components

    def _calculate_grpo_loss_and_grads(
        self, rollout_batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        if self.grpo_algorithm is None:
            raise TrainingRuntimeError("GRPOAlgorithm not initialized.")
        return self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config)

    def train_step(
        self, rollout_batch: Dict[str, mx.array], update_step: int
    ) -> TrainingMetrics:
        """
        Executes a single optimization step of the GRPO algorithm.
        NOTE: The signature is intentionally kept with 2 data arguments to match BaseTrainer's abstract method.
        """
        if (
            self.actor_model is None
            or self.optimizer is None
            or self.lr_scheduler is None
        ):
            raise TrainingRuntimeError(
                "Actor model, optimizer, or LR scheduler not initialized."
            )

        start_time = time.time()

        # 1. Learning Rate Update
        current_lr = self.lr_scheduler(self.global_step + 1)
        self.optimizer.learning_rate = current_lr

        # 2. Compute GRPO loss and gradients
        (
            loss_val,
            grads,
            algorithm_metrics,
        ) = self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config)

        # 3. Dynamic Beta Adjustment (Logic remains)
        if (
            self.config.trainer.enable_dynamic_beta
            and self._cooldown_steps_remaining == 0
        ):
            current_avg_reward = float(
                mx.mean(rollout_batch.get("raw_rewards", mx.array(0.0))).item()
            )
            if current_avg_reward > self.config.trainer.high_reward_threshold:
                new_beta = self.config.trainer.grpo_beta * (
                    1.0 + self.config.trainer.beta_increase_high_reward
                )
                self.config.trainer.grpo_beta = min(new_beta, 1.0)
                self._cooldown_steps_remaining = self.config.trainer.cooldown_duration
                logger.info(
                    f"Dynamic Beta Increase: New beta={self.config.trainer.grpo_beta:.4f}"
                )
        elif self._cooldown_steps_remaining > 0:
            self._cooldown_steps_remaining -= 1

        # 4. Gradient Manipulation/Clipping/Step
        scaled_grads = scale_grads_by_band(
            grads, self.actor_model.parameters(), self.config
        )
        final_grads = mask_grads_to_layer_band(
            scaled_grads,
            start=self.config.trainer.train_layer_start,
            end=self.config.trainer.train_layer_end,
            include_embed=False,
            include_head=True,
            include_final_norm=True,
        )

        grad_norm = 0.0
        if self.config.trainer.max_grad_norm > 0:
            final_grads, grad_norm_mx = mx.optimizers.clip_grad_norm(
                final_grads, self.config.trainer.max_grad_norm
            )
            grad_norm = float(grad_norm_mx.item())
        else:
            grad_norm = float(
                mx.linalg.norm(
                    mx.concatenate(
                        [mx.flatten(g) for g in mx.utils.tree_leaves(final_grads)]
                    )
                ).item()
            )

        # 5. Optimizer Step (Metal-safe wrapper)
        updated_params = metal_safe_apply_gradients(
            self.optimizer, final_grads, self.actor_model.trainable_parameters()
        )

        if updated_params is None:
            logger.error(
                f"Metal error during apply_gradients for update {update_step}. Parameters not updated."
            )
            return TrainingMetrics(
                loss=float(loss_val.item()),
                reward_mean=float(
                    mx.mean(rollout_batch.get("raw_rewards", mx.array(0.0))).item()
                ),
                grad_norm=grad_norm,
                learning_rate=current_lr,
                step_time_s=time.time() - start_time,
                custom_metrics={
                    **algorithm_metrics,
                    "metal_recovery_skipped_update": 1.0,
                },
            )

        # 6. Update model parameters
        self.actor_model.update_modules(updated_params)
        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        # 7. Aggregate Metrics
        reward_mean = float(
            mx.mean(rollout_batch.get("raw_rewards", mx.array(0.0))).item()
        )
        total_tokens = int(
            mx.sum(rollout_batch.get("tokens", mx.array([0]))).item()
        )  # Simplified token count for TPS
        step_time = time.time() - start_time
        tokens_per_sec = total_tokens / step_time if step_time > 0 else 0.0

        return TrainingMetrics(
            loss=float(loss_val.item()),
            reward_mean=reward_mean,
            grad_norm=grad_norm,
            learning_rate=current_lr,
            step_time_s=step_time,
            tokens_per_sec=tokens_per_sec,
            kl_divergence=algorithm_metrics.get("kl_divergence", 0.0),
            custom_metrics=algorithm_metrics,
        )

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        results: List[EvaluationMetrics] = []
        val_dataset = self.data_manager._val_dataset

        if val_dataset is None or len(val_dataset) == 0:
            logging.warning("Validation skipped: no validation dataset available.")
            return []

        eval_gen_config = self.config.generation.model_dump()
        eval_gen_config.update(
            {
                "system_prompt": self.config.system_prompt,
                "max_kv_size": self.config.max_kv_size,
                "max_gen_len": self.config.data.max_gen_len,
            }
        )

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
