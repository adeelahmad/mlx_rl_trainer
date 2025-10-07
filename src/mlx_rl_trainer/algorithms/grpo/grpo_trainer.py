"""
Concrete GRPO Trainer implementation.
"""

import uuid
import logging
import time
import json
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from rich import print as rprint
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
from ...core.dataset_manager import DatasetManager
from ...core.checkpoint_manager import CheckpointManager
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
)
from ...utils.text_utils import (
    _extract_final_numeric,
    _normalize_ans_for_match,
    _first_token_ids_for_lexemes,
    _letter_token_ids,
    TwoBlockFormatter,
)
from ...monitoring.metrics_logger import (
    _maybe_log_samples,
    wandb_run,
)
from ...generation.caching import PagedKVCache
from .grpo_algorithm import GRPOAlgorithm

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

        self._cooldown_steps_remaining: int = 0
        self._last_avg_reward: float = 0.0
        self._reward_history_for_smoothing: List[float] = []

        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        """
        Performs all necessary setup: loads models, initializes tokenizer,
        sets up optimizer and LR scheduler, and resumes from a checkpoint if specified.

        Returns:
            A tuple of (start_update_step, start_epoch).
        """
        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")

        if (
            self.config.model.model_path.exists()
            and self.config.model.model_path.is_dir()
        ):
            actor_path = self.config.model.model_path
        else:
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
            lora_config={
                "rank": self.config.model.lora_rank,
                "alpha": self.config.model.lora_alpha,
                "dropout": self.config.model.lora_dropout,
                "scale_by_rank": self.config.model.lora_scale_by_rank,
                "target_modules": self.config.model.lora_target_modules,
            },
        )

        ref_model_path = self.config.model.ref_model_path
        if (
            not ref_model_path
            or not ref_model_path.exists()
            or not ref_model_path.is_dir()
        ):
            ref_model_path = Path("./models/mock_model")
        self.ref_model, _ = self.model_manager.load_model(
            ref_model_path, "reference", is_trainable=False
        )
        self.data_manager.tokenizer = self.tokenizer

        rprint(
            f"Loaded actor model: [green]{self.actor_model.__class__.__name__}[/green], "
            f"ref model: [green]{self.ref_model.__class__.__name__}[/green]."
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
            f"Optimizer: [cyan]{self.optimizer.__class__.__name__}[/cyan], "
            f"Learning Rate: [cyan]{self.config.trainer.learning_rate:.2e}[/cyan]."
        )

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
            f"[bold green]Trainer setup complete.[/bold green] "
            f"Resuming from update step [bold magenta]{self.global_step}[/bold magenta], "
            f"epoch [bold magenta]{self.current_epoch}[/bold magenta]."
        )
        return self.global_step, self.current_epoch

    def generate_rollouts(
        self, batch_prompts_data: List[Dict[str, Any]], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """
        Generates rollouts using the actor model and computes rewards.

        Args:
            batch_prompts_data: List containing a single dictionary of batch data from DatasetManager.
            update_step: The current training update step.

        Returns:
            Tuple: (rollout_data_dict, average_reward, raw_reward_components_dict)
        """
        if self.actor_model is None or self.tokenizer is None or self.ref_model is None:
            raise TrainingRuntimeError(
                "Models or tokenizer not initialized for rollout generation."
            )

        if not isinstance(batch_prompts_data, list) or len(batch_prompts_data) == 0:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        batch_data = batch_prompts_data[0]

        prompts_list_of_arrays = batch_data["input_ids"]
        raw_prompts = batch_data["raw_prompts"]
        raw_completions = batch_data["raw_completions"]
        raw_test_cases = batch_data["raw_test_cases"]
        is_invalid_sample_flags = batch_data["is_invalid_sample_flags"]

        num_prompts_in_batch = len(prompts_list_of_arrays)
        if num_prompts_in_batch == 0:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)

        padded_arrs = []
        for arr in prompts_list_of_arrays:
            padding_amount = max_prompt_len_mb - arr.shape[0]
            if padding_amount > 0:
                padded_arr = mx.pad(
                    arr,
                    [(0, padding_amount)],
                    constant_values=self.tokenizer.pad_token_id,
                )
                padded_arrs.append(padded_arr)
            else:
                padded_arrs.append(arr)

        padded_prompts_mx = mx.stack(padded_arrs, axis=0)

        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        total_samples = num_prompts_in_batch * num_samples_per_prompt

        self.actor_model.eval()

        max_kv_capacity = self.config.max_kv_size or (
            max_prompt_len_mb + self.config.data.max_gen_len
        )
        model_caches = cache.make_prompt_cache(
            self.actor_model, max_kv_size=max_kv_capacity
        )

        prompts_rep = mx.repeat(
            padded_prompts_mx, repeats=num_samples_per_prompt, axis=0
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

        rep_penalty_value = max(self.config.trainer.repetition_penalty or 1.0, 1.0)
        rep_context_size = self.config.trainer.repetition_context_size or 20
        rep_processors = make_logits_processors(
            repetition_penalty=rep_penalty_value,
            repetition_context_size=rep_context_size,
            logit_bias=None,
        )

        mcq_flags: List[bool] = [
            p.get("is_mcq", False)
            for p in raw_prompts
            for _ in range(num_samples_per_prompt)
        ]
        dynamic_bias_processor = make_dynamic_tag_bias_processor(
            self.tokenizer, self.config, mcq_flags
        )

        think_temp = self.config.trainer.think_temperature
        answer_temp = self.config.trainer.answer_temperature

        hist_tokens_py: List[List[int]] = [list(row) for row in prompts_rep.tolist()]
        responses_tok_list: List[mx.array] = []
        actor_lp_cached_list: List[mx.array] = []
        ended = mx.full((total_samples,), False, dtype=mx.bool_)

        proc0 = next_logits
        for f in rep_processors:
            proc0 = f(hist_tokens_py, proc0)
        proc0 = dynamic_bias_processor(hist_tokens_py, proc0)

        sampler0 = safe_make_sampler(self.config, temp=think_temp)
        tok0 = sampler0(proc0)
        lp0 = mx.log_softmax(proc0.astype(mx.float32), axis=-1)
        lp0_selected = mx.take_along_axis(lp0, tok0[..., None], axis=-1).squeeze(-1)

        if self.tokenizer.eos_token_id is not None:
            ended = mx.logical_or(ended, mx.equal(tok0, self.tokenizer.eos_token_id))

        responses_tok_list.append(tok0[:, None])
        actor_lp_cached_list.append(lp0_selected[:, None])
        curr_tok = tok0

        for i in range(total_samples):
            if not ended[i].item():
                hist_tokens_py[i].append(int(tok0[i].item()))

        for step in range(self.config.data.max_gen_len - 1):
            if mx.all(ended).item():
                break

            out = self.actor_model(
                curr_tok[:, None].astype(mx.int64), cache=model_caches
            )
            logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
                mx.float32
            )

            proc_step = logits
            for f in rep_processors:
                proc_step = f(hist_tokens_py, proc_step)
            proc_step = dynamic_bias_processor(hist_tokens_py, proc_step)

            current_temp = answer_temp
            sampler = safe_make_sampler(self.config, temp=current_temp)

            sampled_tok = sampler(proc_step)
            lp_step = mx.log_softmax(proc_step.astype(mx.float32), axis=-1)
            lp_step_selected = mx.take_along_axis(
                lp_step, sampled_tok[..., None], axis=-1
            ).squeeze(-1)

            ended_prev = ended
            if self.tokenizer.eos_token_id is not None:
                ended = mx.logical_or(
                    ended_prev, mx.equal(sampled_tok, self.tokenizer.eos_token_id)
                )

            responses_tok_list.append(
                mx.where(
                    ended_prev[:, None],
                    mx.full(
                        (total_samples, 1),
                        self.tokenizer.pad_token_id,
                        dtype=sampled_tok.dtype,
                    ),
                    sampled_tok[:, None],
                )
            )
            actor_lp_cached_list.append(
                mx.where(
                    ended_prev[:, None],
                    mx.zeros((total_samples, 1), dtype=lp_step_selected.dtype),
                    lp_step_selected[:, None],
                )
            )

            curr_tok = sampled_tok

            for i in range(total_samples):
                if not ended_prev[i].item():
                    hist_tokens_py[i].append(int(curr_tok[i].item()))

            if step % 32 == 0:
                mx.eval(sampled_tok, ended, lp_step_selected)

        mx.synchronize()
        self.actor_model.train()

        responses_mx = (
            mx.concatenate(responses_tok_list, axis=1)
            if responses_tok_list
            else mx.zeros((total_samples, 0), dtype=mx.int32)
        )
        actor_lp_resp = (
            mx.concatenate(actor_lp_cached_list, axis=1)
            if actor_lp_cached_list
            else mx.zeros((total_samples, 0), dtype=mx.float32)
        )
        mx.eval(responses_mx, actor_lp_resp)

        gen_len = int(responses_mx.shape[1]) if responses_mx.size else 0
        decoded_responses = (
            self.tokenizer.batch_decode(
                responses_mx.tolist(), skip_special_tokens=False
            )
            if gen_len > 0
            else [""] * total_samples
        )

        rewards_total: List[float] = []
        rewards_fmt: List[float] = []
        rewards_cont: List[float] = []

        reward_cfg_for_text_utils = (
            self.config.rewards[0]
            if self.config.rewards
            else RewardConfig(
                name="temp_for_text_utils",
                weight=1.0,
                think_start_tag="<think>",
                think_end_tag="</think>",
                answer_start_tag="<answer>",
                answer_end_tag="</answer>",
            )
        )

        for i in range(total_samples):
            prompt_idx = i // num_samples_per_prompt
            context = RewardContext(
                generated_text=decoded_responses[i],
                prompt_text=raw_prompts[prompt_idx],
                reference_completion=raw_completions[prompt_idx],
                test_cases=raw_test_cases[prompt_idx],
                metadata={
                    "is_mcq": mcq_flags[i],
                    "is_invalid_sample": is_invalid_sample_flags[prompt_idx],
                },
            )

            rewards_dict = self.reward_composer.compute(context)
            rewards_total.append(rewards_dict.get("total", 0.0))
            rewards_fmt.append(rewards_dict.get("format_structure", 0.0))
            rewards_cont.append(rewards_dict.get("content_similarity", 0.0))

        rewards_total_mx = mx.array(rewards_total, dtype=mx.float32)

        advantages = self.grpo_algorithm.compute_advantages(
            rewards_total_mx, num_samples_per_prompt
        )

        resp_mask = _mask_after_answer(
            responses_mx,
            (responses_mx != self.tokenizer.pad_token_id).astype(mx.float32),
            self.tokenizer,
            reward_cfg_for_text_utils,
        )

        ref_lp_resp = mx.zeros((total_samples, gen_len), dtype=mx.float32)
        aligned_ok = False
        kl_mode = "unknown"

        if gen_len > 0:
            if not self.config.allow_cross_arch_ref:
                try:
                    full_seq = mx.concatenate([prompts_rep, responses_mx], axis=1)
                    ref_attn_mask = _create_4d_attention_mask(
                        full_seq, self.tokenizer.pad_token_id, dtype=mx.bfloat16
                    )
                    ref_out = self.ref_model(
                        full_seq.astype(mx.int64), mask=ref_attn_mask
                    )
                    ref_logits = (
                        ref_out[0] if isinstance(ref_out, tuple) else ref_out
                    ).astype(mx.float32)

                    logits_start_idx = max_prompt_len_mb - 1
                    logits_end_idx = max_prompt_len_mb + gen_len - 1

                    ref_resp_logits = ref_logits[:, logits_start_idx:logits_end_idx, :]

                    if int(ref_resp_logits.shape[1]) == gen_len:
                        ref_lp_resp = mx.log_softmax(ref_resp_logits, axis=-1)
                        ref_lp_resp = mx.take_along_axis(
                            ref_lp_resp, responses_mx[..., None], axis=-1
                        ).squeeze(-1)
                        mx.eval(ref_lp_resp)
                        aligned_ok = True
                        kl_mode = "per_token_aligned"
                    else:
                        logger.warning(
                            f"Ref model logits shape mismatch: expected {gen_len}, "
                            f"got {ref_resp_logits.shape[1]}. Falling back."
                        )
                except Exception as e:
                    logger.warning(
                        f"In-architecture KL alignment failed: {e}. Falling back to cross-arch surrogate.",
                        exc_info=True,
                    )
                    aligned_ok = False

            if (
                not aligned_ok
                and self.config.allow_cross_arch_ref
                and self.config.align_bridge_path
            ):
                if self.config.align_bridge_path.exists():
                    logger.warning(
                        "Cross-arch KL requires custom sequence logprob or will use dummy KL."
                    )
                    kl_mode = "cross_arch_fallback_zero_kl"
                    ref_lp_resp = mx.zeros((total_samples, gen_len), dtype=mx.float32)
                    aligned_ok = True
                else:
                    logger.warning(
                        f"Align bridge path not found: {self.config.align_bridge_path}. "
                        "Cannot use cross-arch KL."
                    )

            if not aligned_ok:
                logger.warning(
                    "No valid reference log-probabilities could be computed. "
                    "KL divergence will effectively be zero."
                )
                ref_lp_resp = actor_lp_resp
                kl_mode = "policy_self_kl"

        prompt_token_lengths = [len(p_arr.tolist()) for p_arr in prompts_list_of_arrays]
        response_token_lengths = [
            int(mx.sum(resp_mask[i]).item()) for i in range(total_samples)
        ]

        _maybe_log_samples(
            config=self.config,
            update_idx=update_step,
            prompts_data=raw_prompts,
            decoded_responses=decoded_responses,
            rewards_total=rewards_total,
            rewards_fmt=rewards_fmt,
            rewards_cont=rewards_cont,
            prompt_token_lens=prompt_token_lengths,
            response_token_lens=response_token_lengths,
            kl_mode=kl_mode,
            run_id=self._run_id,
            is_invalid_batch=any(is_invalid_sample_flags),
            ref_letters_list=[""] * total_samples,
            gen_letters_list=[""] * total_samples,
            is_mcq_list=mcq_flags,
        )

        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx,
        }

        try:
            del attn_mask
        except:
            pass

        gc.collect()
        mx.clear_cache()

        avg_reward = float(np.mean(rewards_total)) if rewards_total else 0.0
        raw_components = {
            "raw_format": float(np.mean(rewards_fmt)) if rewards_fmt else 0.0,
            "raw_content_combined": float(np.mean(rewards_cont))
            if rewards_cont
            else 0.0,
        }

        return rollout_data_for_loss, avg_reward, raw_components

    def train_step(
        self, rollout_batch: List[Dict[str, mx.array]], update_step: int
    ) -> TrainingMetrics:
        """
        Executes a single optimization step of the GRPO algorithm.

        Args:
            rollout_batch: List containing rollout data dictionaries from generate_rollouts
            update_step: Current training step number

        Returns:
            TrainingMetrics object with loss, rewards, and other metrics
        """
        if (
            self.actor_model is None
            or self.optimizer is None
            or self.lr_scheduler is None
        ):
            raise TrainingRuntimeError(
                "Actor model, optimizer, or LR scheduler not initialized."
            )

        if not isinstance(rollout_batch, list) or len(rollout_batch) == 0:
            raise TrainingRuntimeError("Empty rollout batch received in train_step")

        actual_rollout_batch = rollout_batch[0]

        start_time = time.time()

        current_lr = self.lr_scheduler(self.global_step + 1)
        self.optimizer.learning_rate = current_lr

        (
            loss_val,
            grads,
            algorithm_metrics,
        ) = self.grpo_algorithm.calculate_loss_and_grads(
            actual_rollout_batch, self.config
        )

        if (
            self.config.trainer.enable_dynamic_beta
            and self._cooldown_steps_remaining == 0
        ):
            current_avg_reward = float(
                mx.mean(actual_rollout_batch.get("raw_rewards", mx.array(0.0))).item()
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

        scaled_grads = scale_grads_by_band(grads, self.config)
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

        updated_params = metal_safe_apply_gradients(
            self.optimizer, final_grads, self.actor_model.trainable_parameters()
        )

        if updated_params is None:
            logger.error(
                f"Metal error during apply_gradients for update {update_step}. "
                "Parameters not updated."
            )
            return TrainingMetrics(
                loss=float(loss_val.item()),
                reward_mean=float(
                    mx.mean(
                        actual_rollout_batch.get("raw_rewards", mx.array(0.0))
                    ).item()
                ),
                grad_norm=grad_norm,
                learning_rate=current_lr,
                step_time_s=time.time() - start_time,
                custom_metrics={
                    **algorithm_metrics,
                    "metal_recovery_skipped_update": 1.0,
                },
            )

        self.actor_model.update_modules(updated_params)
        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        reward_mean = float(
            mx.mean(actual_rollout_batch.get("raw_rewards", mx.array(0.0))).item()
        )
        total_tokens = int(
            mx.sum(actual_rollout_batch.get("tokens", mx.array([0]))).item()
        )
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
        """
        Runs registered evaluators against the validation dataset.

        Args:
            update_step: The current training update step.

        Returns:
            A list of EvaluationMetrics objects.
        """
        results: List[EvaluationMetrics] = []
        val_dataset = self.data_manager._val_dataset

        if val_dataset is None or len(val_dataset) == 0:
            logger.warning("Validation skipped: no validation dataset available.")
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
                logger.error(
                    f"Error during evaluation '{eval_cfg.name}': {e}", exc_info=True
                )
        return results
