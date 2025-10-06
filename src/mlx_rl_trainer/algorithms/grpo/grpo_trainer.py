# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 004
# goals_of_writing_code_block: Update GRPOTrainer to import `_extract_predicted_letters` from the central text_utils module to resolve the final dependency issue.
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
from ...utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
    make_dynamic_tag_bias_processor,
    _is_metal_internal_error,
    metal_recover,
    metal_safe_apply_gradients,
    _find_layer_index,
    scale_grads_by_band,
    mask_grads_to_layer_band,
    mask_grads_to_specific_layers,
    _mask_after_answer,
)
from ...utils.text_utils import (
    TwoBlockFormatter,
    _extract_predicted_letters,  # FIX: Correctly imported from text_utils
    _letters_to_canonical,
    _mcq_meta_from_sample,
)
from ...monitoring.metrics_logger import _maybe_log_samples, wandb_run, _calculate_mcq_accuracy
from ...generation.caching import PagedKVCache
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)

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
        self._run_id: str = str(uuid.uuid4())
        self._cooldown_steps_remaining: int = 0
        self._last_avg_reward: float = 0.0
        self._reward_history_for_smoothing: List[float] = []
        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")
        actor_path = self.config.model.model_path
        if not actor_path.exists() or not actor_path.is_dir() or not (actor_path / "config.json").exists():
            actor_path = Path("./models/mock_model")
            actor_path.mkdir(parents=True, exist_ok=True)
            if not (actor_path / "config.json").exists():
                with open(actor_path / "config.json", 'w') as f:
                    json.dump({"config": {"vocab_size": 32000, "embed_dim": 128, "num_layers": 4, "num_kv_heads": 2, "hidden_size": 128, "num_attention_heads": 4, "model_type": "mock"}}, f)
                logger.info(f"Created dummy config.json for mock model at {actor_path}")
            logger.warning(f"Actor model path '{self.config.model.model_path}' not found or invalid. Using mock model at '{actor_path}'.")

        self.actor_model, self.tokenizer = self.model_manager.load_model(
            actor_path, "actor", is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config=self.config.model.model_dump()
        )

        ref_model_path = self.config.model.ref_model_path
        if not ref_model_path.exists() or not ref_model_path.is_dir() or not (ref_model_path / "config.json").exists():
            ref_model_path = Path("./models/mock_model")
            logger.warning(f"Reference model path '{self.config.model.ref_model_path}' not found or invalid. Using mock model at '{ref_model_path}'.")
        self.ref_model, _ = self.model_manager.load_model(ref_model_path, "reference", is_trainable=False)
        self.data_manager.tokenizer = self.tokenizer

        rprint(f"Loaded actor model: [green]{self.actor_model.__class__.__name__}[/green], ref model: [green]{self.ref_model.__class__.__name__}[/green].")

        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)
        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2),
            weight_decay=self.config.trainer.optimizer_weight_decay
        )
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        start_update_step, metadata = self.checkpoint_manager.load_latest_state(self.actor_model, self.optimizer)
        self.global_step = start_update_step
        self.current_epoch = metadata.get('epoch', 0)
        
        rprint(f"[bold green]Trainer setup complete.[/bold green] Resuming from update step [bold magenta]{self.global_step}[/bold magenta], epoch [bold magenta]{self.current_epoch}[/bold magenta].")
        return self.global_step, self.current_epoch
    
    def generate_rollouts(self, batch_prompts_data: List[Dict[str, Any]], update_step: int) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        if not all([self.actor_model, self.tokenizer, self.ref_model]):
            raise TrainingRuntimeError("Models or tokenizer not initialized for rollout generation.")

        prompts_list_of_arrays = [sample["input_ids"] for sample in batch_prompts_data]
        if not prompts_list_of_arrays:
            return {}, 0.0, {}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)
        padded_prompts_mx = mx.array([mx.pad(arr, [(0, max_prompt_len_mb - arr.shape[0])], constant_value=self.tokenizer.pad_token_id) for arr in prompts_list_of_arrays], dtype=mx.int32)
        
        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        total_samples = len(prompts_list_of_arrays) * num_samples_per_prompt
        prompts_rep = mx.repeat(padded_prompts_mx, repeats=num_samples_per_prompt, axis=0)

        self.actor_model.eval()
        model_caches = cache.make_prompt_cache(self.actor_model, max_kv_size=self.config.max_kv_size)
        
        attn_mask = _create_4d_attention_mask(prompts_rep, self.tokenizer.pad_token_id, dtype=mx.bfloat16)
        out_actor = self.actor_model(prompts_rep.astype(mx.int64), mask=attn_mask, cache=model_caches)
        next_logits = (out_actor[0] if isinstance(out_actor, tuple) else out_actor)[:, -1, :].astype(mx.float32)
        mx.eval(next_logits)

        rep_processors = make_logits_processors(
            repetition_penalty=self.config.generation.repetition_penalty,
            repetition_context_size=self.config.generation.repetition_context_size,
        )
        mcq_flags_repeated = [p['is_mcq'] for p in batch_prompts_data for _ in range(num_samples_per_prompt)]
        dynamic_bias_processor = make_dynamic_tag_bias_processor(self.tokenizer, self.config, mcq_flags_repeated)
        
        think_temp = self.config.generation.temperature
        answer_temp = self.config.generation.temperature

        hist_tokens_py = [list(row) for row in prompts_rep.tolist()]
        responses_tok_list, actor_lp_cached_list = [], []
        ended = mx.full((total_samples,), False, dtype=mx.bool_)
        curr_tok = None

        for step in range(self.config.data.max_gen_len):
            if mx.all(ended).item(): break
            
            logits_to_process = next_logits if step == 0 else self.actor_model(curr_tok[:, None].astype(mx.int64), cache=model_caches)[0][:, -1, :].astype(mx.float32)
            for f in rep_processors: logits_to_process = f(hist_tokens_py, logits_to_process)
            logits_to_process = dynamic_bias_processor(hist_tokens_py, logits_to_process)
            
            sampler = safe_make_sampler(self.config, temp=think_temp if step < self.config.trainer.min_think_tokens else answer_temp)
            sampled_tok = sampler(logits_to_process)
            
            lp_step_selected = mx.take_along_axis(mx.log_softmax(logits_to_process.astype(mx.float32), axis=-1), sampled_tok[..., None], axis=-1).squeeze(-1)

            ended_prev = ended
            if self.tokenizer.eos_token_id is not None:
                ended = mx.logical_or(ended_prev, mx.equal(sampled_tok, self.tokenizer.eos_token_id))

            token_to_add = mx.where(ended_prev, self.tokenizer.pad_token_id, sampled_tok)
            logprob_to_add = mx.where(ended_prev, 0.0, lp_step_selected)

            responses_tok_list.append(token_to_add[:, None])
            actor_lp_cached_list.append(logprob_to_add[:, None])
            curr_tok = sampled_tok
            
            for i in range(total_samples):
                if not ended_prev[i].item():
                    hist_tokens_py[i].append(int(token_to_add[i].item()))

        mx.synchronize()
        self.actor_model.train()
        
        responses_mx = mx.concatenate(responses_tok_list, axis=1) if responses_tok_list else mx.zeros((total_samples, 0), dtype=mx.int32)
        actor_lp_resp = mx.concatenate(actor_lp_cached_list, axis=1) if actor_lp_cached_list else mx.zeros((total_samples, 0), dtype=mx.float32)
        mx.eval(responses_mx, actor_lp_resp)
        
        decoded_responses = self.tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=True)
        
        rewards_total, rewards_fmt, rewards_cont, mcq_gen_letters_all_list = [], [], [], []
        coerced_responses = decoded_responses

        for i in range(total_samples):
            prompt_idx = i // num_samples_per_prompt
            original_data = batch_prompts_data[prompt_idx]['original_raw_data']
            context = RewardContext(
                generated_text=coerced_responses[i],
                prompt_text=original_data.get(self.config.data.dataset_prompt_key, ""),
                reference_completion=original_data.get(self.config.data.dataset_answer_key, ""),
                test_cases=original_data.get("test_cases", []),
                metadata=original_data.get("meta", {}),
            )
            rewards_dict = self.reward_composer.compute(context)
            rewards_total.append(rewards_dict.get('total', 0.0))
            rewards_fmt.append(rewards_dict.get('TagStructureReward', 0.0))
            rewards_cont.append(rewards_dict.get('CodeExecutionReward', 0.0))
            
            if mcq_flags_repeated[i]:
                gen_letters = _extract_predicted_letters(coerced_responses[i], original_data.get("meta", {}).get("mcq_options"), self.config.rewards[0])
                mcq_gen_letters_all_list.append(",".join(gen_letters))
            else:
                mcq_gen_letters_all_list.append("")
        
        rewards_total_mx = mx.array(rewards_total, dtype=mx.float32)
        advantages = self.grpo_algorithm.compute_advantages(rewards_total_mx, num_samples_per_prompt)
        resp_mask = (responses_mx != self.tokenizer.pad_token_id).astype(mx.float32)
        
        ref_lp_resp = self.model_manager.get_logprobs_for_sequence(self.ref_model, prompts_rep, responses_mx)

        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx,
        }

        avg_reward = float(np.mean(rewards_total)) if rewards_total else 0.0
        
        del model_caches
        gc.collect()
        mx.clear_cache()
        
        return rollout_data_for_loss, avg_reward, {"raw_format": np.mean(rewards_fmt), "raw_content_combined": np.mean(rewards_cont)}


    def train_step(self, batch_prompts_data: List[Dict[str, Any]], update_step: int, epoch: int) -> TrainingMetrics:
        start_time = time.time()
        if not all([self.optimizer_is_set, self.lr_scheduler_is_set, self.actor_model_is_set]):
            raise TrainingRuntimeError("Trainer is not set up properly.")

        lr = self.lr_scheduler(update_step)
        self.optimizer.learning_rate = lr

        rollout_batch, avg_raw_reward, raw_rewards_breakdown = self.generate_rollouts(
            batch_prompts_data, update_step
        )
        
        if not rollout_batch:
            return TrainingMetrics(loss=0, reward_mean=0,grad_norm=0, learning_rate=lr, step_time_s=0, tokens_per_sec=0, kl_divergence=0, epoch=epoch, step=update_step)

        loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config, self.tokenizer.pad_token_id)
        
        if not mx.isfinite(loss):
            logger.warning(f"Non-finite loss detected: {loss.item()}. Skipping update.")
            return TrainingMetrics(loss=loss.item(), reward_mean=avg_raw_reward, grad_norm=0, learning_rate=lr, step_time_s=0, tokens_per_sec=0, kl_divergence=0, epoch=epoch, step=update_step, custom_metrics=metrics)

        metal_safe_apply_gradients(self.optimizer, grads, self.actor_model.trainable_parameters())
        mx.eval(self.actor_model.parameters(), self.optimizer.state)

        grad_norm_val = mx.linalg.norm(mx.concatenate([mx.flatten(g) for g in mlx.utils.tree_leaves(grads)]))
        
        total_tokens = sum(s["input_ids"].size for s in batch_prompts_data) + int(mx.sum(rollout_batch["response_mask"]).item())
        step_time = time.time() - start_time
        tokens_per_sec = total_tokens / step_time if step_time > 0 else 0

        return TrainingMetrics(
            loss=loss.item(), reward_mean=avg_raw_reward, reward_std=float(mx.std(rollout_batch["raw_rewards"]).item()),
            kl_divergence=metrics.get("kl_divergence", 0.0), grad_norm=float(grad_norm_val.item()),
            learning_rate=lr, epoch=epoch, step=update_step, step_time_s=step_time, 
            tokens_per_sec=tokens_per_sec, custom_metrics={**metrics, **raw_rewards_breakdown}
        )

    def evaluate(self, update_step: int) -> List[EvaluationMetrics]:
        results: List[EvaluationMetrics] = []
        val_dataset = self.data_manager._val_dataset
        if val_dataset is None or len(val_dataset) == 0:
            return []

        eval_gen_config = self.config.generation.model_dump()
        eval_gen_config.update({"system_prompt": self.config.system_prompt})

        for eval_cfg in self.config.evaluation:
            try:
                merged_eval_config = eval_cfg.config.copy()
                merged_eval_config.update(eval_gen_config)
                evaluator = EvaluatorRegistry.create(eval_cfg.name, merged_eval_config)
                eval_output = evaluator.evaluate(self.actor_model, self.tokenizer, val_dataset)
                results.append(eval_output)
            except Exception as e:
                logging.error(f"Error during evaluation '{eval_cfg.name}': {e}", exc_info=True)
        return results
