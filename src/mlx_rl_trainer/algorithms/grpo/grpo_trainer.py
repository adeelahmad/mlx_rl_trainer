# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py
# revision_no: 002
# goals_of_writing_code_block: Implement the concrete GRPOTrainer, orchestrating the GRPO algorithm steps, with full MLX-LM generation integration and bug fixes.
# type_of_code_response: change existing
"""
Concrete GRPO Trainer implementation.
"""
import uuid
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tuner.utils import build_schedule # For LR scheduling
from mlx_lm.models import cache # For KV cache in generation
from mlx_lm.sample_utils import make_logits_processors # For generation

from ...core.trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics, TrainingRuntimeError
from ...core.config import ExperimentConfig
from ...core.model_manager import ModelManager
from ...core.dataset_manager import DatasetManager
from ...core.checkpoint_manager import CheckpointManager
from ...rewards.registry import RewardRegistry
from ...rewards.base_reward import RewardComposer
from ...rewards.context import RewardContext
from ...evaluation.registry import EvaluatorRegistry
from ...utils.mlx_utils import _create_4d_attention_mask, safe_make_sampler, make_dynamic_tag_bias_processor, _is_metal_internal_error, metal_recover # FIX: Correct processor import, metal_recover
from ...utils.mlx_utils import metal_safe_apply_gradients, _find_layer_index # For gradient safety & analysis
from ...utils.text_utils import _extract_final_numeric, _normalize_ans_for_match # For benchmark utilities
from ...monitoring.metrics_logger import _maybe_log_samples, wandb_run # For NDJSON sample logging
from ...generation.caching import PagedKVCache # FIX: Correct PagedKVCache import
from ...utils.text_utils import _first_token_ids_for_lexemes, _letter_token_ids # For logits biasing
from ...utils.text_utils import TwoBlockFormatter # For output coercion
from .grpo_algorithm import GRPOAlgorithm # Import GRPOAlgorithm

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
        paged_kv_cache: Optional[PagedKVCache] = None
    ):
        super().__init__(config, model_manager, data_manager, checkpoint_manager)

        self.reward_composer = reward_composer
        self.grpo_algorithm: Optional[GRPOAlgorithm] = None # Will be initialized in _setup
        self.paged_kv_cache = paged_kv_cache # PagedKV cache instance

        # Attributes set during _setup()
        self.tokenizer: Any = None
        self.actor_model: Optional[nn.Module] = None
        self.ref_model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.lr_scheduler: Optional[Callable[[int], float]] = None

        # Internal state for tracking during training
        self._current_update_step: int = 0
        self._current_epoch: int = 0
        self._run_id: str = str(uuid.uuid4()) # Unique ID for this training run instance

        logger.info(f"GRPOTrainer initialized for run ID: {self._run_id}.")

    def _setup(self) -> Tuple[int, int]:
        """
        Performs all necessary setup: loads models, initializes tokenizer,
        sets up optimizer and LR scheduler, and resumes from a checkpoint if specified.

        Returns:
            A tuple of (start_update_step, start_epoch).
        """
        from rich import print as rprint
        rprint("[bold blue]Setting up GRPO training components...[/bold blue]")

        # 1. Load Actor (Policy) Model and Tokenizer
        if self.config.model.model_path.exists() and self.config.model.model_path.is_dir():
            actor_path = self.config.model.model_path
        else: # Mock model dir
            actor_path = Path("./models/mock_model")
            actor_path.mkdir(parents=True, exist_ok=True) # Ensure mock dir exists
            if not (actor_path / "config.json").exists(): # Create dummy config
                with open(actor_path / "config.json", 'w') as f:
                    json.dump({"config": {"vocab_size": 32000, "embed_dim": 128, "num_layers": 4, "num_kv_heads": 2, "hidden_size": 128, "num_attention_heads": 4}}, f)

        self.actor_model, self.tokenizer = self.model_manager.load_model(
            actor_path, "actor", is_trainable=True,
            apply_lora=self.config.model.use_lora,
            lora_config={
                "rank": self.config.model.lora_rank,
                "alpha": self.config.model.lora_alpha,
                "dropout": self.config.model.lora_dropout,
                "scale_by_rank": self.config.model.lora_scale_by_rank,
                "target_modules": self.config.model.lora_target_modules,
            }
        )

        # 2. Load Reference Model
        ref_model_path = self.config.model.ref_model_path
        if not ref_model_path.exists() or not ref_model_path.is_dir(): # Use mock if ref path invalid
            ref_model_path = Path("./models/mock_model") # Fallback to mock
        self.ref_model, _ = self.model_manager.load_model(
            ref_model_path, "reference", is_trainable=False
        )
        self.data_manager.tokenizer = self.tokenizer # Ensure data manager uses the correct tokenizer

        rprint(f"Loaded actor model: [green]{self.actor_model.__class__.__name__}[/green], ref model: [green]{self.ref_model.__class__.__name__}[/green].")

        # 3. Initialize GRPO Algorithm
        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)

        # 4. Optimizer and LR Scheduler
        self.optimizer = optim.AdamW(
            learning_rate=self.config.trainer.learning_rate,
            betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2), # FIX: Use configured betas
            weight_decay=self.config.trainer.optimizer_weight_decay # FIX: Use configured weight_decay
        )
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)

        rprint(f"Optimizer: [cyan]{self.optimizer.__class__.__name__}[/cyan], Learning Rate: [cyan]{self.config.trainer.learning_rate:.2e}[/cyan].")

        # 5. Load Checkpoint (if resuming)
        start_update_step, metadata = self.checkpoint_manager.load_latest_state(self.actor_model, self.optimizer)
        self.global_step = start_update_step
        self.current_epoch = metadata.get('epoch', 0)

        rprint(f"[bold green]Trainer setup complete.[/bold green] Resuming from update step [bold magenta]{self.global_step}[/bold magenta], epoch [bold magenta]{self.current_epoch}[/bold magenta].")
        return self.global_step, self.current_epoch

    def generate_rollouts(self, batch_prompts_data: Dict[str, Any], update_step: int) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """
        Generates rollouts using the actor model and computes rewards.

        Args:
            batch_prompts_data: A batch of raw prompts from the dataset.
            update_step: The current training update step.

        Returns:
            Tuple: (rollout_data_dict, average_reward, raw_reward_components_dict)
        """
        if self.actor_model is None or self.tokenizer is None or self.ref_model is None:
            raise TrainingRuntimeError("Models or tokenizer not initialized for rollout generation.")

        prompts_list_of_arrays = batch_prompts_data["input_ids"]
        raw_prompts = batch_prompts_data["raw_prompts"]
        raw_completions = batch_prompts_data["raw_completions"]
        raw_test_cases = batch_prompts_data["raw_test_cases"]
        is_invalid_sample_flags = batch_prompts_data["is_invalid_sample_flags"]

        num_prompts_in_batch = len(prompts_list_of_arrays)
        if num_prompts_in_batch == 0:
            return {}, 0.0, {"raw_format": 0.0, "raw_content_combined": 0.0}

        max_prompt_len_mb = max(arr.shape[0] for arr in prompts_list_of_arrays)
        # Pad prompts to max_prompt_len_mb for efficient batch processing in MLX
        padded_prompts_mx = mx.array([
            mx.pad(arr, [(0, max_prompt_len_mb - arr.shape[0])], constant_value=self.tokenizer.pad_token_id)
            for arr in prompts_list_of_arrays
        ], dtype=mx.int32)

        num_samples_per_prompt = self.config.trainer.num_rollout_samples
        total_samples = num_prompts_in_batch * num_samples_per_prompt

        # --- Generate Rollouts (MLX-LM Integration) ---
        self.actor_model.eval() # Set actor to eval mode for generation

        # Create caches for the prompt
        # Use config.max_kv_size, config.data.max_prompt_len, config.data.max_gen_len
        max_kv_capacity = self.config.max_kv_size or (max_prompt_len_mb + self.config.data.max_gen_len)
        model_caches = cache.make_prompt_cache(self.actor_model, max_kv_size=max_kv_capacity)

        # Repeat prompts for num_samples_per_prompt
        prompts_rep = mx.repeat(padded_prompts_mx, repeats=num_samples_per_prompt, axis=0)

        # Initial forward pass to fill cache and get first logits
        attn_mask = _create_4d_attention_mask(prompts_rep, self.tokenizer.pad_token_id, dtype=mx.bfloat16)
        out_actor = self.actor_model(prompts_rep.astype(mx.int64), mask=attn_mask, cache=model_caches)
        next_logits = (out_actor[0] if isinstance(out_actor, tuple) else out_actor)[:, -1, :].astype(mx.float32)

        # Prepare logit processors
        # This includes repetition penalty and the dynamic tag bias processor
        rep_penalty_value = max(self.config.trainer.repetition_penalty or 1.0, 1.0)
        rep_context_size = self.config.trainer.repetition_context_size or 20
        rep_processors = make_logits_processors(
            repetition_penalty=rep_penalty_value,
            repetition_context_size=rep_context_size,
            logit_bias=None,
        )

        # Create dynamic tag bias processor
        # FIX: Construct prompts_data_for_rollout correctly including MCQ info for make_dynamic_tag_bias_processor
        mcq_flags: List[bool] = [p.get('is_mcq', False) for p in batch_prompts_data['raw_prompts'] for _ in range(num_samples_per_prompt)]
        dynamic_bias_processor = make_dynamic_tag_bias_processor(self.tokenizer, self.config, mcq_flags)

        # Samplers
        think_temp = self.config.trainer.think_temperature
        answer_temp = self.config.trainer.answer_temperature

        # Generate tokens autoregressively
        hist_tokens_py: List[List[int]] = [list(row) for row in prompts_rep.tolist()]
        responses_tok_list: List[mx.array] = []
        actor_lp_cached_list: List[mx.array] = []
        ended = mx.full((total_samples,), False, dtype=mx.bool_)

        tokens_generated = 1
        curr_tok = None # Initialize curr_tok

        # First token generation
        proc0 = next_logits
        for f in rep_processors: proc0 = f(hist_tokens_py, proc0)
        proc0 = dynamic_bias_processor(hist_tokens_py, proc0) # Apply dynamic bias

        sampler0 = safe_make_sampler(self.config, temp=think_temp) # Start with think temp
        tok0 = sampler0(proc0)
        lp0 = mx.nn.log_softmax(proc0.astype(mx.float32), axis=-1) # Log softmax on processed logits
        lp0_selected = mx.take_along_axis(lp0, tok0[..., None], axis=-1).squeeze(-1) # Get logprob of sampled token

        if self.tokenizer.eos_token_id is not None:
            ended = mx.logical_or(ended, mx.equal(tok0, self.tokenizer.eos_token_id))

        responses_tok_list.append(tok0[:, None])
        actor_lp_cached_list.append(lp0_selected[:, None])
        curr_tok = tok0 # Set curr_tok for next iteration

        # Update histories
        for i in range(total_samples):
            if not ended[i].item():
                hist_tokens_py[i].append(int(tok0[i].item()))

        tokens_generated += 1

        # Subsequent token generation loop
        for step in range(self.config.data.max_gen_len - 1):
            if mx.all(ended).item(): break

            # Actor model forward pass
            out = self.actor_model(curr_tok[:, None].astype(mx.int64), cache=model_caches)
            logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

            # Apply logit processors
            proc_step = logits
            for f in rep_processors: proc_step = f(hist_tokens_py, proc_step)
            proc_step = dynamic_bias_processor(hist_tokens_py, proc_step) # Apply dynamic bias

            # Determine temperature dynamically based on state (inside think vs answer)
            # This requires dynamic_bias_processor to track 'inside_think' state
            # For this context, we will use a mixed temperature approach.
            # Simplified: Use answer_temp after initial think phase, or a blend.
            # For rollout generation, it's safer to have exploration.
            current_temp = answer_temp # Default to answer temp; dynamic_bias_processor should help structure
            sampler = safe_make_sampler(self.config, temp=current_temp)

            sampled_tok = sampler(proc_step)
            lp_step = mx.nn.log_softmax(proc_step.astype(mx.float32), axis=-1)
            lp_step_selected = mx.take_along_axis(lp_step, sampled_tok[..., None], axis=-1).squeeze(-1)

            ended_prev = ended # Store previous 'ended' state
            if self.tokenizer.eos_token_id is not None:
                ended = mx.logical_or(ended_prev, mx.equal(sampled_tok, self.tokenizer.eos_token_id))

            # Add token and logprob to lists, masking if already ended
            responses_tok_list.append(mx.where(ended_prev[:,None], mx.array(self.tokenizer.pad_token_id, dtype=sampled_tok.dtype), sampled_tok[:,None]))
            actor_lp_cached_list.append(mx.where(ended_prev[:,None], mx.array(0.0, dtype=lp_step_selected.dtype), lp_step_selected[:,None]))

            curr_tok = sampled_tok # Update for next iteration

            # Update histories for next step
            for i in range(total_samples):
                if not ended_prev[i].item(): # Only update if not ended previously
                    hist_tokens_py[i].append(int(curr_tok[i].item()))

            tokens_generated += 1
            if step % 32 == 0: # Periodically evaluate intermediate results
                mx.eval(sampled_tok, ended, lp_step_selected, responses_tok_list)

        mx.synchronize()
        self.actor_model.train() # Set model back to train mode

        # Concatenate all generated tokens and log probabilities
        responses_mx = (mx.concatenate(responses_tok_list, axis=1) if responses_tok_list else mx.zeros((total_samples, 0), dtype=mx.int32))
        actor_lp_resp = (mx.concatenate(actor_lp_cached_list, axis=1) if actor_lp_cached_list else mx.zeros((total_samples, 0), dtype=mx.float32))
        mx.eval(responses_mx, actor_lp_resp)

        gen_len = int(responses_mx.shape[1]) if responses_mx.size else 0
        decoded_responses = (self.tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=False) if gen_len > 0 else [""] * total_samples)

        # --- Reward Calculation ---
        rewards_total: List[float] = []
        rewards_fmt: List[float] = []
        rewards_cont: List[float] = []

        # Create RewardConfig for text utilities with tags from the main config
        reward_cfg_for_text_utils = RewardConfig(
            name="temp_for_text_utils", weight=1.0, # Dummy name/weight
            think_start_tag=self.config.rewards[0].think_start_tag if self.config.rewards else '<think>',
            think_end_tag=self.config.rewards[0].think_end_tag if self.config.rewards else '</think>',
            answer_start_tag=self.config.rewards[0].answer_start_tag if self.config.rewards else '<answer>',
            answer_end_tag=self.config.rewards[0].answer_end_tag if self.config.rewards else '</answer>'
        )

        for i in range(total_samples):
            # Create RewardContext for each sample
            prompt_idx = i // num_samples_per_prompt
            context = RewardContext(
                generated_text=decoded_responses[i],
                prompt_text=raw_prompts[prompt_idx],
                reference_completion=raw_completions[prompt_idx],
                test_cases=raw_test_cases[prompt_idx],
                metadata={
                    "is_mcq": mcq_flags[i], # Pass MCQ flag
                    "mcq_options": mcq_opts[i],
                    "mcq_correct_letters": mcq_gold[i],
                    "mcq_multi_select": mcq_msel[i],
                    "is_invalid_sample": is_invalid_sample_flags[prompt_idx] # Pass invalid flag
                }
            )

            # Compute composite reward using the RewardComposer
            rewards_dict = self.reward_composer.compute(context)
            rewards_total.append(rewards_dict.get('total', 0.0))
            rewards_fmt.append(rewards_dict.get('format_structure', 0.0)) # Example: get specific reward
            rewards_cont.append(rewards_dict.get('content_similarity', 0.0)) # Example: get specific reward

        rewards_total_mx = mx.array(rewards_total, dtype=mx.float32)

        # --- Advantage Calculation ---
        advantages = self.grpo_algorithm.compute_advantages(rewards_total_mx, num_samples_per_prompt)

        # --- Response Mask (Post-processing) ---
        resp_mask = _mask_after_answer(responses_mx, (responses_mx != self.tokenizer.pad_token_id).astype(mx.float32), self.tokenizer, self.config)

        # --- Reference Log-Probabilities for KL Divergence ---
        ref_lp_resp = mx.zeros((total_samples, gen_len), dtype=mx.float32)
        aligned_ok = False
        kl_mode = "unknown"

        if gen_len > 0:
            if not self.config.allow_cross_arch_ref: # Same-architecture reference model
                try:
                    # FIX: Need a separate forward pass for ref_model
                    # on the combined prompts_rep and responses_mx
                    full_seq = mx.concatenate([prompts_rep, responses_mx], axis=1)
                    ref_attn_mask = _create_4d_attention_mask(full_seq, self.tokenizer.pad_token_id, dtype=mx.bfloat16)
                    ref_out = self.ref_model(full_seq.astype(mx.int64), mask=ref_attn_mask) # Forward pass ref_model
                    ref_logits = (ref_out[0] if isinstance(ref_out, tuple) else ref_out).astype(mx.float32)

                    # Extract logits corresponding to generated tokens only
                    # Assuming full_seq_len = max_prompt_len_mb + gen_len
                    logits_start_idx = max_prompt_len_mb -1
                    logits_end_idx = max_prompt_len_mb + gen_len -1

                    ref_resp_logits = ref_logits[:, logits_start_idx:logits_end_idx, :]

                    if int(ref_resp_logits.shape[1]) == gen_len:
                        ref_lp_resp = mx.nn.log_softmax(ref_resp_logits, axis=-1)
                        ref_lp_resp = mx.take_along_axis(ref_lp_resp, responses_mx[..., None], axis=-1).squeeze(-1)
                        mx.eval(ref_lp_resp)
                        aligned_ok = True
                        kl_mode = "per_token_aligned"
                    else:
                        logger.warning(f"Ref model logits shape mismatch: expected {gen_len}, got {ref_resp_logits.shape[1]}. Falling back.")
                except Exception as e:
                    logger.warning(f"In-architecture KL alignment failed: {e}. Falling back to cross-arch surrogate.", exc_info=True)
                    aligned_ok = False

            if not aligned_ok and self.config.allow_cross_arch_ref and self.config.align_bridge_path: # Cross-architecture fallback
                if self.config.align_bridge_path.exists(): # Ensure bridge exists
                    # This path currently uses the custom ContentAlignBridge for a scalar reward
                    # For a per-token KL, a full cross-arch sequence logprob utility is needed.
                    # For now, we simulate this, or it would be zero KL.
                    logger.warning("Cross-arch KL requires custom sequence logprob or will use dummy KL.")
                    # Fallback to zero KL for now if no sophisticated cross-arch logprob
                    kl_mode = "cross_arch_fallback_zero_kl"
                    ref_lp_resp = mx.zeros((total_samples, gen_len), dtype=mx.float32)
                    aligned_ok = True # Mark as aligned to avoid policy_self_kl
                else:
                    logger.warning(f"Align bridge path not found: {self.config.align_bridge_path}. Cannot use cross-arch KL.")

            if not aligned_ok: # Final fallback if no KL could be computed
                logger.warning("No valid reference log-probabilities could be computed. KL divergence will effectively be zero.")
                ref_lp_resp = actor_lp_resp # Fallback to policy's own log-probs (KL will be zero)
                kl_mode = "policy_self_kl"

        # --- Logging ---
        prompt_token_lengths = [len(p_arr.tolist()) for p_arr in prompts_list_of_arrays] # Length of actual prompt tokens
        response_token_lengths = [int(mx.sum(resp_mask[i]).item()) for i in range(total_samples)]

        _maybe_log_samples(
            config=self.config,
            update_idx=update_step,
            prompts_data=batch_prompts_data['raw_prompts'], # Pass raw prompts data for logging
            decoded_responses=decoded_responses,
            rewards_total=rewards_total,
            rewards_fmt=rewards_fmt,
            rewards_cont=rewards_cont,
            prompt_token_lens=prompt_token_lengths,
            response_token_lens=response_token_lengths,
            kl_mode=kl_mode,
            run_id=self._run_id,
            is_invalid_batch=any(is_invalid_sample_flags),
            ref_letters_list=[m.get('mcq_correct_letters', '') for m in mcq_gold], # FIX: Get from specific data
            gen_letters_list=[m.get('mcq_predicted_letters', '') for m in mcq_gold], # FIX: Get from specific data
            is_mcq_list=[m.get('is_mcq', False) for m in mcq_flags] # FIX: Get from specific data
        )

        # --- Final Rollout Batch for Loss Computation ---
        rollout_data_for_loss = {
            "tokens": mx.concatenate([prompts_rep, responses_mx], axis=1),
            "response_mask": resp_mask,
            "advantages": advantages,
            "ref_log_probs": ref_lp_resp.astype(mx.float32),
            "raw_rewards": rewards_total_mx # Add raw rewards for metrics
        }

        # Cleanup caches explicitly
        del model_caches
        gc.collect()
        mx.clear_cache()

        avg_reward = float(np.mean(rewards_total)) if rewards_total else 0.0
        raw_components = {
            "raw_format": float(np.mean(rewards_fmt)) if rewards_fmt else 0.0,
            "raw_content_combined": float(np.mean(rewards_cont)) if rewards_cont else 0.0,
        }

        return rollout_data_for_loss, avg_reward, raw_components

    def _calculate_grpo_loss_and_grads(self, rollout_batch: Dict[str, mx.array]) -> Tuple[mx.array, Dict[str, Any], Dict[str, float]]:
        """
        Calculates the GRPO loss and gradients using the GRPOAlgorithm.

        Args:
            rollout_batch: Processed rollout data for loss computation.

        Returns:
            Tuple: (scalar_loss_tensor, gradients_dict, additional_metrics_dict).
        """
        if self.grpo_algorithm is None:
            raise TrainingRuntimeError("GRPOAlgorithm not initialized.")

        # Pass the current config for beta and other parameters
        return self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config)

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
            logging.warning("Validation skipped: no validation dataset available.")
            return []

        # Create a dummy config dict for evaluators (often expecting a simple dict)
        eval_gen_config = self.config.generation.model_dump() # Convert Pydantic to dict
        eval_gen_config.update({ # Add other relevant global params
            "system_prompt": self.config.system_prompt,
            "max_kv_size": self.config.max_kv_size,
            "max_gen_len": self.config.data.max_gen_len, # Max generated tokens for eval
        })

        for eval_cfg in self.config.evaluation:
            try:
                # Merge global generation/system prompt config into evaluator's config
                merged_eval_config = eval_cfg.config.copy()
                merged_eval_config.update(eval_gen_config) # Add generation params to evaluator config

                evaluator = EvaluatorRegistry.create(eval_cfg.name, merged_eval_config)
                eval_output = evaluator.evaluate(self.actor_model, self.tokenizer, val_dataset)
                results.append(eval_output)
            except Exception as e:
                logging.error(f"Error during evaluation '{eval_cfg.name}': {e}", exc_info=True)
        return results
