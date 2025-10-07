import logging, time, gc
from typing import Dict, Any, List, Optional, Tuple, Callable
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
import numpy as np
from mlx_lm.tuner.utils import build_schedule
from mlx_lm.sample_utils import make_logits_processors
from mlx_rl_trainer.core.trainer import BaseTrainer, TrainingMetrics
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler, make_dynamic_tag_bias_processor, _mask_after_answer
from mlx_rl_trainer.monitoring.metrics_logger import _maybe_log_samples
from .grpo_algorithm import GRPOAlgorithm

logger = logging.getLogger(__name__)

class GRPOTrainer(BaseTrainer):
    def _setup(self) -> Tuple[int, int]:
        self.actor_model, self.tokenizer = self.model_manager.load_model(self.config.model.model_path, "actor", is_trainable=True, apply_lora=self.config.model.use_lora, lora_config=self.config.model.model_dump())
        self.ref_model, _ = self.model_manager.load_model(self.config.model.ref_model_path, "reference", is_trainable=False)
        self.data_manager.tokenizer = self.tokenizer
        
        self.grpo_algorithm = GRPOAlgorithm(self.config, self.actor_model, self.ref_model)
        self.optimizer = optim.AdamW(learning_rate=self.config.trainer.learning_rate, betas=(self.config.trainer.optimizer_beta1, self.config.trainer.optimizer_beta2), weight_decay=self.config.trainer.optimizer_weight_decay)
        self.lr_scheduler = build_schedule(self.config.trainer.lr_schedule_config)
        
        start_step, metadata = self.checkpoint_manager.load_latest_state(self.actor_model, self.optimizer)
        return start_step, metadata.get("epoch", 0)

    def generate_rollouts(self, batch_prompts: List[Dict], update_step: int) -> Tuple[Dict, float, Dict]:
        self.actor_model.eval()
        prompts_mx_list = [d['input_ids'] for d in batch_prompts]
        max_prompt_len = max(p.shape[0] for p in prompts_mx_list) if prompts_mx_list else 0
        
        pad_id = self.tokenizer.pad_token_id
        prompts_padded = [mx.pad(p, ((max_prompt_len - p.shape[0], 0)), constant_values=pad_id) for p in prompts_mx_list]
        prompts_mx = mx.stack(prompts_padded, axis=0)
        
        prompts_replicated = mx.repeat(prompts_mx, repeats=self.config.trainer.num_rollout_samples, axis=0)
        batch_size = prompts_replicated.shape[0]

        mcq_flags = [p['meta_data'].get('is_mcq', False) for p in batch_prompts] * self.config.trainer.num_rollout_samples

        # Generation loop
        y, log_probs_list = [], []
        logits, cache = self.actor_model(prompts_replicated, cache=self.model_manager.make_prompt_cache(self.actor_model))
        
        sampler = safe_make_sampler(self.config, self.config.generation.think_temperature)
        logit_processor = make_dynamic_tag_bias_processor(self.tokenizer, self.config, mcq_flags)

        hist = prompts_replicated.tolist()
        
        for i in range(self.config.data.max_gen_len):
            logits = logit_processor(hist, logits[:, -1, :])
            next_toks = sampler(logits)
            
            log_probs = nn.log_softmax(logits, axis=-1)
            next_log_probs = mx.take_along_axis(log_probs, next_toks[:, None], axis=-1).squeeze(-1)
            
            y.append(next_toks)
            log_probs_list.append(next_log_probs)
            
            [h.append(t) for h, t in zip(hist, next_toks.tolist())]
            
            logits, cache = self.actor_model(next_toks[:, None], cache=cache)

        responses_mx = mx.stack(y, axis=1) if y else mx.zeros((batch_size, 0), dtype=mx.int32)
        actor_log_probs = mx.stack(log_probs_list, axis=1) if log_probs_list else mx.zeros((batch_size, 0))

        ref_log_probs = self.model_manager.get_logprobs_for_sequence(self.ref_model, prompts_replicated, responses_mx)
        
        decoded = self.tokenizer.batch_decode(responses_mx.tolist(), skip_special_tokens=True)
        
        # Rewards
        rewards_total, rewards_breakdown = [], {}
        prompts_data_replicated = [p for p in batch_prompts for _ in range(self.config.trainer.num_rollout_samples)]

        contexts = [self.reward_composer.context_cls(
            generated_text=decoded[i], prompt_text=prompts_data_replicated[i]['raw_prompts'],
            reference_completion=prompts_data_replicated[i]['raw_completions'],
            test_cases=prompts_data_replicated[i]['raw_test_cases'],
            metadata=prompts_data_replicated[i]['meta_data'],
        ) for i in range(batch_size)]
        
        batch_rewards = self.reward_composer.batch_compute(contexts)
        
        # Unpack rewards
        rewards_total = [r['total'] for r in batch_rewards]
        for key in batch_rewards[0]:
            rewards_breakdown[key] = [r[key] for r in batch_rewards]
            
        advantages = self.grpo_algorithm.compute_advantages(mx.array(rewards_total), self.config.trainer.num_rollout_samples)
        
        response_mask = (responses_mx != pad_id).astype(mx.float32)
        response_mask = _mask_after_answer(responses_mx, response_mask, self.tokenizer, self.config)
        
        _maybe_log_samples(self.config, update_step, prompts_data_replicated, decoded, rewards_breakdown, "n/a", self.run_id, False)

        rollout_batch = {
            "tokens": mx.concatenate([prompts_replicated, responses_mx], axis=1),
            "response_mask": response_mask,
            "advantages": advantages,
            "ref_log_probs": ref_log_probs,
            "actor_log_probs": actor_log_probs
        }
        self.actor_model.train()
        return rollout_batch, np.mean(rewards_total), {k: np.mean(v) for k, v in rewards_breakdown.items()}

    def train_step(self, rollout_batch: Dict[str, mx.array], update_step: int) -> TrainingMetrics:
        start_time = time.time()
        loss, grads, metrics = self.grpo_algorithm.calculate_loss_and_grads(rollout_batch, self.config, self.tokenizer.pad_token_id)
        
        self.optimizer.learning_rate = self.lr_scheduler(update_step)
        self.optimizer.apply_gradients(grads, self.actor_model.trainable_parameters())
        mx.eval(self.actor_model.parameters(), self.optimizer.state)
        
        grad_norm = np.linalg.norm([np.linalg.norm(v.astype(np.float32).flatten()) for v in tree_flatten(grads)[1]])
        
        return TrainingMetrics(
            loss=loss.item(), reward_mean=mx.mean(rollout_batch['advantages']).item(),
            grad_norm=grad_norm, learning_rate=self.optimizer.learning_rate,
            step_time_s=time.time() - start_time, kl_divergence=metrics.get('kl_divergence', 0.0),
            epoch=self.current_epoch, step=update_step
        )

    def evaluate(self, update_step: int) -> List[Any]:
        # Placeholder for full evaluation logic
        logger.info(f"Evaluation at step {update_step} is a placeholder.")
        return []
