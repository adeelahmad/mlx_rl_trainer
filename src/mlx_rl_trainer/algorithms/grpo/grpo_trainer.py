"""
Concrete GRPO Trainer implementation.
"""

import uuid
import logging
import time
import json
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from mlx_rl_trainer.core.trainer import TrainingMetrics

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

    # 2. Fix generate_rollouts to return proper rollout_batch
    def generate_rollouts(
        self, batch_prompts_data: List[Dict], update_step: int
    ) -> Tuple[Dict[str, mx.array], float, Dict[str, float]]:
        """
        Generate rollouts and prepare the rollout batch for training.

        Returns:
            rollout_batch: Dict with keys ['tokens', 'response_mask', 'advantages', 'ref_log_probs']
            avg_reward: Average reward across all samples
            raw_rewards_breakdown: Detailed reward breakdown
        """
        # Unwrap batch
        if not isinstance(batch_prompts_data, list) or len(batch_prompts_data) == 0:
            logger.warning("Empty batch_prompts_data in generate_rollouts")
            return {}, 0.0, {}

        batch_data = batch_prompts_data[0]

        # Extract prompts - try multiple possible keys
        prompts_text = (
            batch_data.get("raw_prompt")
            or batch_data.get("prompt")
            or batch_data.get("text")
            or []
        )

        if not prompts_text:
            logger.error(
                f"No prompts found in batch_data. Available keys: {list(batch_data.keys())}"
            )
            return {}, 0.0, {}

        logger.info(f"Processing {len(prompts_text)} prompts for rollout generation")

        # Tokenize prompts
        prompts_tokens = []
        for prompt in prompts_text:
            tokens = self.tokenizer.encode(str(prompt), add_special_tokens=True)
            prompts_tokens.append(tokens)

        # Pad prompts to same length
        max_prompt_len = max(len(t) for t in prompts_tokens)
        prompts_padded = []
        for tokens in prompts_tokens:
            padding = [self.tokenizer.pad_token_id] * (max_prompt_len - len(tokens))
            prompts_padded.append(padding + tokens)

        prompts_mx = mx.array(prompts_padded, dtype=mx.int32)
        logger.debug(f"Prompts tokenized: shape={prompts_mx.shape}")

        # Generate responses using actor model
        logger.debug("Generating responses with actor model...")
        responses_mx, actor_log_probs = self.model_manager.generate_with_logprobs(
            model=self.actor_model,
            prompts=prompts_mx,
            tokenizer=self.tokenizer,
            temp=self.config.trainer.temperature,
            max_tokens=self.config.data.max_gen_len,
        )
        logger.debug(f"Generated responses: shape={responses_mx.shape}")

        # Get reference log probs
        logger.debug("Computing reference log probs...")
        ref_log_probs = self.model_manager.get_logprobs_for_sequence(
            model=self.ref_model, prompts=prompts_mx, responses=responses_mx
        )
        logger.debug(f"Reference log probs: shape={ref_log_probs.shape}")

        # Decode responses for reward calculation
        responses_text = self.tokenizer.batch_decode(
            responses_mx.tolist(), skip_special_tokens=True
        )

        # Calculate rewards
        logger.debug("Computing rewards...")
        rewards_list = []
        for i, (prompt, response) in enumerate(zip(prompts_text, responses_text)):
            try:
                reward_dict = self.reward_composer.compute_reward(
                    prompt=str(prompt), response=str(response), context={}
                )
                reward_value = reward_dict.get("total", 0.0)
                rewards_list.append(reward_value)
                logger.debug(f"Sample {i}: reward={reward_value:.4f}")
            except Exception as e:
                logger.error(f"Error computing reward for sample {i}: {e}")
                rewards_list.append(0.0)

        rewards_mx = mx.array(rewards_list, dtype=mx.float32)
        logger.debug(
            f"Rewards computed: shape={rewards_mx.shape}, mean={float(mx.mean(rewards_mx).item()):.4f}"
        )

        # Compute advantages
        logger.debug("Computing advantages...")
        advantages = self.algorithm.compute_advantages(
            rewards_flat=rewards_mx,
            samples_per_prompt=1,  # Adjust if you use multiple samples per prompt
        )
        logger.debug(f"Advantages computed: shape={advantages.shape}")

        # Create response mask (1 for real tokens, 0 for padding)
        response_mask = mx.array(
            [
                [1.0 if token != self.tokenizer.pad_token_id else 0.0 for token in row]
                for row in responses_mx.tolist()
            ],
            dtype=mx.float32,
        )
        logger.debug(f"Response mask created: shape={response_mask.shape}")

        # Concatenate prompts and responses for full sequence
        full_tokens = mx.concatenate([prompts_mx, responses_mx], axis=1)
        logger.debug(f"Full tokens created: shape={full_tokens.shape}")

        # Prepare rollout_batch with CORRECT keys
        rollout_batch = {
            "tokens": full_tokens,
            "response_mask": response_mask,
            "advantages": advantages,
            "ref_log_probs": ref_log_probs,
        }

        avg_reward = float(mx.mean(rewards_mx).item())
        raw_rewards_breakdown = {
            "total": avg_reward,
            "min": float(mx.min(rewards_mx).item()),
            "max": float(mx.max(rewards_mx).item()),
            "std": float(mx.std(rewards_mx).item()),
        }

        logger.info(f"Rollout generation complete. Avg reward: {avg_reward:.4f}")

        return rollout_batch, avg_reward, raw_rewards_breakdown

    def train_step(
        self, rollout_batch: Union[Dict[str, mx.array], List[Dict]], update_step: int
    ) -> TrainingMetrics:
        """
        Perform a single training step using the rollout batch.

        Args:
            rollout_batch: Either a dict with rollout data or list containing the dict
            update_step: Current training step number

        Returns:
            TrainingMetrics with all required fields including step_time_s and tokens_per_sec
        """
        step_start_time = time.time()

        # Handle list input (unwrap if needed)
        if isinstance(rollout_batch, list):
            if len(rollout_batch) == 0:
                logger.error("Empty rollout batch list in train_step")
                return TrainingMetrics(
                    loss=0.0,
                    reward_mean=0.0,
                    reward_std=0.0,
                    kl_divergence=0.0,
                    grad_norm=0.0,
                    learning_rate=self._get_learning_rate(),
                    epoch=self.current_epoch,
                    step=update_step,
                    step_time_s=time.time() - step_start_time,
                    tokens_per_sec=0.0,
                )
            rollout_batch = rollout_batch[0]

        # Validate required keys
        required_keys = {"tokens", "response_mask", "advantages", "ref_log_probs"}
        missing_keys = required_keys - set(rollout_batch.keys())

        if missing_keys:
            logger.error(
                f"Rollout batch missing required keys: {missing_keys}. "
                f"Has: {list(rollout_batch.keys())}"
            )
            return TrainingMetrics(
                loss=0.0,
                reward_mean=0.0,
                reward_std=0.0,
                kl_divergence=0.0,
                grad_norm=0.0,
                learning_rate=self._get_learning_rate(),
                epoch=self.current_epoch,
                step=update_step,
                step_time_s=time.time() - step_start_time,
                tokens_per_sec=0.0,
            )

        logger.debug(
            f"train_step: Processing batch with tokens.shape={rollout_batch['tokens'].shape}"
        )

        # Calculate loss and gradients
        loss, grads, metrics = self.algorithm.calculate_loss_and_grads(
            rollout_batch=rollout_batch,
            full_config=self.config,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Compute gradient norm
        grad_norm = 0.0
        if grads:
            grad_norm = float(
                mx.sqrt(sum(mx.sum(g * g) for g in grads.values())).item()
            )
            logger.debug(f"Gradient norm: {grad_norm:.6f}")

            # Apply gradients
            self.optimizer.apply_gradients(
                grads, self.actor_model.trainable_parameters()
            )
        else:
            logger.warning("No gradients to apply in train_step")

        # Extract reward statistics from advantages
        advantages = rollout_batch.get("advantages", mx.array([0.0]))
        reward_mean = float(mx.mean(advantages).item())
        reward_std = float(mx.std(advantages).item())

        # Calculate tokens processed
        tokens = rollout_batch.get("tokens", mx.array([]))
        total_tokens = tokens.size if tokens.size > 0 else 0

        # Calculate step time and throughput
        step_time = time.time() - step_start_time
        tokens_per_sec = total_tokens / step_time if step_time > 0 else 0.0

        logger.debug(
            f"train_step complete: loss={float(loss.item()):.6f}, "
            f"kl={metrics.get('kl_divergence', 0.0):.6f}, "
            f"time={step_time:.3f}s, tokens/s={tokens_per_sec:.1f}"
        )

        return TrainingMetrics(
            loss=float(loss.item()),
            reward_mean=reward_mean,
            reward_std=reward_std,
            kl_divergence=metrics.get("kl_divergence", 0.0),
            grad_norm=grad_norm,
            learning_rate=self._get_learning_rate(),
            epoch=self.current_epoch,
            step=update_step,
            step_time_s=step_time,
            tokens_per_sec=tokens_per_sec,
        )

    def _get_learning_rate(self) -> float:
        """Helper to safely extract learning rate from optimizer."""
        lr = self.optimizer.learning_rate
        if hasattr(lr, "item"):
            return float(lr.item())
        return float(lr)

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
