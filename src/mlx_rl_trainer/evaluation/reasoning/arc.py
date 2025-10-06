# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/reasoning/arc.py
# revision_no: 001
# goals_of_writing_code_block: ARC reasoning benchmark evaluator implementation.
# type_of_code_response: add new code
"""ARC reasoning benchmark evaluator implementation."""

import logging
import random
import re  # For regex in answer extraction
import mlx.core as mx  # For model generation
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper  # For tokenizer type hinting
from mlx_lm.models import cache  # For KV cache in generation
from mlx_lm.sample_utils import make_logits_processors  # For generation

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics
from mlx_rl_trainer.utils.text_utils import _normalize_ans_for_match  # For scoring
from mlx_rl_trainer.core.config import ExperimentConfig  # For full config access
from mlx_rl_trainer.utils.text_utils import (
    apply_chat_template_wrapper,
)  # For consistent prompting
from mlx_rl_trainer.utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
)  # For generation utils

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("arc")
class ARCEvaluator(BaseEvaluator):
    """
    Evaluates model performance on the ARC (AI2 Reasoning Challenge) benchmark.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples_to_evaluate = config.get("num_samples", 10)
        self.max_gen_len = config.get("max_gen_len", 256)
        self.system_prompt = config.get("system_prompt", None)  # From ExperimentConfig

        # Generation parameters for evaluation (from ExperimentConfig -> GenerationConfig)
        self.gen_temp = config.get(
            "temperature", 0.0
        )  # Use configured temp, fallback to greedy
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0)  # 0 means disabled
        logger.info(f"ARCEvaluator initialized with config: {config}")

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        logger.info(f"Running ARC evaluation on {len(dataset)} samples.")

        # Select a subset of problems
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_samples_to_evaluate))
        )

        total_problems = len(sampled_problems)
        if total_problems == 0:
            return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        correct_count = 0

        for problem_idx, problem in enumerate(sampled_problems):
            prompt = problem.get("prompt", "")
            # ARC often has multiple choice options that need to be processed
            choices = problem.get("choices", {"text": [], "label": []})
            correct_label = problem.get("answerKey", "")

            if not prompt or not choices["text"] or not correct_label:
                continue  # Skip if missing crucial info

            # 1. Generate solution
            generated_text = self._generate_solution(model, tokenizer, prompt, choices)

            # 2. Extract answer and compare
            predicted_answer_label = self._extract_arc_answer(generated_text, choices)

            if (
                predicted_answer_label is not None
                and predicted_answer_label == correct_label
            ):
                correct_count += 1

        accuracy = float(correct_count / total_problems)
        logger.info(
            f"ARC Evaluation: Accuracy = {accuracy:.4f} ({correct_count}/{total_problems})"
        )

        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=accuracy,
            additional_info={"num_samples": total_problems},
        )

    def _generate_solution(
        self,
        model: Any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        choices: Dict[str, List[str]],
    ) -> str:
        """Helper to generate a single solution for an ARC problem."""
        # For ARC, prompt often includes question + choices.
        full_prompt = f"{prompt}\nChoices:\n"
        for label, text in zip(choices["label"], choices["text"]):
            full_prompt += f"{label}) {text}\n"
        full_prompt += "Answer:"

        model.eval()  # Set model to eval mode for generation

        # Apply chat template
        formatted_full_prompt = apply_chat_template_wrapper(
            tokenizer, full_prompt, self.system_prompt
        )

        encoded_prompt = tokenizer.encode(
            formatted_full_prompt, add_special_tokens=True
        )
        prompt_mx = mx.array([encoded_prompt], dtype=mx.int32)

        # Initialize KV cache
        caches = cache.make_prompt_cache(
            model, max_kv_size=prompt_mx.shape[1] + self.max_gen_len
        )

        # First forward pass
        out = model(prompt_mx, cache=caches)
        next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
            mx.float32
        )

        # Sampler
        sampler = safe_make_sampler(
            self.config, temp=self.gen_temp
        )  # Pass eval config as args
        logits_processors = make_logits_processors(
            repetition_penalty=1.1,  # Default for evaluation
            repetition_context_size=20,
            logit_bias=None,
        )

        generated_ids = []
        current_ids = encoded_prompt  # Keep track of generated tokens

        for _ in range(self.max_gen_len):
            logits_to_process = next_logits
            for proc_fn in logits_processors:
                logits_to_process = proc_fn(
                    [current_ids], logits_to_process
                )  # Pass list of lists for batching

            next_token = sampler(logits_to_process)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())
            current_ids.append(next_token.item())  # Update history for next iteration

            # Forward pass for next token
            out = model(next_token[None, :], cache=caches)
            next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(
                mx.float32
            )

        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _extract_arc_answer(
        self, generated_text: str, choices: Dict[str, List[str]]
    ) -> Optional[str]:
        """Extracts the predicted answer label (A, B, C, D) from generated text."""
        # This is highly dependent on output format. Look for A, B, C, D at the end.
        match = re.search(r"\b([A-D])\s*$", generated_text.strip())
        if match:
            return match.group(1)
        return None
