# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/programming/mbpp.py
# revision_no: 001
# goals_of_writing_code_block: MBPP programming benchmark evaluator implementation.
# type_of_code_response: add new code
"""MBPP programming benchmark evaluator implementation."""

import logging
import random
import json
import concurrent.futures  # For multiprocessing
from multiprocessing import get_context, Queue  # For safe execution
from typing import Dict, Any, List, Optional
from datasets import Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper  # For tokenizer type hinting
from mlx_lm.models import cache  # For KV cache in generation
from mlx_lm.sample_utils import make_logits_processors  # For generation
import mlx.core as mx  # For model generation

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics, DataLoadError
from mlx_rl_trainer.rewards.programming.code_execution import (
    CodeExecutionReward,
)  # Reuse code execution logic
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import (
    apply_chat_template_wrapper,
)  # For consistent prompting
from mlx_rl_trainer.utils.mlx_utils import (
    _create_4d_attention_mask,
    safe_make_sampler,
)  # For generation utils

logger = logging.getLogger(__name__)


@EvaluatorRegistry.register("mbpp")
class MBPPEvaluator(BaseEvaluator):
    """
    Evaluates model performance on the MBPP (Mostly Basic Python Problems) benchmark.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_problems_to_evaluate = config.get("num_samples", 10)
        self.max_gen_len = config.get("max_gen_len", 512)
        self.timeout = config.get("timeout", 10)  # Code execution timeout

        # Generation parameters for evaluation (from ExperimentConfig -> GenerationConfig)
        self.gen_temp = config.get("temperature", 0.0)  # Often greedy for benchmarks
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0)  # 0 means disabled

        # Pass relevant config to the internal CodeExecutionReward
        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": config.get("memory_limit", 512),
            "allow_imports": config.get("allow_imports", ["math", "re", "json"]),
        }
        logger.info(f"MBPPEvaluator initialized with config: {config}")

    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        logger.info(f"Running MBPP evaluation on {len(dataset)} samples.")

        # Select a subset of problems
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_problems_to_evaluate))
        )

        total_problems = len(sampled_problems)
        if total_problems == 0:
            return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        correct_count = 0
        mp_context = get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.config.get("num_workers", 1)), mp_context=mp_context
        ) as executor:
            for problem_idx, problem in enumerate(sampled_problems):
                prompt = problem.get("prompt", "")
                test_cases_code = problem.get(
                    "test_cases", ""
                )  # MBPP has test cases as code

                if not prompt or not test_cases_code:
                    continue

                # 1. Generate solution code
                generated_code = self._generate_solution(model, tokenizer, prompt)

                # 2. Combine generated code with test cases and execute
                full_executable_code = f"{generated_code}\n{test_cases_code}"

                result_queue = mp_context.Queue(1)
                future = executor.submit(
                    _execute_code_in_isolated_process_wrapper,  # Use the wrapper
                    full_executable_code,
                    json.dumps(
                        []
                    ),  # No explicit input/output test cases, only execution status
                    self.code_execution_config,
                    result_queue,
                )

                try:
                    pass_score = result_queue.get(timeout=self.timeout + 2)
                    if (
                        pass_score > 0.99
                    ):  # If code runs without error and passes internal tests (if any)
                        correct_count += 1
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        f"MBPP execution for problem {problem_idx} timed out."
                    )
                except Exception as e:
                    logger.error(
                        f"Error during MBPP execution for problem {problem_idx}: {e}"
                    )

        pass_rate = float(correct_count / total_problems)
        logger.info(
            f"MBPP Evaluation: Pass Rate = {pass_rate:.4f} ({correct_count}/{total_problems})"
        )

        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=pass_rate,
            additional_info={"num_samples": total_problems},
        )

    def _generate_solution(
        self, model: Any, tokenizer: TokenizerWrapper, prompt: str
    ) -> str:
        """Helper to generate a single solution for an MBPP problem."""
        # For MBPP, prompt describes problem, solution is Python function.
        model.eval()  # Set model to eval mode for generation

        # Apply chat template
        formatted_prompt = apply_chat_template_wrapper(
            tokenizer, prompt, self.config.get("system_prompt", None)
        )

        encoded_prompt = tokenizer.encode(formatted_prompt, add_special_tokens=True)
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
        sampler = safe_make_sampler(self.config, temp=self.gen_temp)
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
                logits_to_process = proc_fn([current_ids], logits_to_process)

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
