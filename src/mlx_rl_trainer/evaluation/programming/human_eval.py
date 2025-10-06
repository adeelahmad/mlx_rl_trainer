# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/programming/human_eval.py
# revision_no: 001
# goals_of_writing_code_block: HumanEval benchmark evaluator implementation.
# type_of_code_response: add new code
"""HumanEval benchmark evaluator implementation."""

import logging
from datasets import Dataset
import random
import json
from typing import Dict, Any, List, Optional
import concurrent.futures # For multiprocessing
from multiprocessing import get_context, Queue, Process # For safe execution

from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry
from mlx_rl_trainer.core.trainer import EvaluationMetrics, DataLoadError
from mlx_rl_trainer.rewards.programming.code_execution import CodeExecutionReward # Reuse code execution logic
from mlx_rl_trainer.rewards.context import RewardContext # For reward context
from mlx_lm.tokenizer_utils import TokenizerWrapper # For tokenizer type hinting
import mlx.core as mx # For model generation
from mlx_lm.models import cache # For KV cache in generation
from mlx_lm.sample_utils import make_logits_processors # For generation
from mlx_rl_trainer.utils.mlx_utils import safe_make_sampler # For sampler

logger = logging.getLogger(__name__)


def _execute_code_in_isolated_process_wrapper(code: str, test_cases_json: str, config: Dict[str, Any], result_queue: Queue):
    """
    Wrapper to run CodeExecutionReward's compute in a separate process.
    Requires JSON-serialized test_cases and config.
    """
    try:
        test_cases = json.loads(test_cases_json)
        reward_fn = CodeExecutionReward(config=config)
        context = RewardContext(
            generated_text=code,
            prompt_text="", # Not used by CodeExecutionReward for direct execution
            reference_completion="", # Not used
            test_cases=test_cases,
            metadata={}
        )
        pass_score = reward_fn.compute(context)
        result_queue.put(pass_score)
    except Exception as e:
        logger.error(f"Error in isolated code execution process: {e}")
        result_queue.put(0.0) # Return 0.0 on error

@EvaluatorRegistry.register("human_eval")
class HumanEvalEvaluator(BaseEvaluator):
    """
    Evaluates model performance on the HumanEval code generation benchmark.

    It uses an internal `CodeExecutionReward` to determine the pass rate of
    generated solutions against provided test cases, supporting pass@k metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.k_values = config.get("k_values", [1])
        self.num_problems_to_evaluate = config.get("num_samples", 10)
        self.timeout = config.get("timeout", 5) # Code execution timeout
        self.max_gen_len = config.get("max_gen_len", 512) # Max tokens to generate per solution

        # Generation parameters for evaluation (from ExperimentConfig)
        self.gen_temp = config.get("temperature", 0.0) # Often greedy for benchmarks
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", 0) # 0 means disabled

        # Pass relevant config to the internal CodeExecutionReward
        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": config.get("memory_limit", 512),
            "allow_imports": config.get("allow_imports", ["math", "re", "json"])
        }

        logger.debug(f"HumanEvalEvaluator initialized with k_values: {self.k_values}, num_problems: {self.num_problems_to_evaluate}.")

    def evaluate(self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset) -> EvaluationMetrics:
        """
        Runs the HumanEval evaluation.

        Args:
            model: The MLX model to evaluate.
            tokenizer: The tokenizer.
            dataset: The HumanEval dataset (expected to contain 'prompt' and 'test_cases').

        Returns:
            An `EvaluationMetrics` object with pass@k results.
        """
        if not dataset or len(dataset) == 0:
            raise DataLoadError("HumanEval dataset is empty.")

        # Select a subset of problems
        sampled_problems = dataset.shuffle(seed=self.config.get("seed", 42)).select(
            range(min(len(dataset), self.num_problems_to_evaluate))
        )

        total_problems = len(sampled_problems)
        if total_problems == 0: return EvaluationMetrics(task_name=self.name, pass_rate=0.0)

        max_k_required = max(self.k_values)
        passed_at_k_counts: Dict[int, int] = {k: 0 for k in self.k_values}

        logger.info(f"Evaluating {total_problems} problems for {self.name} (max_k={max_k_required})...")

        mp_context = get_context('spawn')

        # Use ProcessPoolExecutor for parallel code execution (for safety and speed)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, self.config.get("num_workers", 1)), mp_context=mp_context) as executor:

            for problem_idx, problem in enumerate(sampled_problems):
                problem_prompt = problem.get("prompt", "")
                problem_test_cases_raw = problem.get("test_cases", [])

                problem_test_cases: List[Dict[str, Any]] = []
                for tc_json_str in problem_test_cases_raw:
                    try: problem_test_cases.append(json.loads(tc_json_str))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed test case for problem {problem_idx}: {tc_json_str[:50]}...")

                if not problem_prompt or not problem_test_cases: continue

                # 1. Generate max_k solutions using the model
                generated_solutions: List[str] = self._generate_solutions(model, tokenizer, problem_prompt, max_k_required)

                # 2. Evaluate each solution in parallel using subprocesses
                solution_pass_scores: List[float] = []
                futures_with_queues = []
                for solution_text in generated_solutions:
                    result_queue = mp_context.Queue(1) # Queue for IPC
                    future = executor.submit(
                        _execute_code_in_isolated_process_wrapper,
                        solution_text,
                        json.dumps(problem_test_cases), # Pass test cases as JSON string
                        self.code_execution_config,
                        result_queue
                    )
                    futures_with_queues.append((future, result_queue))

                for future, result_queue in futures_with_queues:
                    try:
                        pass_score = result_queue.get(timeout=self.timeout + 2) # Get result from queue
                        solution_pass_scores.append(pass_score)
                    except concurrent.futures.TimeoutError:
                        logger.warning("Code execution subprocess timed out.")
                        solution_pass_scores.append(0.0) # Penalty for timeout
                    except Exception as e:
                        logger.error(f"Error getting result from subprocess: {e}", exc_info=True)
                        solution_pass_scores.append(0.0)
                    finally:
                        if result_queue: # Ensure queue is closed
                            try: result_queue.close()
                            except Exception: pass

                # 3. Compute pass@k for this problem: Check if *any* of the k solutions pass
                solution_pass_statuses = [score > 0.99 for score in solution_pass_scores] # True if all tests passed (score ~ 1.0)

                for k_val in sorted(self.k_values):
                    if any(solution_pass_statuses[:k_val]): # Check if any of the first k solutions passed
                        passed_at_k_counts[k_val] += 1

        # 4. Final pass@k rates
        final_pass_rates: Dict[str, float] = {}
        for k_val, passed_count in passed_at_k_counts.items():
            final_pass_rates[f"pass@{k_val}"] = passed_count / total_problems

        primary_pass_rate = final_pass_rates.get("pass@1", 0.0) # Usually pass@1 is the primary metric

        logger.info(f"{self.name} Evaluation Summary:")
        for metric, value in final_pass_rates.items():
            logger.info(f"  - {metric}: {value:.4f}")

        return EvaluationMetrics(
            task_name=self.name,
            pass_rate=primary_pass_rate,
            additional_info=final_pass_rates
        )

    def _generate_solutions(self, model: Any, tokenizer: TokenizerWrapper, prompt: str, num_solutions: int) -> List[str]:
        """
        Generates multiple solutions for a given prompt using the model.

        Args:
            model: The MLX model.
            tokenizer: The tokenizer.
            prompt: The problem prompt.
            num_solutions: How many distinct solutions to generate (for pass@k).

        Returns:
            A list of generated code strings.
        """
        logger.debug(f"Generating {num_solutions} solutions for prompt: '{prompt[:60]}...'")

        model.eval() # Set model to eval mode for generation

        # Apply chat template (assuming general system prompt or no system prompt for HumanEval)
        from mlx_rl_trainer.utils.text_utils import apply_chat_template_wrapper
        formatted_prompt = apply_chat_template_wrapper(tokenizer, prompt, self.config.get("system_prompt", None)) # System prompt from main config

        encoded_prompt = tokenizer.encode(formatted_prompt, add_special_tokens=True)
        prompt_mx = mx.array([encoded_prompt] * num_solutions, dtype=mx.int32) # Batch prompts for generation

        # Get generation parameters from config (from ExperimentConfig -> GenerationConfig)
        max_gen_tokens = self.max_gen_len # Use evaluator's configured max_gen_len
        gen_temp = self.gen_temp # Use evaluator's configured temperature
        top_p = self.top_p
        top_k = self.top_k

        # Sampler and logit processors
        sampler = safe_make_sampler(self.config, temp=gen_temp) # Pass config
        logits_processors = make_logits_processors(
            repetition_penalty=1.1, # Default for evaluation
            repetition_context_size=20,
            logit_bias=None
        )

        generated_ids_batch = []

        # For each solution in the batch (can be parallelized more if mlx_lm had batched generation)
        # Current mlx_lm.generate usually does one-by-one, simulating batch for eval.
        for i in range(num_solutions):
            current_prompt_input = prompt_mx[i:i+1] # Take one prompt from batch
            caches = cache.make_prompt_cache(model, max_kv_size=current_prompt_input.shape[1] + max_gen_tokens)

            out = model(current_prompt_input, cache=caches)
            next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

            tokens_to_add = []
            current_ids_history = current_prompt_input.tolist()[0] # History for logit processors

            for _ in range(max_gen_tokens):
                logits_to_process = next_logits
                for proc_fn in logits_processors:
                    logits_to_process = proc_fn([current_ids_history], logits_to_process) # Pass list of lists for batching

                next_token_mx = sampler(logits_to_process) # Sample one token (mx.array of shape (1,))

                if next_token_mx.item() == tokenizer.eos_token_id:
                    break

                tokens_to_add.append(next_token_mx.item())
                current_ids_history.append(next_token_mx.item())

                # Forward pass for the next token
                out = model(next_token_mx[None, :], cache=caches) # Input needs to be 2D (B,1)
                next_logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].astype(mx.float32)

            generated_ids_batch.append(tokens_to_add)

        return tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)

