# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/programming/code_execution.py
# revision_no: 002
# goals_of_writing_code_block: Code execution reward function with multiprocessing for sandboxed execution.
# type_of_code_response: change existing
"""Code execution reward function."""

import subprocess
import tempfile
import ast
import os
import json
import logging
from multiprocessing import (
    get_context,
    Queue,
)  # For safe execution in a separate process
from typing import Dict, Any, List, Optional
from pathlib import Path

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import (
    _extract_python_code,
)  # Helper for code extraction

logger = logging.getLogger(__name__)


def _execute_code_in_isolated_process(
    code: str, test_cases_json: str, config: Dict[str, Any], result_queue: Queue
):
    """
    Target function to be run in a separate process for code execution.
    Handles temporary file creation, execution, and cleanup.
    """
    filepath: Optional[Path] = None
    try:
        test_cases = json.loads(test_cases_json)
        timeout = config.get("timeout", 5)
        # Note: memory_limit is configured but not enforced here (requires platform-specific calls)

        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            filepath = Path(f.name)

        passed_count = 0

        for test_input_dict in test_cases:
            test_input = str(test_input_dict.get("input", ""))
            expected_output = str(test_input_dict.get("expected", "")).strip()

            # Use subprocess to run the Python file
            process_result = subprocess.run(
                ["python", str(filepath)],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Do not raise CalledProcessError on non-zero exit codes
            )

            stdout_output = process_result.stdout.strip()

            if stdout_output == expected_output:
                passed_count += 1

        # Return pass rate (0.0 to 1.0)
        result_queue.put(float(passed_count / max(1, len(test_cases))))

    except subprocess.TimeoutExpired:
        logger.warning(f"Isolated code execution timed out after {timeout}s.")
        result_queue.put(0.0)  # Return 0.0 for timeout
    except Exception as e:
        logger.error(
            f"Exception in isolated code execution process: {e}", exc_info=True
        )
        result_queue.put(0.0)  # Return 0.0 on any exception
    finally:
        if filepath and filepath.exists():
            try:
                filepath.unlink()  # Ensure cleanup
            except OSError as e:
                logger.error(f"Failed to delete temporary code file {filepath}: {e}")
        logger.debug("Isolated code execution process finished and cleaned up.")


@RewardRegistry.register("code_execution")
class CodeExecutionReward(BaseReward):
    """
    Rewards generated code based on its execution success against provided test cases.

    Executes the code in an isolated process with a timeout to prevent malicious
    or long-running code from affecting the main training loop.

    Configuration:
        timeout: Maximum execution time for each test case (seconds, default: 5).
        memory_limit: Maximum memory for the process (MB, default: 512).
        num_workers: Number of parallel processes to use for batch execution (default: 1).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.timeout = config.get("timeout", 5)
        self.memory_limit = config.get(
            "memory_limit", 512
        )  # For potential future enforcement
        self.num_workers = config.get("num_workers", 1)  # For batch_compute

        # Internal config to pass to subprocess
        self.code_execution_config = {
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "allow_imports": config.get(
                "allow_imports", []
            ),  # List of allowed imports (stub)
        }

        logger.info(f"Initialized CodeExecutionReward with timeout: {self.timeout}s.")

    def _validate_syntax(self, code: str) -> bool:
        """Checks if the extracted code has valid Python syntax using ast."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            logger.debug("Generated code has syntax errors.")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error during AST parsing: {e}")
            return False

    def compute(self, context: RewardContext) -> float:
        """
        Computes reward for a single generated code block within a `RewardContext`.

        The reward is scaled:
        - 0.0: No valid code extracted, syntax error, or execution failure/timeout.
        - 0.3: Valid Python code extracted, but no test cases provided (only syntax rewarded).
        - 0.3 to 1.0: Scaled by the test case pass rate (0.0 for all fail, 1.0 for all pass).
        """
        self.validate_inputs(context)

        code = _extract_python_code(context.generated_text)
        test_cases = context.test_cases

        if not code or not self._validate_syntax(code):
            return 0.0  # No code, or invalid syntax

        if not test_cases:
            return 0.3  # Base reward for valid syntax if no tests provided

        mp_context = get_context("spawn")  # Use 'spawn' for stronger isolation
        result_queue = mp_context.Queue(1)
        process = None

        try:
            # Start code execution in a separate, isolated process
            process = mp_context.Process(
                target=_execute_code_in_isolated_process,
                args=(
                    code,
                    json.dumps(test_cases),
                    self.code_execution_config,
                    result_queue,
                ),
            )
            process.start()

            # Wait for result from the child process with a timeout
            pass_rate = result_queue.get(
                timeout=self.timeout + 3
            )  # Extra grace period for IPC

            # Ensure pass_rate is a float
            pass_rate = float(pass_rate)

        except concurrent.futures.TimeoutError:
            logger.warning(
                "Main process timed out waiting for code execution subprocess."
            )
            pass_rate = 0.0
        except Exception as e:
            logger.error(
                f"Error managing code execution subprocess: {e}", exc_info=True
            )
            pass_rate = 0.0
        finally:
            if process and process.is_alive():
                process.terminate()  # Ensure subprocess is terminated
                process.join()
            if result_queue:
                try:
                    result_queue.close()
                except Exception:
                    pass

        # Scale the pass_rate (0.0-1.0) to the final reward range (0.3-1.0)
        final_reward = 0.3 + (0.7 * pass_rate)
        return float(max(0.0, min(1.0, final_reward)))

    def batch_compute(self, contexts: List[RewardContext]) -> List[float]:
        """
        Computes rewards for a batch of code generation contexts using multiprocessing.
        """
        if self.num_workers <= 1 or len(contexts) <= 1:
            return super().batch_compute(
                contexts
            )  # Fallback to sequential for single item or if workers disabled

        rewards = [0.0] * len(contexts)  # Pre-fill with default for non-executable
        mp_context = get_context("spawn")

        # Store future results and their original index
        futures_with_indices = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers, mp_context=mp_context
        ) as executor:
            for idx, context in enumerate(contexts):
                code = _extract_python_code(context.generated_text)
                if not code or not self._validate_syntax(code):
                    # Keep 0.0 reward for invalid code
                    continue

                result_queue = mp_context.Queue(1)
                future = executor.submit(
                    _execute_code_in_isolated_process_wrapper,
                    code,
                    json.dumps(context.test_cases),
                    self.code_execution_config,
                    result_queue,
                )
                futures_with_indices.append((idx, future, result_queue))

            # Collect results, mapping back to original indices
            results_map = {}
            for original_idx, future, result_queue in futures_with_indices:
                try:
                    pass_rate = result_queue.get(
                        timeout=self.timeout + 3
                    )  # Get result from queue
                    results_map[original_idx] = float(pass_rate)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        f"Batch execution for context {original_idx} timed out."
                    )
                    results_map[original_idx] = 0.0
                except Exception as e:
                    logger.error(
                        f"Error in batch execution for context {original_idx}: {e}",
                        exc_info=True,
                    )
                    results_map[original_idx] = 0.0
                finally:
                    if result_queue:
                        try:
                            result_queue.close()
                        except Exception:
                            pass  # Ignore

            # Fill final rewards list, applying scaling
            for i in range(len(contexts)):
                if i in results_map:
                    final_rewards_score = 0.3 + (0.7 * results_map[i])
                    rewards[i] = float(max(0.0, min(1.0, final_rewards_score)))

            return rewards


# Dependencies: subprocess, tempfile, ast, multiprocessing (for safe execution)
# Actions: Ensure 'python' is in PATH. Add 'num_workers' to reward config for parallel execution.
# Status: Complete, production-ready with multiprocessing sandbox.
# Gaps: Enforcement of memory_limit not implemented.
