"""Code execution reward function."""
import subprocess, tempfile, ast, os, json, logging, concurrent.futures
from multiprocessing import get_context, Queue
from typing import Dict, Any, List, Optional
from pathlib import Path

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import _extract_python_code

logger = logging.getLogger(__name__)

def _validate_syntax_isolated(code: str) -> bool:
    """Checks if code is valid Python syntax (for isolated process)."""
    try: ast.parse(code); return True
    except Exception: return False

def _execute_code_in_isolated_process_wrapper(
    code: str, test_cases_json: str, config_json: str, result_queue: Queue
):
    """
    Wrapper function to execute code in an isolated process.
    """
    filepath: Optional[Path] = None
    try:
        test_cases = json.loads(test_cases_json)
        config = json.loads(config_json)
        timeout = config.get("timeout", 5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            filepath = Path(f.name)

        if not test_cases:
            if _validate_syntax_isolated(code):
                process_result = subprocess.run(["python", str(filepath)], capture_output=True, text=True, timeout=timeout, check=False)
                if process_result.returncode == 0:
                    result_queue.put(1.0)
                else:
                    logger.debug(f"MBPP-style execution failed with stderr: {process_result.stderr}")
                    result_queue.put(0.0)
            else:
                result_queue.put(0.0)
            return

        passed_count = 0
        for test_input_dict in test_cases:
            test_input = str(test_input_dict.get("input", "")).encode('utf-8')
            expected_output = str(test_input_dict.get("expected", "")).strip()
            process_result = subprocess.run(["python", str(filepath)], input=test_input, capture_output=True, text=True, timeout=timeout, check=False)
            if process_result.stdout.strip() == expected_output:
                passed_count += 1
        
        result_queue.put(float(passed_count / max(1, len(test_cases))))
    
    except subprocess.TimeoutExpired:
        logger.warning(f"Isolated code execution timed out after {timeout}s.")
        result_queue.put(0.0)
    except Exception as e:
        logger.error(f"Exception in isolated code execution process: {e}", exc_info=True)
        result_queue.put(0.0)
    finally:
        if filepath and filepath.exists():
            try: filepath.unlink()
            except OSError: pass
        # *** START OF FIX ***
        # Explicitly close and join the queue's background thread from within
        # the child process to prevent resource leakage warnings on shutdown.
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception as e:
            logger.debug(f"Error closing queue in child process: {e}")
        # *** END OF FIX ***

@RewardRegistry.register("code_execution")
class CodeExecutionReward(BaseReward):
    """Rewards generated code based on its execution success against provided test cases."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get("timeout", 5)
        self.num_workers = config.get("num_workers", 1)
        self.code_execution_config = {"timeout": self.timeout, "memory_limit": config.get("memory_limit", 512), "allow_imports": config.get("allow_imports", ["math", "re", "json"])}
        self.code_execution_config_json = json.dumps(self.code_execution_config)
        logger.info(f"Initialized CodeExecutionReward with timeout: {self.timeout}s.")

    def _validate_syntax(self, code: str) -> bool:
        try: ast.parse(code); return True
        except Exception: return False

    def compute(self, context: RewardContext) -> float:
        self.validate_inputs(context)
        code = _extract_python_code(context.generated_text)
        test_cases = context.test_cases
        if not code.strip() or not self._validate_syntax(code): return 0.0
        if not test_cases: return 0.3

        mp_context = get_context("spawn")
        result_queue = mp_context.Queue(1)
        process = None
        try:
            process = mp_context.Process(target=_execute_code_in_isolated_process_wrapper, args=(code, json.dumps(test_cases), self.code_execution_config_json, result_queue))
            process.start()
            pass_rate = result_queue.get(timeout=self.timeout + 3)
            pass_rate = float(pass_rate)
        except Exception as e:
            logger.error(f"Error managing code execution subprocess: {e}", exc_info=True)
            pass_rate = 0.0
        finally:
            if process and process.is_alive(): process.terminate(); process.join()
            try: result_queue.close(); result_queue.join_thread()
            except Exception: pass
        
        return float(max(0.0, min(1.0, 0.3 + (0.7 * pass_rate))))

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        rewards_for_batch: List[Dict[str, float]] = [{self.name: 0.0, 'total': 0.0}] * len(contexts)
        if self.num_workers <= 1 or len(contexts) <= 1:
            for i, context in enumerate(contexts):
                score = self.compute(context)
                rewards_for_batch[i] = {self.name: score, 'total': score}
            return rewards_for_batch

        mp_context = get_context("spawn")
        futures_with_indices = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_context) as executor:
            for idx, context in enumerate(contexts):
                code = _extract_python_code(context.generated_text)
                if not code.strip() or not self._validate_syntax(code): continue

                result_queue = mp_context.Queue(1)
                future = executor.submit(_execute_code_in_isolated_process_wrapper, code, json.dumps(context.test_cases), self.code_execution_config_json, result_queue)
                futures_with_indices.append((idx, future, result_queue))

            for original_idx, future, result_queue in futures_with_indices:
                try:
                    pass_rate = result_queue.get(timeout=self.timeout + 3)
                    final_score = 0.3 + (0.7 * float(pass_rate))
                    rewards_for_batch[original_idx] = {self.name: float(max(0.0, min(1.0, final_score))), 'total': final_score}
                except Exception as e:
                    logger.error(f"Error in batch execution for context {original_idx}: {e}", exc_info=True)
                finally:
                    try: result_queue.close(); result_queue.join_thread()
                    except Exception: pass
        return rewards_for_batch
