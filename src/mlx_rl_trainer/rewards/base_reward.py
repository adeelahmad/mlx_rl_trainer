"""Abstract base class for reward functions."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging
import numpy as np

from .context import RewardContext

logger = logging.getLogger(__name__)


class BaseReward(ABC):
    """
    Abstract base class for all reward functions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.weight = config.get("weight", 1.0)
        self.smoothing_window_size = config.get("smoothing_window_size", 5)
        self._reward_history: List[float] = []
        logger.debug(f"Initialized {self.name} with config: {config}")

    def _validate_result_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and normalizes the result dictionary from compute methods.

        Ensures the result dictionary has the required format and types.

        Args:
            result: The result dictionary to validate

        Returns:
            A normalized result dictionary with at least a 'reward' key

        Raises:
            ValueError: If the result is missing required keys or has invalid values
        """
        if not isinstance(result, dict):
            logger.warning(f"{self.name}: compute returned non-dict result, creating fallback")
            return {"reward": 0.0, "log": {"error": "Invalid return type"}}

        if "reward" not in result:
            logger.warning(f"{self.name}: compute result missing 'reward' key, using 0.0")
            result["reward"] = 0.0

        reward_value = result["reward"]
        if not isinstance(reward_value, (int, float)):
            logger.warning(f"{self.name}: reward value is not a number, converting to float")
            try:
                result["reward"] = float(reward_value) if reward_value is not None else 0.0
            except (ValueError, TypeError):
                logger.error(f"{self.name}: could not convert reward value to float")
                result["reward"] = 0.0

        # Ensure reward is in valid range [0.0, 1.0]
        result["reward"] = max(0.0, min(1.0, result["reward"]))

        # Ensure log is a dictionary if present
        if "log" in result and not isinstance(result["log"], dict):
            logger.warning(f"{self.name}: log is not a dictionary, converting")
            result["log"] = {"value": str(result["log"])}

        return result

    @abstractmethod
    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Compute reward for a single response.

        This method must be implemented by all reward subclasses.

        ⭐ MUST return a dictionary with at least a 'reward' key.
        e.g., {"reward": 0.8, "log": {"details": ...}}

        Args:
            context: The RewardContext object containing all necessary data

        Returns:
            A dictionary with at least a 'reward' key containing a float value
            between 0.0 and 1.0, and optionally a 'log' key with additional info

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
        """
        raise NotImplementedError

    def _smooth_reward(self, current_reward: float) -> float:
        """Applies simple moving average smoothing to the reward."""
        self._reward_history.append(current_reward)
        if len(self._reward_history) > self.smoothing_window_size:
            self._reward_history.pop(0)
        return float(np.mean(self._reward_history))

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Default implementation calls `compute()` for each item.

        This is the generic fallback for all reward functions. It handles validation,
        error handling, and result normalization for each context in the batch.

        Args:
            contexts: A list of RewardContext objects to process

        Returns:
            A list of dictionaries containing reward scores and logs
        """
        if not isinstance(contexts, list):
            logger.error(f"{self.name}: batch_compute received non-list contexts")
            return [{"total": 0.0, self.name: 0.0, "log": {"error": "Invalid contexts type"}}]

        rewards_list = []
        for i, context in enumerate(contexts):
            try:
                # Validate the context
                try:
                    self.validate_inputs(context)
                except (ValueError, TypeError) as e:
                    logger.warning(f"{self.name}: Context validation failed: {e}")
                    rewards_list.append({
                        self.name: 0.0,
                        "total": 0.0,
                        "log": {"error": f"Invalid context: {str(e)}"}
                    })
                    continue

                # Compute the reward
                result_dict = self.compute(context)
                logger.debug(f"{self.name}: Raw result_dict {result_dict}")

                # Validate and normalize the result
                result_dict = self._validate_result_dict(result_dict)

                # Extract the float score from the dictionary
                raw_score = result_dict.get('reward', 0.0)

                # Smooth the float score
                smoothed_score = self._smooth_reward(raw_score)

                # Structure the final output for the composer
                output = {
                    self.name: smoothed_score,
                    "total": smoothed_score,
                    "log": result_dict.get('log', {})
                }
                rewards_list.append(output)

            except Exception as e:
                logger.error(
                    f"Batch computation failed in {self.name} for context {i}: {e}",
                    exc_info=True,
                )
                rewards_list.append({
                    self.name: 0.0,
                    "total": 0.0,
                    "log": {"error": f"Exception: {str(e)}"}
                })

        return rewards_list

    def validate_inputs(self, context: RewardContext) -> None:
        """
        Validates the input context for reward computation.

        Performs comprehensive validation of the RewardContext object to ensure
        it contains all required fields with appropriate values before computation.

        Args:
            context: The RewardContext object to validate

        Raises:
            ValueError: If the context is invalid or missing required fields
            TypeError: If the context is not a RewardContext instance
        """
        # Type validation
        if not isinstance(context, RewardContext):
            raise TypeError(f"Context must be RewardContext, got {type(context)}")

        # Required field validation
        if context.generated_text is None:
            raise ValueError("Context missing required field: generated_text")

        if context.prompt_text is None:
            raise ValueError("Context missing required field: prompt_text")

        if context.reference_completion is None:
            raise ValueError("Context missing required field: reference_completion")

        # Type validation for fields
        if not isinstance(context.generated_text, str):
            raise TypeError(f"generated_text must be a string, got {type(context.generated_text)}")

        if not isinstance(context.prompt_text, str):
            raise TypeError(f"prompt_text must be a string, got {type(context.prompt_text)}")

        if not isinstance(context.reference_completion, str):
            raise TypeError(f"reference_completion must be a string, got {type(context.reference_completion)}")

        if not isinstance(context.test_cases, list):
            raise TypeError(f"test_cases must be a list, got {type(context.test_cases)}")

        if not isinstance(context.metadata, dict):
            raise TypeError(f"metadata must be a dictionary, got {type(context.metadata)}")

        # Validate test cases if present
        for i, test_case in enumerate(context.test_cases):
            if not isinstance(test_case, dict):
                raise TypeError(f"Test case at index {i} must be a dictionary, got {type(test_case)}")

        # Log validation success at debug level
        logger.debug(f"Context validation passed for {self.name}")

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class RewardComposer:
    """
    Composes multiple `BaseReward` functions with specified weights.
    """

    def __init__(
        self, rewards: List[Tuple[BaseReward, float]], context_cls: type = RewardContext
    ):
        self.rewards = rewards
        self.total_weight_sum = sum(weight for _, weight in rewards)
        self.context_cls = context_cls
        if not (0.99 <= self.total_weight_sum <= 1.01):
            logger.warning(
                f"Reward weights do not sum to 1.0 (got {self.total_weight_sum:.2f})."
            )
        logger.info(f"Initialized RewardComposer with {len(rewards)} rewards.")

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        """
        Computes rewards for a batch, leveraging individual batch_compute methods.
        """
        all_individual_batch_results: Dict[str, List[Dict[str, Any]]] = {}

        for reward_fn, _ in self.rewards:
            # Validate that batch_compute returns proper format
            batch_result = reward_fn.batch_compute(contexts)
            if not isinstance(batch_result, list):
                logger.error(f"Reward {reward_fn.name} batch_compute() returned {type(batch_result)}, expected List[Dict[str, Any]]")
                # Create fallback result
                batch_result = [{"total": 0.0, reward_fn.name: 0.0, "log": {"error": "Invalid return type"}} for _ in contexts]

            for i, item in enumerate(batch_result):
                if not isinstance(item, dict):
                    logger.error(f"Reward {reward_fn.name} batch_compute()[{i}] returned {type(item)}, expected Dict[str, Any]")
                    batch_result[i] = {"total": 0.0, reward_fn.name: 0.0, "log": {"error": "Invalid item type"}}
                elif "total" not in item:
                    logger.warning(f"Reward {reward_fn.name} batch_compute()[{i}] missing 'total' key, using 0.0")
                    item["total"] = 0.0

            all_individual_batch_results[reward_fn.name] = batch_result

        composed_batch_results: List[Dict[str, float]] = []
        for i in range(len(contexts)):
            individual_results_for_sample = {}
            weighted_sum_for_sample = 0.0

            for reward_fn, weight in self.rewards:
                try:
                    result_item = all_individual_batch_results[reward_fn.name][i]
                    if not isinstance(result_item, dict):
                        logger.error(f"Reward {reward_fn.name} sample {i}: Expected dict, got {type(result_item)}")
                        raw_score_for_sample = 0.0
                    else:
                        raw_score_for_sample = result_item.get("total", 0.0)
                        if not isinstance(raw_score_for_sample, (int, float)):
                            logger.warning(f"Reward {reward_fn.name} sample {i}: 'total' is {type(raw_score_for_sample)}, converting to float")
                            raw_score_for_sample = float(raw_score_for_sample) if raw_score_for_sample is not None else 0.0

                    individual_results_for_sample[reward_fn.name] = raw_score_for_sample
                    weighted_sum_for_sample += raw_score_for_sample * weight
                except Exception as e:
                    logger.error(f"Batch compose failed for reward '{reward_fn.name}' sample idx {i}: {e}", exc_info=True)
                    individual_results_for_sample[reward_fn.name] = 0.0

            final_total_for_sample = weighted_sum_for_sample / (
                self.total_weight_sum if self.total_weight_sum > 0 else 1.0
            )
            individual_results_for_sample["total"] = float(
                np.clip(final_total_for_sample, 0.0, 1.0)
            )
            composed_batch_results.append(individual_results_for_sample)

        return composed_batch_results
