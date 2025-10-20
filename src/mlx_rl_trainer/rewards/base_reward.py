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

    @abstractmethod
    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Compute reward for a single response.
        ⭐ MUST return a dictionary with at least a 'reward' key.
        e.g., {"reward": 0.8, "log": {"details": ...}}
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
        This is the generic fallback for all reward functions.
        """
        rewards_list = []
        for context in contexts:
            try:
                # ⭐ FIX: `compute()` now returns a dictionary
                result_dict = self.compute(context)

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
                    f"Batch computation failed in {self.name} for a context: {e}",
                    exc_info=True,
                )
                rewards_list.append({self.name: 0.0, "total": 0.0, "log": {"error": str(e)}})
        return rewards_list

    def validate_inputs(self, context: RewardContext) -> None:
        if not isinstance(context, RewardContext):
            raise ValueError(f"Context must be RewardContext, got {type(context)}")
        # ... other validations

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
            all_individual_batch_results[reward_fn.name] = reward_fn.batch_compute(
                contexts
            )

        composed_batch_results: List[Dict[str, float]] = []
        for i in range(len(contexts)):
            individual_results_for_sample = {}
            weighted_sum_for_sample = 0.0

            for reward_fn, weight in self.rewards:
                try:
                    # This is where the error was happening.
                    # It now correctly gets the 'total' score from the dict returned by batch_compute.
                    raw_score_for_sample = all_individual_batch_results[reward_fn.name][i].get("total", 0.0)
                    individual_results_for_sample[reward_fn.name] = raw_score_for_sample
                    weighted_sum_for_sample += raw_score_for_sample * weight
                except Exception as e:
                    logger.warning(
                        f"Batch compose failed for reward '{reward_fn.name}' sample idx {i}: {e}"
                    )
                    individual_results_for_sample[reward_fn.name] = 0.0

            final_total_for_sample = weighted_sum_for_sample / (
                self.total_weight_sum if self.total_weight_sum > 0 else 1.0
            )
            individual_results_for_sample["total"] = float(
                np.clip(final_total_for_sample, 0.0, 1.0)
            )
            composed_batch_results.append(individual_results_for_sample)

        return composed_batch_results
