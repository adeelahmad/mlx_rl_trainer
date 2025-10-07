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

    All reward functions should inherit from this class and implement
    the `compute()` method.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the BaseReward.

        Args:
            config: Configuration dictionary for this reward function.
        """
        self.config = config
        self.name = self.__class__.__name__
        self.weight = config.get(
            "weight", 1.0
        )  # Weight is now part of the reward's own config too
        self.smoothing_window_size = config.get(
            "smoothing_window_size", 5
        )  # For internal smoothing
        self._reward_history: List[float] = []  # For internal reward smoothing

        logger.debug(f"Initialized {self.name} with config: {config}")

    @abstractmethod
    def compute(self, context: RewardContext) -> float:
        """
        Compute reward for a single generated response based on context.

        Args:
            context: RewardContext object containing generated, reference, and metadata.

        Returns:
            Raw reward score (typically between 0 and 1).
        """
        pass

    def _smooth_reward(self, current_reward: float) -> float:
        """Applies simple moving average smoothing to the reward."""
        self._reward_history.append(current_reward)
        if len(self._reward_history) > self.smoothing_window_size:
            self._reward_history.pop(0)
        return float(np.mean(self._reward_history))

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        """
        Compute rewards for a batch of responses and return detailed breakdown.

        Default implementation calls `compute()` for each item.
        Override for more efficient batch processing.

        Args:
            contexts: List of RewardContext objects.

        Returns:
            List of dictionaries, each containing raw reward scores for this function,
            e.g., [{'total': 0.8, 'sub_metric_1': 0.9}, ...].
            The 'total' key for this specific reward function is expected.
        """
        rewards_list = []
        for context in contexts:
            try:
                raw_score = self.compute(context)
                # Apply internal smoothing
                smoothed_score = self._smooth_reward(raw_score)
                rewards_list.append({self.name: smoothed_score, 'total': smoothed_score}) # Return internal name and total
            except Exception as e:
                logger.error(f"Batch computation failed in {self.name} for a context: {e}", exc_info=True)
                rewards_list.append({self.name: 0.0, 'total': 0.0}) # Return 0.0 on error
        return rewards_list

    def validate_inputs(self, context: RewardContext) -> None:
        """
        Validates reward function inputs, ensuring required fields are present and correctly typed.

        Args:
            context: RewardContext object.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(context, RewardContext):
            raise ValueError(f"Context must be RewardContext, got {type(context)}")
        if not isinstance(context.generated_text, str):
            raise ValueError(
                f"Generated text must be string, got {type(context.generated_text)}"
            )
        if not isinstance(context.reference_completion, str):
            raise ValueError(
                f"Reference completion must be string, got {type(context.reference_completion)}"
            )
        if not isinstance(context.metadata, dict):
            raise ValueError(f"Metadata must be dict, got {type(context.metadata)}")

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"


class RewardComposer:
    """
    Composes multiple `BaseReward` functions with specified weights.
    Calculates a single weighted sum of all individual rewards, returning detailed breakdown.
    """

    def __init__(self, rewards: List[Tuple[BaseReward, float]], context_cls: type = RewardContext):
        """
        Initializes the RewardComposer.

        Args:
            rewards: A list of (reward_function_instance, weight) tuples.
            context_cls: The context class to use if not RewardContext.
        """
        self.rewards = rewards
        self.total_weight_sum = sum(weight for _, weight in rewards)
        self.context_cls = context_cls

        if not (0.99 <= self.total_weight_sum <= 1.01):
            logger.warning(
                f"Reward weights do not sum to 1.0 (got {self.total_weight_sum:.2f}). Total reward may not be normalized."
            )
        logger.info(f"Initialized RewardComposer with {len(rewards)} rewards.")

    def compute(self, context: RewardContext) -> Dict[str, float]:
        """
        Computes a weighted combination of all configured rewards for a single context.

        Args:
            context: The `RewardContext` object for which to compute rewards.

        Returns:
            A dictionary containing individual reward scores (raw output of each BaseReward)
            and their weighted total.
        """
        individual_results = {}
        weighted_sum = 0.0

        for reward_fn, weight in self.rewards:
            try:
                score = reward_fn.compute(context)
                
                if hasattr(reward_fn, "_smooth_reward") and callable(getattr(reward_fn, "_smooth_reward")):
                    score = reward_fn._smooth_reward(score)

                individual_results[reward_fn.name] = score
                weighted_sum += score * weight

            except Exception as e:
                logger.warning(f"Reward '{reward_fn.name}' failed to compute for context ID {id(context)}: {e}")
                individual_results[reward_fn.name] = 0.0

        final_total = weighted_sum / (self.total_weight_sum if self.total_weight_sum > 0 else 1.0)
        individual_results['total'] = float(np.clip(final_total, 0.0, 1.0))

        return individual_results

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        """
        Computes rewards for a batch of `RewardContext` objects, leveraging batch_compute
        of individual reward functions where available, then composes them.
        """
        all_individual_batch_results: Dict[str, List[Dict[str, float]]] = {}
        
        for reward_fn, _ in self.rewards:
             all_individual_batch_results[reward_fn.name] = reward_fn.batch_compute(contexts)
        
        composed_batch_results: List[Dict[str, float]] = []
        for i in range(len(contexts)):
            individual_results_for_sample = {}
            weighted_sum_for_sample = 0.0
            
            for reward_fn, weight in self.rewards:
                try:
                    raw_score_for_sample = all_individual_batch_results[reward_fn.name][i].get('total', 0.0)
                    
                    individual_results_for_sample[reward_fn.name] = raw_score_for_sample
                    weighted_sum_for_sample += raw_score_for_sample * weight
                    
                except Exception as e:
                    logger.warning(f"Batch compose failed for reward '{reward_fn.name}' sample idx {i}: {e}")
                    individual_results_for_sample[reward_fn.name] = 0.0
            
            final_total_for_sample = weighted_sum_for_sample / (self.total_weight_sum if self.total_weight_sum > 0 else 1.0)
            individual_results_for_sample['total'] = float(np.clip(final_total_for_sample, 0.0, 1.0))
            
            composed_batch_results.append(individual_results_for_sample)
            
        return composed_batch_results
