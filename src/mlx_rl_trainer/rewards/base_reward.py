# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/base_reward.py
# revision_no: 002
# goals_of_writing_code_block: Abstract base class for reward functions, now compatible with RewardContext and internal smoothing.
# type_of_code_response: change existing
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

    def batch_compute(self, contexts: List[RewardContext]) -> List[float]:
        """
        Compute rewards for a batch of responses.

        Default implementation calls `compute()` for each item.
        Override for more efficient batch processing (e.g., vectorized operations).

        Args:
            contexts: List of RewardContext objects.

        Returns:
            List of reward scores.
        """
        try:
            rewards = []
            for context in contexts:
                reward = self.compute(context)
                rewards.append(reward)
            return rewards
        except Exception as e:
            logger.error(f"Batch computation failed in {self.name}: {e}")
            # In case of batch failure, return zeros for all to avoid crashing
            return [0.0] * len(contexts)
        finally:
            pass

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
    Calculates a single weighted sum of all individual rewards.
    """

    def __init__(self, rewards: List[Tuple[BaseReward, float]]):
        """
        Initializes the RewardComposer.

        Args:
            rewards: A list of (reward_function_instance, weight) tuples.
        """
        self.rewards = rewards

        total_weight = sum(weight for _, weight in rewards)
        if not (0.99 <= total_weight <= 1.01):  # Allow slight floating point deviation
            logger.warning(
                f"Reward weights do not sum to 1.0 (got {total_weight:.2f}). This may lead to unexpected total reward scaling."
            )

        logger.info(f"Initialized RewardComposer with {len(rewards)} rewards.")

    def compute(self, context: RewardContext) -> Dict[str, float]:
        """
        Computes a weighted combination of all configured rewards for a single context.

        Args:
            context: The `RewardContext` object for which to compute rewards.

        Returns:
            A dictionary containing individual reward scores and their weighted total.
        """
        results = {}
        total_reward = 0.0

        for reward_fn, weight in self.rewards:
            try:
                score = reward_fn.compute(context)

                # Apply internal smoothing if reward function has a smoothing method
                if hasattr(reward_fn, "_smooth_reward") and callable(
                    getattr(reward_fn, "_smooth_reward")
                ):
                    score = reward_fn._smooth_reward(score)

                weighted_score = score * weight
                results[reward_fn.name] = score  # Store raw (possibly smoothed) score
                total_reward += weighted_score
            except Exception as e:
                logger.warning(
                    f"Reward '{reward_fn.name}' failed to compute for context: {e}"
                )
                results[reward_fn.name] = 0.0  # Assign 0.0 on failure

        results["total"] = total_reward
        return results
