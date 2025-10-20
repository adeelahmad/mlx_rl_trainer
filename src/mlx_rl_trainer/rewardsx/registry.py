# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/registry.py
# revision_no: 001
# goals_of_writing_code_block: Plugin registry system for reward functions.
# type_of_code_response: add new code
"""Reward function registry for plugin architecture."""

from typing import Dict, Type, Any
import logging

from mlx_rl_trainer.rewards.base_reward import BaseReward

logger = logging.getLogger(__name__)


class RewardRegistry:
    """
    Central registry for reward function plugins.

    Allows dynamic registration and instantiation of `BaseReward` subclasses
    by their unique string names. This promotes a modular and extensible
    reward system.
    """

    _rewards: Dict[str, Type[BaseReward]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward function class.

        Args:
            name: A unique string identifier for this reward function.
        """

        def decorator(reward_class: Type[BaseReward]):
            if name in cls._rewards:
                logger.warning(
                    f"Overwriting existing reward: '{name}' with {reward_class.__name__}."
                )
            cls._rewards[name] = reward_class
            logger.info(f"Registered reward: '{name}' -> {reward_class.__name__}.")
            return reward_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseReward]:
        """
        Retrieves a registered reward class by its name.

        Args:
            name: The string identifier of the reward to retrieve.

        Returns:
            The `BaseReward` subclass associated with the given name.

        Raises:
            KeyError: If no reward is registered with the given name.
        """
        if name not in cls._rewards:
            available = ", ".join(cls._rewards.keys())
            raise KeyError(
                f"Reward '{name}' not found. Available rewards: [{available}]."
            )
        return cls._rewards[name]

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseReward:
        """
        Instantiates a registered reward function using its configuration.

        Args:
            name: The string identifier of the reward to create.
            config: A dictionary of configuration parameters to pass to the reward's constructor.

        Returns:
            An initialized instance of the `BaseReward` subclass.

        Raises:
            Exception: If instantiation fails (e.g., due to invalid config).
        """
        try:
            reward_class = cls.get(name)
            instance = reward_class(config)
            logger.debug(f"Created reward instance: '{name}'.")
            return instance
        except Exception as e:
            logger.error(f"Failed to create reward '{name}': {e}", exc_info=True)
            raise

    @classmethod
    def list_available(cls) -> list[str]:
        """
        Returns a list of all currently registered reward names.
        """
        return list(cls._rewards.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Clears all registered reward functions from the registry.
        Primarily used for testing or dynamic reloading scenarios.
        """
        cls._rewards.clear()
        logger.info("Cleared reward registry.")


register_reward = RewardRegistry.register
