from .base_reward import BaseReward, RewardComposer
from .registry import RewardRegistry, register_reward
from .context import RewardContext

# Import all reward modules to ensure they are registered
import mlx_rl_trainer.rewards.format.tag_structure
import mlx_rl_trainer.rewards.content.semantic_similarity
import mlx_rl_trainer.rewards.content.verify_response
import mlx_rl_trainer.rewards.programming.code_execution
import mlx_rl_trainer.rewards.reasoning.thinking_quality
import mlx_rl_trainer.rewards.content.mcq_accuracy
import mlx_rl_trainer.rewards.content.steps_coverage

__all__ = [
    "BaseReward",
    "RewardComposer",
    "RewardRegistry",
    "register_reward",
    "RewardContext",
]
