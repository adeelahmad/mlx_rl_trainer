import re
from typing import Dict, Any
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import register_reward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import extract_think_region

@register_reward("thinking_quality")
class ThinkingQualityReward(BaseReward):
    """
    Rewards the quality of the reasoning process in the <think> block.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_length_min = config.get("target_length_min", 50)
        self.target_length_max = config.get("target_length_max", 500)
        self.bad_phrases = config.get("bad_phrases", ["i think", "i believe", "maybe", "i'm not sure"])

    def compute(self, context: RewardContext) -> float:
        """
        Computes the thinking quality reward.
        """
        from mlx_rl_trainer.utils.text_utils import _get_static_reward_config
        reward_config = _get_static_reward_config()
        
        think_content = extract_think_region(context.generated_text, reward_config)
        if not think_content:
            return 0.0

        reward = 1.0

        # Length penalty/reward
        length = len(think_content)
        if length < self.target_length_min:
            reward *= length / self.target_length_min
        elif length > self.target_length_max:
            reward *= self.target_length_max / length

        # Structure reward
        if re.search(r"\n\s*[-*•]|\n\s*\d+\.", think_content):
            reward += 0.1
        
        # Penalty for bad phrases
        for phrase in self.bad_phrases:
            if phrase in think_content.lower():
                reward -= 0.2

        return max(0.0, min(1.0, reward))
