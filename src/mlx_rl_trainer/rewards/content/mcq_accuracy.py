import re
from typing import Dict, Any
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import register_reward
from mlx_rl_trainer.rewards.context import RewardContext

@register_reward("mcq_accuracy")
class MCQAccuracyReward(BaseReward):
    """
    Calculates the accuracy of a multiple-choice question answer.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def compute(self, context: RewardContext) -> float:
        """
        Computes the MCQ accuracy reward.
        The reward is 1.0 if the generated answer matches the correct answer, and 0.0 otherwise.
        """
        metadata = context.metadata
        if not metadata or not metadata.get("is_mcq"):
            return 0.0

        correct_letters = metadata.get("correct_letters", "")
        if not correct_letters:
            return 0.0

        generated_answer = context.generated_text
        if not generated_answer:
            return 0.0

        # Extract the chosen letter from the generated answer.
        # This is a simple implementation that just looks for the first letter in the answer.
        match = re.search(r"([A-Z])", generated_answer, re.IGNORECASE)
        if not match:
            return 0.0
        
        chosen_letter = match.group(1).upper()

        return 1.0 if chosen_letter in correct_letters else 0.0
