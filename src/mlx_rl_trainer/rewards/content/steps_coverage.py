from typing import Dict, Any, Set
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import register_reward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import _extract_action_phrases

@register_reward("steps_coverage")
class StepsCoverageReward(BaseReward):
    """
    Calculates a reward based on how many of the required steps are covered in the generated text.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def compute(self, context: RewardContext) -> float:
        """
        Computes the steps coverage reward.
        The reward is the Jaccard similarity between the set of required steps and the set of extracted steps.
        """
        required_steps = context.metadata.get("required_steps")
        if not required_steps or not isinstance(required_steps, list):
            return 0.0

        generated_text = context.generated_text
        if not generated_text:
            return 0.0

        extracted_steps = _extract_action_phrases(generated_text)
        if not extracted_steps:
            return 0.0

        required_steps_set = set(step.lower() for step in required_steps)
        extracted_steps_set = set(step.lower() for step in extracted_steps)

        intersection = len(required_steps_set.intersection(extracted_steps_set))
        union = len(required_steps_set.union(extracted_steps_set))

        if union == 0:
            return 1.0 if not required_steps else 0.0

        return intersection / union
