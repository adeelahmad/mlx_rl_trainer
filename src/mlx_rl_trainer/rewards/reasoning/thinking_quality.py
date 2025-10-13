import re
from typing import Dict, Any
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import register_reward
from mlx_rl_trainer.rewards.context import RewardContext
# Assuming this still imports the basic extractor from the previous file
from mlx_rl_trainer.utils.text_utils import extract_think_region
from mlx_rl_trainer.core.config import GenerationConfig


@register_reward("thinking_quality")
class ThinkingQualityReward(BaseReward):
    """
    Rewards the quality of the reasoning process in the <think> block.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_length_min = config.get("target_length_min", 50)
        self.target_length_max = config.get("target_length_max", 500)
        # Use a more comprehensive list of phrases that suggest low confidence or repetition
        self.bad_phrases = config.get(
            "bad_phrases",
            [
                "i think",
                "i believe",
                "maybe",
                "i'm not sure",
                "i will now",
                "i'll start by",
                "let's see"
            ]
        )
        self.tag_misuse_penalty = config.get("tag_misuse_penalty", 0.3)


    def _check_tag_misuse_penalty(self, text: str, gen_config: GenerationConfig) -> float:
        """
        Calculates a penalty for misuse of think tags (e.g., duplicates, unmatched).
        This indicates poor generation quality/adherence to the required format.
        """
        start_tag = gen_config.think_start_tag
        end_tag = gen_config.think_end_tag

        if not start_tag or not end_tag:
            return 0.0

        # Count tags case-insensitively
        th_s = len(re.findall(re.escape(start_tag), text, flags=re.I))
        th_e = len(re.findall(re.escape(end_tag), text, flags=re.I))

        penalty = 0.0

        # Penalize multiple or severely mismatched tags (e.g., duplicates or unclosed)
        if th_s > 1 or th_e > 1 or abs(th_s - th_e) > 1:
            penalty = self.tag_misuse_penalty

        # Check for nested tags *inside* the extracted think region (which extract_think_region might miss)
        # This is a good quality check for a messy thought process
        if th_s == 1 and th_e == 1:
            think_content = extract_think_region(text, gen_config)
            if re.search(r"<think>|<\/think>", think_content, flags=re.I):
                 penalty = self.tag_misuse_penalty

        return penalty

    def compute(self, context: RewardContext) -> float:
        """
        Computes the thinking quality reward.
        """
        gen_config = GenerationConfig()

        # The extract_think_region function is assumed to return the content of the FIRST <think>...</think> block.
        think_content = extract_think_region(context.generated_text, gen_config)

        if not think_content:
            return 0.0

        reward = 1.0

        # 1. Tag Misuse Penalty (Poor structural quality)
        tag_misuse_penalty = self._check_tag_misuse_penalty(context.generated_text, gen_config)
        reward -= tag_misuse_penalty

        # 2. Length Penalty/Reward (Optimal Verbosity)
        length = len(think_content.strip())
        if length < self.target_length_min:
            # Linear decay to 0 if too short
            reward *= max(0.0, length / self.target_length_min)
        elif length > self.target_length_max:
            # Inverse decay if too long (penalizing verbosity)
            reward *= self.target_length_max / length

        # 3. Structure Bonus (Good Readability/Process)
        # Check for bullet points or numbered lists, implying a structured breakdown
        if re.search(r"(\n\s*[-*•]|\n\s*\d+\.\s+)", think_content):
            reward += 0.1

        # 4. Penalty for bad phrases (Low Confidence/Filler)
        for phrase in self.bad_phrases:
            if phrase in think_content.lower():
                reward -= 0.2

        # Ensure the reward stays within [0.0, 1.0]
        return max(0.0, min(1.0, reward))
