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
    Penalizes overly long thinking and rewards concise, focused reasoning.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_length_min = config.get("target_length_min", 50)
        self.target_length_max = config.get("target_length_max", 500)

        # New: Optimal concise range (gets bonus reward)
        self.optimal_length_min = config.get("optimal_length_min", 100)
        self.optimal_length_max = config.get("optimal_length_max", 300)
        self.conciseness_bonus = config.get("conciseness_bonus", 0.15)

        # New: Excessive length penalty (for very verbose thinking)
        self.excessive_length_threshold = config.get("excessive_length_threshold", 800)
        self.excessive_length_penalty = config.get("excessive_length_penalty", 0.3)

        # New: Special tokens penalty (tokens that shouldn't appear in thinking)
        self.special_tokens = config.get(
            "special_tokens",
            [
                "<|endoftext|>",
                "<|im_start|>",
                "<|im_end|>",
                "<|end|>",
                "<|begin|>",
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "[INST]",
                "[/INST]",
                "<s>",
                "</s>",
                "<pad>",
                "<unk>",
                "<bos>",
                "<eos>"
            ]
        )
        self.special_token_penalty = config.get("special_token_penalty", 0.4)

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
                "let's see",
                "<think>\\n<|im_start|>",
                "Confused",
                "stuck",
                "frustrated",
                "<|im_start|>",
                "<|endoftext|>",
                "<think>\n<|im_start|><think>\n<|im_start|><think>\n<think>",
                # "<think>\n<think>",
                "I think the answer",
                "I believe that",
                "In my view",
                "From what I can tell",
                "It seems to me",
                "It appears that",
                # "My understanding is",
                "As far as I know",
                # "Let me start by",
                # "Let me first",
                "I should probably",
                "I need to figure out",
                "I'm trying to",
                "I'm going to try",
                "I'll attempt to",
                "Confused",
                "stక్",
                "frustrated",
                "frustrating",
                "Alternatively",
                "Actually",
                "Probably not sure",
                "Uncertain about",
                "Unclear whether",
                "I'm guessing that",
                "maybe this is",
                "Could be that",
                "Might be because",
                "I'm not 100% sure",
                "I'm not sure if",
                "I'm not certain",
                "Hard to say",
                "Difficult to tell",
                "Circular reasoning detected",
                "In some way or another",
                "Magically works",
                "For some unknown reason",
                "Too complicated",
                "It just somehow",
                "Something seems off",
                "False assumption",
                "Insufficient information to",
                "Wait, what if",
                "Wait, "
                "Wait, another idea:",
                "Wait, unless...",
                "Wait, perhaps","Wait, let's see","Wait, here's","Wait, no. Wait,","Wait, wait! Wait,",
                "Wait, actually no",
                "Wait, on second thought",
                "Hold on, maybe",
                "Hmm, perhaps",
                "Or wait, could",
                "Looking at this more closely",
                "Upon further reflection",
                "Taking a step back",
                "Thinking about it more",
                "Now that I consider",
                "When I really think",
                "If I had to guess",
                "To be completely honest",
                "In all honesty",
                "You know what",
                "The thing is",
                "What I mean is",
                "In other words",
                "Put simply",
                "Basically what happens",
                "Long story short",
                "At the end of the day"
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

    def _check_special_tokens_penalty(self, think_content: str) -> float:
        """
        Calculates a penalty for special tokens appearing in the thinking content.
        Special tokens indicate poor generation quality or model confusion.
        """
        penalty = 0.0

        for token in self.special_tokens:
            # Case-sensitive check since special tokens are usually exact
            if token in think_content:
                penalty += self.special_token_penalty

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

        # 1a. NEW: Special Tokens Penalty (Control tokens shouldn't appear in thinking)
        special_token_penalty = self._check_special_tokens_penalty(think_content)
        reward -= special_token_penalty

        # 2. Length Penalty/Reward (Optimal Verbosity)
        length = len(think_content.strip())

        if length < self.target_length_min:
            # Linear decay to 0 if too short
            reward *= max(0.0, length / self.target_length_min)
        elif length > self.target_length_max:
            # Inverse decay if too long (penalizing verbosity)
            reward *= self.target_length_max / length

        # 2a. NEW: Conciseness Bonus (Reward for optimal length)
        if self.optimal_length_min <= length <= self.optimal_length_max:
            reward += self.conciseness_bonus

        # 2b. NEW: Excessive Length Penalty (Strong penalty for very long thinking)
        if length > self.excessive_length_threshold:
            # Progressive penalty that increases with length
            excess_ratio = length / self.excessive_length_threshold
            reward -= self.excessive_length_penalty * excess_ratio

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
