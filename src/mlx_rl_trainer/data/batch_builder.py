import re
from typing import Dict, Any
import logging
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extract the text between <think> and </think> tags.

    Args:
        text: Full generated text
        gen_config: Generation configuration with tag definitions

    Returns:
        Text inside think tags, or empty string if not found
    """
    if not text:
        return ""

    start_tag = gen_config.think_start_tag
    end_tag = gen_config.think_end_tag

    if start_tag in text and end_tag in text:
        start_idx = text.find(start_tag) + len(start_tag)
        end_idx = text.find(end_tag)
        if start_idx < end_idx:
            return text[start_idx:end_idx].strip()

    return ""


def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extract the answer text that comes AFTER </think> tag.
    This is for formats without explicit <answer> tags.

    Args:
        text: Full generated text
        gen_config: Generation configuration with tag definitions

    Returns:
        Text after </think> tag, or empty string if not found
    """
    if not text:
        return ""

    end_tag = gen_config.think_end_tag

    if end_tag in text:
        # Split on closing think tag and take everything after
        parts = text.split(end_tag, 1)
        if len(parts) > 1:
            # Strip leading newline and whitespace
            return parts[1].lstrip('\n').strip()

    # If no think tag found, return full text (fallback)
    return text.strip()


@RewardRegistry.register("format_structure")
class TagStructureReward(BaseReward):
    """
    Rewards the model for adhering to <think>...</think> structure
    followed by direct answer text (no answer tags).

    Encourages concise, compressed thinking by penalizing verbosity.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_think_length = config.get("min_think_length", 20)
        self.min_answer_length = config.get("min_answer_length", 15)

        # Optimal think length range (from GenerationConfig)
        gen_config = GenerationConfig()
        self.think_target_min = config.get(
            "think_length_target_min",
            gen_config.think_length_target_min
        )
        self.think_target_max = config.get(
            "think_length_target_max",
            gen_config.think_length_target_max
        )

        # Penalty strength for length deviation
        self.length_penalty_strength = config.get(
            "length_penalty_strength",
            gen_config.think_length_penalty_strength
        )

        # Verbosity penalty multiplier (how much to penalize excessive length)
        self.verbosity_penalty_factor = config.get("verbosity_penalty_factor", 2.0)

    def _compute_length_score(self, think_length: int) -> float:
        """
        Compute a score based on think length relative to target range.

        Scoring philosophy:
        - Optimal range (target_min to target_max): 1.0
        - Too short (below target_min): Gradually decrease
        - Too long (above target_max): More aggressive penalty (verbosity is worse than brevity)

        Args:
            think_length: Character count of thinking section

        Returns:
            Score multiplier between 0.0 and 1.0
        """
        if self.think_target_min <= think_length <= self.think_target_max:
            # Perfect length - in the sweet spot
            return 1.0

        if think_length < self.think_target_min:
            # Too short - linear penalty
            if think_length < self.min_think_length:
                return 0.0  # Way too short

            # Scale from min_think_length to target_min
            ratio = (think_length - self.min_think_length) / (self.think_target_min - self.min_think_length)
            return 0.5 + (0.5 * ratio)  # Range: 0.5 to 1.0

        else:
            # Too long - exponential penalty (verbosity is bad!)
            excess = think_length - self.think_target_max
            penalty_range = self.think_target_max  # How far over we tolerate

            # Apply penalty strength with verbosity factor
            normalized_excess = excess / penalty_range
            penalty = normalized_excess * self.length_penalty_strength * self.verbosity_penalty_factor

            # Exponential decay for severe verbosity
            score = max(0.0, 1.0 - penalty)
            return score

    def compute(self, context: RewardContext) -> float:
        """
        Computes the format structure reward for the generated text.

        Scoring:
        - 1.0: Perfect structure + content + optimal length
        - 0.7-0.9: Perfect structure + content but sub-optimal length
        - 0.6: Has structure but content too short
        - 0.3: Started thinking but didn't close properly
        - 0.1: Has some text but no structure
        - 0.0: Empty or broken
        """
        generated = context.generated_text
        if not generated or len(generated.strip()) < 10:
            return 0.0

        gen_config = GenerationConfig()

        # Count tags (case-insensitive)
        th_s = len(re.findall(
            re.escape(gen_config.think_start_tag),
            generated,
            flags=re.I
        ))
        th_e = len(re.findall(
            re.escape(gen_config.think_end_tag),
            generated,
            flags=re.I
        ))

        # Extract thinking section using our function
        think_text = extract_think_region(generated, gen_config)

        # Extract answer section using our function
        answer_text = extract_answer_region(generated, gen_config)

        # === SCORING LOGIC ===

        # Perfect: Exactly one pair of tags with good content
        if th_s == 1 and th_e == 1:
            think_len = len(think_text.strip())
            answer_len = len(answer_text.strip())

            # Both sections have meaningful content
            if think_len >= self.min_think_length and answer_len >= self.min_answer_length:
                # Base score is 1.0, now apply length penalty
                length_score = self._compute_length_score(think_len)
                final_score = 1.0 * length_score

                # Log for debugging (remove after tuning)
                if context.update_step % 50 == 0:
                    logger.debug(
                        f"Format reward: think_len={think_len}, "
                        f"length_score={length_score:.3f}, "
                        f"final={final_score:.3f}"
                    )

                return final_score

            # Has structure but one section too short
            if think_len >= self.min_think_length or answer_len >= self.min_answer_length:
                return 0.6

            # Has tags but both sections too short
            return 0.3

        # Started thinking but never closed (incomplete)
        if th_s >= 1 and th_e == 0:
            return 0.3

        # Multiple think tags (confused model)
        if th_s > 1 or th_e > 1:
            return 0.2

        # No structure at all but has some text
        if th_s == 0 and th_e == 0:
            if len(generated.strip()) > 30:
                return 0.1  # At least it tried to generate something
            return 0.0

        # Fallback for weird edge cases
        return 0.2
