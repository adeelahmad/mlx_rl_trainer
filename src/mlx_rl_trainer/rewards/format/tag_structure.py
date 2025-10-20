import re
from typing import Dict, Any
import logging
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


def e2xtract_think_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extract the text between the FIRST <think> and the FIRST </think> tags.

    Args:
        text: Full generated text
        gen_config: Generation configuration with tag definitions

    Returns:
        Text inside think tags, or empty string if not found
    """
    if not text:
        return ""

    # Use the 'or' keyword to provide a default value
    start_tag = "<think>"
    end_tag = "</think>"

    if not start_tag or not end_tag:
        return ""

    # Use re.search for robust, case-insensitive, non-greedy match to find the first block
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)

    m = re.search(pattern, text, flags=re.I | re.S)

    if m:
        # Group 1 is the content inside the tags
        return m.group(1).strip()

    return ""


def e2xtract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extract the answer text that comes AFTER the LAST </think> tag.
    This is for formats without explicit <answer> tags.

    Args:
        text: Full generated text
        gen_config: Generation configuration with tag definitions

    Returns:
        Text after the last </think> tag, or empty string if not found
    """
    if not text:
        return ""

    end_tag = '</think>'
    if not end_tag:
        # If no end tag defined, return the whole text as answer (fallback)
        return text.strip()

    # Use rfind for the last occurrence of the end tag (case-insensitive find)
    end_tag_lower = end_tag.lower()
    text_lower = text.lower()

    last_idx = text_lower.rfind(end_tag_lower)

    if last_idx != -1:
        # The length of the original tag must be used to slice the original text
        original_end_tag_len = len(end_tag)

        # Take everything AFTER the last end tag
        answer_text = text[last_idx + original_end_tag_len :].strip()

        # Strip leading newline specifically, as often the answer starts on a new line
        return answer_text.lstrip("\n").strip()

    # If no think end tag found, return full text (fallback, although reward logic should penalize this)
    return text.strip()



def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    """Extracts the text between the FIRST <think> and FIRST </think> tags."""
    if not text:
        return ""
    start_tag = getattr(gen_config, 'think_start_tag', '<think>')
    end_tag = getattr(gen_config, 'think_end_tag', '</think>')
    if not start_tag or not end_tag:
        return ""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    """Extracts text that comes AFTER the LAST </think> tag."""
    if not text:
        return ""
    end_tag = getattr(gen_config, 'think_end_tag', '</think>')
    if not end_tag:
        return text.strip()
    last_idx = text.lower().rfind(end_tag.lower())
    if last_idx != -1:
        answer_text = text[last_idx + len(end_tag):].strip()
        return answer_text.lstrip("\n").strip()
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
            "think_length_target_min", gen_config.think_length_target_min
        )
        self.think_target_max = config.get(
            "think_length_target_max", gen_config.think_length_target_max
        )

        # Penalty strength for length deviation
        self.length_penalty_strength = config.get(
            "length_penalty_strength", gen_config.think_length_penalty_strength
        )

        # Verbosity penalty multiplier (how much to penalize excessive length)
        self.verbosity_penalty_factor = config.get("verbosity_penalty_factor", 2.0)

        # Debug logging flag
        self.debug_logging = config.get("debug_logging", True)

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
            range_diff = self.think_target_min - self.min_think_length
            if range_diff <= 0:
                return 0.5  # Neutral if min/target are equal or inverted

            ratio = (think_length - self.min_think_length) / range_diff
            return 0.5 + (0.5 * max(0.0, min(1.0, ratio)))  # Range: 0.5 to 1.0

        else:
            # Too long - exponential penalty (verbosity is bad!)
            excess = think_length - self.think_target_max
            penalty_range = self.think_target_max if self.think_target_max > 0 else 100

            normalized_excess = excess / penalty_range
            penalty = (
                normalized_excess
                * self.length_penalty_strength
                * self.verbosity_penalty_factor
            )

            # Exponential decay for severe verbosity
            score = max(0.0, 1.0 - penalty)
            return score


    def compute(self, context: RewardContext) -> Dict[str, Any]:
    	"""Computes the format structure reward and returns a dictionary."""
    	generated = context.generated_text
    	log_data = {}
    	if not generated or len(generated.strip()) < 10:
    		return {"reward": 0.0, "log": {"error": "Empty or too short generation"}}

    	gen_config = GenerationConfig()
    	start_tag = gen_config.think_start_tag
    	end_tag = gen_config.think_end_tag
    	th_s = len(re.findall(re.escape(start_tag), generated, flags=re.I))
    	th_e = len(re.findall(re.escape(end_tag), generated, flags=re.I))

    	think_text = extract_think_region(generated, gen_config)
    	answer_text = extract_answer_region(generated, gen_config)
    	think_len = len(think_text.strip())
    	answer_len = len(answer_text.strip())

    	log_data = {"start_tags": th_s, "end_tags": th_e, "think_len": think_len, "answer_len": answer_len}
    	if self.debug_logging:
    		logger.info(f"TagStructure | {log_data}")

    	score = 0.0
    	if th_s == 1 and th_e == 1:
    		if think_len >= self.min_think_length and answer_len >= self.min_answer_length:
    			length_score = self._compute_length_score(think_len)
    			score = 1.0 * length_score
    			log_data["reason"] = "Perfect structure"
    			log_data["length_score"] = length_score
    		elif think_len >= self.min_think_length or answer_len >= self.min_answer_length:
    			score = 0.6
    			log_data["reason"] = "Partial content"
    		else:
    			score = 0.3
    			log_data["reason"] = "Empty content"
    	elif th_s >= 1 and th_e == 0:
    		score = 0.3
    		log_data["reason"] = "Incomplete think block"
    	elif th_s > 1 or th_e > 1:
    		score = 0.2
    		log_data["reason"] = "Multiple tags"
    	elif th_s == 0 and th_e == 0:
    		score = 0.1 if len(generated.strip()) > 30 else 0.0
    		log_data["reason"] = "No tags"
    	else:
    		score = 0.2
    		log_data["reason"] = "Fallback case"

    	log_data["final_score"] = score
    	return {"reward": score, "log": log_data}

    def compute1(self, context: RewardContext) -> float:
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
            if self.debug_logging:
                logger.warning(
                    f"TagStructureReward: Empty or too short generation (len={len(generated) if generated else 0})"
                )
            return 0.0

        # Use GenerationConfig to get standard tags
        gen_config = GenerationConfig()
        start_tag = gen_config.think_start_tag
        end_tag = gen_config.think_end_tag

        # Count tags (case-insensitive)
        th_s = len(re.findall(re.escape(start_tag), generated, flags=re.I))
        th_e = len(re.findall(re.escape(end_tag), generated, flags=re.I))

        # Extract thinking section (first valid block)
        think_text = extract_think_region(generated, gen_config)

        # Extract answer section (text after last </think>)
        answer_text = extract_answer_region(generated, gen_config)

        # Calculate lengths
        think_len = len(think_text.strip())
        answer_len = len(answer_text.strip())

        # ALWAYS log when debug is enabled (remove the % 50 condition!)
        if self.debug_logging:
            logger.info(
                f"TagStructure | start_tags={th_s}, end_tags={th_e}, "
                f"think_len={think_len}, answer_len={answer_len}, "
                f"min_think={self.min_think_length}, min_ans={self.min_answer_length}"
            )
            # Log first 100 chars of each section for inspection
            logger.debug(f"Think preview: {think_text[:100]}...")
            logger.debug(f"Answer preview: {answer_text[:100]}...")

        # === SCORING LOGIC ===

        # Perfect: Exactly one pair of tags with good content
        if th_s == 1 and th_e == 1:
            # Both sections have meaningful content
            if (
                think_len >= self.min_think_length
                and answer_len >= self.min_answer_length
            ):
                # Base score is 1.0, now apply length penalty
                length_score = self._compute_length_score(think_len)
                final_score = 1.0 * length_score

                if self.debug_logging:
                    logger.info(
                        f"TagStructure PASS: length_score={length_score:.3f}, final={final_score:.3f}"
                    )

                return final_score

            # Has structure but one section too short
            if (
                think_len >= self.min_think_length
                or answer_len >= self.min_answer_length
            ):
                if self.debug_logging:
                    logger.info(
                        f"TagStructure PARTIAL: One section too short, returning 0.6"
                    )
                return 0.6

            # Has tags but both sections too short
            if self.debug_logging:
                logger.info(
                    f"TagStructure FAIL: Both sections too short, returning 0.3"
                )
            return 0.3

        # Started thinking but never closed (incomplete)
        if th_s >= 1 and th_e == 0:
            if self.debug_logging:
                logger.warning(
                    f"TagStructure INCOMPLETE: Started thinking but no close tag"
                )
            return 0.3

        # Multiple think tags (confused model)
        if th_s > 1 or th_e > 1:
            if self.debug_logging:
                logger.warning(f"TagStructure CONFUSED: Multiple tags detected")
            return 0.2

        # No structure at all but has some text
        if th_s == 0 and th_e == 0:
            score = 0.1 if len(generated.strip()) > 30 else 0.0
            if self.debug_logging:
                logger.warning(f"TagStructure NO_TAGS: Returning {score}")
            return score

        # Fallback for weird edge cases
        if self.debug_logging:
            logger.warning(f"TagStructure FALLBACK: Unexpected case, returning 0.2")
        return 0.2
