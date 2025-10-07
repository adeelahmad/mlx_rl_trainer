import re
from typing import Dict, Any
import logging
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import extract_think_region

logger = logging.getLogger(__name__)

@RewardRegistry.register("format_structure")
class TagStructureReward(BaseReward):
    """
    Rewards the model for adhering to <think>...</think> structure
    followed by direct answer text (no answer tags).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_think_length = config.get("min_think_length", 20)
        self.min_answer_length = config.get("min_answer_length", 15)

    def compute(self, context: RewardContext) -> float:
        """
        Computes the format structure reward for the generated text.

        Scoring:
        - 1.0: Perfect structure with meaningful content
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

        # Extract thinking section
        think_text = extract_think_region(generated, gen_config)

        # Extract answer as text AFTER </think> tag
        answer_text = ""
        if gen_config.think_end_tag in generated:
            # Split on closing think tag and take everything after
            parts = generated.split(gen_config.think_end_tag, 1)
            if len(parts) > 1:
                answer_text = parts[1].strip()

        # === SCORING LOGIC ===

        # Perfect: Exactly one pair of tags with good content
        if th_s == 1 and th_e == 1:
            think_len = len(think_text.strip())
            answer_len = len(answer_text.strip())

            # Both sections have meaningful content
            if think_len >= self.min_think_length and answer_len >= self.min_answer_length:
                return 1.0

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
