import re
from typing import Dict, Any
import logging

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import extract_think_region, extract_answer_region

logger = logging.getLogger(__name__)


@RewardRegistry.register("format_structure")
class TagStructureReward(BaseReward):
    """
    Rewards the model for adhering to a predefined XML-like tag structure
    for separating thinking (`<think>...</think>`) and answer sections.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def compute(self, context: RewardContext) -> float:
        """
        Computes the format structure reward for the generated text.
        """
        generated = context.generated_text
        gen_config = GenerationConfig()

        th_s = len(
            re.findall(
                re.escape(gen_config.think_start_tag), generated or "", flags=re.I
            )
        )
        th_e = len(
            re.findall(
                re.escape(gen_config.think_end_tag), generated or "", flags=re.I
            )
        )

        think = extract_think_region(generated, gen_config)
        ans = extract_answer_region(generated, gen_config)

        if th_s == 1 and th_e == 1:
            if len(think) > 10 and len(ans) > 10:
                return 1.0
            if len(think) > 10 or len(ans) > 10:
                return 0.5
            return 0.2

        if th_s >= 1 and th_e == 0:
            return 0.3

        if th_s == 0 and th_e == 0:
            if len(generated or "") > 20:
                return 0.1
            return 0.0

        return 0.2
