import logging
import re
from typing import Dict, Any

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.utils.text_utils import extract_answer_region, _jaccard_similarity
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


@RewardRegistry.register("verify_response")
class VerifyResponseReward(BaseReward):
    """
    A flexible reward that verifies a generated answer. It can perform
    exact/normalized string matching and blend this verification score with
    a semantic similarity score.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.key_name = config.get("key_name", "verifiable_answer_str")
        self.verification_mode = config.get("verification_mode", "str_exact_match")
        self.similarity_weight = config.get("similarity_weight", 0.0)
        self.verification_weight = config.get("verification_weight", 1.0)
        self.gen_config = GenerationConfig()

        if self.similarity_weight + self.verification_weight > 1.0:
            logger.warning(
                "Similarity and verification weights sum to > 1.0. Normalizing."
            )
            total = self.similarity_weight + self.verification_weight
            self.similarity_weight /= total
            self.verification_weight /= total

    def _normalize_str(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text

    def compute(self, context: RewardContext) -> float:
        self.validate_inputs(context)

        # The whole original sample is passed in metadata now
        original_sample = context.metadata.get("meta", {}).get("_original_sample", {})
        verifiable_answer = original_sample.get(self.key_name)

        if not verifiable_answer or not isinstance(verifiable_answer, str):
            return 0.0

        generated_answer = extract_answer_region(
            context.generated_text, self.gen_config
        )
        if not generated_answer:
            return 0.0

        # --- Verification part ---
        verification_score = 0.0
        if self.verification_mode == "str_exact_match":
            if generated_answer.strip() == verifiable_answer.strip():
                verification_score = 1.0
        elif self.verification_mode == "str_normalized_match":
            if self._normalize_str(generated_answer) == self._normalize_str(
                verifiable_answer
            ):
                verification_score = 1.0
        else:
            logger.warning(
                f"Unknown verification mode: {self.verification_mode}. Defaulting to 0.0."
            )

        # --- Similarity part (if blended) ---
        similarity_score = 0.0
        if self.similarity_weight > 0:
            similarity_score = _jaccard_similarity(generated_answer, verifiable_answer)

        # --- Blend scores ---
        final_reward = (verification_score * self.verification_weight) + (
            similarity_score * self.similarity_weight
        )

        return max(0.0, min(1.0, final_reward))
