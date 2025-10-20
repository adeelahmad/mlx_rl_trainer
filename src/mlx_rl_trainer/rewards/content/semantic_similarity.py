# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 009
# goals_of_writing_code_block: Provide a robust, stable, and simplified semantic_similarity reward.
# type_of_code_response: change existing
"""
Semantic similarity-based content reward, simplified for stability and correctness.
"""

import logging
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.rewards.registry import RewardRegistry

logger = logging.getLogger(__name__)


def _clean_text_for_tfidf(text: str) -> str:
    """A robust text cleaner for TF-IDF vectorization."""
    if not isinstance(text, str) or not text:
        return ""
    # Standardize and clean the text
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    # Remove all non-alphanumeric characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


@RewardRegistry.register("semantic_similarity")
class SemanticSimilarityReward(BaseReward):
    """
    Rewards semantic similarity between generated and reference answers.

    This version is simplified for stability. It uses a new, local TF-IDF
    vectorizer for each comparison to prevent stateful errors where the
    vectorizer's vocabulary becomes stale and produces incorrect zero vectors.

    Configuration:
        min_length (int): Min characters for text to be considered valid.
        apply_verbosity_penalty (bool): Penalize generated text that is
                                        much longer than the reference.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_length = int(config.get("min_length", 10))
        self.apply_verbosity_penalty = bool(config.get("apply_verbosity_penalty", True))

        gen_config = GenerationConfig()
        self.extract_after_tag = gen_config.think_end_tag

        logger.info(f"SemanticSimilarityReward initialized: min_length={self.min_length}")

    def _extract_answer(self, text: str) -> str:
        """Extracts text content after the </think> tag."""
        if self.extract_after_tag in text:
            try:
                # Split and take the second part after the tag
                return text.split(self.extract_after_tag, 1)[1].strip()
            except IndexError:
                return ""  # Tag exists, but there's nothing after it
        # If no tag is found, return an empty string to avoid comparing it to the reference.
        return ""

    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """Computes the semantic similarity score for a single context."""
        try:
            generated_answer = self._extract_answer(context.generated_text)
            reference_answer = self._extract_answer(context.reference_completion)

            # If either text is too short or missing, the reward is zero.
            if len(generated_answer) < self.min_length or len(reference_answer) < self.min_length:
                return {"reward": 0.0, "log": "text_too_short_or_missing"}

            # Clean the texts to get a good representation for TF-IDF
            cleaned_gen = _clean_text_for_tfidf(generated_answer)
            cleaned_ref = _clean_text_for_tfidf(reference_answer)

            if not cleaned_gen or not cleaned_ref:
                return {"reward": 0.0, "log": "empty_after_cleaning"}

            # **CRITICAL FIX**: Use a new, stateless vectorizer for every call.
            # This prevents vocabulary contamination from previous batches.
            vectorizer = TfidfVectorizer(stop_words="english", norm="l2")
            vectors = vectorizer.fit_transform([cleaned_gen, cleaned_ref])

            if vectors.getnnz() == 0: # Check if vectors are all zeros
                return {"reward": 0.0, "log": "all_stopwords_or_empty"}

            raw_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score = float(np.clip(raw_score, 0.0, 1.0))

            # Apply a simple verbosity penalty
            if self.apply_verbosity_penalty:
                len_gen, len_ref = len(generated_answer), len(reference_answer)
                if len_gen > len_ref * 1.5:  # Penalize if >50% longer
                    score *= (len_ref * 1.5) / len_gen

            return {"reward": score, "log": f"raw_score={raw_score:.4f}"}

        except Exception as e:
            logger.error(f"SemanticSimilarityReward failed: {e}", exc_info=False)
            return {"reward": 0.0, "log": "exception_in_compute"}

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Computes rewards for a batch of contexts by calling compute sequentially.
        This is more robust than a complex batch implementation.
        """
        return [self.compute(c) for c in contexts]
