# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 003
# goals_of_writing_code_block: Semantic similarity reward updated to extract answer text after </think> tag (no <answer> tags)
# type_of_code_response: change existing
"""Semantic similarity-based content reward."""

from typing import Dict, Any, List
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import (
    _tokenize_set,
    _tfidf_cosine,
)  # Import utilities

logger = logging.getLogger(__name__)


@RewardRegistry.register("semantic_similarity")
class SemanticSimilarityReward(BaseReward):
    """
    Rewards semantic similarity between generated and reference text.

    Uses either TF-IDF Cosine Similarity or Jaccard similarity as fallback.
    Extracts answer text that appears AFTER </think> tag for comparison.

    Configuration:
        method: Similarity method - 'tfidf' or 'jaccard' (default: 'tfidf').
        min_length: Minimum text length (in characters) to compute similarity (default: 10).
        max_features: Max features for TF-IDF Vectorizer (default: 1000).
        extract_after_tag: Tag to split on for answer extraction (default: uses GenerationConfig).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.method = config.get("method", "tfidf")
        self.min_length = config.get("min_length", 10)
        self.max_features = config.get("max_features", 1000)

        # Get generation config for tag extraction
        self.gen_config = GenerationConfig()
        self.extract_after_tag = config.get("extract_after_tag", self.gen_config.think_end_tag)

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english", lowercase=True
            )
        elif self.method == "jaccard":
            # Jaccard doesn't need a vectorizer
            pass
        else:
            logger.warning(
                f"Unknown similarity method '{self.method}', falling back to TF-IDF."
            )
            self.method = "tfidf"
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english", lowercase=True
            )

    def _extract_answer_text(self, text: str) -> str:
        """
        Extract the answer portion of text (everything after </think> tag).
        If no tag found, returns the full text.

        Args:
            text: Full text that may contain <think>...</think> followed by answer

        Returns:
            The answer text (after </think>) or full text if no tag found
        """
        if not text:
            return ""

        if self.extract_after_tag and self.extract_after_tag in text:
            # Split on closing think tag and take everything after
            parts = text.split(self.extract_after_tag, 1)
            if len(parts) > 1:
                return parts[1].strip()
            return ""

        # If no tag found, return full text (fallback for reference texts without tags)
        return text.strip()

    def compute(self, context: RewardContext) -> float:
        """
        Computes semantic similarity reward for a single `RewardContext`.

        Args:
            context: The `RewardContext` containing `generated_text` and `reference_completion`.

        Returns:
            A float score between 0.0 and 1.0, representing similarity.
        """
        generated = context.generated_text
        reference = context.reference_completion

        try:
            self.validate_inputs(context)

            # Extract answer portions (text after </think> tag)
            generated_answer = self._extract_answer_text(generated)
            reference_answer = self._extract_answer_text(reference)

            # Check minimum length
            if (
                len(generated_answer) < self.min_length
                or len(reference_answer) < self.min_length
            ):
                logger.debug(
                    f"Content: Generated answer ({len(generated_answer)} chars) or "
                    f"reference answer ({len(reference_answer)} chars) too short. Returning 0.0."
                )
                return 0.0

            if self.method == "tfidf":
                return self._compute_tfidf_similarity(generated_answer, reference_answer)
            elif self.method == "jaccard":
                return self._compute_jaccard_similarity(generated_answer, reference_answer)
            else:
                return 0.0

        except Exception as e:
            logger.error(
                f"SemanticSimilarityReward computation failed: {e}", exc_info=True
            )
            return 0.0

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Helper to compute TF-IDF based cosine similarity between two texts."""
        try:
            # Create a local vectorizer to fit only these two texts
            local_vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english", lowercase=True
            )
            vectors = local_vectorizer.fit_transform([text1, text2])

            # Check if vectors are non-empty
            if vectors.shape[0] < 2:
                logger.debug("TF-IDF: Not enough valid vectors after transformation.")
                return 0.0

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logger.warning(
                f"TF-IDF computation failed for pair: {e}. Falling back to Jaccard.",
                exc_info=True,
            )
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Helper to compute Jaccard similarity between two texts."""
        try:
            A, B = _tokenize_set(text1), _tokenize_set(text2)

            # Edge cases
            if not A and not B:
                return 1.0  # Both empty
            if not A or not B:
                return 0.0  # One empty

            intersection = len(A & B)
            union = len(A | B)

            if union == 0:
                return 0.0

            return float(intersection / union)
        except Exception as e:
            logger.error(f"Jaccard similarity computation failed: {e}", exc_info=True)
            return 0.0

    def batch_compute(self, contexts: List[RewardContext]) -> List[float]:
        """
        Optimized batch computation for TF-IDF based similarity.

        Args:
            contexts: A list of `RewardContext` objects.

        Returns:
            A list of float similarity scores for the batch.
        """
        if self.method == "jaccard":
            # Jaccard doesn't benefit from batching, compute sequentially
            return super().batch_compute(contexts)

        try:
            # Extract answer portions for all texts
            generated_answers = [self._extract_answer_text(c.generated_text) for c in contexts]
            reference_answers = [self._extract_answer_text(c.reference_completion) for c in contexts]

            # Filter out pairs where either text is too short
            valid_indices = []
            valid_generated = []
            valid_reference = []

            for i, (gen, ref) in enumerate(zip(generated_answers, reference_answers)):
                if len(gen) >= self.min_length and len(ref) >= self.min_length:
                    valid_indices.append(i)
                    valid_generated.append(gen)
                    valid_reference.append(ref)

            # Initialize all scores to 0.0
            scores = [0.0] * len(contexts)

            if not valid_generated:
                logger.debug("Batch compute: No valid text pairs found.")
                return scores

            # Combine all valid texts for a single, comprehensive TF-IDF vocabulary
            all_texts = valid_generated + valid_reference

            # Fit the vectorizer on all texts, then transform
            self.vectorizer.fit(all_texts)
            generated_vectors = self.vectorizer.transform(valid_generated)
            reference_vectors = self.vectorizer.transform(valid_reference)

            # Compute pairwise cosine similarity
            if generated_vectors.shape[0] == 0 or reference_vectors.shape[0] == 0:
                return scores

            similarities = cosine_similarity(
                generated_vectors, reference_vectors
            ).diagonal()

            # Map similarities back to original indices
            for idx, sim in zip(valid_indices, similarities):
                scores[idx] = float(np.clip(sim, 0.0, 1.0))

            return scores

        except Exception as e:
            logger.error(
                f"Batch semantic similarity (TF-IDF) computation failed: {e}",
                exc_info=True,
            )
            # Fallback to sequential compute if batch fails
            return [self.compute(c) for c in contexts]
