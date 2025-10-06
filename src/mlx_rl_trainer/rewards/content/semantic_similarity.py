# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 002
# goals_of_writing_code_block: Semantic similarity-based content reward, refactored for RewardContext and improved TF-IDF batching.
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
from mlx_rl_trainer.utils.text_utils import _tokenize_set, _tfidf_cosine # Import utilities

logger = logging.getLogger(__name__)


@RewardRegistry.register("content_similarity")
class SemanticSimilarityReward(BaseReward):
    """
    Rewards semantic similarity between generated and reference text.

    Uses either TF-IDF Cosine Similarity or (placeholder for) embedding-based methods.

    Configuration:
        method: Similarity method - 'tfidf' or 'embedding' (default: 'tfidf').
        min_length: Minimum text length (in characters) to compute similarity (default: 10).
        max_features: Max features for TF-IDF Vectorizer (default: 1000).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.method = config.get("method", "tfidf")
        self.min_length = config.get("min_length", 10)
        self.max_features = config.get("max_features", 1000)

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                lowercase=True
            )
        elif self.method == "embedding":
            logger.warning("Embedding similarity method not yet fully implemented, falling back to TF-IDF.")
            self.method = "tfidf" # Fallback
            self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english', lowercase=True)

    def compute(
        self,
        context: RewardContext
    ) -> float:
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

            if not generated or not reference or len(generated) < self.min_length or len(reference) < self.min_length:
                logger.debug("Content: Generated or reference text too short or empty. Returning 0.0.")
                return 0.0

            if self.method == "tfidf":
                return self._compute_tfidf_similarity(generated, reference)
            else:
                return 0.0 # Should not be reached if fallback is working

        except Exception as e:
            logger.error(f"SemanticSimilarityReward computation failed: {e}", exc_info=True)
            return 0.0

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Helper to compute TF-IDF based cosine similarity between two texts."""
        try:
            # Fit and transform the two texts
            # Temporarily create a new vectorizer to fit only these two texts
            # (or use the class-wide one after a fit_transform on a larger corpus if batching)
            local_vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english', lowercase=True)
            vectors = local_vectorizer.fit_transform([text1, text2])

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            return float(np.clip(similarity, 0.0, 1.0))

        except Exception as e:
            logger.warning(f"TF-IDF computation failed for pair: {e}. Falling back to Jaccard.", exc_info=True)
            # Fallback to Jaccard if TF-IDF fails for some reason
            A, B = _tokenize_set(text1), _tokenize_set(text2)
            if not A and not B: return 1.0
            if not A or not B: return 0.0
            return float(len(A & B) / len(A | B))

    def batch_compute(
        self,
        contexts: List[RewardContext]
    ) -> List[float]:
        """
        Optimized batch computation for TF-IDF based similarity.

        Args:
            contexts: A list of `RewardContext` objects.

        Returns:
            A list of float similarity scores for the batch.
        """
        if self.method != "tfidf":
            return super().batch_compute(contexts) # Fallback to default sequential if method is not TF-IDF

        try:
            generated_texts = [c.generated_text for c in contexts]
            reference_texts = [c.reference_completion for c in contexts]

            # Combine all texts for a single, comprehensive TF-IDF vocabulary
            all_texts = generated_texts + reference_texts
            if not all_texts:
                return [0.0] * len(contexts)

            # Fit the vectorizer on all texts, then transform
            # Using the class's vectorizer for consistency
            self.vectorizer.fit(all_texts)
            generated_vectors = self.vectorizer.transform(generated_texts)
            reference_vectors = self.vectorizer.transform(reference_texts)

            # Compute pairwise cosine similarity
            # Handle cases where `generated_vectors` or `reference_vectors` might be empty after transform
            if generated_vectors.shape[0] == 0 or reference_vectors.shape[0] == 0:
                return [0.0] * len(contexts)

            similarities = cosine_similarity(generated_vectors, reference_vectors).diagonal()

            return [float(np.clip(s, 0.0, 1.0)) for s in similarities]

        except Exception as e:
            logger.error(f"Batch semantic similarity (TF-IDF) computation failed: {e}", exc_info=True)
            # Fallback to sequential compute if batch fails
            return [self.compute(c) for c in contexts]
