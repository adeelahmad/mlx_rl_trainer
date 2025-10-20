# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 004
# goals_of_writing_code_block: Semantic similarity reward with comprehensive debugging and logging
# type_of_code_response: change existing
"""Semantic similarity-based content reward with detailed debugging."""

from typing import Dict, Any, List, Set
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig

# from mlx_rl_trainer.utils.text_utils import (
#     _tfidf_cosine,
# )  # Import utilities

logger = logging.getLogger(__name__)


# ⭐ USE NLTK STOP WORDS
try:
    from nltk.corpus import stopwords
    import nltk

    # Attempt to load stopwords, download if not present
    try:
        STOP_WORDS = set(stopwords.words("english"))
        logger.info(f"Loaded {len(STOP_WORDS)} NLTK stop words")
    except LookupError:
        # First time - download the corpus
        logger.info("Downloading NLTK stopwords corpus...")
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
        logger.info(f"Downloaded and loaded {len(STOP_WORDS)} NLTK stop words")

except ImportError:
    # Fallback if NLTK not installed
    logger.warning(
        "NLTK not installed. Using basic stop words list. "
        "Install with: pip install nltk"
    )
    STOP_WORDS = {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "might",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }


def _tokenize_set(text: str, remove_stop_words: bool = True) -> Set[str]:
    """
    Tokenize text into set of words with optional stop word removal.

    Uses NLTK stop words if available, otherwise falls back to basic list.

    Args:
        text: Input text to tokenize
        remove_stop_words: Whether to filter out stop words (default: True)

    Returns:
        Set of tokens
    """
    if not text:
        return set()

    # Lowercase
    text = text.lower()

    # Handle common contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)

    # Extract alphanumeric tokens
    tokens = set(re.findall(r"\b[a-z0-9]+\b", text))

    # Remove stop words if requested
    if remove_stop_words:
        tokens = tokens - STOP_WORDS

    # Remove very short tokens except numbers
    tokens = {t for t in tokens if len(t) > 2 or t.isdigit()}

    return tokens


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
        debug_logging: Enable detailed logging (default: True).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.method = config.get("method", "tfidf")
        self.min_length = config.get("min_length", 10)
        self.max_features = config.get("max_features", 1000)
        self.debug_logging = config.get("debug_logging", True)

        # ⭐ ADD THIS LINE - A/B testing for stop words
        self.remove_stop_words = config.get("remove_stop_words", True)

        # Get generation config for tag extraction
        self.gen_config = GenerationConfig()
        self.extract_after_tag = config.get(
            "extract_after_tag", self.gen_config.think_end_tag
        )

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",  # TF-IDF has built-in stop words
                lowercase=True,
            )
            logger.info(
                f"SemanticSimilarityReward initialized with TF-IDF "
                f"(max_features={self.max_features})"
            )
        elif self.method == "jaccard":
            logger.info("SemanticSimilarityReward initialized with Jaccard similarity")
        else:
            logger.warning(
                f"Unknown similarity method '{self.method}', falling back to TF-IDF."
            )
            self.method = "tfidf"
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english", lowercase=True
            )

        logger.info(
            f"SemanticSimilarity config: method={self.method}, "
            f"min_length={self.min_length}, "
            f"extract_after_tag='{self.extract_after_tag}', "
            f"remove_stop_words={self.remove_stop_words}"  # ⭐ ADD THIS
        )

    def _extract_answer_text(self, text: str) -> str:
        """Extract answer after </think> tag, stripping garbage."""
        if not text:
            return ""

        if self.extract_after_tag and self.extract_after_tag in text:
            parts = text.split(self.extract_after_tag, 1)
            if len(parts) > 1:
                answer = parts[1].strip()

                # CRITICAL FIX: Strip leading single-char garbage
                # Model generates "Q\nActual answer" or "A\nActual answer"
                lines = answer.split("\n", 1)
                if len(lines) > 1 and len(lines[0]) <= 2:  # Single char or char+space
                    answer = lines[1].strip()
                    if self.debug_logging:
                        logger.warning(f"Stripped leading garbage: '{lines[0]}'")

                return answer

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

            # Log extraction results
            if self.debug_logging:
                logger.info(
                    f"SemanticSimilarity extraction | "
                    f"generated_full={len(generated)}, "
                    f"generated_answer={len(generated_answer)}, "
                    f"reference_full={len(reference)}, "
                    f"reference_answer={len(reference_answer)}"
                )
                logger.debug(f"Generated answer preview: {generated_answer[:100]}...")
                logger.debug(f"Reference answer preview: {reference_answer[:100]}...")

            # Check minimum length
            if (
                len(generated_answer) < self.min_length
                or len(reference_answer) < self.min_length
            ):
                if self.debug_logging:
                    logger.warning(
                        f"SemanticSimilarity: Text too short | "
                        f"generated={len(generated_answer)} chars "
                        f"(min={self.min_length}), "
                        f"reference={len(reference_answer)} chars "
                        f"(min={self.min_length}). Returning 0.0"
                    )
                return 0.0

            if self.method == "tfidf":
                score = self._compute_tfidf_similarity(
                    generated_answer, reference_answer
                )
            elif self.method == "jaccard":
                score = self._compute_jaccard_similarity(
                    generated_answer, reference_answer
                )
            else:
                score = 0.0

            if self.debug_logging:
                logger.info(
                    f"SemanticSimilarity | method={self.method}, score={score:.4f}"
                )

            return score

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

            if self.debug_logging:
                logger.debug(
                    f"TF-IDF vectorizing texts: "
                    f"text1={len(text1)} chars, text2={len(text2)} chars"
                )

            vectors = local_vectorizer.fit_transform([text1, text2])

            # Check if vectors are non-empty
            if vectors.shape[0] < 2:
                if self.debug_logging:
                    logger.warning(
                        f"TF-IDF: Not enough valid vectors after transformation. "
                        f"Shape: {vectors.shape}"
                    )
                return 0.0

            if self.debug_logging:
                logger.debug(
                    f"TF-IDF vectors shape: {vectors.shape}, "
                    f"vocab_size: {len(local_vectorizer.vocabulary_)}"
                )

            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            clipped_similarity = float(np.clip(similarity, 0.0, 1.0))

            if self.debug_logging:
                logger.debug(
                    f"TF-IDF cosine similarity: {similarity:.4f} "
                    f"(clipped: {clipped_similarity:.4f})"
                )

            return clipped_similarity

        except Exception as e:
            logger.warning(
                f"TF-IDF computation failed: {e}. Falling back to Jaccard.",
                exc_info=True,
            )
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Helper to compute Jaccard similarity between two texts."""
        try:
            # ⭐ PASS self.remove_stop_words HERE
            A = _tokenize_set(text1, remove_stop_words=self.remove_stop_words)
            B = _tokenize_set(text2, remove_stop_words=self.remove_stop_words)

            if self.debug_logging:
                stop_words_msg = "with" if self.remove_stop_words else "without"
                logger.debug(
                    f"Jaccard sets ({stop_words_msg} stop word removal): "
                    f"A={len(A)} tokens, B={len(B)} tokens"
                )

            # Edge cases
            if not A and not B:
                if self.debug_logging:
                    logger.debug("Jaccard: Both sets empty, returning 1.0")
                return 1.0

            if not A or not B:
                if self.debug_logging:
                    logger.warning(
                        f"Jaccard: One set empty (A={len(A)}, B={len(B)}), returning 0.0"
                    )
                return 0.0

            intersection = len(A & B)
            union = len(A | B)

            if union == 0:
                if self.debug_logging:
                    logger.warning("Jaccard: Union is 0, returning 0.0")
                return 0.0

            similarity = float(intersection / union)

            if self.debug_logging:
                logger.debug(
                    f"Jaccard: intersection={intersection}, union={union}, "
                    f"similarity={similarity:.4f}"
                )
                # Log sample tokens
                token_type = "content words" if self.remove_stop_words else "all tokens"
                logger.debug(f"Sample {token_type} A: {list(A)[:10]}")
                logger.debug(f"Sample {token_type} B: {list(B)[:10]}")
                logger.debug(f"Sample intersection: {list(A & B)[:10]}")

            return similarity

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
            if self.debug_logging:
                logger.info(
                    f"Batch compute with Jaccard (sequential): {len(contexts)} contexts"
                )
            return super().batch_compute(contexts)

        try:
            if self.debug_logging:
                logger.info(f"Batch compute with TF-IDF: {len(contexts)} contexts")

            # Extract answer portions for all texts
            generated_answers = [
                self._extract_answer_text(c.generated_text) for c in contexts
            ]
            reference_answers = [
                self._extract_answer_text(c.reference_completion) for c in contexts
            ]

            # Filter out pairs where either text is too short
            valid_indices = []
            valid_generated = []
            valid_reference = []

            for i, (gen, ref) in enumerate(zip(generated_answers, reference_answers)):
                if len(gen) >= self.min_length and len(ref) >= self.min_length:
                    valid_indices.append(i)
                    valid_generated.append(gen)
                    valid_reference.append(ref)

            if self.debug_logging:
                logger.info(
                    f"Batch compute: {len(valid_indices)}/{len(contexts)} valid pairs "
                    f"(min_length={self.min_length})"
                )

            # Initialize all scores to 0.0
            scores = [0.0] * len(contexts)

            if not valid_generated:
                if self.debug_logging:
                    logger.warning("Batch compute: No valid text pairs found")
                return scores

            # Combine all valid texts for a single, comprehensive TF-IDF vocabulary
            all_texts = valid_generated + valid_reference

            if self.debug_logging:
                logger.debug(
                    f"Fitting TF-IDF on {len(all_texts)} texts "
                    f"(max_features={self.max_features})"
                )

            # Fit the vectorizer on all texts, then transform
            self.vectorizer.fit(all_texts)
            generated_vectors = self.vectorizer.transform(valid_generated)
            reference_vectors = self.vectorizer.transform(valid_reference)

            if self.debug_logging:
                logger.debug(
                    f"TF-IDF vectors: generated={generated_vectors.shape}, "
                    f"reference={reference_vectors.shape}, "
                    f"vocab_size={len(self.vectorizer.vocabulary_)}"
                )

            # Compute pairwise cosine similarity
            if generated_vectors.shape[0] == 0 or reference_vectors.shape[0] == 0:
                if self.debug_logging:
                    logger.warning("Batch compute: Empty vectors after transformation")
                return scores

            similarities = cosine_similarity(
                generated_vectors, reference_vectors
            ).diagonal()

            if self.debug_logging:
                logger.debug(
                    f"Computed {len(similarities)} similarities: "
                    f"min={similarities.min():.4f}, "
                    f"max={similarities.max():.4f}, "
                    f"mean={similarities.mean():.4f}"
                )

            # Map similarities back to original indices
            for idx, sim in zip(valid_indices, similarities):
                scores[idx] = float(np.clip(sim, 0.0, 1.0))

            if self.debug_logging:
                non_zero_scores = [s for s in scores if s > 0.0]
                logger.info(
                    f"Batch compute results: {len(non_zero_scores)}/{len(scores)} "
                    f"non-zero scores"
                )

            return scores

        except Exception as e:
            logger.error(
                f"Batch semantic similarity (TF-IDF) computation failed: {e}",
                exc_info=True,
            )
            # Fallback to sequential compute if batch fails
            if self.debug_logging:
                logger.warning("Falling back to sequential computation")
            return [self.compute(c) for c in contexts]
