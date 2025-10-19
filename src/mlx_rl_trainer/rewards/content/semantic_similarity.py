# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 008
# goals_of_writing_code_block: Fix 'float has no attribute get' error by returning a dict
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

logger = logging.getLogger(__name__)

# ⭐ USE NLTK STOP WORDS
try:
    from nltk.corpus import stopwords
    import nltk

    try:
        STOP_WORDS = set(stopwords.words("english"))
        logger.info(f"Loaded {len(STOP_WORDS)} NLTK stop words")
    except LookupError:
        logger.info("Downloading NLTK stopwords corpus...")
        nltk.download("stopwords", quiet=True)
        STOP_WORDS = set(stopwords.words("english"))
        logger.info(f"Downloaded and loaded {len(STOP_WORDS)} NLTK stop words")

except ImportError:
    logger.warning("NLTK not installed. Using basic stop words list.")
    # Basic fallback stop words list from previous version
    STOP_WORDS = {
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "out",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    }


def _tokenize_set(text: str, remove_stop_words: bool = True) -> Set[str]:
    """Tokenize text into a set of words with optional stop word removal."""
    if not text:
        return set()
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)
    tokens = set(re.findall(r"\b[a-z0-9]+\b", text))
    if remove_stop_words:
        tokens = tokens - STOP_WORDS
    tokens = {t for t in tokens if len(t) > 1 or t.isdigit()}
    return tokens


@RewardRegistry.register("semantic_similarity")
class SemanticSimilarityReward(BaseReward):
    """
    Rewards semantic similarity, with advanced, configurable length adjustments.

    Supports multiple length adjustment methods:
    - 'static': (Default) Penalizes based on raw character length.
    - 'dynamic_content': Penalizes based on the ratio of meaningful "content words".
    - 'hybrid': Uses a strict Gaussian penalty for short texts and a static
      penalty for longer texts.

    Configuration:
        method: 'tfidf' or 'jaccard'.
        length_adjustment_method: 'static', 'dynamic_content', or 'hybrid'.
        # ... other method-specific configs below ...
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.method = config.get("method", "tfidf")
        self.min_length = config.get("min_length", 10)
        self.debug_logging = config.get("debug_logging", True)

        # ⭐ NEW: Main switch for length adjustment logic
        self.length_adjustment_method = config.get("length_adjustment_method", "static")

        # Config for 'static' and 'hybrid' methods (backwards compatible)
        self.apply_brevity_penalty = config.get("apply_length_penalty", True)
        self.apply_verbosity_penalty = config.get("apply_verbosity_penalty", True)
        self.verbosity_penalty_strength = config.get("verbosity_penalty_strength", 0.5)

        # Config for 'dynamic_content' method
        self.dynamic_strength = config.get("dynamic_strength", 1.0)
        self.remove_stop_words = config.get("remove_stop_words", True)

        # Config for 'hybrid' method
        self.hybrid_length_threshold = config.get("hybrid_length_threshold", 100)
        self.gaussian_tolerance = config.get("gaussian_tolerance", 0.25)

        self.gen_config = GenerationConfig()
        self.extract_after_tag = config.get(
            "extract_after_tag", self.gen_config.think_end_tag
        )

        if self.method == "tfidf":
            self.max_features = config.get("max_features", 1000)
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english", lowercase=True
            )

        logger.info(
            f"SemanticSimilarityReward initialized with length adjustment "
            f"method: '{self.length_adjustment_method}'"
        )

    def _extract_answer_text(self, text: str) -> str:
        """Extracts answer text after the think tag."""
        if not text:
            return ""
        if self.extract_after_tag and self.extract_after_tag in text:
            parts = text.split(self.extract_after_tag, 1)
            if len(parts) > 1:
                return parts[1].strip()
        return text.strip()

    def _calculate_length_adjustment(self, gen_ans: str, ref_ans: str) -> float:
        """Dispatcher to select the configured length adjustment method."""
        if self.length_adjustment_method == "dynamic_content":
            return self._calculate_dynamic_content_adjustment(gen_ans, ref_ans)

        gen_len, ref_len = len(gen_ans), len(ref_ans)
        if self.length_adjustment_method == "hybrid":
            return self._calculate_hybrid_adjustment(gen_len, ref_len)

        return self._calculate_static_adjustment(gen_len, ref_len)

    def _calculate_static_adjustment(self, gen_len: int, ref_len: int) -> float:
        """Calculates penalty based on raw character length."""
        if gen_len <= 0 or ref_len <= 0:
            return 1.0
        if self.apply_brevity_penalty and gen_len < ref_len:
            return np.exp(1.0 - ref_len / gen_len)
        if self.apply_verbosity_penalty and gen_len > ref_len:
            return (ref_len / gen_len) ** self.verbosity_penalty_strength
        return 1.0

    def _calculate_dynamic_content_adjustment(
        self, gen_ans: str, ref_ans: str
    ) -> float:
        """Calculates penalty based on the ratio of content words."""
        gen_words = _tokenize_set(gen_ans, self.remove_stop_words)
        ref_words = _tokenize_set(ref_ans, self.remove_stop_words)
        gen_content_len, ref_content_len = len(gen_words), len(ref_words)

        if ref_content_len == 0 or gen_content_len == 0:
            return 1.0

        content_ratio = gen_content_len / ref_content_len

        if content_ratio > 1.0:
            return (1.0 / content_ratio) ** self.dynamic_strength
        else:
            return content_ratio**self.dynamic_strength

    def _calculate_hybrid_adjustment(self, gen_len: int, ref_len: int) -> float:
        """Uses Gaussian penalty for short texts, static for long texts."""
        if ref_len < self.hybrid_length_threshold:
            stdev = ref_len * self.gaussian_tolerance
            if stdev == 0:
                return 1.0 if gen_len == ref_len else 0.0
            return np.exp(-0.5 * ((gen_len - ref_len) / stdev) ** 2)
        else:
            return self._calculate_static_adjustment(gen_len, ref_len)

    # ⭐ MODIFIED: Updated return type hint from float to Dict
    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """Computes semantic similarity reward for a single context."""
        try:
            gen_ans = self._extract_answer_text(context.generated_text)
            ref_ans = self._extract_answer_text(context.reference_completion)

            if len(gen_ans) < self.min_length or len(ref_ans) < self.min_length:
                # ⭐ MODIFIED: Return a dict even on failure
                return {"reward": 0.0, "log": {"error": "Text too short"}}

            if self.method == "tfidf":
                score = self._compute_tfidf_similarity(gen_ans, ref_ans)
            else:
                score = self._compute_jaccard_similarity(gen_ans, ref_ans)

            adjustment_factor = self._calculate_length_adjustment(gen_ans, ref_ans)
            weighted_score = score * adjustment_factor

            log_data = {
                "raw_score": score,
                "adjustment_factor": adjustment_factor,
                "weighted_score": weighted_score,
                "adjustment_method": self.length_adjustment_method,
            }

            if self.debug_logging:
                logger.info(
                    f"SemanticSimilarity | raw_score={score:.4f}, "
                    f"adj_factor={adjustment_factor:.4f}, "
                    f"weighted_score={weighted_score:.4f}"
                )

            # ⭐ MODIFIED: Return a dictionary with 'reward' and 'log' keys
            return {"reward": weighted_score, "log": log_data}

        except Exception as e:
            logger.error(f"SemanticSimilarityReward failed: {e}", exc_info=True)
            # ⭐ MODIFIED: Return a dict even on failure
            return {"reward": 0.0, "log": {"error": str(e)}}

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Computes TF-IDF cosine similarity."""
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(np.clip(similarity, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Computes Jaccard similarity."""
        try:
            A = _tokenize_set(text1, remove_stop_words=self.remove_stop_words)
            B = _tokenize_set(text2, remove_stop_words=self.remove_stop_words)
            if not A and not B:
                return 1.0
            intersection = len(A.intersection(B))
            union = len(A.union(B))
            return float(intersection / union) if union > 0 else 0.0
        except Exception:
            return 0.0

    # ⭐ MODIFIED: Updated return type hint from List[float] to List[Dict]
    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """Computes rewards for a batch of contexts."""
        return [self.compute(c) for c in contexts]
