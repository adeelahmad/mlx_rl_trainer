# file_path: mlx_rl_trainer/src/mlx_rl_trainer/rewards/content/semantic_similarity.py
# revision_no: 004
# goals: NON-BREAKING memory-efficient hack-proof enhancement - no training data changes required
# type: drop-in replacement
"""Semantic similarity-based content reward with internal anti-gaming mechanisms."""

from typing import Dict, Any, List, Tuple
import logging
import numpy as np
from collections import deque
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig
from mlx_rl_trainer.utils.text_utils import _tokenize_set, _tfidf_cosine

logger = logging.getLogger(__name__)


@RewardRegistry.register("semantic_similarity")
class SemanticSimilarityReward(BaseReward):
    """
    Semantic similarity with INTERNAL anti-gaming mechanisms.

    FULLY BACKWARD COMPATIBLE - no training data changes required.
    All anti-gaming is computed internally from generated text analysis.

    New Features (all automatic, no data changes needed):
    - Multi-signal validation (semantic + structure + reasoning)
    - Automatic diversity tracking across generations
    - Reasoning quality scoring from think tags
    - Length-based gaming detection
    - Exact match penalty
    - Vocabulary diversity enforcement
    - Memory-efficient with __slots__ and deque

    Configuration (all optional, backward compatible):
        method: 'tfidf' or 'jaccard' (default: 'tfidf')
        min_length: Minimum answer length (default: 10)
        max_features: TF-IDF max features (default: 1000)
        min_reasoning_length: Minimum reasoning length (default: 50)
        exact_match_penalty: Penalty multiplier for exact matches (default: 0.5)
        length_ratio_tolerance: Acceptable length ratio range (default: 0.4)
        reasoning_weight: Weight for reasoning quality (default: 0.35)
        semantic_weight: Weight for semantic similarity (default: 0.45)
        diversity_weight: Weight for diversity bonus (default: 0.20)
        diversity_history_size: Responses to track (default: 500)
    """

    __slots__ = (
        "method",
        "min_length",
        "max_features",
        "gen_config",
        "extract_after_tag",
        "_think_start_tag",
        "min_reasoning_length",
        "exact_match_penalty",
        "length_ratio_tolerance",
        "reasoning_weight",
        "semantic_weight",
        "diversity_weight",
        "_diversity_tracker",
        "_response_hashes",
        "_vocab_history",
        "diversity_history_size",
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Original config (backward compatible)
        self.method = config.get("method", "tfidf")
        self.min_length = config.get("min_length", 10)
        self.max_features = config.get("max_features", 1000)

        # Anti-gaming config (new, but with defaults)
        self.min_reasoning_length = config.get("min_reasoning_length", 50)
        self.exact_match_penalty = config.get("exact_match_penalty", 0.5)
        self.length_ratio_tolerance = config.get("length_ratio_tolerance", 0.4)

        # Multi-signal weights (defaults maintain backward compatibility)
        self.reasoning_weight = config.get("reasoning_weight", 0.35)
        self.semantic_weight = config.get("semantic_weight", 0.45)
        self.diversity_weight = config.get("diversity_weight", 0.20)

        # Generation config for tag extraction
        self.gen_config = GenerationConfig()
        self.extract_after_tag = config.get(
            "extract_after_tag", self.gen_config.think_end_tag
        )
        self._think_start_tag = getattr(self.gen_config, "think_start_tag", "<think>")

        # Memory-efficient diversity tracking (internal only)
        self.diversity_history_size = config.get("diversity_history_size", 500)
        self._diversity_tracker = deque(maxlen=self.diversity_history_size)
        self._response_hashes = set()
        self._vocab_history = deque(maxlen=self.diversity_history_size)

    def _extract_sections(self, text: str) -> Tuple[str, str]:
        """Extract reasoning and answer sections from text."""
        if not text:
            return "", ""

        reasoning = ""
        answer = ""

        # Extract reasoning (between <think> and </think>)
        if self._think_start_tag in text and self.extract_after_tag in text:
            start_idx = text.find(self._think_start_tag) + len(self._think_start_tag)
            end_idx = text.find(self.extract_after_tag)
            if 0 < start_idx < end_idx:
                reasoning = text[start_idx:end_idx].strip()

        # Extract answer (after </think>)
        if self.extract_after_tag and self.extract_after_tag in text:
            parts = text.split(self.extract_after_tag, 1)
            if len(parts) > 1:
                answer = parts[1].strip()
        else:
            answer = text.strip()

        return reasoning, answer

    def _compute_reasoning_quality(self, reasoning: str) -> float:
        """
        Compute reasoning quality from structure and content.
        No external data needed - analyzes text directly.
        """
        if not reasoning or len(reasoning) < self.min_reasoning_length:
            return 0.0

        words = reasoning.lower().split()
        word_count = len(words)

        if word_count < 10:
            return 0.1

        # Signal 1: Reasoning depth (logical connectives)
        depth_markers = {
            "first",
            "second",
            "third",
            "then",
            "next",
            "therefore",
            "thus",
            "because",
            "since",
            "given",
            "if",
            "however",
            "although",
        }
        depth_score = min(1.0, sum(1 for w in words if w in depth_markers) / 5)

        # Signal 2: Vocabulary diversity (unique words ratio)
        unique_ratio = len(set(words)) / word_count
        diversity_score = min(1.0, unique_ratio / 0.5)  # Target 50% unique

        # Signal 3: Length adequacy (not too short)
        length_score = min(1.0, len(reasoning) / (self.min_reasoning_length * 2))

        # Signal 4: Avoid vague language
        vague_words = {"thing", "stuff", "maybe", "possibly", "somehow"}
        vague_count = sum(1 for w in words if w in vague_words)
        vague_penalty = min(0.3, vague_count / word_count * 5)

        # Combine signals
        quality = (depth_score * 0.4 + diversity_score * 0.3 + length_score * 0.3) * (
            1.0 - vague_penalty
        )

        return float(np.clip(quality, 0.0, 1.0))

    def _compute_length_penalty(self, text1: str, text2: str) -> float:
        """Penalize extreme length mismatches (gaming detection)."""
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0

        ratio = min(len1, len2) / max(len1, len2)

        if ratio < self.length_ratio_tolerance:
            return ratio / self.length_ratio_tolerance

        return 1.0

    def _compute_exact_match_penalty(self, text1: str, text2: str) -> float:
        """
        Detect copy-paste gaming via exact/near-exact matching.
        Returns penalty factor (0.0 = no penalty, 1.0 = maximum penalty).
        """
        if not text1 or not text2:
            return 0.0

        norm1 = text1.lower().strip()
        norm2 = text2.lower().strip()

        # Exact match = maximum penalty
        if norm1 == norm2:
            return 1.0

        # Character-level similarity for near-exact detection
        shorter = min(len(norm1), len(norm2))
        longer = max(len(norm1), len(norm2))

        if longer == 0:
            return 0.0

        # Count matching characters in order
        matches = sum(c1 == c2 for c1, c2 in zip(norm1, norm2))
        char_similarity = matches / longer

        # Penalize if >95% character match (likely copy-paste)
        if char_similarity > 0.95:
            return (char_similarity - 0.95) * 20  # Scale to 0-1

        return 0.0

    def _compute_diversity_bonus(self, text: str) -> float:
        """
        Compute diversity bonus from internal tracking.
        Rewards novel responses, penalizes repetition.
        NO external data needed.
        """
        if not text:
            return 0.5  # Neutral

        # Generate fingerprint
        normalized = text.lower().strip()
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        vocab = set(normalized.split())

        # Check for exact duplicates
        if text_hash in self._response_hashes:
            return 0.0  # Severe penalty for exact duplicate

        # Check vocabulary overlap with history
        if not self._vocab_history:
            # First response - neutral
            novelty = 0.8
        else:
            # Compute overlap with historical responses
            overlaps = []
            for hist_vocab in list(self._vocab_history)[-50:]:  # Check recent 50
                if not vocab or not hist_vocab:
                    continue
                intersection = len(vocab & hist_vocab)
                union = len(vocab | hist_vocab)
                if union > 0:
                    overlaps.append(intersection / union)

            if overlaps:
                # Penalize high overlap with any recent response
                max_overlap = max(overlaps)
                novelty = 1.0 - max_overlap
            else:
                novelty = 0.8

        # Update tracking (internal state)
        self._response_hashes.add(text_hash)
        self._vocab_history.append(vocab)
        if len(self._response_hashes) > self.diversity_history_size:
            # Clean oldest hashes periodically
            self._response_hashes = set(
                list(self._response_hashes)[-self.diversity_history_size :]
            )

        return float(np.clip(novelty, 0.0, 1.0))

    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Memory-efficient TF-IDF with float32."""
        try:
            if not text1 or not text2:
                return 0.0

            # Local vectorizer for thread safety and memory efficiency
            vec = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                lowercase=True,
                dtype=np.float32,  # 50% memory vs float64
            )

            vectors = vec.fit_transform([text1, text2])

            if vectors.shape[0] < 2 or vectors.nnz == 0:
                return 0.0

            # Sparse similarity computation (memory efficient)
            sim = cosine_similarity(vectors[0:1], vectors[1:2], dense_output=False)[
                0, 0
            ]
            return float(np.clip(sim, 0.0, 1.0))

        except Exception as e:
            logger.debug(f"TF-IDF failed: {e}, using Jaccard fallback")
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Memory-efficient Jaccard similarity."""
        try:
            if not text1 or not text2:
                return 0.0

            set1, set2 = _tokenize_set(text1), _tokenize_set(text2)

            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0

            intersection = len(set1 & set2)
            union = len(set1 | set2)

            return float(intersection / union) if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Jaccard computation failed: {e}", exc_info=True)
            return 0.0

    def compute(self, context: RewardContext) -> float:
        """
        Compute multi-signal hack-proof reward.
        FULLY BACKWARD COMPATIBLE - works with existing data.

        Combines:
        1. Semantic similarity (core signal)
        2. Reasoning quality (automatic from think tags)
        3. Diversity bonus (automatic from internal tracking)

        With automatic penalties for:
        - Exact/near-exact matches
        - Length gaming
        - Low reasoning quality
        """
        try:
            self.validate_inputs(context)

            # Extract sections
            gen_reasoning, gen_answer = self._extract_sections(context.generated_text)
            ref_reasoning, ref_answer = self._extract_sections(
                context.reference_completion
            )

            # Signal 1: Reasoning quality (automatic, no data changes needed)
            reasoning_score = self._compute_reasoning_quality(gen_reasoning)

            # Minimum reasoning threshold
            if reasoning_score < 0.15:
                return 0.05  # Very low reward for insufficient reasoning

            # Check answer length
            if len(gen_answer) < self.min_length or len(ref_answer) < self.min_length:
                return 0.05 * reasoning_score

            # Signal 2: Semantic similarity (core)
            if self.method == "tfidf":
                semantic_score = self._compute_tfidf_similarity(gen_answer, ref_answer)
            else:
                semantic_score = self._compute_jaccard_similarity(
                    gen_answer, ref_answer
                )

            # Apply anti-gaming penalties
            length_penalty = self._compute_length_penalty(gen_answer, ref_answer)
            exact_penalty = self._compute_exact_match_penalty(gen_answer, ref_answer)

            # Adjust semantic with penalties
            semantic_score *= length_penalty
            semantic_score *= 1.0 - exact_penalty * self.exact_match_penalty

            # Signal 3: Diversity bonus (automatic from internal tracking)
            diversity_score = self._compute_diversity_bonus(context.generated_text)

            # Combine signals with weights
            final_score = (
                semantic_score * self.semantic_weight
                + reasoning_score * self.reasoning_weight
                + diversity_score * self.diversity_weight
            )

            # Apply reasoning as quality gate
            final_score *= 0.5 + 0.5 * reasoning_score  # Boost if good reasoning

            return float(np.clip(final_score, 0.0, 1.0))

        except Exception as e:
            logger.error(
                f"SemanticSimilarityReward computation failed: {e}", exc_info=True
            )
            return 0.0

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        """
        Memory-efficient batch processing with all anti-gaming checks.
        FULLY BACKWARD COMPATIBLE.
        """
        if self.method == "jaccard":
            # Jaccard doesn't benefit from batching
            return super().batch_compute(contexts)

        try:
            n = len(contexts)
            results = []

            # Extract all sections once (memory efficient)
            gen_sections = [self._extract_sections(c.generated_text) for c in contexts]
            ref_sections = [
                self._extract_sections(c.reference_completion) for c in contexts
            ]

            # Compute reasoning quality for all (cheap operation)
            reasoning_scores = [
                self._compute_reasoning_quality(gen_r) for gen_r, _ in gen_sections
            ]

            # Filter valid pairs for batch TF-IDF
            valid_indices = []
            valid_gen_answers = []
            valid_ref_answers = []

            for i, ((gen_r, gen_a), (ref_r, ref_a), reasoning) in enumerate(
                zip(gen_sections, ref_sections, reasoning_scores)
            ):
                if (
                    reasoning < 0.15
                    or len(gen_a) < self.min_length
                    or len(ref_a) < self.min_length
                ):
                    # Skip invalid, will assign low score
                    continue

                valid_indices.append(i)
                valid_gen_answers.append(gen_a)
                valid_ref_answers.append(ref_a)

            # Batch TF-IDF computation (memory efficient with float32)
            semantic_scores = [0.0] * n

            if valid_gen_answers:
                all_texts = valid_gen_answers + valid_ref_answers

                vec = TfidfVectorizer(
                    max_features=self.max_features,
                    stop_words="english",
                    lowercase=True,
                    dtype=np.float32,
                )

                vectors = vec.fit_transform(all_texts)
                n_valid = len(valid_gen_answers)

                gen_vectors = vectors[:n_valid]
                ref_vectors = vectors[n_valid:]

                similarities = cosine_similarity(
                    gen_vectors, ref_vectors, dense_output=False
                ).diagonal()

                for idx, sim in zip(valid_indices, similarities):
                    semantic_scores[idx] = float(sim)

            # Compute final scores with all components
            for i, context in enumerate(contexts):
                gen_r, gen_a = gen_sections[i]
                ref_r, ref_a = ref_sections[i]

                reasoning_score = reasoning_scores[i]
                semantic_score = semantic_scores[i]

                if reasoning_score < 0.15 or not semantic_score:
                    final_score = 0.05 * reasoning_score
                else:
                    # Apply penalties
                    length_penalty = self._compute_length_penalty(gen_a, ref_a)
                    exact_penalty = self._compute_exact_match_penalty(gen_a, ref_a)

                    semantic_score *= length_penalty
                    semantic_score *= 1.0 - exact_penalty * self.exact_match_penalty

                    # Diversity
                    diversity_score = self._compute_diversity_bonus(
                        context.generated_text
                    )

                    # Combine
                    final_score = (
                        semantic_score * self.semantic_weight
                        + reasoning_score * self.reasoning_weight
                        + diversity_score * self.diversity_weight
                    )

                    final_score *= 0.5 + 0.5 * reasoning_score

                final_score = float(np.clip(final_score, 0.0, 1.0))

                results.append({"total": final_score, self.name: final_score})

            return results

        except Exception as e:
            logger.error(
                f"Batch semantic similarity computation failed: {e}",
                exc_info=True,
            )
            return super().batch_compute(contexts)
