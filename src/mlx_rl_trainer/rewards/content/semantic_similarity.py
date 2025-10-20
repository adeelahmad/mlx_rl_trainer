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
        self.verbosity_penalty_strength = float(config.get("verbosity_penalty_strength", 0.01))
        self.method = str(config.get("method", "tfidf")).lower()

        # Default tag for extraction
        self.extract_after_tag = "</think>"

        # Initialize GenerationConfig once and store it as an instance attribute
        try:
            # Use load_config from core.config to get a properly initialized config
            from mlx_rl_trainer.core.config import load_config

            # Try to load the experiment config which contains generation config
            try:
                experiment_config = load_config("config.yaml")
                self.gen_config = experiment_config.generation
                logger.debug("Loaded generation config from experiment config")
            except Exception:
                # If that fails, create a default GenerationConfig with all required parameters
                from mlx_rl_trainer.core.config import GenerationConfig
                self.gen_config = GenerationConfig(
                    think_start_tag="<think>",
                    think_end_tag="</think>",
                    answer_start_tag="",
                    answer_end_tag="",
                    think_boost_tokens=8,
                    think_temperature=0.2,
                    answer_temperature=0.3,
                    sampling_top_p=0.6,
                    sampling_min_p=0.0,
                    sampling_top_k=80,
                    repetition_penalty=1.1,
                    repetition_context_size=20,
                    min_tokens_to_keep=1,
                    xtc_probability=None,
                    xtc_threshold=None,
                    min_think_tokens=32,
                    think_end_early_bias=-20.0,
                    bias_answer_start_after_min_think=True,
                    bias_close_think=12.0,
                    bias_answer_start=10.0,
                    punish_extra_think_end=-22.0,
                    punish_reopen_think=-10.0,
                    punish_reopen_answer=-9.0,
                    bias_eos_after_answer=4.0,
                    hard_mask_mcq_first_token=True,
                    mcq_letter_lift=8.0,
                    mcq_ban_first_bias=-14.0,
                    nonmcq_ban_first_bias=-12.0,
                    mcq_close_after_k=1,
                    min_answer_tokens=8,
                    min_answer_tokens_mcq=1,
                    mcq_answer_end_bias=9.0,
                    encourage_think_bias=4.5,
                    ban_think_bias=-5.0,
                    allow_tool_calls=True,
                    tool_call_penalty=0.0,
                    think_length_target_min=8,
                    think_length_target_max=64,
                    think_length_penalty_strength=0.8
                )
                logger.debug("Created default GenerationConfig")

            # Set the extract_after_tag based on the config
            if hasattr(self.gen_config, "think_end_tag") and self.gen_config.think_end_tag:
                self.extract_after_tag = self.gen_config.think_end_tag

        except Exception as e:
            logger.warning(f"Could not load GenerationConfig: {e}. Using default tag: {self.extract_after_tag}")
            # Create a minimal config with default values as a fallback
            self.gen_config = type('GenerationConfig', (), {
                'think_start_tag': '<think>',
                'think_end_tag': '</think>'
            })()

        logger.info(f"SemanticSimilarityReward initialized: min_length={self.min_length}, "
                   f"method={self.method}, verbosity_penalty={self.apply_verbosity_penalty}")

    def _extract_answer(self, text: str) -> str:
        """Extracts text content after the </think> tag."""
        if self.extract_after_tag in text:
            try:
                # Split and take the second part after the tag
                return text.split(self.extract_after_tag, 1)[1].strip()
            except IndexError:
                return ""  # Tag exists, but there's nothing after it
        # If no tag found, use the full text as answer (graceful fallback)
        return text.strip()

    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """Computes the semantic similarity score for a single context."""
        try:
            # Extract and validate text content
            generated_answer = self._extract_answer(context.generated_text or "")
            reference_answer = self._extract_answer(context.reference_completion or "")

            # If either text is too short or missing, the reward is zero.
            if len(generated_answer) < self.min_length or len(reference_answer) < self.min_length:
                return {"reward": 0.0, "log": "text_too_short_or_missing"}

            # Clean the texts to get a good representation for TF-IDF
            cleaned_gen = _clean_text_for_tfidf(generated_answer)
            cleaned_ref = _clean_text_for_tfidf(reference_answer)

            if not cleaned_gen or not cleaned_ref:
                return {"reward": 0.0, "log": "empty_after_cleaning"}

            # Calculate similarity based on the selected method
            if self.method == "tfidf":
                # Create a new vectorizer for each comparison to avoid state issues
                try:
                    vectorizer = TfidfVectorizer(stop_words="english", norm="l2", min_df=1)
                    vectors = vectorizer.fit_transform([cleaned_gen, cleaned_ref])

                    # Verify vectors are not empty
                    if vectors.shape[0] != 2 or vectors.getnnz() == 0:
                        return {"reward": 0.0, "log": "empty_vectors"}

                    # Calculate cosine similarity safely
                    try:
                        # Get the vectors as arrays (handles both sparse and dense matrices)
                        vec1 = vectors[0].toarray().flatten() if hasattr(vectors[0], "toarray") else vectors[0].flatten()
                        vec2 = vectors[1].toarray().flatten() if hasattr(vectors[1], "toarray") else vectors[1].flatten()

                        # Calculate dot product and magnitudes manually for safety
                        dot_product = np.dot(vec1, vec2)
                        magnitude1 = np.sqrt(np.dot(vec1, vec1))
                        magnitude2 = np.sqrt(np.dot(vec2, vec2))

                        # Avoid division by zero
                        if magnitude1 > 0 and magnitude2 > 0:
                            raw_score = float(dot_product / (magnitude1 * magnitude2))
                        else:
                            raw_score = 0.0

                    except Exception as e:
                        logger.warning(f"Manual cosine similarity calculation failed: {e}. Using fallback.")
                        # Fallback to simple word overlap
                        gen_words = set(cleaned_gen.split())
                        ref_words = set(cleaned_ref.split())
                        if not gen_words or not ref_words:
                            return {"reward": 0.0, "log": "no_words_for_overlap"}

                        overlap = len(gen_words.intersection(ref_words))
                        total = len(gen_words.union(ref_words))
                        raw_score = float(overlap / total) if total > 0 else 0.0
                except Exception as vec_error:
                    logger.warning(f"TF-IDF vectorization failed: {vec_error}. Falling back to simple overlap.")
                    # Fallback to simple word overlap
                    gen_words = set(cleaned_gen.split())
                    ref_words = set(cleaned_ref.split())
                    if not gen_words or not ref_words:
                        return {"reward": 0.0, "log": "no_words_for_overlap"}

                    overlap = len(gen_words.intersection(ref_words))
                    total = len(gen_words.union(ref_words))
                    raw_score = float(overlap / total) if total > 0 else 0.0
            else:
                # Simple word overlap as fallback
                gen_words = set(cleaned_gen.split())
                ref_words = set(cleaned_ref.split())
                if not gen_words or not ref_words:
                    return {"reward": 0.0, "log": "no_words_for_overlap"}

                overlap = len(gen_words.intersection(ref_words))
                total = len(gen_words.union(ref_words))
                raw_score = float(overlap / total) if total > 0 else 0.0

            # Ensure score is in valid range
            score = float(np.clip(raw_score, 0.0, 1.0))

            # Apply verbosity penalty if configured
            if self.apply_verbosity_penalty:
                len_gen, len_ref = len(generated_answer), len(reference_answer)
                if len_gen > len_ref * 1.5:  # Penalize if >50% longer
                    # Use configurable penalty strength
                    penalty_factor = 1.0 - self.verbosity_penalty_strength * (len_gen / len_ref - 1.5)
                    penalty_factor = max(0.5, penalty_factor)  # Don't reduce by more than 50%
                    score *= penalty_factor
                    return {
                        "reward": score,
                        "log": f"raw_score={raw_score:.4f}, verbosity_penalty={penalty_factor:.4f}"
                    }

            return {"reward": score, "log": f"raw_score={raw_score:.4f}"}

        except Exception as e:
            logger.error(f"SemanticSimilarityReward failed: {e}", exc_info=True)
            return {"reward": 0.0, "log": f"exception_in_compute: {str(e)[:100]}"}

    def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, Any]]:
        """
        Computes rewards for a batch of contexts by calling compute sequentially.
        This is more robust than a complex batch implementation.

        Args:
            contexts: List of RewardContext objects to evaluate

        Returns:
            List of dictionaries containing reward scores and logs
        """
        results = []
        for context in contexts:
            try:
                # Get the raw result from compute
                result_dict = self.compute(context)

                # Extract the reward value
                raw_score = result_dict.get('reward', 0.0)

                # Apply smoothing
                smoothed_score = self._smooth_reward(raw_score)

                # Create properly formatted output with total key
                output = {
                    self.name: smoothed_score,
                    "total": smoothed_score,  # Add the total key
                    "log": result_dict.get('log', {})
                }

                results.append(output)
            except Exception as e:
                logger.error(f"Exception in batch_compute for context: {e}")
                results.append({
                    self.name: 0.0,
                    "total": 0.0,  # Include total key in error case too
                    "log": {"error": f"batch_exception: {str(e)[:100]}"}
                })

        return results
