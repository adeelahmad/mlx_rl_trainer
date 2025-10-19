# mlx_rl_trainer/rewards/content/answer_quality.py

"""
Answer Quality Reward - Penalizes meta-cognitive phrases in answers.

Ensures answers are direct and professional, not introspective or thinking-aloud.
Meta-cognitive phrases like "the user is asking" or "let me think" should only
appear in <think> sections, never in the actual answer.
"""

from typing import Dict, Any, List
import logging

from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


@RewardRegistry.register("answer_quality")
class AnswerQualityReward(BaseReward):
    """
    Penalizes meta-cognitive and thinking-style phrases in the answer section.

    Rewards direct, professional answers while penalizing:
    - Meta-commentary about the user ("the user is asking...")
    - Thinking aloud ("let me think...", "I need to recall...")
    - Filler words at start of answers ("okay,", "hmm,", "so,")
    - Self-referential planning ("first, I should...", "I will now...")

    Configuration:
        forbidden_phrases: List of phrases that should not appear in answers
        phrase_penalty: Penalty per forbidden phrase found (default: 0.2)
        max_penalty: Maximum total penalty (default: 1.0)
        case_sensitive: Whether to do case-sensitive matching (default: False)
        debug_logging: Enable detailed logging (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.forbidden_phrases = config.get(
            "forbidden_phrases",
            [
                # User meta-commentary
                "the user is asking",
                "the user wants",
                "the user might be",
                "the user could be",
                "the user seems",
                "they are asking",
                "they want to know",
                "you are asking",
                "you want to know",
                # Thinking aloud
                "let me think",
                "let me start",
                "let me recall",
                "let me consider",
                "let me break",
                "i need to recall",
                "i need to think",
                "i need to consider",
                "i should recall",
                "i should think",
                "i should consider",
                "i will recall",
                "i will think",
                "i will consider",
                # Planning/process description
                "first, i need",
                "first, i should",
                "first, i will",
                "first, let me",
                "okay, let me",
                "alright, let me",
                "so let me",
                # Filler starts (common in leaked thinking)
                "hmm,",
                "okay,",
                "alright,",
                "so,",
                "wait,",
                "hold on,",
                "well,",
                # Meta-cognitive verbs
                "i remember that",
                "i recall that",
                "i think that",
                "i believe that",
                "thinking about",
                "considering that",
                "recalling that",
                # Question analysis (belongs in thinking)
                "the question is about",
                "this question asks",
                "the problem is asking",
                "we need to find",
                "we need to determine",
                "we need to figure out",
                # Uncertainty markers (should be in thinking, not answer)
                "i'm not sure",
                "i'm unsure",
                "i don't know if",
                "maybe it's",
                "perhaps it's",
                # Process commentary
                "looking at this",
                "analyzing this",
                "breaking this down",
                "unpacking this",
            ],
        )

        self.phrase_penalty = config.get("phrase_penalty", 0.2)
        self.max_penalty = config.get("max_penalty", 1.0)
        self.case_sensitive = config.get("case_sensitive", False)
        self.debug_logging = config.get("debug_logging", True)

        logger.info(
            f"AnswerQualityReward initialized: "
            f"{len(self.forbidden_phrases)} forbidden phrases, "
            f"penalty={self.phrase_penalty}, "
            f"max_penalty={self.max_penalty}"
        )

    def _extract_answer_text(self, text: str) -> str:
        """
        Extract the answer section (text after </think> tag).

        Args:
            text: Full generated text

        Returns:
            Answer text, or empty string if no </think> tag found
        """
        if not text:
            return ""

        gen_config = GenerationConfig()
        end_tag = gen_config.think_end_tag

        if not end_tag:
            if self.debug_logging:
                logger.warning("No think_end_tag defined in GenerationConfig")
            return text.strip()

        if end_tag not in text:
            if self.debug_logging:
                logger.warning(f"No '{end_tag}' tag found in generated text")
            return ""

        parts = text.split(end_tag, 1)
        if len(parts) > 1:
            answer = parts[1].strip()

            # Strip common garbage characters that appear after </think>
            lines = answer.split("\n", 1)
            if len(lines) > 1 and len(lines[0]) <= 2:
                answer = lines[1].strip()
                if self.debug_logging:
                    logger.debug(f"Stripped leading garbage: '{lines[0]}'")

            return answer

        return ""

    def _find_violations(self, answer_text: str) -> List[Dict[str, Any]]:
        """
        Find all forbidden phrases in the answer text.

        Args:
            answer_text: The answer section to check

        Returns:
            List of violation dicts with 'phrase' and 'position' keys
        """
        if not answer_text:
            return []

        violations = []
        check_text = answer_text if self.case_sensitive else answer_text.lower()

        for phrase in self.forbidden_phrases:
            check_phrase = phrase if self.case_sensitive else phrase.lower()

            position = check_text.find(check_phrase)
            if position != -1:
                violations.append(
                    {
                        "phrase": phrase,
                        "position": position,
                        "context": answer_text[
                            max(0, position - 20) : position + len(phrase) + 20
                        ],
                    }
                )

        return violations

    def compute(self, context: RewardContext) -> float:
        """
        Compute answer quality reward.

        Returns:
            Score between 0.0 and 1.0:
            - 1.0 = Perfect answer (no forbidden phrases)
            - 0.0 = Answer filled with meta-cognitive phrases
        """
        generated_text = context.generated_text

        if not generated_text or len(generated_text.strip()) < 10:
            if self.debug_logging:
                logger.warning(
                    f"AnswerQuality: Empty or too short text "
                    f"(len={len(generated_text) if generated_text else 0})"
                )
            return 0.0

        # Extract answer section
        answer_text = self._extract_answer_text(generated_text)

        if not answer_text:
            if self.debug_logging:
                logger.warning(
                    "AnswerQuality: No answer section found "
                    "(no text after </think> tag)"
                )
            return 0.0

        if len(answer_text) < 5:
            if self.debug_logging:
                logger.warning(
                    f"AnswerQuality: Answer too short ({len(answer_text)} chars)"
                )
            return 0.0

        # Find violations
        violations = self._find_violations(answer_text)

        # Calculate penalty
        if not violations:
            if self.debug_logging:
                logger.info(
                    f"AnswerQuality: PASS - No forbidden phrases found "
                    f"(answer_len={len(answer_text)})"
                )
            return 1.0

        total_penalty = len(violations) * self.phrase_penalty
        total_penalty = min(total_penalty, self.max_penalty)

        final_score = max(0.0, 1.0 - total_penalty)

        if self.debug_logging:
            logger.warning(
                f"AnswerQuality: VIOLATIONS FOUND | "
                f"count={len(violations)}, "
                f"penalty={total_penalty:.3f}, "
                f"score={final_score:.3f}"
            )

            # Log first 3 violations with context
            for i, v in enumerate(violations[:3]):
                logger.warning(
                    f"  Violation {i+1}: '{v['phrase']}' at position {v['position']}"
                )
                logger.debug(f"    Context: ...{v['context']}...")

            if len(violations) > 3:
                logger.warning(f"  ... and {len(violations) - 3} more violations")

        return final_score
