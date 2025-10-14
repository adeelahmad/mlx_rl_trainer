# ============================================================================
# UPDATED thinking_quality.py REWARD FILE
# ============================================================================
# Replace your existing ./reasoning/thinking_quality.py with this:

"""
Enhanced Thinking Quality Reward with Hardware-Constrained Length Penalties

Optimized for M2 MacBook with 96GB RAM and 128 token generation limit.
Prevents endless thinking rambling while maintaining reasoning quality.
"""

import re
from typing import Dict, Any, Optional
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import register_reward
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.utils.text_utils import extract_think_region
from mlx_rl_trainer.core.config import GenerationConfig
import logging

logger = logging.getLogger(__name__)


@register_reward('thinking_quality')
class ThinkingQualityReward(BaseReward):
    """
    Evaluates thinking section quality with adaptive length penalties.

    Features:
    - Quality scoring (tags, structure, clarity)
    - Length penalties for excessive thinking (critical for 128 token budget)
    - Conciseness bonuses for optimal length
    - Integration with trainer thinking limits
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Length targets (adjusted for 128 token budget)
        self.target_length_min = config.get('target_length_min', 30)
        self.target_length_max = config.get('target_length_max', 80)
        self.optimal_length_min = config.get('optimal_length_min', 40)
        self.optimal_length_max = config.get('optimal_length_max', 60)

        # Penalties and bonuses
        self.conciseness_bonus = config.get('conciseness_bonus', 0.15)
        self.excessive_length_threshold = config.get('excessive_length_threshold', 90)
        self.excessive_length_penalty = config.get('excessive_length_penalty', 0.5)

        # Use trainer limits if available
        self.use_trainer_limits = config.get('use_trainer_thinking_limits', True)

        # Special tokens to penalize
        self.special_tokens = config.get('special_tokens', [
            '<|endoftext|>', '<|im_start|>', '<think><think>',
            '<|im_end|>', '<|end|>', '<|begin|>',
            '<|system|>', '<|user|>', '<|assistant|>',
            '[INST]', '[/INST]', '<s>', '</s>',
            '<pad>', '<unk>', '<bos>', '<eos>'
        ])
        self.special_token_penalty = config.get('special_token_penalty', 0.4)

        # Bad phrases indicating poor reasoning
        self.bad_phrases = config.get('bad_phrases', [
            'i think', 'i believe', 'maybe', "i'm not sure",
            'i will now', "i'll start by", "let's see",
            'confused', 'stuck', 'frustrated',
            'wait, wait', 'hmm, perhaps', 'or wait',
            'to be completely honest', 'basically what happens',
            'long story short', 'at the end of the day',
            'circular reasoning', 'insufficient information',
            'too complicated', 'for some unknown reason'
        ])

        self.tag_misuse_penalty = config.get('tag_misuse_penalty', 0.3)

        logger.info(
            f"ThinkingQualityReward initialized: "
            f"target_length=[{self.target_length_min}, {self.target_length_max}], "
            f"optimal=[{self.optimal_length_min}, {self.optimal_length_max}], "
            f"excessive_threshold={self.excessive_length_threshold}"
        )

    def _check_tag_misuse_penalty(self, text: str, gen_config: GenerationConfig) -> float:
        """Check for tag misuse (duplicate tags, nested tags)."""
        start_tag = gen_config.think_start_tag
        end_tag = gen_config.think_end_tag

        if not start_tag or not end_tag:
            return 0.0

        start_count = len(re.findall(re.escape(start_tag), text, flags=re.I))
        end_count = len(re.findall(re.escape(end_tag), text, flags=re.I))

        penalty = 0.0

        # Multiple tags or mismatched counts
        if start_count > 1 or end_count > 1 or abs(start_count - end_count) > 1:
            penalty = self.tag_misuse_penalty

        # Nested tags within thinking region
        if start_count == 1 and end_count == 1:
            think_content = extract_think_region(text, gen_config)
            if re.search(r'<think>|<\/think>', think_content, flags=re.I):
                penalty = self.tag_misuse_penalty

        return penalty

    def _check_special_tokens_penalty(self, think_content: str) -> float:
        """Penalize presence of special tokens in thinking."""
        penalty = 0.0
        for token in self.special_tokens:
            if token in think_content:
                penalty += self.special_token_penalty
        return penalty

    def _compute_length_score(
        self,
        think_length: int,
        trainer_max_tokens: Optional[int] = None
    ) -> float:
        """
        Compute score based on thinking length with hardware constraints.

        Args:
            think_length: Actual thinking token count
            trainer_max_tokens: Optional max from trainer config (overrides)

        Returns:
            Score between 0.0 and 1.0 (can exceed 1.0 with bonus)
        """
        # Use trainer limit if available and enabled
        if self.use_trainer_limits and trainer_max_tokens is not None:
            # Adjust thresholds based on trainer limit
            effective_max = min(self.target_length_max, trainer_max_tokens)
            effective_excessive = min(self.excessive_length_threshold, trainer_max_tokens + 10)
        else:
            effective_max = self.target_length_max
            effective_excessive = self.excessive_length_threshold

        score = 1.0

        # Too short penalty
        if think_length < self.target_length_min:
            # Linear scale from 0 to 1
            score = max(0.0, think_length / self.target_length_min)

        # Too long penalty
        elif think_length > effective_max:
            # Gradual penalty
            excess_ratio = (think_length - effective_max) / effective_max
            score = max(0.0, 1.0 - (excess_ratio * 0.5))

        # Optimal length bonus
        if self.optimal_length_min <= think_length <= self.optimal_length_max:
            score += self.conciseness_bonus

        # Excessive length harsh penalty (critical for 128 token budget!)
        if think_length > effective_excessive:
            excess_ratio = think_length / effective_excessive
            harsh_penalty = self.excessive_length_penalty * excess_ratio
            score -= harsh_penalty

            # Log warning for monitoring
            if think_length > trainer_max_tokens if trainer_max_tokens else effective_excessive:
                logger.warning(
                    f"Thinking length {think_length} exceeds limit "
                    f"({trainer_max_tokens or effective_excessive}), "
                    f"penalty={harsh_penalty:.3f}"
                )

        return max(0.0, score)

    def compute(self, context: RewardContext) -> float:
        """Compute thinking quality reward with length constraints."""
        text = context.generated_text

        if not text or len(text.strip()) < 10:
            return 0.0

        gen_config = GenerationConfig()
        think_content = extract_think_region(text, gen_config)

        if not think_content:
            return 0.0

        # Base score
        score = 1.0

        # Tag misuse penalty
        tag_penalty = self._check_tag_misuse_penalty(text, gen_config)
        score -= tag_penalty

        # Special tokens penalty
        special_token_penalty = self._check_special_tokens_penalty(think_content)
        score -= special_token_penalty

        # Length scoring with trainer limits
        think_length = len(think_content.strip())

        # Try to get trainer max_thinking_tokens from metadata
        trainer_max_tokens = context.metadata.get('max_thinking_tokens')

        length_score = self._compute_length_score(think_length, trainer_max_tokens)
        score *= length_score

        # Structure bonus (lists, bullet points)
        if re.search(r'(\n\s*[-*•]|\n\s*\d+\.\s+)', think_content):
            score += 0.1

        # Bad phrases penalty
        think_lower = think_content.lower()
        for phrase in self.bad_phrases:
            if phrase in think_lower:
                score -= 0.15  # Reduced from 0.2 to allow some natural language

        # Final clipping
        final_score = max(0.0, min(1.0, score))

        # Periodic logging for monitoring
        if context.update_step % 50 == 0:
            logger.debug(
                f"ThinkingQuality: length={think_length}, "
                f"length_score={length_score:.3f}, "
                f"tag_penalty={tag_penalty:.3f}, "
                f"final={final_score:.3f}"
            )

        return final_score
