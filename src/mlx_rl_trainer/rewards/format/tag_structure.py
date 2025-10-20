import re
from typing import Dict, Any, List
import logging
from mlx_rl_trainer.rewards.base_reward import BaseReward
from mlx_rl_trainer.rewards.registry import RewardRegistry
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)


def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extracts the text between the FIRST <think> and FIRST </think> tags.

    Args:
        text: The text to extract from
        gen_config: Configuration object containing tag definitions

    Returns:
        The extracted thinking region text, or empty string if not found
    """
    # Get tags from config with fallbacks
    start_tag = getattr(gen_config, 'think_start_tag', '<think>')
    end_tag = getattr(gen_config, 'think_end_tag', '</think>')

    # Handle edge cases
    if not text or not start_tag or not end_tag:
        return ""
    if text.endswith(start_tag) or text.startswith(end_tag):
        return ""

    # Extract the thinking region
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    """
    Extracts text that comes AFTER the LAST </think> tag.

    Args:
        text: The text to extract from
        gen_config: Configuration object containing tag definitions

    Returns:
        The extracted answer region text, or the original text if no tags found
    """
    # Get end tag from config with fallback
    end_tag = getattr(gen_config, 'think_end_tag', '</think>')
    start_tag = getattr(gen_config, 'think_start_tag', '<think>')

    # Handle edge cases
    if not text:
        return ""
    if text.endswith(start_tag) or text.startswith(end_tag):
        return ""
    if not end_tag:
        return text.strip()

    # Find the last end tag and extract everything after it
    last_idx = text.lower().rfind(end_tag.lower())
    if last_idx != -1:
        answer_text = text[last_idx + len(end_tag):].strip()
        return answer_text.lstrip("\n").strip()
    return text.strip()


@RewardRegistry.register("format_structure")
class TagStructureReward(BaseReward):
    """
    Rewards the model for adhering to <think>...</think> structure
    followed by direct answer text (no answer tags).

    Encourages concise, compressed thinking by penalizing verbosity.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_think_length = config.get("min_think_length", 20)
        self.min_answer_length = config.get("min_answer_length", 15)

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
                    think_length_target_min=100,
                    think_length_target_max=250,
                    think_length_penalty_strength=0.5
                )
                logger.debug("Created default GenerationConfig")
        except Exception as e:
            logger.warning(f"Could not load GenerationConfig: {e}. Using default values.")
            # Create a minimal config with default values as a fallback
            self.gen_config = type('GenerationConfig', (), {
                'think_length_target_min': 100,
                'think_length_target_max': 250,
                'think_length_penalty_strength': 0.5,
                'think_start_tag': '<think>',
                'think_end_tag': '</think>'
            })()

        # Optimal think length range (from GenerationConfig)
        self.think_target_min = config.get(
            "think_length_target_min", self.gen_config.think_length_target_min
        )
        self.think_target_max = config.get(
            "think_length_target_max", self.gen_config.think_length_target_max
        )

        # Penalty strength for length deviation
        self.length_penalty_strength = config.get(
            "length_penalty_strength", self.gen_config.think_length_penalty_strength
        )

        # Verbosity penalty multiplier (how much to penalize excessive length)
        self.verbosity_penalty_factor = config.get("verbosity_penalty_factor", 2.0)

        # Debug logging flag
        self.debug_logging = config.get("debug_logging", True)

        # Store tag values for reuse
        self.start_tag = self.gen_config.think_start_tag
        self.end_tag = self.gen_config.think_end_tag

    def _compute_length_score(self, think_length: int) -> float:
        """
        Compute a score based on think length relative to target range.

        Scoring philosophy:
        - Optimal range (target_min to target_max): 1.0
        - Too short (below target_min): Gradually decrease
        - Too long (above target_max): More aggressive penalty (verbosity is worse than brevity)

        Args:
            think_length: Character count of thinking section

        Returns:
            Score multiplier between 0.0 and 1.0
        """
        if self.think_target_min <= think_length <= self.think_target_max:
            # Perfect length - in the sweet spot
            return 1.0

        if think_length < self.think_target_min:
            # Too short - linear penalty
            if think_length < self.min_think_length:
                return 0.0  # Way too short

            # Scale from min_think_length to target_min
            range_diff = self.think_target_min - self.min_think_length
            if range_diff <= 0:
                return 0.5  # Neutral if min/target are equal or inverted

            ratio = (think_length - self.min_think_length) / range_diff
            return 0.5 + (0.5 * max(0.0, min(1.0, ratio)))  # Range: 0.5 to 1.0

        else:
            # Too long - exponential penalty (verbosity is bad!)
            excess = think_length - self.think_target_max
            penalty_range = self.think_target_max if self.think_target_max > 0 else 100

            normalized_excess = excess / penalty_range
            penalty = (
                normalized_excess
                * self.length_penalty_strength
                * self.verbosity_penalty_factor
            )

            # Exponential decay for severe verbosity
            score = max(0.0, 1.0 - penalty)
            return score

    def compute(self, context: RewardContext) -> Dict[str, Any]:
        """
        Computes the format structure reward and returns a dictionary.

        Analyzes the structure of the generated text, checking for proper thinking tags
        and content length in both thinking and answer sections.

        Args:
            context: The RewardContext containing the generated text to evaluate

        Returns:
            A dictionary with at least a 'reward' key containing a score between 0.0 and 1.0,
            and a 'log' key with detailed information about the evaluation
        """
        try:
            # Validate input context
            if not context or not hasattr(context, 'generated_text'):
                return {"reward": 0.0, "log": {"error": "Invalid context object"}}

            generated = context.generated_text or ""
            log_data = {}

            # Handle empty or very short text
            if not generated or len(generated.strip()) < 10:
                return {"reward": 0.0, "log": {"error": "Empty or too short generation"}}

            # Use the stored tags instead of creating a new GenerationConfig
            start_tag = self.start_tag
            end_tag = self.end_tag

            # Count tag occurrences
            try:
                th_s = len(re.findall(re.escape(start_tag), generated, flags=re.I))
                th_e = len(re.findall(re.escape(end_tag), generated, flags=re.I))
            except Exception as e:
                logger.warning(f"Error counting tags: {e}")
                th_s = 0
                th_e = 0

            # Extract text regions with error handling
            try:
                think_text = extract_think_region(generated, self.gen_config)
                answer_text = extract_answer_region(generated, self.gen_config)
                think_len = len(think_text.strip())
                answer_len = len(answer_text.strip())
            except Exception as e:
                logger.warning(f"Error extracting text regions: {e}")
                think_text = ""
                answer_text = ""
                think_len = 0
                answer_len = 0

            # Log data for debugging
            log_data = {"start_tags": th_s, "end_tags": th_e, "think_len": think_len, "answer_len": answer_len}
            if self.debug_logging:
                logger.info(f"TagStructure | {log_data}")

            # Calculate score based on structure
            score = 0.0
            if th_s == 1 and th_e == 1:
                if think_len >= self.min_think_length and answer_len >= self.min_answer_length:
                    try:
                        length_score = self._compute_length_score(think_len)
                        score = 1.0 * length_score
                        log_data["reason"] = "Perfect structure"
                        log_data["length_score"] = length_score
                    except Exception as e:
                        logger.warning(f"Error computing length score: {e}")
                        score = 0.8  # Fallback to a reasonable score
                        log_data["reason"] = "Perfect structure (score calculation error)"
                elif think_len >= self.min_think_length or answer_len >= self.min_answer_length:
                    score = 0.6
                    log_data["reason"] = "Partial content"
                else:
                    score = 0.3
                    log_data["reason"] = "Empty content"
            elif th_s >= 1 and th_e == 0:
                score = 0.3
                log_data["reason"] = "Incomplete think block"
            elif th_s > 1 or th_e > 1:
                score = 0.2
                log_data["reason"] = "Multiple tags"
            elif th_s == 0 and th_e == 0:
                score = 0.1 if len(generated.strip()) > 30 else 0.0
                log_data["reason"] = "No tags"
            else:
                score = 0.2
                log_data["reason"] = "Fallback case"

            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            log_data["final_score"] = score
            return {"reward": score, "log": log_data}

        except Exception as e:
            # Catch-all exception handler for robustness
            logger.error(f"TagStructureReward compute error: {e}", exc_info=True)
            return {"reward": 0.0, "log": {"error": f"Exception: {str(e)[:100]}"}}

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
