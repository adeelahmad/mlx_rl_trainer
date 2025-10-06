# file_path: mlx_rl_trainer/src/mlx_rl_trainer/utils/text_utils.py
# revision_no: 003
# goals_of_writing_code_block: Provide a complete and correct version of text_utils.py to resolve the persistent ImportError, including _extract_final_numeric.
# type_of_code_response: replace code
"""Text processing utility functions."""

import logging
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from functools import lru_cache  # For caching static config

from mlx_lm.tokenizer_utils import TokenizerWrapper

# Use a custom name here to prevent naming conflict with local uses of 'RewardConfig'
from mlx_rl_trainer.core.config import RewardConfig as StaticRewardConfig

logger = logging.getLogger(__name__)

# --- Global Text Constants ---
LETTER_ALPH = string.ascii_uppercase

# --- Static Configuration for Text Rules ---


@lru_cache(maxsize=1)
def _get_static_reward_config() -> StaticRewardConfig:
    """Provides a static RewardConfig instance for accessing default tags."""
    # Since we cannot import the full ExperimentConfig here, we instantiate the base wrapper
    return StaticRewardConfig(name="static_config")


# --- Text Manipulation ---


def _preview(s: str, n: int = 600) -> str:
    """Shortens text for logs and escapes newlines."""
    if s is None:
        return ""
    s = s.replace("\r\n", "\n")
    s = s[:n] + ("..." if len(s) > n else "")
    return s.replace("\n", "\\n")


_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]+`")
_HTML_TAGS = re.compile(r"<[^>]+>")


def _strip_markup(s: str) -> str:
    """Removes common markdown, code fences, and HTML tags for cleaner text analysis."""
    if not s:
        return ""
    s = str(s)
    s = _CODE_FENCE.sub(" ", s)
    s = _INLINE_CODE.sub(" ", s)
    s = _MD_HEADER.sub(" ", s)
    s = _HTML_TAGS.sub(" ", s)
    s = re.sub(r"(^|\n)\s*[-*•]\s+", r"\1", s)
    s = re.sub(r"(^|\n)\s*\d+\.\s+", r"\1", s)
    s = s.replace("\u2026", " ")
    s = re.sub(
        r"[^\w\s/:%\-.]", " ", s
    )  # Remove most symbols, keep space/numbers/letters/some punctuation
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _count_words(txt: str) -> int:
    """Counts non-whitespace tokens (approximates word count)."""
    return len(re.findall(r"\w+", txt or ""))


def _tokenize_set(s: str) -> Set[str]:
    """Convert string to set of lowercase tokens without punctuation"""
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return set(w for w in s.split() if w)


def _normalize_ans_for_match(s: str) -> str:
    """Normalizes an answer string for case-insensitive, whitespace-insensitive comparison."""
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:")
    return s


def _contains_keywords(haystack: str, keywords: Sequence[str]) -> bool:
    """Return True if any keyword occurs in s (case-insensitive)"""
    if not haystack or not keywords:
        return False
    s_low = haystack.lower()
    return any(k.lower() in s_low for k in keywords)


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Check if needle phrase appears in haystack with some tolerance."""
    if not haystack or not needle:
        return False
    haystack_lower = haystack.lower()
    needle_lower = needle.lower()
    if needle_lower in haystack_lower:
        return True

    # Fallback: check if first two significant tokens appear
    toks = [t for t in needle_lower.split() if len(t) >= 3]
    if len(toks) >= 2:
        short = " ".join(toks[:2])
        return short in haystack_lower
    return False


def _extract_action_phrases(s: str, min_len: int = 3) -> List[str]:
    """Identifies potential steps/actionable phrases from text, usually from structured lists."""
    if not s:
        return []

    # Look for bullet points/numbered lists
    bullets = re.findall(r"(^|\n)\s*(?:[-*•]|\d+\.)\s+(.*?)(?:\n|$)", s, re.S)
    items = [b[1].strip() for b in bullets if b[1].strip()]

    # If no lists, try splitting by sentence/phrase markers
    if not items:
        # Filter out short, non-informative segments
        items = [it for it in re.split(r"[;.\n]+", s) if _count_words(it) > 3]

    out = []
    for it in items:
        itn = _strip_markup(it)
        if itn and _count_words(itn) >= min_len:
            out.append(itn)

    # Deduplicate while preserving order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    return uniq


def _extract_python_code(text: str) -> str:
    """Extracts Python code from a markdown code block or assumes plain code."""
    # Look for markdown code blocks
    matches = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[0]

    # If no markdown block, try to parse the entire text as Python code
    try:
        import ast

        ast.parse(text)  # Check for valid Python syntax
        return text.strip()
    except (
        SyntaxError,
        ImportError,
    ):  # Catch ImportError if ast fails for some reason
        return ""  # Not valid Python code or ast unavailable
    except Exception as e:
        logger.debug(f"Unexpected error during AST parsing: {e}")
        return ""


def _extract_final_numeric(s: str) -> Optional[float]:
    """
    Extracts the last numeric value (int or float) from a string.
    Crucial for mathematical reasoning benchmarks like GSM8K.
    """
    if not s:
        return None
    # Regex to find floating point numbers or integers, including optional leading sign
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    if not numbers:
        return None
    try:
        # Return the last number found
        return float(numbers[-1])
    except (ValueError, IndexError):
        return None


def _extract_predicted_letters(
    generated_text: str, options: List[str], reward_config: Any
) -> List[str]:
    """
    Extracts predicted MCQ answer letters from generated text.
    Prioritizes explicit tags, then looks for patterns.
    """
    if not generated_text or not options:
        return []

    # 1. Try to extract from explicit answer tags if present
    answer_region = extract_answer_region(generated_text, reward_config)
    if answer_region:
        # Look for patterns like "Answer: A", "A, B", "The answer is C."
        matches = re.findall(r"[A-Z]", answer_region.upper())
        if matches:
            return list(set(matches))  # Return unique letters

    # 2. Fallback: Look for patterns in the entire generated text
    matches = re.findall(r"[A-Z]", generated_text.upper())
    if matches:
        return list(set(matches))

    return []


# --- Region Extraction for Rewards ---


def extract_think_region(text: str, reward_config: StaticRewardConfig) -> str:
    """Extracts content between think_start_tag and think_end_tag."""
    if not reward_config.think_start_tag or not reward_config.think_end_tag:
        return ""
    m = re.search(
        re.escape(reward_config.think_start_tag)
        + r"\s*(.*?)\s*"
        + re.escape(reward_config.think_end_tag),
        text or "",
        flags=re.I | re.S,
    )
    return (m.group(1).strip() if m else "")[:8000]


def extract_answer_region(text: str, reward_config: StaticRewardConfig) -> str:
    """Extracts answer region: everything AFTER the last </think> tag. If no </think> tag, return full text."""
    tl = text or ""
    tend = reward_config.think_end_tag

    if tend and tend.lower() in tl.lower():
        idx = tl.lower().rfind(tend.lower())
        answer = tl[idx + len(tend) :].strip()
        return answer[:2000]

    return tl.strip()[:2000]


def _extract_think_answer_lengths(
    text: str,
    think_start_tag: str,
    think_end_tag: str,
    answer_start_tag: str,
    answer_end_tag: str,
) -> Tuple[int, int]:
    """Extracts character lengths of thinking and answer sections from text."""
    try:
        # Create a mock config object to pass to extract_region functions
        cfg = type(
            "MockCfg",
            (object,),
            {
                "think_start_tag": think_start_tag,
                "think_end_tag": think_end_tag,
                "answer_start_tag": answer_start_tag,
                "answer_end_tag": answer_end_tag,
            },
        )()
        think_content = extract_think_region(text, cfg)
        answer_content = extract_answer_region(text, cfg)
        return len(think_content.strip()), len(answer_content.strip())
    except Exception as e:
        logger.debug(f"Failed to extract think/answer lengths: {e}")
        return 0, 0


# --- MCQ Helpers ---


def _indices_to_letters(indices: List[int]) -> str:
    """Converts a list of 0-based indices to comma-separated letters (e.g., [0, 2] -> 'A,C')."""
    letters = []
    for idx in indices:
        if 0 <= idx < len(LETTER_ALPH):
            letters.append(LETTER_ALPH[idx])

    seen, out = set(), []
    for L in letters:
        if L not in seen:
            seen.add(L)
            out.append(L)
    return ",".join(out)


def _letters_to_canonical(letter_str: str) -> str:
    """Converts a string of letters (e.g., 'a, B ,d') to canonical uppercase form ('A,B,D')."""
    parts = []
    for p in (letter_str or "").split(","):
        p = p.strip().upper()
        if len(p) == 1 and p in LETTER_ALPH:
            parts.append(p)

    seen, out = set(), []
    for L in parts:
        if L not in seen:
            seen.add(L)
            out.append(L)
    return ",".join(out)


def _match_ref_to_option_index(ref_text: str, options: List[str]) -> Optional[int]:
    """Tries to match a reference answer string to one of the provided options."""
    if not (ref_text and options):
        return None
    ref_n = _normalize_ans_for_match(ref_text)

    # 1. Exact normalized match
    for idx, opt in enumerate(options):
        if _normalize_ans_for_match(opt) == ref_n:
            return idx

    # 2. Substring or overlap match (less strict)
    for idx, opt in enumerate(options):
        on = _normalize_ans_for_match(opt)
        if ref_n and (ref_n in on or on in ref_n):
            return idx

    return None


def _extract_mcq_options(prompt_text: str) -> List[str]:
    """Tries to extract numbered or bulleted MCQ options from a prompt."""
    if not isinstance(prompt_text, str):
        return []

    # Look for a 'Choices:' block
    m = re.search(r"choices\s*:?(.*)$", prompt_text, flags=re.I | re.S)
    block = m.group(1) if m else prompt_text
    lines = [ln.strip() for ln in block.splitlines()]
    opts = []

    for ln in lines:
        if re.match(r"^\s*[-•]\s+", ln):  # Bullet points
            opts.append(re.sub(r"^\s*[-•]\s+", "", ln).strip())
            continue

        m2 = re.match(r"^\s*([A-Za-z])\s*[\)\.\-:]\s*(.+)$", ln)  # A) Option, B. Option
        if m2:
            opts.append(m2.group(2).strip())
            continue

        m3 = re.match(r"^\s*\d+\s*[\)\.\-:]\s*(.+)$", ln)  # 1) Option
        if m3:
            opts.append(m3.group(1).strip())
            continue

    # Limit options to prevent overly large metadata or impossible MCQs
    return [o for o in opts if o.strip()][: len(LETTER_ALPH)]


def _infer_gold_letters_from_meta(
    meta: Dict[str, Any], options: List[str], fallback_ref_text: str = ""
) -> str:
    """Generates canonical gold letters (e.g., "A,C") from various metadata fields."""

    # 1. Try explicit letter fields
    for key in ("correct_answer_letters", "correct_letters", "correct_letter"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return _letters_to_canonical(v)

    # 2. Try index fields
    correct_indices = []
    if isinstance(meta.get("correct_indices"), list) and meta["correct_indices"]:
        try:
            correct_indices = [
                int(i) for i in meta["correct_indices"] if isinstance(i, (int, float))
            ]
        except Exception:
            pass
    elif isinstance(meta.get("correct_index"), (int, float)):
        correct_indices = [int(meta["correct_index"])]

    if correct_indices:
        return _indices_to_letters(
            [i for i in correct_indices if 0 <= i < len(options)]
        )

    # 3. Try text fields, matching against options
    correct_texts = meta.get("correct_texts")
    if isinstance(correct_texts, list) and correct_texts:
        for t in correct_texts:
            idx = _match_ref_to_option_index(str(t), options)
            if idx is not None:
                correct_indices.append(idx)

    if correct_indices:
        return _indices_to_letters(
            [i for i in correct_indices if 0 <= i < len(options)]
        )

    # 4. Fallback: match reference text against options
    if fallback_ref_text:
        idx = _match_ref_to_option_index(fallback_ref_text, options)
        if idx is not None:
            return _indices_to_letters([idx])

    return ""


def _mcq_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Generates comprehensive metadata for a sample based on inferred options/answers."""
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    options = meta.get("options") if isinstance(meta.get("options"), list) else []
    ref_ans = sample.get("ref_answer_str") or sample.get("completion") or ""

    is_mcq = False
    if isinstance(meta.get("type"), str) and meta["type"].strip().lower() == "mcq":
        is_mcq = True

    if not options:
        options = _extract_mcq_options(sample.get("prompt") or sample.get("text") or "")

    options = [str(o).strip() for o in options if str(o).strip()]
    if len(options) < 2:
        return {
            "is_mcq": False,
            "options": [],
            "multi_select": False,
            "correct_indices": [],
            "correct_letters": "",
        }

    if not is_mcq:
        is_mcq = True

    correct_indices = []
    multi_select = bool(meta.get("multi_select", False))

    # Delegate gold letter inference
    correct_letters = _infer_gold_letters_from_meta(
        meta, options, fallback_ref_text=ref_ans
    )

    if correct_letters:
        correct_indices = [
            LETTER_ALPH.index(L) for L in correct_letters.split(",") if L in LETTER_ALPH
        ]

    if len(correct_indices) > 1:
        multi_select = True

    return {
        "is_mcq": is_mcq,
        "options": options,
        "multi_select": multi_select,
        "correct_indices": correct_indices,
        "correct_letters": correct_letters,
    }


def apply_chat_template_wrapper(
    tokenizer: TokenizerWrapper, prompt: str, system_prompt: Optional[str]
) -> str:
    """Applies a chat template to a prompt, handling potential errors gracefully."""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(
            f"apply_chat_template failed: {e}. Falling back to manual formatting."
        )
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt else ""
        return f"{prefix}User: {prompt.strip()}\n\nAssistant:"


# --- TF-IDF Utility (for rewards) ---
def _tfidf_cosine(a: str, b: str) -> float:
    """Compute TF-IDF cosine similarity, fallback to Jaccard."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(
            min_df=1, max_df=0.9, ngram_range=(1, 2), stop_words="english"
        )
        X = vec.fit_transform([_strip_markup(a), _strip_markup(b)])
        sim = float(cosine_similarity(X[0:1], X[1:2])[0, 0])
        return max(0.0, min(1.0, sim))
    except Exception:
        A, B = _tokenize_set(a), _tokenize_set(b)
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)


import re
import json
from typing import List, Sequence, Dict, Any, Union


def _strip_specials(s: str) -> str:
    """Remove special unwanted characters from text."""
    return s.strip()


class TwoBlockFormatter:
    def __init__(
        self,
        think_start: str,
        think_end: str,
        answer_start: str,
        answer_end: str,
        validate_json: bool = False,
    ):
        self.ts = think_start
        self.te = think_end
        self.as_ = answer_start
        self.ae = answer_end
        self.validate_json = validate_json

        # Build regex patterns, handling empty strings
        if self.ts and self.te:
            self._re_think = re.compile(
                re.escape(self.ts) + r"(.*?)" + re.escape(self.te), re.DOTALL
            )
        else:
            self._re_think = None

        if self.as_ and self.ae:
            self._re_answer = re.compile(
                re.escape(self.as_) + r"(.*?)" + re.escape(self.ae), re.DOTALL
            )
        else:
            self._re_answer = None

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks.
        Strips ```json or ``` blocks if present.
        """
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            else:
                text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _looks_like_json(self, text: str) -> bool:
        """Check if text looks like it's trying to be JSON."""
        text = text.strip()

        # Remove markdown code blocks first
        text = self._extract_json_from_text(text)

        # Check if it starts with JSON indicators
        return text.startswith("{") or text.startswith("[")

    def _validate_json_content(self, content: str) -> float:
        """
        Validate if content is valid JSON.
        Returns 0.5 for valid JSON, 0.0 for invalid.
        """
        if not content or content == "Insufficient information.":
            return 0.0

        json_text = self._extract_json_from_text(content)

        try:
            json.loads(json_text)
            return 0.5
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def _coerce_one(self, s: str) -> str:
        if s is None:
            s = ""

        s = s.strip()

        if not s:
            return f"{self.ts}…{self.te}\n{self.as_}Insufficient information.{self.ae}"

        i_ts = s.find(self.ts) if self.ts else -1

        if i_ts == -1:
            ans = _strip_specials(s or "Insufficient information.")
            if self.te:
                ans = ans.replace(self.te, "")
            return f"{self.ts}…{self.te}\n{self.as_}{ans}{self.ae}"

        s = s[i_ts:]
        i_te = s.find(self.te, len(self.ts)) if self.te else -1

        if i_te == -1:
            s = s + self.te
            i_te = len(s) - len(self.te)

        think_body = s[len(self.ts) : i_te].strip()
        if not think_body:
            think_body = "…"
        think_body = _strip_specials(think_body)

        remainder = s[i_te + len(self.te) :].strip()

        if self.as_ and self.ae:
            i_as = remainder.find(self.as_)
            if i_as != -1:
                remainder = remainder[i_as + len(self.as_) :]
                i_ae = remainder.find(self.ae)

                if i_ae != -1:
                    answer_body = remainder[:i_ae].strip()
                else:
                    answer_body = remainder.strip()
            else:
                answer_body = remainder
        elif self.as_ and not self.ae:
            i_as = remainder.find(self.as_)
            if i_as != -1:
                answer_body = remainder[i_as + len(self.as_) :].strip()
            else:
                answer_body = remainder
        else:
            answer_body = remainder

        answer_body = _strip_specials(answer_body)
        if self.te:
            answer_body = answer_body.replace(self.te, "")

        if not answer_body:
            answer_body = "Insufficient information.Way too much overthinking"

        final = f"{self.ts}{think_body}{self.te}\n{self.as_}{answer_body}{self.ae}"

        if self.ae:
            j_ae = final.rfind(self.ae)
            if j_ae != -1:
                final = final[: j_ae + len(self.ae)]

        return final

    def _score_one(
        self, s: str, detailed: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Score how well-formatted the text is with granular partial credit.

        Special case: If output looks like JSON (starts with { or [), no think block required!

        Normal format: <think>COT</think>\nAnswer
        - Answer tags are OPTIONAL
        - Everything after </think> is the answer

        Format scoring rubric:
        - 0.0: No think block and doesn't look like JSON
        - 0.1: Think block exists but not at the start
        - 0.2: Starts with think tag but not properly closed OR has nested/duplicate tags
        - 0.3: Complete think block but empty content
        - 0.4: Complete think block with content but missing/empty answer
        - 0.5: Perfect format OR valid JSON output (no think needed)
        """
        result = {"reward_format": 0.0, "reward_content": 0.0, "reward_total": 0.0}

        if not s:
            return result if detailed else 0.0

        s = s.strip()
        answer_content = ""

        # SPECIAL CASE: If it looks like JSON, format is perfect even without think!
        if self._looks_like_json(s):
            result["reward_format"] = 0.5
            answer_content = s

            # Validate the JSON if enabled
            if self.validate_json:
                result["reward_content"] = self._validate_json_content(answer_content)
                result["reward_total"] = (
                    result["reward_format"] * result["reward_content"]
                )
            else:
                result["reward_total"] = result["reward_format"]

            return result if detailed else result["reward_total"]

        # Otherwise, require think block format
        if self._re_think:
            m_think = self._re_think.search(s)
            if not m_think:
                # No complete think block found
                if self.ts and self.ts in s:
                    if s.startswith(self.ts):
                        result["reward_format"] = 0.2  # Starts but not closed
                    else:
                        result["reward_format"] = 0.1  # Has tag but not at start
                else:
                    result["reward_format"] = 0.0  # No think block

                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            # Must start at the beginning
            if not s.startswith(self.ts):
                result["reward_format"] = 0.1
                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            # Check think body for nested tags
            think_body = (m_think.group(1) or "").strip()

            if self.ts and self.ts in think_body:
                # Nested think start tag - malformed!
                result["reward_format"] = 0.2
                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            if self.te and self.te in think_body:
                # Nested think end tag - malformed!
                result["reward_format"] = 0.2
                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            # Think body must have content
            if not think_body:
                result["reward_format"] = 0.3
                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            # Extract everything after </think> - this is the answer
            after_think = s[m_think.end() :].strip()

            # Answer tags are OPTIONAL
            # We just need SOME content after </think>
            if not after_think:
                result["reward_format"] = 0.4  # No answer content
                result["reward_total"] = result["reward_format"]
                return result if detailed else result["reward_total"]

            # If answer tags are provided, extract content from within them
            if self.as_ and self.ae and after_think.startswith(self.as_):
                if self._re_answer:
                    m_ans = self._re_answer.search(after_think)
                    if m_ans:
                        ans_body = (m_ans.group(1) or "").strip()
                        if not ans_body:
                            result["reward_format"] = 0.4  # Empty answer
                            result["reward_total"] = result["reward_format"]
                            return result if detailed else result["reward_total"]

                        # Check for nested tags in answer
                        if self.ts in ans_body or self.te in ans_body:
                            result["reward_format"] = 0.2  # Nested tags
                            result["reward_total"] = result["reward_format"]
                            return result if detailed else result["reward_total"]

                        answer_content = ans_body

                        # Check for trailing junk after </answer>
                        tail = after_think[m_ans.end() :].strip()
                        if tail:
                            result["reward_format"] = 0.4  # Extra content
                            result["reward_total"] = result["reward_format"]
                            return result if detailed else result["reward_total"]
                    else:
                        # Has <answer> but not closed
                        result["reward_format"] = 0.4
                        result["reward_total"] = result["reward_format"]
                        return result if detailed else result["reward_total"]
            else:
                # No answer tags or doesn't start with answer tag
                # Just use whatever content is after </think>
                answer_content = after_think

                # Check for nested think tags in the answer content
                if self.ts in answer_content or self.te in answer_content:
                    result["reward_format"] = 0.2  # Nested tags
                    result["reward_total"] = result["reward_format"]
                    return result if detailed else result["reward_total"]

            # Perfect format!
            result["reward_format"] = 0.5
        else:
            result["reward_total"] = 0.0
            return result if detailed else 0.0

        # Validate JSON content if enabled
        if self.validate_json and answer_content:
            result["reward_content"] = self._validate_json_content(answer_content)
            result["reward_total"] = result["reward_format"] * result["reward_content"]
        else:
            result["reward_total"] = result["reward_format"]

        return result if detailed else result["reward_total"]

    def coerce_batch(self, texts: Sequence[str]) -> List[str]:
        """Coerce a batch of texts to proper format."""
        return [self._coerce_one(t) for t in texts]

    def score_batch(
        self, texts: Sequence[str], detailed: bool = False
    ) -> Union[List[float], List[Dict[str, float]]]:
        """Score a batch of texts for formatting quality."""
        return [self._score_one(t, detailed=detailed) for t in texts]


def _ensure_schedule_dict(args):
    if not isinstance(args.lr_schedule_config, dict):
        args.lr_schedule_config = {}
    cfg = args.lr_schedule_config
    init_lr = float(args.learning_rate)
    total_steps = int(args.num_training_steps)
    warmup_steps = int(cfg.get("warmup", 16))
    decay_steps = max(total_steps - warmup_steps, 1)
    end_lr = max(init_lr * 0.1, 1e-7)
    cfg.setdefault("name", "cosine_decay")
    cfg.setdefault("arguments", [init_lr, decay_steps, end_lr])
    cfg.setdefault("warmup", warmup_steps)
    cfg.setdefault("warmup_init", min(init_lr, max(init_lr * 0.1, 1e-8)))


def _extract_predicted_letters(
    generated_text: str, options: Optional[List[str]], reward_config: Any
) -> List[str]:
    """
    Extracts predicted MCQ letter(s) from the generated answer region.
    Prioritizes single-letter choices, then tries to map text to options.
    """
    ans_region = extract_answer_region(generated_text, reward_config)

    # 1. Look for single, anchored capital letters
    matches = re.findall(r"\b([A-Z])\b", ans_region.strip().upper())
    if matches:
        return sorted(list(set(matches)))  # Return unique sorted letters found

    # 2. Fallback: Try to semantically match extracted answer text to options
    if options:
        normalized_ans = _normalize_ans_for_match(ans_region)
        matched_indices = []
        for idx, opt_text in enumerate(options):
            if normalized_ans == _normalize_ans_for_match(opt_text):
                matched_indices.append(idx)
            elif len(normalized_ans) > 5 and normalized_ans in _normalize_ans_for_match(
                opt_text
            ):
                matched_indices.append(idx)

        if matched_indices:
            return [
                _indices_to_letters([i]) for i in matched_indices
            ]  # Convert indices to letters

    return []
