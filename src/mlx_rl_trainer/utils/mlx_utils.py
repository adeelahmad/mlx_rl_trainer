"""Text processing utility functions."""

import gc
import re
import random
import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from pathlib import Path

import logging
import re
import string
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Callable, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import mlx.nn as nn
from mlx_rl_trainer.core.config import (
    GenerationConfig,
    ExperimentConfig,
)  # Import GenerationConfig for tags
import mlx.optimizers as optim
logger = logging.getLogger(__name__)


import logging
import re
import string
import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Callable, Union

from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.nn as nn
from mlx_rl_trainer.core.config import GenerationConfig

logger = logging.getLogger(__name__)

LETTER_ALPH = string.ascii_uppercase


# --- MLX Global Config & Constants ---
TARGET_FLOAT_DTYPE = mx.bfloat16
MIN_REQUIRED_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
SAVE_ON_EXIT_FLAG_PATH = Path(".save_on_exit_request")
LETTER_ALPH = string.ascii_uppercase


def _preview(s: str, n: int = 600) -> str:
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
    s = re.sub(r"[^\w\s/:%\-.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _count_words(txt: str) -> int:
    return len(re.findall(r"\w+", txt or ""))


def _tokenize_set(s: str) -> Set[str]:
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return set(w for w in s.split() if w)


def _jaccard_similarity(a: str, b: str) -> float:
    A, B = _tokenize_set(a), _tokenize_set(b)
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))


def _normalize_ans_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;:")
    return s


def _contains_keywords(haystack: str, keywords: Sequence[str]) -> bool:
    if not haystack or not keywords:
        return False
    s_low = haystack.lower()
    return any(k.lower() in s_low for k in keywords)


def _contains_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    haystack_lower = haystack.lower()
    needle_lower = needle.lower()
    if needle_lower in haystack_lower:
        return True
    toks = [t for t in needle_lower.split() if len(t) >= 3]
    if len(toks) >= 2:
        return " ".join(toks[:2]) in haystack_lower
    return False


def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s or "")


def _extract_action_phrases(s: str, min_len: int = 3) -> List[str]:
    if not s:
        return []
    bullets = re.findall(r"(^|\n)\s*(?:[-*•]|\d+\.)\s+(.*?)(?:\n|$)", s, re.S | re.M)
    items = [b[1].strip() for b in bullets if b[1].strip()]
    if not items:
        items = [it for it in re.split(r"[;.\n]+", s) if _count_words(it) >= 3]

    out = [
        _strip_markup(it)
        for it in items
        if _strip_markup(it) and _count_words(_strip_markup(it)) >= min_len
    ]
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _extract_python_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
    matches = re.findall(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return ""


def _extract_final_numeric(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)\s*$", s.strip())
    if m:
        return m.group(1)
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    if not text or not gen_config.think_start_tag or not gen_config.think_end_tag:
        return ""
    m = re.search(
        re.escape(gen_config.think_start_tag)
        + r"\s*(.*?)\s*"
        + re.escape(gen_config.think_end_tag),
        text,
        flags=re.I | re.S,
    )
    return (m.group(1).strip() if m else "")[:8000]


def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    tl = text or ""
    tend = gen_config.think_end_tag
    if tend and tend.lower() in tl.lower():
        idx = tl.lower().rfind(tend.lower())
        return tl[idx + len(tend) :].strip()[:2000]
    return tl.strip()[:2000]


def _extract_think_answer_lengths(
    text: str, gen_config: GenerationConfig
) -> Tuple[int, int]:
    try:
        think_content = extract_think_region(text, gen_config)
        answer_content = extract_answer_region(text, gen_config)
        return len(think_content.strip()), len(answer_content.strip())
    except Exception as e:
        logger.debug(f"Failed to extract think/answer lengths: {e}")
        return 0, 0


def _indices_to_letters(indices: List[int]) -> str:
    letters = [LETTER_ALPH[idx] for idx in indices if 0 <= idx < len(LETTER_ALPH)]
    seen, out = set(), []
    for L in sorted(letters):
        if L not in seen:
            seen.add(L)
            out.append(L)
    return ",".join(out)


def _letters_to_canonical(letter_str: str) -> str:
    parts = []
    for p in (letter_str or "").split(","):
        p = p.strip().upper()
        if len(p) == 1 and p in LETTER_ALPH:
            parts.append(p)
    seen, out = set(), []
    for L in sorted(parts):
        if L not in seen:
            seen.add(L)
            out.append(L)
    return ",".join(out)


def _match_ref_to_option_index(ref_text: str, options: List[str]) -> Optional[int]:
    if not (ref_text and options):
        return None
    ref_n = _normalize_ans_for_match(ref_text)
    for idx, opt in enumerate(options):
        if _normalize_ans_for_match(opt) == ref_n:
            return idx
    return None


def _extract_mcq_options(prompt_text: str) -> List[str]:
    if not isinstance(prompt_text, str):
        return []
    m = re.search(r"choices\s*:?(.*)$", prompt_text, flags=re.I | re.S)
    block = m.group(1) if m else prompt_text
    opts = []
    for ln in block.splitlines():
        ln_stripped = ln.strip()
        if re.match(r"^\s*[-•]\s+", ln_stripped):
            opts.append(re.sub(r"^\s*[-•]\s+", "", ln_stripped).strip())
            continue
        m2 = re.match(r"^\s*([A-Za-z])\s*[\)\.\-:]\s*(.+)$", ln_stripped)
        if m2:
            opts.append(m2.group(2).strip())
            continue
        m3 = re.match(r"^\s*\d+\s*[\)\.\-:]\s*(.+)$", ln_stripped)
        if m3:
            opts.append(m3.group(1).strip())
            continue
    return [o for o in opts if o.strip()][: len(LETTER_ALPH)]


def _infer_mcq_ref_letters(sample: Dict[str, Any]) -> str:
    meta = sample.get("meta", {})
    options = sample.get("mcq_options", [])
    ref_ans_text = sample.get("ref_answer_str", "") or sample.get("completion", "")

    for key in ("correct_letters", "correct_letter"):
        if (val := meta.get(key)) and isinstance(val, str):
            return _letters_to_canonical(val)

    indices = []
    if (val := meta.get("correct_indices")) and isinstance(val, list):
        try:
            indices = [int(x) for x in val if isinstance(x, (int, float))]
        except (ValueError, TypeError):
            pass
    elif (val := meta.get("correct_index")) is not None:
        try:
            indices = [int(val)]
        except (ValueError, TypeError):
            pass
    if indices:
        return _indices_to_letters(indices)

    if ref_ans_text and options:
        idx = _match_ref_to_option_index(ref_ans_text, options)
        if idx is not None:
            return _indices_to_letters([idx])
    return ""


def _mcq_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = sample.get("prompt", "") or sample.get("text", "")
    completion_text = sample.get("completion", "")
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    options_from_meta = meta.get("options")
    options = (
        options_from_meta
        if isinstance(options_from_meta, list)
        else _extract_mcq_options(prompt_text)
    )
    options = [str(o).strip() for o in options if str(o).strip()][: len(LETTER_ALPH)]
    is_mcq = (meta.get("type", "").lower() == "mcq") or (
        isinstance(options, list) and len(options) >= 2
    )
    if not is_mcq:
        return {"is_mcq": False, "mcq_options": options, "mcq_correct_letters": ""}
    temp_sample = {
        **sample,
        "meta": meta,
        "mcq_options": options,
        "ref_answer_str": completion_text,
    }
    correct_letters = _infer_mcq_ref_letters(temp_sample)
    correct_indices = [
        LETTER_ALPH.index(L) for L in correct_letters.split(",") if L in LETTER_ALPH
    ]
    multi_select = len(correct_indices) > 1 or bool(meta.get("multi_select", False))
    return {
        "is_mcq": True,
        "mcq_options": options,
        "mcq_multi_select": multi_select,
        "mcq_correct_indices": correct_indices,
        "mcq_correct_letters": correct_letters,
    }


def apply_chat_template_wrapper(
    tokenizer: TokenizerWrapper, prompt: str, system_prompt: Optional[str]
) -> str:
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


def _tfidf_cosine(a: str, b: str) -> float:
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
        return len(A & B) / max(1, len(A | B))


def _ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    return sum(1 for ch in s if 32 <= ord(ch) <= 126 or ch in "\n\r\t") / max(1, len(s))


def _looks_garbage(s: str) -> bool:
    if not s or len(s.strip()) < 3 or len(s) > 20000 or _ascii_ratio(s) < 0.75:
        return True
    bad = re.findall(r"[^\w\s\-\.\,\:\;\(\)\[\]\/\+\=\&\<\>]", s)
    return (len(bad) / max(1, len(s))) > 0.15


LETTER_ALPH = string.ascii_uppercase


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
    s = re.sub(r"[^\w\s/:%\-.]", " ", s)
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
    toks = [t for t in needle_lower.split() if len(t) >= 3]
    if len(toks) >= 2:
        return " ".join(toks[:2]) in haystack_lower
    return False


def _has_non_ascii(s: str) -> bool:
    """Checks if a string contains any non-ASCII characters."""
    return any(ord(ch) > 127 for ch in s or "")


def _extract_action_phrases(s: str, min_len: int = 3) -> List[str]:
    """Identifies potential steps/actionable phrases from text, usually from structured lists."""
    if not s:
        return []
    # Identify bulleted/numbered lists
    bullets = re.findall(r"(^|\n)\s*(?:[-*•]|\d+\.)\s+(.*?)(?:\n|$)", s, re.S | re.M)
    items = [b[1].strip() for b in bullets if b[1].strip()]
    if not items:
        # Fallback to splitting by common separators if no list formatting found
        items = [it for it in re.split(r"[;.\n]+", s) if _count_words(it) >= 3]

    out = [
        _strip_markup(it)
        for it in items
        if _strip_markup(it) and _count_words(_strip_markup(it)) >= min_len
    ]
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _extract_python_code(text: str) -> str:
    """Extracts Python code from a markdown code block or assumes plain code."""
    # Look for python block first
    matches = re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()

    # Look for generic code block
    matches = re.findall(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[0].strip()

    return ""


def _extract_final_numeric(s: str) -> Optional[str]:
    """Extracts the last numeric value (int or float) from a string, supporting `####` format."""
    if not s:
        return None
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)\s*$", s.strip())
    if m:
        return m.group(1)
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return m[-1] if m else None


# Helper functions for tags, using GenerationConfig
def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    """Extracts content between think_start_tag and think_end_tag."""
    if not text or not gen_config.think_start_tag or not gen_config.think_end_tag:
        return ""
    m = re.search(
        re.escape(gen_config.think_start_tag)
        + r"\s*(.*?)\s*"
        + re.escape(gen_config.think_end_tag),
        text,
        flags=re.I | re.S,
    )
    return (m.group(1).strip() if m else "")[:8000]


def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    """Extracts answer region: everything AFTER the last </think> tag. If no </think> tag, returns full text."""
    tl = text or ""
    tend = gen_config.think_end_tag
    if tend and tend.lower() in tl.lower():
        idx = tl.lower().rfind(tend.lower())

        answer_part = tl[idx + len(tend) :].strip()
        # Optionally look for explicit answer tags if defined, but usually content after </think> is the answer.
        # This implementation aligns with the goal of pulling everything *after* the thinking phase.

        return answer_part[:2000]
    return tl.strip()[:2000]


def _extract_think_answer_lengths(
    text: str, gen_config: GenerationConfig
) -> Tuple[int, int]:
    """Extracts character lengths of thinking and answer sections from text using GenerationConfig tags."""
    try:
        think_content = extract_think_region(text, gen_config)
        answer_content = extract_answer_region(text, gen_config)
        return len(think_content.strip()), len(answer_content.strip())
    except Exception as e:
        logger.debug(f"Failed to extract think/answer lengths: {e}")
        return 0, 0


# --- MCQ Helpers ---
def _indices_to_letters(indices: List[int]) -> str:
    """Converts a list of 0-based indices to comma-separated letters (e.g., [0, 2] -> 'A,C')."""
    letters = [LETTER_ALPH[idx] for idx in indices if 0 <= idx < len(LETTER_ALPH)]
    seen, out = set(), []
    for L in sorted(letters):
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
    for L in sorted(parts):
        if L not in seen:
            seen.add(L)
            out.append(L)
    return ",".join(out)


def _match_ref_to_option_index(ref_text: str, options: List[str]) -> Optional[int]:
    """Tries to match a reference answer string to one of the provided options."""
    if not (ref_text and options):
        return None
    ref_n = _normalize_ans_for_match(ref_text)
    for idx, opt in enumerate(options):
        if _normalize_ans_for_match(opt) == ref_n:
            return idx
    return None


def _extract_mcq_options(prompt_text: str) -> List[str]:
    """Tries to extract numbered or bulleted MCQ options from a prompt."""
    if not isinstance(prompt_text, str):
        return []
    m = re.search(r"choices\s*:?(.*)$", prompt_text, flags=re.I | re.S)
    block = m.group(1) if m else prompt_text
    opts = []
    for ln in block.splitlines():
        ln_stripped = ln.strip()
        if re.match(r"^\s*[-•]\s+", ln_stripped):
            opts.append(re.sub(r"^\s*[-•]\s+", "", ln_stripped).strip())
            continue
        m2 = re.match(r"^\s*([A-Za-z])\s*[\)\.\-:]\s*(.+)$", ln_stripped)
        if m2:
            opts.append(m2.group(2).strip())
            continue
        m3 = re.match(r"^\s*\d+\s*[\)\.\-:]\s*(.+)$", ln_stripped)
        if m3:
            opts.append(m3.group(1).strip())
            continue
    return [o for o in opts if o.strip()][: len(LETTER_ALPH)]


def _infer_mcq_ref_letters(sample: Dict[str, Any]) -> str:
    """Infers the canonical correct MCQ letters from various metadata fields or text matching."""
    meta = sample.get("meta", {})
    options = sample.get("mcq_options", [])
    ref_ans_text = sample.get("ref_answer_str", "") or sample.get("completion", "")

    for key in ("correct_letters", "correct_letter"):
        if (val := meta.get(key)) and isinstance(val, str):
            return _letters_to_canonical(val)

    indices = []
    if (val := meta.get("correct_indices")) and isinstance(val, list):
        try:
            indices = [int(x) for x in val if isinstance(x, (int, float))]
        except (ValueError, TypeError):
            pass
    elif (val := meta.get("correct_index")) is not None:
        try:
            indices = [int(val)]
        except (ValueError, TypeError):
            pass
    if indices:
        return _indices_to_letters(indices)

    if ref_ans_text and options:
        idx = _match_ref_to_option_index(ref_ans_text, options)
        if idx is not None:
            return _indices_to_letters([idx])
    return ""


def _extract_predicted_letters(
    generated_text: str, options: Optional[List[str]], cfg: GenerationConfig
) -> str:
    """Extracts the predicted MCQ letter(s) from the answer region."""
    ans = extract_answer_region(generated_text or "", cfg)

    # 1. Look for first capital letter followed by separator/space
    m = re.match(r"^\s*([A-Z])(?:[\)\.\:\-]\s*|\s+|$)", ans)
    if m:
        pred_char = m.group(1).upper()
        if not options or LETTER_ALPH.index(pred_char) < len(options):
            return pred_char

    return ""  # Return empty if no clear prediction found


def _mcq_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Generates comprehensive MCQ metadata for a sample."""
    prompt_text = sample.get("prompt", "") or sample.get("text", "")
    completion_text = sample.get("completion", "")
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}

    options_from_meta = meta.get("options")
    if isinstance(options_from_meta, list):
        options = options_from_meta
    else:
        options = _extract_mcq_options(prompt_text)
    options = [str(o).strip() for o in options if str(o).strip()][: len(LETTER_ALPH)]

    is_mcq = (meta.get("type", "").lower() == "mcq") or (
        isinstance(options, list) and len(options) >= 2
    )
    if not is_mcq:
        return {"is_mcq": False, "mcq_options": options, "mcq_correct_letters": ""}

    temp_sample = {
        **sample,
        "meta": meta,
        "mcq_options": options,
        "ref_answer_str": completion_text,
    }
    correct_letters = _infer_mcq_ref_letters(temp_sample)

    correct_indices = [
        LETTER_ALPH.index(L) for L in correct_letters.split(",") if L in LETTER_ALPH
    ]
    multi_select = len(correct_indices) > 1 or bool(meta.get("multi_select", False))

    return {
        "is_mcq": True,
        "mcq_options": options,
        "mcq_multi_select": multi_select,
        "mcq_correct_indices": correct_indices,
        "mcq_correct_letters": correct_letters,
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
        return len(A & B) / max(1, len(A | B))


class TwoBlockFormatter:
    """Utility class used in BEFORE_STATE for coercing and scoring text format."""

    def __init__(
        self,
        think_start: str,
        think_end: str,
        answer_start: str,
        answer_end: str,
        validate_json: bool = False,
    ):
        self.ts, self.te, self.as_, self.ae = (
            think_start,
            think_end,
            answer_start,
            answer_end,
        )
        self.validate_json = validate_json
        self._re_think = (
            re.compile(re.escape(self.ts) + r"(.*?)" + re.escape(self.te), re.DOTALL)
            if self.ts and self.te
            else None
        )
        self._re_answer = (
            re.compile(re.escape(self.as_) + r"(.*?)" + re.escape(self.ae), re.DOTALL)
            if self.as_ and self.ae
            else None
        )

    def _extract_json_from_text(self, text: str) -> str:
        text = text.strip()
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
        text = self._extract_json_from_text(text.strip())
        return text.startswith("{") or text.startswith("[")

    def _validate_json_content(self, content: str) -> float:
        if not content or content == "Insufficient information.":
            return 0.0
        try:
            json.loads(self._extract_json_from_text(content))
            return 0.5
        except (json.JSONDecodeError, ValueError):
            return 0.0

    def _coerce_one(self, s: str) -> str:
        s = (s or "").strip()
        if not s:
            return f"{self.ts}…{self.te}\n{self.as_}Insufficient information.{self.ae}"

        i_ts = s.find(self.ts) if self.ts else -1
        if i_ts == -1:  # No <think> tag
            ans = _strip_specials(s or "Insufficient information.")
            if self.te:
                ans = ans.replace(self.te, "")
            return f"{self.ts}…{self.te}\n{self.as_}{ans}{self.ae}"

        s = s[i_ts:]
        i_te = s.find(self.te, len(self.ts)) if self.te else -1
        if i_te == -1:
            s += self.te
            i_te = len(s) - len(self.te)

        think_body = _strip_specials(s[len(self.ts) : i_te].strip() or "…")
        remainder = s[i_te + len(self.te) :].strip()

        answer_body = remainder
        if self.as_ and self.ae:  # If explicit answer tags defined
            i_as = remainder.find(self.as_)
            if i_as != -1:
                answer_body = remainder[i_as + len(self.as_) :]
                i_ae = answer_body.find(self.ae)
                if i_ae != -1:
                    answer_body = answer_body[:i_ae].strip()
                else:
                    answer_body = answer_body.strip()
            else:
                pass
        elif self.as_ and not self.ae:  # If only opening answer tag
            i_as = remainder.find(self.as_)
            if i_as != -1:
                answer_body = remainder[i_as + len(self.as_) :].strip()

        answer_body = _strip_specials(answer_body.replace(self.te, ""))
        if not answer_body:
            answer_body = "Insufficient information.Way too much overthinking"

        final = f"{self.ts}{think_body}{self.te}\n{self.as_}{answer_body}{self.ae}"
        if self.ae and (j_ae := final.rfind(self.ae)) != -1:
            final = final[: j_ae + len(self.ae)]
        return final

    def _score_one(
        self, s: str, detailed: bool = False
    ) -> Union[float, Dict[str, float]]:
        """Scores based on proper tag usage, aligned with BEFORE_STATE logic."""
        result = {"reward_format": 0.0, "reward_content": 0.0, "reward_total": 0.0}

        if not s:
            return result if detailed else 0.0
        s = s.strip()

        if self._looks_like_json(s):
            result["reward_format"] = 0.5
            result["reward_total"] = 0.5
            if self.validate_json:
                result["reward_content"] = self._validate_json_content(s)
            result["reward_total"] = result["reward_format"] * (
                result["reward_content"] if self.validate_json else 1.0
            )
            return result if detailed else result["reward_total"]

        # Check basic think/answer structure (Simplified from BEFORE_STATE's reward calculation)
        th_s = s.count(self.ts)
        th_e = s.count(self.te)
        if th_s == 1 and th_e == 1:
            think_content = extract_think_region(
                s, GenerationConfig(think_start_tag=self.ts, think_end_tag=self.te)
            )
            answer_content = extract_answer_region(
                s, GenerationConfig(think_end_tag=self.te)
            )

            if len(think_content) > 10 and len(answer_content) > 10:
                result["reward_format"] = 1.0
            elif len(think_content) > 10 or len(answer_content) > 10:
                result["reward_format"] = 0.5
            else:
                result["reward_format"] = 0.2
        elif th_s >= 1 and th_e == 0:
            result["reward_format"] = 0.3
        elif th_s == 0 and th_e == 0:
            result["reward_format"] = 0.1 if len(s) > 20 else 0.0
        else:
            result["reward_format"] = 0.2

        result["reward_total"] = result["reward_format"]
        return result if detailed else result["reward_total"]

    def coerce_batch(self, texts: Sequence[str]) -> List[str]:
        return [self._coerce_one(t) for t in texts]

    def score_batch(
        self, texts: Sequence[str], detailed: bool = False
    ) -> Union[List[float], List[Dict[str, float]]]:
        return [self._score_one(t, detailed=detailed) for t in texts]


def make_dynamic_tag_bias_processor(
    tokenizer: TokenizerWrapper, config: ExperimentConfig, mcq_flags: List[bool]
) -> Callable:
    gen_cfg = config.generation
    tag_ids = _resolve_tag_ids(tokenizer, gen_cfg)
    mcq_letter_ids = sorted(set(sum(_letter_token_ids(tokenizer).values(), [])))
    ban_ids = _first_token_ids_for_lexemes(tokenizer, gen_cfg.ban_phrases_for_bias)
    encourage_ids = _first_token_ids_for_lexemes(
        tokenizer, gen_cfg.encourage_phrases_for_bias
    )
    tool_ids = _first_token_ids_for_lexemes(tokenizer, _TOOL_LIKE_MARKERS)

    te, ts, as_id, ae, eos_tok = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    B_CLOSE, B_AS, P_REOPEN_THINK, P_EXTRA_TE, P_REOPEN_ANS, B_EOS_ANS = (
        gen_cfg.bias_close_think,
        gen_cfg.bias_answer_start,
        gen_cfg.punish_reopen_think,
        gen_cfg.punish_extra_think_end,
        gen_cfg.punish_reopen_answer,
        gen_cfg.bias_eos_after_answer,
    )
    MIN_ANS, MIN_ANS_MCQ, HARD_MASK, LIFT_MCQ, BAN_MCQ, BAN_NONMCQ = (
        gen_cfg.min_answer_tokens,
        gen_cfg.min_answer_tokens_mcq,
        gen_cfg.hard_mask_mcq_first_token,
        gen_cfg.mcq_letter_lift,
        gen_cfg.mcq_ban_first_bias,
        gen_cfg.nonmcq_ban_first_bias,
    )
    MCQ_CLOSE_K, B_MCQ_CLOSE, MIN_THINK, B_END_EARLY, B_AS_MIN_THINK = (
        gen_cfg.mcq_close_after_k,
        gen_cfg.mcq_answer_end_bias,
        gen_cfg.min_think_tokens,
        gen_cfg.think_end_early_bias,
        gen_cfg.bias_answer_start_after_min_think,
    )
    B_ENCOURAGE, P_TOOL = (
        gen_cfg.encourage_think_bias,
        gen_cfg.tool_call_penalty * -10.0,
    )

    def _proc_vectorized(hist_list: List[List[int]], logits: mx.array) -> mx.array:
        if logits.ndim != 2:
            return logits
        B, V = logits.shape
        neg_inf, pad_id = mx.array(-1e9, dtype=logits.dtype), tokenizer.pad_token_id
        max_hist_len = max(len(row) for row in hist_list) if hist_list else 0
        if max_hist_len == 0:
            return logits

        history_mx = mx.array(
            [row + [pad_id] * (max_hist_len - len(row)) for row in hist_list],
            dtype=mx.int32,
        )
        if tool_ids and P_TOOL < 0:
            logits = logits.at[:, tool_ids].add(P_TOOL)

        def find_last_pos_mx(tag_id):
            if tag_id is None:
                return mx.full((B,), -1, dtype=mx.int32)
            matches = history_mx == tag_id
            rev_indices = mx.argmax(matches[:, ::-1], axis=1).astype(mx.int32)
            # Add this check
            if max_hist_len <= 0:
                return mx.full((B,), -1, dtype=mx.int32)
            return mx.where(
                mx.any(matches, axis=1), max_hist_len - 1 - rev_indices, -1
            ).astype(mx.int32)

        last_ts, last_te, last_as, last_ae = (
            find_last_pos_mx(t) for t in (ts, te, as_id, ae)
        )
        history_len_mx = mx.array([len(row) for row in hist_list], dtype=mx.int32)

        inside_think = mx.logical_and(
            last_ts != -1, mx.logical_and(last_te < last_ts, last_as < last_ts)
        )
        inside_answer = mx.logical_and(last_as != -1, last_ae < last_as)
        ae_seen = last_ae != -1
        k_think = mx.where(inside_think, history_len_mx - (last_ts + 1), 0)
        k_answer = mx.where(inside_answer, history_len_mx - (last_as + 1), 0)
        is_mcq_mask = mx.array(mcq_flags, dtype=mx.bool_)

        if ts is not None and te is not None:
            logits = logits.at[:, ts].add(mx.where(last_te != -1, P_REOPEN_THINK, 0.0))
            if as_id is not None:
                logits = logits.at[:, as_id].add(
                    mx.where(last_ae > last_as, P_REOPEN_ANS, 0.0)
                )
            te_count = mx.sum(history_mx == te, axis=1)
            bias_at_te = mx.where(te_count == 0, B_CLOSE, P_EXTRA_TE)
            min_think_penalty_mask = mx.logical_and(inside_think, (k_think < MIN_THINK))
            bias_at_te = mx.where(min_think_penalty_mask, B_END_EARLY, bias_at_te)
            logits = logits.at[:, te].add(bias_at_te)
            can_start_answer = mx.logical_and(
                last_te > last_as, mx.logical_not(inside_answer)
            )
            min_think_ok = mx.logical_not(B_AS_MIN_THINK)
            if B_AS_MIN_THINK:
                min_think_ok = k_think >= MIN_THINK
            can_start_answer = mx.logical_and(can_start_answer, min_think_ok)
            if as_id is not None:
                logits = logits.at[:, as_id].add(mx.where(can_start_answer, B_AS, 0.0))

        if eos_tok is not None:
            logits = logits.at[:, eos_tok].add(mx.where(ae_seen, B_EOS_ANS, 0.0))
        if encourage_ids and B_ENCOURAGE > 0 and mx.any(inside_think).item():
            logits = logits.at[inside_think.tolist(), encourage_ids].add(B_ENCOURAGE)

        mcq_first_token_mask = mx.logical_and(
            is_mcq_mask, mx.logical_and(inside_answer, (k_answer == 0))
        )
        if mx.any(mcq_first_token_mask).item() and HARD_MASK:
            mcq_allowed_logits = mx.full((V,), neg_inf, dtype=logits.dtype)
            if mcq_letter_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[mcq_letter_ids].set(LIFT_MCQ)
            if ban_ids:
                mcq_allowed_logits = mcq_allowed_logits.at[ban_ids].add(BAN_MCQ)
            logits = mx.where(
                mcq_first_token_mask[:, None], mcq_allowed_logits[None, :], logits
            )

        non_mcq_first_answer = mx.logical_and(
            mx.logical_not(is_mcq_mask), mx.logical_and(inside_answer, (k_answer == 0))
        )
        if mx.any(non_mcq_first_answer).item() and ban_ids:
            logits = logits.at[non_mcq_first_answer.tolist(), ban_ids].add(BAN_NONMCQ)

        if ae is not None:
            min_ans_len = mx.where(is_mcq_mask, MIN_ANS_MCQ, MIN_ANS)
            min_len_penalty_mask = mx.logical_and(
                inside_answer, (k_answer < min_ans_len)
            )
            logits = logits.at[:, ae].add(mx.where(min_len_penalty_mask, -8.0, 0.0))
            mcq_close_mask = mx.logical_and(
                is_mcq_mask, mx.logical_and(inside_answer, (k_answer >= MCQ_CLOSE_K))
            )
            logits = logits.at[:, ae].add(mx.where(mcq_close_mask, B_MCQ_CLOSE, 0.0))

        return logits

    return _proc_vectorized


def _mask_after_answer(
    responses_mx: mx.array,
    initial_mask: mx.array,
    tokenizer: TokenizerWrapper,
    config: ExperimentConfig,
) -> mx.array:
    if responses_mx.ndim != 2:
        return initial_mask
    B, L_gen = responses_mx.shape
    initial_mask = initial_mask.astype(mx.float32)
    answer_end_id = _resolve_tag_ids(tokenizer, config.generation).get("answer_end")
    if answer_end_id is None:
        return initial_mask
    indices = mx.arange(L_gen)
    is_answer_end = responses_mx == answer_end_id
    first_end_indices = mx.argmin(mx.where(is_answer_end, indices, L_gen + 1), axis=1)
    boundary_index = first_end_indices + 1
    end_mask = (
        mx.broadcast_to(indices[None, :], responses_mx.shape) < boundary_index[:, None]
    )
    return initial_mask * end_mask.astype(mx.float32)


# --- MLX-Specific Errors and Recovery (Unchanged) ---
def _is_metal_internal_error(err: BaseException) -> bool:
    s = str(err)
    return (
        "Command buffer execution failed" in s
        or "[METAL]" in s in s
        or "Internal Error" in s
    )


def metal_recover(stage: str):
    logger.warning(f"[METAL] Recovering after error at stage: {stage}")
    try:
        mx.synchronize()
    except Exception:
        pass
    mx.clear_cache()
    gc.collect()


def metal_safe_apply_gradients(
    optimizer: optim.Optimizer, grads: Dict[str, mx.array], params: Dict[str, mx.array]
):
    try:
        # NOTE: apply_gradients handles the update of both params and optimizer.state
        return optimizer.apply_gradients(grads, params)
    except Exception as e:
        if _is_metal_internal_error(e):
            metal_recover("apply_gradients")
            return None
        raise
    finally:
        mx.clear_cache()
        gc.collect()


# --- Gradient Manipulation ---
_LAYER_PAT = re.compile(r"(?:^|[^a-zA-Z0-9_])layers\.(\d+)(?:[^0-9_]|$)")
_HEAD_PAT = re.compile(r"\blm_head\b", re.I)


def _find_layer_index(name: str) -> Optional[int]:
    m = _LAYER_PAT.search(name)
    if m:
        return int(m.group(1))
    parts = re.split(r"[\.\/]", name)
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                pass
    return None


def _band_for_name(
    name: str,
    low_band: Tuple[int, int],
    mid_band: Tuple[int, int],
    top_band: Tuple[int, int],
) -> str:
    li = _find_layer_index(name)

    def _in_range(li: int, band_range: Optional[Tuple[int, int]]) -> bool:
        if band_range is None:
            return False
        try:
            s, e = band_range
            return (s is None or li >= int(s)) and (e is None or li <= int(e))
        except Exception:
            return False

    if li is not None:
        if _in_range(li, low_band):
            return "low"
        if _in_range(li, mid_band):
            return "mid"
        if _in_range(li, top_band):
            return "top"
    if _HEAD_PAT.search(name):
        return "head"
    return "other"


def scale_grads_by_band(
    grads_tree: Dict[str, mx.array], config: ExperimentConfig
) -> Dict[str, mx.array]:
    low_band, mid_band, top_band = (
        config.trainer.low_band,
        config.trainer.mid_band,
        config.trainer.top_band,
    )
    low_mul, mid_mul, top_mul, head_mul = (
        config.trainer.low_mul,
        config.trainer.mid_mul,
        config.trainer.top_mul,
        config.trainer.head_mul,
    )
    g_flat = tree_flatten(grads_tree)
    out = []
    for name, g in g_flat:
        if not isinstance(g, mx.array):
            out.append((name, g))
            continue
        band = _band_for_name(
            name,
            low_band,
            mid_band,
            top_band,
        )
        mul = {"low": low_mul, "mid": mid_mul, "top": top_mul, "head": head_mul}.get(
            band, 1.0
        )
        out.append((name, g * mul))
    return tree_unflatten(out)


def mask_grads_to_layer_band(
    grads_tree: Dict[str, mx.array],
    start: Optional[int],
    end: Optional[int],
    *,
    include_embed: bool = True,
    include_head: bool = True,
    include_final_norm: bool = True,
) -> Dict[str, mx.array]:
    flat = tree_flatten(grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue
        li = _find_layer_index(name)
        keep = False
        if li is not None:
            keep = (start is None or li >= start) and (end is None or li <= end)
        else:
            lname = name.lower()
            if (
                "embed" in lname
                or "token_embedding" in lname
                or "word_embedding" in lname
            ):
                keep = include_embed
            elif (
                "final_norm" in lname
                or "norm_out" in lname
                or "ln_f" in lname
                or "final_layer_norm" in lname
            ):
                keep = include_final_norm
            elif (
                "lm_head" in lname
                or "output" in lname
                and "head" in lname
                or "logits" in lname
            ):
                keep = include_head
        kept.append((name, g if keep else mx.zeros_like(g)))
    return tree_unflatten(kept)


def mask_grads_to_specific_layers(
    grads_tree: Dict[str, mx.array], layer_indices: Set[int]
) -> Dict[str, mx.array]:
    flat = tree_flatten(grads_tree)
    kept = []
    for name, g in flat:
        if not isinstance(g, mx.array):
            kept.append((name, g))
            continue
        if (
            layer_idx := _find_layer_index(name)
        ) is not None and layer_idx in layer_indices:
            kept.append((name, g))
        else:
            kept.append((name, mx.zeros_like(g)))
    return tree_unflatten(kept)


# --- Attention Masking ---


def _create_4d_attention_mask(
    tokens: mx.array, pad_token_id: int, dtype: mx.Dtype = TARGET_FLOAT_DTYPE
) -> mx.array:
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2D, got {tokens.shape}")
    B, T = tokens.shape
    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T, dtype=dtype)
    padding_mask = (tokens == pad_token_id)[:, None, None, :]
    neg_inf = mx.array(
        -1e9, dtype=dtype
    )  # Use large negative number instead of bfloat16 limit for wider compatibility
    if dtype == mx.bfloat16:
        neg_inf = mx.array(-65504.0, dtype=dtype)

    combined = mx.minimum(causal_mask, mx.where(padding_mask, neg_inf, 0.0))
    return combined


# --- Sampling ---


def safe_make_sampler(config: Any, temp: float) -> Callable:
    """Creates a sampler safely, handling differing mlx-lm API versions."""
    top_p = float(getattr(config, "sampling_top_p", 0.9))
    top_k = int(getattr(config, "sampling_top_k", 0))
    min_p_val = float(getattr(config, "sampling_min_p", 0.0))

    # Ensure temp is not below 0 or ridiculously high
    temp = max(1e-5, temp)

    # Note: MLX sampling functions evolve. Try the most complete signature first.
    try:
        return make_sampler(temp=temp, top_p=top_p, min_p=min_p_val, top_k=top_k)
    except TypeError:
        try:
            # Older signature: no top_k or min_p
            return make_sampler(temp=temp, top_p=top_p)
        except Exception as e:
            logger.error(f"Failed to create sampler: {e}. Falling back to greedy.")
            return make_sampler(temp=1e-6, top_p=1.0)  # Near greedy


# --- Logit Processors Helper Functions ---

_TOOL_LIKE_MARKERS = [
    "<tool_code>",
    "</tool_code>",
    "<json_output>",
    "</json_output>",
    "<img_base64>",
    "</img_base64>",
    "<ocr_text>",
    "</ocr_text>",
]


def _first_token_ids_for_lexemes(
    tokenizer: TokenizerWrapper, lexemes: Sequence[str]
) -> List[int]:
    ids: List[int] = []
    for lx in lexemes:
        # Check raw token
        if (
            (t := tokenizer.encode(lx, add_special_tokens=False))
            and t
            and t[0] not in ids
        ):
            ids.append(t[0])
        # Check token with leading space
        if (
            (t_space := tokenizer.encode(" " + lx, add_special_tokens=False))
            and t_space
            and t_space[0] not in ids
        ):
            ids.append(t_space[0])
    return ids


def _letter_token_ids(
    tokenizer: TokenizerWrapper, letters: Sequence[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) -> Dict[str, List[int]]:
    out = {}
    for L in letters:
        cand = []
        if (ids := tokenizer.encode(L, add_special_tokens=False)) and len(ids) == 1:
            cand.append(ids[0])
        if (
            (ids := tokenizer.encode(" " + L, add_special_tokens=False))
            and len(ids) == 1
            and ids[0] not in cand
        ):
            cand.append(ids[0])
        # Include common suffix tokenizations
        for suf in [")", ".", " )", " ."]:
            if (
                (ids := tokenizer.encode(L + suf, add_special_tokens=False))
                and len(ids) == 1
                and ids[0] not in cand
            ):
                cand.append(ids[0])
        out[L] = cand
    return out


def _resolve_tag_ids(
    tokenizer: TokenizerWrapper, config: ExperimentConfig
) -> Dict[str, Optional[int]]:
    def _one_id(tok_str):
        if not tok_str:
            return None
        try:
            ids = tokenizer.encode(tok_str, add_special_tokens=False)
            return ids[0] if len(ids) == 1 else None
        except Exception:
            return None

    tags = config.generation
    return {
        "think_start": _one_id(tags.think_start_tag),
        "think_end": _one_id(tags.think_end_tag),
        "answer_start": _one_id(tags.answer_start_tag),
        "answer_end": _one_id(tags.answer_end_tag),
        "eos": tokenizer.eos_token_id,
    }
