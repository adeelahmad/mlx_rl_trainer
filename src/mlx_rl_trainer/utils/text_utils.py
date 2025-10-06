# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


import logging
import re
import string
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import mlx.core as mx

from mlx_rl_trainer.core.config import RewardConfig
from mlx_lm.tokenizer_utils import TokenizerWrapper


def _tokenize_set(s: str) -> Set[str]:
    """Convert string to set of lowercase tokens without punctuation"""
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return set(w for w in s.split() if w)


def _jaccard_similarity(a: str, b: str) -> float:
    """Return Jaccard similarity between two strings (token-based)"""
    A, B = _tokenize_set(a), _tokenize_set(b)
    if not A or not B:
        return 0.0
    return float(len(A & B) / len(A | B))


def _has_non_ascii(s: str) -> bool:
    """Check for any non-ASCII characters in the string"""
    return any(ord(ch) > 127 for ch in s or "")


def _contains_keywords(s: str, keywords: Sequence[str]) -> bool:
    """Return True if any keyword occurs in s (case-insensitive)"""
    if not s or not keywords:
        return False
    s_low = s.lower()
    return any(k.lower() in s_low for k in keywords)


def _tfidf_cosine(a: str, b: str) -> float:
    """Compute TF-IDF cosine similarity, fallback to Jaccard"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        X = vec.fit_transform([a, b])
        sim = float(cosine_similarity(X[0:1], X[1:2])[0, 0])
        return max(0.0, min(1.0, sim))
    except Exception:
        A, B = _tokenize_set(a), _tokenize_set(b)
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)


# Regex patterns for markup stripping
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`]+`")
_HTML_TAGS = re.compile(r"<[^>]+>")


def _strip_markup(s: str) -> str:
    """Remove markdown and HTML formatting"""
    if not s:
        return ""
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


def _extract_action_phrases(s: str) -> List[str]:
    """Extract bullet points or action phrases from text"""
    if not s:
        return []
    bullets = re.findall(r"(^|\n)\s*(?:[-*•]|\d+\.)\s+(.*?)(?:\n|$)", s)
    items = [b[1].strip() for b in bullets if b[1].strip()]
    if not items:
        items = re.split(r"[;\n\.]+", s)
    out = []
    for it in items:
        itn = _strip_markup(it)
        if itn and len(itn) >= 3:
            out.append(itn)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════


def extract_think_region(text: str, cfg: RewardConfig) -> str:
    """Extract content between <think> and </think> tags"""
    m = re.search(
        re.escape(cfg.think_start_tag) + r"\s*(.*?)\s*" + re.escape(cfg.think_end_tag),
        text or "",
        flags=re.I | re.S,
    )
    return (m.group(1).strip() if m else "")[:8000]


def extract_answer_region(text: str, cfg: RewardConfig) -> str:
    """
    Extract answer region: everything AFTER the last </think> tag.
    If no </think> tag exists, return the entire text.
    """
    tl = text or ""
    tend = cfg.think_end_tag

    if tend and tend.lower() in tl.lower():
        # Find LAST occurrence of </think>
        idx = tl.lower().rfind(tend.lower())
        # Everything after </think> is the answer
        answer = tl[idx + len(tend) :].strip()
        return answer[:2000]

    # No </think> tag found - return whole text as answer
    return tl.strip()[:2000]


# ═══════════════════════════════════════════════════════════════════════════════
# SHORTHAND THINKING REWARDS (NEW!)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_shorthand_reward(think_text: str, cfg: RewardConfig) -> float:
    """
    Reward compact, symbolic thinking. Penalize verbose patterns.
    Returns value in range [-1.0, 1.0]
    """
    if not think_text:
        return 0.0

    reward = 0.0
    think_lower = think_text.lower()

    # === 1. SYMBOLIC CHARACTER BONUS ===
    symbol_count = sum(think_text.count(sym) for sym in cfg.config.get('symbolic_chars', []))
    reward += symbol_count * cfg.config.get('symbolic_bonus_per_use', 0.0)

    # === 2. ABBREVIATION BONUS ===
    abbrev_count = sum(1 for abbr in cfg.config.get('abbreviations', []) if abbr.lower() in think_lower)
    reward += abbrev_count * cfg.config.get('abbreviation_bonus_per_use', 0.0)

    # === 3. BAN KEYWORD PENALTY ===
    banned_count = sum(
        1 for phrase in cfg.config.get('ban_keywords', []) if phrase.lower() in think_lower
    )
    reward -= banned_count * cfg.config.get('ban_keyword_penalty', 0.0)

    # === 4. VERBOSITY PENALTY (words per line) ===
    lines = [line.strip() for line in think_text.split("\n") if line.strip()]
    if lines:
        total_words = sum(len(line.split()) for line in lines)
        avg_words_per_line = total_words / len(lines)

        max_avg_words_per_line = cfg.config.get('max_avg_words_per_line', 0.0)
        if avg_words_per_line > max_avg_words_per_line:
            excess = avg_words_per_line - max_avg_words_per_line
            reward -= excess * cfg.config.get('verbosity_penalty_strength', 0.0)

    # === 5. TOKEN DIVERSITY CHECK ===
    tokens = think_text.split()
    if len(tokens) > 5:  # Only check if sufficient tokens
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < cfg.config.get('min_unique_token_ratio', 0.0):
            reward -= cfg.config.get('low_diversity_penalty', 0.0)

    # === 6. COMPACT NOTATION BONUS (holistic check) ===
    # Award bonus if thinking shows multiple compact characteristics
    compact_score = 0
    if symbol_count >= 3:
        compact_score += 1
    if abbrev_count >= 2:
        compact_score += 1
    if banned_count == 0:
        compact_score += 1
    if avg_words_per_line < cfg.config.get('compact_notation_avg_words_per_line_threshold', 8.0):
        compact_score += 1

    if compact_score >= 3:
        reward += cfg.config.get('compact_notation_bonus', 0.0)

    # Clamp to reasonable range
    return max(-1.0, min(1.0, reward))


def compute_think_length_reward(
    think_text: str,
    target_min: int = 16,
    target_max: int = 64,
    penalty_strength: float = 0.01,
    penalty_type: str = "exponential",
) -> float:
    """
    Reward thinking within target token range, penalize excess.

    Args:
        think_text: The thinking content
        target_min: Minimum desired tokens
        target_max: Maximum desired tokens
        penalty_strength: How strongly to penalize excess
        penalty_type: 'linear' or 'exponential'

    Returns:
        Reward in range [-1.0, 1.0]
    """
    if not think_text:
        return -0.5  # Penalize missing thinking

    tokens = think_text.split()
    length = len(tokens)

    # Within target range
    if target_min <= length <= target_max:
        # Slightly prefer shorter within range
        ratio = (
            (target_max - length) / (target_max - target_min)
            if target_max > target_min
            else 0.5
        )
        return 0.3 + 0.2 * ratio  # 0.3 to 0.5 reward

    # Too short
    if length < target_min:
        shortage = target_min - length
        return -0.1 * (shortage / target_min)

    # Too long
    excess = length - target_max
    if penalty_type == "exponential":
        # Exponential penalty grows quickly
        penalty = 1.0 - (1.0 / (1.0 + penalty_strength * excess**1.5))
    else:
        # Linear penalty
        penalty = min(1.0, penalty_strength * excess)

    return -penalty


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT REWARD
# ═══════════════════════════════════════════════════════════════════════════════


def format_reward(text: str, cfg: RewardConfig) -> float:
    """
    Reward proper use of <think> tags.
    """
    th_s = len(re.findall(re.escape(cfg.think_start_tag), text or "", flags=re.I))
    th_e = len(re.findall(re.escape(cfg.think_end_tag), text or "", flags=re.I))

    think = extract_think_region(text, cfg)
    ans = extract_answer_region(text, cfg)

    # Perfect format: exactly one think block with both thinking and answer content
    if th_s == 1 and th_e == 1:
        if len(think) > 10 and len(ans) > 10:
            return 1.0  # Increased from 0.5 - reward good structure strongly
        if len(think) > 10 or len(ans) > 10:
            return 0.5  # Has one component
        return 0.2  # Has structure but too short

    # Incomplete structure
    if th_s >= 1 and th_e == 0:
        return 0.3  # Started thinking but didn't close

    if th_s == 0 and th_e == 0:
        # No structure at all - but might still have content
        if len(text or "") > 20:
            return 0.1  # Some content without structure
        return 0.0  # Empty or near-empty

    # Multiple think blocks or malformed
    return 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT REWARDS
# ═══════════════════════════════════════════════════════════════════════════════


def hybrid_answer_reward(
    text: str,
    reference_completion: Optional[str],
    cfg: RewardConfig,
    sem_weight: float = 0.3,
) -> float:
    """
    Hybrid semantic similarity between generated answer and reference.
    Combines Jaccard and TF-IDF cosine similarity.
    """
    if not reference_completion:
        return 0.0

    gen_ans_raw = extract_answer_region(text, cfg)
    ref_ans = reference_completion

    # Dynamic cap based on reference length
    cap = max(200, min(1000, 4 * len(ref_ans)))

    # Remove any think blocks that leaked into answer
    think_block_re = re.compile(
        re.escape(cfg.think_start_tag) + r"\s*.*?\s*" + re.escape(cfg.think_end_tag),
        flags=re.I | re.S,
    )
    gen_ans_sanitized = think_block_re.sub("", gen_ans_raw)

    over_cap = len(gen_ans_sanitized) > cap
    gen_ans = re.sub(
        r"\s+", " ", gen_ans_sanitized[:cap] if over_cap else gen_ans_sanitized
    ).strip()

    # Compute similarities
    A, B = _tokenize_set(gen_ans), _tokenize_set(ref_ans)
    jac = (
        1.0
        if (not A and not B)
        else (0.0 if (not A or not B) else len(A & B) / len(A | B))
    )
    cos = _tfidf_cosine(gen_ans, ref_ans)

    base = (1.0 - sem_weight) * jac + sem_weight * cos

    # Penalty for excessive length
    penalty = 1.0
    if over_cap:
        ratio = max(0.2, min(1.0, cap / max(1, len(gen_ans_raw))))
        penalty *= ratio

    return float(max(0.0, min(1.0, base * penalty)))


def _contains_phrase(haystack: str, needle: str) -> bool:
    """
    Check if needle phrase appears in haystack with some tolerance.
    """
    if not haystack or not needle:
        return False
    if needle in haystack:
        return True

    # Fallback: check if first two significant tokens appear
    toks = [t for t in needle.split() if len(t) >= 3]
    if len(toks) >= 2:
        short = " ".join(toks[:2])
        return short in haystack
    return False


def choice_correctness_reward(
    text: str, reference_answer_str: Optional[str], cfg: RewardConfig
) -> float:
    """
    Reward for multiple-choice question correctness.
    Expects reference like "A" or "B,C" for multi-select.
    """
    if not reference_answer_str:
        return 0.0

    # Parse reference answer(s)
    ref_set: Set[str] = set()
    for part in reference_answer_str.split(","):
        p = part.strip().upper()
        if len(p) == 1 and p.isalpha():
            ref_set.add(p)

    if not ref_set:
        return 0.0

    # Extract answer region and find letter choices
    ans_region = extract_answer_region(text, cfg).upper()

    # Remove common noise words
    ans_region = re.sub(
        r"\b(OPTION|OPTIONS|ANSWER|CORRECT|IS|ARE|THE|CHOICE|CHOICES)\b",
        "",
        ans_region,
        flags=re.I,
    )

    letters = set(re.findall(r"[A-Z]", ans_region))

    if not letters:
        return 0.0

    # Jaccard similarity between predicted and reference letters
    inter = letters & ref_set
    union = letters | ref_set

    return float(len(inter) / len(union))


def steps_coverage_reward(
    gen_text: str, ref_text: Optional[str], cfg: RewardConfig
) -> float:
    """
    Reward based on coverage of reference reasoning steps.
    Checks both thinking and answer regions.
    """
    if not ref_text:
        return 0.0

    # Combine both regions for comprehensive coverage check
    gen_ans = _strip_markup(extract_answer_region(gen_text or "", cfg))
    gen_think = _strip_markup(extract_think_region(gen_text or "", cfg))
    gen = (gen_ans + " " + gen_think).strip()

    # Extract steps from reference
    ref_steps = _extract_action_phrases(ref_text)[:12]

    if not ref_steps:
        return 0.0

    # Count how many reference steps are covered
    hits = sum(1 for step in ref_steps if _contains_phrase(gen, step))
    coverage = hits / max(1, len(ref_steps))

    # Soft floor: if no exact matches but semantic similarity exists
    if coverage == 0.0:
        if _tfidf_cosine(gen, _strip_markup(ref_text)) >= 0.08:
            return 0.10

    return float(coverage)


def smart_content_reward(
    text: str, reference_completion: Optional[str], cfg: RewardConfig
) -> float:
    """
    Smart content reward: tries step coverage first, falls back to semantic similarity.
    """
    # Try steps coverage first
    s = steps_coverage_reward(text, reference_completion, cfg)
    if s > 0.0:
        return s

    # Fallback: full semantic similarity on combined thinking + answer
    full = (
        _strip_markup(extract_answer_region(text, cfg))
        + " "
        + _strip_markup(extract_think_region(text, cfg))
    ).strip()
    ref = _strip_markup(reference_completion or "")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
        X = vec.fit_transform([full, ref])
        return float(max(0.0, min(1.0, cosine_similarity(X[0:1], X[1:2])[0, 0])))
    except Exception:
        A, B = _tokenize_set(full), _tokenize_set(ref)
        if not A or not B:
            return 0.0
        return float(len(A & B) / len(A | B))


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK-SPECIFIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"([\-+]?\d+(?:[\.,]\d+)?)")


def _to_num_str(s: str) -> Optional[str]:
    """Normalize a numeric string"""
    if not s:
        return None
    s = s.strip().replace(",", "")
    m = _NUM_RE.search(s)
    return m.group(1) if m else None


def _extract_gsm8k_gold(ans: str) -> Optional[str]:
    """Extract answer from GSM8K format (#### NUMBER)"""
    if not ans:
        return None
    m = re.search(r"####\s*([\-+]?\d+(?:[\.,]\d+)?)", ans)
    if m:
        return _to_num_str(m.group(1))
    # Fallback: last number in text
    ms = _NUM_RE.findall(ans)
    return _to_num_str(ms[-1]) if ms else None


def _dataset_keys_for_bench(args, ds_alias: str) -> tuple[str, str]:
    """
    Heuristic prompt/answer keys for common benchmarks.
    """
    if hasattr(args, "benchmark_prompt_key") and hasattr(args, "benchmark_answer_key"):
        if args.benchmark_prompt_key and args.benchmark_answer_key:
            return args.benchmark_prompt_key, args.benchmark_answer_key

    alias = (ds_alias or "").lower()
    if "gsm8k" in alias:
        return "question", "answer"

    # Fallback to training keys
    if hasattr(args, "dataset_prompt_key") and hasattr(args, "dataset_answer_key"):
        return args.dataset_prompt_key, args.dataset_answer_key

    return "prompt", "completion"


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

CONTENT_REWARD_FUNCTIONS: Dict[
    str, Callable[[str, Optional[str], RewardConfig], float]
] = {
    "jaccard": lambda txt, ref, cfg: hybrid_answer_reward(
        txt, ref, cfg, sem_weight=0.3
    ),
    "hybrid": lambda txt, ref, cfg: hybrid_answer_reward(txt, ref, cfg, sem_weight=0.4),
    "choice_correctness": choice_correctness_reward,
    "steps_coverage": steps_coverage_reward,
    "steps": steps_coverage_reward,  # Alias
    "smart": smart_content_reward,
}


def get_content_reward_fn(
    kind: str,
) -> Callable[[str, Optional[str], RewardConfig], float]:
    """Get content reward function by name"""
    fn = CONTENT_REWARD_FUNCTIONS.get(kind.lower())
    if fn is None:
        raise ValueError(
            f"Unknown reward_content_type: '{kind}'. "
            f"Available: {list(CONTENT_REWARD_FUNCTIONS.keys())}"
        )
    return fn


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE REWARD COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_total_reward(
    generated_text: str,
    reference_completion: Optional[str],
    cfg: RewardConfig,
    content_reward_type: str = "smart",
    format_weight: float = 0.05,
    content_weight: float = 0.70,
    think_reward_weight: float = 0.15,
    think_length_weight: float = 0.10,
    think_len_min: int = 16,
    think_len_max: int = 64,
) -> Dict[str, float]:
    """
    Compute comprehensive reward with all components.

    Returns dict with breakdown:
        - format: Format correctness reward
        - content: Answer quality reward
        - think_shorthand: Shorthand/symbolic thinking reward
        - think_length: Length penalty/reward
        - total: Weighted sum
    """

    # 1. Format reward (proper tag usage)
    r_format = format_reward(generated_text, cfg)

    # 2. Content reward (answer quality)
    content_fn = get_content_reward_fn(content_reward_type)
    r_content = content_fn(generated_text, reference_completion, cfg)

    # 3. Shorthand thinking reward (symbolic, compact notation)
    think_text = extract_think_region(generated_text, cfg)
    r_think_shorthand = compute_shorthand_reward(think_text, cfg)

    # 4. Think length reward (target range)
    r_think_length = compute_think_length_reward(
        think_text,
        target_min=think_len_min,
        target_max=think_len_max,
        penalty_strength=0.015,
        penalty_type="exponential",
    )

    # Weighted total
    total = (
        format_weight * r_format
        + content_weight * r_content
        + think_reward_weight * r_think_shorthand
        + think_length_weight * r_think_length
    )

    return {
        "format": r_format,
        "content": r_content,
        "think_shorthand": r_think_shorthand,
        "think_length": r_think_length,
        "total": total,
    }
def make_dynamic_tag_bias_processor(
    tag_ids,
    *,
    mcq_mask=None,
    letter_id_list=None,
    ban_first_ids=None,
    bias_close_think: float = 9.0,
    bias_answer_start: float = 6.0,
    punish_reopen_think: float = -8.0,
    punish_extra_think_end: float = -12.0,
    bias_eos_after_answer: float = 3.0,
    min_answer_tokens: int = 6,  # <= allow a short but non-empty answer
    min_answer_tokens_mcq: int = 1,
    hard_mask_mcq_first_token: bool = True,
    mcq_close_after_k: int = 1,
    mcq_answer_end_bias: float = 10.0,
    mcq_letter_bias: float = 8.0,
    mcq_ban_first_bias: float = -12.0,
):
    te, ts, as_id, ae, eos = (
        tag_ids.get(k)
        for k in ("think_end", "think_start", "answer_start", "answer_end", "eos")
    )
    mcq_mask = list(mcq_mask or [])
    letter_ids = set(letter_id_list or [])
    ban_ids = set(ban_first_ids or [])

    def _last_pos(row, tok):
        if tok is None:
            return -1
        for idx in range(len(row) - 1, -1, -1):
            if row[idx] == tok:
                return idx
        return -1

    def _proc(hist_list, logits):
        if logits is None or logits.ndim != 2:
            return logits
        B, V = logits.shape
        V = int(V)
        # Cast all scalar biases to the logits dtype when used
        neg_inf = mx.array(-1e9, dtype=logits.dtype)

        for i, row_tokens in enumerate(hist_list):
            if not isinstance(row_tokens, (list, tuple)):
                continue

            te_count = row_tokens.count(te) if te is not None else 0
            ae_seen = (ae in row_tokens) if ae is not None else False

            if te is not None and te_count == 0:
                logits = logits.at[i, te].set(
                    logits[i, te] + mx.array(bias_close_think, dtype=logits.dtype)
                )
                if as_id is not None:
                    logits = logits.at[i, as_id].set(
                        logits[i, as_id]
                        + mx.array(bias_answer_start, dtype=logits.dtype)
                    )
            else:
                if te is not None:
                    logits = logits.at[i, te].set(
                        logits[i, te]
                        + mx.array(punish_extra_think_end, dtype=logits.dtype)
                    )
                if ts is not None:
                    logits = logits.at[i, ts].set(
                        logits[i, ts]
                        + mx.array(punish_reopen_think, dtype=logits.dtype)
                    )

            if ae_seen and eos is not None:
                logits = logits.at[i, eos].set(
                    logits[i, eos] + mx.array(bias_eos_after_answer, dtype=logits.dtype)
                )

            last_as = _last_pos(row_tokens, as_id)
            last_ae = _last_pos(row_tokens, ae)
            inside_answer = (last_as != -1) and (last_ae < last_as)
            if not inside_answer:
                continue

            k_inside = len(row_tokens) - (last_as + 1)
            is_mcq = (i < len(mcq_mask)) and bool(mcq_mask[i])

            # MCQ first token: hard-mask to letters; also ban "Insufficient..."
            if is_mcq and hard_mask_mcq_first_token and k_inside == 0 and letter_ids:
                # full -inf row
                row = mx.full_like(logits[i, :], neg_inf)

                # enable letter ids at 0.0
                for t in letter_ids:
                    if 0 <= t < V:
                        row = row.at[t].set(mx.array(0.0, dtype=row.dtype))

                # ban first-token forbidden ids (e.g., "Insufficient...")
                for t in ban_ids:
                    if 0 <= t < V:
                        row = row.at[t].set(
                            mx.array(mcq_ban_first_bias, dtype=row.dtype)
                        )

                # add positive bias to letters
                if mcq_letter_bias != 0.0:
                    for t in letter_ids:
                        if 0 <= t < V:
                            row = row.at[t].set(
                                row[t] + mx.array(mcq_letter_bias, dtype=row.dtype)
                            )

                # write the masked row back
                logits = logits.at[i, :].set(row)

            # Non-MCQ first token: softly discourage starting with banned phrases
            elif (not is_mcq) and k_inside == 0 and ban_ids:
                for t in ban_ids:
                    if 0 <= t < V:
                        logits = logits.at[i, t].set(
                            logits[i, t]
                            + mx.array(mcq_ban_first_bias, dtype=logits.dtype)
                        )

            # MCQ quick close
            if is_mcq and ae is not None and k_inside >= mcq_close_after_k:
                logits = logits.at[i, ae].set(
                    logits[i, ae] + mx.array(mcq_answer_end_bias, dtype=logits.dtype)
                )

            # Minimum length before allowing </answer>
            local_min = min_answer_tokens_mcq if is_mcq else min_answer_tokens
            if ae is not None and k_inside < local_min:
                logits = logits.at[i, ae].set(
                    logits[i, ae] - mx.array(8.0, dtype=logits.dtype)
                )

        return logits

    return _proc
def _preview(text: str, max_len: int) -> str:
    """Return preview of text with ellipsis if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


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
        logging.error(f"apply_chat_template failed: {e}. Fallback.")
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt else ""
        return f"{prefix}User: {prompt.strip()}\n\nAssistant:"
