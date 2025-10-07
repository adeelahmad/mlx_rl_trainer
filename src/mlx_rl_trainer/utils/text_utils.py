"""Text processing utility functions."""
import logging, re, string, json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Callable, Union
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.nn as nn
from mlx_rl_trainer.core.config import GenerationConfig, ExperimentConfig
import mlx.core as mx

logger = logging.getLogger(__name__)
LETTER_ALPH = string.ascii_uppercase

def _preview(s: str, n: int = 600) -> str:
    if s is None: return ""
    s = s.replace("\r\n", "\n")
    s = s[:n] + ("..." if len(s) > n else "")
    return s.replace("\n", "\\n")

def _strip_markup(s: str) -> str:
    if not s: return ""
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"`[^`]+`", " ", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", s, flags=re.M)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^\w\s/:%\-.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _count_words(txt: str) -> int:
    return len(re.findall(r"\w+", txt or ""))

def _tokenize_set(s: str) -> Set[str]:
    s = (s or "").lower().translate(str.maketrans("", "", string.punctuation))
    return set(w for w in s.split() if w)

def _normalize_ans_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _contains_keywords(haystack: str, keywords: Sequence[str]) -> bool:
    if not haystack or not keywords: return False
    s_low = haystack.lower()
    return any(k.lower() in s_low for k in keywords)

def _has_non_ascii(s: str) -> bool:
    return any(ord(ch) > 127 for ch in s or "")

def extract_think_region(text: str, gen_config: GenerationConfig) -> str:
    m = re.search(re.escape(gen_config.think_start_tag) + r"\s*(.*?)\s*" + re.escape(gen_config.think_end_tag), text or "", flags=re.I | re.S)
    return (m.group(1).strip() if m else "")

def extract_answer_region(text: str, gen_config: GenerationConfig) -> str:
    tl = text or ""
    tend = gen_config.think_end_tag
    if tend and tend.lower() in tl.lower():
        idx = tl.lower().rfind(tend.lower())
        return tl[idx + len(tend):].strip()
    return tl.strip()

def _extract_think_answer_lengths(text: str, gen_config: GenerationConfig) -> Tuple[int, int]:
    think_content = extract_think_region(text, gen_config)
    answer_content = extract_answer_region(text, gen_config)
    return len(think_content.strip()), len(answer_content.strip())
    
def _jaccard_similarity(a: str, b: str) -> float:
    A, B = _tokenize_set(a), _tokenize_set(b)
    if not A or not B: return 0.0
    return float(len(A & B) / len(A | B))

def _extract_final_numeric(s: str) -> Optional[str]:
    if not s: return None
    m = re.search(r"####\s*([-]?\d+(?:\.\d+)?)\s*$", s.strip())
    if m: return m.group(1)
    m = re.findall(r"[-]?\d+(?:\.\d+)?", s)
    return m[-1] if m else None

def _indices_to_letters(indices: List[int]) -> str:
    seen, out = set(), []
    for idx in sorted(indices):
        if 0 <= idx < len(LETTER_ALPH):
            letter = LETTER_ALPH[idx]
            if letter not in seen:
                seen.add(letter)
                out.append(letter)
    return ",".join(out)

def _letters_to_canonical(letter_str: str) -> str:
    seen, out = set(), []
    for p in sorted((letter_str or "").upper().split(",")):
        p = p.strip()
        if len(p) == 1 and p in LETTER_ALPH and p not in seen:
            seen.add(p)
            out.append(p)
    return ",".join(out)
    
def _match_ref_to_option_index(ref_text: str, options: List[str]) -> Optional[int]:
    if not (ref_text and options): return None
    ref_n = _normalize_ans_for_match(ref_text)
    for i, opt in enumerate(options):
        if _normalize_ans_for_match(opt) == ref_n: return i
    for i, opt in enumerate(options):
        on = _normalize_ans_for_match(opt)
        if ref_n and (ref_n in on or on in ref_n): return i
    return None

def _extract_mcq_options(prompt_text: str) -> List[str]:
    if not isinstance(prompt_text, str): return []
    m = re.search(r"choices\s*:?(.*)$", prompt_text, flags=re.I | re.S)
    block = m.group(1) if m else prompt_text
    opts = []
    for ln in block.splitlines():
        ln = ln.strip()
        m2 = re.match(r"^\s*(?:[A-Za-z]|\d+)\s*[\)\.\-:]\s*(.+)$", ln)
        if m2: opts.append(m2.group(1).strip())
    return [o for o in opts if o.strip()][:len(LETTER_ALPH)]

def _infer_mcq_ref_letters(sample: Dict[str, Any]) -> str:
    meta = sample.get("meta", {})
    options = sample.get("mcq_options", [])
    ref_ans_text = sample.get("ref_answer_str", "")
    
    # Priority 1: Explicit letters
    for key in ("correct_letters", "correct_letter"):
        if (val := meta.get(key)) and isinstance(val, str): return _letters_to_canonical(val)
    
    # Priority 2: Explicit indices
    indices = []
    if (val := meta.get("correct_indices")) and isinstance(val, list):
        try: indices = [int(x) for x in val]
        except (ValueError, TypeError): pass
    elif (val := meta.get("correct_index")) is not None:
        try: indices = [int(val)]
        except (ValueError, TypeError): pass
    if indices: return _indices_to_letters(indices)

    # Priority 3: Match reference text to options
    if ref_ans_text and options:
        idx = _match_ref_to_option_index(ref_ans_text, options)
        if idx is not None: return _indices_to_letters([idx])
    return ""


def _mcq_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    prompt = sample.get("text", "")
    meta = sample.get("meta", {})
    options = meta.get("options", _extract_mcq_options(prompt))
    
    is_mcq = (meta.get("type", "").lower() == "mcq") or (isinstance(options, list) and len(options) >= 2)
    if not is_mcq: return {"is_mcq": False}
    
    sample_with_opts = {**sample, "mcq_options": options}
    correct_letters = _infer_mcq_ref_letters(sample_with_opts)
    correct_indices = [LETTER_ALPH.index(L) for L in correct_letters.split(",") if L in LETTER_ALPH]
    
    return {
        "is_mcq": True,
        "mcq_options": options,
        "mcq_multi_select": len(correct_indices) > 1,
        "mcq_correct_indices": correct_indices,
        "mcq_correct_letters": correct_letters,
    }

def apply_chat_template_wrapper(tokenizer: TokenizerWrapper, prompt: str, system_prompt: Optional[str]) -> str:
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prefix = f"System: {system_prompt.strip()}\n\n" if system_prompt else ""
        return f"{prefix}User: {prompt.strip()}\n\nAssistant:"
