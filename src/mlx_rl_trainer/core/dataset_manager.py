import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from datasets import Dataset, Features, Value, load_dataset
from mlx_rl_trainer.core.config import DataConfig, THINK_STYLE_PROMPT_LITERAL
from mlx_rl_trainer.utils.text_utils import _contains_keywords

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (from user's provided code)
# ═══════════════════════════════════════════════════════════════════════════════


def _ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    ascii_cnt = sum(1 for ch in s if 32 <= ord(ch) <= 126 or ch in "\n\r\t")
    return ascii_cnt / max(1, len(s))


def _looks_garbage(s: str) -> bool:
    if not s:
        return True
    if _ascii_ratio(s) < 0.75:
        return True
    if len(s) > 20000:
        return True
    if len(s.strip()) < 3:
        return True
    # excessive weird symbols
    bad = re.findall(r"[^\w\s\-\.\,\:\;\(\)\[\]\/\+\=\&\<\>]", s)
    return (len(bad) / max(1, len(s))) > 0.15


def _normalize_record(
    obj: Dict[str, Any], prompt_key: str, completion_key: str
) -> Optional[Dict[str, Any]]:
    """
    Normalizes a record from a dataset into a standard dictionary format.

    Args:
        obj: The input object, expected to be a dictionary.
        prompt_key: The primary key for the prompt text.
        completion_key: The primary key for the completion text.

    Returns:
        A dictionary with 'prompt', 'completion', and 'system' keys,
        or None if the object is not a dictionary or is empty.
    """
    if not isinstance(obj, dict):
        return None

    def _s(x: Any) -> str:
        """Safely convert any value to a string."""
        if x is None:
            return ""
        try:
            return str(x)
        except Exception:
            return ""

    prompt = obj.get(prompt_key, obj.get("prompt", obj.get("question", "")))
    completion = obj.get(completion_key, obj.get("completion", obj.get("response", "")))

    completion = (
        _s(completion)
        .replace("<think>\n<think>\n", "<think>")
        .replace("</think>\n</think>", "</think>")
        .replace("<think>\n\n<think>", "<think>")
    )

    # If completion lacks think tags, wrap in minimal structure
    if completion and "<think>" not in completion:
        completion = f"<think>\n\n</think>\n{completion}"

    system = obj.get(
        "system", THINK_STYLE_PROMPT_LITERAL
    )  # Use THINK_STYLE_PROMPT_LITERAL from config

    rec = {
        "prompt": _s(prompt),
        "completion": _s(completion),
        "system": _s(system),
        "is_invalid_sample": obj.get("is_invalid_sample"),
        "test_cases": obj.get("test_cases", []),
        "meta": obj.get("meta", {}),
    }

    # Return None if all fields are empty after normalization
    if not rec["prompt"] and not rec["completion"] and not rec["system"]:
        return None

    return rec


def _load_jsonl_normalized(
    path: Path, prompt_key: str, completion_key: str
) -> "Dataset":
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = _normalize_record(obj, prompt_key, completion_key)
            rec["is_invalid_sample"] = random.choice([True, False])
            if rec is not None:
                rows.append(rec)
    features = Features(
        {
            "prompt": Value("string"),
            "completion": Value("string"),
            "system": Value("string"),
            "is_invalid_sample": Value("bool"),
            "test_cases": Value("string"),  # Assuming test_cases are stringified JSON
            "meta": Value("string"),  # Assuming meta is stringified JSON
        }
    )
    return Dataset.from_list(rows, features=features)


def filter_and_prepare_dataset(
    ds: Dataset, filter_keywords: Optional[Sequence[str]] = None
) -> Dataset:
    """
    Drop rows with garbage prompt/completion. For MCQ, enforce sane options/correctness.
    Optionally skip rows containing any of ``filter_keywords``.
    Returns a new HF Dataset.
    """
    rows: List[Dict[str, Any]] = []
    for ex in ds:
        p = ex.get("prompt") or ex.get("question") or ""
        p = f"{p}\nDo not over think or assume anything, feel free to ask followup questions without responding if anything is ambhigious!"

        c = ex.get("completion") or ex.get("response") or ex.get("answer") or ""
        c = (
            c.replace("**other**\n", "\n")
            .replace("<thinking>", "<think>")
            .replace("</thinking>", "/<think>")
            .replace("<think>\n\n", "<think>")
            .replace("<answer>", "")
            .replace("</answer>", "")
            .replace("<think>\n", "<think>")
            .replace("<think>\n\n", "<think>")
            .replace("<think>\n", "<think>")
        )

        if filter_keywords and (
            _contains_keywords(p, filter_keywords)
            or _contains_keywords(c, filter_keywords)
        ):
            continue
        if _looks_garbage(p):
            continue
        # small completions allowed (we train to generate)
        if len(p) < 5:
            continue

        meta = ex.get("meta", {}) if isinstance(ex.get("meta"), dict) else {}
        type_hint = (
            str(meta.get("type", "")).lower().strip()
            if isinstance(meta.get("type"), str)
            else ""
        )
        if type_hint == "mcq" or isinstance(meta.get("options"), list):
            opts = meta.get("options") if isinstance(meta.get("options"), list) else []
            if not (isinstance(opts, list) and len(opts) >= 2 and len(opts) <= 12):
                # keep as QA if the prompt itself contains textual 'Choices:' block
                if "Choices:" not in p:
                    continue
            # ensure some correctness info
            has_corr = bool(meta.get("correct_indices")) or bool(
                meta.get("correct_texts")
            )
            if (not has_corr) and "Choices:" not in p:
                continue

        rows.append(ex)

    return Dataset.from_list(rows)


def get_dataset(
    split: str,
    config: DataConfig,
) -> Optional[Dataset]:
    """
    Load dataset from JSONL paths or the HuggingFace hub.
    """
    assert split in ("train", "val")
    path = (
        Path(config.train_path)
        if split == "train"
        else (Path(config.val_path) if config.val_path else None)
    )

    # Local file path takes precedence
    if path and path.exists():
        ext = path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            ds = _load_jsonl_normalized(
                path, config.dataset_prompt_key, config.dataset_answer_key
            )
        elif ext in (".json",):
            # JSON (list of dicts)
            try:
                rows_raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(rows_raw, dict):
                    rows_raw = rows_raw.get("data", [])
                if not isinstance(rows_raw, list):
                    rows_raw = []
                rows = [
                    r
                    for r in (
                        _normalize_record(
                            x, config.dataset_prompt_key, config.dataset_answer_key
                        )
                        for x in rows_raw
                    )
                    if r
                ]
                ds = Dataset.from_list(rows)
            except Exception as e:
                logging.error(f"Failed to read JSON dataset {path}: {e}")
                ds = Dataset.from_list([])
        else:
            logging.error(f"Unsupported dataset extension '{ext}' for {path}")
            ds = Dataset.from_list([])
        logging.info(f"[data] Loaded {split} dataset from {path} -> {len(ds)} rows")
        return ds

    # HuggingFace hub fallback
    if config.dataset_name:
        try:
            hf_split = (
                config.dataset_train_split
                if split == "train"
                else config.dataset_val_split
            )
            ds = load_dataset(
                config.dataset_name, config.dataset_config, split=hf_split
            )
            # Normalize columns
            rows = []
            for ex in ds:
                rec = _normalize_record(
                    ex, config.dataset_prompt_key, config.dataset_answer_key
                )
                if rec:
                    rows.append(rec)
            ds = Dataset.from_list(rows)
            logging.info(
                f"[data] Loaded {split} dataset from hub {config.dataset_name} ({hf_split}) -> {len(ds)} rows"
            )
            return ds
        except Exception as e:
            logging.error(f"HF dataset load failed: {e}", exc_info=True)
            return Dataset.from_list([])

    logging.warning(f"No dataset source specified for split='{split}'.")
    return Dataset.from_list([])
