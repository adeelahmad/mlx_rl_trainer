# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/dataset_manager.py
# revision_no: 003
# goals_of_writing_code_block: Refactor the file to remove obsolete module-level functions (get_dataset, filter_and_prepare_dataset) which caused an ImportError in __init__.py.
# type_of_code_response: change existing
import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Generator, Iterator
import concurrent.futures

from datasets import Dataset, Features, Value, load_dataset
from mlx_rl_trainer.core.config import DataConfig, THINK_STYLE_PROMPT_LITERAL
from mlx_rl_trainer.core.trainer import DataLoadError, TrainingRuntimeError
from mlx_rl_trainer.core.model_manager import TokenizerWrapper
from mlx_rl_trainer.utils.text_utils import (
    _contains_keywords,
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
)
import mlx.core as mx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPER FUNCTIONS (Kept as private module helpers)
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
    bad = re.findall(r"[^\w\s\-\.\,\:\;\(\)\[\]\/\+\=\&\<\>]", s)
    return (len(bad) / max(1, len(s))) > 0.15


def _normalize_record(
    obj: Dict[str, Any], prompt_key: str, completion_key: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    def _s(x: Any) -> str:
        if x is None:
            return ""
        try:
            return str(x)
        except Exception:
            return ""

    prompt = obj.get(prompt_key, obj.get("prompt", obj.get("question", "")))
    completion = obj.get(completion_key, obj.get("completion", obj.get("response", "")))
    system = obj.get("system", "")

    # Clean up completion format
    completion = (
        _s(completion)
        .replace("<think>\n<think>\n", "<think>")
        .replace("</think>\n</think>", "</think>")
        .replace("<think>\n\n<think>", "<think>")
    )
    if completion and "<think>" not in completion:
        completion = f"<think>\n\n</think>\n{completion}"

    system = obj.get("system", THINK_STYLE_PROMPT_LITERAL)

    rec = {
        "prompt": _s(prompt),
        "completion": _s(completion),
        "system": _s(system),
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "test_cases": obj.get("test_cases", []),
        "meta": obj.get("meta", {}),
    }

    if not rec["prompt"] and not rec["completion"]:
        return None

    # Enrich MCQ metadata (requires _mcq_meta_from_sample from text_utils.py to work correctly)
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": rec["prompt"], "completion": rec["completion"], "meta": rec["meta"]}
    )
    rec["meta"].update(mcq_meta)

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET MANAGER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class DatasetManager:
    """
    Manages loading, filtering, tokenizing, and batching of datasets.
    """

    __slots__ = ("config", "tokenizer", "_train_dataset", "_val_dataset", "_is_loaded")

    def __init__(self, config: DataConfig, tokenizer: Optional[TokenizerWrapper]):
        self.config: DataConfig = config
        self.tokenizer: Optional[TokenizerWrapper] = tokenizer
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        logger.debug("DatasetManager initialized.")

    def load_datasets(self, force_reload: bool = False):
        """Loads and pre-filters the training and validation datasets."""
        if self._is_loaded and not force_reload:
            logger.info("Datasets already loaded.")
            return

        self._train_dataset = self._load_and_process_split(
            self.config.train_path, "train"
        )
        if self.config.val_path:
            self._val_dataset = self._load_and_process_split(
                self.config.val_path, "val"
            )

        self._is_loaded = True
        logger.info(
            f"Loaded datasets. Train samples: {len(self._train_dataset) if self._train_dataset else 0}, Val samples: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _load_and_process_split(self, path: Path, split: str) -> Dataset:
        """Helper to load data from a single path (local or HF) and apply filtering."""

        if not path:
            return Dataset.from_dict({})

        ds: Dataset

        # Load logic (synchronous wrapper over common HF/local calls)
        try:
            if path.suffix.lower() in (".json", ".jsonl", ".ndjson"):
                # Local JSON/JSONL file
                ds = load_dataset("json", data_files={"data": str(path)})["data"]
            else:
                # HuggingFace dataset
                hf_split = (
                    self.config.dataset_train_split
                    if split == "train"
                    else self.config.dataset_val_split
                )
                ds = load_dataset(path.as_posix(), split=hf_split)
        except Exception as e:
            logger.error(f"Failed to load dataset from {path} for split {split}: {e}")
            return Dataset.from_dict({})

        # Normalize and filter
        rows = []
        for raw_rec in ds:
            rec = _normalize_record(
                raw_rec, self.config.dataset_prompt_key, self.config.dataset_answer_key
            )
            if rec is None:
                continue

            p, c = rec["prompt"], rec["completion"]

            if _looks_garbage(p) or _looks_garbage(c):
                continue

            if self.config.dataset_filter_keywords and (
                _contains_keywords(p, self.config.dataset_filter_keywords)
                or _contains_keywords(c, self.config.dataset_filter_keywords)
            ):
                continue

            rows.append(rec)

        logger.info(
            f"Loaded {split} split from {path} with {len(rows)} usable samples."
        )
        return Dataset.from_list(rows)

    def _prepare_batch(self, raw_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes a list of raw data dictionaries and prepares them for the trainer.
        Includes chat templating, tokenization, and metadata extraction.
        """
        if self.tokenizer is None:
            raise TrainingRuntimeError(
                "Tokenizer must be set before preparing batches."
            )

        input_ids_list: List[mx.array] = []
        (
            raw_prompts,
            raw_completions,
            raw_test_cases,
            is_invalid_sample_flags,
            meta_data,
        ) = ([], [], [], [], [])

        for record in raw_batch:
            # 1. Apply Chat Template
            system_prompt_to_use = record.get("system") or getattr(
                self.config, "system_prompt", THINK_STYLE_PROMPT_LITERAL
            )
            formatted_prompt = apply_chat_template_wrapper(
                self.tokenizer, record["prompt"], system_prompt_to_use
            )

            # 2. Tokenize and Truncate Prompt
            encoded_ids = self.tokenizer.encode(
                formatted_prompt, add_special_tokens=True
            )
            if len(encoded_ids) > self.config.max_prompt_len:
                encoded_ids = encoded_ids[: self.config.max_prompt_len]

            input_ids_list.append(mx.array(encoded_ids, dtype=mx.int32))

            # 3. Collect Raw Data
            raw_prompts.append(record["prompt"])
            raw_completions.append(record["completion"])
            raw_test_cases.append(record.get("test_cases", []))
            is_invalid_sample_flags.append(record.get("is_invalid_sample", False))
            meta_data.append(record.get("meta", {}))

        return {
            "input_ids": input_ids_list,
            "raw_prompts": raw_prompts,
            "raw_completions": raw_completions,
            "raw_test_cases": raw_test_cases,
            "is_invalid_sample_flags": is_invalid_sample_flags,
            "meta_data": meta_data,
        }

    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, Any]]:
        """
        Returns an iterator that yields batched data dictionaries for one epoch.
        Uses a generator for memory efficiency.
        """
        dataset = self._train_dataset if split == "train" else self._val_dataset

        if dataset is None or len(dataset) == 0:
            logger.warning(f"Attempted to get dataloader for empty split: {split}.")
            return iter([])

        indices = list(range(len(dataset)))
        if self.config.shuffle_data and split == "train":
            random.shuffle(indices)

        # Generator function uses an anonymous inner function for clear iterator definition
        def batch_generator() -> Generator[Dict[str, Any], None, None]:
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]

                raw_batch_list = dataset.select(batch_indices).to_list()

                if not raw_batch_list:
                    return

                try:
                    prepared_batch = self._prepare_batch(raw_batch_list)
                    yield prepared_batch
                except Exception as e:
                    logger.error(
                        f"Error preparing batch {i//batch_size}: {e}. Skipping batch.",
                        exc_info=True,
                    )
                    continue

        return batch_generator()
