"""
Data pipeline management: Loading, preprocessing, and efficient batching.
"""
import json
import logging
import random
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

from datasets import Dataset, Features, Value, Sequence
from tqdm.auto import tqdm
import aiofiles
import mlx.core as mx

from mlx_rl_trainer.core.config import ExperimentConfig
from mlx_rl_trainer.core.exceptions import DataLoadError, InvalidConfigurationError
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
from mlx_rl_trainer.utils.text_utils import (
    _contains_keywords,
    _mcq_meta_from_sample,
    clean_completion_string,
    _looks_garbage,
)

logger = logging.getLogger(__name__)


def _normalize_record(
    obj: Dict, prompt_key: str, completion_key: str
) -> Optional[Dict]:
    """
    Normalizes a raw data dictionary into a standardized format. This function
    is the primary defense against schema errors by creating a consistent dictionary structure.
    """
    if not isinstance(obj, dict):
        return None

    prompt = str(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = str(
        obj.get(completion_key, obj.get("completion", obj.get("answer", "")))
    )

    if not prompt.strip():
        return None

    clean_completion = clean_completion_string(completion)
    meta_from_source = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}

    # This creates a dictionary with a consistent set of keys for MCQ data
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": prompt, "completion": clean_completion, "meta": meta_from_source}
    )

    # **THE FIX**: Create a final, standardized meta dictionary for EVERY record.
    # This guarantees that the schema is always the same, preventing Arrow errors.
    final_meta = {
        "is_mcq": mcq_meta.get("is_mcq", False),
        "mcq_options": mcq_meta.get("mcq_options", []),
        "mcq_multi_select": mcq_meta.get("mcq_multi_select", False),
        "mcq_correct_indices": mcq_meta.get("mcq_correct_indices", []),
        "mcq_correct_letters": mcq_meta.get("mcq_correct_letters", ""),
    }

    # Safely merge other original meta keys that don't conflict.
    # We will serialize any complex values (dicts/lists) to prevent type errors.
    for k, v in meta_from_source.items():
        if k not in final_meta:
            if isinstance(v, (dict, list)):
                try:
                    final_meta[k] = json.dumps(v, ensure_ascii=False)
                except (TypeError, OverflowError):
                    final_meta[k] = str(v)
            else:
                final_meta[k] = v

    # Ensure test cases are always a list of strings
    test_cases = obj.get("test_cases", [])
    if not isinstance(test_cases, list):
        test_cases = [test_cases] if test_cases is not None else []
    test_cases_str = [
        json.dumps(tc) if isinstance(tc, dict) else str(tc) for tc in test_cases
    ]

    return {
        "prompt": prompt,
        "completion": clean_completion,
        "system": str(obj.get("system", "")),
        "test_cases": test_cases_str,
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "meta": final_meta,
    }


class DatasetManager:
    def __init__(self, config: ExperimentConfig):
        self.exp_config = config
        self.config = config.data
        self._tokenizer = None
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        self.system_prompt: str = ""
        logger.debug("DatasetManager initialized.")

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def _read_data(self, path: Path) -> List[Dict[str, Any]]:
        if not path:
            return []
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")

        lines = []
        logger.info(f"Robustly parsing JSONL file: {path}")
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in {path.name}")
        return lines

    async def load_datasets(self, force_reload: bool = False):
        if self._is_loaded and not force_reload:
            return

        raw_train_data = await self._read_data(self.config.train_path)
        self._train_dataset = self._process_raw_to_dataset(raw_train_data, "train")

        if self.config.val_path:
            raw_val_data = await self._read_data(self.config.val_path)
            self._val_dataset = self._process_raw_to_dataset(raw_val_data, "val")
        else:
            self._val_dataset = None

        self._is_loaded = True
        logger.info(
            f"Datasets loaded. Train: {len(self._train_dataset)}, Val: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _process_raw_to_dataset(self, raw_data: List[Dict], split_name: str) -> Dataset:
        normalized_records = []
        for obj in tqdm(raw_data, desc=f"Normalizing {split_name} data"):
            rec = _normalize_record(
                obj, self.config.dataset_prompt_key, self.config.dataset_answer_key
            )
            if rec and not _looks_garbage(rec["prompt"]):
                if not self.config.dataset_filter_keywords or not _contains_keywords(
                    rec["prompt"], self.config.dataset_filter_keywords
                ):
                    normalized_records.append(rec)

        if not normalized_records:
            logger.warning(
                f"No valid records found for {split_name}. The dataloader will be empty."
            )
            return Dataset.from_list([])

        # **THE FIX**: Define a consistent schema for PyArrow, including a flexible 'meta' field.
        # By using a dictionary of strings for unknown meta keys, we avoid type clashes.
        features = Features(
            {
                "prompt": Value("string"),
                "completion": Value("string"),
                "system": Value("string"),
                "test_cases": Sequence(Value("string")),
                "is_invalid_sample": Value("bool"),
                "meta": {
                    "is_mcq": Value("bool"),
                    "mcq_options": Sequence(Value("string")),
                    "mcq_multi_select": Value("bool"),
                    "mcq_correct_indices": Sequence(Value("int32")),
                    "mcq_correct_letters": Value("string"),
                },
            }
        )

        # This allows other keys in 'meta' as long as they are strings
        features["meta"].update(
            Features(
                {
                    k: Value("string")
                    for k in normalized_records[0]["meta"]
                    if k not in features["meta"]
                }
            )
        )

        return Dataset.from_list(normalized_records, features=features)

    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, Any]]:
        dataset = self._train_dataset if split == "train" else self._val_dataset
        if not dataset or len(dataset) == 0:
            logger.warning(f"Dataloader for '{split}' is empty.")
            return iter([])

        indices = list(range(len(dataset)))
        if self.config.shuffle_data and split == "train":
            random.shuffle(indices)

        def batch_generator():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                if not batch_indices:
                    continue

                prompts_data, prompts_mx, _ = build_rollout_batch(
                    self._tokenizer, dataset, batch_indices, self.exp_config
                )

                if prompts_mx.size > 0:
                    yield {"prompts_data": prompts_data, "prompts_mx": prompts_mx}

        return batch_generator()
