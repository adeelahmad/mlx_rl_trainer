import asyncio
import gc
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib.util

import mlx.core as mx
from datasets import Dataset, Features, Value, load_dataset
from tqdm.auto import tqdm

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
    if not isinstance(obj, dict):
        return None

    prompt = str(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = str(
        obj.get(completion_key, obj.get("completion", obj.get("answer", "")))
    )

    clean_completion = clean_completion_string(completion)

    meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": prompt, "completion": clean_completion, "meta": meta}
    )

    final_meta = mcq_meta.copy()
    final_meta.update({k: v for k, v in meta.items() if k not in mcq_meta})

    if "verifiable_answer_str" in obj:
        final_meta["verifiable_answer_str"] = obj["verifiable_answer_str"]

    # This is a bit of a hack to pass through the original sample for the reward context
    final_meta["_original_sample"] = obj

    return {
        "prompt": prompt,
        "completion": clean_completion,
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "system": obj.get("system", ""),
        "test_cases": obj.get("test_cases", []),
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
        self._processing_chunk_size = 1000
        self.validator_fn = self._load_validator()

    def _load_validator(self):
        script_path = self.config.data_validation_script_path
        if not script_path:
            return None

        if not script_path.exists():
            raise InvalidConfigurationError(
                f"Data validation script not found at: {script_path}"
            )

        try:
            spec = importlib.util.spec_from_file_location("data_validator", script_path)
            validator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validator_module)

            if hasattr(validator_module, "validate_sample"):
                logger.info(f"Loaded custom data validator from {script_path}")
                return getattr(validator_module, "validate_sample")
            else:
                raise InvalidConfigurationError(
                    f"Validator script {script_path} must contain a 'validate_sample' function."
                )
        except Exception as e:
            raise InvalidConfigurationError(
                f"Failed to load data validator from {script_path}: {e}"
            ) from e

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def _aggressive_memory_cleanup(self):
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
        mx.clear_cache()
        gc.collect()

    async def _read_data(self, path: Path, split_name: str) -> List[Dict]:
        if not path:
            return []
        if path.suffix.lower() in [".jsonl", ".ndjson"]:
            import aiofiles

            lines = []
            async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    if line.strip():
                        try:
                            lines.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Skipping malformed JSON line in {path.name}"
                            )
            return lines
        else:
            split = "train" if split_name == "train" else "test"
            dataset = await asyncio.to_thread(load_dataset, str(path), split=split)
            return dataset.to_list()

    async def load_datasets(self, force_reload: bool = False):
        if self._is_loaded and not force_reload:
            return

        train_data = await self._read_data(self.config.train_path, "train")
        self._train_dataset = self._process_raw_to_dataset(train_data, "train")
        del train_data
        self._aggressive_memory_cleanup()

        if self.config.val_path:
            val_data = await self._read_data(self.config.val_path, "val")
            self._val_dataset = self._process_raw_to_dataset(val_data, "val")
            del val_data
            self._aggressive_memory_cleanup()
        else:
            self._val_dataset = None

        self._is_loaded = True
        logger.info(
            f"Datasets loaded. Train: {len(self._train_dataset)}, Val: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _process_raw_to_dataset(self, raw_data: List[Dict], split_name: str) -> Dataset:
        if not raw_data:
            return Dataset.from_list([])

        processed_records = []
        for record in tqdm(raw_data, desc=f"Processing {split_name} data"):
            if self.validator_fn:
                is_valid = self.validator_fn(record)
                if not is_valid:
                    if self.config.data_validation_strict_mode:
                        continue  # Discard sample
                    else:
                        record["is_invalid_sample"] = True

            norm_rec = _normalize_record(
                record, self.config.dataset_prompt_key, self.config.dataset_answer_key
            )
            if norm_rec and not _looks_garbage(norm_rec["prompt"]):
                if not self.config.dataset_filter_keywords or not _contains_keywords(
                    norm_rec["prompt"], self.config.dataset_filter_keywords
                ):
                    processed_records.append(norm_rec)

        if not processed_records:
            return Dataset.from_list([])

        return Dataset.from_list(processed_records)

    def get_dataloader(self, split: str, batch_size: int):
        dataset = self._train_dataset if split == "train" else self._val_dataset
        if not dataset or len(dataset) == 0:
            logger.warning(f"Dataloader for '{split}' is empty.")
            return iter([])

        indices = list(range(len(dataset)))
        if self.config.shuffle_data and split == "train":
            random.shuffle(indices)

        def generator():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                if not batch_indices:
                    continue

                prompts_data, prompts_mx, _ = build_rollout_batch(
                    self._tokenizer, dataset, batch_indices, self.exp_config
                )

                if prompts_mx.size > 0:
                    yield {"prompts_data": prompts_data, "prompts_mx": prompts_mx}

                if i > 0 and (i // batch_size) % 10 == 0:
                    self._aggressive_memory_cleanup()

        return generator()
