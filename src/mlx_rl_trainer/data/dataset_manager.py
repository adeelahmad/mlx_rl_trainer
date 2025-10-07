"""
Data pipeline management: Loading, preprocessing, and efficient batching.
"""
import asyncio, aiofiles, json, logging, random, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple
import mlx.core as mx
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from ..core.config import DataConfig, THINK_STYLE_PROMPT_LITERAL
from ..core.trainer import DataLoadError, TrainingRuntimeError
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.utils.text_utils import (
    _contains_keywords,
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
)

logger = logging.getLogger(__name__)


def _ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    return sum(1 for ch in s if 32 <= ord(ch) <= 126 or ch in "\n\r\t") / max(1, len(s))


def _looks_garbage(s: str) -> bool:
    if not s or len(s.strip()) < 3 or len(s) > 20000 or _ascii_ratio(s) < 0.75:
        return True
    bad = re.findall(r"[^\w\s\-\.\,\:\;\(\)\[\]\/\+\=\&\<\>]", s)
    return (len(bad) / max(1, len(s))) > 0.15


def _normalize_record(
    obj: Dict[str, Any], prompt_key: str, completion_key: str, system_prompt: str
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    def _s(x: Any) -> str:
        return str(x) if x is not None else ""

    prompt = _s(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = _s(
        obj.get(completion_key, obj.get("completion", obj.get("response", "")))
    )

    # Fix #1: Ensure test_cases is always a list
    test_cases = obj.get("test_cases", [])
    if not isinstance(test_cases, list):
        test_cases = [test_cases] if test_cases else []

    rec = {
        "prompt": prompt,
        "completion": completion,
        "system": _s(obj.get("system", system_prompt)),
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "test_cases": test_cases,
        "meta": obj.get("meta", {}),
    }
    if not rec["prompt"] and not rec["completion"]:
        return None

    mcq_meta = _mcq_meta_from_sample(
        {"prompt": rec["prompt"], "completion": rec["completion"], "meta": rec["meta"]}
    )
    rec["meta"].update(mcq_meta)
    return rec


class DatasetManager:
    """Manages loading, filtering, tokenizing, and batching of datasets."""

    def __init__(self, config: DataConfig, tokenizer: Optional[TokenizerWrapper]):
        self.config = config
        self._tokenizer = tokenizer
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        self.system_prompt: str = ""
        logger.debug("DatasetManager initialized.")

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        if not isinstance(tokenizer, TokenizerWrapper):
            raise ValueError(f"Expected TokenizerWrapper, got {type(tokenizer)}")
        self._tokenizer = tokenizer
        logger.debug(f"DatasetManager tokenizer updated.")

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        logger.debug("DatasetManager system prompt set.")

    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        data = []
        async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
            async for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Malformed JSONL line in {path.name}, skipping."
                        )
        return data

    async def load_datasets(self, force_reload: bool = False):
        if self._is_loaded and not force_reload:
            return
        loop = asyncio.get_event_loop()

        async def load_raw_data_for_split(
            path: Path, split_name: str
        ) -> List[Dict[str, Any]]:
            if not path:
                return []
            if path.suffix.lower() in [".jsonl", ".ndjson"]:
                return await self._async_read_jsonl(path)
            elif path.suffix.lower() == ".json":
                return json.loads(
                    await aiofiles.open(path, "r", encoding="utf-8").read()
                )
            else:
                hf_split = getattr(
                    self.config, f"dataset_{split_name}_split", split_name
                )
                dataset_obj = await asyncio.to_thread(
                    load_dataset, path.as_posix(), split=hf_split
                )
                return (
                    dataset_obj.to_list()
                    if hasattr(dataset_obj, "to_list")
                    else list(dataset_obj)
                )

        raw_train_data = await load_raw_data_for_split(self.config.train_path, "train")
        raw_val_data = (
            await load_raw_data_for_split(self.config.val_path, "val")
            if self.config.val_path
            else []
        )

        self._train_dataset = self._process_raw_to_dataset(raw_train_data, "train")
        self._val_dataset = (
            self._process_raw_to_dataset(raw_val_data, "val") if raw_val_data else None
        )

        self._is_loaded = True
        logger.info(
            f"Datasets loaded. Train samples: {len(self._train_dataset)}, Val samples: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _process_raw_to_dataset(
        self, raw_data: List[Dict[str, Any]], split_name: str
    ) -> Dataset:
        normalized_records = []
        for obj in tqdm(raw_data, desc=f"Normalizing {split_name} data"):
            rec = _normalize_record(
                obj,
                self.config.dataset_prompt_key,
                self.config.dataset_answer_key,
                self.system_prompt,
            )
            if (
                rec
                and not _looks_garbage(rec["prompt"])
                and not (
                    _contains_keywords(
                        rec["prompt"], self.config.dataset_filter_keywords
                    )
                    or _contains_keywords(
                        rec["completion"], self.config.dataset_filter_keywords
                    )
                )
            ):
                normalized_records.append(rec)

        if not normalized_records:
            logger.warning(
                f"No valid records found for {split_name} after normalization and filtering."
            )
            return Dataset.from_list([])

        return Dataset.from_list(normalized_records)

    def _prepare_batch(self, raw_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise TrainingRuntimeError(
                "Tokenizer must be set before preparing batches."
            )

        prepared_batch = {
            "tokens": [],
            "text": [],
            "ref_answer_str": [],
            "ref_think_str": [],
            "is_invalid_sample": [],
            "is_mcq": [],
            "mcq_options": [],
            "mcq_correct_letters": [],
            "mcq_multi_select": [],
            "meta": [],
            "raw_test_cases": [],
        }

        for record in raw_batch:
            formatted_prompt = apply_chat_template_wrapper(
                self.tokenizer,
                record["prompt"],
                record.get("system", self.system_prompt),
            )
            encoded_ids = self.tokenizer.encode(
                formatted_prompt, add_special_tokens=True
            )
            if len(encoded_ids) > self.config.max_prompt_len:
                encoded_ids = encoded_ids[-self.config.max_prompt_len :]

            prepared_batch["tokens"].append(mx.array(encoded_ids, dtype=mx.int32))
            prepared_batch["text"].append(formatted_prompt)
            prepared_batch["ref_answer_str"].append(record["completion"])
            prepared_batch["ref_think_str"].append(
                ""
            )  # This should be derived from completion if needed
            prepared_batch["is_invalid_sample"].append(
                record.get("is_invalid_sample", False)
            )

            meta = record.get("meta", {})
            prepared_batch["is_mcq"].append(meta.get("is_mcq", False))
            prepared_batch["mcq_options"].append(meta.get("mcq_options", []))
            prepared_batch["mcq_correct_letters"].append(
                meta.get("mcq_correct_letters", "")
            )
            prepared_batch["mcq_multi_select"].append(
                meta.get("mcq_multi_select", False)
            )
            prepared_batch["raw_test_cases"].append(record.get("test_cases", []))
            prepared_batch["meta"].append(meta)  # Full meta for context

        return prepared_batch

    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, Any]]:
        dataset = self._train_dataset if split == "train" else self._val_dataset
        if not dataset or len(dataset) == 0:
            return iter([])

        indices = list(range(len(dataset)))
        if self.config.shuffle_data and split == "train":
            random.shuffle(indices)

        def batch_generator():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                raw_batch_list = dataset.select(batch_indices).to_list()
                if raw_batch_list:
                    try:
                        yield self._prepare_batch(raw_batch_list)
                    except Exception as e:
                        logger.error(
                            f"Error preparing batch {i//batch_size}: {e}. Skipping batch.",
                            exc_info=True,
                        )

        return batch_generator()
