"""
Data pipeline management: Loading, preprocessing, and efficient batching.
"""
import json, logging, random, re, asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple, Generator
import aiofiles
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from mlx_rl_trainer.core.config import (
    DataConfig,
    THINK_STYLE_PROMPT_LITERAL,
    GenerationConfig,
)
from mlx_rl_trainer.core.exceptions import (
    DataLoadError,
)  # Import from new exceptions module
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_rl_trainer.utils.text_utils import (
    _contains_keywords,
    _mcq_meta_from_sample,
    apply_chat_template_wrapper,
    extract_think_region,
    _looks_garbage,
)
from mlx_rl_trainer.data.batch_builder import build_rollout_batch
import mlx.core as mx

logger = logging.getLogger(__name__)


def _normalize_record(
    obj: Dict[str, Any],
    prompt_key: str,
    completion_key: str,
    system_prompt_default: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    def _s(x: Any) -> str:
        return str(x) if x is not None else ""

    prompt = _s(obj.get(prompt_key, obj.get("prompt", obj.get("question", ""))))
    completion = _s(
        obj.get(completion_key, obj.get("completion", obj.get("answer", "")))
    )
    system = _s(obj.get("system", system_prompt_default))

    gen_config_default = GenerationConfig()
    completion_cleaned = (
        completion.replace(
            f"{gen_config_default.think_start_tag}\n{gen_config_default.think_start_tag}\n",
            gen_config_default.think_start_tag,
        )
        .replace(
            f"{gen_config_default.think_end_tag}\n{gen_config_default.think_end_tag}",
            gen_config_default.think_end_tag,
        )
        .replace(
            f"{gen_config_default.think_start_tag}\n\n{gen_config_default.think_start_tag}",
            gen_config_default.think_start_tag,
        )
    )
    if (
        completion_cleaned
        and gen_config_default.think_start_tag not in completion_cleaned
    ):
        completion_cleaned = f"{gen_config_default.think_start_tag}\n\n{gen_config_default.think_end_tag}\n{completion_cleaned}"

    meta = obj.get("meta", {}) if isinstance(obj.get("meta"), dict) else {}
    mcq_meta = _mcq_meta_from_sample(
        {"prompt": prompt, "completion": completion_cleaned, "meta": meta}
    )
    meta.update(mcq_meta)

    if not prompt.strip() and not completion_cleaned.strip() and not system.strip():
        return None

    return {
        "prompt": prompt,
        "completion": completion_cleaned,
        "system": system,
        "test_cases": obj.get("test_cases", []),
        "is_invalid_sample": obj.get("is_invalid_sample", False),
        "meta": meta,
    }


class DatasetManager:
    def __init__(self, config: DataConfig, tokenizer: Optional[TokenizerWrapper]):
        self.config = config
        self._tokenizer = tokenizer
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._is_loaded = False
        self.system_prompt: str = ""
        logger.debug("DatasetManager initialized.")

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        self._tokenizer = tokenizer
        logger.debug(f"DatasetManager tokenizer updated.")

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        logger.debug("DatasetManager system prompt set.")

    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
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
                raw_content = await aiofiles.open(
                    path, mode="r", encoding="utf-8"
                ).read()
                return json.loads(raw_content)
            else:
                hf_split_name = "train" if split_name == "train" else "test"
                dataset_obj = await asyncio.to_thread(
                    load_dataset, path.as_posix(), split=hf_split_name
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
            f"Datasets loaded. Train: {len(self._train_dataset)}, Val: {len(self._val_dataset) if self._val_dataset else 0}"
        )

    def _process_raw_to_dataset(
        self, raw_data: List[Dict[str, Any]], split_name: str
    ) -> Dataset:
        normalized_records = []
        for obj in raw_data:
            rec = _normalize_record(
                obj,
                self.config.dataset_prompt_key,
                self.config.dataset_answer_key,
                self.system_prompt,
            )
            if (
                rec
                and not _looks_garbage(rec["prompt"])
                and not _looks_garbage(rec["completion"])
            ):
                if not self.config.dataset_filter_keywords or not (
                    _contains_keywords(
                        rec["prompt"], self.config.dataset_filter_keywords
                    )
                    or _contains_keywords(
                        rec["completion"], self.config.dataset_filter_keywords
                    )
                ):
                    normalized_records.append(rec)
        if not normalized_records:
            logger.warning(f"No valid records found for {split_name}.")
            return Dataset.from_list([])
        return Dataset.from_list(normalized_records)

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
                    self._tokenizer, dataset, batch_indices, self.config
                )

                if prompts_mx.size > 0:
                    yield {
                        "prompts_data": prompts_data,
                        "prompts_mx": prompts_mx,
                    }

        return batch_generator()
