# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/dataset_manager.py
# revision_no: 001
# goals_of_writing_code_block: Manage dataset loading, preprocessing, and efficient batching using async I/O and handling various data formats.
# type_of_code_response: add new code
"""
Data pipeline management: Loading, preprocessing, and efficient batching.
"""
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import logging
import asyncio
import aiofiles
import json
import mlx.core as mx

from datasets import (
    Dataset,
    Features,
    Value,
    load_dataset,
)  # Explicitly import datasets
from mlx_lm.tokenizer_utils import (
    TokenizerWrapper,
)  # Explicitly import TokenizerWrapper

from ..core.config import DataConfig  # Correct import
from ..core.trainer import DataLoadError  # Correct import
from rich import print as rprint


class DatasetManager:
    def __init__(self, config: DataConfig, tokenizer: Optional[TokenizerWrapper]):
        self.config = config
        self.tokenizer = tokenizer  # Tokenizer can be set later

        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None

        logging.info(f"DatasetManager initialized with config: {config}")

    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Asynchronously reads a JSONL file."""
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        data: List[Dict[str, Any]] = []
        try:
            async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            logging.warning(
                                f"Skipping malformed JSONL line in {path}: {line[:80]}..."
                            )
        except Exception as e:
            raise DataLoadError(f"Async JSONL read failed for {path}: {e}") from e
        return data

    async def _async_load_hf_dataset(
        self, name: str, config: Optional[str], split: str
    ) -> List[Dict[str, Any]]:
        """Asynchronously loads a Hugging Face dataset."""
        try:
            ds = load_dataset(name, config, split=split)
            return ds.to_list()
        except Exception as e:
            raise DataLoadError(
                f"Async HF dataset load failed for {name}/{config} ({split}): {e}"
            ) from e

    async def load_datasets(self) -> None:
        """Loads training and validation datasets based on configuration."""
        rprint("[bold blue]Starting asynchronous data load...[/bold blue]")
        train_path = self.config.train_path
        val_path = self.config.val_path

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def load_task(
            path: Path, loader_type: str, split: str
        ) -> List[Dict[str, Any]]:
            if loader_type == "jsonl":
                return await self._async_read_jsonl(path)
            elif loader_type == "hf_dataset":
                return await self._async_load_hf_dataset(
                    str(path), None, split
                )  # Path as name
            elif loader_type == "mock":
                # Create mock data directly
                num_samples = 100 if split == "train" else 20
                return [
                    {
                        "prompt": f"mock_prompt_{i} for {split}",
                        "completion": f"mock_completion_{i} for {split}",
                        "test_cases": [],
                    }
                    for i in range(num_samples)
                ]
            else:
                raise DataLoadError(f"Unknown loader type: {loader_type}")

        tasks = [load_task(train_path, self.config.loader_type, "train")]
        if val_path and val_path.exists():
            tasks.append(load_task(val_path, self.config.loader_type, "val"))
        elif self.config.loader_type == "mock":
            tasks.append(load_task(Path("mock_val_path"), "mock", "val"))

        results = await asyncio.gather(*tasks)
        train_raw_data = results[0]
        val_raw_data = results[1] if len(results) > 1 else []

        # Define flexible features for dataset creation
        features = Features(
            {
                self.config.dataset_prompt_key: Value("string"),
                self.config.dataset_answer_key: Value("string"),
                "test_cases": [
                    Value("string")
                ],  # Assume test_cases are lists of JSON strings
                "is_mcq": Value("bool"),  # Add for future MCQ handling
                "mcq_options": [Value("string")],
                "mcq_correct_letters": Value("string"),
                "mcq_multi_select": Value("bool"),
            }
        )

        self._train_dataset = Dataset.from_list(
            [
                {
                    self.config.dataset_prompt_key: r.get(
                        self.config.dataset_prompt_key, ""
                    ),
                    self.config.dataset_answer_key: r.get(
                        self.config.dataset_answer_key, ""
                    ),
                    "test_cases": [json.dumps(tc) for tc in r.get("test_cases", [])],
                    "is_mcq": r.get("is_mcq", False),
                    "mcq_options": r.get("mcq_options", []),
                    "mcq_correct_letters": r.get("mcq_correct_letters", ""),
                    "mcq_multi_select": r.get("mcq_multi_select", False),
                }
                for r in train_raw_data
            ],
            features=features,
        )

        # Apply filtering
        self._train_dataset = self._filter_and_prepare_dataset(
            self._train_dataset, self.config.dataset_filter_keywords
        )
        rprint(
            f"Loaded training dataset with {len(self._train_dataset)} records ({len(self._train_dataset.filter(lambda x: x['is_invalid_sample'] == False))} valid)."
        )

        if val_raw_data:
            self._val_dataset = Dataset.from_list(
                [
                    {
                        self.config.dataset_prompt_key: r.get(
                            self.config.dataset_prompt_key, ""
                        ),
                        self.config.dataset_answer_key: r.get(
                            self.config.dataset_answer_key, ""
                        ),
                        "test_cases": [
                            json.dumps(tc) for tc in r.get("test_cases", [])
                        ],
                        "is_mcq": r.get("is_mcq", False),
                        "mcq_options": r.get("mcq_options", []),
                        "mcq_correct_letters": r.get("mcq_correct_letters", ""),
                        "mcq_multi_select": r.get("mcq_multi_select", False),
                    }
                    for r in val_raw_data
                ],
                features=features,
            )
            self._val_dataset = self._filter_and_prepare_dataset(
                self._val_dataset, self.config.dataset_filter_keywords
            )
            rprint(
                f"Loaded validation dataset with {len(self._val_dataset)} records ({len(self._val_dataset.filter(lambda x: x['is_invalid_sample'] == False))} valid)."
            )

    def _filter_and_prepare_dataset(
        self, dataset: Dataset, filter_keywords: List[str]
    ) -> Dataset:
        """
        Filters out samples containing specified keywords and adds an 'is_invalid_sample' flag.
        """

        def _filter_fn(example):
            prompt_content = example.get(self.config.dataset_prompt_key, "")
            completion_content = example.get(self.config.dataset_answer_key, "")
            text_content = prompt_content + " " + completion_content
            # Check if any filter keyword is present (case-insensitive)
            if any(k.lower() in text_content.lower() for k in filter_keywords):
                example["is_invalid_sample"] = True
            else:
                example["is_invalid_sample"] = False
            return example

        return dataset.map(_filter_fn)

    def get_dataloader(
        self, split: str, batch_size: int, shuffle: Optional[bool] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Returns an iterator for batches of data.
        """
        ds = self._train_dataset if split == "train" else self._val_dataset
        if ds is None:
            raise DataLoadError(f"Dataset '{split}' not loaded.")
        if len(ds) == 0:
            return iter([])

        indices = list(range(len(ds)))
        current_shuffle = shuffle if shuffle is not None else self.config.shuffle_data
        if split == "train" and current_shuffle:
            import random

            random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Use .select() to get a view, then .to_dict() to materialize for processing
            raw_batch_data = ds.select(batch_indices).to_dict()

            # Check for empty batch after selection
            if not raw_batch_data.get(self.config.dataset_prompt_key):
                continue

            for sample in self._process_batch(raw_batch_data):
                yield sample

    def _process_batch(self, raw_batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Processes a raw batch of data, tokenizing prompts and preparing other fields.
        Returns a list of dictionaries, one for each processed sample.
        """
        if self.tokenizer is None:
            raise DataLoadError("Tokenizer not set in DatasetManager.")

        processed_samples = []
        num_samples_in_batch = len(raw_batch.get(self.config.dataset_prompt_key, []))

        for i in range(num_samples_in_batch):
            prompt = raw_batch.get(self.config.dataset_prompt_key, [])[i]
            completion = raw_batch.get(self.config.dataset_answer_key, [])[i]
            test_cases_raw = raw_batch.get(
                "test_cases", [[] for _ in range(num_samples_in_batch)]
            )[i]

            tokenized_input_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
            )

            # Process test cases from JSON strings back to dicts
            current_processed_test_cases = []
            for tc_json_str in test_cases_raw:
                try:
                    current_processed_test_cases.append(json.loads(tc_json_str))
                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping malformed test case: {tc_json_str[:50]}..."
                    )

            processed_samples.append(
                {
                    "input_ids": mx.array(tokenized_input_ids, dtype=mx.int32),
                    "raw_prompt": prompt,  # Store the raw prompt string
                    "raw_completion": completion,  # Store the raw completion string
                    "raw_test_cases": current_processed_test_cases,
                    "is_mcq": raw_batch.get("is_mcq", [False] * num_samples_in_batch)[
                        i
                    ],
                    "mcq_options": raw_batch.get(
                        "mcq_options", [[] for _ in range(num_samples_in_batch)]
                    )[i],
                    "mcq_correct_letters": raw_batch.get(
                        "mcq_correct_letters", [""] * num_samples_in_batch
                    )[i],
                    "mcq_multi_select": raw_batch.get(
                        "mcq_multi_select", [False] * num_samples_in_batch
                    )[i],
                    "is_invalid_sample": raw_batch.get(
                        "is_invalid_sample", [False] * num_samples_in_batch
                    )[i],
                    "original_index": raw_batch.get(
                        "original_index", list(range(num_samples_in_batch))
                    )[i],
                    # Store the entire original raw_batch entry for this sample if needed for metadata
                    "original_raw_data": {
                        k: v[i]
                        for k, v in raw_batch.items()
                        if isinstance(v, list) and len(v) == num_samples_in_batch
                    },
                }
            )
        return processed_samples
