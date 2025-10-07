# file_path: mlx_rl_trainer/src/mlx_rl_trainer/data/dataset_manager.py
# revision_no: 001
# goals_of_writing_code_block: Manage dataset loading, preprocessing, and efficient batching using async I/O and handling various data formats.
# type_of_code_response: add new code
"""
Data pipeline management: Loading, preprocessing, and efficient batching.
Leverages AsyncIO and implicit mmap for speed.
(SRP: Handles data access and formatting only)
"""
import asyncio
import aiofiles
import json
import logging
import re
import string
import random # For shuffling indices
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Sequence, Tuple
import mlx.core as mx
from datasets import Dataset, Features, Value, load_dataset # Leveraging HuggingFace Datasets library
from tqdm.auto import tqdm # For progress bars

from ..core.config import DataConfig # For configuration
from ..core.trainer import DataLoadError # For custom exception
from mlx_lm.tokenizer_utils import TokenizerWrapper # For explicit tokenizer operations

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Central dataset management and preprocessing.
    (OCP: easily extendable for different loader types)
    """

    def __init__(self, config: DataConfig, tokenizer: TokenizerWrapper):
        """
        Initializes the DatasetManager.
        
        Args:
            config: The validated DataConfig instance.
            tokenizer: The MLX tokenizer instance to be used for tokenization.
            This tokenizer is initially a mock and will be updated in trainer.setup().
        """
        self.config = config
        self._tokenizer = tokenizer # This will be updated with the actual model tokenizer in trainer.setup()
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        
        self.max_prompt_len = self.config.max_prompt_len
        self.max_gen_len = self.config.max_gen_len
        
        # Store system prompt here if it is needed for data processing (e.g. chat template)
        self.system_prompt: str = "" # To be set by trainer.setup() (from ExperimentConfig)
        
        logger.info("DatasetManager initialized, awaiting real tokenizer from ModelManager.")

    def set_tokenizer(self, tokenizer: TokenizerWrapper):
        """Updates the tokenizer instance, typically after a model has been loaded."""
        if not isinstance(tokenizer, TokenizerWrapper):
            raise ValueError(f"Expected TokenizerWrapper, got {type(tokenizer)}")
        self._tokenizer = tokenizer
        logger.debug(f"DatasetManager tokenizer updated to {tokenizer.__class__.__name__}.")

    def set_system_prompt(self, system_prompt: str):
        """Updates the system prompt to be used during data preprocessing."""
        self.system_prompt = system_prompt
        logger.debug("DatasetManager system prompt set.")

    async def _async_read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Reads a JSONL file asynchronously, enabling efficient I/O."""
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        data = []
        try:
            async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
                async for line in f:
                    try:
                        obj = json.loads(line.strip())
                        data.append(obj)
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping malformed JSONL line in {path.name}.")
                        # Handle malformed lines gracefully, continue processing others.
                logger.debug(f"Successfully read {len(data)} records from {path.name}.")
        except Exception as e:
            logger.error(f"Async JSONL read failed for {path}: {e}", exc_info=True)
            raise DataLoadError(f"Async JSONL read failed for {path}") from e
        return data

    def _normalize_record(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalizes a raw dataset record into a standard dictionary format expected by the trainer.
        Extracts 'prompt', 'completion', 'system', and 'test_cases'.
        """
        def _s(x: Any) -> str:
            if x is None: return ''
            try: return str(x)
            except Exception: return ''

        # Prioritize 'prompt', then 'question' for prompt_text
        prompt_text = obj.get(self.config.dataset_prompt_key, obj.get('prompt', obj.get('question', ''))) 
        # Prioritize 'completion', then 'answer' for completion_text
        completion_text = obj.get(self.config.dataset_answer_key, obj.get('completion', obj.get('answer', '')))
        system_text = obj.get('system', "") # System message, if any

        # Ensure test_cases is a list of JSON strings if present, otherwise an empty list
        test_cases_raw = obj.get('test_cases', [])
        if not isinstance(test_cases_raw, list):
            test_cases_raw = [test_cases_raw] if test_cases_raw is not None else [] 
            
        test_cases_json_string = []
        for tc in test_cases_raw:
            try:
                # If tc is already a dict, dump it. If raw string, just use it.
                test_cases_json_string.append(json.dumps(tc) if isinstance(tc, dict) else str(tc))
            except Exception as e:
                logger.warning(f"Could not convert test case to string/JSON for record: {tc}. Error: {e}")
                continue # Skip this problematic test case

        # Basic record validation: ensure at least prompt or completion exists
        if not _s(prompt_text).strip() and not _s(completion_text).strip():
            logger.debug(f"Skipping empty record in normalization: {list(obj.keys())}")
            return None
            
        return {
            'prompt': _s(prompt_text),
            'completion': _s(completion_text),
            'system': _s(system_text),
            'test_cases': test_cases_json_string, # Stored as JSON strings, `RewardContext` will parse into dicts
            'is_invalid_sample': obj.get('is_invalid_sample', False) # Flag for robustness training, if present
        }

    def load_datasets(self) -> None:
        """
        Loads training and validation datasets from configured paths.
        Supports local JSONL/JSON files and HuggingFace datasets.
        Applies tokenization via `_preprocess_function`.
        """
        logger.info("Starting data loading and preprocessing...")
        
        train_path = self.config.train_path
        val_path = self.config.val_path
        
        # Use asyncio event loop for async file operations
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError: # Handle RuntimeError if no event loop is set (e.g., in a non-async thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def load_all_data():
            train_rows = []
            if train_path.suffix.lower() in ['.jsonl', '.ndjson']:
                if train_path.exists(): train_rows = await self._async_read_jsonl(train_path)
            elif train_path.suffix.lower() == '.json' and train_path.exists():
                train_rows = json.loads(await aiofiles.open(train_path, 'r', encoding='utf-8').read())
            else: # Assume HuggingFace dataset
                logger.info(f"Training data path '{train_path}' not found locally. Attempting HuggingFace Dataset load.")
                train_rows = await asyncio.to_thread(load_dataset, str(train_path), split=self.config.dataset_train_split)
                if hasattr(train_rows, 'to_list'): train_rows = train_rows.to_list() # Ensure is list of dicts

            val_rows = []
            if val_path:
                if val_path.suffix.lower() in ['.jsonl', '.ndjson']:
                    if val_path.exists(): val_rows = await self._async_read_jsonl(val_path)
                elif val_path.suffix.lower() == '.json' and val_path.exists():
                    val_rows = json.loads(await aiofiles.open(val_path, 'r', encoding='utf-8').read())
                else: # Assume HuggingFace dataset
                    logger.info(f"Validation data path '{val_path}' not found locally. Attempting HuggingFace Dataset load.")
                    val_rows = await asyncio.to_thread(load_dataset, str(val_path), split=self.config.dataset_val_split)
                    if hasattr(val_rows, 'to_list'): val_rows = val_rows.to_list()

            return train_rows, val_rows

        raw_train_data, raw_val_data = loop.run_until_complete(load_all_data())
        
        logger.debug(f"Raw train data records: {len(raw_train_data)}, raw val data records: {len(raw_val_data)}")

        # Normalize records and create HuggingFace Dataset objects
        self._train_dataset = Dataset.from_list([rec for rec in (self._normalize_record(obj) for obj in raw_train_data) if rec])
        self._train_dataset = self._train_dataset.filter(lambda x: x is not None, batched=False, desc="Filtering invalid train records")
        self._train_dataset = self._train_dataset.map(self._preprocess_function, batched=True, num_proc=self.config.num_workers, desc="Tokenizing train data")
        logger.info(f"Loaded training dataset with {len(self._train_dataset)} records.")

        if raw_val_data:
            self._val_dataset = Dataset.from_list([rec for rec in (self._normalize_record(obj) for obj in raw_val_data) if rec])
            self._val_dataset = self._val_dataset.filter(lambda x: x is not None, batched=False, desc="Filtering invalid val records")
            self._val_dataset = self._val_dataset.map(self._preprocess_function, batched=True, num_proc=self.config.num_workers, desc="Tokenizing val data")
            logger.info(f"Loaded validation dataset with {len(self._val_dataset)} records.")

        logger.info("All datasets loaded and preprocessed.")

    def _preprocess_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[mx.array]]: # Return type specifies mx.array list
        """
        Tokenizes and prepares data batch for model input.
        This function is designed to be used with `dataset.map()`.
        """
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not set in DatasetManager. Call set_tokenizer() first.")
        
        # Combine system, prompt, and completion into a single chat template string for tokenization
        batched_text_to_tokenize = []
        for i in range(len(examples['prompt'])):
            system_part = f"{examples['system'][i]}\n" if examples['system'][i].strip() else ""
            # Use make_chat_template to apply the tokenizer's chat template
            # This is robust to different models' chat formats
            chat_messages = []
            if system_part: chat_messages.append({"role": "system", "content": system_part.strip()})
            chat_messages.append({"role": "user", "content": examples['prompt'][i].strip()})

            # The apply_chat_template is complex, use the helper from mlx_lm
            formatted_input = self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True # Add assistant turn for generation
            )
            batched_text_to_tokenize.append(formatted_input)

        # Tokenize the combined text
        # mlx_lm's tokenizer supports batch_encode_plus
        tokenized_output = self._tokenizer.batch_encode_plus(
            batched_text_to_tokenize,
            max_length=self.config.max_prompt_len,
            padding='max_length', # Pad to max_length
            truncation=True,
            return_attention_mask=True,
            return_tensors='np' # Return numpy arrays which HF Dataset can handle
        )
        
        # Convert numpy arrays to MLX arrays here for direct use in the trainer's dataloader loop
        return {
            'input_ids': tokenized_output['input_ids'].tolist(), # Convert back to list of int
            'attention_mask': tokenized_output['attention_mask'].tolist(), # Convert back to list of int
            'raw_prompts': examples['prompt'], # Keep raw prompts
            'raw_completions': examples['completion'], # Keep raw completions
            'raw_test_cases': examples['test_cases'], # Keep raw test_cases (json strings)
            'is_invalid_sample': examples['is_invalid_sample']
        }


    def get_dataloader(self, split: str, batch_size: int) -> Iterator[Dict[str, List[Any]]]:
        """
        Provides an iterator (dataloader) for batches of processed data.
        
        Args:
            split (str): 'train' or 'val'.
            batch_size (int): The desired batch size.
            
        Yields:
            Dict[str, List[Any]]: A dictionary representing a batch of data,
                                 with token IDs and masks as lists of integers,
                                 and raw data fields as lists of Python types.
        """
        ds = self._train_dataset if split == 'train' else self._val_dataset
        if ds is None or len(ds) == 0:
            logger.warning(f"Dataloader for '{split}' is empty or not loaded. Returning empty iterator.")
            return iter([])

        # Indices for shuffling and batching
        indices = list(range(len(ds)))
        if split == 'train' and self.config.shuffle_data:
            random.shuffle(indices)

        pbar_desc = f"Batching {split} data"
        
        for i in tqdm(range(0, len(indices), batch_size), desc=pbar_desc, leave=False):
            batch_indices = indices[i:i + batch_size]
            data_batch = ds.select(batch_indices).to_dict() # Efficient slicing with HF Datasets

            # Convert input_ids and attention_mask to MLX arrays here
            data_batch['input_ids'] = mx.array(data_batch['input_ids'], dtype=mx.int32)
            data_batch['attention_mask'] = mx.array(data_batch['attention_mask'], dtype=mx.int32) # Usually mx.int32
            
            yield data_batch
