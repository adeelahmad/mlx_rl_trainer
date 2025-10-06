# file_path: mlx_rl_trainer/src/mlx_rl_trainer/evaluation/base_evaluator.py
# revision_no: 001
# goals_of_writing_code_block: Abstract base evaluator interface.
# type_of_code_response: add new code
"""Abstract interface for all evaluation benchmarks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
from datasets import Dataset  # HuggingFace Dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper  # For tokenizer type hinting

from mlx_rl_trainer.core.trainer import (
    EvaluationMetrics,
    DataLoadError,
)  # Import structured metrics

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation benchmarks.

    Subclasses must implement the `evaluate` method to run a specific benchmark
    and return results in a structured `EvaluationMetrics` object.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the evaluator.

        Args:
            config: Evaluator-specific configuration dictionary.
        """
        self.config = config
        self.name = self.__class__.__name__
        self.dataset: Optional[
            Dataset
        ] = None  # Dataset can be loaded by evaluator or passed

        logger.debug(f"Initialized {self.name} evaluator with config: {config}")

    def load_dataset(
        self,
        dataset_path: str,
        dataset_subset: Optional[str] = None,
        split: str = "test",
    ) -> Dataset:
        """
        Loads an evaluation dataset from HuggingFace Hub or a local JSONL file.

        Args:
            dataset_path: The path/name of the dataset (e.g., "gsm8k" or "path/to/data.jsonl").
            dataset_subset: Optional subset of the dataset (e.g., "main").
            split: The dataset split to load (e.g., "test", "validation").

        Returns:
            A HuggingFace `Dataset` object.

        Raises:
            DataLoadError: If the dataset fails to load or is not found.
        """
        try:
            path = Path(dataset_path)
            if path.exists() and path.suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f if line.strip()]
                self.dataset = Dataset.from_list(data)
            else:
                self.dataset = load_dataset(dataset_path, dataset_subset, split=split)

            logger.info(
                f"Loaded dataset for {self.name}: {dataset_path} (split: {split}) with {len(self.dataset)} samples."
            )
            return self.dataset
        except Exception as e:
            raise DataLoadError(
                f"Failed to load dataset for {self.name} from {dataset_path}: {e}"
            ) from e

    @abstractmethod
    def evaluate(
        self, model: Any, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        """
        Abstract method to run evaluation on the model.

        This method should implement the benchmark-specific logic for:
        1. Generating responses from the `model`.
        2. Comparing generated responses to ground truth.
        3. Calculating relevant metrics (e.g., accuracy, pass rate).
        4. Packaging results into an `EvaluationMetrics` object.

        Args:
            model: The MLX model instance to be evaluated.
            tokenizer: The tokenizer associated with the model.
            dataset: The dataset to run evaluation on.

        Returns:
            An `EvaluationMetrics` object containing the results.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
