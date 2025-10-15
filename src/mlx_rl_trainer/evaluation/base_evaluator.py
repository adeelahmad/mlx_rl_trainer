from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import json
from pathlib import Path
from datasets import Dataset, load_dataset
from mlx_lm.tokenizer_utils import TokenizerWrapper

from mlx_rl_trainer.core.types import EvaluationMetrics
from mlx_rl_trainer.core.exceptions import DataLoadError

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.dataset: Optional[Dataset] = None
        logger.debug(f"Initialized {self.name} evaluator with config: {config}")

    def load_dataset(
        self,
        dataset_path: str,
        dataset_subset: Optional[str] = None,
        split: str = "test",
    ) -> Dataset:
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
        self, model, tokenizer: TokenizerWrapper, dataset: Dataset
    ) -> EvaluationMetrics:
        """
        Runs the evaluation for a given model and tokenizer on the loaded dataset.

        Args:
            model: The MLX model to evaluate.
            tokenizer: The tokenizer associated with the model.
            dataset: The dataset to evaluate on.

        Returns:
            An EvaluationMetrics object containing the results.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
