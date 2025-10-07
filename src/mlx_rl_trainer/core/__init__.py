# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/__init__.py
# revision_no: 002
# goals_of_writing_code_block: Remove obsolete imports of get_dataset and filter_and_prepare_dataset from dataset_manager to resolve ImportError.
# type_of_code_response: change existing
"""Core training infrastructure."""
from .config import (
    ExperimentConfig,
    RewardConfig,
    EvaluatorConfig,
    DataConfig,
    ModelConfig,
    TrainerParams,
)
from .trainer import (
    BaseTrainer,
    TrainingMetrics,
    EvaluationMetrics,
    CustomBaseException,
    ModelLoadError,
    InvalidConfigurationError,
    DataLoadError,
    CheckpointError,
    TrainingRuntimeError,
)
from .model_manager import ModelManager
from .dataset_manager import DatasetManager  # Import only the class

__all__ = [
    "ExperimentConfig",
    "RewardConfig",
    "EvaluatorConfig",
    "DataConfig",
    "ModelConfig",
    "TrainerParams",
    "BaseTrainer",
    "TrainingMetrics",
    "EvaluationMetrics",
    "CustomBaseException",
    "ModelLoadError",
    "InvalidConfigurationError",
    "DataLoadError",
    "CheckpointError",
    "TrainingRuntimeError",
    "ModelManager",
    "DatasetManager",
    "CheckpointManager",
]
