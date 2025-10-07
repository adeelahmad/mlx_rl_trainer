# file_path: mlx_rl_trainer/src/mlx_rl_trainer/core/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Initialize the core module, making key abstractions easily importable.
# type_of_code_response: add new code
"""Core abstractions that define the trainer's architecture."""
from .config import TrainerConfig, RewardConfig, EvaluatorConfig, DataConfig, ModelConfig, TrainerParams
from .trainer import BaseTrainer, TrainingMetrics, EvaluationMetrics, CustomBaseException, ModelLoadError, InvalidConfigurationError, DataLoadError, CheckpointError
from .model_manager import ModelManager
from .dataset_manager import DatasetManager
from .checkpoint_manager import CheckpointManager

__all__ = [
    "TrainerConfig", "RewardConfig", "EvaluatorConfig", "DataConfig", "ModelConfig", "TrainerParams",
    "BaseTrainer", "TrainingMetrics", "EvaluationMetrics", "CustomBaseException", "ModelLoadError", "InvalidConfigurationError", "DataLoadError", "CheckpointError",
    "ModelManager", "DatasetManager", "CheckpointManager"
]
