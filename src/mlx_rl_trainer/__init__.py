# file_path: mlx_rl_trainer/src/mlx_rl_trainer/__init__.py
# revision_no: 001
# goals_of_writing_code_block: Top-level __init__.py for the MLX RL Trainer package.
# type_of_code_response: add new code
"""
MLX RL Trainer - Production-ready reinforcement learning framework
"""
__version__ = "0.1.0"

# Expose key components for easier imports
from mlx_rl_trainer.core.trainer import (
    BaseTrainer,
    TrainingMetrics,
    EvaluationMetrics,
    CustomBaseException,
    ModelLoadError,
    DataLoadError,
    CheckpointError,
    InvalidConfigurationError,
    TrainingRuntimeError,
)
from mlx_rl_trainer.core.config import (
    ExperimentConfig,
    RewardConfig,
    EvaluatorConfig,
    DataConfig,
    ModelConfig,
    TrainerParams,
)
from mlx_rl_trainer.core.model_manager import ModelManager
from mlx_rl_trainer.core.dataset_manager import get_dataset, filter_and_prepare_dataset
from mlx_rl_trainer.core.checkpoint_manager import CheckpointManager
from mlx_rl_trainer.rewards.registry import RewardRegistry, register_reward
from mlx_rl_trainer.rewards.base_reward import RewardComposer
from mlx_rl_trainer.rewards.context import RewardContext
from mlx_rl_trainer.evaluation.registry import EvaluatorRegistry

__all__ = [
    "BaseTrainer",
    "TrainingMetrics",
    "EvaluationMetrics",
    "CustomBaseException",
    "ModelLoadError",
    "DataLoadError",
    "CheckpointError",
    "InvalidConfigurationError",
    "TrainingRuntimeError",
    "ExperimentConfig",
    "RewardConfig",
    "EvaluatorConfig",
    "DataConfig",
    "ModelConfig",
    "TrainerParams",
    "ModelManager",
    "get_dataset",
    "filter_and_prepare_dataset",
    "CheckpointManager",
    "RewardRegistry",
    "register_reward",
    "RewardComposer",
    "RewardContext",
    "EvaluatorRegistry",
]
