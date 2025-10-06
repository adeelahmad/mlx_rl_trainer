# file_path: mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/__init__.py
# revision_no: 001
# goals_of_writing_code_block: __init__.py for the GRPO algorithm module.
# type_of_code_response: add new code
"""GRPO algorithm module init."""
from .grpo_trainer import GRPOTrainer
from .grpo_algorithm import GRPOAlgorithm

__all__ = ["GRPOTrainer", "GRPOAlgorithm"]
