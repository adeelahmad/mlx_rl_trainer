# Project Updates and New Features

This document outlines the significant updates, new configurations, scripts, and features introduced in the MLX RL Trainer. It also provides guidance on how to enable and use these new capabilities, with a focus on backwards compatibility.

## Summary of Key Changes

### 1. Dependency Updates (`setup.py`)
- **Added**: `wandb`, `pandas`, `matplotlib`, `scikit-learn` to `install_requires`.
- **Purpose**: These additions enable Weights & Biases (W&B) logging, advanced data manipulation, plotting capabilities for metrics, and machine learning utilities (e.g., for reward functions like TF-IDF).
- **Backwards Compatibility**: Fully backwards compatible. Existing installations will require `pip install -e .` or `pip install -r requirements.txt` to update dependencies.

### 2. Advanced GRPO Trainer Logic (`src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py`)
- **Features**: Confirmed the integration of dual-gradient accumulation, hybrid RL+SFT training, adaptive SFT weights, adaptive layer-wise gradient scaling, and constrained thinking during rollout generation.
- **Purpose**: These features enhance training stability, efficiency, and the ability to fine-tune specific aspects of the model's behavior (e.g., thinking vs. answering).
- **Usage**: These features are controlled via parameters in the `ExperimentConfig` (e.g., `trainer.alternate_dual_gradients`, `trainer.sft_think_loss_weight`, `trainer.boost_answer_grad_layers`, `trainer.min_think_tokens`). Refer to the `ExperimentConfig` for detailed parameter descriptions and sensible defaults.
- **Backwards Compatibility**: Fully backwards compatible. Existing configurations will use default values for new parameters, maintaining previous behavior unless explicitly configured.

### 3. Robust Checkpoint Management (`src/mlx_rl_trainer/core/checkpoint_manager.py`)
- **Features**: Confirmed retry-with-backoff logic for `save_checkpoint` operations and terminal alerting (using `rich.print`) for critical failures (e.g., disk full, permissions issues).
- **Purpose**: Improves the reliability of checkpoint saving, preventing data loss due to transient errors and providing immediate feedback on critical issues.
- **Backwards Compatibility**: Fully backwards compatible. This is an internal enhancement that does not require configuration changes.

### 4. Enhanced Configuration Options (`src/mlx_rl_trainer/core/config.py`)
- **`DataConfig` Additions**:
    - `data_validation_script_path` (Optional[Path]): Path to an external Python script for custom data validation.
    - `data_validation_strict_mode` (bool): If `True`, samples failing validation are discarded. If `False`, they are marked as invalid but kept.
- **`RewardConfig` Flexibility**: Confirmed that the `config: Dict[str, Any]` field within `RewardConfig` allows for flexible parameterization of new reward functions like `VerifyResponseReward` (e.g., `similarity_weight`, `verification_weight`, `verification_mode`).
- **Other Configurations**: Confirmed the presence of new configuration fields related to evaluation, monitoring, memory management, and trainer parameters (dual gradients, SFT, adaptive weights, thinking constraints) with sensible defaults.
- **Usage**:
    - To use custom data validation, specify `data_validation_script_path` in your `ExperimentConfig` and set `data_validation_strict_mode` as desired. The script should contain a `validate_sample(sample: Dict[str, Any]) -> bool` function.
    - Parameters for `VerifyResponseReward` are passed via the `config` dictionary in its `RewardConfig` entry.
- **Backwards Compatibility**: Fully backwards compatible. New fields have defaults that maintain existing behavior.

### 5. Data Pipeline Improvements (`src/mlx_rl_trainer/data/dataset_manager.py`)
- **Features**: Confirmed enhanced `_aggressive_memory_cleanup` calls with explicit `gc.collect()`, refined chunking logic in `_async_read_jsonl` and `_process_raw_to_dataset` for memory efficiency, and integrated logic to load and run custom data validation scripts.
- **Purpose**: Improves memory management during data loading and processing, and allows for flexible, user-defined data quality control.
- **Usage**: Custom data validation is enabled via `DataConfig` parameters (`data_validation_script_path`, `data_validation_strict_mode`).
- **Backwards Compatibility**: Fully backwards compatible. Internal optimizations and optional features.

### 6. Trainer Enhancements (`src/mlx_rl_trainer/core/trainer.py`)
- **Features**: Confirmed integration of periodic evaluation calls, hooks to track `mx.metal` memory usage (`allocated_mb`, `peak_mb`, `cache_mb`) at key stages, and calls to the plot generation function from `MetricsLogger` before each checkpoint save.
- **Purpose**: Provides better insights into training progress, resource utilization, and automates plot generation for easier analysis.
- **Usage**: Memory tracking is enabled via `trainer.log_memory_usage` in `ExperimentConfig`. Evaluation frequency is controlled by `trainer.eval_every`.
- **Backwards Compatibility**: Fully backwards compatible. These are enhancements to the training loop that do not alter core behavior unless configured.

### 7. Generator Updates (`src/mlx_rl_trainer/generation/generator.py`)
- **Features**: Confirmed logic to track generated token lengths for thinking and answer regions via `_create_thinking_answer_masks`.
- **Purpose**: Enables more granular analysis of the model's generation process, particularly for thought-based reasoning.
- **Backwards Compatibility**: Fully backwards compatible. This is an internal enhancement for metrics.

### 8. Comprehensive Metrics Logging (`src/mlx_rl_trainer/monitoring/metrics_logger.py`)
- **Features**: Confirmed W&B logging for metrics, charts, and samples, periodic `_aggressive_memory_cleanup`, and on-demand calling of `_emit_plots_from_csv` by the trainer.
- **Purpose**: Centralized and robust logging solution for experiment tracking, visualization, and resource management.
- **Usage**: W&B logging is enabled via `monitoring.wandb_log` in `ExperimentConfig`.
- **Backwards Compatibility**: Fully backwards compatible.

### 9. New Reward Function: `VerifyResponseReward` (`src/mlx_rl_trainer/rewards/content/verify_response.py`)
- **New File**: `src/mlx_rl_trainer/rewards/content/verify_response.py`
- **Features**: Implements a new `VerifyResponseReward` class that checks `sample['verified_answer_str']` from `RewardContext.metadata`. Supports multiple `verification_modes` (`str_exact_match`, `str_normalized_match`, `math_eval_match`, `python_script_output_based`) and configurable `similarity_weight` and `verification_weight`.
- **Purpose**: Allows for direct reward signals based on the correctness of the generated response against a ground truth, crucial for tasks requiring factual accuracy or specific output formats.
- **Usage**: Add a `RewardConfig` entry for `name: "verify_response"` in your `ExperimentConfig`, and configure its `config` dictionary with `similarity_weight`, `verification_weight`, and `verification_mode`. Ensure your dataset provides `verified_answer_str` in the sample metadata.
- **Backwards Compatibility**: New feature, does not affect existing reward configurations.

### 10. New Script: `dump_config.py` (`src/mlx_rl_trainer/scripts/dump_config.py`)
- **New File**: `src/mlx_rl_trainer/scripts/dump_config.py`
- **Features**: A script to load `ExperimentConfig` and output its default values to a YAML file.
- **Purpose**: Facilitates easy generation of a base configuration file, which users can then modify.
- **Usage**: Run `python src/mlx_rl_trainer/scripts/dump_config.py <output_file.yaml>`.
- **Backwards Compatibility**: New script, no impact on existing workflows.

### 11. Standalone Evaluation Script (`src/mlx_rl_trainer/scripts/evaluate.py`)
- **Features**: Confirmed the standalone model evaluation script.
- **Purpose**: Provides a dedicated tool for evaluating trained models outside the main training loop.
- **Usage**: Run `python src/mlx_rl_trainer/scripts/evaluate.py --config <config_file.yaml> --model_path <path_to_model>`.
- **Backwards Compatibility**: Fully backwards compatible.

### 12. Training Script Updates (`src/mlx_rl_trainer/scripts/train.py`)
- **Features**: Confirmed `wandb` initialization and the `limit_memory` utility.
- **Purpose**: Ensures proper W&B integration and allows for memory limits to be set for the training process.
- **Usage**: W&B is configured via `monitoring` section in `ExperimentConfig`. Memory limits are set via `mlx_utils.limit_memory`.
- **Backwards Compatibility**: Fully backwards compatible.

### 13. MLX Utility Functions (`src/mlx_rl_trainer/utils/mlx_utils.py`)
- **Features**: Confirmed the `limit_memory` function.
- **Purpose**: Provides a utility to set memory limits for MLX, helping to manage resource consumption.
- **Usage**: Called internally by the training script.
- **Backwards Compatibility**: Fully backwards compatible.

### 14. New Script: `data_validation.py` (`src/mlx_rl_trainer/scripts/data_validation.py`)
- **New File**: `src/mlx_rl_trainer/scripts/data_validation.py`
- **Features**: An example external Python script defining a `validate_sample(sample: Dict[str, Any]) -> bool` function.
- **Purpose**: Serves as a template and example for users to implement their own custom data validation logic.
- **Usage**: Refer to this file when implementing a custom validation script and specify its path in `DataConfig.data_validation_script_path`.
- **Backwards Compatibility**: New script, no impact on existing workflows.

## Backwards Compatibility Statement

All new features and modifications have been implemented with a strong focus on backwards compatibility.
- **Configuration**: New configuration fields have sensible default values, meaning existing `ExperimentConfig` files will continue to function without modification, adopting the new default behaviors. To utilize new features, users will need to explicitly add or modify these fields in their configuration.
- **Codebase**: Existing code paths and functionalities remain unchanged unless explicitly enhanced. Internal optimizations do not require user-level code changes.
- **Dependencies**: New dependencies are additive. Users should update their environment to include them for full functionality.

Users are encouraged to review the `ExperimentConfig` structure and the new scripts to leverage the full potential of these updates.