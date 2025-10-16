# MLX RL Trainer

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Data Preprocessing](#data-preprocessing)
  - [Dumping Configuration](#dumping-configuration)
- [Configuration](#configuration)
  - [Overview](#overview)
  - [ExperimentConfig](#experimentconfig)
  - [TrainerParams](#trainerparams)
  - [ModelConfig](#modelconfig)
  - [DataConfig](#dataconfig)
  - [GenerationConfig](#generationconfig)
  - [RewardConfig](#rewardconfig)
  - [EvaluatorConfig](#evaluatorconfig)
  - [CheckpointConfig](#checkpointconfig)
  - [MonitoringConfig](#monitoringconfig)
- [CLI Commands](#cli-commands)
  - [`train.py`](#trainpy)
  - [`evaluate.py`](#evaluatepy)
  - [`data_preprocessing.py`](#data_preprocessingpy)
  - [`data_validation.py`](#data_validationpy)
  - [`dump_config.py`](#dump_configpy)
- [Dataset Examples](#dataset-examples)
- [Extensibility](#extensibility)
  - [Adding New Rewards](#adding-new-rewards)
  - [Adding New Evaluators](#adding-new-evaluators)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The MLX RL Trainer is a robust and flexible framework for training Large Language Models (LLMs) using Reinforcement Learning (RL) techniques, specifically focusing on algorithms like GRPO (Generalized Reinforcement Policy Optimization) and PPO (Proximal Policy Optimization). Built on MLX, it provides efficient model training, comprehensive configuration options, and detailed monitoring capabilities.

## Features
- **Flexible RL Algorithms:** Supports GRPO and PPO for fine-tuning Large Language Models (LLMs).
- **Pydantic-based Configuration:** Ensures strict validation and predictability of all training, model, data, and generation parameters.
- **Modular Reward System:** Easily define and integrate custom reward functions.
- **Extensible Evaluation Framework:** Add new evaluators to assess model performance across various metrics.
- **WandB Integration:** Seamless logging and visualization of training metrics and generated samples.
- **Advanced Memory Optimizations:**
    - **Gradient Checkpointing:** Reduces memory footprint during training.
    - **Paged KV Cache:** Efficiently manages KV cache memory during generation.
    - **Aggressive Tensor Cleanup:** Minimizes memory usage during rollout generation.
- **LoRA Support:** Efficient fine-tuning with Low-Rank Adaptation.
- **Sophisticated Generation Controls:**
    - **Thinking/Answer Masking:** Fine-grained control over token generation for structured reasoning.
    - **Dynamic Biasing:** Encourages or penalizes specific tokens/phrases during generation.
    - **Constrained Thinking:** Enforces minimum and maximum thinking token lengths.
    - **Thinking Penalties & Bonuses:** Rewards efficient thinking and penalizes excessive verbosity.
- **Hybrid RL+SFT Training:** Combines Reinforcement Learning with Supervised Fine-Tuning for robust model updates.
    - **Dual Gradient Accumulation:** Accumulates separate gradients for thinking and answer portions.
    - **Adaptive SFT Weights:** Dynamically adjusts SFT loss weights based on KL divergence.
    - **Adaptive Layer-wise Gradient Scaling:** Boosts gradients for specific model layers (e.g., answer-focused layers).
    - **Configurable SFT Layer Control:** Apply SFT to all layers, answer layers only, or with weighted layer groups.
- **Cross-Architecture Reference Model Alignment:** Supports aligning reference models from different architectures.
- **Data Preprocessing & Validation:** Tools for preparing and ensuring the quality of your training data.

## Installation
To get started with MLX RL Trainer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/mlx_rl_trainer.git
    cd mlx_rl_trainer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Training
To start a training run, you'll typically use the `train.py` script with a configuration file.

```bash
python src/mlx_rl_trainer/scripts/train.py --config configs/experiments/code_gen_base.yaml
```

### Evaluation
Evaluate a trained model using the `evaluate.py` script.

```bash
python src/mlx_rl_trainer/scripts/evaluate.py --config configs/experiments/code_gen_base.yaml --checkpoint_path /path/to/your/checkpoint
```

### Data Preprocessing
Prepare your datasets using the `data_preprocessing.py` script.

```bash
python src/mlx_rl_trainer/scripts/data_preprocessing.py --input_path /path/to/raw_data.jsonl --output_path /path/to/processed_data.jsonl --config configs/experiments/code_gen_base.yaml
```

### Dumping Configuration
To inspect the effective configuration, including defaults and overrides, use `dump_config.py`.

```bash
python src/mlx_rl_trainer/scripts/dump_config.py --config configs/experiments/code_gen_base.yaml --output_path effective_config.yaml
```

## Configuration
The project uses a Pydantic-based configuration system, allowing for robust validation and clear definition of all parameters. Configurations are typically loaded from YAML files.

### Overview
The main configuration class is `ExperimentConfig`, which aggregates several sub-configurations:
- `trainer`: Parameters for the RL training process.
- `model`: Model-specific settings, including LoRA.
- `data`: Dataset paths and processing options.
- `generation`: Text generation and sampling parameters.
- `rewards`: List of reward functions and their weights.
- `evaluation`: List of evaluators to run.
- `checkpointing`: Checkpoint saving strategy.
- `monitoring`: Weights & Biases logging settings.

### ExperimentConfig
The top-level configuration for an entire experiment.

| Field | Type | Description | Default |
|---|---|---|---|
| `use_grad_checkpointing` | `bool` | Enable gradient checkpointing to save memory. | `True` |
| `grad_checkpoint_layers` | `PositiveInt` | Number of layers to apply gradient checkpointing to. | `1` |
| `trainer` | `TrainerParams` | See [TrainerParams](#trainerparams) section. | (Required) |
| `model` | `ModelConfig` | See [ModelConfig](#modelconfig) section. | (Required) |
| `generation` | `GenerationConfig` | See [GenerationConfig](#generationconfig) section. | (Default instance) |
| `rewards` | `List[RewardConfig]` | See [RewardConfig](#rewardconfig) section. | `[]` |
| `data` | `DataConfig` | See [DataConfig](#dataconfig) section. | (Required) |
| `evaluation` | `List[EvaluatorConfig]` | See [EvaluatorConfig](#evaluatorconfig) section. | `[]` |
| `checkpointing` | `CheckpointConfig` | See [CheckpointConfig](#checkpointconfig) section. | (Default instance) |
| `monitoring` | `MonitoringConfig` | See [MonitoringConfig](#monitoringconfig) section. | (Default instance) |
| `max_kv_size` | `PositiveInt` | Maximum KV cache size for generation. | `1536` |
| `system_prompt` | `str` | The system prompt used for generation, defining thinking rules and format. | (Detailed internal string) |
| `ban_phrases_for_bias` | `List[str]` | Phrases to penalize during generation. | (Extensive list) |
| `encourage_phrases_for_bias` | `List[str]` | Phrases to encourage during generation. | (Extensive list) |
| `encourage_think_bias` | `float` | Bias value to encourage thinking tokens. | `4.5` |
| `ban_think_bias` | `float` | Bias value to penalize thinking tokens. | `-3.0` |
| `allow_tool_calls` | `bool` | Whether to allow tool calls during generation. | `True` |
| `tool_call_penalty` | `NonNegativeFloat` | Penalty for generating tool calls. | `0.0` |
| `think_length_target_min` | `PositiveInt` | Minimum target length for thinking tokens. | `8` |
| `think_length_target_max` | `PositiveInt` | Maximum target length for thinking tokens. | `64` |
| `think_length_penalty_strength` | `NonNegativeFloat` | Strength of the penalty for thinking token length. | `0.8` |
| `use_paged_kv_cache` | `bool` | Enable paged KV cache for efficient memory usage. | `True` |
| `kv_cache_block_size` | `PositiveInt` | Block size for the paged KV cache. | `16` |
| `kv_cache_num_blocks` | `PositiveInt` | Number of blocks for the paged KV cache. | `2048` |
| `allow_cross_arch_ref` | `bool` | Allow reference model to be on a different architecture. | `False` |
| `align_bridge_path` | `Optional[Path]` | Path to the alignment bridge model. | `None` |
| `align_bridge_weight` | `NonNegativeFloat` | Weight for the alignment bridge. | `1.0` |
| `align_pool` | `Literal["mean", "last"]` | Pooling strategy for alignment bridge. | `"mean"` |
| `align_after_tag` | `str` | Tag after which alignment is applied. | `"</think>"` |
| `use_sft_on_answer` | `bool` | Enable Supervised Fine-Tuning (SFT) on answer tokens. | `True` |
| `sft_mode` | `str` | SFT mode (e.g., 'weighted'). | `'weighted'` |
| `sft_weight` | `NonNegativeFloat` | Overall SFT weight. | `0.2` |
| `sft_thinking_weight` | `NonNegativeFloat` | SFT weight for thinking layers. | `0.2` |
| `sft_answer_weight` | `NonNegativeFloat` | SFT weight for answer layers. | `1.7` |

### TrainerParams
Parameters specifically for the RL training loop.

| Field | Type | Description | Default |
|---|---|---|---|
| `algorithm` | `Literal["grpo", "ppo"]` | The RL algorithm to use. | `"grpo"` |
| `output_dir` | `Path` | Directory to save training outputs and logs. | `./outputs` |
| `num_training_steps` | `PositiveInt` | Total number of training steps. | `45869` |
| `learning_rate` | `NonNegativeFloat` | Initial learning rate for the optimizer. | `2e-6` |
| `ppo_batch_size` | `PositiveInt` | Batch size for PPO updates. | `1` |
| `num_rollout_samples` | `PositiveInt` | Number of samples to generate per prompt during rollouts. | `2` |
| `grad_accum_steps` | `PositiveInt` | Number of gradient accumulation steps. | `1` |
| `gradient_checkpointing` | `bool` | Enable gradient checkpointing to save memory. | `False` |
| `min_think_tokens` | `PositiveInt` | Minimum number of tokens for the thinking part of the generation. | `16` |
| `max_think_tokens` | `PositiveInt` | Maximum number of tokens for the thinking part of the generation. | `128` |
| `alternate_dual_gradients` | `bool` | Whether to alternate between dual gradients. | `True` |
| `use_mixed_precision` | `bool` | Use mixed precision training. | `False` |
| `log_memory_usage` | `bool` | Log memory usage during training. | `True` |
| `grpo_beta` | `NonNegativeFloat` | Beta parameter for GRPO algorithm. | `0.0025` |
| `seed` | `int` | Random seed for reproducibility. `-1` for random. | `-1` |
| `optimizer_beta1` | `NonNegativeFloat` | Beta1 parameter for the Adam optimizer. | `0.9` |
| `optimizer_beta2` | `NonNegativeFloat` | Beta2 parameter for the Adam optimizer. | `0.95` |
| `optimizer_weight_decay` | `NonNegativeFloat` | Weight decay for the optimizer. | `0.01` |
| `grad_clip_norm` | `Optional[NonNegativeFloat]` | Gradient clipping norm. `None` to disable. | `0.5` |
| `lr_schedule_config` | `Dict[str, Any]` | Configuration for the learning rate scheduler. | `{}` |
| `low_band` | `Tuple[int, int]` | Layer range for low gradient multiplier. | `(0, 15)` |
| `mid_band` | `Tuple[int, int]` | Layer range for mid gradient multiplier. | `(16, 23)` |
| `top_band` | `Tuple[int, int]` | Layer range for top gradient multiplier. | `(24, 35)` |
| `low_mul` | `NonNegativeFloat` | Gradient multiplier for low band layers. | `0.3` |
| `mid_mul` | `NonNegativeFloat` | Gradient multiplier for mid band layers. | `1.3` |
| `top_mul` | `NonNegativeFloat` | Gradient multiplier for top band layers. | `1.5` |
| `head_mul` | `NonNegativeFloat` | Gradient multiplier for head layers. | `1.2` |
| `train_layer_start` | `Optional[int]` | Starting layer index for training. | `22` |
| `train_layer_end` | `Optional[int]` | Ending layer index for training. | `35` |
| `use_custom_batch_builder` | `bool` | Use a custom batch builder for handling invalid samples. | `True` |
| `invalid_sample_layers` | `str` | Comma-separated string of layers to consider for invalid samples. | `"33,34,35"` |
| `invalid_sample_frequency` | `PositiveInt` | Frequency to check for invalid samples. | `2` |
| `eval_every` | `PositiveInt` | Run evaluation every N training updates. | `10000000000` |
| `reward_smoothing_window` | `PositiveInt` | Window size for smoothing rewards. | `20` |
| `use_dual_gradients` | `bool` | Enable dual gradient optimization. | `True` |
| `thinking_layer_start` | `Optional[int]` | Starting layer index for thinking part gradients. | `20` |
| `thinking_layer_end` | `Optional[int]` | Ending layer index for thinking part gradients. | `24` |
| `answer_layer_start` | `Optional[int]` | Starting layer index for answer part gradients. | `22` |
| `answer_layer_end` | `Optional[int]` | Ending layer index for answer part gradients. | `36` |
| `answer_gradient_weight` | `Optional[NonNegativeFloat]` | Weight for answer part gradients. | `4.2` |
| `use_sft_on_answer` | `bool` | Enable SFT on answer tokens (hybrid RL+SFT). | `True` |
| `adaptive_gradient_weights` | `bool` | Enable adaptive balancing of gradient weights. | `True` |
| `max_thinking_tokens` | `Optional[int]` | Maximum allowed thinking tokens. | `80` |
| `optimal_thinking_tokens` | `Optional[int]` | Optimal number of thinking tokens. | `50` |
| `use_thinking_penalty` | `bool` | Apply penalty for thinking token length. | `True` |
| `thinking_penalty_rate` | `Optional[NonNegativeFloat]` | Rate of thinking token penalty. | `0.05` |
| `use_thinking_bonus` | `bool` | Apply bonus for efficient thinking. | `False` |
| `efficiency_bonus_weight` | `Optional[NonNegativeFloat]` | Weight of the efficiency bonus. | `0.1` |

### ModelConfig
Defines the models used for training and generation.

| Field | Type | Description | Default |
|---|---|---|---|
| `model_path` | `Path` | Path to the actor model directory. | (Required) |
| `ref_model_path` | `Optional[Path]` | Path to the reference model directory. Defaults to `model_path` if not provided. | `None` |
| `use_lora` | `bool` | Enable LoRA fine-tuning. | `False` |
| `lora_rank` | `PositiveInt` | LoRA adapter rank. | `8` |
| `lora_alpha` | `float` | LoRA alpha parameter. | `16.0` |
| `lora_dropout` | `NonNegativeFloat` | LoRA dropout rate. | `0.0` |
| `lora_scale_by_rank` | `bool` | Whether to scale LoRA weights by rank. | `True` |
| `lora_target_modules` | `List[str]` | List of module names to apply LoRA to. | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |

### DataConfig
Configures data loading and preprocessing.

| Field | Type | Description | Default |
|---|---|---|---|
| `train_path` | `Path` | Path to training data. | (Required) |
| `val_path` | `Optional[Path]` | Path to validation data. | `None` |
| `max_prompt_len` | `PositiveInt` | Maximum token length for input prompts. | `350` |
| `max_gen_len` | `PositiveInt` | Maximum token length for generated responses. | `96` |
| `loader_type` | `Literal["jsonl", "hf_dataset", "mock"]` | Type of data loader to use. | `"jsonl"` |
| `shuffle_data` | `bool` | Whether to shuffle training data. | `True` |
| `dataset_prompt_key` | `str` | Key for prompt text in the dataset. | `"prompt"` |
| `dataset_answer_key` | `str` | Key for reference answer/completion in the dataset. | `"completion"` |
| `dataset_filter_keywords` | `List[str]` | Keywords to filter out samples during loading. | `["http://", "**other**", "https://", "png", "jpg", "Another way", "Adeel"]` |
| `data_validation_script_path` | `Optional[Path]` | Path to an external Python script for custom validation. | `None` |
| `data_validation_strict_mode` | `bool` | If `True`, discard samples that fail validation. | `False` |

### GenerationConfig
Controls the text generation process.

| Field | Type | Description | Default |
|---|---|---|---|
| `think_start_tag` | `str` | Token tag indicating the start of thinking. | `<think>` |
| `think_end_tag` | `str` | Token tag indicating the end of thinking. | `</think>` |
| `answer_start_tag` | `str` | Token tag indicating the start of the answer. | `""` |
| `answer_end_tag` | `str` | Token tag indicating the end of the answer. | `""` |
| `think_boost_tokens` | `int` | Number of tokens to boost thinking. | `8` |
| `think_temperature` | `NonNegativeFloat` | Temperature for thinking generation. | `0.15` |
| `answer_temperature` | `NonNegativeFloat` | Temperature for answer generation. | `0.1` |
| `sampling_top_p` | `NonNegativeFloat` | Top-p (nucleus) sampling parameter. | `0.6` |
| `sampling_min_p` | `NonNegativeFloat` | Min-p sampling parameter. | `0.00` |
| `sampling_top_k` | `int` | Top-k sampling parameter. | `60` |
| `repetition_penalty` | `Optional[float]` | Penalty for repeating tokens. | `1.1` |
| `repetition_context_size` | `Optional[int]` | Context size for repetition penalty. | `20` |
| `min_think_tokens` | `int` | Minimum thinking tokens for dynamic bias. | `32` |
| `think_end_early_bias` | `float` | Bias to end thinking early. | `-12.0` |
| `bias_answer_start_after_min_think` | `bool` | Bias answer start after minimum thinking tokens. | `True` |
| `bias_close_think` | `float` | Bias to close thinking tag. | `9.0` |
| `bias_answer_start` | `float` | Bias to start answer tag. | `6.0` |
| `punish_extra_think_end` | `float` | Penalty for extra thinking end tags. | `-12.0` |
| `punish_reopen_think` | `float` | Penalty for reopening thinking tag. | `-10.0` |
| `punish_reopen_answer` | `float` | Penalty for reopening answer tag. | `-9.0` |
| `bias_eos_after_answer` | `float` | Bias for End-of-Sequence token after answer. | `3.0` |
| `hard_mask_mcq_first_token` | `bool` | Hard mask first token for MCQ. | `True` |
| `mcq_letter_lift` | `float` | Lift for MCQ answer letters. | `8.0` |
| `mcq_ban_first_bias` | `float` | Bias to ban first token in MCQ. | `-14.0` |
| `nonmcq_ban_first_bias` | `float` | Bias to ban first token in non-MCQ. | `-12.0` |
| `min_answer_tokens` | `int` | Minimum answer tokens. | `8` |
| `min_answer_tokens_mcq` | `int` | Minimum answer tokens for MCQ. | `1` |
| `mcq_answer_end_bias` | `float` | Bias to end MCQ answer. | `9.0` |

### RewardConfig
Defines a single reward function to be used in training.

| Field | Type | Description | Default |
|---|---|---|---|
| `name` | `str` | Registered name of the reward function (e.g., `mcq_accuracy`, `semantic_similarity`). | (Required) |
| `weight` | `float` | Weighting factor for this reward signal (0.0 to 1.0). | `1.0` |
| `config` | `Dict[str, Any]` | Reward-specific parameters. | `{}` |

### EvaluatorConfig
Defines a single evaluator to be run.

| Field | Type | Description | Default |
|---|---|---|---|
| `name` | `str` | Registered name of the evaluator (e.g., `perplexity`, `human_eval`). | (Required) |
| `config` | `Dict[str, Any]` | Evaluator-specific parameters. | `{}` |

### CheckpointConfig
Manages how and when model checkpoints are saved.

| Field | Type | Description | Default |
|---|---|---|---|
| `save_dir` | `Path` | Directory relative to the project root to save checkpoints. | `./checkpoints` |
| `save_every` | `PositiveInt` | Save a full checkpoint every N training updates. | `10` |
| `keep_last_n` | `PositiveInt` | Number of most recent checkpoints to retain. | `2` |
| `save_optimizer_state` | `bool` | Whether to save the optimizer's state along with the model. | `False` |

### MonitoringConfig
Configures integration with Weights & Biases (W&B) and other logging.

| Field | Type | Description | Default |
|---|---|---|---|
| `wandb_log` | `bool` | Enable Weights & Biases (W&B) logging. | `True` |
| `wandb_project` | `Optional[str]` | W&B project name. | `"mlx-grpo-qwen3-v4"` |
| `wandb_entity` | `Optional[str]` | Your W&B entity (username or team name). | `None` |
| `wandb_run_name` | `Optional[str]` | Custom name for the W&B run. | `None` |
| `log_samples_every` | `PositiveInt` | Log generated text samples every N updates. | `1` |
| `max_logged_samples` | `PositiveInt` | Maximum number of generated samples to log per event. | `50` |
| `log_prompts` | `bool` | Include full input prompts in sample logs. | `True` |
| `sample_log_path` | `Optional[Path]` | Custom path to save NDJSON sample logs. | `None` |

## Example Configurations

Configuration files are central to defining your training and evaluation experiments. They are written in YAML format and validated against the Pydantic schemas defined in `src/mlx_rl_trainer/core/config.py`.

Here's an example configuration (`configs/experiments/code_gen_base.yaml`) for a code generation task using GRPO:

```yaml
# Production-ready configuration for a code generation task using GRPO.
# This file is validated against the pydantic schemas in src/mlx_rl_trainer/core/config.py.

trainer:
  algorithm: "grpo"
  output_dir: "./outputs/code_gen_run_006"
  num_training_steps: 80000
  learning_rate: 1e-3 # CORRECTED: Lowered to a safe value for fine-tuning.
  ppo_batch_size: 1
  num_rollout_samples: 1
  grad_accum_steps: 3
  # CORRECTED: Increased beta to prevent model collapse.
  grpo_beta: 0.002
  seed: -1

model:
  model_path: "/Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen-4B-Thinking-2507"
  ref_model_path: "/Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen-4B-Thinking-2507"
  base_model_path: "/Users/adeelahmad/.cache/lm-studio/models/lmstudio-community/Qwen-4B-Thinking-2507"
  use_lora: false
  lora_rank: 16

data:
  train_path: "/Users/adeelahmad/work/SiLLM-examples/helpsteer/surgical/train.jsonl"
  val_path: "/Users/adeelahmad/work/SiLLM-examples/helpsteer/surgical/valid.jsonl"
  max_prompt_len: 150
  max_gen_len: 192
  loader_type: "jsonl"
  shuffle_data: true

# The system_prompt from your ExperimentConfig will be used automatically by the trainer.
# You don't need to specify it here unless you want to override the default.

rewards:
  - name: "format_structure"
    weight: 0.05
    config: {}

  - name: "thinking_quality"
    weight: 0.35
    config:
      # NEW: Length constraints for 128 token budget
      target_length_min: 30
      target_length_max: 80
      optimal_length_min: 40
      optimal_length_max: 60
      excessive_length_threshold: 90
      excessive_length_penalty: 0.5
      conciseness_bonus: 0.15
      use_trainer_thinking_limits: true # Use trainer max_thinking_tokens

  - name: "semantic_similarity"
    weight: 0.55
    config:
      method: "jaccard"

  - name: "steps_coverage"
    weight: 0.05
    config:
      required_steps: 1

evaluation:
  - name: "human_eval"
    config:
      k_values: [1, 2]
      num_samples: 10

monitoring:
  log_samples_every: 1 # Log samples at every update step to debug easily.
  max_logged_samples: 50 # Log a few samples to see the outputs.
```

For more detailed and varied examples, please refer to the `configs/experiments/` directory.


## CLI Commands

### `train.py`
The main script for starting and managing RL training runs.

```bash
python src/mlx_rl_trainer/scripts/train.py [OPTIONS]
```

**Options:**
- `--config <path>`: Path to the YAML configuration file (e.g., `configs/experiments/my_experiment.yaml`). **Required.**
- `--resume_from_checkpoint <path>`: Path to a checkpoint directory to resume training from.
- `--override <key=value>`: Override specific configuration values directly from the command line. Can be used multiple times.
  - Example: `--override trainer.learning_rate=1e-5 model.use_lora=True`

### `evaluate.py`
Evaluates a trained model or a base model on specified datasets and metrics.

```bash
python src/mlx_rl_trainer/scripts/evaluate.py [OPTIONS]
```

**Options:**
- `--config <path>`: Path to the YAML configuration file. **Required.**
- `--checkpoint_path <path>`: Path to the model checkpoint to evaluate. **Required.**
- `--output_path <path>`: Path to save evaluation results (e.g., `eval_results.json`).
- `--override <key=value>`: Override specific configuration values.

### `data_preprocessing.py`
A utility script for preprocessing raw datasets into a format suitable for training.

```bash
python src/mlx_rl_trainer/scripts/data_preprocessing.py [OPTIONS]
```

**Options:**
- `--input_path <path>`: Path to the raw input data file (e.g., `raw_data.jsonl`). **Required.**
- `--output_path <path>`: Path to save the processed data file. **Required.**
- `--config <path>`: Path to a YAML configuration file, primarily for `DataConfig` settings. **Required.**
- `--validation_script <path>`: Path to an external Python script for custom validation logic.
- `--strict_validation`: If set, samples failing validation will be discarded.

### `data_validation.py`
Validates a dataset against the `DataConfig` rules and optionally a custom script.

```bash
python src/mlx_rl_trainer/scripts/data_validation.py [OPTIONS]
```

**Options:**
- `--data_path <path>`: Path to the dataset file to validate. **Required.**
- `--config <path>`: Path to a YAML configuration file, primarily for `DataConfig` settings. **Required.**
- `--validation_script <path>`: Path to an external Python script for custom validation logic.
- `--strict_mode`: If set, the script will exit with an error if any sample fails validation.

### `dump_config.py`
Prints or saves the resolved configuration, useful for debugging and understanding effective parameters.

```bash
python src/mlx_rl_trainer/scripts/dump_config.py [OPTIONS]
```

**Options:**
- `--config <path>`: Path to the base YAML configuration file. **Required.**
- `--output_path <path>`: Optional path to save the dumped configuration to a YAML file. If not provided, prints to console.
- `--override <key=value>`: Override specific configuration values.

## Dataset Examples
The trainer expects datasets in a `jsonl` format, where each line is a JSON object representing a single training sample. The keys for prompt and completion are defined in `DataConfig` (default: `prompt` and `completion`).

**Example `data.jsonl`:**
```json
{"prompt": "What is the capital of France?", "completion": "Paris"}
{"prompt": "Explain the concept of recursion.", "completion": "Recursion is a process where a function calls itself directly or indirectly to solve a problem."}
{"prompt": "Write a Python function to calculate factorial.", "completion": "```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```"}
```

For more complex scenarios, especially with thinking tags, your data might look like:
```json
{"prompt": "Solve the equation 2x + 5 = 11.", "completion": "<think>Equation: 2x + 5 = 11. Goal: Isolate x. Step 1: Subtract 5 from both sides. Step 2: Divide by 2.</think>x = 3"}
```

## Extensibility

### Adding New Rewards
To add a new reward function:
1.  Create a new Python file in `src/mlx_rl_trainer/rewards/` (e.g., `src/mlx_rl_trainer/rewards/my_custom_reward.py`).
2.  Implement a class that inherits from `BaseReward` and overrides the `compute` or `batch_compute` method.
3.  Register your reward in `src/mlx_rl_trainer/rewards/registry.py`.
4.  Reference your new reward by its registered name in your `ExperimentConfig` YAML file under the `rewards` section.

### Adding New Evaluators
To add a new evaluator:
1.  Create a new Python file in `src/mlx_rl_trainer/evaluation/` (e.g., `src/mlx_rl_trainer/evaluation/my_custom_evaluator.py`).
2.  Implement a class that inherits from `BaseEvaluator` and overrides the `evaluate` method.
3.  Register your evaluator in `src/mlx_rl_trainer/evaluation/registry.py`.
4.  Reference your new evaluator by its registered name in your `ExperimentConfig` YAML file under the `evaluation` section.

## Troubleshooting
- **`TypeError: zeros_like(): incompatible function arguments`**: This indicates an issue with how `mlx.core.zeros_like` is being called, likely with an unsupported `dtype` argument. Ensure you are using `mx.zeros(array.shape, dtype=...)` instead.
- **Configuration Validation Errors**: If you encounter `ValidationError` when loading a config, carefully check the error message. It will pinpoint the exact field and reason for the validation failure. Refer to the [Configuration](#configuration) section for correct types and expected values.
- **Model Loading Issues**: Ensure `model_path` and `ref_model_path` in `ModelConfig` point to valid directories containing your MLX models.
- **WandB Not Logging**: Verify `wandb_log` is `True` in `MonitoringConfig` and that your W&B API key is correctly set up (e.g., via `wandb login` or `WANDB_API_KEY` environment variable).

## Contributing
We welcome contributions! Please see `CONTRIBUTING.md` (if available) for guidelines.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.