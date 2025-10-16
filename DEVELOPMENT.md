# DEVELOPMENT.md

This document provides guidelines for setting up your development environment, understanding the project structure, and extending the `mlx_rl_trainer` with new features like reward functions and evaluators.

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Project Structure Overview](#project-structure-overview)
- [Adding a New Reward Type](#adding-a-new-reward-type)
- [Adding a New Evaluator Type](#adding-a-new-evaluator-type)
- [Extending Configuration](#extending-configuration)
- [Writing Tests](#writing-tests)
- [Code Style and Linting](#code-style-and-linting)
- [Contributing Guidelines](#contributing-guidelines)

## Development Environment Setup

To contribute to `mlx_rl_trainer`, follow these steps to set up your local development environment.

### Prerequisites
- **Python 3.9+**: Ensure you have a compatible Python version installed.
- **Git**: For version control.
- **MLX**: Follow the official MLX installation instructions for your system.

### Cloning the Repository
```bash
git clone https://github.com/your-repo/mlx_rl_trainer.git
cd mlx_rl_trainer
```

### Setting up a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
```

### Installing Development Dependencies
Install all necessary packages, including those for development and testing.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt # If a dev requirements file exists
```

### IDE Setup Recommendations
- **VS Code**: Install the Python extension. Recommended extensions include `Pylance` for IntelliSense and linting, `Black Formatter` for code formatting, and `isort` for import sorting.
- **PyCharm**: PyCharm provides excellent Python development support out-of-the-box, including integrated virtual environments, debugging, and code analysis.

## Project Structure Overview

- `configs/`: Contains example YAML configuration files for various experiments.
- `docs/`: Documentation files, including `README.md` and this `DEVELOPMENT.md`.
- `src/mlx_rl_trainer/`: The core source code of the trainer.
  - `algorithms/`: Implementations of RL algorithms (e.g., GRPO, PPO).
  - `core/`: Core components like `config.py`, `trainer.py`, `checkpoint_manager.py`.
  - `data/`: Data loading, batching, and preprocessing utilities.
  - `evaluation/`: Base classes and implementations for evaluators.
  - `generation/`: Logic for text generation and sampling.
  - `monitoring/`: Logging and metrics collection (e.g., Weights & Biases).
  - `rewards/`: Base classes and implementations for reward functions.
  - `scripts/`: Command-line interface (CLI) entry points (e.g., `train.py`, `evaluate.py`).
  - `utils/`: General utility functions.
- `tests/`: Unit and integration tests for the project.

## Adding a New Reward Type

To extend the framework with a new reward function, follow these steps:

1.  **Create a New Reward File:**
    Create a new Python file in `src/mlx_rl_trainer/rewards/` (e.g., `src/mlx_rl_trainer/rewards/my_custom_reward.py`).

2.  **Implement the Reward Class:**
    Your new class must inherit from `mlx_rl_trainer.rewards.base_reward.BaseReward` and implement the `compute` method. Optionally, you can override `batch_compute` for efficiency if your reward can be calculated for a batch of contexts simultaneously.

    ```python
    # src/mlx_rl_trainer/rewards/my_custom_reward.py
    from typing import Dict, Any
    from mlx_rl_trainer.rewards.base_reward import BaseReward, RewardContext

    class MyCustomReward(BaseReward):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            # Initialize any reward-specific parameters from config
            self.my_param = config.get("my_param", 0.5)

        def compute(self, context: RewardContext) -> float:
            # Implement your reward logic here
            # Access generated_text, prompt_text, reference_completion, metadata from context
            # Example: A simple length-based reward
            reward = len(context.generated_text) * self.my_param
            return float(reward)

        # Optionally, override batch_compute for efficiency
        # def batch_compute(self, contexts: List[RewardContext]) -> List[Dict[str, float]]:
        #     # Implement batch processing here
        #     pass
    ```

3.  **Register the New Reward:**
    Open `src/mlx_rl_trainer/rewards/registry.py` and add your new reward class to the `REWARD_REGISTRY` dictionary.

    ```python
    # src/mlx_rl_trainer/rewards/registry.py
    from .base_reward import BaseReward
    from .content.mcq_accuracy import MCQAccuracyReward
    from .my_custom_reward import MyCustomReward # Import your new reward

    REWARD_REGISTRY: Dict[str, type[BaseReward]] = {
        "mcq_accuracy": MCQAccuracyReward,
        "my_custom_reward": MyCustomReward, # Register it here
        # ... other rewards
    }
    ```

4.  **Configure in YAML:**
    Now you can use your new reward in your experiment configuration YAML file (e.g., `configs/experiments/my_experiment.yaml`).

    ```yaml
    # configs/experiments/my_experiment.yaml
    rewards:
      - name: my_custom_reward
        weight: 0.7
        config:
          my_param: 0.75
      - name: semantic_similarity
        weight: 0.3
        # ... other reward configs
    ```

## Adding a New Evaluator Type

Similar to rewards, you can add custom evaluators to measure specific aspects of your model's performance.

1.  **Create a New Evaluator File:**
    Create a new Python file in `src/mlx_rl_trainer/evaluation/` (e.g., `src/mlx_rl_trainer/evaluation/my_custom_evaluator.py`).

2.  **Implement the Evaluator Class:**
    Your new class must inherit from `mlx_rl_trainer.evaluation.base_evaluator.BaseEvaluator` and implement the `evaluate` method.

    ```python
    # src/mlx_rl_trainer/evaluation/my_custom_evaluator.py
    from typing import Dict, Any, List
    from mlx_rl_trainer.evaluation.base_evaluator import BaseEvaluator

    class MyCustomEvaluator(BaseEvaluator):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.threshold = config.get("threshold", 0.8)

        def evaluate(self, prompts: List[Dict], responses: List[str]) -> Dict[str, Any]:
            # Implement your evaluation logic here
            # 'prompts' contains original prompt data, 'responses' are generated texts
            # Example: Count responses longer than a certain length
            long_responses = [r for r in responses if len(r) > self.threshold * 100]
            score = len(long_responses) / len(responses) if responses else 0.0
            return {"my_custom_metric": score, "long_response_count": len(long_responses)}
    ```

3.  **Register the New Evaluator:**
    Open `src/mlx_rl_trainer/evaluation/registry.py` and add your new evaluator class to the `EVALUATOR_REGISTRY` dictionary.

    ```python
    # src/mlx_rl_trainer/evaluation/registry.py
    from .base_evaluator import BaseEvaluator
    from .general.perplexity import PerplexityEvaluator
    from .my_custom_evaluator import MyCustomEvaluator # Import your new evaluator

    EVALUATOR_REGISTRY: Dict[str, type[BaseEvaluator]] = {
        "perplexity": PerplexityEvaluator,
        "my_custom_evaluator": MyCustomEvaluator, # Register it here
        # ... other evaluators
    }
    ```

4.  **Configure in YAML:**
    Add your new evaluator to your experiment configuration YAML file.

    ```yaml
    # configs/experiments/my_experiment.yaml
    evaluation:
      - name: my_custom_evaluator
        config:
          threshold: 0.9
      - name: human_eval
        # ... other evaluator configs
    ```

## Extending Configuration

The `mlx_rl_trainer` uses Pydantic models defined in `src/mlx_rl_trainer/core/config.py` for robust and validated configuration. You can extend these models or create new ones.

1.  **Adding New Fields to Existing Models:**
    To add a new parameter, simply add it as a field to the relevant Pydantic `BaseModel` in `src/mlx_rl_trainer/core/config.py`. Remember to provide a type hint, a default value (if optional), and a `Field` description.

    ```python
    # src/mlx_rl_trainer/core/config.py (example for DataConfig)
    class DataConfig(BaseModel):
        # ... existing fields ...
        new_data_param: str = Field("default_value", description="A new parameter for data configuration.")
    ```

2.  **Creating New Configuration Models:**
    If you have a new logical group of settings, create a new `BaseModel` class.

    ```python
    # src/mlx_rl_trainer/core/config.py
    class MyNewFeatureConfig(BaseModel):
        enable_feature: bool = Field(False, description="Enable or disable my new feature.")
        feature_param_a: PositiveInt = Field(10, description="Parameter A for my new feature.")
    ```

3.  **Integrating New Models into `ExperimentConfig`:**
    Add an instance of your new configuration model as a field in the main `ExperimentConfig`.

    ```python
    # src/mlx_rl_trainer/core/config.py
    class ExperimentConfig(BaseModel):
        # ... existing fields ...
        my_new_feature: MyNewFeatureConfig = Field(default_factory=MyNewFeatureConfig)
    ```

4.  **Using New Configs in YAML:**
    Your YAML files can now include settings for your new feature.

    ```yaml
    # configs/experiments/my_experiment.yaml
    my_new_feature:
      enable_feature: true
      feature_param_a: 25
    ```

## Writing Tests

Tests are crucial for ensuring the correctness and stability of the `mlx_rl_trainer`. Please write tests for any new features or bug fixes.

-   **Location**: Place unit tests in `tests/unit/` and integration tests in `tests/integration/`. Follow the existing directory structure (e.g., `tests/unit/test_rewards/` for reward tests).
-   **Framework**: We use `pytest`.
-   **Running Tests**: From the project root, run:
    ```bash
    pytest
    # To run specific tests:
    pytest tests/unit/test_rewards/test_my_custom_reward.py
    ```

## Code Style and Linting

We adhere to a consistent code style. Please ensure your code passes linting and formatting checks.

-   **Tools**: We use `ruff` for linting and `black` for formatting.
-   **Running Checks**:
    ```bash
    ruff check src/
    black src/
    ```
    It's recommended to configure your IDE to automatically format code with Black on save.

## Contributing Guidelines

1.  **Branching**: Create a new feature branch from `main` for your changes (e.g., `feature/my-new-reward`).
2.  **Commits**: Write clear, concise commit messages.
3.  **Pull Requests**: Submit a Pull Request to the `main` branch. Ensure all tests pass and the code is linted correctly. Provide a detailed description of your changes.
