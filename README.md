# file_path: mlx_rl_trainer/README.md
# revision_no: 001
# goals_of_writing_code_block: Provide a comprehensive overview of the project, its architectural principles, and quick start instructions.
# type_of_code_response: add new code
# MLX RL Trainer: Production-Scale Architecture

This project implements a highly modular, production-grade Reinforcement Learning (RL) trainer for MLX-based language models, designed from the ground up with best software engineering practices.

## Core Architectural Pillars
- **SOLID & OOP:** Every component is designed with Single Responsibility, Open/Closed principles, and Dependency Injection in mind, using Abstract Base Classes (ABCs) to define clear contracts. This ensures code is testable, maintainable, and extensible.
- **Meta-Programming (Registry Pattern):** Reward functions and evaluators are pluggable components. They can be added to the system without modifying any core training code, simply by decorating the class and updating a YAML configuration file. This adheres to the Open/Closed Principle.
- **Predictable Code (Pydantic):** All configurations are rigorously validated through Pydantic schemas, ensuring type safety, data integrity, and preventing runtime errors stemming from malformed or incomplete configurations. This makes the system robust and easy to reason about.
- **Efficient I/O (Asyncio & Aiofiles):** The data pipeline leverages Python's `asyncio` framework in conjunction with `aiofiles` for non-blocking asynchronous file operations. This is crucial for handling large datasets efficiently without blocking the event loop.
- **Safe & Isolated Execution (Multiprocessing):** The `code_execution` reward function uses Python's `multiprocessing` library (specifically the `spawn` context) to run untrusted code in a separate, isolated process with strict timeouts. This prevents the main training loop from crashing and ensures system stability.
- **Robustness (Custom Exceptions):** A hierarchy of custom, domain-specific exceptions is used throughout the codebase for predictable, granular error handling. This allows for precise error management and recovery strategies.
- **Developer-Friendly (Tqdm & Rich Logging):** Integrated `tqdm` for clear progress bars in CLI and `rich` for enhanced, structured, and colorful logging output, improving developer experience and debugging.

## Project Structure Overview
```
mlx_rl_trainer/
├── configs/                # Configuration files (YAML)
├── docs/                   # Project documentation
├── scripts/                # Entry-point scripts
├── src/
│   └── mlx_rl_trainer/
│       ├── core/           # Core abstractions: config, trainer interface, managers
│       ├── algorithms/     # RL algorithm implementations (e.g., GRPO, PPO)
│       ├── data/           # Data loading, processing, batching
│       ├── evaluation/     # Benchmark evaluators
│       ├── generation/     # Text generation utilities
│       ├── monitoring/     # Logging, metrics, W&B integration
│       ├── rewards/        # Pluggable reward functions (meta-programming)
│       └── utils/          # General utilities, logging setup
└── tests/                  # Unit and integration tests
```

## Quick Start (Mock Setup)

This project is designed to be immediately runnable even without a full MLX-LM model. The `ModelManager` and `DatasetManager` include mock implementations that adhere to the defined interfaces.

1.  **Install the package in editable mode with development dependencies:**
    ```bash
    pip install -e ./mlx_rl_trainer[dev]
    ```

2.  **Run the training script:**
    The script will automatically create dummy model and data files for the initial run.
    ```bash
    mlx-train --config mlx_rl_trainer/configs/experiments/code_gen_base.yaml --log-level INFO
    ```
    Observe the rich logging output, progress bars, and the simulated training loop. The first run will create `models/mock_model` and `data/` directories.

## Next Steps for Full MLX-LM Integration
To transition from the mock setup to full MLX-LM capabilities:
- Replace the mock implementations in `mlx_rl_trainer/src/mlx_rl_trainer/core/model_manager.py` and `mlx_rl_trainer/src/mlx_rl_trainer/algorithms/grpo/grpo_trainer.py` (specifically in `generate_rollouts`) with actual `mlx_lm` calls.
- This architecture ensures that these integrations can be done with minimal impact on other components.
# mlx_rl_trainer
