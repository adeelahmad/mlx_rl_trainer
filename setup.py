# file_path: mlx_rl_trainer/setup.py
# revision_no: 001
# goals_of_writing_code_block: Define the package structure and dependencies for the MLX RL Trainer.
# type_of_code_response: add new code
"""Setup configuration for the MLX RL Trainer package."""
from setuptools import setup, find_packages

setup(
    name="mlx_rl_trainer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "mlx-train=mlx_rl_trainer.scripts.train:main",
        ],
    },
    install_requires=[
        "mlx>=0.5.0",
        "mlx-lm>=0.8.0",  # Required for mlx_lm.utils.load, TokenizerWrapper, etc.
        "pydantic>=2.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",  # HuggingFace datasets library
        "pyyaml>=6.0",  # For config files
        "rich>=13.0.0",  # For enhanced logging and progress bars
        "tqdm>=4.60.0",  # For console progress bars
        "aiofiles>=22.0.0",  # For asynchronous file I/O
        "scikit-learn>=1.3.0",  # For TF-IDF in reward functions
    ],
    extras_require={
        "dev": [
            "pandas",  # For metrics plotting
            "matplotlib",  # For metrics plotting
            "pytest",  # For unit and integration testing
            "pytest-asyncio",  # For testing async code
        ]
    },
    python_requires=">=3.9",
)
