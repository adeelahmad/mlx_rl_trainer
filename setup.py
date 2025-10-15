from setuptools import setup, find_packages

setup(
    name="mlx_rl_trainer",
    version="0.8.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "mlx-train=mlx_rl_trainer.scripts.train:main",
            "mlx-evaluate=mlx_rl_trainer.scripts.evaluate:main",
            "mlx-dump-config=mlx_rl_trainer.scripts.dump_config:main",
        ]
    },
    install_requires=[
        "mlx>=0.5.0",
        "mlx-lm>=0.8.0",
        "pydantic>=2.0",
        "numpy>=1.24.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "tqdm>=4.60.0",
        "aiofiles>=22.0.0",
        "scikit-learn>=1.3.0",
        # Added for core tracking functionality
        "wandb>=0.16.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio"],
        "monitoring": ["pandas", "matplotlib"],
    },
    python_requires=">=3.9",
)
