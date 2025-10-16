"""
A script to load ExperimentConfig and output its default values to a YAML file.
"""
import argparse
from pathlib import Path
import yaml

from mlx_rl_trainer.core.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(
        description="Dump the default ExperimentConfig to a YAML file."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output YAML file (e.g., `config.yaml`)",
    )
    args = parser.parse_args()

    # Create a default config instance
    default_config = ExperimentConfig()

    # Convert to a dictionary suitable for YAML export
    config_dict = default_config.model_dump(mode="json")

    # Write to the specified file
    output_path = Path(args.output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Default configuration successfully written to: {output_path}")


if __name__ == "__main__":
    main()
