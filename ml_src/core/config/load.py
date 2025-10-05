"""Configuration loading utilities."""

import yaml


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
