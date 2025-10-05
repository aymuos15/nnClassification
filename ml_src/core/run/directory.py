"""Run directory creation utilities."""

import os

import yaml
from loguru import logger


def create_run_dir(overrides, config, config_path=None):
    """
    Create run directory based on config overrides and save config.

    Args:
        overrides: List of override strings for run naming (e.g., ['batch_32', 'lr_0.01'])
        config: Configuration dictionary
        config_path: Original config path (for logging purposes, optional)

    Returns:
        str: Path to created run directory
    """
    dataset_name = config["data"].get("dataset_name", "dataset")

    run_name = f"{dataset_name}_{'_'.join(overrides)}" if overrides else f"{dataset_name}_base"

    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.success(f"Saved config to {config_save_path}")

    # Create subdirectories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    # TensorBoard directory will be created automatically by SummaryWriter

    return run_dir
