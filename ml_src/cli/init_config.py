#!/usr/bin/env python3
"""
Configuration initialization script for creating dataset-specific configs.
"""

import argparse
import os

import yaml
from loguru import logger

from ml_src.core.config import create_config
from ml_src.core.data import detect_dataset_info
from ml_src.core.logging import setup_logging
from ml_src.core.ui import prompt_user_settings


def main():
    """Main function for config initialization."""
    setup_logging()  # Console only

    parser = argparse.ArgumentParser(
        description="Initialize configuration for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect settings and create config
  ml-init-config --data_dir data/my_dataset

  # Specify output location
  ml-init-config --data_dir data/my_dataset --output configs/custom.yaml

  # Non-interactive mode (use defaults)
  ml-init-config --data_dir data/my_dataset --yes

  # Custom settings
  ml-init-config --data_dir data/my_dataset --architecture efficientnet_b0 --batch_size 32

  # With hyperparameter search support
  ml-init-config --data_dir data/my_dataset --optuna
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory (must contain raw/ subdirectory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for config file (default: configs/{dataset_name}_config.yaml)",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Non-interactive mode (use defaults)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=25, help="Number of training epochs (default: 25)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument(
        "--optuna",
        action="store_true",
        default=False,
        help="Include hyperparameter search configuration (requires: pip install -e '.[optuna]')",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Dataset Configuration Initialization")
    logger.info("=" * 60)

    # Detect dataset information
    logger.info(f"Scanning dataset directory: {args.data_dir}")
    dataset_info = detect_dataset_info(args.data_dir)

    if dataset_info is None:
        return

    logger.success(f"Detected dataset: {dataset_info['dataset_name']}")
    logger.info(f"Number of classes: {dataset_info['num_classes']}")
    logger.info(f"Classes: {', '.join(dataset_info['class_names'])}")

    # Get settings (interactive or from args)
    if args.yes:
        settings = {
            "architecture": args.architecture,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "num_folds": 5,
        }
        logger.info("Using default settings (non-interactive mode)")
    else:
        settings = prompt_user_settings()

    # Create config
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_template.yaml")

    if not os.path.exists(template_path):
        logger.error(f"Config template not found: {template_path}")
        return

    config = create_config(dataset_info, template_path, **settings)

    # Add search configuration if --optuna flag is set
    if args.optuna:
        logger.info("Adding hyperparameter search configuration...")
        config["search"] = {
            "study_name": f"{dataset_info['dataset_name']}_optimization",
            "storage": "sqlite:///optuna_studies.db",
            "n_trials": 50,
            "timeout": None,
            "direction": "maximize",
            "metric": "val_acc",
            "sampler": {"type": "TPESampler", "n_startup_trials": 10},
            "pruner": {
                "type": "MedianPruner",
                "n_startup_trials": 5,
                "n_warmup_steps": 5,
            },
            "cross_validation": {"enabled": False, "n_folds": 5, "aggregation": "mean"},
            "search_space": {
                "optimizer.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-1},
                "training.batch_size": {"type": "categorical", "choices": [16, 32, 64]},
                "optimizer.momentum": {"type": "uniform", "low": 0.8, "high": 0.99},
                "scheduler.step_size": {"type": "int", "low": 5, "high": 15},
                "scheduler.gamma": {"type": "uniform", "low": 0.05, "high": 0.5},
            },
        }
        logger.success("Search configuration added. Install with: pip install -e '.[optuna]'")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: configs/{dataset_name}_config.yaml
        configs_dir = "configs"
        os.makedirs(configs_dir, exist_ok=True)
        output_path = os.path.join(configs_dir, f"{dataset_info['dataset_name']}_config.yaml")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.success(f"Configuration saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Dataset:      {dataset_info['dataset_name']}")
    print(f"Classes:      {dataset_info['num_classes']} ({', '.join(dataset_info['class_names'])})")
    print(f"Architecture: {settings['architecture']}")
    print(f"Batch size:   {settings['batch_size']}")
    print(f"Epochs:       {settings['num_epochs']}")
    print(f"Learning rate: {settings['lr']}")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    if args.optuna:
        print("  1. Install search dependencies: pip install -e '.[optuna]'")
        print(f"  2. (Optional) Edit search space: {output_path}")
        print(f"  3. Run hyperparameter search: ml-search --config {output_path}")
        print(f"  4. Visualize results: ml-visualise --mode search --study-name {dataset_info['dataset_name']}_optimization")
    else:
        print(f"  1. (Optional) Edit config: {output_path}")
        print(f"  2. Train model: ml-train --config {output_path}")
    print()


if __name__ == "__main__":
    main()
