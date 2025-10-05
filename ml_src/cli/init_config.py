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
        """
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path to dataset directory (must contain raw/ subdirectory)"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output path for config file (default: configs/{dataset_name}_config.yaml)"
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help="Non-interactive mode (use defaults)"
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='resnet18',
        help="Model architecture (default: resnet18)"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Dataset Configuration Initialization")
    logger.info("="*60)

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
            'architecture': args.architecture,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'lr': args.lr,
            'num_folds': 5
        }
        logger.info("Using default settings (non-interactive mode)")
    else:
        settings = prompt_user_settings()

    # Create config
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'config_template.yaml'
    )

    if not os.path.exists(template_path):
        logger.error(f"Config template not found: {template_path}")
        return

    config = create_config(dataset_info, template_path, **settings)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: configs/{dataset_name}_config.yaml
        configs_dir = 'configs'
        os.makedirs(configs_dir, exist_ok=True)
        output_path = os.path.join(configs_dir, f"{dataset_info['dataset_name']}_config.yaml")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.success(f"Configuration saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Dataset:      {dataset_info['dataset_name']}")
    print(f"Classes:      {dataset_info['num_classes']} ({', '.join(dataset_info['class_names'])})")
    print(f"Architecture: {settings['architecture']}")
    print(f"Batch size:   {settings['batch_size']}")
    print(f"Epochs:       {settings['num_epochs']}")
    print(f"Learning rate: {settings['lr']}")
    print("="*60)

    # Print next steps
    print("\nNext steps:")
    print(f"  1. (Optional) Edit config: {output_path}")
    print(f"  2. Train model: ml-train --config {output_path}")
    print()


if __name__ == '__main__':
    main()
