#!/usr/bin/env python3
"""
Configuration initialization script for creating dataset-specific configs.
"""

import argparse
import os
from pathlib import Path

import yaml
from loguru import logger


def setup_logging():
    """Setup loguru logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO",
    )


def detect_dataset_info(data_dir):
    """
    Detect dataset information from directory structure.

    Args:
        data_dir: Path to dataset directory (should contain raw/ subdirectory)

    Returns:
        dict: Dataset information (name, num_classes, class_names)
    """
    data_path = Path(data_dir)

    # Check if raw directory exists
    raw_dir = data_path / "raw"
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        logger.error("Expected structure: {data_dir}/raw/class1/, {data_dir}/raw/class2/, ...")
        return None

    # Detect classes from subdirectories
    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if len(class_dirs) == 0:
        logger.error(f"No class directories found in {raw_dir}")
        logger.error("Expected structure: {data_dir}/raw/class1/, {data_dir}/raw/class2/, ...")
        return None

    class_names = sorted([d.name for d in class_dirs])
    num_classes = len(class_names)
    dataset_name = data_path.name

    return {
        'dataset_name': dataset_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'data_dir': str(data_path)
    }


def create_config(dataset_info, template_path, architecture='resnet18',
                 batch_size=4, num_epochs=25, lr=0.001, num_folds=5):
    """
    Create configuration from template with dataset-specific values.

    Args:
        dataset_info: Dictionary with dataset information
        template_path: Path to config template
        architecture: Model architecture name
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        num_folds: Number of cross-validation folds

    Returns:
        dict: Configuration dictionary
    """
    # Load template
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update with dataset-specific values
    config['data']['dataset_name'] = dataset_info['dataset_name']
    config['data']['data_dir'] = dataset_info['data_dir']
    config['data']['fold'] = 0  # Default to fold 0

    config['model']['num_classes'] = dataset_info['num_classes']
    config['model']['architecture'] = architecture

    config['training']['batch_size'] = batch_size
    config['training']['num_epochs'] = num_epochs

    config['optimizer']['lr'] = lr

    return config


def prompt_user_settings():
    """
    Prompt user for configuration settings interactively.

    Returns:
        dict: User-selected settings
    """
    print("\n" + "="*60)
    print("Configuration Settings")
    print("="*60 + "\n")

    # Architecture
    print("Model Architecture:")
    print("  Popular choices: resnet18, resnet50, efficientnet_b0, mobilenet_v2, vit_b_16")
    architecture = input("  Architecture [resnet18]: ").strip() or "resnet18"

    # Batch size
    batch_size = input("  Batch size [4]: ").strip()
    batch_size = int(batch_size) if batch_size else 4

    # Epochs
    num_epochs = input("  Number of epochs [25]: ").strip()
    num_epochs = int(num_epochs) if num_epochs else 25

    # Learning rate
    lr = input("  Learning rate [0.001]: ").strip()
    lr = float(lr) if lr else 0.001

    # Number of folds
    num_folds = input("  Number of CV folds (for splitting) [5]: ").strip()
    num_folds = int(num_folds) if num_folds else 5

    return {
        'architecture': architecture,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'num_folds': num_folds
    }


def main():
    """Main function for config initialization."""
    setup_logging()

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
