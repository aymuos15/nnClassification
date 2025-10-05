#!/usr/bin/env python3
"""
Training script for image classification models.
"""

import argparse
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
import yaml
from loguru import logger

from ml_src.dataset import get_datasets, get_class_names
from ml_src.loader import get_dataloaders, get_dataset_sizes
from ml_src.network import get_model
from ml_src.loss import get_criterion
from ml_src.optimizer import get_optimizer, get_scheduler
from ml_src.trainer import train_model
from ml_src.seeding import set_seed
from ml_src.checkpointing import load_checkpoint


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config, args):
    """Override configuration with command-line arguments."""
    overrides = []

    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        overrides.append(f"batch_{args.batch_size}")
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
        overrides.append(f"epochs_{args.num_epochs}")
    if args.lr:
        config['optimizer']['lr'] = args.lr
        overrides.append(f"lr_{args.lr}")
    if args.momentum:
        config['optimizer']['momentum'] = args.momentum
    if args.step_size:
        config['scheduler']['step_size'] = args.step_size
    if args.gamma:
        config['scheduler']['gamma'] = args.gamma
    if args.fold is not None:
        config['data']['fold'] = args.fold
        overrides.append(f"fold_{args.fold}")

    return config, overrides


def create_run_dir(overrides, config, config_path):
    """Create run directory based on config overrides and save config."""
    if overrides:
        run_name = "_".join(overrides)
    else:
        run_name = "base"

    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config to run directory
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.success(f"Saved config to {config_save_path}")

    # Create subdirectories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    # TensorBoard directory will be created automatically by SummaryWriter

    return run_dir


def setup_logging(run_dir):
    """Setup loguru logging to both console and file."""
    # Remove default handler
    logger.remove()

    # Add colorized console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO"
    )

    # Add file handler (plain text, no colors)
    log_path = os.path.join(run_dir, "logs", "train.log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG"
    )

    logger.info(f"Logging to {log_path}")


def main():
    """Main function to orchestrate training."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--config', type=str, default='ml_src/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from (e.g., runs/base/last.pt)')
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, help='SGD momentum')
    parser.add_argument('--step_size', type=int, help='LR scheduler step size')
    parser.add_argument('--gamma', type=float, help='LR scheduler gamma')
    parser.add_argument('--fold', type=int, help='Fold number for cross-validation (0-indexed, default: 0)')

    args = parser.parse_args()

    # Load and override configuration
    config = load_config(args.config)
    config, overrides = override_config(config, args)

    # Create run directory and save config
    run_dir = create_run_dir(overrides, config, args.config)

    # Setup logging (must be done after creating run_dir)
    setup_logging(run_dir)

    logger.info(f"Run directory: {run_dir}")

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', False)
    set_seed(seed, deterministic)

    # Determine device
    device_str = config['training']['device']
    if device_str.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading datasets...")
    datasets = get_datasets(config)
    class_names = get_class_names(datasets)
    logger.info(f"Classes: {class_names}")

    # Create dataloaders
    dataloaders = get_dataloaders(datasets, config)
    dataset_sizes = get_dataset_sizes(datasets)
    logger.info(f"Dataset sizes: {dataset_sizes}")

    # Create model
    logger.info("Creating model...")
    model = get_model(config, device)

    # Create criterion
    criterion = get_criterion()

    # Training
    logger.info("="*50)
    logger.info("Starting Training")
    logger.info("="*50)

    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_best_acc = 0.0
    resume_train_losses = None
    resume_val_losses = None
    resume_train_accs = None
    resume_val_accs = None

    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"Checkpoint not found: {args.resume}")
            return

        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, resume_best_acc, resume_train_losses, resume_val_losses, resume_train_accs, resume_val_accs, _ = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        # Continue from the next epoch
        start_epoch += 1
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        config=config,
        run_dir=run_dir,
        class_names=class_names,
        start_epoch=start_epoch,
        resume_best_acc=resume_best_acc,
        resume_train_losses=resume_train_losses,
        resume_val_losses=resume_val_losses,
        resume_train_accs=resume_train_accs,
        resume_val_accs=resume_val_accs
    )

    logger.info("="*50)
    logger.success("Training Complete!")
    logger.info("="*50)
    logger.info(f"View training metrics: tensorboard --logdir {run_dir}/tensorboard")


if __name__ == '__main__':
    main()
