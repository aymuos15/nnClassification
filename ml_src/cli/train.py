#!/usr/bin/env python3
"""
Training script for image classification models.
"""

import argparse
import os

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.callbacks import get_callbacks
from ml_src.core.checkpointing import load_checkpoint
from ml_src.core.config import load_config, override_config
from ml_src.core.data import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.logging import setup_logging
from ml_src.core.loss import get_criterion
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)
from ml_src.core.network import get_model, load_model
from ml_src.core.optimizer import get_optimizer, get_scheduler
from ml_src.core.run import create_run_dir
from ml_src.core.seeding import set_seed
from ml_src.core.test import evaluate_model
from ml_src.core.trainers import get_trainer


def main():
    """Main function to orchestrate training."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (use 'ml-init-config' to generate one)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from (e.g., runs/base/last.pt)",
    )
    parser.add_argument("--data_dir", type=str, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="SGD momentum")
    parser.add_argument("--step_size", type=int, help="LR scheduler step size")
    parser.add_argument("--gamma", type=float, help="LR scheduler gamma")
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold number for cross-validation (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name (used in run directory naming)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Early stopping patience (epochs to wait for improvement)",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        choices=["val_acc", "val_loss"],
        help="Early stopping metric to monitor (val_acc or val_loss)",
    )

    args = parser.parse_args()

    # Load and override configuration
    config = load_config(args.config)
    config, overrides = override_config(config, args)

    # Create run directory and save config
    run_dir = create_run_dir(overrides, config)

    # Setup logging (must be done after creating run_dir)
    setup_logging(run_dir, filename="train.log")

    logger.info(f"Run directory: {run_dir}")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    set_seed(seed, deterministic)

    # Determine device
    device_str = config["training"]["device"]
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Loading datasets...")
    fold = config["data"].get("fold", 0)
    logger.info(f"Using fold: {fold}")
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
    logger.info("=" * 50)
    logger.info("Starting Training")
    logger.info("=" * 50)

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
        (
            start_epoch,
            resume_best_acc,
            resume_train_losses,
            resume_val_losses,
            resume_train_accs,
            resume_val_accs,
            _,
            _,  # early_stopping_state (handled by trainer)
            _,  # ema_state (handled by trainer)
        ) = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        # Continue from the next epoch
        start_epoch += 1
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Load callbacks from configuration
    callbacks = get_callbacks(config)

    # Create trainer
    trainer = get_trainer(
        config=config,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=device,
        run_dir=run_dir,
        class_names=class_names,
        callbacks=callbacks,
    )

    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = trainer.train(
        start_epoch=start_epoch,
        resume_best_acc=resume_best_acc,
        resume_train_losses=resume_train_losses,
        resume_val_losses=resume_val_losses,
        resume_train_accs=resume_train_accs,
        resume_val_accs=resume_val_accs,
    )

    logger.info("=" * 50)
    logger.info("Running Test Evaluation")
    logger.info("=" * 50)

    # Load best checkpoint for testing
    best_checkpoint_path = os.path.join(run_dir, "weights", "best.pt")
    logger.info(f"Loading best checkpoint from {best_checkpoint_path}")
    model = load_model(model, best_checkpoint_path, device)

    # Run test evaluation
    test_acc, per_sample_results = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        dataset_size=dataset_sizes["test"],
        device=device,
        class_names=class_names,
    )

    # Extract labels and predictions for metrics
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_labels = [class_to_idx[true_label] for true_label, _, _ in per_sample_results]
    test_preds = [class_to_idx[pred_label] for _, pred_label, _ in per_sample_results]

    # Initialize TensorBoard writer for test metrics
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Log test metrics to TensorBoard
    logger.info("Logging test metrics to TensorBoard...")
    writer.add_scalar("Test/Accuracy", test_acc, 0)

    # Log confusion matrix
    log_confusion_matrix_to_tensorboard(
        writer, test_labels, test_preds, class_names, "Confusion_Matrix/test", 0
    )

    # Log classification report
    test_report = get_classification_report_str(test_labels, test_preds, class_names)
    writer.add_text("Classification_Report/test", test_report, 0)

    # Save classification report to file
    save_classification_report(
        test_labels,
        test_preds,
        class_names,
        os.path.join(run_dir, "logs", "classification_report_test.txt"),
    )

    writer.close()
    logger.success(f"Test Accuracy: {test_acc:.4f}")

    logger.info("=" * 50)
    logger.success("Training and Testing Complete!")
    logger.info("=" * 50)
    logger.info(f"View training metrics: tensorboard --logdir {run_dir}/tensorboard")
    logger.info(f"Test results saved to: {run_dir}/logs/classification_report_test.txt")


if __name__ == "__main__":
    main()
