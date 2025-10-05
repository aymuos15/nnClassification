#!/usr/bin/env python3
"""
Inference script for evaluating trained models.
"""

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.dataset import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)
from ml_src.core.network import get_model, load_model
from ml_src.core.test import test_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(run_dir):
    """Setup loguru logging to both console and file."""
    # Remove default handler
    logger.remove()

    # Add colorized console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | <level>{level: <8}</level> | {message}",
        colorize=True,
        level="INFO",
    )

    # Add file handler (plain text, no colors)
    log_path = os.path.join(run_dir, "logs", "inference.log")
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
    )

    logger.info(f"Logging to {log_path}")


def main():
    """Main function for inference."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference with best checkpoint
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt

  # Run inference with last checkpoint
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/last.pt

  # Override data directory
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt --data_dir data/new_dataset
        """,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., runs/base/weights/best.pt)",
    )
    parser.add_argument("--data_dir", type=str, help="Override data directory")

    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found: {args.checkpoint_path}")
        return

    # Auto-extract run_dir from checkpoint path
    # Expected format: runs/run_name/weights/checkpoint.pt
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))  # Go up 2 levels

    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Run directory: {run_dir}")

    # Load config from run directory
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"config.yaml not found in {run_dir}")
        logger.error(f"Expected path: {config_path}")
        logger.error("Make sure checkpoint path follows format: runs/run_name/weights/checkpoint.pt")
        return

    config = load_config(config_path)

    # Setup logging
    setup_logging(run_dir)

    logger.info(f"Loaded config from {config_path}")

    # Override data directory if provided
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    # Setup CUDA
    cudnn.benchmark = True

    # Determine device
    device_str = config["training"]["device"]
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    fold = config["data"].get("fold", 0)
    logger.info(f"Using fold: {fold}")
    datasets = get_datasets(config)
    class_names = get_class_names(datasets)
    logger.info(f"Classes: {class_names}")

    # Create dataloaders
    dataloaders = get_dataloaders(datasets, config)
    dataset_sizes = get_dataset_sizes(datasets)
    logger.info(f"Test dataset size: {dataset_sizes['test']}")

    # Create model
    logger.info("Creating model...")
    model = get_model(config, device)

    # Load checkpoint (already validated at the beginning)
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    model = load_model(model, args.checkpoint_path, device)

    # Run inference
    logger.info("=" * 50)
    logger.info("Running Inference")
    logger.info("=" * 50)

    test_acc, per_sample_results = test_model(
        model=model,
        dataloader=dataloaders["test"],
        dataset_size=dataset_sizes["test"],
        device=device,
        class_names=class_names,
    )

    logger.info("=" * 50)
    logger.success("Inference Complete!")
    logger.info("=" * 50)

    # Extract labels and predictions for metrics
    # per_sample_results contains (true_label_name, pred_label_name, is_correct)
    # We need to convert back to indices for confusion matrix
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_labels = [class_to_idx[true_label] for true_label, _, _ in per_sample_results]
    test_preds = [class_to_idx[pred_label] for _, pred_label, _ in per_sample_results]

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")

    # Log test metrics to TensorBoard
    logger.info("Generating test metrics...")

    # Log test accuracy
    writer.add_scalar("Test/Accuracy", test_acc, 0)

    # Log confusion matrix
    log_confusion_matrix_to_tensorboard(
        writer, test_labels, test_preds, class_names, "Confusion_Matrix/test", 0
    )

    # Log classification report
    test_report = get_classification_report_str(test_labels, test_preds, class_names)
    writer.add_text("Classification_Report/test", test_report, 0)

    # Save classification report to file (for backward compatibility)
    save_classification_report(
        test_labels,
        test_preds,
        class_names,
        os.path.join(run_dir, "logs", "classification_report_test.txt"),
    )

    # Close TensorBoard writer
    writer.close()
    logger.info("TensorBoard writer closed")

    # Display per-sample results
    console = Console()

    # Per-sample results table
    sample_table = Table(
        title="Per-Sample Results", show_header=True, header_style="bold magenta"
    )
    sample_table.add_column("Sample #", style="cyan", width=10)
    sample_table.add_column("True Label", style="blue", width=15)
    sample_table.add_column("Predicted", style="yellow", width=15)
    sample_table.add_column("Correct", style="white", width=10)

    for idx, (true_label, pred_label, is_correct) in enumerate(per_sample_results, 1):
        status = "✓" if is_correct else "✗"
        status_style = "green" if is_correct else "red"
        sample_table.add_row(
            str(idx),
            str(true_label),
            str(pred_label),
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print("\n")
    console.print(sample_table)
    console.print("\n")

    # Summary table
    summary_table = Table(
        title="Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", width=15)

    summary_table.add_row("Run Directory", run_dir)
    summary_table.add_row("Checkpoint", os.path.basename(args.checkpoint_path))
    summary_table.add_row("Model", "ResNet18")
    summary_table.add_row("Test Samples", str(dataset_sizes["test"]))
    summary_table.add_row("Mean Accuracy", f"{test_acc:.4f}")

    console.print(summary_table)
    console.print("\n")

    # Generate and print classification report
    report = classification_report(test_labels, test_preds, target_names=class_names)
    console.print(Panel(report, title="Classification Report", expand=False))

    # Remind user about TensorBoard
    console.print("\n")
    console.print("[bold green]View test metrics in TensorBoard:[/bold green]")
    console.print(f"  tensorboard --logdir {run_dir}/tensorboard")
    console.print("\n")


if __name__ == "__main__":
    main()
