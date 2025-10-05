#!/usr/bin/env python3
"""
Inference script for evaluating trained models.
"""

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.config import load_config
from ml_src.core.data import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.logging import setup_logging
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)
from ml_src.core.network import get_model, load_model
from ml_src.core.run import get_run_dir_from_checkpoint
from ml_src.core.test import test_model
from ml_src.core.ui import display_inference_results


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
    run_dir = get_run_dir_from_checkpoint(args.checkpoint_path)

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
    setup_logging(run_dir, filename="inference.log")

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

    # Save classification report to file
    save_classification_report(
        test_labels,
        test_preds,
        class_names,
        os.path.join(run_dir, "logs", "classification_report_test.txt"),
    )

    writer.close()
    logger.info("TensorBoard writer closed")

    # Display results using rich tables
    display_inference_results(
        per_sample_results, test_acc, dataset_sizes["test"], run_dir, args.checkpoint_path, class_names
    )


if __name__ == "__main__":
    main()
