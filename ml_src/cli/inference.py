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
from ml_src.core.inference import get_inference_strategy
from ml_src.core.loader import get_dataloaders, get_dataset_sizes
from ml_src.core.logging import setup_logging
from ml_src.core.metrics import (
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
)
from ml_src.core.network import get_model, load_model
from ml_src.core.run import get_run_dir_from_checkpoint
from ml_src.core.ui import display_inference_results


def main():
    """Main function for inference."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard inference with best checkpoint
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt

  # TTA inference with horizontal flip
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt --tta

  # TTA with custom augmentations
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt --tta --tta-augmentations horizontal_flip vertical_flip

  # Ensemble inference from multiple folds
  ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt runs/fold_2/weights/best.pt

  # Combined TTA + Ensemble for maximum performance
  ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt --tta

  # Override data directory
  ml-inference --checkpoint_path runs/my_dataset_base_fold_0/weights/best.pt --data_dir data/new_dataset
        """,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to checkpoint file (e.g., runs/base/weights/best.pt). Not required for ensemble.",
    )
    parser.add_argument("--data_dir", type=str, help="Override data directory")

    # TTA arguments
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test-Time Augmentation for improved robustness (slower, higher accuracy)",
    )
    parser.add_argument(
        "--tta-augmentations",
        nargs="+",
        default=["horizontal_flip"],
        help="TTA augmentations to apply (default: horizontal_flip). "
        "Options: horizontal_flip, vertical_flip, rotate_90, rotate_180, rotate_270",
    )
    parser.add_argument(
        "--tta-aggregation",
        choices=["mean", "max", "voting"],
        default="mean",
        help="How to aggregate TTA predictions (default: mean)",
    )

    # Ensemble arguments
    parser.add_argument(
        "--ensemble",
        nargs="+",
        metavar="CHECKPOINT",
        help="Ensemble multiple checkpoints (e.g., --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt)",
    )
    parser.add_argument(
        "--ensemble-aggregation",
        choices=["soft_voting", "hard_voting", "weighted"],
        default="soft_voting",
        help="How to aggregate ensemble predictions (default: soft_voting)",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        help="Weights for weighted ensemble (must match number of checkpoints)",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights from checkpoints when available",
    )

    args = parser.parse_args()

    # Determine checkpoint source
    if args.ensemble:
        # Ensemble mode
        checkpoints = args.ensemble
        is_ensemble = True
    elif args.checkpoint_path:
        # Single checkpoint mode
        checkpoints = [args.checkpoint_path]
        is_ensemble = False
    else:
        parser.error("Either --checkpoint_path or --ensemble is required")
        return

    # Validate checkpoints exist
    for checkpoint in checkpoints:
        if not os.path.exists(checkpoint):
            logger.error(f"Checkpoint not found: {checkpoint}")
            return

    # Auto-extract run_dir from first checkpoint path
    run_dir = get_run_dir_from_checkpoint(checkpoints[0])

    if is_ensemble:
        logger.info(f"Ensemble mode with {len(checkpoints)} checkpoints")
        for i, ckpt in enumerate(checkpoints):
            logger.info(f"  Checkpoint {i + 1}: {ckpt}")
    else:
        logger.info(f"Checkpoint: {checkpoints[0]}")

    logger.info(f"Run directory: {run_dir}")

    # Load config from run directory
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"config.yaml not found in {run_dir}")
        logger.error(f"Expected path: {config_path}")
        logger.error(
            "Make sure checkpoint path follows format: runs/run_name/weights/checkpoint.pt"
        )
        return

    config = load_config(config_path)

    # Setup logging
    setup_logging(run_dir, filename="inference.log")

    logger.info(f"Loaded config from {config_path}")

    # Override data directory if provided
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    # Override config with CLI arguments for TTA/Ensemble
    if args.use_ema:
        config.setdefault("inference", {})
        config["inference"]["use_ema"] = True

    if args.tta and is_ensemble:
        # TTA + Ensemble mode
        config["inference"] = {
            "strategy": "tta_ensemble",
            "tta": {
                "augmentations": args.tta_augmentations,
                "aggregation": args.tta_aggregation,
            },
            "ensemble": {
                "checkpoints": checkpoints,
                "aggregation": args.ensemble_aggregation,
                "weights": args.ensemble_weights,
            },
            "use_ema": args.use_ema,
        }
        logger.info("Using TTA + Ensemble inference strategy")
    elif args.tta:
        # TTA only mode
        config["inference"] = {
            "strategy": "tta",
            "tta": {
                "augmentations": args.tta_augmentations,
                "aggregation": args.tta_aggregation,
            },
            "use_ema": args.use_ema,
        }
        logger.info("Using TTA inference strategy")
    elif is_ensemble:
        # Ensemble only mode
        config["inference"] = {
            "strategy": "ensemble",
            "ensemble": {
                "checkpoints": checkpoints,
                "aggregation": args.ensemble_aggregation,
                "weights": args.ensemble_weights,
            },
            "use_ema": args.use_ema,
        }
        logger.info("Using Ensemble inference strategy")
    # Otherwise, use strategy from config file (default: standard)

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

    # Create and load model (skip for ensemble - models loaded in strategy)
    if is_ensemble:
        # For ensemble, model loading is handled by the strategy
        model = None
        logger.info("Ensemble mode: models will be loaded by inference strategy")
    else:
        # Standard single-model inference
        logger.info("Creating model...")
        model = get_model(config, device)

        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoints[0]}")
        use_ema = config.get("inference", {}).get("use_ema", False)
        model = load_model(model, checkpoints[0], device, use_ema=use_ema)

    # Run inference
    logger.info("=" * 50)
    logger.info("Running Inference")
    logger.info("=" * 50)

    # Get inference strategy from config (pass device for ensemble strategies)
    strategy = get_inference_strategy(config, device=device)

    test_acc, per_sample_results = strategy.run_inference(
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
    checkpoint_display = checkpoints[0] if not is_ensemble else f"Ensemble ({len(checkpoints)} models)"
    display_inference_results(
        per_sample_results,
        test_acc,
        dataset_sizes["test"],
        run_dir,
        checkpoint_display,
        class_names,
    )


if __name__ == "__main__":
    main()
