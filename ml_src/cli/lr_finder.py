#!/usr/bin/env python3
"""
Learning Rate Finder CLI tool.

This tool helps find an optimal learning rate for training by performing a learning rate
range test. It trains the model for a small number of iterations while exponentially
increasing the learning rate, recording the loss at each step. The optimal learning rate
is typically found at the steepest descent in the loss curve.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from loguru import logger

from ml_src.core.config import load_config
from ml_src.core.data import get_datasets
from ml_src.core.loader import get_dataloaders
from ml_src.core.logging import setup_logging
from ml_src.core.loss import get_criterion
from ml_src.core.lr_finder import LRFinder, plot_lr_finder
from ml_src.core.network import get_model
from ml_src.core.seeding import set_seed


def main():
    """Main function to run the learning rate finder."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Find optimal learning rate using LR range test",
        epilog="""
Examples:
  # Basic usage with default settings
  ml-lr-finder --config configs/my_config.yaml

  # Custom LR range and iterations
  ml-lr-finder --config configs/my_config.yaml --start_lr 1e-7 --end_lr 1 --num_iter 200

  # Use specific fold for cross-validation
  ml-lr-finder --config configs/my_config.yaml --fold 2

  # Adjust smoothing factor
  ml-lr-finder --config configs/my_config.yaml --beta 0.95

Usage Tips:
  - The tool will test learning rates from start_lr to end_lr over num_iter iterations
  - Loss is smoothed using exponential moving average (controlled by beta)
  - Suggested LR is typically 1/10th of the LR at the steepest descent point
  - Run this before training to find a good initial learning rate
  - Results are saved to runs/lr_finder_TIMESTAMP/

Output:
  - lr_plot.png: Visualization of LR vs Loss curve with suggested LR marked
  - results.json: Learning rates, losses, and suggested LR in JSON format
  - lr_finder.log: Detailed logs of the LR range test
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (use 'ml-init-config' to generate one)",
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default=1e-8,
        help="Starting learning rate for range test (default: 1e-8)",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=10.0,
        help="Ending learning rate for range test (default: 10.0)",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=100,
        help="Number of iterations to run (default: 100)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation (0-indexed, default: 0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.98,
        help="Smoothing factor for loss (exponential moving average, default: 0.98)",
    )
    parser.add_argument(
        "--diverge_threshold",
        type=float,
        default=4.0,
        help="Early stopping threshold: stop when loss > threshold * min_loss (default: 4.0)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override fold if specified
    if "data" not in config:
        config["data"] = {}
    config["data"]["fold"] = args.fold

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), "runs", f"lr_finder_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

    # Setup logging
    setup_logging(output_dir, filename="lr_finder.log")

    logger.info("=" * 70)
    logger.info("Learning Rate Finder")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"LR range: {args.start_lr:.2e} -> {args.end_lr:.2e}")
    logger.info(f"Iterations: {args.num_iter}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Beta (smoothing): {args.beta}")
    logger.info(f"Divergence threshold: {args.diverge_threshold}x")

    # Set random seed for reproducibility (but disable deterministic for speed)
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=False)
    logger.info(f"Random seed: {seed} (deterministic=False for faster execution)")

    # Determine device
    device_str = config["training"]["device"]
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    datasets = get_datasets(config)

    # Create dataloaders (we only need train loader for LR finder)
    dataloaders = get_dataloaders(datasets, config)
    train_loader = dataloaders["train"]
    logger.info(f"Training samples: {len(datasets['train'])}")
    logger.info(f"Training batches: {len(train_loader)}")

    # Create model
    logger.info("Creating model...")
    model = get_model(config, device)

    # Get number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,} total, {num_trainable:,} trainable")

    # Create criterion
    criterion = get_criterion(config)

    # Create optimizer function
    # This function creates an optimizer with a given learning rate
    optimizer_type = config["optimizer"]["type"]
    momentum = config["optimizer"].get("momentum", 0.9)

    def optimizer_fn(lr):
        """Create optimizer with specified learning rate."""
        if optimizer_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            logger.warning(f"Unknown optimizer type '{optimizer_type}', using SGD")
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    logger.info(f"Optimizer: {optimizer_type}")
    if optimizer_type == "sgd":
        logger.info(f"Momentum: {momentum}")

    # Run LR Finder
    logger.info("=" * 70)
    logger.info("Running LR Range Test...")
    logger.info("=" * 70)

    finder = LRFinder()
    lrs, losses, suggested_lr = finder.find_lr(
        model=model,
        train_loader=train_loader,
        optimizer_fn=optimizer_fn,
        criterion=criterion,
        device=device,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        beta=args.beta,
        diverge_threshold=args.diverge_threshold,
    )

    # Check if we got valid results
    if len(lrs) == 0:
        logger.error("LR range test failed to produce results")
        logger.error("This could be due to:")
        logger.error("  - Model initialization issues")
        logger.error("  - Data loading problems")
        logger.error("  - NaN/Inf losses from the start")
        logger.error("Please check your configuration and try again")
        return

    logger.info("=" * 70)
    logger.success(f"Suggested Learning Rate: {suggested_lr:.2e}")
    logger.info("=" * 70)

    # Plot results
    plot_path = os.path.join(output_dir, "lr_plot.png")
    logger.info(f"Creating visualization...")
    plot_lr_finder(lrs, losses, suggested_lr, plot_path)

    # Save results to JSON
    results_path = os.path.join(output_dir, "results.json")
    results = {
        "learning_rates": lrs,
        "losses": losses,
        "suggested_lr": suggested_lr,
        "config": {
            "start_lr": args.start_lr,
            "end_lr": args.end_lr,
            "num_iter": args.num_iter,
            "beta": args.beta,
            "diverge_threshold": args.diverge_threshold,
            "fold": args.fold,
        },
        "model": {
            "type": config["model"]["type"],
            "architecture": config["model"].get("architecture")
            or config["model"].get("custom_architecture"),
            "num_parameters": num_params,
            "num_trainable": num_trainable,
        },
        "optimizer": {
            "type": optimizer_type,
            "momentum": momentum if optimizer_type == "sgd" else None,
        },
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Print summary
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Tested {len(lrs)} learning rates")
    logger.info(f"Min loss: {min(losses):.6f}")
    logger.info(f"Max loss: {max(losses):.6f}")
    logger.success(f"Suggested LR: {suggested_lr:.2e}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Review the plot: {plot_path}")
    logger.info(f"  2. Update your config with: optimizer.lr = {suggested_lr:.2e}")
    logger.info(f"  3. Consider trying LR values around {suggested_lr/2:.2e} to {suggested_lr*2:.2e}")
    logger.info(f"  4. Start training: ml-train --config {args.config} --lr {suggested_lr:.2e}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
