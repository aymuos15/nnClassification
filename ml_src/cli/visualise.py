#!/usr/bin/env python3
"""
Visualization script for TensorBoard - visualize datasets and model predictions.
"""

import argparse
import os

from loguru import logger

from ml_src.core.config import load_config
from ml_src.core.logging import setup_logging
from ml_src.core.visual import (
    clean_tensorboard_logs,
    launch_tensorboard,
    visualize_predictions,
    visualize_samples,
)


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(
        description="TensorBoard visualization tool for datasets and predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch TensorBoard for a specific run
  ml-visualise --mode launch --run_dir runs/base

  # Visualize sample images from training set
  ml-visualise --mode samples --run_dir runs/base --split train --num_images 16

  # Visualize model predictions on validation set
  ml-visualise --mode predictions --run_dir runs/base --split val --checkpoint best.pt

  # Clean TensorBoard logs from all runs
  ml-visualise --mode clean

  # Clean TensorBoard logs from specific run
  ml-visualise --mode clean --run_dir runs/base
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["launch", "samples", "predictions", "clean"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/base",
        help="Path to run directory (default: runs/base)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to visualize (default: val)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="Number of images to visualize (default: 16)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint to use for predictions (default: best.pt)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for TensorBoard server (default: 6006)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()  # Console only

    # Handle clean mode (doesn't require config)
    if args.mode == "clean":
        if args.run_dir == "runs/base":
            # Default value, clean all
            logger.info("Cleaning TensorBoard logs from all runs...")
            clean_tensorboard_logs(run_dir=None)
        else:
            # Specific run directory provided
            clean_tensorboard_logs(run_dir=args.run_dir)
        return

    # Handle launch mode (doesn't require config)
    if args.mode == "launch":
        launch_tensorboard(args.run_dir, args.port)
        return

    # For samples and predictions modes, load config
    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"config.yaml not found in {args.run_dir}")
        logger.error("Please specify a valid run directory with --run_dir")
        return

    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Execute requested mode
    if args.mode == "samples":
        visualize_samples(args.run_dir, config, args.split, args.num_images)
    elif args.mode == "predictions":
        visualize_predictions(args.run_dir, config, args.checkpoint, args.split, args.num_images)


if __name__ == "__main__":
    main()
