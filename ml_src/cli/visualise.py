#!/usr/bin/env python3
"""
Visualization script for TensorBoard - visualize datasets and model predictions.
"""

import argparse
import os
import shutil
import subprocess

import torch
import torchvision
import yaml
from loguru import logger
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.dataset import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders
from ml_src.core.network import get_model, load_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging():
    """Setup loguru logging."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level="INFO",
    )


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor: Tensor image of size (C, H, W) or (B, C, H, W)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch of images
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def visualize_samples(run_dir, config, split="train", num_images=16):
    """
    Visualize sample images from dataset and log to TensorBoard.

    Args:
        run_dir: Run directory path
        config: Configuration dictionary
        split: Dataset split to visualize ('train', 'val', or 'test')
        num_images: Number of images to visualize
    """
    logger.info(f"Visualizing {num_images} sample images from {split} split...")

    # Load datasets
    datasets = get_datasets(config)
    class_names = get_class_names(datasets)

    # Create dataloader with shuffle for variety
    dataloaders = get_dataloaders(datasets, config)
    dataloader = dataloaders[split]

    # Get one batch
    images, labels = next(iter(dataloader))

    # Limit to num_images
    images = images[:num_images]
    labels = labels[:num_images]

    # Denormalize images for visualization
    mean = config["transforms"][split]["normalize"]["mean"]
    std = config["transforms"][split]["normalize"]["std"]
    images_denorm = denormalize(images, mean, std)

    # Clamp values to [0, 1] for display
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Create grid
    grid = torchvision.utils.make_grid(
        images_denorm, nrow=4, padding=2, normalize=False
    )

    # Setup TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Log to TensorBoard
    writer.add_image(f"Dataset_Samples/{split}", grid, 0)

    # Also log individual images with labels
    for idx, (img, label) in enumerate(zip(images_denorm, labels)):
        class_name = class_names[label.item()]
        writer.add_image(
            f"Dataset_Samples/{split}_individual/{class_name}_{idx}", img, 0
        )

    writer.close()

    logger.success(f"Logged {len(images)} sample images to TensorBoard")
    logger.info(f"View in TensorBoard: tensorboard --logdir {run_dir}/tensorboard")


def add_colored_border(image_tensor, color, border_width=5):
    """
    Add colored border to image tensor.

    Args:
        image_tensor: Tensor image of size (C, H, W)
        color: RGB tuple for border color
        border_width: Width of border in pixels

    Returns:
        Image tensor with colored border
    """
    # Convert to PIL Image
    img = torchvision.transforms.ToPILImage()(image_tensor)

    # Create a new image with border
    new_width = img.width + 2 * border_width
    new_height = img.height + 2 * border_width
    bordered_img = Image.new("RGB", (new_width, new_height), color)
    bordered_img.paste(img, (border_width, border_width))

    # Convert back to tensor
    return torchvision.transforms.ToTensor()(bordered_img)


def visualize_predictions(
    run_dir, config, checkpoint="best.pt", split="val", num_images=16
):
    """
    Visualize model predictions and log to TensorBoard.

    Args:
        run_dir: Run directory path
        config: Configuration dictionary
        checkpoint: Checkpoint file name ('best.pt' or 'last.pt')
        split: Dataset split to visualize ('train', 'val', or 'test')
        num_images: Number of images to visualize
    """
    logger.info(f"Visualizing predictions on {split} split using {checkpoint}...")

    # Determine device
    device_str = config["training"]["device"]
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load datasets
    datasets = get_datasets(config)
    class_names = get_class_names(datasets)
    dataloaders = get_dataloaders(datasets, config)
    dataloader = dataloaders[split]

    # Create and load model
    model = get_model(config, device)
    checkpoint_path = os.path.join(run_dir, "weights", checkpoint)

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint {checkpoint_path} not found")
        return

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model = load_model(model, checkpoint_path, device)
    model.eval()

    # Get one batch
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Get predictions
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        _, preds = torch.max(outputs, 1)

    preds = preds.cpu()

    # Denormalize images
    mean = config["transforms"][split]["normalize"]["mean"]
    std = config["transforms"][split]["normalize"]["std"]
    images_denorm = denormalize(images, mean, std)
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # Add colored borders (green for correct, red for incorrect)
    bordered_images = []
    for img, true_label, pred_label in zip(images_denorm, labels, preds):
        is_correct = true_label.item() == pred_label.item()
        color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green or Red
        bordered_img = add_colored_border(img, color, border_width=5)
        bordered_images.append(bordered_img)

    bordered_images = torch.stack(bordered_images)

    # Create grid
    grid = torchvision.utils.make_grid(
        bordered_images, nrow=4, padding=2, normalize=False
    )

    # Setup TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Log to TensorBoard
    writer.add_image(f"Predictions/{split}", grid, 0)

    # Calculate accuracy for this batch
    correct = (preds == labels).sum().item()
    accuracy = correct / len(labels)

    # Log individual predictions with metadata
    for idx, (img, true_label, pred_label) in enumerate(
        zip(bordered_images, labels, preds)
    ):
        true_class = class_names[true_label.item()]
        pred_class = class_names[pred_label.item()]
        is_correct = true_label.item() == pred_label.item()
        status = "Correct" if is_correct else "Incorrect"

        tag = f"Predictions/{split}_individual/{status}/{idx}_true_{true_class}_pred_{pred_class}"
        writer.add_image(tag, img, 0)

    writer.close()

    logger.success(
        f"Logged {len(images)} predictions to TensorBoard (Accuracy: {accuracy:.2%})"
    )
    logger.info("Green border = Correct prediction, Red border = Incorrect prediction")
    logger.info(f"View in TensorBoard: tensorboard --logdir {run_dir}/tensorboard")


def launch_tensorboard(run_dir, port=6006):
    """
    Launch TensorBoard server.

    Args:
        run_dir: Run directory path
        port: Port number for TensorBoard server
    """
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    if not os.path.exists(tensorboard_dir):
        logger.warning(f"TensorBoard directory not found: {tensorboard_dir}")
        logger.info("Run training or visualization first to generate TensorBoard logs")
        return

    logger.info(f"Launching TensorBoard on port {port}...")
    logger.info(f"TensorBoard directory: {tensorboard_dir}")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info("Press Ctrl+C to stop TensorBoard")

    try:
        subprocess.run(
            ["tensorboard", "--logdir", tensorboard_dir, "--port", str(port)]
        )
    except KeyboardInterrupt:
        logger.info("\nTensorBoard stopped")
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install with: pip install tensorboard")


def clean_tensorboard_logs(run_dir=None):
    """
    Clean TensorBoard logs.

    Args:
        run_dir: Specific run directory to clean, or None to clean all runs
    """
    if run_dir:
        # Clean specific run directory
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        if os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
            logger.success(f"Removed TensorBoard logs from {run_dir}")
        else:
            logger.warning(f"No TensorBoard logs found in {run_dir}")
    else:
        # Clean all runs
        if not os.path.exists("runs"):
            logger.warning("No runs directory found")
            return

        cleaned_count = 0
        for run_name in os.listdir("runs"):
            run_path = os.path.join("runs", run_name)
            if os.path.isdir(run_path):
                tensorboard_dir = os.path.join(run_path, "tensorboard")
                if os.path.exists(tensorboard_dir):
                    shutil.rmtree(tensorboard_dir)
                    logger.info(f"Removed TensorBoard logs from {run_path}")
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.success(f"Cleaned TensorBoard logs from {cleaned_count} run(s)")
        else:
            logger.warning("No TensorBoard logs found in any runs")


def main():
    """Main function for visualization."""
    parser = argparse.ArgumentParser(
        description="TensorBoard visualization tool for datasets and predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch TensorBoard for a specific run
  python visualise.py --mode launch --run_dir runs/base

  # Visualize sample images from training set
  python visualise.py --mode samples --run_dir runs/base --split train --num_images 16

  # Visualize model predictions on validation set
  python visualise.py --mode predictions --run_dir runs/base --split val --checkpoint best.pt

  # Clean TensorBoard logs from all runs
  python visualise.py --mode clean

  # Clean TensorBoard logs from specific run
  python visualise.py --mode clean --run_dir runs/base
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
    setup_logging()

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
        visualize_predictions(
            args.run_dir, config, args.checkpoint, args.split, args.num_images
        )


if __name__ == "__main__":
    main()
