"""TensorBoard visualization utilities."""

import os

import torch
import torchvision
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from ml_src.core.data import get_class_names, get_datasets
from ml_src.core.loader import get_dataloaders
from ml_src.core.network import get_model, load_model
from ml_src.core.visual.transforms import add_colored_border, denormalize


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

    # Create dataloader
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
    grid = torchvision.utils.make_grid(images_denorm, nrow=4, padding=2, normalize=False)

    # Setup TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Log to TensorBoard
    writer.add_image(f"Dataset_Samples/{split}", grid, 0)

    # Also log individual images with labels
    for idx, (img, label) in enumerate(zip(images_denorm, labels)):
        class_name = class_names[label.item()]
        writer.add_image(f"Dataset_Samples/{split}_individual/{class_name}_{idx}", img, 0)

    writer.close()

    logger.success(f"Logged {len(images)} sample images to TensorBoard")
    logger.info(f"View in TensorBoard: tensorboard --logdir {run_dir}/tensorboard")


def visualize_predictions(run_dir, config, checkpoint="best.pt", split="val", num_images=16):
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
    grid = torchvision.utils.make_grid(bordered_images, nrow=4, padding=2, normalize=False)

    # Setup TensorBoard writer
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    # Log to TensorBoard
    writer.add_image(f"Predictions/{split}", grid, 0)

    # Calculate accuracy for this batch
    correct = (preds == labels).sum().item()
    accuracy = correct / len(labels)

    # Log individual predictions with metadata
    for idx, (img, true_label, pred_label) in enumerate(zip(bordered_images, labels, preds)):
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
