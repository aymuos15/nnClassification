"""Segmentation metrics: IoU, Dice, pixel accuracy."""

import numpy as np
import torch
from loguru import logger


def calculate_iou(pred_mask, true_mask, num_classes, ignore_index=-100):
    """
    Calculate mean Intersection over Union (IoU) for segmentation.

    IoU measures the overlap between predicted and ground truth masks.
    For each class: IoU = intersection / union

    Args:
        pred_mask: Predicted mask tensor [B, H, W] or [H, W]
        true_mask: Ground truth mask tensor [B, H, W] or [H, W]
        num_classes: Number of classes
        ignore_index: Class index to ignore in calculation (default: -100)

    Returns:
        float: Mean IoU across all classes (ignoring NaN values for absent classes)

    Example:
        >>> pred = torch.tensor([[0, 0, 1], [1, 1, 2]])
        >>> true = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> iou = calculate_iou(pred, true, num_classes=3)
    """
    ious = []

    # Ensure masks are numpy arrays for computation
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    # Filter out ignored indices
    if ignore_index >= 0:
        valid_mask = true_mask != ignore_index
        pred_mask = pred_mask[valid_mask]
        true_mask = true_mask[valid_mask]

    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        true_cls = true_mask == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            # Class not present in ground truth
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    # Mean IoU (ignoring NaN values for classes not present)
    ious = np.array(ious)
    return np.nanmean(ious)


def calculate_dice(pred_mask, true_mask, num_classes, ignore_index=-100):
    """
    Calculate Dice coefficient for segmentation.

    Dice coefficient is 2 * intersection / (pred + true).
    Similar to IoU but weights overlap differently.

    Args:
        pred_mask: Predicted mask tensor [B, H, W] or [H, W]
        true_mask: Ground truth mask tensor [B, H, W] or [H, W]
        num_classes: Number of classes
        ignore_index: Class index to ignore (default: -100)

    Returns:
        float: Mean Dice coefficient across all classes

    Example:
        >>> pred = torch.tensor([[0, 0, 1], [1, 1, 2]])
        >>> true = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> dice = calculate_dice(pred, true, num_classes=3)
    """
    dice_scores = []

    # Ensure masks are numpy arrays
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()

    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    # Filter out ignored indices
    if ignore_index >= 0:
        valid_mask = true_mask != ignore_index
        pred_mask = pred_mask[valid_mask]
        true_mask = true_mask[valid_mask]

    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        true_cls = true_mask == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        dice = (2.0 * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)
        dice_scores.append(dice)

    return np.mean(dice_scores)


def calculate_pixel_accuracy(pred_mask, true_mask, ignore_index=-100):
    """
    Calculate pixel-wise accuracy.

    Simple metric: percentage of correctly classified pixels.

    Args:
        pred_mask: Predicted mask tensor [B, H, W] or [H, W]
        true_mask: Ground truth mask tensor [B, H, W] or [H, W]
        ignore_index: Class index to ignore (default: -100)

    Returns:
        float: Pixel accuracy (0.0 to 1.0)

    Example:
        >>> pred = torch.tensor([[0, 0, 1], [1, 1, 2]])
        >>> true = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> acc = calculate_pixel_accuracy(pred, true)
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu()
        true_mask = true_mask.cpu()

        # Create mask for valid pixels
        if ignore_index >= 0:
            valid_mask = true_mask != ignore_index
            correct = ((pred_mask == true_mask) & valid_mask).sum().item()
            total = valid_mask.sum().item()
        else:
            correct = (pred_mask == true_mask).sum().item()
            total = pred_mask.numel()
    else:
        pred_mask = np.array(pred_mask).flatten()
        true_mask = np.array(true_mask).flatten()

        if ignore_index >= 0:
            valid_mask = true_mask != ignore_index
            pred_mask = pred_mask[valid_mask]
            true_mask = true_mask[valid_mask]

        correct = (pred_mask == true_mask).sum()
        total = len(pred_mask)

    return correct / total if total > 0 else 0.0


def get_per_class_iou(pred_masks, true_masks, num_classes, ignore_index=-100):
    """
    Calculate per-class IoU scores.

    Args:
        pred_masks: Predicted masks (tensor or array)
        true_masks: Ground truth masks (tensor or array)
        num_classes: Number of classes
        ignore_index: Class index to ignore (default: -100)

    Returns:
        numpy.ndarray: Per-class IoU scores [num_classes]

    Example:
        >>> pred = torch.tensor([[0, 0, 1], [1, 1, 2]])
        >>> true = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> per_class_iou = get_per_class_iou(pred, true, num_classes=3)
    """
    ious = []

    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(true_masks, torch.Tensor):
        true_masks = true_masks.cpu().numpy()

    pred_masks = pred_masks.flatten()
    true_masks = true_masks.flatten()

    # Filter out ignored indices
    if ignore_index >= 0:
        valid_mask = true_masks != ignore_index
        pred_masks = pred_masks[valid_mask]
        true_masks = true_masks[valid_mask]

    for cls in range(num_classes):
        pred_cls = pred_masks == cls
        true_cls = true_masks == cls

        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()

        if union == 0:
            ious.append(0.0)  # Class not present
        else:
            ious.append(intersection / union)

    return np.array(ious)


def get_segmentation_report_str(pred_masks, true_masks, class_names, num_classes):
    """
    Generate segmentation report (similar to classification report).

    Args:
        pred_masks: All predicted masks (tensor, list, or array)
        true_masks: All ground truth masks (tensor, list, or array)
        class_names: List of class names
        num_classes: Number of classes

    Returns:
        str: Formatted report string with per-class IoU and Dice

    Example:
        >>> pred = torch.randint(0, 3, (10, 256, 256))
        >>> true = torch.randint(0, 3, (10, 256, 256))
        >>> report = get_segmentation_report_str(pred, true, ['bg', 'cat', 'dog'], 3)
        >>> print(report)
    """
    # Concatenate if list of tensors
    if isinstance(pred_masks, list):
        pred_masks = torch.cat(pred_masks) if isinstance(pred_masks[0], torch.Tensor) else np.concatenate(pred_masks)
    if isinstance(true_masks, list):
        true_masks = torch.cat(true_masks) if isinstance(true_masks[0], torch.Tensor) else np.concatenate(true_masks)

    report_lines = ["Segmentation Report", "=" * 80, ""]
    report_lines.append(f"{'Class':<20} {'IoU':<10} {'Dice':<10}")
    report_lines.append("-" * 80)

    # Calculate per-class metrics
    per_class_ious = get_per_class_iou(pred_masks, true_masks, num_classes)

    # Convert to numpy for calculations
    if isinstance(pred_masks, torch.Tensor):
        pred_masks_np = pred_masks.cpu().numpy().flatten()
        true_masks_np = true_masks.cpu().numpy().flatten()
    else:
        pred_masks_np = np.array(pred_masks).flatten()
        true_masks_np = np.array(true_masks).flatten()

    for cls in range(num_classes):
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"

        # IoU (already calculated)
        iou = per_class_ious[cls]

        # Calculate per-class Dice
        pred_cls = pred_masks_np == cls
        true_cls = true_masks_np == cls
        intersection = np.logical_and(pred_cls, true_cls).sum()
        dice = (2.0 * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)

        report_lines.append(f"{class_name:<20} {iou:<10.4f} {dice:<10.4f}")

    # Overall metrics
    mean_iou = np.mean(per_class_ious)
    mean_dice = calculate_dice(pred_masks, true_masks, num_classes)
    pixel_acc = calculate_pixel_accuracy(pred_masks, true_masks)

    report_lines.append("-" * 80)
    report_lines.append(f"{'Mean IoU':<20} {mean_iou:<10.4f}")
    report_lines.append(f"{'Mean Dice':<20} {mean_dice:<10.4f}")
    report_lines.append(f"{'Pixel Accuracy':<20} {pixel_acc:<10.4f}")

    return "\n".join(report_lines)


def save_segmentation_report(pred_masks, true_masks, class_names, num_classes, save_path):
    """
    Save segmentation report to file.

    Args:
        pred_masks: All predicted masks
        true_masks: All ground truth masks
        class_names: List of class names
        num_classes: Number of classes
        save_path: Path to save report (e.g., 'segmentation_report.txt')

    Example:
        >>> save_segmentation_report(pred, true, ['bg', 'cat'], 2, 'report.txt')
    """
    report = get_segmentation_report_str(pred_masks, true_masks, class_names, num_classes)

    with open(save_path, "w") as f:
        f.write(report)

    logger.info(f"Saved segmentation report to {save_path}")


__all__ = [
    "calculate_iou",
    "calculate_dice",
    "calculate_pixel_accuracy",
    "get_per_class_iou",
    "get_segmentation_report_str",
    "save_segmentation_report",
]
