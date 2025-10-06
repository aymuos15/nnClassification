"""Test-Time Augmentation (TTA) transforms and utilities.

This module provides augmentation strategies for TTA during inference.
TTA applies multiple augmented versions of each test image and aggregates
predictions to improve model robustness and accuracy.
"""

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


class TTATransform:
    """
    Test-Time Augmentation transform wrapper.

    Applies a set of augmentations to create multiple versions of each image.
    Each augmentation can be inverted after prediction to align outputs.

    Example:
        >>> tta = TTATransform(['horizontal_flip', 'vertical_flip'])
        >>> augmented_images = tta.apply(image)
        >>> # augmented_images contains [original, h_flip, v_flip, h+v_flip]
    """

    def __init__(self, augmentations=None):
        """
        Initialize TTA transform.

        Args:
            augmentations: List of augmentation names to apply.
                Options: 'horizontal_flip', 'vertical_flip', 'rotate_90',
                        'rotate_180', 'rotate_270', 'brightness', 'contrast'
                If None, defaults to ['horizontal_flip']
        """
        if augmentations is None:
            augmentations = ["horizontal_flip"]

        self.augmentations = augmentations
        self._validate_augmentations()

    def _validate_augmentations(self):
        """Validate that all augmentations are supported."""
        valid_augs = {
            "horizontal_flip",
            "vertical_flip",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "brightness",
            "contrast",
        }
        for aug in self.augmentations:
            if aug not in valid_augs:
                raise ValueError(
                    f"Unknown augmentation: '{aug}'. "
                    f"Valid options: {sorted(valid_augs)}"
                )

    def apply(self, image_tensor):
        """
        Apply all augmentations to create multiple versions of the image.

        Args:
            image_tensor: Input image tensor (C, H, W)

        Returns:
            List of augmented image tensors, including original
        """
        augmented = [image_tensor]  # Always include original

        # Single augmentations
        if "horizontal_flip" in self.augmentations:
            augmented.append(TF.hflip(image_tensor))

        if "vertical_flip" in self.augmentations:
            augmented.append(TF.vflip(image_tensor))

        if "rotate_90" in self.augmentations:
            augmented.append(TF.rotate(image_tensor, 90))

        if "rotate_180" in self.augmentations:
            augmented.append(TF.rotate(image_tensor, 180))

        if "rotate_270" in self.augmentations:
            augmented.append(TF.rotate(image_tensor, 270))

        # Combined augmentations (horizontal + vertical flip)
        if "horizontal_flip" in self.augmentations and "vertical_flip" in self.augmentations:
            augmented.append(TF.vflip(TF.hflip(image_tensor)))

        return augmented


def get_tta_transforms(augmentations):
    """
    Factory function to create TTA transform.

    Args:
        augmentations: List of augmentation names or 'default'

    Returns:
        TTATransform instance

    Example:
        >>> tta = get_tta_transforms(['horizontal_flip', 'rotate_90'])
        >>> tta = get_tta_transforms('default')  # Uses horizontal_flip only
    """
    if augmentations == "default":
        augmentations = ["horizontal_flip"]

    return TTATransform(augmentations=augmentations)


def aggregate_predictions(predictions, method="mean"):
    """
    Aggregate predictions from multiple TTA augmentations.

    Args:
        predictions: List of prediction tensors (logits or probabilities)
                    Each tensor shape: (batch_size, num_classes)
        method: Aggregation method - 'mean', 'max', or 'voting'

    Returns:
        Aggregated predictions tensor (batch_size, num_classes)

    Example:
        >>> # Soft voting (average logits)
        >>> preds = [model(img1), model(img2), model(img3)]
        >>> final = aggregate_predictions(preds, method='mean')
        >>>
        >>> # Hard voting (majority vote)
        >>> final = aggregate_predictions(preds, method='voting')
    """
    if method == "mean":
        # Soft voting: average logits/probabilities
        return torch.stack(predictions).mean(dim=0)

    elif method == "max":
        # Take maximum across augmentations
        return torch.stack(predictions).max(dim=0)[0]

    elif method == "voting":
        # Hard voting: majority vote on predicted classes
        # Convert logits to class predictions
        class_preds = torch.stack([pred.argmax(dim=1) for pred in predictions])
        # Mode (most common prediction) for each sample
        # Use bincount approach for each sample
        batch_size = class_preds.shape[1]
        num_classes = predictions[0].shape[1]

        voted = []
        for i in range(batch_size):
            votes = class_preds[:, i]
            # Count votes for each class
            counts = torch.bincount(votes, minlength=num_classes)
            voted.append(counts)

        # Return as "logits" (vote counts)
        return torch.stack(voted).float()

    else:
        raise ValueError(
            f"Unknown aggregation method: '{method}'. "
            f"Valid options: 'mean', 'max', 'voting'"
        )


__all__ = ["TTATransform", "get_tta_transforms", "aggregate_predictions"]
