"""Segmentation loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.

    The Dice coefficient is a measure of overlap between predicted and ground truth masks.
    Dice loss = 1 - Dice coefficient. Works well for imbalanced segmentation tasks.

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        ignore_index: Class index to ignore in loss calculation (default: -100)

    Example:
        >>> criterion = DiceLoss(smooth=1.0)
        >>> outputs = torch.randn(4, 3, 256, 256)  # [B, C, H, W] logits
        >>> targets = torch.randint(0, 3, (4, 256, 256))  # [B, H, W] class indices
        >>> loss = criterion(outputs, targets)
    """

    def __init__(self, smooth=1.0, ignore_index=-100, **kwargs):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        """
        Compute Dice loss.

        Args:
            outputs: Model predictions [batch_size, num_classes, H, W] (logits)
            targets: Ground truth masks [batch_size, H, W] (class indices)

        Returns:
            Scalar Dice loss value
        """
        num_classes = outputs.size(1)

        # Convert logits to probabilities
        outputs = F.softmax(outputs, dim=1)  # [B, C, H, W]

        # Flatten spatial dimensions
        outputs = outputs.view(outputs.size(0), num_classes, -1)  # [B, C, H*W]
        targets = targets.view(targets.size(0), -1)  # [B, H*W]

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # [B, H*W, C]
        targets_one_hot = targets_one_hot.permute(0, 2, 1).float()  # [B, C, H*W]

        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H*W]
            outputs = outputs * mask
            targets_one_hot = targets_one_hot * mask

        # Calculate Dice coefficient per class
        intersection = (outputs * targets_one_hot).sum(dim=2)  # [B, C]
        cardinality = outputs.sum(dim=2) + targets_one_hot.sum(dim=2)  # [B, C]

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over batch and classes
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for segmentation with class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.
    Useful when there's significant class imbalance in segmentation.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor for class balance (default: 0.25)
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples
        ignore_index: Class index to ignore (default: -100)

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> outputs = torch.randn(4, 3, 256, 256)  # [B, C, H, W]
        >>> targets = torch.randint(0, 3, (4, 256, 256))  # [B, H, W]
        >>> loss = criterion(outputs, targets)
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        """
        Compute focal loss.

        Args:
            outputs: Model predictions [batch_size, num_classes, H, W] (logits)
            targets: Ground truth masks [batch_size, H, W] (class indices)

        Returns:
            Scalar focal loss value
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(outputs, targets, reduction="none", ignore_index=self.ignore_index)

        # Get probabilities
        p = F.softmax(outputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # Probability of true class

        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation (e.g., Cross-Entropy + Dice).

    Combines multiple loss functions with configurable weights.
    Default: 50% Cross-Entropy + 50% Dice Loss

    Args:
        loss_weights: Dictionary of loss weights (default: {'ce': 0.5, 'dice': 0.5})
        ignore_index: Class index to ignore (default: -100)

    Example:
        >>> criterion = CombinedLoss(loss_weights={'ce': 0.4, 'dice': 0.6})
        >>> outputs = torch.randn(4, 3, 256, 256)
        >>> targets = torch.randint(0, 3, (4, 256, 256))
        >>> loss = criterion(outputs, targets)
    """

    def __init__(self, loss_weights=None, ignore_index=-100, **kwargs):
        super().__init__()
        self.loss_weights = loss_weights or {"ce": 0.5, "dice": 0.5}

        # Initialize individual losses
        from .classification import CrossEntropyLoss

        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, outputs, targets):
        """
        Compute combined loss.

        Args:
            outputs: Model predictions [batch_size, num_classes, H, W] (logits)
            targets: Ground truth masks [batch_size, H, W] (class indices)

        Returns:
            Weighted combination of CE and Dice losses
        """
        ce = self.ce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)

        combined = self.loss_weights["ce"] * ce + self.loss_weights["dice"] * dice

        return combined
