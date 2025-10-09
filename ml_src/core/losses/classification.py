"""Classification loss functions."""

import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for classification tasks.

    This is a wrapper around PyTorch's CrossEntropyLoss to provide
    a consistent interface with other loss functions in the registry.

    Args:
        weight: Optional tensor of per-class weights
        ignore_index: Specifies a target value that is ignored (default: -100)
        label_smoothing: Amount of label smoothing (default: 0.0)

    Example:
        >>> criterion = CrossEntropyLoss()
        >>> outputs = torch.randn(32, 10)  # [batch_size, num_classes]
        >>> targets = torch.randint(0, 10, (32,))  # [batch_size]
        >>> loss = criterion(outputs, targets)
    """

    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    def forward(self, outputs, targets):
        """
        Compute cross-entropy loss.

        Args:
            outputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Scalar loss value
        """
        return self.loss_fn(outputs, targets)
