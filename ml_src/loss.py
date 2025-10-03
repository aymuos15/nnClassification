"""Loss function module."""

import torch.nn as nn


def get_criterion(config=None):
    """
    Get the loss criterion based on configuration.

    Args:
        config: Configuration dictionary with loss settings (optional)

    Returns:
        Loss criterion (default: CrossEntropyLoss)
    """
    # Currently returns CrossEntropyLoss
    # In the future, can be extended to support multiple loss functions:
    # - Focal Loss
    # - Label Smoothing CrossEntropy
    # - Custom losses
    #
    # Example future usage:
    # loss_type = config.get('loss', {}).get('type', 'cross_entropy')
    # if loss_type == 'focal':
    #     return FocalLoss()
    # elif loss_type == 'label_smoothing':
    #     return LabelSmoothingCrossEntropy()

    return nn.CrossEntropyLoss()
