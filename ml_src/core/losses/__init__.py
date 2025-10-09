"""
Loss functions module with registry pattern.

This module provides a flexible loss function system using a registry pattern.
Losses are automatically registered and can be instantiated via configuration.

Available losses:
    Classification:
        - cross_entropy: Standard cross-entropy loss

    Segmentation:
        - dice: Dice loss for overlap-based optimization
        - focal: Focal loss for handling class imbalance
        - combined: Combined CE + Dice loss

Example:
    >>> # In config.yaml:
    >>> # loss:
    >>> #   type: 'dice'
    >>> #   params:
    >>> #     smooth: 1.0
    >>>
    >>> from ml_src.core.losses import get_criterion
    >>> criterion = get_criterion(config)
    >>> loss = criterion(outputs, targets)
"""

from loguru import logger

# Loss registry - populated as losses are imported
LOSS_REGISTRY = {}


def register_loss(name, loss_class):
    """
    Register a loss function in the registry.

    Args:
        name: Loss type name (used in config)
        loss_class: Loss class to register

    Example:
        >>> register_loss('cross_entropy', CrossEntropyLoss)
    """
    LOSS_REGISTRY[name] = loss_class


def get_criterion(config=None):
    """
    Factory function to create loss criterion based on configuration.

    Reads loss configuration from config['loss'] and instantiates
    the appropriate loss class with the specified parameters.

    Args:
        config: Configuration dictionary containing loss specifications.
                If None, defaults to CrossEntropyLoss.

    Returns:
        Instantiated loss criterion

    Config structure:
        loss:
            type: 'dice'           # Loss type (must be in LOSS_REGISTRY)
            params:                # Loss-specific parameters
                smooth: 1.0
                ignore_index: -100

    Example:
        >>> # Cross-entropy loss (default)
        >>> criterion = get_criterion()
        >>>
        >>> # Dice loss with custom parameters
        >>> config = {
        ...     'loss': {
        ...         'type': 'dice',
        ...         'params': {'smooth': 1.0}
        ...     }
        ... }
        >>> criterion = get_criterion(config)
        >>>
        >>> # Combined loss
        >>> config = {
        ...     'loss': {
        ...         'type': 'combined',
        ...         'params': {
        ...             'loss_weights': {'ce': 0.4, 'dice': 0.6}
        ...         }
        ...     }
        ... }
        >>> criterion = get_criterion(config)

    Raises:
        ValueError: If loss type is not found in registry
    """
    # Default to cross_entropy if no config provided
    if config is None:
        config = {}

    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "cross_entropy")

    if loss_type not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. " f"Available: {list(LOSS_REGISTRY.keys())}"
        )

    loss_class = LOSS_REGISTRY[loss_type]
    loss_params = loss_config.get("params", {})

    logger.debug(f"Creating loss: {loss_type} with params: {loss_params}")

    return loss_class(**loss_params)


# Import and register losses
try:
    from .classification import CrossEntropyLoss

    register_loss("cross_entropy", CrossEntropyLoss)
except ImportError as e:
    logger.warning(f"Failed to import classification losses: {e}")

try:
    from .segmentation import CombinedLoss, DiceLoss, FocalLoss

    register_loss("dice", DiceLoss)
    register_loss("focal", FocalLoss)
    register_loss("combined", CombinedLoss)
except ImportError as e:
    logger.warning(f"Failed to import segmentation losses: {e}")


__all__ = [
    "get_criterion",
    "LOSS_REGISTRY",
    "register_loss",
    "CrossEntropyLoss",
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
]
