"""Network module for model architectures.

This module provides a unified interface for loading both torchvision models
and custom architectures. It maintains backward compatibility with the original
network.py API while offering expanded functionality.

Main API:
    - get_model(config, device): Load model based on configuration
    - save_model(model, path): Save model state dict
    - load_model(model, path, device): Load model state dict

Usage:
    >>> from ml_src.network import get_model, save_model, load_model
    >>> model = get_model(config, device)
    >>> save_model(model, 'model.pt')
    >>> model = load_model(model, 'model.pt', device)
"""

import torch
from loguru import logger

from .base import get_base_model
from .custom import get_custom_model


def get_model(config, device):
    """
    Create and configure a model based on the configuration.

    This function routes to the appropriate model loader based on config['model']['type']:
        - 'base': Load torchvision model (ResNet, VGG, EfficientNet, etc.)
        - 'custom': Load custom architecture (SimpleCNN, TinyNet, etc.)

    Args:
        config: Configuration dictionary with model settings
        device: Device to place the model on (cuda or cpu)

    Returns:
        Configured model moved to the specified device

    Config structure:
        For base models (torchvision):
            model:
                type: 'base'
                architecture: 'resnet18'  # any torchvision model
                num_classes: 10
                weights: 'DEFAULT'  # 'DEFAULT' for pretrained, null for random init

        For custom models:
            model:
                type: 'custom'
                custom_architecture: 'simple_cnn'  # or 'tiny_net'
                num_classes: 10
                input_size: 224  # optional, defaults to 224
                dropout: 0.5  # optional, model-specific

    Example:
        >>> # Load pretrained ResNet18
        >>> config = {
        ...     'model': {
        ...         'type': 'base',
        ...         'architecture': 'resnet18',
        ...         'num_classes': 10,
        ...         'weights': 'DEFAULT'
        ...     }
        ... }
        >>> model = get_model(config, 'cuda:0')

        >>> # Load custom SimpleCNN
        >>> config = {
        ...     'model': {
        ...         'type': 'custom',
        ...         'custom_architecture': 'simple_cnn',
        ...         'num_classes': 10,
        ...         'dropout': 0.3
        ...     }
        ... }
        >>> model = get_model(config, 'cpu')
    """
    model_config = config["model"]
    num_classes = model_config["num_classes"]
    model_type = model_config.get("type", "base")  # default to 'base' for backward compatibility

    if model_type == "base":
        # Load torchvision model
        architecture = model_config.get("architecture", "resnet18")
        weights = model_config.get("weights", None)

        logger.info(f"Loading base model: {architecture}")
        model = get_base_model(
            architecture=architecture,
            num_classes=num_classes,
            weights=weights,
            device=device,
        )

    elif model_type == "custom":
        # Load custom model
        custom_arch = model_config.get("custom_architecture")
        if not custom_arch:
            raise ValueError(
                "For custom models, 'custom_architecture' must be specified in config. "
                "Available: 'simple_cnn', 'tiny_net'"
            )

        # Extract optional parameters
        input_size = model_config.get("input_size", 224)
        dropout = model_config.get("dropout", 0.5)

        logger.info(f"Loading custom model: {custom_arch}")
        model = get_custom_model(
            model_name=custom_arch,
            num_classes=num_classes,
            input_size=input_size,
            device=device,
            dropout=dropout,
        )

    else:
        raise ValueError(f"Invalid model type '{model_type}'. Must be 'base' or 'custom'.")

    return model


def save_model(model, path):
    """
    Save model state dict to a file.

    This saves only the model weights (state_dict), not the full model.
    Use checkpointing.save_checkpoint() to save full training state.

    Args:
        model: The model to save
        path: Path to save the model

    Example:
        >>> save_model(model, 'runs/experiment/weights/model.pt')
    """
    torch.save(model.state_dict(), path)
    logger.success(f"Model saved to {path}")


def load_model(model, path, device, use_ema: bool = False):
    """
    Load model state dict from a file.

    This loads only the model weights. The model architecture must already
    be created before calling this function.

    Args:
        model: The model to load weights into (must match saved architecture)
        path: Path to the saved model
        device: Device to load the model on

    Returns:
        Model with loaded weights

    Example:
        >>> model = get_model(config, device)
        >>> model = load_model(model, 'runs/experiment/weights/best.pt', device)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle both full checkpoints and standalone state dicts
    if use_ema and isinstance(checkpoint, dict) and "ema_state" in checkpoint:
        ema_state = checkpoint["ema_state"].get("ema_model_state")
        if ema_state is None:
            logger.warning(
                "EMA state missing model weights in checkpoint {}. Falling back to standard weights.",
                path,
            )
            model_state = checkpoint.get("model_state_dict", checkpoint)
        else:
            model_state = ema_state
            logger.info("Loaded EMA weights from checkpoint {}", path)
    else:
        if use_ema:
            logger.warning(
                "EMA weights requested but not found in checkpoint {}. Using standard weights.",
                path,
            )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
        else:
            model_state = checkpoint

    model.load_state_dict(model_state)

    # Ensure model is on the correct device
    model = model.to(device)

    logger.info(f"Model loaded from {path}")
    return model


# Export main API
__all__ = ["get_model", "save_model", "load_model"]
