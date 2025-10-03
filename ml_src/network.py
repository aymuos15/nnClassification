"""Network module for model architectures."""

import torch
import torch.nn as nn
from torchvision import models
from loguru import logger


def get_model(config, device):
    """
    Create and configure a model based on the configuration.

    Args:
        config: Configuration dictionary with model settings
        device: Device to place the model on (cuda or cpu)

    Returns:
        Configured model moved to the specified device
    """
    num_classes = config['model']['num_classes']

    # Create ResNet18 model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move model to device
    model = model.to(device)

    return model


def save_model(model, path):
    """
    Save model state dict to a file.

    Args:
        model: The model to save
        path: Path to save the model
    """
    torch.save(model.state_dict(), path)
    logger.success(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load model state dict from a file.

    Args:
        model: The model to load weights into
        path: Path to the saved model
        device: Device to load the model on

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    logger.info(f"Model loaded from {path}")
    return model
