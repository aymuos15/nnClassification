"""Custom model architectures.

This module provides example custom architectures that users can modify
or use as templates for their own models.

Users can create their own architectures here and import them in __init__.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SimpleCNN(nn.Module):
    """
    Simple 3-layer Convolutional Neural Network.

    This is a basic CNN architecture suitable for small to medium-sized
    image classification tasks. Use this as a template for creating your
    own custom architectures.

    Architecture:
        - Conv1: 3 -> 32 channels, 3x3 kernel
        - Conv2: 32 -> 64 channels, 3x3 kernel
        - Conv3: 64 -> 128 channels, 3x3 kernel
        - FC1: flattened -> 256 features
        - FC2: 256 -> num_classes

    Each conv layer is followed by ReLU and MaxPool.
    Dropout is applied before final FC layer.

    Args:
        num_classes: Number of output classes
        input_size: Input image size (assumes square images)
        dropout: Dropout probability (default: 0.5)

    Example:
        >>> model = SimpleCNN(num_classes=10, input_size=224)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 10])
    """

    def __init__(self, num_classes, input_size=224, dropout=0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer (reused)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after conv+pool layers
        # Input: 224x224 -> after 3 pools: 28x28
        # Formula: input_size / (2^num_pools)
        feature_size = input_size // (2**3)  # 3 pooling layers
        flattened_size = 128 * feature_size * feature_size

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        logger.info(
            f"Created SimpleCNN: input_size={input_size}, "
            f"flattened_size={flattened_size}, num_classes={num_classes}"
        )

    def forward(self, x):
        """Forward pass through the network."""
        # Conv block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112

        # Conv block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56

        # Conv block 3: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28

        # Flatten for FC layers
        x = torch.flatten(x, 1)

        # FC block 1: Linear -> ReLU -> Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # FC block 2: Linear (output)
        x = self.fc2(x)

        return x


class TinyNet(nn.Module):
    """
    Tiny CNN for quick prototyping and testing.

    A minimal 2-layer CNN suitable for very small datasets or quick experiments.
    Fast to train but limited capacity.

    Architecture:
        - Conv1: 3 -> 16 channels, 5x5 kernel
        - Conv2: 16 -> 32 channels, 5x5 kernel
        - FC1: flattened -> num_classes

    Args:
        num_classes: Number of output classes
        input_size: Input image size (assumes square images)

    Example:
        >>> model = TinyNet(num_classes=2, input_size=224)
    """

    def __init__(self, num_classes, input_size=224, **kwargs):
        super().__init__()
        
        # Ignore extra kwargs (like dropout) for compatibility
        if kwargs:
            logger.debug(f"TinyNet ignoring extra arguments: {list(kwargs.keys())}")

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate flattened size: input_size / (2^2) for 2 pools
        feature_size = input_size // 4
        flattened_size = 32 * feature_size * feature_size

        self.fc = nn.Linear(flattened_size, num_classes)

        logger.info(f"Created TinyNet: input_size={input_size}, num_classes={num_classes}")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_custom_model(model_name, num_classes, input_size=224, device="cpu", **kwargs):
    """
    Factory function to create custom models.

    Add your own custom models to this function.

    Args:
        model_name: Name of custom model ('simple_cnn', 'tiny_net', or your own)
        num_classes: Number of output classes
        input_size: Input image size (default: 224)
        device: Device to place model on
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Custom model moved to specified device

    Example:
        >>> model = get_custom_model('simple_cnn', num_classes=10, dropout=0.3)
        >>> model = get_custom_model('tiny_net', num_classes=2)

    To add your own model:
        1. Define your model class above (inherit from nn.Module)
        2. Add it to the MODEL_REGISTRY dictionary below
        3. Use it via config: model.type = 'custom', model.custom_architecture = 'your_model_name'
    """
    # Registry of available custom models
    MODEL_REGISTRY = {
        "simple_cnn": SimpleCNN,
        "tiny_net": TinyNet,
        # Add your custom models here:
        # 'my_model': MyCustomModel,
    }

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Custom model '{model_name}' not found. "
            f"Available custom models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Create model
    model = model_class(num_classes=num_classes, input_size=input_size, **kwargs)

    # Move to device
    model = model.to(device)

    logger.success(f"Created custom model '{model_name}' with {num_classes} output classes")
    return model
