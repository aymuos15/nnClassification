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


class SimpleUNet(nn.Module):
    """
    Simple U-Net for semantic segmentation.

    Standard U-Net architecture with encoder-decoder structure and skip connections.
    Suitable for binary and multi-class segmentation tasks.

    Architecture:
        - Encoder: 4 levels (double conv + batch norm + ReLU + maxpool)
        - Bottleneck: double conv + batch norm + ReLU
        - Decoder: 4 levels (transpose conv + skip connection + double conv)
        - Output: 1x1 conv to num_classes channels

    Args:
        num_classes: Number of output classes (including background)
        in_channels: Number of input channels (default: 3 for RGB)
        base_features: Base number of features in first layer (default: 64)

    Input: [B, in_channels, H, W]
    Output: [B, num_classes, H, W] (logits, per-pixel classification)

    Example:
        >>> model = SimpleUNet(num_classes=3, in_channels=3, base_features=64)
        >>> x = torch.randn(2, 3, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([2, 3, 256, 256])
    """

    def __init__(self, num_classes, in_channels=3, base_features=64, **kwargs):
        super().__init__()

        # Ignore extra kwargs for compatibility
        if kwargs:
            logger.debug(f"SimpleUNet ignoring extra arguments: {list(kwargs.keys())}")

        # Encoder (downsampling path)
        self.enc1 = self._double_conv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._double_conv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._double_conv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = self._double_conv(base_features * 4, base_features * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._double_conv(base_features * 8, base_features * 16)

        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, stride=2)
        self.dec4 = self._double_conv(base_features * 16, base_features * 8)

        self.upconv3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, stride=2)
        self.dec3 = self._double_conv(base_features * 8, base_features * 4)

        self.upconv2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = self._double_conv(base_features * 4, base_features * 2)

        self.upconv1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = self._double_conv(base_features * 2, base_features)

        # Output layer
        self.out = nn.Conv2d(base_features, num_classes, kernel_size=1)

        logger.info(
            f"Created SimpleUNet: num_classes={num_classes}, "
            f"in_channels={in_channels}, base_features={base_features}"
        )

    def _double_conv(self, in_channels, out_channels):
        """
        Double convolution block: Conv-BN-ReLU-Conv-BN-ReLU.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels

        Returns:
            Sequential module with double conv + batch norm + ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            Output logits [B, num_classes, H, W]
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)  # [B, 64, H, W]
        enc2 = self.enc2(self.pool1(enc1))  # [B, 128, H/2, W/2]
        enc3 = self.enc3(self.pool2(enc2))  # [B, 256, H/4, W/4]
        enc4 = self.enc4(self.pool3(enc3))  # [B, 512, H/8, W/8]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))  # [B, 1024, H/16, W/16]

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)  # [B, 512, H/8, W/8]
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)  # [B, 256, H/4, W/4]
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)  # [B, 128, H/2, W/2]
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)  # [B, 64, H, W]
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)  # [B, num_classes, H, W]

        return out


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
        "simple_unet": SimpleUNet,
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
