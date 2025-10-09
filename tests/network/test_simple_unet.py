"""Tests for SimpleUNet architecture."""

import pytest
import torch

from ml_src.core.network.custom import SimpleUNet


def test_simple_unet_forward():
    """Test SimpleUNet forward pass."""
    num_classes = 3
    model = SimpleUNet(num_classes=num_classes, in_channels=3, base_features=64)

    # Test with 256x256 input
    x = torch.randn(2, 3, 256, 256)
    output = model(x)

    assert output.shape == (2, num_classes, 256, 256), f"Expected shape (2, {num_classes}, 256, 256), got {output.shape}"


def test_simple_unet_different_input_sizes():
    """Test SimpleUNet with different input sizes."""
    num_classes = 2
    model = SimpleUNet(num_classes=num_classes)

    # Test multiple sizes (must be divisible by 16 due to 4 pooling layers)
    test_sizes = [128, 256, 512]

    for size in test_sizes:
        x = torch.randn(1, 3, size, size)
        output = model(x)
        assert output.shape == (1, num_classes, size, size), f"Failed for size {size}"


def test_simple_unet_different_num_classes():
    """Test SimpleUNet with different number of classes."""
    for num_classes in [2, 3, 5, 10]:
        model = SimpleUNet(num_classes=num_classes)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape[1] == num_classes, f"Expected {num_classes} output channels, got {output.shape[1]}"


def test_simple_unet_batch_sizes():
    """Test SimpleUNet with different batch sizes."""
    model = SimpleUNet(num_classes=3)

    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 3, 256, 256)
        output = model(x)
        assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"


def test_simple_unet_in_channels():
    """Test SimpleUNet with different input channels."""
    # Grayscale input
    model = SimpleUNet(num_classes=2, in_channels=1)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    assert output.shape == (1, 2, 256, 256)

    # 4-channel input
    model = SimpleUNet(num_classes=2, in_channels=4)
    x = torch.randn(1, 4, 256, 256)
    output = model(x)
    assert output.shape == (1, 2, 256, 256)


def test_simple_unet_base_features():
    """Test SimpleUNet with different base features."""
    for base_features in [32, 64, 128]:
        model = SimpleUNet(num_classes=2, base_features=base_features)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape == (1, 2, 256, 256)
