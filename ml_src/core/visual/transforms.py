"""Image transformation utilities for visualization."""

import torch
import torchvision
from PIL import Image


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image with mean and standard deviation.

    Args:
        tensor: Tensor image of size (C, H, W) or (B, C, H, W)
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch of images
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def add_colored_border(image_tensor, color, border_width=5):
    """
    Add colored border to image tensor.

    Args:
        image_tensor: Tensor image of size (C, H, W)
        color: RGB tuple for border color
        border_width: Width of border in pixels

    Returns:
        Image tensor with colored border
    """
    # Convert to PIL Image
    img = torchvision.transforms.ToPILImage()(image_tensor)

    # Create a new image with border
    new_width = img.width + 2 * border_width
    new_height = img.height + 2 * border_width
    bordered_img = Image.new("RGB", (new_width, new_height), color)
    bordered_img.paste(img, (border_width, border_width))

    # Convert back to tensor
    return torchvision.transforms.ToTensor()(bordered_img)
