"""Base torchvision model loader with automatic final layer replacement.

This module provides flexible loading of any torchvision model architecture
with automatic detection and replacement of the final classification layer.
"""

import torch.nn as nn
from loguru import logger
from torchvision import models


def get_base_model(architecture, num_classes, weights=None, device="cpu"):
    """
    Load any torchvision model and adapt it for custom number of classes.

    Automatically detects the final layer structure and replaces it with
    a new linear layer matching the desired number of output classes.

    Args:
        architecture: Name of torchvision model (e.g., 'resnet18', 'vgg16', 'efficientnet_b0')
        num_classes: Number of output classes
        weights: Pretrained weights ('DEFAULT' for ImageNet weights, None for random init)
        device: Device to place the model on

    Returns:
        Model with adapted final layer, moved to specified device

    Supported architectures:
        - ResNet family: resnet18, resnet34, resnet50, resnet101, resnet152
        - ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
        - Wide ResNet: wide_resnet50_2, wide_resnet101_2
        - VGG: vgg11, vgg13, vgg16, vgg19 (with/without BN)
        - DenseNet: densenet121, densenet161, densenet169, densenet201
        - EfficientNet: efficientnet_b0 through efficientnet_b7
        - EfficientNetV2: efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
        - MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
        - SqueezeNet: squeezenet1_0, squeezenet1_1
        - AlexNet: alexnet
        - Vision Transformer: vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14
        - Swin Transformer: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b
        - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
        - RegNet: regnet_x_*, regnet_y_*
        - MNASNet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
        - ShuffleNet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
        - GoogLeNet/Inception: googlenet, inception_v3
        - MaxVit: maxvit_t

    Example:
        >>> model = get_base_model('resnet18', num_classes=10, weights='DEFAULT')
        >>> model = get_base_model('efficientnet_b0', num_classes=100, weights=None)
    """
    # Get model constructor function from torchvision.models
    try:
        model_fn = getattr(models, architecture)
    except AttributeError as err:
        raise ValueError(
            f"Architecture '{architecture}' not found in torchvision.models. "
            f"Available models: {[name for name in dir(models) if name.islower() and not name.startswith('_')]}"
        ) from err

    # Load model with optional pretrained weights
    if weights == "DEFAULT":
        logger.info(f"Loading {architecture} with pretrained ImageNet weights")
        model = model_fn(weights="DEFAULT")
    else:
        logger.info(f"Loading {architecture} with random initialization")
        model = model_fn(weights=None)

    # Replace final layer based on architecture family
    _replace_final_layer(model, architecture, num_classes)

    # Move to device
    model = model.to(device)

    logger.success(f"Created {architecture} model with {num_classes} output classes")
    return model


def _replace_final_layer(model, architecture, num_classes):
    """
    Replace the final classification layer of a model.

    Different model families use different attribute names for their final layer.
    This function detects the architecture family and replaces the appropriate layer.

    Args:
        model: PyTorch model instance
        architecture: Architecture name (string)
        num_classes: Number of output classes
    """
    arch_lower = architecture.lower()

    # ResNet family: resnet*, resnext*, wide_resnet*
    if "resnet" in arch_lower or "resnext" in arch_lower:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .fc layer: {num_ftrs} -> {num_classes}")

    # VGG and AlexNet: classifier[6] is the final layer
    elif "vgg" in arch_lower or "alexnet" in arch_lower:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[6] layer: {num_ftrs} -> {num_classes}")

    # DenseNet: single classifier Linear layer
    elif "densenet" in arch_lower:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier layer: {num_ftrs} -> {num_classes}")

    # EfficientNet (all versions): classifier is Sequential, last layer is [1]
    elif "efficientnet" in arch_lower:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[1] layer: {num_ftrs} -> {num_classes}")

    # MobileNetV2 and MobileNetV3: classifier is Sequential, last layer is [1]
    elif "mobilenet" in arch_lower:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[-1] layer: {num_ftrs} -> {num_classes}")

    # SqueezeNet: classifier[1] is Conv2d, not Linear
    elif "squeezenet" in arch_lower:
        model.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        logger.debug(f"Replaced .classifier[1] Conv2d layer: 512 -> {num_classes}")

    # MaxVit: classifier is Sequential, last layer is [5]
    # NOTE: Check BEFORE 'vit_' to avoid false match
    elif "maxvit" in arch_lower:
        num_ftrs = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[5] layer: {num_ftrs} -> {num_classes}")

    # Vision Transformer (ViT): heads.head is the final layer
    elif "vit_" in arch_lower:
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .heads.head layer: {num_ftrs} -> {num_classes}")

    # Swin Transformer: head is the final layer
    elif "swin" in arch_lower:
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .head layer: {num_ftrs} -> {num_classes}")

    # ConvNeXt: classifier is Sequential, last layer is [2]
    elif "convnext" in arch_lower:
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[2] layer: {num_ftrs} -> {num_classes}")

    # RegNet: fc is the final layer
    elif "regnet" in arch_lower:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .fc layer: {num_ftrs} -> {num_classes}")

    # MNASNet: classifier is Sequential, last layer is [1]
    elif "mnasnet" in arch_lower:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .classifier[1] layer: {num_ftrs} -> {num_classes}")

    # ShuffleNetV2: fc is the final layer
    elif "shufflenet" in arch_lower:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .fc layer: {num_ftrs} -> {num_classes}")

    # GoogLeNet/Inception: fc is the final layer (Inception v3 has aux_logits too)
    elif "googlenet" in arch_lower or "inception" in arch_lower:
        # Replace main output
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        logger.debug(f"Replaced .fc layer: {num_ftrs} -> {num_classes}")

        # Inception v3 has auxiliary classifier
        if hasattr(model, "AuxLogits"):
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
            logger.debug(
                f"Replaced .AuxLogits.fc layer: {num_ftrs_aux} -> {num_classes}"
            )

    else:
        # No fallback - require explicit support for all architectures
        supported_families = [
            "resnet/resnext/wide_resnet",
            "vgg",
            "alexnet",
            "densenet",
            "efficientnet",
            "mobilenet",
            "squeezenet",
            "maxvit",
            "vit",
            "swin",
            "convnext",
            "regnet",
            "mnasnet",
            "shufflenet",
            "googlenet",
            "inception",
        ]
        raise ValueError(
            f"Architecture '{architecture}' is not explicitly supported. "
            f"Supported architecture families: {', '.join(supported_families)}. "
            f"To add support, update the _replace_final_layer() function in network/base.py "
            f"with the correct final layer attribute for this architecture."
        )
