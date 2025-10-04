# Architecture Tests

Validates: **"All torchvision models are automatically supported with proper final layer adaptation."**

## Coverage

**201 tests** across **70+ torchvision architectures**: ResNet, VGG, EfficientNet, MobileNet, ViT, Swin, ConvNeXt, DenseNet, RegNet, MNASNet, ShuffleNet, and others.

## What's Tested

✅ Model instantiation via `get_model()`
✅ Final layer replacement for custom `num_classes` (2, 10, 100, 1000)
✅ Forward pass with correct output shape `[batch_size, num_classes]`
✅ Pretrained weights loading (`weights='DEFAULT'`)
✅ Various batch sizes (1, 2, 4, 8)
✅ Error handling for invalid configs

## Running Tests

```bash
# All tests (~5-10 min)
pytest tests/test_architectures.py -v

# Quick smoke test (~1-2 min)
pytest tests/test_architectures.py::TestArchitectureFamilies -v

# Specific architectures
pytest tests/test_architectures.py -k "resnet18 or efficientnet_b0" -v

# Specific test type
pytest tests/test_architectures.py::TestFinalLayerReplacement -v
```

CPU-compatible, no GPU required.
