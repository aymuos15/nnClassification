# Model Configuration

## Overview

The `model` section configures model architecture and parameters. The framework supports both torchvision pretrained models and custom architectures.

## Configuration Parameters

```yaml
model:
  type: <string>                    # 'base' or 'custom'
  architecture: <string>            # For base models (torchvision)
  custom_architecture: <string>     # For custom models
  num_classes: <int>                # Number of output classes
  weights: <string or null>         # Pretrained weights ('DEFAULT' or null)
  input_size: <int>                 # Input image size (custom models)
  dropout: <float>                  # Dropout probability (custom models)
  model_path: <string>              # Legacy parameter (deprecated)
```

---

## `model.type`

- **Type:** String
- **Default:** `'base'`
- **Description:** Model type - 'base' for torchvision models, 'custom' for custom architectures
- **Purpose:** Select between pretrained torchvision models and custom implementations

### Usage

```yaml
model:
  type: 'base'    # Use torchvision models
  # OR
  type: 'custom'  # Use custom architectures
```

### When to Use

**`'base'` - Torchvision Models:**
- ✅ Need pretrained weights (transfer learning)
- ✅ Production-quality architectures
- ✅ State-of-the-art performance
- ✅ Standard computer vision tasks
- ✅ Most use cases

**`'custom'` - Custom Architectures:**
- ✅ Learning/educational purposes
- ✅ Prototyping new ideas
- ✅ Resource-constrained environments
- ✅ Specific architectural requirements
- ✅ Research experiments

---

## `model.architecture`

- **Type:** String
- **Default:** `'resnet18'`
- **Description:** Name of torchvision model architecture (used when `type: 'base'`)
- **Purpose:** Specify which pretrained architecture to load

### Usage

```yaml
model:
  type: 'base'
  architecture: 'resnet18'  # Any torchvision model
```

### Supported Architectures

**All torchvision models are automatically supported with proper final layer adaptation.**

#### ResNet Family
```yaml
architecture: 'resnet18'     # 11M params, good baseline
architecture: 'resnet34'     # 21M params
architecture: 'resnet50'     # 26M params, standard choice
architecture: 'resnet101'    # 45M params
architecture: 'resnet152'    # 60M params

# ResNeXt variants (improved ResNet)
architecture: 'resnext50_32x4d'
architecture: 'resnext101_32x8d'

# Wide ResNet (wider layers)
architecture: 'wide_resnet50_2'
architecture: 'wide_resnet101_2'
```

#### VGG (Deep but simple)
```yaml
architecture: 'vgg11'        # 133M params
architecture: 'vgg13'        # 133M params
architecture: 'vgg16'        # 138M params, classic
architecture: 'vgg19'        # 144M params

# With Batch Normalization (recommended)
architecture: 'vgg11_bn'
architecture: 'vgg13_bn'
architecture: 'vgg16_bn'
architecture: 'vgg19_bn'
```

#### EfficientNet (Efficient & accurate)
```yaml
# EfficientNet V1
architecture: 'efficientnet_b0'  # 5M params, fast
architecture: 'efficientnet_b1'  # 8M params
architecture: 'efficientnet_b2'  # 9M params
architecture: 'efficientnet_b3'  # 12M params
architecture: 'efficientnet_b4'  # 19M params
architecture: 'efficientnet_b5'  # 30M params
architecture: 'efficientnet_b6'  # 43M params
architecture: 'efficientnet_b7'  # 66M params, best accuracy

# EfficientNet V2 (faster training)
architecture: 'efficientnet_v2_s'  # Small
architecture: 'efficientnet_v2_m'  # Medium
architecture: 'efficientnet_v2_l'  # Large
```

#### MobileNet (Mobile & edge devices)
```yaml
architecture: 'mobilenet_v2'       # 3.5M params
architecture: 'mobilenet_v3_small' # 2.5M params
architecture: 'mobilenet_v3_large' # 5.5M params
```

#### Vision Transformers (Attention-based)
```yaml
architecture: 'vit_b_16'   # Base, 16x16 patches, 86M params
architecture: 'vit_b_32'   # Base, 32x32 patches, 88M params
architecture: 'vit_l_16'   # Large, 16x16 patches, 304M params
architecture: 'vit_l_32'   # Large, 32x32 patches
architecture: 'vit_h_14'   # Huge, 14x14 patches, 632M params
```

#### Swin Transformers (Hierarchical transformers)
```yaml
# Swin V1
architecture: 'swin_t'     # Tiny, 28M params
architecture: 'swin_s'     # Small, 50M params
architecture: 'swin_b'     # Base, 88M params

# Swin V2 (improved)
architecture: 'swin_v2_t'
architecture: 'swin_v2_s'
architecture: 'swin_v2_b'
```

#### ConvNeXt (Modern ConvNets)
```yaml
architecture: 'convnext_tiny'   # 29M params
architecture: 'convnext_small'  # 50M params
architecture: 'convnext_base'   # 89M params
architecture: 'convnext_large'  # 198M params
```

#### DenseNet (Dense connections)
```yaml
architecture: 'densenet121'  # 8M params
architecture: 'densenet161'  # 29M params
architecture: 'densenet169'  # 14M params
architecture: 'densenet201'  # 20M params
```

#### Other Architectures
```yaml
architecture: 'alexnet'         # Classic, 61M params
architecture: 'squeezenet1_0'   # Very small, 1.2M params
architecture: 'squeezenet1_1'   # Smaller variant
architecture: 'googlenet'       # Inception v1
architecture: 'inception_v3'    # Requires 299x299 input
architecture: 'maxvit_t'        # MaxViT Tiny

# RegNet (diverse family)
architecture: 'regnet_x_400mf'
architecture: 'regnet_y_400mf'
# ... many more regnet variants

# MNASNet (mobile)
architecture: 'mnasnet0_5'
architecture: 'mnasnet1_0'

# ShuffleNet (efficient)
architecture: 'shufflenet_v2_x0_5'
architecture: 'shufflenet_v2_x1_0'
```

### Choosing an Architecture

| Need | Recommended Architecture |
|------|-------------------------|
| **Fast training** | `efficientnet_b0`, `resnet18`, `mobilenet_v3_small` |
| **Best accuracy** | `efficientnet_b7`, `vit_l_16`, `convnext_large` |
| **Small model size** | `mobilenet_v3_small`, `squeezenet1_0` |
| **Balanced** | `resnet50`, `efficientnet_b3` |
| **Production standard** | `resnet50`, `efficientnet_b0` |
| **Research/SOTA** | `swin_b`, `vit_b_16`, `convnext_base` |

### Automatic Final Layer Replacement

The framework automatically detects and replaces the final classification layer for any architecture, adapting it to your `num_classes`. No manual configuration needed!

---

## `model.custom_architecture`

- **Type:** String or null
- **Default:** `null`
- **Description:** Name of custom model architecture (used when `type: 'custom'`)
- **Purpose:** Specify which custom model to use

### Usage

```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'  # or 'tiny_net'
```

### Available Custom Models

#### SimpleCNN
**Description:** 3-layer CNN with fully connected layers

**Specifications:**
- Channels: 3 → 32 → 64 → 128
- FC layers with dropout
- Configurable dropout rate
- Input size: Configurable (default 224x224)

**Use Cases:**
- Small to medium datasets (1k-10k images)
- Educational purposes
- Quick prototyping
- Baseline model

**Configuration:**
```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'
  num_classes: 10
  input_size: 224
  dropout: 0.5
```

#### TinyNet
**Description:** Minimal 2-layer CNN for fast experimentation

**Specifications:**
- Very small capacity
- Fast training
- Limited performance
- Good for testing pipelines

**Use Cases:**
- Pipeline testing
- Quick iterations
- Very small datasets
- Proof of concept

**Configuration:**
```yaml
model:
  type: 'custom'
  custom_architecture: 'tiny_net'
  num_classes: 2
  input_size: 224
```

### Creating Your Own Custom Model

1. **Define model class in `ml_src/network/custom.py`:**
```python
class MyCustomModel(nn.Module):
    def __init__(self, num_classes, input_size=224, **kwargs):
        super().__init__()
        # Your architecture here
        pass
    
    def forward(self, x):
        # Forward pass
        return x
```

2. **Add to MODEL_REGISTRY:**
```python
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'tiny_net': TinyNet,
    'my_custom_model': MyCustomModel,  # Add here
}
```

3. **Use in config:**
```yaml
model:
  type: 'custom'
  custom_architecture: 'my_custom_model'
  num_classes: 10
```

**See:** [Adding Custom Models Guide](../development/adding-models.md)

---

## `model.num_classes`

- **Type:** Integer (> 0)
- **Default:** `2`
- **Description:** Number of output classes for classification
- **Purpose:** Configures final layer size

### Usage

```yaml
model:
  num_classes: 2  # Binary classification (ants vs bees)
```

### Important Notes

**Must Match Dataset:**
- Must equal the number of class folders in your dataset
- All splits (train/val/test) must have same number of classes

**Common Values:**
- Binary classification: `num_classes: 2`
- CIFAR-10: `num_classes: 10`
- CIFAR-100: `num_classes: 100`
- ImageNet: `num_classes: 1000`

### Changing num_classes

**Consequences:**
1. Existing checkpoints become incompatible (different final layer size)
2. Must train from scratch or fine-tune from earlier layer
3. Must reorganize dataset to have correct number of classes

**Procedure:**
```bash
# 1. Update configuration
# In ml_src/config.yaml:
model:
  num_classes: 10  # Changed from 2

# 2. Verify dataset has 10 class folders
ls data/my_dataset/train/  # Should show 10 directories

# 3. Train from scratch (old checkpoints won't work)
ml-train
```

---

## `model.weights`

- **Type:** String or null
- **Default:** `null`
- **Description:** Pretrained weights for base models
- **Purpose:** Use transfer learning from ImageNet-pretrained models

### Usage

```yaml
model:
  type: 'base'
  architecture: 'resnet18'
  weights: 'DEFAULT'  # Use ImageNet weights
  # OR
  weights: null       # Random initialization
```

### Options

**`'DEFAULT'` - Pretrained Weights:**
- Loads ImageNet-pretrained weights
- Transfer learning approach
- Faster convergence
- Better performance on small datasets

**`null` - Random Initialization:**
- Train from scratch
- No pretrained weights
- Requires more data and time
- Full control over learning

### When to Use Pretrained Weights (`'DEFAULT'`)

✅ **Small Datasets (< 10k images)**
- Pretrained features help generalization
- Prevents overfitting

✅ **Similar to ImageNet Domain**
- Natural images
- Objects, animals, scenes
- RGB photographs

✅ **Fast Convergence**
- Need results quickly
- Limited training time

✅ **Limited Compute**
- Can't afford long training

### When to Train from Scratch (`null`)

✅ **Very Large Datasets (> 100k images)**
- Enough data to learn from scratch
- May outperform pretrained

✅ **Very Different Domain**
- Medical images (X-rays, MRI)
- Satellite imagery
- Specialized domains

✅ **Plenty of Time/Compute**
- Can afford 200+ epochs
- Have powerful GPUs

✅ **Want Full Control**
- Research purposes
- Understanding from ground up

### Performance Comparison

| Approach | Small Dataset | Large Dataset | Training Time |
|----------|--------------|---------------|---------------|
| Pretrained (`'DEFAULT'`) | ✅ Better | ✅ Good | ⚡ Fast (20-50 epochs) |
| From Scratch (`null`) | ❌ Worse | ✅ May be better | 🐌 Slow (200+ epochs) |

### Examples

**Transfer Learning (Recommended):**
```yaml
model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 10
  weights: 'DEFAULT'  # Start with ImageNet weights

training:
  num_epochs: 25  # Shorter training needed
  
optimizer:
  lr: 0.001  # Lower LR for fine-tuning
```

**From Scratch:**
```yaml
model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 10
  weights: null  # Random initialization

training:
  num_epochs: 200  # Longer training needed

optimizer:
  lr: 0.01  # Higher LR for from-scratch training
```

---

## `model.input_size`

- **Type:** Integer
- **Default:** `224`
- **Description:** Input image size for custom models
- **Purpose:** Configure input dimensions for custom architectures

### Usage

```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'
  input_size: 224  # Must match transforms.*.resize
```

### Important

- **Only used for custom models** (base models have fixed sizes)
- **Must match transform resize:**
  ```yaml
  model:
    input_size: 224
  
  transforms:
    train:
      resize: [224, 224]  # Must match!
  ```

### Common Sizes

- 224×224 (standard)
- 128×128 (smaller, faster)
- 512×512 (larger, more detail)

---

## `model.dropout`

- **Type:** Float [0.0, 1.0]
- **Default:** `0.5`
- **Description:** Dropout probability for custom models
- **Purpose:** Regularization to prevent overfitting

### Usage

```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'
  dropout: 0.5  # 50% dropout
```

### Typical Values

| Dropout | Effect | Use Case |
|---------|--------|----------|
| `0.0` | No regularization | Very large datasets |
| `0.3` | Light regularization | Medium datasets |
| `0.5` | Standard regularization ✅ | General use |
| `0.7` | Strong regularization | Small datasets, overfitting |

### Note

Only applies to custom models that support dropout (like SimpleCNN). Not used for base models.

---

## `model.model_path`

- **Type:** String (path)
- **Default:** `'best_model.pth'`
- **Description:** **Legacy parameter (deprecated)**
- **Current Status:** Not used by current code

The checkpointing system now handles model paths automatically:
- Best model: `runs/{run_name}/weights/best.pt`
- Last checkpoint: `runs/{run_name}/weights/last.pt`

**This parameter can be ignored.**

---

## Complete Examples

### Example 1: Pretrained ResNet18 (Recommended Starting Point)

```yaml
model:
  type: 'base'
  architecture: 'resnet18'
  num_classes: 2
  weights: 'DEFAULT'  # Transfer learning
```

### Example 2: Large Model for High Accuracy

```yaml
model:
  type: 'base'
  architecture: 'efficientnet_b7'
  num_classes: 100
  weights: 'DEFAULT'
```

### Example 3: Mobile/Edge Deployment

```yaml
model:
  type: 'base'
  architecture: 'mobilenet_v3_small'
  num_classes: 10
  weights: 'DEFAULT'
```

### Example 4: Vision Transformer

```yaml
model:
  type: 'base'
  architecture: 'vit_b_16'
  num_classes: 50
  weights: 'DEFAULT'
```

### Example 5: Custom Model for Learning

```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'
  num_classes: 5
  input_size: 224
  dropout: 0.3
```

### Example 6: Training from Scratch

```yaml
model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 10
  weights: null  # No pretrained weights

training:
  num_epochs: 200  # Need more epochs

optimizer:
  lr: 0.01  # Higher LR
```

---

## Best Practices

1. **Start with pretrained ResNet18**
   - Good baseline for most tasks
   - Fast training
   - Solid performance

2. **Use transfer learning when possible**
   - `weights: 'DEFAULT'` for most cases
   - Especially for small datasets

3. **Match num_classes to your dataset**
   - Double-check class folder count
   - All splits must match

4. **Choose architecture based on requirements**
   - Speed: EfficientNet-B0, MobileNet
   - Accuracy: EfficientNet-B7, ViT
   - Balanced: ResNet50, EfficientNet-B3

5. **Consider deployment constraints**
   - Mobile: MobileNet, EfficientNet-B0
   - Server: Any large model
   - Edge: SqueezeNet, MobileNet

## Related Configuration

- [Training Configuration](training.md) - Training parameters
- [Transform Configuration](transforms.md) - Input size must match
- [Examples](examples.md) - Complete configurations
- [Adding Models Guide](../development/adding-models.md) - Create custom models

## Further Reading

- [torchvision.models Documentation](https://pytorch.org/vision/stable/models.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
