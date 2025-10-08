# Configuration Examples

## Overview

This page provides complete, real-world configuration examples for common use cases. Copy and adapt these for your needs.

---

## Example 1: Quick Testing

**Use Case:** Verify pipeline works, quick iterations during development

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/hymenoptera_data'
  num_workers: 2  # Lower for quick testing

training:
  batch_size: 8
  num_epochs: 5   # Just 5 epochs
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 3    # Decay once during 5 epochs
  gamma: 0.1

model:
  type: 'base'
  architecture: 'resnet18'
  num_classes: 2
  weights: 'DEFAULT'

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Small batch size and few epochs for speed
- Pretrained weights for faster convergence
- Low num_workers to reduce overhead

**When to use:** Development, testing, debugging pipelines

---

## Example 2: Production Training

**Use Case:** Full-scale training for deployment

```yaml
seed: 42
deterministic: false

data:
  data_dir: '/mnt/datasets/my_data'
  num_workers: 8  # High parallelism for speed

training:
  batch_size: 64  # Large batch for efficient training
  num_epochs: 100
  device: 'cuda:0'

optimizer:
  lr: 0.01        # Higher LR for larger batch
  momentum: 0.9

scheduler:
  step_size: 30   # Decay 3 times during 100 epochs
  gamma: 0.1

model:
  type: 'base'
  architecture: 'efficientnet_b3'  # Good balance of speed/accuracy
  num_classes: 10
  weights: 'DEFAULT'  # Transfer learning

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Large batch size for GPU efficiency
- EfficientNet for SOTA performance
- Many epochs for full convergence
- High num_workers for fast data loading

**When to use:** Final production training, deployment models

---

## Example 3: Reproducible Research

**Use Case:** Academic paper, reproducible results required

```yaml
seed: 12345
deterministic: true  # Full bit-exact reproducibility

data:
  data_dir: 'data/research_dataset'
  num_workers: 4

training:
  batch_size: 16
  num_epochs: 50
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 15
  gamma: 0.1

model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 5
  weights: 'DEFAULT'

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- `deterministic: true` for exact reproducibility
- Fixed seed documented
- Standard ResNet50 architecture
- Moderate batch size (not hardware-specific)

**When to use:** Research papers, experiments requiring exact reproduction

**Note:** Share config file with paper. Readers can reproduce exact results.

---

## Example 4: CPU Training (No GPU)

**Use Case:** Training on laptop or server without GPU

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/hymenoptera_data'
  num_workers: 4  # CPU can handle data loading

training:
  batch_size: 8   # Smaller batch for CPU
  num_epochs: 10  # Fewer epochs (CPU is slow)
  device: 'cpu'   # Force CPU

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 5
  gamma: 0.1

model:
  type: 'base'
  architecture: 'mobilenet_v3_small'  # Lightweight model for CPU
  num_classes: 2
  weights: 'DEFAULT'

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Small batch size (CPU memory limited)
- Lightweight MobileNet architecture
- Fewer epochs (CPU training is slow)
- Pretrained weights still help

**When to use:** No GPU available, laptop development

---

## Example 5: High-Resolution Images

**Use Case:** Detailed images, small objects, fine-grained classification

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/high_res_dataset'
  num_workers: 6

training:
  batch_size: 16  # Smaller batch due to larger images
  num_epochs: 50
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 15
  gamma: 0.1

model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 20
  weights: 'DEFAULT'

transforms:
  train:
    resize: [512, 512]  # High resolution
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [512, 512]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [512, 512]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- 512×512 resolution for detail
- Smaller batch size (high res uses more memory)
- Standard architecture works well

**When to use:** Medical imaging, satellite imagery, fine-grained tasks

---

## Example 6: Training from Scratch

**Use Case:** Very large dataset, want to train without pretrained weights

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/large_dataset'
  num_workers: 8

training:
  batch_size: 128  # Large batch for from-scratch training
  num_epochs: 200  # Many more epochs needed
  device: 'cuda:0'

optimizer:
  lr: 0.1         # Much higher LR (10-100x typical)
  momentum: 0.9

scheduler:
  step_size: 60   # Decay 3 times during 200 epochs
  gamma: 0.1

model:
  type: 'base'
  architecture: 'resnet50'
  num_classes: 1000
  weights: null   # NO pretrained weights

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- `weights: null` - train from scratch
- Very high learning rate (0.1)
- Many epochs (200+)
- Large batch size

**When to use:** ImageNet-scale datasets, domain very different from ImageNet

---

## Example 7: Mobile/Edge Deployment

**Use Case:** Model will run on mobile or edge devices

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/mobile_dataset'
  num_workers: 4

training:
  batch_size: 32
  num_epochs: 50
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 15
  gamma: 0.1

model:
  type: 'base'
  architecture: 'mobilenet_v3_small'  # Optimized for mobile
  num_classes: 10
  weights: 'DEFAULT'

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- MobileNet architecture (small, fast)
- Standard training process
- Model size < 10MB

**When to use:** Mobile apps, edge devices, resource-constrained environments

**Deployment tip:** Export to TorchScript or ONNX for mobile runtimes

---

## Example 8: Vision Transformer

**Use Case:** State-of-the-art transformer architecture

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/my_dataset'
  num_workers: 8

training:
  batch_size: 32  # Transformers use more memory
  num_epochs: 100
  device: 'cuda:0'

optimizer:
  lr: 0.001       # Lower LR for transformers
  momentum: 0.9

scheduler:
  step_size: 30
  gamma: 0.1

model:
  type: 'base'
  architecture: 'vit_b_16'  # Vision Transformer Base
  num_classes: 100
  weights: 'DEFAULT'

transforms:
  train:
    resize: [224, 224]  # ViT expects 224x224
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Vision Transformer architecture
- Smaller batch size (transformers are memory-intensive)
- Pretrained weights essential (ViT needs lots of data)

**When to use:** State-of-the-art performance, research, abundant compute

---

## Example 9: Custom CNN Model

**Use Case:** Learning, educational purposes, simple architecture

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/small_dataset'
  num_workers: 2

training:
  batch_size: 16
  num_epochs: 30
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 10
  gamma: 0.1

model:
  type: 'custom'              # Custom model
  custom_architecture: 'simple_cnn'
  num_classes: 5
  input_size: 224
  dropout: 0.5                # Regularization

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Custom SimpleCNN architecture
- Dropout for regularization
- Good for learning PyTorch

**When to use:** Educational purposes, understanding CNNs, simple tasks

---

## Example 10: Debugging Configuration

**Use Case:** Debugging code, finding issues, quick iterations

```yaml
seed: 42
deterministic: true  # Consistent behavior for debugging

data:
  data_dir: 'data/hymenoptera_data'
  num_workers: 0  # Single-threaded (easier to debug)

training:
  batch_size: 2   # Very small (fast epochs)
  num_epochs: 2   # Just 2 epochs
  device: 'cpu'   # CPU (better error messages)

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 1
  gamma: 0.1

model:
  type: 'custom'
  custom_architecture: 'tiny_net'  # Smallest model
  num_classes: 2

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: false  # Disable for consistency
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key Features:**
- Deterministic for consistent behavior
- Single-threaded data loading
- Minimal model and data
- CPU for better error messages

**When to use:** Debugging crashes, testing modifications, development

---

## Comparison Table

| Example | Batch Size | Epochs | Model | Use Case |
|---------|------------|--------|-------|----------|
| Quick Testing | 8 | 5 | ResNet18 | Pipeline verification |
| Production | 64 | 100 | EfficientNet-B3 | Deployment |
| Research | 16 | 50 | ResNet50 | Academic papers |
| CPU Training | 8 | 10 | MobileNet-Small | No GPU |
| High-Res | 16 | 50 | ResNet50 | Fine-grained tasks |
| From Scratch | 128 | 200 | ResNet50 | Large datasets |
| Mobile | 32 | 50 | MobileNet-Small | Edge devices |
| Transformer | 32 | 100 | ViT-B | SOTA performance |
| Custom CNN | 16 | 30 | SimpleCNN | Learning |
| Debugging | 2 | 2 | TinyNet | Development |

---

## How to Use These Examples

### Option 1: Copy to Custom Config

```bash
# Create configs directory
mkdir -p configs

# Copy example to file
cat > configs/production.yaml << 'EOF'
# Paste Example 2 content here
EOF

# Use it
ml-train --config configs/production.yaml
```

### Option 2: Modify Base Config

```bash
# Edit ml_src/config_template.yaml directly
nano ml_src/config_template.yaml

# Paste example content

# Train
ml-train
```

### Option 3: Use CLI Overrides

```bash
# Start with base, override key parameters
ml-train --batch_size 64 --num_epochs 100 --lr 0.01
```

---

## Related Documentation

- [Configuration Overview](README.md) - How the config system works
- [All Parameters](../configuration/) - Detailed parameter documentation
- [CLI Overrides](cli-overrides.md) - Command-line overrides
- [Best Practices](../reference/best-practices.md) - Configuration tips

---

## Quick Start Recommendations

**First time using the framework?** Start here:

1. **Try Example 1 (Quick Testing)**
   - Verifies everything works
   - Fast feedback (5 epochs)

2. **Adapt Example 2 (Production)**
   - Change `data_dir` to your dataset
   - Adjust `num_classes`
   - Train for real

3. **Experiment with CLI overrides**
   ```bash
   ml-train --batch_size 32 --lr 0.01
   ```

4. **Create your own config**
   - Copy closest example
   - Modify for your needs
   - Save in `configs/` directory

---

## Tips for Configuration

1. **Start simple** - Use default config first
2. **Change one thing at a time** - Easier to understand effects
3. **Document changes** - Note why you changed defaults
4. **Version control configs** - Track configuration evolution
5. **Name configs meaningfully** - `production.yaml`, `debug.yaml`, etc.
6. **Test on small data first** - Verify config before full training
7. **Monitor TensorBoard** - Watch metrics during training
8. **Save successful configs** - Keep configs that worked well

---

## Common Adjustments

### Faster Training
- Increase `batch_size` (e.g., 32 → 64)
- Increase `num_workers` (e.g., 4 → 8)
- Reduce `num_epochs` initially
- Use smaller model (MobileNet, EfficientNet-B0)

### Better Accuracy
- More `num_epochs`
- Larger model (ResNet50 → EfficientNet-B7)
- Higher resolution (224 → 512)
- Pretrained weights (`weights: 'DEFAULT'`)

### Less Memory
- Reduce `batch_size`
- Reduce image `resize`
- Use smaller model
- Reduce `num_workers`

### Debugging Issues
- Set `deterministic: true`
- Set `num_workers: 0`
- Reduce `batch_size` to 2
- Use `device: 'cpu'`
- Use TinyNet model

---

**Have questions?** Check the [Troubleshooting Guide](../reference/troubleshooting.md) or [FAQ](../reference/faq.md).
