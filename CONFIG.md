# Configuration Reference

This document provides comprehensive documentation for all configuration options in the PyTorch Image Classifier project.

## Table of Contents
1. [Overview](#overview)
2. [Configuration File Location](#configuration-file-location)
3. [Configuration Structure](#configuration-structure)
4. [All Configuration Options](#all-configuration-options)
5. [CLI Override System](#cli-override-system)
6. [Configuration Examples](#configuration-examples)
7. [Best Practices](#best-practices)
8. [Advanced Configuration](#advanced-configuration)

## Overview

The project uses a YAML-based configuration system with command-line override support. This design provides:

- **Version Control Friendly**: YAML configs can be committed and tracked
- **Hierarchical Organization**: Related settings grouped logically
- **Override Flexibility**: CLI arguments override YAML defaults
- **Experiment Tracking**: Each run saves its final configuration
- **Type Safety**: YAML supports numbers, strings, booleans, lists, nested dicts

### Configuration Loading Flow

```
1. Load base config from YAML file (default: ml_src/config.yaml)
2. Override with CLI arguments (if provided)
3. Create run directory based on overrides
4. Save final config to run directory (runs/{run_name}/config.yaml)
```

## Configuration File Location

**Default Location:** `ml_src/config.yaml`

**Override with CLI:**
```bash
python train.py --config path/to/custom_config.yaml
```

**Per-Run Saved Config:**
After training starts, the final configuration is saved to:
```
runs/{run_name}/config.yaml
```

This saved config can be inspected to understand exact settings used for that run.

## Configuration Structure

The configuration is organized into 7 main sections:

```yaml
# Reproducibility Settings
seed: <int>
deterministic: <bool>

# Data Configuration
data:
  ...

# Training Configuration
training:
  ...

# Optimizer Configuration
optimizer:
  ...

# Scheduler Configuration
scheduler:
  ...

# Model Configuration
model:
  ...

# Transform Configuration
transforms:
  ...
```

## All Configuration Options

### 1. Reproducibility Configuration

#### `seed`
- **Type:** Integer
- **Default:** `42`
- **Description:** Random seed for all random number generators (Python, NumPy, PyTorch, CUDA)
- **Purpose:** Ensures reproducibility of experiments
- **Usage:**
  ```yaml
  seed: 42
  ```
- **Notes:**
  - Seeds Python's `random`, NumPy's `np.random`, PyTorch's generators
  - Seeds all CUDA devices
  - DataLoader workers are also seeded
  - Combined with `deterministic`, enables exact reproduction

**When to Change:**
- Running multiple independent experiments: Use different seeds
- Debugging: Keep same seed for consistent behavior
- Ensemble methods: Use different seeds per model

#### `deterministic`
- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable fully deterministic operations at the cost of performance
- **Purpose:** Guarantee bit-exact reproducibility across runs
- **Usage:**
  ```yaml
  deterministic: false  # Fast, approximately reproducible
  deterministic: true   # Slower, fully reproducible
  ```
- **Technical Details:**
  - `false`: Uses cuDNN benchmark mode for fastest algorithms (non-deterministic)
  - `true`: Forces deterministic algorithms and disables benchmark mode

**Trade-offs:**
| Mode | Speed | Reproducibility |
|------|-------|-----------------|
| `false` | Fast (1.0x) | Approximate (same seed → similar results) |
| `true` | Slower (0.7-0.9x) | Exact (same seed → identical results) |

**When to Use `true`:**
- Debugging training issues
- Comparing optimization algorithms
- Publishing reproducible research
- Legal/compliance requirements

**When to Use `false` (default):**
- Production training
- Hyperparameter search
- General experimentation
- When speed matters more than exact reproducibility

---

### 2. Data Configuration

The `data` section controls dataset loading and preprocessing.

```yaml
data:
  data_dir: <string>
  num_workers: <int>
```

#### `data.data_dir`
- **Type:** String (path)
- **Default:** `'data/hymenoptera_data'`
- **Description:** Path to dataset directory

---

### ⚠️ **CRITICAL: MANDATORY DIRECTORY STRUCTURE** ⚠️

**THIS STRUCTURE IS NOT OPTIONAL. THE CODE WILL FAIL WITHOUT IT.**

Your dataset **MUST** follow this exact hierarchy:

```
data_dir/
├── train/              ← REQUIRED: Training split
│   ├── class1/        ← REQUIRED: One folder per class
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/        ← REQUIRED: Another class folder
│   │   ├── img1.jpg
│   │   └── ...
│   └── classN/        ← REQUIRED: N class folders (N = num_classes)
│
├── val/                ← REQUIRED: Validation split
│   ├── class1/        ← REQUIRED: Same class names as train/
│   ├── class2/
│   └── classN/
│
└── test/               ← REQUIRED: Test split
    ├── class1/        ← REQUIRED: Same class names as train/
    ├── class2/
    └── classN/
```

**REQUIREMENTS (ALL MANDATORY):**
1. ✅ **Three splits:** Must have `train/`, `val/`, and `test/` directories
2. ✅ **Same classes:** All three splits must have identical class folder names
3. ✅ **Class folders:** Each class must be in its own subdirectory
4. ✅ **Images in class folders:** Images go directly inside class folders (no subdirectories)
5. ✅ **Matching num_classes:** Number of class folders must equal `model.num_classes` in config

---

### ❌ **WHAT WILL FAIL:**

**WRONG: Missing splits**
```
data_dir/
├── train/
│   ├── class1/
│   └── class2/
└── val/              # Missing test/ → FAILS
```
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'data_dir/test'`

---

**WRONG: Images not in class folders**
```
data_dir/
├── train/
│   ├── img1.jpg      # Images directly in train/ → FAILS
│   ├── img2.jpg
│   └── img3.jpg
```
**Error:** `RuntimeError: Found 0 files in subfolders of: data_dir/train`

---

**WRONG: Mismatched class names**
```
data_dir/
├── train/
│   ├── cats/
│   └── dogs/
├── val/
│   ├── cat/          # Different name → FAILS
│   └── dog/          # Different name → FAILS
└── test/
    ├── cats/
    └── dogs/
```
**Error:** Class indices won't match between splits, metrics will be wrong

---

**WRONG: Nested subdirectories**
```
data_dir/
├── train/
│   └── class1/
│       └── subset/   # Extra level → Images won't be found
│           ├── img1.jpg
│           └── img2.jpg
```
**Error:** Images in nested folders won't be loaded

---

### ✅ **CORRECT EXAMPLE:**

```
data/hymenoptera_data/
├── train/
│   ├── ants/
│   │   ├── 0013035.jpg
│   │   ├── 1030023.jpg
│   │   └── ... (124 images)
│   └── bees/
│       ├── 1092977343_cb42b38d62.jpg
│       ├── 1093831624_fb5fbe2308.jpg
│       └── ... (121 images)
├── val/
│   ├── ants/
│   │   └── ... (70 images)
│   └── bees/
│       └── ... (83 images)
└── test/
    ├── ants/
    │   └── ... (50 images)
    └── bees/
        └── ... (50 images)
```

With config:
```yaml
data:
  data_dir: 'data/hymenoptera_data'

model:
  num_classes: 2  # Must match number of class folders (ants, bees)
```

---

### 📋 **HOW TO ORGANIZE YOUR DATA:**

If your data is currently unorganized, use this script to organize it:

```bash
#!/bin/bash
# organize_data.sh

# Create directory structure
mkdir -p data/my_dataset/{train,val,test}/{class1,class2,class3}

# Example: Move images to correct locations
# Adjust paths and class names for your dataset
mv /path/to/class1/train_images/* data/my_dataset/train/class1/
mv /path/to/class1/val_images/* data/my_dataset/val/class1/
mv /path/to/class1/test_images/* data/my_dataset/test/class1/

# Repeat for all classes...
```

Or use Python:
```python
import os
import shutil
from sklearn.model_selection import train_test_split

# Assuming you have images organized by class
source_dir = 'original_data/'
target_dir = 'data/my_dataset/'

classes = ['class1', 'class2', 'class3']

for class_name in classes:
    # Get all images for this class
    images = os.listdir(f'{source_dir}/{class_name}')

    # Split: 70% train, 15% val, 15% test
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Create directories
    os.makedirs(f'{target_dir}/train/{class_name}', exist_ok=True)
    os.makedirs(f'{target_dir}/val/{class_name}', exist_ok=True)
    os.makedirs(f'{target_dir}/test/{class_name}', exist_ok=True)

    # Copy files
    for img in train:
        shutil.copy(
            f'{source_dir}/{class_name}/{img}',
            f'{target_dir}/train/{class_name}/{img}'
        )
    for img in val:
        shutil.copy(
            f'{source_dir}/{class_name}/{img}',
            f'{target_dir}/val/{class_name}/{img}'
        )
    for img in test:
        shutil.copy(
            f'{source_dir}/{class_name}/{img}',
            f'{target_dir}/test/{class_name}/{img}'
        )
```

---

### 🔍 **VERIFY YOUR STRUCTURE:**

Before training, verify your structure is correct:

```bash
# Check structure
tree -L 2 data/my_dataset/

# Expected output:
# data/my_dataset/
# ├── train
# │   ├── class1
# │   ├── class2
# │   └── class3
# ├── val
# │   ├── class1
# │   ├── class2
# │   └── class3
# └── test
#     ├── class1
#     ├── class2
#     └── class3

# Count images per split
echo "Train images:"
find data/my_dataset/train -type f | wc -l
echo "Val images:"
find data/my_dataset/val -type f | wc -l
echo "Test images:"
find data/my_dataset/test -type f | wc -l

# Count images per class
for split in train val test; do
  echo "$split split:"
  for class in data/my_dataset/$split/*; do
    echo "  $(basename $class): $(find $class -type f | wc -l) images"
  done
done
```

---

### 📝 **USAGE:**

```yaml
data:
  data_dir: 'data/hymenoptera_data'
```

**CLI Override:**
```bash
python train.py --data_dir data/my_dataset
```

**Common Paths:**
- Local data: `data/my_dataset`
- Network storage: `/mnt/shared/datasets/my_dataset`
- Absolute paths: `/home/user/datasets/my_dataset`

**Supported Image Formats:**
- JPG/JPEG
- PNG
- BMP
- TIFF
- Any format supported by PIL/Pillow

---

**REMEMBER: The code uses PyTorch's `ImageFolder` class, which REQUIRES this exact structure. This is not a limitation of this codebase—it's how PyTorch works. If you don't follow this structure, the code will crash immediately.**

---

#### `data.num_workers`
- **Type:** Integer (≥ 0)
- **Default:** `4`
- **Description:** Number of subprocesses for data loading
- **Purpose:** Parallelize data loading to prevent GPU starvation
- **Usage:**
  ```yaml
  data:
    num_workers: 4
  ```
- **CLI Override:**
  ```bash
  python train.py --num_workers 8
  ```

**Performance Guidance:**

| Setting | Use Case | Notes |
|---------|----------|-------|
| `0` | Debugging, small datasets | Single-threaded, easier to debug |
| `2-4` | General use, consumer hardware | Good balance |
| `4-8` | High-performance training | For systems with many CPU cores |
| `8+` | Large-scale training | May see diminishing returns |

**Factors to Consider:**
- CPU core count (don't exceed available cores)
- RAM availability (each worker loads data in memory)
- Disk I/O (too many workers can bottleneck disk)
- Batch size (larger batches benefit more from parallel loading)

**Troubleshooting:**
- **Too high:** Out of memory, slower training, system unresponsive
- **Too low:** GPU underutilized (check with `nvidia-smi`)
- **Optimal:** GPU utilization near 100%, no slowdowns

---

### 3. Training Configuration

The `training` section controls core training parameters.

```yaml
training:
  batch_size: <int>
  num_epochs: <int>
  device: <string>
```

#### `training.batch_size`
- **Type:** Integer (> 0)
- **Default:** `4`
- **Description:** Number of samples per training batch
- **Purpose:** Controls memory usage, training speed, and gradient quality
- **Usage:**
  ```yaml
  training:
    batch_size: 4
  ```
- **CLI Override:**
  ```bash
  python train.py --batch_size 32
  ```

**Trade-offs:**

| Aspect | Small Batches (1-8) | Medium Batches (16-64) | Large Batches (128+) |
|--------|-------------------|----------------------|-------------------|
| Memory Usage | Low | Medium | High |
| Training Speed | Slow | Medium | Fast |
| Gradient Noise | High | Medium | Low |
| Generalization | Better | Balanced | May overfit |
| Optimal Learning Rate | Lower | Medium | Higher |

**Hardware Considerations:**
- **GPU Memory Limited:** Start with 4-8, increase until OOM
- **Fast GPU (A100, V100):** Use 64-128 to maximize utilization
- **Consumer GPU (GTX 1080):** Typically 16-32
- **CPU Training:** 4-16 depending on RAM

**Best Practices:**
- Powers of 2 often faster (8, 16, 32, 64)
- Larger batches require higher learning rates
- If changing batch size, adjust LR proportionally
- Use gradient accumulation if batch size too small

#### `training.num_epochs`
- **Type:** Integer (> 0)
- **Default:** `3`
- **Description:** Total number of training epochs
- **Purpose:** Controls total training time and model convergence
- **Usage:**
  ```yaml
  training:
    num_epochs: 25
  ```
- **CLI Override:**
  ```bash
  python train.py --num_epochs 50
  ```

**Typical Values:**
- **Quick testing:** 3-5 epochs
- **Small datasets:** 20-50 epochs
- **Medium datasets:** 50-100 epochs
- **Large datasets:** 100-200 epochs
- **From scratch training:** 200-300 epochs

**Stopping Criteria:**
- Monitor validation loss/accuracy curves
- Stop when validation performance plateaus
- Watch for overfitting (train acc ↑, val acc ↓)
- Consider early stopping (not currently implemented)

#### `training.device`
- **Type:** String
- **Default:** `'cuda:0'`
- **Description:** Device to run training on
- **Purpose:** Select computation hardware
- **Usage:**
  ```yaml
  training:
    device: 'cuda:0'  # First GPU
    device: 'cuda:1'  # Second GPU
    device: 'cpu'     # CPU only
  ```

**Valid Values:**
- `'cuda:0'`, `'cuda:1'`, etc. - Specific GPU
- `'cuda'` - Default CUDA device
- `'cpu'` - CPU only

**Auto-Fallback:**
The code automatically falls back to CPU if CUDA is unavailable:
```python
if device_str.startswith('cuda') and torch.cuda.is_available():
    device = torch.device(device_str)
else:
    device = torch.device('cpu')
```

**GPU Selection:**
```bash
# Use second GPU
python train.py --device cuda:1

# Or via environment variable
CUDA_VISIBLE_DEVICES=1 python train.py
```

---

### 4. Optimizer Configuration

The `optimizer` section configures SGD optimizer parameters.

```yaml
optimizer:
  lr: <float>
  momentum: <float>
```

#### `optimizer.lr`
- **Type:** Float (> 0)
- **Default:** `0.001`
- **Description:** Learning rate (step size for gradient descent)
- **Purpose:** Controls how much to update weights each iteration
- **Usage:**
  ```yaml
  optimizer:
    lr: 0.001
  ```
- **CLI Override:**
  ```bash
  python train.py --lr 0.01
  ```

**Typical Ranges:**

| Learning Rate | Use Case |
|--------------|----------|
| `1e-5 - 1e-4` | Fine-tuning pretrained models |
| `1e-4 - 1e-3` | Training with transfer learning (current default) |
| `1e-3 - 1e-2` | Training from scratch with small batches |
| `1e-2 - 1e-1` | Training from scratch with large batches |

**Finding the Right LR:**
1. Start with `0.001` (current default)
2. If loss decreases too slowly → increase LR (try `0.01`)
3. If loss diverges/NaN → decrease LR (try `0.0001`)
4. Use LR finder (not implemented, but recommended extension)

**Relationship with Batch Size:**
- Larger batches → Higher LR (rule of thumb: LR ∝ √batch_size)
- Example: batch 8 with LR 0.001 → batch 32 with LR 0.002

**Scheduler Interaction:**
- Initial LR set here
- StepLR scheduler will decay this value during training
- See `scheduler` section for decay details

#### `optimizer.momentum`
- **Type:** Float [0.0, 1.0]
- **Default:** `0.9`
- **Description:** SGD momentum factor
- **Purpose:** Accelerates optimization by accumulating velocity
- **Usage:**
  ```yaml
  optimizer:
    momentum: 0.9
  ```
- **CLI Override:**
  ```bash
  python train.py --momentum 0.95
  ```

**How Momentum Works:**
- `0.0`: No momentum (vanilla SGD)
- `0.5`: Low momentum (slow acceleration)
- `0.9`: Standard momentum (recommended)
- `0.99`: High momentum (may overshoot)

**Effects:**
- **Higher momentum:** Faster convergence, may oscillate
- **Lower momentum:** Slower convergence, more stable
- **Zero momentum:** Slowest convergence, most stable

**Best Practices:**
- `0.9` is the standard value (current default)
- Rarely needs tuning
- If training oscillates, try `0.8` or `0.85`
- If training too slow, try `0.95`

---

### 5. Scheduler Configuration

The `scheduler` section configures StepLR learning rate scheduler.

```yaml
scheduler:
  step_size: <int>
  gamma: <float>
```

**Scheduler Type:** `StepLR` (multiplicative decay every N epochs)

#### `scheduler.step_size`
- **Type:** Integer (> 0)
- **Default:** `7`
- **Description:** Number of epochs between LR decay steps
- **Purpose:** Controls frequency of learning rate reduction
- **Usage:**
  ```yaml
  scheduler:
    step_size: 7
  ```
- **CLI Override:**
  ```bash
  python train.py --step_size 10
  ```

**How It Works:**
- Every `step_size` epochs, LR is multiplied by `gamma`
- Example with `step_size: 7`, `gamma: 0.1`, initial `lr: 0.001`:
  - Epochs 0-6: LR = 0.001
  - Epochs 7-13: LR = 0.0001
  - Epochs 14-20: LR = 0.00001
  - Epochs 21+: LR = 0.000001

**Choosing `step_size`:**
- **Short training (< 20 epochs):** 5-7 epochs
- **Medium training (20-50 epochs):** 10-15 epochs
- **Long training (50+ epochs):** 15-30 epochs

**Rules of Thumb:**
- Decay 2-3 times during training
- Let model train with initial LR for some time before first decay
- If training 25 epochs, try `step_size: 8` (decays at epochs 8, 16)

#### `scheduler.gamma`
- **Type:** Float (0.0, 1.0]
- **Default:** `0.1`
- **Description:** Multiplicative factor for learning rate decay
- **Purpose:** Controls magnitude of LR reduction
- **Usage:**
  ```yaml
  scheduler:
    gamma: 0.1
  ```
- **CLI Override:**
  ```bash
  python train.py --gamma 0.5
  ```

**Common Values:**

| Gamma | Effect | Use Case |
|-------|--------|----------|
| `0.1` | Aggressive decay (90% reduction) | Default, works well |
| `0.2` | Strong decay (80% reduction) | Slower convergence |
| `0.5` | Moderate decay (50% reduction) | Fine-tuning |
| `0.9` | Gentle decay (10% reduction) | Gradual refinement |

**Choosing `gamma`:**
- `0.1` is the standard value (current default)
- More aggressive (`0.1`) → Faster convergence, may underfit
- Less aggressive (`0.5`) → Slower convergence, may overfit
- Very gentle (`0.9`) → Useful for fine-tuning already-trained models

**Combined Example:**
```yaml
scheduler:
  step_size: 7
  gamma: 0.1
```

Training 25 epochs:
```
Epochs 0-6:   LR = 0.001   (initial)
Epochs 7-13:  LR = 0.0001  (×0.1 at epoch 7)
Epochs 14-20: LR = 0.00001 (×0.1 at epoch 14)
Epochs 21-24: LR = 0.000001(×0.1 at epoch 21)
```

---

### 6. Model Configuration

The `model` section configures model architecture and parameters.

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

#### `model.type`
- **Type:** String
- **Default:** `'base'`
- **Description:** Model type - 'base' for torchvision models, 'custom' for custom architectures
- **Purpose:** Select between pretrained torchvision models and custom implementations
- **Usage:**
  ```yaml
  model:
    type: 'base'    # Use torchvision models
    # OR
    type: 'custom'  # Use custom architectures
  ```

**When to Use:**
- **'base'**: Use any torchvision model (ResNet, VGG, EfficientNet, ViT, etc.)
- **'custom'**: Use custom architectures (SimpleCNN, TinyNet, or your own)

#### `model.architecture`
- **Type:** String
- **Default:** `'resnet18'`
- **Description:** Name of torchvision model architecture (used when `type: 'base'`)
- **Purpose:** Specify which pretrained architecture to load
- **Usage:**
  ```yaml
  model:
    type: 'base'
    architecture: 'resnet18'  # Any torchvision model
  ```

**Supported Architectures:**

All torchvision models are automatically supported with proper final layer adaptation:

**ResNet Family:**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d`
- `wide_resnet50_2`, `wide_resnet101_2`

**VGG:**
- `vgg11`, `vgg13`, `vgg16`, `vgg19`
- `vgg11_bn`, `vgg13_bn`, `vgg16_bn`, `vgg19_bn` (with batch norm)

**DenseNet:**
- `densenet121`, `densenet161`, `densenet169`, `densenet201`

**EfficientNet:**
- `efficientnet_b0` through `efficientnet_b7`
- `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l`

**MobileNet:**
- `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`

**Vision Transformers:**
- `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`

**Swin Transformers:**
- `swin_t`, `swin_s`, `swin_b`
- `swin_v2_t`, `swin_v2_s`, `swin_v2_b`

**ConvNeXt:**
- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`

**Other:**
- `alexnet`, `squeezenet1_0`, `squeezenet1_1`
- `googlenet`, `inception_v3`
- `regnet_x_*`, `regnet_y_*` (various sizes)
- `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3`
- `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`
- `maxvit_t`

**Automatic Final Layer Replacement:**
The framework automatically detects and replaces the final classification layer for any architecture, adapting it to your `num_classes`.

#### `model.custom_architecture`
- **Type:** String or null
- **Default:** `null`
- **Description:** Name of custom model architecture (used when `type: 'custom'`)
- **Purpose:** Specify which custom model to use
- **Usage:**
  ```yaml
  model:
    type: 'custom'
    custom_architecture: 'simple_cnn'  # or 'tiny_net'
  ```

**Available Custom Models:**

**SimpleCNN:**
- 3-layer CNN with fully connected layers
- Good for small to medium datasets
- Configurable dropout
- Input: 224x224 (configurable)

**TinyNet:**
- Minimal 2-layer CNN
- Fast prototyping and testing
- Limited capacity
- Input: 224x224 (configurable)

**Creating Your Own:**
1. Define your model class in `ml_src/network/custom.py`
2. Add it to the `MODEL_REGISTRY` dictionary
3. Use via config: `custom_architecture: 'your_model_name'`

#### `model.num_classes`
- **Type:** Integer (> 0)
- **Default:** `2`
- **Description:** Number of output classes for classification
- **Purpose:** Configures final layer size
- **Usage:**
  ```yaml
  model:
    num_classes: 2  # Binary classification (ants vs bees)
  ```

**Important:**
- Must match number of subdirectories in train/val/test folders
- For binary classification: `num_classes: 2`
- For ImageNet: `num_classes: 1000`
- For CIFAR-10: `num_classes: 10`

**Changing `num_classes`:**
1. Update config: `num_classes: 10`
2. Ensure dataset has 10 class folders
3. Existing checkpoints won't work (different layer size)
4. Train from scratch

#### `model.weights`
- **Type:** String or null
- **Default:** `null`
- **Description:** Pretrained weights for base models
- **Purpose:** Use transfer learning from ImageNet-pretrained models
- **Usage:**
  ```yaml
  model:
    type: 'base'
    architecture: 'resnet18'
    weights: 'DEFAULT'  # Use ImageNet weights
    # OR
    weights: null       # Random initialization
  ```

**Options:**
- `'DEFAULT'`: Load ImageNet-pretrained weights (recommended for transfer learning)
- `null`: Random initialization (train from scratch)

**When to Use Pretrained Weights:**
- ✅ Small datasets (< 10k images)
- ✅ Similar to ImageNet domain (natural images, objects)
- ✅ Want faster convergence
- ✅ Limited training time/compute

**When to Train from Scratch:**
- ❌ Very large datasets (> 100k images)
- ❌ Very different domain (medical, satellite, etc.)
- ❌ Plenty of training time/compute
- ❌ Want full control over learning

**Note:** Only applies to `type: 'base'`. Custom models always use random initialization.

#### `model.input_size`
- **Type:** Integer
- **Default:** `224`
- **Description:** Input image size for custom models
- **Purpose:** Configure input dimensions for custom architectures
- **Usage:**
  ```yaml
  model:
    type: 'custom'
    custom_architecture: 'simple_cnn'
    input_size: 224  # Must match transforms.*.resize
  ```

**Important:**
- Must match `transforms.*.resize` configuration
- Only used for custom models
- Base models have fixed input sizes (usually 224x224)

#### `model.dropout`
- **Type:** Float [0.0, 1.0]
- **Default:** `0.5`
- **Description:** Dropout probability for custom models
- **Purpose:** Regularization to prevent overfitting
- **Usage:**
  ```yaml
  model:
    type: 'custom'
    custom_architecture: 'simple_cnn'
    dropout: 0.5  # 50% dropout
  ```

**Typical Values:**
- `0.3`: Light regularization
- `0.5`: Standard (default)
- `0.7`: Strong regularization

**Note:** Only applies to custom models that support dropout (SimpleCNN). Not used for base models.

#### `model.model_path`
- **Type:** String (path)
- **Default:** `'best_model.pth'`
- **Description:** Legacy parameter (deprecated)
- **Purpose:** Originally for model save path, now handled by checkpointing system
- **Current Behavior:**
  - Models saved to `runs/{run_name}/weights/best.pt` and `last.pt`
  - This parameter can be ignored

**Note:** This parameter is present for backward compatibility but not used by current code. Model paths are determined by the run directory structure.

---

### Complete Model Configuration Examples

**Example 1: ResNet18 from scratch**
```yaml
model:
  type: 'base'
  architecture: 'resnet18'
  num_classes: 2
  weights: null  # Train from scratch
```

**Example 2: Pretrained EfficientNet-B0**
```yaml
model:
  type: 'base'
  architecture: 'efficientnet_b0'
  num_classes: 10
  weights: 'DEFAULT'  # Transfer learning
```

**Example 3: Custom SimpleCNN**
```yaml
model:
  type: 'custom'
  custom_architecture: 'simple_cnn'
  num_classes: 5
  input_size: 224
  dropout: 0.3
```

**Example 4: Vision Transformer**
```yaml
model:
  type: 'base'
  architecture: 'vit_b_16'
  num_classes: 100
  weights: 'DEFAULT'
```

---

### 7. Transform Configuration

The `transforms` section configures data preprocessing and augmentation for each data split.

```yaml
transforms:
  train:
    ...
  val:
    ...
  test:
    ...
```

Each split (train/val/test) can have its own transform pipeline.

#### Transform Pipeline Structure

Each split supports these transforms:

```yaml
<split>:
  resize: [height, width]                 # Required
  random_horizontal_flip: <bool>          # Optional (train only)
  normalize:                              # Required
    mean: [R, G, B]
    std: [R, G, B]
```

**Transform Order:**
1. Resize
2. Random horizontal flip (if enabled)
3. Convert to tensor
4. Normalize

---

#### `transforms.<split>.resize`
- **Type:** List of 2 integers [height, width]
- **Default:** `[224, 224]`
- **Description:** Target image size for resizing
- **Purpose:** Standardize image dimensions for batching and model input
- **Usage:**
  ```yaml
  transforms:
    train:
      resize: [224, 224]
  ```

**Common Sizes:**

| Size | Use Case |
|------|----------|
| `[224, 224]` | ResNet, VGG standard (current) |
| `[299, 299]` | Inception, Xception |
| `[384, 384]` | ViT-Large |
| `[512, 512]` | High-resolution models |

**Considerations:**
- Larger sizes → More detail, slower training, more memory
- Smaller sizes → Faster training, less memory, less detail
- Non-square sizes supported: `[256, 224]`
- Must match model's expected input size

#### `transforms.train.random_horizontal_flip`
- **Type:** Boolean
- **Default:** `true`
- **Description:** Randomly flip images horizontally with 50% probability
- **Purpose:** Data augmentation to improve generalization
- **Usage:**
  ```yaml
  transforms:
    train:
      random_horizontal_flip: true  # Enable augmentation
    val:
      # Not applied to val/test splits
    test:
      # Not applied to val/test splits
  ```

**Best Practices:**
- **Enable** for: Most natural images, objects, animals, scenes
- **Disable** for: Text, oriented objects (cars, planes), asymmetric objects
- Only use in training split
- Val/test should match real-world data (no augmentation)

**Additional Augmentations (Not Currently Implemented):**
To add more augmentations, modify `ml_src/dataset.py::get_transforms()`:
- Random rotation
- Color jitter
- Random crop
- Random erasing
- Cutout
- MixUp
- AutoAugment

#### `transforms.<split>.normalize`
- **Type:** Dictionary with `mean` and `std` keys
- **Default:** ImageNet statistics
  ```yaml
  mean: [0.485, 0.456, 0.406]  # ImageNet mean (RGB)
  std: [0.229, 0.224, 0.225]   # ImageNet std (RGB)
  ```
- **Description:** Normalize images to zero mean and unit variance per channel
- **Purpose:** Stabilize training and match pretrained model expectations
- **Usage:**
  ```yaml
  transforms:
    train:
      normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
  ```

**Understanding Normalization:**
```python
# For each pixel and channel:
normalized_value = (original_value - mean) / std

# Example:
# Red channel pixel = 0.8
# mean[0] = 0.485, std[0] = 0.229
# normalized = (0.8 - 0.485) / 0.229 = 1.376
```

**When to Use ImageNet Stats:**
- Training models from scratch (current setup)
- Fine-tuning pretrained models
- General-purpose image classification
- When in doubt

**When to Compute Custom Stats:**
- Specialized domains (medical, satellite, etc.)
- Non-RGB images (grayscale, multispectral)
- Significantly different data distribution

**Computing Custom Stats:**
```python
# Compute mean and std from your training data
# (Not included in current codebase, but can be added)
from torchvision import datasets, transforms
import torch

dataset = datasets.ImageFolder(
    'data/your_dataset/train',
    transform=transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
)

loader = torch.utils.data.DataLoader(dataset, batch_size=32)

mean = 0.
std = 0.
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(loader.dataset)
std /= len(loader.dataset)

print(f'mean: {mean.tolist()}')
print(f'std: {std.tolist()}')
```

---

#### Complete Transform Examples

**Training Split (with augmentation):**
```yaml
train:
  resize: [224, 224]
  random_horizontal_flip: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

**Validation/Test Splits (no augmentation):**
```yaml
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

**High-Resolution Configuration:**
```yaml
train:
  resize: [512, 512]
  random_horizontal_flip: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

---

## CLI Override System

Command-line arguments override YAML configuration, enabling quick experimentation without editing config files.

### Override Mapping

| CLI Argument | Config Path | Example |
|-------------|-------------|---------|
| `--config` | (Specifies which YAML to load) | `--config custom.yaml` |
| `--data_dir` | `data.data_dir` | `--data_dir data/my_dataset` |
| `--batch_size` | `training.batch_size` | `--batch_size 32` |
| `--num_workers` | `data.num_workers` | `--num_workers 8` |
| `--num_epochs` | `training.num_epochs` | `--num_epochs 50` |
| `--lr` | `optimizer.lr` | `--lr 0.01` |
| `--momentum` | `optimizer.momentum` | `--momentum 0.95` |
| `--step_size` | `scheduler.step_size` | `--step_size 10` |
| `--gamma` | `scheduler.gamma` | `--gamma 0.5` |

### Run Directory Naming

Run directories are automatically named based on overrides:

```bash
# No overrides → runs/base/
python train.py

# Single override → runs/batch_32/
python train.py --batch_size 32

# Multiple overrides → runs/batch_32_epochs_50_lr_0.01/
python train.py --batch_size 32 --num_epochs 50 --lr 0.01
```

**Override Tracking:**
Only parameters that affect training results are included in run name:
- `batch_size`
- `num_epochs`
- `lr` (learning rate)

Parameters not affecting results (e.g., `num_workers`) are excluded from run name.

---

## Configuration Examples

### Example 1: Quick Testing
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
  num_classes: 2

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

### Example 2: Production Training
```yaml
seed: 42
deterministic: false

data:
  data_dir: '/mnt/datasets/my_data'
  num_workers: 8  # High parallelism

training:
  batch_size: 64  # Large batch for fast training
  num_epochs: 100
  device: 'cuda:0'

optimizer:
  lr: 0.01        # Higher LR for larger batch
  momentum: 0.9

scheduler:
  step_size: 30   # Decay 3 times during 100 epochs
  gamma: 0.1

model:
  num_classes: 10  # 10-class problem

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

### Example 3: Reproducible Research
```yaml
seed: 12345
deterministic: true  # Full reproducibility

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
  num_classes: 5

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

### Example 4: CPU Training (No GPU)
```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/hymenoptera_data'
  num_workers: 4

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
  num_classes: 2

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

### Example 5: High-Resolution Images
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
  num_classes: 20

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

---

## Best Practices

### 1. Start with Defaults
- Use the provided `ml_src/config.yaml` as a starting point
- Modify incrementally based on results
- Document why you changed defaults

### 2. Version Control Your Configs
```bash
# Create experiment configs
configs/
├── baseline.yaml
├── high_lr.yaml
├── large_batch.yaml
└── augmented.yaml

# Train with specific config
python train.py --config configs/high_lr.yaml
```

### 3. Systematic Hyperparameter Search
```bash
# Grid search example
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    python train.py --lr $lr --batch_size $bs
  done
done
```

### 4. Use Descriptive Run Names
The automatic naming helps, but for complex experiments:
```bash
# Manual run directory creation not supported
# But CLI overrides create meaningful names automatically
python train.py --lr 0.01 --batch_size 64 --num_epochs 100
# Creates: runs/batch_64_epochs_100_lr_0.01/
```

### 5. Always Check Saved Config
After starting training, verify:
```bash
cat runs/{run_name}/config.yaml
```
This shows exactly what configuration was used.

### 6. Document Experiments
Create a simple log:
```bash
# experiments.md
## Experiment 1: Baseline
- Config: base
- Result: 85% val acc
- Notes: Good starting point

## Experiment 2: Higher LR
- Config: batch_32_lr_0.01
- Result: 88% val acc
- Notes: Converges faster, better final performance
```

### 7. Reproducibility Checklist
For reproducible results:
- ✅ Set `seed` to a fixed value
- ✅ Set `deterministic: true` (if needed)
- ✅ Use same PyTorch/CUDA versions
- ✅ Same hardware (GPU model affects determinism)
- ✅ Save and share config file
- ✅ Document environment (`pip freeze > requirements.txt`)

### 8. Performance Optimization
For fastest training:
- ✅ Set `deterministic: false` (default)
- ✅ Increase `num_workers` (4-8)
- ✅ Maximize `batch_size` (until OOM)
- ✅ Use `batch_size` as power of 2
- ✅ Enable cuDNN benchmark (automatic with `deterministic: false`)

### 9. Debugging Configuration
When debugging:
```yaml
# Minimal config for fast iteration
data:
  num_workers: 0      # Single-threaded for easier debugging

training:
  batch_size: 2       # Small for quick epochs
  num_epochs: 2       # Just a couple epochs

deterministic: true   # Consistent behavior
```

### 10. Memory Management
If encountering OOM:
1. Reduce `batch_size` (halve it)
2. Reduce `resize` dimensions (224→192→160)
3. Reduce `num_workers` (frees RAM)
4. Close other applications
5. Check GPU utilization: `nvidia-smi`

---

## Advanced Configuration

### Multi-GPU Training (Not Currently Supported)
To add multi-GPU support, modify `ml_src/model.py`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

Then use:
```yaml
training:
  device: 'cuda'  # Uses all available GPUs
```

### Mixed Precision Training (Not Currently Supported)
To add AMP support for faster training:
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Add to config:
```yaml
training:
  mixed_precision: true
```

### Custom Learning Rate Schedules
Current: StepLR

To add other schedulers, modify `ml_src/model.py::get_scheduler()`:
- `CosineAnnealingLR`: Smooth cosine decay
- `ReduceLROnPlateau`: Decay when val loss plateaus
- `OneCycleLR`: Cyclical learning rates
- `ExponentialLR`: Exponential decay

### Early Stopping (Not Currently Implemented)
To add early stopping:
```yaml
training:
  early_stopping:
    patience: 10      # Stop after 10 epochs without improvement
    min_delta: 0.001  # Minimum change to qualify as improvement
```

### Data Augmentation Extensions
Current: Only horizontal flip

To add more, modify `ml_src/dataset.py::get_transforms()`:
```yaml
transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    random_rotation: 15        # New
    color_jitter:              # New
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
    random_erasing:            # New
      p: 0.5
      scale: [0.02, 0.33]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

---

## Summary

This configuration system provides:
- ✅ Clear, hierarchical organization
- ✅ CLI override flexibility
- ✅ Automatic experiment tracking
- ✅ Reproducibility support
- ✅ Easy to extend

**Key Takeaways:**
1. Start with defaults in `ml_src/config.yaml`
2. Use CLI overrides for quick experiments
3. Check saved configs in `runs/{run_name}/config.yaml`
4. Version control your configs
5. Document your experiments

For questions or issues, refer to:
- `CLAUDE.md` - Comprehensive codebase guide
- `train.py` - Entry point implementation
- `ml_src/model.py` - Optimizer/scheduler logic
- `ml_src/dataset.py` - Transform implementation
