# Transform Configuration

## Overview

The `transforms` section configures data preprocessing and augmentation for each data split (train/val/test). Proper transforms are critical for model performance and generalization.

## Configuration Structure

```yaml
transforms:
  train:
    resize: [height, width]
    random_horizontal_flip: <bool>
    normalize:
      mean: [R, G, B]
      std: [R, G, B]
  val:
    resize: [height, width]
    normalize:
      mean: [R, G, B]
      std: [R, G, B]
  test:
    resize: [height, width]
    normalize:
      mean: [R, G, B]
      std: [R, G, B]
```

## Transform Pipeline Order

Transforms are applied in this order:
1. **Resize** - Standardize image dimensions
2. **Random Horizontal Flip** (train only) - Data augmentation
3. **Convert to Tensor** - Convert PIL image to PyTorch tensor
4. **Normalize** - Standardize pixel values

---

## `transforms.<split>.resize`

- **Type:** List of 2 integers [height, width]
- **Default:** `[224, 224]`
- **Description:** Target image size for resizing
- **Purpose:** Standardize image dimensions for batching and model input

### Usage

```yaml
transforms:
  train:
    resize: [224, 224]
  val:
    resize: [224, 224]
  test:
    resize: [224, 224]
```

### Common Sizes

| Size | Use Case | Notes |
|------|----------|-------|
| `[224, 224]` | ResNet, VGG, EfficientNet | Standard (default) |
| `[299, 299]` | Inception v3, Xception | Required for these models |
| `[384, 384]` | ViT-Large | Higher resolution transformers |
| `[512, 512]` | High-resolution models | More detail, slower |
| `[128, 128]` | Fast prototyping | Faster training, less detail |

### Considerations

**Larger Sizes (512×512):**
- ✅ More detail preserved
- ✅ Better for small objects
- ❌ Slower training
- ❌ More memory usage
- ❌ Smaller max batch size

**Smaller Sizes (128×128):**
- ✅ Faster training
- ✅ Less memory
- ✅ Larger batch sizes possible
- ❌ Less detail
- ❌ May hurt accuracy

**Non-Square Sizes:**
```yaml
transforms:
  train:
    resize: [256, 224]  # Height × Width
```
- Supported but uncommon
- Useful for specific aspect ratios

### Must Match Model Input

For custom models:
```yaml
model:
  input_size: 224  # Must match!

transforms:
  train:
    resize: [224, 224]  # Must match model.input_size
```

For base models, most expect 224×224 (Inception expects 299×299).

---

## `transforms.train.random_horizontal_flip`

- **Type:** Boolean
- **Default:** `true`
- **Description:** Randomly flip images horizontally with 50% probability
- **Purpose:** Data augmentation to improve generalization

### Usage

```yaml
transforms:
  train:
    random_horizontal_flip: true  # Enable for training
  val:
    random_horizontal_flip: false  # Or omit (no aug for val/test)
  test:
    random_horizontal_flip: false  # Or omit
```

### How It Works

- 50% chance: Image is flipped horizontally
- 50% chance: Image stays as-is
- Applied randomly each epoch
- Effectively doubles training data

### When to Enable

✅ **Enable for:**
- Natural images (animals, scenes, objects)
- Symmetric objects
- General photography
- Most computer vision tasks

**Examples:**
- ✅ Cats, dogs, birds
- ✅ Flowers, plants
- ✅ Landscapes, scenes
- ✅ Faces (generally symmetric)

### When to Disable

❌ **Disable for:**
- Text or documents
- Oriented objects (vehicles, planes)
- Asymmetric objects
- Direction matters
- Already augmented data

**Examples:**
- ❌ OCR, document classification
- ❌ Cars (usually face one direction)
- ❌ Handwritten digits (mirrored numbers confusing)
- ❌ Medical images (laterality matters)

### Best Practices

1. **Only apply to training split**
   - Val/test should match real-world data (no augmentation)

2. **Combine with other augmentations** (future extensions)
   - Rotation
   - Color jitter
   - Random crop

3. **Verify appropriateness** for your domain
   - Visualize flipped images
   - Ensure semantic meaning preserved

### Example: Training vs Validation

```yaml
transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true  # Augmentation
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

  val:
    resize: [224, 224]
    # NO flip - evaluate on real data distribution
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

---

## `transforms.<split>.normalize`

- **Type:** Dictionary with `mean` and `std` keys
- **Default:** ImageNet statistics
  ```yaml
  mean: [0.485, 0.456, 0.406]  # RGB channels
  std: [0.229, 0.224, 0.225]
  ```
- **Description:** Normalize images to zero mean and unit variance per channel
- **Purpose:** Stabilize training and match pretrained model expectations

### Usage

```yaml
transforms:
  train:
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

### Understanding Normalization

**Formula:**
```python
normalized_value = (original_value - mean) / std
```

**Example:**
```
Red channel pixel = 0.8
mean[0] = 0.485
std[0] = 0.229

normalized = (0.8 - 0.485) / 0.229 = 1.376
```

**Effect:**
- Centers each channel around 0
- Scales to unit variance
- Helps optimizer converge faster
- Matches pretrained model training conditions

### ImageNet Statistics (Default)

```yaml
normalize:
  mean: [0.485, 0.456, 0.406]  # R, G, B
  std: [0.229, 0.224, 0.225]
```

**Computed from ImageNet dataset (1.2M images).**

### When to Use ImageNet Stats

✅ **Use ImageNet stats when:**
- Using pretrained models (ResNet, EfficientNet, etc.)
- Training on natural RGB images
- General-purpose computer vision
- Similar domain to ImageNet
- **When in doubt, use ImageNet stats**

### When to Compute Custom Stats

⚠️ **Compute custom stats when:**
- Very different domain (medical, satellite, industrial)
- Non-RGB images (grayscale, multispectral)
- Specialized imaging (infrared, X-ray, microscopy)
- Data distribution very different from natural images

### Computing Custom Stats

```python
# Script to compute mean/std from your dataset
from torchvision import datasets, transforms
import torch

# Load dataset without normalization
dataset = datasets.ImageFolder(
    'data/your_dataset/train',
    transform=transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()  # Converts to [0, 1] range
    ])
)

loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

# Compute mean and std
mean = 0.
std = 0.
n_samples = 0

for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

print(f'mean: {mean.tolist()}')
print(f'std: {std.tolist()}')

# Use in config.yaml:
# transforms:
#   train:
#     normalize:
#       mean: [0.xxx, 0.xxx, 0.xxx]
#       std: [0.xxx, 0.xxx, 0.xxx]
```

### Must Be Consistent

**Critical:** All splits must use **same normalization**:

```yaml
transforms:
  train:
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  val:
    normalize:
      mean: [0.485, 0.456, 0.406]  # MUST match train
      std: [0.229, 0.224, 0.225]   # MUST match train
  
  test:
    normalize:
      mean: [0.485, 0.456, 0.406]  # MUST match train
      std: [0.229, 0.224, 0.225]   # MUST match train
```

**Why:** Model learns on normalized training data, must see same normalization at inference.

---

## Complete Transform Examples

### Example 1: Standard Configuration (Default)

```yaml
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

**Use for:** Most natural image classification tasks

---

### Example 2: High-Resolution Images

```yaml
transforms:
  train:
    resize: [512, 512]  # Larger for more detail
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

**Use for:** Small objects, fine details matter

---

### Example 3: Fast Prototyping

```yaml
transforms:
  train:
    resize: [128, 128]  # Smaller for speed
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  val:
    resize: [128, 128]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  test:
    resize: [128, 128]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Use for:** Quick experiments, testing pipeline

---

### Example 4: No Augmentation (Oriented Objects)

```yaml
transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: false  # Disabled (direction matters)
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

**Use for:** Cars, planes, text, asymmetric objects

---

### Example 5: Inception v3 Model

```yaml
transforms:
  train:
    resize: [299, 299]  # Inception requires 299×299
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  val:
    resize: [299, 299]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  test:
    resize: [299, 299]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

model:
  type: 'base'
  architecture: 'inception_v3'
  num_classes: 10
  weights: 'DEFAULT'
```

---

## Future Augmentations

**Currently not implemented, but can be added:**

To extend augmentation in `ml_src/dataset.py::get_transforms()`:

```python
# Random rotation
transforms.RandomRotation(15)  # ±15 degrees

# Color jitter
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)

# Random crop
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))

# Random erasing
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))

# Gaussian blur
transforms.GaussianBlur(kernel_size=3)

# Random perspective
transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
```

**See:** [Adding Transforms Guide](../development/adding-transforms.md)

---

## Best Practices

1. **Always normalize**
   - Critical for training stability
   - Use ImageNet stats for pretrained models

2. **Keep val/test clean**
   - No augmentation on validation/test
   - Match real-world data distribution

3. **Start with defaults**
   - 224×224 resize
   - Horizontal flip for training
   - ImageNet normalization

4. **Increase augmentation if overfitting**
   - Add more transforms
   - See [Adding Transforms](../development/adding-transforms.md)

5. **Match transforms to model**
   - Inception → 299×299
   - Most others → 224×224
   - ViT-L → 384×384

6. **Visualize transformed images**
   ```python
   # Quick check
   from torchvision import datasets
   from ml_src.dataset import get_transforms
   
   transforms = get_transforms(config)
   dataset = datasets.ImageFolder(
       'data/my_dataset/train',
       transform=transforms['train']
   )
   
   # Look at a few samples
   for i in range(5):
       img, label = dataset[i]
       print(f"Image shape: {img.shape}, Label: {label}")
   ```

---

## Troubleshooting

### Images Look Wrong

**Problem:** Normalized images look strange when visualized

**Solution:** This is normal! Normalized images are not meant for visualization. To visualize:
```python
# Denormalize for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_denorm = img * std + mean
```

### Model Not Converging

**Problem:** Training loss not decreasing

**Solution:** 
- Check normalization is applied
- Verify mean/std values are correct
- Ensure all splits use same normalization

### Size Mismatch Error

**Problem:** `RuntimeError: size mismatch`

**Solution:**
- Ensure `transforms.*.resize` matches `model.input_size` (for custom models)
- All transforms should use same resize dimensions

### Poor Generalization

**Problem:** Train accuracy high, val accuracy low

**Solution:**
- Enable `random_horizontal_flip: true`
- Consider adding more augmentations
- See [Troubleshooting Guide](../reference/troubleshooting.md)

---

## Related Configuration

- [Model Configuration](models.md) - Input size must match transforms
- [Data Configuration](data.md) - Dataset location
- [Examples](examples.md) - Complete configurations
- [Adding Transforms](../development/adding-transforms.md) - Extend augmentation

## Further Reading

- [torchvision.transforms Documentation](https://pytorch.org/vision/stable/transforms.html)
- [Data Augmentation Guide](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [AutoAugment Paper](https://arxiv.org/abs/1805.09501)
