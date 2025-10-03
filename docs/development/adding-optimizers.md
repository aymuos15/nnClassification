# Adding Custom Transforms

Add new data augmentation techniques.

## Overview

Transforms are defined in `ml_src/dataset.py::get_transforms()`.

## Step 1: Update get_transforms()

Edit `ml_src/dataset.py`:

```python
from torchvision import transforms as T

def get_transforms(config):
    transform_config = config['transforms']
    
    transforms_dict = {}
    
    for split in ['train', 'val', 'test']:
        transform_list = []
        
        # Resize
        resize = transform_config[split]['resize']
        transform_list.append(T.Resize(resize))
        
        # NEW: Add rotation (training only)
        if split == 'train' and transform_config[split].get('random_rotation'):
            degrees = transform_config[split]['random_rotation']
            transform_list.append(T.RandomRotation(degrees))
        
        # Horizontal flip
        if transform_config[split].get('random_horizontal_flip'):
            transform_list.append(T.RandomHorizontalFlip())
        
        # NEW: Add color jitter (training only)
        if split == 'train' and transform_config[split].get('color_jitter'):
            cj = transform_config[split]['color_jitter']
            transform_list.append(T.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            ))
        
        # ToTensor and Normalize
        transform_list.append(T.ToTensor())
        normalize = transform_config[split]['normalize']
        transform_list.append(T.Normalize(
            mean=normalize['mean'],
            std=normalize['std']
        ))
        
        transforms_dict[split] = T.Compose(transform_list)
    
    return transforms_dict
```

## Step 2: Update Config

```yaml
transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    random_rotation: 15  # NEW
    color_jitter:        # NEW
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## Common Transforms

```python
# Random crop
T.RandomResizedCrop(224, scale=(0.8, 1.0))

# Gaussian blur
T.GaussianBlur(kernel_size=3)

# Random erasing
T.RandomErasing(p=0.5)

# Random perspective
T.RandomPerspective(distortion_scale=0.2)

# Grayscale
T.RandomGrayscale(p=0.1)
```

## Best Practices

1. Only apply to training split
2. Test impact on validation accuracy
3. Don't over-augment
4. Match domain (e.g., no flip for text)
5. Document why added

## Related

- [Transform Configuration](../configuration/transforms.md)
- [Dataset Module](../architecture/ml-src-modules.md#datasetpy)
EOF4
