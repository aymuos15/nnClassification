# Data Preparation Guide

## Overview

Proper data organization is **critical** for this framework. The code uses PyTorch's `ImageFolder` class, which requires a specific directory structure. **If you don't follow this structure, training will fail immediately.**

---

## ⚠️ MANDATORY DIRECTORY STRUCTURE

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

---

## Requirements (ALL MANDATORY)

1. ✅ **Three splits:** Must have `train/`, `val/`, and `test/` directories
2. ✅ **Same classes:** All three splits must have identical class folder names
3. ✅ **Class folders:** Each class must be in its own subdirectory
4. ✅ **Images in class folders:** Images go directly inside class folders (no nested subdirectories)
5. ✅ **Matching num_classes:** Number of class folders must equal `model.num_classes` in config

---

## ❌ Common Mistakes (WILL FAIL)

### Mistake 1: Missing Splits

**WRONG:**
```
data_dir/
├── train/
│   ├── class1/
│   └── class2/
└── val/              # Missing test/ → FAILS
```

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data_dir/test'
```

**Solution:** Create all three splits (train, val, test).

---

### Mistake 2: Images Not in Class Folders

**WRONG:**
```
data_dir/
├── train/
│   ├── img1.jpg      # Images directly in train/ → FAILS
│   ├── img2.jpg
│   └── img3.jpg
```

**Error:**
```
RuntimeError: Found 0 files in subfolders of: data_dir/train
```

**Solution:** Put images inside class subdirectories.

---

### Mistake 3: Mismatched Class Names

**WRONG:**
```
data_dir/
├── train/
│   ├── cats/         # "cats" (plural)
│   └── dogs/         # "dogs" (plural)
├── val/
│   ├── cat/          # "cat" (singular) → DIFFERENT NAME
│   └── dog/          # "dog" (singular) → DIFFERENT NAME
└── test/
    ├── cats/
    └── dogs/
```

**Problem:** Class indices won't match between splits. Metrics will be incorrect.

**Solution:** Use **identical** class names across all splits.

---

### Mistake 4: Nested Subdirectories

**WRONG:**
```
data_dir/
├── train/
│   └── class1/
│       └── subset/   # Extra level → Images won't be found
│           ├── img1.jpg
│           └── img2.jpg
```

**Error:** Images in nested folders won't be loaded.

**Solution:** Images must be directly inside class folders (no additional nesting).

---

### Mistake 5: Wrong num_classes

**WRONG:**
```
data_dir/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/       # 3 classes in data

# But config says:
model:
  num_classes: 2      # Config expects 2 classes → MISMATCH
```

**Problem:** Model output size doesn't match dataset classes.

**Solution:** Set `num_classes` to match number of class folders.

---

## ✅ Correct Example

### Example: Ants vs Bees Dataset

```
data/hymenoptera_data/
├── train/
│   ├── ants/
│   │   ├── 0013035.jpg
│   │   ├── 1030023.jpg
│   │   ├── 1030023.jpg
│   │   └── ... (124 images total)
│   └── bees/
│       ├── 1092977343_cb42b38d62.jpg
│       ├── 1093831624_fb5fbe2308.jpg
│       └── ... (121 images total)
├── val/
│   ├── ants/
│   │   ├── 10308379_1b6c72e180.jpg
│   │   └── ... (70 images total)
│   └── bees/
│       ├── 1093831624_fb5fbe2308.jpg
│       └── ... (83 images total)
└── test/
    ├── ants/
    │   └── ... (50 images total)
    └── bees/
        └── ... (50 images total)
```

**Configuration:**
```yaml
data:
  data_dir: 'data/hymenoptera_data'

model:
  num_classes: 2  # Must match: ants, bees = 2 classes
```

---

## How to Organize Your Data

### Method 1: Manual Organization

```bash
#!/bin/bash
# organize_data.sh

# Create directory structure
mkdir -p data/my_dataset/{train,val,test}/{class1,class2,class3}

# Move images to correct locations
# (Adjust paths for your dataset)
mv /path/to/class1/train_images/* data/my_dataset/train/class1/
mv /path/to/class1/val_images/* data/my_dataset/val/class1/
mv /path/to/class1/test_images/* data/my_dataset/test/class1/

# Repeat for all classes...
```

---

### Method 2: Python Script (Automatic Split)

Use this if you have images organized by class but not split:

```python
#!/usr/bin/env python3
"""
organize_dataset.py - Automatically split and organize dataset
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = 'original_data/'  # Your current data location
TARGET_DIR = 'data/my_dataset/'  # Where to create organized dataset
CLASSES = ['class1', 'class2', 'class3']  # Your class names

# Split ratios
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing

# Process each class
for class_name in CLASSES:
    print(f"Processing {class_name}...")
    
    # Get all images for this class
    source_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(source_class_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"  Found {len(images)} images")
    
    # Split: 70% train, 15% val, 15% test
    train, temp = train_test_split(
        images, 
        test_size=(VAL_RATIO + TEST_RATIO), 
        random_state=42
    )
    val, test = train_test_split(
        temp, 
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), 
        random_state=42
    )
    
    print(f"  Split: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        target_class_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
    
    # Copy files to appropriate splits
    for img in train:
        shutil.copy2(
            os.path.join(source_class_dir, img),
            os.path.join(TARGET_DIR, 'train', class_name, img)
        )
    
    for img in val:
        shutil.copy2(
            os.path.join(source_class_dir, img),
            os.path.join(TARGET_DIR, 'val', class_name, img)
        )
    
    for img in test:
        shutil.copy2(
            os.path.join(source_class_dir, img),
            os.path.join(TARGET_DIR, 'test', class_name, img)
        )

print("\nDataset organization complete!")
print(f"Organized dataset location: {TARGET_DIR}")
```

**Usage:**
```bash
python organize_dataset.py
```

---

### Method 3: Symbolic Links (Save Disk Space)

If disk space is limited, use symbolic links instead of copying:

```python
# Replace shutil.copy2() with os.symlink()
os.symlink(
    os.path.abspath(os.path.join(source_class_dir, img)),
    os.path.join(TARGET_DIR, 'train', class_name, img)
)
```

**Note:** Original files must not be moved/deleted.

---

## Verify Your Dataset Structure

### Quick Check with tree

```bash
# Check structure (2 levels deep)
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
```

### Count Images Per Split

```bash
echo "Train images:"
find data/my_dataset/train -type f | wc -l

echo "Val images:"
find data/my_dataset/val -type f | wc -l

echo "Test images:"
find data/my_dataset/test -type f | wc -l
```

### Count Images Per Class

```bash
for split in train val test; do
  echo "$split split:"
  for class_dir in data/my_dataset/$split/*; do
    class_name=$(basename "$class_dir")
    count=$(find "$class_dir" -type f | wc -l)
    echo "  $class_name: $count images"
  done
done
```

**Example output:**
```
train split:
  ants: 124 images
  bees: 121 images
val split:
  ants: 70 images
  bees: 83 images
test split:
  ants: 50 images
  bees: 50 images
```

---

## Verification Script

```python
#!/usr/bin/env python3
"""
verify_dataset.py - Check dataset structure before training
"""
import os
from pathlib import Path

def verify_dataset(data_dir):
    """Verify dataset structure is correct."""
    
    print(f"Verifying dataset at: {data_dir}\n")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Directory does not exist: {data_dir}")
        return False
    
    # Check for required splits
    required_splits = ['train', 'val', 'test']
    for split in required_splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"❌ ERROR: Missing required split: {split}/")
            return False
        print(f"✓ Found {split}/ directory")
    
    # Get class names from train split
    train_dir = os.path.join(data_dir, 'train')
    classes = [d for d in os.listdir(train_dir) 
               if os.path.isdir(os.path.join(train_dir, d))]
    
    if len(classes) == 0:
        print(f"❌ ERROR: No class folders found in train/")
        return False
    
    print(f"\n✓ Found {len(classes)} classes: {', '.join(classes)}")
    
    # Verify same classes in all splits
    for split in required_splits:
        split_classes = set(os.listdir(os.path.join(data_dir, split)))
        if set(classes) != split_classes:
            print(f"❌ ERROR: Class mismatch in {split}/ split")
            print(f"  Expected: {set(classes)}")
            print(f"  Found: {split_classes}")
            return False
    
    print("✓ All splits have same classes")
    
    # Count images per split and class
    print("\nImage counts:")
    for split in required_splits:
        print(f"  {split}:")
        for class_name in classes:
            class_path = os.path.join(data_dir, split, class_name)
            images = [f for f in os.listdir(class_path)
                     if os.path.isfile(os.path.join(class_path, f))]
            print(f"    {class_name}: {len(images)} images")
    
    print("\n✅ Dataset structure is valid!")
    print(f"\nSet in config.yaml:")
    print(f"  data:")
    print(f"    data_dir: '{data_dir}'")
    print(f"  model:")
    print(f"    num_classes: {len(classes)}")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python verify_dataset.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    verify_dataset(data_dir)
```

**Usage:**
```bash
python verify_dataset.py data/my_dataset
```

---

## Supported Image Formats

The framework supports any format that PIL/Pillow can read:

- **Common formats:** JPG, JPEG, PNG, BMP, GIF, TIFF
- **Others:** WebP, PPM, PGM, PBM, etc.

**Recommendation:** Use JPG or PNG for best compatibility.

---

## Dataset Split Ratios

### Recommended Splits

| Total Images | Train | Val | Test |
|--------------|-------|-----|------|
| < 1,000 | 70% | 15% | 15% |
| 1,000 - 10,000 | 70% | 20% | 10% |
| 10,000+ | 80% | 10% | 10% |

### Why Three Splits?

- **Train:** Used for training the model
- **Val:** Used for hyperparameter tuning and model selection
- **Test:** Final evaluation (never seen during training)

**Important:** Never use test set during training or validation!

---

## Configuration

After organizing your dataset, update the configuration:

```yaml
# ml_src/config.yaml

data:
  data_dir: 'data/my_dataset'  # Path to your dataset

model:
  num_classes: 3  # Number of class folders (must match!)
```

Or use CLI override:
```bash
python train.py --data_dir data/my_dataset
```

---

## Troubleshooting

### Problem: "Found 0 files"

**Error:**
```
RuntimeError: Found 0 files in subfolders of: data/my_dataset/train
```

**Causes:**
1. Images not in class subfolders
2. Wrong image file extensions
3. Empty class folders

**Solution:**
```bash
# Check structure
tree -L 3 data/my_dataset/

# Check for images
find data/my_dataset/ -type f -name "*.jpg" | head
```

---

### Problem: Class Index Mismatch

**Symptom:** Metrics look wrong, confusion matrix off

**Cause:** Different class order in train/val/test

**Solution:** Ensure **identical** class folder names across all splits.

---

### Problem: "Doesn't match num_classes"

**Error:**
```
RuntimeError: Model output size (2) doesn't match dataset classes (3)
```

**Solution:** Update `num_classes` in config:
```yaml
model:
  num_classes: 3  # Match number of class folders
```

---

## Example Datasets

### Built-in Example: Hymenoptera

The repository includes a sample dataset (ants vs bees):

```bash
# Already included in repo
data/hymenoptera_data/
├── train/
│   ├── ants/
│   └── bees/
├── val/
│   ├── ants/
│   └── bees/
└── test/
    ├── ants/
    └── bees/
```

**Use it to verify installation:**
```bash
python train.py --num_epochs 3
```

---

## Best Practices

1. **Verify structure** before training (use verification script)
2. **Use consistent naming** across splits
3. **Balance classes** if possible (similar number of images per class)
4. **Check image quality** (corrupt files will cause errors)
5. **Document splits** (note how you split the data)
6. **Version control** your organization script
7. **Backup original data** before reorganizing

---

## Next Steps

After organizing your data:

1. **Verify structure:**
   ```bash
   python verify_dataset.py data/my_dataset
   ```

2. **Update configuration:**
   ```yaml
   data:
     data_dir: 'data/my_dataset'
   model:
     num_classes: <your_num_classes>
   ```

3. **Start training:**
   ```bash
   python train.py
   ```

4. **See:** [Quick Start Guide](quick-start.md)

---

## Summary

✅ **Checklist:**
- [ ] Data organized in train/val/test splits
- [ ] Each split has class subfolders
- [ ] Same class names across all splits
- [ ] Images directly in class folders (no nesting)
- [ ] Structure verified with script
- [ ] `num_classes` set correctly in config

**Remember:** This structure is **mandatory**. The code will not work without it. Take time to organize your data properly—it will save hours of debugging later!
