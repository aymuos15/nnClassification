# Troubleshooting Guide

Common issues and solutions.

## Installation Issues

### "torch not found"
```bash
# Reinstall PyTorch
pip install torch torchvision
```

### CUDA Not Available
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, train on CPU
python train.py --device cpu
```

## Training Issues

### Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# 1. Reduce batch size
python train.py --batch_size 8

# 2. Reduce image size (in config.yaml)
transforms:
  train:
    resize: [128, 128]

# 3. Use CPU
python train.py --device cpu
```

### Loss is NaN
**Symptom:** Loss becomes NaN during training

**Solutions:**
1. Lower learning rate: `--lr 0.0001`
2. Check data normalization
3. Verify labels are correct
4. Use gradient clipping (code modification needed)

### Training Too Slow
**Solutions:**
```bash
# 1. Increase workers
python train.py --num_workers 8

# 2. Larger batch size
python train.py --batch_size 64

# 3. Check GPU utilization
nvidia-smi
```

### Low Accuracy
**Solutions:**
1. Train more epochs
2. Use pretrained weights: `weights: 'DEFAULT'`
3. Increase model capacity
4. Check data augmentation
5. Verify dataset quality

## Data Issues

### "Found 0 files"
**Error:** `RuntimeError: Found 0 files in subfolders`

**Cause:** Incorrect directory structure

**Solution:** Organize data properly:
```
data/my_dataset/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```
See: [Data Preparation](../getting-started/data-preparation.md)

### Class Mismatch
**Symptom:** Weird metrics, confusion matrix wrong

**Cause:** Different class names in train/val/test

**Solution:** Ensure identical class folder names across all splits

## Resumption Issues

### Checkpoint Not Found
```bash
# Check file exists
ls runs/base/weights/

# Correct path
python train.py --resume runs/base/weights/last.pt
```

### Device Mismatch
**Problem:** Trained on GPU, can't resume on CPU

**Solution:** Load checkpoint explicitly to target device (code modification needed)

## Configuration Issues

### Override Not Working
```bash
# Wrong argument name
python train.py --learning_rate 0.01  # ❌

# Correct
python train.py --lr 0.01  # ✅
```

### Config File Not Found
```bash
# Use absolute path
python train.py --config /full/path/to/config.yaml
```

## TensorBoard Issues

### Port Already in Use
```bash
# Use different port
tensorboard --logdir runs/ --port 6007
```

### No Data Shown
**Cause:** TensorBoard looking in wrong directory

**Solution:**
```bash
# Point to correct directory
tensorboard --logdir runs/base/tensorboard
```

## Quick Diagnostic Commands

```bash
# Check Python/PyTorch
python -c "import torch; print(torch.__version__)"

# Check CUDA
nvidia-smi

# Check dataset structure
tree -L 3 data/my_dataset/

# Check GPU usage during training
watch -n 1 nvidia-smi

# View training log
cat runs/base/logs/train.log

# Check configuration
cat runs/base/config.yaml
```

## Getting Help

1. Check error message carefully
2. Review relevant documentation section
3. Check system info:
```bash
python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
```

## Related

- [Installation](../getting-started/installation.md)
- [Data Preparation](../getting-started/data-preparation.md)
- [Configuration](../configuration/overview.md)
- [FAQ](faq.md)
