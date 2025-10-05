# Training Guide

Complete guide to training workflows, best practices, and common scenarios.

## Prerequisites

Before training:

1. **Organize raw data** in `data/my_dataset/raw/` with class subdirectories
2. **Generate CV splits** using `splitting.py`
3. **Update configuration** with dataset name, data directory, and num_classes

See [Data Preparation Guide](../getting-started/data-preparation.md) for details.

---

## Basic Training

### Quick Start

```bash
# One-time: Generate CV splits
python splitting.py \
  --raw_data data/my_dataset/raw \
  --output data/my_dataset/splits \
  --folds 5

# Train fold 0 (default)
python train.py --fold 0

# Train with custom parameters
python train.py --fold 0 --num_epochs 50 --batch_size 32 --lr 0.01

# Train other folds
python train.py --fold 1
python train.py --fold 2
```

### Configuration File

Edit `ml_src/config.yaml`:

```yaml
data:
  dataset_name: 'my_dataset'  # Used in run directory names
  data_dir: 'data/my_dataset' # Must contain raw/ and splits/
  fold: 0                      # Which fold to use

model:
  num_classes: 3  # Must match number of classes in raw/
```

Then train:
```bash
python train.py
```

---

## Training Workflows

### 1. Initial Experiment

Quick test to verify everything works:

```bash
# Quick test (5 epochs) on fold 0
python train.py --fold 0 --num_epochs 5

# Check results
tensorboard --logdir runs/
```

### 2. Hyperparameter Search

Find optimal hyperparameters using one fold:

```bash
# Test learning rates on fold 0
python train.py --fold 0 --lr 0.0001 --num_epochs 20
python train.py --fold 0 --lr 0.001 --num_epochs 20
python train.py --fold 0 --lr 0.01 --num_epochs 20

# Test batch sizes
python train.py --fold 0 --batch_size 16 --lr 0.001 --num_epochs 20
python train.py --fold 0 --batch_size 32 --lr 0.001 --num_epochs 20
python train.py --fold 0 --batch_size 64 --lr 0.001 --num_epochs 20

# Compare in TensorBoard
tensorboard --logdir runs/
```

### 3. Cross-Validation Training

Train all folds with best hyperparameters:

```bash
# Use best hyperparams from search on all folds
python train.py --fold 0 --lr 0.001 --batch_size 32 --num_epochs 100
python train.py --fold 1 --lr 0.001 --batch_size 32 --num_epochs 100
python train.py --fold 2 --lr 0.001 --batch_size 32 --num_epochs 100
python train.py --fold 3 --lr 0.001 --batch_size 32 --num_epochs 100
python train.py --fold 4 --lr 0.001 --batch_size 32 --num_epochs 100
```

Or use a loop:
```bash
for fold in {0..4}; do
  python train.py --fold $fold --lr 0.001 --batch_size 32 --num_epochs 100
done
```

### 4. Model Comparison

Compare different architectures:

```bash
# Edit config for each model
# ResNet18
python train.py --fold 0 --num_epochs 50
# (config: architecture: 'resnet18')

# EfficientNet
python train.py --fold 0 --num_epochs 50
# (config: architecture: 'efficientnet_b0', weights: 'DEFAULT')

# MobileNetV2
python train.py --fold 0 --num_epochs 50
# (config: architecture: 'mobilenet_v2', weights: 'DEFAULT')

# Compare in TensorBoard
tensorboard --logdir runs/
```

---

## Understanding Output

### Console Output

Training displays real-time progress with color-coded information:

```
2025-10-05 01:25:02 | INFO     | Using fold: 0
2025-10-05 01:25:02 | INFO     | Dataset sizes: {'train': 265, 'val': 56, 'test': 59}
2025-10-05 01:25:02 | INFO     | Starting Training
2025-10-05 01:25:02 | INFO     | Epoch 0/49
--------------------------------------------------
2025-10-05 01:25:04 | INFO     | train Loss: 0.7019 Acc: 0.5623
2025-10-05 01:25:04 | INFO     | val Loss: 0.6481 Acc: 0.6607
2025-10-05 01:25:04 | SUCCESS  | New best model saved! Acc: 0.6607
```

**Color coding:**
- Timestamps: Cyan
- Epoch numbers: Yellow
- Train phase: Blue
- Validation phase: Magenta
- Loss/Accuracy values: Yellow
- Success messages: Green

### Run Directory

Each training run creates a directory under `runs/`:

```
runs/
└── hymenoptera_batch_32_fold_0/  # Auto-named based on dataset + parameters
    ├── config.yaml                # Saved configuration
    ├── summary.txt                # Training summary
    ├── weights/
    │   ├── best.pt               # Best model (highest val accuracy)
    │   └── last.pt               # Latest checkpoint (for resuming)
    ├── logs/
    │   ├── train.log             # Detailed training log
    │   ├── classification_report_train.txt
    │   └── classification_report_val.txt
    └── tensorboard/              # TensorBoard logs
        └── events.out.tfevents.*
```

**Key files:**
- `best.pt` - Use for deployment/inference (highest validation accuracy)
- `last.pt` - Use to resume interrupted training
- `summary.txt` - Quick overview of training results
- `train.log` - Complete training information
- `config.yaml` - Exact configuration used (for reproducibility)

---

## Monitoring Training

### TensorBoard (Recommended)

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open browser: http://localhost:6006
```

**Available metrics:**
- Loss curves (train/val)
- Accuracy curves (train/val)
- Learning rate schedule
- Confusion matrices
- Classification reports

**Tips:**
- Compare multiple runs side-by-side
- Use smoothing slider to reduce noise
- Download plots as images/PDFs
- Export data as CSV

### Log Files

```bash
# View training log
cat runs/hymenoptera_base_fold_0/logs/train.log

# View training summary
cat runs/hymenoptera_base_fold_0/summary.txt

# Watch training progress in real-time
tail -f runs/hymenoptera_base_fold_0/logs/train.log
```

### GPU Monitoring

```bash
# Check GPU usage once
nvidia-smi

# Continuous monitoring (updates every second)
watch -n 1 nvidia-smi

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Common Scenarios

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python train.py --batch_size 16  # or 8, 4
   ```

2. **Use CPU (slower but works):**
   ```bash
   python train.py --device cpu --batch_size 4
   ```

3. **Use gradient accumulation** (edit trainer code to accumulate over multiple steps)

### Slow Training

**Symptoms:**
- Low GPU utilization (<70%)
- Training takes much longer than expected

**Solutions:**

1. **Increase workers:**
   ```bash
   python train.py --num_workers 8
   ```

2. **Larger batch size:**
   ```bash
   python train.py --batch_size 64  # if GPU has memory
   ```

3. **Check disk I/O:**
   ```bash
   iostat -x 1  # Monitor disk usage
   ```

### Resume After Crash

If training is interrupted:

```bash
# Resume from last checkpoint
python train.py --resume runs/hymenoptera_base_fold_0/last.pt

# Resume and continue for more epochs
python train.py --resume runs/hymenoptera_base_fold_0/last.pt --num_epochs 100
```

**Note:** Resuming preserves:
- Model weights
- Optimizer state
- Learning rate scheduler state
- Training history (loss/accuracy curves)

### Overfitting

**Symptoms:**
- Training accuracy >> validation accuracy
- Validation loss increasing while training loss decreasing

**Solutions:**

1. **Use pretrained weights:**
   ```yaml
   model:
     architecture: 'resnet18'
     weights: 'DEFAULT'  # ImageNet pretrained
   ```

2. **Add data augmentation** (edit config transforms)
3. **Early stopping** (monitor validation, stop when it plateaus)
4. **Reduce model complexity** (use smaller architecture)
5. **Get more training data**

### Underfitting

**Symptoms:**
- Both training and validation accuracy are low
- Loss not decreasing

**Solutions:**

1. **Increase model capacity:**
   ```yaml
   model:
     architecture: 'resnet50'  # Instead of resnet18
   ```

2. **Train longer:**
   ```bash
   python train.py --num_epochs 100
   ```

3. **Increase learning rate:**
   ```bash
   python train.py --lr 0.01  # Instead of 0.001
   ```

4. **Check data quality** (mislabeled images, wrong num_classes)

---

## Advanced Options

### Custom Configuration File

```bash
# Create custom config
cp ml_src/config.yaml configs/experiment1.yaml

# Edit configs/experiment1.yaml
# ...

# Train with custom config
python train.py --config configs/experiment1.yaml
```

### Combining CLI and Config

CLI arguments override config file:

```bash
# Uses config file, but overrides batch_size and lr
python train.py --config configs/base.yaml --batch_size 64 --lr 0.01
```

### Training on Specific GPU

```bash
# Use GPU 1 instead of GPU 0
python train.py --device cuda:1

# Or set environment variable
CUDA_VISIBLE_DEVICES=1 python train.py
```

---

## Best Practices

### Before Training

1. ✅ **Verify dataset structure** - Check `raw/` and `splits/` exist
2. ✅ **Generate splits** - Run `splitting.py` once
3. ✅ **Update config** - Set dataset_name, data_dir, num_classes
4. ✅ **Quick test** - Train for 3-5 epochs to verify everything works
5. ✅ **Check GPU** - Run `nvidia-smi` to ensure GPU is available

### During Training

1. ✅ **Monitor TensorBoard** - Check loss/accuracy curves
2. ✅ **Watch GPU utilization** - Should be >90% for optimal speed
3. ✅ **Check logs** - Look for warnings or errors
4. ✅ **Save checkpoints** - Framework saves automatically
5. ✅ **Compare experiments** - Use TensorBoard to compare runs

### After Training

1. ✅ **Evaluate on test set** - Use `inference.py` with best.pt
2. ✅ **Analyze metrics** - Check confusion matrix, per-class performance
3. ✅ **Document results** - Note hyperparameters and performance
4. ✅ **Save important runs** - Keep config.yaml and best.pt
5. ✅ **Clean up** - Delete unnecessary checkpoints to save space

---

## Troubleshooting

### "Index file not found"

```
FileNotFoundError: Index file not found: data/my_dataset/splits/fold_0_train.txt
```

**Solution:** Generate splits first:
```bash
python splitting.py --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 5
```

### "CUDA not available"

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU
python train.py --device cpu
```

### "Model output doesn't match num_classes"

```
RuntimeError: Model output size (2) doesn't match dataset classes (3)
```

**Solution:** Update config:
```yaml
model:
  num_classes: 3  # Match number of class folders in raw/
```

### Data loading is slow

- Increase `num_workers`: `--num_workers 8`
- Move data to faster storage (SSD instead of HDD)
- Reduce image resolution in transforms

---

## Performance Optimization

### GPU Utilization

Target: >90% GPU utilization

**If too low:**
- Increase `num_workers`
- Increase `batch_size`
- Check disk I/O bottleneck

**If too high (OOM):**
- Decrease `batch_size`
- Use gradient accumulation
- Use mixed precision training

### Training Speed

**Baseline:** ~2-3 seconds per epoch for hymenoptera dataset (244 images, ResNet18, GPU)

**Speed up:**
- ✅ Use GPU instead of CPU (10-50x faster)
- ✅ Increase batch size (if GPU memory allows)
- ✅ Use pretrained weights (converges faster)
- ✅ Optimize num_workers (start with 4, adjust based on GPU utilization)
- ✅ Use SSD instead of HDD for data storage
- ✅ Reduce image resolution (if acceptable for task)

---

## Related Guides

- [Data Preparation](../getting-started/data-preparation.md) - Organize dataset
- [Hyperparameter Tuning](hyperparameter-tuning.md) - Systematic parameter search
- [Resuming Training](resuming-training.md) - Continue interrupted training
- [Inference](inference.md) - Evaluate trained models
- [Configuration Reference](../configuration/overview.md) - All parameters explained
- [Troubleshooting](../reference/troubleshooting.md) - Common issues and solutions

---

## Summary

**You've learned:**
- ✅ Basic training workflows (experiment → tune → cross-validate)
- ✅ Understanding training output and run directories
- ✅ Monitoring with TensorBoard and logs
- ✅ Common scenarios (OOM, slow training, resuming)
- ✅ Best practices (before/during/after training)
- ✅ Performance optimization tips

**Ready to train?**
```bash
python splitting.py --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 5
python train.py --fold 0 --batch_size 32 --lr 0.01 --num_epochs 50
```
