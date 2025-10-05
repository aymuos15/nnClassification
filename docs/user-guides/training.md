# Training Guide

Complete guide to training workflows, best practices, and common scenarios.

## Prerequisites

Before training:

1. **Organize raw data** in `data/my_dataset/raw/` with class subdirectories
2. **Generate CV splits** using `ml-split`
3. **Update configuration** with dataset name, data directory, and num_classes

See [Data Preparation Guide](../getting-started/data-preparation.md) for details.

---

## Basic Training

### Quick Start

```bash
# One-time: Generate CV splits
ml-split --raw_data data/my_dataset/raw --folds 5
# Output: data/my_dataset/splits/

# Train fold 0 (default)
ml-train --config configs/my_dataset_config.yaml --fold 0

# Train with custom parameters
ml-train --config configs/my_dataset_config.yaml --fold 0 --num_epochs 50 --batch_size 32 --lr 0.01

# Train other folds
ml-train --config configs/my_dataset_config.yaml --fold 1
ml-train --config configs/my_dataset_config.yaml --fold 2
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
ml-train
```

---

## Training Workflows

### 1. Initial Experiment

Quick test to verify everything works:

```bash
# Quick test (5 epochs) on fold 0
ml-train --config configs/my_dataset_config.yaml --fold 0 --num_epochs 5

# Check results
tensorboard --logdir runs/
```

### 2. Hyperparameter Search

Find optimal hyperparameters using one fold:

```bash
# Test learning rates on fold 0
ml-train --config configs/my_dataset_config.yaml --fold 0 --lr 0.0001 --num_epochs 20
ml-train --config configs/my_dataset_config.yaml --fold 0 --lr 0.001 --num_epochs 20
ml-train --config configs/my_dataset_config.yaml --fold 0 --lr 0.01 --num_epochs 20

# Test batch sizes
ml-train --config configs/my_dataset_config.yaml --fold 0 --batch_size 16 --lr 0.001 --num_epochs 20
ml-train --config configs/my_dataset_config.yaml --fold 0 --batch_size 32 --lr 0.001 --num_epochs 20
ml-train --config configs/my_dataset_config.yaml --fold 0 --batch_size 64 --lr 0.001 --num_epochs 20

# Compare in TensorBoard
tensorboard --logdir runs/
```

### 3. Cross-Validation Training

Train all folds with best hyperparameters:

```bash
# Use best hyperparams from search on all folds
ml-train --config configs/my_dataset_config.yaml --fold 0 --lr 0.001 --batch_size 32 --num_epochs 100
ml-train --config configs/my_dataset_config.yaml --fold 1 --lr 0.001 --batch_size 32 --num_epochs 100
ml-train --config configs/my_dataset_config.yaml --fold 2 --lr 0.001 --batch_size 32 --num_epochs 100
ml-train --config configs/my_dataset_config.yaml --fold 3 --lr 0.001 --batch_size 32 --num_epochs 100
ml-train --config configs/my_dataset_config.yaml --fold 4 --lr 0.001 --batch_size 32 --num_epochs 100
```

Or use a loop:
```bash
for fold in {0..4}; do
  ml-train --config configs/my_dataset_config.yaml --fold $fold --lr 0.001 --batch_size 32 --num_epochs 100
done
```

**Important:** All folds are evaluated on the SAME held-out test set after training. Only train/val splits vary across folds. This ensures fair comparison.

### 4. Model Comparison

Compare different architectures:

```bash
# Edit config for each model
# ResNet18
ml-train --fold 0 --num_epochs 50
# (config: architecture: 'resnet18')

# EfficientNet
ml-train --fold 0 --num_epochs 50
# (config: architecture: 'efficientnet_b0', weights: 'DEFAULT')

# MobileNetV2
ml-train --fold 0 --num_epochs 50
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
   ml-train --batch_size 16  # or 8, 4
   ```

2. **Use CPU (slower but works):**
   ```bash
   ml-train --device cpu --batch_size 4
   ```

3. **Use gradient accumulation** (edit trainer code to accumulate over multiple steps)

### Slow Training

**Symptoms:**
- Low GPU utilization (<70%)
- Training takes much longer than expected

**Solutions:**

1. **Increase workers:**
   ```bash
   ml-train --num_workers 8
   ```

2. **Larger batch size:**
   ```bash
   ml-train --batch_size 64  # if GPU has memory
   ```

3. **Check disk I/O:**
   ```bash
   iostat -x 1  # Monitor disk usage
   ```

### Resume After Crash

If training is interrupted:

```bash
# Resume from last checkpoint
ml-train --resume runs/hymenoptera_base_fold_0/last.pt

# Resume and continue for more epochs
ml-train --resume runs/hymenoptera_base_fold_0/last.pt --num_epochs 100
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
   ml-train --num_epochs 100
   ```

3. **Increase learning rate:**
   ```bash
   ml-train --lr 0.01  # Instead of 0.001
   ```

4. **Check data quality** (mislabeled images, wrong num_classes)

---

## Trainer Selection

The framework supports multiple specialized trainers optimized for different use cases. Each trainer provides unique capabilities while maintaining a consistent interface.

### Available Trainers

| Trainer Type | Use Case | Requirements | Speed | Complexity |
|--------------|----------|--------------|-------|------------|
| `standard` | Beginners, single GPU, simple workflows | None | Baseline | Low |
| `mixed_precision` | Faster training, memory savings | CUDA GPU | 2-3x faster | Low |
| `accelerate` | Multi-GPU, distributed, TPU | `accelerate` package | Variable | Medium |
| `dp` | Privacy-sensitive data, research | `opacus` package | Slower | High |

### Configuration

Set the trainer type in your config file:

```yaml
training:
  trainer_type: 'standard'  # Options: 'standard', 'mixed_precision', 'accelerate', 'dp'
```

**Default behavior:** If `trainer_type` is not specified, the framework uses `standard` trainer.

### Trainer-Specific Examples

#### Standard Trainer (Default)

Traditional PyTorch training - simple and reliable:

```yaml
training:
  trainer_type: 'standard'
  num_epochs: 50
  batch_size: 32
```

```bash
ml-train --config configs/my_config.yaml
```

**Use when:**
- Learning the framework
- CPU-only environment
- Simple single-GPU workflows
- No special requirements

#### Mixed Precision Trainer

Automatic Mixed Precision (AMP) for faster training with reduced memory:

```yaml
training:
  trainer_type: 'mixed_precision'
  amp_dtype: 'float16'  # Options: 'float16', 'bfloat16'
  num_epochs: 50
  batch_size: 64  # Can use larger batch sizes due to memory savings
```

```bash
ml-train --config configs/my_config.yaml
```

**Use when:**
- Have modern NVIDIA GPU (Volta/Turing/Ampere or newer)
- Want 2-3x faster training
- Need to fit larger batches in memory
- Memory is a constraint

**Benefits:**
- 2-3x training speedup on compatible GPUs
- 50% memory reduction
- Minimal accuracy impact
- Easy to enable (just change config)

#### Accelerate Trainer

Multi-GPU and distributed training with Hugging Face Accelerate:

```yaml
training:
  trainer_type: 'accelerate'
  num_epochs: 50
  batch_size: 32  # Per-device batch size
  gradient_accumulation_steps: 2  # Optional
```

**One-time setup:**
```bash
pip install accelerate
accelerate config  # Interactive configuration
```

**Launch training:**
```bash
# Single GPU (same as standard)
ml-train --config configs/my_config.yaml

# Multi-GPU
accelerate launch ml-train --config configs/my_config.yaml

# Distributed across nodes
accelerate launch --multi_gpu --num_processes 4 ml-train --config configs/my_config.yaml
```

**Use when:**
- Have multiple GPUs
- Training on TPU
- Need distributed training across machines
- Want gradient accumulation
- Scaling to larger datasets

#### Differential Privacy Trainer

Training with privacy guarantees using Opacus:

```yaml
training:
  trainer_type: 'dp'
  num_epochs: 50
  batch_size: 32
  dp:
    noise_multiplier: 1.1  # Higher = more privacy, lower accuracy
    max_grad_norm: 1.0     # Gradient clipping threshold
    target_epsilon: 3.0    # Privacy budget (lower = stronger privacy)
    target_delta: 1e-5     # Privacy parameter (typically 1/n_samples)
```

```bash
pip install opacus
ml-train --config configs/my_config.yaml
```

**Use when:**
- Training on sensitive/private data (medical, financial)
- Need formal privacy guarantees
- Research on differential privacy
- Regulatory compliance requirements

**Note:** DP training is slower and may require hyperparameter tuning for good accuracy.

### Quick Decision Guide

```
Need privacy guarantees? → Yes → dp
                         ↓ No
Have multiple GPUs? → Yes → accelerate
                     ↓ No
Have single GPU? → Yes → mixed_precision (recommended)
                  ↓ No (CPU only)
StandardTrainer
```

**For most users:** Start with `standard` trainer. Once comfortable, switch to `mixed_precision` for faster training on GPU.

**For advanced users:** See [Advanced Training Guide](advanced-training.md) for detailed documentation on each specialized trainer.

---

## Advanced Options

### Custom Configuration File

```bash
# Create custom config
cp ml_src/config.yaml configs/experiment1.yaml

# Edit configs/experiment1.yaml
# ...

# Train with custom config
ml-train --config configs/experiment1.yaml
```

### Combining CLI and Config

CLI arguments override config file:

```bash
# Uses config file, but overrides batch_size and lr
ml-train --config configs/base.yaml --batch_size 64 --lr 0.01
```

### Training on Specific GPU

```bash
# Use GPU 1 instead of GPU 0
ml-train --device cuda:1

# Or set environment variable
CUDA_VISIBLE_DEVICES=1 ml-train
```

---

## Best Practices

### Before Training

1. ✅ **Verify dataset structure** - Check `raw/` and `splits/` exist
2. ✅ **Generate splits** - Run `ml-split` once
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

1. ✅ **Test evaluation** - Automatically runs on held-out test set after training
2. ✅ **Analyze metrics** - Check confusion matrix in TensorBoard and classification report
3. ✅ **Document results** - Note hyperparameters and performance
4. ✅ **Save important runs** - Keep config.yaml and best.pt
5. ✅ **Clean up** - Delete unnecessary checkpoints to save space

**Note:** The framework automatically evaluates the best model on the test set after training completes. Test results are saved to:
- `runs/{run_name}/logs/classification_report_test.txt`
- TensorBoard (confusion matrix, accuracy, classification report)

---

## Troubleshooting

### "Index file not found"

```
FileNotFoundError: Index file not found: data/my_dataset/splits/fold_0_train.txt
```

**Solution:** Generate splits first:
```bash
ml-split --raw_data data/my_dataset/raw --folds 5
```

### "CUDA not available"

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, use CPU
ml-train --device cpu
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
- [Configuration Reference](../configuration/README.md) - All parameters explained
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
ml-split --raw_data data/my_dataset/raw --folds 5
ml-train --config configs/my_dataset_config.yaml --fold 0 --batch_size 32 --lr 0.01 --num_epochs 50
```
