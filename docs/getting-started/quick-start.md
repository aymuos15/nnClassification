# Quick Start Guide

Get up and running with your first training run in 5 minutes!

---

## Prerequisites

- ✅ Python 3.8+ installed
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ Dataset organized ([see Data Preparation](data-preparation.md))

---

## 5-Minute Quick Start

### Step 1: Verify Installation

```bash
# Check PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
```

### Step 2: Use Example Dataset

The repo includes a sample dataset (ants vs bees):

```bash
# Check it exists
ls data/hymenoptera_data/

# Should show: train/ val/ test/
```

If missing, extract it:
```bash
unzip hymenoptera_data.zip -d data/
```

### Step 3: Run Training

```bash
# Train for 3 epochs (quick test)
python train.py --num_epochs 3
```

**Expected output:**
```
[INFO] Configuration loaded
[INFO] Training on cuda:0
[INFO] Starting training...
Epoch 1/3: train_loss=0.512, val_loss=0.498, val_acc=75.2%
Epoch 2/3: train_loss=0.312, val_loss=0.287, val_acc=87.4%
Epoch 3/3: train_loss=0.198, val_loss=0.245, val_acc=91.8%
[INFO] Training complete! Best val acc: 91.8%
```

### Step 4: Check Results

```bash
# View training outputs
ls runs/base/

# Should show:
# - config.yaml
# - summary.txt
# - weights/best.pt
# - weights/last.pt
# - logs/
# - tensorboard/
```

### Step 5: View Metrics

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open in browser: http://localhost:6006
```

**🎉 Congratulations! You've completed your first training run.**

---

## Train on Your Own Data

### Step 1: Organize Your Dataset

**Required structure:**
```
data/my_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

**See:** [Data Preparation Guide](data-preparation.md) for detailed instructions.

### Step 2: Update Configuration

Edit `ml_src/config.yaml`:

```yaml
data:
  data_dir: 'data/my_dataset'  # Your dataset path

model:
  num_classes: 5  # Number of your classes
```

### Step 3: Train

```bash
python train.py
```

---

## Basic Training Options

### Adjust Training Duration

```bash
# Quick test (5 epochs)
python train.py --num_epochs 5

# Full training (50 epochs)
python train.py --num_epochs 50
```

### Change Batch Size

```bash
# Smaller batch (if GPU memory limited)
python train.py --batch_size 16

# Larger batch (if you have powerful GPU)
python train.py --batch_size 64
```

### Adjust Learning Rate

```bash
# Lower learning rate (more stable)
python train.py --lr 0.0001

# Higher learning rate (faster convergence)
python train.py --lr 0.01
```

### Combine Options

```bash
python train.py --batch_size 32 --lr 0.01 --num_epochs 50
```

---

## Resume Training

If training was interrupted:

```bash
# Resume from last checkpoint
python train.py --resume runs/base/last.pt

# Resume and train more epochs
python train.py --resume runs/base/last.pt --num_epochs 50
```

---

## Run Inference

After training, evaluate on test data:

```bash
# Use best model
python inference.py --run_dir runs/base --checkpoint best.pt
```

**Expected output:**
```
[INFO] Loading model from runs/base/weights/best.pt
[INFO] Running inference on test set...

Test Accuracy: 92.5%

Classification Report:
               precision    recall  f1-score   support
        ants       0.90      0.94      0.92        50
        bees       0.94      0.91      0.93        50

Confusion Matrix saved to: runs/base/plots/confusion_matrix_test.png
```

---

## Monitor Training

### TensorBoard (Recommended)

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open browser: http://localhost:6006
```

**You can view:**
- Training and validation loss curves
- Accuracy curves
- Learning rate schedule
- Confusion matrices
- Classification reports

### Log Files

```bash
# View training log
cat runs/base/logs/train.log

# View training summary
cat runs/base/summary.txt
```

### Real-Time Monitoring

```bash
# Watch training progress
tail -f runs/base/logs/train.log
```

---

## Common Commands Cheat Sheet

```bash
# Basic training
python train.py

# Quick test
python train.py --num_epochs 3 --batch_size 8

# Custom dataset
python train.py --data_dir data/my_dataset

# High-performance training
python train.py --batch_size 64 --num_workers 8 --num_epochs 100

# Resume training
python train.py --resume runs/base/last.pt

# Inference
python inference.py --run_dir runs/base --checkpoint best.pt

# Monitor training
tensorboard --logdir runs/

# Check GPU
nvidia-smi
watch -n 1 nvidia-smi  # Continuous monitoring
```

---

## Training Workflow

### Typical Workflow

1. **Quick test** (verify everything works):
   ```bash
   python train.py --num_epochs 3
   ```

2. **Check results** in TensorBoard:
   ```bash
   tensorboard --logdir runs/
   ```

3. **Tune hyperparameters** (try different values):
   ```bash
   python train.py --lr 0.001 --batch_size 32
   python train.py --lr 0.01 --batch_size 32
   python train.py --lr 0.01 --batch_size 64
   ```

4. **Full training** (with best hyperparams):
   ```bash
   python train.py --lr 0.01 --batch_size 64 --num_epochs 100
   ```

5. **Evaluate** on test set:
   ```bash
   python inference.py --run_dir runs/batch_64_lr_0.01_epochs_100 --checkpoint best.pt
   ```

---

## Understanding Output Structure

After training, you'll see:

```
runs/
└── base/                        # Run directory (auto-named)
    ├── config.yaml             # Saved configuration
    ├── summary.txt             # Training summary
    ├── weights/
    │   ├── best.pt            # Best model (highest val accuracy)
    │   └── last.pt            # Latest checkpoint (for resuming)
    ├── logs/
    │   ├── train.log          # Detailed training log
    │   ├── classification_report_train.txt
    │   ├── classification_report_val.txt
    │   └── classification_report_test.txt
    ├── plots/
    │   ├── confusion_matrix_train.png
    │   ├── confusion_matrix_val.png
    │   └── confusion_matrix_test.png
    └── tensorboard/           # TensorBoard logs
        └── events.out.tfevents.*
```

### Key Files

- **`best.pt`** - Use for deployment/inference (highest validation accuracy)
- **`last.pt`** - Use to resume interrupted training
- **`summary.txt`** - Quick overview of training results
- **`train.log`** - Detailed training information
- **`config.yaml`** - Exact configuration used (for reproducibility)

---

## Troubleshooting Quick Fixes

### Out of Memory Error

```bash
# Reduce batch size
python train.py --batch_size 8

# Or train on CPU
python train.py --device cpu --batch_size 4
```

### Training Too Slow

```bash
# Increase workers (if CPU has many cores)
python train.py --num_workers 8

# Use larger batch size (if GPU has memory)
python train.py --batch_size 64
```

### CUDA Not Available

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, train on CPU
python train.py --device cpu
```

### Data Loading Error

```bash
# Verify dataset structure
ls -R data/my_dataset/

# Should show train/val/test with class subfolders
# See: Data Preparation Guide
```

---

## Next Steps

### Learn More

- **Configuration:** [Configuration Overview](../configuration/overview.md)
- **Training Guide:** [Training Workflows](../user-guides/training.md)
- **Hyperparameter Tuning:** [Tuning Guide](../user-guides/hyperparameter-tuning.md)
- **Model Selection:** [Model Configuration](../configuration/models.md)

### Try Different Models

```bash
# Use EfficientNet (better accuracy)
# Edit ml_src/config.yaml:
model:
  type: 'base'
  architecture: 'efficientnet_b0'
  weights: 'DEFAULT'

python train.py
```

### Experiment with Hyperparameters

```bash
# Systematic search
python train.py --lr 0.001 --batch_size 16
python train.py --lr 0.001 --batch_size 32
python train.py --lr 0.01 --batch_size 16
python train.py --lr 0.01 --batch_size 32

# Compare in TensorBoard
tensorboard --logdir runs/
```

---

## Getting Help

### Check Documentation

- [Installation Guide](installation.md) - Setup issues
- [Data Preparation](data-preparation.md) - Dataset organization
- [Configuration Reference](../configuration/) - All parameters explained
- [Troubleshooting](../reference/troubleshooting.md) - Common issues
- [FAQ](../reference/faq.md) - Frequently asked questions

### Verify System Info

```bash
python -c "
import sys, torch, torchvision
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Summary

**You've learned how to:**
- ✅ Run your first training
- ✅ Monitor training with TensorBoard
- ✅ Adjust hyperparameters via CLI
- ✅ Resume interrupted training
- ✅ Run inference on trained models
- ✅ Navigate output structure

**Ready for more?** Check out the [full training guide](../user-guides/training.md) for advanced workflows.

---

## Quick Reference Card

```bash
# === TRAINING ===
python train.py                          # Basic training
python train.py --num_epochs 50          # More epochs
python train.py --batch_size 32          # Different batch size
python train.py --lr 0.01                # Different learning rate
python train.py --data_dir data/custom   # Custom dataset
python train.py --resume runs/base/last.pt  # Resume training

# === INFERENCE ===
python inference.py --run_dir runs/base --checkpoint best.pt

# === MONITORING ===
tensorboard --logdir runs/               # View all runs
tail -f runs/base/logs/train.log         # Watch training log
cat runs/base/summary.txt                # View summary

# === SYSTEM ===
nvidia-smi                               # Check GPU
watch -n 1 nvidia-smi                    # Monitor GPU
ls runs/                                 # List training runs
```

**Happy training!** 🚀
