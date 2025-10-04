# PyTorch Image Classifier

Production-ready image classification framework with ResNet18. Supports training, resumption, and comprehensive evaluation.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Data Structure (MANDATORY)
```
data/your_dataset/
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
**Must have train/val/test splits with identical class folders. Code will fail otherwise.**

### Train
```bash
# Basic training
python train.py

# Custom hyperparameters
python train.py --batch_size 32 --lr 0.01 --num_epochs 25

# Resume training
python train.py --resume runs/base/last.pt
```

### Inference
```bash
python inference.py --run_dir runs/base --checkpoint best.pt
```

### Visualization

```bash
# Launch TensorBoard
python visualise.py --mode launch --run_dir runs/base

# Visualize dataset samples
python visualise.py --mode samples --run_dir runs/base --split train --num_images 16

# Visualize model predictions (green=correct, red=incorrect)
python visualise.py --mode predictions --run_dir runs/base --split val

# Clean TensorBoard logs
python visualise.py --mode clean
```

Or use TensorBoard directly:
```bash
tensorboard --logdir runs/
```

## Configuration

Edit `ml_src/config.yaml` or use CLI overrides:
- `--data_dir`: Dataset path
- `--batch_size`: Batch size
- `--num_epochs`: Training epochs
- `--lr`: Learning rate
- `--num_workers`: Data loading workers

**Full config documentation:** See `CONFIG.md`

## Output Structure
```
runs/{run_name}/
├── config.yaml                      # Saved configuration
├── summary.txt                      # Training summary
├── weights/
│   ├── best.pt                     # Best model
│   └── last.pt                     # Latest checkpoint
├── logs/
│   ├── train.log
│   └── classification_report_*.txt
└── tensorboard/                     # TensorBoard logs
    └── events.out.tfevents.*       # Training metrics, plots, confusion matrices
```

## Documentation

📚 **[Complete Documentation](docs/README.md)** - Comprehensive guides organized by topic

**Quick Links:**
- [Quick Start Guide](docs/getting-started/quick-start.md) - Train in 5 minutes
- [Data Preparation](docs/getting-started/data-preparation.md) - Organize your dataset
- [Configuration Reference](docs/configuration/overview.md) - All settings explained
- [Training Guide](docs/user-guides/training.md) - Complete workflows
- [Monitoring & Visualization](docs/user-guides/monitoring.md) - TensorBoard and visualise.py
- [Troubleshooting](docs/reference/troubleshooting.md) - Common issues

**Documentation Sections:**
- **Getting Started** - Installation, data prep, quick start
- **Configuration** - Complete parameter reference
- **User Guides** - Training, inference, monitoring, tuning
- **Architecture** - System design and components
- **Development** - Extend and customize
- **Reference** - Best practices, troubleshooting, FAQ

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TensorBoard 2.14+
- CUDA (optional, for GPU training)

See `requirements.txt` for full dependencies.
