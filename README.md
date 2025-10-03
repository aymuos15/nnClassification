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

### View Training Metrics
```bash
# Start TensorBoard (view at http://localhost:6006)
tensorboard --logdir runs/

# Or for a specific run
tensorboard --logdir runs/base/tensorboard
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

- **`CLAUDE.md`** - Complete codebase guide (architecture, components, workflows)
- **`CONFIG.md`** - Detailed configuration reference (all parameters explained)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TensorBoard 2.14+
- CUDA (optional, for GPU training)

See `requirements.txt` for full dependencies.
