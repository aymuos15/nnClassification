# Getting Started with PyTorch Image Classifier

Welcome to the PyTorch Image Classification framework! This guide will take you from installation to your first trained model in under 15 minutes.

---

## About This Guide

This **Getting Started** section is designed for users who are new to the framework. Whether you're a machine learning beginner or an experienced practitioner, these guides will help you quickly set up your environment, organize your data, and train your first image classification model.

**What you'll learn:**
- How to install and verify the framework
- How to organize datasets using the index-based cross-validation system
- How to train your first model and monitor results
- Basic workflows for training, evaluation, and hyperparameter tuning

**Time investment:** 15-30 minutes to complete all guides

---

## Guide Overview

This section contains three essential guides that should be followed in order:

### 1. Installation Guide

**[installation.md](installation.md)** - Complete setup and dependency installation

**Covers:**
- System requirements (hardware and software)
- Step-by-step installation process
- Virtual environment setup (venv and conda)
- GPU configuration (CUDA, cuDNN)
- Installation verification
- Platform-specific instructions (Linux, macOS, Windows)
- Docker and cloud platform setup (optional)
- Troubleshooting common installation issues

**When to use:**
- First-time setup
- Installing on a new machine
- Debugging installation problems
- Setting up GPU support

**Time required:** 5-15 minutes (depending on GPU setup)

---

### 2. Data Preparation Guide

**[data-preparation.md](data-preparation.md)** - Organize your dataset (CRITICAL)

**Covers:**
- Index-based cross-validation system explained
- Mandatory directory structure (`raw/` and `splits/`)
- Step-by-step dataset organization
- Using `ml-split` to generate cross-validation folds
- Complete examples with real datasets
- Verification scripts
- Troubleshooting data-related issues
- Best practices for dataset management

**When to use:**
- Before training on any dataset
- Setting up a new dataset
- Understanding the data pipeline
- Debugging data loading errors

**Time required:** 10-20 minutes

**Warning:** This is the most critical guide. The framework will NOT work without proper data organization. Take time to understand the required structure before proceeding to training.

---

### 3. Quick Start Guide

**[quick-start.md](quick-start.md)** - Train your first model in 5 minutes

**Covers:**
- Running your first training session
- Using the example dataset (ants vs bees)
- Training on custom datasets
- Basic CLI options (batch size, epochs, learning rate)
- Monitoring training progress
- Running inference on trained models
- Understanding output structure
- Common command reference
- Quick troubleshooting fixes

**When to use:**
- After installation and data preparation
- Learning basic workflows
- Testing the framework
- Quick reference for common commands

**Time required:** 5-10 minutes for first run, 5 minutes for subsequent reference

---

## Recommended Learning Path

Follow these guides in order for the smoothest experience:

### Step 1: Installation (5-15 minutes)

Start here to set up your environment:

```bash
# Clone repository
git clone <repository-url>
cd gui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} ready!')"
```

**Read:** [installation.md](installation.md) for complete details

---

### Step 2: Data Preparation (10-20 minutes)

Learn the mandatory data organization structure:

**Required structure:**
```
data/your_dataset/
├── raw/                    # All images organized by class
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── classN/
└── splits/                 # Generated index files (created by ml-split)
    ├── fold_0_train.txt
    ├── fold_0_val.txt
    ├── fold_0_test.txt
    └── ...
```

**Generate cross-validation splits:**
```bash
ml-split \
  --raw_data data/your_dataset/raw \
  --output data/your_dataset/splits \
  --folds 5 \
  --ratio 0.7 0.15 0.15 \
  --seed 42
```

**Read:** [data-preparation.md](data-preparation.md) for complete details

**Critical:** The framework uses an index-based cross-validation system to avoid data duplication. Understanding this system is essential before training.

---

### Step 3: Quick Start (5-10 minutes)

Train your first model:

**Using example dataset:**
```bash
# Quick test run (3 epochs)
ml-train --fold 0 --num_epochs 3

# View results in TensorBoard
tensorboard --logdir runs/
```

**Using your own dataset:**
```bash
# Update ml_src/config.yaml with your dataset info
# Then train
ml-train --fold 0
```

**Run inference:**
```bash
ml-inference --run_dir runs/hymenoptera_base_fold_0 --checkpoint best.pt
```

**Read:** [quick-start.md](quick-start.md) for complete details

---

## Key Concepts

Before diving into the guides, familiarize yourself with these core concepts:

### Index-Based Cross-Validation

The framework uses **index files** instead of duplicating data:

**Benefits:**
- No data duplication (single copy in `raw/`)
- Reproducible splits via fixed seeds
- Version control friendly (lightweight text files)
- Easy to regenerate with different parameters

**How it works:**
1. Store all images once in `data/your_dataset/raw/class_name/`
2. Generate index files that reference images: `fold_0_train.txt`, `fold_0_val.txt`, etc.
3. Each index file contains paths like `raw/class1/img1.jpg`
4. Data loaders read images based on index file contents

**Learn more:** [data-preparation.md](data-preparation.md)

---

### Configuration System

YAML-based configuration with CLI overrides:

**Base configuration:** `ml_src/config.yaml`
```yaml
data:
  dataset_name: 'my_dataset'
  data_dir: 'data/my_dataset'
  fold: 0

model:
  architecture: 'resnet18'
  num_classes: 3

training:
  batch_size: 32
  num_epochs: 50
  lr: 0.01
```

**CLI overrides:**
```bash
ml-train --batch_size 64 --lr 0.001 --num_epochs 100
```

**Learn more:** [Configuration Documentation](../configuration/README.md)

---

### Run Directory Structure

Training outputs are organized in automatically named directories:

```
runs/
└── dataset_base_fold_0/          # Auto-generated name
    ├── config.yaml               # Saved configuration
    ├── summary.txt               # Training summary
    ├── weights/
    │   ├── best.pt              # Best model (highest val accuracy)
    │   └── last.pt              # Latest checkpoint (for resuming)
    ├── logs/
    │   ├── train.log            # Detailed training log
    │   ├── classification_report_train.txt
    │   └── classification_report_val.txt
    └── tensorboard/             # TensorBoard logs
        └── events.out.tfevents.*
```

**Key files:**
- `best.pt` - Use for inference and deployment
- `last.pt` - Use to resume interrupted training
- `summary.txt` - Quick overview of results
- `train.log` - Detailed training information

**Learn more:** [quick-start.md](quick-start.md)

---

### CLI Tools

The framework provides four command-line tools:

| Command | Purpose | Example |
|---------|---------|---------|
| `ml-split` | Generate cross-validation splits | `ml-split --raw_data data/my_dataset/raw --output data/my_dataset/splits` |
| `ml-train` | Train a model | `ml-train --fold 0 --batch_size 32` |
| `ml-inference` | Run evaluation on test set | `ml-inference --run_dir runs/base --checkpoint best.pt` |
| `ml-visualise` | Visualize datasets/predictions | `ml-visualise --mode launch --run_dir runs/base` |

**Learn more:** [CLI Overrides](../configuration/cli-overrides.md)

---

## Complete Workflow Example

Here's a complete workflow from installation to trained model:

### 1. Setup Environment

```bash
# Clone and enter directory
git clone <repository-url>
cd gui

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Verify installation
python -c "import torch; print('Ready!')"
```

### 2. Organize Dataset

```bash
# Create directory structure
mkdir -p data/my_dataset/raw/class1
mkdir -p data/my_dataset/raw/class2
mkdir -p data/my_dataset/raw/class3

# Move images to appropriate folders
mv /path/to/class1_images/* data/my_dataset/raw/class1/
mv /path/to/class2_images/* data/my_dataset/raw/class2/
mv /path/to/class3_images/* data/my_dataset/raw/class3/

# Generate cross-validation splits
ml-split \
  --raw_data data/my_dataset/raw \
  --output data/my_dataset/splits \
  --folds 5
```

### 3. Configure Training

Edit `ml_src/config.yaml`:
```yaml
data:
  dataset_name: 'my_dataset'
  data_dir: 'data/my_dataset'
  fold: 0

model:
  num_classes: 3  # Match number of class folders
```

### 4. Train Model

```bash
# Quick test (3 epochs)
ml-train --fold 0 --num_epochs 3

# Full training (50 epochs)
ml-train --fold 0 --num_epochs 50

# Monitor in TensorBoard
tensorboard --logdir runs/
```

### 5. Evaluate Results

```bash
# Run inference on test set
ml-inference --run_dir runs/my_dataset_base_fold_0 --checkpoint best.pt

# View summary
cat runs/my_dataset_base_fold_0/summary.txt

# Check training log
cat runs/my_dataset_base_fold_0/logs/train.log
```

### 6. Cross-Validation (Optional)

```bash
# Train all folds
for fold in {0..4}; do
  ml-train --fold $fold --batch_size 32 --lr 0.01
done

# Evaluate all folds
for fold in {0..4}; do
  ml-inference --run_dir runs/my_dataset_base_fold_$fold --checkpoint best.pt
done
```

---

## Common Pitfalls for New Users

### Pitfall 1: Skipping Data Preparation

**Problem:** Jumping straight to training without understanding data organization

**Solution:** Always read and follow [data-preparation.md](data-preparation.md) first. The framework requires a specific directory structure and will not work otherwise.

---

### Pitfall 2: Incorrect Directory Structure

**Problem:** Organizing data differently than required

**Solution:** Your dataset MUST have this exact structure:
```
data/your_dataset/
├── raw/              # All images here
│   ├── class1/
│   ├── class2/
│   └── classN/
└── splits/           # Generated by ml-split
```

---

### Pitfall 3: Mismatched num_classes

**Problem:** Number of class folders doesn't match `model.num_classes` in config

**Solution:** Count your class folders and update config:
```yaml
model:
  num_classes: 3  # Must equal number of folders in raw/
```

---

### Pitfall 4: Not Generating Splits

**Problem:** Trying to train without running `ml-split` first

**Solution:** Always generate splits before training:
```bash
ml-split --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 5
```

---

### Pitfall 5: Ignoring GPU Setup

**Problem:** Training is extremely slow on CPU when GPU is available

**Solution:** Verify CUDA availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If False but you have GPU, see [installation.md](installation.md) for GPU setup.

---

## Quick Reference

### Essential Commands

```bash
# === INSTALLATION ===
pip install -e .                              # Install framework
python -c "import torch; print('Ready!')"     # Verify installation

# === DATA PREPARATION ===
ml-split --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 5

# === TRAINING ===
ml-train --fold 0                             # Basic training
ml-train --fold 0 --num_epochs 50             # More epochs
ml-train --fold 0 --batch_size 32 --lr 0.01   # Custom hyperparameters

# === INFERENCE ===
ml-inference --run_dir runs/my_dataset_base_fold_0 --checkpoint best.pt

# === MONITORING ===
tensorboard --logdir runs/                    # View all runs
tail -f runs/my_dataset_base_fold_0/logs/train.log  # Watch training
```

### Directory Structure Template

```
project/
├── data/
│   └── my_dataset/
│       ├── raw/              # You create this
│       │   ├── class1/
│       │   ├── class2/
│       │   └── classN/
│       └── splits/           # ml-split creates this
│           ├── fold_0_train.txt
│           ├── fold_0_val.txt
│           ├── fold_0_test.txt
│           └── ...
├── ml_src/
│   └── config.yaml           # You edit this
└── runs/                     # Training creates this
    └── my_dataset_base_fold_0/
        ├── config.yaml
        ├── summary.txt
        └── weights/
            ├── best.pt
            └── last.pt
```

---

## Next Steps After Getting Started

Once you've completed these guides, explore:

### Configuration

Learn about all available options:
- **[Configuration Overview](../configuration/README.md)** - Complete config system
- **[Model Configuration](../configuration/models.md)** - All available architectures
- **[Training Configuration](../configuration/training.md)** - Batch size, epochs, device
- **[Optimizer & Scheduler](../configuration/optimizer-scheduler.md)** - Learning rate tuning

### User Guides

Practical workflows for common tasks:
- **[Training Guide](../user-guides/training.md)** - Advanced training workflows
- **[Inference Guide](../user-guides/inference.md)** - Detailed evaluation
- **[Monitoring Guide](../user-guides/monitoring.md)** - TensorBoard and logging
- **[Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md)** - Systematic search

### Architecture

Understand how the framework works:
- **[System Overview](../architecture/README.md)** - High-level architecture
- **[Data Flow](../architecture/data-flow.md)** - How data moves through system
- **[ML Source Modules](../architecture/ml-src-modules.md)** - Core components

### Development

Extend and customize the framework:
- **[Adding Models](../development/adding-models.md)** - Custom architectures
- **[Adding Transforms](../development/adding-transforms.md)** - New augmentations
- **[Extending Framework](../development/extending-framework.md)** - General patterns

---

## Troubleshooting

Having issues? Check these resources:

### Installation Problems
- **[Installation Guide - Troubleshooting](installation.md#troubleshooting-installation)** - Common installation errors
- Check Python version: `python --version` (need 3.8+)
- Check PyTorch: `python -c "import torch; print(torch.__version__)"`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Data Problems
- **[Data Preparation - Troubleshooting](data-preparation.md#troubleshooting)** - Data-related errors
- Verify structure: `ls -R data/my_dataset/`
- Check splits exist: `ls data/my_dataset/splits/`
- Count images: `find data/my_dataset/raw -type f | wc -l`

### Training Problems
- **[Quick Start - Troubleshooting](quick-start.md#troubleshooting-quick-fixes)** - Common training issues
- Out of memory: Use `--batch_size 8` or `--device cpu`
- CUDA not available: Use `--device cpu`
- Slow training: Increase `--num_workers` or `--batch_size`

### General Resources
- **[Troubleshooting Guide](../reference/troubleshooting.md)** - Comprehensive troubleshooting
- **[FAQ](../reference/faq.md)** - Frequently asked questions
- **[Best Practices](../reference/best-practices.md)** - Tips and conventions

---

## Getting Help

### Documentation Resources

1. **Quick Reference:** Use this README for overview and navigation
2. **Detailed Guides:** Read individual guide files for specific topics
3. **Examples:** See [Configuration Examples](../configuration/examples.md)
4. **API Reference:** Check [Architecture Documentation](../architecture/)

### System Information

When reporting issues, always include:

```bash
python -c "
import sys, torch, torchvision
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### External Resources

- **[PyTorch Documentation](https://pytorch.org/docs/)** - Official PyTorch docs
- **[PyTorch Tutorials](https://pytorch.org/tutorials/)** - Learning resources
- **[CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)** - GPU setup

---

## Summary

**This Getting Started section includes:**

1. **[Installation Guide](installation.md)** - Complete setup with verification
2. **[Data Preparation Guide](data-preparation.md)** - Critical data organization (mandatory reading)
3. **[Quick Start Guide](quick-start.md)** - Train your first model in 5 minutes

**Recommended order:**
Installation → Data Preparation → Quick Start

**Time investment:**
- Installation: 5-15 minutes
- Data Preparation: 10-20 minutes
- Quick Start: 5-10 minutes
- **Total: 20-45 minutes**

**After completing these guides, you will:**
- ✅ Have a working installation
- ✅ Understand the index-based cross-validation system
- ✅ Know how to organize datasets correctly
- ✅ Be able to train models and monitor results
- ✅ Understand basic CLI usage
- ✅ Know where to find more information

---

## Navigation

- **[← Back to Main Documentation](../README.md)** - Documentation home
- **[Installation Guide →](installation.md)** - Start here
- **[Data Preparation Guide →](data-preparation.md)** - Critical reading
- **[Quick Start Guide →](quick-start.md)** - Train your first model

---

**Ready to begin?** Start with the **[Installation Guide](installation.md)** →

**Have data ready?** Jump to **[Data Preparation Guide](data-preparation.md)** →

**Already set up?** Go to **[Quick Start Guide](quick-start.md)** →
