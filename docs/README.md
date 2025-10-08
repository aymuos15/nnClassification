# PyTorch Image Classifier - Documentation

Complete documentation for the PyTorch Image Classification framework.

## Quick Links

- **[Quick Start](getting-started/quick-start.md)** - Get training in 5 minutes
- **[Configuration Reference](configuration/README.md)** - All configuration options
- **[Training Guide](user-guides/training.md)** - Complete training workflows

---

## Documentation Structure

### 🚀 Getting Started

New to the framework? Start here:

1. **[Installation](getting-started/installation.md)** - Setup and dependencies
2. **[Data Preparation](getting-started/data-preparation.md)** - Organize your dataset ⚠️ CRITICAL
3. **[Quick Start](getting-started/quick-start.md)** - Train your first model

### ⚙️ Configuration

Complete configuration reference:

- **[Overview](configuration/README.md)** - Config system explained
- **[Reproducibility](configuration/reproducibility.md)** - Seed & determinism
- **[Data](configuration/data.md)** - Data loading parameters
- **[Training](configuration/training.md)** - Batch size, epochs, device
- **[Optimizer & Scheduler](configuration/optimizer-scheduler.md)** - Learning rate, momentum
- **[Models](configuration/models.md)** - All model architectures
- **[Transforms](configuration/transforms.md)** - Data preprocessing
- **[CLI Overrides](configuration/cli-overrides.md)** - Command-line usage
- **[Examples](configuration/examples.md)** - 10 complete configs

### 📖 User Guides

Practical workflows and how-tos:

- **[Training](user-guides/training.md)** - Training workflows & trainer selection
- **[Advanced Training](user-guides/advanced-training.md)** - Mixed precision, multi-GPU, DP
- **[Inference](user-guides/inference.md)** - Running evaluation & inference strategies
- **[Test-Time Augmentation](user-guides/test-time-augmentation.md)** - TTA for improved accuracy
- **[Ensemble Inference](user-guides/ensemble-inference.md)** - Combining multiple models
- **[Resuming Training](user-guides/resuming-training.md)** - Continue interrupted runs
- **[Monitoring](user-guides/monitoring.md)** - TensorBoard & logging
- **[Hyperparameter Tuning](user-guides/hyperparameter-tuning.md)** - Systematic search
- **[Model Export](user-guides/model-export.md)** - ONNX export & deployment
- **[Learning Rate Finder](user-guides/lr-finder.md)** - Find optimal learning rate

### 🏗️ Architecture

Deep dive into the codebase:

- **[Overview](architecture/README.md)** - System architecture
- **[Entry Points](architecture/entry-points.md)** - train.py & inference.py
- **[ML Source Modules](architecture/ml-src-modules.md)** - All ml_src components
- **[Data Flow](architecture/data-flow.md)** - How data moves through system
- **[Design Decisions](architecture/design-decisions.md)** - Why it's built this way

### 🛠️ Development

Extend and customize:

- **[Adding Models](development/adding-models.md)** - Custom architectures
- **[Adding Transforms](development/adding-transforms.md)** - New augmentations
- **[Adding Optimizers](development/adding-optimizers.md)** - New optimization methods
- **[Adding Metrics](development/adding-metrics.md)** - Custom evaluation metrics
- **[Extending Framework](development/extending-framework.md)** - General patterns

### 📚 Reference

Quick lookups and troubleshooting:

- **[Best Practices](reference/best-practices.md)** - Tips and conventions
- **[Troubleshooting](reference/troubleshooting.md)** - Common issues
- **[Performance Tuning](reference/performance-tuning.md)** - Speed & memory optimization
- **[FAQ](reference/faq.md)** - Frequently asked questions

---

## Common Tasks

### Train a Model
```bash
ml-train --config configs/my_dataset_config.yaml --batch_size 32 --lr 0.01 --num_epochs 50
```
See: [Training Guide](user-guides/training.md)

### Resume Training
```bash
ml-train --config configs/my_dataset_config.yaml --resume runs/hymenoptera_base_fold_0/last.pt
```
See: [Resuming Training](user-guides/resuming-training.md)

### Run Inference
```bash
# Standard inference
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt

# TTA for improved accuracy
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt --tta

# Ensemble multiple models
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt
```
See: [Inference Guide](user-guides/inference.md), [TTA Guide](user-guides/test-time-augmentation.md), [Ensemble Guide](user-guides/ensemble-inference.md)

### Monitor Training
```bash
tensorboard --logdir runs/
# Or use ml-visualise
ml-visualise --mode launch --run_dir runs/hymenoptera_base_fold_0
```
See: [Monitoring Guide](user-guides/monitoring.md)

### Visualize Dataset
```bash
ml-visualise --mode samples --run_dir runs/hymenoptera_base_fold_0 --split train
```
See: [Monitoring Guide](user-guides/monitoring.md)

### Visualize Predictions
```bash
ml-visualise --mode predictions --run_dir runs/hymenoptera_base_fold_0 --split val
```
See: [Visualization Reference](reference/visualization.md)

### Change Model
```yaml
# In ml_src/config_template.yaml
model:
  architecture: 'efficientnet_b0'
```
See: [Model Configuration](configuration/models.md)

---

## Documentation Navigation Tips

1. **New users:** Start with [Getting Started](getting-started/)
2. **Quick reference:** Check [Configuration](configuration/)
3. **Workflows:** See [User Guides](user-guides/)
4. **Understanding code:** Read [Architecture](architecture/)
5. **Customization:** Explore [Development](development/)
6. **Problems:** Visit [Troubleshooting](reference/troubleshooting.md)

---

## Key Concepts

### Configuration System
YAML-based with CLI overrides:
```bash
ml-train --config custom.yaml --lr 0.01
```
[Learn more →](configuration/README.md)

### Data Organization
**Mandatory structure:**
```
data/your_dataset/
├── raw/                    # All images organized by class
│   ├── class1/
│   ├── class2/
│   └── classN/
└── splits/                 # Generated by ml-split
    ├── test.txt
    ├── fold_0_train.txt
    ├── fold_0_val.txt
    └── ...
```
**Index-based cross-validation:** No data duplication - all images stored once in `raw/`, referenced by `.txt` index files.

[Learn more →](getting-started/data-preparation.md)

### Checkpointing
Two checkpoints per run:
- `best.pt` - Highest validation accuracy (for deployment)
- `last.pt` - Latest epoch (for resuming)

[Learn more →](architecture/ml-src-modules.md#checkpointingpy)

### Run Organization
Automatic directory naming based on dataset, fold, and hyperparameters:
```
runs/
├── hymenoptera_base_fold_0/
├── hymenoptera_batch_32_lr_0.01_fold_0/
└── my_dataset_epochs_50_lr_0.001_fold_1/
```
[Learn more →](configuration/cli-overrides.md)

---

## Support

- **Documentation:** You're reading it!
- **Examples:** See [Configuration Examples](configuration/examples.md)
- **Troubleshooting:** Check [Troubleshooting Guide](reference/troubleshooting.md)
- **FAQ:** See [Frequently Asked Questions](reference/faq.md)

---

## Contributing to Documentation

Found an issue or want to improve the docs?

1. Documentation source: `/docs/` directory
2. Each section has its own folder
3. Written in Markdown
4. Feel free to submit improvements!

---

**Happy training!** 🚀

Start with the [Quick Start Guide](getting-started/quick-start.md) →
