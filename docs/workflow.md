# Complete Workflow Guide

This guide shows the complete workflow from dataset preparation to model deployment with all major commands.

---

## 1. Installation

### Basic Installation
```bash
pip install -e .
```
Installs the package in editable mode with core dependencies.

### With Development Tools
```bash
pip install -e ".[dev]"
```
Includes pytest, ruff, and mkdocs for development and testing.

### With Optional Features
```bash
# For differential privacy training
pip install -e ".[dp]"

# For hyperparameter search
pip install -e ".[optuna]"

# Install all optional features
pip install -e ".[dev,dp,optuna]"
```

---

## 2. Dataset Preparation

### Organize Your Data
```bash
# Your dataset should be organized as:
# data/my_dataset/raw/class1/*.jpg
# data/my_dataset/raw/class2/*.jpg
```
Place images in class-named subdirectories under `raw/`.

### Generate Train/Val/Test Splits
```bash
ml-split --raw_data data/my_dataset/raw --folds 5
```
Creates 5-fold cross-validation splits with a shared test set in `data/my_dataset/splits/`.

---

## 3. Configuration

### Generate Base Configuration
```bash
ml-init-config data/my_dataset --yes
```
Auto-generates `configs/my_dataset_config.yaml` with detected dataset info.

### Generate Configuration with Hyperparameter Search (Optional)
```bash
ml-init-config data/my_dataset --optuna --yes
```
Includes pre-configured search space for hyperparameter optimization.

### Edit Configuration (Optional)
```bash
nano configs/my_dataset_config.yaml
```
Customize model architecture, training parameters, or search space.

---

## 4. Training

### Standard Training
```bash
ml-train --config configs/my_dataset_config.yaml
```
Trains model with default settings from config file.

### Training with Overrides
```bash
ml-train --config configs/my_dataset_config.yaml \
  --batch_size 32 \
  --lr 0.01 \
  --num_epochs 50
```
Overrides specific config parameters via command line.

### Training Specific Fold
```bash
ml-train --config configs/my_dataset_config.yaml --fold 0
```
Trains on fold 0 (useful for cross-validation).

### Resume Training from Checkpoint
```bash
ml-train --config configs/my_dataset_config.yaml \
  --resume runs/my_run/weights/last.pt
```
Continues training from saved checkpoint (restores optimizer, scheduler, epoch).

### Mixed Precision Training (GPU)
```bash
# Update config: training.trainer_type: 'mixed_precision'
ml-train --config configs/my_dataset_config.yaml
```
2-3x faster training with minimal accuracy impact on modern GPUs.

### Multi-GPU Training
```bash
# One-time setup
accelerate config

# Launch training
accelerate launch ml-train --config configs/my_dataset_config.yaml
```
Trains on multiple GPUs with automatic data parallelism.

---

## 5. Hyperparameter Search (Optional)

### Run Hyperparameter Optimization
```bash
ml-search --config configs/my_dataset_config.yaml --n-trials 50
```
Searches for best hyperparameters using Optuna (requires config with `search` section).

### Resume Existing Study
```bash
ml-search --config configs/my_dataset_config.yaml --resume
```
Continues optimization from saved study.

### Custom Study Name
```bash
ml-search --config configs/my_dataset_config.yaml \
  --study-name my_custom_study \
  --n-trials 100
```
Runs study with custom name for better organization.

### Train with Best Hyperparameters
```bash
ml-train --config runs/optuna_studies/my_study/best_config.yaml
```
Uses automatically exported best configuration from search.

---

## 6. Monitoring & Visualization

### Launch TensorBoard
```bash
ml-visualise --mode launch --run_dir runs/my_run
```
Opens TensorBoard to view training curves, metrics, and confusion matrices.

### Visualize Dataset Samples
```bash
ml-visualise --mode samples \
  --run_dir runs/my_run \
  --split train \
  --num_images 16
```
Shows sample images from training set with augmentations in TensorBoard.

### Visualize Model Predictions
```bash
ml-visualise --mode predictions \
  --run_dir runs/my_run \
  --split val \
  --checkpoint best.pt \
  --num_images 16
```
Displays model predictions on validation set in TensorBoard.

### Visualize Hyperparameter Search Results
```bash
ml-visualise --mode search --study-name my_study
```
Generates interactive plots showing optimization progress and parameter importance.

### Generate Specific Search Plots
```bash
# Optimization history
ml-visualise --mode search \
  --study-name my_study \
  --plot-type optimization_history

# Parameter importances
ml-visualise --mode search \
  --study-name my_study \
  --plot-type param_importances

# Contour plot for two parameters
ml-visualise --mode search \
  --study-name my_study \
  --plot-type contour \
  --params lr batch_size
```
Creates targeted visualizations for specific aspects of search.

### Clean TensorBoard Logs
```bash
# Clean all runs
ml-visualise --mode clean

# Clean specific run
ml-visualise --mode clean --run_dir runs/my_run
```
Removes TensorBoard log files to save space.

---

## 7. Inference

### Run Inference on Test Set
```bash
ml-inference --checkpoint_path runs/my_run/weights/best.pt
```
Evaluates best model on test set (uses config from checkpoint directory).

### Inference with Custom Config
```bash
ml-inference \
  --checkpoint_path runs/my_run/weights/best.pt \
  --config configs/my_dataset_config.yaml
```
Overrides checkpoint config with custom configuration.

### Inference on Specific Split
```bash
ml-inference \
  --checkpoint_path runs/my_run/weights/best.pt \
  --split val
```
Runs inference on validation set instead of test set.

---

## 8. Cross-Validation Workflow

### Train All Folds
```bash
for fold in {0..4}; do
  ml-train --config configs/my_dataset_config.yaml --fold $fold
done
```
Trains model on all 5 folds for robust evaluation.

### Aggregate Results
```bash
# Results stored in separate run directories:
# runs/my_dataset_fold_0/
# runs/my_dataset_fold_1/
# ...
# runs/my_dataset_fold_4/
```
Each fold produces independent results; manually aggregate metrics.

---

## 9. Common Workflows

### Quick Start: Single Training Run
```bash
# 1. Prepare data splits
ml-split --raw_data data/my_dataset/raw --folds 5

# 2. Generate config
ml-init-config data/my_dataset --yes

# 3. Train model
ml-train --config configs/my_dataset_config.yaml

# 4. View results
ml-visualise --mode launch --run_dir runs/my_dataset_fold_0
```

### Complete Workflow with Hyperparameter Search
```bash
# 1. Install with search support
pip install -e ".[optuna]"

# 2. Prepare data
ml-split --raw_data data/my_dataset/raw --folds 5

# 3. Generate config with search space
ml-init-config data/my_dataset --optuna --yes

# 4. Run hyperparameter search
ml-search --config configs/my_dataset_config.yaml --n-trials 50

# 5. Visualize search results
ml-visualise --mode search --study-name my_dataset_optimization

# 6. Train with best hyperparameters
ml-train --config runs/optuna_studies/my_dataset_optimization/best_config.yaml

# 7. View final results
ml-visualise --mode launch --run_dir runs/my_dataset_*
```

### Production Workflow with Mixed Precision
```bash
# 1. Prepare and configure
ml-split --raw_data data/my_dataset/raw --folds 5
ml-init-config data/my_dataset --yes

# 2. Edit config to enable mixed precision
# Set training.trainer_type: 'mixed_precision'

# 3. Train with GPU acceleration
ml-train --config configs/my_dataset_config.yaml

# 4. Evaluate on test set
ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt

# 5. Monitor results
ml-visualise --mode launch --run_dir runs/my_dataset_fold_0
```

### Cross-Validation with Hyperparameter Search
```bash
# 1. Search on single fold
ml-search --config configs/my_dataset_config.yaml --fold 0 --n-trials 50

# 2. Use best config for all folds
for fold in {0..4}; do
  ml-train --config runs/optuna_studies/my_study/best_config.yaml --fold $fold
done

# 3. Aggregate results manually from each fold's test metrics
```

---

## 10. Useful Tips

### Check Run Directory Structure
```bash
tree runs/my_run
```
View complete structure of training outputs (weights, logs, tensorboard, summaries).

### View Training Logs
```bash
cat runs/my_run/logs/train.log
```
Read detailed training logs with timestamps.

### Check Training Summary
```bash
cat runs/my_run/summary.txt
```
Quick summary of model, dataset, and final metrics.

### List Available Models
```bash
python -c "import torchvision.models as models; print([m for m in dir(models) if not m.startswith('_')])"
```
Shows all available torchvision architectures.

### Monitor GPU Usage During Training
```bash
watch -n 1 nvidia-smi
```
Real-time GPU utilization monitoring.

### Run Quick Test
```bash
# Test with minimal epochs and small batch size
ml-train --config configs/my_dataset_config.yaml \
  --num_epochs 2 \
  --batch_size 8
```
Verify pipeline works before full training run.

---

## File Outputs Reference

### After ml-split
```
data/my_dataset/
└── splits/
    ├── test.txt              # Shared test set
    ├── fold_0_train.txt      # Fold 0 training
    ├── fold_0_val.txt        # Fold 0 validation
    ├── fold_1_train.txt      # Fold 1 training
    └── ...
```

### After ml-init-config
```
configs/
└── my_dataset_config.yaml    # Generated configuration
```

### After ml-train
```
runs/my_dataset_fold_0/
├── config.yaml                # Training configuration
├── summary.txt                # Training summary
├── weights/
│   ├── best.pt                # Best model checkpoint
│   └── last.pt                # Last epoch checkpoint
├── logs/
│   ├── train.log              # Training log
│   ├── classification_report_train.txt
│   ├── classification_report_val.txt
│   └── classification_report_test.txt
└── tensorboard/               # TensorBoard logs
```

### After ml-search
```
runs/optuna_studies/my_study/
├── best_config.yaml           # Best hyperparameters
├── trial_0/                   # First trial outputs
├── trial_1/                   # Second trial outputs
├── ...
└── visualizations/            # Search plots (HTML)
    ├── optimization_history.html
    ├── param_importances.html
    ├── parallel_coordinate.html
    ├── slice.html
    └── intermediate_values.html
```

---

## Troubleshooting Commands

### Check Installation
```bash
pip list | grep ml-classifier
```
Verify package is installed.

### Verify CLI Commands
```bash
ml-train --help
ml-search --help
ml-visualise --help
```
Show available options for each command.

### Test Configuration
```bash
python -c "from ml_src.core.config import load_config; print(load_config('configs/my_config.yaml'))"
```
Verify config file is valid YAML and loads correctly.

### Check CUDA Availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```
Verify GPU is accessible to PyTorch.
