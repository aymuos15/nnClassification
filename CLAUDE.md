# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-ready PyTorch image classification framework with flexible architecture support, index-based cross-validation, and comprehensive experiment tracking. Built as an installable Python package with CLI tools.

## Core Development Commands

### Installation & Setup
```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_architectures.py

# Run tests with coverage
pytest --cov=ml_src
```

### Code Quality
```bash
# Format and lint code using ruff
ruff format .
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Documentation
```bash
# Build documentation locally
mkdocs build

# Serve documentation with live reload
mkdocs serve

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### CLI Tools (after installation)
```bash
# Generate dataset configuration
ml-init-config data/my_dataset

# Generate configuration with hyperparameter search (optional)
ml-init-config data/my_dataset --optuna

# Create cross-validation splits
ml-split --raw_data data/my_dataset/raw --folds 5

# Find optimal learning rate (optional, before training)
ml-lr-finder --config configs/my_dataset_config.yaml
ml-lr-finder --config configs/my_dataset_config.yaml --start_lr 1e-7 --end_lr 1 --num_iter 200
ml-lr-finder --config configs/my_dataset_config.yaml --diverge_threshold 2.0  # More sensitive early stopping

# Train model
ml-train --config configs/my_dataset_config.yaml

# Run hyperparameter search (requires: uv pip install -e ".[optuna]")
ml-search --config configs/my_dataset_config.yaml --n-trials 50

# Run inference
ml-inference --checkpoint_path runs/my_run/weights/best.pt

# Export model to ONNX for deployment
ml-export --checkpoint runs/my_run/weights/best.pt
ml-export --checkpoint runs/my_run/weights/best.pt --validate --benchmark

# Visualize with TensorBoard
ml-visualise --mode launch --run_dir runs/my_run

# Visualize hyperparameter search results
ml-visualise --mode search --study-name my_study
```

## Architecture

### Package Structure

```
ml_src/
├── cli/                    # CLI entry points (installed as console scripts)
│   ├── init_config.py      # ml-init-config: Generate dataset configs
│   ├── splitting.py        # ml-split: CV split generator
│   ├── lr_finder.py        # ml-lr-finder: Learning rate range test
│   ├── train.py            # ml-train: Main training orchestrator (uses get_trainer)
│   ├── inference.py        # ml-inference: Test/inference runner
│   ├── export.py           # ml-export: ONNX model export
│   ├── visualise.py        # ml-visualise: TensorBoard + search visualization
│   └── search.py           # ml-search: Hyperparameter optimization with Optuna
│
└── core/                   # Reusable ML components (no CLI dependencies)
    ├── dataset.py          # IndexedImageDataset (index-based loading)
    ├── loader.py           # DataLoader creation
    ├── network/            # Model architectures
    │   ├── __init__.py     # get_model() API (routes base vs custom)
    │   ├── base.py         # Torchvision models (ResNet, EfficientNet, etc.)
    │   └── custom.py       # Custom architectures (SimpleCNN, TinyNet)
    ├── trainers/           # Specialized trainer implementations
    │   ├── __init__.py     # get_trainer() factory function
    │   ├── base.py         # BaseTrainer abstract class (with Optuna trial support)
    │   ├── standard.py     # StandardTrainer: Traditional PyTorch training
    │   ├── mixed_precision.py  # MixedPrecisionTrainer: PyTorch AMP
    │   ├── accelerate.py   # AccelerateTrainer: Multi-GPU with Accelerate
    │   └── differential_privacy.py  # DPTrainer: Opacus differential privacy
    ├── loss.py             # Loss functions
    ├── optimizer.py        # Optimizers and schedulers
    ├── lr_finder.py        # Learning rate finder implementation
    ├── export.py           # ONNX export utilities
    ├── search.py           # Hyperparameter search utilities (Optuna wrapper)
    ├── test.py             # Evaluation/testing
    ├── metrics/            # Comprehensive metrics suite
    │   ├── __init__.py     # Metrics API
    │   ├── classification.py  # Classification-specific metrics
    │   ├── utils.py        # Metric utilities
    │   ├── visualization.py   # Metric visualization
    │   └── onnx_validation.py # ONNX model validation and benchmarking
    ├── checkpointing.py    # Checkpoint save/load, summaries
    ├── seeding.py          # Reproducibility (seed setting)
    └── visual/             # Visualization utilities
        ├── tensorboard.py  # TensorBoard dataset/prediction visualization
        ├── search.py       # Optuna study visualization
        └── server.py       # TensorBoard server management
```

### Key Design Principles

1. **CLI vs Core Separation**: `cli/` orchestrates workflows, `core/` contains reusable components. Never import CLI modules from core modules.

2. **Index-based Cross-Validation**: Dataset uses text files (e.g., `fold_0_train.txt`) containing relative paths to images in `data_dir/raw/`. This avoids duplicating data across folds.

3. **Configuration System**:
   - Base config template: `ml_src/config_template.yaml`
   - Dataset-specific configs: `configs/{dataset_name}_config.yaml` (auto-generated by `ml-init-config`)
   - CLI overrides: Use `--batch_size`, `--lr`, `--fold`, etc. to override config values

4. **Model Loading**: `get_model(config, device)` routes to either:
   - `base.py` for torchvision models (requires `model.type: 'base'`, `model.architecture: 'resnet18'`)
   - `custom.py` for custom models (requires `model.type: 'custom'`, `model.custom_architecture: 'simple_cnn'`)

5. **Trainer Selection**: `get_trainer(config, ...)` routes to specialized trainers:
   - `standard.py` for traditional PyTorch training (default)
   - `mixed_precision.py` for PyTorch AMP (requires `training.trainer_type: 'mixed_precision'`)
   - `accelerate.py` for multi-GPU/distributed (requires `training.trainer_type: 'accelerate'`)
   - `differential_privacy.py` for privacy-preserving training (requires `training.trainer_type: 'dp'`)

6. **Run Directory Naming**: Auto-generated as `runs/{dataset_name}_{overrides}_fold_{N}/` where overrides include non-default parameters (e.g., `hymenoptera_batch_32_lr_0.01_fold_0`)

### Data Flow

1. **Dataset Creation** (`core/dataset.py`):
   - `get_datasets(config)` → reads index files from `data_dir/splits/`
   - Creates `IndexedImageDataset` for train/val/test splits
   - Test set is shared across all folds

2. **Training** (`cli/train.py` → `core/trainers/`):
   - Loads config, overrides with CLI args
   - Selects appropriate trainer via `get_trainer()` factory
   - Creates run directory with auto-generated name
   - Sets up logging (console + file + TensorBoard)
   - Trains model using selected trainer, saves best/last checkpoints
   - Auto-runs test evaluation after training completes

3. **Checkpointing** (`core/checkpointing.py`):
   - `save_checkpoint()`: Saves full state (model, optimizer, scheduler, epoch, metrics history)
   - `load_checkpoint()`: Restores complete training state for resumption
   - Best model saved to `weights/best.pt`, last to `weights/last.pt`

## Critical Implementation Details

### Trainer Types

All trainers inherit from `BaseTrainer` and are selected via `config['training']['trainer_type']`.

**Available trainers:**
- `standard` (default) - Traditional PyTorch training
- `mixed_precision` - PyTorch AMP for 2-3x speedup
- `accelerate` - Multi-GPU/distributed with Hugging Face Accelerate
- `dp` - Differential privacy with Opacus

**Factory function:** `get_trainer(config, model, dataloaders, criterion, optimizer, scheduler, device, run_dir, num_classes)`

**Example config:**
```yaml
training:
  trainer_type: 'mixed_precision'  # or 'standard', 'accelerate', 'dp'
  amp_dtype: 'float16'  # for mixed_precision
  num_epochs: 50
```

**When to use each:**
- `standard`: Beginners, CPU, simple workflows
- `mixed_precision`: Single modern GPU (most common for production)
- `accelerate`: Multiple GPUs, distributed training
- `dp`: Privacy-sensitive data requiring formal guarantees

### Adding New Models

**Torchvision models** (e.g., ConvNeXt):
1. No code changes needed
2. Update config: `model.architecture: 'convnext_tiny'`

**Custom models**:
1. Add architecture to `ml_src/core/network/custom.py` in `MODEL_REGISTRY`
2. Ensure it accepts `num_classes`, `dropout` parameters
3. Update config: `model.type: 'custom'`, `model.custom_architecture: 'your_model'`

### Dataset Requirements

Mandatory directory structure:
```
data/{dataset_name}/
├── raw/                    # Original images (never modified)
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class2/
└── splits/                 # Generated by ml-split
    ├── test.txt            # Shared test set (all folds)
    ├── fold_0_train.txt
    ├── fold_0_val.txt
    ├── fold_1_train.txt
    └── ...
```

Index files contain relative paths: `raw/class1/img1.jpg`

### Configuration Override Behavior

CLI arguments override config file:
```bash
# Config has batch_size: 4, lr: 0.001
ml-train --config config.yaml --batch_size 32 --lr 0.01
# Results in: batch_size=32, lr=0.01, run_name includes "batch_32_lr_0.01"
```

Override-based parameters are added to run directory name for easy tracking.

### Logging System

- **loguru** for structured logging with color-coded levels
- Console output: Color-coded, formatted timestamps
- File output: `runs/{run_name}/logs/train.log`
- TensorBoard: `runs/{run_name}/tensorboard/` (metrics, confusion matrices, classification reports)

### Reproducibility

Set in config:
```yaml
seed: 42
deterministic: false  # true for full reproducibility (slower)
```

`core/seeding.py` handles:
- Python random seed
- NumPy seed
- PyTorch seed (CPU + CUDA)
- Deterministic algorithms (if enabled)

## Testing Guidelines

- Test files in `tests/` follow `test_*.py` naming
- Use `pytest` fixtures from `tests/conftest.py`
- Current coverage: Model architecture loading (`test_architectures.py`)
- When adding models, add corresponding tests to verify they load correctly

## Common Workflows

### Choosing a Trainer

**Standard Training** (default):
```yaml
training:
  trainer_type: 'standard'
```
No special requirements, works on CPU or GPU. Best for beginners and simple workflows.

**Mixed Precision** (2-3x faster on GPU):
```yaml
training:
  trainer_type: 'mixed_precision'
  amp_dtype: 'float16'  # or 'bfloat16' for newer GPUs (A100, RTX 40)
```
Requires modern NVIDIA GPU (Volta/Turing/Ampere or newer). Provides significant speedup with minimal accuracy impact.

**Multi-GPU/Distributed**:
```bash
# One-time setup
uv pip install accelerate
accelerate config

# Launch training
accelerate launch ml-train --config configs/my_config.yaml
```
Config:
```yaml
training:
  trainer_type: 'accelerate'
  batch_size: 32  # Per-device batch size
  gradient_accumulation_steps: 2  # Optional: for larger effective batches
```
Requires `accelerate` package. Supports multi-GPU, distributed training, and TPU.

**Differential Privacy**:
```bash
uv pip install opacus
```
Config:
```yaml
training:
  trainer_type: 'dp'
  dp:
    noise_multiplier: 1.1      # Privacy-accuracy tradeoff
    max_grad_norm: 1.0         # Gradient clipping
    target_epsilon: 3.0        # Privacy budget (lower = stronger privacy)
    target_delta: 1e-5         # Privacy parameter
```
For privacy-sensitive data (medical, financial). Slower and requires hyperparameter tuning.

**Decision guide:**
- Need privacy guarantees? → `dp`
- Have multiple GPUs? → `accelerate`
- Have single GPU? → `mixed_precision` (recommended)
- CPU only or learning? → `standard`

See [Advanced Training Guide](docs/user-guides/advanced-training.md) for detailed documentation.

### Training a New Dataset
1. Organize data in `data/{name}/raw/class1/`, `data/{name}/raw/class2/`, etc.
2. Run `ml-split --raw_data data/{name}/raw --folds 5`
3. Run `ml-init-config data/{name}` → creates `configs/{name}_config.yaml`
4. Edit config if needed (model architecture, hyperparameters, trainer type)
5. (Optional) Run `ml-lr-finder --config configs/{name}_config.yaml` to find optimal LR
6. Run `ml-train --config configs/{name}_config.yaml`

### Resuming Training
```bash
ml-train --config configs/my_config.yaml --resume runs/my_run/weights/last.pt
```
Restores: model weights, optimizer state, scheduler state, epoch counter, best accuracy, loss/accuracy history

### Cross-Validation
Train each fold separately:
```bash
ml-train --config configs/my_config.yaml --fold 0
ml-train --config configs/my_config.yaml --fold 1
ml-train --config configs/my_config.yaml --fold 2
```
Test set is identical across all folds; only train/val splits differ.

### Learning Rate Finder

Find optimal learning rate before training using LR range test:

```bash
# Basic usage
ml-lr-finder --config configs/my_config.yaml

# Custom LR range and iterations
ml-lr-finder --config configs/my_config.yaml --start_lr 1e-7 --end_lr 1 --num_iter 200

# Use specific fold
ml-lr-finder --config configs/my_config.yaml --fold 2

# Adjust smoothing (lower beta = less smoothing)
ml-lr-finder --config configs/my_config.yaml --beta 0.95

# Control early stopping sensitivity (default: 4.0)
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 2.0  # Stops earlier
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 6.0  # Allows more loss increase
```

**Output:**
- `runs/lr_finder_TIMESTAMP/lr_plot.png`: LR vs Loss curve with suggested LR marked
- `runs/lr_finder_TIMESTAMP/results.json`: Learning rates, losses, and suggested LR
- `runs/lr_finder_TIMESTAMP/lr_finder.log`: Detailed logs

**Usage tips:**
- Suggested LR is typically 1/10th of the LR at steepest descent point
- Run before training to find good initial learning rate
- The tool tests LRs from start_lr to end_lr over num_iter iterations
- Loss is smoothed using exponential moving average (beta parameter)
- Early stopping triggers when `loss > diverge_threshold × min_loss` (default: 4.0)

### Model Export

Export trained models to ONNX format for deployment:

```bash
# Basic export
ml-export --checkpoint runs/my_run/weights/best.pt
# Creates: runs/my_run/weights/best.onnx

# Export with basic validation
ml-export --checkpoint runs/my_run/weights/best.pt --validate

# Export with comprehensive validation (uses test loader)
ml-export --checkpoint runs/my_run/weights/best.pt --comprehensive-validate

# Export with benchmarking
ml-export --checkpoint runs/my_run/weights/best.pt --benchmark

# Batch export multiple checkpoints
ml-export --checkpoint "runs/*/weights/best.pt" --validate

# Custom output path and opset version
ml-export --checkpoint runs/my_run/weights/best.pt --output model.onnx --opset 14

# Specify input size (auto-detected if not provided)
ml-export --checkpoint runs/my_run/weights/best.pt --input_size 224 224
```

**Features:**
- **Basic validation**: Checks model can be loaded and runs forward pass
- **Comprehensive validation**: Compares PyTorch vs ONNX outputs on test data
- **Benchmarking**: Measures inference speed (PyTorch vs ONNX)
- **Batch export**: Use glob patterns to export multiple checkpoints
- **Auto-detection**: Input size and opset version auto-detected from checkpoint

**Output:**
- `{checkpoint_name}.onnx`: Exported ONNX model
- `validation_report.json`: Validation results (if --validate or --comprehensive-validate used)
- Console logs with detailed validation metrics and benchmarks

### Hyperparameter Search (Optional)

**Installation:**
```bash
uv pip install -e ".[optuna]"
```

**Generate config with search space:**
```bash
ml-init-config data/my_dataset --optuna
```

**Run optimization:**
```bash
# Run hyperparameter search
ml-search --config configs/my_config.yaml --n-trials 50

# Resume existing study
ml-search --config configs/my_config.yaml --resume

# Override study name
ml-search --config configs/my_config.yaml --study-name custom_study
```

**Visualize results:**
```bash
# Generate all plots
ml-visualise --mode search --study-name my_study

# Generate specific plot types
ml-visualise --mode search --study-name my_study --plot-type optimization_history
ml-visualise --mode search --study-name my_study --plot-type param_importances
ml-visualise --mode search --study-name my_study --plot-type contour --params lr batch_size
```

**Train with best hyperparameters:**
```bash
ml-train --config runs/optuna_studies/my_study/best_config.yaml
```

**Features:**
- **Opt-in**: Requires `--optuna` flag in `ml-init-config` (disabled by default)
- **Search space**: Supports categorical, uniform, loguniform, int parameters
- **Samplers**: TPE (default), Random, CMA-ES, Grid
- **Pruning**: MedianPruner, PercentilePruner, HyperbandPruner for early trial termination
- **Storage**: SQLite (default), PostgreSQL, MySQL for distributed optimization
- **Visualization**: Interactive Plotly plots (optimization history, parameter importances, etc.)
- **Hybrid approach**: Trial training logs in TensorBoard, study-level plots in HTML

**Configuration example:**
```yaml
search:
  study_name: 'my_optimization'
  storage: 'sqlite:///optuna_studies.db'
  n_trials: 50
  direction: 'maximize'  # or 'minimize'
  metric: 'val_acc'      # or 'val_loss'
  
  sampler:
    type: 'TPESampler'
    n_startup_trials: 10
  
  pruner:
    type: 'MedianPruner'
    n_startup_trials: 5
    n_warmup_steps: 5
  
  search_space:
    optimizer.lr:
      type: 'loguniform'
      low: 1e-5
      high: 1e-1
    
    training.batch_size:
      type: 'categorical'
      choices: [16, 32, 64]
    
    optimizer.momentum:
      type: 'uniform'
      low: 0.8
      high: 0.99
```

See config_template.yaml for full search configuration options.

## Documentation Structure

Comprehensive docs in `docs/`:
- **getting-started/**: Installation, data prep, quick start
- **configuration/**: All config parameters explained
- **user-guides/**: Training, inference, monitoring, tuning
- **architecture/**: System design, data flow, design decisions
- **development/**: Extending framework (models, transforms, optimizers, metrics)
- **reference/**: Best practices, troubleshooting, FAQ

Documentation built with MkDocs Material theme, deployed to GitHub Pages.

## Code Style

- **Formatter**: ruff (line length: 100)
- **Linter**: ruff (pycodestyle, Pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear)
- **Docstrings**: Google-style with examples
- **Type hints**: Not enforced but encouraged for public APIs
- **Import order**: Standard lib → third-party → first-party (`ml_src`)

## Package Management

- Build system: setuptools
- Dependencies defined in `pyproject.toml`
- Console scripts: `ml-init-config`, `ml-split`, `ml-lr-finder`, `ml-train`, `ml-inference`, `ml-export`, `ml-visualise`, `ml-search`
- Core dependencies include: torch, torchvision, matplotlib, onnx, onnxruntime (for export functionality)
- Optional dependencies:
  - Dev: `uv pip install -e ".[dev]"` (pytest, ruff, mkdocs)
  - Differential Privacy: `uv pip install -e ".[dp]"` (opacus)
  - Hyperparameter Search: `uv pip install -e ".[optuna]"` (optuna, plotly, kaleido)

## Git Workflow

- Main branch: `master`
- Runs directory (`runs/`) is gitignored
- User configs (`configs/*.yaml`) are gitignored (template at `ml_src/config_template.yaml`)
- Data directory (`data/`) is gitignored
