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

# Run inference (standard)
ml-inference --checkpoint_path runs/my_run/weights/best.pt

# Run inference with Test-Time Augmentation (TTA)
ml-inference --checkpoint_path runs/my_run/weights/best.pt --tta
ml-inference --checkpoint_path runs/my_run/weights/best.pt --tta --tta-augmentations horizontal_flip vertical_flip rotate_90 brightness contrast

# Run ensemble inference from multiple folds
ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt runs/fold_2/weights/best.pt

# Combined TTA + Ensemble for maximum performance
ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt --tta

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
│   ├── splitting.py        # ml-split: CV split generator (now supports --federated)
│   ├── lr_finder.py        # ml-lr-finder: Learning rate range test
│   ├── train.py            # ml-train: Main training orchestrator (uses get_trainer)
│   ├── inference.py        # ml-inference: Test/inference runner
│   ├── export.py           # ml-export: ONNX model export
│   ├── visualise.py        # ml-visualise: TensorBoard + search visualization
│   ├── search.py           # ml-search: Hyperparameter optimization with Optuna
│   ├── fl_server.py        # ml-fl-server: Federated learning server
│   ├── fl_client.py        # ml-fl-client: Federated learning client
│   └── fl_run.py           # ml-fl-run: Unified FL launcher (simulation/deployment)
│
└── core/                   # Reusable ML components (no CLI dependencies)
    ├── federated/          # Federated learning with Flower
    │   ├── __init__.py     # FL module API
    │   ├── client.py       # FlowerClient wrapper (composes with trainers)
    │   ├── server.py       # Server and strategy creation
    │   ├── strategies.py   # FL strategy utilities
    │   └── partitioning.py # Data partitioning (IID, non-IID, label-skew)
    ├── data/               # Dataset handling and analysis
    │   ├── __init__.py     # Data module API
    │   ├── datasets.py     # IndexedImageDataset (index-based loading)
    │   ├── detection.py    # Dataset structure detection
    │   ├── indexing.py     # File indexing utilities
    │   └── splitting.py    # CV split creation utilities
    ├── loader.py           # DataLoader creation
    ├── network/            # Model architectures
    │   ├── __init__.py     # get_model() API (routes base vs custom)
    │   ├── base.py         # Torchvision models (ResNet, EfficientNet, etc.)
    │   └── custom.py       # Custom architectures (SimpleCNN, TinyNet)
    ├── trainers/           # Specialized trainer implementations
    │   ├── __init__.py     # get_trainer() factory function
    │   ├── base.py         # BaseTrainer abstract class (with callback support, EMA, Optuna)
    │   ├── standard.py     # StandardTrainer: Traditional PyTorch training
    │   ├── mixed_precision.py  # MixedPrecisionTrainer: PyTorch AMP
    │   ├── accelerate.py   # AccelerateTrainer: Multi-GPU with Accelerate
    │   └── differential_privacy.py  # DPTrainer: Opacus differential privacy
    ├── callbacks/          # Extensible callback system for training hooks
    │   ├── __init__.py     # get_callbacks() factory, CALLBACK_REGISTRY
    │   ├── base.py         # Callback base class, CallbackManager
    │   ├── early_stopping.py     # EarlyStoppingCallback
    │   ├── checkpoint.py   # ModelCheckpointCallback (top-k model saving)
    │   ├── lr_monitor.py   # LearningRateMonitor
    │   ├── progress.py     # ProgressBar (tqdm)
    │   ├── swa.py          # StochasticWeightAveraging
    │   ├── gradient.py     # GradientClipping, GradientNormMonitor
    │   └── augmentation.py # MixUpCallback, CutMixCallback
    ├── training/           # Training utilities
    │   ├── __init__.py     # Training utilities API
    │   └── ema.py          # ModelEMA: Exponential Moving Average
    ├── inference/          # Inference strategies for test-time optimization
    │   ├── __init__.py     # get_inference_strategy() factory function
    │   ├── base.py         # BaseInferenceStrategy abstract class
    │   ├── standard.py     # StandardInference: Basic PyTorch inference
    │   ├── mixed_precision.py  # MixedPrecisionInference: AMP inference
    │   ├── accelerate.py   # AccelerateInference: Multi-GPU inference
    │   ├── tta.py          # TTAInference: Test-Time Augmentation
    │   ├── ensemble.py     # EnsembleInference: Multi-model ensembling
    │   └── tta_ensemble.py # TTAEnsembleInference: Combined TTA + Ensemble
    ├── transforms/         # Transform utilities
    │   ├── __init__.py     # Transform API
    │   └── tta.py          # TTA augmentation utilities
    ├── loss.py             # Loss functions
    ├── optimizer.py        # Optimizers and schedulers
    ├── lr_finder.py        # Learning rate finder implementation
    ├── export.py           # ONNX export utilities
    ├── search.py           # Hyperparameter search utilities (Optuna wrapper)
    ├── test.py             # Evaluation/testing (deprecated)
    ├── metrics/            # Comprehensive metrics suite
    │   ├── __init__.py     # Metrics API
    │   ├── classification.py  # Classification-specific metrics
    │   ├── utils.py        # Metric utilities
    │   ├── visualization.py   # Metric visualization
    │   └── onnx_validation.py # ONNX model validation and benchmarking
    ├── checkpointing.py    # Checkpoint save/load with EMA support
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

1. **Dataset Creation** (`core/data/datasets.py`):
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

### Callbacks System

**NEW:** Extensible callback system for injecting custom behavior into the training loop.

**Design Pattern:**
- All callbacks inherit from `Callback` base class
- `CallbackManager` orchestrates multiple callbacks
- Callbacks invoked at lifecycle hooks: `on_train_begin`, `on_epoch_begin`, `on_phase_begin`, `on_batch_begin`, `on_backward_end`, `on_batch_end`, `on_phase_end`, `on_epoch_end`, `on_train_end`
- Configuration-driven via `config['training']['callbacks']`
- Loaded automatically by CLI via `get_callbacks(config)`

**Available callbacks:**
- `early_stopping` - Stop training when metric stops improving
- `model_checkpoint` - Save top-k best models based on metric
- `lr_monitor` - Log learning rate to TensorBoard
- `progress_bar` - Display training progress with tqdm
- `swa` - Stochastic Weight Averaging (0.5-2% accuracy improvement)
- `gradient_clipping` - Clip gradients to prevent exploding gradients
- `gradient_norm_monitor` - Monitor and log gradient norms
- `mixup` - MixUp data augmentation (1-3% accuracy improvement)
- `cutmix` - CutMix data augmentation (1-3% accuracy improvement)

**Example config:**
```yaml
training:
  callbacks:
    - type: 'early_stopping'
      monitor: 'val_acc'
      patience: 10
      mode: 'max'

    - type: 'model_checkpoint'
      monitor: 'val_acc'
      save_top_k: 3

    - type: 'swa'
      swa_start_epoch: 75
      swa_lr: 0.0005

    - type: 'mixup'
      alpha: 0.2
      apply_prob: 0.5
```

**Backward Compatibility:**
- Legacy `early_stopping` config still supported
- If `callbacks` section exists, it takes precedence
- All trainers support callbacks (passed via `get_trainer(..., callbacks=callbacks)`)

**Custom Callbacks:**
Users can create custom callbacks by subclassing `Callback` and implementing desired lifecycle hooks. See `docs/development/custom-callbacks.md` for guide.

### Inference Strategies

All inference strategies inherit from `BaseInferenceStrategy` and are selected via `config['inference']['strategy']`.

**Available strategies:**
- `standard` (default) - Traditional PyTorch inference
- `mixed_precision` - AMP inference for 2-3x speedup
- `accelerate` - Multi-GPU/distributed inference
- `tta` - Test-Time Augmentation for improved robustness (1-3% accuracy gain, slower)
- `ensemble` - Combine multiple model checkpoints (e.g., from CV folds)
- `tta_ensemble` - Combined TTA + Ensemble for maximum performance (slowest, best accuracy)

**Factory function:** `get_inference_strategy(config, device=None)`

**Example configs:**

```yaml
# TTA inference
inference:
  strategy: 'tta'
  tta:
    augmentations: ['horizontal_flip', 'vertical_flip']
    aggregation: 'mean'  # or 'max', 'voting'

# Ensemble inference
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
      - 'runs/fold_2/weights/best.pt'
    aggregation: 'soft_voting'  # or 'hard_voting', 'weighted'
    weights: [0.4, 0.3, 0.3]  # Optional, for weighted aggregation

# Combined TTA + Ensemble
inference:
  strategy: 'tta_ensemble'
  tta:
    augmentations: ['horizontal_flip']
    aggregation: 'mean'
  ensemble:
    checkpoints: ['runs/fold_0/weights/best.pt', 'runs/fold_1/weights/best.pt']
    aggregation: 'soft_voting'
```

**CLI usage (easier than config):**
```bash
# TTA inference
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta

# Ensemble inference
ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt

# Combined TTA + Ensemble
ml-inference --ensemble runs/fold_0/weights/best.pt runs/fold_1/weights/best.pt --tta
```

**When to use each:**
- `standard`: Fast inference, baseline performance
- `tta`: Single model, want robustness improvement (~1-3% accuracy gain)
- `ensemble`: Have multiple trained folds, want best accuracy (~2-5% gain)
- `tta_ensemble`: Maximum accuracy needed, inference time not critical (~3-8% total gain)

**Performance comparison:**
- `standard`: 1x speed (baseline)
- `tta` (5 augmentations): ~5x slower
- `ensemble` (5 models): ~5x slower
- `tta_ensemble` (5 models × 5 augmentations): ~25x slower

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

### Model EMA (Exponential Moving Average)

**What is EMA?**
Maintains a "shadow" copy of model weights updated as an exponential moving average during training. Typically improves test accuracy by **0.5-2%** with zero additional training cost. Used in SOTA models (YOLO, Stable Diffusion).

**How it works:**
- After each `optimizer.step()`: `ema_weight = decay * ema_weight + (1-decay) * current_weight`
- Separate validation pass with EMA model each epoch
- Both regular and EMA metrics logged to TensorBoard

**Enable in config:**
```yaml
training:
  ema:
    enabled: true
    decay: 0.9999      # 0.999-0.9999 typical (higher = slower update)
    warmup_steps: 2000  # Optional: skip first N training steps
```

**TensorBoard metrics:**
- `Accuracy/val` - Regular model validation accuracy
- `Accuracy/val_ema` - EMA model validation accuracy (typically 0.5-2% higher!)

**Checkpointing:**
- EMA state automatically saved in checkpoints
- Resume training with EMA intact
- Backward compatible (old checkpoints without EMA work fine)

**When to use:**
- Production deployments (more stable predictions)
- Competitive benchmarks (free accuracy gain)
- Any scenario where test performance matters

### Dataset Analysis

The framework includes dataset detection and indexing utilities:
- Automatic dataset structure detection (`core/data/detection.py`)
- File indexing for efficient loading (`core/data/indexing.py`)
- Index-based cross-validation splits (`core/data/splitting.py`)

These utilities are used automatically by `ml-split` and `ml-init-config` commands.

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

### Federated Learning (Optional)

**What is Federated Learning?**
Federated Learning (FL) enables training models across multiple decentralized devices/clients without centralizing data. Each client trains locally on its private data, and only model updates (not data) are shared with a central server for aggregation.

**Installation:**
```bash
uv pip install -e ".[flower]"
```

**Key Concept: Composition Over Inheritance**
The FL implementation wraps existing trainers (standard, mixed_precision, dp, accelerate) in Flower clients. This means:
- Each client can use a different trainer type based on capabilities
- Hospital 1: `standard` trainer (CPU)
- Hospital 2: `dp` trainer (privacy-sensitive data)
- Hospital 3: `mixed_precision` trainer (GPU available)
- All participate in the same federation!

---

#### **Quick Start: Simulation Mode**

Simulation mode runs all clients on one machine for testing:

```bash
# 1. Prepare federated data splits
ml-split --raw_data data/medical_images/raw --federated --num-clients 10 --partition-strategy non-iid

# 2. Create federated config (use template)
cp ml_src/federated_config_template.yaml configs/my_fl_experiment.yaml
# Edit configs/my_fl_experiment.yaml as needed

# 3. Run federated training (ONE command!)
ml-fl-run --config configs/my_fl_experiment.yaml

# 4. Monitor with TensorBoard
tensorboard --logdir runs/simulation/
```

**Output:**
- `runs/simulation/fl_client_0/` - Client 0 logs and checkpoints
- `runs/simulation/fl_client_1/` - Client 1 logs and checkpoints
- ... (one directory per client)

---

#### **Data Partitioning Strategies**

```bash
# IID (uniform): Each client gets equal, random data
ml-split --raw_data data/my_dataset/raw --federated --num-clients 10 --partition-strategy iid

# Non-IID (Dirichlet): Realistic heterogeneous distributions
ml-split --raw_data data/my_dataset/raw --federated --num-clients 10 \\
  --partition-strategy non-iid --alpha 0.5
# alpha: 0.1 (very heterogeneous), 0.5 (moderate), 10.0 (nearly IID)

# Label-Skew: Each client sees only subset of classes
ml-split --raw_data data/my_dataset/raw --federated --num-clients 5 \\
  --partition-strategy label-skew --classes-per-client 2
```

**Output structure:**
```
data/my_dataset/splits/
├── client_0_train.txt  # Client 0's training data
├── client_0_val.txt    # Client 0's validation data
├── client_1_train.txt  # Client 1's training data
├── client_1_val.txt    # Client 1's validation data
...
└── test.txt            # Shared global test set (all clients)
```

---

#### **Configuration: Heterogeneous Clients**

**Simulation mode** (profiles for client groups):

```yaml
federated:
  mode: 'simulation'

  server:
    strategy: 'FedAvg'  # or FedProx, FedAdam, FedAdagrad
    num_rounds: 100
    strategy_config:
      fraction_fit: 0.8        # Use 80% of clients per round
      min_fit_clients: 8
      min_available_clients: 10

  clients:
    num_clients: 10

    # Heterogeneous client profiles
    profiles:
      # Clients 0-5: Standard GPU training
      - id: [0, 1, 2, 3, 4, 5]
        trainer_type: 'standard'
        batch_size: 32

      # Clients 6-7: Mixed precision (faster)
      - id: [6, 7]
        trainer_type: 'mixed_precision'
        batch_size: 64

      # Client 8: Privacy-sensitive (uses DP)
      - id: [8]
        trainer_type: 'dp'
        batch_size: 16
        dp:
          noise_multiplier: 1.1
          max_grad_norm: 1.0
          target_epsilon: 3.0

      # Client 9: CPU only
      - id: [9]
        trainer_type: 'standard'
        device: 'cpu'
        batch_size: 16

  partitioning:
    strategy: 'non-iid'
    alpha: 0.5
```

---

#### **Federated Learning Strategies**

**FedAvg** (default): Federated Averaging, standard FL
```yaml
server:
  strategy: 'FedAvg'
```

**FedProx**: Handles heterogeneous clients better (different compute/data)
```yaml
server:
  strategy: 'FedProx'
  strategy_config:
    proximal_mu: 0.01  # Regularization term
```

**FedAdam**: Adaptive optimizer on server side
```yaml
server:
  strategy: 'FedAdam'
  strategy_config:
    eta: 0.01      # Server learning rate
    beta_1: 0.9    # First moment decay
    beta_2: 0.99   # Second moment decay
```

**FedAdagrad**: Adagrad-style server optimizer
```yaml
server:
  strategy: 'FedAdagrad'
  strategy_config:
    eta: 0.01  # Server learning rate
```

---

#### **Deployment Mode: Real Distributed Setup**

For production across multiple machines:

**Step 1: Update config for deployment mode**
```yaml
federated:
  mode: 'deployment'

  server:
    address: '10.0.0.1:8080'  # Server IP and port
    strategy: 'FedAvg'
    num_rounds: 200

  clients:
    manifest:
      - id: 0
        config_override: 'configs/client_overrides/hospital_1.yaml'
      - id: 1
        config_override: 'configs/client_overrides/hospital_2_dp.yaml'
      - id: 2
        config_override: 'configs/client_overrides/hospital_3.yaml'
```

**Step 2: Create client override configs**
```yaml
# configs/client_overrides/hospital_2_dp.yaml
training:
  trainer_type: 'dp'
  batch_size: 16
  dp:
    noise_multiplier: 1.1
    max_grad_norm: 1.0
    target_epsilon: 3.0
    target_delta: 1e-5
```

**Step 3: Launch server (Machine 1)**
```bash
ml-fl-server --config configs/my_fl_deployment.yaml
```

**Step 4: Launch clients (Machines 2-N)**
```bash
# Hospital 1 (Machine 2)
ml-fl-client --config configs/my_fl_deployment.yaml --client-id 0

# Hospital 2 (Machine 3)
ml-fl-client --config configs/my_fl_deployment.yaml --client-id 1

# Hospital 3 (Machine 4)
ml-fl-client --config configs/my_fl_deployment.yaml --client-id 2
```

**Alternative: Automated deployment (all on one machine)**
```bash
# Automatically starts server + all clients
ml-fl-run --config configs/my_fl_deployment.yaml --mode deployment
```

---

#### **Federated Learning + Differential Privacy**

Achieve privacy-preserving federated learning by combining FL with DP:

```yaml
federated:
  clients:
    profiles:
      # Privacy-sensitive clients use DP trainer
      - id: [0, 1, 2]
        trainer_type: 'dp'
        dp:
          noise_multiplier: 1.1
          max_grad_norm: 1.0
          target_epsilon: 3.0    # Strong privacy guarantee
          target_delta: 1e-5

      # Non-sensitive clients use standard
      - id: [3, 4]
        trainer_type: 'standard'
```

**Result:** Formal privacy guarantees for sensitive client data, while non-sensitive clients train faster!

---

#### **Federated Learning + Optuna**

**Recommended:** Run hyperparameter search **before** FL training (not during):

```bash
# 1. Server-side global search (recommended)
ml-search --config configs/federated_base.yaml --n-trials 50

# 2. Use best config for FL
ml-fl-run --config runs/optuna_studies/my_study/best_config.yaml
```

**Alternative:** Per-client local search (for heterogeneous clients):
```bash
# Each client finds its own optimal hyperparameters
ml-search --config configs/client_0_base.yaml --n-trials 20
ml-search --config configs/client_1_base.yaml --n-trials 20

# Then join federation with optimized configs
ml-fl-run --config configs/federated_deployment.yaml
```

**NOT recommended:** Running Optuna during FL rounds (breaks synchronization, huge overhead)

---

#### **Monitoring Federated Training**

Each client logs to its own directory with TensorBoard:

```bash
# View client 0 training
tensorboard --logdir runs/simulation/fl_client_0/tensorboard

# View all clients (combined)
tensorboard --logdir runs/simulation/

# Server-side metrics logged by Flower
# Look for aggregated metrics in console output
```

---

#### **CLI Reference**

```bash
# Simulation mode (all clients on one machine)
ml-fl-run --config configs/federated_config.yaml
ml-fl-run --config configs/federated_config.yaml --num-rounds 200

# Deployment mode (automated server + clients)
ml-fl-run --config configs/federated_config.yaml --mode deployment

# Manual server launch
ml-fl-server --config configs/federated_config.yaml
ml-fl-server --config configs/federated_config.yaml --server-address 0.0.0.0:9000

# Manual client launch
ml-fl-client --config configs/federated_config.yaml --client-id 0
ml-fl-client --config configs/federated_config.yaml --client-id 1 --trainer-type dp
```

---

#### **When to Use Federated Learning**

**Good use cases:**
- ✅ Medical imaging across hospitals (data can't leave premises)
- ✅ Mobile devices (smartphones, IoT) training personal models
- ✅ Financial data across banks (regulatory constraints)
- ✅ Cross-organization collaboration without data sharing

**Not recommended:**
- ❌ Single organization with centralized data access
- ❌ Small datasets (< 1000 images total)
- ❌ When centralized training is feasible and faster

---

#### **Performance Considerations**

- **Simulation mode:** Limited by single machine resources (GPU sharing via Flower)
- **Deployment mode:** True parallelism across machines
- **Communication overhead:** More FL rounds = more communication, design for fewer rounds with more local epochs
- **Heterogeneity:** Use FedProx strategy for clients with different capabilities

---

#### **Configuration Template**

See `ml_src/federated_config_template.yaml` for comprehensive examples covering:
- IID vs non-IID data distributions
- Heterogeneous client profiles (different trainers)
- All FL strategies (FedAvg, FedProx, FedAdam, FedAdagrad)
- Simulation vs deployment modes
- Integration with DP, EMA, callbacks

---

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
- Console scripts: `ml-init-config`, `ml-split`, `ml-lr-finder`, `ml-train`, `ml-inference`, `ml-export`, `ml-visualise`, `ml-search`, `ml-fl-server`, `ml-fl-client`, `ml-fl-run`
- Core dependencies include: torch, torchvision, matplotlib, onnx, onnxruntime (for export functionality)
- Optional dependencies:
  - Dev: `uv pip install -e ".[dev]"` (pytest, ruff, mkdocs)
  - Differential Privacy: `uv pip install -e ".[dp]"` (opacus)
  - Hyperparameter Search: `uv pip install -e ".[optuna]"` (optuna, plotly, kaleido)
  - Federated Learning: `uv pip install -e ".[flower]"` (flwr)

## Git Workflow

- Main branch: `master`
- Runs directory (`runs/`) is gitignored
- User configs (`configs/*.yaml`) are gitignored (template at `ml_src/config_template.yaml`)
- Data directory (`data/`) is gitignored
