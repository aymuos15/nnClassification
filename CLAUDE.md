# PyTorch Image Classifier - Codebase Guide

## Project Overview

This is a production-ready PyTorch image classification framework designed for training and evaluating deep learning models on image datasets. The project demonstrates best practices in deep learning engineering, including modular architecture, comprehensive logging, reproducibility, checkpointing, and metrics tracking.

**Key Features:**
- Clean, modular architecture with separation of concerns
- ResNet18-based transfer learning
- Full reproducibility through seeding and deterministic operations
- Training resumption from checkpoints with complete state restoration
- Real-time monitoring with TensorBoard (metrics, confusion matrices, classification reports)
- Rich CLI with configuration override support
- Structured logging with both console and file outputs
- Automatic experiment tracking with organized run directories

**Current Use Case:**
The project is configured for the Hymenoptera dataset (ants vs bees classification), but the architecture is generic and can be easily adapted for any image classification task.

## Directory Structure

```
.
├── train.py                    # Main training script (entry point)
├── inference.py                # Inference/testing script (entry point)
├── requirements.txt            # Python dependencies
├── ml_src/                     # Core ML package
│   ├── __init__.py
│   ├── config.yaml            # Base configuration file
│   ├── checkpointing.py       # Checkpoint save/load, training summaries
│   ├── dataset.py             # Dataset creation and transforms
│   ├── loader.py              # DataLoader creation
│   ├── loss.py                # Loss functions
│   ├── metrics.py             # Evaluation metrics (confusion matrix, reports)
│   ├── network/               # Model architectures package
│   │   ├── __init__.py        # Main API (get_model, save_model, load_model)
│   │   ├── base.py            # Flexible torchvision model loader
│   │   └── custom.py          # Custom model architectures (SimpleCNN, TinyNet)
│   ├── optimizer.py           # Optimizers and learning rate schedulers
│   ├── seeding.py             # Reproducibility utilities
│   ├── test.py                # Testing/evaluation logic
│   └── trainer.py             # Training loop
├── data/                       # Data directory
│   └── hymenoptera_data/      # Dataset with train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
└── runs/                       # Training run outputs
    └── {run_name}/            # Each run creates a directory
        ├── config.yaml        # Saved configuration
        ├── summary.txt        # Training summary
        ├── logs/              # Log files
        │   ├── train.log
        │   ├── inference.log
        │   ├── classification_report_train.txt
        │   ├── classification_report_val.txt
        │   └── classification_report_test.txt
        ├── tensorboard/       # TensorBoard logs
        │   └── events.out.tfevents.*  # Training metrics, confusion matrices
        └── weights/           # Model checkpoints
            ├── best.pt        # Best validation accuracy model
            └── last.pt        # Latest epoch checkpoint
```

## Core Components

### 1. Entry Points

#### `train.py` - Training Script
**Purpose:** Orchestrates the complete training pipeline from configuration to final evaluation.

**Key Responsibilities:**
- Parse CLI arguments and load/override configuration
- Create organized run directories based on hyperparameter overrides
- Setup logging infrastructure (console + file)
- Initialize datasets, dataloaders, model, optimizer, scheduler
- Execute training loop via `trainer.py` with TensorBoard logging
- Support training resumption from checkpoints

**CLI Arguments:**
```bash
python train.py \
  --config ml_src/config.yaml \          # Config file path
  --resume runs/base/last.pt \           # Resume from checkpoint
  --data_dir data/hymenoptera_data \     # Override data directory
  --batch_size 16 \                      # Override batch size
  --num_workers 4 \                      # Override worker count
  --num_epochs 25 \                      # Override epoch count
  --lr 0.01 \                            # Override learning rate
  --momentum 0.9 \                       # Override SGD momentum
  --step_size 7 \                        # Override LR step size
  --gamma 0.1                            # Override LR gamma
```

**Run Directory Naming:**
- Run names are generated from overridden parameters
- Example: `batch_16_epochs_25_lr_0.01` for multiple overrides
- Default name: `base` (no overrides)

#### `inference.py` - Inference Script
**Purpose:** Load trained models and evaluate on test data with comprehensive metrics.

**Key Responsibilities:**
- Load saved configuration and checkpoint from run directory
- Run model on test dataset
- Generate per-sample predictions
- Create confusion matrices and classification reports
- Display rich formatted results (tables, summaries)

**CLI Arguments:**
```bash
python inference.py \
  --run_dir runs/base \          # Run directory to load from
  --checkpoint best.pt \          # Which checkpoint (best.pt or last.pt)
  --data_dir data/custom          # Override data directory
```

### 2. ML Source Modules (`ml_src/`)

#### `checkpointing.py` - Checkpoint Management
**Purpose:** Comprehensive state persistence for training resumption and experiment tracking.

**Key Functions:**

- `save_checkpoint()`: Saves complete training state including:
  - Model state dict
  - Optimizer state dict
  - Scheduler state dict
  - Training metrics (losses, accuracies)
  - Best accuracy and epoch
  - Configuration
  - Random states (PyTorch, NumPy, Python, CUDA)
  - Timestamp

- `load_checkpoint()`: Restores complete training state for seamless resumption

- `save_summary()`: Creates human-readable training summaries with:
  - Status (running/completed/failed)
  - Timing information (start, end, duration)
  - Progress (current epoch / total epochs)
  - Performance metrics (best/final accuracies)
  - Configuration details
  - System information (device, parameters, dataset sizes)
  - Error messages (if failed)

- `count_parameters()`: Count trainable parameters

- `format_duration()`: Human-readable time formatting

**Design Decision:** Complete state preservation enables exact training resumption, critical for long-running experiments and debugging.

#### `dataset.py` - Dataset Management
**Purpose:** Create PyTorch datasets with configurable transforms.

**Key Functions:**

- `get_transforms()`: Build transform pipelines per split (train/val/test):
  - Resize
  - Random horizontal flip (training only)
  - ToTensor
  - Normalization (ImageNet stats)

- `get_datasets()`: Create ImageFolder datasets for all splits

- `get_class_names()`: Extract class names from training dataset

**Design Decision:** Transform configuration in YAML allows experimenting with augmentation without code changes.

#### `loader.py` - DataLoader Creation
**Purpose:** Create reproducible DataLoaders with proper seeding.

**Key Functions:**

- `get_dataloaders()`: Create DataLoaders with:
  - Configurable batch size and worker count
  - Shuffling (train only)
  - Seeded generator for reproducibility
  - Worker initialization function for multi-process seeding

**Design Decision:** Reproducible data loading is crucial for debugging and experiment comparison. Each worker process is properly seeded.

#### `metrics.py` - Evaluation Metrics
**Purpose:** Generate and save evaluation visualizations and reports.

**Key Functions:**

- `save_confusion_matrix()`: Creates heatmap visualization using seaborn
  - Annotated with counts
  - Class names on axes
  - Saved as high-resolution PNG

- `save_classification_report()`: Saves sklearn classification report
  - Per-class precision, recall, F1-score
  - Support counts
  - Macro/weighted averages
  - 4 decimal precision

**Design Decision:** Automatic metric generation provides immediate feedback on model performance across all classes.

#### `network/` - Model Architectures Package
**Purpose:** Provide flexible model loading for both torchvision models and custom architectures.

**Structure:**
```
network/
├── __init__.py    # Main API
├── base.py        # Torchvision model loader
└── custom.py      # Custom architectures
```

**`__init__.py` - Main API:**

- `get_model(config, device)`: Unified interface for loading models
  - Routes to base or custom models based on config
  - Maintains backward compatibility
  - Returns configured model on specified device

- `save_model(model, path)`: Saves model state dict

- `load_model(model, path, device)`: Loads model weights

**`base.py` - Flexible Torchvision Model Loader:**

- `get_base_model(architecture, num_classes, weights, device)`: Load any torchvision model
  - Supports all torchvision architectures (ResNet, VGG, EfficientNet, ViT, etc.)
  - Automatically detects and replaces final layer for each architecture family
  - Optional pretrained weights ('DEFAULT' for ImageNet, None for random init)

- `_replace_final_layer(model, architecture, num_classes)`: Internal helper
  - Pattern matches architecture name to determine final layer structure
  - ResNet family: Replaces `.fc`
  - VGG/AlexNet: Replaces `.classifier[6]`
  - DenseNet: Replaces `.classifier`
  - EfficientNet/MobileNet: Replaces `.classifier[1]` or `.classifier[-1]`
  - Vision Transformers: Replaces `.heads.head` or `.head`
  - And many more (see base.py for full list)
  - Generic fallback for unknown architectures

**`custom.py` - Custom Architectures:**

- `SimpleCNN`: 3-layer CNN for small/medium datasets
  - Conv layers: 3→32→64→128 channels
  - FC layers with dropout
  - Configurable input size and dropout rate

- `TinyNet`: Minimal 2-layer CNN for prototyping
  - Fast training, limited capacity
  - Good for quick experiments

- `get_custom_model(model_name, num_classes, input_size, device, **kwargs)`: Factory function
  - Loads custom models from registry
  - Extensible: Users add models to MODEL_REGISTRY

**Usage Examples:**

```python
# Base model (torchvision)
config = {
    'model': {
        'type': 'base',
        'architecture': 'resnet18',
        'num_classes': 10,
        'weights': 'DEFAULT'
    }
}
model = get_model(config, 'cuda:0')

# Custom model
config = {
    'model': {
        'type': 'custom',
        'custom_architecture': 'simple_cnn',
        'num_classes': 10,
        'dropout': 0.3
    }
}
model = get_model(config, 'cuda:0')
```

**Design Decisions:**
- **Package structure**: Clean separation between base and custom models
- **Automatic final layer detection**: No manual configuration needed per architecture
- **Extensible**: Easy to add new custom models or torchvision architectures
- **Boilerplate philosophy**: Base models work out-of-the-box, custom models as templates
- **Backward compatible**: Same API as original network.py

#### `loss.py` - Loss Functions
**Purpose:** Define loss functions for training.

**Key Functions:**

- `get_criterion()`: Returns loss function based on config
  - Currently: CrossEntropyLoss
  - Extensible: Can add Focal Loss, Label Smoothing, etc.

**Design Decision:** Separate loss module enables easy experimentation with different loss functions without modifying training code.

#### `optimizer.py` - Optimizers & Schedulers
**Purpose:** Configure optimizers and learning rate schedulers.

**Key Functions:**

- `get_optimizer()`: Creates optimizer based on config
  - Currently: SGD with momentum
  - Extensible: Can add Adam, AdamW, etc.

- `get_scheduler()`: Creates LR scheduler based on config
  - Currently: StepLR (multiplicative decay)
  - Extensible: Can add Cosine Annealing, OneCycle, etc.

**Design Decision:** Optimizers and schedulers are tightly coupled, so they live in the same module. Separate from model architecture for flexibility.

#### `seeding.py` - Reproducibility
**Purpose:** Ensure reproducible experiments across runs.

**Key Functions:**

- `set_seed()`: Seeds all random number generators:
  - Python's random
  - NumPy
  - PyTorch (CPU and CUDA)
  - cuDNN behavior (deterministic vs benchmark mode)

- `seed_worker()`: Seeds individual DataLoader worker processes

**Design Decisions:**
- **Non-deterministic mode (default)**: Faster training, approximate reproducibility
- **Deterministic mode**: Slower but fully reproducible (useful for debugging)

#### `test.py` - Model Testing
**Purpose:** Evaluate trained models on test data.

**Key Functions:**

- `test_model()`: Runs inference on test dataset
  - Returns overall accuracy
  - Returns per-sample results (true label, predicted label, correctness)
  - Optionally uses class names instead of indices

**Design Decision:** Per-sample results enable detailed error analysis and understanding of model failures.

#### `trainer.py` - Training Loop
**Purpose:** Core training logic with validation, checkpointing, and metrics.

**Key Functions:**

- `train_model()`: Main training loop that:
  - Trains for configured epochs
  - Validates after each epoch
  - Saves best model based on validation accuracy
  - Saves last model checkpoint after every epoch
  - Updates training summary continuously
  - Collects training/validation losses and accuracies
  - Generates confusion matrices and classification reports
  - Supports resumption from previous checkpoints

- `collect_predictions()`: Gathers all predictions for metric computation

**Training Flow per Epoch:**
1. **Training Phase:**
   - Set model to train mode
   - Iterate through training batches
   - Forward pass
   - Compute loss
   - Backward pass
   - Optimizer step
   - Track loss and accuracy

2. **Validation Phase:**
   - Set model to eval mode
   - Iterate through validation batches
   - Forward pass (no gradients)
   - Compute loss and accuracy
   - Save checkpoint if best accuracy improved

3. **Scheduler Step:** Update learning rate

4. **Checkpointing:**
   - Save `best.pt` when validation accuracy improves
   - Save `last.pt` after every epoch (for resumption)

5. **Summary Update:** Update `summary.txt` with current progress

**Post-Training:**
- Load best model weights
- Generate confusion matrices for train and val sets
- Generate classification reports
- Save final summary with completion status

**Design Decisions:**
- Validation after every epoch provides frequent feedback
- Dual checkpointing (best + last) enables both deployment and resumption
- Continuous summary updates allow monitoring long-running jobs
- Best model selection based on validation accuracy prevents overfitting

## Data Flow

### Training Pipeline
```
Configuration → Datasets → DataLoaders → Model → Training Loop
     ↓              ↓           ↓           ↓          ↓
  Override      Transforms  Batching   Forward   Validation
     ↓              ↓           ↓           ↓          ↓
  Run Dir     Augmentation Shuffling  Backward  Checkpointing
                                         ↓          ↓
                                    Optimizer   Metrics
                                         ↓          ↓
                                    Scheduler   Logging
```

### Inference Pipeline
```
Run Directory → Load Config → Load Model → Test DataLoader → Predictions
                                ↓              ↓                  ↓
                           Best/Last      Test Batches      Per-Sample Results
                           Checkpoint          ↓                  ↓
                                           Forward          Metrics & Reports
```

## Configuration System

The project uses a hierarchical configuration system with YAML + CLI overrides:

1. **Base Configuration:** `ml_src/config.yaml` defines all default settings
2. **CLI Overrides:** Command-line arguments override base config
3. **Saved Configuration:** Each run saves its final config for reproducibility

See `CONFIG.md` for detailed configuration documentation.

## Key Features

### 1. Reproducibility
- Complete seeding of all random number generators
- Optional deterministic mode for full reproducibility
- Random state saved/restored in checkpoints
- Seeded DataLoader workers

### 2. Training Resumption
- Save complete training state in checkpoints
- Resume from any epoch with `--resume` flag
- Preserves optimizer momentum and scheduler state
- Continues metric tracking from previous run

### 3. Experiment Tracking
- Automatic run directory creation
- Configuration persistence
- Continuous training summaries
- Comprehensive logging (console + file)

### 4. Metrics & Visualization
- Real-time TensorBoard monitoring with interactive plots
- Training/validation loss and accuracy curves
- Learning rate tracking
- Confusion matrices for train/val/test sets
- Per-class classification reports
- Per-sample prediction results

**View metrics:**
```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in your browser
```

### 5. Logging
- Structured logging with loguru
- Color-coded console output
- Timestamped file logs
- Rotating log files (10 MB, 30 day retention)
- Debug vs. Info level separation

## Development Workflow

### Adding a New Model Architecture

**For Torchvision Models (Base Models):**
1. Update config: Set `model.architecture` to any torchvision model name
2. Example: `architecture: 'efficientnet_b0'`, `architecture: 'vit_b_16'`
3. No code changes needed - automatic final layer detection handles it
4. All torchvision models supported out-of-the-box

**For Custom Models:**
1. Define your model class in `ml_src/network/custom.py` (inherit from `nn.Module`)
2. Add it to the `MODEL_REGISTRY` dictionary in `get_custom_model()`
3. Update config: Set `model.type: 'custom'` and `model.custom_architecture: 'your_model_name'`
4. No other changes needed (modular design)

**Example - Adding a Custom ResNet-like Model:**
```python
# In ml_src/network/custom.py

class MyResNet(nn.Module):
    def __init__(self, num_classes, ...):
        # Your implementation
        pass

# Add to MODEL_REGISTRY
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'tiny_net': TinyNet,
    'my_resnet': MyResNet,  # Add here
}
```

### Adding New Transforms
1. Update `ml_src/config.yaml` transforms section
2. Modify `ml_src/dataset.py::get_transforms()` to handle new transform
3. No other changes needed

### Adding New Metrics
1. Add metric computation in `ml_src/metrics.py`
2. Call from `ml_src/trainer.py` post-training or `inference.py`

### Changing Optimizer/Scheduler
1. Modify `ml_src/optimizer.py::get_optimizer()` or `get_scheduler()`
2. Update config with new hyperparameters
3. Example: Add `optimizer.type` config option to select SGD/Adam/AdamW
4. Ensure checkpoint save/load handles new optimizer state

### Working with Different Datasets
1. Organize data in ImageFolder format:
   ```
   data/your_dataset/
   ├── train/
   │   ├── class1/
   │   ├── class2/
   │   └── ...
   ├── val/
   └── test/
   ```
2. Update `config.yaml`: `data.data_dir` and `model.num_classes`
3. Run training: `python train.py --data_dir data/your_dataset`

### Debugging Training Issues
1. Enable deterministic mode: `deterministic: true` in config
2. Check logs in `runs/{run_name}/logs/train.log`
3. Review summary: `runs/{run_name}/summary.txt`
4. Examine confusion matrices in `runs/{run_name}/plots/`
5. Use `--resume` to continue from last checkpoint

## Architecture Decisions

### Why Modular Design?
- **Testability:** Each module can be tested independently
- **Reusability:** Modules work across different projects
- **Maintainability:** Changes localized to specific modules
- **Clarity:** Clear separation of concerns

### Why Split Network/Loss/Optimizer?
The `ml_src/` package separates these concerns into dedicated modules:

- **`network/`**: Model architectures package
  - **`base.py`**: Flexible torchvision model loader
    - Supports all torchvision architectures automatically
    - Automatic final layer detection and replacement
    - No manual configuration needed per architecture
  - **`custom.py`**: Custom model implementations
    - Example models (SimpleCNN, TinyNet)
    - Template for users to create their own
    - Registry-based extensibility
  - **`__init__.py`**: Clean, unified API
    - Routes between base and custom models
    - Maintains backward compatibility
  - **Package structure benefits**:
    - Clear separation between base and custom models
    - Easy to find and modify specific model types
    - Can grow to include more specialized modules (pretrained/, timm_models/, etc.)

- **`loss.py`**: Loss functions only
  - Easy to add new losses (Focal Loss, Label Smoothing, custom losses)
  - Can A/B test different loss functions
  - Loss selection independent of model choice

- **`optimizer.py`**: Optimizers and schedulers
  - Easy to add new optimizers (Adam, AdamW, Lion)
  - Easy to add new schedulers (Cosine, OneCycle, ReduceLROnPlateau)
  - Optimizers and schedulers are tightly coupled (kept together)
  - Separate from model and loss for maximum flexibility

**Benefits:**
- **Extensibility**: Adding new models doesn't touch optimizer code
- **Experimentation**: Try different models/optimizers/losses independently
- **Single Responsibility**: Each file/module has one clear purpose
- **Scalability**: Package structure prevents monolithic files
- **Professional**: Matches PyTorch Lightning, timm, HuggingFace architectures
- **Boilerplate-friendly**: Users can customize without breaking the base system

### Why YAML Configuration?
- Human-readable and editable
- Hierarchical structure matches code organization
- Easy to version control
- CLI overrides provide flexibility

### Why Checkpoint Everything?
- Long training runs can fail (hardware, interruptions)
- Hyperparameter search requires resumption
- Debugging benefits from exact state restoration
- Complete training history for analysis

### Why Separate Best and Last Checkpoints?
- **best.pt**: For deployment and final evaluation
- **last.pt**: For resuming interrupted training
- Different use cases, both critical

### Why Rich Logging?
- Console: Immediate feedback during training
- File: Complete record for post-analysis
- Color coding: Quick visual parsing
- Rotation: Prevent disk space issues

### Why Per-Sample Results?
- Error analysis: Understand which samples fail
- Dataset quality: Identify mislabeled data
- Model debugging: Examine failure patterns
- Class balance: Detect bias issues

## Common Tasks

### Train from Scratch
```bash
python train.py --config ml_src/config.yaml
```

### Resume Training
```bash
python train.py --resume runs/base/last.pt
```

### Hyperparameter Search
```bash
python train.py --lr 0.01 --batch_size 16 --num_epochs 25
python train.py --lr 0.001 --batch_size 32 --num_epochs 25
```

### Evaluate Model
```bash
python inference.py --run_dir runs/base --checkpoint best.pt
```

### View Training Metrics
```bash
# View all runs
tensorboard --logdir runs/

# View specific run
tensorboard --logdir runs/base/tensorboard

# Compare multiple runs
tensorboard --logdir runs/ --port 6006

# Then open http://localhost:6006 in your browser
```

### Train on GPU
```bash
# Automatic if CUDA available, or override:
python train.py --device cuda:0
```

### Train on CPU
```bash
python train.py
# Config will fallback to CPU if CUDA unavailable
```

## Dependencies

See `requirements.txt` for exact versions. Key dependencies:
- **torch**: Deep learning framework
- **torchvision**: Model architectures and transforms
- **tensorboard**: Real-time training visualization
- **pyyaml**: Configuration parsing
- **loguru**: Structured logging
- **rich**: Terminal formatting
- **matplotlib**: Plotting (for confusion matrices)
- **seaborn**: Statistical visualizations
- **scikit-learn**: Metrics computation
- **numpy**: Numerical operations
- **Pillow**: Image loading

## Performance Considerations

### Training Speed
- Use `deterministic: false` for faster training (default)
- Increase `num_workers` for faster data loading (4-8 typically good)
- Larger batch sizes utilize GPU better (memory permitting)
- `cudnn.benchmark: true` finds fastest algorithms (non-deterministic mode)

### Memory Usage
- Reduce batch size if OOM errors occur
- Model checkpoints include full training state (large files)
- Logs rotate to prevent disk space issues

### Reproducibility vs. Speed Trade-off
- `deterministic: false` (default): Faster, approximately reproducible
- `deterministic: true`: Slower, fully reproducible across hardware

## Troubleshooting

### Training Diverges (Loss → NaN)
- Reduce learning rate
- Check data normalization
- Verify labels are correct
- Enable gradient clipping (requires code modification)

### Low Accuracy
- Train for more epochs
- Increase model capacity
- Add data augmentation
- Check dataset quality and class balance

### Out of Memory
- Reduce batch size
- Reduce number of workers
- Use smaller model
- Enable gradient checkpointing (requires code modification)

### Slow Data Loading
- Increase `num_workers`
- Check disk I/O (SSD vs HDD)
- Reduce image resize resolution
- Cache preprocessed data (requires code modification)

### Can't Resume Training
- Ensure checkpoint path is correct
- Check checkpoint file integrity
- Verify device compatibility (CUDA vs CPU)

## Future Extensions

Potential improvements and extensions:
- Support for additional model architectures (EfficientNet, Vision Transformer)
- Mixed precision training (AMP) for faster training
- Distributed training across multiple GPUs
- Early stopping based on validation metrics
- Learning rate finder
- Gradient accumulation for larger effective batch sizes
- TensorBoard integration
- Model ensembling
- Test-time augmentation
- Cross-validation support
- Automatic hyperparameter optimization (Optuna, Ray Tune)

## License & Attribution

This codebase follows PyTorch best practices and incorporates design patterns from:
- PyTorch official tutorials
- Transfer learning examples
- Production ML systems
