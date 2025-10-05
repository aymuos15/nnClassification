# Architecture Overview

## Project Overview

This is a production-ready PyTorch image classification framework designed for training and evaluating deep learning models on image datasets. The project demonstrates best practices in deep learning engineering, including modular architecture, comprehensive logging, reproducibility, checkpointing, and metrics tracking.

## Key Features

### Core Capabilities

- **Clean, modular architecture** with separation of concerns
- **Flexible model support** - All torchvision models + custom architectures
- **Full reproducibility** through seeding and deterministic operations
- **Training resumption** from checkpoints with complete state restoration
- **Real-time monitoring** with TensorBoard (metrics, confusion matrices, classification reports)
- **Rich CLI** with configuration override support
- **Structured logging** with both console and file outputs
- **Automatic experiment tracking** with organized run directories

### Design Philosophy

1. **Modularity** - Each component has a single, well-defined purpose
2. **Configurability** - YAML-based config with CLI overrides
3. **Reproducibility** - Complete state tracking and seeding
4. **Extensibility** - Easy to add new models, transforms, optimizers
5. **Production-ready** - Logging, checkpointing, error handling
6. **User-friendly** - Clear interfaces, helpful error messages

### Current Use Case

The project is configured for the Hymenoptera dataset (ants vs bees classification), but the architecture is generic and can be easily adapted for any image classification task.

---

## Directory Structure

```
.
├── pyproject.toml              # Project configuration, CLI entry points, and dependencies
│
├── ml_src/                     # Core ML package
│   ├── __init__.py
│   ├── config.yaml            # Base configuration file
│   │
│   ├── cli/                   # Command-line interface scripts
│   │   ├── __init__.py
│   │   ├── init_config.py     # Config initialization (ml-init-config)
│   │   ├── train.py           # Main training script (ml-train)
│   │   ├── inference.py       # Inference/testing script (ml-inference)
│   │   ├── splitting.py       # Dataset splitting utility (ml-split)
│   │   └── visualise.py       # TensorBoard visualization (ml-visualise)
│   │
│   └── core/                  # Core ML modules
│       ├── __init__.py
│       ├── checkpointing.py   # Checkpoint save/load, training summaries
│       ├── dataset.py         # Dataset creation and transforms
│       ├── loader.py          # DataLoader creation
│       ├── loss.py            # Loss functions
│       ├── metrics.py         # Evaluation metrics (confusion matrix, reports)
│       ├── optimizer.py       # Optimizers and learning rate schedulers
│       ├── seeding.py         # Reproducibility utilities
│       ├── test.py            # Testing/evaluation logic
│       ├── trainer.py         # Training loop
│       │
│       └── network/           # Model architectures package
│           ├── __init__.py    # Main API (get_model, save_model, load_model)
│           ├── base.py        # Flexible torchvision model loader
│           └── custom.py      # Custom model architectures (SimpleCNN, TinyNet)
│
├── data/                       # Data directory
│   └── hymenoptera_data/      # Example dataset with raw images and index-based splits
│       ├── raw/
│       │   ├── ants/
│       │   └── bees/
│       └── splits/
│           ├── test.txt
│           ├── fold_0_train.txt
│           └── fold_0_val.txt
│
└── runs/                       # Training run outputs
    └── {run_name}/            # Each run creates a directory
        ├── config.yaml        # Saved configuration
        ├── summary.txt        # Training summary
        │
        ├── logs/              # Log files
        │   ├── train.log
        │   ├── inference.log
        │   ├── classification_report_train.txt
        │   ├── classification_report_val.txt
        │   └── classification_report_test.txt
        │
        ├── plots/             # Visualizations
        │   ├── confusion_matrix_train.png
        │   ├── confusion_matrix_val.png
        │   └── confusion_matrix_test.png
        │
        ├── tensorboard/       # TensorBoard logs
        │   └── events.out.tfevents.*
        │
        └── weights/           # Model checkpoints
            ├── best.pt        # Best validation accuracy model
            └── last.pt        # Latest epoch checkpoint
```

---

## Component Organization

### Entry Points Layer

**Files:** `ml_src/cli/` (train.py, inference.py, splitting.py, visualise.py)

- Orchestrate workflows
- Parse CLI arguments
- Setup infrastructure (logging, directories)
- Call ml_src.core modules

**Purpose:** User-facing CLI interfaces accessible via `ml-init-config`, `ml-train`, `ml-inference`, `ml-split`, and `ml-visualise` commands.

**Entry Points:** Defined in `pyproject.toml` [project.scripts] section for easy command-line access.

---

### ML Source Package (`ml_src/`)

**CLI Scripts (`ml_src/cli/`):**

| Module | CLI Command | Purpose |
|--------|-------------|---------|
| `init_config.py` | `ml-init-config` | Dataset-specific config generation |
| `train.py` | `ml-train` | Main training workflow |
| `inference.py` | `ml-inference` | Model evaluation and testing |
| `splitting.py` | `ml-split` | Dataset splitting utility |
| `visualise.py` | `ml-visualise` | TensorBoard visualization |

**Core Modules (`ml_src/core/`):**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `checkpointing.py` | State persistence | save_checkpoint, load_checkpoint, save_summary |
| `dataset.py` | Data loading | get_datasets, get_transforms, get_class_names |
| `loader.py` | DataLoader creation | get_dataloaders (with seeding) |
| `loss.py` | Loss functions | get_criterion |
| `metrics.py` | Evaluation | save_confusion_matrix, save_classification_report |
| `optimizer.py` | Optimization | get_optimizer, get_scheduler |
| `seeding.py` | Reproducibility | set_seed, seed_worker |
| `test.py` | Model evaluation | test_model |
| `trainer.py` | Training loop | train_model, collect_predictions |

**Configuration:**

| File | Purpose |
|------|---------|
| `config.yaml` | Default configuration with all hyperparameters and settings |

**Model Architecture Package (`ml_src/core/network/`):**

| Module | Purpose | Contents |
|--------|---------|----------|
| `__init__.py` | Unified API | get_model, save_model, load_model |
| `base.py` | Torchvision models | get_base_model, automatic final layer replacement |
| `custom.py` | Custom models | SimpleCNN, TinyNet, MODEL_REGISTRY |

---

## Key Architectural Patterns

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:

```
Configuration → Dataset → DataLoader → Model → Training → Evaluation
     ↓            ↓          ↓           ↓         ↓          ↓
config.yaml  dataset.py  loader.py  network/  trainer.py  metrics.py
                (core/)     (core/)   (core/)   (core/)     (core/)
```

**Benefits:**
- Easy to test individual components
- Easy to swap implementations
- Clear code organization
- Easier debugging

---

### 2. Configuration-Driven Design

All settings centralized in `config.yaml`:

```yaml
data:           # Data loading settings
training:       # Training hyperparameters
optimizer:      # Optimization settings
scheduler:      # LR scheduling
model:          # Model architecture
transforms:     # Data preprocessing
```

**Benefits:**
- No code changes for hyperparameter tuning
- Configuration versioning
- Reproducibility
- Easy experiment tracking

---

### 3. Checkpoint-Based State Management

Complete state saved in checkpoints:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_acc': best_acc,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'random_states': {...},
    'config': config,
    'timestamp': timestamp
}
```

**Benefits:**
- Exact resumption after interruptions
- Complete training history
- Reproducibility
- Debugging aid

---

### 4. CLI Entry Point Design

Multiple CLI commands for different workflows, accessible via `pyproject.toml`:

**`ml-train` (Training Workflow):**
```
Config → Setup → Train → Validate → Checkpoint → Repeat
```

**`ml-inference` (Evaluation Workflow):**
```
Load Config → Load Model → Test → Metrics → Results
```

**`ml-split` (Dataset Splitting):**
```
Source Data → Split Ratios → Train/Val/Test Splits
```

**`ml-visualise` (Visualization):**
```
Run Directory → TensorBoard → Interactive Plots
```

**Benefits:**
- Clear command-line interface
- Defined in pyproject.toml [project.scripts]
- No need for `python script.py` syntax
- Professional CLI tool experience
- Can use trained models independently
- Better for deployment

---

## Package Structure Rationale

### Why `ml_src/` Package?

1. **Namespace isolation** - Prevents naming conflicts
2. **Importability** - Can `import ml_src` from anywhere
3. **Distribution** - Can package and distribute separately via pip
4. **Professional** - Matches standard Python project structure
5. **Entry points** - Enables CLI commands via pyproject.toml

### Why `cli/` and `core/` Separation?

**`cli/` (Command-line interfaces):**
- User-facing scripts
- Argument parsing and orchestration
- Entry points for `ml-train`, `ml-inference`, etc.
- No business logic, just workflow coordination

**`core/` (Core ML functionality):**
- Reusable ML components
- Business logic and algorithms
- Importable by CLI scripts and other code
- Can be used programmatically without CLI

**Benefits:**
1. **Clear separation** - Interface vs implementation
2. **Reusability** - Core modules work without CLI
3. **Testing** - Test core logic independently
4. **Flexibility** - Can add GUI, API, or notebook interfaces later

### Why `network/` Sub-Package?

1. **Scalability** - Room for many model types
2. **Organization** - Clear separation: base vs custom
3. **Extensibility** - Easy to add new model families
4. **Clarity** - `network.base` vs `network.custom` is self-documenting

### Why Separate Files for Similar Functions?

Example: `dataset.py`, `loader.py` could be one file.

**Reasons for separation:**
- **Single responsibility** - Each file does one thing
- **Easier navigation** - Find what you need quickly
- **Parallel development** - Multiple people can work
- **Testing** - Test each component independently

---

## Output Organization

### Run Directory Structure

Each training run creates an organized directory:

```
runs/{run_name}/
├── config.yaml          # Exact configuration used
├── summary.txt          # Human-readable summary
├── weights/             # Model checkpoints
│   ├── best.pt         # Highest val accuracy
│   └── last.pt         # Latest (for resuming)
├── logs/                # Text logs and reports
├── plots/               # Visualizations
└── tensorboard/         # TensorBoard data
```

**Run Naming:**
- Automatic based on dataset, fold, and hyperparameter overrides
- Example: `hymenoptera_batch_32_epochs_50_lr_0.01_fold_0/`
- Default: `hymenoptera_base_fold_0/`

**Benefits:**
- Self-documenting experiments
- Easy comparison
- No overwrites
- Complete artifact preservation

---

## Dependencies

### Core Dependencies

| Package | Purpose |
|---------|---------|
| **torch** | Deep learning framework |
| **torchvision** | Models, transforms, datasets |
| **tensorboard** | Training visualization |
| **pyyaml** | Configuration parsing |
| **numpy** | Numerical operations |
| **Pillow** | Image loading |

### Utilities

| Package | Purpose |
|---------|---------|
| **loguru** | Structured logging |
| **rich** | Terminal formatting |
| **matplotlib** | Plotting |
| **seaborn** | Statistical visualizations |
| **scikit-learn** | Metrics (classification report) |

See `pyproject.toml` for exact versions and all dependencies.

---

## System Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- 10GB disk space
- CPU (GPU optional)

### Recommended

- Python 3.10+
- 16GB+ RAM
- 50GB+ disk space
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.0+

---

## Extensibility Points

The architecture is designed for easy extension:

### 1. Add New Models

**Torchvision models:**
- Just change config: `architecture: 'efficientnet_b0'`
- No code changes needed

**Custom models:**
1. Define in `ml_src/network/custom.py`
2. Add to `MODEL_REGISTRY`
3. Use in config

### 2. Add New Transforms

1. Update `ml_src/dataset.py::get_transforms()`
2. Add to config `transforms` section

### 3. Add New Optimizers

1. Update `ml_src/optimizer.py::get_optimizer()`
2. Add to config `optimizer` section

### 4. Add New Schedulers

1. Update `ml_src/optimizer.py::get_scheduler()`
2. Add to config `scheduler` section

### 5. Add New Metrics

1. Add function in `ml_src/metrics.py`
2. Call from `trainer.py` or `inference.py`

### 6. Add New Loss Functions

1. Update `ml_src/loss.py::get_criterion()`
2. Add to config if needed

**See:** [Development Guides](../development/) for detailed instructions.

---

## Comparison with Other Frameworks

### PyTorch Lightning

**This Framework:**
- Simpler, more explicit
- Easier to understand and modify
- Good for learning and small projects

**PyTorch Lightning:**
- More abstraction
- More features out-of-the-box
- Better for large-scale projects

### Timm (PyTorch Image Models)

**This Framework:**
- Educational and customizable
- Clear architecture
- Easy to understand end-to-end

**Timm:**
- Production library
- More models
- More advanced features

### Custom Scripts

**This Framework:**
- Structure and organization
- Best practices built-in
- Easy to extend

**Custom Scripts:**
- Maximum flexibility
- No constraints
- Requires more setup

---

## Related Documentation

- **[Entry Points](entry-points.md)** - Detailed explanation of train.py and inference.py
- **[ML Source Modules](ml-src-modules.md)** - Deep dive into each module
- **[Data Flow](data-flow.md)** - How data moves through the system
- **[Design Decisions](design-decisions.md)** - Why the architecture is this way

---

## Summary

**The architecture is:**
- ✅ **Modular** - Independent, reusable components
- ✅ **Configurable** - YAML-based with CLI overrides
- ✅ **Reproducible** - Complete state tracking
- ✅ **Extensible** - Easy to add new functionality
- ✅ **Production-ready** - Logging, error handling, checkpointing
- ✅ **User-friendly** - Clear interfaces and documentation

**Key Principle:** **Separation of concerns** - Each component does one thing well.

**Next Steps:** Explore detailed documentation for each component in this section.
