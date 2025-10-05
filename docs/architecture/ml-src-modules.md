# ML Source Modules

Complete reference for all modules in the `ml_src/` package.

---

## Module Overview

### CLI Scripts (`ml_src/cli/`)

| Module | CLI Command | Purpose |
|--------|-------------|---------|
| `train.py` | `ml-train` | Main training workflow |
| `inference.py` | `ml-inference` | Model evaluation and testing |
| `splitting.py` | `ml-split` | Dataset splitting utility |
| `visualise.py` | `ml-visualise` | TensorBoard visualization |

### Core Modules (`ml_src/core/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `checkpointing.py` | State persistence | save_checkpoint, load_checkpoint |
| `dataset.py` | Data loading | get_datasets, get_transforms |
| `loader.py` | DataLoader creation | get_dataloaders |
| `loss.py` | Loss functions | get_criterion |
| `metrics.py` | Evaluation | save_confusion_matrix, save_classification_report |
| `network/__init__.py` | Model API | get_model, save_model, load_model |
| `network/base.py` | Torchvision models | get_base_model |
| `network/custom.py` | Custom models | SimpleCNN, TinyNet |
| `optimizer.py` | Optimization | get_optimizer, get_scheduler |
| `seeding.py` | Reproducibility | set_seed, seed_worker |
| `test.py` | Evaluation | test_model |
| `trainer.py` | Training loop | train_model |

### Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Default configuration with all hyperparameters and settings |

---

## checkpointing.py

**Purpose:** Complete training state persistence and experiment tracking.

### Functions

#### `save_checkpoint(path, model, optimizer, scheduler, epoch, ...)`
Saves complete training state.

**Saves:**
- Model weights
- Optimizer state (momentum, etc.)
- Scheduler state
- Training metrics history
- Best accuracy
- Random states (all RNGs)
- Configuration
- Timestamp

**Usage:**
```python
save_checkpoint(
    path='runs/base/weights/best.pt',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=10,
    best_acc=0.92,
    train_losses=[...],
    val_losses=[...],
    config=config
)
```

#### `load_checkpoint(path, model, optimizer, scheduler, device)`
Restores training state.

**Returns:** Dictionary with all checkpoint data

**Usage:**
```python
checkpoint = load_checkpoint(
    'runs/base/weights/last.pt',
    model, optimizer, scheduler, device
)
start_epoch = checkpoint['epoch'] + 1
best_acc = checkpoint['best_acc']
```

#### `save_summary(run_dir, status, ...)`
Creates human-readable training summary.

**Includes:**
- Training status
- Timing (start, end, duration)
- Progress (epochs)
- Performance (accuracies)
- Configuration
- System info

**Usage:**
```python
save_summary(
    run_dir='runs/base',
    status='completed',
    start_time=start,
    end_time=end,
    current_epoch=50,
    total_epochs=50,
    best_acc=0.92,
    final_acc=0.91,
    config=config,
    device='cuda:0'
)
```

---

## dataset.py

**Purpose:** Dataset creation and transform configuration.

### Functions

#### `get_transforms(config)`
Creates transform pipelines for each split.

**Returns:** Dict with 'train', 'val', 'test' transforms

**Transform pipeline:**
1. Resize
2. Random horizontal flip (train only)
3. ToTensor
4. Normalize

**Usage:**
```python
transforms = get_transforms(config)
train_transform = transforms['train']
```

#### `get_datasets(config)`
Creates ImageFolder datasets.

**Returns:** Dict with 'train', 'val', 'test' datasets

**Usage:**
```python
datasets = get_datasets(config)
train_dataset = datasets['train']
```

#### `get_class_names(dataset)`
Extracts class names from dataset.

**Returns:** List of class names

**Usage:**
```python
class_names = get_class_names(datasets['train'])
# ['ants', 'bees']
```

---

## loader.py

**Purpose:** Create reproducible DataLoaders.

### Functions

#### `get_dataloaders(datasets, config)`
Creates DataLoaders with proper seeding.

**Features:**
- Seeded generator for reproducibility
- Worker initialization with `seed_worker`
- Shuffling (train only)
- Configurable batch size and workers

**Returns:** Dict with 'train', 'val', 'test' DataLoaders

**Usage:**
```python
dataloaders = get_dataloaders(datasets, config)
train_loader = dataloaders['train']
```

---

## loss.py

**Purpose:** Loss function configuration.

### Functions

#### `get_criterion(config)`
Returns configured loss function.

**Currently supported:**
- CrossEntropyLoss

**Extensible:** Add Focal Loss, Label Smoothing, etc.

**Usage:**
```python
criterion = get_criterion(config)
loss = criterion(outputs, labels)
```

---

## metrics.py

**Purpose:** Evaluation metrics and visualization.

### Functions

#### `save_confusion_matrix(true_labels, pred_labels, class_names, path)`
Creates and saves confusion matrix heatmap.

**Features:**
- Seaborn heatmap
- Annotated with counts
- Class names on axes

**Usage:**
```python
save_confusion_matrix(
    true_labels=y_true,
    pred_labels=y_pred,
    class_names=['ants', 'bees'],
    path='runs/base/plots/confusion_matrix.png'
)
```

#### `save_classification_report(true_labels, pred_labels, class_names, path)`
Saves sklearn classification report.

**Includes:**
- Per-class precision, recall, F1
- Support counts
- Macro/weighted averages

**Usage:**
```python
save_classification_report(
    true_labels=y_true,
    pred_labels=y_pred,
    class_names=['ants', 'bees'],
    path='runs/base/logs/classification_report.txt'
)
```

---

## network/ Package

### network/__init__.py

**Purpose:** Unified model API.

#### `get_model(config, device)`
Loads model (base or custom) based on config.

**Usage:**
```python
# Base model
config = {'model': {'type': 'base', 'architecture': 'resnet18', ...}}
model = get_model(config, 'cuda:0')

# Custom model
config = {'model': {'type': 'custom', 'custom_architecture': 'simple_cnn', ...}}
model = get_model(config, 'cuda:0')
```

#### `save_model(model, path)`
Saves model state dict.

#### `load_model(model, path, device)`
Loads model weights.

### network/base.py

**Purpose:** Flexible torchvision model loader.

#### `get_base_model(architecture, num_classes, weights, device)`
Loads any torchvision model with automatic final layer replacement.

**Supported architectures:** All torchvision models

**Features:**
- Automatic final layer detection
- Pretrained weights support
- No manual configuration needed

**Usage:**
```python
model = get_base_model(
    architecture='resnet50',
    num_classes=10,
    weights='DEFAULT',  # or None
    device='cuda:0'
)
```

### network/custom.py

**Purpose:** Custom model architectures.

#### `SimpleCNN`
3-layer CNN for small/medium datasets.

**Architecture:**
- Conv: 3→32→64→128 channels
- FC layers with dropout
- Configurable dropout rate

#### `TinyNet`
Minimal 2-layer CNN for prototyping.

#### `get_custom_model(model_name, num_classes, ...)`
Factory function for custom models.

**Extensible:** Add models to `MODEL_REGISTRY`

---

## optimizer.py

**Purpose:** Optimizer and scheduler configuration.

### Functions

#### `get_optimizer(parameters, config)`
Creates optimizer (currently SGD).

**Usage:**
```python
optimizer = get_optimizer(model.parameters(), config)
```

#### `get_scheduler(optimizer, config)`
Creates LR scheduler (currently StepLR).

**Usage:**
```python
scheduler = get_scheduler(optimizer, config)
```

---

## seeding.py

**Purpose:** Reproducibility through seeding.

### Functions

#### `set_seed(seed, deterministic=False)`
Seeds all random number generators.

**Seeds:**
- Python random
- NumPy
- PyTorch (CPU and CUDA)
- Sets cuDNN behavior

**Usage:**
```python
set_seed(42, deterministic=False)  # Fast, approximate
set_seed(42, deterministic=True)   # Slow, exact
```

#### `seed_worker(worker_id)`
Seeds individual DataLoader worker.

**Usage:**
```python
DataLoader(..., worker_init_fn=seed_worker)
```

---

## test.py

**Purpose:** Model evaluation.

### Functions

#### `test_model(model, dataloader, device, class_names=None)`
Evaluates model on dataset.

**Returns:**
- Overall accuracy
- Per-sample results (true, pred, correct)

**Usage:**
```python
accuracy, results = test_model(
    model=model,
    dataloader=test_loader,
    device='cuda:0',
    class_names=['ants', 'bees']
)
```

---

## trainer.py

**Purpose:** Main training loop.

### Functions

#### `train_model(model, criterion, optimizer, scheduler, dataloaders, ...)`
Complete training loop.

**Responsibilities:**
- Train for all epochs
- Validate after each epoch
- Save checkpoints (best + last)
- Log to TensorBoard
- Update summaries
- Generate metrics

**Usage:**
```python
history = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    dataloaders=dataloaders,
    device=device,
    config=config,
    run_dir='runs/base',
    class_names=['ants', 'bees']
)
```

#### `collect_predictions(model, dataloader, device)`
Collects all predictions for metrics.

---

## Related Documentation

- [Architecture Overview](README.md)
- [Entry Points](entry-points.md)
- [Data Flow](data-flow.md)
- [Design Decisions](design-decisions.md)
