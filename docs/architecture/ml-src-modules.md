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
| `test.py` | Evaluation | evaluate_model (legacy wrapper) |
| `inference/` | Inference strategies | StandardInference, MixedPrecisionInference, AccelerateInference |
| `trainers/` | Training strategies | StandardTrainer, MixedPrecisionTrainer, AccelerateTrainer, DPTrainer |

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

**Purpose:** Legacy evaluation wrapper (backward compatibility).

**Note:** This module is maintained for backward compatibility. New code should use the inference strategies in `ml_src.core.inference` instead.

### Functions

#### `evaluate_model(model, dataloader, dataset_size, device, class_names=None)`
Evaluates model on dataset using StandardInference internally.

**Returns:**
- Overall accuracy
- Per-sample results (true, pred, correct)

**Usage:**
```python
from ml_src.core.test import evaluate_model

accuracy, results = evaluate_model(
    model=model,
    dataloader=test_loader,
    dataset_size=len(test_dataset),
    device='cuda:0',
    class_names=['ants', 'bees']
)
```

**Recommended:** Use inference strategies directly:
```python
from ml_src.core.inference import get_inference_strategy

strategy = get_inference_strategy(config)
accuracy, results = strategy.run_inference(
    model, test_loader, len(test_dataset), device, class_names
)
```

---

## inference/

**Purpose:** Specialized inference strategies for different hardware and performance requirements.

### Module Structure

```
inference/
├── __init__.py           # get_inference_strategy() factory
├── base.py               # BaseInferenceStrategy abstract class
├── standard.py           # StandardInference
├── mixed_precision.py    # MixedPrecisionInference (PyTorch AMP)
└── accelerate.py         # AccelerateInference (multi-GPU)
```

### Factory Function

#### `get_inference_strategy(config)`
Creates appropriate inference strategy based on config.

**Config:**
```yaml
inference:
  strategy: 'standard'  # or 'mixed_precision', 'accelerate'
  amp_dtype: 'float16'  # for mixed_precision
```

**Usage:**
```python
from ml_src.core.inference import get_inference_strategy

strategy = get_inference_strategy(config)
test_acc, results = strategy.run_inference(
    model=model,
    dataloader=test_loader,
    dataset_size=len(test_dataset),
    device=device,
    class_names=['ants', 'bees']
)
```

### Inference Strategies

#### StandardInference
Standard PyTorch inference without optimizations.

**Use when:**
- Running on CPU
- Simplicity is priority

#### MixedPrecisionInference
Uses PyTorch AMP for 2-3x faster inference.

**Use when:**
- Running on modern NVIDIA GPU
- Speed is important
- Recommended for production on GPU

**Performance:**
- 2-3x faster than standard
- 50% less memory usage
- Minimal accuracy impact

#### AccelerateInference
Distributed inference across multiple GPUs.

**Use when:**
- Multiple GPUs available
- Very large test sets

**Requirements:**
```bash
pip install accelerate
```

---

## trainers/

**Purpose:** Specialized training strategies for different hardware and requirements.

### Module Structure

```
trainers/
├── __init__.py              # get_trainer() factory
├── base.py                  # BaseTrainer abstract class
├── standard.py              # StandardTrainer
├── mixed_precision.py       # MixedPrecisionTrainer (PyTorch AMP)
├── accelerate.py            # AccelerateTrainer (multi-GPU)
└── differential_privacy.py  # DPTrainer (Opacus)
```

### Factory Function

#### `get_trainer(config, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, run_dir, class_names)`
Creates appropriate trainer based on config.

**Config:**
```yaml
training:
  trainer_type: 'standard'  # or 'mixed_precision', 'accelerate', 'dp'

  # Mixed precision settings
  amp_dtype: 'float16'

  # Accelerate settings
  gradient_accumulation_steps: 1

  # Differential privacy settings
  dp:
    noise_multiplier: 1.1
    max_grad_norm: 1.0
    target_epsilon: 3.0
    target_delta: 1e-5
```

**Usage:**
```python
from ml_src.core.trainers import get_trainer

trainer = get_trainer(
    config=config,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    dataloaders=dataloaders,
    dataset_sizes=dataset_sizes,
    device=device,
    run_dir=run_dir,
    class_names=class_names
)

model, train_losses, val_losses, train_accs, val_accs = trainer.train()
```

### Training Strategies

#### StandardTrainer
Traditional PyTorch training.

**Use when:**
- Beginners or CPU training
- Simple workflows
- Default choice

#### MixedPrecisionTrainer
PyTorch AMP for 2-3x faster training.

**Use when:**
- Single modern GPU
- Recommended for most production use cases

**Performance:**
- 2-3x faster than standard
- 50% less memory usage
- Minimal accuracy impact

#### AccelerateTrainer
Multi-GPU/distributed training with Hugging Face Accelerate.

**Use when:**
- Multiple GPUs available
- Distributed training needed

**Requirements:**
```bash
pip install accelerate
accelerate config  # One-time setup
```

**Launch:**
```bash
accelerate launch ml-train --config config.yaml
```

#### DPTrainer
Differential privacy training with Opacus.

**Use when:**
- Privacy-sensitive data
- Formal privacy guarantees required

**Requirements:**
```bash
pip install opacus
```

**Features:**
- Privacy budget tracking (epsilon, delta)
- DP-SGD algorithm
- Gradient clipping
- Per-sample gradient computation

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
