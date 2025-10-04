# Entry Points

## Overview

The framework has three main entry points that provide complete workflows for training, inference, and visualization. These scripts orchestrate all other components and provide the primary user interface.

---

## `train.py` - Training Script

### Purpose

Orchestrates the complete training pipeline from configuration to final evaluation.

### Key Responsibilities

1. **Parse CLI arguments** and load/override configuration
2. **Create organized run directories** based on hyperparameter overrides
3. **Setup logging infrastructure** (console + file)
4. **Initialize all components:**
   - Datasets and DataLoaders
   - Model architecture
   - Optimizer and scheduler
   - Loss function
5. **Execute training loop** via `trainer.py` with TensorBoard logging
6. **Support training resumption** from checkpoints

---

### CLI Arguments

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
  --gamma 0.1 \                          # Override LR gamma
  --device cuda:0                        # Override device
```

**All arguments are optional.** Defaults come from config file.

---

### Execution Flow

#### 1. Initialization Phase

```python
# Load and parse arguments
args = parse_args()

# Load base configuration
config = load_config(args.config)

# Apply CLI overrides
config = apply_overrides(config, args)

# Set reproducibility
set_seed(config['seed'])
if config['deterministic']:
    set_deterministic_mode()
```

#### 2. Run Directory Creation

```python
# Generate run name from overrides
run_name = generate_run_name(overrides)  # e.g., "batch_32_lr_0.01"

# Create directory structure
run_dir = Path('runs') / run_name
run_dir.mkdir(parents=True, exist_ok=True)

# Save configuration
save_config(config, run_dir / 'config.yaml')
```

**Run Naming Logic:**
- No overrides → `base/`
- Single override → `batch_32/`
- Multiple overrides → `batch_32_epochs_50_lr_0.01/`

#### 3. Logging Setup

```python
# Console logging (color-coded, INFO level)
logger.add(sys.stdout, level="INFO", colorize=True)

# File logging (detailed, DEBUG level)
log_file = run_dir / 'logs' / 'train.log'
logger.add(log_file, level="DEBUG", rotation="10 MB", retention="30 days")
```

#### 4. Data Preparation

```python
# Create datasets
datasets = get_datasets(config)
# Returns: {'train': Dataset, 'val': Dataset, 'test': Dataset}

# Create dataloaders
dataloaders = get_dataloaders(datasets, config)
# Returns: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}

# Get class names
class_names = get_class_names(datasets['train'])
```

#### 5. Model Initialization

```python
# Load model (base or custom)
model = get_model(config, device)

# Display model info
logger.info(f"Model: {config['model']['architecture']}")
logger.info(f"Parameters: {count_parameters(model):,}")
```

#### 6. Optimizer & Scheduler Setup

```python
# Create optimizer (e.g., SGD with momentum)
optimizer = get_optimizer(model.parameters(), config)

# Create LR scheduler (e.g., StepLR)
scheduler = get_scheduler(optimizer, config)
```

#### 7. Loss Function

```python
# Get loss criterion (e.g., CrossEntropyLoss)
criterion = get_criterion(config)
```

#### 8. Resume Training (if requested)

```python
if args.resume:
    checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    logger.info(f"Resumed from epoch {checkpoint['epoch']}")
else:
    start_epoch = 0
    best_acc = 0.0
```

#### 9. Training Execution

```python
# Train model
history = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    dataloaders=dataloaders,
    device=device,
    config=config,
    run_dir=run_dir,
    class_names=class_names,
    start_epoch=start_epoch,
    best_acc=best_acc
)
```

**Inside `train_model()` (from `trainer.py`):**
- Training loop for all epochs
- Validation after each epoch
- Checkpoint saving (best + last)
- TensorBoard logging
- Summary updates

#### 10. Post-Training

```python
# Load best model
load_model(model, run_dir / 'weights' / 'best.pt', device)

# Generate final metrics
save_confusion_matrix(...)
save_classification_report(...)

# Update final summary
save_summary(run_dir, status='completed', ...)

logger.info("Training complete!")
```

---

### Error Handling

```python
try:
    # Main training logic
    ...
except KeyboardInterrupt:
    logger.warning("Training interrupted by user")
    save_summary(run_dir, status='interrupted', ...)
except Exception as e:
    logger.error(f"Training failed: {e}")
    save_summary(run_dir, status='failed', error=str(e), ...)
    raise
```

**Ensures:**
- Partial progress is saved
- Error messages captured
- Clean shutdown

---

### Output Artifacts

After training, `runs/{run_name}/` contains:

```
runs/{run_name}/
├── config.yaml              # Final configuration
├── summary.txt              # Training summary
├── weights/
│   ├── best.pt             # Best model
│   └── last.pt             # Latest checkpoint
├── logs/
│   ├── train.log           # Detailed log
│   ├── classification_report_train.txt
│   └── classification_report_val.txt
├── plots/
│   ├── confusion_matrix_train.png
│   └── confusion_matrix_val.png
└── tensorboard/
    └── events.out.tfevents.*
```

---

## `inference.py` - Inference Script

### Purpose

Load trained models and evaluate on test data with comprehensive metrics.

### Key Responsibilities

1. **Load saved configuration** and checkpoint from run directory
2. **Run model on test dataset**
3. **Generate per-sample predictions**
4. **Create confusion matrices** and classification reports
5. **Display rich formatted results** (tables, summaries)

---

### CLI Arguments

```bash
python inference.py \
  --run_dir runs/base \          # Run directory to load from
  --checkpoint best.pt \          # Which checkpoint (best.pt or last.pt)
  --data_dir data/custom          # Override data directory (optional)
```

**Required:** `--run_dir`  
**Optional:** `--checkpoint` (default: `best.pt`), `--data_dir`

---

### Execution Flow

#### 1. Load Configuration

```python
# Load configuration from run directory
config_path = Path(args.run_dir) / 'config.yaml'
config = load_config(config_path)

# Override data_dir if specified
if args.data_dir:
    config['data']['data_dir'] = args.data_dir
```

#### 2. Setup Logging

```python
# Console + file logging
log_file = Path(args.run_dir) / 'logs' / 'inference.log'
logger.add(log_file, level="INFO")
```

#### 3. Load Data

```python
# Create test dataset
datasets = get_datasets(config)
test_loader = get_dataloaders(datasets, config)['test']

# Get class names
class_names = get_class_names(datasets['train'])
```

#### 4. Load Model

```python
# Initialize model architecture
model = get_model(config, device)

# Load trained weights
checkpoint_path = Path(args.run_dir) / 'weights' / args.checkpoint
load_model(model, checkpoint_path, device)

logger.info(f"Loaded model from {checkpoint_path}")
```

#### 5. Run Inference

```python
# Evaluate on test set
test_acc, results = test_model(
    model=model,
    dataloader=test_loader,
    device=device,
    class_names=class_names
)

# results = list of (true_label, pred_label, correct)
```

#### 6. Generate Metrics

```python
# Confusion matrix
save_confusion_matrix(
    results,
    class_names,
    Path(args.run_dir) / 'plots' / 'confusion_matrix_test.png'
)

# Classification report
save_classification_report(
    results,
    class_names,
    Path(args.run_dir) / 'logs' / 'classification_report_test.txt'
)
```

#### 7. Display Results

```python
# Print summary
logger.info(f"Test Accuracy: {test_acc:.2f}%")

# Show per-class metrics (from classification report)
print_classification_report(results, class_names)

# Display confusion matrix path
logger.info(f"Confusion matrix saved to: {plot_path}")
```

---

### Output

**Console Output:**
```
[INFO] Loading model from runs/base/weights/best.pt
[INFO] Running inference on test set...
[INFO] Test Accuracy: 92.5%

Classification Report:
               precision    recall  f1-score   support
        ants       0.90      0.94      0.92        50
        bees       0.94      0.91      0.93        50

    accuracy                           0.93       100
   macro avg       0.92      0.93      0.92       100
weighted avg       0.92      0.93      0.92       100

[INFO] Confusion matrix saved to: runs/base/plots/confusion_matrix_test.png
[INFO] Classification report saved to: runs/base/logs/classification_report_test.txt
```

**Generated Files:**
```
runs/{run_name}/
├── logs/
│   ├── inference.log
│   └── classification_report_test.txt
└── plots/
    └── confusion_matrix_test.png
```

---

## Common Usage Patterns

### Pattern 1: Basic Training

```bash
# Train with defaults
python train.py

# Results in: runs/base/
```

### Pattern 2: Hyperparameter Experiment

```bash
# Try different learning rates
python train.py --lr 0.001
python train.py --lr 0.01
python train.py --lr 0.1

# Creates: runs/lr_0.001/, runs/lr_0.01/, runs/lr_0.1/
```

### Pattern 3: Training Interruption & Resumption

```bash
# Start training
python train.py --num_epochs 100

# Interrupted at epoch 47 (Ctrl+C or crash)

# Resume training
python train.py --resume runs/base/last.pt

# Continues from epoch 48
```

### Pattern 4: Different Datasets

```bash
# Train on dataset 1
python train.py --data_dir data/dataset1

# Train on dataset 2
python train.py --data_dir data/dataset2

# Both create runs/base/ (data_dir doesn't affect run name)
# Tip: Combine with other overrides for unique names
python train.py --data_dir data/dataset1 --num_epochs 25
python train.py --data_dir data/dataset2 --num_epochs 50
```

### Pattern 5: Full Workflow

```bash
# 1. Train model
python train.py --batch_size 32 --lr 0.01 --num_epochs 50

# Creates: runs/batch_32_lr_0.01_epochs_50/

# 2. Evaluate on test set
python inference.py --run_dir runs/batch_32_lr_0.01_epochs_50

# 3. View training curves
tensorboard --logdir runs/batch_32_lr_0.01_epochs_50/tensorboard
```

---

## Implementation Tips

### For Developers Modifying Entry Points

**train.py modifications:**
1. **Adding CLI argument:**
   ```python
   parser.add_argument('--new_arg', type=int, default=10)
   ```

2. **Applying override:**
   ```python
   if args.new_arg is not None:
       config['section']['new_arg'] = args.new_arg
   ```

3. **Including in run name:**
   ```python
   if args.new_arg != config_defaults['new_arg']:
       overrides.append(f'new_arg_{args.new_arg}')
   ```

**inference.py modifications:**
1. Usually no changes needed (loads from saved config)
2. Add CLI args only for runtime overrides (like `data_dir`)

---

## Best Practices

### For train.py

1. **Always use `--resume` for interrupted training**
   - Don't restart from scratch
   - Preserves training history

2. **Name runs meaningfully**
   - Use multiple overrides for unique names
   - Example: `--lr 0.01 --batch_size 32`

3. **Check config before long training**
   ```bash
   cat runs/{run_name}/config.yaml
   ```

4. **Monitor with TensorBoard**
   ```bash
   tensorboard --logdir runs/
   ```

### For inference.py

1. **Use `best.pt` for final evaluation**
   - Highest validation accuracy
   - Best for deployment

2. **Use `last.pt` for debugging**
   - Latest state
   - Useful for checking training progress

3. **Keep test set pristine**
   - Don't use for hyperparameter tuning
   - Final evaluation only

---

## `visualise.py` - Visualization Script

### Purpose

Provide easy TensorBoard visualization of datasets, model predictions, and training metrics.

### Key Responsibilities

1. **Launch TensorBoard server** for viewing training logs
2. **Visualize dataset samples** in image grids
3. **Visualize model predictions** with colored borders (green=correct, red=incorrect)
4. **Clean TensorBoard logs** for fresh starts

---

### CLI Arguments

```bash
python visualise.py \
  --mode launch|samples|predictions|clean \  # Visualization mode (required)
  --run_dir runs/base \                      # Run directory
  --split train|val|test \                   # Dataset split
  --num_images 16 \                          # Number of images
  --checkpoint best.pt \                     # Model checkpoint
  --port 6006                                # TensorBoard port
```

**Required:** `--mode`
**Optional:** All others (defaults provided)

---

### Modes

#### 1. Launch Mode

Start TensorBoard server:

```bash
python visualise.py --mode launch --run_dir runs/base --port 6006
```

**Purpose:** Launch TensorBoard to view existing logs

**Output:** TensorBoard web interface at http://localhost:6006

#### 2. Samples Mode

Visualize dataset images:

```bash
python visualise.py --mode samples --run_dir runs/base --split train --num_images 16
```

**Purpose:** Log dataset images to TensorBoard for inspection

**Output:**
- Image grids in TensorBoard
- Individual images organized by class

**Use Cases:**
- Verify data loading works correctly
- Check image transformations
- Inspect dataset quality

#### 3. Predictions Mode

Visualize model predictions:

```bash
python visualise.py --mode predictions --run_dir runs/base --split val --checkpoint best.pt
```

**Purpose:** Visualize model predictions with color-coded correctness

**Output:**
- Images with green borders (correct) or red borders (incorrect)
- Grid view and individual images
- Organized by Correct/Incorrect in TensorBoard

**Use Cases:**
- Identify misclassified examples
- Analyze failure patterns
- Compare different checkpoints

#### 4. Clean Mode

Remove TensorBoard logs:

```bash
python visualise.py --mode clean --run_dir runs/base  # Clean specific run
python visualise.py --mode clean                      # Clean all runs
```

**Purpose:** Remove TensorBoard logs while preserving weights and other artifacts

**What's Removed:** `runs/*/tensorboard/` directories

**What's Preserved:** Weights, logs, configs, summaries

---

### Execution Flow

#### Samples Mode Flow

```python
# Load configuration
config = load_config(run_dir / 'config.yaml')

# Create datasets
datasets = get_datasets(config)
dataloaders = get_dataloaders(datasets, config)

# Get batch of images
images, labels = next(iter(dataloaders[split]))

# Denormalize for display
mean, std = config['transforms'][split]['normalize']
images_denorm = denormalize(images, mean, std)

# Create grid
grid = torchvision.utils.make_grid(images_denorm, nrow=4)

# Log to TensorBoard
writer = SummaryWriter(run_dir / 'tensorboard')
writer.add_image(f'Dataset_Samples/{split}', grid, 0)
```

#### Predictions Mode Flow

```python
# Load model
model = get_model(config, device)
model = load_model(model, checkpoint_path, device)

# Get predictions
images, labels = next(iter(dataloader))
outputs = model(images)
preds = torch.max(outputs, 1)

# Denormalize
images_denorm = denormalize(images, mean, std)

# Add colored borders
for img, true_label, pred_label in zip(images_denorm, labels, preds):
    is_correct = (true_label == pred_label)
    color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green or Red
    bordered_img = add_colored_border(img, color, border_width=5)

# Create grid and log
grid = torchvision.utils.make_grid(bordered_images, nrow=4)
writer.add_image(f'Predictions/{split}', grid, 0)
```

---

### Features

**Automatic Denormalization:**
- Images are denormalized using config normalization parameters
- Ensures natural appearance in TensorBoard

**Color-Coded Predictions:**
- 🟢 Green border = Correct prediction
- 🔴 Red border = Incorrect prediction
- 5-pixel border width

**Grid Layout:**
- 4 images per row by default
- 2-pixel padding between images
- Adapts to image dimensions

**Organized Output:**
- Individual images tagged by class
- Predictions organized by Correct/Incorrect
- Easy navigation in TensorBoard

---

### Use Cases

#### Data Debugging

```bash
# Check if training data looks correct
python visualise.py --mode samples --run_dir runs/base --split train --num_images 32

# Verify transformations
python visualise.py --mode samples --run_dir runs/base --split val --num_images 16
```

#### Model Analysis

```bash
# Identify misclassified examples
python visualise.py --mode predictions --run_dir runs/base --split test

# Compare best vs last checkpoint
python visualise.py --mode predictions --run_dir runs/base --checkpoint best.pt
python visualise.py --mode clean --run_dir runs/base
python visualise.py --mode predictions --run_dir runs/base --checkpoint last.pt
```

#### Complete Visualization

```bash
# Full workflow
python train.py --batch_size 32 --num_epochs 50
python visualise.py --mode samples --run_dir runs/batch_32 --split train
python visualise.py --mode predictions --run_dir runs/batch_32 --split val
python visualise.py --mode launch --run_dir runs/batch_32
```

---

## Related Documentation

- [ML Source Modules](ml-src-modules.md) - Components called by entry points
- [Data Flow](data-flow.md) - How data moves through training/inference
- [Configuration](../configuration/overview.md) - Config system details
- [Training Guide](../user-guides/training.md) - Training workflows
- [Monitoring Guide](../user-guides/monitoring.md) - TensorBoard and visualization
- [Visualization Reference](../reference/visualization.md) - Complete visualise.py reference

---

## Summary

**train.py:**
- ✅ Orchestrates complete training pipeline
- ✅ Handles configuration and CLI overrides
- ✅ Creates organized run directories
- ✅ Supports resumption
- ✅ Comprehensive logging and checkpointing

**inference.py:**
- ✅ Loads trained models
- ✅ Evaluates on test data
- ✅ Generates metrics and visualizations
- ✅ Rich formatted output
- ✅ Can override data directory

**visualise.py:**
- ✅ TensorBoard server management
- ✅ Dataset sample visualization
- ✅ Model prediction visualization with color coding
- ✅ Clean mode for fresh starts
- ✅ Automatic image denormalization

**All scripts:**
- Clean, focused interfaces
- Proper error handling
- Complete artifact preservation
- User-friendly output
