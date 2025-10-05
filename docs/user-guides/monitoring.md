# Monitoring Training Guide

Monitor training progress with TensorBoard and logs.

## TensorBoard

### Launch TensorBoard

```bash
# View all runs
tensorboard --logdir runs/

runs/hymenoptera_base_fold_0/tensorboard

# Specify port
tensorboard --logdir runs/hymenoptera_base_fold_0
```

Open http://localhost:6006 in your browser.

### Using ml-visualise

The `ml-visualise` command provides easy TensorBoard management:

```bash
runs/hymenoptera_base_fold_0
# Launch on custom port
runs/hymenoptera_base_fold_0
```

## What Gets Logged

### During Training

**Scalars (metrics over time):**
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Learning rate

**Images (at end of training):**
- Confusion matrices (train & val)

**Text:**
- Classification reports (train & val)

### Viewing Metrics

1. **Scalars tab:** Loss and accuracy curves
2. **Images tab:** Confusion matrices
3. **Text tab:** Classification reports

## Visualizing Dataset Samples

View sample images from your dataset:

```bash
# Visualize 16 training samples
ml-visualise --mode samples --run_dir runs/hymenoptera_base_fold_0 --split train --num_images 16

# Visualize validation samples
ml-visualise --mode samples --run_dir runs/hymenoptera_base_fold_0 --split val --num_images 32

# Visualize test samples
ml-visualise --mode runs/hymenoptera_base_fold_0 --num_images 8
```

This creates image grids in TensorBoard showing:
- Grid view of multiple images
- Individual images organized by class

## Visualizing Model Predictions

Visualize model predictions with colored borders:

```bash
# Visualize predictions on validation set using best checkpoint
runs/hymenoptera_base_fold_0

# Visualize predictions on test set
ml-visualise --mode predictions --run_dir runs/base --split test --checkpoint best.pt --num_images 32

# Use last checkpoint instead
ml-visualise --mode predictions --run_dir runs/base --split val --checkpoint last.pt
```

**Color coding:**
- 🟢 **Green border** = Correct prediction
- 🔴 **Red border** = Incorrect prediction

Predictions are organized in TensorBoard by:
- Grid view of all predictions
- Individual images organized by Correct/Incorrect status

## Cleaning TensorBoard Logs

Remove TensorBoard logs to start fresh:

```bash
# Clean all runs
ml-visualise --mode clean

# Clean specific run
ml-visualise --mode clean --run_dir runs/base
```

This removes only TensorBoard logs, preserving:
- Model weights
- Training logs
- Configuration files
- Metrics reports

## File-based Logging

### Log Files

Training creates log files in `runs/<run_name>/logs/`:

```
runs/base/logs/
├── train.log                           # Training log
├── inference.log                       # Inference log
├── classification_report_train.txt     # Train metrics
├── classification_report_val.txt       # Validation metrics
└── classification_report_test.txt      # Test metrics
```

### View Logs

```bash
# View training log
cat runs/base/logs/train.log

# Follow training in real-time
tail -f runs/base/logs/train.log

# View classification report
cat runs/base/logs/classification_report_val.txt
```

### Summary File

Each run includes `summary.txt` with key information:

```bash
cat runs/base/summary.txt
```

Contains:
- Configuration
- Dataset sizes
- Model parameters
- Training duration
- Best accuracy
- Final metrics

## Tips

### Compare Multiple Runs

```bash
# View all runs together
tensorboard --logdir runs/

# Compare specific runs
tensorboard --logdir_spec \
  base:runs/base/tensorboard,\
  lr_001:runs/lr_0.01/tensorboard
```

### Refresh TensorBoard

If new data doesn't appear:
1. Refresh browser (F5)
2. Check TensorBoard is reading correct directory
3. Ensure training/visualization completed successfully

### Monitor During Training

```bash
# Terminal 1: Start training
ml-train --num_epochs 100

# Terminal 2: Launch TensorBoard
ml-visualise --mode launch --run_dir runs/base
```

Watch metrics update in real-time as training progresses.

## Common Workflows

### Complete Visualization Pipeline

```bash
# 1. Train model
ml-train --batch_size 32 --num_epochs 50

# 2. Visualize training samples
ml-visualise --mode samples --run_dir runs/batch_32 --split train

# 3. Visualize validation predictions
ml-visualise --mode predictions --run_dir runs/batch_32 --split val

# 4. Launch TensorBoard
ml-visualise --mode launch --run_dir runs/batch_32
```

### After Training

```bash
# View training metrics
tensorboard --logdir runs/base/tensorboard

# Check summary
cat runs/base/summary.txt

# Visualize predictions
ml-visualise --mode predictions --run_dir runs/base --split test
```

## Related

- [Training Guide](training.md)
- [Inference Guide](inference.md)
- [Visualization Reference](../reference/visualization.md)
