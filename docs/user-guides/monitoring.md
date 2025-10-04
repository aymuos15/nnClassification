# Monitoring Training Guide

Monitor training progress with TensorBoard and logs.

## TensorBoard

### Launch TensorBoard

```bash
# View all runs
tensorboard --logdir runs/

# View specific run
tensorboard --logdir runs/base/tensorboard

# Specify port
tensorboard --logdir runs/base/tensorboard --port 6007
```

Open http://localhost:6006 in your browser.

### Using visualise.py

The `visualise.py` script provides easy TensorBoard management:

```bash
# Launch TensorBoard for a specific run
python visualise.py --mode launch --run_dir runs/base

# Launch on custom port
python visualise.py --mode launch --run_dir runs/base --port 6007
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
python visualise.py --mode samples --run_dir runs/base --split train --num_images 16

# Visualize validation samples
python visualise.py --mode samples --run_dir runs/base --split val --num_images 32

# Visualize test samples
python visualise.py --mode samples --run_dir runs/base --split test --num_images 8
```

This creates image grids in TensorBoard showing:
- Grid view of multiple images
- Individual images organized by class

## Visualizing Model Predictions

Visualize model predictions with colored borders:

```bash
# Visualize predictions on validation set using best checkpoint
python visualise.py --mode predictions --run_dir runs/base --split val --checkpoint best.pt

# Visualize predictions on test set
python visualise.py --mode predictions --run_dir runs/base --split test --checkpoint best.pt --num_images 32

# Use last checkpoint instead
python visualise.py --mode predictions --run_dir runs/base --split val --checkpoint last.pt
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
python visualise.py --mode clean

# Clean specific run
python visualise.py --mode clean --run_dir runs/base
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
python train.py --num_epochs 100

# Terminal 2: Launch TensorBoard
python visualise.py --mode launch --run_dir runs/base
```

Watch metrics update in real-time as training progresses.

## Common Workflows

### Complete Visualization Pipeline

```bash
# 1. Train model
python train.py --batch_size 32 --num_epochs 50

# 2. Visualize training samples
python visualise.py --mode samples --run_dir runs/batch_32 --split train

# 3. Visualize validation predictions
python visualise.py --mode predictions --run_dir runs/batch_32 --split val

# 4. Launch TensorBoard
python visualise.py --mode launch --run_dir runs/batch_32
```

### After Training

```bash
# View training metrics
tensorboard --logdir runs/base/tensorboard

# Check summary
cat runs/base/summary.txt

# Visualize predictions
python visualise.py --mode predictions --run_dir runs/base --split test
```

## Related

- [Training Guide](training.md)
- [Inference Guide](inference.md)
- [Visualization Reference](../reference/visualization.md)
