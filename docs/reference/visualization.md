# Visualization Reference

Complete reference for `ml-visualise` CLI command and TensorBoard visualization.

## ml-visualise CLI Command

### Overview

The `ml-visualise` command provides easy-to-use TensorBoard visualization for:
- Dataset samples
- Model predictions
- TensorBoard server management
- Log cleanup

### Command-line Interface

```bash
ml-visualise --mode <MODE> [OPTIONS]
```

### Modes

#### 1. Launch Mode

Start TensorBoard server for viewing logs.

```bash
ml-visualise --mode launch --run_dir runs/base [--port 6006]
```

**Arguments:**
- `--run_dir`: Run directory containing tensorboard logs (required)
- `--port`: TensorBoard server port (default: 6006)

**Example:**
```bash
# Default port
ml-visualise --mode launch --run_dir runs/base

# Custom port
ml-visualise --mode launch --run_dir runs/batch_32 --port 6007
```

#### 2. Samples Mode

Visualize dataset images in TensorBoard.

```bash
ml-visualise --mode samples --run_dir runs/base \
  [--split train|val|test] [--num_images 16]
```

**Arguments:**
- `--run_dir`: Run directory (required)
- `--split`: Dataset split (default: val, choices: train/val/test)
- `--num_images`: Number of images to visualize (default: 16)

**Example:**
```bash
# Visualize 16 validation samples
ml-visualise --mode samples --run_dir runs/base

# Visualize 32 training samples
ml-visualise --mode samples --run_dir runs/base --split train --num_images 32

# Visualize test samples
ml-visualise --mode samples --run_dir runs/base --split test --num_images 8
```

**TensorBoard Output:**
- `Dataset_Samples/<split>`: Grid view of all images
- `Dataset_Samples/<split>_individual/<class>_<idx>`: Individual images by class

#### 3. Predictions Mode

Visualize model predictions with colored borders.

```bash
ml-visualise --mode predictions --run_dir runs/base \
  [--checkpoint best.pt] [--split val] [--num_images 16]
```

**Arguments:**
- `--run_dir`: Run directory (required)
- `--checkpoint`: Model checkpoint file (default: best.pt)
- `--split`: Dataset split (default: val, choices: train/val/test)
- `--num_images`: Number of images to visualize (default: 16)

**Example:**
```bash
# Predictions with best model on validation set
ml-visualise --mode predictions --run_dir runs/base

# Predictions on test set
ml-visualise --mode predictions --run_dir runs/base --split test

# Use last checkpoint instead
ml-visualise --mode predictions --run_dir runs/base --checkpoint last.pt --num_images 32
```

**Color Coding:**
- 🟢 Green border = Correct prediction
- 🔴 Red border = Incorrect prediction

**TensorBoard Output:**
- `Predictions/<split>`: Grid view with colored borders
- `Predictions/<split>_individual/Correct/<idx>_true_<class>_pred_<class>`: Correct predictions
- `Predictions/<split>_individual/Incorrect/<idx>_true_<class>_pred_<class>`: Incorrect predictions

#### 4. Clean Mode

Remove TensorBoard logs to start fresh.

```bash
ml-visualise --mode clean [--run_dir runs/base]
```

**Arguments:**
- `--run_dir`: Specific run to clean (optional, omit to clean all runs)

**Example:**
```bash
# Clean all TensorBoard logs
ml-visualise --mode clean

# Clean specific run
ml-visualise --mode clean --run_dir runs/base
```

**What Gets Removed:**
- `runs/*/tensorboard/` directories only

**What's Preserved:**
- Model weights (`runs/*/weights/`)
- Log files (`runs/*/logs/`)
- Configuration (`runs/*/config.yaml`)
- Summary (`runs/*/summary.txt`)

### Complete Examples

#### Full Visualization Workflow

```bash
# 1. Train a model
ml-train --batch_size 32 --lr 0.01 --num_epochs 50

# 2. Visualize training data
ml-visualise --mode samples --run_dir runs/batch_32_lr_0.01 --split train --num_images 32

# 3. Visualize validation predictions
ml-visualise --mode predictions --run_dir runs/batch_32_lr_0.01 --split val

# 4. Visualize test predictions
ml-visualise --mode predictions --run_dir runs/batch_32_lr_0.01 --split test

# 5. Launch TensorBoard
ml-visualise --mode launch --run_dir runs/batch_32_lr_0.01
```

#### Compare Multiple Checkpoints

```bash
# Visualize best checkpoint
ml-visualise --mode predictions --run_dir runs/base --checkpoint best.pt

# Clean logs
ml-visualise --mode clean --run_dir runs/base

# Visualize last checkpoint
ml-visualise --mode predictions --run_dir runs/base --checkpoint last.pt

# Launch and compare in TensorBoard
ml-visualise --mode launch --run_dir runs/base
```

## TensorBoard Interface

### Tabs Overview

#### Scalars Tab
- **Loss curves:** Training and validation loss over epochs
- **Accuracy curves:** Training and validation accuracy over epochs
- **Learning rate:** LR schedule visualization

#### Images Tab
- **Dataset samples:** Image grids from each split
- **Predictions:** Model predictions with colored borders
- **Confusion matrices:** Visual confusion matrix heatmaps

#### Text Tab
- **Classification reports:** Detailed precision, recall, F1-scores

### Navigation Tips

1. **Smoothing slider:** Reduce noise in loss/accuracy curves
2. **Run selector:** Filter which runs to display
3. **Refresh button:** Update with latest data
4. **Download data:** Export metrics as CSV/JSON

## Image Visualization Details

### Denormalization

Images are automatically denormalized for proper display using the normalization parameters from your config:

```yaml
transforms:
  train:
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

The script reverses this transformation so images appear natural.

### Grid Layout

Images are arranged in a grid:
- **Default nrow:** 4 images per row
- **Padding:** 2 pixels between images
- **Automatic sizing:** Adapts to image dimensions

### Colored Borders

Prediction visualization adds 5-pixel borders:
- Color determined by prediction accuracy
- Border width: 5 pixels
- Applied before grid creation

## Common Workflows

### Debug Dataset Issues

```bash
# Visualize training data
ml-visualise --mode samples --run_dir runs/base --split train --num_images 32

# Check if images look correct
# - Are transformations applied correctly?
# - Are images properly normalized?
# - Do classes look distinct?
```

### Analyze Model Performance

```bash
# Visualize predictions
ml-visualise --mode predictions --run_dir runs/base --split val --num_images 32

# In TensorBoard, check:
# - Which images are misclassified (red borders)?
# - Are errors systematic?
# - Does model confuse specific classes?
```

### Compare Training Runs

```bash
# Visualize predictions for run 1
ml-visualise --mode predictions --run_dir runs/base

# Visualize predictions for run 2
ml-visualise --mode predictions --run_dir runs/lr_0.01

# Launch TensorBoard for comparison
tensorboard --logdir runs/
```

### Clean Start

```bash
# Remove all old visualizations
ml-visualise --mode clean

# Generate fresh visualizations
ml-visualise --mode samples --run_dir runs/base --split train
ml-visualise --mode predictions --run_dir runs/base --split val

# View updated logs
ml-visualise --mode launch --run_dir runs/base
```

## Troubleshooting

### TensorBoard Not Starting

**Error:** `tensorboard: command not found`

**Solution:**
```bash
uv pip install tensorboard
```

### Images Not Appearing

**Issue:** TensorBoard shows no images

**Solutions:**
1. Ensure visualization completed successfully
2. Refresh browser (F5)
3. Check correct run directory
4. Verify TensorBoard directory exists:
   ```bash
   ls -la runs/base/tensorboard/
   ```

### Wrong Images Displayed

**Issue:** Images look corrupted or wrong colors

**Solutions:**
1. Check normalization parameters in config
2. Ensure dataset images are RGB (not grayscale or RGBA)
3. Verify transforms are correct

### Port Already in Use

**Error:** `TensorBoard port 6006 already in use`

**Solution:**
```bash
# Use different port
ml-visualise --mode launch --run_dir runs/base --port 6007
```

### Clean Doesn't Work

**Issue:** Old data still appears

**Solutions:**
1. Verify clean completed:
   ```bash
   ml-visualise --mode clean --run_dir runs/base
   ```
2. Check tensorboard directory removed:
   ```bash
   ls runs/base/tensorboard  # Should error
   ```
3. Hard refresh browser (Ctrl+Shift+R)

## Technical Details

### Dependencies

Required packages (already in pyproject.toml):
- `torch` - Core framework
- `torchvision` - Image utilities
- `tensorboard` - Visualization backend
- `pillow` - Image processing
- `loguru` - Logging

### Data Flow

**Samples Mode:**
1. Load config from run directory
2. Create datasets using config
3. Get batch from dataloader
4. Denormalize images
5. Create grid with `make_grid()`
6. Log to TensorBoard with `add_image()`

**Predictions Mode:**
1. Load config and model
2. Load checkpoint weights
3. Get batch and run inference
4. Denormalize images
5. Add colored borders (PIL)
6. Create grid
7. Log to TensorBoard

### File Structure

```
runs/base/
├── tensorboard/                    # TensorBoard logs
│   └── events.out.tfevents.*      # Event files
├── weights/
│   ├── best.pt
│   └── last.pt
├── logs/
│   └── train.log
├── config.yaml
└── summary.txt
```

## Integration with Training

### Automatic Logging During Training

Training automatically logs to TensorBoard:
- Every epoch: Loss, accuracy, learning rate
- After training: Confusion matrices, classification reports

### Additional Visualization

Use `ml-visualise` to add:
- Dataset sample views
- Prediction visualizations with color coding
- Multiple checkpoints for comparison

### Combined Workflow

```bash
# Train
ml-train --batch_size 32 --num_epochs 50

# Add visualizations
ml-visualise --mode samples --run_dir runs/batch_32
ml-visualise --mode predictions --run_dir runs/batch_32

# View everything together
ml-visualise --mode launch --run_dir runs/batch_32
```

Now TensorBoard shows:
- Training curves (from training)
- Dataset samples (from ml-visualise samples mode)
- Prediction visualizations (from ml-visualise predictions mode)
- Confusion matrices (from training)

## Related Documentation

- [Monitoring Guide](../user-guides/monitoring.md) - TensorBoard usage
- [Training Guide](../user-guides/training.md) - Training workflows
- [Inference Guide](../user-guides/inference.md) - Model evaluation
- [Architecture: Entry Points](../architecture/entry-points.md) - Script details
