# Training Configuration

## Overview

The `training` section controls core training parameters that affect training speed, memory usage, and convergence behavior.

## Configuration Parameters

```yaml
training:
  batch_size: <int>
  num_epochs: <int>
  device: <string>
```

---

## `batch_size`

- **Type:** Integer (> 0)
- **Default:** `4`
- **Description:** Number of samples per training batch
- **Purpose:** Controls memory usage, training speed, and gradient quality

### Usage

```yaml
training:
  batch_size: 4
```

### CLI Override

```bash
ml-train --batch_size 32
```

### Trade-offs

| Aspect | Small Batches (1-8) | Medium Batches (16-64) | Large Batches (128+) |
|--------|-------------------|----------------------|-------------------|
| Memory Usage | Low | Medium | High |
| Training Speed | Slow | Medium | Fast |
| Gradient Noise | High | Medium | Low |
| Generalization | Better | Balanced | May overfit |
| Optimal Learning Rate | Lower | Medium | Higher |

### Understanding Batch Size

**Small Batches (1-8):**
- ✅ Fits on any hardware
- ✅ Better generalization (noisy gradients act as regularization)
- ✅ Lower memory usage
- ❌ Slower training
- ❌ May be unstable

**Medium Batches (16-64):**
- ✅ Good balance of speed and quality
- ✅ Reasonable memory usage
- ✅ Stable training
- ✅ Works well for most problems

**Large Batches (128+):**
- ✅ Fastest training
- ✅ Maximum GPU utilization
- ✅ Stable gradients
- ❌ High memory usage
- ❌ May overfit or converge to sharp minima
- ❌ Requires higher learning rates

### Hardware Considerations

**GPU Memory Limited:**
- Start with 4-8, increase until OOM (Out Of Memory)
- Reduce image resolution if needed
- Consider gradient accumulation (not currently implemented)

**Fast GPU (A100, V100, RTX 3090):**
- Use 64-128 to maximize utilization
- GPU has plenty of memory

**Consumer GPU (GTX 1080, RTX 2060):**
- Typically 16-32
- Balance between speed and memory

**CPU Training:**
- 4-16 depending on RAM
- Batch size less critical on CPU

### Best Practices

1. **Use powers of 2** (8, 16, 32, 64)
   - Often faster due to hardware optimization

2. **Adjust learning rate with batch size**
   - Larger batches → higher learning rate
   - Rule of thumb: LR ∝ √batch_size
   - Example: batch 8 with LR 0.001 → batch 32 with LR 0.002

3. **Find maximum batch size**
   ```bash
   # Start small, then double until OOM
   ml-train --batch_size 4
   ml-train --batch_size 8
   ml-train --batch_size 16
   ml-train --batch_size 32  # OOM! Use 16
   ```

4. **Monitor GPU memory**
   ```bash
   watch -n 1 nvidia-smi
   ```

### Examples by Hardware

```yaml
# RTX 3090 (24GB)
training:
  batch_size: 64

# RTX 2080 (11GB)
training:
  batch_size: 32

# GTX 1080 (8GB)
training:
  batch_size: 16

# CPU training
training:
  batch_size: 8
```

---

## `num_epochs`

- **Type:** Integer (> 0)
- **Default:** `3`
- **Description:** Total number of training epochs
- **Purpose:** Controls total training time and model convergence

### Usage

```yaml
training:
  num_epochs: 25
```

### CLI Override

```bash
ml-train --num_epochs 50
```

### Typical Values

| Use Case | Epochs | Notes |
|----------|--------|-------|
| Quick testing | 3-5 | Just to verify code works |
| Small datasets | 20-50 | Fewer samples, more epochs needed |
| Medium datasets | 50-100 | Most common range |
| Large datasets | 100-200 | More data, needs more training time |
| From scratch training | 200-300 | No pretrained weights |

### How Many Epochs Do You Need?

**Too Few Epochs:**
- Model hasn't converged
- Validation accuracy still improving
- Underfitting

**Optimal Epochs:**
- Validation accuracy plateaus
- Train/val performance balanced
- Sweet spot before overfitting

**Too Many Epochs:**
- Validation accuracy decreases
- Training accuracy keeps improving (overfitting)
- Wasting compute time

### Monitoring Convergence

Watch for these signs:

```
Epoch 1:  train_loss=0.693, val_loss=0.685  # Still learning
Epoch 5:  train_loss=0.512, val_loss=0.498  # Good progress
Epoch 10: train_loss=0.231, val_loss=0.245  # Converging
Epoch 15: train_loss=0.123, val_loss=0.156  # Nearly converged
Epoch 20: train_loss=0.087, val_loss=0.158  # Plateau (can stop)
Epoch 25: train_loss=0.054, val_loss=0.162  # Overfitting! (stop)
```

**Stop when:**
- Validation loss stops decreasing
- Validation accuracy plateaus
- Gap between train and val widens (overfitting)

### Best Practices

1. **Start conservatively**
   - Use 20-30 epochs initially
   - Check if converged, extend if needed

2. **Use early stopping** (see [Early Stopping](#early-stopping))
   - Automatically stop when validation plateaus
   - Prevents wasting compute

3. **Resume from checkpoint**
   ```bash
   # Train 25 epochs
   ml-train --num_epochs 25

   # If not converged, resume and train 25 more
   ml-train --resume runs/base/last.pt --num_epochs 50
   ```

4. **Monitor TensorBoard**
   ```bash
   tensorboard --logdir runs/
   ```
   - Watch loss curves
   - Stop when curves flatten

---

## `device`

- **Type:** String
- **Default:** `'cuda:0'`
- **Description:** Device to run training on
- **Purpose:** Select computation hardware (GPU or CPU)

### Usage

```yaml
training:
  device: 'cuda:0'  # First GPU
  # OR
  device: 'cuda:1'  # Second GPU
  # OR
  device: 'cpu'     # CPU only
```

### Valid Values

- `'cuda:0'`, `'cuda:1'`, etc. - Specific GPU
- `'cuda'` - Default CUDA device
- `'cpu'` - CPU only

### Auto-Fallback

The framework automatically falls back to CPU if CUDA is unavailable:

```python
if device_str.startswith('cuda') and torch.cuda.is_available():
    device = torch.device(device_str)
else:
    device = torch.device('cpu')
```

### GPU Selection

**Check available GPUs:**
```bash
nvidia-smi
```

**Use specific GPU:**
```bash
# Use second GPU
ml-train --device cuda:1
```

**Set GPU via environment variable:**
```bash
# Make only GPU 1 visible
CUDA_VISIBLE_DEVICES=1 ml-train

# Make GPUs 0 and 2 visible
CUDA_VISIBLE_DEVICES=0,2 ml-train
```

### When to Use CPU

✅ **No GPU available**
- Laptop without discrete GPU
- Server without CUDA

✅ **Debugging**
- Easier to debug on CPU
- More informative error messages

✅ **Small models/datasets**
- Training time acceptable on CPU

❌ **Production training**
- GPU is 10-100x faster
- Always use GPU if available

### Performance Comparison

| Device | Relative Speed | When to Use |
|--------|---------------|-------------|
| CPU | 1x (baseline) | No GPU, debugging |
| GTX 1080 | ~20x | Consumer GPU |
| RTX 2080 Ti | ~40x | Enthusiast GPU |
| RTX 3090 | ~60x | High-end consumer |
| A100 | ~100x | Data center GPU |

### Example Configurations

**GPU training (most common):**
```yaml
training:
  device: 'cuda:0'
  batch_size: 32
  num_epochs: 50
```

**CPU training:**
```yaml
training:
  device: 'cpu'
  batch_size: 8      # Smaller batch for CPU
  num_epochs: 10     # Fewer epochs (CPU is slow)
```

**Multi-GPU selection:**
```yaml
training:
  device: 'cuda:1'   # Use second GPU (first might be busy)
  batch_size: 32
  num_epochs: 50
```

---

## Complete Examples

### Example 1: Quick Testing

```yaml
training:
  batch_size: 4
  num_epochs: 5
  device: 'cuda:0'
```

### Example 2: Production Training

```yaml
training:
  batch_size: 64
  num_epochs: 100
  device: 'cuda:0'
```

### Example 3: CPU Training

```yaml
training:
  batch_size: 8
  num_epochs: 10
  device: 'cpu'
```

### Example 4: High-Memory GPU

```yaml
training:
  batch_size: 128   # Large batch for A100
  num_epochs: 50
  device: 'cuda:0'
```

## Troubleshooting

### Out of Memory (OOM)

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `batch_size` (halve it)
2. Reduce image resolution in transforms
3. Use CPU (slower but works)
4. Close other programs using GPU

### Slow Training

**Problem:** Training taking too long

**Solutions:**
1. Increase `batch_size` (if memory allows)
2. Use GPU instead of CPU
3. Increase `num_workers` in data config
4. Reduce `num_epochs` initially
5. Use smaller model

### Not Converging

**Problem:** Loss not decreasing

**Solutions:**
1. Train more epochs
2. Adjust learning rate (see optimizer config)
3. Check data preparation
4. Try smaller batch size
5. Check for bugs in data/model

## Best Practices Summary

1. **Batch Size:**
   - Start with 16-32
   - Maximize without OOM
   - Use powers of 2

2. **Num Epochs:**
   - Monitor validation curves
   - Stop when plateau reached
   - Can always resume and extend

3. **Device:**
   - Always use GPU if available
   - Use CPU only for debugging or necessity
   - Monitor GPU utilization

---

## `early_stopping`

- **Type:** Dictionary
- **Default:** `enabled: false`
- **Description:** Configuration for early stopping to prevent overfitting
- **Purpose:** Automatically stop training when validation metric stops improving

### Overview

Early stopping monitors a validation metric (accuracy or loss) and stops training when the metric hasn't improved for a specified number of epochs (patience). This prevents overfitting and saves compute time.

### Configuration Parameters

```yaml
training:
  early_stopping:
    enabled: false         # Enable/disable early stopping
    patience: 10           # Epochs to wait for improvement
    metric: 'val_acc'      # Metric to monitor
    mode: 'max'            # 'max' or 'min'
    min_delta: 0.0         # Minimum improvement threshold
```

### `enabled`

- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable or disable early stopping
- **Usage:**
  ```yaml
  training:
    early_stopping:
      enabled: true
  ```

### `patience`

- **Type:** Integer (> 0)
- **Default:** `10`
- **Description:** Number of epochs to wait for improvement before stopping
- **Usage:**
  ```yaml
  training:
    early_stopping:
      patience: 15  # Wait 15 epochs
  ```

**Typical Values:**
- Small datasets: 5-10 epochs
- Medium datasets: 10-20 epochs
- Large datasets: 20-30 epochs
- Conservative (less likely to stop early): 30-50 epochs

### `metric`

- **Type:** String
- **Default:** `'val_acc'`
- **Description:** Which metric to monitor
- **Options:** `'val_acc'` or `'val_loss'`
- **Usage:**
  ```yaml
  training:
    early_stopping:
      metric: 'val_loss'  # Monitor validation loss instead
  ```

**When to use each:**
- `'val_acc'`: Most common, directly measures model performance
- `'val_loss'`: Better for imbalanced datasets or when loss is more stable

### `mode`

- **Type:** String
- **Default:** `'max'`
- **Description:** Whether metric should increase or decrease
- **Options:**
  - `'max'`: Higher is better (for accuracy)
  - `'min'`: Lower is better (for loss)
- **Usage:**
  ```yaml
  training:
    early_stopping:
      metric: 'val_loss'
      mode: 'min'  # Stop when loss stops decreasing
  ```

**Mode should match metric:**
- `val_acc` → `mode: 'max'`
- `val_loss` → `mode: 'min'`

### `min_delta`

- **Type:** Float (>= 0)
- **Default:** `0.0`
- **Description:** Minimum change in metric to qualify as improvement
- **Usage:**
  ```yaml
  training:
    early_stopping:
      min_delta: 0.001  # Require at least 0.1% improvement
  ```

**When to use:**
- `0.0`: Count any improvement (default, recommended)
- `0.001-0.01`: Require meaningful improvement, reduces sensitivity to noise

### Complete Examples

**Example 1: Monitor validation accuracy (most common)**
```yaml
training:
  num_epochs: 100
  early_stopping:
    enabled: true
    patience: 10
    metric: 'val_acc'
    mode: 'max'
    min_delta: 0.0
```

**Example 2: Monitor validation loss**
```yaml
training:
  num_epochs: 100
  early_stopping:
    enabled: true
    patience: 15
    metric: 'val_loss'
    mode: 'min'
    min_delta: 0.001
```

**Example 3: Conservative (longer patience)**
```yaml
training:
  num_epochs: 200
  early_stopping:
    enabled: true
    patience: 30  # Wait longer before stopping
    metric: 'val_acc'
    mode: 'max'
```

### How It Works

1. **Tracks best metric value** across all epochs
2. **Counts epochs without improvement** (patience counter)
3. **Stops training** when patience counter reaches patience limit
4. **Loads best model** at the end (already implemented via checkpointing)

**Example scenario:**
```
Epoch 0:  val_acc=0.70  → Best so far, counter=0
Epoch 1:  val_acc=0.75  → Improved, counter=0
Epoch 2:  val_acc=0.78  → Improved, counter=0
Epoch 3:  val_acc=0.77  → No improvement, counter=1
Epoch 4:  val_acc=0.76  → No improvement, counter=2
Epoch 5:  val_acc=0.79  → Improved! counter=0
Epoch 6:  val_acc=0.78  → No improvement, counter=1
...
Epoch 15: val_acc=0.77  → counter=10, STOP (patience=10)
```

### Best Practices

1. **Always enable for long training runs**
   - Set `num_epochs` higher than you think you need
   - Let early stopping decide when to stop

2. **Adjust patience based on dataset**
   - Small datasets: Use smaller patience (5-10)
   - Large datasets: Use larger patience (15-30)
   - Noisy validation: Increase patience to avoid premature stopping

3. **Monitor in TensorBoard**
   ```bash
   tensorboard --logdir runs/
   ```
   - Verify early stopping triggered at right time
   - Check if patience needs adjustment

4. **Use with high `num_epochs`**
   ```yaml
   training:
     num_epochs: 200  # Set high
     early_stopping:
       enabled: true
       patience: 20   # Will stop early if no improvement
   ```

5. **Combine with learning rate scheduling**
   - Early stopping works best with learning rate decay
   - StepLR scheduler already configured in template

### Troubleshooting

**Problem: Stops too early**
- Increase `patience` (try 20-30)
- Decrease `min_delta` to 0.0
- Check if validation set is too small/noisy

**Problem: Never stops (trains all epochs)**
- Model is still improving (this is good!)
- Try increasing `num_epochs`
- Check if `metric` and `mode` are correct

**Problem: Training unstable**
- Validation accuracy jumps around
- Increase `patience` to tolerate noise
- Consider using `val_loss` instead (often more stable)
- Add `min_delta` threshold (e.g., 0.001)

### When to Use Early Stopping

**Use early stopping when:**
- ✅ Training for many epochs (>50)
- ✅ Uncertain about optimal epoch count
- ✅ Want to prevent overfitting automatically
- ✅ Running multiple experiments (saves compute)

**Don't use early stopping when:**
- ❌ Training for very few epochs (<20)
- ❌ Need exact epoch count for comparisons
- ❌ Validation set is very small/noisy
- ❌ Debugging (want full training run)

---

## Related Configuration

- [Data Configuration](data.md) - `num_workers` affects training speed
- [Optimizer Configuration](optimizer-scheduler.md) - Learning rate depends on batch size
- [CLI Overrides](cli-overrides.md) - Quick experimentation
- [Performance Tuning](../reference/performance-tuning.md) - Optimize training speed

## Further Reading

- [PyTorch CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)
- [Efficient Training Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
