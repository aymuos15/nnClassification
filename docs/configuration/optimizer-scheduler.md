# Optimizer & Scheduler Configuration

## Overview

This section configures the optimizer (SGD) and learning rate scheduler (StepLR) that control how the model learns during training.

## Optimizer Configuration

```yaml
optimizer:
  lr: <float>
  momentum: <float>
```

### `lr` (Learning Rate)

- **Type:** Float (> 0)
- **Default:** `0.001`
- **Description:** Learning rate (step size for gradient descent)
- **Purpose:** Controls how much to update weights each iteration

#### Usage

```yaml
optimizer:
  lr: 0.001
```

#### CLI Override

```bash
python train.py --lr 0.01
```

#### Typical Ranges

| Learning Rate | Use Case |
|--------------|----------|
| `1e-5 - 1e-4` | Fine-tuning pretrained models |
| `1e-4 - 1e-3` | Training with transfer learning (current default) |
| `1e-3 - 1e-2` | Training from scratch with small batches |
| `1e-2 - 1e-1` | Training from scratch with large batches |

#### Understanding Learning Rate

**Too Small (e.g., 0.00001):**
- ❌ Very slow convergence
- ❌ May not reach optimal solution
- ❌ Wastes training time
- ✅ Very stable (won't diverge)

**Optimal (e.g., 0.001):**
- ✅ Good convergence speed
- ✅ Stable training
- ✅ Reaches good solution

**Too Large (e.g., 0.1):**
- ❌ Unstable training
- ❌ Loss may diverge to NaN
- ❌ May overshoot optimal solution
- ✅ Fast initial progress (if stable)

#### Finding the Right Learning Rate

**Method 1: Start Conservative**
```bash
# Start with default
python train.py --lr 0.001

# If loss decreases too slowly → increase
python train.py --lr 0.01

# If loss diverges/NaN → decrease
python train.py --lr 0.0001
```

**Method 2: LR Range Test** (not implemented, but recommended)
- Start with very small LR (1e-6)
- Gradually increase each batch
- Plot loss vs LR
- Choose LR where loss decreases fastest

#### Relationship with Batch Size

Learning rate should scale with batch size:

**Rule of thumb:** LR ∝ √batch_size

```yaml
# Small batch
training:
  batch_size: 8
optimizer:
  lr: 0.001

# Medium batch (4x larger) → LR should be ~2x larger
training:
  batch_size: 32
optimizer:
  lr: 0.002

# Large batch (16x larger) → LR should be ~4x larger
training:
  batch_size: 128
optimizer:
  lr: 0.004
```

#### Interaction with Scheduler

The scheduler modifies this initial learning rate during training:

```yaml
optimizer:
  lr: 0.001      # Initial LR

scheduler:
  step_size: 7   # Decay every 7 epochs
  gamma: 0.1     # Multiply by 0.1
```

**LR schedule over 25 epochs:**
- Epochs 0-6: LR = 0.001
- Epochs 7-13: LR = 0.0001 (×0.1)
- Epochs 14-20: LR = 0.00001 (×0.1)
- Epochs 21-24: LR = 0.000001 (×0.1)

---

### `momentum`

- **Type:** Float [0.0, 1.0]
- **Default:** `0.9`
- **Description:** SGD momentum factor
- **Purpose:** Accelerates optimization by accumulating velocity

#### Usage

```yaml
optimizer:
  momentum: 0.9
```

#### CLI Override

```bash
python train.py --momentum 0.95
```

#### How Momentum Works

Without momentum (vanilla SGD):
```
weight_update = -lr * gradient
```

With momentum:
```
velocity = momentum * velocity + gradient
weight_update = -lr * velocity
```

**Effect:**
- Accumulates gradients from previous steps
- Accelerates movement in consistent directions
- Dampens oscillations

#### Momentum Values

| Momentum | Effect | Use Case |
|----------|--------|----------|
| `0.0` | No momentum (vanilla SGD) | Rarely used, very slow |
| `0.5` | Low momentum | Slower convergence, more stable |
| `0.9` | Standard momentum ✅ | Recommended default |
| `0.95` | High momentum | Faster convergence, may oscillate |
| `0.99` | Very high momentum | May overshoot, hard to tune |

#### When to Change

**Use lower momentum (0.8-0.85):**
- Training is oscillating
- Loss jumping around
- Need more stability

**Use higher momentum (0.95):**
- Training is too slow
- Loss decreasing smoothly
- Want faster convergence

**Keep default (0.9):**
- Works well for most cases
- Rarely needs tuning
- Good starting point

#### Visualization

```
Without momentum:        With momentum (0.9):
    
    |\ |\ |\              |\
    | \| \| \             | \___
 Loss                  Loss        
    |                     |
    +---> Epochs          +---> Epochs
    (Noisy)               (Smooth, faster)
```

---

## Scheduler Configuration

The framework uses **StepLR scheduler** (multiplicative decay every N epochs).

```yaml
scheduler:
  step_size: <int>
  gamma: <float>
```

### `step_size`

- **Type:** Integer (> 0)
- **Default:** `7`
- **Description:** Number of epochs between LR decay steps
- **Purpose:** Controls frequency of learning rate reduction

#### Usage

```yaml
scheduler:
  step_size: 7
```

#### CLI Override

```bash
python train.py --step_size 10
```

#### How It Works

Every `step_size` epochs, LR is multiplied by `gamma`:

```
Epoch 0-6:   LR = initial_lr
Epoch 7-13:  LR = initial_lr × gamma
Epoch 14-20: LR = initial_lr × gamma²
Epoch 21+:   LR = initial_lr × gamma³
```

#### Choosing Step Size

**Short Training (< 20 epochs):**
```yaml
scheduler:
  step_size: 5  # Decay 2-3 times
```

**Medium Training (20-50 epochs):**
```yaml
scheduler:
  step_size: 10  # Decay 2-4 times
```

**Long Training (50+ epochs):**
```yaml
scheduler:
  step_size: 20  # Decay 2-3 times
```

#### Rules of Thumb

1. **Decay 2-3 times during training**
   - More than 3 decays: LR becomes too small
   - Fewer than 2 decays: not enough refinement

2. **Let model train with initial LR first**
   - Don't decay too early
   - Model needs time to make coarse adjustments

3. **Plan ahead:**
   ```
   Training 25 epochs, want 3 decays:
   step_size = 25 / 3 ≈ 8
   Decays at epochs: 8, 16, 24
   ```

#### Examples by Training Length

```yaml
# 10 epochs → decay once
scheduler:
  step_size: 7

# 25 epochs → decay twice  
scheduler:
  step_size: 10

# 50 epochs → decay 3 times
scheduler:
  step_size: 15

# 100 epochs → decay 3-4 times
scheduler:
  step_size: 25
```

---

### `gamma`

- **Type:** Float (0.0, 1.0]
- **Default:** `0.1`
- **Description:** Multiplicative factor for learning rate decay
- **Purpose:** Controls magnitude of LR reduction

#### Usage

```yaml
scheduler:
  gamma: 0.1
```

#### CLI Override

```bash
python train.py --gamma 0.5
```

#### Common Values

| Gamma | Effect | Description |
|-------|--------|-------------|
| `0.1` | Aggressive decay ✅ | 90% reduction (standard) |
| `0.2` | Strong decay | 80% reduction |
| `0.5` | Moderate decay | 50% reduction |
| `0.9` | Gentle decay | 10% reduction |

#### Choosing Gamma

**Aggressive (0.1 - default):**
- ✅ Standard value, works well
- ✅ Fast convergence
- ❌ May underfit if too many decays

**Moderate (0.5):**
- ✅ Good for fine-tuning
- ✅ More gradual refinement
- ❌ Slower convergence

**Gentle (0.9):**
- ✅ Very gradual adjustment
- ✅ Good for already-trained models
- ❌ Very slow refinement

#### Combined Example

```yaml
optimizer:
  lr: 0.001

scheduler:
  step_size: 7
  gamma: 0.1
```

**Training 25 epochs:**
```
Epochs 0-6:   LR = 0.001    (initial)
Epochs 7-13:  LR = 0.0001   (×0.1 at epoch 7)
Epochs 14-20: LR = 0.00001  (×0.1 at epoch 14)
Epochs 21-24: LR = 0.000001 (×0.1 at epoch 21)
```

---

## Complete Examples

### Example 1: Default Configuration

```yaml
optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 7
  gamma: 0.1
```
**When to use:** General purpose, good starting point

---

### Example 2: High Learning Rate (Large Batch)

```yaml
training:
  batch_size: 64

optimizer:
  lr: 0.01      # Higher LR for larger batch
  momentum: 0.9

scheduler:
  step_size: 10
  gamma: 0.1
```
**When to use:** Large batch sizes (64+)

---

### Example 3: Conservative (Fine-tuning)

```yaml
optimizer:
  lr: 0.0001    # Low LR for fine-tuning
  momentum: 0.9

scheduler:
  step_size: 15  # Decay less frequently
  gamma: 0.5     # Moderate decay
```
**When to use:** Fine-tuning pretrained models

---

### Example 4: Aggressive Training

```yaml
optimizer:
  lr: 0.01      # High LR
  momentum: 0.95 # High momentum

scheduler:
  step_size: 5   # Frequent decay
  gamma: 0.1
```
**When to use:** Training from scratch, fast experimentation

---

## Best Practices

### For Learning Rate
1. **Start with 0.001** (default)
2. **Scale with batch size** (larger batch → higher LR)
3. **Monitor training loss**
   - If decreasing too slowly → increase LR
   - If diverging/NaN → decrease LR
4. **Use TensorBoard** to visualize LR schedule

### For Momentum
1. **Use 0.9** (standard value)
2. **Rarely needs tuning**
3. **Lower if training unstable** (oscillating loss)
4. **Higher if training too slow** (smooth loss curve)

### For Scheduler
1. **Plan decays ahead** (2-3 decays total)
2. **Don't decay too early** (let model learn first)
3. **gamma=0.1 is standard** (rarely needs changing)
4. **Monitor validation loss** (decay when plateaus)

### Hyperparameter Search Strategy

```bash
# 1. Find good LR first (fix other params)
python train.py --lr 0.0001 --num_epochs 10
python train.py --lr 0.001 --num_epochs 10
python train.py --lr 0.01 --num_epochs 10

# 2. Use best LR, tune batch size
python train.py --lr 0.001 --batch_size 16
python train.py --lr 0.001 --batch_size 32
python train.py --lr 0.001 --batch_size 64

# 3. Scale LR with batch size
python train.py --lr 0.002 --batch_size 64

# 4. Fine-tune scheduler if needed
python train.py --lr 0.002 --batch_size 64 --step_size 10
```

## Troubleshooting

### Loss Not Decreasing

**Problem:** Training loss stays high

**Solutions:**
1. Increase learning rate (try 10x higher)
2. Check data preparation (labels correct?)
3. Check model architecture
4. Reduce batch size
5. Train more epochs

### Loss Diverges (NaN)

**Problem:** Loss becomes NaN or infinity

**Solutions:**
1. **Decrease learning rate** (most common fix)
2. Reduce batch size
3. Check data normalization
4. Add gradient clipping (not implemented)
5. Check for bugs in data pipeline

### Training Too Slow

**Problem:** Loss decreasing but very slowly

**Solutions:**
1. Increase learning rate
2. Increase momentum (0.95)
3. Increase batch size (if memory allows)
4. Use more aggressive scheduler (smaller step_size)

### Overfitting

**Problem:** Train acc high, val acc low

**Solutions:**
1. This is not optimizer problem (see data augmentation)
2. But: train for fewer epochs
3. Or: use more aggressive LR decay (smaller gamma)

## Related Configuration

- [Training Configuration](training.md) - Batch size affects LR
- [CLI Overrides](cli-overrides.md) - Quick experimentation
- [Examples](examples.md) - Complete configurations
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

## Further Reading

- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [LR Schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
- [SGD with Momentum Paper](https://www.sciencedirect.com/science/article/abs/pii/0893608089900208)
