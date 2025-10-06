# Learning Rate Finder Guide

Find the optimal learning rate for your model before training begins.

---

## What is LR Finder?

The Learning Rate Finder is a technique developed by Leslie Smith that helps you find an optimal learning rate range for training neural networks. The method systematically tests different learning rates by training your model for a few iterations with exponentially increasing learning rates, monitoring how the loss responds at each learning rate.

This approach is based on the cyclical learning rate research introduced in the paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186) by Leslie N. Smith (2015) and popularized by the [fastai](https://docs.fast.ai/callback.schedule.html#learningratefinder) library.

### How it Works

1. **Initialize**: Start with a very small learning rate (e.g., 1e-7)
2. **Train**: Run a few mini-batches of training, exponentially increasing the learning rate after each batch
3. **Monitor**: Record the loss at each learning rate
4. **Analyze**: Plot loss vs. learning rate to identify the optimal range
5. **Stop**: Terminate when loss starts to diverge (typically after 100-300 iterations)

The resulting plot shows how your model's loss responds to different learning rates, revealing the "sweet spot" where learning is fastest.

---

## Why Use LR Finder?

### Benefits

**Faster convergence**: Starting with a good learning rate helps your model converge in fewer epochs, saving compute time and resources.

**Avoid bad learning rates**: Too small causes painfully slow training, too large causes divergence or oscillation. LR Finder helps you avoid both extremes.

**Find good starting point**: Provides a principled, data-driven approach to choosing learning rates instead of arbitrary values like 0.001.

**Model-specific tuning**: Different architectures, datasets, and optimizers require different learning rates. LR Finder adapts to your specific setup.

**Save experimentation time**: Instead of manually trying learning rates like 0.0001, 0.001, 0.01, you can find the optimal range in a single run.

### When to Use

- Starting a new project with an unfamiliar dataset
- Trying a new model architecture
- Switching optimizers (SGD vs Adam vs AdamW)
- After significant changes to data augmentation
- When transfer learning from different domains

---

## Installation

LR Finder is included in the core framework - no additional dependencies required!

```bash
# Already available after standard installation
uv pip install -e "."

# Verify installation
ml-lr-finder --help
```

---

## Basic Usage

### Quick Start

Find optimal learning rate for your dataset in three steps:

```bash
# 1. Ensure you have splits and config ready
ml-split --raw_data data/my_dataset/raw --folds 5
ml-init-config data/my_dataset

# 2. Run LR Finder
ml-lr-finder --config configs/my_dataset_config.yaml

# 3. Check the output plot and suggested learning rate
# Output: runs/lr_finder/my_dataset/lr_finder.png
#         Suggested LR: 1e-3 (shown in console and on plot)
```

### Example Output

After running LR Finder, you'll see console output like this:

```
2025-10-06 10:30:15 | INFO     | Starting Learning Rate Finder
2025-10-06 10:30:15 | INFO     | Testing learning rates from 1e-7 to 1.0
2025-10-06 10:30:15 | INFO     | Using 100 iterations with exponential growth
2025-10-06 10:30:20 | INFO     | LR: 1.00e-07 | Loss: 2.3026
2025-10-06 10:30:21 | INFO     | LR: 2.51e-07 | Loss: 2.3015
2025-10-06 10:30:22 | INFO     | LR: 6.31e-07 | Loss: 2.2998
...
2025-10-06 10:31:05 | INFO     | LR: 1.00e-03 | Loss: 0.8521
2025-10-06 10:31:10 | INFO     | LR: 2.51e-03 | Loss: 0.7834
2025-10-06 10:31:15 | INFO     | LR: 6.31e-03 | Loss: 0.9123
2025-10-06 10:31:20 | WARNING  | Loss increasing - stopping early
2025-10-06 10:31:20 | SUCCESS  | Suggested learning rate: 2.00e-03
2025-10-06 10:31:20 | SUCCESS  | Saved plot to runs/lr_finder/my_dataset/lr_finder.png
```

A plot will be saved showing:
- X-axis: Learning rate (log scale)
- Y-axis: Training loss
- Vertical dashed line: Suggested optimal learning rate
- Annotation: Exact LR value

---

## Interpreting the Plot

### The Ideal Learning Rate Curve

A typical LR Finder plot has three regions:

```
Loss
 |
 |     ╱╲
 |    ╱  ╲ ← Too high: Loss explodes
 |   ╱    ╲
 |  ╱      ╲___
 | ╱           ╲
 |╱_____________╲________
  |      |      |
  Too   Good   Too
  small range  large

  Learning Rate (log scale) →
```

**Region 1 - Too Small (left side)**: Loss decreases very slowly or stays flat. Learning is happening but inefficiently.

**Region 2 - Good Range (middle)**: Loss decreases rapidly. This is the "sweet spot" - the steepest descent indicates fastest learning.

**Region 3 - Too Large (right side)**: Loss increases or oscillates wildly. Learning rate is too aggressive, causing instability.

### Finding the Optimal Learning Rate

!!!tip "Look for the steepest descent"
    The optimal learning rate is typically found where the loss is decreasing most rapidly - the steepest negative slope on the curve. This is usually just before the loss starts to increase.

The framework automatically suggests a learning rate by finding:
1. The learning rate with the minimum loss
2. The learning rate with the steepest gradient (fastest descent)
3. A point between these two that balances speed and stability

### Choosing Your Final Learning Rate

Use one of these strategies based on the suggested learning rate:

**Option 1 - Use the suggested LR directly (recommended for most cases)**:
```bash
# LR Finder suggests: 2e-3
ml-train --config configs/my_dataset_config.yaml --lr 0.002
```

**Option 2 - Use 1/10 of the minimum loss point (conservative)**:
```bash
# If minimum loss occurs at 1e-2, use 1e-3
ml-train --config configs/my_dataset_config.yaml --lr 0.001
```

**Option 3 - Use the middle of the good range (for schedulers)**:
```bash
# If good range is 1e-4 to 1e-2, use 1e-3
ml-train --config configs/my_dataset_config.yaml --lr 0.001
```

!!!warning "Don't pick the absolute minimum"
    The learning rate at the absolute minimum loss is often too high for stable long-term training. Pick a point on the steepest descent just before the minimum.

---

## Advanced Options

### Custom Learning Rate Range

Test a specific range of learning rates:

```bash
# Test learning rates from 1e-6 to 1e-1
ml-lr-finder --config configs/my_config.yaml --start_lr 1e-6 --end_lr 1e-1
```

**When to adjust:**
- `--start_lr`: Set higher (e.g., 1e-5) if you know very small LRs won't work
- `--end_lr`: Set lower (e.g., 1e-2) for small models or sensitive tasks

### Number of Iterations

Control how many mini-batches to test:

```bash
# Quick test: 50 iterations (~30 seconds)
ml-lr-finder --config configs/my_config.yaml --num_iter 50

# Thorough test: 300 iterations (~3 minutes)
ml-lr-finder --config configs/my_config.yaml --num_iter 300
```

**Guidelines:**
- Small datasets: 50-100 iterations
- Medium datasets: 100-200 iterations (default: 100)
- Large datasets: 200-300 iterations
- More iterations = smoother curve but longer runtime

### Divergence Threshold

Control early stopping sensitivity when loss starts increasing:

```bash
# More sensitive (stops earlier when loss increases slightly)
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 2.0

# Default behavior (stops when loss exceeds 4x minimum)
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 4.0

# Less sensitive (allows more loss increase before stopping)
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 6.0
```

**How it works:**
- LR Finder tracks the minimum loss seen so far
- If `current_loss > diverge_threshold × minimum_loss`, it stops early
- This prevents wasting time testing learning rates that clearly don't work

**Guidelines:**
- **Lower threshold (1.5-2.5)**: Use for unstable models or when you want to stop early
- **Default threshold (4.0)**: Works well for most cases
- **Higher threshold (5.0-6.0)**: Use for models with noisy loss curves or when you want complete curves

**When to adjust:**
- Model with smooth loss: Use default (4.0)
- Model with noisy/oscillating loss: Increase to 5.0-6.0
- Very large models prone to instability: Decrease to 2.0-3.0
- Want to see full curve even if loss explodes: Increase to 8.0-10.0

### Smoothing Factor (Beta)

Adjust how much the loss curve is smoothed using exponential moving average:

```bash
# Less smoothing (more responsive to changes, noisier curve)
ml-lr-finder --config configs/my_config.yaml --beta 0.9

# Default smoothing (good balance)
ml-lr-finder --config configs/my_config.yaml --beta 0.98

# More smoothing (smoother curve, less responsive)
ml-lr-finder --config configs/my_config.yaml --beta 0.99
```

**Guidelines:**
- **Lower beta (0.8-0.95)**: Use for datasets with stable loss to see fine details
- **Default beta (0.98)**: Works well for most cases
- **Higher beta (0.99)**: Use for small batches or noisy datasets to smooth out oscillations

### Specific Fold or Architecture

Test LR for a specific cross-validation fold or model:

```bash
# Test specific fold
ml-lr-finder --config configs/my_config.yaml --fold 2

# Test with different architecture (edit config first)
ml-lr-finder --config configs/resnet50_config.yaml
```

### Combined Advanced Options

Combine multiple options for fine-grained control:

```bash
# Complete custom configuration
ml-lr-finder --config configs/my_config.yaml \
    --start_lr 1e-7 \
    --end_lr 1 \
    --num_iter 200 \
    --beta 0.95 \
    --diverge_threshold 3.0 \
    --fold 1
```

---

## Examples with Different Scenarios

### Example 1: Small Dataset (Hymenoptera)

```bash
# Small dataset with 244 images
ml-lr-finder --config configs/hymenoptera_config.yaml --num_iter 50

# Typical output:
# Suggested LR: 1e-3 to 1e-2
# Good for: Small datasets benefit from higher learning rates

# Train with suggested LR
ml-train --config configs/hymenoptera_config.yaml --lr 0.005
```

### Example 2: Large Model (ResNet50)

```bash
# ResNet50 with pretrained weights
# Edit config: architecture: 'resnet50', weights: 'DEFAULT'
ml-lr-finder --config configs/resnet50_config.yaml

# Typical output:
# Suggested LR: 1e-4 to 1e-3
# Lower than small models due to pretrained weights

# Train with conservative LR for fine-tuning
ml-train --config configs/resnet50_config.yaml --lr 0.0001
```

### Example 3: Custom Architecture

```bash
# Custom model without pretrained weights
# Edit config: type: 'custom', custom_architecture: 'simple_cnn'
ml-lr-finder --config configs/custom_config.yaml --num_iter 150

# Typical output:
# Suggested LR: 1e-3 to 1e-2
# Higher than pretrained models since training from scratch

# Train with higher LR
ml-train --config configs/custom_config.yaml --lr 0.01
```

### Example 4: Different Optimizers

```bash
# SGD typically needs higher learning rates than Adam
# SGD with momentum
ml-lr-finder --config configs/sgd_config.yaml

# Typical output (SGD):
# Suggested LR: 1e-2 to 1e-1

# Adam or AdamW
ml-lr-finder --config configs/adam_config.yaml

# Typical output (Adam):
# Suggested LR: 1e-4 to 1e-3
```

### Example 5: Transfer Learning Workflow

```bash
# Step 1: Find LR for fine-tuning
ml-lr-finder --config configs/transfer_config.yaml

# Step 2: Train with suggested LR
ml-train --config configs/transfer_config.yaml --lr 0.0001 --num_epochs 30

# Step 3: Unfreeze more layers, find new LR
# Edit config to unfreeze additional layers
ml-lr-finder --config configs/transfer_unfreeze_config.yaml

# Step 4: Continue training with new LR
ml-train --config configs/transfer_unfreeze_config.yaml --lr 0.0005 --num_epochs 20
```

---

## FAQ

### When to Use LR Finder?

**Always use when:**
- Starting a new project or dataset
- Trying a new architecture for the first time
- Switching between optimizers (SGD ↔ Adam)
- Significant changes to your pipeline (augmentation, preprocessing)

**Optional when:**
- Using common architecture + optimizer combo with known good LRs
- Doing quick experiments with temporary configurations
- Reproducing published results with documented hyperparameters

### When NOT to Use LR Finder?

**Skip LR Finder if:**
- You have already found optimal LR for this exact setup
- Reproducing results from a paper (use their LR)
- Using established recipes (e.g., ImageNet training protocols)
- Doing hyperparameter search with Optuna (it will search LR automatically)

**LR Finder has limitations:**
- Requires at least one epoch's worth of data to run effectively
- May suggest suboptimal LR if your dataset is very small (<100 images)
- Assumes monotonic loss decrease during initial training

### How is LR Finder Different from Grid Search?

| Aspect | LR Finder | Grid Search |
|--------|-----------|-------------|
| Speed | 2-5 minutes | Hours to days |
| Resource usage | Minimal (100-300 iterations) | High (full training runs) |
| Provides | Single optimal LR | Multiple full training results |
| Best for | Quick initial LR discovery | Final hyperparameter tuning |

**Recommendation**: Use LR Finder first to narrow the range, then use grid search or Optuna for fine-tuning if needed.

### Why is the Suggested LR Different from What I Expected?

Several factors affect optimal learning rate:

**Model architecture**: Larger models (ResNet50) need lower LRs than smaller ones (ResNet18)

**Pretrained weights**: Fine-tuning needs 10-100x lower LR than training from scratch

**Optimizer**: SGD needs ~10x higher LR than Adam/AdamW

**Batch size**: Larger batches can use proportionally higher LRs (linear scaling rule)

**Dataset**: Different domains (natural images vs medical) may require different LRs

!!!tip "Trust the plot"
    If the suggested LR seems unusual, look at the plot. The curve never lies - if loss decreases fastest at 1e-2, that's likely your optimal LR regardless of intuition.

### Can I Run LR Finder Multiple Times?

Yes! Run LR Finder separately for each:
- Cross-validation fold (optional, but fold 0 usually representative)
- Model architecture you want to try
- Optimizer change
- Major augmentation change

**Example workflow:**
```bash
# Find LR for ResNet18
ml-lr-finder --config configs/resnet18_config.yaml
# Suggested: 1e-3

# Find LR for ResNet50
ml-lr-finder --config configs/resnet50_config.yaml
# Suggested: 1e-4

# Find LR for EfficientNet
ml-lr-finder --config configs/efficientnet_config.yaml
# Suggested: 5e-4
```

### What if the Loss Never Decreases?

**Possible causes:**

1. **Learning rate range too high**: Try `--start_lr 1e-8 --end_lr 1e-3`
2. **Model/data issue**: Check that your config is correct (num_classes, data_dir)
3. **Bad initialization**: Very rare, try changing seed in config
4. **Insufficient iterations**: Increase to `--num_iter 200`

**Debugging steps:**
```bash
# Try lower learning rates
ml-lr-finder --config configs/my_config.yaml --start_lr 1e-9 --end_lr 1e-2

# Verify config and data
ml-train --config configs/my_config.yaml --num_epochs 1  # Quick test

# Check data loading
ls data/my_dataset/splits/  # Should see fold_*_train.txt files
```

### Should I Run LR Finder for Each Fold?

**Usually not necessary**. The optimal learning rate is primarily determined by:
- Model architecture
- Optimizer
- Batch size
- Data characteristics

These are the same across folds. Running on fold 0 is sufficient for most cases.

**Exception**: If you notice significantly different convergence behavior across folds (unusual), you might want to verify LR works well on another fold.

### How Does LR Finder Interact with Schedulers?

LR Finder finds the **maximum** learning rate to start with. Most schedulers (StepLR, CosineAnnealing) will decrease LR during training.

**Best practice**: Use the suggested LR as your initial/max learning rate, then let the scheduler reduce it:

```yaml
optimizer:
  lr: 0.002  # From LR Finder

scheduler:
  name: 'steplr'
  step_size: 10
  gamma: 0.1  # Will reduce to 0.0002, then 0.00002
```

For cyclical schedulers (OneCycleLR, CyclicLR), use the suggested LR as the maximum LR in the cycle.

---

## Troubleshooting

### "CUDA out of memory" Error

**Problem**: GPU memory exhausted during LR Finder

**Solutions:**
```bash
# Reduce batch size
ml-lr-finder --config configs/my_config.yaml --batch_size 16

# Use fewer iterations
ml-lr-finder --config configs/my_config.yaml --num_iter 50

# Run on CPU (slower)
ml-lr-finder --config configs/my_config.yaml --device cpu
```

### Plot Shows Flat Line

**Problem**: Loss doesn't change across learning rates

**Possible causes:**
- Learning rate range is too narrow or too low
- Model is not learning at all (check data loading)
- Loss is already at minimum (unusual for random initialization)

**Solutions:**
```bash
# Expand learning rate range
ml-lr-finder --config configs/my_config.yaml --start_lr 1e-8 --end_lr 1.0

# Verify data is loading correctly
ml-train --config configs/my_config.yaml --num_epochs 1

# Check that splits exist
ls data/my_dataset/splits/
```

### Loss Explodes Immediately

**Problem**: Loss goes to infinity or NaN from the start

**Solutions:**
```bash
# Start with even lower learning rates
ml-lr-finder --config configs/my_config.yaml --start_lr 1e-9 --end_lr 1e-3

# Use lower divergence threshold to catch explosion earlier
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 2.0

# Check data preprocessing/normalization in config
# Ensure transforms include proper normalization:
# - mean: [0.485, 0.456, 0.406]
# - std: [0.229, 0.224, 0.225]

# Try with smaller model first
# Edit config: architecture: 'resnet18'
```

### Loss Curve is Too Noisy

**Problem**: Loss oscillates wildly, making it hard to identify optimal LR

**Solutions:**
```bash
# Increase smoothing
ml-lr-finder --config configs/my_config.yaml --beta 0.99

# Use more iterations for better averaging
ml-lr-finder --config configs/my_config.yaml --num_iter 200

# Increase batch size for more stable gradients
ml-lr-finder --config configs/my_config.yaml --batch_size 64

# Combine all three for smoothest curve
ml-lr-finder --config configs/my_config.yaml \
    --beta 0.99 \
    --num_iter 200 \
    --batch_size 64
```

### LR Finder Stops Too Early

**Problem**: LR Finder terminates before testing higher learning rates

**Cause**: Early stopping triggered by divergence threshold

**Solutions:**
```bash
# Increase divergence threshold to allow more loss increase
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 6.0

# Or disable early stopping effectively
ml-lr-finder --config configs/my_config.yaml --diverge_threshold 10.0

# Also increase max LR to test higher values
ml-lr-finder --config configs/my_config.yaml \
    --diverge_threshold 6.0 \
    --end_lr 10.0
```

### Different Results on Different Runs

**Problem**: LR Finder suggests different values each time

**Cause**: Random initialization and data shuffling create variation

**Solutions:**
```bash
# Set seed for reproducibility
# Edit config:
# seed: 42
# deterministic: true

ml-lr-finder --config configs/my_config.yaml

# Or average results from multiple runs
ml-lr-finder --config configs/my_config.yaml  # Run 1: suggests 2e-3
ml-lr-finder --config configs/my_config.yaml  # Run 2: suggests 3e-3
ml-lr-finder --config configs/my_config.yaml  # Run 3: suggests 2.5e-3
# Use average: ~2.5e-3
```

---

## Related Guides

- [Training Guide](training.md) - Use LR from finder in training workflow
- [Hyperparameter Tuning](hyperparameter-tuning.md) - Automated search including learning rate
- [Advanced Training](advanced-training.md) - Trainer types and optimization strategies
- [Configuration Reference](../configuration/optimizer-scheduler.md) - Learning rate and optimizer settings
- [Troubleshooting](../reference/troubleshooting.md) - Common training issues and solutions

---

## Summary

**Key Takeaways:**

- LR Finder helps you discover optimal learning rates in minutes instead of hours of trial and error
- Look for the steepest descent in the loss curve - this indicates fastest learning
- Use the suggested LR directly or pick 1/10 of the minimum loss point for conservative training
- Run LR Finder when starting new projects, trying new architectures, or changing optimizers
- The technique is fast (2-5 minutes), reliable, and requires no extra dependencies

**Ready to find your optimal learning rate?**

```bash
# Run LR Finder
ml-lr-finder --config configs/my_dataset_config.yaml

# Use suggested LR for training
ml-train --config configs/my_dataset_config.yaml --lr 0.002 --num_epochs 50
```

Happy training!
