# CLI Override System

## Overview

Command-line arguments override YAML configuration, enabling quick experimentation without editing config files. This is perfect for hyperparameter search and rapid iteration.

## How It Works

```
1. Load base config from YAML (ml_src/config.yaml)
2. Apply CLI argument overrides
3. Create run directory based on overrides
4. Save final config to runs/{run_name}/config.yaml
```

---

## Override Mapping

| CLI Argument | Config Path | Type | Example |
|-------------|-------------|------|---------|
| `--config` | (Specifies YAML file) | string | `--config custom.yaml` |
| `--dataset_name` | `data.dataset_name` | string | `--dataset_name my_dataset` |
| `--data_dir` | `data.data_dir` | string | `--data_dir data/my_dataset` |
| `--fold` | `data.fold` | int | `--fold 1` |
| `--batch_size` | `training.batch_size` | int | `--batch_size 32` |
| `--num_workers` | `data.num_workers` | int | `--num_workers 8` |
| `--num_epochs` | `training.num_epochs` | int | `--num_epochs 50` |
| `--lr` | `optimizer.lr` | float | `--lr 0.01` |
| `--momentum` | `optimizer.momentum` | float | `--momentum 0.95` |
| `--step_size` | `scheduler.step_size` | int | `--step_size 10` |
| `--gamma` | `scheduler.gamma` | float | `--gamma 0.5` |
| `--device` | `training.device` | string | `--device cuda:1` |
| `--resume` | (Special) | string | `--resume runs/base/last.pt` |

---

## Usage Examples

### Basic Override

```bash
# Use base config, override batch size
ml-train --batch_size 32
```

### Multiple Overrides

```bash
# Override several parameters
ml-train --batch_size 32 --lr 0.01 --num_epochs 50
```

### Custom Config File

```bash
# Use custom config
ml-train --config configs/my_config.yaml

# Custom config + overrides
ml-train --config configs/my_config.yaml --batch_size 64
```

### Resume Training

```bash
# Resume from last checkpoint
ml-train --resume runs/base/last.pt

# Resume and train more epochs
ml-train --resume runs/base/last.pt --num_epochs 50
```

---

## Run Directory Naming

Run directories are automatically named based on the dataset name, fold number, and hyperparameter overrides. This creates organized, self-documenting experiment folders.

### Naming Rules

The format is: `runs/{dataset_name}_{overrides}_fold_{fold_num}`

**No overrides:**
```bash
# Assuming dataset_name: 'hymenoptera', fold: 0
ml-train
# Creates: runs/hymenoptera_base_fold_0/
```

**With fold override:**
```bash
ml-train --fold 1
# Creates: runs/hymenoptera_base_fold_1/
```

**With hyperparameter overrides:**
```bash
ml-train --fold 0 --batch_size 32
# Creates: runs/hymenoptera_batch_32_fold_0/

ml-train --fold 0 --lr 0.01
# Creates: runs/hymenoptera_lr_0.01_fold_0/
```

**Multiple overrides:**
```bash
ml-train --fold 1 --batch_size 32 --num_epochs 50
# Creates: runs/hymenoptera_batch_32_epochs_50_fold_1/

ml-train --fold 1 --batch_size 32 --lr 0.01 --num_epochs 50
# Creates: runs/hymenoptera_batch_32_epochs_50_lr_0.01_fold_1/
```

### Which Parameters Affect Run Name?

**Included in run name** (affect training results):
- `batch_size`
- `num_epochs`
- `lr` (learning rate)

**Excluded from run name** (don't affect results):
- `num_workers` (just loading speed)
- `data_dir` (just data location)
- `device` (just hardware selection)

### Why This Design?

- **Self-documenting:** Run name tells you what changed
- **Prevents overwrites:** Different hyperparams → different folders
- **Easy comparison:** Compare runs by folder name

---

## Common Workflows

### 1. Quick Experimentation

```bash
# Try different batch sizes
ml-train --batch_size 16
ml-train --batch_size 32
ml-train --batch_size 64

# Compare results in:
# - runs/batch_16/
# - runs/batch_32/
# - runs/batch_64/
```

### 2. Hyperparameter Search

```bash
# Grid search over LR and batch size
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    ml-train --lr $lr --batch_size $bs
  done
done

# Creates organized folders:
# runs/batch_16_lr_0.001/
# runs/batch_16_lr_0.01/
# runs/batch_16_lr_0.1/
# runs/batch_32_lr_0.001/
# ... etc
```

### 3. Different Datasets

```bash
# Train on different datasets (same hyperparams)
ml-train --data_dir data/dataset1
ml-train --data_dir data/dataset2
ml-train --data_dir data/dataset3

# Note: All create runs/base/ (data_dir doesn't affect run name)
# Solution: Use different configs or add meaningful overrides
ml-train --data_dir data/dataset1 --num_epochs 25
ml-train --data_dir data/dataset2 --num_epochs 50
```

### 4. GPU Selection

```bash
# Use specific GPU (doesn't affect run name)
ml-train --device cuda:0
ml-train --device cuda:1

# Or via environment variable
CUDA_VISIBLE_DEVICES=1 ml-train
```

### 5. Resume and Extend

```bash
# Train 25 epochs
ml-train --num_epochs 25
# Saves to: runs/epochs_25/

# If not converged, resume and train more
ml-train --resume runs/epochs_25/last.pt --num_epochs 50
# Note: Continues training up to epoch 50 total
```

---

## Advanced Usage

### Config File + Overrides

You can combine custom configs with CLI overrides:

```bash
# configs/production.yaml has most settings
# Override just batch_size
ml-train --config configs/production.yaml --batch_size 128
```

**Use case:** Base config for a project, tweak individual params

### Debugging with Overrides

```bash
# Quick debug run (small batch, few epochs, no workers)
ml-train --batch_size 2 --num_epochs 2 --num_workers 0

# Creates: runs/batch_2_epochs_2/
```

### Full Override Example

```bash
ml-train \
  --config ml_src/config.yaml \
  --data_dir /mnt/datasets/imagenet \
  --batch_size 64 \
  --num_workers 8 \
  --num_epochs 100 \
  --lr 0.01 \
  --momentum 0.9 \
  --step_size 30 \
  --gamma 0.1 \
  --device cuda:0

# Creates: runs/batch_64_epochs_100_lr_0.01/
# With all your custom settings
```

---

## Verifying Configuration

After training starts, check what config was actually used:

```bash
# View saved config
cat runs/{run_name}/config.yaml

# Or during training
cat runs/batch_32_lr_0.01/config.yaml
```

This shows the final merged config (YAML + overrides).

---

## Best Practices

### 1. Start with Defaults

```bash
# First, train with base config
ml-train

# Then experiment with overrides
ml-train --lr 0.01
ml-train --batch_size 32
```

### 2. Systematic Exploration

```bash
# Test one parameter at a time
ml-train --lr 0.0001
ml-train --lr 0.001
ml-train --lr 0.01

# Then combine best values
ml-train --lr 0.001 --batch_size 32
```

### 3. Document Experiments

```bash
# Create a script to track experiments
cat > experiments.sh << 'EOF'
#!/bin/bash

# Experiment 1: Baseline
ml-train

# Experiment 2: Higher LR
ml-train --lr 0.01

# Experiment 3: Larger batch
ml-train --batch_size 32 --lr 0.002
EOF

chmod +x experiments.sh
./experiments.sh
```

### 4. Use Shell Scripts for Sweeps

```bash
# sweep.sh - Hyperparameter sweep
#!/bin/bash

learning_rates=(0.0001 0.001 0.01)
batch_sizes=(16 32 64)

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    echo "Training with lr=$lr, batch_size=$bs"
    ml-train --lr $lr --batch_size $bs
  done
done
```

### 5. Compare Results

```bash
# After training multiple runs, compare
ls runs/

# Output:
# base/
# batch_32/
# batch_32_lr_0.01/
# epochs_50/
# lr_0.01/

# Use TensorBoard to compare
tensorboard --logdir runs/
```

---

## Troubleshooting

### Run Directory Already Exists

**Problem:** 
```
Error: Run directory 'runs/batch_32/' already exists
```

**Solution:**
```bash
# Option 1: Delete old run
rm -rf runs/batch_32/

# Option 2: Use different hyperparams (creates new folder)
ml-train --batch_size 32 --lr 0.01

# Option 3: Resume training
ml-train --resume runs/batch_32/last.pt
```

### Override Not Taking Effect

**Problem:** Changed CLI arg but config still shows old value

**Solution:**
```bash
# Check you're using correct argument name
ml-train --lr 0.01  # ✅ Correct

ml-train --learning_rate 0.01  # ❌ Wrong (not supported)

# Verify in saved config
cat runs/{run_name}/config.yaml
```

### Complex Config Not Overridable

**Problem:** Want to override nested config (e.g., transforms)

**Solution:** CLI only supports common parameters. For complex changes:
```bash
# Option 1: Create custom config file
# configs/custom_transforms.yaml
# ... your changes ...

ml-train --config configs/custom_transforms.yaml

# Option 2: Modify base config temporarily
# Edit ml_src/config.yaml directly
```

---

## Comparison with Other Approaches

### CLI Overrides (Current)

✅ Quick experimentation  
✅ No file editing  
✅ Good for sweeps
❌ Limited to common params

```bash
ml-train --lr 0.01 --batch_size 32
```

### Custom Config Files

✅ Full control  
✅ Version controlled  
✅ Shareable  
❌ Requires file creation

```bash
ml-train --config configs/experiment1.yaml
```

### Hybrid Approach (Recommended)

✅ Best of both worlds

```bash
# Base config + targeted overrides
ml-train --config configs/base.yaml --lr 0.01
```

---

## Integration with Other Tools

### TensorBoard

```bash
# Train multiple experiments
ml-train --lr 0.001
ml-train --lr 0.01
ml-train --lr 0.1

# Compare in TensorBoard
tensorboard --logdir runs/
# Open http://localhost:6006
# All runs visible with meaningful names
```

### Shell Scripting

```bash
# Parallel training (if you have multiple GPUs)
ml-train --device cuda:0 --lr 0.001 &
ml-train --device cuda:1 --lr 0.01 &
wait
```

### Python Scripting

```python
import subprocess

learning_rates = [0.0001, 0.001, 0.01, 0.1]

for lr in learning_rates:
    cmd = f"ml-train --lr {lr}"
    subprocess.run(cmd, shell=True, check=True)
```

---

## Summary

**Key Takeaways:**
1. CLI overrides make experimentation fast
2. Run names are auto-generated from overrides
3. Systematic sweeps are easy
4. Original config never modified
5. Final config always saved

**Common Pattern:**
```bash
# 1. Train baseline
ml-train

# 2. Experiment with one parameter
ml-train --lr 0.01

# 3. Combine best settings
ml-train --lr 0.01 --batch_size 32

# 4. Compare results
tensorboard --logdir runs/
```

---

## Related Documentation

- [Configuration Overview](README.md) - How config system works
- [All Configuration Options](../configuration/) - What you can override
- [Training Guide](../user-guides/training.md) - Training workflows
- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md) - Systematic search

## Quick Reference

```bash
# Most common overrides
--batch_size <int>      # Batch size
--lr <float>            # Learning rate
--num_epochs <int>      # Training epochs
--data_dir <path>       # Dataset location
--num_workers <int>     # Data loading workers
--device <string>       # GPU/CPU selection
--resume <path>         # Resume training

# Examples
ml-train --batch_size 32
ml-train --lr 0.01 --num_epochs 50
ml-train --resume runs/base/last.pt
```
