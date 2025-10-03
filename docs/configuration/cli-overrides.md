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
| `--data_dir` | `data.data_dir` | string | `--data_dir data/my_dataset` |
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
python train.py --batch_size 32
```

### Multiple Overrides

```bash
# Override several parameters
python train.py --batch_size 32 --lr 0.01 --num_epochs 50
```

### Custom Config File

```bash
# Use custom config
python train.py --config configs/my_config.yaml

# Custom config + overrides
python train.py --config configs/my_config.yaml --batch_size 64
```

### Resume Training

```bash
# Resume from last checkpoint
python train.py --resume runs/base/last.pt

# Resume and train more epochs
python train.py --resume runs/base/last.pt --num_epochs 50
```

---

## Run Directory Naming

Run directories are automatically named based on overrides. This creates organized, self-documenting experiment folders.

### Naming Rules

**No overrides:**
```bash
python train.py
# Creates: runs/base/
```

**Single override:**
```bash
python train.py --batch_size 32
# Creates: runs/batch_32/

python train.py --lr 0.01
# Creates: runs/lr_0.01/

python train.py --num_epochs 50
# Creates: runs/epochs_50/
```

**Multiple overrides:**
```bash
python train.py --batch_size 32 --num_epochs 50
# Creates: runs/batch_32_epochs_50/

python train.py --batch_size 32 --lr 0.01 --num_epochs 50
# Creates: runs/batch_32_epochs_50_lr_0.01/
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
python train.py --batch_size 16
python train.py --batch_size 32
python train.py --batch_size 64

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
    python train.py --lr $lr --batch_size $bs
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
python train.py --data_dir data/dataset1
python train.py --data_dir data/dataset2
python train.py --data_dir data/dataset3

# Note: All create runs/base/ (data_dir doesn't affect run name)
# Solution: Use different configs or add meaningful overrides
python train.py --data_dir data/dataset1 --num_epochs 25
python train.py --data_dir data/dataset2 --num_epochs 50
```

### 4. GPU Selection

```bash
# Use specific GPU (doesn't affect run name)
python train.py --device cuda:0
python train.py --device cuda:1

# Or via environment variable
CUDA_VISIBLE_DEVICES=1 python train.py
```

### 5. Resume and Extend

```bash
# Train 25 epochs
python train.py --num_epochs 25
# Saves to: runs/epochs_25/

# If not converged, resume and train more
python train.py --resume runs/epochs_25/last.pt --num_epochs 50
# Note: Continues training up to epoch 50 total
```

---

## Advanced Usage

### Config File + Overrides

You can combine custom configs with CLI overrides:

```bash
# configs/production.yaml has most settings
# Override just batch_size
python train.py --config configs/production.yaml --batch_size 128
```

**Use case:** Base config for a project, tweak individual params

### Debugging with Overrides

```bash
# Quick debug run (small batch, few epochs, no workers)
python train.py --batch_size 2 --num_epochs 2 --num_workers 0

# Creates: runs/batch_2_epochs_2/
```

### Full Override Example

```bash
python train.py \
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
python train.py

# Then experiment with overrides
python train.py --lr 0.01
python train.py --batch_size 32
```

### 2. Systematic Exploration

```bash
# Test one parameter at a time
python train.py --lr 0.0001
python train.py --lr 0.001
python train.py --lr 0.01

# Then combine best values
python train.py --lr 0.001 --batch_size 32
```

### 3. Document Experiments

```bash
# Create a script to track experiments
cat > experiments.sh << 'EOF'
#!/bin/bash

# Experiment 1: Baseline
python train.py

# Experiment 2: Higher LR
python train.py --lr 0.01

# Experiment 3: Larger batch
python train.py --batch_size 32 --lr 0.002
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
    python train.py --lr $lr --batch_size $bs
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
python train.py --batch_size 32 --lr 0.01

# Option 3: Resume training
python train.py --resume runs/batch_32/last.pt
```

### Override Not Taking Effect

**Problem:** Changed CLI arg but config still shows old value

**Solution:**
```bash
# Check you're using correct argument name
python train.py --lr 0.01  # ✅ Correct

python train.py --learning_rate 0.01  # ❌ Wrong (not supported)

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

python train.py --config configs/custom_transforms.yaml

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
python train.py --lr 0.01 --batch_size 32
```

### Custom Config Files

✅ Full control  
✅ Version controlled  
✅ Shareable  
❌ Requires file creation  

```bash
python train.py --config configs/experiment1.yaml
```

### Hybrid Approach (Recommended)

✅ Best of both worlds

```bash
# Base config + targeted overrides
python train.py --config configs/base.yaml --lr 0.01
```

---

## Integration with Other Tools

### TensorBoard

```bash
# Train multiple experiments
python train.py --lr 0.001
python train.py --lr 0.01
python train.py --lr 0.1

# Compare in TensorBoard
tensorboard --logdir runs/
# Open http://localhost:6006
# All runs visible with meaningful names
```

### Shell Scripting

```bash
# Parallel training (if you have multiple GPUs)
python train.py --device cuda:0 --lr 0.001 &
python train.py --device cuda:1 --lr 0.01 &
wait
```

### Python Scripting

```python
import subprocess

learning_rates = [0.0001, 0.001, 0.01, 0.1]

for lr in learning_rates:
    cmd = f"python train.py --lr {lr}"
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
python train.py

# 2. Experiment with one parameter
python train.py --lr 0.01

# 3. Combine best settings
python train.py --lr 0.01 --batch_size 32

# 4. Compare results
tensorboard --logdir runs/
```

---

## Related Documentation

- [Configuration Overview](overview.md) - How config system works
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
python train.py --batch_size 32
python train.py --lr 0.01 --num_epochs 50
python train.py --resume runs/base/last.pt
```
