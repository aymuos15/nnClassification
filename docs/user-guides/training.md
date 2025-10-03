# Training Guide

Complete guide to training workflows, best practices, and common scenarios.

## Basic Training

### Quick Start
```bash
# Default training
python train.py

# Custom epochs
python train.py --num_epochs 50

# Custom batch size and LR
python train.py --batch_size 32 --lr 0.01
```

## Training Workflows

### 1. Initial Experiment
```bash
# Quick test (5 epochs)
python train.py --num_epochs 5

# Check results
tensorboard --logdir runs/
```

### 2. Hyperparameter Search
```bash
# Test learning rates
python train.py --lr 0.0001 --num_epochs 10
python train.py --lr 0.001 --num_epochs 10
python train.py --lr 0.01 --num_epochs 10

# Compare in TensorBoard
tensorboard --logdir runs/
```

### 3. Full Training
```bash
# Use best hyperparams from search
python train.py --lr 0.001 --batch_size 32 --num_epochs 100
```

## Monitoring Training

### TensorBoard
```bash
# Watch training live
tensorboard --logdir runs/

# Open http://localhost:6006
```

### Log Files
```bash
# Watch training log
tail -f runs/{run_name}/logs/train.log

# View summary
cat runs/{run_name}/summary.txt
```

### GPU Monitoring
```bash
# Check GPU usage
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

## Common Scenarios

### Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 8

# Or use CPU
python train.py --device cpu
```

### Slow Training
```bash
# Increase workers
python train.py --num_workers 8

# Larger batch size
python train.py --batch_size 64
```

### Resume After Crash
```bash
# Resume from last checkpoint
python train.py --resume runs/base/last.pt
```

## Best Practices

1. **Start small** - Test with few epochs first
2. **Monitor metrics** - Use TensorBoard
3. **Save checkpoints** - Resume if interrupted
4. **Track experiments** - Document what works
5. **Use validation** - Don't overfit

## Related

- [Configuration](../configuration/overview.md)
- [Hyperparameter Tuning](hyperparameter-tuning.md)
- [Resuming Training](resuming-training.md)
- [Troubleshooting](../reference/troubleshooting.md)
