# Monitoring Guide

Track training progress with TensorBoard, logs, and metrics.

## TensorBoard

### Start TensorBoard
```bash
# All runs
tensorboard --logdir runs/

# Specific run
tensorboard --logdir runs/base/tensorboard

# Custom port
tensorboard --logdir runs/ --port 6007
```

### Open in Browser
```
http://localhost:6006
```

### What You'll See

**Scalars:**
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule

**Images:**
- Confusion matrices

**Graphs:**
- Model architecture (if logged)

## Log Files

### Training Log
```bash
# View full log
cat runs/base/logs/train.log

# Watch live
tail -f runs/base/logs/train.log

# Search for errors
grep ERROR runs/base/logs/train.log
```

### Summary
```bash
# Quick overview
cat runs/base/summary.txt
```

## GPU Monitoring

### Check Usage
```bash
# Single check
nvidia-smi

# Continuous (updates every second)
watch -n 1 nvidia-smi
```

### What to Look For
- GPU utilization (should be near 100%)
- Memory usage
- Temperature

## Metrics

### During Training
- Loss decreasing
- Accuracy increasing
- No divergence (NaN)

### After Training
- Confusion matrices
- Classification reports
- Per-sample results

## Best Practices

1. **Always use TensorBoard** - Visual comparison
2. **Monitor GPU** - Ensure full utilization
3. **Check logs** - Catch errors early
4. **Compare runs** - Use TensorBoard multi-run view
5. **Save summaries** - Document results

## Related

- [Training Guide](training.md)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
EOF3
