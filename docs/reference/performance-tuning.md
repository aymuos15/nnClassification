# Performance Tuning

Optimize training speed and memory usage.

## Training Speed

### Batch Size
```bash
# Larger batch = faster (if GPU memory allows)
python train.py --batch_size 64
```

### Data Loading
```bash
# More workers = faster data loading
python train.py --num_workers 8
```

### Determinism
```yaml
# Non-deterministic = faster
deterministic: false  # (default)
```

### Image Size
```yaml
# Smaller images = faster
transforms:
  train:
    resize: [128, 128]  # Instead of [224, 224]
```

## Memory Usage

### Reduce Batch Size
```bash
python train.py --batch_size 8
```

### Reduce Image Resolution
```yaml
transforms:
  train:
    resize: [192, 192]
```

### Reduce Workers
```bash
python train.py --num_workers 2
```

### Smaller Model
```yaml
model:
  architecture: 'mobilenet_v3_small'
```

## GPU Optimization

### Check Utilization
```bash
# Should be near 100%
watch -n 1 nvidia-smi
```

### Maximize GPU Usage
1. Increase batch size until OOM
2. Then reduce slightly
3. Verify 95%+ utilization

### Multiple GPUs
Currently not supported. Future extension.

## Disk I/O

### Use SSD
Faster than HDD for image loading.

### Reduce Image Size
Smaller files load faster.

### Cache Preprocessing
For repeated experiments (code modification needed).

## Profiling

### Time Bottlenecks
```python
import time

start = time.time()
# ... training code ...
print(f"Epoch time: {time.time() - start:.2f}s")
```

### PyTorch Profiler
For advanced profiling (code modification needed).

## Benchmarks

Example training speed (ResNet18, batch 32):
- GPU (RTX 3090): ~0.5s/epoch
- GPU (GTX 1080): ~1.5s/epoch
- CPU: ~60s/epoch

## Related

- [Training Guide](../user-guides/training.md)
- [Configuration](../configuration/overview.md)
- [Troubleshooting](troubleshooting.md)
