# Performance Tuning

Optimize training speed and memory usage.

## Training Speed

### Mixed Precision Training (2-3x Faster) ⭐

**The #1 performance optimization** - Use mixed precision for 2-3x speedup on modern GPUs.

```yaml
training:
  trainer_type: 'mixed_precision'
  amp_dtype: 'float16'  # or 'bfloat16' for A100/RTX 40 series
```

**Benefits:**
- 2-3x faster training
- ~50% memory reduction
- Minimal accuracy impact
- No code changes needed

**Requirements:**
- NVIDIA GPU (Volta/Turing/Ampere or newer: GTX 1080 Ti, RTX 20/30/40 series, A100)
- CUDA support

**When to use:**
- Single GPU training (most common)
- Want maximum speed without multi-GPU complexity
- **Recommended for all production training**

See: [Advanced Training Guide](../user-guides/advanced-training.md#mixed-precision-training)

---

### Batch Size
```bash
# Larger batch = faster (if GPU memory allows)
ml-train --batch_size 64
```

### Data Loading
```bash
# More workers = faster data loading
ml-train --num_workers 8
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
ml-train --batch_size 8
```

### Reduce Image Resolution
```yaml
transforms:
  train:
    resize: [192, 192]
```

### Reduce Workers
```bash
ml-train --num_workers 2
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

Multi-GPU training is supported via Hugging Face Accelerate:

```bash
# One-time setup
uv pip install accelerate
accelerate config

# Launch multi-GPU training
accelerate launch ml-train --config configs/my_config.yaml
```

**Configuration:**
```yaml
training:
  trainer_type: 'accelerate'
  batch_size: 32  # Per-device batch size
```

**Benefits:**
- ~3.5x faster with 2 GPUs
- Linear scaling with more GPUs
- Supports distributed training

See: [Advanced Training Guide](../user-guides/advanced-training.md#multi-gpu-training)

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
- [Configuration](../configuration/README.md)
- [Troubleshooting](troubleshooting.md)
