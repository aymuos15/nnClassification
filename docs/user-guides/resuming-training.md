# Resuming Training Guide

How to resume interrupted training and extend training runs.

## Resume from Last Checkpoint

```bash
# Basic resumption
ml-train --resume runs/base/last.pt

# Resume and train more epochs
ml-train --resume runs/base/last.pt --num_epochs 100
```

## What Gets Restored

- Model weights
- Optimizer state (momentum)
- Scheduler state (LR schedule)
- Training history (losses, accuracies)
- Best accuracy achieved
- Random states (reproducibility)
- Current epoch number

## Use Cases

### 1. Training Interrupted
```bash
# Crashed at epoch 47
ml-train --resume runs/base/last.pt
# Continues from epoch 48
```

### 2. Extend Training
```bash
# Initially trained 25 epochs
ml-train --num_epochs 25

# Not converged, train 25 more
ml-train --resume runs/base/last.pt --num_epochs 50
# Continues to epoch 50
```

### 3. Fine-tune Further
```bash
# Lower LR for fine-tuning
ml-train --resume runs/base/best.pt --lr 0.0001 --num_epochs 75
```

## Troubleshooting

### Checkpoint Not Found
```bash
# Check file exists
ls runs/base/weights/
```

### Device Mismatch
If trained on GPU, resumption needs GPU too (or modify checkpoint).

## Related

- [Training Guide](training.md)
- [Checkpointing](../architecture/ml-src-modules.md)
