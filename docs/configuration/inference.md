# Inference Configuration

Configure inference strategies for test-time predictions and model evaluation.

## Overview

The `inference` section controls how models make predictions during testing and deployment. Different strategies optimize for speed vs accuracy.

## Basic Configuration

```yaml
inference:
  strategy: 'standard'  # Inference strategy to use
  use_ema: false        # Use EMA weights if available
```

## Inference Strategies

### 1. Standard Inference (Default)

Basic PyTorch inference - fastest option.

```yaml
inference:
  strategy: 'standard'
```

**When to use:**
- Production deployment (speed critical)
- Baseline performance evaluation
- Real-time applications

**Performance:** 1x speed (baseline)

---

### 2. Mixed Precision Inference

Uses PyTorch AMP for 2-3x speedup on modern GPUs.

```yaml
inference:
  strategy: 'mixed_precision'
  amp_dtype: 'float16'  # or 'bfloat16' for newer GPUs
```

**When to use:**
- Single GPU inference
- Speed optimization needed
- Modern NVIDIA GPU available (Volta/Turing/Ampere)

**Performance:** ~2.5x faster than standard

---

### 3. Accelerate Inference

Multi-GPU/distributed inference using Hugging Face Accelerate.

```yaml
inference:
  strategy: 'accelerate'
```

**When to use:**
- Multiple GPUs available
- Large batch inference
- Distributed deployment

**Setup required:**
```bash
uv pip install accelerate
accelerate config
```

**Performance:** ~3.5x faster with 2 GPUs

---

### 4. Test-Time Augmentation (TTA)

Apply multiple augmented versions of each image for improved robustness.

```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations:
      - 'horizontal_flip'
      - 'vertical_flip'
      - 'rotate_90'
      - 'rotate_180'
      - 'rotate_270'
      - 'brightness'
      - 'contrast'
    aggregation: 'mean'  # How to combine predictions
```

**Available augmentations:**
- `horizontal_flip` - Mirror horizontally
- `vertical_flip` - Mirror vertically
- `rotate_90` - Rotate 90° clockwise
- `rotate_180` - Rotate 180°
- `rotate_270` - Rotate 270° clockwise
- `brightness` - Adjust brightness (±10%)
- `contrast` - Adjust contrast (±10%)

**Aggregation methods:**
- `mean` (soft voting) - Average probabilities (recommended)
- `max` - Take maximum probability per class
- `voting` (hard voting) - Majority vote on class predictions

**When to use:**
- Accuracy more important than speed
- Model predictions on borderline cases
- Competitive benchmarks

**Performance:**
- ~5x slower (with 5 augmentations)
- +1-3% accuracy improvement

---

### 5. Ensemble Inference

Combine predictions from multiple trained models (e.g., from cross-validation folds).

```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
      - 'runs/fold_2/weights/best.pt'
    aggregation: 'soft_voting'  # How to combine model predictions
    weights: [0.4, 0.3, 0.3]     # Optional: weighted aggregation
```

**Aggregation methods:**
- `soft_voting` - Weighted average of probabilities (recommended)
- `hard_voting` - Majority vote on class predictions
- `weighted` - Custom weights per model (specify `weights`)

**When to use:**
- Have multiple trained folds/models
- Maximum accuracy needed
- Production deployment with ensemble

**Performance:**
- ~5x slower (with 5 models)
- +2-5% accuracy improvement

---

### 6. TTA + Ensemble (Maximum Accuracy)

Combine both TTA and ensemble for best possible accuracy.

```yaml
inference:
  strategy: 'tta_ensemble'
  tta:
    augmentations:
      - 'horizontal_flip'
      - 'vertical_flip'
    aggregation: 'mean'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
    aggregation: 'soft_voting'
```

**When to use:**
- Maximum accuracy required
- Inference speed not critical
- Competitive benchmarks/leaderboards

**Performance:**
- ~25x slower (5 models × 5 augmentations)
- +3-8% total accuracy improvement

---

## EMA Weights

Use Exponential Moving Average weights if available:

```yaml
inference:
  use_ema: true  # Use EMA weights from checkpoint
```

EMA weights typically provide 0.5-2% better accuracy than regular weights. Automatically used if available in checkpoint.

---

## CLI Usage

You can also configure inference via CLI flags:

```bash
# Standard inference
ml-inference --checkpoint_path runs/fold_0/weights/best.pt

# TTA inference
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta

# TTA with specific augmentations
ml-inference --checkpoint_path runs/fold_0/weights/best.pt \
  --tta --tta-augmentations horizontal_flip vertical_flip

# Ensemble inference
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt

# TTA + Ensemble
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  --tta
```

---

## Strategy Selection Guide

| Priority | Recommended Strategy | Expected Performance |
|----------|---------------------|---------------------|
| **Speed** | `standard` or `mixed_precision` | Baseline to 2.5x faster |
| **Accuracy** | `tta` or `ensemble` | +1-5% accuracy, 5x slower |
| **Maximum Accuracy** | `tta_ensemble` | +3-8% accuracy, 25x slower |
| **Multi-GPU** | `accelerate` | 3.5x faster with 2 GPUs |

---

## Examples

### Production Deployment (Speed)
```yaml
inference:
  strategy: 'mixed_precision'
  amp_dtype: 'float16'
  use_ema: true
```

### Research/Competition (Accuracy)
```yaml
inference:
  strategy: 'tta_ensemble'
  tta:
    augmentations: ['horizontal_flip', 'vertical_flip', 'rotate_90']
    aggregation: 'mean'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
      - 'runs/fold_2/weights/best.pt'
    aggregation: 'soft_voting'
  use_ema: true
```

### Balanced (Speed + Accuracy)
```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations: ['horizontal_flip', 'vertical_flip']
    aggregation: 'mean'
  use_ema: true
```

---

## Related Guides

- [Inference Guide](../user-guides/inference.md) - Detailed inference workflows
- [Test-Time Augmentation](../user-guides/test-time-augmentation.md) - TTA deep dive
- [Ensemble Inference](../user-guides/ensemble-inference.md) - Ensemble strategies
- [Performance Tuning](../reference/performance-tuning.md) - Optimization tips
