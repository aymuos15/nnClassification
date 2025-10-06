# Test-Time Augmentation (TTA) Guide

Comprehensive guide to using Test-Time Augmentation for improved inference accuracy.

## What is TTA?

Test-Time Augmentation (TTA) is a technique that applies multiple augmented versions of each test image during inference and combines the predictions to improve accuracy and robustness.

**How it works:**
1. Take a test image
2. Create multiple augmented versions (e.g., flipped, rotated)
3. Run the model on all versions
4. Aggregate predictions (average, voting, etc.)
5. Use combined prediction as final output

**Benefits:**
- ✅ Improved accuracy (+1-3% typically)
- ✅ More robust to input variations
- ✅ No retraining required
- ✅ Works with any trained model

**Tradeoffs:**
- ❌ Slower inference (~5x for 5 augmentations)
- ❌ Requires more memory during inference
- ❌ Not suitable for real-time applications

---

## Quick Start

### CLI Usage (Recommended)

```bash
# Basic TTA (default: horizontal flip)
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta

# TTA with custom augmentations
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta \
  --tta-augmentations horizontal_flip vertical_flip

# TTA with all rotation augmentations
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta \
  --tta-augmentations horizontal_flip vertical_flip rotate_90 rotate_180 rotate_270

# TTA with different aggregation method
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta \
  --tta-aggregation voting  # or 'max'
```

### Config File Usage

```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations:
      - 'horizontal_flip'
      - 'vertical_flip'
    aggregation: 'mean'  # Soft voting (recommended)
```

Then run:
```bash
ml-inference --checkpoint_path runs/fold_0/weights/best.pt
```

---

## Available Augmentations

### Flip Augmentations

#### Horizontal Flip
Mirrors image left-to-right.

**Use for:**
- Natural scenes (animals, objects)
- Most classification tasks
- **Recommended as default**

**Avoid for:**
- Text recognition (letters get mirrored)
- Oriented objects (directional signs)

```yaml
augmentations: ['horizontal_flip']
```

#### Vertical Flip
Mirrors image top-to-bottom.

**Use for:**
- Aerial/satellite imagery
- Microscopy images
- Textures

**Avoid for:**
- Natural scenes (trees, animals - unnatural)
- Face recognition

```yaml
augmentations: ['vertical_flip']
```

### Rotation Augmentations

#### Rotate 90°, 180°, 270°
Rotates image by specified degrees.

**Use for:**
- Medical imaging (no inherent orientation)
- Aerial imagery
- Microscopy
- Texture classification

**Avoid for:**
- Natural scenes (upside-down animals look unnatural)
- Face recognition
- Scene understanding

```yaml
augmentations: ['rotate_90', 'rotate_180', 'rotate_270']
```

### Combining Augmentations

You can combine multiple augmentations:

```yaml
# Minimal (fast, good for most tasks)
augmentations: ['horizontal_flip']

# Moderate (balanced speed/accuracy)
augmentations: ['horizontal_flip', 'vertical_flip']

# Comprehensive (slow, maximum robustness)
augmentations:
  - 'horizontal_flip'
  - 'vertical_flip'
  - 'rotate_90'
  - 'rotate_180'
  - 'rotate_270'
```

**Note:** Horizontal + Vertical flip automatically includes combined H+V flip, so 2 augmentations = 4 versions (original, H, V, H+V).

---

## Aggregation Methods

### Mean (Soft Voting) - Recommended

Averages the logits (pre-softmax outputs) from all augmentations.

**How it works:**
```
final_logits = mean([logits1, logits2, logits3, ...])
final_prediction = argmax(final_logits)
```

**Best for:**
- Maximum accuracy (recommended default)
- Well-calibrated models
- Most classification tasks

**Example:**
```yaml
tta:
  aggregation: 'mean'
```

```bash
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta --tta-aggregation mean
```

### Max

Takes element-wise maximum of logits.

**How it works:**
```
final_logits = max([logits1, logits2, logits3, ...])
final_prediction = argmax(final_logits)
```

**Best for:**
- Conservative predictions
- When model is very confident on correct augmentations

**Example:**
```yaml
tta:
  aggregation: 'max'
```

### Voting (Hard Voting)

Each augmentation votes for a class, majority wins.

**How it works:**
```
predictions = [argmax(logits1), argmax(logits2), ...]
final_prediction = mode(predictions)  # Most common
```

**Best for:**
- Poorly calibrated models
- When individual predictions are reliable
- Binary classification

**Example:**
```yaml
tta:
  aggregation: 'voting'
```

**Aggregation Comparison:**
```
Method   | Accuracy | Use Case
---------|----------|----------------------------------
mean     | Highest  | Default, most tasks (recommended)
max      | Moderate | Conservative predictions
voting   | Moderate | Poorly calibrated models
```

---

## When to Use TTA

### ✅ Good Use Cases

**1. Medical Imaging**
- X-rays, MRIs, CT scans
- No inherent orientation
- Accuracy is critical
```yaml
augmentations: ['horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270']
```

**2. Aerial/Satellite Imagery**
- Object detection from above
- No canonical orientation
- Need robustness to camera angle
```yaml
augmentations: ['rotate_90', 'rotate_180', 'rotate_270']
```

**3. Microscopy**
- Cell classification
- Tissue analysis
- Orientation-invariant
```yaml
augmentations: ['horizontal_flip', 'vertical_flip', 'rotate_90', 'rotate_180', 'rotate_270']
```

**4. Competitions (Kaggle, etc.)**
- Need maximum accuracy
- Inference time not critical
- Final leaderboard submission
```yaml
augmentations: ['horizontal_flip', 'vertical_flip']
aggregation: 'mean'
```

**5. Quality Control**
- Product defect detection
- Can afford slower inference
- False negatives costly
```yaml
augmentations: ['horizontal_flip', 'vertical_flip']
```

### ❌ Avoid TTA For

**1. Real-time Applications**
- Video processing
- Live camera feeds
- Inference time critical
- **Use standard inference instead**

**2. Text/OCR**
- Flipped text is meaningless
- Rotations rarely help
- **TTA may hurt accuracy**

**3. Face Recognition**
- Faces have canonical orientation
- Flipped/rotated faces unnatural
- **Use standard inference**

**4. Well-Aligned Data**
- Dataset pre-processed with consistent orientation
- No expected variations
- **TTA provides minimal benefit**

---

## Performance Considerations

### Speed Impact

TTA is **N times slower** where N = number of augmentations + 1 (original).

**Examples:**
```
Augmentations          | Versions | Speed    | Example Time
-----------------------|----------|----------|-------------
horizontal_flip        | 2        | 0.5x     | 90 sec (was 45)
horizontal + vertical  | 4        | 0.25x    | 180 sec
h + v + rotate_90      | 5        | 0.2x     | 225 sec
All augmentations      | 8        | 0.125x   | 360 sec
```

### Memory Impact

TTA requires:
- Loading all augmented versions at once (per image)
- Storing intermediate predictions

**Memory usage:**
```
memory = batch_size × num_augmentations × image_size × channels
```

**Tips to reduce memory:**
1. Process images one at a time (batch_size=1)
2. Use fewer augmentations
3. Use mixed precision inference

### Accuracy Improvement

Typical improvements vary by domain:

```
Domain              | Typical Gain | Notes
--------------------|--------------|-------------------------
Medical imaging     | +2-4%        | High benefit (no orientation)
Aerial imagery      | +2-3%        | Moderate benefit
Natural scenes      | +1-2%        | Small benefit
Face recognition    | 0-1%         | Minimal or negative
Text/OCR            | -1-0%        | May hurt accuracy
```

---

## Best Practices

### 1. Start Simple

Begin with minimal augmentations:
```bash
# Start here
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta
```

Only add more if accuracy improves significantly.

### 2. Match Training Augmentations

Use TTA augmentations similar to training augmentations:

**Training used horizontal flip?** → Use horizontal flip in TTA
```yaml
tta:
  augmentations: ['horizontal_flip']
```

**Training used rotations?** → Use rotations in TTA
```yaml
tta:
  augmentations: ['rotate_90', 'rotate_180', 'rotate_270']
```

### 3. Validate on Validation Set First

Test TTA on validation set before using on test set:
```bash
# Test TTA effectiveness on validation set
# (Would require modifying inference.py to use val set)
```

Compare standard vs TTA accuracy to ensure benefit.

### 4. Use Soft Voting (Mean)

Unless you have a specific reason, use `aggregation: 'mean'`:
```yaml
tta:
  aggregation: 'mean'  # Best default
```

### 5. Consider TTA + Ensemble

For maximum accuracy, combine TTA with ensembling:
```bash
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  --tta
```

Typically provides +3-8% total accuracy gain.

---

## Examples by Domain

### Medical Imaging (Full TTA)

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
    aggregation: 'mean'
```

### Natural Scenes (Minimal TTA)

```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations:
      - 'horizontal_flip'
    aggregation: 'mean'
```

### Aerial Imagery (Rotation Only)

```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations:
      - 'rotate_90'
      - 'rotate_180'
      - 'rotate_270'
    aggregation: 'mean'
```

### Competition Submission (Balanced)

```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations:
      - 'horizontal_flip'
      - 'vertical_flip'
    aggregation: 'mean'
```

---

## Troubleshooting

### TTA Doesn't Improve Accuracy

**Possible causes:**
1. Augmentations don't match task (e.g., flipping text)
2. Model already robust to variations
3. Dataset is well-aligned
4. Wrong aggregation method

**Solutions:**
- Try different augmentations
- Try different aggregation (mean vs voting)
- Check training augmentations
- Consider TTA may not help this task

### TTA Makes Accuracy Worse

**Likely causes:**
1. Using inappropriate augmentations (e.g., flipping faces)
2. Model overfit to specific orientation
3. Using voting with poor calibration

**Solutions:**
- Remove inappropriate augmentations
- Use only augmentations that preserve semantics
- Switch to `mean` aggregation

### Out of Memory

**Solutions:**
1. Reduce batch size to 1
2. Use fewer augmentations
3. Enable mixed precision:
   ```yaml
   inference:
     strategy: 'tta'
     amp_dtype: 'float16'  # Add this
     tta:
       augmentations: ['horizontal_flip']
   ```

### Too Slow

**Solutions:**
1. Reduce number of augmentations
2. Use only horizontal flip (fastest)
3. Consider ensemble instead (may be faster with similar gain)

---

## Related Guides

- [Inference Guide](inference.md) - General inference documentation
- [Ensemble Guide](ensemble-inference.md) - Combining multiple models
- [Model Export](model-export.md) - Export for production

---

## Summary

**Key Takeaways:**
- ✅ TTA improves accuracy by +1-3% typically
- ✅ Best for: medical imaging, aerial imagery, competitions
- ✅ Use `mean` aggregation (soft voting) as default
- ✅ Start with `horizontal_flip` only
- ❌ Avoid for: real-time, text/OCR, face recognition
- ❌ ~5x slower (5 augmentations)

**Quick Start Command:**
```bash
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta
```
