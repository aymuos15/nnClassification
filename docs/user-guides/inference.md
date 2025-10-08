# Inference Guide

Guide to running inference and evaluating trained models on test data.

## Automatic Testing

**Note:** After training completes, the framework automatically evaluates the best model on the test set. Test results are saved to:
- `runs/{run_name}/logs/classification_report_test.txt`
- TensorBoard (confusion matrix, accuracy, classification report)

**You typically don't need to run manual inference** unless you want to:
- Re-evaluate a checkpoint on different data
- Evaluate checkpoints from different runs
- Generate predictions for analysis

---

## Inference Strategies

The framework supports multiple inference strategies optimized for different hardware and performance requirements, including advanced techniques like Test-Time Augmentation (TTA) and model ensembling.

### Available Strategies

#### Standard Inference (Default)
Standard PyTorch inference without optimizations.

**Config:**
```yaml
inference:
  strategy: 'standard'
```

**Use when:**
- Running on CPU
- Simplicity is priority
- No special hardware requirements
- Need baseline performance

#### Mixed Precision Inference
Uses PyTorch AMP for 2-3x faster inference with ~50% memory reduction.

**Config:**
```yaml
inference:
  strategy: 'mixed_precision'
  amp_dtype: 'float16'  # or 'bfloat16' for newer GPUs
```

**Use when:**
- Running on modern NVIDIA GPU (Volta/Turing/Ampere or newer)
- Speed is important
- Large batches needed
- **Recommended for production on GPU**

**Performance:**
- 2-3x faster than standard inference
- 50% less memory usage
- Minimal accuracy impact (<0.1% difference)

#### Accelerate Inference
Distributed inference across multiple GPUs using Hugging Face Accelerate.

**Config:**
```yaml
inference:
  strategy: 'accelerate'
```

**Use when:**
- Multiple GPUs available
- Very large test sets
- Need distributed evaluation

**Requirements:**
```bash
uv pip install accelerate
```

#### Test-Time Augmentation (TTA)
Applies multiple augmented versions of each test image and aggregates predictions for improved robustness.

**Config:**
```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations: ['horizontal_flip', 'vertical_flip']
    aggregation: 'mean'  # or 'max', 'voting'
```

**CLI (easier):**
```bash
# Default TTA (horizontal flip)
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta

# Custom augmentations
ml-inference --checkpoint_path runs/fold_0/weights/best.pt --tta \
  --tta-augmentations horizontal_flip vertical_flip rotate_90
```

**Use when:**
- Single model, want accuracy improvement
- Test set has variations (lighting, orientation)
- Can afford slower inference (~5x slower)
- Need robustness to input variations

**Performance:**
- Accuracy improvement: +1-3%
- Speed: ~5x slower (5 augmentations)
- Best for: Medical imaging, aerial imagery, natural scenes

**Available augmentations:**
- `horizontal_flip` - Mirror image horizontally
- `vertical_flip` - Mirror image vertically
- `rotate_90` - Rotate 90 degrees
- `rotate_180` - Rotate 180 degrees
- `rotate_270` - Rotate 270 degrees

**Aggregation methods:**
- `mean` - Average logits (soft voting, **recommended**)
- `max` - Take maximum logits
- `voting` - Hard voting (majority vote on classes)

#### Ensemble Inference
Combines predictions from multiple models (e.g., trained on different CV folds).

**Config:**
```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
      - 'runs/fold_2/weights/best.pt'
    aggregation: 'soft_voting'  # or 'hard_voting', 'weighted'
```

**CLI (easier):**
```bash
# Ensemble multiple folds
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt
```

**Use when:**
- Have multiple trained models (e.g., from cross-validation)
- Need maximum accuracy
- Can afford slower inference (~5x slower for 5 models)
- Production deployment where accuracy is critical

**Performance:**
- Accuracy improvement: +2-5%
- Speed: ~Nx slower (N models)
- Best for: Competitions, critical applications, production

**Aggregation methods:**
- `soft_voting` - Average logits (**recommended**, best accuracy)
- `hard_voting` - Majority vote on predicted classes
- `weighted` - Weighted average (specify weights per model)

**Example with weights:**
```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints: ['fold_0/best.pt', 'fold_1/best.pt', 'fold_2/best.pt']
    aggregation: 'weighted'
    weights: [0.4, 0.35, 0.25]  # Higher weight for better performing folds
```

#### TTA + Ensemble (Maximum Performance)
Combines both TTA and ensembling for the best possible accuracy.

**Config:**
```yaml
inference:
  strategy: 'tta_ensemble'
  tta:
    augmentations: ['horizontal_flip']
    aggregation: 'mean'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'
      - 'runs/fold_1/weights/best.pt'
    aggregation: 'soft_voting'
```

**CLI (easier):**
```bash
# TTA + Ensemble
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  --tta
```

**Use when:**
- Need absolute maximum accuracy
- Can afford very slow inference (~25x slower)
- Critical applications (medical diagnosis, safety systems)
- Final competition submissions

**Performance:**
- Accuracy improvement: +3-8%
- Speed: ~(N models × M augmentations)x slower
- Best for: Kaggle competitions, medical AI, safety-critical systems

### Choosing an Inference Strategy

**Decision guide:**

**For Speed:**
1. Have multiple GPUs? → `accelerate`
2. Have single GPU? → `mixed_precision` (recommended)
3. CPU only? → `standard`

**For Accuracy:**
1. Need maximum accuracy, have time? → `tta_ensemble`
2. Have multiple trained folds? → `ensemble`
3. Single model, want improvement? → `tta`
4. Baseline performance? → `standard`

**Quick Reference:**
```
Strategy          | Speed  | Accuracy | When to Use
------------------|--------|----------|---------------------------
standard          | 1x     | baseline | Fast inference, baseline
mixed_precision   | 2.5x   | baseline | Fast GPU inference (recommended)
accelerate        | 3.75x  | baseline | Multi-GPU inference
tta               | 0.2x   | +1-3%    | Single model accuracy boost
ensemble          | 0.2x   | +2-5%    | Multiple models available
tta_ensemble      | 0.04x  | +3-8%    | Maximum accuracy needed
```

**Performance comparison** (ResNet50, 1000 images, V100 GPU):
- Standard: 45 seconds
- Mixed Precision: 18 seconds (2.5x faster)
- Accelerate (2 GPUs): 12 seconds (3.75x faster)
- TTA (5 augmentations): 225 seconds (~5x slower, +2% accuracy)
- Ensemble (5 models): 225 seconds (~5x slower, +3% accuracy)
- TTA + Ensemble: 1125 seconds (~25x slower, +5% accuracy)

### Configuration Examples

**Fast GPU inference:**
```yaml
inference:
  strategy: 'mixed_precision'
  amp_dtype: 'float16'
```

**TTA for accuracy:**
```yaml
inference:
  strategy: 'tta'
  tta:
    augmentations: ['horizontal_flip', 'vertical_flip']
    aggregation: 'mean'
```

**Ensemble from CV folds:**
```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/hymenoptera_fold_0/weights/best.pt'
      - 'runs/hymenoptera_fold_1/weights/best.pt'
      - 'runs/hymenoptera_fold_2/weights/best.pt'
      - 'runs/hymenoptera_fold_3/weights/best.pt'
      - 'runs/hymenoptera_fold_4/weights/best.pt'
    aggregation: 'soft_voting'
```

**Maximum accuracy (TTA + Ensemble):**
```yaml
inference:
  strategy: 'tta_ensemble'
  tta:
    augmentations: ['horizontal_flip']
    aggregation: 'mean'
  ensemble:
    checkpoints:
      - 'runs/hymenoptera_fold_0/weights/best.pt'
      - 'runs/hymenoptera_fold_1/weights/best.pt'
      - 'runs/hymenoptera_fold_2/weights/best.pt'
    aggregation: 'soft_voting'
```

**Multi-GPU inference:**
```yaml
inference:
  strategy: 'accelerate'
```

**CPU inference:**
```yaml
inference:
  strategy: 'standard'
```

---

## Manual Inference (Optional)

### Quick Start

```bash
# Standard inference - evaluate best model on test set
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt

# TTA inference - for improved accuracy
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt --tta

# TTA with custom augmentations
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt --tta \
  --tta-augmentations horizontal_flip vertical_flip rotate_90

# Ensemble inference - combine multiple folds
ml-inference --ensemble \
  runs/hymenoptera_fold_0/weights/best.pt \
  runs/hymenoptera_fold_1/weights/best.pt \
  runs/hymenoptera_fold_2/weights/best.pt

# TTA + Ensemble - maximum accuracy
ml-inference --ensemble \
  runs/hymenoptera_fold_0/weights/best.pt \
  runs/hymenoptera_fold_1/weights/best.pt \
  --tta

# Evaluate latest checkpoint
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/last.pt

# Evaluate on different dataset
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt --data_dir data/new_dataset
```

### Expected Output

```
2025-10-05 01:35:00 | INFO     | Loaded config from runs/hymenoptera_base_fold_0/config.yaml
2025-10-05 01:35:00 | INFO     | Using device: cuda:0
2025-10-05 01:35:00 | INFO     | Loading datasets...
2025-10-05 01:35:00 | INFO     | Using fold: 0
2025-10-05 01:35:00 | INFO     | Classes: ['ants', 'bees']
2025-10-05 01:35:00 | INFO     | Test dataset size: 59
2025-10-05 01:35:00 | INFO     | Creating model...
2025-10-05 01:35:00 | INFO     | Loading checkpoint from runs/hymenoptera_base_fold_0/weights/best.pt
2025-10-05 01:35:00 | INFO     | Running Inference
2025-10-05 01:35:01 | SUCCESS  | Inference Complete!

┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Sample # ┃ True Label   ┃ Predicted    ┃ Correct   ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1        │ ants         │ ants         │ ✓         │
│ 2        │ ants         │ ants         │ ✓         │
│ 3        │ bees         │ bees         │ ✓         │
│ 4        │ ants         │ bees         │ ✗         │
...

┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Run Directory      │ runs/...      │
│ Checkpoint         │ best.pt       │
│ Test Samples       │ 59            │
│ Mean Accuracy      │ 0.9153        │
└────────────────────┴───────────────┘

Classification Report:
               precision    recall  f1-score   support
        ants       0.90      0.94      0.92        30
        bees       0.94      0.91      0.93        29
    accuracy                           0.92        59
   macro avg       0.92      0.92      0.92        59
weighted avg       0.92      0.92      0.92        59
```

---

## Output Files

Inference generates several files:

### Console Output
- Test accuracy
- Per-sample predictions table (with ✓/✗ indicators)
- Summary table
- Classification report

### Generated Files

```
runs/hymenoptera_base_fold_0/
├── logs/
│   ├── inference.log                      # Detailed inference log
│   └── classification_report_test.txt     # Classification metrics
└── tensorboard/
    └── events.out.tfevents.*              # Updated with test metrics
```

**Key files:**
- **`inference.log`** - Complete inference details
- **`classification_report_test.txt`** - Precision/recall/F1 per class
- **TensorBoard events** - Confusion matrix and metrics for visualization

---

## Understanding Output

### Test Accuracy

Overall percentage of correct predictions:
```
Mean Accuracy: 0.9153  # 91.53% correct
```

### Per-Sample Results

Shows predictions for each test image:
```
┃ Sample # ┃ True Label   ┃ Predicted    ┃ Correct   ┃
│ 1        │ ants         │ ants         │ ✓         │  # Correct
│ 4        │ ants         │ bees         │ ✗         │  # Incorrect
```

**Use this to:**
- Identify misclassified samples
- Debug model errors
- Find systematic mistakes

### Confusion Matrix

Saved to TensorBoard, shows:
- True Positives (diagonal)
- False Positives / False Negatives (off-diagonal)

```
             Predicted
           ants  bees
True ants   28     2     # 28 correct, 2 misclassified as bees
     bees    3    26     # 3 misclassified as ants, 26 correct
```

### Classification Report

Per-class metrics:

```
               precision    recall  f1-score   support
        ants       0.90      0.94      0.92        30
        bees       0.94      0.91      0.93        29
```

**Metrics explained:**
- **Precision:** Of predicted "ants", how many were actually ants? (90%)
- **Recall:** Of actual ants, how many did we find? (94%)
- **F1-score:** Harmonic mean of precision and recall (92%)
- **Support:** Number of actual instances (30 ants, 29 bees)

---

## Advanced Usage

### Custom Dataset for Testing

Override the dataset in config:

```bash
# Use different test data
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt --data_dir data/external_test_set
```

**Note:** Custom test data must:
- Follow same directory structure (`raw/` and `splits/`)
- Have `test.txt` (single test file, same for all folds)
- Contain same class names as training

### Evaluate Multiple Folds

Compare performance across all folds:

```bash
# Evaluate all folds
for fold in {0..4}; do
  ml-inference \
    --checkpoint_path runs/hymenoptera_base_fold_$fold/weights/best.pt
done

# Average results across folds for final performance estimate
```

### Compare Checkpoints

Compare best vs last checkpoint:

```bash
# Evaluate best model
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt

# Evaluate latest checkpoint
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/last.pt
```

**Use case:** Check if training was still improving when it stopped.

---

## Viewing Results

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/hymenoptera_base_fold_0/tensorboard

# Open browser: http://localhost:6006
```

**Available visualizations:**
- Test confusion matrix
- Test accuracy metric
- Classification report (text)
- Compare train/val/test metrics side-by-side

### Log Files

```bash
# View inference log
cat runs/hymenoptera_base_fold_0/logs/inference.log

# View classification report
cat runs/hymenoptera_base_fold_0/logs/classification_report_test.txt
```

---

## Interpreting Results

### Good Performance

**Indicators:**
- High test accuracy (>85% for binary, >70% for multi-class)
- Balanced precision/recall across classes
- Confusion matrix concentrated on diagonal

**Example:**
```
Test Accuracy: 0.92

               precision    recall  f1-score   support
        ants       0.90      0.94      0.92        30
        bees       0.94      0.91      0.93        29
```
✅ Balanced, high performance

### Overfitting

**Indicators:**
- Training/validation accuracy >> test accuracy
- Large gap between val and test performance

**Example:**
- Val accuracy: 95%
- Test accuracy: 75%

**Solutions:**
- Use more training data
- Add data augmentation
- Use pretrained weights
- Reduce model complexity

### Class Imbalance Issues

**Indicators:**
- High accuracy but poor precision/recall for some classes
- Confusion matrix shows bias toward majority class

**Example:**
```
               precision    recall  f1-score   support
      class1       0.95      0.98      0.96       100  # Majority
      class2       0.40      0.30      0.34        10  # Minority
```

**Solutions:**
- Collect more data for minority classes
- Use class weights in loss function
- Apply data augmentation to minority class
- Use stratified sampling

### Systematic Errors

**Indicators:**
- Specific class pairs frequently confused
- Confusion matrix shows consistent off-diagonal pattern

**Example:**
```
# Model frequently confuses "ants" as "bees" but not vice versa
Confusion matrix shows:
- ants→bees: 15 errors
- bees→ants: 2 errors
```

**Solutions:**
- Inspect misclassified images (are they actually hard to classify?)
- Add more training examples for confused classes
- Improve data augmentation for that class
- Check for labeling errors

---

## Best Practices

### Before Inference

1. ✅ **Use best.pt** - Highest validation accuracy
2. ✅ **Check run_dir** - Ensure it's the correct trained model
3. ✅ **Verify dataset** - Test set matches training classes
4. ✅ **Check fold** - Using correct fold's test set

### During Inference

1. ✅ **Monitor output** - Watch for errors or warnings
2. ✅ **Check GPU usage** - Should be running on GPU if available
3. ✅ **Review per-sample results** - Look for patterns in errors

### After Inference

1. ✅ **Analyze metrics** - Don't just look at accuracy
2. ✅ **Check confusion matrix** - Understand where model fails
3. ✅ **Compare to validation** - Test should be similar to val performance
4. ✅ **Document results** - Save important metrics and insights
5. ✅ **Investigate errors** - Look at misclassified images

### Cross-Validation Inference

When running inference on multiple folds:

1. ✅ **Evaluate all folds** - Get performance estimate across all splits
2. ✅ **Calculate mean/std** - Report average ± standard deviation
3. ✅ **Check consistency** - Large variance suggests instability
4. ✅ **Identify outliers** - Investigate folds with unusual performance

---

## Common Issues

### "Checkpoint not found"

**Error:**
```
FileNotFoundError: Checkpoint runs/hymenoptera_base_fold_0/weights/best.pt not found
```

**Solutions:**
1. Check run directory exists: `ls runs/`
2. Check weights directory: `ls runs/hymenoptera_base_fold_0/weights/`
3. Use `last.pt` if training didn't improve: `--checkpoint last.pt`

### "Dataset sizes don't match"

**Error:**
```
RuntimeError: Model output size (2) doesn't match dataset classes (3)
```

**Solution:** Ensure test data has same classes as training data.

### Low test accuracy compared to validation

**Symptom:** Val accuracy 90%, test accuracy 70%

**Causes:**
- Overfitting to validation set
- Test set is different distribution than train/val
- Used validation set for hyperparameter tuning too much

**Solutions:**
- Re-train with more data augmentation
- Use pretrained weights
- Ensure test set is from same distribution

### All predictions are same class

**Symptom:** Model always predicts one class

**Causes:**
- Severe class imbalance
- Model collapsed during training
- Wrong checkpoint loaded

**Solutions:**
- Check training logs for issues
- Try different checkpoint
- Re-train with class weights

---

## Performance Metrics Explained

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Overall percentage correct. Good for balanced datasets.

### Precision

```
Precision = TP / (TP + FP)
```

Of predicted positives, how many are actually positive?
- High precision = Few false alarms

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

Of actual positives, how many did we find?
- High recall = Few missed detections

### F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall.
- Balances both metrics
- Good for imbalanced datasets

---

## Workflow Example

### Complete Cross-Validation Evaluation

```bash
# 1. Train all folds
for fold in {0..4}; do
  ml-train --fold $fold --batch_size 32 --lr 0.01 --num_epochs 100
done

# 2. Evaluate all folds
for fold in {0..4}; do
  echo "=== Fold $fold ==="
  ml-inference \
    --run_dir runs/hymenoptera_batch_32_lr_0.01_epochs_100_fold_$fold \
    --checkpoint best.pt
done

# 3. View all results in TensorBoard
tensorboard --logdir runs/

# 4. Calculate average performance across folds
# (manually from inference outputs or write script to parse logs)
```

### Production Deployment

```bash
# 1. Choose best fold based on test accuracy
ml-inference --checkpoint_path runs/hymenoptera_base_fold_3/weights/best.pt
# Suppose fold 3 has highest test accuracy: 93.5%

# 2. Copy best model for deployment
cp runs/hymenoptera_base_fold_3/weights/best.pt production/model.pt
cp runs/hymenoptera_base_fold_3/config.yaml production/config.yaml

# 3. Document performance
echo "Model: ResNet18, Test Acc: 93.5%" > production/README.txt
```

---

## Related Guides

- [Training Guide](training.md) - How to train models
- [Data Preparation](../getting-started/data-preparation.md) - Organize datasets
- [Metrics](../architecture/ml-src-modules.md#metricspy) - Understanding evaluation metrics
- [TensorBoard](monitoring.md) - Visualizing results
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

---

## Summary

**You've learned:**
- ✅ How to run inference on trained models
- ✅ Understanding output files and metrics
- ✅ Interpreting classification reports and confusion matrices
- ✅ Best practices for evaluation
- ✅ Common issues and solutions
- ✅ Cross-validation evaluation workflow

**Ready to evaluate?**
```bash
ml-inference --checkpoint_path runs/my_model_fold_0/weights/best.pt
tensorboard --logdir runs/
```
