# Ensemble Inference Guide

Comprehensive guide to combining multiple models for improved accuracy through ensembling.

## What is Ensemble Inference?

Ensemble inference combines predictions from multiple trained models to achieve better accuracy and robustness than any single model.

**How it works:**
1. Train multiple models (e.g., different CV folds, architectures, or training runs)
2. Load all models during inference
3. Get predictions from each model
4. Combine predictions (soft voting, hard voting, weighted)
5. Use combined prediction as final output

**Benefits:**
- ✅ Significant accuracy improvement (+2-5% typically)
- ✅ More robust predictions
- ✅ Reduces variance across folds
- ✅ No model retraining required

**Tradeoffs:**
- ❌ Slower inference (~Nx for N models)
- ❌ Higher memory usage (N models loaded)
- ❌ Requires training multiple models first

---

## Quick Start

### CLI Usage (Recommended)

```bash
# Basic ensemble (2 models)
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt

# Ensemble all 5 CV folds
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt \
  runs/fold_3/weights/best.pt \
  runs/fold_4/weights/best.pt

# Ensemble with different aggregation
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  --ensemble-aggregation hard_voting

# Weighted ensemble (give more weight to better models)
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt \
  --ensemble-aggregation weighted \
  --ensemble-weights 0.4 0.35 0.25
```

### Config File Usage

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
    aggregation: 'soft_voting'  # Recommended
```

Then run:
```bash
ml-inference --ensemble  # Uses checkpoints from config
```

---

## Aggregation Methods

### Soft Voting (Recommended)

Averages the logits (pre-softmax outputs) from all models.

**How it works:**
```python
final_logits = mean([model1_logits, model2_logits, model3_logits, ...])
final_prediction = argmax(final_logits)
```

**Best for:**
- Maximum accuracy (recommended default)
- Well-calibrated models
- Most classification tasks

**Why it's best:**
- Preserves confidence information
- Best statistical properties
- Typically 1-2% better than hard voting

**Example:**
```yaml
ensemble:
  aggregation: 'soft_voting'
```

```bash
ml-inference --ensemble fold_0/best.pt fold_1/best.pt --ensemble-aggregation soft_voting
```

### Hard Voting

Each model votes for a class, majority wins.

**How it works:**
```python
predictions = [argmax(model1_logits), argmax(model2_logits), ...]
final_prediction = mode(predictions)  # Most common
```

**Best for:**
- Poorly calibrated models
- Models trained with different objectives
- When individual predictions are reliable

**Example:**
```yaml
ensemble:
  aggregation: 'hard_voting'
```

### Weighted Voting

Weighted average of logits, giving more weight to better-performing models.

**How it works:**
```python
final_logits = sum([w1*model1_logits, w2*model2_logits, ...])
final_prediction = argmax(final_logits)
```

**Best for:**
- Models with known performance differences
- When some folds perform significantly better
- Fine-tuning ensemble performance

**Example:**
```yaml
ensemble:
  aggregation: 'weighted'
  weights: [0.4, 0.3, 0.2, 0.1]  # Weights for 4 models (sum to 1.0)
```

**Aggregation Comparison:**
```
Method        | Accuracy | Use Case
--------------|----------|----------------------------------
soft_voting   | Highest  | Default, most tasks (recommended)
weighted      | High     | Known performance differences
hard_voting   | Moderate | Poorly calibrated models
```

---

## When to Use Ensemble

### ✅ Good Use Cases

**1. Cross-Validation Models**
Most common use case - ensemble all CV folds:
```bash
# Train 5 folds
for fold in {0..4}; do
  ml-train --config config.yaml --fold $fold
done

# Ensemble all folds
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt \
  runs/fold_3/weights/best.pt \
  runs/fold_4/weights/best.pt
```

**2. Production Deployment**
- Maximum accuracy required
- Can afford inference time
- Critical applications (medical, finance, safety)
```yaml
ensemble:
  checkpoints: ['fold_0/best.pt', 'fold_1/best.pt', 'fold_2/best.pt']
  aggregation: 'soft_voting'
```

**3. Competitions (Kaggle, etc.)**
- Final leaderboard submission
- Need every fraction of accuracy
- Inference time not limited
```bash
ml-inference --ensemble fold_*/weights/best.pt --ensemble-aggregation soft_voting
```

**4. Different Architectures**
Combine different model architectures:
```yaml
ensemble:
  checkpoints:
    - 'runs/resnet18_fold_0/weights/best.pt'
    - 'runs/resnet50_fold_0/weights/best.pt'
    - 'runs/efficientnet_fold_0/weights/best.pt'
```

**5. Different Training Runs**
Combine models from different random seeds:
```yaml
ensemble:
  checkpoints:
    - 'runs/seed_42/weights/best.pt'
    - 'runs/seed_123/weights/best.pt'
    - 'runs/seed_456/weights/best.pt'
```

### ❌ Avoid Ensemble For

**1. Real-time Applications**
- Video processing
- Live inference
- Latency critical
- **Use single best model instead**

**2. Limited Resources**
- Low memory environments
- Single model already at memory limit
- **Use model compression instead**

**3. Models Too Similar**
- Same fold, same architecture, same hyperparameters
- **Minimal benefit, not worth cost**

**4. Only 2 Models**
- Ensemble benefit marginal with 2 models
- Consider if worth 2x slowdown
- **Use TTA instead for single model**

---

## Best Practices

### 1. Use All CV Folds

Train and ensemble all folds for best results:
```bash
# Train all 5 folds
for fold in {0..4}; do
  ml-train --config config.yaml --fold $fold --batch_size 32 --lr 0.01
done

# Ensemble all
ml-inference --ensemble runs/*/fold_*/weights/best.pt
```

**Why:** Each fold trains on different data splits, reducing variance.

### 2. Use Soft Voting (Default)

Always start with soft voting:
```bash
ml-inference --ensemble fold_0/best.pt fold_1/best.pt
# Defaults to soft_voting
```

Only switch to hard voting if soft voting doesn't work well.

### 3. Weight by Validation Performance

If folds have different performance, use weighted voting:

**Step 1:** Check validation accuracy for each fold:
```
Fold 0: 92.5%
Fold 1: 91.0%
Fold 2: 90.5%
Fold 3: 91.5%
Fold 4: 89.5%
```

**Step 2:** Assign weights proportional to performance:
```bash
ml-inference --ensemble \
  fold_0/best.pt \
  fold_1/best.pt \
  fold_2/best.pt \
  fold_3/best.pt \
  fold_4/best.pt \
  --ensemble-aggregation weighted \
  --ensemble-weights 0.25 0.20 0.18 0.20 0.17
```

### 4. Verify Model Compatibility

Ensure all models:
- ✅ Have same number of classes
- ✅ Use same class ordering
- ✅ Trained on compatible data

**Check before ensembling:**
```bash
# Verify each model loads correctly
ml-inference --checkpoint_path fold_0/best.pt
ml-inference --checkpoint_path fold_1/best.pt
ml-inference --checkpoint_path fold_2/best.pt
```

### 5. Combine with TTA

For maximum accuracy, use ensemble + TTA:
```bash
ml-inference --ensemble \
  fold_0/best.pt \
  fold_1/best.pt \
  fold_2/best.pt \
  --tta
```

Expected total gain: +3-8% accuracy.

---

## Performance Considerations

### Speed Impact

Ensemble is **~N times slower** where N = number of models.

**Examples:**
```
Models | Speed  | Example Time | Use Case
-------|--------|--------------|------------------
2      | 0.5x   | 90 sec       | Quick boost
3      | 0.33x  | 135 sec      | Good balance
5      | 0.2x   | 225 sec      | Full CV ensemble
10     | 0.1x   | 450 sec      | Maximum ensemble
```

### Memory Impact

All models loaded simultaneously:
```
memory = num_models × model_size
```

**Example:**
- ResNet18: ~45 MB per model
- 5 models: ~225 MB
- ResNet50: ~100 MB per model
- 5 models: ~500 MB

**Tips to reduce memory:**
1. Use smaller architectures
2. Reduce number of models
3. Sequential loading (slower but less memory)

### Accuracy Improvement

Typical improvements by ensemble size:

```
Ensemble Size | Typical Gain | Diminishing Returns
--------------|--------------|--------------------
2 models      | +1-2%        | Good start
3 models      | +2-3%        | Solid improvement
5 models      | +2-4%        | Standard CV ensemble
7+ models     | +2-5%        | Diminishing returns
```

**Diminishing returns:** Going from 5 to 10 models typically adds <0.5% accuracy.

---

## Advanced Techniques

### 1. Diverse Model Ensemble

Combine different architectures for better diversity:

```yaml
ensemble:
  checkpoints:
    - 'runs/resnet18/weights/best.pt'
    - 'runs/resnet50/weights/best.pt'
    - 'runs/efficientnet_b0/weights/best.pt'
  aggregation: 'soft_voting'
```

**Why:** Different architectures make different mistakes → better ensemble.

### 2. Snapshot Ensemble

Ensemble checkpoints from different epochs of same training run:

```yaml
ensemble:
  checkpoints:
    - 'runs/my_run/weights/epoch_30.pt'
    - 'runs/my_run/weights/epoch_40.pt'
    - 'runs/my_run/weights/epoch_50.pt'
    - 'runs/my_run/weights/best.pt'
  aggregation: 'soft_voting'
```

**Why:** Different training stages capture different patterns.

### 3. Weighted by Fold Performance

Calculate weights from validation accuracy:

```python
# Fold validation accuracies
val_accs = [0.925, 0.910, 0.905, 0.915, 0.895]

# Normalize to sum to 1.0
weights = [acc / sum(val_accs) for acc in val_accs]
# Result: [0.203, 0.200, 0.199, 0.201, 0.197]
```

Use in config:
```yaml
ensemble:
  aggregation: 'weighted'
  weights: [0.203, 0.200, 0.199, 0.201, 0.197]
```

### 4. Ensemble Pruning

Remove underperforming models to speed up inference:

**Step 1:** Test ensemble with all models:
```bash
ml-inference --ensemble fold_0/best.pt fold_1/best.pt fold_2/best.pt fold_3/best.pt fold_4/best.pt
# Accuracy: 93.5%
```

**Step 2:** Test without worst performer (fold 4):
```bash
ml-inference --ensemble fold_0/best.pt fold_1/best.pt fold_2/best.pt fold_3/best.pt
# Accuracy: 93.3%  (only -0.2% drop, 20% faster!)
```

**Decision:** Use 4-model ensemble (acceptable accuracy loss for speed gain).

---

## Examples by Scenario

### Standard 5-Fold CV Ensemble

```bash
# Train all folds
for fold in {0..4}; do
  ml-train --config config.yaml --fold $fold
done

# Ensemble all
ml-inference --ensemble \
  runs/hymenoptera_fold_0/weights/best.pt \
  runs/hymenoptera_fold_1/weights/best.pt \
  runs/hymenoptera_fold_2/weights/best.pt \
  runs/hymenoptera_fold_3/weights/best.pt \
  runs/hymenoptera_fold_4/weights/best.pt
```

### Multi-Architecture Ensemble

```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/resnet18_fold_0/weights/best.pt'
      - 'runs/resnet34_fold_0/weights/best.pt'
      - 'runs/efficientnet_b0_fold_0/weights/best.pt'
    aggregation: 'soft_voting'
```

### Production (Weighted by Performance)

```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/fold_0/weights/best.pt'  # 93.5% val acc
      - 'runs/fold_1/weights/best.pt'  # 92.0% val acc
      - 'runs/fold_2/weights/best.pt'  # 91.5% val acc
    aggregation: 'weighted'
    weights: [0.40, 0.35, 0.25]  # Proportional to performance
```

### Maximum Accuracy (Ensemble + TTA)

```bash
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt \
  --tta \
  --tta-augmentations horizontal_flip vertical_flip
```

---

## Troubleshooting

### Models Have Different Number of Classes

**Error:**
```
RuntimeError: Model output size mismatch: model1 has 10 classes, model2 has 8
```

**Cause:** Models trained on different datasets or configs.

**Solution:** Ensure all models trained with same `num_classes` in config.

### Checkpoint Not Found

**Error:**
```
FileNotFoundError: Checkpoint not found: runs/fold_3/weights/best.pt
```

**Solution:**
1. Check all checkpoint paths exist: `ls runs/*/weights/best.pt`
2. Verify fold numbers match: `ls runs/`
3. Use absolute paths if necessary

### Ensemble Doesn't Improve Accuracy

**Possible causes:**
1. Models too similar (same fold, same everything)
2. Only 2 models (need 3+ for significant gain)
3. Using hard voting instead of soft voting

**Solutions:**
- Train more diverse models (different folds, architectures)
- Use soft voting (recommended)
- Add more models to ensemble (5 is typical)

### Out of Memory

**Solutions:**
1. Reduce number of models
2. Use smaller architecture (ResNet18 vs ResNet50)
3. Enable mixed precision (if available)
4. Increase GPU memory or use CPU

### Too Slow for Production

**Solutions:**
1. Reduce number of models (ensemble pruning)
2. Use faster models (MobileNet, EfficientNet-B0)
3. Consider model distillation (train single model to mimic ensemble)
4. Deploy only best single model if accuracy acceptable

---

## Workflow Example

### Complete Cross-Validation Ensemble

```bash
# Step 1: Train all 5 folds
for fold in {0..4}; do
  echo "Training fold $fold..."
  ml-train --config configs/hymenoptera.yaml \
    --fold $fold \
    --batch_size 32 \
    --lr 0.001 \
    --num_epochs 50
done

# Step 2: Record validation accuracies
echo "Fold 0: $(grep 'Best Val Acc' runs/hymenoptera_fold_0/logs/train.log | tail -1)"
echo "Fold 1: $(grep 'Best Val Acc' runs/hymenoptera_fold_1/logs/train.log | tail -1)"
# ... etc

# Step 3: Ensemble all folds
ml-inference --ensemble \
  runs/hymenoptera_batch_32_lr_0.001_fold_0/weights/best.pt \
  runs/hymenoptera_batch_32_lr_0.001_fold_1/weights/best.pt \
  runs/hymenoptera_batch_32_lr_0.001_fold_2/weights/best.pt \
  runs/hymenoptera_batch_32_lr_0.001_fold_3/weights/best.pt \
  runs/hymenoptera_batch_32_lr_0.001_fold_4/weights/best.pt

# Step 4: Document performance
# Single best fold: 91.5%
# Ensemble (5 folds): 94.2%  (+2.7% improvement)
```

---

## Related Guides

- [Inference Guide](inference.md) - General inference documentation
- [TTA Guide](test-time-augmentation.md) - Test-Time Augmentation
- [Training Guide](training.md) - Training models for ensembling
- [Model Export](model-export.md) - Export for production

---

## Summary

**Key Takeaways:**
- ✅ Ensemble improves accuracy by +2-5% typically
- ✅ Best for: CV folds, production, competitions
- ✅ Use `soft_voting` aggregation (recommended)
- ✅ Train all CV folds for best results
- ❌ Avoid for: real-time, limited resources
- ❌ ~5x slower (5 models)

**Quick Start Command:**
```bash
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt
```

**Maximum Accuracy:**
```bash
ml-inference --ensemble \
  runs/fold_0/weights/best.pt \
  runs/fold_1/weights/best.pt \
  runs/fold_2/weights/best.pt \
  --tta
```
