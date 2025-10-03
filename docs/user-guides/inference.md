# Inference Guide

Guide to running inference and evaluating trained models.

## Basic Inference

```bash
# Evaluate best model
python inference.py --run_dir runs/base --checkpoint best.pt

# Evaluate latest checkpoint
python inference.py --run_dir runs/base --checkpoint last.pt
```

## Output

**Console:**
- Test accuracy
- Per-class metrics
- Confusion matrix location

**Files Generated:**
- `logs/inference.log`
- `logs/classification_report_test.txt`
- `plots/confusion_matrix_test.png`

## Custom Dataset

```bash
# Use different test data
python inference.py --run_dir runs/base --data_dir data/custom_test
```

## Interpreting Results

### Accuracy
Overall correct predictions percentage.

### Confusion Matrix
Shows prediction patterns - where model confuses classes.

### Classification Report
Per-class precision, recall, F1-score.

## Best Practices

1. Use `best.pt` for final evaluation
2. Don't use test set during development
3. Check confusion matrix for insights
4. Compare multiple checkpoints if needed

## Related

- [Training Guide](training.md)
- [Metrics](../architecture/ml-src-modules.md#metricspy)
