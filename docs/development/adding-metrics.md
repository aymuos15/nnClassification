# Adding Custom Metrics

Add new evaluation metrics beyond accuracy and confusion matrix.

## Overview

Metrics are in `ml_src/core/metrics/`.

## Adding New Metric

### Step 1: Define Metric Function

Edit `ml_src/core/metrics/`:

```python
def calculate_f1_per_class(true_labels, pred_labels, class_names):
    """Calculate F1 score for each class."""
    from sklearn.metrics import f1_score
    
    f1_scores = f1_score(
        true_labels,
        pred_labels,
        average=None,
        labels=range(len(class_names))
    )
    
    # Create dict
    f1_dict = {
        class_name: f1
        for class_name, f1 in zip(class_names, f1_scores)
    }
    
    return f1_dict


def save_f1_scores(true_labels, pred_labels, class_names, path):
    """Save F1 scores to file."""
    f1_dict = calculate_f1_per_class(true_labels, pred_labels, class_names)
    
    with open(path, 'w') as f:
        f.write("F1 Scores per Class\n")
        f.write("=" * 40 + "\n\n")
        
        for class_name, f1 in f1_dict.items():
            f.write(f"{class_name:15s}: {f1:.4f}\n")
        
        f.write(f"\nMacro Average: {sum(f1_dict.values()) / len(f1_dict):.4f}\n")
```

### Step 2: Call from trainer.py

Edit `ml_src/core/trainers/` at end of `train_model()`:

```python
# After existing metrics
save_confusion_matrix(...)
save_classification_report(...)

# NEW: Add F1 scores
save_f1_scores(
    all_true_labels,
    all_pred_labels,
    class_names,
    run_dir / 'logs' / f'f1_scores_{split}.txt'
)
```

## Example: ROC Curve

```python
def save_roc_curve(true_labels, pred_probs, class_names, path):
    """Save ROC curve for binary or multi-class."""
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    num_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    else:
        # Multi-class: one curve per class
        for i, class_name in enumerate(class_names):
            binary_labels = (true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
```

## Useful Metrics

- Precision, Recall, F1 (per-class)
- ROC curves and AUC
- Precision-Recall curves
- Top-K accuracy
- Calibration curves
- Per-sample confidence scores

## Best Practices

1. Save metrics to files
2. Visualize when possible
3. Include in summary
4. Compare across runs
5. Document interpretation

## Related

- [Metrics Module](../architecture/ml-src-modules.md#metricspy)
- [Trainers Module](../architecture/ml-src-modules.md#trainers)
EOF6