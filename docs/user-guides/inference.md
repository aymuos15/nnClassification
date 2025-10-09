# Inference Guide

Use this guide alongside **[Workflow Step 8](../workflow.md#step-8-evaluate--run-inference)**. The workflow covers the basic commands; here we compare inference strategies and highlight when to apply each.

---

## Quick Commands

```bash
# Evaluate a single checkpoint on the test split
ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt

# Evaluate on validation data instead of test
ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt --split val

# Supply a different config (rare)
ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt --config configs/alt.yaml
```

Outputs land under the run directory (`logs/classification_report_test.txt`, TensorBoard metrics, confusion matrices).

---

## Strategy Overview

| Strategy | When to choose it | CLI shortcut |
| --- | --- | --- |
| `standard` | Baseline evaluation, CPU runs, simplicity | `ml-inference --checkpoint_path ...` |
| `mixed_precision` | Faster GPU inference with minimal memory | Enable via config (`inference.strategy: mixed_precision`) or matching CLI flags if exposed |
| `tta` | Single model, boost robustness with augmentations | `ml-inference --checkpoint_path ... --tta [--tta-augmentations ...]` |
| `ensemble` | Combine multiple folds/models | `ml-inference --ensemble run1/best.pt run2/best.pt ...` |
| `tta_ensemble` | Maximum accuracy regardless of cost | Add `--tta` to the ensemble command |

For in-depth tuning of TTA parameters or ensemble weighting, see the dedicated guides below.

---

## Practical Tips

- Always evaluate the checkpoint saved as `best.pt`; it reflects the highest validation score.
- When comparing multiple runs, write results to a table (accuracy, precision, recall) using the generated classification reports.
- Keep inference configuration aligned with training (transforms, class order). If you need to override, provide the exact config via `--config`.

---

## Related Guides

- [Test Time Augmentation](test-time-augmentation.md) – choose augmentations, aggregation modes, and understand cost/benefit.
- [Ensemble Inference](ensemble-inference.md) – weighting strategies and fold management.
- [Model Export](model-export.md) – convert checkpoints to ONNX after you validate accuracy.
