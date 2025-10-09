# Test-Time Augmentation (TTA)

Pairs with **[Workflow Step 8](../workflow.md#step-8-evaluate--run-inference)**. Use this page when you need to tune augmentation choices or understand the cost/benefit of TTA beyond the default workflow flags.

---

## Why TTA?

- Mitigates sensitivity to orientation, brightness, or other augmentable factors.
- Often yields +1–3 pp accuracy on balanced datasets for the cost of slower inference.
- Particularly effective when training data is limited or imbalanced.

Enable TTA:
```bash
ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt --tta
```
This applies the default augmentation bundle defined in the config (horizontal flip by default).

---

## Custom Augmentation Sets

Specify augmentations explicitly:
```bash
ml-inference --checkpoint_path runs/.../best.pt --tta   --tta-augmentations horizontal_flip vertical_flip rotate_90 brightness contrast
```

Available options match the entries in `ml_src/core/transforms/tta.py`, e.g. `horizontal_flip`, `vertical_flip`, `rotate_90`, `rotate_180`, `brightness`, `contrast`.

### Choosing Augmentations

| Scenario | Suggested augmentations |
| --- | --- |
| Natural images | `horizontal_flip`, `rotate_90` |
| Remote sensing / aerial | `horizontal_flip`, `vertical_flip`, rotations |
| Lighting variability | `brightness`, `contrast` |
| Strict orientation (e.g., digits) | Use only illumination changes |

More augmentations = more forward passes. Five transforms roughly quintuple inference time.

---

## Aggregation Modes

Set via config (`inference.tta.aggregation`) or CLI flag `--tta-aggregation`.

| Mode | Description | Notes |
| --- | --- | --- |
| `mean` | Average logits (soft voting) | Stable default |
| `max` | Take max logit across augmentations | Emphasises confident predictions |
| `voting` | Majority vote on class labels | Works best with many augmentations |

---

## Practical Tips

- Run TTA on the validation split first to verify the gain before using it on the test set.
- Combine with ensembles only when the accuracy boost justifies the multiplicative cost (`tta_ensemble`).
- Store the TTA command you used alongside the results to ensure reproducibility.

Need to stack multiple checkpoints as well? See **[Ensemble Inference](ensemble-inference.md)**.
