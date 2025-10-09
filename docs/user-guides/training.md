# Training Guide

Use this guide alongside **[Workflow Step 6](../workflow.md#step-6-train-across-folds)**. The workflow covers the full loop (prep → train → evaluate); this page focuses on day-to-day training tips, key switches, and troubleshooting once you are inside the training phase.

---

## Baseline Checklist

- Config generated and edited (dataset name, data directory, `num_classes`).
- Cross-validation splits exist in `data/<dataset>/splits/`.
- Command to start a fold:
  ```bash
  ml-train --config configs/my_dataset_config.yaml --fold 0
  ```
- For quick smoke tests, shorten the run:
  ```bash
  ml-train --config configs/my_dataset_config.yaml --fold 0 --num_epochs 3 --batch_size 8
  ```

More variations (resume, CLI overrides, cross-validation loops) live in the workflow. Below are the knobs you are likely to tweak repeatedly.

---

## Frequently Used Overrides

| Flag | Effect | When to use |
| --- | --- | --- |
| `--fold` | Select CV fold | Iterate across folds after tuning |
| `--num_epochs` | Adjust training length | Quick smoke tests vs long runs |
| `--batch_size` | Change batch size | Balance speed vs VRAM limits |
| `--lr` | Override learning rate | Apply LR finder or Optuna results |
| `--num_workers` | DataLoader workers | Increase if GPU under-utilised |
| `--resume <path>` | Continue from checkpoint | Restarts, extended training |

Combine overrides freely; the CLI always takes precedence over config values.

---

## Run Outputs at a Glance

Every training run creates `runs/<dataset>_*_fold_<n>/` containing:

| Path | Contents | Notes |
| --- | --- | --- |
| `config.yaml` | Exact configuration snapshot | Useful for reproducibility |
| `summary.txt` | Final metrics & run metadata | Quick glance at results |
| `weights/best.pt` | Checkpoint with best validation score | Use for inference/export |
| `weights/last.pt` | Final epoch checkpoint | Use with `--resume` |
| `logs/train.log` | Detailed loguru output | Tail during training or review later |
| `logs/classification_report_{split}.txt` | Per-split precision/recall/f1 | Generated after evaluation |
| `tensorboard/` | Scalar curves, confusion matrices | View with TensorBoard |

If a run finishes but `best.pt` is missing, inspect the log for errors (e.g., validation never improved).

---

## Monitoring Tips

- Launch TensorBoard for any run directory:
  ```bash
  ml-visualise --mode launch --run_dir runs/my_dataset_fold_0
  ```
- Logs can be tailed live: `tail -f runs/<run>/logs/train.log`.
- Watch GPU utilisation with `watch -n 1 nvidia-smi`; low utilisation often means DataLoader workers are saturated.

For more visualisation modes (samples, predictions, search plots) see the Monitoring guide or Workflow Step 7.

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Quick Fix |
| --- | --- | --- |
| `CUDA out of memory` | Batch too large / model heavy | Reduce `--batch_size`, enable mixed precision |
| Training stuck at low accuracy | Underfitting or bad LR | Increase `--num_epochs`, adjust `--lr`, review data quality |
| Val accuracy drops while train rises | Overfitting | Add augmentation, enable early stopping callback, collect more data |
| Resume fails with shape mismatch | Config changed since checkpoint | Revert config to match run or regenerate checkpoint |
| `FileNotFoundError` for split files | Splits missing/moved | Re-run `ml-split` and confirm `data_dir` path |

If an error persists, consult the main [Troubleshooting reference](../reference/troubleshooting.md).

---

## Trainer Selection Snapshot

| Trainer | Best for | Link |
| --- | --- | --- |
| `standard` | CPU/single-GPU simplicity | Default choice |
| `mixed_precision` | Faster single-GPU training, lower memory | [Advanced Training](advanced-training.md#mixed-precision-training) |
| `accelerate` | Multi-GPU or distributed jobs | [Advanced Training](advanced-training.md#accelerate-trainer) |
| `dp` | Differential privacy workflows | [Advanced Training](advanced-training.md#dp-trainer) |

Set the trainer in your config under `training.trainer_type`. Refer to the linked sections for configuration examples and caveats.

---

## Next Steps

- Ready to tune hyperparameters? Jump to **[Hyperparameter Tuning](hyperparameter-tuning.md)** after completing Workflow Step 5.
- Looking to monitor or resume training? See **[Monitoring Guide](monitoring.md)** and the resume section in the workflow.
- After finishing your folds, continue with **[Workflow Step 8](../workflow.md#step-8-evaluate--run-inference)** to evaluate and export.
