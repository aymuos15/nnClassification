# Monitoring Guide

Complement **[Workflow Step 7](../workflow.md#step-7-monitor--review-progress)** with these quick references for visualising runs and inspecting logs.

---

## Essential Commands

```bash
# Launch TensorBoard for a single run
ml-visualise --mode launch --run_dir runs/my_dataset_fold_0

# Inspect raw logs
tail -f runs/my_dataset_fold_0/logs/train.log

# Clean TensorBoard artefacts only
ml-visualise --mode clean --run_dir runs/my_dataset_fold_0
```

TensorBoard lives at http://localhost:6006 by default; use `--port` on `ml-visualise` if the port is taken.

---

## What to Expect in TensorBoard

| Tab | Contents |
| --- | --- |
| **Scalars** | Train/val loss, accuracy, learning rate |
| **Images** | Confusion matrices, optional sample grids |
| **Text** | Classification reports (train/val/test) |

Enable smoothing for noisy scalar curves and compare multiple run directories by pointing `--logdir` to the parent `runs/` folder.

---

## Visualise Samples & Predictions

```bash
# Dataset samples
ml-visualise --mode samples --run_dir runs/my_dataset_fold_0 --split train --num_images 20

# Model predictions (correct vs incorrect borders)
ml-visualise --mode predictions --run_dir runs/my_dataset_fold_0 --split val --checkpoint best.pt --num_images 32
```

Use these after an experiment finishes to audit data quality or error patterns.

---

## Run Artefacts to Inspect

| File | Purpose |
| --- | --- |
| `logs/train.log` | Chronological record of training (same output as console) |
| `logs/classification_report_*.txt` | Precision/recall/F1 for each split |
| `summary.txt` | Dataset sizes, best metrics, configuration snapshot |

`tail -f` is helpful during long runs; use `grep`/`rg` on `train.log` to locate specific epochs or warnings.

---

## Troubleshooting

| Symptom | Suggestion |
| --- | --- |
| TensorBoard blank | Verify correct `--run_dir`, refresh browser, check training still running |
| Metrics missing | Ensure run completed; some logs write only at epoch end |
| Log spam | Reduce log level via config or truncate with `ml-visualise --mode clean` before reruns |

For deeper analysis (Optuna plots, prediction galleries), see the corresponding sections in the workflow or specialised guides.
