# Unified Workflow Guide

Single end-to-end workflow for preparing data, training across folds, tuning, evaluating, and exporting PyTorch image classifiers built with this framework.

---

## Overview & Prerequisites

- Python 3.8+, PyTorch 2.0+, optional CUDA-enabled GPU
- `uv` installed for dependency management
- Dataset organized as `data/<name>/raw/<class_name>/*.jpg`
- Sufficient disk space under `runs/` for fold artifacts

---

## Step 1: Install Dependencies

```bash
uv pip install -e .
```

**Options:**

- `uv pip install -e ".[dev]"` – development utilities (pytest, ruff, mkdocs)
- `uv pip install -e ".[optuna]"` – Optuna hyperparameter search support
- `uv pip install -e ".[dp]"` – differential privacy training
- Combine extras as needed, e.g. `".[dev,optuna]"`

---

## Step 2: Prepare the Dataset

1. Ensure images are stored under `data/<name>/raw/<class>/image.jpg`.
2. Generate cross-validation splits (required for later steps):

   ```bash
   ml-split --raw_data data/my_dataset/raw --folds 5
   ```

   Outputs index files in `data/my_dataset/splits/` including a shared `test.txt`.

**Optional checks:**

- Run dataset analysis utilities (e.g., `ml_src.core.data.analyze_dataset`) if you want class balance or statistics before training.

---

## Step 3: Generate & Customize Configuration

1. Create a baseline config using detected dataset settings:

   ```bash
   ml-init-config data/my_dataset --yes
   ```

2. For built-in Optuna search space, add `--optuna`.
3. Edit `configs/my_dataset_config.yaml` to confirm:
   - `data.dataset_name`, `data.data_dir`, `data.fold`
   - `model.num_classes`, architecture, pretrained weights
   - `training.trainer_type` (default `standard`)
   - Optional features such as EMA (`training.ema`) or callbacks

---

## Step 4: Optional Pre-Training Utilities

- **Learning rate finder:**

  ```bash
  ml-lr-finder --config configs/my_dataset_config.yaml
  ```

  Adjust ranges with `--start_lr`, `--end_lr`, `--num_iter`, or change divergence sensitivity via `--diverge_threshold`.

- **Dataset reports:** Use analysis helpers to generate statistics or plots before tuning (recommended for imbalanced datasets).

---

## Step 5: Hyperparameter Search (Optional, before full CV)

1. Ensure Optuna extras are installed if running search.
2. Run search on a representative fold (commonly fold 0):

   ```bash
   ml-search --config configs/my_dataset_config.yaml --fold 0 --n-trials 50
   ```

3. Inspect results via `ml-visualise --mode search --study-name <study>`.
4. Reuse the exported `runs/optuna_studies/<study>/best_config.yaml` for subsequent fold training.

*Tip:* For quick experiments, you can skip this step and rely on defaults, but tuned hyperparameters usually transfer better when training every fold.

---

## Step 6: Train Across Folds

Cross-validation is the standard workflow. Train each fold with the finalized configuration (from Step 3 or best config from Step 5).

```bash
for fold in {0..4}; do
  ml-train --config configs/my_dataset_config.yaml --fold $fold
done
```

Each run writes to `runs/<dataset>_*_fold_<n>/` with saved config, logs, checkpoints, and TensorBoard events.

**Options:**

- **Quick iteration:** Run a single fold (e.g., `--fold 0`) to validate the pipeline fast.
- **CLI overrides:** `--batch_size`, `--lr`, `--num_epochs`, `--num_workers`, etc.
- **Resume:** `ml-train --config ... --resume runs/<run_name>/weights/last.pt`
- **Trainer types:** set `training.trainer_type` to `standard`, `mixed_precision`, `accelerate`, or `dp` before training.
- **EMA:** enable in config via `training.ema.enabled: true` for smoothing improvements.

---

## Step 7: Monitor & Review Progress

- Launch TensorBoard per run or root directory:

  ```bash
  ml-visualise --mode launch --run_dir runs/my_dataset_fold_0
  ```

- Inspect logs: `cat runs/<run>/logs/train.log` or tail for live updates.
- GPU monitoring (if applicable): `watch -n 1 nvidia-smi`
- Clean up logs when finished: `ml-visualise --mode clean --run_dir runs/<run>`

---

## Step 8: Evaluate & Run Inference

1. Use the best checkpoint from each fold to assess validation/test metrics:

   ```bash
   ml-inference --checkpoint_path runs/my_dataset_fold_0/weights/best.pt
   ```

2. Evaluate alternate splits or configs with `--split` or `--config` overrides.
3. **Options for improved accuracy:**
   - **TTA:** add `--tta` and optional `--tta-augmentations ...`
   - **Ensemble:** `ml-inference --ensemble runs/my_dataset_fold_0/weights/best.pt runs/my_dataset_fold_1/weights/best.pt ...`
   - **TTA + ensemble:** combine flags for maximum robustness (slowest path)

---

## Step 9: Export & Deployment Preparation

- Export to ONNX from the preferred checkpoint (single fold or ensemble representative):

  ```bash
  ml-export --checkpoint runs/my_dataset_fold_0/weights/best.pt
  ```

- Add validation or benchmarking options as needed:
  - `--validate` (comprehensive)
  - `--validate-basic`
  - `--benchmark`
  - Custom `--output`, `--input_size`, `--opset`

---

## Step 10: Follow-up & Maintenance

- Aggregate metrics across folds (e.g., averaging validation/test scores stored in each run directory).
- Archive or prune large `runs/` entries when finished.
- Iterate on configuration, callbacks, or data augmentations based on insights from monitoring and inference.
- For fast smoke tests, temporarily reduce epochs (`--num_epochs 2`) or batch size before returning to full cross-validation runs.

---

## Reference Outputs per Step

- **Step 2:** `data/<name>/splits/fold_{k}_{train,val}.txt`, `test.txt`
- **Step 3:** `configs/<name>_config.yaml`
- **Step 4:** `runs/lr_finder_<timestamp>/{lr_plot.png,results.json,logs/}`
- **Step 6:** `runs/<dataset>_*_fold_<n>/{config.yaml,summary.txt,weights/,logs/,tensorboard/}`
- **Step 5 (if used):** `runs/optuna_studies/<study>/{best_config.yaml,trial_*/}`
- **Step 9:** `runs/<run>/weights/best.onnx` plus optional validation reports

---

## Troubleshooting Snapshot

- Verify install: `pip list | grep ml-classifier`
- CLI help: `ml-train --help`, `ml-search --help`, `ml-visualise --help`
- Validate config loads: `python -c "from ml_src.core.config import load_config; print(load_config('configs/my_dataset_config.yaml'))"`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
