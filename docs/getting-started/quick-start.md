# Quick Start Guide

This page gives you a lightning-fast path to running the sample project. For the full end-to-end process, follow **[Workflow Steps 1–6](../workflow.md#step-1-install-dependencies)**.

---

## TL;DR

```bash
# 1. Ensure dependencies are installed (see workflow Step 1)
uv pip install -e .

# 2. Use the bundled ants vs bees dataset
ls data/hymenoptera_data/raw

# 3. Train a short sanity-check run (fold 0, 3 epochs)
ml-train --fold 0 --num_epochs 3

# 4. Inspect outputs
ls runs/hymenoptera_base_fold_0
tensorboard --logdir runs/
```

Within ~5 minutes you should see TensorBoard metrics and checkpoints under `runs/`.

---

## Quick Checks

- Verify PyTorch and CUDA availability:
  ```bash
  python -c "import torch; print(f'PyTorch {torch.__version__}'); print('CUDA:', torch.cuda.is_available())"
  ```
- If the command above fails, revisit installation (workflow Step 1).
- Need a GPU? Confirm `nvidia-smi` lists your device; otherwise you can continue on CPU.

---

## Going Beyond the Demo

- **Bring your own data:** follow **[Data Preparation](data-preparation.md)** to structure `raw/` and generate splits.
- **Generate a config:** `ml-init-config data/<dataset>` (workflow Step 3).
- **Tune or customize:** Optional utilities (LR finder, hyperparameter search, trainer selection) live in workflow Steps 4–7 and the focused user guides.

Once you’re comfortable with the sample run, jump into the unified workflow to carry your dataset from preparation through export.
