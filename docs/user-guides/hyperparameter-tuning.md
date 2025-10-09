# Hyperparameter Tuning Guide

Refer to **[Workflow Step 5](../workflow.md#step-5-hyperparameter-search-optional-before-full-cv)** for the end-to-end flow. This page captures tuning patterns, search-space design tips, and troubleshooting once you move beyond the basics.

---

## Manual Exploration (Quick Iterations)

- Override a single parameter per run to isolate impact:
  ```bash
  ml-train --config configs/my_config.yaml --lr 0.005
  ml-train --config configs/my_config.yaml --batch_size 64
  ```
- Record run metadata (command, git commit, seed) alongside results.
- Compare runs with TensorBoard (`tensorboard --logdir runs/`) or by inspecting `summary.txt` in each run directory.

When manual sweeps become tedious, switch to automated search.

---

## Automated Search with Optuna

1. Install extras (once): `uv pip install -e "[optuna]"`.
2. Generate a config that includes a `search` block (`ml-init-config data/<dataset> --optuna --yes`).
3. Launch trials:
   ```bash
   ml-search --config configs/my_dataset_config.yaml --n-trials 50
   ```
4. Review results:
   ```bash
   ml-visualise --mode search --study-name my_dataset_optimization
   ```
5. Train with the exported best configuration:
   ```bash
   ml-train --config runs/optuna_studies/my_dataset_optimization/best_config.yaml
   ```

Resume an existing study with `ml-search --config ... --resume`.

---

## Designing the Search Space

| Field | Example | Notes |
| --- | --- | --- |
| `type: categorical` | Architectures, optimiser choices | Explicit list of options |
| `type: uniform` | Continuous range (linear) | Good for momentum, dropout |
| `type: loguniform` | Exponential range | Ideal for learning rates, weight decay |
| `type: int` | Discrete integers | Epochs, scheduler steps |

Keep the space focused—start with the parameters that historically move the needle (LR, batch size, scheduler) before adding architecture choices.

**Sampler/Pruner defaults:** `TPESampler` + `MedianPruner` cover most cases. Adjust `n_startup_trials` and `n_warmup_steps` if trials are pruned too aggressively.

---

## Best Practices

- **Start small:** run 10–20 trials first; scale up once you see promising regions.
- **Use pruning:** saves compute by terminating weak trials early.
- **Log trial context:** Optuna stores trial params; export `best_config.yaml` to freeze winning settings.
- **Parallelise thoughtfully:** point `search.storage` to a shared SQLite/PostgreSQL DB before running multiple workers.
- **Cross-validation search:** Enable `search.cross_validation` for small datasets when variance across folds is high (slower but robust).

---

## Troubleshooting

| Issue | Resolution |
| --- | --- |
| All trials pruned early | Reduce pruner strictness (`n_warmup_steps`), widen search space, or increase max epochs per trial |
| Best trial worse than baseline | Increase `n_trials`, ensure baseline config lies within the search space, or refine sampler/pruner settings |
| Trials crash with CUDA OOM | Restrict batch-size choices; consider mixed precision during search |
| Search too slow | Lower trial epochs, use pruning, or run workers in parallel |

Still stuck? Consult the [Troubleshooting reference](../reference/troubleshooting.md) or Optuna’s documentation for sampler-specific guidance.
