# Learning Rate Finder Guide

Use this tool after completing **[Workflow Step 3](../workflow.md#step-3-generate--customize-configuration)** and before full training (**Workflow Step 4**). The workflow shows where LR Finder fits; this page highlights how to interpret results and when to tweak options.

---

## Why Run It?

- Quickly identify a safe learning-rate window for a new dataset/model.
- Avoid repeated manual sweeps across magnitudes (1e-4 → 1e-2 → ...).
- Catch obviously unstable configurations before long training runs.

A single command launches the scan:
```bash
ml-lr-finder --config configs/my_dataset_config.yaml
```

The command writes a plot and JSON under `runs/lr_finder_<timestamp>/` and prints a suggested LR.

---

## Reading the Plot

- **Steep descent region** → best learning-rate range.
- **Flat left side** → LR too small, training will be slow.
- **Sharp spike upwards** → LR too large, loss diverging (stop uses `--diverge_threshold`).
- Pick a value slightly before the curve turns upward; applying ~½ to 1× the suggested LR is usually stable.

Update your config or call `ml-train ... --lr <value>` accordingly.

---

## Useful Flags

| Flag | Default | Use it when |
| --- | --- | --- |
| `--start_lr` | `1e-7` | Your model tolerates larger minimum LRs |
| `--end_lr` | `1` | Smaller models require a lower upper bound |
| `--num_iter` | `100` | Increase for smoother curves, decrease for speed |
| `--beta` | `0.98` | Controls loss smoothing; lower for noisier data |
| `--diverge_threshold` | `4.0` | Set lower to stop earlier on sensitive models |
| `--fold` | current fold | Align finder with the fold you will train |

Example:
```bash
ml-lr-finder --config configs/my_dataset_config.yaml   --start_lr 1e-6 --end_lr 5e-2 --num_iter 150 --diverge_threshold 3.0
```

---

## Tips & Troubleshooting

- **Suggestion looks too high** → use 0.1× the suggested LR or tighten `--diverge_threshold`.
- **Curve noisy** → increase `--num_iter` or reduce `--beta` so smoothing reacts faster.
- **Finder crashes with OOM** → lower `--batch_size` temporarily or run on CPU.
- **Different folds behave differently** → rerun finder per fold when datasets are imbalanced.

If you already have an LR from a previous run, you can skip the finder and proceed directly to training.
