# Resuming Training

Follow the instructions in **[Workflow Step 6](../workflow.md#step-6-train-across-folds)** to resume standard runs. This page simply highlights the commands and edge cases you may encounter when continuing training.

---

## Core Commands

```bash
# Resume exactly where the run stopped
ml-train --resume runs/my_dataset_fold_0/weights/last.pt

# Resume but extend the schedule to 100 epochs total
ml-train --resume runs/my_dataset_fold_0/weights/last.pt --num_epochs 100

# Resume from the best checkpoint for fine-tuning
ml-train --resume runs/my_dataset_fold_0/weights/best.pt --lr 5e-4
```

Everything needed to continue (model weights, optimiser, scheduler, epoch counter, RNG state) is stored in the checkpoint file.

---

## Good Practices

- **Keep configs in sync:** resume with the same config used to create the checkpoint; changing architecture or `num_classes` will fail.
- **Adjust learning rate carefully:** when lowering the LR, specify it explicitly (`--lr`) after `--resume`.
- **Name outputs:** if resuming into a new experiment, use `--run_name` to avoid overwriting the original directory.

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `FileNotFoundError` | Confirm the checkpoint path; `weights/last.pt` is created only after at least one epoch finishes. |
| Tensor shape mismatch | You changed model architecture/`num_classes`; revert config or retrain from scratch. |
| Resumed run diverges | Reduce `--lr`, or resume from `best.pt` instead of `last.pt`. |

For deeper details about what is stored inside checkpoints, see the [Checkpointing module](../architecture/ml-src-modules.md#checkpointingpy).
