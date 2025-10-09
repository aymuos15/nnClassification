# Ensemble Inference

Reference **[Workflow Step 8](../workflow.md#step-8-evaluate--run-inference)** for the base command. This guide focuses on how to assemble folds/models and control aggregation once you want more accuracy than a single checkpoint provides.

---

## Basic Usage

```bash
ml-inference --ensemble   runs/my_dataset_fold_0/weights/best.pt   runs/my_dataset_fold_1/weights/best.pt   runs/my_dataset_fold_2/weights/best.pt
```

Predictions from each checkpoint are combined according to the aggregation strategy defined in the config or CLI.

---

## Aggregation Strategies

| Strategy | Description | When to use |
| --- | --- | --- |
| `soft_voting` | Average probabilities/logits | Default; smooth and stable |
| `hard_voting` | Majority vote on predicted labels | Use when models have similar accuracy |
| `weighted` | Weighted average | Give stronger folds higher influence |

To supply weights:
```yaml
inference:
  strategy: 'ensemble'
  ensemble:
    checkpoints:
      - 'runs/my_dataset_fold_0/weights/best.pt'
      - 'runs/my_dataset_fold_1/weights/best.pt'
      - 'runs/my_dataset_fold_2/weights/best.pt'
    aggregation: 'weighted'
    weights: [0.4, 0.35, 0.25]
```

---

## Selecting Members

- Prefer checkpoints trained on different folds or architectures to maximise diversity.
- Ensure all checkpoints were trained with the same class ordering and preprocessing.
- Keep a record of each model’s validation metrics so weights reflect performance.

---

## Performance Considerations

- Runtime scales roughly with the number of checkpoints (5 models ≈ 5× slower).
- To manage compute, start with the top-2 folds and expand only if accuracy gains justify the cost.
- Combine with TTA only for final evaluations (`ml-inference --ensemble ... --tta`).

---

## Troubleshooting

| Issue | Fix |
| --- | --- |
| Shape mismatch | One checkpoint was trained with different `num_classes`; exclude it |
| Accuracy decreases | Remove underperforming models or adjust weights |
| Memory pressure | Run ensemble in mixed precision or evaluate in batches |

After validating ensemble performance, proceed to export the strongest checkpoint(s) via **[Model Export](model-export.md)**.
