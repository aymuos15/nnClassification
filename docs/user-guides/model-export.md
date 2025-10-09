# Model Export Guide

After validating accuracy in **[Workflow Step 8](../workflow.md#step-8-evaluate--run-inference)**, move to **Step 9** to export deployment artefacts. This page captures the key commands, optional flags, and sanity checks for ONNX export.

---

## Core Workflow

```bash
# Export best checkpoint (writes .onnx next to the checkpoint)
ml-export --checkpoint runs/my_dataset_fold_0/weights/best.pt

# Optional: pick a custom output path
ml-export --checkpoint runs/my_dataset_fold_0/weights/best.pt --output deploy/model.onnx
```

Results include the ONNX file plus validation/benchmark reports (if requested) within the checkpoint directory.

---

## Validation & Benchmarking

| Flag | Purpose |
| --- | --- |
| `--validate` | Compare ONNX vs PyTorch outputs on the test loader (thorough, slower) |
| `--validate-basic` | Quick dummy-input check; verifies graph integrity |
| `--benchmark` | Measure latency for PyTorch vs ONNX Runtime |

Combine flags as needed:
```bash
ml-export --checkpoint runs/.../best.pt --validate --benchmark
```

---

## Useful Options

| Option | Default | Notes |
| --- | --- | --- |
| `--opset-version` | Auto-detected | Pin to a specific opset when targeting constrained runtimes (e.g., 12 for TensorRT) |
| `--input-size H W` | From checkpoint metadata | Override when exporting models that expect fixed-size inputs |
| Glob patterns in `--checkpoint` | Single path | Expand to export many checkpoints at once, e.g. `"runs/*/weights/best.pt"` |

---

## Quality Checks

1. Load the ONNX model with ONNX Runtime and run a quick prediction to ensure outputs align with PyTorch.
2. Review validation metrics printed by `ml-export` (max diff, cosine similarity) when using `--validate`.
3. Store the exact command you used so the deployment pipeline can reproduce the artefact.

---

## Troubleshooting

| Issue | Action |
| --- | --- |
| Unsupported operator error | Upgrade opset (`--opset-version 15`) or adjust model to use supported layers |
| Output mismatch warnings | Ensure you exported the same checkpoint/config pair used during evaluation; rerun with `--validate` |
| Large ONNX size | Switch to a lighter architecture or apply post-export optimisations (ONNX Runtime Graph Optimiser, pruning, quantisation) |

Once satisfied with accuracy and runtime, proceed to integrate the ONNX model with your serving stack (ONNX Runtime, TensorRT, OpenVINO, etc.).
