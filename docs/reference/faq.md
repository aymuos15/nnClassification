# Frequently Asked Questions (FAQ)

Common questions and quick answers.

---

## General

### Q: What models are supported?

**A:** All torchvision models (ResNet, EfficientNet, ViT, etc.) are automatically supported. You can also create custom models.

See: [Model Configuration](../configuration/models.md)

---

### Q: Can I use my own dataset?

**A:** Yes! Just organize it in the required train/val/test structure with class subfolders.

See: [Data Preparation](../getting-started/data-preparation.md)

---

### Q: Do I need a GPU?

**A:** No, but highly recommended. You can train on CPU with `--device cpu`, but it will be much slower.

---

### Q: How do I resume interrupted training?

**A:** Use the `--resume` flag:
```bash
ml-train --resume runs/base/last.pt
```

See: [Resuming Training](../user-guides/resuming-training.md)

---

## Configuration

### Q: How do I change the learning rate?

**A:** Use CLI override or edit config:
```bash
ml-train --lr 0.01
```

Or in `ml_src/config.yaml`:
```yaml
optimizer:
  lr: 0.01
```

---

### Q: What's the difference between best.pt and last.pt?

**A:**
- **best.pt** - Model with highest validation accuracy (use for deployment/inference)
- **last.pt** - Latest epoch checkpoint (use to resume training)

---

### Q: Can I use pretrained weights?

**A:** Yes, set `weights: 'DEFAULT'` in config:
```yaml
model:
  type: 'base'
  architecture: 'resnet18'
  weights: 'DEFAULT'  # ImageNet pretrained
```

---

### Q: How do I change batch size?

**A:**
```bash
ml-train --batch_size 32
```

Start small (8-16) and increase until GPU memory is full.

---

## Data & Cross-Validation

### Q: What data structure is required?

**A:** The framework uses an index-based structure where raw images are stored once and referenced by text files for each split.

**Mandatory structure:**
```
data/your_dataset/
├── raw/
│   ├── class1/
│   └── class2/
└── splits/
    ├── test.txt
    ├── fold_0_train.txt
    └── fold_0_val.txt
```
You must run `ml-split` to generate the `splits/` directory. See the [Data Preparation Guide](../getting-started/data-preparation.md) for details.

---

### Q: How does cross-validation work with test sets?

**A:** Test set is held out ONCE and is the SAME for all folds:
- ✅ **Test set:** 15% held out once (test.txt) - SAME for all folds
- ✅ **Train/Val:** Remaining 85% split across k folds (varies per fold)
- ✅ **Fair comparison:** All folds evaluated on identical test set

**Why not different test sets per fold?**
Different test sets would make fold results incomparable. Our approach ensures consistent evaluation.

---

### Q: Where is my test data after running ml-split?

**A:** Check `data/my_dataset/splits/test.txt` (single file, no fold number).
- ✅ `test.txt` - Same for all folds
- ✅ `fold_0_train.txt`, `fold_0_val.txt` - Vary per fold
- ✅ `fold_1_train.txt`, `fold_1_val.txt` - Vary per fold

---

## Training

### Q: How many epochs should I train for?

**A:**
- Quick test: 3-5 epochs
- Small datasets: 25-50 epochs
- Large datasets: 50-100 epochs
- From scratch: 200+ epochs

Monitor validation curves and stop when they plateau.

---

### Q: Training is very slow. What can I do?

**A:**
1. Increase batch size: `ml-train --batch_size 64`
2. Increase workers: `ml-train --num_workers 8`
3. Use smaller images (edit transforms in config)
4. Use faster model: `efficientnet_b0` or `mobilenet_v3_small`

See: [Performance Tuning](performance-tuning.md)

---

### Q: I'm getting "CUDA out of memory" errors. Help?

**A:**
1. Reduce batch size: `ml-train --batch_size 8`
2. Reduce image size (edit config transforms)
3. Use smaller model
4. Train on CPU: `ml-train --device cpu`

See: [Troubleshooting](troubleshooting.md)

---

### Q: How do I know if my model is training well?

**A:**
- Training loss should decrease
- Validation accuracy should increase
- Gap between train/val not too large (indicates overfitting)
- Check TensorBoard: `tensorboard --logdir runs/`

---

## Models

### Q: Which model should I use?

**A:**
- **Beginner:** `resnet18` with `weights: 'DEFAULT'`
- **Best accuracy:** `efficientnet_b7` or `vit_b_16`
- **Fast training:** `efficientnet_b0` or `mobilenet_v3_small`
- **Mobile/Edge:** `mobilenet_v3_small`

See: [Model Configuration](../configuration/models.md)

---

### Q: Can I create my own model architecture?

**A:** Yes! Add your model to `ml_src/network/custom.py` and register it.

See: [Adding Custom Models](../development/adding-models.md)

---

### Q: How do I switch models?

**A:** Edit `ml_src/config.yaml`:
```yaml
model:
  architecture: 'efficientnet_b0'  # Change this
```

Note: Existing checkpoints won't work (different architecture).

---

## Inference

### Q: How do I run inference on test data?

**A:** Test evaluation is **automatic** after training! Results are saved to:
- `runs/{run_name}/logs/classification_report_test.txt`
- TensorBoard (confusion matrix, metrics)

For manual inference:
```bash
ml-inference runs/base/weights/best.pt
```

See: [Inference Guide](../user-guides/inference.md)

---

### Q: Can I test on a single image?

**A:** Not directly supported. You'd need to modify the inference module or create a new script.

---

## Monitoring

### Q: How do I view training metrics?

**A:** Use TensorBoard:
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

See: [Monitoring Guide](../user-guides/monitoring.md)

---

### Q: What metrics are tracked?

**A:**
- Training/validation loss
- Training/validation accuracy
- Learning rate schedule
- Confusion matrices
- Per-class precision/recall/F1

---

### Q: Where are the logs?

**A:**
- Training log: `runs/{run_name}/logs/train.log`
- Summary: `runs/{run_name}/summary.txt`
- TensorBoard: `runs/{run_name}/tensorboard/`

---

## Hyperparameter Tuning

### Q: How do I tune hyperparameters?

**A:** Run multiple experiments with different values:
```bash
ml-train --lr 0.001 --batch_size 16
ml-train --lr 0.01 --batch_size 16
ml-train --lr 0.01 --batch_size 32
```

Compare in TensorBoard.

See: [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md)

---

### Q: What hyperparameters should I tune first?

**A:** Priority order:
1. Learning rate (`--lr`)
2. Batch size (`--batch_size`)
3. Number of epochs (`--num_epochs`)
4. Scheduler settings (`--step_size`, `--gamma`)

---

## Reproducibility

### Q: How do I make results reproducible?

**A:** Set a fixed seed and use deterministic mode:
```yaml
seed: 42
deterministic: true
```

Note: `deterministic: true` is slower but guarantees exact reproducibility.

See: [Reproducibility Configuration](../configuration/reproducibility.md)

---

### Q: Why do I get slightly different results each run?

**A:** With `deterministic: false` (default), some operations are non-deterministic for speed. Set `deterministic: true` for exact reproduction.

---

## Errors

### Q: "Found 0 files in subfolders" error?

**A:** Incorrect data structure. Images must be inside class subfolders.

See: [Data Preparation](../getting-started/data-preparation.md)

---

### Q: "RuntimeError: CUDA out of memory"?

**A:** Reduce batch size:
```bash
ml-train --batch_size 8
```

See: [Troubleshooting](troubleshooting.md)

---

### Q: Loss becomes NaN during training?

**A:**
1. Lower learning rate: `ml-train --lr 0.0001`
2. Check data normalization
3. Verify labels are correct

---

## Advanced

### Q: Can I use mixed precision training?

**A:** Not currently supported, but can be added by modifying `trainer.py` to use `torch.cuda.amp`.

---

### Q: Can I train on multiple GPUs?

**A:** Not currently supported. Future extension.

---

### Q: How do I add data augmentation?

**A:** Modify `ml_src/dataset.py::get_transforms()` to add more transforms.

See: [Adding Transforms](../development/adding-transforms.md)

---

### Q: Can I use a different optimizer (Adam, AdamW)?

**A:** Yes, modify `ml_src/optimizer.py::get_optimizer()` to add more optimizers.

See: [Adding Optimizers](../development/adding-optimizers.md)

---

### Q: How do I implement early stopping?

**A:** Not currently implemented. Requires modification to `trainer.py`.

---

## Getting Help

### Q: Where can I find more documentation?

**A:**
- [Documentation Index](../README.md)
- [Configuration Reference](../configuration/README.md)
- [Troubleshooting Guide](troubleshooting.md)

---

### Q: Something isn't working. What should I check?

**A:**
1. Verify data structure: `tree -L 2 data/my_dataset/`
2. Check configuration: `cat runs/{run_name}/config.yaml`
3. Review logs: `cat runs/{run_name}/logs/train.log`
4. Check system: `python -c "import torch; print(torch.__version__)"`

See: [Troubleshooting](troubleshooting.md)

---

**Still have questions?** Check the full documentation at [docs/README.md](../README.md)
