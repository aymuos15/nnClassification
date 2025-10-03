# Frequently Asked Questions

## General

**Q: What models are supported?**
A: All torchvision models (ResNet, EfficientNet, ViT, etc.) plus custom models.

**Q: Can I use my own dataset?**
A: Yes! Organize in train/val/test structure. See [Data Preparation](../getting-started/data-preparation.md).

**Q: Do I need a GPU?**
A: No, CPU training is supported. GPU is much faster though.

## Configuration

**Q: How do I change learning rate?**
```bash
python train.py --lr 0.01
```

**Q: Where is the config file?**
A: `ml_src/config.yaml`

**Q: How do I create custom config?**
A: Copy `config.yaml` to `configs/my_config.yaml`, edit, then:
```bash
python train.py --config configs/my_config.yaml
```

## Training

**Q: How do I resume training?**
```bash
python train.py --resume runs/base/last.pt
```

**Q: How long does training take?**
A: Depends on dataset size, model, hardware. Example: 50 epochs with 1000 images on GPU takes ~10 minutes.

**Q: Can I train on CPU?**
```bash
python train.py --device cpu
```

**Q: Out of memory error?**
```bash
python train.py --batch_size 8
```

## Data

**Q: What image formats are supported?**
A: JPG, PNG, BMP, TIFF - anything PIL can read.

**Q: Do I need exactly train/val/test splits?**
A: Yes, all three are required.

**Q: Can I use different class names?**
A: Yes, but they must be identical across train/val/test.

**Q: How much data do I need?**
A: Minimum: ~100 images per class. More is better.

## Models

**Q: How do I use different model?**
```yaml
model:
  architecture: 'efficientnet_b0'
```

**Q: Should I use pretrained weights?**
A: Yes, for small datasets (<10k images). Use `weights: 'DEFAULT'`.

**Q: How do I add custom model?**
A: See [Adding Models](../development/adding-models.md).

## Results

**Q: Where are the results?**
A: `runs/{run_name}/`

**Q: How do I view training curves?**
```bash
tensorboard --logdir runs/
```

**Q: Which checkpoint to use?**
A: `best.pt` for deployment, `last.pt` for resuming.

**Q: How do I compare runs?**
A: Use TensorBoard with multiple runs.

## Troubleshooting

**Q: Training loss not decreasing?**
A: Try lower learning rate (`--lr 0.0001`).

**Q: Loss becomes NaN?**
A: Lower learning rate, check data normalization.

**Q: Training too slow?**
A: Increase `num_workers`, larger `batch_size`, or use GPU.

**Q: Low accuracy?**
A: Train more epochs, use pretrained weights, check data quality.

## Advanced

**Q: Can I use mixed precision?**
A: Not currently implemented. Future extension.

**Q: Multi-GPU training?**
A: Not currently supported. Future extension.

**Q: Custom loss function?**
A: Edit `ml_src/loss.py::get_criterion()`.

**Q: Custom optimizer?**
A: Edit `ml_src/optimizer.py::get_optimizer()`.

## Related

- [Quick Start](../getting-started/quick-start.md)
- [Training Guide](../user-guides/training.md)
- [Troubleshooting](troubleshooting.md)
- [Configuration](../configuration/overview.md)
EOF3
