# Extending the Framework

This guide covers general patterns for framework-wide customization and extension.

## Overview

The framework is designed with extensibility in mind. See the specific guides for detailed instructions:

- **[Adding Models](adding-models.md)** - Add custom architectures
- **[Adding Transforms](adding-transforms.md)** - Add data augmentations
- **[Adding Optimizers](adding-optimizers.md)** - Add custom optimizers and schedulers
- **[Adding Metrics](adding-metrics.md)** - Add evaluation metrics

## Common Extension Patterns

### 1. Registry Pattern

Many components use a registry pattern for extensibility:

**Models** - Register in `get_custom_model()` function:
```python
# ml_src/core/network/custom.py
MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "your_model": YourModel,
}
```

**Trainers** - Use factory function:
```python
# ml_src/core/trainers/__init__.py
def get_trainer(config, ...):
    trainer_type = config["training"]["trainer_type"]
    # Routes to appropriate trainer class
```

**Inference Strategies** - Use factory function:
```python
# ml_src/core/inference/__init__.py
def get_inference_strategy(config, device=None):
    strategy = config.get("inference", {}).get("strategy", "standard")
    # Routes to appropriate strategy class
```

### 2. Configuration-Driven Design

All components are configured via YAML:
```yaml
model:
  type: 'custom'  # or 'base'
  custom_architecture: 'your_model'

training:
  trainer_type: 'standard'  # or 'mixed_precision', 'accelerate', 'dp'

inference:
  strategy: 'standard'  # or 'tta', 'ensemble', etc.
```

### 3. CLI vs Core Separation

**Rule:** CLI modules orchestrate, core modules implement.

- **CLI** (`ml_src/cli/`) - Command-line entry points, argument parsing, workflow orchestration
- **Core** (`ml_src/core/`) - Reusable ML components (models, trainers, datasets, etc.)

**Never import CLI modules from core modules.**

### 4. Adding Custom Loss Functions

To add a custom loss:

1. Add to `ml_src/core/loss.py`:
```python
def get_loss_function(loss_name, num_classes=None, **kwargs):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "focal_loss":
        return FocalLoss(**kwargs)  # Add here
    # ...
```

2. Use in config:
```yaml
training:
  loss: 'focal_loss'
```

### 5. Adding Custom Data Loading

To customize data loading:

1. Extend `IndexedImageDataset` in `ml_src/core/data/datasets.py`
2. Override `__getitem__` for custom preprocessing
3. Modify `get_datasets()` function to use your dataset class

### 6. Integrating External Tools

**TensorBoard** - Already integrated via `ml_src/core/visual/tensorboard.py`

**Weights & Biases** - Add to trainer's `_log_metrics()` method:
```python
# In ml_src/core/trainers/base.py
def _log_metrics(self, metrics, step, prefix=""):
    # Existing TensorBoard logging
    self.writer.add_scalar(...)

    # Add W&B logging
    import wandb
    wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()})
```

## Best Practices

1. **Follow existing patterns** - Match the style of similar components
2. **Add tests** - Create tests in `tests/` directory
3. **Document** - Add docstrings and update relevant guides
4. **Keep core independent** - Don't add CLI dependencies to core modules
5. **Use type hints** - Helps with IDE support and documentation

## Need Help?

- Check existing implementations in `ml_src/core/`
- See [Development README](README.md) for architecture overview
- Review [Architecture Documentation](../architecture/README.md) for design decisions
