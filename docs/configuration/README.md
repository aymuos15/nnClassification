# Configuration Overview

## Introduction

The PyTorch Image Classifier uses a YAML-based configuration system with command-line override support. This design provides flexibility, version control friendliness, and comprehensive experiment tracking.

## Key Benefits

- **Version Control Friendly**: YAML configs can be committed and tracked
- **Hierarchical Organization**: Related settings grouped logically
- **Override Flexibility**: CLI arguments override YAML defaults
- **Experiment Tracking**: Each run saves its final configuration
- **Type Safety**: YAML supports numbers, strings, booleans, lists, nested dicts

## Configuration Loading Flow

```
1. Load base config from YAML file (default: ml_src/config.yaml)
2. Override with CLI arguments (if provided)
3. Create run directory based on overrides
4. Save final config to run directory (runs/{run_name}/config.yaml)
```

## Configuration File Locations

### Default Base Configuration
**Location:** `ml_src/config.yaml`

This is the default configuration file that contains all standard settings.

### Custom Configuration Files
You can create custom configuration files anywhere:

```bash
ml-train --config path/to/custom_config.yaml
```

### Per-Run Saved Configuration
After training starts, the final configuration (with all overrides applied) is saved to:

```
runs/{run_name}/config.yaml
```

This saved config shows exactly what settings were used for that specific run.

## Configuration Structure

The configuration is organized into 7 main sections:

```yaml
# Reproducibility Settings
seed: <int>
deterministic: <bool>

# Data Configuration
data:
  data_dir: <string>
  num_workers: <int>

# Training Configuration
training:
  batch_size: <int>
  num_epochs: <int>
  device: <string>

# Optimizer Configuration
optimizer:
  lr: <float>
  momentum: <float>

# Scheduler Configuration
scheduler:
  step_size: <int>
  gamma: <float>

# Model Configuration
model:
  type: <string>
  architecture: <string>
  custom_architecture: <string>
  num_classes: <int>
  weights: <string or null>
  input_size: <int>
  dropout: <float>
  model_path: <string>

# Transform Configuration
transforms:
  train:
    ...
  val:
    ...
  test:
    ...
```

## Quick Reference

| Section | Purpose | Key Parameters |
|---------|---------|----------------|
| **Reproducibility** | Random seeding and determinism | `seed`, `deterministic` |
| **Data** | Dataset loading and preprocessing | `data_dir`, `num_workers` |
| **Training** | Core training parameters | `batch_size`, `num_epochs`, `device` |
| **Optimizer** | Optimization algorithm settings | `lr`, `momentum` |
| **Scheduler** | Learning rate scheduling | `step_size`, `gamma` |
| **Model** | Model architecture and configuration | `type`, `architecture`, `num_classes` |
| **Transforms** | Data augmentation and preprocessing | `resize`, `normalize`, `flip` |

## Detailed Documentation

For detailed information about each configuration section, see:

- [Reproducibility Configuration](reproducibility.md) - Seeding and determinism
- [Data Configuration](data.md) - Dataset loading parameters
- [Training Configuration](training.md) - Core training settings
- [Optimizer & Scheduler](optimizer-scheduler.md) - Optimization parameters
- [Model Configuration](models.md) - Model architecture settings
- [Transform Configuration](transforms.md) - Data preprocessing
- [CLI Overrides](cli-overrides.md) - Command-line override system
- [Configuration Examples](examples.md) - Complete real-world examples

## Example: Basic Configuration

```yaml
seed: 42
deterministic: false

data:
  data_dir: 'data/hymenoptera_data'
  num_workers: 4

training:
  batch_size: 4
  num_epochs: 3
  device: 'cuda:0'

optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 7
  gamma: 0.1

model:
  type: 'base'
  architecture: 'resnet18'
  num_classes: 2
  weights: null

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

## Next Steps

- Learn about [CLI overrides](cli-overrides.md) for quick experimentation
- Explore [complete configuration examples](examples.md)
- Understand [reproducibility settings](reproducibility.md) for consistent results
