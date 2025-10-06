# Hyperparameter Tuning Guide

Optimize model performance through systematic hyperparameter search.

---

## Overview

This framework provides two approaches to hyperparameter tuning:

1. **Manual tuning** - Run experiments with different parameter combinations
2. **Automated search** - Use `ml-search` with Optuna for intelligent optimization

---

## Manual Hyperparameter Tuning

### CLI Override Method

Run multiple training experiments with different hyperparameters using CLI overrides:

```bash
# Experiment 1: Baseline
ml-train --config configs/my_config.yaml

# Experiment 2: Higher learning rate
ml-train --config configs/my_config.yaml --lr 0.01

# Experiment 3: Larger batch size
ml-train --config configs/my_config.yaml --batch_size 64

# Experiment 4: Combination
ml-train --config configs/my_config.yaml --lr 0.01 --batch_size 64
```

### Compare Results in TensorBoard

```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

TensorBoard will show all runs side-by-side for comparison.

### Recommended Tuning Order

1. **Learning rate** - Most impactful parameter
2. **Batch size** - Affects training speed and stability
3. **Number of epochs** - Ensure convergence
4. **Scheduler settings** - Fine-tune learning rate decay
5. **Model architecture** - Try different models
6. **Optimizer settings** - Momentum, weight decay

---

## Automated Hyperparameter Search

### Installation

```bash
uv pip install -e ".[optuna]"
```

This installs Optuna, Plotly, and Kaleido for hyperparameter optimization and visualization.

### Quick Start

```bash
# 1. Generate config with search space
ml-init-config data/my_dataset --optuna --yes

# 2. Run optimization
ml-search --config configs/my_dataset_config.yaml --n-trials 50

# 3. Visualize results
ml-visualise --mode search --study-name my_dataset_optimization

# 4. Train with best hyperparameters
ml-train --config runs/optuna_studies/my_dataset_optimization/best_config.yaml
```

### Search Configuration

The search configuration in your config file defines the optimization strategy:

```yaml
search:
  study_name: 'my_optimization'
  storage: 'sqlite:///optuna_studies.db'
  n_trials: 50
  direction: 'maximize'  # or 'minimize'
  metric: 'val_acc'      # or 'val_loss'

  # Sampler: How to suggest hyperparameters
  sampler:
    type: 'TPESampler'          # Tree-structured Parzen Estimator (recommended)
    n_startup_trials: 10        # Random trials before optimization

  # Pruner: Early stopping for unpromising trials
  pruner:
    type: 'MedianPruner'
    n_startup_trials: 5
    n_warmup_steps: 5

  # Define search space
  search_space:
    optimizer.lr:
      type: 'loguniform'
      low: 1e-5
      high: 1e-1

    training.batch_size:
      type: 'categorical'
      choices: [16, 32, 64]

    optimizer.momentum:
      type: 'uniform'
      low: 0.8
      high: 0.99
```

### Search Space Types

**Categorical** - Discrete choices:
```yaml
model.architecture:
  type: 'categorical'
  choices: ['resnet18', 'resnet34', 'efficientnet_b0']
```

**Uniform** - Continuous range (linear scale):
```yaml
optimizer.momentum:
  type: 'uniform'
  low: 0.8
  high: 0.99
```

**Log-Uniform** - Continuous range (logarithmic scale):
```yaml
optimizer.lr:
  type: 'loguniform'
  low: 1e-5
  high: 1e-1
```

**Integer** - Discrete integer range:
```yaml
scheduler.step_size:
  type: 'int'
  low: 5
  high: 15
```

### Samplers

**TPESampler** (recommended):
- Tree-structured Parzen Estimator
- Balances exploration and exploitation
- Good for most use cases

**RandomSampler**:
- Pure random search
- Good baseline

**CmaEsSampler**:
- Covariance Matrix Adaptation Evolution Strategy
- Good for continuous parameters

**GridSampler**:
- Exhaustive grid search
- Good for small search spaces

### Pruners

**MedianPruner**:
- Stops trials performing worse than median
- Conservative, good default

**PercentilePruner**:
- Stops trials in bottom X percentile
- More aggressive than median

**HyperbandPruner**:
- Resource-efficient early stopping
- Good for large-scale searches

### Resume Studies

```bash
# Resume existing study
ml-search --config configs/my_config.yaml --resume

# Or add more trials
ml-search --config configs/my_config.yaml --resume --n-trials 50
```

### Visualizations

```bash
# Generate all plots
ml-visualise --mode search --study-name my_study

# Specific plot types
ml-visualise --mode search --study-name my_study --plot-type optimization_history
ml-visualise --mode search --study-name my_study --plot-type param_importances
ml-visualise --mode search --study-name my_study --plot-type contour --params lr batch_size
```

Available plot types:
- `optimization_history` - Trial performance over time
- `param_importances` - Most influential parameters
- `slice` - Parameter vs. objective plots
- `contour` - 2D parameter interaction
- `parallel_coordinate` - Multi-dimensional visualization
- `intermediate_values` - Training curves for all trials

---

## Best Practices

### Start Small

```bash
# Quick test with 10 trials
ml-search --config configs/my_config.yaml --n-trials 10
```

### Focus Search Space

Don't search everything at once. Start with most impactful parameters:
- Learning rate
- Batch size
- Optimizer type

### Use Pruning

Enable pruners to save compute on unpromising trials:
```yaml
pruner:
  type: 'MedianPruner'
  n_warmup_steps: 5  # Let trials train for 5 epochs before pruning
```

### Cross-Validation

For robust hyperparameter selection on small datasets:
```yaml
search:
  cross_validation:
    enabled: true
    n_folds: 5
    aggregation: 'mean'
```

### Parallel Search

Use shared storage for distributed optimization:
```yaml
search:
  storage: 'postgresql://user:pass@host/db'
```

Then run multiple search processes in parallel - they'll coordinate via the database.

---

## Common Search Spaces

### Conservative (Fast)

```yaml
search_space:
  optimizer.lr:
    type: 'loguniform'
    low: 1e-4
    high: 1e-2
  training.batch_size:
    type: 'categorical'
    choices: [32, 64]
```

### Comprehensive (Thorough)

```yaml
search_space:
  optimizer.lr:
    type: 'loguniform'
    low: 1e-5
    high: 1e-1
  training.batch_size:
    type: 'categorical'
    choices: [16, 32, 64, 128]
  optimizer.momentum:
    type: 'uniform'
    low: 0.8
    high: 0.99
  optimizer.weight_decay:
    type: 'loguniform'
    low: 1e-6
    high: 1e-2
  scheduler.step_size:
    type: 'int'
    low: 5
    high: 20
  scheduler.gamma:
    type: 'uniform'
    low: 0.1
    high: 0.5
```

### Architecture Search

```yaml
search_space:
  model.architecture:
    type: 'categorical'
    choices: ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0']
  model.dropout:
    type: 'uniform'
    low: 0.1
    high: 0.5
  optimizer.lr:
    type: 'loguniform'
    low: 1e-5
    high: 1e-2
```

---

## Troubleshooting

### All Trials Pruned

**Issue:** Every trial gets pruned early

**Solution:**
- Reduce `n_warmup_steps` or `n_startup_trials`
- Use less aggressive pruner (MedianPruner instead of PercentilePruner)
- Check if search space is too broad

### Poor Results

**Issue:** Best trial worse than manual tuning

**Solution:**
- Increase `n_trials` (try 100+)
- Expand search space
- Check sampler configuration
- Verify search space includes known good values

### Slow Search

**Issue:** Each trial takes too long

**Solution:**
- Enable pruning
- Reduce `num_epochs` for trials
- Use smaller dataset split for search
- Run parallel searches

### Out of Memory

**Issue:** Search crashes with CUDA OOM

**Solution:**
- Limit `training.batch_size` choices to smaller values
- Don't include very large models in architecture search
- Use `mixed_precision` trainer

---

## Related

- [Workflow Guide](../workflow.md) - Complete workflow with search
- [Advanced Training](advanced-training.md) - Trainer types and optimization
- [Configuration Reference](../configuration/README.md) - All config options
- [FAQ](../reference/faq.md#hyperparameter-tuning) - Common questions
