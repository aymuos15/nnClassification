# User Guides

Practical workflow guides for training, evaluating, and optimizing your image classification models. These guides focus on real-world scenarios and best practices for daily usage of the framework.

## Overview

The user guides provide step-by-step instructions for common tasks and workflows. Unlike the configuration reference (which explains *what* each option does) or the architecture documentation (which explains *how* the system works), these guides focus on *using* the framework effectively to accomplish your goals.

Each guide is designed to be:
- **Task-oriented**: Focused on completing specific workflows
- **Practical**: Real examples with actual commands and expected outputs
- **Progressive**: From basic usage to advanced techniques
- **Self-contained**: Can be read independently or as a series

---

## Available Guides

### 1. [Training Guide](training.md)
**Complete training workflows and best practices**

Learn how to train models from scratch, including:
- Basic training with default settings
- Cross-validation workflows using the fold system
- Multi-fold training strategies
- Trainer selection (standard, mixed precision, multi-GPU, DP)
- Configuration management
- Monitoring progress during training
- Understanding training outputs and checkpoints

**Start here if you're:** Setting up a new training run or learning the training workflow.

---

### 2. [Advanced Training Guide](advanced-training.md)
**Specialized trainers for high-performance and privacy-preserving training**

Master advanced training techniques:
- Mixed Precision training (2-3x speedup with PyTorch AMP)
- Multi-GPU/distributed training with Accelerate
- Differential Privacy training with Opacus
- Combining techniques for maximum performance
- Trainer selection decision tree
- Performance expectations and benchmarks
- Troubleshooting advanced trainers

**Start here if you're:** Need faster training, have multiple GPUs, or require privacy guarantees.

---

### 3. [Inference Guide](inference.md)
**Running evaluation on trained models**

Evaluate model performance on test data:
- Running inference with saved checkpoints
- Understanding evaluation metrics
- Interpreting classification reports
- Comparing different checkpoints (best vs last)
- Analyzing per-class performance
- Troubleshooting inference issues

**Start here if you're:** Ready to evaluate a trained model or need to generate predictions.

---

### 4. [Resuming Training Guide](resuming-training.md)
**Continue interrupted or extend completed training runs**

Master checkpoint resumption:
- Recovering from interrupted training
- Extending training for more epochs
- Fine-tuning with different hyperparameters
- Understanding what state gets preserved
- Best practices for checkpoint management
- Common resumption scenarios

**Start here if you're:** Training was interrupted, or you want to train an existing model longer.

---

### 5. [Monitoring Guide](monitoring.md)
**TensorBoard integration and training visualization**

Track and visualize training progress:
- Launching TensorBoard for your runs
- Using the `ml-visualise` command
- Understanding logged metrics and plots
- Visualizing dataset samples
- Viewing model predictions
- Analyzing confusion matrices
- Real-time monitoring during training

**Start here if you're:** Tracking training progress or analyzing model behavior.

---

### 6. [Hyperparameter Tuning Guide](hyperparameter-tuning.md)
**Automated and manual hyperparameter optimization**

Optimize model performance:
- Automated search with `ml-search` and Optuna
- Manual tuning with CLI overrides
- Learning rate optimization
- Batch size and optimizer tuning
- Architecture search
- Samplers and pruning strategies
- Search space configuration
- Visualization and analysis
- Best practices for optimization

**Start here if you're:** Optimizing model performance, running experiments, or using automated hyperparameter search.

---

## Common Workflows

### New Model Training
1. **[Prepare data](../getting-started/data-preparation.md)** - Organize dataset structure
2. **[Generate splits](training.md#basic-training)** - Create cross-validation folds with `ml-split`
3. **[Configure](../configuration/README.md)** - Set up config.yaml for your dataset
4. **[Train](training.md)** - Start training with `ml-train`
5. **[Monitor](monitoring.md)** - Watch progress with TensorBoard
6. **[Evaluate](inference.md)** - Test final model with `ml-inference`

### Experiment Iteration
1. **[Baseline](training.md)** - Train with default hyperparameters
2. **[Tune](hyperparameter-tuning.md)** - Systematic hyperparameter search
3. **[Compare](monitoring.md)** - Analyze runs in TensorBoard
4. **[Extend](resuming-training.md)** - Train promising models longer
5. **[Validate](inference.md)** - Final evaluation on test set

### Recovery & Extension
1. **[Check checkpoint](resuming-training.md)** - Verify last.pt exists
2. **[Resume](resuming-training.md)** - Continue from interruption point
3. **[Extend epochs](resuming-training.md)** - Train beyond original limit
4. **[Fine-tune](resuming-training.md)** - Adjust learning rate for further training

### Cross-Validation Workflow
1. **[Generate folds](training.md)** - Create K-fold splits
2. **[Train all folds](training.md)** - Train on each fold separately
3. **[Evaluate each](inference.md)** - Test performance per fold
4. **[Aggregate results](hyperparameter-tuning.md)** - Combine fold metrics
5. **[Compare](monitoring.md)** - Visualize fold variance

---

## Quick Command Reference

### Training
```bash
# Basic training (fold 0)
ml-train --fold 0

# Custom hyperparameters
ml-train --fold 0 --num_epochs 50 --batch_size 32 --lr 0.01

# Train specific fold
ml-train --fold 2 --num_epochs 100
```

### Inference
```bash
# Evaluate best checkpoint
ml-inference --run_dir runs/my_dataset_base_fold_0 --checkpoint best.pt

# Evaluate last checkpoint
ml-inference --run_dir runs/my_dataset_base_fold_0 --checkpoint last.pt
```

### Resuming
```bash
# Resume from interruption
ml-train --resume runs/my_dataset_base_fold_0/last.pt

# Extend training
ml-train --resume runs/my_dataset_base_fold_0/last.pt --num_epochs 100
```

### Monitoring
```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Using ml-visualise
ml-visualise --mode launch --run_dir runs/my_dataset_base_fold_0

# View dataset samples
ml-visualise --mode samples --run_dir runs/my_dataset_base_fold_0 --split train

# View predictions
ml-visualise --mode predictions --run_dir runs/my_dataset_base_fold_0 --split val

# Visualize hyperparameter search results
ml-visualise --mode search --study-name my_study
```

### Hyperparameter Search
```bash
# Automated search with Optuna
ml-search --config configs/my_config.yaml --n-trials 50

# Visualize search results
ml-visualise --mode search --study-name my_study

# Train with best hyperparameters
ml-train --config runs/optuna_studies/my_study/best_config.yaml
```

---

## Integration with Other Documentation

### Before Using These Guides

**Essential prerequisites:**
- [Installation](../getting-started/installation.md) - Framework setup
- [Data Preparation](../getting-started/data-preparation.md) - Dataset organization (CRITICAL)
- [Quick Start](../getting-started/quick-start.md) - First training run

### Complementary Resources

**Configuration details:**
- [Configuration Overview](../configuration/README.md) - How the config system works
- [Data Configuration](../configuration/data.md) - Dataset parameters
- [Training Configuration](../configuration/training.md) - Training parameters
- [Model Configuration](../configuration/models.md) - Architecture options
- [CLI Overrides](../configuration/cli-overrides.md) - Command-line usage

**Understanding the system:**
- [Architecture Overview](../architecture/README.md) - System design
- [Entry Points](../architecture/entry-points.md) - train.py & inference.py
- [Data Flow](../architecture/data-flow.md) - How data moves through the system

**Advanced topics:**
- [Best Practices](../reference/best-practices.md) - Tips and conventions
- [Performance Tuning](../reference/performance-tuning.md) - Speed optimization
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

---

## Guide Reading Paths

### For Beginners
**Recommended sequence:**
1. [Training Guide](training.md) - Learn the basics
2. [Monitoring Guide](monitoring.md) - Track your training
3. [Inference Guide](inference.md) - Evaluate results
4. [Resuming Training](resuming-training.md) - Handle interruptions
5. [Hyperparameter Tuning](hyperparameter-tuning.md) - Optimize performance
6. [Advanced Training](advanced-training.md) - Specialized trainers (when ready)

### For Experienced Users
**Quick reference:**
- Jump directly to the guide matching your current task
- Use the [Quick Command Reference](#quick-command-reference) above
- See [Advanced Training Guide](advanced-training.md) for performance optimization
- Refer to specific sections within guides as needed

### For Troubleshooting
**When things go wrong:**
1. Check the relevant guide for your task
2. Review [Troubleshooting](../reference/troubleshooting.md)
3. Verify your [configuration](../configuration/README.md)
4. Check [data preparation](../getting-started/data-preparation.md)

---

## Common Questions

### Which checkpoint should I use?

- **best.pt**: Highest validation accuracy - use for deployment and final evaluation
- **last.pt**: Most recent training state - use for resuming training

See: [Resuming Training Guide](resuming-training.md)

### How do I train multiple folds?

```bash
# Generate splits once
ml-split --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 5

# Train each fold
ml-train --fold 0
ml-train --fold 1
ml-train --fold 2
ml-train --fold 3
ml-train --fold 4
```

See: [Training Guide](training.md#cross-validation-workflow)

### Can I change hyperparameters when resuming?

Yes! You can override most hyperparameters:
```bash
ml-train --resume runs/base/last.pt --lr 0.0001 --num_epochs 100
```

See: [Resuming Training Guide](resuming-training.md#fine-tune-further)

### How do I monitor training in real-time?

```bash
# Terminal 1: Start training
ml-train --fold 0

# Terminal 2: Launch TensorBoard
tensorboard --logdir runs/

# Or watch log file
tail -f runs/my_dataset_base_fold_0/logs/train.log
```

See: [Monitoring Guide](monitoring.md)

### Where are my results saved?

All outputs go to the run directory:
```
runs/my_dataset_base_fold_0/
├── weights/          # best.pt, last.pt
├── logs/            # train.log
├── tensorboard/     # TensorBoard logs
├── metrics/         # Saved metrics
├── predictions/     # Inference results
└── config.yaml      # Exact config used
```

See: [Training Guide](training.md#understanding-outputs)

---

## Tips for Effective Usage

### 1. Always Use Version Control
```bash
# Track config changes
git add ml_src/config.yaml
git commit -m "Update learning rate for experiment 5"
```

### 2. Document Your Experiments
Keep notes on:
- Hyperparameter choices and rationale
- Training observations
- Validation performance
- Ideas for next experiments

### 3. Use Descriptive Run Names
Override the automatic naming:
```bash
ml-train --run_dir runs/experiment_5_high_lr
```

### 4. Monitor During Training
Don't wait until training finishes - check TensorBoard periodically to catch issues early.

### 5. Save Configurations
```bash
# Export config after successful run
cp runs/best_model/config.yaml configs/production_model.yaml
```

### 6. Test on Small Subsets First
Verify your pipeline before long training runs:
```bash
ml-train --num_epochs 2 --batch_size 8  # Quick sanity check
```

---

## Best Practices Summary

1. **Data preparation is critical** - Follow the [Data Preparation Guide](../getting-started/data-preparation.md) exactly
2. **Start with defaults** - Use baseline configuration before tuning
3. **Use cross-validation** - Train on multiple folds for robust results
4. **Monitor actively** - Watch TensorBoard during training
5. **Keep checkpoints** - Don't delete best.pt and last.pt
6. **Document experiments** - Track what you've tried
7. **Validate properly** - Always evaluate on test set
8. **Tune systematically** - Use grid/random search, not random guessing

---

## Getting Help

### Within the Guides
Each guide includes:
- Prerequisites and assumptions
- Step-by-step instructions
- Expected outputs
- Common issues and solutions
- Links to related documentation

### Additional Resources
- **[FAQ](../reference/faq.md)** - Frequently asked questions
- **[Troubleshooting](../reference/troubleshooting.md)** - Problem diagnosis
- **[Configuration Examples](../configuration/examples.md)** - Complete config files
- **[Best Practices](../reference/best-practices.md)** - Expert tips

---

## Navigation

**Back to:** [Main Documentation](../README.md)

**Related sections:**
- [Getting Started](../getting-started/) - Setup and first steps
- [Configuration](../configuration/) - Complete config reference
- [Architecture](../architecture/) - System design
- [Development](../development/) - Extending the framework
- [Reference](../reference/) - Quick lookups and troubleshooting

---

**Ready to start?** Begin with the [Training Guide](training.md) to learn the core workflow.
