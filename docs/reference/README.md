# Reference Documentation

Quick references, troubleshooting guides, and optimization resources for the PyTorch Image Classification framework.

## Overview

This section provides practical reference materials designed for quick lookups during development and training. Whether you're debugging an issue, optimizing performance, or looking for best practices, these guides offer concise, actionable information to help you work efficiently.

---

## Reference Documents

### [Best Practices](best-practices.md)
**Essential tips and conventions for effective framework usage**

Learn recommended approaches for configuration management, training workflows, hyperparameter tuning, data handling, and reproducibility. This guide helps you avoid common pitfalls and establish good habits from the start.

**Key Topics:**
- Configuration management and versioning
- Training workflow recommendations
- Systematic hyperparameter tuning
- Data verification and augmentation strategies
- Code extension patterns
- Reproducibility guidelines

**When to use:** Before starting new experiments, when establishing team conventions, or when unsure about recommended approaches.

---

### [Troubleshooting](troubleshooting.md)
**Common issues and their solutions**

Comprehensive troubleshooting guide covering installation problems, training issues, data errors, configuration mistakes, and inference problems. Each issue includes symptoms, causes, and step-by-step solutions.

**Key Topics:**
- Installation and dependency issues
- CUDA and GPU problems
- Out of memory errors
- Training failures (NaN loss, slow training, poor convergence)
- Data loading errors
- Configuration validation errors
- Checkpoint and resume issues

**When to use:** When encountering errors, unexpected behavior, or performance issues. Check here first before deep debugging.

---

### [Performance Tuning](performance-tuning.md)
**Speed and memory optimization strategies**

Detailed guide for optimizing training speed and reducing memory usage. Learn how to maximize GPU utilization, accelerate data loading, and train larger models within memory constraints.

**Key Topics:**
- Training speed optimization (batch size, data loading, determinism)
- Memory usage reduction techniques
- GPU utilization monitoring
- Model-specific optimizations
- Profiling and bottleneck identification
- Mixed precision training considerations

**When to use:** When training is too slow, hitting memory limits, or optimizing resource utilization for production workflows.

---

### [FAQ](faq.md)
**Frequently asked questions with quick answers**

Quick answers to common questions organized by category. Includes general framework questions, configuration help, training workflows, data handling, model selection, and deployment topics.

**Key Topics:**
- Supported models and architectures
- Dataset requirements and organization
- GPU vs CPU training
- Resuming interrupted training
- Configuration overrides
- Checkpoint selection (best vs last)
- Custom model integration
- Multi-GPU training

**When to use:** For quick answers to common questions without reading full documentation sections.

---

### [Visualization](visualization.md)
**TensorBoard tools and ml-visualise command reference**

Complete reference for the `ml-visualise` CLI command and TensorBoard visualization capabilities. Learn how to visualize datasets, inspect predictions, monitor training metrics, and manage TensorBoard servers.

**Key Topics:**
- `ml-visualise` CLI command modes and options
- Visualizing dataset samples
- Inspecting model predictions
- TensorBoard server management
- Log cleanup and organization
- Training metrics visualization
- Comparative experiment analysis

**When to use:** When setting up visualization, debugging data pipelines, analyzing model predictions, or comparing experiments.

---

## Quick Access Guide

### Most Commonly Needed Resources

**Starting a new project?**
- Read: [Best Practices](best-practices.md) → Configuration and Training sections

**Encountering an error?**
- Check: [Troubleshooting](troubleshooting.md) → Find your error message or symptom

**Training too slow or out of memory?**
- Optimize: [Performance Tuning](performance-tuning.md) → Speed or Memory sections

**Quick question?**
- Search: [FAQ](faq.md) → Organized by topic

**Setting up visualization?**
- Reference: [Visualization](visualization.md) → ml-visualise modes

---

## Common Scenarios

### Scenario: First Time Training

1. Review [Best Practices](best-practices.md) - Configuration & Training
2. Set up [Visualization](visualization.md) - Launch TensorBoard
3. Keep [Troubleshooting](troubleshooting.md) handy for any issues

### Scenario: Optimizing Production Workflow

1. Read [Performance Tuning](performance-tuning.md) - All sections
2. Apply [Best Practices](best-practices.md) - Reproducibility
3. Check [FAQ](faq.md) - Multi-GPU and deployment topics

### Scenario: Debugging Training Issues

1. Check [Troubleshooting](troubleshooting.md) - Training Issues section
2. Verify configuration using [FAQ](faq.md)
3. Inspect data with [Visualization](visualization.md) - Samples mode
4. Review [Best Practices](best-practices.md) - Data section

### Scenario: Team Onboarding

1. Share [Best Practices](best-practices.md) for conventions
2. Bookmark [FAQ](faq.md) for quick answers
3. Reference [Troubleshooting](troubleshooting.md) for common issues
4. Demo [Visualization](visualization.md) tools

---

## Reference Quick Links

| Document | Primary Use | Quick Jump |
|----------|-------------|------------|
| **Best Practices** | Conventions & recommendations | [View →](best-practices.md) |
| **Troubleshooting** | Error resolution | [View →](troubleshooting.md) |
| **Performance Tuning** | Speed & memory optimization | [View →](performance-tuning.md) |
| **FAQ** | Quick answers | [View →](faq.md) |
| **Visualization** | TensorBoard & ml-visualise | [View →](visualization.md) |

---

## Integration with Other Documentation

### Related Documentation Sections

**For comprehensive guides:**
- [User Guides](../user-guides/) - Complete workflows and how-tos
- [Configuration Reference](../configuration/) - All configuration options

**For system understanding:**
- [Architecture](../architecture/) - System design and code structure
- [Development](../development/) - Extending the framework

**For getting started:**
- [Getting Started](../getting-started/) - Installation and quick start

---

## Documentation Tips

### Effective Reference Usage

1. **Bookmark frequently used sections** - Keep quick access to relevant guides
2. **Use browser search** (Ctrl+F / Cmd+F) - Find specific topics within documents
3. **Check FAQ first** - Often the fastest path to answers
4. **Cross-reference** - Troubleshooting often links to Performance Tuning and Best Practices
5. **Stay updated** - Reference docs evolve with common user needs

### When Reference Isn't Enough

If these quick references don't address your needs:

- **Complex workflows:** See [User Guides](../user-guides/)
- **Configuration questions:** See [Configuration Reference](../configuration/)
- **Understanding internals:** See [Architecture Documentation](../architecture/)
- **Custom development:** See [Development Guides](../development/)

---

## Quick Troubleshooting Checklist

Before deep debugging, verify:

- [ ] Data directory structure is correct (see [Data Preparation](../getting-started/data-preparation.md))
- [ ] Configuration file is valid YAML
- [ ] GPU is available and utilized (`nvidia-smi`)
- [ ] Dependencies are installed (`pip install -e .`)
- [ ] Sufficient disk space for checkpoints and logs
- [ ] Correct Python and PyTorch versions

See [Troubleshooting Guide](troubleshooting.md) for detailed solutions.

---

## Performance Quick Wins

Common optimizations with immediate impact:

1. **Increase batch size** - If GPU memory allows (`--batch_size 64`)
2. **More data workers** - Speed up data loading (`--num_workers 8`)
3. **Reduce image size** - If resolution isn't critical (modify transforms)
4. **Disable determinism** - Faster but non-reproducible (usually default)
5. **Monitor GPU usage** - Ensure near 100% utilization

See [Performance Tuning](performance-tuning.md) for comprehensive strategies.

---

## Contributing to Reference Documentation

Found a common issue not covered? Have optimization tips to share?

Reference documentation grows from user experience:

1. Document solutions to problems you encountered
2. Share optimization strategies that worked
3. Suggest FAQ additions for repeated questions
4. Clarify confusing sections

Reference docs should be **concise**, **actionable**, and **regularly used**.

---

## Navigation

**← Back to [Main Documentation](../README.md)**

Explore other documentation sections:
- [Getting Started](../getting-started/) - New user guides
- [Configuration](../configuration/) - Complete config reference
- [User Guides](../user-guides/) - Practical workflows
- [Architecture](../architecture/) - System design
- [Development](../development/) - Extending the framework

---

**Need help fast?** Start with [FAQ](faq.md) or [Troubleshooting](troubleshooting.md) →
