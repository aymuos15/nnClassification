# Development Documentation

Complete guide to extending and customizing the PyTorch Image Classification framework.

---

## Overview

This section provides comprehensive guides for developers who need to extend the framework beyond its default capabilities. While the framework supports all torchvision models and a wide range of configurations out-of-the-box, you may need to implement custom components for specialized research or production requirements.

The development guides cover five key extension points:
- **Custom model architectures** for novel network designs
- **Custom transforms** for specialized data augmentation
- **Custom optimizers** for new training algorithms
- **Custom metrics** for domain-specific evaluation
- **General extension patterns** for framework-wide modifications

Each guide follows a consistent structure: overview, step-by-step implementation, configuration, testing, and best practices.

---

## Extension Guides

### [Adding Custom Models](adding-models.md)

**Create and integrate custom neural network architectures.**

While the framework automatically supports all torchvision models (ResNet, EfficientNet, Vision Transformers, etc.), you may need custom architectures for:
- Novel research architectures not in torchvision
- Domain-specific network designs
- Lightweight models for embedded deployment
- Architecture search experiments
- Custom attention mechanisms or novel layers

**Key Topics:**
- Defining model classes with PyTorch `nn.Module`
- Registering models in `MODEL_REGISTRY`
- Configuration setup for custom architectures
- Testing forward passes and output shapes
- Best practices for model implementation

**Example Use Cases:**
- Implementing a custom ResNet variant with additional skip connections
- Creating a lightweight model for mobile deployment
- Building a multi-branch architecture for multi-task learning

[View Full Guide →](adding-models.md)

---

### [Adding Custom Transforms](adding-transforms.md)

**Implement new data augmentation and preprocessing techniques.**

Transforms control how images are augmented and preprocessed before training. Custom transforms are needed for:
- Domain-specific augmentations (medical imaging, satellite imagery, etc.)
- Advanced augmentation strategies (MixUp, CutMix, AutoAugment)
- Custom normalization schemes
- Specialized preprocessing for novel data types
- Research experiments with augmentation policies

**Key Topics:**
- Modifying `get_transforms()` in `ml_src/core/data/datasets.py`
- Adding torchvision transforms with configuration
- Implementing custom transform classes
- Split-specific transforms (train vs. val/test)
- Configuration integration and validation

**Example Use Cases:**
- Adding random rotation and color jitter for training robustness
- Implementing custom transforms for medical image preprocessing
- Creating augmentation pipelines for few-shot learning

[View Full Guide →](adding-transforms.md)

---

### [Adding Custom Optimizers](adding-optimizers.md)

**Integrate new optimization algorithms and learning rate schedules.**

Optimizers control how model weights are updated during training. Custom optimizers are needed for:
- Research with novel optimization algorithms
- Domain-specific optimization strategies
- Adaptive learning rate methods
- Custom momentum and regularization techniques
- Learning rate scheduling experiments

**Key Topics:**
- Extending `get_optimizer()` in `ml_src/core/optimizer.py`
- Supporting multiple optimizer types (SGD, Adam, AdamW, RMSprop)
- Configuration for optimizer-specific parameters
- Implementing custom learning rate schedulers
- Best practices for optimizer selection

**Example Use Cases:**
- Using Adam optimizer instead of SGD for faster convergence
- Implementing cosine annealing learning rate schedule
- Adding AdamW with weight decay for transformer models
- Custom optimizer for specific loss landscapes

[View Full Guide →](adding-optimizers.md)

---

### [Adding Custom Metrics](adding-metrics.md)

**Create new evaluation metrics beyond accuracy and confusion matrices.**

Metrics quantify model performance on validation and test sets. Custom metrics are needed for:
- Domain-specific evaluation criteria
- Class-imbalanced datasets (F1, precision, recall)
- Probabilistic calibration assessment
- Top-K accuracy for large label spaces
- ROC curves and AUC for binary/multi-class problems
- Per-class and per-sample analysis

**Key Topics:**
- Defining metric functions in `ml_src/core/metrics/`
- Integration with training pipeline in `ml_src/core/trainers/`
- Saving metrics to files and visualizations
- Multi-class and binary classification metrics
- Best practices for metric interpretation

**Example Use Cases:**
- Computing F1 scores per class for imbalanced datasets
- Generating ROC curves for model comparison
- Calculating top-5 accuracy for ImageNet-scale problems
- Creating calibration curves for probability assessment

[View Full Guide →](adding-metrics.md)

---

### [Extending Framework](extending-framework.md)

**General patterns for framework-wide customization and extension.**

Beyond specific components, you may need to extend the framework's core functionality. This guide covers:
- Custom loss functions
- Multi-task learning modifications
- Data loading customization
- Checkpoint and logging extensions
- Integration with external tools and libraries
- Advanced debugging and profiling

**Key Topics:**
- Modifying core training loops
- Custom dataset implementations
- Loss function customization
- Advanced logging and monitoring
- Integration patterns and best practices

**Example Use Cases:**
- Implementing custom loss functions (focal loss, triplet loss)
- Multi-GPU training setup
- Custom data loaders for non-standard formats
- Integration with experiment tracking tools (Weights & Biases, MLflow)

[View Full Guide →](extending-framework.md)

---

## When to Extend the Framework

### Use Default Configuration When:
- Training standard image classification tasks
- Using torchvision models (ResNet, EfficientNet, ViT, etc.)
- Standard preprocessing and augmentation is sufficient
- Default metrics (accuracy, confusion matrix) are adequate
- SGD optimization with step learning rate decay works well

### Extend the Framework When:
- **Models**: Implementing novel architectures not in torchvision
- **Transforms**: Requiring domain-specific augmentation strategies
- **Optimizers**: Experimenting with advanced optimization algorithms
- **Metrics**: Needing specialized evaluation criteria
- **Loss Functions**: Using non-standard training objectives
- **Data Loading**: Working with non-standard data formats or structures

---

## Extension Best Practices

### 1. Start Small and Test
- Implement minimal changes first
- Test each component independently
- Verify with simple examples before full training
- Use small datasets for rapid iteration

### 2. Follow Framework Conventions
- Match existing code style and structure
- Use configuration for all customizable parameters
- Maintain compatibility with existing features
- Document changes and rationale

### 3. Validate Thoroughly
- Test with edge cases and boundary conditions
- Compare against known implementations when possible
- Verify shapes, dtypes, and device placement
- Check memory usage and performance

### 4. Document Your Extensions
- Add docstrings to custom functions and classes
- Document configuration parameters and valid ranges
- Provide usage examples
- Note any assumptions or limitations

### 5. Version Control
- Commit extensions in logical, atomic changes
- Use descriptive commit messages
- Tag stable versions of custom components
- Keep track of experimental branches

---

## Development Workflow

### Typical Extension Process:

1. **Identify Need**
   - Determine what component needs customization
   - Check if existing configurations can solve the problem
   - Review relevant guides in this section

2. **Read Relevant Guide**
   - Study the specific extension guide
   - Understand the integration points
   - Review example implementations

3. **Implement Changes**
   - Edit the appropriate module files
   - Follow the step-by-step instructions
   - Add configuration parameters as needed

4. **Update Configuration**
   - Add new parameters to `ml_src/config_template.yaml`
   - Document valid values and defaults
   - Consider CLI override support

5. **Test Implementation**
   - Create simple test cases
   - Verify correct behavior
   - Check for errors and edge cases

6. **Integrate and Train**
   - Run training with custom component
   - Monitor for errors and unexpected behavior
   - Validate results match expectations

7. **Document and Share**
   - Document custom components
   - Share configurations and examples
   - Update team documentation

---

## Common Extension Patterns

### Adding Registry-Based Components

Many framework components use registry patterns for easy extension:

```python
# Define registry
COMPONENT_REGISTRY = {
    'default': DefaultComponent,
    'custom': CustomComponent,
}

# Lookup at runtime
def get_component(config):
    component_type = config['type']
    if component_type not in COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component: {component_type}")
    return COMPONENT_REGISTRY[component_type](**config)
```

**Benefits:**
- Easy addition of new components
- Configuration-driven selection
- No modification of core logic

---

### Configuration-Driven Extensions

Always use configuration for customizable behavior:

```python
# Good: Configuration-driven
def create_component(config):
    if config['component'].get('custom_option'):
        # Apply custom behavior
        pass

# Avoid: Hard-coded behavior
def create_component():
    # Hard-coded values
    pass
```

**Benefits:**
- Reproducible experiments
- Easy experimentation
- Shareable configurations

---

### Modular Design

Keep extensions modular and composable:

```python
# Good: Modular components
def preprocess(x, config):
    return apply_transforms(x, config['transforms'])

def augment(x, config):
    return apply_augmentations(x, config['augmentations'])

# Composition
def prepare_data(x, config):
    x = preprocess(x, config)
    x = augment(x, config)
    return x
```

**Benefits:**
- Testable in isolation
- Reusable across projects
- Clear separation of concerns

---

## Related Documentation

### Architecture Documentation
Understanding the codebase structure helps with extensions:
- [Architecture Overview](../architecture/README.md) - System design
- [ML Source Modules](../architecture/ml-src-modules.md) - Detailed module documentation
- [Data Flow](../architecture/data-flow.md) - How data moves through the system
- [Design Decisions](../architecture/design-decisions.md) - Why things are built this way

### Configuration Documentation
Extensions typically require configuration changes:
- [Configuration Overview](../configuration/README.md) - Config system explained
- [Configuration Examples](../configuration/examples.md) - Sample configurations
- [CLI Overrides](../configuration/cli-overrides.md) - Command-line usage

### User Guides
See extensions in action:
- [Training Guide](../user-guides/training.md) - Training workflows
- [Hyperparameter Tuning](../user-guides/hyperparameter-tuning.md) - Systematic search
- [Monitoring Guide](../user-guides/monitoring.md) - Track experiments

---

## Quick Reference

### File Locations for Common Extensions

| Extension Type | Primary File | Registry/Function |
|---------------|--------------|-------------------|
| Custom Models | `ml_src/core/network/custom.py` | `MODEL_REGISTRY` |
| Transforms | `ml_src/core/data/datasets.py` | `get_transforms()` |
| Optimizers | `ml_src/core/optimizer.py` | `get_optimizer()` |
| Schedulers | `ml_src/core/optimizer.py` | `get_scheduler()` |
| Loss Functions | `ml_src/core/loss.py` | `get_criterion()` |
| Metrics | `ml_src/core/metrics/` | Custom functions |
| Data Loaders | `ml_src/core/data/datasets.py` | `get_data_loaders()` |

### Configuration Sections

| Component | Config Section | Example |
|-----------|---------------|---------|
| Models | `model:` | `architecture: 'resnet50'` |
| Transforms | `transforms:` | `random_horizontal_flip: true` |
| Optimizers | `optimizer:` | `type: 'adam', lr: 0.001` |
| Schedulers | `scheduler:` | `type: 'cosine'` |
| Data | `data:` | `batch_size: 32` |
| Training | `training:` | `num_epochs: 50` |

---

## Example: Complete Custom Extension

Here's a complete example of adding a custom component with all best practices:

### 1. Define the Component
```python
# ml_src/core/network/custom.py
class MyCustomNet(nn.Module):
    """Custom architecture for [specific purpose].

    Args:
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout probability (default: 0.5)
    """
    def __init__(self, num_classes, hidden_dim=256, dropout=0.5, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Register it
MODEL_REGISTRY = {
    # ... existing models
    'my_custom_net': MyCustomNet,
}
```

### 2. Configure It
```yaml
# ml_src/config_template.yaml or custom config
model:
  type: 'custom'
  custom_architecture: 'my_custom_net'
  num_classes: 10
  hidden_dim: 256  # Custom parameter
  dropout: 0.5     # Custom parameter
```

### 3. Test It
```python
# test_custom_net.py
from ml_src.core.network import get_model
import torch

config = {
    'model': {
        'type': 'custom',
        'custom_architecture': 'my_custom_net',
        'num_classes': 10,
        'hidden_dim': 256,
        'dropout': 0.5
    }
}

model = get_model(config, 'cpu')
x = torch.randn(2, 3, 224, 224)
y = model(x)
assert y.shape == (2, 10), f"Expected (2, 10), got {y.shape}"
print("Test passed!")
```

### 4. Train It
```bash
ml-train --config configs/my_custom_net.yaml
```

---

## Getting Help

If you encounter issues while extending the framework:

1. **Review the specific guide** for the component you're extending
2. **Check existing implementations** in the codebase for reference
3. **Read the architecture documentation** to understand the system
4. **Test incrementally** to isolate issues
5. **Consult troubleshooting guides** for common problems

### Helpful Resources
- [Troubleshooting Guide](../reference/troubleshooting.md) - Common issues and solutions
- [Architecture Documentation](../architecture/README.md) - Deep dive into codebase
- [Best Practices](../reference/best-practices.md) - Tips and conventions

---

## Navigation

**← Back to [Main Documentation](../README.md)**

**Explore Other Sections:**
- [Getting Started](../getting-started/) - Setup and first steps
- [Configuration](../configuration/) - Complete configuration reference
- [User Guides](../user-guides/) - Practical workflows
- [Architecture](../architecture/) - System design and structure
- [Reference](../reference/) - Quick lookups and troubleshooting

---

**Ready to extend the framework?** Start with the guide most relevant to your needs, or explore the architecture documentation to understand the system better.
