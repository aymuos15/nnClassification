# Design Decisions

## Overview

This document explains the architectural choices made in this framework and their rationale.

---

## Modular Design

### Decision: Separate modules for each concern

**Structure:**
```
ml_src/
├── dataset.py      # Data loading only
├── loader.py       # DataLoader creation only
├── network/        # Model architectures only
├── loss.py         # Loss functions only
├── optimizer.py    # Optimization only
├── trainer.py      # Training loop only
└── ...
```

### Rationale

**Benefits:**
1. **Testability** - Test each module independently
2. **Reusability** - Use modules in other projects
3. **Maintainability** - Changes localized to specific modules
4. **Clarity** - Clear separation of concerns
5. **Parallel development** - Multiple developers can work simultaneously

**Alternative rejected:** Monolithic `train.py` with all logic

**Why rejected:** Hard to test, maintain, and understand

---

## Split Network/Loss/Optimizer

### Decision: Three separate modules instead of one "model" module

**Structure:**
```
ml_src/
├── network/        # Model architectures
├── loss.py         # Loss functions
└── optimizer.py    # Optimizers & schedulers
```

### Rationale

**Enables independent experimentation:**

```python
# Try different models with same optimizer
model = ResNet18(...)
model = EfficientNet(...)

# Try different optimizers with same model
optimizer = SGD(...)
optimizer = Adam(...)

# Try different losses with same model
criterion = CrossEntropyLoss(...)
criterion = FocalLoss(...)
```

**Benefits:**
- **Extensibility** - Add new models without touching optimizer code
- **Experimentation** - Test combinations independently
- **Single Responsibility** - Each module does one thing
- **Professional** - Matches PyTorch Lightning, timm architectures

**Alternative rejected:** Combined `model.py` with everything

**Why rejected:** Changes to optimizer affect model code, tight coupling

---

## Network Package Structure

### Decision: Package instead of single file

**Structure:**
```
network/
├── __init__.py     # Main API
├── base.py         # Torchvision models
└── custom.py       # Custom architectures
```

### Rationale

**Scalability:**
- Room for many model types
- Can add `network/pretrained/`, `network/timm_models/`, etc.

**Organization:**
- Clear separation: base vs custom
- Easy to find specific model type

**Extensibility:**
- Add new model families without affecting existing code
- Users can add custom models without modifying base.py

**Alternative rejected:** Single `network.py` file

**Why rejected:** Would become large and hard to navigate

---

## YAML Configuration

### Decision: YAML-based config instead of argparse-only

**Structure:**
```
ml_src/config.yaml  ←  Base configuration
        ↓
CLI overrides  ←  python train.py --lr 0.01
        ↓
Final config saved  →  runs/{run_name}/config.yaml
```

### Rationale

**Benefits:**
1. **Human-readable** - Easy to edit and understand
2. **Version control** - Track configuration changes in git
3. **Hierarchical** - Nested structure matches code organization
4. **Reusable** - Same config for multiple runs
5. **Reproducible** - Save exact config with results

**CLI overrides:**
- Quick experimentation without editing files
- Hyperparameter sweeps

**Alternative rejected:** Pure CLI arguments

**Why rejected:** Too many arguments, hard to track, not reusable

---

## Dual Checkpointing

### Decision: Save both `best.pt` and `last.pt`

**Files:**
- `best.pt` - Highest validation accuracy
- `last.pt` - Latest epoch (for resuming)

### Rationale

**Different use cases:**

**best.pt:**
- Deployment
- Final evaluation
- Inference
- Best model selection

**last.pt:**
- Resume interrupted training
- Continue training for more epochs
- Debugging
- Training history

**Both critical:**
```bash
# Use best for deployment
python inference.py --checkpoint best.pt

# Resume training from last
python train.py --resume runs/base/last.pt
```

**Alternative rejected:** Single checkpoint

**Why rejected:** Can't resume if best was many epochs ago

---

## Complete State Persistence

### Decision: Save everything in checkpoints

**Checkpoint contents:**
- Model weights
- Optimizer state
- Scheduler state
- Training metrics
- Random states (all RNGs)
- Configuration
- Timestamp

### Rationale

**Enables:**
1. **Exact resumption** - Continue training seamlessly
2. **Reproducibility** - Restore exact state
3. **Debugging** - Analyze training at any point
4. **History** - Complete training record

**Cost:** Larger checkpoint files (~50MB vs ~25MB for weights only)

**Verdict:** Worth it for robustness

**Alternative rejected:** Save only model weights

**Why rejected:** Can't resume training properly, lose history

---

## Automatic Run Naming

### Decision: Name runs based on hyperparameter overrides

**Examples:**
```bash
python train.py                      → runs/base/
python train.py --lr 0.01            → runs/lr_0.01/
python train.py --lr 0.01 --batch_size 32  → runs/batch_32_lr_0.01/
```

### Rationale

**Benefits:**
1. **Self-documenting** - Name tells you what changed
2. **No overwrites** - Different params → different folders
3. **Easy comparison** - Compare runs by name
4. **Organized** - Automatic experiment organization

**Alternative rejected:** Manual naming or timestamps

**Why rejected:** 
- Manual: User error, inconsistent
- Timestamps: Not descriptive, hard to compare

---

## Configuration-Driven Transforms

### Decision: Transforms in YAML, not hardcoded

**Config:**
```yaml
transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

### Rationale

**Benefits:**
1. **No code changes** - Experiment with augmentation via config
2. **Reproducible** - Config saved with results
3. **Flexible** - Different transforms per split
4. **Version controlled** - Track transform changes

**Alternative rejected:** Hardcoded transforms in `dataset.py`

**Why rejected:** Requires code changes for experimentation

---

## Structured Logging

### Decision: Console + file logging with loguru

**Setup:**
- Console: Color-coded, INFO level
- File: Detailed, DEBUG level, rotating

### Rationale

**Console:**
- Immediate feedback
- Quick visual parsing (colors)
- Clean output (INFO only)

**File:**
- Complete record
- Debug information
- Post-mortem analysis
- Rotating prevents disk issues

**loguru benefits:**
- Clean API
- Automatic formatting
- Rotation built-in
- Better than print()

**Alternative rejected:** print() statements

**Why rejected:** No control, no file logging, messy output

---

## Separate Train/Inference Scripts

### Decision: Two entry points instead of one

**Files:**
- `train.py` - Training workflow
- `inference.py` - Evaluation workflow

### Rationale

**Benefits:**
1. **Clarity** - Each script does one thing
2. **Independence** - Use trained models without training code
3. **Simpler** - Easier to understand and maintain
4. **Deployment** - Only need inference.py for production

**Alternative rejected:** Single script with modes

**Why rejected:** Complex flag handling, confusing logic

---

## ImageFolder Dataset Structure

### Decision: Use PyTorch's ImageFolder with required structure

**Required structure:**
```
data/
├── train/
│   ├── class1/
│   └── class2/
├── val/
└── test/
```

### Rationale

**Pros:**
- Standard PyTorch pattern
- Well-documented
- Simple to understand
- Works with existing tools

**Cons:**
- Rigid structure requirement
- Manual organization needed

**Verdict:** Pros outweigh cons for most use cases

**Alternative rejected:** Custom dataset class

**Why rejected:** Reinventing the wheel, more complex

---

## Default Non-Deterministic Mode

### Decision: `deterministic: false` by default

**Performance:**
- Non-deterministic: 1.0x (fast)
- Deterministic: 0.7-0.9x (slower)

### Rationale

**Most users want:**
- Fast training
- Approximate reproducibility (good enough)

**Deterministic mode still available:**
- For research
- For debugging
- When needed

**Trade-off accepted:** Slight variation across runs

**Alternative rejected:** Deterministic by default

**Why rejected:** Unnecessary performance cost for most users

---

## TensorBoard Integration

### Decision: Use TensorBoard for visualization

### Rationale

**Benefits:**
1. **Interactive** - Zoom, pan, compare runs
2. **Real-time** - Watch training live
3. **Standard** - Everyone knows TensorBoard
4. **Rich** - Plots, histograms, embeddings

**Alternative rejected:** Custom plotting or WandB

**Why rejected:**
- Custom: Too much work, less features
- WandB: External dependency, requires account

---

## Seeded DataLoader Workers

### Decision: Seed each DataLoader worker process

**Implementation:**
```python
DataLoader(..., worker_init_fn=seed_worker)
```

### Rationale

**Problem:** Multi-process data loading uses different RNG states

**Solution:** Seed each worker with derived seed

**Result:** Reproducible data loading even with `num_workers > 0`

**Alternative rejected:** Single-threaded loading

**Why rejected:** Too slow, not practical

---

## Summary of Design Philosophy

### Core Principles

1. **Modularity** - Independent, reusable components
2. **Configurability** - YAML + CLI for flexibility
3. **Reproducibility** - Complete state tracking
4. **Usability** - Clear interfaces, helpful defaults
5. **Extensibility** - Easy to add new functionality
6. **Production-ready** - Logging, error handling, robustness

### Trade-offs Accepted

| Trade-off | Decision | Rationale |
|-----------|----------|-----------|
| Speed vs Determinism | Default non-deterministic | Most users prefer speed |
| Checkpoint size | Save everything | Robustness > disk space |
| Structure rigidity | Require ImageFolder format | Simplicity > flexibility |
| Dual checkpoints | best + last | Robustness > disk space |

### Rejected Alternatives

| Alternative | Why Rejected |
|-------------|--------------|
| Monolithic train.py | Hard to test and maintain |
| Single checkpoint | Can't resume properly |
| Hardcoded config | Not flexible enough |
| Manual run naming | Error-prone, inconsistent |
| Pure CLI arguments | Too many, hard to track |

---

## Related Documentation

- [Architecture Overview](overview.md)
- [Entry Points](entry-points.md)
- [ML Source Modules](ml-src-modules.md)
- [Data Flow](data-flow.md)
