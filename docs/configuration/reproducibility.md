# Reproducibility Configuration

## Overview

Reproducibility settings control random number generation and deterministic behavior across the entire training pipeline. These settings are crucial for debugging, research, and comparing experiments.

## Configuration Parameters

### `seed`

- **Type:** Integer
- **Default:** `42`
- **Description:** Random seed for all random number generators (Python, NumPy, PyTorch, CUDA)
- **Purpose:** Ensures reproducibility of experiments

#### Usage

```yaml
seed: 42
```

#### What Gets Seeded

- Python's `random` module
- NumPy's `np.random`
- PyTorch's random number generators (CPU)
- All CUDA devices
- DataLoader worker processes

#### When to Change

**Use Different Seeds:**
- Running multiple independent experiments for statistical significance
- Training ensemble models (each model should use different seed)
- Exploring variance in results

**Keep Same Seed:**
- Debugging training issues (consistent behavior across runs)
- Comparing different architectures or hyperparameters fairly
- Reproducing published results

#### Example

```bash
# Train 5 models with different seeds for ensemble
ml-train --seed 42
ml-train --seed 123
ml-train --seed 456
ml-train --seed 789
ml-train --seed 101112
```

---

### `deterministic`

- **Type:** Boolean
- **Default:** `false`
- **Description:** Enable fully deterministic operations at the cost of performance
- **Purpose:** Guarantee bit-exact reproducibility across runs

#### Usage

```yaml
deterministic: false  # Fast, approximately reproducible (default)
# OR
deterministic: true   # Slower, fully reproducible
```

#### Technical Details

**When `false` (default):**
- Uses cuDNN benchmark mode to find fastest algorithms
- Algorithms may be non-deterministic (slight variations across runs)
- Faster training (baseline speed: 1.0x)
- Approximate reproducibility (same seed → similar results)

**When `true`:**
- Forces PyTorch to use deterministic algorithms
- Disables cuDNN benchmark mode
- Sets `torch.use_deterministic_algorithms(True)`
- Slower training (typically 0.7-0.9x speed)
- Exact reproducibility (same seed → identical results)

#### Performance Trade-off

| Mode | Speed | Reproducibility | Use Case |
|------|-------|-----------------|----------|
| `false` | Fast (1.0x) | Approximate | Production, general training |
| `true` | Slower (0.7-0.9x) | Bit-exact | Debugging, research papers |

#### When to Use `true` (Deterministic Mode)

✅ **Debugging training issues**
- Ensures consistent behavior when troubleshooting
- Makes it easier to identify problems

✅ **Comparing optimization algorithms**
- Fair comparison requires identical conditions
- Eliminates randomness as a confounding factor

✅ **Publishing reproducible research**
- Academic papers should provide reproducible results
- Enables others to verify your findings

✅ **Legal/compliance requirements**
- Some industries require deterministic model training
- Audit trails need exact reproducibility

#### When to Use `false` (Non-Deterministic Mode - Default)

✅ **Production training**
- Speed matters more than exact reproducibility
- Approximate reproducibility is sufficient

✅ **Hyperparameter search**
- Running many experiments quickly
- Exact reproducibility not critical

✅ **General experimentation**
- Exploring ideas and iterating fast
- Acceptable variation across runs

✅ **Large-scale training**
- Long training times make speed critical
- 10-30% speedup is significant

## Complete Examples

### Example 1: Research Paper (Full Reproducibility)

```yaml
seed: 42
deterministic: true  # Exact reproducibility for paper
```

**Why:**
- Readers can reproduce exact results
- Eliminates randomness concerns
- Speed is less critical for one-time training

### Example 2: Production Training (Fast)

```yaml
seed: 42
deterministic: false  # Fast training with approximate reproducibility
```

**Why:**
- Need to train many models quickly
- Approximate reproducibility sufficient
- Speed is critical for iteration

### Example 3: Hyperparameter Search

```yaml
seed: 42
deterministic: false  # Fast experiments
```

**Why:**
- Running hundreds of experiments
- Need quick feedback
- Statistical trends more important than exact numbers

### Example 4: Debugging

```yaml
seed: 12345
deterministic: true  # Consistent behavior for debugging
```

**Why:**
- Need identical behavior across runs
- Easier to isolate problems
- Can step through code deterministically

## Implementation Details

### What the Framework Does

When you set these parameters, the framework:

1. **Seeds all random number generators:**
   ```python
   import random
   import numpy as np
   import torch
   
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   ```

2. **Configures cuDNN behavior:**
   ```python
   if deterministic:
       torch.use_deterministic_algorithms(True)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   else:
       torch.backends.cudnn.deterministic = False
       torch.backends.cudnn.benchmark = True
   ```

3. **Seeds DataLoader workers:**
   ```python
   def seed_worker(worker_id):
       worker_seed = torch.initial_seed() % 2**32
       np.random.seed(worker_seed)
       random.seed(worker_seed)
   ```

### Limitations

Even with `deterministic: true`, exact reproducibility requires:
- Same PyTorch version
- Same CUDA version
- Same GPU hardware
- Same operating system (potentially)
- Single-GPU training (multi-GPU has additional challenges)

**Bottom line:** `deterministic: true` helps significantly, but 100% reproducibility across different hardware/software is challenging.

## Best Practices

1. **Always set a seed** (even if `deterministic: false`)
   - Enables approximate reproducibility
   - Useful for debugging

2. **Document your seed** in experiment logs
   - Makes it possible to reproduce later
   - Include in paper/report methods sections

3. **Use deterministic mode for critical work**
   - Research papers
   - Production models (after hyperparameter search)
   - Debugging

4. **Use non-deterministic mode for exploration**
   - Faster iteration
   - Hyperparameter search
   - Prototyping

5. **Test reproducibility**
   ```bash
   # Run twice with same seed
   ml-train --seed 42
   ml-train --seed 42

   # Compare results - should be identical (if deterministic: true)
   # or very similar (if deterministic: false)
   ```

## Related Configuration

- [CLI Overrides](cli-overrides.md) - How to set seed via command line
- [Training Configuration](training.md) - Other training parameters
- [Examples](examples.md) - See reproducibility in complete configs

## Further Reading

- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html)
- [Deterministic Algorithms](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
