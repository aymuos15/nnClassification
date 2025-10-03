# Adding Custom Optimizers

Add new optimization algorithms.

## Overview

Optimizers are defined in `ml_src/optimizer.py::get_optimizer()`.

## Current Implementation

```python
def get_optimizer(parameters, config):
    opt_config = config['optimizer']
    
    optimizer = torch.optim.SGD(
        parameters,
        lr=opt_config['lr'],
        momentum=opt_config['momentum']
    )
    
    return optimizer
```

## Adding New Optimizer

### Step 1: Update get_optimizer()

```python
def get_optimizer(parameters, config):
    opt_config = config['optimizer']
    opt_type = opt_config.get('type', 'sgd')  # NEW
    
    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=opt_config['lr'],
            momentum=opt_config['momentum']
        )
    
    elif opt_type == 'adam':  # NEW
        optimizer = torch.optim.Adam(
            parameters,
            lr=opt_config['lr'],
            betas=opt_config.get('betas', (0.9, 0.999))
        )
    
    elif opt_type == 'adamw':  # NEW
        optimizer = torch.optim.AdamW(
            parameters,
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer
```

### Step 2: Update Config

```yaml
optimizer:
  type: 'adam'  # NEW
  lr: 0.001
  betas: [0.9, 0.999]  # NEW (Adam-specific)
```

## Popular Optimizers

### Adam
```yaml
optimizer:
  type: 'adam'
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-8
```

### AdamW
```yaml
optimizer:
  type: 'adamw'
  lr: 0.001
  weight_decay: 0.01
```

### RMSprop
```yaml
optimizer:
  type: 'rmsprop'
  lr: 0.001
  alpha: 0.99
  momentum: 0.9
```

## Adding Custom Scheduler

Similar process in `get_scheduler()`:

```python
def get_scheduler(optimizer, config):
    sched_config = config['scheduler']
    sched_type = sched_config.get('type', 'step')  # NEW
    
    if sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    
    elif sched_type == 'cosine':  # NEW
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config['T_max']
        )
    
    return scheduler
```

## Best Practices

1. Test on validation set
2. Adjust LR for different optimizers
3. Save optimizer state in checkpoints
4. Document optimizer choice

## Related

- [Optimizer Configuration](../configuration/optimizer-scheduler.md)
- [Optimizer Module](../architecture/ml-src-modules.md#optimizerpy)
EOF5
