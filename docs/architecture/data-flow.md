# Data Flow

## Overview

This document explains how data flows through the system during training and inference, showing the complete pipeline from configuration to results.

---

## Training Pipeline

### High-Level Flow

```
Configuration → Data Preparation → Model Initialization → Training Loop → Evaluation
```

### Detailed Training Flow

```
┌─────────────────────┐
│  Load config.yaml   │
│  + CLI overrides    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Set seed &        │
│   deterministic     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create datasets    │
│  (train/val/test)   │
│  with transforms    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create DataLoaders │
│  with seeding       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Initialize model   │
│  (base or custom)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Create optimizer   │
│  and scheduler      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Get loss function  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Resume checkpoint? ├──► Yes ──► Load state
│                     │
└──────────┬──────────┘
           │ No
           ▼
    ┌──────────────────────────┐
    │     TRAINING LOOP        │
    │  (for each epoch)        │
    └───────────┬──────────────┘
                │
       ┌────────▼─────────┐
       │  Training Phase  │
       │  ────────────── │
       │  • Set train mode│
       │  • For each batch│
       │    - Forward     │
       │    - Loss        │
       │    - Backward    │
       │    - Optimizer   │
       │  • Track metrics │
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │ Validation Phase │
       │  ────────────── │
       │  • Set eval mode │
       │  • For each batch│
       │    - Forward     │
       │    - Loss        │
       │  • Track metrics │
       └────────┬─────────┘
                │
       ┌────────▼─────────┐
       │  Scheduler step  │
       └────────┬─────────┘
                │
       ┌────────▼──────────┐
       │  Save checkpoints │
       │  • best.pt (if    │
       │    val acc ↑)     │
       │  • last.pt        │
       └────────┬──────────┘
                │
       ┌────────▼──────────┐
       │  Update summary   │
       │  & TensorBoard    │
       └────────┬──────────┘
                │
                ▼
            [Repeat]
                │
                ▼
      ┌──────────────────────┐
      │   POST-TRAINING      │
      │  ──────────────────  │
      │  • Load best model   │
      │  • Generate metrics  │
      │  • Confusion matrices│
      │  • Classification    │
      │    reports           │
      │  • Final summary     │
      └──────────────────────┘
```

---

## Training Loop Detail

### Single Epoch Flow

```
TRAINING PHASE:
───────────────
model.train()
    │
    ▼
for batch in train_loader:
    │
    ├─► images, labels = batch
    │
    ├─► images = images.to(device)
    │   labels = labels.to(device)
    │
    ├─► optimizer.zero_grad()
    │
    ├─► outputs = model(images)      # Forward
    │
    ├─► loss = criterion(outputs, labels)
    │
    ├─► loss.backward()              # Backward
    │
    ├─► optimizer.step()             # Update weights
    │
    └─► track metrics (loss, accuracy)

VALIDATION PHASE:
─────────────────
model.eval()
    │
    ▼
with torch.no_grad():
    │
    ▼
    for batch in val_loader:
        │
        ├─► images, labels = batch
        │
        ├─► images = images.to(device)
        │   labels = labels.to(device)
        │
        ├─► outputs = model(images)      # Forward only
        │
        ├─► loss = criterion(outputs, labels)
        │
        └─► track metrics (loss, accuracy)

SCHEDULER & CHECKPOINTING:
──────────────────────────
scheduler.step()
    │
    ▼
if val_acc > best_acc:
    │
    ├─► best_acc = val_acc
    └─► save best.pt
    │
save last.pt
    │
    ▼
update summary.txt
log to TensorBoard
```

---

## Inference Pipeline

### High-Level Flow

```
Load Config → Load Model → Load Data → Inference → Metrics → Results
```

### Detailed Inference Flow

```
┌──────────────────────┐
│  Load config from    │
│  runs/{run_name}/    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Load checkpoint     │
│  (best.pt or last.pt)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Create test dataset │
│  & DataLoader        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Initialize model    │
│  architecture        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Load trained        │
│  weights into model  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Run inference       │
│  ──────────────────  │
│  model.eval()        │
│  with torch.no_grad()│
│                      │
│  for batch in test:  │
│    outputs = model() │
│    predictions.append│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Generate metrics    │
│  ──────────────────  │
│  • Accuracy          │
│  • Confusion matrix  │
│  • Classification    │
│    report            │
│  • Per-sample results│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Display & save      │
│  results             │
└──────────────────────┘
```

---

## Data Transformation Pipeline

### Image Processing Flow

```
Raw Image
    │
    ▼
[Resize to target size]
    │     (e.g., 224×224)
    ▼
[Random Horizontal Flip]  ─────► (Training only)
    │
    ▼
[Convert to Tensor]
    │     (HWC → CHW, [0,255] → [0,1])
    ▼
[Normalize]
    │     (per-channel: (x - mean) / std)
    ▼
Preprocessed Tensor
    │     shape: (C, H, W)
    ▼
[Batching in DataLoader]
    │
    ▼
Batch Tensor
    │     shape: (B, C, H, W)
    ▼
Model Input
```

---

## Module Interaction

### Component Communication

```
ml-train (ml_src/cli/train.py)
    │
    ├─► core.seeding.set_seed()
    │
    ├─► core.dataset.get_datasets()
    │   └─► core.dataset.get_transforms()
    │
    ├─► core.loader.get_dataloaders()
    │   └─► core.seeding.seed_worker()
    │
    ├─► core.network.get_model()
    │   ├─► core.network.base.get_base_model()
    │   └─► core.network.custom.get_custom_model()
    │
    ├─► core.optimizer.get_optimizer()
    │
    ├─► core.optimizer.get_scheduler()
    │
    ├─► core.loss.get_criterion()
    │
    ├─► core.checkpointing.load_checkpoint() (if resume)
    │
    ├─► core.trainer.train_model()
    │   │
    │   ├─► [Training loop]
    │   │
    │   ├─► core.checkpointing.save_checkpoint()
    │   │
    │   ├─► core.checkpointing.save_summary()
    │   │
    │   └─► core.trainer.collect_predictions()
    │
    └─► core.metrics.save_confusion_matrix()
        core.metrics.save_classification_report()

ml-inference (ml_src/cli/inference.py)
    │
    ├─► load config.yaml
    │
    ├─► core.dataset.get_datasets()
    │
    ├─► core.loader.get_dataloaders()
    │
    ├─► core.network.get_model()
    │
    ├─► core.network.load_model()
    │
    ├─► core.inference.get_inference_strategy()
    │   └─► StandardInference.run_inference() / MixedPrecisionInference.run_inference() / AccelerateInference.run_inference()
    │
    └─► core.metrics.save_confusion_matrix()
        core.metrics.save_classification_report()
```

---

## Checkpoint State Flow

### What's Saved and Restored

```
save_checkpoint():
    │
    ├─► model.state_dict()        # Model weights
    ├─► optimizer.state_dict()    # Optimizer state (momentum, etc.)
    ├─► scheduler.state_dict()    # LR scheduler state
    ├─► epoch                     # Current epoch number
    ├─► best_acc                  # Best validation accuracy
    ├─► train_losses              # Training loss history
    ├─► val_losses                # Validation loss history
    ├─► train_accs                # Training accuracy history
    ├─► val_accs                  # Validation accuracy history
    ├─► random_states             # All RNG states
    │   ├─► python_state
    │   ├─► numpy_state
    │   ├─► torch_state
    │   └─► cuda_state
    ├─► config                    # Complete configuration
    └─► timestamp                 # When checkpoint was created

load_checkpoint():
    │
    ├─► Restore all above state
    │
    └─► Return checkpoint dict
```

**Resume Training:**
```
1. Load checkpoint
2. Restore model, optimizer, scheduler
3. Get start_epoch = checkpoint['epoch'] + 1
4. Get best_acc = checkpoint['best_acc']
5. Continue training from start_epoch
```

---

## TensorBoard Logging Flow

### What Gets Logged

```
During Training:
    │
    ├─► Every batch:
    │   └─► training_loss
    │
    ├─► Every epoch:
    │   ├─► epoch_train_loss
    │   ├─► epoch_train_accuracy
    │   ├─► epoch_val_loss
    │   ├─► epoch_val_accuracy
    │   └─► learning_rate
    │
    └─► End of training:
        ├─► confusion_matrix_train
        ├─► confusion_matrix_val
        └─► confusion_matrix_test
```

**View in TensorBoard:**
```bash
tensorboard --logdir runs/
# Browse to http://localhost:6006
```

---

## Configuration Override Flow

### How Overrides Are Applied

```
1. Load base config.yaml
   ↓
2. Parse CLI arguments
   ↓
3. For each CLI argument:
   if provided:
       override config value
   ↓
4. Generate run name from overrides
   ↓
5. Create run directory
   ↓
6. Save final config to run directory
```

**Example:**
```bash
ml-train --lr 0.01 --batch_size 32

# Flow:
config = load('ml_src/config.yaml')  # lr: 0.001, batch_size: 4
config['optimizer']['lr'] = 0.01      # Override
config['training']['batch_size'] = 32 # Override
run_name = 'batch_32_lr_0.01'        # Generate name
save(config, f'runs/{run_name}/config.yaml')  # Save
```

---

## Error Propagation

### How Errors Are Handled

```
try:
    ├─► Configuration loading
    ├─► Data loading
    ├─► Model initialization
    ├─► Training loop
    │   │
    │   └─► On error:
    │       ├─► Log error
    │       ├─► Save checkpoint
    │       ├─► Update summary (status='failed')
    │       └─► Raise exception
    │
└─► except KeyboardInterrupt:
    │   ├─► Log interruption
    │   ├─► Save checkpoint
    │   └─► Update summary (status='interrupted')
    │
    except Exception:
        ├─► Log error
        ├─► Save summary with error message
        └─► Raise
```

---

## Summary

### Key Flows

**Training:**
```
Config → Seed → Data → Model → Train → Validate → Checkpoint → Repeat
```

**Inference:**
```
Load Config → Load Model → Test Data → Predict → Metrics → Display
```

**Resumption:**
```
Checkpoint → Restore State → Continue Training
```

**Monitoring:**
```
Training → Log → TensorBoard/Files → User
```

### Design Principles

1. **Linear flow** - Clear progression from start to finish
2. **Checkpointing** - Save state at every key point
3. **Logging** - Track everything for debugging
4. **Error handling** - Graceful failure with state preservation
5. **Reproducibility** - Seeding at every random operation

---

## Related Documentation

- [Entry Points](entry-points.md) - How train.py and inference.py orchestrate
- [ML Source Modules](ml-src-modules.md) - What each module does
- [Design Decisions](design-decisions.md) - Why it's organized this way
