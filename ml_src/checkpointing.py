"""Checkpointing utilities for saving and resuming training."""

import os
import time
from datetime import datetime
import torch
import numpy as np
import random
from loguru import logger


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_acc,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    config,
    checkpoint_path
):
    """
    Save a comprehensive training checkpoint.

    Args:
        model: The model being trained
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        best_acc: Best validation accuracy achieved so far
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        config: Configuration dictionary
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        # Save random states for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }

    # Save CUDA RNG state if using CUDA
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, checkpoint_path)
    logger.debug(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """
    Load a checkpoint to resume training.

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to map tensors to

    Returns:
        Tuple of (epoch, best_acc, train_losses, val_losses, train_accs, val_accs, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model, optimizer, and scheduler states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore random states for reproducibility
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])

    if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_accs = checkpoint.get('val_accs', [])
    config = checkpoint.get('config', None)

    logger.success(f"Resumed from epoch {epoch}, best accuracy: {best_acc:.4f}")

    return epoch, best_acc, train_losses, val_losses, train_accs, val_accs, config


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_duration(seconds):
    """Format duration in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:.0f}h {minutes:.0f}m {secs:.0f}s"


def save_summary(
    summary_path,
    status,
    config=None,
    device=None,
    dataset_sizes=None,
    num_parameters=None,
    start_time=None,
    end_time=None,
    current_epoch=None,
    total_epochs=None,
    best_acc=None,
    best_epoch=None,
    final_train_acc=None,
    final_train_loss=None,
    final_val_acc=None,
    final_val_loss=None,
    error_message=None
):
    """
    Save or update training summary to a text file.

    Args:
        summary_path: Path to save the summary file
        status: Training status ('running', 'completed', 'failed')
        config: Configuration dictionary
        device: Device used for training
        dataset_sizes: Dictionary of dataset sizes
        num_parameters: Number of trainable parameters
        start_time: Training start timestamp
        end_time: Training end timestamp (if completed/failed)
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        best_acc: Best validation accuracy achieved
        best_epoch: Epoch where best accuracy was achieved
        final_train_acc: Final training accuracy
        final_train_loss: Final training loss
        final_val_acc: Final validation accuracy
        final_val_loss: Final validation loss
        error_message: Error message if failed
    """
    lines = []
    lines.append("=" * 70)
    lines.append("TRAINING SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Status
    status_upper = status.upper()
    if status == 'completed':
        status_display = f"✓ {status_upper}"
    elif status == 'running':
        status_display = f"⟳ {status_upper}"
    elif status == 'failed':
        status_display = f"✗ {status_upper}"
    else:
        status_display = status_upper

    lines.append(f"Status: {status_display}")
    lines.append("")

    # Timing information
    lines.append("-" * 70)
    lines.append("TIMING")
    lines.append("-" * 70)
    if start_time:
        lines.append(f"Started:  {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    if end_time:
        lines.append(f"Finished: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    if start_time and end_time:
        duration = end_time - start_time
        lines.append(f"Duration: {format_duration(duration)}")
    elif start_time and status == 'running':
        elapsed = time.time() - start_time
        lines.append(f"Elapsed:  {format_duration(elapsed)}")
    lines.append("")

    # Progress
    if current_epoch is not None and total_epochs is not None:
        lines.append("-" * 70)
        lines.append("PROGRESS")
        lines.append("-" * 70)
        progress_pct = (current_epoch / total_epochs) * 100
        lines.append(f"Epoch:    {current_epoch}/{total_epochs} ({progress_pct:.1f}%)")
        lines.append("")

    # Performance metrics
    if best_acc is not None or final_train_acc is not None:
        lines.append("-" * 70)
        lines.append("METRICS")
        lines.append("-" * 70)
        if best_acc is not None:
            best_info = f"{best_acc:.4f}"
            if best_epoch is not None:
                best_info += f" (epoch {best_epoch})"
            lines.append(f"Best Val Accuracy:  {best_info}")
        if final_val_acc is not None:
            lines.append(f"Final Val Accuracy: {final_val_acc:.4f}")
        if final_val_loss is not None:
            lines.append(f"Final Val Loss:     {final_val_loss:.4f}")
        if final_train_acc is not None:
            lines.append(f"Final Train Accuracy: {final_train_acc:.4f}")
        if final_train_loss is not None:
            lines.append(f"Final Train Loss:     {final_train_loss:.4f}")
        lines.append("")

    # Configuration
    if config:
        lines.append("-" * 70)
        lines.append("CONFIGURATION")
        lines.append("-" * 70)
        if 'training' in config:
            lines.append(f"Batch Size:   {config['training'].get('batch_size', 'N/A')}")
            lines.append(f"Epochs:       {config['training'].get('num_epochs', 'N/A')}")
        if 'optimizer' in config:
            lines.append(f"Learning Rate: {config['optimizer'].get('lr', 'N/A')}")
            lines.append(f"Momentum:     {config['optimizer'].get('momentum', 'N/A')}")
        if 'scheduler' in config:
            lines.append(f"LR Step Size: {config['scheduler'].get('step_size', 'N/A')}")
            lines.append(f"LR Gamma:     {config['scheduler'].get('gamma', 'N/A')}")
        lines.append("")

    # System information
    lines.append("-" * 70)
    lines.append("SYSTEM")
    lines.append("-" * 70)
    if device:
        lines.append(f"Device: {device}")
    if num_parameters is not None:
        lines.append(f"Model Parameters: {num_parameters:,}")
    if dataset_sizes:
        lines.append(f"Dataset Sizes:")
        for split, size in dataset_sizes.items():
            lines.append(f"  {split}: {size:,}")
    lines.append("")

    # Error information (if failed)
    if status == 'failed' and error_message:
        lines.append("-" * 70)
        lines.append("ERROR")
        lines.append("-" * 70)
        lines.append(error_message)
        lines.append("")

    lines.append("=" * 70)

    # Write to file
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.debug(f"Updated summary: {summary_path}")
