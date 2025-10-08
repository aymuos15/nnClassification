"""Trainer factory for creating different training strategies."""

from ml_src.core.trainers.base import BaseTrainer
from ml_src.core.trainers.mixed_precision import MixedPrecisionTrainer
from ml_src.core.trainers.standard import StandardTrainer

# Conditional import for AccelerateTrainer (requires accelerate)
try:
    from ml_src.core.trainers.accelerate import AccelerateTrainer

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False

# Conditional import for DPTrainer (requires opacus)
try:
    from ml_src.core.trainers.differential_privacy import DPTrainer

    _OPACUS_AVAILABLE = True
except ImportError:
    _OPACUS_AVAILABLE = False


def get_trainer(
    config,
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    run_dir,
    class_names,
    **kwargs,
):
    """
    Factory function to create appropriate trainer based on configuration.

    This factory enables different training strategies:
    - 'standard': Traditional PyTorch training with manual device management
    - 'mixed_precision': Training with automatic mixed precision (AMP)
    - 'accelerate': Training with HuggingFace Accelerate for multi-GPU/distributed
    - 'dp': Differential privacy training with Opacus (requires: pip install opacus)

    Args:
        config: Configuration dictionary
        model: The model to train
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary of dataloaders for train and val
        dataset_sizes: Dictionary of dataset sizes
        device: Device to train on
        run_dir: Directory to save model checkpoints
        class_names: List of class names
        **kwargs: Additional trainer-specific arguments

    Returns:
        BaseTrainer: An instance of the appropriate trainer class

    Raises:
        ValueError: If trainer_type is not supported

    Example:
        >>> trainer = get_trainer(
        ...     config=config,
        ...     model=model,
        ...     criterion=criterion,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     dataloaders=dataloaders,
        ...     dataset_sizes=dataset_sizes,
        ...     device=device,
        ...     run_dir=run_dir,
        ...     class_names=class_names
        ... )
        >>> model, train_losses, val_losses, train_accs, val_accs = trainer.train()
    """
    # Get trainer type from config, default to 'standard' for backward compatibility
    trainer_type = config.get("training", {}).get("trainer_type", "standard")

    if trainer_type == "standard":
        return StandardTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=class_names,
            **kwargs,  # Pass through callbacks and other args
        )
    elif trainer_type == "mixed_precision":
        return MixedPrecisionTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=class_names,
            **kwargs,  # Pass through callbacks and other args
        )
    elif trainer_type == "accelerate":
        if not _ACCELERATE_AVAILABLE:
            raise ImportError(
                "AccelerateTrainer requires accelerate. Install with: pip install accelerate\n"
                "Or install with accelerate extras: pip install -e '.[accelerate]'"
            )
        return AccelerateTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=class_names,
            **kwargs,  # Pass through callbacks and other args
        )
    elif trainer_type == "dp":
        if not _OPACUS_AVAILABLE:
            raise ImportError(
                "DPTrainer requires opacus. Install with: pip install opacus\n"
                "Or install with dp extras: pip install -e '.[dp]'"
            )
        return DPTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=class_names,
            **kwargs,  # Pass through callbacks and other args
        )
    else:
        raise ValueError(
            f"Unsupported trainer_type: '{trainer_type}'. "
            f"Available options: 'standard', 'mixed_precision', 'accelerate', 'dp'. "
            f"Note: 'dp' requires opacus (pip install opacus or pip install -e '.[dp]')."
        )


__all__ = [
    "BaseTrainer",
    "StandardTrainer",
    "MixedPrecisionTrainer",
    "get_trainer",
]

# Conditionally add optional trainers to __all__ if available
if _ACCELERATE_AVAILABLE:
    __all__.append("AccelerateTrainer")

if _OPACUS_AVAILABLE:
    __all__.append("DPTrainer")
