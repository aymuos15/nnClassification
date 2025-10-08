"""
Callbacks module for training lifecycle hooks.

This module provides a flexible callback system for injecting custom behavior
into the training loop. Callbacks can be configured via YAML and are instantiated
automatically by the get_callbacks() factory function.

Available callbacks:
    - early_stopping: Stop training when validation metric stops improving
    - model_checkpoint: Save best model checkpoints based on metric
    - lr_monitor: Log learning rate to TensorBoard
    - progress_bar: Display training progress with tqdm
    - swa: Stochastic Weight Averaging for improved generalization
    - gradient_clipping: Clip gradients to prevent exploding gradients
    - gradient_norm_monitor: Monitor and log gradient norms
    - mixup: MixUp data augmentation
    - cutmix: CutMix data augmentation

Example:
    >>> # In config.yaml:
    >>> # training:
    >>> #   callbacks:
    >>> #     - type: 'early_stopping'
    >>> #       monitor: 'val_acc'
    >>> #       patience: 10
    >>> #     - type: 'model_checkpoint'
    >>> #       monitor: 'val_acc'
    >>> #       save_top_k: 3
    >>>
    >>> from ml_src.core.callbacks import get_callbacks
    >>> callbacks = get_callbacks(config)
    >>> trainer = get_trainer(..., callbacks=callbacks)
"""

from loguru import logger

from ml_src.core.callbacks.base import Callback, CallbackManager

# Callback registry - populated as callbacks are imported
CALLBACK_REGISTRY = {}


def register_callback(name, callback_class):
    """
    Register a callback class in the registry.

    Args:
        name: Callback type name (used in config)
        callback_class: Callback class to register

    Example:
        >>> register_callback('early_stopping', EarlyStoppingCallback)
    """
    CALLBACK_REGISTRY[name] = callback_class


def get_callbacks(config):
    """
    Create callbacks from configuration.

    Reads the callbacks configuration from config['training']['callbacks']
    and instantiates the appropriate callback classes.

    Args:
        config: Configuration dictionary containing callback specifications

    Returns:
        List of instantiated Callback objects

    Example:
        >>> config = {
        ...     'training': {
        ...         'callbacks': [
        ...             {'type': 'early_stopping', 'patience': 10, 'monitor': 'val_acc'},
        ...             {'type': 'model_checkpoint', 'save_top_k': 3}
        ...         ]
        ...     }
        ... }
        >>> callbacks = get_callbacks(config)
        >>> # Returns [EarlyStoppingCallback(...), ModelCheckpointCallback(...)]
    """
    callbacks = []
    callback_configs = config.get("training", {}).get("callbacks", [])

    if not callback_configs:
        logger.debug("No callbacks configured")
        return callbacks

    for cb_config in callback_configs:
        cb_type = cb_config.get("type")
        if not cb_type:
            logger.warning(f"Callback config missing 'type' field: {cb_config}")
            continue

        if cb_type not in CALLBACK_REGISTRY:
            logger.warning(
                f"Unknown callback type '{cb_type}'. Available: {list(CALLBACK_REGISTRY.keys())}"
            )
            continue

        # Get callback class and remove 'type' from kwargs
        cb_class = CALLBACK_REGISTRY[cb_type]
        cb_kwargs = {k: v for k, v in cb_config.items() if k != "type"}

        try:
            callback = cb_class(**cb_kwargs)
            callbacks.append(callback)
            logger.debug(f"Registered callback: {cb_type}")
        except Exception as e:
            logger.error(f"Failed to instantiate callback '{cb_type}': {e}")
            raise

    logger.info(f"Loaded {len(callbacks)} callbacks: {[type(cb).__name__ for cb in callbacks]}")
    return callbacks


# Import and register callbacks
# Note: Callbacks will be registered as they are implemented
try:
    from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback

    register_callback("early_stopping", EarlyStoppingCallback)
except ImportError:
    pass  # Callback not yet implemented

try:
    from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback

    register_callback("model_checkpoint", ModelCheckpointCallback)
except ImportError:
    pass

try:
    from ml_src.core.callbacks.lr_monitor import LearningRateMonitor

    register_callback("lr_monitor", LearningRateMonitor)
except ImportError:
    pass

try:
    from ml_src.core.callbacks.progress import ProgressBar

    register_callback("progress_bar", ProgressBar)
except ImportError:
    pass

try:
    from ml_src.core.callbacks.swa import StochasticWeightAveraging

    register_callback("swa", StochasticWeightAveraging)
except ImportError:
    pass

try:
    from ml_src.core.callbacks.gradient import GradientClipping, GradientNormMonitor

    register_callback("gradient_clipping", GradientClipping)
    register_callback("gradient_norm_monitor", GradientNormMonitor)
except ImportError:
    pass

try:
    from ml_src.core.callbacks.augmentation import CutMixCallback, MixUpCallback

    register_callback("mixup", MixUpCallback)
    register_callback("cutmix", CutMixCallback)
except ImportError:
    pass


__all__ = [
    "Callback",
    "CallbackManager",
    "get_callbacks",
    "register_callback",
    "CALLBACK_REGISTRY",
]