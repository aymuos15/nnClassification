"""Inference factory for creating different inference strategies."""

import torch

from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.inference.ensemble import EnsembleInference
from ml_src.core.inference.mixed_precision import MixedPrecisionInference
from ml_src.core.inference.standard import StandardInference
from ml_src.core.inference.tta import TTAInference
from ml_src.core.inference.tta_ensemble import TTAEnsembleInference

# Conditional import for AccelerateInference (requires accelerate)
try:
    from ml_src.core.inference.accelerate import AccelerateInference

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


def get_inference_strategy(config, device=None):
    """
    Factory function to create appropriate inference strategy based on configuration.

    This factory enables different inference strategies:
    - 'standard': Traditional PyTorch inference with manual device management (default)
    - 'mixed_precision': Inference with automatic mixed precision (AMP) for speed
    - 'accelerate': Inference with HuggingFace Accelerate for multi-GPU/distributed
    - 'tta': Test-Time Augmentation for improved robustness
    - 'ensemble': Model ensembling from multiple checkpoints
    - 'tta_ensemble': Combined TTA + Ensemble (maximum performance)

    Args:
        config: Configuration dictionary containing inference settings
        device: Device to use (required for ensemble strategies)

    Returns:
        BaseInferenceStrategy: An instance of the appropriate inference strategy class

    Raises:
        ValueError: If inference strategy is not supported

    Example:
        >>> # Standard inference (default)
        >>> config = {'inference': {'strategy': 'standard'}}
        >>> strategy = get_inference_strategy(config)
        >>> acc, results = strategy.run_inference(model, loader, size, device)
        >>>
        >>> # TTA inference
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'tta',
        ...         'tta': {
        ...             'augmentations': ['horizontal_flip', 'vertical_flip'],
        ...             'aggregation': 'mean'
        ...         }
        ...     }
        ... }
        >>> strategy = get_inference_strategy(config)
        >>> acc, results = strategy.run_inference(model, loader, size, device)
        >>>
        >>> # Ensemble inference
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'ensemble',
        ...         'ensemble': {
        ...             'checkpoints': ['runs/fold_0/weights/best.pt', ...],
        ...             'aggregation': 'soft_voting'
        ...         }
        ...     }
        ... }
        >>> strategy = get_inference_strategy(config, device='cuda:0')
        >>> acc, results = strategy.run_inference(None, loader, size, device)
    """
    # Get strategy from config, default to 'standard' for backward compatibility
    strategy = config.get("inference", {}).get("strategy", "standard")

    if strategy == "standard":
        return StandardInference()

    elif strategy == "mixed_precision":
        # Get amp_dtype from config (default to float16)
        amp_dtype_str = config.get("inference", {}).get("amp_dtype", "float16")

        # Map string to torch dtype
        if amp_dtype_str == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

        return MixedPrecisionInference(amp_dtype=amp_dtype)

    elif strategy == "accelerate":
        if not _ACCELERATE_AVAILABLE:
            raise ImportError(
                "AccelerateInference requires accelerate. "
                "Install with: pip install accelerate\n"
                "Or install with accelerate extras: pip install -e '.[accelerate]'"
            )
        return AccelerateInference()

    elif strategy == "tta":
        # TTA inference
        tta_config = config.get("inference", {}).get("tta", {})
        augmentations = tta_config.get("augmentations", ["horizontal_flip"])
        aggregation = tta_config.get("aggregation", "mean")

        return TTAInference(augmentations=augmentations, aggregation=aggregation)

    elif strategy == "ensemble":
        # Ensemble inference
        ensemble_config = config.get("inference", {}).get("ensemble", {})
        checkpoints = ensemble_config.get("checkpoints", [])
        aggregation = ensemble_config.get("aggregation", "soft_voting")
        weights = ensemble_config.get("weights", None)

        if not checkpoints:
            raise ValueError(
                "Ensemble inference requires 'checkpoints' list in config. "
                "Example: inference.ensemble.checkpoints = ['runs/fold_0/weights/best.pt', ...]"
            )

        if device is None:
            raise ValueError("Ensemble inference requires device parameter")

        return EnsembleInference(
            checkpoints=checkpoints,
            config=config,
            device=device,
            aggregation=aggregation,
            weights=weights,
        )

    elif strategy == "tta_ensemble":
        # Combined TTA + Ensemble
        tta_config = config.get("inference", {}).get("tta", {})
        ensemble_config = config.get("inference", {}).get("ensemble", {})

        tta_augmentations = tta_config.get("augmentations", ["horizontal_flip"])
        tta_aggregation = tta_config.get("aggregation", "mean")

        checkpoints = ensemble_config.get("checkpoints", [])
        ensemble_aggregation = ensemble_config.get("aggregation", "soft_voting")
        ensemble_weights = ensemble_config.get("weights", None)

        if not checkpoints:
            raise ValueError(
                "TTA+Ensemble inference requires 'checkpoints' list in config. "
                "Example: inference.ensemble.checkpoints = ['runs/fold_0/weights/best.pt', ...]"
            )

        if device is None:
            raise ValueError("TTA+Ensemble inference requires device parameter")

        return TTAEnsembleInference(
            checkpoints=checkpoints,
            config=config,
            device=device,
            tta_augmentations=tta_augmentations,
            tta_aggregation=tta_aggregation,
            ensemble_aggregation=ensemble_aggregation,
            ensemble_weights=ensemble_weights,
        )

    else:
        available = "'standard', 'mixed_precision', 'tta', 'ensemble', 'tta_ensemble'"
        if _ACCELERATE_AVAILABLE:
            available += ", 'accelerate'"

        raise ValueError(
            f"Unknown inference strategy: '{strategy}'. " f"Available options: {available}"
        )


__all__ = [
    "BaseInferenceStrategy",
    "StandardInference",
    "MixedPrecisionInference",
    "TTAInference",
    "EnsembleInference",
    "TTAEnsembleInference",
    "get_inference_strategy",
]

# Conditionally add AccelerateInference to __all__ if available
if _ACCELERATE_AVAILABLE:
    __all__.append("AccelerateInference")
