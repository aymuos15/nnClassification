"""Inference factory for creating different inference strategies."""

import torch

from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.inference.mixed_precision import MixedPrecisionInference
from ml_src.core.inference.standard import StandardInference

# Conditional import for AccelerateInference (requires accelerate)
try:
    from ml_src.core.inference.accelerate import AccelerateInference

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


def get_inference_strategy(config):
    """
    Factory function to create appropriate inference strategy based on configuration.

    This factory enables different inference strategies:
    - 'standard': Traditional PyTorch inference with manual device management (default)
    - 'mixed_precision': Inference with automatic mixed precision (AMP) for speed
    - 'accelerate': Inference with HuggingFace Accelerate for multi-GPU/distributed

    Args:
        config: Configuration dictionary containing inference settings

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
        >>> # Mixed precision inference (2-3x faster on GPU)
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'mixed_precision',
        ...         'amp_dtype': 'float16'
        ...     }
        ... }
        >>> strategy = get_inference_strategy(config)
        >>> acc, results = strategy.run_inference(model, loader, size, device)
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
    else:
        raise ValueError(
            f"Unknown inference strategy: '{strategy}'. "
            f"Available options: 'standard', 'mixed_precision', 'accelerate'. "
            f"Note: 'accelerate' requires accelerate (pip install accelerate)."
        )


__all__ = [
    "BaseInferenceStrategy",
    "StandardInference",
    "MixedPrecisionInference",
    "get_inference_strategy",
]

# Conditionally add AccelerateInference to __all__ if available
if _ACCELERATE_AVAILABLE:
    __all__.append("AccelerateInference")
