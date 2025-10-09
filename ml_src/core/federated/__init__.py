"""Federated learning module for distributed training with Flower."""

from ml_src.core.federated.client import FlowerClient, create_client_fn
from ml_src.core.federated.server import create_server_fn, get_strategy

# Conditional imports for Flower (optional dependency)
try:
    import flwr as fl

    _FLOWER_AVAILABLE = True
except ImportError:
    _FLOWER_AVAILABLE = False


def check_flower_available():
    """Check if Flower is installed and raise helpful error if not."""
    if not _FLOWER_AVAILABLE:
        raise ImportError(
            "Flower is required for federated learning. Install with:\n"
            "  uv pip install -e '.[flower]'\n"
            "Or:\n"
            "  pip install flwr>=1.7.0"
        )


__all__ = [
    "FlowerClient",
    "create_client_fn",
    "create_server_fn",
    "get_strategy",
    "check_flower_available",
]
