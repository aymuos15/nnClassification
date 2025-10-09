"""Federated learning module for distributed training with Flower."""

from typing import TYPE_CHECKING

# Conditional imports for Flower (optional dependency)
try:
    import flwr as fl

    _FLOWER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency branch
    _FLOWER_AVAILABLE = False

if _FLOWER_AVAILABLE:
    from ml_src.core.federated.client import FlowerClient, create_client_fn
    from ml_src.core.federated.server import create_server_fn, get_strategy
elif TYPE_CHECKING:  # pragma: no cover - hinting only
    from ml_src.core.federated.client import FlowerClient  # type: ignore
    from ml_src.core.federated.server import create_server_fn, get_strategy  # type: ignore
else:
    FlowerClient = None  # type: ignore[assignment]
    create_client_fn = None  # type: ignore[assignment]
    create_server_fn = None  # type: ignore[assignment]
    get_strategy = None  # type: ignore[assignment]


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
