"""Custom federated learning strategies and utilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from flwr.common import Metrics
except ImportError:
    raise ImportError(
        "Flower is required for federated learning. Install with:\n"
        "  uv pip install -e '.[flower]'\n"
        "Or:\n"
        "  pip install flwr>=1.7.0"
    )


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute weighted average of metrics across clients.

    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients

    Returns:
        Dictionary containing weighted average of all metrics

    Example:
        >>> metrics = [
        ...     (100, {"accuracy": 0.8, "loss": 0.5}),
        ...     (200, {"accuracy": 0.9, "loss": 0.3})
        ... ]
        >>> result = weighted_average(metrics)
        >>> print(result)
        {'accuracy': 0.8666..., 'loss': 0.3666...}
    """
    # Get all metric names from first client
    if not metrics:
        return {}

    metric_names = metrics[0][1].keys()
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Compute weighted average for each metric
    aggregated = {}
    for metric_name in metric_names:
        weighted_sum = sum(
            num_examples * m.get(metric_name, 0.0) for num_examples, m in metrics
        )
        aggregated[metric_name] = weighted_sum / total_examples

    return aggregated


def compute_data_distribution(
    client_data_sizes: Dict[int, int]
) -> Tuple[Dict[int, float], float, float]:
    """
    Compute statistics about data distribution across clients.

    Args:
        client_data_sizes: Dictionary mapping client_id to number of samples

    Returns:
        Tuple of (data_fractions, mean_size, std_size) where:
            - data_fractions: Dict mapping client_id to fraction of total data
            - mean_size: Mean number of samples per client
            - std_size: Standard deviation of samples per client

    Example:
        >>> sizes = {0: 100, 1: 200, 2: 150}
        >>> fractions, mean, std = compute_data_distribution(sizes)
        >>> print(fractions)
        {0: 0.222..., 1: 0.444..., 2: 0.333...}
    """
    total_samples = sum(client_data_sizes.values())
    data_fractions = {
        client_id: size / total_samples for client_id, size in client_data_sizes.items()
    }

    sizes = list(client_data_sizes.values())
    mean_size = np.mean(sizes)
    std_size = np.std(sizes)

    return data_fractions, mean_size, std_size


def log_client_selection(
    client_ids: List[int],
    total_clients: int,
    round_num: int,
    strategy_name: str = "FedAvg",
) -> None:
    """
    Log information about client selection for a federated round.

    Args:
        client_ids: List of selected client IDs
        total_clients: Total number of available clients
        round_num: Current round number
        strategy_name: Name of the FL strategy being used
    """
    from loguru import logger

    num_selected = len(client_ids)
    fraction = num_selected / total_clients if total_clients > 0 else 0.0

    logger.info(
        f"[Round {round_num}] {strategy_name} selected {num_selected}/{total_clients} "
        f"clients ({fraction:.1%}): {sorted(client_ids)}"
    )


# Strategy configuration templates for common use cases
STRATEGY_TEMPLATES = {
    "basic": {
        "strategy": "FedAvg",
        "strategy_config": {
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": 2,
            "min_evaluate_clients": 2,
            "min_available_clients": 2,
        },
    },
    "production": {
        "strategy": "FedAvg",
        "strategy_config": {
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 8,
            "min_evaluate_clients": 4,
            "min_available_clients": 10,
        },
    },
    "heterogeneous": {
        "strategy": "FedProx",
        "strategy_config": {
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 8,
            "min_evaluate_clients": 4,
            "min_available_clients": 10,
            "proximal_mu": 0.01,
        },
    },
    "adaptive": {
        "strategy": "FedAdam",
        "strategy_config": {
            "fraction_fit": 0.8,
            "fraction_evaluate": 0.5,
            "min_fit_clients": 8,
            "min_evaluate_clients": 4,
            "min_available_clients": 10,
            "eta": 0.01,
            "eta_l": 0.01,
            "beta_1": 0.9,
            "beta_2": 0.99,
        },
    },
}


def get_strategy_template(template_name: str) -> Dict:
    """
    Get predefined strategy configuration template.

    Args:
        template_name: Name of template ('basic', 'production', 'heterogeneous', 'adaptive')

    Returns:
        Strategy configuration dictionary

    Raises:
        ValueError: If template_name is not found

    Example:
        >>> config = get_strategy_template('production')
        >>> print(config['strategy'])
        'FedAvg'
    """
    if template_name not in STRATEGY_TEMPLATES:
        available = ", ".join(STRATEGY_TEMPLATES.keys())
        raise ValueError(
            f"Unknown strategy template: '{template_name}'. Available: {available}"
        )

    return STRATEGY_TEMPLATES[template_name].copy()
