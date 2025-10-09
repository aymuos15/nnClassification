"""Flower server implementation with custom strategies."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from loguru import logger

try:
    import flwr as fl
    from flwr.common import Metrics
    from flwr.server.strategy import FedAvg, FedAdam, FedAdagrad, FedProx, Strategy
except ImportError:
    raise ImportError(
        "Flower is required for federated learning. Install with:\n"
        "  uv pip install -e '.[flower]'\n"
        "Or:\n"
        "  pip install flwr>=1.7.0"
    )


def get_strategy(config: Dict) -> Strategy:
    """
    Create Flower strategy based on configuration.

    Supports various federated learning strategies:
    - FedAvg: Federated Averaging (default, McMahan et al. 2017)
    - FedProx: Federated Proximal (Li et al. 2020) - handles heterogeneous clients
    - FedAdam: Federated Adam optimizer (Reddi et al. 2020)
    - FedAdagrad: Federated Adagrad optimizer (Reddi et al. 2020)

    Args:
        config: Configuration dictionary with 'federated.server' section

    Returns:
        Flower Strategy instance

    Example:
        >>> config = {
        ...     'federated': {
        ...         'server': {
        ...             'strategy': 'FedAvg',
        ...             'num_rounds': 100,
        ...             'strategy_config': {
        ...                 'fraction_fit': 0.8,
        ...                 'fraction_evaluate': 0.5,
        ...                 'min_fit_clients': 8,
        ...                 'min_evaluate_clients': 4,
        ...                 'min_available_clients': 10
        ...             }
        ...         }
        ...     }
        ... }
        >>> strategy = get_strategy(config)
    """
    server_config = config.get("federated", {}).get("server", {})
    strategy_name = server_config.get("strategy", "FedAvg")
    strategy_config = server_config.get("strategy_config", {})

    # Common strategy parameters
    fraction_fit = strategy_config.get("fraction_fit", 1.0)
    fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
    min_fit_clients = strategy_config.get("min_fit_clients", 2)
    min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)
    min_available_clients = strategy_config.get("min_available_clients", 2)

    # Custom metric aggregation functions
    def fit_metrics_aggregation_fn(metrics: list[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate training metrics from clients."""
        # Weighted average by number of examples
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}
        if total_examples > 0:
            for metric_name in ["train_loss", "train_acc", "val_loss", "val_acc"]:
                weighted_sum = sum(
                    num_examples * m.get(metric_name, 0.0) for num_examples, m in metrics
                )
                aggregated[metric_name] = weighted_sum / total_examples

        logger.info(f"[Server] Aggregated fit metrics: {aggregated}")
        return aggregated

    def evaluate_metrics_aggregation_fn(metrics: list[Tuple[int, Metrics]]) -> Metrics:
        """Aggregate evaluation metrics from clients."""
        # Weighted average by number of examples
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}
        if total_examples > 0:
            val_acc_sum = sum(num_examples * m.get("val_acc", 0.0) for num_examples, m in metrics)
            aggregated["val_acc"] = val_acc_sum / total_examples

        logger.info(f"[Server] Aggregated eval metrics: {aggregated}")
        return aggregated

    # Create strategy based on name
    if strategy_name == "FedAvg":
        logger.info("Using FedAvg strategy")
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif strategy_name == "FedProx":
        logger.info("Using FedProx strategy")
        proximal_mu = strategy_config.get("proximal_mu", 0.01)
        logger.info(f"FedProx proximal_mu: {proximal_mu}")

        strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            proximal_mu=proximal_mu,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif strategy_name == "FedAdam":
        logger.info("Using FedAdam strategy")
        eta = strategy_config.get("eta", 0.01)  # Server learning rate
        eta_l = strategy_config.get("eta_l", 0.01)  # Client learning rate
        beta_1 = strategy_config.get("beta_1", 0.9)
        beta_2 = strategy_config.get("beta_2", 0.99)
        tau = strategy_config.get("tau", 1e-9)

        logger.info(f"FedAdam parameters: eta={eta}, eta_l={eta_l}, beta_1={beta_1}, beta_2={beta_2}")

        strategy = FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    elif strategy_name == "FedAdagrad":
        logger.info("Using FedAdagrad strategy")
        eta = strategy_config.get("eta", 0.01)  # Server learning rate
        eta_l = strategy_config.get("eta_l", 0.01)  # Client learning rate
        tau = strategy_config.get("tau", 1e-9)

        logger.info(f"FedAdagrad parameters: eta={eta}, eta_l={eta_l}")

        strategy = FedAdagrad(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            eta=eta,
            eta_l=eta_l,
            tau=tau,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

    else:
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. "
            f"Available: 'FedAvg', 'FedProx', 'FedAdam', 'FedAdagrad'"
        )

    logger.success(f"Strategy created: {strategy_name}")
    logger.info(
        f"Strategy config: fraction_fit={fraction_fit}, "
        f"min_fit_clients={min_fit_clients}, min_available_clients={min_available_clients}"
    )

    return strategy


def create_server_fn(config: Dict) -> Callable[[], Tuple[Strategy, Dict]]:
    """
    Factory function to create Flower server configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Function that returns (strategy, server_config) tuple

    Example:
        >>> config = load_config('configs/federated_config.yaml')
        >>> server_fn = create_server_fn(config)
        >>> strategy, server_config = server_fn()
    """

    def server_fn() -> Tuple[Strategy, Dict]:
        """Create and return Flower server strategy and configuration."""
        strategy = get_strategy(config)

        # Server configuration
        num_rounds = config.get("federated", {}).get("server", {}).get("num_rounds", 10)

        server_config = {
            "num_rounds": num_rounds,
        }

        logger.info(f"Server configuration: {num_rounds} rounds")

        return strategy, server_config

    return server_fn


def start_server(
    config: Dict,
    server_address: str = "0.0.0.0:8080",
    num_rounds: Optional[int] = None,
) -> None:
    """
    Start Flower server for federated learning.

    Args:
        config: Configuration dictionary
        server_address: Server address in format 'host:port'
        num_rounds: Number of federated rounds (overrides config if provided)

    Example:
        >>> config = load_config('configs/federated_config.yaml')
        >>> start_server(config, server_address='0.0.0.0:8080', num_rounds=100)
    """
    logger.info(f"Starting Flower server at {server_address}")

    # Override num_rounds if provided
    if num_rounds is not None:
        if "federated" not in config:
            config["federated"] = {}
        if "server" not in config["federated"]:
            config["federated"]["server"] = {}
        config["federated"]["server"]["num_rounds"] = num_rounds

    # Get strategy
    strategy = get_strategy(config)

    # Get number of rounds
    num_rounds = config.get("federated", {}).get("server", {}).get("num_rounds", 10)

    logger.info(f"Server will run for {num_rounds} rounds")
    logger.success("Server ready and waiting for clients...")

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    logger.success("Federated training complete!")
