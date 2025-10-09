"""CLI command for starting Flower federated learning server."""

import argparse
import sys

import yaml
from loguru import logger

from ml_src.core.federated import check_flower_available
from ml_src.core.federated.server import start_server


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start Flower federated learning server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default config
  ml-fl-server --config configs/federated_experiment.yaml

  # Start server with custom address and rounds
  ml-fl-server --config configs/federated_experiment.yaml \\
    --server-address 0.0.0.0:9000 --num-rounds 200

  # Start server for deployment mode
  ml-fl-server --config configs/federated_deployment.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Server address in format 'host:port' (default: from config or '0.0.0.0:8080')",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of federated learning rounds (overrides config)",
    )

    return parser.parse_args()


def main():
    """Main entry point for FL server."""
    args = parse_args()

    # Check if Flower is installed
    try:
        check_flower_available()
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Get server address from args, config, or default
    server_address = args.server_address
    if server_address is None:
        server_address = (
            config.get("federated", {}).get("server", {}).get("address", "0.0.0.0:8080")
        )

    # Display configuration
    logger.info("=" * 70)
    logger.info("Flower Federated Learning Server")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Server address: {server_address}")

    strategy_name = config.get("federated", {}).get("server", {}).get("strategy", "FedAvg")
    logger.info(f"FL Strategy: {strategy_name}")

    num_rounds = args.num_rounds or config.get("federated", {}).get("server", {}).get(
        "num_rounds", 10
    )
    logger.info(f"Number of rounds: {num_rounds}")

    logger.info("=" * 70)

    # Start server
    try:
        start_server(
            config=config,
            server_address=server_address,
            num_rounds=num_rounds,
        )
    except KeyboardInterrupt:
        logger.info("\nServer interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
