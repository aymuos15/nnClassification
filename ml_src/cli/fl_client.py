"""CLI command for starting Flower federated learning client."""

import argparse
import sys

import yaml
from loguru import logger

from ml_src.core.federated import FlowerClient, check_flower_available


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start Flower federated learning client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start client 0 with default config
  ml-fl-client --config configs/federated_experiment.yaml --client-id 0

  # Start client with custom server address
  ml-fl-client --config configs/federated_experiment.yaml --client-id 1 \\
    --server-address localhost:9000

  # Start client with specific trainer type (overrides config)
  ml-fl-client --config configs/federated_experiment.yaml --client-id 2 \\
    --trainer-type mixed_precision
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="Unique client identifier (0-indexed)",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        help="Server address in format 'host:port' (default: from config or 'localhost:8080')",
    )

    parser.add_argument(
        "--trainer-type",
        type=str,
        default=None,
        choices=["standard", "mixed_precision", "accelerate", "dp"],
        help="Override trainer type from config",
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Directory for client logs and checkpoints (default: runs/fl_client_N)",
    )

    return parser.parse_args()


def main():
    """Main entry point for FL client."""
    args = parse_args()

    # Check if Flower is installed
    try:
        check_flower_available()
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)

    # Import Flower after availability check
    import flwr as fl

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

    # Apply CLI overrides
    if args.trainer_type is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["trainer_type"] = args.trainer_type
        logger.info(f"Overriding trainer type to: {args.trainer_type}")

    # Get server address from args, config, or default
    server_address = args.server_address
    if server_address is None:
        server_address = (
            config.get("federated", {}).get("server", {}).get("address", "localhost:8080")
        )

    # Get run directory
    run_dir = args.run_dir or f"runs/fl_client_{args.client_id}"

    # Display configuration
    logger.info("=" * 70)
    logger.info(f"Flower Federated Learning Client {args.client_id}")
    logger.info("=" * 70)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Client ID: {args.client_id}")
    logger.info(f"Server address: {server_address}")
    logger.info(f"Run directory: {run_dir}")

    trainer_type = config.get("training", {}).get("trainer_type", "standard")
    logger.info(f"Trainer type: {trainer_type}")

    logger.info("=" * 70)

    # Create client
    try:
        logger.info("Initializing Flower client...")
        client = FlowerClient(config=config, client_id=args.client_id, run_dir=run_dir)

        logger.info(f"Connecting to server at {server_address}...")
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client(),
        )

        logger.success("Client completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nClient interrupted by user")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.error(
            "Did you run: ml-split --raw_data <path> --federated --num-clients <N> ?"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Client error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
