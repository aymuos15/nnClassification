"""CLI command for running Flower federated learning (simulation or deployment mode)."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

from ml_src.core.federated import check_flower_available


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Flower federated learning in simulation or deployment mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulation mode (default) - all clients on one machine
  ml-fl-run --config configs/federated_experiment.yaml

  # Simulation mode with custom rounds
  ml-fl-run --config configs/federated_experiment.yaml --num-rounds 200

  # Deployment mode - automatically launch server and clients
  ml-fl-run --config configs/federated_deployment.yaml --mode deployment

Note:
  - Simulation mode uses Flower's built-in simulation engine
  - Deployment mode launches server and clients as separate processes
  - For distributed deployment across machines, use ml-fl-server and ml-fl-client separately
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["simulation", "deployment"],
        help="Execution mode (default: from config or 'simulation')",
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=None,
        help="Number of federated learning rounds (overrides config)",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help="Number of clients for simulation mode (overrides config)",
    )

    return parser.parse_args()


def run_simulation(config, num_rounds=None, num_clients=None):
    """Run federated learning in simulation mode using Flower's simulation engine."""
    logger.info("Starting Flower simulation mode...")

    # Check Flower availability
    check_flower_available()
    import flwr as fl

    from ml_src.core.federated.client import create_client_fn
    from ml_src.core.federated.server import get_strategy

    # Get configuration values
    if num_rounds is None:
        num_rounds = config.get("federated", {}).get("server", {}).get("num_rounds", 10)

    if num_clients is None:
        num_clients = config.get("federated", {}).get("clients", {}).get("num_clients", 2)

    logger.info(f"Simulation parameters: {num_rounds} rounds, {num_clients} clients")

    # Create client factory
    client_fn = create_client_fn(config, run_base_dir="runs/simulation")

    # Get strategy
    strategy = get_strategy(config)

    # Configure simulation resources
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 0.0,  # Share GPUs across clients
    }

    # Check if GPU is available and update resources
    import torch

    if torch.cuda.is_available():
        num_gpus_available = torch.cuda.device_count()
        # Distribute GPU resources across clients
        client_resources["num_gpus"] = num_gpus_available / num_clients
        logger.info(
            f"GPU resources: {num_gpus_available} GPUs, "
            f"{client_resources['num_gpus']:.2f} per client"
        )
    else:
        logger.info("No GPU available, running on CPU")

    logger.info("=" * 70)
    logger.info("Starting Flower Simulation")
    logger.info("=" * 70)

    # Run simulation
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_resources,
        )

        logger.success("Simulation complete!")
        logger.info(f"History: {history}")

        return history

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_deployment(config, num_rounds=None):
    """Run federated learning in deployment mode (server + multiple clients as processes)."""
    logger.info("Starting Flower deployment mode...")

    # Get server configuration
    server_address = (
        config.get("federated", {}).get("server", {}).get("address", "localhost:8080")
    )

    if num_rounds is None:
        num_rounds = config.get("federated", {}).get("server", {}).get("num_rounds", 10)

    # Get client manifest
    manifest = config.get("federated", {}).get("clients", {}).get("manifest", [])

    if not manifest:
        logger.error("No client manifest found in config. Cannot run deployment mode.")
        logger.error("Add 'federated.clients.manifest' to config or use simulation mode.")
        sys.exit(1)

    num_clients = len(manifest)
    logger.info(f"Deployment parameters: {num_rounds} rounds, {num_clients} clients")

    # Get config file path (absolute)
    config_path = Path(sys.argv[sys.argv.index("--config") + 1]).absolute()

    logger.info("=" * 70)
    logger.info("Starting Flower Deployment")
    logger.info("=" * 70)
    logger.info(f"Server: {server_address}")
    logger.info(f"Clients: {num_clients}")
    logger.info("=" * 70)

    processes = []

    try:
        # Start server process
        logger.info("Starting server process...")
        server_cmd = [
            sys.executable,
            "-m",
            "ml_src.cli.fl_server",
            "--config",
            str(config_path),
            "--server-address",
            server_address,
            "--num-rounds",
            str(num_rounds),
        ]

        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(("server", server_process))

        # Wait for server to start
        time.sleep(5)
        logger.success("Server started")

        # Start client processes
        for client_entry in manifest:
            client_id = client_entry["id"]
            logger.info(f"Starting client {client_id} process...")

            client_cmd = [
                sys.executable,
                "-m",
                "ml_src.cli.fl_client",
                "--config",
                str(config_path),
                "--client-id",
                str(client_id),
                "--server-address",
                server_address,
            ]

            client_process = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append((f"client_{client_id}", client_process))

            time.sleep(1)  # Stagger client starts

        logger.success(f"All {num_clients} clients started")
        logger.info("Waiting for training to complete...")

        # Monitor processes
        while True:
            # Check if server is still running
            if server_process.poll() is not None:
                logger.info("Server process completed")
                break

            time.sleep(5)

        # Wait for all processes to complete
        logger.info("Waiting for all processes to finish...")
        for name, process in processes:
            process.wait()
            logger.info(f"{name} completed with code {process.returncode}")

        logger.success("Deployment complete!")

    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
        # Terminate all processes
        for name, process in processes:
            logger.info(f"Terminating {name}...")
            process.terminate()
        sys.exit(0)

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        # Terminate all processes
        for name, process in processes:
            process.terminate()
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for FL orchestration."""
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

    # Determine mode
    mode = args.mode
    if mode is None:
        mode = config.get("federated", {}).get("mode", "simulation")

    logger.info(f"Running in {mode} mode")

    # Run appropriate mode
    if mode == "simulation":
        run_simulation(config, num_rounds=args.num_rounds, num_clients=args.num_clients)
    elif mode == "deployment":
        run_deployment(config, num_rounds=args.num_rounds)
    else:
        logger.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
