"""TensorBoard server management utilities."""

import os
import shutil
import subprocess

from loguru import logger


def launch_tensorboard(run_dir, port=6006):
    """
    Launch TensorBoard server.

    Args:
        run_dir: Run directory path
        port: Port number for TensorBoard server
    """
    tensorboard_dir = os.path.join(run_dir, "tensorboard")

    if not os.path.exists(tensorboard_dir):
        logger.warning(f"TensorBoard directory not found: {tensorboard_dir}")
        logger.info("Run training or visualization first to generate TensorBoard logs")
        return

    logger.info(f"Launching TensorBoard on port {port}...")
    logger.info(f"TensorBoard directory: {tensorboard_dir}")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info("Press Ctrl+C to stop TensorBoard")

    try:
        subprocess.run(["tensorboard", "--logdir", tensorboard_dir, "--port", str(port)])
    except KeyboardInterrupt:
        logger.info("\nTensorBoard stopped")
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install with: pip install tensorboard")


def clean_tensorboard_logs(run_dir=None):
    """
    Clean TensorBoard logs.

    Args:
        run_dir: Specific run directory to clean, or None to clean all runs
    """
    if run_dir:
        # Clean specific run directory
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        if os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
            logger.success(f"Removed TensorBoard logs from {run_dir}")
        else:
            logger.warning(f"No TensorBoard logs found in {run_dir}")
    else:
        # Clean all runs
        if not os.path.exists("runs"):
            logger.warning("No runs directory found")
            return

        cleaned_count = 0
        for run_name in os.listdir("runs"):
            run_path = os.path.join("runs", run_name)
            if os.path.isdir(run_path):
                tensorboard_dir = os.path.join(run_path, "tensorboard")
                if os.path.exists(tensorboard_dir):
                    shutil.rmtree(tensorboard_dir)
                    logger.info(f"Removed TensorBoard logs from {run_path}")
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.success(f"Cleaned TensorBoard logs from {cleaned_count} run(s)")
        else:
            logger.warning("No TensorBoard logs found in any runs")
