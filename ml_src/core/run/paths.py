"""Run directory path utilities."""

import os


def get_run_dir_from_checkpoint(checkpoint_path):
    """
    Extract run directory from checkpoint path.

    Expected format: runs/run_name/weights/checkpoint.pt

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        str: Run directory path (e.g., 'runs/run_name')
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    # Go up 2 levels: checkpoint.pt -> weights -> run_dir
    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return run_dir
