"""Seeding utilities for reproducibility."""

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger


def set_seed(seed, deterministic=False):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but fully reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Full determinism - slower but completely reproducible
        torch.use_deterministic_algorithms(True)
        cudnn.deterministic = True
        cudnn.benchmark = False
        logger.info(f"Seed set to {seed} with deterministic mode enabled (slower)")
    else:
        # Allow non-deterministic operations for speed
        cudnn.deterministic = False
        cudnn.benchmark = True
        logger.info(f"Seed set to {seed} (non-deterministic mode for speed)")


def seed_worker(worker_id):
    """
    Seed each DataLoader worker process.

    This ensures reproducibility when using num_workers > 0.

    Args:
        worker_id: Worker process ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
