"""Data loader module for creating PyTorch DataLoaders."""

import torch
import torch.utils.data

from ml_src.seeding import seed_worker


def get_dataloaders(datasets, config):
    """
    Create DataLoaders for train, val, and test datasets.

    Args:
        datasets: Dictionary of datasets for train, val, and test
        config: Configuration dictionary with data loading settings

    Returns:
        Dictionary of DataLoaders for train, val, and test splits
    """
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    seed = config.get("seed", 42)

    # Create generator for reproducible DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    dataloaders = {}
    for split in ["train", "val", "test"]:
        # Shuffle training data, but not validation or test data
        shuffle = split == "train"

        dataloaders[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return dataloaders


def get_dataset_sizes(datasets):
    """
    Get the size of each dataset split.

    Args:
        datasets: Dictionary of datasets

    Returns:
        Dictionary with sizes for train, val, and test splits
    """
    return {split: len(datasets[split]) for split in ["train", "val", "test"]}
