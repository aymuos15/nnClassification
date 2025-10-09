"""Data partitioning utilities for federated learning."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger


def partition_data_iid(
    file_paths: List[str],
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Partition data uniformly (IID) across clients.

    Args:
        file_paths: List of all file paths to partition
        num_clients: Number of clients
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping client_id to list of file paths

    Example:
        >>> paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
        >>> partitions = partition_data_iid(paths, num_clients=2)
        >>> print(len(partitions[0]), len(partitions[1]))
        2 2
    """
    np.random.seed(seed)

    # Shuffle file paths
    shuffled_paths = file_paths.copy()
    np.random.shuffle(shuffled_paths)

    # Partition into equal-sized chunks
    partitions = {}
    chunk_size = len(shuffled_paths) // num_clients

    for i in range(num_clients):
        start_idx = i * chunk_size
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = len(shuffled_paths)
        else:
            end_idx = (i + 1) * chunk_size

        partitions[i] = shuffled_paths[start_idx:end_idx]

    # Log partition statistics
    logger.info("IID partitioning complete:")
    for client_id, paths in partitions.items():
        logger.info(f"  Client {client_id}: {len(paths)} samples")

    return partitions


def partition_data_non_iid_dirichlet(
    file_paths_by_class: Dict[str, List[str]],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Partition data non-IID using Dirichlet distribution.

    This creates realistic heterogeneous data distributions across clients,
    where each client has a different class distribution. Lower alpha values
    create more heterogeneity.

    Args:
        file_paths_by_class: Dictionary mapping class_name to list of file paths
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
                Typical values: 0.1 (very heterogeneous), 0.5 (moderately), 10.0 (nearly IID)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping client_id to list of file paths

    Example:
        >>> paths_by_class = {
        ...     'cat': ['cat1.jpg', 'cat2.jpg', 'cat3.jpg'],
        ...     'dog': ['dog1.jpg', 'dog2.jpg', 'dog3.jpg']
        ... }
        >>> partitions = partition_data_non_iid_dirichlet(
        ...     paths_by_class, num_clients=2, alpha=0.5
        ... )
    """
    np.random.seed(seed)

    class_names = list(file_paths_by_class.keys())
    num_classes = len(class_names)

    # Initialize client partitions
    client_partitions = {i: [] for i in range(num_clients)}
    client_class_counts = {i: {cls: 0 for cls in class_names} for i in range(num_clients)}

    # For each class, partition samples using Dirichlet distribution
    for class_name in class_names:
        class_paths = file_paths_by_class[class_name]
        num_samples = len(class_paths)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Convert proportions to sample counts
        sample_counts = (proportions * num_samples).astype(int)

        # Adjust to ensure all samples are assigned
        diff = num_samples - sample_counts.sum()
        if diff > 0:
            # Add remaining samples to random clients
            indices = np.random.choice(num_clients, size=diff, replace=True)
            for idx in indices:
                sample_counts[idx] += 1

        # Shuffle class paths and assign to clients
        shuffled_paths = class_paths.copy()
        np.random.shuffle(shuffled_paths)

        start_idx = 0
        for client_id in range(num_clients):
            count = sample_counts[client_id]
            end_idx = start_idx + count

            client_partitions[client_id].extend(shuffled_paths[start_idx:end_idx])
            client_class_counts[client_id][class_name] = count

            start_idx = end_idx

    # Shuffle each client's data
    for client_id in client_partitions:
        np.random.shuffle(client_partitions[client_id])

    # Log partition statistics
    logger.info(f"Non-IID Dirichlet partitioning complete (alpha={alpha}):")
    for client_id in range(num_clients):
        total_samples = len(client_partitions[client_id])
        class_dist = client_class_counts[client_id]
        logger.info(f"  Client {client_id}: {total_samples} samples, dist={class_dist}")

    return client_partitions


def partition_data_label_skew(
    file_paths_by_class: Dict[str, List[str]],
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Partition data with label skew (each client sees only subset of classes).

    Args:
        file_paths_by_class: Dictionary mapping class_name to list of file paths
        num_clients: Number of clients
        classes_per_client: Number of classes each client has access to
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping client_id to list of file paths

    Example:
        >>> paths_by_class = {
        ...     'cat': ['cat1.jpg', 'cat2.jpg'],
        ...     'dog': ['dog1.jpg', 'dog2.jpg'],
        ...     'bird': ['bird1.jpg', 'bird2.jpg']
        ... }
        >>> partitions = partition_data_label_skew(
        ...     paths_by_class, num_clients=3, classes_per_client=2
        ... )
    """
    np.random.seed(seed)

    class_names = list(file_paths_by_class.keys())
    num_classes = len(class_names)

    if classes_per_client > num_classes:
        logger.warning(
            f"classes_per_client ({classes_per_client}) > num_classes ({num_classes}). "
            f"Setting classes_per_client={num_classes}"
        )
        classes_per_client = num_classes

    # Assign classes to clients
    client_classes = {}
    for client_id in range(num_clients):
        client_classes[client_id] = list(
            np.random.choice(class_names, size=classes_per_client, replace=False)
        )

    # Partition data for each class
    class_partitions = {}
    for class_name in class_names:
        # Count how many clients have access to this class
        clients_with_class = [
            cid for cid in range(num_clients) if class_name in client_classes[cid]
        ]
        num_clients_with_class = len(clients_with_class)

        if num_clients_with_class == 0:
            logger.warning(f"Class '{class_name}' not assigned to any client!")
            continue

        # Partition this class's data among clients that have it
        class_paths = file_paths_by_class[class_name]
        np.random.shuffle(class_paths)

        chunk_size = len(class_paths) // num_clients_with_class
        class_partitions[class_name] = {}

        for idx, client_id in enumerate(clients_with_class):
            start_idx = idx * chunk_size
            if idx == num_clients_with_class - 1:
                end_idx = len(class_paths)
            else:
                end_idx = (idx + 1) * chunk_size

            class_partitions[class_name][client_id] = class_paths[start_idx:end_idx]

    # Combine partitions by client
    client_partitions = {i: [] for i in range(num_clients)}
    client_class_counts = {i: {} for i in range(num_clients)}

    for class_name, client_data in class_partitions.items():
        for client_id, paths in client_data.items():
            client_partitions[client_id].extend(paths)
            client_class_counts[client_id][class_name] = len(paths)

    # Shuffle each client's data
    for client_id in client_partitions:
        np.random.shuffle(client_partitions[client_id])

    # Log partition statistics
    logger.info(f"Label skew partitioning complete (classes_per_client={classes_per_client}):")
    for client_id in range(num_clients):
        total_samples = len(client_partitions[client_id])
        classes = list(client_class_counts[client_id].keys())
        logger.info(f"  Client {client_id}: {total_samples} samples, classes={classes}")

    return client_partitions


def create_federated_splits(
    raw_data_dir: str,
    splits_dir: str,
    num_clients: int,
    partition_strategy: str = "iid",
    alpha: float = 0.5,
    classes_per_client: int = 2,
    train_val_split: float = 0.8,
    test_split: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Create federated learning data splits and save as index files.

    This function partitions data across clients and creates train/val/test splits.
    Output format matches existing ml-split functionality.

    Args:
        raw_data_dir: Directory containing raw data in class folders
        splits_dir: Directory to save split index files
        num_clients: Number of federated clients
        partition_strategy: Strategy for partitioning ('iid', 'non-iid', 'label-skew')
        alpha: Dirichlet alpha parameter (for non-iid strategy)
        classes_per_client: Number of classes per client (for label-skew strategy)
        train_val_split: Fraction of client data to use for training
        test_split: Fraction of total data to reserve for global test set
        seed: Random seed for reproducibility

    Example:
        >>> create_federated_splits(
        ...     raw_data_dir='data/my_dataset/raw',
        ...     splits_dir='data/my_dataset/splits',
        ...     num_clients=10,
        ...     partition_strategy='non-iid',
        ...     alpha=0.5
        ... )
    """
    raw_data_path = Path(raw_data_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating federated splits for {num_clients} clients")
    logger.info(f"Strategy: {partition_strategy}")

    # Collect all file paths by class
    file_paths_by_class = {}
    all_file_paths = []

    for class_dir in sorted(raw_data_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        file_paths = []

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                # Store relative path from data_dir parent
                rel_path = os.path.join("raw", class_name, img_path.name)
                file_paths.append(rel_path)

        if file_paths:
            file_paths_by_class[class_name] = file_paths
            all_file_paths.extend(file_paths)

    logger.info(f"Found {len(all_file_paths)} images across {len(file_paths_by_class)} classes")

    # Create global test set (shared across all clients)
    np.random.seed(seed)
    np.random.shuffle(all_file_paths)

    num_test = int(len(all_file_paths) * test_split)
    test_paths = all_file_paths[:num_test]
    remaining_paths = all_file_paths[num_test:]

    logger.info(f"Global test set: {num_test} samples")

    # Reorganize remaining paths by class for partitioning
    remaining_by_class = {cls: [] for cls in file_paths_by_class.keys()}
    for path in remaining_paths:
        class_name = path.split("/")[1]  # Extract class from 'raw/class/image.jpg'
        remaining_by_class[class_name].append(path)

    # Partition data according to strategy
    if partition_strategy == "iid":
        client_partitions = partition_data_iid(remaining_paths, num_clients, seed)

    elif partition_strategy == "non-iid":
        client_partitions = partition_data_non_iid_dirichlet(
            remaining_by_class, num_clients, alpha, seed
        )

    elif partition_strategy == "label-skew":
        client_partitions = partition_data_label_skew(
            remaining_by_class, num_clients, classes_per_client, seed
        )

    else:
        raise ValueError(
            f"Unknown partition_strategy: '{partition_strategy}'. "
            f"Available: 'iid', 'non-iid', 'label-skew'"
        )

    # Split each client's data into train/val
    for client_id, client_paths in client_partitions.items():
        num_train = int(len(client_paths) * train_val_split)

        train_paths = client_paths[:num_train]
        val_paths = client_paths[num_train:]

        # Write train index file
        train_file = splits_path / f"client_{client_id}_train.txt"
        with open(train_file, "w") as f:
            f.write("\n".join(train_paths))

        # Write val index file
        val_file = splits_path / f"client_{client_id}_val.txt"
        with open(val_file, "w") as f:
            f.write("\n".join(val_paths))

        logger.info(
            f"Client {client_id}: {len(train_paths)} train, {len(val_paths)} val samples"
        )

    # Write shared test set
    test_file = splits_path / "test.txt"
    with open(test_file, "w") as f:
        f.write("\n".join(test_paths))

    logger.success(f"Federated splits created in {splits_dir}")
    logger.info(f"  - {num_clients} clients (client_N_train.txt, client_N_val.txt)")
    logger.info(f"  - 1 shared test set (test.txt)")
