"""Dataset module for handling image datasets."""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IndexedImageDataset(Dataset):
    """
    Dataset that loads images from index file paths.

    This dataset reads image paths from a text file (one path per line)
    and loads them dynamically. Useful for cross-validation without
    duplicating data.

    Args:
        index_file: Path to text file containing image paths (one per line)
        data_root: Root directory to prepend to relative paths
        transform: Optional transform to apply to images
    """

    def __init__(self, index_file, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        # Read index file
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")

        with open(index_file) as f:
            paths = [line.strip() for line in f if line.strip()]

        # Build class mapping and samples list efficiently
        # Expected format: raw/class_name/image.jpg
        class_names = set()
        data_root_parent = str(self.data_root.parent)  # Pre-compute parent path

        temp_samples = []
        for path in paths:
            # Extract class name efficiently (assume format: raw/class_name/image.jpg)
            parts = path.split("/")
            if len(parts) >= 2:
                class_name = parts[-2]  # Parent directory is class name
                class_names.add(class_name)
                # Store path and class_name for later
                temp_samples.append((path, class_name))

        # Create sorted class list and mapping
        self.classes = sorted(class_names)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Build final samples list with class indices
        for path, class_name in temp_samples:
            class_idx = self.class_to_idx[class_name]
            # Construct absolute path efficiently
            absolute_path = os.path.join(data_root_parent, path)
            self.samples.append((absolute_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return image and label at index.

        Args:
            idx: Index of sample to load

        Returns:
            Tuple of (image_tensor, label_idx)
        """
        img_path, label = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}") from e

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config):
    """
    Create data transforms based on configuration.

    Args:
        config: Configuration dictionary with transform settings

    Returns:
        Dictionary of transforms for train, val, and test splits
    """
    data_transforms = {}

    for split in ["train", "val", "test"]:
        transform_config = config["transforms"][split]
        transform_list = []

        # Resize
        if "resize" in transform_config:
            resize_size = tuple(transform_config["resize"])
            transform_list.append(transforms.Resize(resize_size))

        # Random horizontal flip (only for training)
        if split == "train" and transform_config.get("random_horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalization
        if "normalize" in transform_config:
            mean = transform_config["normalize"]["mean"]
            std = transform_config["normalize"]["std"]
            transform_list.append(transforms.Normalize(mean, std))

        data_transforms[split] = transforms.Compose(transform_list)

    return data_transforms


def get_datasets(config):
    """
    Create image datasets for train, val, and test splits using index-based CV splits.

    Loads datasets from index files that reference images in data_dir/raw/.
    Requires data_dir/splits/ to contain:
    - test.txt (single test set, same for all folds)
    - fold_{N}_train.txt and fold_{N}_val.txt (per-fold train/val splits)

    Args:
        config: Configuration dictionary with data settings

    Returns:
        Dictionary of datasets for train, val, and test splits
    """
    data_transforms = get_transforms(config)

    # Load from index files (CV mode is mandatory)
    # Assumes data_dir/raw/ contains images and data_dir/splits/ contains index files
    fold = config["data"].get("fold", 0)
    data_dir = config["data"]["data_dir"]

    # Hardcoded subdirectories (always 'raw' and 'splits')
    data_root = os.path.join(data_dir, "raw")
    splits_dir = os.path.join(data_dir, "splits")

    image_datasets = {}

    # Train and val are fold-specific
    for split in ["train", "val"]:
        index_file = os.path.join(splits_dir, f"fold_{fold}_{split}.txt")
        image_datasets[split] = IndexedImageDataset(
            index_file=index_file, data_root=data_root, transform=data_transforms[split]
        )

    # Test is the SAME for all folds (no fold suffix)
    test_index_file = os.path.join(splits_dir, "test.txt")
    image_datasets["test"] = IndexedImageDataset(
        index_file=test_index_file, data_root=data_root, transform=data_transforms["test"]
    )

    return image_datasets


def get_class_names(datasets_dict):
    """
    Get class names from the training dataset.

    Args:
        datasets_dict: Dictionary of datasets

    Returns:
        List of class names
    """
    return datasets_dict["train"].classes
