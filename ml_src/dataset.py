"""Dataset module for handling image datasets."""

import os
from torchvision import datasets, transforms


def get_transforms(config):
    """
    Create data transforms based on configuration.

    Args:
        config: Configuration dictionary with transform settings

    Returns:
        Dictionary of transforms for train, val, and test splits
    """
    data_transforms = {}

    for split in ['train', 'val', 'test']:
        transform_config = config['transforms'][split]
        transform_list = []

        # Resize
        if 'resize' in transform_config:
            resize_size = tuple(transform_config['resize'])
            transform_list.append(transforms.Resize(resize_size))

        # Random horizontal flip (only for training)
        if split == 'train' and transform_config.get('random_horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalization
        if 'normalize' in transform_config:
            mean = transform_config['normalize']['mean']
            std = transform_config['normalize']['std']
            transform_list.append(transforms.Normalize(mean, std))

        data_transforms[split] = transforms.Compose(transform_list)

    return data_transforms


def get_datasets(config):
    """
    Create image datasets for train, val, and test splits.

    Args:
        config: Configuration dictionary with data settings

    Returns:
        Dictionary of datasets for train, val, and test splits
    """
    data_dir = config['data']['data_dir']
    data_transforms = get_transforms(config)

    image_datasets = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        image_datasets[split] = datasets.ImageFolder(
            split_dir,
            data_transforms[split]
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
    return datasets_dict['train'].classes
