"""Configuration creation utilities."""

import yaml


def create_config(
    dataset_info,
    template_path,
    architecture="resnet18",
    batch_size=4,
    num_epochs=25,
    lr=0.001,
    num_folds=5,
):
    """
    Create configuration from template with dataset-specific values.

    Args:
        dataset_info: Dictionary with dataset information (dataset_name, num_classes, etc.)
        template_path: Path to config template file
        architecture: Model architecture name
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        num_folds: Number of cross-validation folds

    Returns:
        dict: Configuration dictionary
    """
    # Load template
    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Update with dataset-specific values
    config["data"]["dataset_name"] = dataset_info["dataset_name"]
    config["data"]["data_dir"] = dataset_info["data_dir"]
    config["data"]["fold"] = 0  # Default to fold 0

    config["model"]["num_classes"] = dataset_info["num_classes"]
    config["model"]["architecture"] = architecture

    config["training"]["batch_size"] = batch_size
    config["training"]["num_epochs"] = num_epochs

    config["optimizer"]["lr"] = lr

    return config
