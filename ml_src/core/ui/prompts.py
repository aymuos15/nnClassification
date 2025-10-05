"""Interactive prompt utilities."""


def prompt_user_settings():
    """
    Prompt user for configuration settings interactively.

    Returns:
        dict: User-selected settings (architecture, batch_size, num_epochs, lr, num_folds)
    """
    print("\n" + "=" * 60)
    print("Configuration Settings")
    print("=" * 60 + "\n")

    # Architecture
    print("Model Architecture:")
    print("  Popular choices: resnet18, resnet50, efficientnet_b0, mobilenet_v2, vit_b_16")
    architecture = input("  Architecture [resnet18]: ").strip() or "resnet18"

    # Batch size
    batch_size = input("  Batch size [4]: ").strip()
    batch_size = int(batch_size) if batch_size else 4

    # Epochs
    num_epochs = input("  Number of epochs [25]: ").strip()
    num_epochs = int(num_epochs) if num_epochs else 25

    # Learning rate
    lr = input("  Learning rate [0.001]: ").strip()
    lr = float(lr) if lr else 0.001

    # Number of folds
    num_folds = input("  Number of CV folds (for splitting) [5]: ").strip()
    num_folds = int(num_folds) if num_folds else 5

    return {
        "architecture": architecture,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "num_folds": num_folds,
    }
