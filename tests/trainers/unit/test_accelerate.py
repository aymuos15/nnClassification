"""Tests for AccelerateTrainer."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.losses import get_criterion
from ml_src.core.trainers import get_trainer

# Check if accelerate package is available
try:
    import accelerate  # noqa

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


@pytest.mark.skipif(not _ACCELERATE_AVAILABLE, reason="Accelerate not available")
def test_accelerate_trainer_single_device():
    """Test that AccelerateTrainer loads and runs 1 epoch in single-device mode."""
    # Create config with accelerate trainer
    config = {
        "training": {
            "trainer_type": "accelerate",
            "num_epochs": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
        },
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset (ensure both classes are represented)
    num_samples = 8
    X_train = torch.randn(num_samples, 10)
    y_train = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    X_val = torch.randn(num_samples, 10)
    y_val = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": num_samples, "val": num_samples}

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"]
    )

    # Create criterion
    criterion = get_criterion()

    # Use temporary directory for run_dir
    with tempfile.TemporaryDirectory() as run_dir:
        # Create required directories
        import os

        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Create trainer via factory
        trainer = get_trainer(
            config=config,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        # Verify it's an AccelerateTrainer
        from ml_src.core.trainers import AccelerateTrainer

        assert isinstance(trainer, AccelerateTrainer)

        # Run 1 epoch of training
        trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()

        # Verify no errors occurred
        assert trained_model is not None
        assert len(train_losses) == 1  # 1 epoch
        assert len(val_losses) == 1
        assert len(train_accs) == 1
        assert len(val_accs) == 1

        # Verify metrics are valid numbers
        assert isinstance(train_losses[0], float)
        assert isinstance(val_losses[0], float)
        assert isinstance(train_accs[0], float)
        assert isinstance(val_accs[0], float)

        # Verify accuracies are in valid range
        assert 0.0 <= train_accs[0] <= 1.0
        assert 0.0 <= val_accs[0] <= 1.0
