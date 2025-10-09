"""Tests for trainer factory and StandardTrainer."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.losses import get_criterion
from ml_src.core.trainers import get_trainer


def test_standard_trainer_factory():
    """Test that StandardTrainer loads via factory and runs 1 epoch."""
    # Create minimal config
    config = {
        "training": {"trainer_type": "standard", "num_epochs": 1, "batch_size": 2},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset
    num_samples = 10
    X_train = torch.randn(num_samples, 10)
    y_train = torch.randint(0, 2, (num_samples,))
    X_val = torch.randn(num_samples, 10)
    y_val = torch.randint(0, 2, (num_samples,))

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
        # Create weights directory
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


def test_trainer_factory_backward_compatibility():
    """Test that factory defaults to StandardTrainer when trainer_type is omitted."""
    # Create config WITHOUT trainer_type field (backward compatibility)
    config = {
        "training": {"num_epochs": 1, "batch_size": 2},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset (ensure both classes are represented)
    num_samples = 8
    X_train = torch.randn(num_samples, 10)
    # Ensure we have both classes
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
        # Create weights directory
        import os

        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Should default to StandardTrainer
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

        # Verify it's a StandardTrainer
        from ml_src.core.trainers import StandardTrainer

        assert isinstance(trainer, StandardTrainer)

        # Run 1 epoch and verify it works
        trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()
        assert trained_model is not None
        assert len(train_losses) == 1


def test_trainer_factory_invalid_type():
    """Test that factory raises error for invalid trainer_type."""
    # Create config with invalid trainer_type
    config = {
        "training": {"trainer_type": "invalid_trainer", "num_epochs": 1, "batch_size": 2},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    model = nn.Sequential(nn.Linear(10, 2))
    device = torch.device("cpu")
    dataloaders = {"train": None, "val": None}
    dataset_sizes = {"train": 0, "val": 0}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    criterion = get_criterion()

    with tempfile.TemporaryDirectory() as run_dir:
        import os

        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unsupported trainer_type"):
            get_trainer(
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
