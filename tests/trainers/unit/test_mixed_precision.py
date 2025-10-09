"""Tests for MixedPrecisionTrainer."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.losses import get_criterion
from ml_src.core.trainers import MixedPrecisionTrainer, get_trainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_trainer_cuda():
    """Test MixedPrecisionTrainer with CUDA device."""
    # Create config for mixed precision training
    config = {
        "training": {
            "trainer_type": "mixed_precision",
            "num_epochs": 1,
            "batch_size": 2,
            "amp_dtype": "float16",
        },
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model and move to device
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cuda:0")
    model = model.to(device)

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

        # Verify it's a MixedPrecisionTrainer
        assert isinstance(trainer, MixedPrecisionTrainer)

        # Run 1 epoch of training (this will call prepare_training internally)
        trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()

        # Verify no errors occurred
        assert trained_model is not None
        assert len(train_losses) == 1
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

        # Verify GradScaler was used (check after training)
        assert hasattr(trainer, "scaler")
        assert trainer.scaler is not None
        assert isinstance(trainer.scaler, torch.cuda.amp.GradScaler)

        # Verify amp_dtype is set correctly
        assert hasattr(trainer, "amp_dtype")
        assert trainer.amp_dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_trainer_bfloat16():
    """Test MixedPrecisionTrainer with bfloat16 dtype."""
    # Create config with bfloat16
    config = {
        "training": {
            "trainer_type": "mixed_precision",
            "num_epochs": 1,
            "batch_size": 2,
            "amp_dtype": "bfloat16",
        },
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model and move to device
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cuda:0")
    model = model.to(device)

    # Create minimal dataset
    num_samples = 8
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
        # Create required directories
        import os

        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Create trainer directly
        trainer = MixedPrecisionTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        # Prepare training (initializes scaler and dtype)
        trainer.prepare_training()

        # Verify amp_dtype is bfloat16
        assert trainer.amp_dtype == torch.bfloat16

        # Run 1 epoch
        trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()

        # Verify training completed successfully
        assert trained_model is not None
        assert len(train_losses) == 1


def test_mixed_precision_trainer_cpu_fallback():
    """Test that MixedPrecisionTrainer falls back to standard training on CPU."""
    # Create config for mixed precision
    config = {
        "training": {
            "trainer_type": "mixed_precision",
            "num_epochs": 1,
            "batch_size": 2,
            "amp_dtype": "float16",
        },
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")  # Force CPU

    # Create minimal dataset
    num_samples = 8
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
        # Create required directories
        import os

        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Create trainer
        trainer = MixedPrecisionTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            config=config,
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        # Prepare training
        trainer.prepare_training()

        # Verify GradScaler is NOT initialized on CPU
        assert trainer.scaler is None
        assert trainer.amp_dtype is None

        # Run 1 epoch - should fall back to standard training
        trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()

        # Verify training completed successfully despite being on CPU
        assert trained_model is not None
        assert len(train_losses) == 1
        assert len(val_losses) == 1
        assert len(train_accs) == 1
        assert len(val_accs) == 1

        # Verify accuracies are in valid range
        assert 0.0 <= train_accs[0] <= 1.0
        assert 0.0 <= val_accs[0] <= 1.0
