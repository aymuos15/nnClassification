"""Shared fixtures for callback tests."""

import os
import tempfile
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(20, num_classes)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def mock_trainer():
    """Create a mock trainer with essential attributes."""
    trainer = Mock()

    # Basic attributes
    trainer.model = DummyModel(num_classes=10)
    trainer.device = torch.device("cpu")
    trainer.num_epochs = 10
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=5)
    trainer.criterion = nn.CrossEntropyLoss()

    # Dataset attributes
    trainer.dataset_sizes = {"train": 100, "val": 50}

    # Dataloaders
    train_data = TensorDataset(torch.randn(100, 20), torch.randint(0, 10, (100,)))
    val_data = TensorDataset(torch.randn(50, 20), torch.randint(0, 10, (50,)))
    trainer.dataloaders = {
        "train": DataLoader(train_data, batch_size=32),
        "val": DataLoader(val_data, batch_size=32),
    }

    # Config
    trainer.config = {
        "training": {"num_epochs": 10},
        "model": {"num_classes": 10},
    }

    # Run directory (temporary)
    trainer.run_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(trainer.run_dir, "weights"), exist_ok=True)

    # TensorBoard writer (mock)
    trainer.writer = Mock(spec=SummaryWriter)

    # Best model path
    trainer.best_model_path = os.path.join(trainer.run_dir, "weights", "best.pt")

    # Callback control
    trainer.should_stop = False

    # Mock save_checkpoint method
    trainer.save_checkpoint = Mock()

    return trainer


@pytest.fixture
def sample_logs():
    """Sample metrics logs for testing."""
    return {
        "train_loss": 0.5,
        "train_acc": 0.85,
        "val_loss": 0.6,
        "val_acc": 0.82,
        "lr": 0.001,
    }


@pytest.fixture
def config_with_callbacks():
    """Sample config with callbacks defined."""
    return {
        "training": {
            "num_epochs": 10,
            "callbacks": [
                {
                    "type": "early_stopping",
                    "monitor": "val_acc",
                    "patience": 5,
                    "mode": "max",
                },
                {
                    "type": "model_checkpoint",
                    "monitor": "val_acc",
                    "mode": "max",
                    "save_top_k": 3,
                },
            ],
        },
        "model": {"num_classes": 10},
    }


@pytest.fixture
def config_with_legacy_early_stopping():
    """Config with legacy early_stopping format (backward compatibility test)."""
    return {
        "training": {
            "num_epochs": 10,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "metric": "val_acc",
                "mode": "max",
            },
        },
        "model": {"num_classes": 10},
    }


@pytest.fixture(autouse=True)
def cleanup_temp_dirs():
    """Clean up temporary directories after tests."""
    yield
    # Cleanup happens automatically with tempfile
