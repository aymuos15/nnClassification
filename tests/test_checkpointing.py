"""Comprehensive tests for checkpointing system (save/load, EMA, early stopping)."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml_src.core.checkpointing import (
    count_parameters,
    format_duration,
    load_checkpoint,
    save_checkpoint,
    save_summary,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer for test model."""
    return torch.optim.SGD(simple_model.parameters(), lr=0.01, momentum=0.9)


@pytest.fixture
def scheduler(optimizer):
    """Create scheduler."""
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "data": {"data_dir": "/path/to/data", "fold": 0},
        "model": {"architecture": "resnet18", "num_classes": 10},
        "training": {"num_epochs": 25, "batch_size": 32},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 7, "gamma": 0.1},
    }


@pytest.fixture
def training_state():
    """Sample training state."""
    return {
        "epoch": 5,
        "best_acc": 0.85,
        "train_losses": [0.5, 0.4, 0.3, 0.25, 0.2],
        "val_losses": [0.6, 0.5, 0.45, 0.4, 0.35],
        "train_accs": [0.7, 0.75, 0.8, 0.82, 0.85],
        "val_accs": [0.65, 0.7, 0.75, 0.8, 0.85],
    }


# ============================================================================
# save_checkpoint Tests
# ============================================================================


class TestSaveCheckpoint:
    """Tests for save_checkpoint function."""

    def test_save_checkpoint_creates_file(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that save_checkpoint creates a file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            assert os.path.exists(checkpoint_path)
            assert os.path.getsize(checkpoint_path) > 0
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_model_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains model state dict."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "model_state_dict" in checkpoint
            assert len(checkpoint["model_state_dict"]) > 0
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_optimizer_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains optimizer state dict."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "optimizer_state_dict" in checkpoint
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_scheduler_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains scheduler state dict."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "scheduler_state_dict" in checkpoint
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_training_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains all training state."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert checkpoint["epoch"] == 5
            assert checkpoint["best_acc"] == 0.85
            assert len(checkpoint["train_losses"]) == 5
            assert len(checkpoint["val_losses"]) == 5
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_config(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains configuration."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "config" in checkpoint
            assert checkpoint["config"]["model"]["architecture"] == "resnet18"
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_contains_random_states(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint contains random states."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "torch_rng_state" in checkpoint
            assert "numpy_rng_state" in checkpoint
            assert "python_rng_state" in checkpoint
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_with_ema_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test saving checkpoint with EMA state."""
        ema_state = {"ema_model_state": simple_model.state_dict(), "decay": 0.999}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                ema_state=ema_state,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "ema_state" in checkpoint
            assert "ema_model_state" in checkpoint["ema_state"]
            assert checkpoint["ema_state"]["decay"] == 0.999
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_with_early_stopping_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test saving checkpoint with early stopping state."""
        early_stopping_state = {
            "patience": 10,
            "counter": 3,
            "best_value": 0.85,
            "stopped_epoch": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                early_stopping_state=early_stopping_state,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "early_stopping_state" in checkpoint
            assert checkpoint["early_stopping_state"]["patience"] == 10
            assert checkpoint["early_stopping_state"]["counter"] == 3
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_save_checkpoint_timestamp(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that checkpoint includes timestamp."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "timestamp" in checkpoint
            assert isinstance(checkpoint["timestamp"], str)
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


# ============================================================================
# load_checkpoint Tests
# ============================================================================


class TestLoadCheckpoint:
    """Tests for load_checkpoint function."""

    def test_load_checkpoint_missing_file(self, simple_model, optimizer, scheduler):
        """Test that loading missing checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(
                "/nonexistent/checkpoint.pt", simple_model, optimizer, scheduler, device="cpu"
            )

    def test_load_checkpoint_restores_model_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that model state is restored correctly."""
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save original model state
            original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            # Modify model
            for param in simple_model.parameters():
                param.data.fill_(999.0)

            # Create new model and load
            new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=7)

            load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler, device="cpu")

            # Verify state restored
            for key in original_state:
                assert torch.allclose(new_model.state_dict()[key], original_state[key])
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_returns_correct_values(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that load_checkpoint returns correct values."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            # Load checkpoint
            (
                epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                early_stopping_state,
                ema_state,
            ) = load_checkpoint(checkpoint_path, simple_model, optimizer, scheduler, device="cpu")

            # Verify values
            assert epoch == 5
            assert best_acc == 0.85
            assert len(train_losses) == 5
            assert len(val_losses) == 5
            assert len(train_accs) == 5
            assert len(val_accs) == 5
            assert config is not None
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_restores_random_states(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test that random states are restored."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Set specific random seeds before saving
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            # Generate some random numbers to change state
            torch.rand(100)
            np.random.rand(100)
            random.random()

            # Save checkpoint
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            # Generate random numbers after saving
            value1 = torch.rand(1).item()

            # Load checkpoint (should restore RNG state)
            load_checkpoint(checkpoint_path, simple_model, optimizer, scheduler, device="cpu")

            # Generate random number with restored state
            value2 = torch.rand(1).item()

            # Should be the same as value1 (same RNG state)
            assert abs(value1 - value2) < 1e-6
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_with_ema_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test loading checkpoint with EMA state."""
        ema_state = {"ema_model_state": simple_model.state_dict(), "decay": 0.999}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                ema_state=ema_state,
                **training_state,
            )

            (
                epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                early_stopping_state,
                loaded_ema_state,
            ) = load_checkpoint(checkpoint_path, simple_model, optimizer, scheduler, device="cpu")

            assert loaded_ema_state is not None
            assert "ema_model_state" in loaded_ema_state
            assert loaded_ema_state["decay"] == 0.999
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_without_ema_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test loading checkpoint without EMA state returns None."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            (
                epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                early_stopping_state,
                ema_state,
            ) = load_checkpoint(checkpoint_path, simple_model, optimizer, scheduler, device="cpu")

            assert ema_state is None
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_load_checkpoint_with_early_stopping_state(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test loading checkpoint with early stopping state."""
        early_stopping_state = {"patience": 10, "counter": 3, "best_value": 0.85}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                early_stopping_state=early_stopping_state,
                **training_state,
            )

            (
                epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                loaded_early_stopping_state,
                ema_state,
            ) = load_checkpoint(checkpoint_path, simple_model, optimizer, scheduler, device="cpu")

            assert loaded_early_stopping_state is not None
            assert loaded_early_stopping_state["patience"] == 10
            assert loaded_early_stopping_state["counter"] == 3
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestCountParameters:
    """Tests for count_parameters function."""

    def test_count_parameters_simple_model(self):
        """Test counting parameters in a simple model."""
        model = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        count = count_parameters(model)
        assert count == 55

    def test_count_parameters_multi_layer(self):
        """Test counting parameters in multi-layer model."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        # Layer 1: 10*5 + 5 = 55
        # Layer 2: 5*2 + 2 = 12
        # Total: 67
        count = count_parameters(model)
        assert count == 67

    def test_count_parameters_with_frozen_layers(self):
        """Test that frozen layers are not counted."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        # Should only count second layer (12 parameters)
        count = count_parameters(model)
        assert count == 12

    def test_count_parameters_empty_model(self):
        """Test counting parameters in model with no parameters."""
        model = nn.Sequential()
        count = count_parameters(model)
        assert count == 0


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_duration_seconds(self):
        """Test formatting duration in seconds."""
        assert format_duration(45) == "45s"
        assert format_duration(10.5) == "10s"

    def test_format_duration_minutes(self):
        """Test formatting duration in minutes."""
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"

    def test_format_duration_hours(self):
        """Test formatting duration in hours."""
        assert format_duration(3661) == "1h 1m 1s"
        assert format_duration(7325) == "2h 2m 5s"

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        assert format_duration(0) == "0s"

    def test_format_duration_fractional(self):
        """Test formatting fractional seconds."""
        result = format_duration(45.7)
        assert "45s" in result or "46s" in result  # Depends on rounding


class TestSaveSummary:
    """Tests for save_summary function."""

    def test_save_summary_creates_file(self):
        """Test that save_summary creates a file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            save_summary(summary_path, status="completed")
            assert os.path.exists(summary_path)
            assert os.path.getsize(summary_path) > 0
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_save_summary_contains_status(self):
        """Test that summary contains status information."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            save_summary(summary_path, status="completed")
            content = Path(summary_path).read_text()
            assert "COMPLETED" in content
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_save_summary_with_timing(self):
        """Test summary with timing information."""
        import time

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            start = time.time()
            end = start + 3661  # 1 hour, 1 minute, 1 second

            save_summary(summary_path, status="completed", start_time=start, end_time=end)
            content = Path(summary_path).read_text()
            assert "TIMING" in content
            assert "Duration" in content
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_save_summary_with_metrics(self):
        """Test summary with performance metrics."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            save_summary(
                summary_path,
                status="completed",
                best_acc=0.95,
                best_epoch=10,
                final_val_acc=0.93,
                final_val_loss=0.25,
            )
            content = Path(summary_path).read_text()
            assert "METRICS" in content
            assert "0.95" in content
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_save_summary_with_config(self, sample_config):
        """Test summary with configuration."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            save_summary(summary_path, status="completed", config=sample_config)
            content = Path(summary_path).read_text()
            assert "CONFIGURATION" in content
            assert "Batch Size" in content
        finally:
            Path(summary_path).unlink(missing_ok=True)

    def test_save_summary_failed_status(self):
        """Test summary with failed status."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            summary_path = f.name

        try:
            save_summary(
                summary_path, status="failed", error_message="Out of memory error"
            )
            content = Path(summary_path).read_text()
            assert "FAILED" in content
            assert "ERROR" in content
            assert "Out of memory" in content
        finally:
            Path(summary_path).unlink(missing_ok=True)


# ============================================================================
# Integration Tests
# ============================================================================


class TestCheckpointingIntegration:
    """Integration tests for checkpointing workflow."""

    def test_save_load_roundtrip(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test full save/load roundtrip."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Save checkpoint
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                config=sample_config,
                **training_state,
            )

            # Create new objects
            new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=7)

            # Load checkpoint
            (
                epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                early_stopping_state,
                ema_state,
            ) = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler, "cpu")

            # Verify all values match
            assert epoch == training_state["epoch"]
            assert best_acc == training_state["best_acc"]
            assert train_losses == training_state["train_losses"]
            assert val_losses == training_state["val_losses"]
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)

    def test_resume_training_scenario(
        self, simple_model, optimizer, scheduler, training_state, sample_config
    ):
        """Test realistic training resume scenario."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name

        try:
            # Phase 1: Train for 5 epochs, save checkpoint
            epoch_5_losses = training_state["train_losses"][:5]
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=5,
                best_acc=0.80,
                train_losses=epoch_5_losses,
                val_losses=training_state["val_losses"][:5],
                train_accs=training_state["train_accs"][:5],
                val_accs=training_state["val_accs"][:5],
                checkpoint_path=checkpoint_path,
                config=sample_config,
            )

            # Phase 2: Load and continue training
            new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=7)

            (
                resumed_epoch,
                best_acc,
                train_losses,
                val_losses,
                train_accs,
                val_accs,
                config,
                _,
                _,
            ) = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler, "cpu")

            # Verify we can resume from epoch 5
            assert resumed_epoch == 5
            assert len(train_losses) == 5
            # Next epoch would be 6
            next_epoch = resumed_epoch + 1
            assert next_epoch == 6
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
