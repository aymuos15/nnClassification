"""Tests for essential callbacks."""

import os
from unittest.mock import Mock, patch

import pytest
import torch

from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback
from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback
from ml_src.core.callbacks.lr_monitor import LearningRateMonitor
from ml_src.core.callbacks.progress import ProgressBar
from ml_src.core.callbacks.swa import StochasticWeightAveraging


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_initialization(self):
        """Test callback initializes with correct defaults."""
        callback = EarlyStoppingCallback()

        assert callback.monitor == "val_acc"
        assert callback.patience == 10
        assert callback.mode == "max"
        assert callback.min_delta == 0.0
        assert callback.counter == 0
        assert callback.best_value is None
        assert callback.stopped_epoch is None

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = EarlyStoppingCallback(
            monitor="val_loss",
            patience=5,
            mode="min",
            min_delta=0.001,
        )

        assert callback.monitor == "val_loss"
        assert callback.patience == 5
        assert callback.mode == "min"
        assert callback.min_delta == 0.001

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            EarlyStoppingCallback(mode="invalid")

    def test_improvement_detection_max_mode(self):
        """Test improvement detection in max mode."""
        callback = EarlyStoppingCallback(monitor="val_acc", mode="max")

        # First value is always improvement
        assert callback._is_improvement(0.8)

        callback.best_value = 0.8

        # Higher is better in max mode
        assert callback._is_improvement(0.85)
        assert not callback._is_improvement(0.75)
        assert not callback._is_improvement(0.8)  # Equal is not improvement

    def test_improvement_detection_min_mode(self):
        """Test improvement detection in min mode."""
        callback = EarlyStoppingCallback(monitor="val_loss", mode="min")

        callback.best_value = 0.5

        # Lower is better in min mode
        assert callback._is_improvement(0.4)
        assert not callback._is_improvement(0.6)
        assert not callback._is_improvement(0.5)  # Equal is not improvement

    def test_min_delta_respected(self):
        """Test that min_delta threshold is respected."""
        callback = EarlyStoppingCallback(
            monitor="val_acc",
            mode="max",
            min_delta=0.01,
        )

        callback.best_value = 0.80

        # Must improve by at least min_delta
        assert not callback._is_improvement(0.805)  # +0.005 not enough
        assert callback._is_improvement(0.815)  # +0.015 is enough

    def test_early_stopping_triggers_after_patience(self, mock_trainer):
        """Test that early stopping triggers after patience epochs."""
        callback = EarlyStoppingCallback(monitor="val_acc", patience=3, mode="max")

        # Epoch 0: improvement
        callback.on_epoch_end(mock_trainer, 0, {"val_acc": 0.80})
        assert not mock_trainer.should_stop
        assert callback.counter == 0

        # Epochs 1-3: no improvement
        for epoch in range(1, 4):
            callback.on_epoch_end(mock_trainer, epoch, {"val_acc": 0.79})
            if epoch < 3:
                assert not mock_trainer.should_stop
            else:
                assert mock_trainer.should_stop  # Should stop after 3 epochs

        assert callback.counter == 3
        assert callback.stopped_epoch == 3

    def test_counter_resets_on_improvement(self, mock_trainer):
        """Test that counter resets when improvement occurs."""
        callback = EarlyStoppingCallback(monitor="val_acc", patience=3, mode="max")

        # Initial improvement
        callback.on_epoch_end(mock_trainer, 0, {"val_acc": 0.80})
        assert callback.counter == 0

        # No improvement for 2 epochs
        callback.on_epoch_end(mock_trainer, 1, {"val_acc": 0.79})
        callback.on_epoch_end(mock_trainer, 2, {"val_acc": 0.78})
        assert callback.counter == 2

        # Improvement! Counter should reset
        callback.on_epoch_end(mock_trainer, 3, {"val_acc": 0.85})
        assert callback.counter == 0
        assert not mock_trainer.should_stop

    def test_missing_metric_warning(self, mock_trainer):
        """Test that missing metric in logs generates warning."""
        callback = EarlyStoppingCallback(monitor="val_acc")

        # Logs missing the monitored metric
        callback.on_epoch_end(mock_trainer, 0, {"train_loss": 0.5})

        # Should not crash, counter should not increment
        assert callback.counter == 0

    def test_state_save_and_load(self):
        """Test that callback state can be saved and restored."""
        callback = EarlyStoppingCallback(monitor="val_acc", patience=5)

        # Simulate some training
        callback.best_value = 0.85
        callback.counter = 3
        callback.stopped_epoch = 10

        # Save state
        state = callback.get_state()

        # Create new callback and load state
        new_callback = EarlyStoppingCallback()
        new_callback.load_state(state)

        assert new_callback.monitor == "val_acc"
        assert new_callback.patience == 5
        assert new_callback.best_value == 0.85
        assert new_callback.counter == 3
        assert new_callback.stopped_epoch == 10


class TestModelCheckpointCallback:
    """Tests for ModelCheckpointCallback."""

    def test_initialization(self):
        """Test callback initializes with correct defaults."""
        callback = ModelCheckpointCallback()

        assert callback.monitor == "val_acc"
        assert callback.mode == "max"
        assert callback.save_top_k == 1
        assert callback.save_last is True
        assert callback.saved_checkpoints == []

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = ModelCheckpointCallback(
            monitor="val_loss",
            mode="min",
            save_top_k=5,
            save_last=False,
            filename="model_{epoch}.pt",
        )

        assert callback.monitor == "val_loss"
        assert callback.mode == "min"
        assert callback.save_top_k == 5
        assert callback.save_last is False
        assert callback.filename == "model_{epoch}.pt"

    def test_is_better_max_mode(self):
        """Test _is_better in max mode."""
        callback = ModelCheckpointCallback(mode="max")

        # First value is always better
        assert callback._is_better(0.8, None)

        # Higher is better
        assert callback._is_better(0.9, 0.8)
        assert not callback._is_better(0.7, 0.8)

    def test_is_better_min_mode(self):
        """Test _is_better in min mode."""
        callback = ModelCheckpointCallback(mode="min")

        # Lower is better
        assert callback._is_better(0.3, 0.5)
        assert not callback._is_better(0.7, 0.5)

    def test_should_save_with_empty_list(self):
        """Test that first model is always saved."""
        callback = ModelCheckpointCallback(save_top_k=3)

        assert callback._should_save(0.8)

    def test_should_save_below_top_k(self):
        """Test saving when below top_k limit."""
        callback = ModelCheckpointCallback(save_top_k=3, mode="max")

        callback.saved_checkpoints = [
            (0.80, "model1.pt"),
            (0.82, "model2.pt"),
        ]

        # Should save because we have < 3 models
        assert callback._should_save(0.75)

    def test_should_save_better_than_worst(self):
        """Test saving when better than worst in top_k."""
        callback = ModelCheckpointCallback(save_top_k=3, mode="max")

        callback.saved_checkpoints = [
            (0.80, "model1.pt"),
            (0.82, "model2.pt"),
            (0.85, "model3.pt"),
        ]

        # Better than worst (0.80)
        assert callback._should_save(0.83)

        # Worse than worst
        assert not callback._should_save(0.75)

    def test_should_save_unlimited(self):
        """Test save_top_k=-1 saves all models."""
        callback = ModelCheckpointCallback(save_top_k=-1)

        callback.saved_checkpoints = [(0.8, f"model{i}.pt") for i in range(100)]

        # Should always save with save_top_k=-1
        assert callback._should_save(0.5)

    def test_filename_formatting(self):
        """Test filename formatting with placeholders."""
        callback = ModelCheckpointCallback(
            filename="epoch_{epoch:02d}_acc_{val_acc:.4f}.pt"
        )

        logs = {"val_acc": 0.8523, "val_loss": 0.3}
        filename = callback._format_filename(epoch=5, logs=logs)

        assert filename == "epoch_05_acc_0.8523.pt"

    def test_filename_formatting_fallback(self):
        """Test filename formatting falls back on error."""
        callback = ModelCheckpointCallback(filename="epoch_{missing_key}.pt")

        logs = {"val_acc": 0.85}
        filename = callback._format_filename(epoch=5, logs=logs)

        # Should fall back to default format
        assert filename == "epoch_05.pt"

    @patch("os.path.exists")
    @patch("os.remove")
    def test_cleanup_removes_worst_checkpoint(self, mock_remove, mock_exists, mock_trainer):
        """Test that cleanup removes the worst checkpoint."""
        mock_exists.return_value = True

        callback = ModelCheckpointCallback(save_top_k=2, mode="max")

        # Set up saved checkpoints (worst is 0.80)
        callback.saved_checkpoints = [
            (0.80, "/tmp/model1.pt"),
            (0.85, "/tmp/model2.pt"),
            (0.90, "/tmp/model3.pt"),  # Exceeds top_k
        ]

        callback._cleanup_checkpoints()

        # Should remove worst checkpoint (0.80)
        mock_remove.assert_called_once_with("/tmp/model1.pt")
        assert len(callback.saved_checkpoints) == 2
        assert (0.80, "/tmp/model1.pt") not in callback.saved_checkpoints

    def test_on_epoch_end_saves_checkpoint(self, mock_trainer):
        """Test that on_epoch_end saves checkpoint when metric improves."""
        callback = ModelCheckpointCallback(monitor="val_acc", save_top_k=3)

        logs = {"val_acc": 0.85, "val_loss": 0.3}
        callback.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        # Should have called save_checkpoint
        assert mock_trainer.save_checkpoint.called
        assert len(callback.saved_checkpoints) == 1
        assert callback.saved_checkpoints[0][0] == 0.85


class TestLearningRateMonitor:
    """Tests for LearningRateMonitor."""

    def test_initialization(self):
        """Test callback initializes with defaults."""
        callback = LearningRateMonitor()

        assert callback.log_momentum is False
        assert callback.log_to_console is False

    def test_logs_learning_rate(self, mock_trainer):
        """Test that LR is logged to TensorBoard."""
        callback = LearningRateMonitor()

        logs = {"val_acc": 0.85}
        callback.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        # Should log LR
        mock_trainer.writer.add_scalar.assert_called_with("Learning_Rate", 0.01, 5)

    def test_logs_multiple_param_groups(self, mock_trainer):
        """Test logging with multiple parameter groups."""
        # Add second param group
        mock_trainer.optimizer.param_groups.append({"lr": 0.001})

        callback = LearningRateMonitor()

        logs = {}
        callback.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        # Should log both LRs with group tags
        calls = mock_trainer.writer.add_scalar.call_args_list
        assert len(calls) == 2
        assert calls[0][0] == ("Learning_Rate/group_0", 0.01, 5)
        assert calls[1][0] == ("Learning_Rate/group_1", 0.001, 5)

    def test_logs_momentum_when_enabled(self, mock_trainer):
        """Test that momentum is logged when enabled."""
        # Add momentum to optimizer
        mock_trainer.optimizer.param_groups[0]["momentum"] = 0.9

        callback = LearningRateMonitor(log_momentum=True)

        logs = {}
        callback.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        # Should log both LR and momentum
        calls = mock_trainer.writer.add_scalar.call_args_list
        assert len(calls) == 2

        # Check that momentum was logged
        momentum_logged = any("Momentum" in str(call) for call in calls)
        assert momentum_logged


class TestProgressBar:
    """Tests for ProgressBar callback."""

    def test_initialization(self):
        """Test callback initializes with defaults."""
        callback = ProgressBar()

        assert callback.show_metrics is True
        assert callback.position == 0
        assert callback.epoch_bar is None
        assert callback.batch_bar is None

    @patch("ml_src.core.callbacks.progress.tqdm")
    def test_creates_epoch_bar(self, mock_tqdm, mock_trainer):
        """Test that epoch progress bar is created."""
        callback = ProgressBar()
        callback.on_train_begin(mock_trainer)

        # Should create epoch-level progress bar
        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["total"] == mock_trainer.num_epochs

    @patch("ml_src.core.callbacks.progress.tqdm")
    def test_creates_batch_bar(self, mock_tqdm, mock_trainer):
        """Test that batch progress bar is created for each phase."""
        callback = ProgressBar()
        callback.on_phase_begin(mock_trainer, "train")

        # Should create batch-level progress bar
        assert mock_tqdm.called
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["total"] == len(mock_trainer.dataloaders["train"])

    @patch("ml_src.core.callbacks.progress.tqdm")
    def test_closes_progress_bars(self, mock_tqdm, mock_trainer):
        """Test that progress bars are closed properly."""
        mock_bar = Mock()
        mock_tqdm.return_value = mock_bar

        callback = ProgressBar()
        callback.on_train_begin(mock_trainer)
        callback.on_train_end(mock_trainer)

        # Should close the bar
        mock_bar.close.assert_called()


class TestStochasticWeightAveraging:
    """Tests for SWA callback."""

    def test_initialization(self):
        """Test callback initializes correctly."""
        callback = StochasticWeightAveraging()

        assert callback.swa_start_epoch is None
        assert callback.swa_lr is None
        assert callback.annealing_epochs == 10
        assert callback.annealing_strategy == "cos"
        assert callback.swa_model is None
        assert callback.swa_started is False

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = StochasticWeightAveraging(
            swa_start_epoch=50,
            swa_lr=0.001,
            annealing_epochs=5,
            annealing_strategy="linear",
        )

        assert callback.swa_start_epoch == 50
        assert callback.swa_lr == 0.001
        assert callback.annealing_epochs == 5
        assert callback.annealing_strategy == "linear"

    def test_invalid_annealing_strategy(self):
        """Test that invalid annealing strategy raises error."""
        with pytest.raises(ValueError, match="annealing_strategy must be"):
            StochasticWeightAveraging(annealing_strategy="invalid")

    @patch("torch.optim.swa_utils.AveragedModel")
    @patch("torch.optim.swa_utils.SWALR")
    def test_initializes_swa_on_train_begin(
        self, mock_swalr, mock_averaged_model, mock_trainer
    ):
        """Test that SWA model and scheduler are initialized."""
        callback = StochasticWeightAveraging(swa_start_epoch=75)

        callback.on_train_begin(mock_trainer)

        # Should initialize SWA model and scheduler
        mock_averaged_model.assert_called_once_with(mock_trainer.model)
        mock_swalr.assert_called_once()

        assert callback.swa_start_epoch == 75

    @patch("torch.optim.swa_utils.AveragedModel")
    def test_default_swa_start_epoch(self, mock_averaged_model, mock_trainer):
        """Test that default SWA start epoch is 75% of training."""
        callback = StochasticWeightAveraging()  # No swa_start_epoch specified

        callback.on_train_begin(mock_trainer)

        # Should default to 75% of num_epochs
        assert callback.swa_start_epoch == int(0.75 * mock_trainer.num_epochs)

    @patch("torch.optim.swa_utils.AveragedModel")
    @patch("torch.optim.swa_utils.SWALR")
    @patch("torch.optim.swa_utils.update_bn")
    def test_swa_activates_after_start_epoch(
        self, mock_update_bn, mock_swalr, mock_averaged_model, mock_trainer
    ):
        """Test that SWA starts after specified epoch."""
        # Mock the SWA model to return tensors
        mock_swa_model_instance = Mock()
        mock_swa_model_instance.return_value = torch.randn(50, 10)  # Match expected output
        mock_swa_model_instance.eval = Mock()
        mock_averaged_model.return_value = mock_swa_model_instance

        callback = StochasticWeightAveraging(swa_start_epoch=5)
        callback.on_train_begin(mock_trainer)

        # Before start epoch
        callback.on_epoch_end(mock_trainer, epoch=4, logs={"val_acc": 0.8})
        assert not callback.swa_started

        # At start epoch - this will try to evaluate SWA model
        # Mock the evaluation to prevent errors
        callback._evaluate_swa_model = Mock()
        callback.on_epoch_end(mock_trainer, epoch=5, logs={"val_acc": 0.8})
        assert callback.swa_started
