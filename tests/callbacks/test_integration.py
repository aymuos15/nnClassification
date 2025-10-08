"""Integration tests for callbacks with trainers and CLI."""

from unittest.mock import Mock, patch

import pytest

from ml_src.core.callbacks import get_callbacks
from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback
from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback


class TestCallbacksWithTrainer:
    """Tests for callbacks working with actual trainers."""

    def test_multiple_callbacks_work_together(self, mock_trainer):
        """Test that multiple callbacks can work together without conflicts."""
        from ml_src.core.callbacks.base import CallbackManager

        early_stop = EarlyStoppingCallback(monitor="val_acc", patience=3)
        checkpoint = ModelCheckpointCallback(monitor="val_acc", save_top_k=2)

        manager = CallbackManager([early_stop, checkpoint])

        # Simulate training loop
        for epoch in range(10):
            # Simulating declining performance
            val_acc = 0.9 - (epoch * 0.02)
            logs = {"val_acc": val_acc, "train_acc": 0.95, "val_loss": 0.2, "train_loss": 0.1, "lr": 0.001}

            manager.on_epoch_begin(mock_trainer, epoch)
            manager.on_epoch_end(mock_trainer, epoch, logs)

            # Early stopping should trigger
            if mock_trainer.should_stop:
                break

        # Early stopping should have triggered
        assert mock_trainer.should_stop
        assert early_stop.stopped_epoch is not None

        # Checkpoint callback should have saved models
        assert len(checkpoint.saved_checkpoints) > 0

    def test_callbacks_receive_correct_lifecycle_calls(self, mock_trainer):
        """Test that callbacks receive lifecycle calls in correct order."""
        from ml_src.core.callbacks.base import Callback, CallbackManager

        call_order = []

        class OrderTrackingCallback(Callback):
            def on_train_begin(self, trainer):
                call_order.append("train_begin")

            def on_epoch_begin(self, trainer, epoch):
                call_order.append(f"epoch_begin_{epoch}")

            def on_phase_begin(self, trainer, phase):
                call_order.append(f"phase_begin_{phase}")

            def on_phase_end(self, trainer, phase, logs):
                call_order.append(f"phase_end_{phase}")

            def on_epoch_end(self, trainer, epoch, logs):
                call_order.append(f"epoch_end_{epoch}")

            def on_train_end(self, trainer):
                call_order.append("train_end")

        callback = OrderTrackingCallback()
        manager = CallbackManager([callback])

        # Simulate minimal training loop
        manager.on_train_begin(mock_trainer)

        for epoch in range(2):
            manager.on_epoch_begin(mock_trainer, epoch)

            for phase in ["train", "val"]:
                manager.on_phase_begin(mock_trainer, phase)
                manager.on_phase_end(mock_trainer, phase, {"loss": 0.5})

            manager.on_epoch_end(mock_trainer, epoch, {"val_acc": 0.8})

        manager.on_train_end(mock_trainer)

        # Verify order
        assert call_order[0] == "train_begin"
        assert call_order[-1] == "train_end"

        # Check epoch structure is correct
        assert "epoch_begin_0" in call_order
        assert "phase_begin_train" in call_order
        assert "phase_end_train" in call_order
        assert "phase_begin_val" in call_order
        assert "phase_end_val" in call_order
        assert "epoch_end_0" in call_order

    def test_early_stopping_sets_trainer_flag(self, mock_trainer):
        """Test that early stopping sets trainer.should_stop flag."""
        callback = EarlyStoppingCallback(monitor="val_acc", patience=2, mode="max")

        # Initial value
        assert not mock_trainer.should_stop

        # No improvement for patience epochs
        callback.on_epoch_end(mock_trainer, 0, {"val_acc": 0.8})
        callback.on_epoch_end(mock_trainer, 1, {"val_acc": 0.75})
        callback.on_epoch_end(mock_trainer, 2, {"val_acc": 0.74})

        # Should have set the flag
        assert mock_trainer.should_stop

    def test_checkpoint_callback_saves_to_run_dir(self, mock_trainer):
        """Test that checkpoint callback respects trainer's run_dir."""
        callback = ModelCheckpointCallback(monitor="val_acc", save_top_k=1)

        logs = {"val_acc": 0.85}
        callback.on_epoch_end(mock_trainer, epoch=5, logs=logs)

        # Should have called save_checkpoint
        assert mock_trainer.save_checkpoint.called

        # Check that filepath is in run_dir
        call_args = mock_trainer.save_checkpoint.call_args
        filepath = call_args[0][-1]  # Last positional argument
        assert mock_trainer.run_dir in filepath


class TestGetCallbacksIntegration:
    """Integration tests for get_callbacks with various configs."""

    def test_load_callbacks_from_config(self, config_with_callbacks):
        """Test loading callbacks from configuration."""
        callbacks = get_callbacks(config_with_callbacks)

        assert len(callbacks) == 2
        assert isinstance(callbacks[0], EarlyStoppingCallback)
        assert isinstance(callbacks[1], ModelCheckpointCallback)

        # Verify parameters were passed correctly
        assert callbacks[0].monitor == "val_acc"
        assert callbacks[0].patience == 5
        assert callbacks[1].save_top_k == 3

    def test_empty_config_returns_empty_list(self):
        """Test that empty config returns no callbacks."""
        config = {"training": {}}
        callbacks = get_callbacks(config)

        assert callbacks == []

    def test_all_callback_types_can_be_loaded(self):
        """Test that all registered callback types can be instantiated."""
        config = {
            "training": {
                "callbacks": [
                    {"type": "early_stopping"},
                    {"type": "model_checkpoint"},
                    {"type": "lr_monitor"},
                    {"type": "progress_bar"},
                    {"type": "swa"},
                    {"type": "gradient_clipping"},
                    {"type": "gradient_norm_monitor"},
                    {"type": "mixup"},
                    {"type": "cutmix"},
                ]
            }
        }

        callbacks = get_callbacks(config)

        # All 9 callbacks should be created
        assert len(callbacks) == 9

        # Verify types
        from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback
        from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback
        from ml_src.core.callbacks.lr_monitor import LearningRateMonitor
        from ml_src.core.callbacks.progress import ProgressBar
        from ml_src.core.callbacks.swa import StochasticWeightAveraging
        from ml_src.core.callbacks.gradient import GradientClipping, GradientNormMonitor
        from ml_src.core.callbacks.augmentation import MixUpCallback, CutMixCallback

        assert isinstance(callbacks[0], EarlyStoppingCallback)
        assert isinstance(callbacks[1], ModelCheckpointCallback)
        assert isinstance(callbacks[2], LearningRateMonitor)
        assert isinstance(callbacks[3], ProgressBar)
        assert isinstance(callbacks[4], StochasticWeightAveraging)
        assert isinstance(callbacks[5], GradientClipping)
        assert isinstance(callbacks[6], GradientNormMonitor)
        assert isinstance(callbacks[7], MixUpCallback)
        assert isinstance(callbacks[8], CutMixCallback)


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy configurations."""

    def test_legacy_early_stopping_still_works(self, config_with_legacy_early_stopping):
        """Test that legacy early_stopping config format is supported."""
        # The legacy format is in config['training']['early_stopping']
        # The new system should still work alongside it

        # With callbacks, the legacy config should be ignored
        config = config_with_legacy_early_stopping.copy()
        config["training"]["callbacks"] = [
            {"type": "early_stopping", "patience": 3}
        ]

        callbacks = get_callbacks(config)
        assert len(callbacks) == 1
        assert callbacks[0].patience == 3  # Should use callback config, not legacy

    def test_no_callbacks_returns_empty(self):
        """Test that no callbacks section returns empty list."""
        config = {
            "training": {
                "num_epochs": 10,
                # No callbacks section
            }
        }

        callbacks = get_callbacks(config)
        assert callbacks == []


class TestCallbackErrorHandling:
    """Tests for error handling in callbacks."""

    def test_invalid_callback_type_logged_not_crashed(self):
        """Test that invalid callback type logs warning but doesn't crash."""
        config = {
            "training": {
                "callbacks": [
                    {"type": "nonexistent_callback", "param": "value"}
                ]
            }
        }

        # Should not raise exception
        callbacks = get_callbacks(config)
        assert callbacks == []

    def test_callback_initialization_error_raised(self):
        """Test that initialization errors are propagated."""
        config = {
            "training": {
                "callbacks": [
                    {
                        "type": "early_stopping",
                        "mode": "invalid",  # Invalid mode will raise ValueError
                    }
                ]
            }
        }

        # Should raise the initialization error
        with pytest.raises(ValueError, match="mode must be"):
            get_callbacks(config)

    def test_missing_required_logs_handled_gracefully(self, mock_trainer):
        """Test that callbacks handle missing metrics gracefully."""
        callback = EarlyStoppingCallback(monitor="val_acc")

        # Logs without the monitored metric
        logs = {"train_loss": 0.5, "train_acc": 0.9}

        # Should not crash
        callback.on_epoch_end(mock_trainer, 0, logs)

        # Counter should not increment
        assert callback.counter == 0


class TestCallbackStateManagement:
    """Tests for callback state management across training."""

    def test_callbacks_maintain_state_across_epochs(self, mock_trainer):
        """Test that callbacks correctly maintain state across epochs."""
        callback = EarlyStoppingCallback(monitor="val_acc", patience=5)

        # Simulate multiple epochs
        for epoch in range(10):
            val_acc = 0.9 - (epoch * 0.01)  # Gradually decreasing
            callback.on_epoch_end(mock_trainer, epoch, {"val_acc": val_acc})

            if mock_trainer.should_stop:
                break

        # Callback should have maintained state
        assert callback.best_value == 0.9
        assert callback.counter == 5
        assert callback.stopped_epoch is not None

    def test_checkpoint_callback_tracks_saved_models(self, mock_trainer):
        """Test that checkpoint callback tracks all saved models."""
        callback = ModelCheckpointCallback(monitor="val_acc", save_top_k=3)

        # Save multiple models
        for epoch in range(5):
            val_acc = 0.75 + (epoch * 0.05)  # Increasing accuracy
            callback.on_epoch_end(mock_trainer, epoch, {"val_acc": val_acc})

        # Should track top 3 models
        assert len(callback.saved_checkpoints) == 3

        # Should have best 3 accuracies
        saved_accs = [acc for acc, _ in callback.saved_checkpoints]
        assert max(saved_accs) == 0.95  # Best
        assert min(saved_accs) == 0.85  # Worst of top 3


class TestCallbacksWithRealScenarios:
    """Tests simulating real training scenarios."""

    def test_overfitting_scenario(self, mock_trainer):
        """Test callbacks in overfitting scenario (train acc increases, val acc decreases)."""
        early_stop = EarlyStoppingCallback(monitor="val_acc", patience=3)

        # Simulate overfitting
        for epoch in range(10):
            train_acc = 0.7 + (epoch * 0.03)  # Increasing
            val_acc = 0.8 - (epoch * 0.02)  # Decreasing

            logs = {"train_acc": train_acc, "val_acc": val_acc}
            early_stop.on_epoch_end(mock_trainer, epoch, logs)

            if mock_trainer.should_stop:
                break

        # Should have stopped early
        assert mock_trainer.should_stop
        assert early_stop.stopped_epoch is not None

    def test_plateau_scenario(self, mock_trainer):
        """Test callbacks when validation acc plateaus."""
        early_stop = EarlyStoppingCallback(monitor="val_acc", patience=5, min_delta=0.01)

        # Simulate plateau
        for epoch in range(10):
            # Small random variations around 0.85
            val_acc = 0.85 + (epoch % 2) * 0.002  # Tiny variations

            logs = {"val_acc": val_acc}
            early_stop.on_epoch_end(mock_trainer, epoch, logs)

            if mock_trainer.should_stop:
                break

        # Should stop due to no significant improvement (min_delta not met)
        assert mock_trainer.should_stop

    def test_recovery_scenario(self, mock_trainer):
        """Test that callbacks handle performance recovery correctly."""
        early_stop = EarlyStoppingCallback(monitor="val_acc", patience=3)

        # Simulate initial performance drop then recovery
        val_accs = [0.80, 0.78, 0.76, 0.85, 0.87]  # Drop then recover

        for epoch, val_acc in enumerate(val_accs):
            logs = {"val_acc": val_acc}
            early_stop.on_epoch_end(mock_trainer, epoch, logs)

            if mock_trainer.should_stop:
                break

        # Should NOT have stopped (recovery at epoch 3)
        assert not mock_trainer.should_stop
        assert early_stop.counter == 0  # Reset after improvement
        assert early_stop.best_value == 0.87  # Updated to new best
