"""Tests for base callback infrastructure."""

from unittest.mock import Mock

import pytest

from ml_src.core.callbacks import get_callbacks
from ml_src.core.callbacks.base import Callback, CallbackManager


class TestCallback:
    """Tests for Callback base class."""

    def test_callback_has_all_hooks(self):
        """Test that Callback has all required lifecycle hooks."""
        callback = Callback()

        # Training lifecycle
        assert hasattr(callback, "on_train_begin")
        assert hasattr(callback, "on_train_end")

        # Epoch lifecycle
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_epoch_end")

        # Phase lifecycle
        assert hasattr(callback, "on_phase_begin")
        assert hasattr(callback, "on_phase_end")

        # Batch lifecycle
        assert hasattr(callback, "on_batch_begin")
        assert hasattr(callback, "on_batch_end")

        # Optimization hooks
        assert hasattr(callback, "on_backward_begin")
        assert hasattr(callback, "on_backward_end")
        assert hasattr(callback, "on_optimizer_step_begin")
        assert hasattr(callback, "on_optimizer_step_end")

    def test_callback_hooks_are_callable(self):
        """Test that all hooks can be called without errors."""
        callback = Callback()
        trainer = Mock()

        # Should not raise errors (default implementations do nothing)
        callback.on_train_begin(trainer)
        callback.on_train_end(trainer)
        callback.on_epoch_begin(trainer, epoch=0)
        callback.on_epoch_end(trainer, epoch=0, logs={})
        callback.on_phase_begin(trainer, phase="train")
        callback.on_phase_end(trainer, phase="train", logs={})
        callback.on_batch_begin(trainer, batch_idx=0, batch=(None, None))
        callback.on_batch_end(trainer, batch_idx=0, batch=(None, None), outputs=None, loss=None)
        callback.on_backward_begin(trainer, loss=None)
        callback.on_backward_end(trainer)
        callback.on_optimizer_step_begin(trainer)
        callback.on_optimizer_step_end(trainer)


class TestCallbackManager:
    """Tests for CallbackManager."""

    def test_callback_manager_initialization(self):
        """Test CallbackManager initializes correctly."""
        callbacks = [Callback(), Callback()]
        manager = CallbackManager(callbacks)

        assert manager.callbacks == callbacks
        assert len(manager.callbacks) == 2

    def test_callback_manager_empty_initialization(self):
        """Test CallbackManager with no callbacks."""
        manager = CallbackManager()

        assert manager.callbacks == []

    def test_callback_manager_invokes_all_callbacks(self):
        """Test that CallbackManager invokes hooks on all callbacks."""
        # Create mock callbacks
        cb1 = Mock(spec=Callback)
        cb2 = Mock(spec=Callback)

        manager = CallbackManager([cb1, cb2])
        trainer = Mock()

        # Test various hooks
        manager.on_train_begin(trainer)
        cb1.on_train_begin.assert_called_once_with(trainer)
        cb2.on_train_begin.assert_called_once_with(trainer)

        manager.on_epoch_end(trainer, epoch=5, logs={"loss": 0.5})
        cb1.on_epoch_end.assert_called_once_with(trainer, 5, {"loss": 0.5})
        cb2.on_epoch_end.assert_called_once_with(trainer, 5, {"loss": 0.5})

        manager.on_train_end(trainer)
        cb1.on_train_end.assert_called_once_with(trainer)
        cb2.on_train_end.assert_called_once_with(trainer)

    def test_callback_manager_execution_order(self):
        """Test that callbacks are executed in registration order."""
        execution_order = []

        class OrderTrackingCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_epoch_begin(self, trainer, epoch):
                execution_order.append(self.name)

        cb1 = OrderTrackingCallback("first")
        cb2 = OrderTrackingCallback("second")
        cb3 = OrderTrackingCallback("third")

        manager = CallbackManager([cb1, cb2, cb3])
        manager.on_epoch_begin(Mock(), epoch=0)

        assert execution_order == ["first", "second", "third"]

    def test_callback_manager_handles_exceptions(self):
        """Test that exceptions in one callback don't stop others (or do they?)."""
        # Note: Current implementation doesn't catch exceptions
        # This test documents expected behavior

        class FailingCallback(Callback):
            def on_epoch_begin(self, trainer, epoch):
                raise ValueError("Intentional error")

        cb1 = FailingCallback()
        cb2 = Mock(spec=Callback)

        manager = CallbackManager([cb1, cb2])

        # Should raise exception and NOT call cb2
        with pytest.raises(ValueError, match="Intentional error"):
            manager.on_epoch_begin(Mock(), epoch=0)

        # cb2 should not be called because cb1 raised
        cb2.on_epoch_begin.assert_not_called()


class TestGetCallbacks:
    """Tests for get_callbacks factory function."""

    def test_get_callbacks_empty_config(self):
        """Test get_callbacks with no callbacks in config."""
        config = {"training": {}}
        callbacks = get_callbacks(config)

        assert callbacks == []

    def test_get_callbacks_missing_training_key(self):
        """Test get_callbacks with missing training key."""
        config = {}
        callbacks = get_callbacks(config)

        assert callbacks == []

    def test_get_callbacks_creates_callbacks(self, config_with_callbacks):
        """Test get_callbacks creates correct callback instances."""
        callbacks = get_callbacks(config_with_callbacks)

        assert len(callbacks) == 2

        # Check types
        from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback
        from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback

        assert isinstance(callbacks[0], EarlyStoppingCallback)
        assert isinstance(callbacks[1], ModelCheckpointCallback)

    def test_get_callbacks_passes_parameters(self):
        """Test that get_callbacks passes parameters correctly."""
        config = {
            "training": {
                "callbacks": [
                    {
                        "type": "early_stopping",
                        "monitor": "val_loss",
                        "patience": 15,
                        "mode": "min",
                    }
                ]
            }
        }

        callbacks = get_callbacks(config)

        assert len(callbacks) == 1
        callback = callbacks[0]

        assert callback.monitor == "val_loss"
        assert callback.patience == 15
        assert callback.mode == "min"

    def test_get_callbacks_unknown_type(self):
        """Test that unknown callback type is handled gracefully."""
        config = {
            "training": {
                "callbacks": [
                    {"type": "unknown_callback_type", "param": "value"}
                ]
            }
        }

        # Should log warning but not crash
        callbacks = get_callbacks(config)
        assert callbacks == []

    def test_get_callbacks_missing_type_field(self):
        """Test callback config without 'type' field."""
        config = {
            "training": {
                "callbacks": [
                    {"monitor": "val_acc", "patience": 10}  # Missing 'type'
                ]
            }
        }

        # Should log warning but not crash
        callbacks = get_callbacks(config)
        assert callbacks == []

    def test_get_callbacks_multiple_callbacks(self):
        """Test creating multiple different callback types."""
        config = {
            "training": {
                "callbacks": [
                    {"type": "early_stopping", "patience": 5},
                    {"type": "model_checkpoint", "save_top_k": 3},
                    {"type": "lr_monitor"},
                ]
            }
        }

        callbacks = get_callbacks(config)

        assert len(callbacks) == 3

        from ml_src.core.callbacks.early_stopping import EarlyStoppingCallback
        from ml_src.core.callbacks.checkpoint import ModelCheckpointCallback
        from ml_src.core.callbacks.lr_monitor import LearningRateMonitor

        assert isinstance(callbacks[0], EarlyStoppingCallback)
        assert isinstance(callbacks[1], ModelCheckpointCallback)
        assert isinstance(callbacks[2], LearningRateMonitor)


class TestCustomCallback:
    """Tests for creating custom callbacks."""

    def test_custom_callback_implementation(self):
        """Test that users can implement custom callbacks."""

        class CustomCallback(Callback):
            def __init__(self):
                self.train_begin_called = False
                self.epoch_count = 0

            def on_train_begin(self, trainer):
                self.train_begin_called = True

            def on_epoch_end(self, trainer, epoch, logs):
                self.epoch_count += 1

        callback = CustomCallback()
        trainer = Mock()

        # Test custom implementation
        callback.on_train_begin(trainer)
        assert callback.train_begin_called

        for epoch in range(5):
            callback.on_epoch_end(trainer, epoch, {})

        assert callback.epoch_count == 5

    def test_custom_callback_state_persistence(self):
        """Test that callbacks can maintain state across invocations."""

        class StateTrackingCallback(Callback):
            def __init__(self):
                self.losses = []

            def on_phase_end(self, trainer, phase, logs):
                if phase == "val":
                    self.losses.append(logs.get("loss", 0.0))

        callback = StateTrackingCallback()
        trainer = Mock()

        # Simulate multiple epochs
        for epoch in range(3):
            callback.on_phase_end(trainer, "train", {"loss": 0.5})
            callback.on_phase_end(trainer, "val", {"loss": 0.6 - epoch * 0.1})

        # Should only track val losses
        assert len(callback.losses) == 3
        # Use approximate equality for floating point
        assert abs(callback.losses[0] - 0.6) < 1e-6
        assert abs(callback.losses[1] - 0.5) < 1e-6
        assert abs(callback.losses[2] - 0.4) < 1e-6
