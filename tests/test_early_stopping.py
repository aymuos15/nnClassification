"""Tests for early stopping functionality."""

import pytest

from ml_src.core.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Test cases for EarlyStopping class."""

    def test_initialization_defaults(self):
        """Test EarlyStopping initializes with default values."""
        es = EarlyStopping()
        assert es.patience == 10
        assert es.metric == 'val_acc'
        assert es.mode == 'max'
        assert es.min_delta == 0.0
        assert es.counter == 0
        assert es.best_value is None
        assert es.stopped_epoch is None

    def test_initialization_custom(self):
        """Test EarlyStopping initializes with custom values."""
        es = EarlyStopping(patience=5, metric='val_loss', mode='min', min_delta=0.001)
        assert es.patience == 5
        assert es.metric == 'val_loss'
        assert es.mode == 'min'
        assert es.min_delta == 0.001

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            EarlyStopping(mode='invalid')

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="metric must be 'val_acc' or 'val_loss'"):
            EarlyStopping(metric='invalid_metric')

    def test_improvement_detected_max_mode(self):
        """Test improvement detection in max mode (for accuracy)."""
        es = EarlyStopping(patience=3, metric='val_acc', mode='max')

        # First epoch - improvement (initial value)
        assert not es.should_stop(0, 0.7)
        assert es.best_value == 0.7
        assert es.counter == 0

        # Second epoch - improvement
        assert not es.should_stop(1, 0.8)
        assert es.best_value == 0.8
        assert es.counter == 0

        # Third epoch - improvement
        assert not es.should_stop(2, 0.85)
        assert es.best_value == 0.85
        assert es.counter == 0

    def test_improvement_detected_min_mode(self):
        """Test improvement detection in min mode (for loss)."""
        es = EarlyStopping(patience=3, metric='val_loss', mode='min')

        # First epoch - improvement (initial value)
        assert not es.should_stop(0, 1.0)
        assert es.best_value == 1.0
        assert es.counter == 0

        # Second epoch - improvement
        assert not es.should_stop(1, 0.8)
        assert es.best_value == 0.8
        assert es.counter == 0

        # Third epoch - improvement
        assert not es.should_stop(2, 0.5)
        assert es.best_value == 0.5
        assert es.counter == 0

    def test_no_improvement_counter_increments(self):
        """Test that counter increments when no improvement."""
        es = EarlyStopping(patience=3, metric='val_acc', mode='max')

        # First epoch - improvement
        assert not es.should_stop(0, 0.8)
        assert es.counter == 0

        # No improvement - counter increments
        assert not es.should_stop(1, 0.79)
        assert es.counter == 1

        # No improvement - counter increments
        assert not es.should_stop(2, 0.78)
        assert es.counter == 2

    def test_early_stopping_triggered(self):
        """Test that early stopping triggers after patience exceeded."""
        es = EarlyStopping(patience=3, metric='val_acc', mode='max')

        # Initial improvement
        assert not es.should_stop(0, 0.8)

        # No improvements for 3 epochs (counter reaches patience at epoch 3)
        assert not es.should_stop(1, 0.79)
        assert not es.should_stop(2, 0.78)

        # Third epoch without improvement - should stop (counter=3, patience=3)
        assert es.should_stop(3, 0.77)
        assert es.stopped_epoch == 3

    def test_counter_resets_on_improvement(self):
        """Test that counter resets when improvement is detected."""
        es = EarlyStopping(patience=3, metric='val_acc', mode='max')

        # Initial improvement
        assert not es.should_stop(0, 0.8)

        # No improvement
        assert not es.should_stop(1, 0.79)
        assert es.counter == 1

        # No improvement
        assert not es.should_stop(2, 0.78)
        assert es.counter == 2

        # Improvement - counter should reset
        assert not es.should_stop(3, 0.85)
        assert es.counter == 0
        assert es.best_value == 0.85

        # No improvement again
        assert not es.should_stop(4, 0.84)
        assert es.counter == 1

    def test_min_delta_threshold_max_mode(self):
        """Test min_delta threshold in max mode."""
        es = EarlyStopping(patience=2, metric='val_acc', mode='max', min_delta=0.01)

        # Initial value
        assert not es.should_stop(0, 0.8)

        # Small improvement (0.005) - below min_delta, should not count as improvement
        assert not es.should_stop(1, 0.805)
        assert es.counter == 1

        # Large improvement (0.02) - above min_delta, should count as improvement
        assert not es.should_stop(2, 0.82)
        assert es.counter == 0
        assert es.best_value == 0.82

    def test_min_delta_threshold_min_mode(self):
        """Test min_delta threshold in min mode."""
        es = EarlyStopping(patience=2, metric='val_loss', mode='min', min_delta=0.01)

        # Initial value
        assert not es.should_stop(0, 1.0)

        # Small improvement (0.005) - below min_delta, should not count as improvement
        assert not es.should_stop(1, 0.995)
        assert es.counter == 1

        # Large improvement (0.15) - above min_delta, should count as improvement
        assert not es.should_stop(2, 0.85)
        assert es.counter == 0
        assert es.best_value == 0.85

    def test_get_state(self):
        """Test getting state for checkpointing."""
        es = EarlyStopping(patience=5, metric='val_loss', mode='min', min_delta=0.001)
        es.should_stop(0, 1.0)
        es.should_stop(1, 0.99)

        state = es.get_state()

        assert state['patience'] == 5
        assert state['metric'] == 'val_loss'
        assert state['mode'] == 'min'
        assert state['min_delta'] == 0.001
        assert state['counter'] == 0
        assert state['best_value'] == 0.99
        assert state['stopped_epoch'] is None

    def test_load_state(self):
        """Test loading state from checkpoint."""
        es = EarlyStopping()

        state = {
            'patience': 7,
            'metric': 'val_loss',
            'mode': 'min',
            'min_delta': 0.005,
            'counter': 3,
            'best_value': 0.5,
            'stopped_epoch': None,
        }

        es.load_state(state)

        assert es.patience == 7
        assert es.metric == 'val_loss'
        assert es.mode == 'min'
        assert es.min_delta == 0.005
        assert es.counter == 3
        assert es.best_value == 0.5
        assert es.stopped_epoch is None

    def test_state_persistence_after_stopping(self):
        """Test that stopped_epoch is correctly saved in state."""
        es = EarlyStopping(patience=2, metric='val_acc', mode='max')

        # Trigger early stopping
        es.should_stop(0, 0.8)
        es.should_stop(1, 0.79)
        es.should_stop(2, 0.78)
        es.should_stop(3, 0.77)  # Should trigger stopping

        state = es.get_state()
        assert state['stopped_epoch'] == 3
