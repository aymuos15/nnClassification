"""Comprehensive tests for Model EMA (Exponential Moving Average) training."""

import torch
import torch.nn as nn

from ml_src.core.training.ema import ModelEMA


# ============================================================================
# Fixtures and Helpers
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# ModelEMA Initialization Tests
# ============================================================================


class TestModelEMAInitialization:
    """Tests for ModelEMA initialization."""

    def test_model_ema_creates_copy(self):
        """Test that ModelEMA creates a deep copy of the model."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        # Should be a different instance
        assert ema.ema_model is not model

        # But should have same architecture
        assert type(ema.ema_model) == type(model)

    def test_model_ema_default_decay(self):
        """Test default decay value."""
        model = SimpleModel()
        ema = ModelEMA(model)

        assert ema.decay == 0.9999

    def test_model_ema_custom_decay(self):
        """Test custom decay value."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.995)

        assert ema.decay == 0.995

    def test_model_ema_default_warmup_steps(self):
        """Test default warmup steps."""
        model = SimpleModel()
        ema = ModelEMA(model)

        assert ema.warmup_steps == 0

    def test_model_ema_custom_warmup_steps(self):
        """Test custom warmup steps."""
        model = SimpleModel()
        ema = ModelEMA(model, warmup_steps=1000)

        assert ema.warmup_steps == 1000

    def test_model_ema_initial_updates_zero(self):
        """Test that initial updates counter is zero."""
        model = SimpleModel()
        ema = ModelEMA(model)

        assert ema.updates == 0

    def test_model_ema_disables_gradients(self):
        """Test that EMA model has gradients disabled."""
        model = SimpleModel()
        ema = ModelEMA(model)

        for param in ema.ema_model.parameters():
            assert param.requires_grad is False

    def test_model_ema_is_in_eval_mode(self):
        """Test that EMA model is in eval mode."""
        model = SimpleModel()
        ema = ModelEMA(model)

        assert not ema.ema_model.training

    def test_model_ema_device_default(self):
        """Test that EMA model uses same device as original model."""
        model = SimpleModel()
        ema = ModelEMA(model)

        # Should be on CPU (default)
        for param in ema.ema_model.parameters():
            assert param.device.type == "cpu"

    def test_model_ema_custom_device_cpu(self):
        """Test specifying CPU device explicitly."""
        model = SimpleModel()
        ema = ModelEMA(model, device=torch.device("cpu"))

        for param in ema.ema_model.parameters():
            assert param.device.type == "cpu"


# ============================================================================
# ModelEMA Update Tests
# ============================================================================


class TestModelEMAUpdate:
    """Tests for ModelEMA update mechanism."""

    def test_update_increments_counter(self):
        """Test that update increments the updates counter."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999)

        assert ema.updates == 0
        ema.update(model)
        assert ema.updates == 1
        ema.update(model)
        assert ema.updates == 2

    def test_update_during_warmup_no_weight_change(self):
        """Test that updates during warmup don't change weights."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=10)

        # Get initial EMA weights
        initial_weights = {k: v.clone() for k, v in ema.ema_model.state_dict().items()}

        # Modify model weights
        for param in model.parameters():
            param.data.fill_(999.0)

        # Update during warmup (should be skipped)
        for _ in range(5):
            ema.update(model)

        # EMA weights should be unchanged
        for key in initial_weights:
            assert torch.allclose(ema.ema_model.state_dict()[key], initial_weights[key])

    def test_update_after_warmup_changes_weights(self):
        """Test that updates after warmup do change weights."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=5)

        # Get initial EMA weights
        initial_weights = {k: v.clone() for k, v in ema.ema_model.state_dict().items()}

        # Modify model weights significantly
        for param in model.parameters():
            param.data.fill_(100.0)

        # Update past warmup
        for _ in range(10):
            ema.update(model)

        # EMA weights should have changed
        changed = False
        for key in initial_weights:
            if not torch.allclose(ema.ema_model.state_dict()[key], initial_weights[key]):
                changed = True
                break

        assert changed

    def test_update_respects_decay_rate(self):
        """Test that update correctly applies decay rate."""
        model = SimpleModel()

        # Initialize model with specific values
        for param in model.parameters():
            param.data.fill_(0.0)

        ema = ModelEMA(model, decay=0.9, warmup_steps=0)

        # Get initial EMA weights (should be 0)
        initial_ema = list(ema.ema_model.parameters())[0].data.clone()

        # Change model weights to 1.0
        for param in model.parameters():
            param.data.fill_(1.0)

        # Update once: ema = 0.9 * 0.0 + 0.1 * 1.0 = 0.1
        ema.update(model)

        new_ema = list(ema.ema_model.parameters())[0].data[0, 0].item()

        # Should be close to 0.1 (with some floating point tolerance)
        assert abs(new_ema - 0.1) < 1e-4

    def test_update_multiple_times_converges(self):
        """Test that multiple updates converge EMA toward current weights."""
        model = SimpleModel()

        # Initialize with zeros
        for param in model.parameters():
            param.data.fill_(0.0)

        ema = ModelEMA(model, decay=0.99, warmup_steps=0)

        # Change model to ones
        for param in model.parameters():
            param.data.fill_(1.0)

        # Update many times
        for _ in range(1000):
            ema.update(model)

        # EMA should be close to 1.0 now
        ema_value = list(ema.ema_model.parameters())[0].data[0, 0].item()
        assert abs(ema_value - 1.0) < 0.01  # Within 1% of target

    def test_update_with_zero_warmup(self):
        """Test updates work correctly with zero warmup."""
        model = SimpleModel()

        for param in model.parameters():
            param.data.fill_(1.0)

        ema = ModelEMA(model, decay=0.9, warmup_steps=0)

        # Should update immediately
        for param in model.parameters():
            param.data.fill_(5.0)

        ema.update(model)

        # Check that update happened
        ema_value = list(ema.ema_model.parameters())[0].data[0, 0].item()
        # Should not be 1.0 anymore
        assert abs(ema_value - 1.0) > 0.1


# ============================================================================
# ModelEMA State Dict Tests
# ============================================================================


class TestModelEMAStateDictSaving:
    """Tests for ModelEMA state dict operations."""

    def test_state_dict_contains_required_keys(self):
        """Test that state dict contains all required keys."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=100)

        state = ema.state_dict()

        assert "ema_model_state" in state
        assert "decay" in state
        assert "warmup_steps" in state
        assert "updates" in state

    def test_state_dict_preserves_values(self):
        """Test that state dict preserves values correctly."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.995, warmup_steps=50)

        # Perform some updates
        for _ in range(10):
            ema.update(model)

        state = ema.state_dict()

        assert state["decay"] == 0.995
        assert state["warmup_steps"] == 50
        assert state["updates"] == 10

    def test_load_state_dict_restores_values(self):
        """Test that load_state_dict restores values correctly."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=100)

        # Perform updates
        for _ in range(25):
            ema.update(model)

        # Save state
        state = ema.state_dict()

        # Create new EMA
        model2 = SimpleModel()
        ema2 = ModelEMA(model2)

        # Load state
        ema2.load_state_dict(state)

        # Verify restoration
        assert ema2.decay == 0.999
        assert ema2.warmup_steps == 100
        assert ema2.updates == 25

    def test_load_state_dict_restores_weights(self):
        """Test that load_state_dict restores model weights."""
        model = SimpleModel()

        # Set specific values
        for param in model.parameters():
            param.data.fill_(7.0)

        ema = ModelEMA(model, decay=0.9)

        # Perform updates to change EMA weights
        for param in model.parameters():
            param.data.fill_(15.0)

        for _ in range(20):
            ema.update(model)

        # Save EMA weights
        ema_weight = list(ema.ema_model.parameters())[0].data.clone()
        state = ema.state_dict()

        # Create new EMA with different weights
        model2 = SimpleModel()
        for param in model2.parameters():
            param.data.fill_(999.0)

        ema2 = ModelEMA(model2)

        # Load state
        ema2.load_state_dict(state)

        # Verify weights restored
        ema2_weight = list(ema2.ema_model.parameters())[0].data
        assert torch.allclose(ema2_weight, ema_weight)

    def test_save_load_roundtrip(self):
        """Test full save/load roundtrip."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9999, warmup_steps=500)

        # Simulate training
        for i in range(100):
            # Modify model
            for param in model.parameters():
                param.data.add_(0.01)
            ema.update(model)

        # Save state
        state = ema.state_dict()

        # Create new EMA and load
        model2 = SimpleModel()
        ema2 = ModelEMA(model2)
        ema2.load_state_dict(state)

        # Verify everything matches
        assert ema2.decay == ema.decay
        assert ema2.warmup_steps == ema.warmup_steps
        assert ema2.updates == ema.updates

        # Verify weights match
        for p1, p2 in zip(ema.ema_model.parameters(), ema2.ema_model.parameters()):
            assert torch.allclose(p1.data, p2.data)


# ============================================================================
# ModelEMA Property Tests
# ============================================================================


class TestModelEMAProperties:
    """Tests for ModelEMA properties and methods."""

    def test_model_property_returns_ema_model(self):
        """Test that model property returns the EMA model."""
        model = SimpleModel()
        ema = ModelEMA(model)

        assert ema.model is ema.ema_model

    def test_repr(self):
        """Test string representation."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=100)

        repr_str = repr(ema)
        assert "ModelEMA" in repr_str
        assert "0.999" in repr_str
        assert "100" in repr_str

    def test_repr_after_updates(self):
        """Test repr includes update count."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=100)

        # Perform updates
        for _ in range(50):
            ema.update(model)

        repr_str = repr(ema)
        assert "updates=50" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================


class TestModelEMAIntegration:
    """Integration tests for realistic EMA usage."""

    def test_training_loop_simulation(self):
        """Test EMA in a realistic training loop."""
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.999, warmup_steps=10)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate training
        for epoch in range(5):
            for batch_idx in range(20):
                # Forward pass
                x = torch.randn(4, 10)
                output = model(x)
                loss = output.sum()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update EMA
                ema.update(model)

        # Verify EMA was updated
        assert ema.updates == 100

    def test_ema_improves_gradually(self):
        """Test that EMA weights change gradually, not suddenly."""
        model = SimpleModel()

        # Initialize with specific values
        for param in model.parameters():
            param.data.fill_(1.0)

        ema = ModelEMA(model, decay=0.99, warmup_steps=0)

        # Record EMA values over time
        ema_values = []

        # Change model to different value
        for param in model.parameters():
            param.data.fill_(10.0)

        # Track EMA convergence
        for _ in range(50):
            ema.update(model)
            ema_val = list(ema.ema_model.parameters())[0].data[0, 0].item()
            ema_values.append(ema_val)

        # EMA should increase monotonically (since we're moving from 1 to 10)
        for i in range(len(ema_values) - 1):
            assert ema_values[i + 1] >= ema_values[i]

        # Should not reach target immediately
        assert ema_values[0] < 2.0  # Still close to 1.0
        # But should get closer over time
        assert ema_values[-1] > 4.0  # Moving toward 10.0 (4.5 with decay=0.99 after 50 steps)

    def test_ema_separate_from_training_model(self):
        """Test that EMA model is independent of training model."""
        model = SimpleModel()

        for param in model.parameters():
            param.data.fill_(5.0)

        ema = ModelEMA(model, decay=0.9, warmup_steps=0)

        # Update EMA
        for param in model.parameters():
            param.data.fill_(10.0)

        ema.update(model)

        # Training model should still be 10.0
        train_val = list(model.parameters())[0].data[0, 0].item()
        assert abs(train_val - 10.0) < 1e-6

        # EMA model should be between 5.0 and 10.0
        ema_val = list(ema.ema_model.parameters())[0].data[0, 0].item()
        assert 5.0 < ema_val < 10.0

    def test_warmup_then_normal_updates(self):
        """Test transition from warmup to normal updates."""
        model = SimpleModel()

        for param in model.parameters():
            param.data.fill_(1.0)

        ema = ModelEMA(model, decay=0.9, warmup_steps=5)

        # Get initial EMA weights
        initial_weight = list(ema.ema_model.parameters())[0].data.clone()

        # Change model
        for param in model.parameters():
            param.data.fill_(10.0)

        # Update during warmup (5 steps)
        for _ in range(5):
            ema.update(model)

        # EMA should be unchanged
        warmup_weight = list(ema.ema_model.parameters())[0].data
        assert torch.allclose(warmup_weight, initial_weight)

        # Update after warmup
        ema.update(model)

        # Now EMA should have changed
        after_warmup_weight = list(ema.ema_model.parameters())[0].data
        assert not torch.allclose(after_warmup_weight, initial_weight)
