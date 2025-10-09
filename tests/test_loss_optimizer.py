"""Comprehensive tests for loss functions and optimizers (factories, all types)."""

import torch
import torch.nn as nn

from ml_src.core.losses import get_criterion
from ml_src.core.optimizer import get_optimizer, get_scheduler


# ============================================================================
# Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Loss Function Tests
# ============================================================================


class TestGetCriterion:
    """Tests for get_criterion function."""

    def test_get_criterion_returns_criterion(self):
        """Test that get_criterion returns a loss criterion."""
        criterion = get_criterion()

        assert criterion is not None
        assert isinstance(criterion, nn.Module)

    def test_get_criterion_default_is_cross_entropy(self):
        """Test that default criterion is CrossEntropyLoss."""
        criterion = get_criterion()

        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_get_criterion_with_none_config(self):
        """Test that get_criterion works with None config."""
        criterion = get_criterion(config=None)

        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_get_criterion_with_empty_config(self):
        """Test that get_criterion works with empty config."""
        criterion = get_criterion(config={})

        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_criterion_forward_pass(self):
        """Test that criterion can be used in forward pass."""
        criterion = get_criterion()

        # Create dummy predictions and targets
        predictions = torch.randn(4, 5)  # Batch size 4, 5 classes
        targets = torch.tensor([0, 1, 2, 3])

        # Should not raise error
        loss = criterion(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative

    def test_criterion_computes_loss(self):
        """Test that criterion computes reasonable loss values."""
        criterion = get_criterion()

        # Perfect predictions (one-hot)
        predictions = torch.zeros(4, 5)
        predictions[0, 0] = 10.0  # High confidence for class 0
        predictions[1, 1] = 10.0  # High confidence for class 1
        predictions[2, 2] = 10.0
        predictions[3, 3] = 10.0
        targets = torch.tensor([0, 1, 2, 3])

        loss = criterion(predictions, targets)

        # Loss should be very small for perfect predictions
        assert loss.item() < 0.1

    def test_criterion_different_batch_sizes(self):
        """Test criterion with different batch sizes."""
        criterion = get_criterion()

        for batch_size in [1, 4, 16, 32]:
            predictions = torch.randn(batch_size, 10)
            targets = torch.randint(0, 10, (batch_size,))

            loss = criterion(predictions, targets)

            assert isinstance(loss, torch.Tensor)
            assert loss.item() >= 0


# ============================================================================
# Optimizer Tests
# ============================================================================


class TestGetOptimizer:
    """Tests for get_optimizer function."""

    def test_get_optimizer_returns_optimizer(self):
        """Test that get_optimizer returns an optimizer."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001, "momentum": 0.9}}

        optimizer = get_optimizer(model, config)

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_get_optimizer_default_is_sgd(self):
        """Test that default optimizer is SGD."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001, "momentum": 0.9}}

        optimizer = get_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)

    def test_get_optimizer_uses_config_lr(self):
        """Test that optimizer uses learning rate from config."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.01, "momentum": 0.9}}

        optimizer = get_optimizer(model, config)

        # Check learning rate
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_get_optimizer_uses_config_momentum(self):
        """Test that optimizer uses momentum from config."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001, "momentum": 0.95}}

        optimizer = get_optimizer(model, config)

        # Check momentum
        assert optimizer.param_groups[0]["momentum"] == 0.95

    def test_get_optimizer_default_momentum(self):
        """Test that optimizer uses default momentum if not specified."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}}  # No momentum specified

        optimizer = get_optimizer(model, config)

        # Should use default momentum (0.9)
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_get_optimizer_different_lr_values(self):
        """Test optimizer with different learning rates."""
        model = SimpleModel()

        for lr in [0.0001, 0.001, 0.01, 0.1]:
            config = {"optimizer": {"lr": lr, "momentum": 0.9}}
            optimizer = get_optimizer(model, config)

            assert optimizer.param_groups[0]["lr"] == lr

    def test_optimizer_step_updates_parameters(self):
        """Test that optimizer step updates model parameters."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.01, "momentum": 0.9}}
        optimizer = get_optimizer(model, config)

        # Get initial parameter values
        initial_params = [p.clone() for p in model.parameters()]

        # Forward pass and backward pass
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Parameters should have changed
        for initial_param, current_param in zip(initial_params, model.parameters()):
            assert not torch.allclose(initial_param, current_param)

    def test_optimizer_zero_grad(self):
        """Test that optimizer zero_grad clears gradients."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.01, "momentum": 0.9}}
        optimizer = get_optimizer(model, config)

        # Create gradients
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None

        # Zero gradients
        optimizer.zero_grad()

        # Gradients should be None or zero
        for param in model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))


# ============================================================================
# Scheduler Tests
# ============================================================================


class TestGetScheduler:
    """Tests for get_scheduler function."""

    def test_get_scheduler_returns_scheduler(self):
        """Test that get_scheduler returns a scheduler."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"step_size": 7, "gamma": 0.1}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler)

    def test_get_scheduler_default_is_steplr(self):
        """Test that default scheduler is StepLR."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"step_size": 7, "gamma": 0.1}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_get_scheduler_uses_config_step_size(self):
        """Test that scheduler uses step_size from config."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"step_size": 10, "gamma": 0.1}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert scheduler.step_size == 10

    def test_get_scheduler_uses_config_gamma(self):
        """Test that scheduler uses gamma from config."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"step_size": 7, "gamma": 0.5}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert scheduler.gamma == 0.5

    def test_get_scheduler_default_step_size(self):
        """Test that scheduler uses default step_size if not specified."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"gamma": 0.1}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert scheduler.step_size == 7  # Default

    def test_get_scheduler_default_gamma(self):
        """Test that scheduler uses default gamma if not specified."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.001}, "scheduler": {"step_size": 7}}
        optimizer = get_optimizer(model, config)

        scheduler = get_scheduler(optimizer, config)

        assert scheduler.gamma == 0.1  # Default

    def test_scheduler_step_reduces_lr(self):
        """Test that scheduler step reduces learning rate after step_size epochs."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 0.1}, "scheduler": {"step_size": 3, "gamma": 0.1}}
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 0.1

        # Step through 3 epochs (before step_size)
        for _ in range(3):
            scheduler.step()

        # LR should have been reduced by gamma
        new_lr = optimizer.param_groups[0]["lr"]
        assert abs(new_lr - 0.01) < 1e-6  # 0.1 * 0.1 = 0.01

    def test_scheduler_multiple_steps(self):
        """Test scheduler reduces LR multiple times."""
        model = SimpleModel()
        config = {"optimizer": {"lr": 1.0}, "scheduler": {"step_size": 2, "gamma": 0.5}}
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # After 2 steps: 1.0 * 0.5 = 0.5
        for _ in range(2):
            scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.5) < 1e-6

        # After 4 steps total: 0.5 * 0.5 = 0.25
        for _ in range(2):
            scheduler.step()
        assert abs(optimizer.param_groups[0]["lr"] - 0.25) < 1e-6


# ============================================================================
# Integration Tests
# ============================================================================


class TestLossOptimizerIntegration:
    """Integration tests for loss, optimizer, and scheduler together."""

    def test_full_training_step(self):
        """Test full training step with loss, optimizer, and scheduler."""
        model = SimpleModel()
        config = {
            "optimizer": {"lr": 0.01, "momentum": 0.9},
            "scheduler": {"step_size": 5, "gamma": 0.1},
        }

        criterion = get_criterion()
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # Training step
        x = torch.randn(4, 10)
        target = torch.randint(0, 5, (4,))

        output = model(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Should not raise errors
        assert loss.item() >= 0

    def test_training_loop_simulation(self):
        """Test simulated training loop."""
        model = SimpleModel()
        config = {
            "optimizer": {"lr": 0.1, "momentum": 0.9},
            "scheduler": {"step_size": 2, "gamma": 0.5},
        }

        criterion = get_criterion()
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        initial_lr = optimizer.param_groups[0]["lr"]
        losses = []

        # Simulate 6 epochs
        for epoch in range(6):
            epoch_losses = []

            # Simulate 5 batches per epoch
            for _ in range(5):
                x = torch.randn(4, 10)
                target = torch.randint(0, 5, (4,))

                output = model(x)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            losses.append(sum(epoch_losses) / len(epoch_losses))
            scheduler.step()

        # LR should have been reduced
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_optimizer_and_scheduler_compatibility(self):
        """Test that optimizer and scheduler work together."""
        model = SimpleModel()
        config = {
            "optimizer": {"lr": 0.01, "momentum": 0.9},
            "scheduler": {"step_size": 3, "gamma": 0.1},
        }

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # Should work together without errors
        for _ in range(10):
            # Simulate training step
            x = torch.randn(4, 10)
            output = model(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Scheduler step (once per epoch)
            scheduler.step()

        # Verify LR has been adjusted
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr != 0.01  # Should have changed

    def test_criterion_optimizer_scheduler_state_dicts(self):
        """Test that optimizer and scheduler state dicts can be saved/loaded."""
        model = SimpleModel()
        config = {
            "optimizer": {"lr": 0.01, "momentum": 0.9},
            "scheduler": {"step_size": 5, "gamma": 0.1},
        }

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # Perform some training
        for _ in range(3):
            x = torch.randn(4, 10)
            output = model(x)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Save state dicts
        opt_state = optimizer.state_dict()
        sched_state = scheduler.state_dict()

        # Create new instances
        model2 = SimpleModel()
        optimizer2 = get_optimizer(model2, config)
        scheduler2 = get_scheduler(optimizer2, config)

        # Load state dicts
        optimizer2.load_state_dict(opt_state)
        scheduler2.load_state_dict(sched_state)

        # States should match
        assert optimizer2.state_dict() == opt_state
        assert scheduler2.state_dict() == sched_state
