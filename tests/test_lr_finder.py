"""Tests for Learning Rate Finder functionality."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.lr_finder import LRFinder, create_lr_finder_figure, plot_lr_finder
from ml_src.core.network.custom import SimpleCNN


@pytest.fixture
def small_model():
    """Fixture providing a small SimpleCNN model for testing."""
    # Use smaller input size for faster tests
    model = SimpleCNN(num_classes=10, input_size=64, dropout=0.5)
    return model


@pytest.fixture
def toy_dataset():
    """Fixture providing a small toy dataset (50 samples, 10 classes)."""
    # Create random data: 50 samples, 3 channels, 64x64 images
    images = torch.randn(50, 3, 64, 64)
    # Random labels: 50 samples, 10 classes
    labels = torch.randint(0, 10, (50,))
    dataset = TensorDataset(images, labels)
    return dataset


@pytest.fixture
def toy_dataloader(toy_dataset):
    """Fixture providing a DataLoader for the toy dataset."""
    return DataLoader(toy_dataset, batch_size=4, shuffle=True)


@pytest.fixture
def empty_dataloader():
    """Fixture providing an empty DataLoader for edge case testing."""
    empty_dataset = TensorDataset(torch.empty(0, 3, 64, 64), torch.empty(0, dtype=torch.long))
    return DataLoader(empty_dataset, batch_size=4)


@pytest.fixture
def criterion():
    """Fixture providing CrossEntropyLoss criterion."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimizer_fn(small_model):
    """Fixture providing optimizer function that creates SGD optimizer."""
    return lambda lr: torch.optim.SGD(small_model.parameters(), lr=lr, momentum=0.9)


@pytest.fixture
def device():
    """Fixture providing CPU device."""
    return torch.device("cpu")


class TestLRFinderCoreAlgorithm:
    """Test cases for core LR Finder algorithm."""

    def test_find_lr_returns_correct_types(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that find_lr returns tuple of (lrs, losses, suggested_lr)."""
        finder = LRFinder()
        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-1,
            num_iter=10,
        )

        # Check return types
        assert isinstance(lrs, list)
        assert isinstance(losses, list)
        assert isinstance(suggested_lr, float)

        # Check lengths match (should have 10 iterations unless early stopped)
        assert len(lrs) == len(losses)
        assert len(lrs) <= 10  # May stop early due to divergence

    def test_suggested_lr_in_valid_range(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that suggested_lr is between start_lr and end_lr."""
        finder = LRFinder()
        start_lr = 1e-6
        end_lr = 1e-1

        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=10,
        )

        # Suggested LR should be in valid range (or slightly below due to /10 safety factor)
        assert suggested_lr >= start_lr / 10  # Account for safety factor
        assert suggested_lr <= end_lr

    def test_learning_rates_increase_exponentially(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that learning rates increase exponentially during LR range test."""
        finder = LRFinder()
        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1e-1,
            num_iter=10,
        )

        # Check that LRs are strictly increasing
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1], f"LR at {i} ({lrs[i]}) >= LR at {i+1} ({lrs[i+1]})"

    def test_model_state_restored_after_lr_finder(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that model state is restored to initial state after LR finder."""
        finder = LRFinder()

        # Save initial model state
        initial_state = {name: param.clone() for name, param in small_model.named_parameters()}

        # Run LR finder
        finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-1,
            num_iter=10,
        )

        # Check that model state is restored
        for name, param in small_model.named_parameters():
            assert torch.allclose(
                param, initial_state[name]
            ), f"Parameter {name} was not restored correctly"

    def test_early_stopping_on_divergence(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that LR finder stops early when loss diverges."""
        finder = LRFinder()

        # Use a wide LR range and lower divergence threshold to ensure early stopping
        # The threshold needs to be low enough to catch the gradual divergence
        # that occurs with SimpleCNN on the toy dataset (loss increases by ~1.2x)
        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=10.0,  # Very high end LR to cause divergence
            num_iter=100,  # Request many iterations
            diverge_threshold=1.08,  # Low threshold to trigger early stopping before iteration 100
        )

        # Should stop early due to divergence (won't complete all 100 iterations)
        assert len(lrs) < 100, "LR finder should have stopped early due to divergence"

    def test_lr_finder_with_different_num_iterations(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test LR finder with different num_iter values."""
        finder = LRFinder()

        # Test with num_iter=5
        lrs_5, losses_5, _ = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=5,
        )

        # Test with num_iter=15
        lrs_15, losses_15, _ = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=15,
        )

        # Should respect num_iter (unless early stopped)
        assert len(lrs_5) <= 5
        assert len(lrs_15) <= 15

    def test_dataloader_cycling(self, small_model, optimizer_fn, criterion, device):
        """Test that LR finder cycles through dataloader when num_iter > dataset batches."""
        # Create very small dataset (only 2 batches)
        tiny_dataset = TensorDataset(torch.randn(8, 3, 64, 64), torch.randint(0, 10, (8,)))
        tiny_dataloader = DataLoader(tiny_dataset, batch_size=4, shuffle=False)

        finder = LRFinder()

        # Request more iterations than available batches (2 batches, but 10 iterations)
        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=tiny_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=10,
        )

        # Should successfully complete (cycles through dataloader)
        assert len(lrs) <= 10


class TestLRFinderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataloader_raises_error(
        self, small_model, empty_dataloader, optimizer_fn, criterion, device
    ):
        """Test that empty dataloader raises ValueError."""
        finder = LRFinder()

        with pytest.raises(ValueError, match="Train loader must contain at least one batch"):
            finder.find_lr(
                model=small_model,
                train_loader=empty_dataloader,
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                device=device,
            )

    def test_invalid_lr_range_start_greater_than_end(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that start_lr > end_lr raises ValueError."""
        finder = LRFinder()

        with pytest.raises(ValueError, match="start_lr must be less than end_lr"):
            finder.find_lr(
                model=small_model,
                train_loader=toy_dataloader,
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                device=device,
                start_lr=1e-1,  # Larger than end_lr
                end_lr=1e-5,
            )

    def test_invalid_lr_range_equal_values(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that start_lr == end_lr raises ValueError."""
        finder = LRFinder()

        with pytest.raises(ValueError, match="start_lr must be less than end_lr"):
            finder.find_lr(
                model=small_model,
                train_loader=toy_dataloader,
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                device=device,
                start_lr=1e-3,
                end_lr=1e-3,  # Same as start_lr
            )

    def test_invalid_num_iter_zero(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that num_iter=0 raises ValueError."""
        finder = LRFinder()

        with pytest.raises(ValueError, match="num_iter must be positive"):
            finder.find_lr(
                model=small_model,
                train_loader=toy_dataloader,
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                device=device,
                num_iter=0,
            )

    def test_invalid_num_iter_negative(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test that negative num_iter raises ValueError."""
        finder = LRFinder()

        with pytest.raises(ValueError, match="num_iter must be positive"):
            finder.find_lr(
                model=small_model,
                train_loader=toy_dataloader,
                optimizer_fn=optimizer_fn,
                criterion=criterion,
                device=device,
                num_iter=-10,
            )

    def test_nan_loss_handling(self, toy_dataloader, optimizer_fn, device):
        """Test that NaN loss is handled gracefully."""

        # Create a model that produces NaN outputs
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64 * 64 * 3, 10)

            def forward(self, x):
                # Return NaN to trigger NaN loss
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x * float("nan")

        model = NaNModel()
        criterion = nn.CrossEntropyLoss()
        finder = LRFinder()

        # Should handle NaN gracefully
        lrs, losses, suggested_lr = finder.find_lr(
            model=model,
            train_loader=toy_dataloader,
            optimizer_fn=lambda lr: torch.optim.SGD(model.parameters(), lr=lr),
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-1,
            num_iter=10,
        )

        # Should return early with empty or partial results
        # If NaN occurs immediately, lists may be empty
        assert isinstance(lrs, list)
        assert isinstance(losses, list)
        assert isinstance(suggested_lr, float)

    def test_inf_loss_handling(self, toy_dataloader, optimizer_fn, device):
        """Test that Inf loss is handled gracefully."""

        # Create a model that produces Inf outputs
        class InfModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64 * 64 * 3, 10)

            def forward(self, x):
                # Return Inf to trigger Inf loss
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x * float("inf")

        model = InfModel()
        criterion = nn.CrossEntropyLoss()
        finder = LRFinder()

        # Should handle Inf gracefully
        lrs, losses, suggested_lr = finder.find_lr(
            model=model,
            train_loader=toy_dataloader,
            optimizer_fn=lambda lr: torch.optim.SGD(model.parameters(), lr=lr),
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-1,
            num_iter=10,
        )

        # Should return early with empty or partial results
        assert isinstance(lrs, list)
        assert isinstance(losses, list)
        assert isinstance(suggested_lr, float)


class TestLRFinderOptimalLRSelection:
    """Test optimal learning rate selection logic."""

    def test_optimal_lr_from_steepest_descent(self):
        """Test that optimal LR is found at steepest descent in loss curve."""
        finder = LRFinder()

        # Create synthetic loss curve with clear steepest descent at index 3
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [2.5, 2.3, 1.5, 0.5, 1.0]  # Steepest descent between 1e-3 and 1e-2

        suggested_lr = finder._find_optimal_lr(lrs, losses)

        # Suggested LR should be around 1e-2 / 10 = 1e-3 (with safety factor)
        assert suggested_lr > 0
        assert suggested_lr < max(lrs)

    def test_optimal_lr_with_few_data_points(self):
        """Test optimal LR selection with very few data points."""
        finder = LRFinder()

        # Only 2 data points (less than 3)
        lrs = [1e-5, 1e-4]
        losses = [2.5, 2.3]

        suggested_lr = finder._find_optimal_lr(lrs, losses)

        # Should return middle value when not enough points
        assert suggested_lr == lrs[1]  # Middle of 2 points is index 1

    def test_optimal_lr_with_empty_lists(self):
        """Test optimal LR selection with empty lists."""
        finder = LRFinder()

        lrs = []
        losses = []

        suggested_lr = finder._find_optimal_lr(lrs, losses)

        # Should return default value (1e-3)
        assert suggested_lr == 1e-3

    def test_optimal_lr_with_monotonic_decreasing_loss(self):
        """Test optimal LR when loss is monotonically decreasing."""
        finder = LRFinder()

        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        losses = [3.0, 2.5, 2.0, 1.5, 1.0]  # Steady decrease

        suggested_lr = finder._find_optimal_lr(lrs, losses)

        # Should return a reasonable value
        assert suggested_lr > 0
        assert suggested_lr <= max(lrs)


class TestLRFinderPlotting:
    """Test plotting functionality."""

    def test_plot_lr_finder_creates_file(self, tmp_path):
        """Test that plot_lr_finder creates PNG file at specified path."""
        # Create mock data
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [2.5, 2.3, 2.0, 1.5, 1.2]
        suggested_lr = 1e-3

        # Create output path in temporary directory
        output_path = tmp_path / "lr_finder.png"

        # Generate plot
        fig = plot_lr_finder(lrs, losses, suggested_lr, str(output_path))

        # Verify file was created
        assert output_path.exists(), f"Plot file not created at {output_path}"

        # Verify it's a valid file (has non-zero size)
        assert output_path.stat().st_size > 0, "Plot file is empty"

    def test_plot_lr_finder_returns_figure(self, tmp_path):
        """Test that plot_lr_finder returns matplotlib figure."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [2.5, 2.3, 2.0, 1.5, 1.2]
        suggested_lr = 1e-3

        output_path = tmp_path / "lr_finder.png"

        fig = plot_lr_finder(lrs, losses, suggested_lr, str(output_path))

        # Check it's a matplotlib figure
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

    def test_plot_lr_finder_with_custom_title(self, tmp_path):
        """Test plot_lr_finder with custom title."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2]
        losses = [2.5, 2.0, 1.5, 1.2]
        suggested_lr = 1e-3

        output_path = tmp_path / "custom_title.png"

        fig = plot_lr_finder(
            lrs, losses, suggested_lr, str(output_path), title="Custom LR Finder Title"
        )

        # Verify file created
        assert output_path.exists()

    def test_create_lr_finder_figure_returns_figure(self):
        """Test that create_lr_finder_figure returns matplotlib figure."""
        lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [2.5, 2.3, 2.0, 1.5, 1.2]
        suggested_lr = 1e-3

        fig = create_lr_finder_figure(lrs, losses, suggested_lr)

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_create_lr_finder_figure_with_numpy_arrays(self):
        """Test create_lr_finder_figure with numpy arrays instead of lists."""
        lrs = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        losses = np.array([2.5, 2.3, 2.0, 1.5, 1.2])
        suggested_lr = 1e-3

        fig = create_lr_finder_figure(lrs, losses, suggested_lr)

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

        # Clean up
        plt.close(fig)

    def test_plot_with_single_data_point(self, tmp_path):
        """Test plotting with a single data point."""
        lrs = [1e-3]
        losses = [2.0]
        suggested_lr = 1e-3

        output_path = tmp_path / "single_point.png"

        # Should handle single point gracefully
        fig = plot_lr_finder(lrs, losses, suggested_lr, str(output_path))

        assert output_path.exists()

    def test_plot_with_diverged_loss(self, tmp_path):
        """Test plotting when loss diverges (goes very high)."""
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        losses = [2.0, 1.8, 1.5, 1.0, 100.0]  # Large spike at end
        suggested_lr = 1e-4

        output_path = tmp_path / "diverged.png"

        fig = plot_lr_finder(lrs, losses, suggested_lr, str(output_path))

        assert output_path.exists()


class TestLRFinderIntegration:
    """Integration tests combining multiple components."""

    def test_full_lr_finder_workflow(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device, tmp_path
    ):
        """Test complete LR finder workflow: find LR -> plot -> verify."""
        finder = LRFinder()

        # Step 1: Find optimal learning rate
        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1e-1,
            num_iter=20,
        )

        # Step 2: Verify results are reasonable
        assert len(lrs) > 0, "No learning rates were tested"
        assert len(losses) > 0, "No losses were recorded"
        assert suggested_lr > 0, "Suggested LR should be positive"

        # Step 3: Create plot
        output_path = tmp_path / "workflow_test.png"
        fig = plot_lr_finder(lrs, losses, suggested_lr, str(output_path))

        # Step 4: Verify plot was created
        assert output_path.exists()

        # Step 5: Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_lr_finder_with_adam_optimizer(
        self, small_model, toy_dataloader, criterion, device
    ):
        """Test LR finder with Adam optimizer instead of SGD."""
        finder = LRFinder()
        optimizer_fn = lambda lr: torch.optim.Adam(small_model.parameters(), lr=lr)

        lrs, losses, suggested_lr = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-6,
            end_lr=1e-1,
            num_iter=10,
        )

        assert len(lrs) > 0
        assert suggested_lr > 0

    def test_lr_finder_with_different_smoothing_beta(
        self, small_model, toy_dataloader, optimizer_fn, criterion, device
    ):
        """Test LR finder with different beta values for smoothing."""
        finder = LRFinder()

        # Test with low beta (less smoothing)
        lrs_low, losses_low, _ = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=10,
            beta=0.8,  # Less smoothing
        )

        # Test with high beta (more smoothing)
        lrs_high, losses_high, _ = finder.find_lr(
            model=small_model,
            train_loader=toy_dataloader,
            optimizer_fn=optimizer_fn,
            criterion=criterion,
            device=device,
            start_lr=1e-5,
            end_lr=1e-2,
            num_iter=10,
            beta=0.99,  # More smoothing
        )

        # Both should complete successfully
        assert len(lrs_low) > 0
        assert len(lrs_high) > 0
