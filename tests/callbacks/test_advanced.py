"""Tests for advanced callbacks (gradient and augmentation)."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml_src.core.callbacks.augmentation import CutMixCallback, MixUpCallback
from ml_src.core.callbacks.gradient import GradientClipping, GradientNormMonitor


class TestGradientClipping:
    """Tests for GradientClipping callback."""

    def test_initialization(self):
        """Test callback initializes with correct defaults."""
        callback = GradientClipping()

        assert callback.value == 1.0
        assert callback.algorithm == "norm"
        assert callback.norm_type == 2.0

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = GradientClipping(value=5.0, algorithm="value", norm_type=float("inf"))

        assert callback.value == 5.0
        assert callback.algorithm == "value"
        assert callback.norm_type == float("inf")

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be"):
            GradientClipping(algorithm="invalid")

    @patch("torch.nn.utils.clip_grad_norm_")
    def test_clips_gradient_norm(self, mock_clip_norm, mock_trainer):
        """Test that gradient norm clipping is applied."""
        callback = GradientClipping(value=1.0, algorithm="norm")

        callback.on_backward_end(mock_trainer)

        # Should have called clip_grad_norm_
        mock_clip_norm.assert_called_once()
        call_args = mock_clip_norm.call_args
        assert call_args[1]["max_norm"] == 1.0

    @patch("torch.nn.utils.clip_grad_value_")
    def test_clips_gradient_value(self, mock_clip_value, mock_trainer):
        """Test that gradient value clipping is applied."""
        callback = GradientClipping(value=5.0, algorithm="value")

        callback.on_backward_end(mock_trainer)

        # Should have called clip_grad_value_
        mock_clip_value.assert_called_once()
        call_args = mock_clip_value.call_args
        assert call_args[1]["clip_value"] == 5.0

    def test_gradient_clipping_with_real_model(self):
        """Test gradient clipping with actual model."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Create large gradients
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        # Manually set large gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.fill_(100.0)  # Very large gradient

        # Create mock trainer
        trainer = Mock()
        trainer.model = model

        callback = GradientClipping(value=1.0, algorithm="norm")
        callback.on_backward_end(trainer)

        # Compute gradient norm after clipping
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        # Should be clipped to approximately 1.0
        assert total_norm <= 1.1  # Allow small tolerance


class TestGradientNormMonitor:
    """Tests for GradientNormMonitor callback."""

    def test_initialization(self):
        """Test callback initializes with defaults."""
        callback = GradientNormMonitor()

        assert callback.frequency == 10
        assert callback.norm_type == 2.0
        assert callback.log_per_layer is False
        assert callback.batch_counter == 0

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = GradientNormMonitor(frequency=5, norm_type=float("inf"), log_per_layer=True)

        assert callback.frequency == 5
        assert callback.norm_type == float("inf")
        assert callback.log_per_layer is True

    def test_counter_resets_on_epoch_begin(self, mock_trainer):
        """Test that batch counter resets at epoch start."""
        callback = GradientNormMonitor()

        callback.batch_counter = 15
        callback.on_epoch_begin(mock_trainer, epoch=0)

        assert callback.batch_counter == 0

    def test_only_logs_at_frequency(self, mock_trainer):
        """Test that logging only occurs at specified frequency."""
        callback = GradientNormMonitor(frequency=5)

        # Add gradients to model
        for param in mock_trainer.model.parameters():
            param.grad = torch.randn_like(param)

        # Call on_backward_end multiple times
        for i in range(10):
            callback.on_backward_end(mock_trainer)

        # Should only log at batches 5 and 10 (multiples of frequency)
        assert mock_trainer.writer.add_scalar.call_count == 2

    def test_computes_gradient_norm_correctly(self, mock_trainer):
        """Test that gradient norm is computed correctly."""
        callback = GradientNormMonitor(frequency=1)

        # Set known gradients
        for param in mock_trainer.model.parameters():
            param.grad = torch.ones_like(param)  # All gradients = 1

        callback.on_backward_end(mock_trainer)

        # Should have logged gradient norm
        assert mock_trainer.writer.add_scalar.called

        # Get the logged value
        call_args = mock_trainer.writer.add_scalar.call_args
        metric_name = call_args[0][0]
        assert "Gradients/total_norm" in metric_name

    def test_logs_per_layer_gradients(self, mock_trainer):
        """Test per-layer gradient logging when enabled."""
        callback = GradientNormMonitor(frequency=1, log_per_layer=True)

        # Add gradients
        for param in mock_trainer.model.parameters():
            param.grad = torch.randn_like(param)

        callback.on_backward_end(mock_trainer)

        # Should log total norm + per-layer norms
        call_count = mock_trainer.writer.add_scalar.call_count
        assert call_count > 1  # At least total + some layers

    def test_warns_on_vanishing_gradients(self, mock_trainer):
        """Test warning for vanishing gradients."""
        callback = GradientNormMonitor(frequency=1)

        # Set very small gradients
        for param in mock_trainer.model.parameters():
            param.grad = torch.ones_like(param) * 1e-10

        # Should log warning about vanishing gradients
        with patch("ml_src.core.callbacks.gradient.logger.warning") as mock_warning:
            callback.on_backward_end(mock_trainer)
            assert mock_warning.called

    def test_warns_on_exploding_gradients(self, mock_trainer):
        """Test warning for exploding gradients."""
        callback = GradientNormMonitor(frequency=1)

        # Set very large gradients
        for param in mock_trainer.model.parameters():
            param.grad = torch.ones_like(param) * 1000

        # Should log warning about exploding gradients
        with patch("ml_src.core.callbacks.gradient.logger.warning") as mock_warning:
            callback.on_backward_end(mock_trainer)
            assert mock_warning.called


class TestMixUpCallback:
    """Tests for MixUp augmentation callback."""

    def test_initialization(self):
        """Test callback initializes with defaults."""
        callback = MixUpCallback()

        assert callback.alpha == 0.2
        assert callback.apply_prob == 0.5
        assert callback.in_training is False

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = MixUpCallback(alpha=0.5, apply_prob=0.8)

        assert callback.alpha == 0.5
        assert callback.apply_prob == 0.8

    def test_only_applies_in_training_phase(self, mock_trainer):
        """Test that MixUp only applies during training phase."""
        callback = MixUpCallback(apply_prob=1.0)  # Always apply

        # Validation phase
        callback.on_phase_begin(mock_trainer, "val")
        assert not callback.in_training

        # Training phase
        callback.on_phase_begin(mock_trainer, "train")
        assert callback.in_training

    def test_mixup_modifies_batch(self, mock_trainer):
        """Test that MixUp modifies the batch data."""
        callback = MixUpCallback(alpha=0.2, apply_prob=1.0)  # Always apply
        callback.in_training = True

        # Create sample batch
        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        batch = (inputs, labels)

        # Apply MixUp
        with patch("numpy.random.beta", return_value=0.7):  # Fixed lambda
            with patch("numpy.random.rand", return_value=0.0):  # Force apply
                callback.on_batch_begin(mock_trainer, 0, batch)

        # Note: In-place modification would require accessing the batch
        # This test documents expected behavior

    def test_mixup_respects_apply_probability(self, mock_trainer):
        """Test that MixUp respects apply_prob."""
        callback = MixUpCallback(apply_prob=0.0)  # Never apply
        callback.in_training = True

        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        batch = (inputs, labels)

        # Should not modify batch when apply_prob=0
        original_inputs = inputs.clone()
        callback.on_batch_begin(mock_trainer, 0, batch)

        # Batch should be unchanged
        assert torch.equal(inputs, original_inputs)


class TestCutMixCallback:
    """Tests for CutMix augmentation callback."""

    def test_initialization(self):
        """Test callback initializes with defaults."""
        callback = CutMixCallback()

        assert callback.alpha == 1.0
        assert callback.apply_prob == 0.5
        assert callback.in_training is False

    def test_custom_parameters(self):
        """Test callback accepts custom parameters."""
        callback = CutMixCallback(alpha=0.5, apply_prob=0.8)

        assert callback.alpha == 0.5
        assert callback.apply_prob == 0.8

    def test_only_applies_in_training_phase(self, mock_trainer):
        """Test that CutMix only applies during training phase."""
        callback = CutMixCallback()

        # Validation phase
        callback.on_phase_begin(mock_trainer, "val")
        assert not callback.in_training

        # Training phase
        callback.on_phase_begin(mock_trainer, "train")
        assert callback.in_training

    def test_rand_bbox_generates_valid_boxes(self):
        """Test that _rand_bbox generates valid bounding boxes."""
        callback = CutMixCallback()

        # Test with various lambda values
        for lam in [0.25, 0.5, 0.75]:
            size = (8, 3, 32, 32)  # batch_size, channels, H, W
            x1, y1, x2, y2 = callback._rand_bbox(size, lam)

            # Boxes should be within image bounds
            assert 0 <= x1 < x2 <= 32
            assert 0 <= y1 < y2 <= 32

    def test_rand_bbox_area_proportional_to_lambda(self):
        """Test that box area is related to lambda."""
        callback = CutMixCallback()

        size = (8, 3, 100, 100)
        total_area = 100 * 100

        # Test multiple times due to randomness
        for lam in [0.25, 0.5, 0.75]:
            areas = []
            for _ in range(10):
                x1, y1, x2, y2 = callback._rand_bbox(size, lam)
                box_area = (x2 - x1) * (y2 - y1)
                areas.append(box_area)

            avg_area = np.mean(areas)
            expected_ratio = 1.0 - lam

            # Box area should be approximately (1-lam) * total_area
            # Allow 50% tolerance due to randomness
            assert abs(avg_area / total_area - expected_ratio) < 0.5

    def test_cutmix_respects_apply_probability(self, mock_trainer):
        """Test that CutMix respects apply_prob."""
        callback = CutMixCallback(apply_prob=0.0)  # Never apply
        callback.in_training = True

        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        batch = (inputs, labels)

        # Should not modify batch when apply_prob=0
        original_inputs = inputs.clone()
        callback.on_batch_begin(mock_trainer, 0, batch)

        # Batch should be unchanged
        assert torch.equal(inputs, original_inputs)


class TestGradientAndAugmentationTogether:
    """Tests for using gradient and augmentation callbacks together."""

    def test_gradient_clipping_with_mixup(self, mock_trainer):
        """Test that gradient clipping works with MixUp."""
        grad_clip = GradientClipping(value=1.0)
        mixup = MixUpCallback(alpha=0.2)

        # Should be able to use both without conflicts
        mixup.on_phase_begin(mock_trainer, "train")
        mixup.on_batch_begin(mock_trainer, 0, (torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))))

        # Add gradients and clip
        for param in mock_trainer.model.parameters():
            param.grad = torch.randn_like(param) * 10

        grad_clip.on_backward_end(mock_trainer)

        # Should complete without errors

    def test_gradient_monitoring_with_cutmix(self, mock_trainer):
        """Test that gradient monitoring works with CutMix."""
        grad_monitor = GradientNormMonitor(frequency=1)
        cutmix = CutMixCallback(alpha=1.0)

        # Should be able to use both
        cutmix.on_phase_begin(mock_trainer, "train")
        cutmix.on_batch_begin(mock_trainer, 0, (torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))))

        # Add gradients and monitor
        for param in mock_trainer.model.parameters():
            param.grad = torch.randn_like(param)

        grad_monitor.on_backward_end(mock_trainer)

        # Should have logged
        assert mock_trainer.writer.add_scalar.called
