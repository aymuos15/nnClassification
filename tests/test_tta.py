"""Tests for Test-Time Augmentation (TTA) functionality."""

import torch

from ml_src.core.transforms.tta import (
    TTATransform,
    aggregate_predictions,
    get_tta_transforms,
)


def test_tta_transform_horizontal_flip():
    """Test TTA with horizontal flip augmentation."""
    tta = TTATransform(augmentations=["horizontal_flip"])

    # Create dummy image (3, 64, 64)
    image = torch.randn(3, 64, 64)

    # Apply TTA
    augmented = tta.apply(image)

    # Should return original + h_flip = 2 images
    assert len(augmented) == 2
    assert augmented[0].shape == (3, 64, 64)
    assert augmented[1].shape == (3, 64, 64)


def test_tta_transform_multiple_augmentations():
    """Test TTA with multiple augmentations."""
    tta = TTATransform(augmentations=["horizontal_flip", "vertical_flip"])

    # Create dummy image
    image = torch.randn(3, 64, 64)

    # Apply TTA
    augmented = tta.apply(image)

    # Should return: original + h_flip + v_flip + h+v_flip = 4 images
    assert len(augmented) == 4


def test_tta_transform_rotations():
    """Test TTA with rotation augmentations."""
    tta = TTATransform(augmentations=["rotate_90", "rotate_180", "rotate_270"])

    image = torch.randn(3, 64, 64)
    augmented = tta.apply(image)

    # Original + 3 rotations = 4 images
    assert len(augmented) == 4


def test_aggregate_predictions_mean():
    """Test mean aggregation of predictions."""
    # Create dummy predictions (3 augmentations, batch_size=2, num_classes=3)
    pred1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pred2 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    pred3 = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    predictions = [pred1, pred2, pred3]

    # Aggregate with mean
    result = aggregate_predictions(predictions, method="mean")

    # Expected: average of all predictions
    expected = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    assert torch.allclose(result, expected)


def test_aggregate_predictions_max():
    """Test max aggregation of predictions."""
    pred1 = torch.tensor([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
    pred2 = torch.tensor([[3.0, 2.0, 6.0], [2.0, 5.0, 1.0]])

    predictions = [pred1, pred2]

    result = aggregate_predictions(predictions, method="max")

    # Expected: element-wise maximum
    expected = torch.tensor([[3.0, 5.0, 6.0], [3.0, 5.0, 4.0]])

    assert torch.allclose(result, expected)


def test_aggregate_predictions_voting():
    """Test voting aggregation of predictions."""
    # Predictions where class indices differ
    # pred1: classes [2, 0]
    # pred2: classes [2, 1]
    # pred3: classes [2, 0]
    # Expected vote: [2, 0] (majority for each sample)

    pred1 = torch.tensor([[1.0, 2.0, 5.0], [3.0, 1.0, 2.0]])  # argmax: [2, 0]
    pred2 = torch.tensor([[2.0, 1.0, 4.0], [1.0, 3.0, 2.0]])  # argmax: [2, 1]
    pred3 = torch.tensor([[1.0, 3.0, 6.0], [4.0, 2.0, 1.0]])  # argmax: [2, 0]

    predictions = [pred1, pred2, pred3]

    result = aggregate_predictions(predictions, method="voting")

    # Get final predictions
    final_preds = result.argmax(dim=1)

    # Expected: class 2 for first sample, class 0 for second sample
    expected = torch.tensor([2, 0])

    assert torch.equal(final_preds, expected)


def test_get_tta_transforms():
    """Test TTA transform factory function."""
    # Test with list of augmentations
    tta = get_tta_transforms(["horizontal_flip"])
    assert isinstance(tta, TTATransform)
    assert tta.augmentations == ["horizontal_flip"]

    # Test with 'default'
    tta_default = get_tta_transforms("default")
    assert isinstance(tta_default, TTATransform)
    assert tta_default.augmentations == ["horizontal_flip"]


def test_tta_invalid_augmentation():
    """Test that invalid augmentation raises error."""
    try:
        TTATransform(augmentations=["invalid_augmentation"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown augmentation" in str(e)
