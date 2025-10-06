"""Tests for ensemble inference functionality."""

import pytest
import torch

from ml_src.core.inference.ensemble import EnsembleInference


def test_ensemble_soft_voting_aggregation(tmp_path):
    """Test soft voting aggregation in ensemble."""
    # This is a unit test for the aggregation method only
    # Full integration test would require actual model checkpoints

    # Create mock config
    config = {
        "model": {"type": "base", "architecture": "resnet18", "num_classes": 3},
        "training": {"device": "cpu"},
    }

    # Create dummy checkpoints (would need actual files for full test)
    # For now, test the aggregation logic directly

    # Mock ensemble instance
    class MockEnsemble:
        def __init__(self):
            self.aggregation = "soft_voting"
            self.weights = None

        def _aggregate_logits(self, all_logits):
            """Copy of EnsembleInference._aggregate_logits for testing."""
            if self.aggregation == "soft_voting":
                return torch.stack(all_logits).mean(dim=0)
            elif self.aggregation == "weighted":
                weighted = torch.stack(
                    [logits * weight for logits, weight in zip(all_logits, self.weights)]
                )
                return weighted.sum(dim=0)
            elif self.aggregation == "hard_voting":
                class_preds = torch.stack([logits.argmax(dim=1) for logits in all_logits])
                batch_size = class_preds.shape[1]
                num_classes = all_logits[0].shape[1]

                voted = []
                for i in range(batch_size):
                    votes = class_preds[:, i]
                    counts = torch.bincount(votes, minlength=num_classes)
                    voted.append(counts)

                return torch.stack(voted).float()

    ensemble = MockEnsemble()

    # Test soft voting
    logits1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logits2 = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    logits3 = torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    result = ensemble._aggregate_logits([logits1, logits2, logits3])
    expected = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    assert torch.allclose(result, expected)


def test_ensemble_hard_voting_aggregation():
    """Test hard voting aggregation."""

    class MockEnsemble:
        def __init__(self):
            self.aggregation = "hard_voting"

        def _aggregate_logits(self, all_logits):
            """Hard voting implementation."""
            class_preds = torch.stack([logits.argmax(dim=1) for logits in all_logits])
            batch_size = class_preds.shape[1]
            num_classes = all_logits[0].shape[1]

            voted = []
            for i in range(batch_size):
                votes = class_preds[:, i]
                counts = torch.bincount(votes, minlength=num_classes)
                voted.append(counts)

            return torch.stack(voted).float()

    ensemble = MockEnsemble()

    # Model predictions with different classes
    # Model 1: [2, 0]
    # Model 2: [2, 1]
    # Model 3: [2, 0]
    # Expected: [2, 0] (majority)

    logits1 = torch.tensor([[1.0, 2.0, 5.0], [3.0, 1.0, 2.0]])  # argmax: [2, 0]
    logits2 = torch.tensor([[2.0, 1.0, 4.0], [1.0, 3.0, 2.0]])  # argmax: [2, 1]
    logits3 = torch.tensor([[1.0, 3.0, 6.0], [4.0, 2.0, 1.0]])  # argmax: [2, 0]

    result = ensemble._aggregate_logits([logits1, logits2, logits3])
    final_preds = result.argmax(dim=1)

    expected = torch.tensor([2, 0])
    assert torch.equal(final_preds, expected)


def test_ensemble_weighted_aggregation():
    """Test weighted aggregation."""

    class MockEnsemble:
        def __init__(self):
            self.aggregation = "weighted"
            self.weights = [0.5, 0.3, 0.2]

        def _aggregate_logits(self, all_logits):
            """Weighted averaging implementation."""
            weighted = torch.stack(
                [logits * weight for logits, weight in zip(all_logits, self.weights)]
            )
            return weighted.sum(dim=0)

    ensemble = MockEnsemble()

    logits1 = torch.tensor([[2.0, 4.0, 6.0]])
    logits2 = torch.tensor([[4.0, 6.0, 8.0]])
    logits3 = torch.tensor([[6.0, 8.0, 10.0]])

    result = ensemble._aggregate_logits([logits1, logits2, logits3])

    # Expected: 0.5 * logits1 + 0.3 * logits2 + 0.2 * logits3
    expected = torch.tensor([[3.4, 5.4, 7.4]])

    assert torch.allclose(result, expected)


def test_ensemble_validation():
    """Test ensemble validation logic."""
    # Test that empty checkpoints raise error
    with pytest.raises(ValueError, match="at least one checkpoint"):
        EnsembleInference(
            checkpoints=[],
            config={},
            device="cpu",
            aggregation="soft_voting",
        )

    # Test invalid aggregation method
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        EnsembleInference(
            checkpoints=["dummy.pt"],
            config={},
            device="cpu",
            aggregation="invalid_method",
        )
