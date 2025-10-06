"""Tests for inference strategies and factory."""


import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.inference import (
    MixedPrecisionInference,
    StandardInference,
    get_inference_strategy,
)
from ml_src.core.test import evaluate_model

# Check if accelerate package is available
try:
    import accelerate  # noqa

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False

# Conditional import for AccelerateInference (only if accelerate is available)
if _ACCELERATE_AVAILABLE:
    from ml_src.core.inference import AccelerateInference


def test_standard_inference():
    """Test StandardInference directly."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset (8 samples, 2 classes)
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    # Ensure we have both classes
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Run inference
    strategy = StandardInference()
    test_acc, per_sample_results = strategy.run_inference(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=None,
    )

    # Verify accuracy is computed
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0

    # Verify per_sample_results format
    assert len(per_sample_results) == num_samples
    for true_label, pred_label, is_correct in per_sample_results:
        assert isinstance(true_label, int)
        assert isinstance(pred_label, int)
        assert isinstance(is_correct, bool)
        assert true_label in [0, 1]
        assert pred_label in [0, 1]


def test_standard_inference_with_class_names():
    """Test StandardInference with class names."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    class_names = ["class_0", "class_1"]

    # Run inference
    strategy = StandardInference()
    test_acc, per_sample_results = strategy.run_inference(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=class_names,
    )

    # Verify per_sample_results format with class names
    assert len(per_sample_results) == num_samples
    for true_label, pred_label, is_correct in per_sample_results:
        assert isinstance(true_label, str)
        assert isinstance(pred_label, str)
        assert isinstance(is_correct, bool)
        assert true_label in class_names
        assert pred_label in class_names


def test_inference_factory_standard():
    """Test factory with 'standard' strategy."""
    # Create config with standard strategy
    config = {"inference": {"strategy": "standard"}}

    # Get strategy
    strategy = get_inference_strategy(config)

    # Verify it's StandardInference
    assert isinstance(strategy, StandardInference)

    # Test it works
    model = nn.Sequential(nn.Linear(10, 2))
    device = torch.device("cpu")
    X_test = torch.randn(8, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    test_acc, per_sample_results = strategy.run_inference(
        model, test_loader, 8, device, class_names=["class_0", "class_1"]
    )

    assert test_acc is not None
    assert len(per_sample_results) == 8


def test_inference_factory_default():
    """Test that factory defaults to StandardInference when strategy is omitted."""
    # Create config WITHOUT inference field
    config = {}

    # Should default to StandardInference
    strategy = get_inference_strategy(config)
    assert isinstance(strategy, StandardInference)


def test_inference_factory_invalid_strategy():
    """Test that factory raises error for invalid strategy."""
    # Create config with invalid strategy
    config = {"inference": {"strategy": "invalid_strategy"}}

    # Should raise ValueError
    with pytest.raises(ValueError, match="Unknown inference strategy"):
        get_inference_strategy(config)


def test_inference_backward_compatibility():
    """Test that test_model() still works for backward compatibility."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Use evaluate_model function
    test_acc, per_sample_results = evaluate_model(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=["class_0", "class_1"],
    )

    # Verify it works
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0
    assert len(per_sample_results) == num_samples

    # Verify format with class names
    for true_label, pred_label, is_correct in per_sample_results:
        assert isinstance(true_label, str)
        assert isinstance(pred_label, str)
        assert isinstance(is_correct, bool)


def test_inference_determinism():
    """Test that inference produces consistent results with same model/data."""
    # Create model and freeze weights
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    model.eval()

    # Freeze model to ensure same predictions
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cpu")

    # Create dataset
    num_samples = 8
    torch.manual_seed(42)  # Seed for reproducible data
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    strategy = StandardInference()

    # Run inference twice
    acc1, results1 = strategy.run_inference(model, test_loader, num_samples, device)

    # Recreate loader to ensure fresh iteration
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    acc2, results2 = strategy.run_inference(model, test_loader, num_samples, device)

    # Results should be identical
    assert acc1.item() == acc2.item()
    assert results1 == results2


# ============================================================================
# Mixed Precision Inference Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_inference_cuda():
    """Test MixedPrecisionInference with float16 on CUDA."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cuda:0")
    model = model.to(device)

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Test float16
    strategy = MixedPrecisionInference(amp_dtype=torch.float16)
    test_acc, per_sample_results = strategy.run_inference(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=["class_0", "class_1"],
    )

    # Verify results
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0
    assert len(per_sample_results) == num_samples


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_inference_bfloat16():
    """Test MixedPrecisionInference with bfloat16 on CUDA."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cuda:0")
    model = model.to(device)

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Test bfloat16
    strategy = MixedPrecisionInference(amp_dtype=torch.bfloat16)
    test_acc, per_sample_results = strategy.run_inference(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=None,
    )

    # Verify results
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0
    assert len(per_sample_results) == num_samples


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_vs_standard_results():
    """Test that mixed precision produces similar results to standard inference."""
    # Create model and freeze weights for deterministic results
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0")
    model = model.to(device)

    # Create dataset
    num_samples = 8
    torch.manual_seed(42)
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)

    # Run standard inference
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    standard_strategy = StandardInference()
    standard_acc, standard_results = standard_strategy.run_inference(
        model, test_loader, num_samples, device
    )

    # Run mixed precision inference
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    mp_strategy = MixedPrecisionInference(amp_dtype=torch.float16)
    mp_acc, mp_results = mp_strategy.run_inference(model, test_loader, num_samples, device)

    # Results should be very close (within numerical precision)
    # Note: Due to reduced precision, results may differ slightly
    assert abs(standard_acc.item() - mp_acc.item()) < 0.01

    # Check that predictions are mostly the same
    correct_predictions = sum(1 for s, m in zip(standard_results, mp_results) if s == m)
    # Allow up to 1 difference due to numerical precision
    assert correct_predictions >= num_samples - 1


def test_mixed_precision_cpu_fallback():
    """Test that mixed precision falls back to standard inference on CPU."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    device = torch.device("cpu")

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Create mixed precision strategy
    strategy = MixedPrecisionInference(amp_dtype=torch.float16)

    # Run inference (should fall back to standard)
    test_acc, per_sample_results = strategy.run_inference(
        model, test_loader, num_samples, device, class_names=None
    )

    # Verify it still works
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0
    assert len(per_sample_results) == num_samples


def test_inference_factory_mixed_precision():
    """Test factory with 'mixed_precision' strategy."""
    # Test with float16
    config = {"inference": {"strategy": "mixed_precision", "amp_dtype": "float16"}}

    strategy = get_inference_strategy(config)
    assert isinstance(strategy, MixedPrecisionInference)
    assert strategy.amp_dtype == torch.float16

    # Test with bfloat16
    config = {"inference": {"strategy": "mixed_precision", "amp_dtype": "bfloat16"}}

    strategy = get_inference_strategy(config)
    assert isinstance(strategy, MixedPrecisionInference)
    assert strategy.amp_dtype == torch.bfloat16

    # Test default (should be float16)
    config = {"inference": {"strategy": "mixed_precision"}}

    strategy = get_inference_strategy(config)
    assert isinstance(strategy, MixedPrecisionInference)
    assert strategy.amp_dtype == torch.float16


# ============================================================================
# Accelerate Inference Tests
# ============================================================================


@pytest.mark.skipif(not _ACCELERATE_AVAILABLE, reason="Accelerate not available")
def test_accelerate_inference_single_device():
    """Test AccelerateInference in single-device mode."""
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Note: Device is not important, Accelerator will handle it
    device = torch.device("cpu")

    # Create minimal dataset
    num_samples = 8
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Run accelerate inference (single device mode)
    strategy = AccelerateInference()
    test_acc, per_sample_results = strategy.run_inference(
        model=model,
        dataloader=test_loader,
        dataset_size=num_samples,
        device=device,
        class_names=["class_0", "class_1"],
    )

    # Verify results
    assert test_acc is not None
    assert isinstance(test_acc.item(), float)
    assert 0.0 <= test_acc.item() <= 1.0
    assert len(per_sample_results) == num_samples

    # Verify format with class names
    for true_label, pred_label, is_correct in per_sample_results:
        assert isinstance(true_label, str)
        assert isinstance(pred_label, str)
        assert isinstance(is_correct, bool)


@pytest.mark.skipif(not _ACCELERATE_AVAILABLE, reason="Accelerate not available")
def test_accelerate_vs_standard_results():
    """Test that accelerate produces same results as standard inference in single-device mode."""
    # Create model and freeze weights for deterministic results
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cpu")

    # Create dataset
    num_samples = 8
    torch.manual_seed(42)
    X_test = torch.randn(num_samples, 10)
    y_test = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    test_dataset = TensorDataset(X_test, y_test)

    # Run standard inference
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    standard_strategy = StandardInference()
    standard_acc, standard_results = standard_strategy.run_inference(
        model, test_loader, num_samples, device
    )

    # Run accelerate inference (need to recreate model for clean state)
    torch.manual_seed(42)
    model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    model2.load_state_dict(model.state_dict())
    model2.eval()

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    accel_strategy = AccelerateInference()
    accel_acc, accel_results = accel_strategy.run_inference(
        model2, test_loader, num_samples, device
    )

    # Results should be identical in single-device mode
    assert abs(standard_acc.item() - accel_acc.item()) < 1e-6
    assert standard_results == accel_results


@pytest.mark.skipif(not _ACCELERATE_AVAILABLE, reason="Accelerate not available")
def test_inference_factory_accelerate():
    """Test factory with 'accelerate' strategy."""
    config = {"inference": {"strategy": "accelerate"}}

    strategy = get_inference_strategy(config)
    assert isinstance(strategy, AccelerateInference)


def test_inference_factory_accelerate_not_installed():
    """Test that factory raises error when accelerate is not available."""
    if _ACCELERATE_AVAILABLE:
        pytest.skip("Accelerate is installed, cannot test import error")

    config = {"inference": {"strategy": "accelerate"}}

    with pytest.raises(ImportError, match="AccelerateInference requires accelerate"):
        get_inference_strategy(config)
