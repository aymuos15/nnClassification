"""Tests for ONNX validation metrics."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.checkpointing import save_checkpoint
from ml_src.core.metrics.onnx_validation import (
    benchmark_inference_speed,
    compare_outputs,
    validate_onnx_model,
)
from ml_src.core.network import get_model

# Check if onnxruntime is available
try:
    import onnxruntime

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Skip marker for tests requiring onnxruntime
requires_onnxruntime = pytest.mark.skipif(
    not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed"
)


class TestCompareOutputs:
    """Test cases for compare_outputs function."""

    def test_identical_torch_tensors(self):
        """Test with identical torch tensors (should have metrics ≈ 0, cosine_sim = 1.0)."""
        pytorch_output = torch.randn(2, 10)
        onnx_output = pytorch_output.clone()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] < 1e-8
        assert metrics["mse"] < 1e-15
        assert metrics["mae"] < 1e-15
        assert abs(metrics["cosine_similarity"] - 1.0) < 1e-6

    def test_identical_numpy_arrays(self):
        """Test with identical numpy arrays."""
        pytorch_output = np.random.randn(2, 10)
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_different_tensors_detect_differences(self):
        """Test with different tensors (should detect differences)."""
        pytorch_output = torch.ones(2, 10)
        onnx_output = torch.ones(2, 10) * 1.1  # 10% difference (scaled)

        metrics = compare_outputs(pytorch_output, onnx_output)

        # Should detect differences in magnitude metrics
        assert metrics["max_diff"] > 0.0
        assert metrics["mse"] > 0.0
        assert metrics["mae"] > 0.0
        # Cosine similarity will be 1.0 because vectors point in same direction (just scaled)
        # This is correct behavior - cosine similarity measures direction, not magnitude
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_slight_difference_metrics(self):
        """Test with slightly different outputs to verify metric calculations."""
        pytorch_output = np.array([[1.0, 2.0, 3.0]])
        onnx_output = np.array([[1.001, 2.001, 3.001]])

        metrics = compare_outputs(pytorch_output, onnx_output)

        # Verify specific metric values
        assert abs(metrics["max_diff"] - 0.001) < 1e-6
        assert abs(metrics["mse"] - 0.000001) < 1e-9
        assert abs(metrics["mae"] - 0.001) < 1e-6
        assert metrics["cosine_similarity"] > 0.999

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 10),  # 1D batch
            (4, 5, 5),  # 3D
            (2, 3, 32, 32),  # 4D (image-like)
            (8, 100),  # Large batch
        ],
    )
    def test_various_shapes(self, shape):
        """Test with various tensor shapes."""
        pytorch_output = np.random.randn(*shape)
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
        ],
    )
    def test_various_dtypes(self, dtype):
        """Test with various dtypes."""
        pytorch_output = np.random.randn(2, 10).astype(dtype)
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_mixed_tensor_and_array(self):
        """Test with torch tensor and numpy array (should work)."""
        pytorch_output = torch.randn(2, 10)
        onnx_output = pytorch_output.numpy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] < 1e-8
        assert metrics["cosine_similarity"] > 0.9999

    def test_array_and_tensor(self):
        """Test with numpy array as pytorch and tensor as onnx."""
        pytorch_output = np.random.randn(2, 10)
        onnx_output = torch.from_numpy(pytorch_output.copy())

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched output shapes raise ValueError."""
        pytorch_output = torch.randn(2, 10)
        onnx_output = torch.randn(2, 15)  # Different shape

        with pytest.raises(ValueError, match="Output shapes must match"):
            compare_outputs(pytorch_output, onnx_output)

    def test_completely_different_shapes_raises_error(self):
        """Test with completely different shapes."""
        pytorch_output = torch.randn(5, 5)
        onnx_output = torch.randn(10, 10)

        with pytest.raises(ValueError, match="Output shapes must match"):
            compare_outputs(pytorch_output, onnx_output)

    def test_zero_norm_vectors(self):
        """Test cosine similarity with zero-norm vectors."""
        pytorch_output = np.zeros((2, 10))
        onnx_output = np.zeros((2, 10))

        metrics = compare_outputs(pytorch_output, onnx_output)

        # When both are zero, cosine similarity should be 0.0
        assert metrics["cosine_similarity"] == 0.0
        assert metrics["max_diff"] == 0.0

    def test_one_zero_norm_vector(self):
        """Test cosine similarity when one vector is zero."""
        pytorch_output = np.random.randn(2, 10)
        onnx_output = np.zeros((2, 10))

        metrics = compare_outputs(pytorch_output, onnx_output)

        # When one is zero, cosine similarity should be 0.0
        assert metrics["cosine_similarity"] == 0.0
        assert metrics["max_diff"] > 0.0

    def test_negative_cosine_similarity(self):
        """Test that cosine similarity can be negative (opposite directions)."""
        pytorch_output = np.ones((2, 10))
        onnx_output = -np.ones((2, 10))  # Opposite direction

        metrics = compare_outputs(pytorch_output, onnx_output)

        # Should have negative cosine similarity
        assert metrics["cosine_similarity"] < 0.0
        assert metrics["cosine_similarity"] > -1.001  # Should be > -1

    def test_nan_handling(self):
        """Test handling of NaN values in outputs."""
        pytorch_output = np.array([[1.0, 2.0, np.nan]])
        onnx_output = np.array([[1.0, 2.0, 3.0]])

        metrics = compare_outputs(pytorch_output, onnx_output)

        # NaN should propagate to metrics
        assert np.isnan(metrics["max_diff"]) or metrics["max_diff"] > 0
        assert np.isnan(metrics["mse"]) or metrics["mse"] > 0

    def test_inf_handling(self):
        """Test handling of Inf values in outputs."""
        pytorch_output = np.array([[1.0, 2.0, np.inf]])
        onnx_output = np.array([[1.0, 2.0, 3.0]])

        metrics = compare_outputs(pytorch_output, onnx_output)

        # Inf should result in very large differences
        assert metrics["max_diff"] == np.inf or metrics["max_diff"] > 1e10

    def test_single_element_tensors(self):
        """Test with single element tensors."""
        pytorch_output = torch.tensor([[5.0]])
        onnx_output = torch.tensor([[5.0]])

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_large_tensors(self):
        """Test with large tensors to verify performance."""
        pytorch_output = np.random.randn(100, 1000)
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)


class TestBenchmarkInferenceSpeed:
    """Test cases for benchmark_inference_speed function."""

    @pytest.fixture
    def simple_model_and_onnx(self, tmp_path, base_config_template):
        """Create a simple model and export it to ONNX."""
        # Create model
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        model.eval()

        # Export to ONNX
        input_tensor = torch.randn(1, 3, 224, 224)
        onnx_path = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            input_tensor,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        return model, str(onnx_path), input_tensor

    @requires_onnxruntime
    def test_benchmark_returns_expected_keys(self, simple_model_and_onnx):
        """Test that benchmark returns dict with expected keys."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=5,  # Small number for fast test
            warmup=2,
        )

        # Verify structure
        assert isinstance(stats, dict)
        assert "pytorch_times" in stats
        assert "onnx_times" in stats
        assert "speedup" in stats

    @requires_onnxruntime
    def test_timing_statistics_keys(self, simple_model_and_onnx):
        """Test that timing statistics have expected keys."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=5,
            warmup=2,
        )

        # Check pytorch_times keys
        assert "mean" in stats["pytorch_times"]
        assert "std" in stats["pytorch_times"]
        assert "min" in stats["pytorch_times"]
        assert "max" in stats["pytorch_times"]
        assert "p50" in stats["pytorch_times"]
        assert "p95" in stats["pytorch_times"]
        assert "p99" in stats["pytorch_times"]

        # Check onnx_times keys
        assert "mean" in stats["onnx_times"]
        assert "std" in stats["onnx_times"]
        assert "min" in stats["onnx_times"]
        assert "max" in stats["onnx_times"]
        assert "p50" in stats["onnx_times"]
        assert "p95" in stats["onnx_times"]
        assert "p99" in stats["onnx_times"]

    @requires_onnxruntime
    def test_statistics_are_reasonable(self, simple_model_and_onnx):
        """Test that timing statistics have reasonable values."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=10,
            warmup=3,
        )

        # Check PyTorch times
        assert stats["pytorch_times"]["mean"] > 0
        assert stats["pytorch_times"]["std"] >= 0
        assert stats["pytorch_times"]["min"] > 0
        assert stats["pytorch_times"]["max"] >= stats["pytorch_times"]["min"]
        assert stats["pytorch_times"]["p50"] > 0
        assert stats["pytorch_times"]["p95"] >= stats["pytorch_times"]["p50"]
        assert stats["pytorch_times"]["p99"] >= stats["pytorch_times"]["p95"]

        # Check ONNX times
        assert stats["onnx_times"]["mean"] > 0
        assert stats["onnx_times"]["std"] >= 0
        assert stats["onnx_times"]["min"] > 0
        assert stats["onnx_times"]["max"] >= stats["onnx_times"]["min"]
        assert stats["onnx_times"]["p50"] > 0
        assert stats["onnx_times"]["p95"] >= stats["onnx_times"]["p50"]
        assert stats["onnx_times"]["p99"] >= stats["onnx_times"]["p95"]

        # Speedup should be positive
        assert stats["speedup"] > 0

    @requires_onnxruntime
    def test_percentile_ordering(self, simple_model_and_onnx):
        """Test that percentiles are in correct order."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=20,
            warmup=3,
        )

        # PyTorch percentiles should be ordered
        assert (
            stats["pytorch_times"]["min"]
            <= stats["pytorch_times"]["p50"]
            <= stats["pytorch_times"]["p95"]
            <= stats["pytorch_times"]["p99"]
            <= stats["pytorch_times"]["max"]
        )

        # ONNX percentiles should be ordered
        assert (
            stats["onnx_times"]["min"]
            <= stats["onnx_times"]["p50"]
            <= stats["onnx_times"]["p95"]
            <= stats["onnx_times"]["p99"]
            <= stats["onnx_times"]["max"]
        )

    @requires_onnxruntime
    def test_speedup_calculation(self, simple_model_and_onnx):
        """Test speedup calculation."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=10,
            warmup=3,
        )

        # Speedup should equal pytorch_mean / onnx_mean
        expected_speedup = stats["pytorch_times"]["mean"] / stats["onnx_times"]["mean"]
        assert abs(stats["speedup"] - expected_speedup) < 1e-6

    @requires_onnxruntime
    def test_small_num_runs(self, simple_model_and_onnx):
        """Test with very small num_runs for fast testing."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=2,  # Minimal runs
            warmup=1,
        )

        # Should still work
        assert stats["pytorch_times"]["mean"] > 0
        assert stats["onnx_times"]["mean"] > 0
        assert stats["speedup"] > 0

    def test_missing_onnxruntime_raises_error(self, simple_model_and_onnx, monkeypatch):
        """Test that missing onnxruntime raises ImportError."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        # Mock onnxruntime import to fail
        import sys

        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        with pytest.raises(ImportError, match="onnxruntime is not installed"):
            # Re-import to trigger the import check
            from ml_src.core.metrics.onnx_validation import benchmark_inference_speed

            benchmark_inference_speed(
                pytorch_model=model,
                onnx_path=onnx_path,
                input_tensor=input_tensor,
                device="cpu",
                num_runs=5,
                warmup=2,
            )

    @requires_onnxruntime
    def test_times_in_milliseconds(self, simple_model_and_onnx):
        """Test that times are returned in milliseconds."""
        model, onnx_path, input_tensor = simple_model_and_onnx

        stats = benchmark_inference_speed(
            pytorch_model=model,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            device="cpu",
            num_runs=5,
            warmup=2,
        )

        # Times should be in reasonable range for milliseconds
        # (not seconds or microseconds)
        # ResNet18 forward pass should be between 0.1ms and 10000ms on CPU
        assert 0.1 < stats["pytorch_times"]["mean"] < 10000
        assert 0.1 < stats["onnx_times"]["mean"] < 10000


class TestValidateONNXModel:
    """Test cases for validate_onnx_model function."""

    @pytest.fixture
    def checkpoint_and_onnx(self, tmp_path, base_config_template):
        """Create checkpoint, ONNX model, and test loader."""
        # Create model
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_acc=0.8,
            train_losses=[0.5],
            val_losses=[0.6],
            train_accs=[0.8],
            val_accs=[0.8],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )

        # Export to ONNX with dynamic batch size
        onnx_path = tmp_path / "model.onnx"
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )

        # Create test loader
        test_data = TensorDataset(
            torch.randn(20, 3, 224, 224), torch.randint(0, 2, (20,))  # 20 samples
        )
        test_loader = DataLoader(test_data, batch_size=4)

        return str(checkpoint_path), str(onnx_path), test_loader

    @requires_onnxruntime
    def test_validate_returns_report_dict(self, checkpoint_and_onnx):
        """Test that validate_onnx_model returns report dict with expected structure."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=3,
        )

        # Verify report structure
        assert isinstance(report, dict)
        assert "num_batches_tested" in report
        assert "total_samples" in report
        assert "metrics" in report
        assert "status" in report
        assert "per_batch_metrics" in report

    @requires_onnxruntime
    def test_report_metrics_structure(self, checkpoint_and_onnx):
        """Test that report metrics have expected structure."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=2,
        )

        # Check metrics structure
        assert "max_diff" in report["metrics"]
        assert "mse" in report["metrics"]
        assert "mae" in report["metrics"]
        assert "cosine_similarity" in report["metrics"]

        # Each metric should have max, mean, min
        for metric_name in ["max_diff", "mse", "mae", "cosine_similarity"]:
            assert "max" in report["metrics"][metric_name]
            assert "mean" in report["metrics"][metric_name]
            assert "min" in report["metrics"][metric_name]

    @requires_onnxruntime
    def test_num_batches_tested(self, checkpoint_and_onnx):
        """Test that correct number of batches is tested."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        num_batches = 3
        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=num_batches,
        )

        assert report["num_batches_tested"] == num_batches
        assert len(report["per_batch_metrics"]) == num_batches

    @requires_onnxruntime
    def test_total_samples_count(self, checkpoint_and_onnx):
        """Test that total samples count is correct."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        num_batches = 3
        batch_size = 4
        expected_samples = num_batches * batch_size

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=num_batches,
        )

        assert report["total_samples"] == expected_samples

    @requires_onnxruntime
    def test_validation_status_pass(self, checkpoint_and_onnx):
        """Test that validation status is PASS for identical models."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=2,
        )

        # Should PASS or WARN (not FAIL) for correctly exported model
        assert report["status"] in ["PASS", "WARN"]

    @requires_onnxruntime
    def test_status_thresholds(self, checkpoint_and_onnx):
        """Test status determination based on max_diff."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=2,
        )

        max_of_max_diffs = report["metrics"]["max_diff"]["max"]

        # Verify status logic
        if max_of_max_diffs < 1e-5:
            assert report["status"] == "PASS"
        elif max_of_max_diffs < 1e-4:
            assert report["status"] == "WARN"
        else:
            assert report["status"] == "FAIL"

    @requires_onnxruntime
    def test_per_batch_metrics_structure(self, checkpoint_and_onnx):
        """Test that per_batch_metrics have correct structure."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=2,
        )

        # Check per-batch metrics
        for batch_metrics in report["per_batch_metrics"]:
            assert "batch_idx" in batch_metrics
            assert "batch_size" in batch_metrics
            assert "max_diff" in batch_metrics
            assert "mse" in batch_metrics
            assert "mae" in batch_metrics
            assert "cosine_similarity" in batch_metrics

    @requires_onnxruntime
    def test_empty_test_loader(self, tmp_path, base_config_template):
        """Test with empty test loader."""
        # Create checkpoint and ONNX
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_acc=0.8,
            train_losses=[0.5],
            val_losses=[0.6],
            train_accs=[0.8],
            val_accs=[0.8],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )

        onnx_path = tmp_path / "model.onnx"
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        # Create empty test loader
        test_data = TensorDataset(torch.randn(0, 3, 224, 224), torch.randint(0, 2, (0,)))
        empty_loader = DataLoader(test_data, batch_size=4)

        # Should handle gracefully
        report = validate_onnx_model(
            checkpoint_path=str(checkpoint_path),
            onnx_path=str(onnx_path),
            test_loader=empty_loader,
            device="cpu",
            num_batches=5,
        )

        assert report["num_batches_tested"] == 0
        assert report["total_samples"] == 0

    @requires_onnxruntime
    def test_validation_with_single_batch(self, checkpoint_and_onnx):
        """Test validation with only one batch."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=1,
        )

        assert report["num_batches_tested"] == 1
        assert len(report["per_batch_metrics"]) == 1
        assert report["status"] in ["PASS", "WARN", "FAIL"]

    def test_missing_onnxruntime_raises_error(self, checkpoint_and_onnx, monkeypatch):
        """Test that missing onnxruntime raises ImportError."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        # Mock onnxruntime import to fail
        import sys

        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        with pytest.raises(ImportError, match="onnxruntime is not installed"):
            # Re-import to trigger the import check
            from ml_src.core.metrics.onnx_validation import validate_onnx_model

            validate_onnx_model(
                checkpoint_path=checkpoint_path,
                onnx_path=onnx_path,
                test_loader=test_loader,
                device="cpu",
                num_batches=2,
            )

    @requires_onnxruntime
    def test_metrics_aggregation(self, checkpoint_and_onnx):
        """Test that metrics are properly aggregated across batches."""
        checkpoint_path, onnx_path, test_loader = checkpoint_and_onnx

        report = validate_onnx_model(
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            test_loader=test_loader,
            device="cpu",
            num_batches=3,
        )

        # Verify max is >= mean >= min for each metric
        for metric_name in ["max_diff", "mse", "mae"]:
            assert (
                report["metrics"][metric_name]["max"]
                >= report["metrics"][metric_name]["mean"]
                >= report["metrics"][metric_name]["min"]
            )

        # Cosine similarity: max >= mean >= min (reversed for similarity)
        assert (
            report["metrics"]["cosine_similarity"]["max"]
            >= report["metrics"]["cosine_similarity"]["mean"]
            >= report["metrics"]["cosine_similarity"]["min"]
        )

    @requires_onnxruntime
    def test_loader_with_images_only(self, tmp_path, base_config_template):
        """Test with loader that returns only images (no labels)."""
        # Create checkpoint and ONNX
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_acc=0.8,
            train_losses=[0.5],
            val_losses=[0.6],
            train_accs=[0.8],
            val_accs=[0.8],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )

        onnx_path = tmp_path / "model.onnx"
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
        )

        # Create loader with only images (no labels)
        test_data = TensorDataset(torch.randn(12, 3, 224, 224))
        test_loader = DataLoader(test_data, batch_size=4)

        # Should work with images only
        report = validate_onnx_model(
            checkpoint_path=str(checkpoint_path),
            onnx_path=str(onnx_path),
            test_loader=test_loader,
            device="cpu",
            num_batches=2,
        )

        assert report["num_batches_tested"] == 2
        assert report["status"] in ["PASS", "WARN", "FAIL"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compare_outputs_with_require_grad(self):
        """Test compare_outputs with tensors that require grad."""
        pytorch_output = torch.randn(2, 10, requires_grad=True)
        onnx_output = pytorch_output.detach().clone()

        # Should work without errors
        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] < 1e-8
        assert metrics["cosine_similarity"] > 0.9999

    def test_compare_outputs_very_small_values(self):
        """Test compare_outputs with very small values."""
        pytorch_output = np.array([[1e-20, 2e-20, 3e-20]])
        onnx_output = np.array([[1e-20, 2e-20, 3e-20]])

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["mae"] == 0.0

    def test_compare_outputs_very_large_values(self):
        """Test compare_outputs with very large values."""
        pytorch_output = np.array([[1e20, 2e20, 3e20]])
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_compare_outputs_mixed_positive_negative(self):
        """Test compare_outputs with mixed positive and negative values."""
        pytorch_output = np.array([[-1.0, 2.0, -3.0, 4.0]])
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),  # 1D single element
            (1, 1),  # 2D single element
            (1, 1, 1),  # 3D single element
            (1, 1, 1, 1),  # 4D single element
        ],
    )
    def test_compare_outputs_single_element_various_dims(self, shape):
        """Test compare_outputs with single element in various dimensions."""
        pytorch_output = np.ones(shape)
        onnx_output = np.ones(shape)

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert metrics["max_diff"] == 0.0
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_compare_outputs_broadcast_shapes_fail(self):
        """Test that broadcastable but different shapes still raise error."""
        pytorch_output = np.ones((2, 1, 10))
        onnx_output = np.ones((2, 5, 10))

        with pytest.raises(ValueError, match="Output shapes must match"):
            compare_outputs(pytorch_output, onnx_output)

    def test_metrics_are_floats(self):
        """Test that all returned metrics are Python floats (not numpy floats)."""
        pytorch_output = np.random.randn(2, 10)
        onnx_output = pytorch_output.copy()

        metrics = compare_outputs(pytorch_output, onnx_output)

        assert isinstance(metrics["max_diff"], float)
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["cosine_similarity"], float)

        # Not numpy types
        assert type(metrics["max_diff"]) == float
        assert type(metrics["mse"]) == float
        assert type(metrics["mae"]) == float
        assert type(metrics["cosine_similarity"]) == float
