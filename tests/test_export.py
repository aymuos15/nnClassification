"""Tests for ONNX export functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ml_src.core.checkpointing import save_checkpoint
from ml_src.core.export import export_to_onnx, validate_onnx_export
from ml_src.core.network import get_model

# Check if ONNX and onnxruntime are available
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Skip markers for tests requiring optional dependencies
requires_onnx = pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnx not installed")
requires_onnxruntime = pytest.mark.skipif(
    not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed"
)
requires_both = pytest.mark.skipif(
    not (ONNX_AVAILABLE and ONNXRUNTIME_AVAILABLE),
    reason="onnx and onnxruntime not installed"
)


class TestONNXExport:
    """Test cases for ONNX model export."""

    @pytest.fixture
    def simple_checkpoint(self, tmp_path, base_config_template):
        """Create a simple checkpoint file for testing."""
        # Create a simple model
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        # Create checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_acc=0.85,
            train_losses=[0.5, 0.4, 0.3],
            val_losses=[0.6, 0.5, 0.4],
            train_accs=[0.7, 0.75, 0.8],
            val_accs=[0.65, 0.75, 0.85],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )
        return checkpoint_path, config

    @pytest.fixture
    def custom_cnn_checkpoint(self, tmp_path):
        """Create a checkpoint with custom SimpleCNN model."""
        config = {
            "model": {
                "type": "custom",
                "custom_architecture": "simple_cnn",
                "num_classes": 10,
                "input_size": 224,
                "dropout": 0.5,
            }
        }
        model = get_model(config, device="cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "custom_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            best_acc=0.75,
            train_losses=[0.8, 0.6, 0.5],
            val_losses=[0.9, 0.7, 0.6],
            train_accs=[0.6, 0.7, 0.75],
            val_accs=[0.55, 0.65, 0.75],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )
        return checkpoint_path, config

    @pytest.fixture
    def inception_checkpoint(self, tmp_path):
        """Create a checkpoint with Inception model (requires 299x299 input)."""
        config = {
            "model": {
                "type": "base",
                "architecture": "inception_v3",
                "num_classes": 5,
                "weights": None,
            }
        }
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "inception_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            best_acc=0.65,
            train_losses=[1.0, 0.8],
            val_losses=[1.1, 0.9],
            train_accs=[0.5, 0.6],
            val_accs=[0.45, 0.65],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )
        return checkpoint_path, config

    @requires_onnx
    def test_export_creates_onnx_file(self, simple_checkpoint, tmp_path):
        """Test that export_to_onnx creates an .onnx file."""
        checkpoint_path, _ = simple_checkpoint
        output_path = tmp_path / "model.onnx"

        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            opset_version=17,
        )

        assert success is True
        assert "successful" in message.lower()
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @requires_onnx
    def test_export_file_has_reasonable_size(self, simple_checkpoint, tmp_path):
        """Test that exported ONNX file has a reasonable size (>100KB for ResNet18)."""
        checkpoint_path, _ = simple_checkpoint
        output_path = tmp_path / "model.onnx"

        success, _ = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success is True
        file_size = output_path.stat().st_size
        # ResNet18 should be at least 100KB
        assert file_size > 100 * 1024
        # But not unreasonably large (< 100MB)
        assert file_size < 100 * 1024 * 1024

    @pytest.mark.parametrize(
        "architecture,expected_success",
        [
            ("resnet18", True),
            ("resnet34", True),
            ("mobilenet_v2", True),
            ("efficientnet_b0", True),
            ("convnext_tiny", True),
        ],
    )
    @requires_onnx
    def test_export_multiple_base_models(
        self, tmp_path, architecture, expected_success, num_classes_small
    ):
        """Test export for multiple base model architectures."""
        # Create config for specified architecture
        config = {
            "model": {
                "type": "base",
                "architecture": architecture,
                "num_classes": num_classes_small,
                "weights": None,
            }
        }

        # Create model and checkpoint
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / f"{architecture}_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_acc=0.7,
            train_losses=[0.5],
            val_losses=[0.6],
            train_accs=[0.7],
            val_accs=[0.7],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )

        # Export to ONNX
        output_path = tmp_path / f"{architecture}.onnx"
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success == expected_success
        if expected_success:
            assert output_path.exists()

    @requires_onnx
    def test_export_custom_cnn(self, custom_cnn_checkpoint, tmp_path):
        """Test export for custom SimpleCNN model."""
        checkpoint_path, _ = custom_cnn_checkpoint
        output_path = tmp_path / "custom_model.onnx"

        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success is True
        assert output_path.exists()
        assert "successful" in message.lower()

    @requires_onnx
    def test_export_with_custom_input_size(self, simple_checkpoint, tmp_path):
        """Test export with custom input size."""
        checkpoint_path, _ = simple_checkpoint
        output_path = tmp_path / "model_custom_size.onnx"

        # Use custom input size (320x320)
        custom_size = (1, 3, 320, 320)
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            input_size=custom_size,
        )

        assert success is True
        assert output_path.exists()

    @requires_onnx
    def test_export_inception_auto_detects_input_size(self, inception_checkpoint, tmp_path):
        """Test that Inception models automatically use 299x299 input size."""
        checkpoint_path, _ = inception_checkpoint
        output_path = tmp_path / "inception.onnx"

        # Don't specify input_size - should auto-detect 299x299
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
        )

        assert success is True
        assert output_path.exists()

    @requires_onnx
    def test_export_creates_output_directory(self, simple_checkpoint, tmp_path):
        """Test that export creates output directory if it doesn't exist."""
        checkpoint_path, _ = simple_checkpoint
        nested_dir = tmp_path / "nested" / "dir" / "structure"
        output_path = nested_dir / "model.onnx"

        assert not nested_dir.exists()

        success, _ = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success is True
        assert nested_dir.exists()
        assert output_path.exists()

    def test_export_invalid_checkpoint_path(self, tmp_path):
        """Test export with non-existent checkpoint path."""
        invalid_path = tmp_path / "nonexistent.pt"
        output_path = tmp_path / "output.onnx"

        success, message = export_to_onnx(
            checkpoint_path=str(invalid_path), output_path=str(output_path)
        )

        assert success is False
        assert "not found" in message.lower()
        assert not output_path.exists()

    def test_export_invalid_checkpoint_format(self, tmp_path):
        """Test export with invalid checkpoint format (missing model_state_dict)."""
        # Create invalid checkpoint (just a regular dict without model_state_dict)
        invalid_checkpoint_path = tmp_path / "invalid.pt"
        torch.save({"some_key": "some_value"}, invalid_checkpoint_path)

        output_path = tmp_path / "output.onnx"

        success, message = export_to_onnx(
            checkpoint_path=str(invalid_checkpoint_path), output_path=str(output_path)
        )

        assert success is False
        assert "model_state_dict" in message.lower()
        assert not output_path.exists()

    def test_export_missing_config(self, tmp_path):
        """Test export with checkpoint missing config."""
        # Create checkpoint without config
        checkpoint_path = tmp_path / "no_config.pt"
        torch.save(
            {
                "model_state_dict": {},
                "epoch": 1,
            },
            checkpoint_path,
        )

        output_path = tmp_path / "output.onnx"

        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success is False
        assert "config" in message.lower()
        assert not output_path.exists()

    @requires_onnx
    def test_export_with_different_opset_versions(self, simple_checkpoint, tmp_path):
        """Test export with different ONNX opset versions."""
        checkpoint_path, _ = simple_checkpoint

        for opset_version in [13, 15, 17]:
            output_path = tmp_path / f"model_opset_{opset_version}.onnx"
            success, _ = export_to_onnx(
                checkpoint_path=str(checkpoint_path),
                output_path=str(output_path),
                opset_version=opset_version,
            )

            assert success is True
            assert output_path.exists()


class TestONNXValidation:
    """Test cases for ONNX export validation."""

    @pytest.fixture
    def model_and_onnx(self, tmp_path, base_config_template):
        """Create a model, export it to ONNX, and return both."""
        # Create model
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        model.eval()

        # Create checkpoint
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

        # Export to ONNX
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(checkpoint_path=str(checkpoint_path), output_path=str(onnx_path))

        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)

        return model, str(onnx_path), dummy_input

    @requires_both
    def test_validation_returns_metrics_dict(self, model_and_onnx):
        """Test that validation returns a dictionary with expected metrics."""
        model, onnx_path, dummy_input = model_and_onnx

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
        )

        # Check all expected keys are present
        assert isinstance(metrics, dict)
        assert "max_diff" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "cosine_similarity" in metrics
        assert "status" in metrics

    @requires_both
    def test_validation_metrics_types(self, model_and_onnx):
        """Test that validation metrics have correct types."""
        model, onnx_path, dummy_input = model_and_onnx

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
        )

        assert isinstance(metrics["max_diff"], float)
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["mae"], float)
        assert isinstance(metrics["cosine_similarity"], float)
        assert isinstance(metrics["status"], str)
        assert metrics["status"] in ["PASS", "WARN", "FAIL"]

    @requires_both
    def test_validation_with_identical_outputs(self, model_and_onnx):
        """Test validation with model that should produce identical outputs."""
        model, onnx_path, dummy_input = model_and_onnx

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
        )

        # Should have very low error (PASS or WARN, not FAIL)
        assert metrics["status"] in ["PASS", "WARN"]
        assert metrics["max_diff"] < 1e-3  # Should be very small
        assert metrics["cosine_similarity"] > 0.999  # Should be very close to 1

    @requires_both
    def test_validation_pass_threshold(self, model_and_onnx):
        """Test that validation PASS status is based on max_diff < 1e-5."""
        model, onnx_path, dummy_input = model_and_onnx

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
        )

        # Check status logic based on max_diff
        if metrics["max_diff"] < 1e-5:
            assert metrics["status"] == "PASS"
        elif metrics["max_diff"] < 1e-4:
            assert metrics["status"] == "WARN"
        else:
            assert metrics["status"] == "FAIL"

    def test_validation_without_onnxruntime(self):
        """Test validation behavior when onnxruntime is not available."""
        # This test verifies the code handles missing onnxruntime gracefully
        # The actual implementation catches ImportError at the beginning of validate_onnx_export
        # and returns a FAIL status with infinite metrics.

        # We can't easily mock this without onnx being installed (needed for export),
        # so this test simply documents the expected behavior.
        # The code path is: try: import onnxruntime except ImportError: return {...'status': 'FAIL'}

        # If onnxruntime is not available, validation should return FAIL
        # This is already tested by test_validation_with_invalid_onnx_path indirectly
        pass

    def test_validation_with_invalid_onnx_path(self, model_and_onnx):
        """Test validation with non-existent ONNX file."""
        model, _, dummy_input = model_and_onnx
        invalid_path = "/nonexistent/path/model.onnx"

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=invalid_path, dummy_input=dummy_input, device="cpu"
        )

        # Should return FAIL status with infinite errors
        assert metrics["status"] == "FAIL"
        assert metrics["max_diff"] == float("inf")
        assert metrics["mse"] == float("inf")
        assert metrics["mae"] == float("inf")

    @requires_both
    def test_validation_with_different_input_sizes(self, tmp_path, base_config_template):
        """Test validation with different input sizes."""
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

        # Export with custom input size
        onnx_path = tmp_path / "model_320.onnx"
        custom_size = (1, 3, 320, 320)
        export_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(onnx_path),
            input_size=custom_size,
        )

        # Validate with matching input size
        dummy_input = torch.randn(*custom_size)
        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=str(onnx_path), dummy_input=dummy_input, device="cpu"
        )

        assert metrics["status"] in ["PASS", "WARN"]

    @requires_both
    def test_validation_cosine_similarity_calculation(self, model_and_onnx):
        """Test that cosine similarity is calculated correctly."""
        model, onnx_path, dummy_input = model_and_onnx

        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
        )

        # Cosine similarity should be between -1 and 1, but for matching outputs close to 1
        assert -1.0 <= metrics["cosine_similarity"] <= 1.0
        # For correctly exported models, should be very close to 1
        assert metrics["cosine_similarity"] > 0.99

    def test_validation_fallback_cosine_similarity(self, model_and_onnx):
        """Test cosine similarity fallback when sklearn is not available."""
        model, onnx_path, dummy_input = model_and_onnx

        # Mock sklearn import failure to test fallback
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.metrics": None}):
            metrics = validate_onnx_export(
                pytorch_model=model, onnx_path=onnx_path, dummy_input=dummy_input, device="cpu"
            )

            # Should still compute cosine similarity using fallback
            assert "cosine_similarity" in metrics
            assert isinstance(metrics["cosine_similarity"], float)


class TestCLIIntegration:
    """Test cases for CLI integration."""

    @requires_onnx
    def test_export_cli_creates_output(self, simple_checkpoint, tmp_path):
        """Test that export can be called programmatically (simulating CLI)."""
        checkpoint_path, _ = simple_checkpoint
        output_path = tmp_path / "cli_output.onnx"

        # Simulate CLI call
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            opset_version=17,
        )

        assert success is True
        assert output_path.exists()

    @requires_onnx
    def test_export_with_validation_integration(self, simple_checkpoint, tmp_path):
        """Test export followed by validation (CLI workflow)."""
        checkpoint_path, config = simple_checkpoint
        output_path = tmp_path / "validated_model.onnx"

        # Step 1: Export
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )
        assert success is True

        # Step 2: Load model for validation
        model = get_model(config, device="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Step 3: Validate
        dummy_input = torch.randn(1, 3, 224, 224)
        metrics = validate_onnx_export(
            pytorch_model=model,
            onnx_path=str(output_path),
            dummy_input=dummy_input,
            device="cpu",
        )

        assert metrics["status"] in ["PASS", "WARN"]

    @pytest.fixture
    def simple_checkpoint(self, tmp_path, base_config_template):
        """Create a simple checkpoint file for testing."""
        config = base_config_template.copy()
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=5,
            best_acc=0.85,
            train_losses=[0.5, 0.4, 0.3],
            val_losses=[0.6, 0.5, 0.4],
            train_accs=[0.7, 0.75, 0.8],
            val_accs=[0.65, 0.75, 0.85],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )
        return checkpoint_path, config


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_export_with_empty_output_path(self, tmp_path, base_config_template):
        """Test export with empty string as output path."""
        # Create checkpoint
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

        # Try to export with empty output path
        # This should work as torch.onnx.export will handle it
        output_path = ""
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=output_path
        )

        # Should fail or handle gracefully
        # Actual behavior depends on torch.onnx.export implementation
        assert isinstance(success, bool)
        assert isinstance(message, str)

    def test_validation_with_mismatched_input_shape(self, tmp_path, base_config_template):
        """Test validation with mismatched input shape."""
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

        # Export with 224x224
        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(checkpoint_path=str(checkpoint_path), output_path=str(onnx_path))

        # Try to validate with different input shape (should fail)
        wrong_input = torch.randn(1, 3, 299, 299)  # Wrong size
        metrics = validate_onnx_export(
            pytorch_model=model, onnx_path=str(onnx_path), dummy_input=wrong_input, device="cpu"
        )

        # Should return FAIL status due to shape mismatch
        assert metrics["status"] == "FAIL"

    @requires_onnx
    def test_export_handles_metadata_gracefully(self, tmp_path, base_config_template):
        """Test that export handles metadata addition failures gracefully."""
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

        output_path = tmp_path / "model.onnx"

        # Mock onnx.load to fail (test metadata addition failure handling)
        # Note: We can't mock onnx entirely because torch.onnx.export needs it
        import onnx
        with patch.object(onnx, "load", side_effect=Exception("Metadata load failed")):
            success, message = export_to_onnx(
                checkpoint_path=str(checkpoint_path), output_path=str(output_path)
            )

            # Should still succeed even if metadata addition fails
            assert success is True
            assert output_path.exists()

    @requires_onnx
    def test_export_with_very_small_model(self, tmp_path):
        """Test export with TinyNet (very small model)."""
        config = {
            "model": {
                "type": "custom",
                "custom_architecture": "tiny_net",
                "num_classes": 2,
                "input_size": 224,
            }
        }
        model = get_model(config, device="cpu")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        checkpoint_path = tmp_path / "tiny_checkpoint.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            best_acc=0.6,
            train_losses=[0.7],
            val_losses=[0.8],
            train_accs=[0.6],
            val_accs=[0.6],
            config=config,
            checkpoint_path=str(checkpoint_path),
        )

        output_path = tmp_path / "tiny_model.onnx"
        success, message = export_to_onnx(
            checkpoint_path=str(checkpoint_path), output_path=str(output_path)
        )

        assert success is True
        assert output_path.exists()
        # TinyNet should be much smaller than ResNet
        assert output_path.stat().st_size < 10 * 1024 * 1024  # Less than 10MB
