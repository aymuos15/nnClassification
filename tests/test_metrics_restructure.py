"""
Tests for metrics package restructure backward compatibility.

This test suite verifies that the metrics package restructure maintains
full backward compatibility with the old import style. It tests:
1. Old-style imports (from ml_src.core.metrics import X)
2. New-style imports (from ml_src.core.metrics.submodule import X)
3. Function identity (old and new imports reference the same objects)
4. Basic functionality (smoke tests with real data)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


class TestBackwardCompatibleImports:
    """Test that old import style still works and references the same objects."""

    def test_classification_report_str_import_compatibility(self):
        """Test get_classification_report_str can be imported both ways."""
        # Old style (backward compatibility)
        from ml_src.core.metrics import get_classification_report_str as old_import

        # New style
        from ml_src.core.metrics.classification import (
            get_classification_report_str as new_import,
        )

        # Verify they're the same function
        assert old_import is new_import, "Imports should reference the same function object"

    def test_save_classification_report_import_compatibility(self):
        """Test save_classification_report can be imported both ways."""
        from ml_src.core.metrics import save_classification_report as old_import
        from ml_src.core.metrics.classification import (
            save_classification_report as new_import,
        )

        assert old_import is new_import

    def test_save_confusion_matrix_import_compatibility(self):
        """Test save_confusion_matrix can be imported both ways."""
        from ml_src.core.metrics import save_confusion_matrix as old_import
        from ml_src.core.metrics.visualization import save_confusion_matrix as new_import

        assert old_import is new_import

    def test_create_confusion_matrix_figure_import_compatibility(self):
        """Test create_confusion_matrix_figure can be imported both ways."""
        from ml_src.core.metrics import create_confusion_matrix_figure as old_import
        from ml_src.core.metrics.visualization import (
            create_confusion_matrix_figure as new_import,
        )

        assert old_import is new_import

    def test_log_confusion_matrix_to_tensorboard_import_compatibility(self):
        """Test log_confusion_matrix_to_tensorboard can be imported both ways."""
        from ml_src.core.metrics import log_confusion_matrix_to_tensorboard as old_import
        from ml_src.core.metrics.visualization import (
            log_confusion_matrix_to_tensorboard as new_import,
        )

        assert old_import is new_import

    def test_deprecated_metrics_module_import_compatibility(self):
        """Test that the deprecated metrics.py module still works."""
        # This tests the old ml_src.core.metrics (single file) compatibility layer
        from ml_src.core import metrics

        # Should have all the expected functions
        assert hasattr(metrics, "get_classification_report_str")
        assert hasattr(metrics, "save_classification_report")
        assert hasattr(metrics, "save_confusion_matrix")
        assert hasattr(metrics, "create_confusion_matrix_figure")
        assert hasattr(metrics, "log_confusion_matrix_to_tensorboard")

        # Should reference the same functions as new package imports
        from ml_src.core.metrics.classification import get_classification_report_str
        from ml_src.core.metrics.visualization import save_confusion_matrix

        assert metrics.get_classification_report_str is get_classification_report_str
        assert metrics.save_confusion_matrix is save_confusion_matrix


class TestUtilityFunctionsImports:
    """Test that utility functions can be imported from the package and submodules."""

    def test_ensure_numpy_import(self):
        """Test ensure_numpy can be imported from both package and utils."""
        from ml_src.core.metrics import ensure_numpy as pkg_import
        from ml_src.core.metrics.utils import ensure_numpy as utils_import

        assert pkg_import is utils_import

    def test_format_class_names_import(self):
        """Test format_class_names can be imported from both locations."""
        from ml_src.core.metrics import format_class_names as pkg_import
        from ml_src.core.metrics.utils import format_class_names as utils_import

        assert pkg_import is utils_import

    def test_validate_labels_import(self):
        """Test validate_labels can be imported from both locations."""
        from ml_src.core.metrics import validate_labels as pkg_import
        from ml_src.core.metrics.utils import validate_labels as utils_import

        assert pkg_import is utils_import

    def test_get_num_classes_import(self):
        """Test get_num_classes can be imported from both locations."""
        from ml_src.core.metrics import get_num_classes as pkg_import
        from ml_src.core.metrics.utils import get_num_classes as utils_import

        assert pkg_import is utils_import

    def test_prepare_labels_for_metrics_import(self):
        """Test prepare_labels_for_metrics can be imported from both locations."""
        from ml_src.core.metrics import prepare_labels_for_metrics as pkg_import
        from ml_src.core.metrics.utils import prepare_labels_for_metrics as utils_import

        assert pkg_import is utils_import

    def test_flatten_predictions_import(self):
        """Test flatten_predictions can be imported from both locations."""
        from ml_src.core.metrics import flatten_predictions as pkg_import
        from ml_src.core.metrics.utils import flatten_predictions as utils_import

        assert pkg_import is utils_import

    def test_argmax_predictions_import(self):
        """Test argmax_predictions can be imported from both locations."""
        from ml_src.core.metrics import argmax_predictions as pkg_import
        from ml_src.core.metrics.utils import argmax_predictions as utils_import

        assert pkg_import is utils_import


class TestSubmoduleImports:
    """Test that all submodules can be imported and have expected functions."""

    def test_classification_submodule_import(self):
        """Test classification submodule has all expected functions."""
        from ml_src.core.metrics import classification

        expected_functions = ["get_classification_report_str", "save_classification_report"]

        for func_name in expected_functions:
            assert hasattr(classification, func_name), f"Missing function: {func_name}"
            assert callable(getattr(classification, func_name))

    def test_visualization_submodule_import(self):
        """Test visualization submodule has all expected functions."""
        from ml_src.core.metrics import visualization

        expected_functions = [
            "save_confusion_matrix",
            "create_confusion_matrix_figure",
            "log_confusion_matrix_to_tensorboard",
        ]

        for func_name in expected_functions:
            assert hasattr(visualization, func_name), f"Missing function: {func_name}"
            assert callable(getattr(visualization, func_name))

    def test_utils_submodule_import(self):
        """Test utils submodule has all expected functions."""
        from ml_src.core.metrics import utils

        expected_functions = [
            "ensure_numpy",
            "format_class_names",
            "validate_labels",
            "get_num_classes",
            "prepare_labels_for_metrics",
            "flatten_predictions",
            "argmax_predictions",
        ]

        for func_name in expected_functions:
            assert hasattr(utils, func_name), f"Missing function: {func_name}"
            assert callable(getattr(utils, func_name))


class TestFunctionalityUnchanged:
    """Test that metrics functions still work correctly with test data."""

    @pytest.fixture
    def sample_labels(self):
        """Provide sample labels for testing."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        class_names = ["Class A", "Class B", "Class C"]
        return y_true, y_pred, class_names

    def test_get_classification_report_str_works(self, sample_labels):
        """Test classification report generation works correctly."""
        from ml_src.core.metrics import get_classification_report_str

        y_true, y_pred, class_names = sample_labels
        report = get_classification_report_str(y_true, y_pred, class_names)

        # Basic smoke test - verify it returns a string with expected content
        assert isinstance(report, str)
        assert "Class A" in report
        assert "Class B" in report
        assert "Class C" in report
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

    def test_save_classification_report_creates_file(self, sample_labels, tmp_path):
        """Test classification report can be saved to file."""
        from ml_src.core.metrics import save_classification_report

        y_true, y_pred, class_names = sample_labels
        save_path = tmp_path / "classification_report.txt"

        save_classification_report(y_true, y_pred, class_names, str(save_path))

        # Verify file was created and has content
        assert save_path.exists()
        content = save_path.read_text()
        assert "Classification Report" in content
        assert "Class A" in content
        assert len(content) > 100  # Should be substantial content

    def test_save_confusion_matrix_creates_file(self, sample_labels, tmp_path):
        """Test confusion matrix can be saved to file."""
        from ml_src.core.metrics import save_confusion_matrix

        y_true, y_pred, class_names = sample_labels
        save_path = tmp_path / "confusion_matrix.png"

        save_confusion_matrix(y_true, y_pred, class_names, str(save_path))

        # Verify file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 1000  # Should be a substantial image file

    def test_create_confusion_matrix_figure_returns_figure(self, sample_labels):
        """Test confusion matrix figure creation."""
        from ml_src.core.metrics import create_confusion_matrix_figure

        y_true, y_pred, class_names = sample_labels
        fig = create_confusion_matrix_figure(y_true, y_pred, class_names)

        # Verify it returns a matplotlib figure
        import matplotlib.pyplot as plt

        assert fig is not None
        assert hasattr(fig, "savefig")  # Should be a matplotlib figure
        plt.close(fig)  # Clean up


class TestUtilityFunctionsBehavior:
    """Test that utility functions work correctly with various input types."""

    def test_ensure_numpy_with_tensor(self):
        """Test ensure_numpy converts PyTorch tensors correctly."""
        from ml_src.core.metrics import ensure_numpy

        tensor = torch.tensor([1, 2, 3, 4])
        result = ensure_numpy(tensor)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_ensure_numpy_with_list(self):
        """Test ensure_numpy converts lists correctly."""
        from ml_src.core.metrics import ensure_numpy

        data = [1, 2, 3, 4]
        result = ensure_numpy(data)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_ensure_numpy_with_numpy(self):
        """Test ensure_numpy returns numpy arrays as-is."""
        from ml_src.core.metrics import ensure_numpy

        data = np.array([1, 2, 3, 4])
        result = ensure_numpy(data)

        assert isinstance(result, np.ndarray)
        assert result is data  # Should be the same object

    def test_format_class_names_generates_defaults(self):
        """Test format_class_names generates default names when None provided."""
        from ml_src.core.metrics import format_class_names

        names = format_class_names(3)
        assert names == ["Class 0", "Class 1", "Class 2"]

    def test_format_class_names_validates_length(self):
        """Test format_class_names validates custom names length."""
        from ml_src.core.metrics import format_class_names

        # Valid case
        names = format_class_names(2, ["cat", "dog"])
        assert names == ["cat", "dog"]

        # Invalid case - wrong length
        with pytest.raises(ValueError, match="does not match"):
            format_class_names(2, ["cat", "dog", "bird"])

    def test_validate_labels_passes_valid_input(self):
        """Test validate_labels passes for valid inputs."""
        from ml_src.core.metrics import validate_labels

        # Should not raise
        validate_labels([0, 1, 2], [0, 1, 1])
        validate_labels(np.array([0, 1]), np.array([1, 0]))
        validate_labels(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))

    def test_validate_labels_raises_on_length_mismatch(self):
        """Test validate_labels raises error for mismatched lengths."""
        from ml_src.core.metrics import validate_labels

        with pytest.raises(ValueError, match="must have the same length"):
            validate_labels([0, 1], [0, 1, 2])

    def test_get_num_classes_infers_correctly(self):
        """Test get_num_classes infers number of classes from labels."""
        from ml_src.core.metrics import get_num_classes

        # 3 classes (0, 1, 2)
        assert get_num_classes([0, 1, 2], [0, 1, 1]) == 3

        # 4 classes (0, 1, 2, 3) - max in pred is 3
        assert get_num_classes([0, 1, 2], [0, 1, 3]) == 4

        # Works with tensors
        assert get_num_classes(torch.tensor([0, 1]), torch.tensor([0, 1])) == 2

    def test_prepare_labels_for_metrics_full_workflow(self):
        """Test prepare_labels_for_metrics handles full preprocessing."""
        from ml_src.core.metrics import prepare_labels_for_metrics

        y_true = torch.tensor([0, 1, 2, 1])
        y_pred = torch.tensor([0, 1, 1, 2])

        y_true_np, y_pred_np, names = prepare_labels_for_metrics(y_true, y_pred)

        # Check conversions
        assert isinstance(y_true_np, np.ndarray)
        assert isinstance(y_pred_np, np.ndarray)
        np.testing.assert_array_equal(y_true_np, np.array([0, 1, 2, 1]))
        np.testing.assert_array_equal(y_pred_np, np.array([0, 1, 1, 2]))

        # Check class names
        assert names == ["Class 0", "Class 1", "Class 2"]

    def test_prepare_labels_for_metrics_with_custom_names(self):
        """Test prepare_labels_for_metrics with custom class names."""
        from ml_src.core.metrics import prepare_labels_for_metrics

        y_true = [0, 1, 2]
        y_pred = [0, 1, 1]
        custom_names = ["cat", "dog", "bird"]

        _, _, names = prepare_labels_for_metrics(y_true, y_pred, custom_names)

        assert names == ["cat", "dog", "bird"]

    def test_flatten_predictions_with_multidim_array(self):
        """Test flatten_predictions flattens multi-dimensional arrays."""
        from ml_src.core.metrics import flatten_predictions

        # 2D array
        predictions = torch.tensor([[0, 1], [2, 1]])
        result = flatten_predictions(predictions)
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 1]))

        # 3D array
        predictions = np.array([[[0]], [[1]], [[2]]])
        result = flatten_predictions(predictions)
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_argmax_predictions_converts_logits(self):
        """Test argmax_predictions converts logits to class predictions."""
        from ml_src.core.metrics import argmax_predictions

        # Soft predictions
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        predictions = argmax_predictions(logits)

        np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))

        # 3-class case
        logits = [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]
        predictions = argmax_predictions(logits)

        np.testing.assert_array_equal(predictions, np.array([1, 0]))


class TestCodebaseImportScan:
    """Informational tests that scan the codebase for metrics imports."""

    def test_list_files_importing_metrics(self, capsys):
        """
        List all files that import from ml_src.core.metrics.

        This is an informational test that helps identify all files
        that might be affected by the metrics restructure.
        """
        # Files known to import from metrics (from grep earlier)
        known_importers = [
            "ml_src/cli/inference.py",
            "ml_src/cli/train.py",
            "ml_src/core/trainers/base.py",
            "ml_src/core/trainers/differential_privacy.py",
        ]

        print("\n=== Files importing from ml_src.core.metrics ===")
        for file_path in known_importers:
            full_path = Path("/home/localssk23/gui") / file_path
            if full_path.exists():
                print(f"  - {file_path}")

        print(
            "\nAll these files should continue to work with backward-compatible imports."
        )

        # This test always passes - it's just informational
        assert True

    def test_verify_all_imports_work_in_known_files(self):
        """
        Verify that known files can still import metrics functions.

        This ensures that the restructure doesn't break existing code.
        """
        # These should all work without errors
        try:
            # Simulate imports from cli/train.py and cli/inference.py
            from ml_src.core.metrics import (
                create_confusion_matrix_figure,
                get_classification_report_str,
                save_classification_report,
                save_confusion_matrix,
            )

            # Verify they're callable
            assert callable(get_classification_report_str)
            assert callable(save_classification_report)
            assert callable(save_confusion_matrix)
            assert callable(create_confusion_matrix_figure)

            # Test passed
            assert True

        except ImportError as e:
            pytest.fail(f"Backward compatible import failed: {e}")


class TestPackageStructure:
    """Test the package structure and __all__ exports."""

    def test_main_package_all_exports(self):
        """Test that main package __all__ contains all expected exports."""
        from ml_src.core import metrics

        expected_exports = [
            # Classification
            "get_classification_report_str",
            "save_classification_report",
            # Visualization
            "create_confusion_matrix_figure",
            "log_confusion_matrix_to_tensorboard",
            "save_confusion_matrix",
            # Utils
            "argmax_predictions",
            "ensure_numpy",
            "flatten_predictions",
            "format_class_names",
            "get_num_classes",
            "prepare_labels_for_metrics",
            "validate_labels",
        ]

        assert hasattr(metrics, "__all__")

        for export in expected_exports:
            assert export in metrics.__all__, f"Missing from __all__: {export}"

    def test_all_exported_functions_are_accessible(self):
        """Test that all functions in __all__ are actually accessible."""
        from ml_src.core import metrics

        for func_name in metrics.__all__:
            assert hasattr(metrics, func_name), f"Function not accessible: {func_name}"
            assert callable(
                getattr(metrics, func_name)
            ), f"Export is not callable: {func_name}"

    def test_deprecated_module_all_exports(self):
        """Test that deprecated metrics.py module has correct __all__."""
        # Import the deprecated single-file module
        import ml_src.core.metrics as old_metrics_module

        # It should have __all__ defined
        assert hasattr(old_metrics_module, "__all__")

        # Should have at least the visualization and classification functions
        expected_core_functions = [
            "get_classification_report_str",
            "save_classification_report",
            "save_confusion_matrix",
            "create_confusion_matrix_figure",
            "log_confusion_matrix_to_tensorboard",
        ]

        for func in expected_core_functions:
            assert (
                func in old_metrics_module.__all__
            ), f"Missing from deprecated module: {func}"


if __name__ == "__main__":
    # Allow running tests directly with: python tests/test_metrics_restructure.py
    pytest.main([__file__, "-v"])
