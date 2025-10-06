"""
Metrics package for model evaluation and validation.

This package provides:
- Classification metrics (accuracy, precision, recall, F1, reports)
- Visualization functions (confusion matrices, plots)
- ONNX validation metrics (PyTorch vs ONNX comparison)
- Utility functions for metrics calculations
"""

from ml_src.core.metrics.classification import (
    get_classification_report_str,
    save_classification_report,
)
from ml_src.core.metrics.onnx_validation import (
    benchmark_inference_speed,
    compare_outputs,
    validate_onnx_model,
)
from ml_src.core.metrics.utils import (
    argmax_predictions,
    ensure_numpy,
    flatten_predictions,
    format_class_names,
    get_num_classes,
    prepare_labels_for_metrics,
    validate_labels,
)
from ml_src.core.metrics.visualization import (
    create_confusion_matrix_figure,
    log_confusion_matrix_to_tensorboard,
    save_confusion_matrix,
)

__all__ = [
    # Classification functions
    "get_classification_report_str",
    "save_classification_report",
    # ONNX validation functions
    "benchmark_inference_speed",
    "compare_outputs",
    "validate_onnx_model",
    # Visualization functions
    "create_confusion_matrix_figure",
    "log_confusion_matrix_to_tensorboard",
    "save_confusion_matrix",
    # Utility functions
    "argmax_predictions",
    "ensure_numpy",
    "flatten_predictions",
    "format_class_names",
    "get_num_classes",
    "prepare_labels_for_metrics",
    "validate_labels",
]
