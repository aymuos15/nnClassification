"""
Metrics utilities for model evaluation.

DEPRECATED: This module is deprecated. Please import from ml_src.core.metrics package instead.

All functions are re-exported from the metrics package for backward compatibility:
- Classification functions → ml_src.core.metrics.classification
- Visualization functions → ml_src.core.metrics.visualization
"""

# Re-export all functions from the metrics package for backward compatibility
from ml_src.core.metrics import (
    create_confusion_matrix_figure,
    get_classification_report_str,
    log_confusion_matrix_to_tensorboard,
    save_classification_report,
    save_confusion_matrix,
)

__all__ = [
    "get_classification_report_str",
    "save_classification_report",
    "save_confusion_matrix",
    "create_confusion_matrix_figure",
    "log_confusion_matrix_to_tensorboard",
]
