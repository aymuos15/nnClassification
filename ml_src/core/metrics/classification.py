"""Classification metrics: accuracy, precision, recall, F1, classification reports."""

from loguru import logger
from sklearn.metrics import classification_report


def get_classification_report_str(y_true, y_pred, class_names):
    """
    Generate classification report as a string.

    Args:
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names

    Returns:
        str: Classification report as formatted string
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    return report


def save_classification_report(y_true, y_pred, class_names, save_path):
    """
    Generate and save classification report as formatted text.

    Args:
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names
        save_path: Path to save the report (e.g., 'classification_report.txt')
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    with open(save_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    logger.opt(colors=True).info(f"<fg 208>Saved classification report to {save_path}</fg 208>")
