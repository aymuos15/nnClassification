"""Metrics utilities for model evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Generate and save confusion matrix plot.

    Args:
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names
        save_path: Path to save the plot (e.g., 'confusion_matrix.png')
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.success(f"Saved confusion matrix to {save_path}")


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


def create_confusion_matrix_figure(y_true, y_pred, class_names):
    """
    Create confusion matrix as a matplotlib figure for TensorBoard.

    Args:
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names

    Returns:
        matplotlib.figure.Figure: Figure object containing confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    return fig


def log_confusion_matrix_to_tensorboard(writer, y_true, y_pred, class_names, tag, global_step):
    """
    Log confusion matrix to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names
        tag: Tag name for the confusion matrix (e.g., 'confusion_matrix/train')
        global_step: Global step value (epoch number)
    """
    fig = create_confusion_matrix_figure(y_true, y_pred, class_names)
    writer.add_figure(tag, fig, global_step)
    plt.close(fig)
    logger.debug(f"Logged confusion matrix to TensorBoard: {tag}")


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
