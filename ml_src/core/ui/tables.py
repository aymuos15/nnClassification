"""Rich table display utilities."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import classification_report


def display_inference_results(
    per_sample_results, test_acc, dataset_size, run_dir, checkpoint_path, class_names
):
    """
    Display inference results using rich tables.

    Args:
        per_sample_results: List of (true_label, pred_label, is_correct) tuples
        test_acc: Test accuracy as float
        dataset_size: Number of test samples
        run_dir: Run directory path
        checkpoint_path: Checkpoint file path
        class_names: List of class names
    """
    console = Console()

    # Extract labels and predictions
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_labels = [class_to_idx[true_label] for true_label, _, _ in per_sample_results]
    test_preds = [class_to_idx[pred_label] for _, pred_label, _ in per_sample_results]

    # Per-sample results table
    sample_table = Table(title="Per-Sample Results", show_header=True, header_style="bold magenta")
    sample_table.add_column("Sample #", style="cyan", width=10)
    sample_table.add_column("True Label", style="blue", width=15)
    sample_table.add_column("Predicted", style="yellow", width=15)
    sample_table.add_column("Correct", style="white", width=10)

    for idx, (true_label, pred_label, is_correct) in enumerate(per_sample_results, 1):
        status = "✓" if is_correct else "✗"
        status_style = "green" if is_correct else "red"
        sample_table.add_row(
            str(idx),
            str(true_label),
            str(pred_label),
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print("\n")
    console.print(sample_table)
    console.print("\n")

    # Summary table
    import os

    summary_table = Table(title="Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan", width=20)
    summary_table.add_column("Value", style="green", width=15)

    summary_table.add_row("Run Directory", run_dir)
    summary_table.add_row("Checkpoint", os.path.basename(checkpoint_path))
    summary_table.add_row("Model", "ResNet18")
    summary_table.add_row("Test Samples", str(dataset_size))
    summary_table.add_row("Mean Accuracy", f"{test_acc:.4f}")

    console.print(summary_table)
    console.print("\n")

    # Classification report
    report = classification_report(test_labels, test_preds, target_names=class_names)
    console.print(Panel(report, title="Classification Report", expand=False))

    # Remind user about TensorBoard
    console.print("\n")
    console.print("[bold green]View test metrics in TensorBoard:[/bold green]")
    console.print(f"  tensorboard --logdir {run_dir}/tensorboard")
    console.print("\n")
