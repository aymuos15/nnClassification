"""Accelerate-based inference for multi-GPU and distributed inference."""

import torch
from loguru import logger

from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.metrics.segmentation import calculate_iou

# Conditional import for accelerate
try:
    from accelerate import Accelerator

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False


class AccelerateInference(BaseInferenceStrategy):
    """
    HuggingFace Accelerate inference strategy for multi-GPU and distributed inference.

    This inference strategy uses HuggingFace Accelerate to provide:
    - Seamless multi-GPU inference (distributes across available GPUs)
    - Automatic device placement
    - Result aggregation across processes
    - Single-device fallback (works with both `python ml-inference` and `accelerate launch`)

    The strategy automatically handles:
    - Model/dataloader preparation
    - Device placement (no manual .to(device) needed)
    - Multi-process synchronization and result gathering
    - Near-linear scaling with GPU count

    Usage:
        Single device:
            ml-inference --checkpoint_path runs/my_run/weights/best.pt

        Multi-GPU (after `accelerate config`):
            accelerate launch ml-inference --checkpoint_path runs/my_run/weights/best.pt

    Example:
        >>> from ml_src.core.inference.accelerate import AccelerateInference
        >>> strategy = AccelerateInference()
        >>> test_acc, results = strategy.run_inference(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     dataset_size=100,
        ...     device=torch.device('cuda:0'),  # Ignored (Accelerator handles device)
        ...     class_names=['cat', 'dog']
        ... )
    """

    def __init__(self):
        """Initialize Accelerate inference strategy."""
        if not _ACCELERATE_AVAILABLE:
            raise ImportError(
                "AccelerateInference requires accelerate. "
                "Install with: pip install accelerate\n"
                "Or install with accelerate extras: pip install -e '.[accelerate]'"
            )

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None, task_type="classification", num_classes=None):
        """
        Run Accelerate-based inference on the dataset.

        Uses Accelerate to distribute inference across available GPUs and
        aggregate results from all processes. Only the main process returns
        the final results.

        Args:
            model: Trained model to evaluate
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on (ignored, Accelerator handles device)
            class_names: List of class names for human-readable results (optional)

        Returns:
            Tuple of (test_acc, per_sample_results) where:
                - test_acc: Overall accuracy as a tensor
                - per_sample_results: List of (true_label, pred_label, is_correct) tuples

        Example:
            >>> strategy = AccelerateInference()
            >>> acc, results = strategy.run_inference(model, test_loader, 100, device)
            >>> print(f"Accuracy: {acc:.4f}")
        """
        # Initialize Accelerator
        accelerator = Accelerator()

        # Prepare model and dataloader
        # Accelerator handles device placement and distributed wrapping
        model, dataloader = accelerator.prepare(model, dataloader)

        # Update device to match Accelerator's device
        device = accelerator.device

        # Log device info (only on main process)
        if accelerator.is_main_process:
            logger.info(f"Accelerator device: {accelerator.device}")
            logger.info(f"Distributed type: {accelerator.distributed_type}")
            logger.info(f"Number of processes: {accelerator.num_processes}")

        model.eval()  # Set model to evaluate mode

        # Collect predictions and labels from this process
        all_preds = []
        all_labels = []

        # Iterate over data
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Note: inputs and labels are already on the correct device
                # thanks to Accelerator's prepare()

                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Gather predictions and labels across all processes
                all_preds_batch = accelerator.gather(preds)
                all_labels_batch = accelerator.gather(labels)

                # Only main process stores results
                if accelerator.is_main_process:
                    all_preds.extend(all_preds_batch.cpu().tolist())
                    all_labels.extend(all_labels_batch.cpu().tolist())

        # Only main process computes final results
        if accelerator.is_main_process:
            # Trim to dataset size (in case of padding from distributed sampling)
            all_preds = all_preds[:dataset_size]
            all_labels = all_labels[:dataset_size]

            # Compute accuracy
            running_corrects = sum(1 for pred, label in zip(all_preds, all_labels) if pred == label)
            test_acc = torch.tensor(running_corrects, dtype=torch.float64) / dataset_size

            # Build per-sample results
            per_sample_results = []
            for true_label, pred_label in zip(all_labels, all_preds):
                is_correct = pred_label == true_label

                if class_names:
                    true_name = class_names[true_label]
                    pred_name = class_names[pred_label]
                    per_sample_results.append((true_name, pred_name, is_correct))
                else:
                    per_sample_results.append((true_label, pred_label, is_correct))

            logger.info(f"Overall Test Acc: {test_acc:.4f}")

            return test_acc, per_sample_results
        else:
            # Non-main processes return None
            # (In distributed inference, only main process needs results)
            return None, None
