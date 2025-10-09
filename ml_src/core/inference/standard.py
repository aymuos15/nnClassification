"""Standard inference strategy using basic PyTorch."""

import torch
from loguru import logger

from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.metrics.segmentation import calculate_iou


class StandardInference(BaseInferenceStrategy):
    """
    Standard inference strategy using basic PyTorch operations.

    This is the default inference strategy that uses manual device management
    and standard PyTorch inference without optimizations like mixed precision
    or distributed processing. It works on both CPU and GPU.

    This strategy is equivalent to the original test_model() function and
    provides baseline performance for comparison with optimized strategies.

    Example:
        >>> from ml_src.core.inference.standard import StandardInference
        >>> strategy = StandardInference()
        >>> test_acc, results = strategy.run_inference(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     dataset_size=100,
        ...     device=torch.device('cuda:0'),
        ...     class_names=['cat', 'dog']
        ... )
        >>> print(f"Test Accuracy: {test_acc:.4f}")
        >>> print(f"First prediction: {results[0]}")  # ('cat', 'dog', False)
    """

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None, task_type="classification", num_classes=None):
        """
        Run standard PyTorch inference on the dataset (task-aware).

        Sets the model to evaluation mode, disables gradient computation,
        and iterates through the dataloader to collect predictions and
        compute metrics (accuracy for classification, IoU for segmentation).

        Args:
            model: Trained model to evaluate
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on
            class_names: List of class names for human-readable results (optional)
            task_type: Task type ('classification' or 'segmentation')
            num_classes: Number of classes (required for segmentation)

        Returns:
            For classification:
                Tuple of (test_acc, per_sample_results)
            For segmentation:
                Tuple of (mean_iou, per_sample_results)

        Example:
            >>> strategy = StandardInference()
            >>> metric, results = strategy.run_inference(
            ...     model, test_loader, 100, device,
            ...     task_type='classification'
            ... )
        """
        model.eval()  # Set model to evaluate mode

        if task_type == "classification":
            return self._run_classification_inference(
                model, dataloader, dataset_size, device, class_names
            )
        else:  # segmentation
            return self._run_segmentation_inference(
                model, dataloader, dataset_size, device, class_names, num_classes
            )

    def _run_classification_inference(self, model, dataloader, dataset_size, device, class_names):
        """Run classification inference."""
        running_corrects = 0
        per_sample_results = []

        # Iterate over data
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            # Store per-sample results
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                is_correct = pred_label == true_label

                if class_names:
                    true_name = class_names[true_label]
                    pred_name = class_names[pred_label]
                    per_sample_results.append((true_name, pred_name, is_correct))
                else:
                    per_sample_results.append((true_label, pred_label, is_correct))

            # Statistics
            running_corrects += torch.sum(preds == labels.data)

        test_acc = running_corrects.double() / dataset_size

        logger.info(f"Overall Test Acc: {test_acc:.4f}")

        return test_acc, per_sample_results

    def _run_segmentation_inference(self, model, dataloader, dataset_size, device, class_names, num_classes):
        """Run segmentation inference."""
        total_iou = 0.0
        num_batches = 0
        per_sample_results = []

        # Iterate over data
        for inputs, masks in dataloader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Forward
            with torch.no_grad():
                outputs = model(inputs)  # [B, C, H, W]
                preds = torch.argmax(outputs, 1)  # [B, H, W]

            # Calculate IoU for this batch
            batch_iou = calculate_iou(preds, masks, num_classes)
            total_iou += batch_iou * inputs.size(0)
            num_batches += inputs.size(0)

            # Store per-sample results
            for i in range(len(masks)):
                per_sample_results.append((
                    inputs[i].cpu(),
                    masks[i].cpu(),
                    preds[i].cpu(),
                    batch_iou
                ))

        mean_iou = total_iou / num_batches

        logger.info(f"Overall Mean IoU: {mean_iou:.4f}")

        return mean_iou, per_sample_results
