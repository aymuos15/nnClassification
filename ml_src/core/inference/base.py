"""Base inference strategy abstract class."""

from abc import ABC, abstractmethod


class BaseInferenceStrategy(ABC):
    """
    Abstract base class for inference strategies.

    This class defines the interface that all inference strategies must implement.
    Different strategies can optimize inference for specific hardware (CPU, GPU),
    enable mixed precision for speed, or use distributed inference across multiple devices.

    Example:
        >>> class StandardInference(BaseInferenceStrategy):
        ...     def run_inference(self, model, dataloader, dataset_size, device, class_names=None):
        ...         # Standard PyTorch inference implementation
        ...         model.eval()
        ...         # ... inference logic
        ...         return accuracy, per_sample_results
        >>> strategy = StandardInference()
        >>> acc, results = strategy.run_inference(model, test_loader, 100, device)
    """

    @abstractmethod
    def run_inference(self, model, dataloader, dataset_size, device, class_names=None, task_type="classification", num_classes=None):
        """
        Run inference on dataset (task-aware).

        This method must be implemented by all concrete inference strategies.
        It should evaluate the model on the provided dataset and return
        both overall metric and per-sample results for detailed analysis.

        Args:
            model: Trained model to evaluate (should be in eval mode or will be set)
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on (e.g., torch.device('cuda:0') or 'cpu')
            class_names: List of class names for human-readable results (optional).
                        If None, returns integer labels. If provided, returns class names.
            task_type: Task type ('classification' or 'segmentation'). Default: 'classification'
            num_classes: Number of classes (required for segmentation metrics)

        Returns:
            For classification:
                Tuple of (accuracy, per_sample_results) where:
                    - accuracy: Overall accuracy as a float (tensor or scalar)
                    - per_sample_results: List of tuples (true_label, pred_label, is_correct)

            For segmentation:
                Tuple of (mean_iou, per_sample_results) where:
                    - mean_iou: Mean IoU across all classes
                    - per_sample_results: List of tuples (image, true_mask, pred_mask, iou)

        Example:
            >>> # Classification
            >>> acc, results = strategy.run_inference(
            ...     model, dataloader, 100, device,
            ...     class_names=['cat', 'dog'],
            ...     task_type='classification'
            ... )
            >>> print(results[0])  # ('cat', 'dog', False)
            >>>
            >>> # Segmentation
            >>> mean_iou, results = strategy.run_inference(
            ...     model, dataloader, 100, device,
            ...     class_names=['bg', 'fg'],
            ...     task_type='segmentation',
            ...     num_classes=2
            ... )
            >>> print(results[0])  # (image_tensor, true_mask, pred_mask, 0.85)
        """
        pass
