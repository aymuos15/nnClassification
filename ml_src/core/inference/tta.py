"""Test-Time Augmentation (TTA) inference strategy."""

import torch
from loguru import logger

from ml_src.core.inference.base import BaseInferenceStrategy
from ml_src.core.metrics.segmentation import calculate_iou
from ml_src.core.transforms.tta import aggregate_predictions, get_tta_transforms


class TTAInference(BaseInferenceStrategy):
    """
    TTA inference strategy that applies multiple augmentations during inference.

    This strategy creates multiple augmented versions of each test image,
    runs inference on all versions, and aggregates the predictions to
    improve robustness and accuracy.

    Example:
        >>> # Config with TTA
        >>> config = {
        ...     'inference': {
        ...         'strategy': 'tta',
        ...         'tta': {
        ...             'augmentations': ['horizontal_flip', 'vertical_flip'],
        ...             'aggregation': 'mean'
        ...         }
        ...     }
        ... }
        >>> strategy = TTAInference(
        ...     augmentations=['horizontal_flip', 'vertical_flip'],
        ...     aggregation='mean'
        ... )
        >>> test_acc, results = strategy.run_inference(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     dataset_size=100,
        ...     device=device,
        ...     class_names=['cat', 'dog']
        ... )
    """

    def __init__(self, augmentations=None, aggregation="mean"):
        """
        Initialize TTA inference strategy.

        Args:
            augmentations: List of augmentation names to apply.
                Options: 'horizontal_flip', 'vertical_flip', 'rotate_90',
                        'rotate_180', 'rotate_270'
                If None, defaults to ['horizontal_flip']
            aggregation: Method to aggregate predictions - 'mean', 'max', or 'voting'
                - 'mean': Average logits (soft voting, recommended)
                - 'max': Take maximum logits
                - 'voting': Hard voting on predicted classes
        """
        if augmentations is None:
            augmentations = ["horizontal_flip"]

        self.augmentations = augmentations
        self.aggregation = aggregation
        self.tta_transform = get_tta_transforms(augmentations)

        logger.info(f"TTA augmentations: {augmentations}")
        logger.info(f"TTA aggregation: {aggregation}")

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None, task_type="classification", num_classes=None):
        """
        Run TTA inference on the dataset.

        For each image:
        1. Create multiple augmented versions
        2. Run model on all versions
        3. Aggregate predictions using specified method

        Args:
            model: Trained model to evaluate
            dataloader: DataLoader for test/inference data
            dataset_size: Total size of the test dataset
            device: Device to run inference on
            class_names: List of class names for human-readable results (optional)

        Returns:
            Tuple of (test_acc, per_sample_results) where:
                - test_acc: Overall accuracy as a tensor
                - per_sample_results: List of (true_label, pred_label, is_correct) tuples
        """
        model.eval()

        running_corrects = 0
        per_sample_results = []

        total_batches = len(dataloader)
        logger.info(f"Running TTA inference on {dataset_size} samples...")

        # Iterate over data
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Apply TTA augmentations to the entire batch
            augmented_batches = self.tta_transform.apply_batch(inputs)

            batch_outputs_list = []

            with torch.no_grad():
                for aug_batch in augmented_batches:
                    aug_batch = aug_batch.to(device)
                    outputs = model(aug_batch)
                    batch_outputs_list.append(outputs)

            # Aggregate predictions across augmentations
            batch_outputs = aggregate_predictions(batch_outputs_list, method=self.aggregation)
            _, preds = torch.max(batch_outputs, 1)

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

            # Progress logging
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                logger.info(
                    f"Progress: {batch_idx + 1}/{total_batches} batches "
                    f"({(batch_idx + 1) / total_batches * 100:.1f}%)"
                )

        test_acc = running_corrects.double() / dataset_size

        logger.info(f"Overall Test Acc (TTA): {test_acc:.4f}")

        return test_acc, per_sample_results
