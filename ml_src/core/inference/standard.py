"""Standard inference strategy using basic PyTorch."""

import torch
from loguru import logger

from ml_src.core.inference.base import BaseInferenceStrategy


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

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None):
        """
        Run standard PyTorch inference on the dataset.

        Sets the model to evaluation mode, disables gradient computation,
        and iterates through the dataloader to collect predictions and
        compute accuracy.

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

        Example:
            >>> strategy = StandardInference()
            >>> acc, results = strategy.run_inference(model, test_loader, 100, device)
            >>> print(f"Accuracy: {acc:.4f}")
            >>> correct = sum(1 for _, _, is_correct in results if is_correct)
            >>> print(f"Correct predictions: {correct}/{len(results)}")
        """
        model.eval()  # Set model to evaluate mode

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
