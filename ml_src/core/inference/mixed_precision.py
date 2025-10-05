"""Mixed precision inference strategy using PyTorch AMP."""

import torch
from loguru import logger

from ml_src.core.inference.base import BaseInferenceStrategy


class MixedPrecisionInference(BaseInferenceStrategy):
    """
    Mixed precision inference strategy using PyTorch Automatic Mixed Precision (AMP).

    This inference strategy uses torch.amp to enable mixed precision inference:
    - 2-3x faster inference on GPU with reduced memory usage
    - Support for both float16 and bfloat16 dtypes
    - Automatic fallback to standard inference on CPU

    Mixed precision inference is ideal for production deployments where
    inference speed is critical and GPU resources are available.

    Args:
        amp_dtype: torch dtype for mixed precision (torch.float16 or torch.bfloat16)

    Example:
        >>> from ml_src.core.inference.mixed_precision import MixedPrecisionInference
        >>> import torch
        >>> # Use float16 for maximum speed
        >>> strategy = MixedPrecisionInference(amp_dtype=torch.float16)
        >>> test_acc, results = strategy.run_inference(
        ...     model=model,
        ...     dataloader=test_loader,
        ...     dataset_size=100,
        ...     device=torch.device('cuda:0'),
        ...     class_names=['cat', 'dog']
        ... )
        >>> # Or use bfloat16 for better numerical stability
        >>> strategy = MixedPrecisionInference(amp_dtype=torch.bfloat16)
        >>> test_acc, results = strategy.run_inference(model, test_loader, 100, device)
    """

    def __init__(self, amp_dtype=torch.float16):
        """
        Initialize mixed precision inference strategy.

        Args:
            amp_dtype: torch dtype for mixed precision.
                      Options: torch.float16 (faster) or torch.bfloat16 (more stable)
                      Default: torch.float16
        """
        self.amp_dtype = amp_dtype
        self.amp_dtype_name = "float16" if amp_dtype == torch.float16 else "bfloat16"

    def run_inference(self, model, dataloader, dataset_size, device, class_names=None):
        """
        Run mixed precision inference on the dataset.

        Uses torch.amp.autocast for the forward pass to enable mixed precision.
        Falls back to standard inference on CPU with a warning.

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
            >>> strategy = MixedPrecisionInference(amp_dtype=torch.float16)
            >>> acc, results = strategy.run_inference(model, test_loader, 100, device)
            >>> print(f"Accuracy: {acc:.4f}")
        """
        model.eval()  # Set model to evaluate mode

        # Check if device is CUDA
        if device.type != "cuda":
            logger.warning(
                f"Mixed precision inference requested but device is {device.type}. "
                "Falling back to standard inference (AMP requires CUDA)."
            )
            use_amp = False
        else:
            use_amp = True
            logger.info(f"Mixed precision inference enabled with dtype={self.amp_dtype_name}")

        running_corrects = 0
        per_sample_results = []

        # Iterate over data
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward with optional autocast
            with torch.no_grad():
                if use_amp:
                    # Mixed precision forward pass
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                else:
                    # Standard forward pass on CPU
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
