"""Testing module for model evaluation.

Note: This module is maintained for backward compatibility.
New code should use the inference strategies in ml_src.core.inference instead.
"""



def evaluate_model(model, dataloader, dataset_size, device, class_names=None):
    """
    Evaluate the model on the test dataset.

    Note:
        This function is maintained for backward compatibility.
        New code should use the inference strategies from ml_src.core.inference:

        >>> from ml_src.core.inference import get_inference_strategy
        >>> strategy = get_inference_strategy(config)
        >>> acc, results = strategy.run_inference(model, loader, size, device, class_names)

    Args:
        model: The trained model to test
        dataloader: DataLoader for test data
        dataset_size: Size of the test dataset
        device: Device to run testing on
        class_names: List of class names (optional)

    Returns:
        Tuple of (test_acc, per_sample_results)
        where per_sample_results is a list of (true_label, pred_label, is_correct)
    """
    # Use StandardInference internally to avoid code duplication
    from ml_src.core.inference.standard import StandardInference

    strategy = StandardInference()
    return strategy.run_inference(model, dataloader, dataset_size, device, class_names)
