"""Testing module for model evaluation."""

import torch
from loguru import logger


def test_model(model, dataloader, dataset_size, device, class_names=None):
    """
    Test the model on the test dataset.

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
