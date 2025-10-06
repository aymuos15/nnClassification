"""Shared utility functions for metrics calculations."""

import numpy as np
import torch


def ensure_numpy(data):
    """
    Convert tensor, list, or other array-like data to numpy array.

    This function provides a unified interface for converting various data types
    (PyTorch tensors, lists, tuples, etc.) into numpy arrays for use in metrics
    calculations and sklearn functions.

    Args:
        data: Input data in various formats:
            - torch.Tensor: Will be detached, moved to CPU, and converted to numpy
            - numpy.ndarray: Will be returned as-is
            - list/tuple: Will be converted to numpy array
            - Other array-like objects: Will attempt conversion via np.asarray

    Returns:
        numpy.ndarray: The input data as a numpy array

    Examples:
        >>> import torch
        >>> tensor_data = torch.tensor([1, 2, 3])
        >>> ensure_numpy(tensor_data)
        array([1, 2, 3])

        >>> list_data = [1, 2, 3]
        >>> ensure_numpy(list_data)
        array([1, 2, 3])

        >>> numpy_data = np.array([1, 2, 3])
        >>> ensure_numpy(numpy_data)
        array([1, 2, 3])
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data)


def format_class_names(num_classes, class_names=None):
    """
    Generate or validate class names for classification tasks.

    If class_names is provided, validates that it has the correct length.
    If not provided, generates default names like ['Class 0', 'Class 1', ...].

    Args:
        num_classes (int): Number of classes in the classification task
        class_names (list[str], optional): List of class names. If None, generates
            default names. Defaults to None.

    Returns:
        list[str]: List of class names with length equal to num_classes

    Raises:
        ValueError: If provided class_names has incorrect length

    Examples:
        >>> format_class_names(3)
        ['Class 0', 'Class 1', 'Class 2']

        >>> format_class_names(2, ['cat', 'dog'])
        ['cat', 'dog']

        >>> format_class_names(2, ['cat', 'dog', 'bird'])
        Traceback (most recent call last):
            ...
        ValueError: Number of class names (3) does not match num_classes (2)
    """
    if class_names is None:
        return [f"Class {i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match "
            f"num_classes ({num_classes})"
        )

    return class_names


def validate_labels(y_true, y_pred):
    """
    Validate that true and predicted labels have compatible shapes.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)

    Raises:
        ValueError: If y_true and y_pred have different lengths

    Examples:
        >>> validate_labels([0, 1, 2], [0, 1, 1])
        # Returns None (validation passed)

        >>> validate_labels([0, 1], [0, 1, 2])
        Traceback (most recent call last):
            ...
        ValueError: y_true and y_pred must have the same length (got 2 and 3)
    """
    y_true = ensure_numpy(y_true)
    y_pred = ensure_numpy(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )


def get_num_classes(y_true, y_pred):
    """
    Infer the number of classes from true and predicted labels.

    Determines the number of unique classes by taking the maximum value
    across both y_true and y_pred and adding 1 (assumes 0-indexed labels).

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)

    Returns:
        int: Number of classes

    Examples:
        >>> get_num_classes([0, 1, 2, 1], [0, 1, 1, 2])
        3

        >>> get_num_classes([0, 1], [0, 1, 2])
        3

        >>> import torch
        >>> get_num_classes(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 1]))
        3
    """
    y_true = ensure_numpy(y_true)
    y_pred = ensure_numpy(y_pred)

    return max(y_true.max(), y_pred.max()) + 1


def prepare_labels_for_metrics(y_true, y_pred, class_names=None):
    """
    Prepare and validate labels for metrics calculation.

    This is a convenience function that combines validation, conversion,
    and class name formatting in a single call.

    Args:
        y_true: True labels (array-like or torch.Tensor)
        y_pred: Predicted labels (array-like or torch.Tensor)
        class_names (list[str], optional): List of class names. If None,
            generates default names. Defaults to None.

    Returns:
        tuple: (y_true_np, y_pred_np, class_names_list) where:
            - y_true_np: True labels as numpy array
            - y_pred_np: Predicted labels as numpy array
            - class_names_list: List of class names

    Raises:
        ValueError: If y_true and y_pred have incompatible shapes or
            if class_names has incorrect length

    Examples:
        >>> import torch
        >>> y_true = torch.tensor([0, 1, 2, 1])
        >>> y_pred = torch.tensor([0, 1, 1, 2])
        >>> y_true_np, y_pred_np, names = prepare_labels_for_metrics(y_true, y_pred)
        >>> names
        ['Class 0', 'Class 1', 'Class 2']
        >>> y_true_np
        array([0, 1, 2, 1])

        >>> y_true = [0, 1, 2]
        >>> y_pred = [0, 1, 1]
        >>> custom_names = ['cat', 'dog', 'bird']
        >>> _, _, names = prepare_labels_for_metrics(y_true, y_pred, custom_names)
        >>> names
        ['cat', 'dog', 'bird']
    """
    # Convert to numpy
    y_true_np = ensure_numpy(y_true)
    y_pred_np = ensure_numpy(y_pred)

    # Validate
    validate_labels(y_true_np, y_pred_np)

    # Get or validate class names
    num_classes = get_num_classes(y_true_np, y_pred_np)
    class_names_list = format_class_names(num_classes, class_names)

    return y_true_np, y_pred_np, class_names_list


def flatten_predictions(predictions):
    """
    Flatten multi-dimensional predictions to 1D array.

    Useful for handling predictions from models that output
    multi-dimensional arrays (e.g., batch predictions).

    Args:
        predictions: Predictions in various formats (tensor, array, list)
            Can be multi-dimensional.

    Returns:
        numpy.ndarray: Flattened 1D numpy array

    Examples:
        >>> import torch
        >>> predictions = torch.tensor([[0, 1], [2, 1]])
        >>> flatten_predictions(predictions)
        array([0, 1, 2, 1])

        >>> predictions = [[0, 1], [2, 1]]
        >>> flatten_predictions(predictions)
        array([0, 1, 2, 1])

        >>> predictions = np.array([[[0]], [[1]], [[2]]])
        >>> flatten_predictions(predictions)
        array([0, 1, 2])
    """
    predictions_np = ensure_numpy(predictions)
    return predictions_np.flatten()


def argmax_predictions(logits, axis=-1):
    """
    Convert model logits/probabilities to class predictions.

    Applies argmax operation along the specified axis to convert
    continuous model outputs (logits or probabilities) into discrete
    class predictions.

    Args:
        logits: Model outputs (array-like or torch.Tensor)
            Shape: (..., num_classes) or similar
        axis (int, optional): Axis along which to apply argmax.
            Defaults to -1 (last axis).

    Returns:
        numpy.ndarray: Class predictions as integers

    Examples:
        >>> import torch
        >>> logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> argmax_predictions(logits)
        array([1, 0, 1])

        >>> logits = [[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]
        >>> argmax_predictions(logits)
        array([1, 0])

        >>> # Works with any axis
        >>> logits = np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]])
        >>> argmax_predictions(logits, axis=-1).shape
        (2, 2)
    """
    logits_np = ensure_numpy(logits)
    return np.argmax(logits_np, axis=axis)
