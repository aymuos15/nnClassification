"""Task type detection and utilities.

This module provides utilities for detecting and validating task types
(classification vs segmentation) from configuration.

Example:
    >>> config = {'task': {'type': 'segmentation'}}
    >>> task_type = get_task_type(config)
    >>> if is_segmentation_task(config):
    ...     # Use segmentation-specific logic
"""


def get_task_type(config):
    """
    Get task type from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        str: Task type ('classification' or 'segmentation')
        Defaults to 'classification' if not specified.

    Example:
        >>> config = {'task': {'type': 'segmentation'}}
        >>> get_task_type(config)
        'segmentation'
        >>> get_task_type({})
        'classification'
    """
    return config.get("task", {}).get("type", "classification")


def is_segmentation_task(config):
    """
    Check if task is segmentation.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if task is segmentation, False otherwise

    Example:
        >>> config = {'task': {'type': 'segmentation'}}
        >>> is_segmentation_task(config)
        True
    """
    return get_task_type(config) == "segmentation"


def is_classification_task(config):
    """
    Check if task is classification.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if task is classification, False otherwise

    Example:
        >>> config = {'task': {'type': 'classification'}}
        >>> is_classification_task(config)
        True
    """
    return get_task_type(config) == "classification"


def validate_task_type(config):
    """
    Validate that task type is supported.

    Args:
        config: Configuration dictionary

    Returns:
        str: Validated task type

    Raises:
        ValueError: If task type is not 'classification' or 'segmentation'

    Example:
        >>> config = {'task': {'type': 'segmentation'}}
        >>> validate_task_type(config)
        'segmentation'
        >>> config = {'task': {'type': 'detection'}}
        >>> validate_task_type(config)
        Traceback (most recent call last):
        ...
        ValueError: Unsupported task type: 'detection'. Must be 'classification' or 'segmentation'
    """
    task_type = get_task_type(config)
    if task_type not in ["classification", "segmentation"]:
        raise ValueError(
            f"Unsupported task type: '{task_type}'. " "Must be 'classification' or 'segmentation'"
        )
    return task_type


__all__ = [
    "get_task_type",
    "is_segmentation_task",
    "is_classification_task",
    "validate_task_type",
]
