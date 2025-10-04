"""Pytest configuration and fixtures for testing."""

import pytest
import torch


@pytest.fixture
def device():
    """Fixture providing CPU device for testing."""
    return 'cpu'


@pytest.fixture
def num_classes_small():
    """Fixture for binary classification (2 classes)."""
    return 2


@pytest.fixture
def num_classes_medium():
    """Fixture for medium classification (10 classes)."""
    return 10


@pytest.fixture
def num_classes_large():
    """Fixture for large classification (100 classes)."""
    return 100


@pytest.fixture
def batch_size():
    """Fixture for test batch size."""
    return 2


@pytest.fixture
def sample_input_224(batch_size):
    """Fixture providing sample 224x224 input tensor."""
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def sample_input_299(batch_size):
    """Fixture providing sample 299x299 input tensor (for Inception v3)."""
    return torch.randn(batch_size, 3, 299, 299)


@pytest.fixture
def base_config_template():
    """Fixture providing base configuration template."""
    return {
        'model': {
            'type': 'base',
            'architecture': 'resnet18',
            'num_classes': 2,
            'weights': None
        }
    }
