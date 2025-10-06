"""Pytest configuration and fixtures for workflow tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import datasets, transforms


@pytest.fixture(scope="session")
def setup_mnist_dataset():
    """
    Download and setup MNIST dataset in expected folder structure.

    Creates data/mnist/raw/{0-9}/*.png with subset of MNIST images.
    Uses 100 images per class from train set + all test set for faster tests.
    """
    # Use absolute path to avoid issues with os.chdir in temp_workspace
    original_cwd = Path.cwd()
    mnist_dir = original_cwd / "data" / "mnist"
    raw_dir = mnist_dir / "raw"

    # Skip if already exists
    if raw_dir.exists() and any(raw_dir.iterdir()):
        return mnist_dir

    # Create directory
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download MNIST
    transform = transforms.Compose([transforms.ToTensor()])

    # Download train and test sets to absolute path
    pytorch_mnist_dir = original_cwd / "data" / "mnist_pytorch"
    train_dataset = datasets.MNIST(
        root=str(pytorch_mnist_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(pytorch_mnist_dir), train=False, download=True, transform=transform
    )

    # Use minimal subset for fast tests: 2 train + 2 test images per class
    # Total: 20 train + 20 test = 40 images
    train_images_per_class = 2
    test_images_per_class = 2
    train_counts = {i: 0 for i in range(10)}
    test_counts = {i: 0 for i in range(10)}

    # Create class directories
    for class_idx in range(10):
        (raw_dir / str(class_idx)).mkdir(exist_ok=True)

    # Process train set (subset)
    for idx, (image, label) in enumerate(train_dataset):
        if train_counts[label] >= train_images_per_class:
            # Check if all classes have enough images
            if all(count >= train_images_per_class for count in train_counts.values()):
                break
            continue

        # Convert tensor to PIL Image and save
        img = transforms.ToPILImage()(image)
        img_path = raw_dir / str(label) / f"train_{idx:05d}.png"
        img.save(img_path)
        train_counts[label] += 1

    # Process test set (subset for faster testing)
    for idx, (image, label) in enumerate(test_dataset):
        if test_counts[label] >= test_images_per_class:
            # Check if all classes have enough images
            if all(count >= test_images_per_class for count in test_counts.values()):
                break
            continue

        img = transforms.ToPILImage()(image)
        img_path = raw_dir / str(label) / f"test_{idx:05d}.png"
        img.save(img_path)
        test_counts[label] += 1

    return mnist_dir


@pytest.fixture(scope="session")
def test_dataset_dir(setup_mnist_dataset):
    """Return path to the MNIST test dataset."""
    return setup_mnist_dataset


@pytest.fixture(scope="function")
def temp_workspace(tmp_path):
    """
    Create a temporary workspace for workflow tests.

    Returns a dictionary with paths for configs, runs, and optuna db.
    """
    workspace = {
        "root": tmp_path,
        "configs": tmp_path / "configs",
        "runs": tmp_path / "runs",
        "optuna_db": tmp_path / "optuna_test.db",
    }

    # Create directories
    workspace["configs"].mkdir(parents=True, exist_ok=True)
    workspace["runs"].mkdir(parents=True, exist_ok=True)

    # Change to temp directory
    original_dir = Path.cwd()
    os.chdir(tmp_path)

    yield workspace

    # Cleanup: change back to original directory
    os.chdir(original_dir)


@pytest.fixture(scope="function")
def dataset_with_splits(test_dataset_dir, tmp_path):
    """
    Ensure dataset has splits, or create them in temp location.

    Returns path to dataset with splits.
    """
    import subprocess

    splits_dir = test_dataset_dir / "splits"

    if not splits_dir.exists() or not list(splits_dir.glob("*.txt")):
        # Need to create splits - copy dataset to temp and create splits there
        temp_dataset = tmp_path / "test_dataset"
        temp_dataset.mkdir(parents=True)

        # Copy raw data
        raw_src = test_dataset_dir / "raw"
        raw_dst = temp_dataset / "raw"
        if raw_src.exists():
            shutil.copytree(raw_src, raw_dst)
        else:
            pytest.skip(f"Test dataset not found at {test_dataset_dir}")

        # Create splits using ml-split
        result = subprocess.run(
            ["ml-split", "--raw_data", str(raw_dst), "--folds", "3"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"ml-split failed: {result.stderr}")

        return temp_dataset
    else:
        return test_dataset_dir


@pytest.fixture(scope="function")
def minimal_train_config():
    """Return minimal training configuration for fast tests."""
    return {
        "num_epochs": 2,
        "batch_size": 8,
        "num_workers": 0,  # Avoid multiprocessing issues in tests
        "device": "cpu",  # Force CPU for reproducibility
    }
