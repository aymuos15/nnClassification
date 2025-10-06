"""Test the quick start workflow from the documentation."""

import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.workflow
@pytest.mark.slow
def test_quick_start_workflow(test_dataset_dir, temp_workspace):
    """
    Test the complete quick start workflow:
    1. Generate splits
    2. Generate config
    3. Train model (2 epochs)
    4. Run inference
    """
    dataset_dir = test_dataset_dir
    configs_dir = temp_workspace["configs"]
    runs_dir = temp_workspace["runs"]

    # Skip if dataset doesn't exist
    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Step 1: Generate splits
    result = subprocess.run(
        [
            "ml-split",
            "--raw_data",
            str(dataset_dir / "raw"),
            "--folds",
            "3",  # Use fewer folds for faster tests
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ml-split failed: {result.stderr}"
    assert (dataset_dir / "splits" / "test.txt").exists()
    assert (dataset_dir / "splits" / "fold_0_train.txt").exists()

    # Step 2: Generate config
    config_path = configs_dir / "mnist_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "2",
            "--batch_size",
            "8"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ml-init-config failed: {result.stderr}"
    assert config_path.exists()

    # Step 3: Train model (quick test with 2 epochs)
    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    assert result.returncode == 0, f"ml-train failed: {result.stderr}"

    # Verify training outputs exist
    run_dirs = list(runs_dir.glob("mnist_*"))
    assert len(run_dirs) > 0, "No run directory created"
    run_dir = run_dirs[0]

    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "weights" / "last.pt").exists()
    assert (run_dir / "logs" / "train.log").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "summary.txt").exists()

    # Step 4: Run inference
    result = subprocess.run(
        ["ml-inference", "--checkpoint_path", str(run_dir / "weights" / "best.pt")],
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout
    )
    assert result.returncode == 0, f"ml-inference failed: {result.stderr}"

    # Verify test classification report was created
    assert (run_dir / "logs" / "classification_report_test.txt").exists()


@pytest.mark.workflow
@pytest.mark.slow
def test_training_with_overrides(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test training with CLI parameter overrides.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "override_test_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Train with overrides
    result = subprocess.run(
        [
            "ml-train",
            "--config",
            str(config_path),
            "--batch_size",
            "16",
            "--lr",
            "0.01",
            "--num_epochs",
            "2",
            "--num_workers",
            "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Training with overrides failed: {result.stderr}"

    # Verify run name includes overrides
    run_dirs = list(Path("runs").glob("*batch_16*lr_0.01*"))
    assert len(run_dirs) > 0, "Run directory with overrides not found"


@pytest.mark.workflow
@pytest.mark.slow
def test_training_specific_fold(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test training on a specific fold.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "fold_test_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Train on fold 1
    result = subprocess.run(
        [
            "ml-train",
            "--config",
            str(config_path),
            "--fold",
            "1",
            "--num_workers",
            "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Training on fold 1 failed: {result.stderr}"

    # Verify run directory includes fold number
    run_dirs = list(Path("runs").glob("*fold_1*"))
    assert len(run_dirs) > 0, "Run directory with fold_1 not found"
