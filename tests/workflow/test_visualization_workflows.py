"""Test visualization workflows."""

import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.workflow
@pytest.mark.slow
def test_tensorboard_visualization_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test TensorBoard visualization commands (without actually launching server).
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Train a model first
    config_path = configs_dir / "viz_test_config.yaml"
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

    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

    # Find run directory
    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    run_dir = run_dirs[-1]

    # Test samples visualization
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "samples",
            "--run_dir",
            str(run_dir),
            "--split",
            "train",
            "--num_images",
            "8"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Samples visualization failed: {result.stderr}"

    # Test predictions visualization
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "predictions",
            "--run_dir",
            str(run_dir),
            "--split",
            "val",
            "--checkpoint",
            "best.pt",
            "--num_images",
            "8"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Predictions visualization failed: {result.stderr}"


@pytest.mark.workflow
@pytest.mark.slow
def test_clean_tensorboard_logs(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test cleaning TensorBoard logs.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Train a model to create TensorBoard logs
    config_path = configs_dir / "clean_test_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "1"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

    # Find run directory
    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    run_dir = run_dirs[-1]
    tensorboard_dir = run_dir / "tensorboard"

    # Verify TensorBoard logs exist
    assert tensorboard_dir.exists()
    tb_files_before = list(tensorboard_dir.rglob("*"))
    assert len(tb_files_before) > 0

    # Clean specific run
    result = subprocess.run(
        ["ml-visualise", "--mode", "clean", "--run_dir", str(run_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Clean command failed: {result.stderr}"

    # Verify TensorBoard logs were removed
    tb_files_after = list(tensorboard_dir.rglob("*"))
    assert len(tb_files_after) < len(tb_files_before), "TensorBoard logs not cleaned"


@pytest.mark.workflow
@pytest.mark.slow
def test_visualization_on_different_splits(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test visualization on different data splits.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Train a model
    config_path = configs_dir / "split_viz_config.yaml"
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

    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    run_dir = run_dirs[-1]

    # Test visualization on train split
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "samples",
            "--run_dir",
            str(run_dir),
            "--split",
            "train",
            "--num_images",
            "4"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0

    # Test visualization on val split
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "predictions",
            "--run_dir",
            str(run_dir),
            "--split",
            "val",
            "--num_images",
            "4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0

    # Test visualization on test split
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "predictions",
            "--run_dir",
            str(run_dir),
            "--split",
            "test",
            "--num_images",
            "4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
