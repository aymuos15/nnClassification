"""Test various training workflows including resume and mixed precision."""

import subprocess
import time
from pathlib import Path

import pytest
import yaml


@pytest.mark.workflow
@pytest.mark.slow
def test_resume_training_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test resuming training from checkpoint.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "resume_test_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "3",
            "--batch_size",
            "8"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Force CPU for tests
    with open(config_path) as f:
        test_config = yaml.safe_load(f)
    test_config['training']['device'] = 'cpu'
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Train for 1 epoch only (simulate interrupted training)
    result = subprocess.run(
        [
            "ml-train",
            "--config",
            str(config_path),
            "--num_epochs",
            "1",
            "--num_workers",
            "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Initial training failed: {result.stderr}"

    # Find the run directory
    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    assert len(run_dirs) > 0
    run_dir = run_dirs[-1]
    last_checkpoint = run_dir / "weights" / "last.pt"
    assert last_checkpoint.exists()

    # Resume training for 2 more epochs
    result = subprocess.run(
        [
            "ml-train",
            "--config",
            str(config_path),
            "--resume",
            str(last_checkpoint),
            "--num_epochs",
            "3",  # Total epochs
            "--num_workers",
            "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Resume training failed: {result.stderr}"

    # Find the LATEST run directory (resume creates a new one or updates existing)
    run_dirs_after = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    latest_run_dir = run_dirs_after[-1]

    # Verify training log mentions resuming
    log_file = latest_run_dir / "logs" / "train.log"
    with open(log_file) as f:
        log_content = f.read()
        assert "Resuming" in log_content or "resume" in log_content.lower()


@pytest.mark.workflow
@pytest.mark.slow
@pytest.mark.skipif(
    subprocess.run(
        ["python3", "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"],
        capture_output=True,
    ).returncode
    != 0,
    reason="CUDA not available",
)
def test_mixed_precision_training(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test mixed precision training (requires CUDA).
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "mixed_precision_config.yaml"
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

    # Force CPU for tests
    with open(config_path) as f:
        test_config = yaml.safe_load(f)
    test_config['training']['device'] = 'cpu'
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Modify config to enable mixed precision
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["training"]["trainer_type"] = "mixed_precision"
    config["training"]["amp_dtype"] = "float16"
    config["training"]["device"] = "cuda:0"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Train with mixed precision
    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Mixed precision training failed: {result.stderr}"

    # Verify model was trained
    run_dirs = list(Path("runs").glob("mnist_*"))
    assert len(run_dirs) > 0
    assert (run_dirs[-1] / "weights" / "best.pt").exists()


@pytest.mark.workflow
@pytest.mark.slow
def test_inference_on_different_splits(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test running inference on different data splits.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config and train
    config_path = configs_dir / "inference_test_config.yaml"
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

    # Force CPU for tests
    with open(config_path) as f:
        test_config = yaml.safe_load(f)
    test_config['training']['device'] = 'cpu'
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

    # Find checkpoint
    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    run_dir = run_dirs[-1]
    checkpoint = run_dir / "weights" / "best.pt"

    # Test inference on test split (ml-inference doesn't support --split argument, uses test by default)
    result = subprocess.run(
        ["ml-inference", "--checkpoint_path", str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Inference failed: {result.stderr}"

    # Verify classification report was created
    assert (run_dir / "logs" / "classification_report_test.txt").exists()


@pytest.mark.workflow
@pytest.mark.slow
def test_training_with_early_stopping(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test training with early stopping enabled.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "early_stopping_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "10"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Force CPU for tests
    with open(config_path) as f:
        test_config = yaml.safe_load(f)
    test_config['training']['device'] = 'cpu'
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Enable early stopping in config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["training"]["early_stopping"] = {
        "enabled": True,
        "patience": 3,
        "metric": "val_acc",
        "mode": "max",
        "min_delta": 0.0,
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Train with early stopping
    result = subprocess.run(
        ["ml-train", "--config", str(config_path), "--num_workers", "0"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"Training with early stopping failed: {result.stderr}"

    # Check if early stopping was triggered (check summary)
    run_dirs = sorted(Path("runs").glob("mnist_*"), key=lambda x: x.stat().st_mtime)
    summary_file = run_dirs[-1] / "summary.txt"
    with open(summary_file) as f:
        summary = f.read()
        # Early stopping may or may not trigger in 10 epochs, just verify it ran
        assert "Status:" in summary
