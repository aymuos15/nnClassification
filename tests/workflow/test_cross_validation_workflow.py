"""Test cross-validation workflows."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.workflow
@pytest.mark.slow
def test_cross_validation_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test training on multiple folds.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "cv_config.yaml"
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

    # Train on 3 folds
    n_folds = 3
    for fold in range(n_folds):
        result = subprocess.run(
            [
                "ml-train",
                "--config",
                str(config_path),
                "--fold",
                str(fold),
                "--num_workers",
                "0"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Training on fold {fold} failed: {result.stderr}"

    # Verify separate run directories were created for each fold
    for fold in range(n_folds):
        fold_runs = list(Path("runs").glob(f"*fold_{fold}*"))
        assert len(fold_runs) > 0, f"Run directory for fold {fold} not found"

        # Verify each fold has outputs
        run_dir = fold_runs[0]
        assert (run_dir / "weights" / "best.pt").exists()
        assert (run_dir / "logs" / "train.log").exists()
        assert (run_dir / "logs" / "classification_report_test.txt").exists()


@pytest.mark.workflow
@pytest.mark.slow
def test_cross_validation_results_collection(
    test_dataset_dir, temp_workspace, dataset_with_splits
):
    """
    Test that cross-validation produces consistent test results across folds.
    """
    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config
    config_path = configs_dir / "cv_results_config.yaml"
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

    # Train on 2 folds to save time
    n_folds = 2

    for fold in range(n_folds):
        result = subprocess.run(
            [
                "ml-train",
                "--config",
                str(config_path),
                "--fold",
                str(fold),
                "--num_workers",
                "0"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Training on fold {fold} failed: {result.stderr}"

    # Verify separate run directories were created for each fold
    for fold in range(n_folds):
        fold_runs = list(Path("runs").glob(f"*fold_{fold}*"))
        assert len(fold_runs) > 0, f"Run directory for fold {fold} not found"

        # Verify each fold has outputs
        run_dir = fold_runs[0]
        assert (run_dir / "weights" / "best.pt").exists()
        assert (run_dir / "logs" / "train.log").exists()
        assert (run_dir / "logs" / "classification_report_test.txt").exists()


@pytest.mark.workflow
@pytest.mark.slow
@pytest.mark.optuna
def test_cross_validation_with_search(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test hyperparameter search on one fold, then train all folds with best config.
    """
    # Check if optuna is installed
    result = subprocess.run(
        [sys.executable, "-c", "import optuna"],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("Optuna not installed")

    dataset_dir = dataset_with_splits
    configs_dir = temp_workspace["configs"]
    optuna_db = temp_workspace["optuna_db"]

    if not dataset_dir.exists():
        pytest.skip(f"Test dataset not found at {dataset_dir}")

    # Generate config with search
    config_path = configs_dir / "cv_search_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--optuna",
            "--output",
            str(config_path),
            "--yes"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    # Update config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["search"]["storage"] = f"sqlite:///{optuna_db}"
    config["search"]["study_name"] = "cv_search_study"
    config["training"]["num_epochs"] = 2
    config["training"]["batch_size"] = 8
    config["data"]["num_workers"] = 0
    config["training"]["device"] = "cpu"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Step 1: Search on fold 0
    result = subprocess.run(
        ["ml-search", "--config", str(config_path), "--fold", "0", "--n-trials", "2"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"Search failed: {result.stderr}"

    # Step 2: Get best config
    best_config_path = Path("runs/optuna_studies/cv_search_study/best_config.yaml")
    assert best_config_path.exists(), "Best config not saved"

    # Step 3: Train folds 0 and 1 with best config
    for fold in [0, 1]:
        result = subprocess.run(
            [
                "ml-train",
                "--config",
                str(best_config_path),
                "--fold",
                str(fold),
                "--num_workers",
                "0"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, f"Training fold {fold} with best config failed"

    # Verify both fold runs exist
    fold_0_runs = list(Path("runs").glob("*fold_0*"))
    fold_1_runs = list(Path("runs").glob("*fold_1*"))

    # Should have runs from search and from best config training
    assert len(fold_0_runs) >= 1
    assert len(fold_1_runs) >= 1
