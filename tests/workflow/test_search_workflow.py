"""Test hyperparameter search workflows."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.workflow
@pytest.mark.slow
@pytest.mark.optuna
def test_hyperparameter_search_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test complete hyperparameter search workflow:
    1. Generate config with search space
    2. Run search with minimal trials
    3. Verify best config is saved
    4. Train with best config
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

    # Step 1: Generate config with search space
    config_path = configs_dir / "search_test_config.yaml"
    result = subprocess.run(
        [
            "ml-init-config",
            "--data_dir",
            str(dataset_dir),
            "--optuna",
            "--output",
            str(config_path),
            "--yes",
            "--num_epochs",
            "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Config generation failed: {result.stderr}"
    assert config_path.exists()

    # Load and update config for faster testing
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert "search" in config, "Search section not in config"

    config["search"]["n_trials"] = 2
    config["search"]["storage"] = f"sqlite:///{optuna_db}"
    config["search"]["study_name"] = "test_search_workflow"
    config["training"]["num_epochs"] = 2
    config["training"]["batch_size"] = 8
    config["data"]["num_workers"] = 0
    config["training"]["device"] = "cpu"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Step 2: Run hyperparameter search
    result = subprocess.run(
        ["ml-search", "--config", str(config_path), "--n-trials", "2"],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes for 2 trials
    )
    assert result.returncode == 0, f"Hyperparameter search failed: {result.stderr}"

    # Step 3: Verify outputs
    study_dir = Path("runs/optuna_studies/test_search_workflow")
    assert study_dir.exists(), "Study directory not created"
    assert (study_dir / "best_config.yaml").exists(), "Best config not saved"
    assert (study_dir / "trial_0").exists(), "Trial 0 directory not created"
    assert (study_dir / "trial_1").exists(), "Trial 1 directory not created"

    # Verify trial outputs
    for trial_num in [0, 1]:
        trial_dir = study_dir / f"trial_{trial_num}"
        assert (trial_dir / "weights" / "best.pt").exists()
        assert (trial_dir / "logs" / "train.log").exists()

    # Step 4: Train with best config
    best_config = study_dir / "best_config.yaml"
    result = subprocess.run(
        ["ml-train", "--config", str(best_config)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Training with best config failed: {result.stderr}"


@pytest.mark.workflow
@pytest.mark.slow
@pytest.mark.optuna
def test_search_resume_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test resuming a hyperparameter search study.
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

    # Generate config
    config_path = configs_dir / "resume_search_config.yaml"
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
    config["search"]["study_name"] = "test_resume_study"
    config["training"]["num_epochs"] = 2
    config["training"]["batch_size"] = 8
    config["data"]["num_workers"] = 0
    config["training"]["device"] = "cpu"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run first search with 1 trial
    result = subprocess.run(
        ["ml-search", "--config", str(config_path), "--n-trials", "1"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0

    # Resume and run 1 more trial
    result = subprocess.run(
        ["ml-search", "--config", str(config_path), "--resume", "--n-trials", "1"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"Resume search failed: {result.stderr}"

    # Verify both trials exist
    study_dir = Path("runs/optuna_studies/test_resume_study")
    assert (study_dir / "trial_0").exists()
    assert (study_dir / "trial_1").exists()


@pytest.mark.workflow
@pytest.mark.slow
@pytest.mark.optuna
def test_search_visualization_workflow(test_dataset_dir, temp_workspace, dataset_with_splits):
    """
    Test hyperparameter search visualization generation.
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

    # Generate config and run small search
    config_path = configs_dir / "viz_search_config.yaml"
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
    config["search"]["study_name"] = "test_viz_study"
    config["training"]["num_epochs"] = 2
    config["training"]["batch_size"] = 8
    config["data"]["num_workers"] = 0
    config["training"]["device"] = "cpu"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run search
    result = subprocess.run(
        ["ml-search", "--config", str(config_path), "--n-trials", "2"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0

    # Generate visualizations
    result = subprocess.run(
        [
            "ml-visualise",
            "--mode",
            "search",
            "--study-name",
            "test_viz_study",
            "--storage",
            f"sqlite:///{optuna_db}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Visualization generation failed: {result.stderr}"

    # Verify plots were created
    viz_dir = Path("runs/optuna_studies/test_viz_study/visualizations")
    assert viz_dir.exists(), "Visualization directory not created"
    assert (viz_dir / "optimization_history.html").exists()
    assert (viz_dir / "param_importances.html").exists()
    assert (viz_dir / "parallel_coordinate.html").exists()
