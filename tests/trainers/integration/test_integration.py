"""Comprehensive integration tests for all trainer types.

This module tests trainer switching, checkpoint compatibility, CLI integration,
and other cross-trainer scenarios.
"""

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml_src.core.checkpointing import load_checkpoint, save_checkpoint
from ml_src.core.loss import get_criterion
from ml_src.core.trainers import get_trainer
from ml_src.core.trainers.base import BaseTrainer
from ml_src.core.trainers.mixed_precision import MixedPrecisionTrainer
from ml_src.core.trainers.standard import StandardTrainer

# Check if optional packages are available
try:
    import accelerate  # noqa

    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

try:
    import opacus  # noqa

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

# Conditional imports for optional trainers (only if packages are available)
if ACCELERATE_AVAILABLE:
    from ml_src.core.trainers.accelerate import AccelerateTrainer

if OPACUS_AVAILABLE:
    from ml_src.core.trainers.differential_privacy import DPTrainer


def create_minimal_setup(device="cpu", num_samples=8):
    """
    Create minimal model, data, optimizer, scheduler for testing.

    Args:
        device: Device string ('cpu' or 'cuda')
        num_samples: Number of samples in dataset

    Returns:
        Tuple of (model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config)
    """
    # Create minimal model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Move to device if needed
    if device != "cpu":
        model = model.to(device)

    # Create balanced dataset (ensure both classes represented)
    X_train = torch.randn(num_samples, 10)
    y_train = torch.tensor([0, 1] * (num_samples // 2))
    X_val = torch.randn(num_samples, 10)
    y_val = torch.tensor([0, 1] * (num_samples // 2))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": num_samples, "val": num_samples}

    # Basic config
    config = {
        "training": {"num_epochs": 1, "batch_size": 2, "device": device},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 1, "gamma": 0.1},
    }

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["optimizer"]["lr"], momentum=config["optimizer"]["momentum"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["scheduler"]["step_size"], gamma=config["scheduler"]["gamma"]
    )

    # Create criterion
    criterion = get_criterion()

    return model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config


def test_trainer_switching_via_config():
    """Test switching between all trainer types via config."""
    trainer_types = ["standard", "mixed_precision"]

    # Add optional trainers if available
    if ACCELERATE_AVAILABLE:
        trainer_types.append("accelerate")
    if OPACUS_AVAILABLE:
        trainer_types.append("dp")

    results = {}

    for trainer_type in trainer_types:
        # Create fresh setup for each trainer
        model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config = (
            create_minimal_setup()
        )

        # Update config for this trainer type
        config["training"]["trainer_type"] = trainer_type

        # Add DP-specific config if needed
        if trainer_type == "dp":
            config["training"]["epsilon"] = 10.0
            config["training"]["delta"] = 1e-5
            config["training"]["max_grad_norm"] = 1.0

        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

            # Create trainer via factory
            trainer = get_trainer(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                device=torch.device("cpu"),
                run_dir=run_dir,
                class_names=["class_0", "class_1"],
            )

            # Verify correct type
            assert isinstance(trainer, BaseTrainer), f"{trainer_type} is not a BaseTrainer"

            # Verify specific type
            if trainer_type == "standard":
                assert isinstance(trainer, StandardTrainer)
            elif trainer_type == "mixed_precision":
                assert isinstance(trainer, MixedPrecisionTrainer)
            elif trainer_type == "accelerate" and ACCELERATE_AVAILABLE:
                assert isinstance(trainer, AccelerateTrainer)
            elif trainer_type == "dp" and OPACUS_AVAILABLE:
                assert isinstance(trainer, DPTrainer)

            # Run 1 epoch
            trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()

            # Verify training completed
            assert trained_model is not None
            assert len(train_losses) == 1
            assert len(val_losses) == 1
            assert len(train_accs) == 1
            assert len(val_accs) == 1

            # Store results
            results[trainer_type] = {
                "trainer": trainer,
                "train_loss": train_losses[0],
                "val_loss": val_losses[0],
            }

    # Verify we got different trainer instances
    trainer_instances = [results[t]["trainer"] for t in trainer_types]
    for i in range(len(trainer_instances)):
        for j in range(i + 1, len(trainer_instances)):
            # Different instances
            assert trainer_instances[i] is not trainer_instances[j]
            # Different types (unless both are same type by coincidence)
            assert type(trainer_instances[i]) != type(trainer_instances[j])


def test_invalid_trainer_type():
    """Test comprehensive invalid trainer type handling."""
    invalid_types = [
        ("invalid", "Unsupported trainer_type: 'invalid'"),
        ("STANDARD", "Unsupported trainer_type: 'STANDARD'"),  # case-sensitive
        ("", "Unsupported trainer_type: ''"),
        ("random_string", "Unsupported trainer_type: 'random_string'"),
    ]

    for invalid_type, expected_msg in invalid_types:
        model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config = (
            create_minimal_setup()
        )

        config["training"]["trainer_type"] = invalid_type

        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

            with pytest.raises(ValueError, match="Unsupported trainer_type"):
                get_trainer(
                    config=config,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    dataloaders=dataloaders,
                    dataset_sizes=dataset_sizes,
                    device=torch.device("cpu"),
                    run_dir=run_dir,
                    class_names=["class_0", "class_1"],
                )


def test_checkpoint_compatibility_standard_mixed_precision():
    """Test checkpoint compatibility between StandardTrainer and MixedPrecisionTrainer on CPU."""
    # Create setup
    model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config = (
        create_minimal_setup()
    )

    config["training"]["trainer_type"] = "standard"

    with tempfile.TemporaryDirectory() as run_dir:
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        checkpoint_path = os.path.join(run_dir, "weights", "test_checkpoint.pt")

        # Phase 1: Train 2 epochs with StandardTrainer
        config["training"]["num_epochs"] = 2
        trainer1 = StandardTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=torch.device("cpu"),
            config=config,
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        trained_model, train_losses, val_losses, train_accs, val_accs = trainer1.train()

        assert len(train_losses) == 2  # 2 epochs

        # Save checkpoint manually
        save_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=2,
            best_acc=max(val_accs),
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            config=config,
            checkpoint_path=checkpoint_path,
        )

        # Verify checkpoint exists
        assert os.path.exists(checkpoint_path)

        # Phase 2: Load checkpoint with MixedPrecisionTrainer (on CPU, so no AMP)
        # Create fresh model, optimizer, scheduler
        model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)

        # Load checkpoint
        (
            epoch,
            best_acc,
            loaded_train_losses,
            loaded_val_losses,
            loaded_train_accs,
            loaded_val_accs,
            loaded_config,
            early_stopping_state,
            ema_state,
        ) = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device=torch.device("cpu"),
        )

        # Verify loaded values
        assert epoch == 2
        assert best_acc == max(val_accs)
        assert len(loaded_train_losses) == 2
        assert len(loaded_val_losses) == 2
        assert ema_state is None

        # Phase 3: Continue training with MixedPrecisionTrainer for 1 more epoch
        config["training"]["trainer_type"] = "mixed_precision"
        config["training"]["num_epochs"] = 1  # Just 1 more epoch

        trainer2 = MixedPrecisionTrainer(
            model=model2,
            criterion=criterion,
            optimizer=optimizer2,
            scheduler=scheduler2,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=torch.device("cpu"),  # CPU - will fall back to standard training
            config=config,
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        # Train 1 more epoch
        trained_model2, new_train_losses, new_val_losses, new_train_accs, new_val_accs = (
            trainer2.train()
        )

        # Verify training continued successfully
        assert trained_model2 is not None
        assert len(new_train_losses) == 1  # 1 new epoch
        assert len(new_val_losses) == 1
        assert len(new_train_accs) == 1
        assert len(new_val_accs) == 1


@pytest.mark.slow
def test_cli_integration_all_trainers():
    """End-to-end CLI test for all trainer types.

    Note: Skipped due to device mismatch in test evaluation phase.
    The trainers themselves are thoroughly tested in other integration tests.
    """
    trainer_types = ["standard", "mixed_precision"]

    # Add optional trainers if available
    if ACCELERATE_AVAILABLE:
        trainer_types.append("accelerate")
    if OPACUS_AVAILABLE:
        trainer_types.append("dp")

    for trainer_type in trainer_types:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_path = os.path.join(temp_dir, "test_config.yaml")
            
            # DP trainer requires models without BatchNorm (use custom model)
            # Other trainers can use standard resnet18
            if trainer_type == "dp":
                model_config = """
model:
  type: "custom"
  custom_architecture: "simple_cnn"
  num_classes: 2
  input_size: 224
  dropout: 0.5
"""
            else:
                model_config = """
model:
  type: "base"
  architecture: "resnet18"
  num_classes: 2
  weights: null
"""
            
            config_content = f"""
dataset_name: "test_cli"

data:
  data_dir: "{temp_dir}/data"
  fold: 0
  num_workers: 0

{model_config}
training:
  trainer_type: "{trainer_type}"
  num_epochs: 1
  batch_size: 2
  device: "cpu"
"""
            # Add DP-specific config
            if trainer_type == "dp":
                config_content += """
  dp:
    noise_multiplier: 1.1
    max_grad_norm: 1.0
    target_epsilon: 10.0
    target_delta: 0.00001
"""

            config_content += """
optimizer:
  lr: 0.001
  momentum: 0.9

scheduler:
  step_size: 1
  gamma: 0.1

transforms:
  train:
    resize: [224, 224]
    random_horizontal_flip: true
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  test:
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

seed: 42
deterministic: false
"""

            # Write config
            with open(config_path, "w") as f:
                f.write(config_content)

            # Create minimal dataset structure
            data_dir = os.path.join(temp_dir, "data")
            raw_dir = os.path.join(data_dir, "raw")
            splits_dir = os.path.join(data_dir, "splits")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(splits_dir, exist_ok=True)

            # Create class directories and dummy images
            for class_name in ["class_0", "class_1"]:
                class_dir = os.path.join(raw_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Create dummy images (1x1 black images)
                for i in range(4):
                    img_path = os.path.join(class_dir, f"img_{i}.jpg")
                    # Create a minimal 1x1 black JPEG
                    from PIL import Image

                    img = Image.new("RGB", (1, 1), color="black")
                    img.save(img_path)

            # Create split files
            train_split = os.path.join(splits_dir, "fold_0_train.txt")
            val_split = os.path.join(splits_dir, "fold_0_val.txt")
            test_split = os.path.join(splits_dir, "test.txt")

            with open(train_split, "w") as f:
                f.write("raw/class_0/img_0.jpg\n")
                f.write("raw/class_1/img_0.jpg\n")
                f.write("raw/class_0/img_1.jpg\n")
                f.write("raw/class_1/img_1.jpg\n")

            with open(val_split, "w") as f:
                f.write("raw/class_0/img_2.jpg\n")
                f.write("raw/class_1/img_2.jpg\n")

            with open(test_split, "w") as f:
                f.write("raw/class_0/img_3.jpg\n")
                f.write("raw/class_1/img_3.jpg\n")

            # Run ml-train command
            cmd = [sys.executable, "-m", "ml_src.cli.train", "--config", config_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            # Check exit code
            if result.returncode != 0:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

            assert result.returncode == 0, f"CLI failed for {trainer_type}: {result.stderr}"

            # Verify outputs exist (run directory should be created)
            runs_dir = os.path.join(os.getcwd(), "runs")
            # Note: We can't easily verify the exact run directory name due to timestamp,
            # but we can verify the command completed successfully via exit code


def test_factory_defaults():
    """Test factory default behavior when trainer_type is omitted or null."""
    test_cases = [
        ({}, "omitted"),  # trainer_type not in config at all
        (
            {"training": {"num_epochs": 1, "batch_size": 2}},
            "omitted from training",
        ),  # training exists but no trainer_type
    ]

    for config_override, description in test_cases:
        model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, base_config = (
            create_minimal_setup()
        )

        # Merge config override properly (deep merge for training)
        if "training" in config_override:
            # Merge training configs
            config = base_config.copy()
            config["training"] = {**base_config.get("training", {}), **config_override["training"]}
        else:
            config = {**base_config, **config_override}

        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

            # Should default to StandardTrainer
            trainer = get_trainer(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                device=torch.device("cpu"),
                run_dir=run_dir,
                class_names=["class_0", "class_1"],
            )

            # Verify it's a StandardTrainer
            assert isinstance(trainer, StandardTrainer), (
                f"Failed for case: {description}, got {type(trainer)}"
            )

            # Verify it works
            trained_model, train_losses, val_losses, train_accs, val_accs = trainer.train()
            assert trained_model is not None
            assert len(train_losses) == 1


@pytest.mark.skipif(not OPACUS_AVAILABLE, reason="Opacus not installed")
def test_opacus_import_check():
    """Test that DPTrainer properly checks for opacus availability."""
    # This test verifies that the import system works correctly
    # If opacus is available, DPTrainer should import successfully

    model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config = (
        create_minimal_setup()
    )

    config["training"]["trainer_type"] = "dp"
    config["training"]["epsilon"] = 10.0
    config["training"]["delta"] = 1e-5
    config["training"]["max_grad_norm"] = 1.0

    with tempfile.TemporaryDirectory() as run_dir:
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

        # Should successfully create DPTrainer if opacus is available
        trainer = get_trainer(
            config=config,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=torch.device("cpu"),
            run_dir=run_dir,
            class_names=["class_0", "class_1"],
        )

        assert isinstance(trainer, DPTrainer)


def test_trainer_independence():
    """Test that trainers don't interfere with each other when created separately."""
    # Create two trainers of different types simultaneously
    model1, dataloaders1, dataset_sizes1, optimizer1, scheduler1, criterion1, config1 = (
        create_minimal_setup()
    )
    model2, dataloaders2, dataset_sizes2, optimizer2, scheduler2, criterion2, config2 = (
        create_minimal_setup()
    )

    config1["training"]["trainer_type"] = "standard"
    config2["training"]["trainer_type"] = "mixed_precision"

    with tempfile.TemporaryDirectory() as run_dir1, tempfile.TemporaryDirectory() as run_dir2:
        os.makedirs(os.path.join(run_dir1, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir1, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir2, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir2, "logs"), exist_ok=True)

        trainer1 = get_trainer(
            config=config1,
            model=model1,
            criterion=criterion1,
            optimizer=optimizer1,
            scheduler=scheduler1,
            dataloaders=dataloaders1,
            dataset_sizes=dataset_sizes1,
            device=torch.device("cpu"),
            run_dir=run_dir1,
            class_names=["class_0", "class_1"],
        )

        trainer2 = get_trainer(
            config=config2,
            model=model2,
            criterion=criterion2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            dataloaders=dataloaders2,
            dataset_sizes=dataset_sizes2,
            device=torch.device("cpu"),
            run_dir=run_dir2,
            class_names=["class_0", "class_1"],
        )

        # Verify they're different instances
        assert trainer1 is not trainer2
        assert type(trainer1) != type(trainer2)

        # Train both
        result1 = trainer1.train()
        result2 = trainer2.train()

        # Verify both completed successfully
        assert result1[0] is not None
        assert result2[0] is not None
        assert len(result1[1]) == 1  # train_losses
        assert len(result2[1]) == 1


def test_config_preservation():
    """Test that config is preserved correctly across trainer types."""
    trainer_types = ["standard", "mixed_precision"]

    if ACCELERATE_AVAILABLE:
        trainer_types.append("accelerate")

    for trainer_type in trainer_types:
        model, dataloaders, dataset_sizes, optimizer, scheduler, criterion, config = (
            create_minimal_setup()
        )

        # Add custom config values
        config["training"]["trainer_type"] = trainer_type
        config["custom_value"] = "test_value_123"
        config["training"]["custom_training_param"] = 42

        with tempfile.TemporaryDirectory() as run_dir:
            os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

            trainer = get_trainer(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                device=torch.device("cpu"),
                run_dir=run_dir,
                class_names=["class_0", "class_1"],
            )

            # Verify config is preserved in trainer
            assert hasattr(trainer, "config")
            assert trainer.config["custom_value"] == "test_value_123"
            assert trainer.config["training"]["custom_training_param"] == 42
            assert trainer.config["training"]["trainer_type"] == trainer_type
