"""Comprehensive tests for configuration system (loading, creation, overrides)."""

import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from ml_src.core.config import create_config, load_config, override_config


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "dataset_name": "test_dataset",
        "data": {
            "dataset_name": "test_dataset",
            "data_dir": "/path/to/data",
            "fold": 0,
            "num_workers": 4,
        },
        "model": {
            "type": "base",
            "architecture": "resnet18",
            "num_classes": 10,
            "weights": None,
        },
        "training": {
            "trainer_type": "standard",
            "num_epochs": 25,
            "batch_size": 32,
            "device": "cuda",
        },
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 7, "gamma": 0.1},
        "transforms": {
            "train": {"resize": [224, 224], "random_horizontal_flip": True},
            "val": {"resize": [224, 224]},
            "test": {"resize": [224, 224]},
        },
    }


@pytest.fixture
def sample_config_file(sample_config_dict):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def config_template_file():
    """Create a minimal config template."""
    template = {
        "data": {"dataset_name": "placeholder", "data_dir": "placeholder", "fold": 0},
        "model": {"type": "base", "architecture": "resnet18", "num_classes": 10},
        "training": {"num_epochs": 25, "batch_size": 4},
        "optimizer": {"lr": 0.001, "momentum": 0.9},
        "scheduler": {"step_size": 7, "gamma": 0.1},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(template, f)
        template_path = f.name

    yield template_path

    # Cleanup
    Path(template_path).unlink(missing_ok=True)


@pytest.fixture
def dataset_info():
    """Sample dataset information."""
    return {
        "dataset_name": "my_dataset",
        "data_dir": "/path/to/my_dataset",
        "num_classes": 5,
        "class_names": ["class_0", "class_1", "class_2", "class_3", "class_4"],
    }


# ============================================================================
# load_config Tests
# ============================================================================


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_returns_dict(self, sample_config_file):
        """Test that load_config returns a dictionary."""
        config = load_config(sample_config_file)

        assert isinstance(config, dict)

    def test_load_config_preserves_structure(self, sample_config_file, sample_config_dict):
        """Test that loaded config matches original structure."""
        config = load_config(sample_config_file)

        # Check main keys exist
        assert "data" in config
        assert "model" in config
        assert "training" in config
        assert "optimizer" in config

    def test_load_config_preserves_values(self, sample_config_file):
        """Test that values are loaded correctly."""
        config = load_config(sample_config_file)

        assert config["model"]["architecture"] == "resnet18"
        assert config["training"]["num_epochs"] == 25
        assert config["optimizer"]["lr"] == 0.001

    def test_load_config_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_nested_values(self, sample_config_file):
        """Test that nested values are loaded correctly."""
        config = load_config(sample_config_file)

        assert config["data"]["num_workers"] == 4
        assert config["scheduler"]["gamma"] == 0.1

    def test_load_config_with_lists(self, sample_config_file):
        """Test that lists are loaded correctly."""
        config = load_config(sample_config_file)

        assert config["transforms"]["train"]["resize"] == [224, 224]

    def test_load_config_with_null_values(self, sample_config_file):
        """Test that None/null values are handled."""
        config = load_config(sample_config_file)

        assert config["model"]["weights"] is None

    def test_load_config_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            empty_file = f.name

        try:
            config = load_config(empty_file)
            # Empty YAML file returns None
            assert config is None
        finally:
            Path(empty_file).unlink(missing_ok=True)

    def test_load_config_invalid_yaml(self):
        """Test that invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            invalid_file = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(invalid_file)
        finally:
            Path(invalid_file).unlink(missing_ok=True)


# ============================================================================
# create_config Tests
# ============================================================================


class TestCreateConfig:
    """Tests for create_config function."""

    def test_create_config_returns_dict(self, dataset_info, config_template_file):
        """Test that create_config returns a dictionary."""
        config = create_config(dataset_info, config_template_file)

        assert isinstance(config, dict)

    def test_create_config_updates_dataset_name(self, dataset_info, config_template_file):
        """Test that dataset name is updated."""
        config = create_config(dataset_info, config_template_file)

        assert config["data"]["dataset_name"] == "my_dataset"

    def test_create_config_updates_data_dir(self, dataset_info, config_template_file):
        """Test that data directory is updated."""
        config = create_config(dataset_info, config_template_file)

        assert config["data"]["data_dir"] == "/path/to/my_dataset"

    def test_create_config_updates_num_classes(self, dataset_info, config_template_file):
        """Test that number of classes is updated."""
        config = create_config(dataset_info, config_template_file)

        assert config["model"]["num_classes"] == 5

    def test_create_config_default_fold(self, dataset_info, config_template_file):
        """Test that fold defaults to 0."""
        config = create_config(dataset_info, config_template_file)

        assert config["data"]["fold"] == 0

    def test_create_config_custom_architecture(self, dataset_info, config_template_file):
        """Test setting custom architecture."""
        config = create_config(dataset_info, config_template_file, architecture="resnet50")

        assert config["model"]["architecture"] == "resnet50"

    def test_create_config_custom_batch_size(self, dataset_info, config_template_file):
        """Test setting custom batch size."""
        config = create_config(dataset_info, config_template_file, batch_size=64)

        assert config["training"]["batch_size"] == 64

    def test_create_config_custom_num_epochs(self, dataset_info, config_template_file):
        """Test setting custom number of epochs."""
        config = create_config(dataset_info, config_template_file, num_epochs=50)

        assert config["training"]["num_epochs"] == 50

    def test_create_config_custom_lr(self, dataset_info, config_template_file):
        """Test setting custom learning rate."""
        config = create_config(dataset_info, config_template_file, lr=0.01)

        assert config["optimizer"]["lr"] == 0.01

    def test_create_config_preserves_template_values(self, dataset_info, config_template_file):
        """Test that non-updated template values are preserved."""
        config = create_config(dataset_info, config_template_file)

        # These should remain from template
        assert config["optimizer"]["momentum"] == 0.9
        assert config["scheduler"]["step_size"] == 7

    def test_create_config_all_custom_params(self, dataset_info, config_template_file):
        """Test setting all custom parameters."""
        config = create_config(
            dataset_info,
            config_template_file,
            architecture="efficientnet_b0",
            batch_size=128,
            num_epochs=100,
            lr=0.0001,
            num_folds=10,
        )

        assert config["model"]["architecture"] == "efficientnet_b0"
        assert config["training"]["batch_size"] == 128
        assert config["training"]["num_epochs"] == 100
        assert config["optimizer"]["lr"] == 0.0001


# ============================================================================
# override_config Tests
# ============================================================================


class TestOverrideConfig:
    """Tests for override_config function."""

    def test_override_config_returns_tuple(self, sample_config_dict):
        """Test that override_config returns (config, overrides) tuple."""
        args = Namespace()
        result = override_config(sample_config_dict, args)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_override_config_no_args(self, sample_config_dict):
        """Test override with no arguments returns unchanged config."""
        args = Namespace()
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config == sample_config_dict
        assert overrides == []

    def test_override_batch_size(self, sample_config_dict):
        """Test overriding batch size."""
        args = Namespace(batch_size=64)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["training"]["batch_size"] == 64
        assert "batch_64" in overrides

    def test_override_num_epochs(self, sample_config_dict):
        """Test overriding number of epochs."""
        args = Namespace(num_epochs=100)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["training"]["num_epochs"] == 100
        assert "epochs_100" in overrides

    def test_override_learning_rate(self, sample_config_dict):
        """Test overriding learning rate."""
        args = Namespace(lr=0.01)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["optimizer"]["lr"] == 0.01
        assert "lr_0.01" in overrides

    def test_override_fold(self, sample_config_dict):
        """Test overriding fold."""
        args = Namespace(fold=3)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["data"]["fold"] == 3
        assert "fold_3" in overrides

    def test_override_momentum(self, sample_config_dict):
        """Test overriding momentum."""
        args = Namespace(momentum=0.95)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["optimizer"]["momentum"] == 0.95
        # Momentum doesn't add to overrides list
        assert "momentum" not in str(overrides)

    def test_override_num_workers(self, sample_config_dict):
        """Test overriding num_workers."""
        args = Namespace(num_workers=8)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["data"]["num_workers"] == 8

    def test_override_dataset_name(self, sample_config_dict):
        """Test overriding dataset name."""
        args = Namespace(dataset_name="new_dataset")
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["data"]["dataset_name"] == "new_dataset"

    def test_override_data_dir(self, sample_config_dict):
        """Test overriding data directory."""
        args = Namespace(data_dir="/new/path")
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["data"]["data_dir"] == "/new/path"

    def test_override_multiple_params(self, sample_config_dict):
        """Test overriding multiple parameters."""
        args = Namespace(batch_size=128, lr=0.0001, num_epochs=50, fold=2)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["training"]["batch_size"] == 128
        assert config["optimizer"]["lr"] == 0.0001
        assert config["training"]["num_epochs"] == 50
        assert config["data"]["fold"] == 2

        # Check all overrides are in the list
        assert "batch_128" in overrides
        assert "lr_0.0001" in overrides
        assert "epochs_50" in overrides
        assert "fold_2" in overrides

    def test_override_early_stopping_patience(self, sample_config_dict):
        """Test overriding early stopping patience."""
        args = Namespace(early_stopping_patience=10)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert "early_stopping" in config["training"]
        assert config["training"]["early_stopping"]["enabled"] is True
        assert config["training"]["early_stopping"]["patience"] == 10
        assert "es_patience_10" in overrides

    def test_override_early_stopping_metric_val_acc(self, sample_config_dict):
        """Test overriding early stopping metric to val_acc."""
        args = Namespace(early_stopping_metric="val_acc")
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["training"]["early_stopping"]["enabled"] is True
        assert config["training"]["early_stopping"]["metric"] == "val_acc"
        assert config["training"]["early_stopping"]["mode"] == "max"

    def test_override_early_stopping_metric_val_loss(self, sample_config_dict):
        """Test overriding early stopping metric to val_loss."""
        args = Namespace(early_stopping_metric="val_loss")
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert config["training"]["early_stopping"]["enabled"] is True
        assert config["training"]["early_stopping"]["metric"] == "val_loss"
        assert config["training"]["early_stopping"]["mode"] == "min"

    def test_override_creates_early_stopping_dict_if_missing(self, sample_config_dict):
        """Test that early stopping dict is created if it doesn't exist."""
        # Remove early_stopping if it exists
        if "early_stopping" in sample_config_dict.get("training", {}):
            del sample_config_dict["training"]["early_stopping"]

        args = Namespace(early_stopping_patience=5)
        config, overrides = override_config(sample_config_dict.copy(), args)

        assert "early_stopping" in config["training"]
        assert isinstance(config["training"]["early_stopping"], dict)

    def test_override_preserves_unmodified_values(self, sample_config_dict):
        """Test that non-overridden values are preserved."""
        args = Namespace(batch_size=16)
        config, overrides = override_config(sample_config_dict.copy(), args)

        # These should remain unchanged
        assert config["optimizer"]["lr"] == 0.001
        assert config["model"]["architecture"] == "resnet18"
        assert config["training"]["num_epochs"] == 25


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigIntegration:
    """Integration tests for full config workflow."""

    def test_create_save_load_config(self, dataset_info, config_template_file):
        """Test creating, saving, and loading a config."""
        # Create config
        config = create_config(dataset_info, config_template_file, batch_size=64)

        # Save to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Load back
            loaded_config = load_config(config_path)

            # Verify values
            assert loaded_config["data"]["dataset_name"] == "my_dataset"
            assert loaded_config["model"]["num_classes"] == 5
            assert loaded_config["training"]["batch_size"] == 64
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_load_override_config_workflow(self, sample_config_file):
        """Test loading config then applying overrides."""
        # Load config
        config = load_config(sample_config_file)

        # Apply overrides
        args = Namespace(batch_size=128, lr=0.0001, fold=2)
        overridden_config, overrides = override_config(config, args)

        # Verify overrides applied
        assert overridden_config["training"]["batch_size"] == 128
        assert overridden_config["optimizer"]["lr"] == 0.0001
        assert overridden_config["data"]["fold"] == 2

        # Verify override list for run naming
        assert len(overrides) == 3
        assert "batch_128" in overrides
        assert "lr_0.0001" in overrides
        assert "fold_2" in overrides

    def test_config_roundtrip_preserves_types(self, sample_config_dict):
        """Test that config roundtrip (save/load) preserves data types."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_dict, f)
            config_path = f.name

        try:
            loaded = load_config(config_path)

            # Check types are preserved
            assert isinstance(loaded["training"]["num_epochs"], int)
            assert isinstance(loaded["optimizer"]["lr"], float)
            assert isinstance(loaded["transforms"]["train"]["random_horizontal_flip"], bool)
            assert loaded["model"]["weights"] is None
        finally:
            Path(config_path).unlink(missing_ok=True)
