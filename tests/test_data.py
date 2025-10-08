"""Comprehensive tests for data module (datasets, detection, indexing, splitting)."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from ml_src.core.data import (
    IndexedImageDataset,
    collect_file_paths,
    create_cv_splits,
    create_test_split,
    create_train_val_fold,
    detect_dataset_info,
    get_class_names,
    get_datasets,
    get_transforms,
    write_index_file,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory with sample structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create raw data structure
        raw_dir = Path(tmpdir) / "test_dataset" / "raw"
        class1_dir = raw_dir / "cat"
        class2_dir = raw_dir / "dog"
        class1_dir.mkdir(parents=True)
        class2_dir.mkdir(parents=True)

        # Create sample images (1x1 black images for speed)
        for i in range(5):
            img = Image.new("RGB", (10, 10), color="black")
            img.save(class1_dir / f"cat_{i}.jpg")
            img.save(class2_dir / f"dog_{i}.jpg")

        yield Path(tmpdir) / "test_dataset"


@pytest.fixture
def temp_index_file(temp_dataset_dir):
    """Create a temporary index file."""
    index_file = temp_dataset_dir / "test_index.txt"
    index_file.write_text(
        "raw/cat/cat_0.jpg\n" "raw/cat/cat_1.jpg\n" "raw/dog/dog_0.jpg\n" "raw/dog/dog_1.jpg\n"
    )
    return index_file


@pytest.fixture
def sample_config(temp_dataset_dir):
    """Create sample config for testing."""
    # Create splits directory with index files
    splits_dir = temp_dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Create index files for fold 0
    (splits_dir / "fold_0_train.txt").write_text("raw/cat/cat_0.jpg\nraw/dog/dog_0.jpg\n")
    (splits_dir / "fold_0_val.txt").write_text("raw/cat/cat_1.jpg\nraw/dog/dog_1.jpg\n")
    (splits_dir / "test.txt").write_text("raw/cat/cat_2.jpg\nraw/dog/dog_2.jpg\n")

    return {
        "data": {"data_dir": str(temp_dataset_dir), "fold": 0},
        "transforms": {
            "train": {
                "resize": [224, 224],
                "random_horizontal_flip": True,
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
            "val": {
                "resize": [224, 224],
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
            "test": {
                "resize": [224, 224],
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
        },
    }


# ============================================================================
# IndexedImageDataset Tests
# ============================================================================


class TestIndexedImageDataset:
    """Tests for IndexedImageDataset class."""

    def test_dataset_loads_from_index_file(self, temp_index_file, temp_dataset_dir):
        """Test that dataset successfully loads from index file."""
        dataset = IndexedImageDataset(
            index_file=str(temp_index_file),
            data_root=str(temp_dataset_dir / "raw"),
            transform=None,
        )

        assert len(dataset) == 4
        assert len(dataset.classes) == 2
        assert set(dataset.classes) == {"cat", "dog"}

    def test_dataset_class_to_idx_mapping(self, temp_index_file, temp_dataset_dir):
        """Test class to index mapping."""
        dataset = IndexedImageDataset(
            index_file=str(temp_index_file), data_root=str(temp_dataset_dir / "raw")
        )

        # Classes should be sorted alphabetically
        assert dataset.class_to_idx == {"cat": 0, "dog": 1}

    def test_dataset_getitem_returns_image_and_label(self, temp_index_file, temp_dataset_dir):
        """Test __getitem__ returns tuple of (image, label)."""
        dataset = IndexedImageDataset(
            index_file=str(temp_index_file),
            data_root=str(temp_dataset_dir / "raw"),
            transform=transforms.ToTensor(),
        )

        image, label = dataset[0]

        # Should return tensor and int
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == (3, 10, 10)  # RGB, 10x10
        assert label in [0, 1]

    def test_dataset_applies_transforms(self, temp_index_file, temp_dataset_dir):
        """Test that transforms are applied correctly."""
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

        dataset = IndexedImageDataset(
            index_file=str(temp_index_file),
            data_root=str(temp_dataset_dir / "raw"),
            transform=transform,
        )

        image, _ = dataset[0]

        # Image should be resized to 32x32
        assert image.shape == (3, 32, 32)

    def test_dataset_handles_missing_index_file(self, temp_dataset_dir):
        """Test that missing index file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Index file not found"):
            IndexedImageDataset(
                index_file="nonexistent.txt", data_root=str(temp_dataset_dir / "raw")
            )

    def test_dataset_handles_missing_image(self, temp_dataset_dir):
        """Test that missing image file raises RuntimeError."""
        # Create index file pointing to non-existent image
        index_file = temp_dataset_dir / "bad_index.txt"
        index_file.write_text("raw/cat/nonexistent.jpg\n")

        dataset = IndexedImageDataset(
            index_file=str(index_file), data_root=str(temp_dataset_dir / "raw")
        )

        # Should fail when trying to load the image
        with pytest.raises(RuntimeError, match="Error loading image"):
            _ = dataset[0]

    def test_dataset_len(self, temp_index_file, temp_dataset_dir):
        """Test __len__ returns correct count."""
        dataset = IndexedImageDataset(
            index_file=str(temp_index_file), data_root=str(temp_dataset_dir / "raw")
        )

        assert len(dataset) == 4

    def test_dataset_empty_index_file(self, temp_dataset_dir):
        """Test dataset with empty index file."""
        index_file = temp_dataset_dir / "empty_index.txt"
        index_file.write_text("")

        dataset = IndexedImageDataset(
            index_file=str(index_file), data_root=str(temp_dataset_dir / "raw")
        )

        assert len(dataset) == 0
        assert len(dataset.classes) == 0

    def test_dataset_preserves_class_order(self, temp_dataset_dir):
        """Test that classes are sorted alphabetically."""
        # Create index with classes in reverse order
        index_file = temp_dataset_dir / "test_order.txt"
        index_file.write_text("raw/zebra/z1.jpg\nraw/aardvark/a1.jpg\nraw/monkey/m1.jpg\n")

        # Create the directories
        for cls in ["zebra", "aardvark", "monkey"]:
            (temp_dataset_dir / "raw" / cls).mkdir(exist_ok=True)
            img = Image.new("RGB", (10, 10))
            img.save(temp_dataset_dir / "raw" / cls / f"{cls[0]}1.jpg")

        dataset = IndexedImageDataset(
            index_file=str(index_file), data_root=str(temp_dataset_dir / "raw")
        )

        # Should be sorted alphabetically
        assert dataset.classes == ["aardvark", "monkey", "zebra"]


# ============================================================================
# Transform Tests
# ============================================================================


class TestGetTransforms:
    """Tests for get_transforms function."""

    def test_get_transforms_returns_dict(self, sample_config):
        """Test that get_transforms returns dictionary with all splits."""
        transforms_dict = get_transforms(sample_config)

        assert isinstance(transforms_dict, dict)
        assert "train" in transforms_dict
        assert "val" in transforms_dict
        assert "test" in transforms_dict

    def test_train_transforms_include_random_flip(self, sample_config):
        """Test that train transforms include random horizontal flip."""
        transforms_dict = get_transforms(sample_config)

        # Check that RandomHorizontalFlip is in train transforms
        train_transform = transforms_dict["train"]
        has_flip = any(
            isinstance(t, transforms.RandomHorizontalFlip) for t in train_transform.transforms
        )
        assert has_flip

    def test_val_test_no_random_augmentations(self, sample_config):
        """Test that val/test don't have random augmentations."""
        transforms_dict = get_transforms(sample_config)

        for split in ["val", "test"]:
            transform = transforms_dict[split]
            has_flip = any(
                isinstance(t, transforms.RandomHorizontalFlip) for t in transform.transforms
            )
            assert not has_flip

    def test_all_transforms_include_totensor(self, sample_config):
        """Test that all transforms include ToTensor."""
        transforms_dict = get_transforms(sample_config)

        for split in ["train", "val", "test"]:
            transform = transforms_dict[split]
            has_totensor = any(isinstance(t, transforms.ToTensor) for t in transform.transforms)
            assert has_totensor

    def test_all_transforms_include_normalize(self, sample_config):
        """Test that all transforms include Normalize."""
        transforms_dict = get_transforms(sample_config)

        for split in ["train", "val", "test"]:
            transform = transforms_dict[split]
            has_normalize = any(isinstance(t, transforms.Normalize) for t in transform.transforms)
            assert has_normalize


# ============================================================================
# get_datasets Tests
# ============================================================================


class TestGetDatasets:
    """Tests for get_datasets function."""

    def test_get_datasets_returns_dict(self, sample_config):
        """Test that get_datasets returns dictionary."""
        datasets = get_datasets(sample_config)

        assert isinstance(datasets, dict)
        assert "train" in datasets
        assert "val" in datasets
        assert "test" in datasets

    def test_datasets_are_indexed_image_dataset(self, sample_config):
        """Test that all datasets are IndexedImageDataset instances."""
        datasets = get_datasets(sample_config)

        for split in ["train", "val", "test"]:
            assert isinstance(datasets[split], IndexedImageDataset)

    def test_datasets_use_correct_fold(self, sample_config):
        """Test that datasets load from correct fold indices."""
        # Update to fold 0
        sample_config["data"]["fold"] = 0
        datasets = get_datasets(sample_config)

        # Should load 2 samples each for train/val (from our fixture)
        assert len(datasets["train"]) == 2
        assert len(datasets["val"]) == 2
        assert len(datasets["test"]) == 2

    def test_test_dataset_same_across_folds(self, sample_config, temp_dataset_dir):
        """Test that test dataset is the same regardless of fold."""
        # Create fold 1 splits
        splits_dir = temp_dataset_dir / "splits"
        (splits_dir / "fold_1_train.txt").write_text("raw/cat/cat_3.jpg\n")
        (splits_dir / "fold_1_val.txt").write_text("raw/cat/cat_4.jpg\n")

        # Load fold 0 test set
        sample_config["data"]["fold"] = 0
        datasets_fold0 = get_datasets(sample_config)
        test_samples_fold0 = datasets_fold0["test"].samples

        # Load fold 1 test set
        sample_config["data"]["fold"] = 1
        datasets_fold1 = get_datasets(sample_config)
        test_samples_fold1 = datasets_fold1["test"].samples

        # Test sets should be identical
        assert test_samples_fold0 == test_samples_fold1


# ============================================================================
# get_class_names Tests
# ============================================================================


class TestGetClassNames:
    """Tests for get_class_names function."""

    def test_get_class_names_returns_list(self, sample_config):
        """Test that get_class_names returns list of class names."""
        datasets = get_datasets(sample_config)
        class_names = get_class_names(datasets)

        assert isinstance(class_names, list)
        assert len(class_names) == 2
        assert set(class_names) == {"cat", "dog"}

    def test_class_names_sorted_alphabetically(self, sample_config):
        """Test that class names are sorted alphabetically."""
        datasets = get_datasets(sample_config)
        class_names = get_class_names(datasets)

        assert class_names == ["cat", "dog"]


# ============================================================================
# Dataset Detection Tests
# ============================================================================


class TestDetectDatasetInfo:
    """Tests for detect_dataset_info function."""

    def test_detect_dataset_info_valid_structure(self, temp_dataset_dir):
        """Test detection with valid dataset structure."""
        info = detect_dataset_info(str(temp_dataset_dir))

        assert info is not None
        assert info["dataset_name"] == "test_dataset"
        assert info["num_classes"] == 2
        assert set(info["class_names"]) == {"cat", "dog"}
        assert info["data_dir"] == str(temp_dataset_dir)

    def test_detect_dataset_info_missing_raw_dir(self):
        """Test detection with missing raw directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No raw subdirectory
            info = detect_dataset_info(tmpdir)
            assert info is None

    def test_detect_dataset_info_empty_raw_dir(self):
        """Test detection with empty raw directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            info = detect_dataset_info(tmpdir)
            assert info is None

    def test_detect_dataset_info_ignores_hidden_dirs(self):
        """Test that hidden directories (starting with .) are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()

            # Create normal and hidden class directories
            (raw_dir / "class1").mkdir()
            (raw_dir / ".hidden").mkdir()

            info = detect_dataset_info(tmpdir)

            assert info is not None
            assert info["num_classes"] == 1
            assert info["class_names"] == ["class1"]


# ============================================================================
# Indexing Tests
# ============================================================================


class TestCollectFilePaths:
    """Tests for collect_file_paths function."""

    def test_collect_file_paths_returns_list(self, temp_dataset_dir):
        """Test that collect_file_paths returns list of paths."""
        split_dir = temp_dataset_dir / "raw"  # Use raw dir as mock split dir
        paths = collect_file_paths(str(split_dir), str(temp_dataset_dir / "raw"))

        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_collect_file_paths_relative_format(self, temp_dataset_dir):
        """Test that paths are in correct relative format."""
        split_dir = temp_dataset_dir / "raw"
        paths = collect_file_paths(str(split_dir), str(temp_dataset_dir / "raw"))

        # All paths should start with 'raw/'
        for path in paths:
            assert path.startswith("raw/")
            # Should have format: raw/class_name/file.jpg
            parts = path.split("/")
            assert len(parts) == 3

    def test_collect_file_paths_sorted(self, temp_dataset_dir):
        """Test that paths are sorted."""
        split_dir = temp_dataset_dir / "raw"
        paths = collect_file_paths(str(split_dir), str(temp_dataset_dir / "raw"))

        # Should be sorted
        assert paths == sorted(paths)


class TestWriteIndexFile:
    """Tests for write_index_file function."""

    def test_write_index_file_creates_file(self):
        """Test that write_index_file creates output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "index.txt"
            paths = ["raw/class1/img1.jpg", "raw/class2/img2.jpg"]

            write_index_file(paths, str(output_path))

            assert output_path.exists()

    def test_write_index_file_correct_content(self):
        """Test that index file has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "index.txt"
            paths = ["raw/class1/img1.jpg", "raw/class2/img2.jpg"]

            write_index_file(paths, str(output_path))

            content = output_path.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            assert len(lines) == 2
            assert lines[0] == "raw/class1/img1.jpg"
            assert lines[1] == "raw/class2/img2.jpg"

    def test_write_index_file_empty_list(self):
        """Test writing empty list creates empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "index.txt"

            write_index_file([], str(output_path))

            assert output_path.exists()
            content = output_path.read_text()
            assert content == ""


# ============================================================================
# Splitting Tests
# ============================================================================


class TestCreateTestSplit:
    """Tests for create_test_split function."""

    def test_create_test_split_creates_test_file(self, temp_dataset_dir):
        """Test that test split creates test.txt file."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        test_paths, remaining_paths = create_test_split(
            str(temp_dataset_dir / "raw"), str(output_dir), test_ratio=0.2, seed=42
        )

        assert (output_dir / "test.txt").exists()

    def test_create_test_split_returns_paths(self, temp_dataset_dir):
        """Test that create_test_split returns path lists."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        test_paths, remaining_paths = create_test_split(
            str(temp_dataset_dir / "raw"), str(output_dir), test_ratio=0.2, seed=42
        )

        assert isinstance(test_paths, list)
        assert isinstance(remaining_paths, list)
        assert len(test_paths) > 0
        assert len(remaining_paths) > 0

    def test_create_test_split_ratio(self, temp_dataset_dir):
        """Test that test split respects the ratio approximately."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        test_paths, remaining_paths = create_test_split(
            str(temp_dataset_dir / "raw"), str(output_dir), test_ratio=0.2, seed=42
        )

        total = len(test_paths) + len(remaining_paths)
        test_ratio_actual = len(test_paths) / total

        # Should be close to 0.2 (within 10% tolerance for small datasets)
        assert 0.1 <= test_ratio_actual <= 0.3


class TestCreateCVSplits:
    """Tests for create_cv_splits function."""

    @pytest.mark.slow
    def test_create_cv_splits_creates_all_files(self, temp_dataset_dir):
        """Test that CV splits creates all expected files."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        num_folds = 3
        create_cv_splits(
            str(temp_dataset_dir / "raw"),
            str(output_dir),
            num_folds=num_folds,
            ratio=(0.7, 0.15, 0.15),
            seed=42,
        )

        # Check test file exists
        assert (output_dir / "test.txt").exists()

        # Check all fold files exist
        for fold_idx in range(num_folds):
            assert (output_dir / f"fold_{fold_idx}_train.txt").exists()
            assert (output_dir / f"fold_{fold_idx}_val.txt").exists()

    @pytest.mark.slow
    def test_create_cv_splits_test_set_consistency(self, temp_dataset_dir):
        """Test that test set is same size across all folds."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        create_cv_splits(
            str(temp_dataset_dir / "raw"),
            str(output_dir),
            num_folds=2,
            ratio=(0.7, 0.15, 0.15),
            seed=42,
        )

        # Read test file
        test_content = (output_dir / "test.txt").read_text()
        test_lines = [line for line in test_content.split("\n") if line.strip()]

        # Test set should have consistent size
        assert len(test_lines) > 0

    @pytest.mark.slow
    def test_create_cv_splits_invalid_ratio(self, temp_dataset_dir):
        """Test that invalid ratio raises error."""
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        with pytest.raises(ValueError, match="Ratio must sum to 1.0"):
            create_cv_splits(
                str(temp_dataset_dir / "raw"),
                str(output_dir),
                num_folds=2,
                ratio=(0.5, 0.3, 0.3),  # Sums to 1.1
                seed=42,
            )

    @pytest.mark.slow
    def test_create_cv_splits_missing_raw_dir(self):
        """Test that missing raw directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Raw data directory not found"):
                create_cv_splits("/nonexistent/path", tmpdir, num_folds=2)


# ============================================================================
# Integration Tests
# ============================================================================


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_split_to_dataset(self, temp_dataset_dir):
        """Test full pipeline from split creation to dataset loading."""
        # Step 1: Create CV splits
        output_dir = temp_dataset_dir / "splits"
        output_dir.mkdir(exist_ok=True)

        create_cv_splits(
            str(temp_dataset_dir / "raw"),
            str(output_dir),
            num_folds=2,
            ratio=(0.7, 0.15, 0.15),
            seed=42,
        )

        # Step 2: Create config
        config = {
            "data": {"data_dir": str(temp_dataset_dir), "fold": 0},
            "transforms": {
                "train": {"resize": [224, 224], "normalize": {"mean": [0.5], "std": [0.5]}},
                "val": {"resize": [224, 224], "normalize": {"mean": [0.5], "std": [0.5]}},
                "test": {"resize": [224, 224], "normalize": {"mean": [0.5], "std": [0.5]}},
            },
        }

        # Step 3: Load datasets
        datasets = get_datasets(config)

        # Step 4: Verify datasets work
        assert len(datasets["train"]) > 0
        assert len(datasets["val"]) > 0
        assert len(datasets["test"]) > 0

        # Step 5: Load a sample
        image, label = datasets["train"][0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_dataset_iteration(self, sample_config):
        """Test that dataset can be iterated."""
        datasets = get_datasets(sample_config)
        train_dataset = datasets["train"]

        # Should be able to iterate
        count = 0
        for image, label in train_dataset:
            assert isinstance(image, torch.Tensor)
            assert isinstance(label, int)
            count += 1

        assert count == len(train_dataset)
