#!/usr/bin/env python3
"""
Data splitting utility for creating cross-validation folds.

This script generates k-fold cross-validation splits using split-folders library
and saves them as index files (no data duplication).
"""

import argparse
import os
import tempfile
from pathlib import Path

import splitfolders
from loguru import logger


def collect_file_paths(split_dir, raw_data_dir):
    """
    Collect all file paths from a split directory (train/val/test).

    Args:
        split_dir: Directory containing class subdirectories with images (from temp dir)
        raw_data_dir: Original raw data directory to reference in paths

    Returns:
        List of relative file paths (relative to parent of raw_data_dir)
    """
    file_paths = []
    split_path = Path(split_dir)
    raw_path = Path(raw_data_dir)
    raw_name = raw_path.name  # 'raw'

    # Walk through class subdirectories
    for class_dir in sorted(split_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        # Collect all files in this class directory
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file():
                # Create path like: raw/class_name/image.jpg
                rel_path = f"{raw_name}/{class_name}/{file_path.name}"
                file_paths.append(rel_path)

    return file_paths


def write_index_file(file_paths, output_path):
    """
    Write list of file paths to an index file.

    Args:
        file_paths: List of file paths to write
        output_path: Path to output index file
    """
    with open(output_path, "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")

    logger.info(f"Written {len(file_paths)} paths to {output_path}")


def create_single_fold(raw_data_dir, temp_output_dir, ratio, seed):
    """
    Create a single fold split using split-folders.

    Args:
        raw_data_dir: Directory containing raw unsplit data (class subdirs)
        temp_output_dir: Temporary directory for split output
        ratio: Tuple of (train, val, test) ratios
        seed: Random seed for reproducibility

    Returns:
        Paths to train, val, test directories
    """
    logger.info(f"Creating fold with ratio {ratio}, seed {seed}")

    # Use split-folders to create the split
    splitfolders.ratio(
        raw_data_dir,
        output=temp_output_dir,
        seed=seed,
        ratio=ratio,
        group_prefix=None,
        move=False,  # Copy files (we'll delete temp dir later)
    )

    train_dir = os.path.join(temp_output_dir, "train")
    val_dir = os.path.join(temp_output_dir, "val")
    test_dir = os.path.join(temp_output_dir, "test")

    return train_dir, val_dir, test_dir


def create_test_split(raw_data_dir, output_dir, test_ratio=0.15, seed=42):
    """
    Create a single held-out test set (same for all folds).

    Args:
        raw_data_dir: Directory containing raw unsplit data
        output_dir: Directory to save index files
        test_ratio: Ratio of data to use for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (test_paths, remaining_paths) where remaining_paths
        will be used for k-fold train/val splits
    """
    logger.info("Creating held-out test set...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Split into test and remaining (train+val pool)
        # ratio is (remaining, test) since split-folders takes (train, test)
        remaining_ratio = 1.0 - test_ratio
        splitfolders.ratio(
            raw_data_dir,
            output=temp_dir,
            seed=seed,
            ratio=(remaining_ratio, test_ratio),
            group_prefix=None,
            move=False,
        )

        # Collect paths
        test_dir = os.path.join(temp_dir, "val")  # split-folders uses "val" for 2nd split
        remaining_dir = os.path.join(temp_dir, "train")  # split-folders uses "train" for 1st split

        test_paths = collect_file_paths(test_dir, raw_data_dir)
        remaining_paths = collect_file_paths(remaining_dir, raw_data_dir)

        # Write test index file
        write_index_file(test_paths, os.path.join(output_dir, "test.txt"))
        logger.success(f"Test set created: {len(test_paths)} samples")

        return test_paths, remaining_paths


def create_train_val_fold(remaining_data_dir, temp_output_dir, train_ratio, seed):
    """
    Create a single train/val fold from remaining data pool.

    Args:
        remaining_data_dir: Directory containing remaining data (after test holdout)
        temp_output_dir: Temporary directory for split output
        train_ratio: Ratio of remaining data to use for training (rest is validation)
        seed: Random seed for reproducibility

    Returns:
        Paths to train and val directories
    """
    val_ratio = 1.0 - train_ratio

    # Use split-folders to create the split
    splitfolders.ratio(
        remaining_data_dir,
        output=temp_output_dir,
        seed=seed,
        ratio=(train_ratio, val_ratio),
        group_prefix=None,
        move=False,
    )

    train_dir = os.path.join(temp_output_dir, "train")
    val_dir = os.path.join(temp_output_dir, "val")

    return train_dir, val_dir


def create_cv_splits(
    raw_data_dir, output_dir, num_folds=5, ratio=(0.7, 0.15, 0.15), seed=42
):
    """
    Create k-fold cross-validation splits as index files.

    IMPORTANT: Test set is held out ONCE and is the SAME for all folds.
    Only train/val splits vary across folds.

    Args:
        raw_data_dir: Directory containing raw unsplit data
        output_dir: Directory to save index files
        num_folds: Number of CV folds to create
        ratio: Tuple of (train, val, test) split ratios
        seed: Base random seed for reproducibility

    Output structure:
        splits/
        ├── test.txt              # Single test set (same for all folds)
        ├── fold_0_train.txt      # ~70% of total data
        ├── fold_0_val.txt        # ~15% of total data
        ├── fold_1_train.txt
        └── ...
    """
    logger.info("=" * 60)
    logger.info("Cross-Validation Split Generation")
    logger.info("=" * 60)
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of folds: {num_folds}")
    logger.info(f"Split ratio (train/val/test): {ratio}")
    logger.info(f"Base seed: {seed}")
    logger.info("=" * 60)

    # Validate inputs
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")

    if sum(ratio) != 1.0:
        raise ValueError(f"Ratio must sum to 1.0, got {sum(ratio)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Create single held-out test set (same for all folds)
    train_ratio, val_ratio, test_ratio = ratio
    test_paths, remaining_paths = create_test_split(
        raw_data_dir, output_dir, test_ratio, seed
    )

    # Step 2: Create temporary directory with remaining data for k-fold CV
    with tempfile.TemporaryDirectory() as temp_remaining_dir:
        # Create class directories in temp dir
        raw_path = Path(raw_data_dir)
        for class_dir in raw_path.iterdir():
            if class_dir.is_dir():
                os.makedirs(os.path.join(temp_remaining_dir, class_dir.name), exist_ok=True)

        # Copy remaining files to temp directory
        data_root_parent = str(raw_path.parent)  # Parent of raw/ directory
        for rel_path in remaining_paths:
            # rel_path format: raw/class_name/image.jpg
            parts = rel_path.split("/")
            class_name = parts[-2]
            file_name = parts[-1]

            # Construct absolute source path
            src = os.path.join(data_root_parent, rel_path)
            dst = os.path.join(temp_remaining_dir, class_name, file_name)

            # Copy file (splitfolders doesn't follow symlinks)
            import shutil
            shutil.copy2(src, dst)

        # Step 3: Generate k-fold train/val splits on remaining data
        # Calculate train ratio within remaining data
        remaining_total = train_ratio + val_ratio
        train_ratio_within_remaining = train_ratio / remaining_total

        for fold_idx in range(num_folds):
            logger.info(f"\nGenerating fold {fold_idx}/{num_folds-1} (train/val only)...")

            # Use different seed for each fold
            fold_seed = seed + fold_idx + 1000  # Offset to avoid collision with test seed

            with tempfile.TemporaryDirectory() as temp_fold_dir:
                # Create train/val split
                train_dir, val_dir = create_train_val_fold(
                    temp_remaining_dir, temp_fold_dir, train_ratio_within_remaining, fold_seed
                )

                # Collect file paths
                train_paths_fold = collect_file_paths(train_dir, raw_data_dir)
                val_paths_fold = collect_file_paths(val_dir, raw_data_dir)

                # Write index files
                write_index_file(
                    train_paths_fold, os.path.join(output_dir, f"fold_{fold_idx}_train.txt")
                )
                write_index_file(
                    val_paths_fold, os.path.join(output_dir, f"fold_{fold_idx}_val.txt")
                )

                logger.success(
                    f"Fold {fold_idx} complete: {len(train_paths_fold)} train, "
                    f"{len(val_paths_fold)} val"
                )

    logger.info("=" * 60)
    logger.success(f"All {num_folds} folds generated successfully!")
    logger.info(f"Test set: {len(test_paths)} samples (SAME for all folds)")
    logger.info(f"Index files saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate k-fold cross-validation splits as index files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5-fold CV splits (output auto-derived as data/my_dataset/splits)
  ml-split --raw_data data/my_dataset/raw

  # Generate 10 folds with custom ratio
  ml-split --raw_data data/my_dataset/raw --folds 10 --ratio 0.8 0.1 0.1

  # Custom output location
  ml-split --raw_data data/my_dataset/raw --output custom/location/splits

  # Use custom seed for reproducibility
  ml-split --raw_data data/my_dataset/raw --seed 123

Expected raw data structure:
  data/my_dataset/raw/
  ├── class1/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── class2/
      └── img3.jpg

Output structure:
  data/my_dataset/splits/
  ├── test.txt              # Single test set (SAME for all folds)
  ├── fold_0_train.txt
  ├── fold_0_val.txt
  ├── fold_1_train.txt
  ├── fold_1_val.txt
  └── ...
        """,
    )

    parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
        help="Path to raw data directory containing class subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output directory for index files (default: auto-derived from raw_data by replacing 'raw' with 'splits')",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds to generate (default: 5)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train/val/test (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Auto-derive output path if not provided
    if args.output is None:
        # Replace 'raw' with 'splits' in the path
        raw_path = Path(args.raw_data)

        # Check if path ends with 'raw'
        if raw_path.name == 'raw':
            # Replace last component 'raw' with 'splits'
            output_dir = str(raw_path.parent / 'splits')
        else:
            # If not ending with 'raw', append '../splits' as sibling directory
            output_dir = str(raw_path.parent / 'splits')

        logger.info(f"Auto-derived output directory: {output_dir}")
    else:
        output_dir = args.output

    # Convert ratio list to tuple
    ratio = tuple(args.ratio)

    # Create CV splits
    create_cv_splits(
        raw_data_dir=args.raw_data,
        output_dir=output_dir,
        num_folds=args.folds,
        ratio=ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
