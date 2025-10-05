#!/usr/bin/env python3
"""
Data splitting utility for creating cross-validation folds.

This script generates k-fold cross-validation splits using split-folders library
and saves them as index files (no data duplication).
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from loguru import logger
import splitfolders


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
    with open(output_path, 'w') as f:
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
        move=False  # Copy files (we'll delete temp dir later)
    )

    train_dir = os.path.join(temp_output_dir, 'train')
    val_dir = os.path.join(temp_output_dir, 'val')
    test_dir = os.path.join(temp_output_dir, 'test')

    return train_dir, val_dir, test_dir


def create_cv_splits(raw_data_dir, output_dir, num_folds=5, ratio=(0.7, 0.15, 0.15), seed=42):
    """
    Create k-fold cross-validation splits as index files.

    Args:
        raw_data_dir: Directory containing raw unsplit data
        output_dir: Directory to save index files
        num_folds: Number of CV folds to create
        ratio: Tuple of (train, val, test) split ratios
        seed: Base random seed for reproducibility
    """
    logger.info("="*60)
    logger.info("Cross-Validation Split Generation")
    logger.info("="*60)
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of folds: {num_folds}")
    logger.info(f"Split ratio (train/val/test): {ratio}")
    logger.info(f"Base seed: {seed}")
    logger.info("="*60)

    # Validate inputs
    if not os.path.exists(raw_data_dir):
        raise ValueError(f"Raw data directory not found: {raw_data_dir}")

    if sum(ratio) != 1.0:
        raise ValueError(f"Ratio must sum to 1.0, got {sum(ratio)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate each fold
    for fold_idx in range(num_folds):
        logger.info(f"\nGenerating fold {fold_idx}/{num_folds-1}...")

        # Use different seed for each fold to get different splits
        fold_seed = seed + fold_idx

        # Create temporary directory for this fold
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate split using split-folders
            train_dir, val_dir, test_dir = create_single_fold(
                raw_data_dir, temp_dir, ratio, fold_seed
            )

            # Collect file paths for each split
            train_paths = collect_file_paths(train_dir, raw_data_dir)
            val_paths = collect_file_paths(val_dir, raw_data_dir)
            test_paths = collect_file_paths(test_dir, raw_data_dir)

            # Write index files
            write_index_file(
                train_paths,
                os.path.join(output_dir, f'fold_{fold_idx}_train.txt')
            )
            write_index_file(
                val_paths,
                os.path.join(output_dir, f'fold_{fold_idx}_val.txt')
            )
            write_index_file(
                test_paths,
                os.path.join(output_dir, f'fold_{fold_idx}_test.txt')
            )

            logger.success(f"Fold {fold_idx} complete: {len(train_paths)} train, "
                         f"{len(val_paths)} val, {len(test_paths)} test")

    logger.info("="*60)
    logger.success(f"All {num_folds} folds generated successfully!")
    logger.info(f"Index files saved to: {output_dir}")
    logger.info("="*60)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Generate k-fold cross-validation splits as index files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5-fold CV splits with default 70/15/15 ratio
  python splitting.py --raw_data data/my_dataset/raw --output data/my_dataset/splits

  # Generate 10 folds with custom ratio
  python splitting.py --raw_data data/my_dataset/raw --output data/my_dataset/splits --folds 10 --ratio 0.8 0.1 0.1

  # Use custom seed for reproducibility
  python splitting.py --raw_data data/my_dataset/raw --output data/my_dataset/splits --seed 123

Expected raw data structure:
  data/my_dataset/raw/
  ├── class1/
  │   ├── img1.jpg
  │   └── img2.jpg
  └── class2/
      └── img3.jpg

Output structure:
  data/my_dataset/splits/
  ├── fold_0_train.txt
  ├── fold_0_val.txt
  ├── fold_0_test.txt
  ├── fold_1_train.txt
  └── ...
        """
    )

    parser.add_argument('--raw_data', type=str, required=True,
                        help='Path to raw data directory containing class subdirectories')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory for index files')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds to generate (default: 5)')
    parser.add_argument('--ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios for train/val/test (default: 0.7 0.15 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Convert ratio list to tuple
    ratio = tuple(args.ratio)

    # Create CV splits
    create_cv_splits(
        raw_data_dir=args.raw_data,
        output_dir=args.output,
        num_folds=args.folds,
        ratio=ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
