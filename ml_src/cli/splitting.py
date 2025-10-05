#!/usr/bin/env python3
"""
Data splitting utility for creating cross-validation folds.

This script generates k-fold cross-validation splits using split-folders library
and saves them as index files (no data duplication).
"""

import argparse
from pathlib import Path

from loguru import logger

from ml_src.core.data import create_cv_splits
from ml_src.core.logging import setup_logging


def main():
    """Main function for command-line interface."""
    setup_logging()  # Console only

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
