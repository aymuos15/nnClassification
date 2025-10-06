#!/usr/bin/env python3
"""
CLI tool for exporting trained PyTorch models to ONNX format.

This tool supports:
- Single checkpoint export or batch export using glob patterns
- Automatic output path generation
- Optional validation of exported models
- Configurable ONNX opset version
- Custom input size specification
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch
from loguru import logger

from ml_src.core.data import get_datasets
from ml_src.core.export import export_to_onnx, validate_onnx_export
from ml_src.core.loader import get_dataloaders
from ml_src.core.metrics.onnx_validation import (
    benchmark_inference_speed,
    validate_onnx_model,
)
from ml_src.core.network import get_model


def export_with_validation(
    checkpoint_path,
    output_path,
    opset_version,
    input_size,
    validate,
    benchmark,
    comprehensive_validate,
    num_val_batches,
):
    """
    Export a single checkpoint to ONNX format with optional validation and benchmarking.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file
        output_path: Path where the ONNX model will be saved
        opset_version: ONNX opset version to use
        input_size: Tuple of input dimensions or None for auto-detection
        validate: Whether to validate the export (basic validation)
        benchmark: Whether to benchmark inference speed
        comprehensive_validate: Whether to use comprehensive validation with test loader
        num_val_batches: Number of batches for comprehensive validation

    Returns:
        tuple: (success: bool, message: str, validation_report: dict or None)
    """
    # Export to ONNX
    success, message = export_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        opset_version=opset_version,
        input_size=input_size,
    )

    if not success:
        return False, message, None

    validation_report = None

    # Comprehensive validation with test loader
    if comprehensive_validate:
        logger.info("=" * 80)
        logger.info("Running Comprehensive ONNX Validation")
        logger.info("=" * 80)

        try:
            # Load checkpoint to get config
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})

            if not config:
                logger.warning("No config found in checkpoint. Falling back to basic validation.")
                comprehensive_validate = False
                validate = True
            else:
                # Create test loader
                logger.info("Creating test dataset...")
                datasets = get_datasets(config)
                dataloaders = get_dataloaders(datasets, config)
                test_loader = dataloaders["test"]

                # Run comprehensive validation
                validation_report = validate_onnx_model(
                    checkpoint_path=checkpoint_path,
                    onnx_path=output_path,
                    test_loader=test_loader,
                    device="cpu",
                    num_batches=num_val_batches,
                )

                # Print status with color coding
                status = validation_report["status"]
                if status == "PASS":
                    logger.success(f"Validation Status: {status}")
                elif status == "WARN":
                    logger.warning(f"Validation Status: {status}")
                else:
                    logger.error(f"Validation Status: {status}")

                # Save validation report to JSON
                report_path = Path(output_path).with_suffix(".validation.json")
                with open(report_path, "w") as f:
                    json.dump(validation_report, f, indent=2)
                logger.info(f"Validation report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            logger.info("Falling back to basic validation...")
            comprehensive_validate = False
            validate = True

    # Basic validation (backward compatibility or fallback)
    if validate and not comprehensive_validate:
        logger.info("=" * 50)
        logger.info("Validating ONNX Export (Basic)")
        logger.info("=" * 50)

        try:
            # Load checkpoint and recreate model for validation
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})

            if not config:
                logger.warning("No config found in checkpoint. Skipping validation.")
                return success, message, validation_report

            # Recreate model
            model = get_model(config, device="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Determine input size (match export_to_onnx logic)
            if input_size is None:
                model_config = config.get("model", {})
                architecture = model_config.get("architecture", "").lower()
                custom_architecture = model_config.get("custom_architecture", "").lower()
                arch_name = architecture or custom_architecture

                input_size = (1, 3, 299, 299) if "inception" in arch_name else (1, 3, 224, 224)

            # Create dummy input
            dummy_input = torch.randn(input_size)

            # Run basic validation
            metrics = validate_onnx_export(
                pytorch_model=model,
                onnx_path=output_path,
                dummy_input=dummy_input,
                device="cpu",
            )

            # Add validation results to message
            validation_msg = (
                f"\nValidation [{metrics['status']}]: "
                f"max_diff={metrics['max_diff']:.2e}, "
                f"mse={metrics['mse']:.2e}, "
                f"mae={metrics['mae']:.2e}, "
                f"cosine_sim={metrics['cosine_similarity']:.6f}"
            )
            message = message + validation_msg

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            message = message + f"\nValidation failed: {str(e)}"

    # Benchmark inference speed
    if benchmark:
        logger.info("=" * 80)
        logger.info("Benchmarking Inference Speed")
        logger.info("=" * 80)

        try:
            # Load checkpoint and recreate model
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config", {})

            if not config:
                logger.warning("No config found in checkpoint. Skipping benchmark.")
                return success, message, validation_report

            model = get_model(config, device="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Determine input size
            if input_size is None:
                model_config = config.get("model", {})
                architecture = model_config.get("architecture", "").lower()
                custom_architecture = model_config.get("custom_architecture", "").lower()
                arch_name = architecture or custom_architecture

                input_size = (1, 3, 299, 299) if "inception" in arch_name else (1, 3, 224, 224)

            dummy_input = torch.randn(input_size)

            # Run benchmark
            benchmark_results = benchmark_inference_speed(
                pytorch_model=model,
                onnx_path=output_path,
                input_tensor=dummy_input,
                device="cpu",
                num_runs=100,
                warmup=10,
            )

            # Print formatted benchmark table
            logger.info("=" * 80)
            logger.info("Benchmark Results (100 runs after 10 warmup)")
            logger.info("=" * 80)
            logger.info(f"{'Metric':<20} {'PyTorch':<20} {'ONNX':<20} {'Improvement':<20}")
            logger.info("-" * 80)

            pytorch_times = benchmark_results["pytorch_times"]
            onnx_times = benchmark_results["onnx_times"]
            speedup = benchmark_results["speedup"]

            logger.info(
                f"{'Mean (ms)':<20} {pytorch_times['mean']:>19.2f} "
                f"{onnx_times['mean']:>19.2f} {speedup:>19.2f}x"
            )
            logger.info(
                f"{'Std Dev (ms)':<20} {pytorch_times['std']:>19.2f} "
                f"{onnx_times['std']:>19.2f} {'-':>20}"
            )
            logger.info(
                f"{'Min (ms)':<20} {pytorch_times['min']:>19.2f} "
                f"{onnx_times['min']:>19.2f} {'-':>20}"
            )
            logger.info(
                f"{'Max (ms)':<20} {pytorch_times['max']:>19.2f} "
                f"{onnx_times['max']:>19.2f} {'-':>20}"
            )
            logger.info(
                f"{'P50 (ms)':<20} {pytorch_times['p50']:>19.2f} "
                f"{onnx_times['p50']:>19.2f} {'-':>20}"
            )
            logger.info(
                f"{'P95 (ms)':<20} {pytorch_times['p95']:>19.2f} "
                f"{onnx_times['p95']:>19.2f} {'-':>20}"
            )
            logger.info(
                f"{'P99 (ms)':<20} {pytorch_times['p99']:>19.2f} "
                f"{onnx_times['p99']:>19.2f} {'-':>20}"
            )
            logger.info("=" * 80)

            if speedup > 1.0:
                logger.success(f"ONNX is {speedup:.2f}x faster than PyTorch")
            elif speedup < 1.0:
                logger.warning(f"ONNX is {1 / speedup:.2f}x slower than PyTorch")
            else:
                logger.info("ONNX and PyTorch have similar performance")

            # Save benchmark results
            benchmark_path = Path(output_path).with_suffix(".benchmark.json")
            with open(benchmark_path, "w") as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Benchmark results saved to: {benchmark_path}")

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

    return success, message, validation_report


def main():
    """Main function for ONNX export CLI."""
    parser = argparse.ArgumentParser(
        description="Export trained PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export a single checkpoint
  ml-export --checkpoint runs/my_run/weights/best.pt

  # Export with custom output path
  ml-export --checkpoint runs/my_run/weights/best.pt --output models/my_model.onnx

  # Export with comprehensive validation (uses test dataset)
  ml-export --checkpoint runs/my_run/weights/best.pt --validate

  # Export with basic validation (uses dummy input, faster)
  ml-export --checkpoint runs/my_run/weights/best.pt --validate-basic

  # Export with benchmark (speed comparison)
  ml-export --checkpoint runs/my_run/weights/best.pt --benchmark

  # Export with validation and benchmark
  ml-export --checkpoint runs/my_run/weights/best.pt --validate --benchmark

  # Export multiple checkpoints using glob pattern
  ml-export --checkpoint "runs/*/weights/best.pt"

  # Export with custom input size (channels, height, width)
  ml-export --checkpoint runs/my_run/weights/best.pt --input-size 3,299,299

  # Export with specific ONNX opset version
  ml-export --checkpoint runs/my_run/weights/best.pt --opset-version 15

  # Comprehensive validation with custom batch count
  ml-export --checkpoint runs/my_run/weights/best.pt --validate --num-val-batches 20

Validation options:
  --validate          Comprehensive validation using test dataset (requires config in checkpoint)
                      - Tests on multiple real batches from test set
                      - Computes aggregated metrics (max_diff, MSE, MAE, cosine similarity)
                      - Saves detailed validation report to .validation.json
                      - Reports PASS/WARN/FAIL status based on numerical accuracy

  --validate-basic    Basic validation using dummy input (no dataset required)
                      - Fast validation for quick checks
                      - Uses single random input tensor
                      - Good for verifying export correctness

  --benchmark         Benchmark PyTorch vs ONNX inference speed
                      - Runs 100 inference iterations (after 10 warmup)
                      - Reports timing statistics (mean, std, percentiles)
                      - Shows speedup factor (e.g., 2.5x faster)
                      - Saves benchmark results to .benchmark.json

Format options:
  --format onnx       Export to ONNX format (currently the only supported format)

Notes:
  - Output path defaults to same directory as checkpoint with .onnx extension
  - Input size is automatically detected based on model architecture
  - Inception models use 299x299 input, all others use 224x224
  - Glob patterns must be quoted to prevent shell expansion
  - Comprehensive validation requires checkpoint to contain config and dataset access
  - If comprehensive validation fails, it automatically falls back to basic validation
  - Validation and benchmark results are saved as JSON files alongside the ONNX model
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help='Path or glob pattern to checkpoint(s) (e.g., "runs/*/weights/best.pt")',
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for exported model (default: same as checkpoint with .onnx extension)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx"],
        help="Export format (default: onnx)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run comprehensive validation after export using test dataset (requires config in checkpoint)",
    )
    parser.add_argument(
        "--validate-basic",
        action="store_true",
        help="Run basic validation with dummy input (faster, no dataset required)",
    )
    parser.add_argument(
        "--num-val-batches",
        type=int,
        default=10,
        help="Number of batches for comprehensive validation (default: 10)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed comparison between PyTorch and ONNX",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        help="Custom input size as comma-separated values (e.g., '3,299,299' for C,H,W)",
    )

    args = parser.parse_args()

    # Parse input size if provided
    input_size = None
    if args.input_size:
        try:
            size_parts = [int(x.strip()) for x in args.input_size.split(",")]
            if len(size_parts) == 3:
                # User provided C,H,W - add batch dimension
                input_size = (1, size_parts[0], size_parts[1], size_parts[2])
            elif len(size_parts) == 4:
                # User provided B,C,H,W
                input_size = tuple(size_parts)
            else:
                logger.error(
                    "Invalid input size format. Expected 3 values (C,H,W) or 4 values (B,C,H,W)"
                )
                sys.exit(1)
            logger.info(f"Using custom input size: {input_size}")
        except ValueError:
            logger.error(
                f"Invalid input size format: {args.input_size}. Expected comma-separated integers"
            )
            sys.exit(1)

    # Expand glob pattern to get list of checkpoints
    checkpoint_paths = glob.glob(args.checkpoint)

    if not checkpoint_paths:
        logger.error(f"No checkpoints found matching pattern: {args.checkpoint}")
        sys.exit(1)

    # Sort for consistent ordering
    checkpoint_paths.sort()

    logger.info(f"Found {len(checkpoint_paths)} checkpoint(s) to export")

    # Track results
    results = {"total": len(checkpoint_paths), "succeeded": 0, "failed": 0}
    failed_exports = []

    # Use tqdm for progress bar if processing multiple checkpoints
    if len(checkpoint_paths) > 1:
        try:
            from tqdm import tqdm

            checkpoint_iter = tqdm(checkpoint_paths, desc="Exporting models")
        except ImportError:
            logger.warning("tqdm not installed. Install with 'pip install tqdm' for progress bars")
            checkpoint_iter = checkpoint_paths
    else:
        checkpoint_iter = checkpoint_paths

    # Process each checkpoint
    for checkpoint_path in checkpoint_iter:
        checkpoint_path = Path(checkpoint_path)

        # Determine output path
        if args.output and len(checkpoint_paths) == 1:
            # Use provided output path only if processing a single checkpoint
            output_path = Path(args.output)
        else:
            # Auto-generate output path
            output_path = checkpoint_path.with_suffix(".onnx")

        logger.info("=" * 70)
        logger.info(f"Processing: {checkpoint_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 70)

        # Export (and optionally validate/benchmark)
        success, message, validation_report = export_with_validation(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            opset_version=args.opset_version,
            input_size=input_size,
            validate=args.validate_basic,
            benchmark=args.benchmark,
            comprehensive_validate=args.validate,
            num_val_batches=args.num_val_batches,
        )

        if success:
            results["succeeded"] += 1
            logger.success(f"Export completed: {output_path}")
        else:
            results["failed"] += 1
            failed_exports.append((str(checkpoint_path), message))
            logger.error(f"Export failed: {checkpoint_path}")

        logger.info("")  # Empty line for readability

    # Print summary report
    logger.info("=" * 70)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total checkpoints: {results['total']}")
    logger.success(f"Succeeded: {results['succeeded']}")
    if results["failed"] > 0:
        logger.error(f"Failed: {results['failed']}")
        logger.info("\nFailed exports:")
        for checkpoint, error_msg in failed_exports:
            logger.error(f"  {checkpoint}")
            logger.error(f"    Reason: {error_msg}")
    else:
        logger.success("All exports completed successfully!")
    logger.info("=" * 70)

    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
