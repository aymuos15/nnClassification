"""ONNX model validation and performance benchmarking utilities.

This module provides comprehensive tools for validating ONNX exports and comparing
PyTorch vs ONNX Runtime inference performance. It includes metrics for numerical
accuracy validation and detailed timing benchmarks.
"""

import time
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger

from ml_src.core.checkpointing import load_checkpoint
from ml_src.core.metrics.utils import ensure_numpy
from ml_src.core.network import get_model


def compare_outputs(
    pytorch_output: Union[torch.Tensor, np.ndarray],
    onnx_output: Union[torch.Tensor, np.ndarray],
) -> dict:
    """
    Compare PyTorch and ONNX model outputs using multiple metrics.

    Computes comprehensive numerical comparison metrics to validate that an
    ONNX export produces outputs consistent with the original PyTorch model.
    All inputs are automatically converted to numpy arrays for comparison.

    Args:
        pytorch_output: Output from PyTorch model inference. Can be torch.Tensor
            or numpy array. Shape: (batch_size, ...) or any compatible shape.
        onnx_output: Output from ONNX Runtime inference. Can be torch.Tensor
            or numpy array. Must have same shape as pytorch_output.

    Returns:
        dict: Dictionary containing comparison metrics:
            - max_diff (float): Maximum absolute difference between outputs
            - mse (float): Mean squared error between outputs
            - mae (float): Mean absolute error between outputs
            - cosine_similarity (float): Cosine similarity between flattened outputs
                (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)

    Raises:
        ValueError: If pytorch_output and onnx_output have different shapes

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> # Compare identical outputs
        >>> pytorch_out = torch.randn(2, 10)
        >>> onnx_out = pytorch_out.numpy()
        >>> metrics = compare_outputs(pytorch_out, onnx_out)
        >>> metrics['max_diff'] < 1e-6
        True
        >>> metrics['cosine_similarity'] > 0.9999
        True

        >>> # Compare slightly different outputs
        >>> pytorch_out = np.array([[1.0, 2.0, 3.0]])
        >>> onnx_out = np.array([[1.001, 2.001, 3.001]])
        >>> metrics = compare_outputs(pytorch_out, onnx_out)
        >>> print(f"Max diff: {metrics['max_diff']:.6f}")
        Max diff: 0.001000
        >>> print(f"MSE: {metrics['mse']:.9f}")
        MSE: 0.000001000

        >>> # Works with torch tensors
        >>> pytorch_out = torch.tensor([[0.5, 0.3, 0.2]])
        >>> onnx_out = torch.tensor([[0.5, 0.3, 0.2]])
        >>> metrics = compare_outputs(pytorch_out, onnx_out)
        >>> metrics['mae']
        0.0
    """
    # Convert to numpy arrays
    pytorch_np = ensure_numpy(pytorch_output)
    onnx_np = ensure_numpy(onnx_output)

    # Validate shapes match
    if pytorch_np.shape != onnx_np.shape:
        raise ValueError(
            f"Output shapes must match. Got PyTorch: {pytorch_np.shape}, "
            f"ONNX: {onnx_np.shape}"
        )

    # Compute difference metrics
    max_diff = float(np.abs(pytorch_np - onnx_np).max())
    mse = float(np.mean((pytorch_np - onnx_np) ** 2))
    mae = float(np.mean(np.abs(pytorch_np - onnx_np)))

    # Compute cosine similarity
    pytorch_flat = pytorch_np.flatten()
    onnx_flat = onnx_np.flatten()

    dot_product = np.dot(pytorch_flat, onnx_flat)
    pytorch_norm = np.linalg.norm(pytorch_flat)
    onnx_norm = np.linalg.norm(onnx_flat)

    if pytorch_norm > 0 and onnx_norm > 0:
        cosine_sim = float(dot_product / (pytorch_norm * onnx_norm))
    else:
        cosine_sim = 0.0

    return {
        "max_diff": max_diff,
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cosine_sim,
    }


def benchmark_inference_speed(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_tensor: torch.Tensor,
    device: str = "cpu",
    num_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark and compare PyTorch vs ONNX Runtime inference speed.

    Performs repeated inference runs with both PyTorch and ONNX Runtime models
    to provide comprehensive timing statistics. Includes warmup runs to ensure
    fair comparison (GPU kernel initialization, memory allocation, etc.).

    Args:
        pytorch_model: PyTorch model in eval mode
        onnx_path: Path to the exported ONNX model file
        input_tensor: Input tensor for inference. Will be used as-is for PyTorch
            and converted to numpy for ONNX Runtime.
        device: Device for PyTorch inference ('cpu' or 'cuda'). ONNX Runtime
            will use CPUExecutionProvider. Default: 'cpu'
        num_runs: Number of inference runs to perform for timing (after warmup).
            Higher values give more accurate statistics. Default: 100
        warmup: Number of warmup runs before timing starts. Default: 10

    Returns:
        dict: Dictionary containing timing statistics:
            - pytorch_times (dict): PyTorch inference timing statistics
                - mean (float): Mean inference time in milliseconds
                - std (float): Standard deviation in milliseconds
                - min (float): Minimum inference time in milliseconds
                - max (float): Maximum inference time in milliseconds
                - p50 (float): Median (50th percentile) in milliseconds
                - p95 (float): 95th percentile in milliseconds
                - p99 (float): 99th percentile in milliseconds
            - onnx_times (dict): ONNX Runtime inference timing statistics
                (same structure as pytorch_times)
            - speedup (float): Speedup factor (pytorch_mean / onnx_mean).
                Values > 1.0 indicate ONNX is faster.

    Raises:
        ImportError: If onnxruntime is not installed
        FileNotFoundError: If ONNX model file does not exist

    Examples:
        >>> import torch
        >>> from torchvision.models import resnet18
        >>> # Setup
        >>> model = resnet18(pretrained=True).eval()
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> # Export to ONNX first
        >>> torch.onnx.export(model, input_tensor, "model.onnx")
        >>> # Benchmark
        >>> stats = benchmark_inference_speed(
        ...     model, "model.onnx", input_tensor,
        ...     device='cpu', num_runs=100, warmup=10
        ... )
        >>> print(f"PyTorch: {stats['pytorch_times']['mean']:.2f} ms")
        >>> print(f"ONNX: {stats['onnx_times']['mean']:.2f} ms")
        >>> print(f"Speedup: {stats['speedup']:.2f}x")

        >>> # GPU benchmark (if available)
        >>> if torch.cuda.is_available():
        ...     model_gpu = model.to('cuda')
        ...     input_gpu = input_tensor.to('cuda')
        ...     stats = benchmark_inference_speed(
        ...         model_gpu, "model.onnx", input_gpu,
        ...         device='cuda', num_runs=100
        ...     )
        ...     print(f"GPU speedup: {stats['speedup']:.2f}x")
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "onnxruntime is not installed. Install it with: pip install onnxruntime"
        )

    logger.info(f"Benchmarking inference speed (warmup={warmup}, runs={num_runs})")

    # Move model and input to device
    pytorch_model = pytorch_model.to(device)
    input_tensor = input_tensor.to(device)
    pytorch_model.eval()

    # Convert input to numpy for ONNX
    input_numpy = input_tensor.cpu().numpy()

    # Initialize ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # === PyTorch Warmup ===
    logger.debug(f"PyTorch warmup ({warmup} runs)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = pytorch_model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()

    # === PyTorch Timing ===
    logger.debug(f"PyTorch inference timing ({num_runs} runs)...")
    pytorch_times_list = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = pytorch_model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            pytorch_times_list.append((end - start) * 1000)  # Convert to ms

    # === ONNX Warmup ===
    logger.debug(f"ONNX Runtime warmup ({warmup} runs)...")
    for _ in range(warmup):
        _ = session.run(None, {input_name: input_numpy})

    # === ONNX Timing ===
    logger.debug(f"ONNX Runtime inference timing ({num_runs} runs)...")
    onnx_times_list = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_numpy})
        end = time.perf_counter()
        onnx_times_list.append((end - start) * 1000)  # Convert to ms

    # Compute statistics
    pytorch_times_array = np.array(pytorch_times_list)
    onnx_times_array = np.array(onnx_times_list)

    pytorch_stats = {
        "mean": float(np.mean(pytorch_times_array)),
        "std": float(np.std(pytorch_times_array)),
        "min": float(np.min(pytorch_times_array)),
        "max": float(np.max(pytorch_times_array)),
        "p50": float(np.percentile(pytorch_times_array, 50)),
        "p95": float(np.percentile(pytorch_times_array, 95)),
        "p99": float(np.percentile(pytorch_times_array, 99)),
    }

    onnx_stats = {
        "mean": float(np.mean(onnx_times_array)),
        "std": float(np.std(onnx_times_array)),
        "min": float(np.min(onnx_times_array)),
        "max": float(np.max(onnx_times_array)),
        "p50": float(np.percentile(onnx_times_array, 50)),
        "p95": float(np.percentile(onnx_times_array, 95)),
        "p99": float(np.percentile(onnx_times_array, 99)),
    }

    speedup = pytorch_stats["mean"] / onnx_stats["mean"] if onnx_stats["mean"] > 0 else 0.0

    logger.info(f"PyTorch: {pytorch_stats['mean']:.2f} ms (±{pytorch_stats['std']:.2f})")
    logger.info(f"ONNX: {onnx_stats['mean']:.2f} ms (±{onnx_stats['std']:.2f})")
    logger.info(f"Speedup: {speedup:.2f}x")

    return {
        "pytorch_times": pytorch_stats,
        "onnx_times": onnx_stats,
        "speedup": speedup,
    }


def validate_onnx_model(
    checkpoint_path: str,
    onnx_path: str,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    num_batches: int = 10,
    use_ema: bool = False,
) -> dict:
    """
    Comprehensive ONNX model validation with PyTorch comparison.

    Loads both PyTorch and ONNX models, runs inference on multiple batches from
    a test loader, and aggregates numerical accuracy metrics. Provides a detailed
    validation report printed to console via loguru.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file (.pt). Must contain
            'model_state_dict' and 'config' keys for model reconstruction.
        onnx_path: Path to exported ONNX model file (.onnx)
        test_loader: PyTorch DataLoader providing test batches. Each batch
            should return (images, labels) or just (images,).
        device: Device for PyTorch inference ('cpu' or 'cuda'). Default: 'cpu'
        num_batches: Number of batches to process for validation. Use fewer
            batches for faster validation or more for comprehensive testing.
            Default: 10

    Returns:
        dict: Comprehensive validation report containing:
            - num_batches_tested (int): Number of batches processed
            - total_samples (int): Total number of samples validated
            - metrics (dict): Aggregated metrics across all batches
                - max_diff (dict): max, mean, min of maximum differences
                - mse (dict): max, mean, min of MSE values
                - mae (dict): max, mean, min of MAE values
                - cosine_similarity (dict): max, mean, min of cosine similarities
            - status (str): Overall validation status ('PASS', 'WARN', or 'FAIL')
            - per_batch_metrics (list): List of metrics for each individual batch

    Raises:
        ImportError: If onnxruntime is not installed
        FileNotFoundError: If checkpoint or ONNX file does not exist
        ValueError: If checkpoint has invalid format

    Examples:
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> import torch
        >>> # Create dummy test loader
        >>> test_data = TensorDataset(
        ...     torch.randn(100, 3, 224, 224),
        ...     torch.randint(0, 10, (100,))
        ... )
        >>> test_loader = DataLoader(test_data, batch_size=16)
        >>> # Validate ONNX export
        >>> report = validate_onnx_model(
        ...     checkpoint_path='runs/my_run/weights/best.pt',
        ...     onnx_path='models/my_model.onnx',
        ...     test_loader=test_loader,
        ...     device='cpu',
        ...     num_batches=10
        ... )
        >>> print(f"Status: {report['status']}")
        >>> print(f"Tested {report['total_samples']} samples")
        >>> print(f"Mean max diff: {report['metrics']['max_diff']['mean']:.2e}")

        >>> # Check if validation passed
        >>> if report['status'] == 'PASS':
        ...     print("ONNX export validated successfully!")
        ... elif report['status'] == 'WARN':
        ...     print("ONNX export has minor differences")
        ... else:
        ...     print("ONNX export validation failed!")

        >>> # GPU validation (if available)
        >>> if torch.cuda.is_available():
        ...     report = validate_onnx_model(
        ...         'runs/my_run/weights/best.pt',
        ...         'models/my_model.onnx',
        ...         test_loader,
        ...         device='cuda',
        ...         num_batches=5
        ...     )
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "onnxruntime is not installed. Install it with: pip install onnxruntime"
        )

    logger.info("=" * 80)
    logger.info("ONNX Model Validation Report")
    logger.info("=" * 80)

    # Load PyTorch model
    logger.info(f"Loading PyTorch checkpoint: {checkpoint_path}")
    # Load checkpoint directly without using load_checkpoint (which requires model/optimizer/scheduler)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    pytorch_model = get_model(config, device=device)

    model_state = checkpoint["model_state_dict"]
    if use_ema:
        ema_state = checkpoint.get("ema_state", {}).get("ema_model_state")
        if ema_state is None:
            logger.warning(
                "EMA weights requested but not found in checkpoint {}. Using standard weights.",
                checkpoint_path,
            )
        else:
            model_state = ema_state
            logger.info("Validating ONNX against EMA weights")

    pytorch_model.load_state_dict(model_state)
    pytorch_model.eval()
    logger.success("PyTorch model loaded successfully")

    # Load ONNX model
    logger.info(f"Loading ONNX model: {onnx_path}")
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logger.success("ONNX model loaded successfully")

    # Validate on multiple batches
    logger.info(f"Validating on {num_batches} batches from test loader...")
    per_batch_metrics = []
    total_samples = 0
    batches_tested = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break

            # Extract images (batch can be (images,) or (images, labels))
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            batch_size = images.size(0)
            total_samples += batch_size
            batches_tested += 1

            # PyTorch inference
            images_device = images.to(device)
            pytorch_output = pytorch_model(images_device)
            pytorch_output_np = pytorch_output.cpu().numpy()

            # ONNX inference
            images_numpy = images.cpu().numpy()
            onnx_output = session.run(None, {input_name: images_numpy})
            onnx_output_np = onnx_output[0]

            # Compute metrics for this batch
            batch_metrics = compare_outputs(pytorch_output_np, onnx_output_np)
            batch_metrics["batch_idx"] = batch_idx
            batch_metrics["batch_size"] = batch_size
            per_batch_metrics.append(batch_metrics)

            logger.debug(
                f"Batch {batch_idx + 1}/{num_batches}: "
                f"max_diff={batch_metrics['max_diff']:.2e}, "
                f"mse={batch_metrics['mse']:.2e}, "
                f"cosine_sim={batch_metrics['cosine_similarity']:.6f}"
            )

    # Aggregate metrics across all batches
    logger.info("Aggregating metrics across all batches...")

    # Handle empty test loader case
    if len(per_batch_metrics) == 0:
        logger.warning("No batches were tested (empty test loader)")
        aggregated_metrics = {
            "max_diff": {"max": 0.0, "mean": 0.0, "min": 0.0},
            "mse": {"max": 0.0, "mean": 0.0, "min": 0.0},
            "mae": {"max": 0.0, "mean": 0.0, "min": 0.0},
            "cosine_similarity": {"max": 0.0, "mean": 0.0, "min": 0.0},
        }
        status = "PASS"  # Empty test loader is not an error, just no data
    else:
        all_max_diffs = [m["max_diff"] for m in per_batch_metrics]
        all_mses = [m["mse"] for m in per_batch_metrics]
        all_maes = [m["mae"] for m in per_batch_metrics]
        all_cosine_sims = [m["cosine_similarity"] for m in per_batch_metrics]

        aggregated_metrics = {
            "max_diff": {
                "max": float(np.max(all_max_diffs)),
                "mean": float(np.mean(all_max_diffs)),
                "min": float(np.min(all_max_diffs)),
            },
            "mse": {
                "max": float(np.max(all_mses)),
                "mean": float(np.mean(all_mses)),
                "min": float(np.min(all_mses)),
            },
            "mae": {
                "max": float(np.max(all_maes)),
                "mean": float(np.mean(all_maes)),
                "min": float(np.min(all_maes)),
            },
            "cosine_similarity": {
                "max": float(np.max(all_cosine_sims)),
                "mean": float(np.mean(all_cosine_sims)),
                "min": float(np.min(all_cosine_sims)),
            },
        }

        # Determine overall validation status
        max_of_max_diffs = aggregated_metrics["max_diff"]["max"]
        if max_of_max_diffs < 1e-5:
            status = "PASS"
        elif max_of_max_diffs < 1e-4:
            status = "WARN"
        else:
            status = "FAIL"

    # Print comprehensive report
    logger.info("=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    logger.info(f"Batches tested: {batches_tested}")
    logger.info(f"Total samples: {total_samples}")
    logger.info("")
    logger.info("Aggregated Metrics (across all batches):")
    logger.info("-" * 80)
    logger.info(
        f"Max Diff:          max={aggregated_metrics['max_diff']['max']:.2e}, "
        f"mean={aggregated_metrics['max_diff']['mean']:.2e}, "
        f"min={aggregated_metrics['max_diff']['min']:.2e}"
    )
    logger.info(
        f"MSE:               max={aggregated_metrics['mse']['max']:.2e}, "
        f"mean={aggregated_metrics['mse']['mean']:.2e}, "
        f"min={aggregated_metrics['mse']['min']:.2e}"
    )
    logger.info(
        f"MAE:               max={aggregated_metrics['mae']['max']:.2e}, "
        f"mean={aggregated_metrics['mae']['mean']:.2e}, "
        f"min={aggregated_metrics['mae']['min']:.2e}"
    )
    logger.info(
        f"Cosine Similarity: max={aggregated_metrics['cosine_similarity']['max']:.6f}, "
        f"mean={aggregated_metrics['cosine_similarity']['mean']:.6f}, "
        f"min={aggregated_metrics['cosine_similarity']['min']:.6f}"
    )
    logger.info("-" * 80)

    # Print final status
    if len(per_batch_metrics) == 0:
        logger.success("PASS - No batches to test (empty test loader)")
    else:
        max_of_max_diffs = aggregated_metrics["max_diff"]["max"]
        if status == "PASS":
            logger.success(
                f"PASS - ONNX model validated successfully "
                f"(max diff: {max_of_max_diffs:.2e})"
            )
        elif status == "WARN":
            logger.warning(
                f"WARN - ONNX model has minor differences "
                f"(max diff: {max_of_max_diffs:.2e})"
            )
        else:
            logger.error(
                f"FAIL - ONNX model validation failed "
                f"(max diff: {max_of_max_diffs:.2e})"
            )

    logger.info("=" * 80)

    return {
        "num_batches_tested": batches_tested,
        "total_samples": total_samples,
        "metrics": aggregated_metrics,
        "status": status,
        "per_batch_metrics": per_batch_metrics,
    }
