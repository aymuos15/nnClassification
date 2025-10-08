"""ONNX model export and validation utilities."""

import os

import numpy as np
import torch
from loguru import logger

from ml_src.core.network import get_model


def validate_onnx_export(pytorch_model, onnx_path, dummy_input, device='cpu'):
    """
    Validate ONNX export by comparing PyTorch and ONNX model outputs.

    Performs inference on both PyTorch and ONNX models with the same input,
    then compares outputs using multiple metrics to ensure the export was successful.

    Args:
        pytorch_model: PyTorch model to validate against
        onnx_path: Path to the exported ONNX model file
        dummy_input: Sample input tensor for inference
        device: Device where the PyTorch model resides ('cpu' or 'cuda')

    Returns:
        dict: Validation metrics containing:
            - max_diff: Maximum absolute difference between outputs
            - mse: Mean squared error between outputs
            - mae: Mean absolute error between outputs
            - cosine_similarity: Cosine similarity between flattened outputs
            - status: Validation status ('PASS', 'WARN', or 'FAIL')

    Example:
        >>> model = resnet18(pretrained=True)
        >>> dummy_input = torch.randn(1, 3, 224, 224)
        >>> torch.onnx.export(model, dummy_input, "model.onnx")
        >>> metrics = validate_onnx_export(model, "model.onnx", dummy_input, "cpu")
        >>> print(f"Status: {metrics['status']}, Max difference: {metrics['max_diff']}")
    """
    try:
        import onnxruntime
    except ImportError:
        logger.error("onnxruntime is not installed. Install it with: pip install onnxruntime")
        return {
            'max_diff': float('inf'),
            'mse': float('inf'),
            'mae': float('inf'),
            'cosine_similarity': 0.0,
            'status': 'FAIL'
        }

    try:
        logger.info(f"Validating ONNX export: {onnx_path}")

        # Step 1: Run PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_out = pytorch_model(dummy_input.to(device))

        # Convert to numpy
        pytorch_out_np = pytorch_out.cpu().numpy()

        # Step 2: Run ONNX inference
        session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        # Get input name from ONNX model
        input_name = session.get_inputs()[0].name

        # Run ONNX inference
        onnx_out = session.run(None, {input_name: dummy_input.numpy()})
        onnx_out_np = onnx_out[0]

        # Step 3: Compute metrics
        max_diff = np.abs(pytorch_out_np - onnx_out_np).max()
        mse = np.mean((pytorch_out_np - onnx_out_np) ** 2)
        mae = np.mean(np.abs(pytorch_out_np - onnx_out_np))

        # Compute cosine similarity
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            cosine_sim = cosine_similarity(
                pytorch_out_np.flatten().reshape(1, -1),
                onnx_out_np.flatten().reshape(1, -1)
            )[0][0]
        except ImportError:
            # Fallback if sklearn is not available
            pytorch_flat = pytorch_out_np.flatten()
            onnx_flat = onnx_out_np.flatten()
            dot_product = np.dot(pytorch_flat, onnx_flat)
            pytorch_norm = np.linalg.norm(pytorch_flat)
            onnx_norm = np.linalg.norm(onnx_flat)
            if pytorch_norm > 0 and onnx_norm > 0:
                cosine_sim = dot_product / (pytorch_norm * onnx_norm)
            else:
                cosine_sim = 0.0

        # Determine validation status
        if max_diff < 1e-5:
            status = 'PASS'
        elif max_diff < 1e-4:
            status = 'WARN'
        else:
            status = 'FAIL'

        # Step 4: Log validation results
        logger.info(f"Validation metrics:")
        logger.info(f"  Max difference: {max_diff:.2e}")
        logger.info(f"  MSE: {mse:.2e}")
        logger.info(f"  MAE: {mae:.2e}")
        logger.info(f"  Cosine similarity: {cosine_sim:.6f}")

        if status == 'PASS':
            logger.success(f"PASS - ONNX export validation successful (max_diff: {max_diff:.2e})")
        elif status == 'WARN':
            logger.warning(f"WARN - ONNX export has minor differences (max_diff: {max_diff:.2e})")
        else:
            logger.error(f"FAIL - ONNX export validation failed (max_diff: {max_diff:.2e})")

        return {
            'max_diff': float(max_diff),
            'mse': float(mse),
            'mae': float(mae),
            'cosine_similarity': float(cosine_sim),
            'status': status
        }

    except Exception as e:
        logger.error(f"Error during ONNX validation: {str(e)}")
        return {
            'max_diff': float('inf'),
            'mse': float('inf'),
            'mae': float('inf'),
            'cosine_similarity': 0.0,
            'status': 'FAIL'
        }


def export_to_onnx(
    checkpoint_path,
    output_path,
    opset_version=17,
    input_size=None,
    use_ema: bool = False,
):
    """
    Export a PyTorch model checkpoint to ONNX format.

    Loads a saved checkpoint, recreates the model architecture, and exports it to ONNX
    format with dynamic batch size support. Automatically determines appropriate input
    dimensions based on the model architecture if not explicitly provided.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file (.pt)
        output_path: Path where the ONNX model will be saved (.onnx)
        opset_version: ONNX opset version to use (default: 17)
        input_size: Tuple of input dimensions (batch, channels, height, width).
                   If None, automatically determined based on architecture:
                   - Inception models: (1, 3, 299, 299)
                   - Other models: (1, 3, 224, 224)

    Returns:
        tuple: (success: bool, message: str)
            - success: True if export succeeded, False otherwise
            - message: Descriptive message about the export result

    Example:
        >>> # Export with auto-detected input size
        >>> success, msg = export_to_onnx(
        ...     'runs/my_run/weights/best.pt',
        ...     'models/my_model.onnx'
        ... )
        >>> print(msg)

        >>> # Export with custom input size
        >>> success, msg = export_to_onnx(
        ...     'runs/my_run/weights/best.pt',
        ...     'models/my_model.onnx',
        ...     input_size=(1, 3, 320, 320)
        ... )
    """
    try:
        # Step 1: Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            error_msg = f"Checkpoint file not found: {checkpoint_path}"
            logger.error(error_msg)
            return False, error_msg

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Step 2: Extract model state dict and config
        if "model_state_dict" not in checkpoint:
            error_msg = (
                "Invalid checkpoint format: missing 'model_state_dict'. "
                "Expected a checkpoint saved by save_checkpoint()."
            )
            logger.error(error_msg)
            return False, error_msg

        config = checkpoint.get("config", {})
        if not config:
            error_msg = "No configuration found in checkpoint. Cannot recreate model architecture."
            logger.error(error_msg)
            return False, error_msg

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
                logger.info("Exporting EMA weights from checkpoint {}", checkpoint_path)

        logger.info("Checkpoint loaded successfully")

        # Step 3: Recreate model architecture
        logger.info("Recreating model architecture from config")
        model = get_model(config, device="cpu")
        model.load_state_dict(model_state)
        model.eval()
        logger.success("Model architecture recreated and weights loaded")

        # Step 4: Determine input size
        if input_size is None:
            # Extract architecture name from config
            model_config = config.get("model", {})
            architecture = model_config.get("architecture", "").lower()
            custom_architecture = model_config.get("custom_architecture", "").lower()

            # Combine both architecture fields for checking
            arch_name = architecture or custom_architecture

            # Inception models require 299x299 input
            if "inception" in arch_name:
                input_size = (1, 3, 299, 299)
                logger.info(f"Detected Inception architecture, using input size: {input_size}")
            else:
                # Default to 224x224 for most models
                input_size = (1, 3, 224, 224)
                logger.info(f"Using default input size: {input_size}")
        else:
            logger.info(f"Using provided input size: {input_size}")

        # Step 5: Create dummy input
        dummy_input = torch.randn(input_size)
        logger.debug(f"Created dummy input with shape: {dummy_input.shape}")

        # Step 6: Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")

        # Step 7: Export to ONNX
        logger.info(f"Exporting model to ONNX format (opset version: {opset_version})")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.success(f"Model exported to {output_path}")

        # Step 8: Add metadata if onnx library is available
        try:
            import onnx

            logger.info("Adding metadata to ONNX model")
            onnx_model = onnx.load(output_path)

            # Add metadata
            meta = onnx_model.metadata_props.add()
            meta.key = "architecture"
            meta.value = str(
                config.get("model", {}).get("architecture")
                or config.get("model", {}).get("custom_architecture", "unknown")
            )

            meta = onnx_model.metadata_props.add()
            meta.key = "num_classes"
            meta.value = str(config.get("model", {}).get("num_classes", "unknown"))

            meta = onnx_model.metadata_props.add()
            meta.key = "input_size"
            meta.value = str(input_size)

            meta = onnx_model.metadata_props.add()
            meta.key = "opset_version"
            meta.value = str(opset_version)

            # Add training metadata if available
            if "best_acc" in checkpoint:
                meta = onnx_model.metadata_props.add()
                meta.key = "best_accuracy"
                meta.value = str(checkpoint["best_acc"])

            if "epoch" in checkpoint:
                meta = onnx_model.metadata_props.add()
                meta.key = "trained_epochs"
                meta.value = str(checkpoint["epoch"])

            # Save with metadata
            onnx.save(onnx_model, output_path)
            logger.success("Metadata added to ONNX model")

        except ImportError:
            logger.warning(
                "onnx library not available. Skipping metadata addition. "
                "Install with: pip install onnx"
            )
        except Exception as e:
            logger.warning(f"Failed to add metadata to ONNX model: {e}")

        # Step 9: Verify the export
        logger.info("Verifying ONNX export")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        success_msg = (
            f"ONNX export successful! "
            f"Model saved to {output_path} ({file_size_mb:.2f} MB)"
        )
        logger.success(success_msg)

        return True, success_msg

    except Exception as e:
        error_msg = f"ONNX export failed: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return False, error_msg
