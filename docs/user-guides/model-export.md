# Model Export Guide

Guide to exporting trained models to ONNX format for deployment and cross-platform inference.

## Why Export Models?

Exporting PyTorch models to ONNX format provides several key benefits:

### Deployment Flexibility
- **Production environments**: Deploy to web servers, mobile devices, and edge hardware
- **Framework-agnostic**: Use models in C++, Java, JavaScript, and other languages
- **Cloud services**: Compatible with Azure ML, AWS SageMaker, Google Cloud AI

### Performance Benefits
- **Optimized runtimes**: ONNX Runtime provides hardware-specific optimizations
- **Reduced latency**: Pre-compiled models execute faster than interpreted PyTorch
- **Smaller footprint**: Deployment doesn't require full PyTorch installation

### Cross-Platform Compatibility
- **Windows, Linux, macOS**: Single model file works everywhere
- **Mobile**: iOS (Core ML) and Android (TensorFlow Lite via conversion)
- **Web browsers**: Run models client-side with ONNX.js

---

## What is ONNX?

**ONNX (Open Neural Network Exchange)** is an open-source format for representing machine learning models. It provides a standard interface that allows models to move between different frameworks and deployment environments.

### Industry Support
ONNX is backed by major technology companies:
- **Microsoft** - Azure ML, Windows ML
- **Facebook/Meta** - PyTorch integration
- **Amazon** - SageMaker Neo
- **NVIDIA** - TensorRT optimization
- **Intel** - OpenVINO toolkit

### Key Features
- **Interoperability**: Convert between PyTorch, TensorFlow, scikit-learn, and more
- **Hardware acceleration**: Optimized for CPUs, GPUs, NPUs, and custom accelerators
- **Version stability**: Models remain compatible across ONNX Runtime updates
- **Operator coverage**: Supports 100+ standard ML operations

---

## Supported Models

### Base Models (Torchvision)

All torchvision models are fully supported for ONNX export:

**Tested architectures:**
- **ResNet family**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`
- **MobileNet**: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **Vision Transformer**: `vit_b_16`, `vit_b_32`, `vit_l_16`
- **ConvNeXt**: `convnext_tiny`, `convnext_small`, `convnext_base`

!!!tip "Recommendation for Deployment"
    For production deployment, we recommend:
    - **Edge devices**: `mobilenet_v3_small` or `efficientnet_b0` (best speed/size)
    - **Servers**: `resnet50` or `efficientnet_b2` (best accuracy/speed balance)
    - **High accuracy**: `convnext_base` or `vit_b_16` (best accuracy)

### Custom Models

Custom models have experimental ONNX support:

**Fully supported:**
- `simple_cnn` - Basic CNN architecture
- `tiny_net` - Lightweight model for testing

**Important notes:**
- Custom models must use standard PyTorch operations
- Avoid custom CUDA kernels or unsupported operations
- Test validation thoroughly before deployment

!!!warning "Custom Model Validation"
    Always validate custom model exports carefully. Check that:
    - Export completes without errors
    - Validation metrics show high cosine similarity (>0.999)
    - Test predictions match PyTorch outputs

---

## Usage

### Basic Export

Export a single trained model to ONNX format:

```bash
# Export best checkpoint from a run
ml-export --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt

# Specify custom output path
ml-export \
  --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt \
  --output_path models/hymenoptera_production.onnx

# Export with custom batch size
ml-export \
  --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt \
  --batch_size 8
```

**Output:**
```
2025-10-06 10:30:00 | INFO     | Loading checkpoint from runs/hymenoptera_base_fold_0/weights/best.pt
2025-10-06 10:30:01 | INFO     | Creating model: resnet18 (base)
2025-10-06 10:30:01 | INFO     | Model loaded successfully
2025-10-06 10:30:01 | INFO     | Exporting to ONNX format...
2025-10-06 10:30:02 | SUCCESS  | ✓ Model exported to: runs/hymenoptera_base_fold_0/weights/best.onnx
2025-10-06 10:30:02 | INFO     | Validating ONNX export...
2025-10-06 10:30:03 | INFO     | Validation metrics:
2025-10-06 10:30:03 | INFO     |   Max difference: 1.19e-07
2025-10-06 10:30:03 | INFO     |   MSE: 2.45e-15
2025-10-06 10:30:03 | INFO     |   Cosine similarity: 0.999999
2025-10-06 10:30:03 | SUCCESS  | ✓ PASS - ONNX export validation successful
```

### Batch Export

Export multiple checkpoints using glob patterns:

```bash
# Export all best checkpoints across folds
ml-export --pattern "runs/hymenoptera_*/weights/best.pt"

# Export all checkpoints (best and last) from a specific run
ml-export --pattern "runs/hymenoptera_base_fold_0/weights/*.pt"

# Export from multiple experiments
ml-export --pattern "runs/experiment_*/weights/best.pt"
```

**Output:**
```
2025-10-06 10:35:00 | INFO     | Found 5 checkpoints matching pattern
2025-10-06 10:35:00 | INFO     | Exporting checkpoint 1/5: runs/hymenoptera_base_fold_0/weights/best.pt
2025-10-06 10:35:02 | SUCCESS  | ✓ Exported to runs/hymenoptera_base_fold_0/weights/best.onnx
2025-10-06 10:35:02 | INFO     | Exporting checkpoint 2/5: runs/hymenoptera_base_fold_1/weights/best.pt
2025-10-06 10:35:04 | SUCCESS  | ✓ Exported to runs/hymenoptera_base_fold_1/weights/best.onnx
...
2025-10-06 10:35:15 | SUCCESS  | ✓ Successfully exported 5/5 models
```

### Validation Only

Validate an existing ONNX export without re-exporting:

```bash
# Validate ONNX model against PyTorch checkpoint
ml-export \
  --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt \
  --onnx_path runs/hymenoptera_base_fold_0/weights/best.onnx \
  --validate_only
```

**Use cases:**
- Verify ONNX model after manual modification
- Re-check validation after ONNX Runtime update
- Confirm export integrity before deployment

### Export Options

**Complete CLI reference:**

```bash
ml-export [OPTIONS]

Required (one of):
  --checkpoint_path PATH      Path to PyTorch checkpoint (.pt file)
  --pattern GLOB              Glob pattern to match multiple checkpoints

Optional:
  --output_path PATH          Custom output path for ONNX model
                              (default: same directory as checkpoint, .onnx extension)

  --batch_size INT            Batch size for export (default: 1)
                              Use higher values if deploying with batching

  --opset_version INT         ONNX opset version (default: 14)
                              Use 11+ for broad compatibility

  --validate                  Run validation after export (default: True)
  --no-validate               Skip validation (faster, use for debugging)

  --validate_only             Only validate existing ONNX file, don't export
  --onnx_path PATH            Path to ONNX model for validation

  --dynamic_axes              Enable dynamic batch/sequence dimensions
                              Required for variable-size inputs

  --verbose                   Show detailed ONNX export logs
```

---

## Deployment Examples

### Python with ONNX Runtime

Use exported models in Python without PyTorch:

```python
import numpy as np
import onnxruntime
from PIL import Image
from torchvision import transforms

# Load ONNX model
session = onnxruntime.InferenceSession(
    "runs/hymenoptera_base_fold_0/weights/best.onnx",
    providers=["CPUExecutionProvider"]  # or ["CUDAExecutionProvider"]
)

# Prepare image (same transforms as training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("test_image.jpg")
input_tensor = transform(image).unsqueeze(0).numpy()

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})[0]

# Get prediction
predicted_class = np.argmax(output, axis=1)[0]
confidence = np.max(output, axis=1)[0]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.4f}")
```

**Performance comparison** (ResNet18, single image, CPU):
- PyTorch: ~45ms
- ONNX Runtime: ~28ms (1.6x faster)

### C++ Inference

Pseudo-code for C++ deployment:

```cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ImageClassifier");
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(4);

// Load model
Ort::Session session(env, "model.onnx", session_options);

// Prepare input (OpenCV)
cv::Mat image = cv::imread("test_image.jpg");
cv::resize(image, image, cv::Size(224, 224));
// ... normalize and convert to float array

// Create input tensor
std::vector<int64_t> input_shape = {1, 3, 224, 224};
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_data, input_size,
    input_shape.data(), input_shape.size()
);

// Run inference
auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names.data(), &input_tensor, 1,
    output_names.data(), 1
);

// Process output
float* output_data = output_tensors[0].GetTensorMutableData<float>();
int predicted_class = std::max_element(output_data, output_data + num_classes)
                      - output_data;
```

**Benefits:**
- No Python runtime required
- Lower memory footprint (~10MB vs 500MB+ for PyTorch)
- Faster startup time

### Web Deployment

Deploy models to web browsers with ONNX.js:

```javascript
// Load ONNX model in browser
const session = await ort.InferenceSession.create('model.onnx');

// Prepare image (using Canvas API)
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.drawImage(image, 0, 0, 224, 224);
const imageData = ctx.getImageData(0, 0, 224, 224);

// Preprocess (normalize)
const input = new Float32Array(3 * 224 * 224);
// ... normalize pixel values

// Run inference
const inputTensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
const outputs = await session.run({ 'input': inputTensor });

// Get prediction
const predictions = outputs.output.data;
const predictedClass = predictions.indexOf(Math.max(...predictions));
```

**Use cases:**
- Client-side image classification
- Privacy-preserving inference (data never leaves browser)
- Offline web applications

!!!tip "Model Size for Web"
    For web deployment, use lightweight models:
    - MobileNetV3-Small: ~5MB
    - EfficientNet-B0: ~20MB
    - ResNet18: ~45MB

    Larger models increase page load time significantly.

---

## Validation

ONNX export validation ensures the exported model produces identical results to the original PyTorch model.

### Validation Metrics

The framework computes three metrics:

#### Maximum Absolute Difference
```
max_diff = max(|pytorch_output - onnx_output|)
```

**Interpretation:**
- `< 1e-5`: Excellent - Export is essentially perfect
- `1e-5 to 1e-4`: Good - Minor floating-point differences (acceptable)
- `> 1e-4`: Poor - Significant differences, investigate

#### Mean Squared Error (MSE)
```
mse = mean((pytorch_output - onnx_output)²)
```

**Interpretation:**
- `< 1e-10`: Excellent
- `1e-10 to 1e-8`: Good
- `> 1e-8`: Poor

#### Cosine Similarity
```
cosine_sim = dot(pytorch, onnx) / (||pytorch|| * ||onnx||)
```

**Interpretation:**
- `> 0.9999`: Excellent - Outputs are nearly identical
- `0.999 to 0.9999`: Good - Very similar
- `< 0.999`: Poor - Outputs diverge significantly

### Example Validation Output

**Successful validation:**
```
2025-10-06 10:30:03 | INFO     | Validation metrics:
2025-10-06 10:30:03 | INFO     |   Max difference: 1.19e-07
2025-10-06 10:30:03 | INFO     |   MSE: 2.45e-15
2025-10-06 10:30:03 | INFO     |   Cosine similarity: 0.999999
2025-10-06 10:30:03 | SUCCESS  | ✓ PASS - ONNX export validation successful
```
✓ Safe to deploy

**Warning (acceptable):**
```
2025-10-06 10:30:03 | INFO     | Validation metrics:
2025-10-06 10:30:03 | INFO     |   Max difference: 5.32e-05
2025-10-06 10:30:03 | INFO     |   MSE: 1.87e-09
2025-10-06 10:30:03 | INFO     |   Cosine similarity: 0.999984
2025-10-06 10:30:03 | WARNING  | ⚠ WARN - ONNX export has minor differences
```
⚠ Minor floating-point differences, usually acceptable

**Failed validation:**
```
2025-10-06 10:30:03 | INFO     | Validation metrics:
2025-10-06 10:30:03 | INFO     |   Max difference: 2.15e-02
2025-10-06 10:30:03 | INFO     |   MSE: 3.45e-04
2025-10-06 10:30:03 | INFO     |   Cosine similarity: 0.987234
2025-10-06 10:30:03 | ERROR    | ✗ FAIL - ONNX export validation failed
```
✗ Do not deploy - investigate issues

### When Validation Fails

If validation shows large differences:

1. **Check for unsupported operations** - Some PyTorch ops don't translate perfectly
2. **Try different opset version** - Older/newer opsets may work better
3. **Verify custom layers** - Custom models may need modification
4. **Test with different batch sizes** - Some operations behave differently
5. **Check for randomness** - Ensure model is in eval mode (dropout disabled)

---

## Troubleshooting

### Unsupported Operations

**Error:**
```
RuntimeError: ONNX export failed: Unsupported operator 'aten::some_op'
```

**Solutions:**
1. **Update ONNX opset version:**
   ```bash
   ml-export --checkpoint_path model.pt --opset_version 15
   ```

2. **Replace unsupported operations** in custom models:
   - Use standard PyTorch operations when possible
   - Avoid `torch.jit.script` decorators that may conflict
   - Check [ONNX operator compatibility](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

3. **Simplify model architecture:**
   - Remove complex custom layers
   - Replace dynamic control flow with static operations

### Dynamic Shape Issues

**Error:**
```
RuntimeError: Exporting model with dynamic axes failed
```

**Solution:**
```bash
# Export with dynamic batch dimension
ml-export \
  --checkpoint_path model.pt \
  --dynamic_axes \
  --batch_size 1
```

**When dynamic shapes are needed:**
- Variable batch sizes during inference
- Sequence models with variable length
- Multi-scale image processing

**When to avoid:**
- Fixed batch size known at deployment time
- Maximum performance required (static shapes optimize better)

### Validation Failures

**Symptom:** High max_diff or low cosine similarity

**Common causes:**

1. **Model not in eval mode** - Dropout/BatchNorm behaving differently
   ```python
   # Framework automatically handles this, but if implementing manually:
   model.eval()
   ```

2. **Randomness in model** - Stochastic operations during inference
   ```python
   # Check for dropout in eval mode
   # Remove stochastic operations from inference path
   ```

3. **Precision differences** - Different floating-point behavior
   ```bash
   # Try different validation tolerance
   # Minor differences (<1e-4) are usually acceptable
   ```

4. **Input preprocessing mismatch** - Ensure same normalization
   ```python
   # Verify transforms match training exactly
   # Check mean/std values
   ```

### Large Model Size

**Issue:** ONNX file is very large (>100MB)

**Solutions:**

1. **Use quantization** (post-export):
   ```python
   from onnxruntime.quantization import quantize_dynamic

   quantize_dynamic(
       "model.onnx",
       "model_quantized.onnx",
       weight_type=QuantType.QUInt8
   )
   ```
   Reduces size by ~75% with minimal accuracy loss

2. **Choose smaller architecture:**
   - Switch from ResNet50 → ResNet18
   - Use EfficientNet-B0 instead of B3
   - Try MobileNetV3 for edge deployment

3. **Remove unnecessary components:**
   - Export only the trained model, not optimizer state
   - Framework already does this, but verify checkpoint is clean

### Performance Issues

**Symptom:** ONNX inference slower than PyTorch

**Diagnostic steps:**

1. **Check execution provider:**
   ```python
   # Use GPU if available
   providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
   session = onnxruntime.InferenceSession("model.onnx", providers=providers)
   ```

2. **Enable optimization:**
   ```python
   import onnxruntime as ort

   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   session = ort.InferenceSession("model.onnx", sess_options, providers=providers)
   ```

3. **Batch inference:**
   ```bash
   # Export with larger batch size
   ml-export --checkpoint_path model.pt --batch_size 8
   ```

4. **Hardware-specific optimization:**
   - Use TensorRT for NVIDIA GPUs
   - Use OpenVINO for Intel CPUs
   - Use Core ML for Apple devices

---

## Best Practices

### Before Export

1. ✅ **Train model completely** - Export best.pt, not intermediate checkpoints
2. ✅ **Validate on test set** - Ensure model performs well before export
3. ✅ **Document architecture** - Record model type, hyperparameters, training config
4. ✅ **Choose appropriate opset** - Use ONNX opset 11+ for broad compatibility

### During Export

1. ✅ **Match deployment batch size** - Export with expected inference batch size
2. ✅ **Enable validation** - Always validate unless debugging
3. ✅ **Use verbose mode for debugging** - `--verbose` shows ONNX export details
4. ✅ **Export all folds** - Keep ONNX versions of all cross-validation models

### After Export

1. ✅ **Test in deployment environment** - Verify ONNX model works on target platform
2. ✅ **Measure performance** - Benchmark inference speed in production
3. ✅ **Version control** - Track ONNX files with git-lfs or separate storage
4. ✅ **Document deployment** - Record runtime versions, dependencies, hardware

### Production Deployment

1. ✅ **Keep PyTorch checkpoint** - Archive original .pt file
2. ✅ **Monitor inference** - Track latency and accuracy in production
3. ✅ **Test fallback** - Have rollback plan if ONNX model fails
4. ✅ **Update cautiously** - Test new ONNX Runtime versions thoroughly

---

## Workflow Example

### Complete Export-to-Deployment Pipeline

```bash
# 1. Train model (already completed)
ml-train --config configs/hymenoptera_config.yaml --fold 0

# 2. Validate training results
ml-inference --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt

# 3. Export to ONNX
ml-export \
  --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt \
  --batch_size 4 \
  --verbose

# 4. Verify export
ml-export \
  --checkpoint_path runs/hymenoptera_base_fold_0/weights/best.pt \
  --onnx_path runs/hymenoptera_base_fold_0/weights/best.onnx \
  --validate_only

# 5. Copy to deployment location
mkdir -p deployment/models
cp runs/hymenoptera_base_fold_0/weights/best.onnx deployment/models/hymenoptera_v1.onnx
cp runs/hymenoptera_base_fold_0/config.yaml deployment/models/hymenoptera_v1_config.yaml

# 6. Document deployment
echo "Model: ResNet18" > deployment/models/README.txt
echo "Trained: 2025-10-06" >> deployment/models/README.txt
echo "Test Accuracy: 91.5%" >> deployment/models/README.txt
echo "ONNX Opset: 14" >> deployment/models/README.txt
```

### Cross-Validation Export

Export best models from all folds:

```bash
# Export all folds
for fold in {0..4}; do
  ml-export --checkpoint_path runs/hymenoptera_base_fold_$fold/weights/best.pt
done

# Or use batch export
ml-export --pattern "runs/hymenoptera_base_fold_*/weights/best.pt"

# Choose best fold for deployment
# (based on validation metrics from training)
cp runs/hymenoptera_base_fold_2/weights/best.onnx deployment/production_model.onnx
```

---

## Related Guides

- [Inference Guide](inference.md) - Test models before export
- [Training Guide](training.md) - Train models for export
- [Advanced Training Guide](advanced-training.md) - Optimize model performance
- [Monitoring Guide](monitoring.md) - Verify model quality
- [Troubleshooting](../reference/troubleshooting.md) - Common issues

---

## Summary

**You've learned:**
- ✅ Why ONNX export is valuable for deployment
- ✅ How to export PyTorch models to ONNX format
- ✅ Validation metrics and their interpretation
- ✅ Deployment examples in Python, C++, and web
- ✅ Troubleshooting export and validation issues
- ✅ Best practices for production deployment

**Ready to export?**
```bash
# Export your trained model
ml-export --checkpoint_path runs/my_model_fold_0/weights/best.pt

# Validate the export
ml-export \
  --checkpoint_path runs/my_model_fold_0/weights/best.pt \
  --validate_only
```
