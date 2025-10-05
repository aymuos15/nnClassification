# Installation Guide

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 2+ cores
- RAM: 8GB
- Disk: 10GB free space
- GPU: Optional (CPU training supported)

**Recommended:**
- CPU: 4+ cores
- RAM: 16GB+
- Disk: 50GB+ free space (for datasets)
- GPU: NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 2060, or better)
- CUDA 11.0+ compatible

### Software Requirements

- **Python:** 3.8 or higher
- **Operating System:** Linux, macOS, or Windows
- **CUDA:** 11.0+ (optional, for GPU training)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd gui
```

Or if you already have the code:
```bash
cd gui
```

### 2. Create Virtual Environment (Recommended)

**Using venv (Python built-in):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate  # On Windows
```

**Using conda:**
```bash
conda create -n pytorch-classifier python=3.10
conda activate pytorch-classifier
```

### 3. Install Dependencies

```bash
pip install -e .
```

This installs the package in editable mode along with all required dependencies including:
- PyTorch and torchvision
- TensorBoard
- NumPy, Pillow
- matplotlib, seaborn
- scikit-learn
- PyYAML, loguru, rich

**Note:** After installation, CLI commands (`ml-train`, `ml-inference`, `ml-split`, `ml-visualise`) are available globally in your environment.

---

## Verify Installation

### Check Python Version

```bash
python --version
# Should show: Python 3.8+ or higher
```

### Check PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Check CUDA Availability (GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

**Expected output (if GPU available):**
```
CUDA available: True
CUDA version: 11.8
GPU count: 1
```

**If no GPU:**
```
CUDA available: False
```
This is fine—CPU training is supported.

### Run Quick Test

```bash
# Verify all imports work
python -c "
import torch
import torchvision
import tensorboard
import yaml
import numpy as np
import matplotlib
import seaborn
import sklearn
print('All imports successful!')
"
```

---

## GPU Setup (Optional but Recommended)

### NVIDIA Driver

Check if NVIDIA driver is installed:
```bash
nvidia-smi
```

You should see GPU information. If not, install NVIDIA drivers:
- **Ubuntu/Debian:** `sudo apt-get install nvidia-driver-XXX`
- **Windows:** Download from [NVIDIA website](https://www.nvidia.com/download/index.aspx)

### CUDA Toolkit

PyTorch comes with CUDA bundled, so separate CUDA installation is usually not needed. However, for development:

**Check CUDA version:**
```bash
nvcc --version
```

**Install CUDA (if needed):**
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Follow platform-specific instructions

### cuDNN (Usually Not Needed)

PyTorch bundles cuDNN. If you need a specific version:
- Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- Follow installation instructions

---

## Troubleshooting Installation

### Issue: "pip: command not found"

**Solution:**
```bash
# Try pip3
pip3 install -e .

# Or use python -m pip
python -m pip install -e .
python3 -m pip install -e .
```

### Issue: "torch not found" or Import Error

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision

# Or install specific version (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA Version Mismatch

**Problem:** "CUDA version mismatch" error

**Solution:**
```bash
# Find your CUDA version
nvidia-smi  # Look for "CUDA Version" in top right

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for all options.

### Issue: Permission Denied

**Solution:**
```bash
# Use --user flag
pip install -e . --user

# Or use sudo (not recommended)
sudo pip install -e .
```

### Issue: Out of Disk Space

**Solution:**
```bash
# Check disk space
df -h

# Clear pip cache
pip cache purge

# Install to different location
pip install -e . --target /path/to/large/disk
```

### Issue: Slow Installation

**Solution:**
```bash
# Use faster mirror (example: Aliyun for China)
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

# Or skip dependencies if already installed
pip install -e . --no-deps
```

---

## Platform-Specific Notes

### Ubuntu/Debian Linux

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Install project
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### macOS

```bash
# Install Python via Homebrew
brew install python

# Install project
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

**Note:** macOS doesn't have CUDA support. Use CPU or Google Colab for GPU.

### Windows

```powershell
# Install Python from python.org

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -e .
```

**Note:** Windows paths use backslashes (`\`). Adjust paths in configs accordingly.

---

## Docker Installation (Alternative)

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY . .

# Install package
RUN pip install -e .

# Set entrypoint
ENTRYPOINT ["ml-train"]
```

### Build and Run

```bash
# Build image
docker build -t pytorch-classifier .

# Run training
docker run --gpus all -v $(pwd)/data:/workspace/data pytorch-classifier

# Interactive session
docker run --gpus all -it pytorch-classifier /bin/bash
```

---

## Cloud Platform Setup

### Google Colab

1. Upload code to Google Drive
2. Open Colab notebook
3. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Install dependencies:
   ```bash
   !pip install -e .
   ```
5. Run training:
   ```bash
   !ml-train
   ```

### AWS EC2

```bash
# Launch Deep Learning AMI (includes PyTorch, CUDA)
# SSH into instance

# Clone repository
git clone <repo-url>
cd gui

# Activate PyTorch environment (already installed on DL AMI)
source activate pytorch

# Install additional dependencies
pip install -e .

# Run training
ml-train
```

### Google Cloud Platform

```bash
# Create Compute Engine instance with GPU
# Use Deep Learning VM image

# SSH into instance
gcloud compute ssh instance-name

# Clone and install
git clone <repo-url>
cd gui
pip install -e .

# Run training
ml-train
```

---

## Updating Dependencies

### Update All Packages

```bash
pip install --upgrade -e .
```

### Update Specific Package

```bash
pip install --upgrade torch torchvision
```

### Check for Outdated Packages

```bash
pip list --outdated
```

---

## Next Steps

After installation:

1. **Verify installation works:**
   ```bash
   ml-train --num_epochs 1 --batch_size 2
   ```

2. **Prepare your data:**
   - See [Data Preparation Guide](data-preparation.md)

3. **Try a quick start:**
   - See [Quick Start Guide](quick-start.md)

4. **Explore configuration:**
   - See [Configuration Documentation](../configuration/README.md)

---

## Getting Help

### Check System Info

```bash
python -c "
import sys, torch, torchvision
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Save this output when reporting issues.

### Common Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Project Issues](../reference/troubleshooting.md)

---

## Summary

✅ **Installation checklist:**
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Package installed (`pip install -e .`)
- [ ] CLI commands available globally
- [ ] PyTorch imported successfully
- [ ] CUDA available (optional, for GPU)
- [ ] Test run completed

**You're ready to start training!** Proceed to [Data Preparation](data-preparation.md).
