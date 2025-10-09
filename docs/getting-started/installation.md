# Installation Guide

Use this page as a lightweight companion to **[Workflow · Step 1](../workflow.md#step-1-install-dependencies)**, which remains the authoritative walkthrough. This guide summarizes requirements, gives a compact install recipe, and captures platform-specific notes and troubleshooting tips.

---

## Requirements at a Glance

| Component | Minimum | Recommended |
| --- | --- | --- |
| CPU | 2 cores | 4+ cores |
| RAM | 8 GB | 16 GB+ |
| Disk | 10 GB free | 50 GB+ (datasets) |
| GPU | Optional | NVIDIA 8 GB+ VRAM, CUDA 11+ |
| Python | 3.8+ | Latest 3.10/3.11 |

CPU-only training works everywhere; GPU support requires CUDA-capable hardware.

---

## Install in Three Steps

1. **Get the source** (clone or open your checkout)
   ```bash
   git clone <repo-url>
   cd gui
   ```
2. **Activate an environment (optional but recommended)**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate  # Linux/macOS
   # or: conda create -n ml-classifier python=3.10 && conda activate ml-classifier
   ```
3. **Install the package**
   ```bash
   uv pip install -e .
   ```

To include extras, add the suffixes from the workflow (e.g. `".[dev,optuna]"`). After installation the CLI commands (`ml-train`, `ml-inference`, etc.) are on your PATH.

---

## Quick Verification

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print('CUDA:', torch.cuda.is_available())"
ml-train --help
```

When a GPU is present, `nvidia-smi` should list it. If CUDA is unavailable, training will fall back to CPU.

---

## Platform Notes

- **Ubuntu/Debian:** `sudo apt-get install python3 python3-venv && curl -LsSf https://astral.sh/uv/install.sh | sh`
- **macOS:** Install Python via Homebrew; CUDA is not available, so run on CPU or external GPU resources.
- **Windows:** Use `python -m venv .venv` then `.venv\Scripts\activate`; `uv` installs via PowerShell (`irm https://astral.sh/uv/install.ps1 | iex`).

Docker or cloud setups follow the same installation command inside the container/VM. Mount your `data/` directory when running containers.

---

## Troubleshooting Highlights

- `uv: command not found` → install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`) or temporarily use `pip install uv`.
- Import errors for torch/torchvision → reinstall via the appropriate PyTorch wheel index (match CUDA version as per [PyTorch local install guide](https://pytorch.org/get-started/locally/)).
- CUDA mismatch → pick the wheel that corresponds to `nvidia-smi`’s CUDA version, or install the CPU-only wheel.
- Permission or disk issues → use `--user`, clean caches (`uv cache clean`), or install to a larger target (`--target /mnt/bigdisk`).

For a full setup sequence (including dataset prep and verification runs), continue with **[Workflow Step 2](../workflow.md#step-2-prepare-the-dataset)**.

### Update Specific Package

```bash
uv pip install --upgrade torch torchvision
```

### Check for Outdated Packages

```bash
uv pip list --outdated
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
- [ ] uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [ ] Virtual environment created and activated
- [ ] Package installed (`uv pip install -e .`)
- [ ] CLI commands available globally
- [ ] PyTorch imported successfully
- [ ] CUDA available (optional, for GPU)
- [ ] Test run completed

**You're ready to start training!** Proceed to [Data Preparation](data-preparation.md).
