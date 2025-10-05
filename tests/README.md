# Test Suite Organization

This directory contains comprehensive tests for the ML classification framework.

## Structure

```
tests/
├── conftest.py                      # Shared pytest fixtures
├── README.md                        # This file
│
├── network/                         # Network/architecture tests (201 tests)
│   ├── __init__.py
│   └── test_architectures.py       # Model loading, instantiation, forward pass tests
│
├── trainers/                        # Trainer-related tests (16 tests)
│   ├── __init__.py
│   ├── unit/                       # Unit tests for individual trainers (5 tests)
│   │   ├── __init__.py
│   │   ├── test_mixed_precision.py       # MixedPrecisionTrainer tests
│   │   ├── test_accelerate.py            # AccelerateTrainer tests
│   │   └── test_differential_privacy.py  # DPTrainer tests
│   │
│   └── integration/                # Integration tests (11 tests)
│       ├── __init__.py
│       ├── test_factory.py         # Trainer factory tests
│       └── test_integration.py     # Cross-trainer integration tests
│
└── inference/                       # Inference strategy tests (17 tests)
    ├── __init__.py
    └── test_strategies.py          # StandardInference, MixedPrecisionInference, AccelerateInference
```

## Running Tests

### Run All Tests (234 tests)
```bash
pytest tests/
```

### Run by Category

**Network tests only** (201 tests):
```bash
pytest tests/network/
```

**All trainer tests** (16 tests):
```bash
pytest tests/trainers/
```

**Trainer unit tests only** (5 tests):
```bash
pytest tests/trainers/unit/
```

**Trainer integration tests only** (11 tests):
```bash
pytest tests/trainers/integration/
```

**Inference tests only** (17 tests):
```bash
pytest tests/inference/
```

### Run Specific Test Files

```bash
# Run only mixed precision trainer tests
pytest tests/trainers/unit/test_mixed_precision.py

# Run only factory tests
pytest tests/trainers/integration/test_factory.py

# Run only architecture tests
pytest tests/network/test_architectures.py
```

### Run Specific Tests

```bash
# Run a specific test function
pytest tests/trainers/unit/test_mixed_precision.py::test_mixed_precision_trainer_cuda

# Run tests matching a pattern
pytest tests/ -k "mixed_precision"
```

## Test Coverage by Category

### Network Tests (201 tests)
- **70+ torchvision architectures**: ResNet, VGG, EfficientNet, MobileNet, ViT, Swin, ConvNeXt, DenseNet, RegNet, MNASNet, ShuffleNet, and more
- **What's tested**:
  - ✅ Model instantiation via `get_model()`
  - ✅ Final layer replacement for custom `num_classes` (2, 10, 100, 1000)
  - ✅ Forward pass with correct output shape `[batch_size, num_classes]`
  - ✅ Pretrained weights loading (`weights='DEFAULT'`)
  - ✅ Various batch sizes (1, 2, 4, 8)
  - ✅ Error handling for invalid configs

### Trainer Tests (16 tests total)

**Unit Tests** (5 tests):
- `test_mixed_precision.py`: MixedPrecisionTrainer with float16/bfloat16, CPU fallback
- `test_accelerate.py`: AccelerateTrainer single-device mode
- `test_differential_privacy.py`: DPTrainer with privacy budget tracking

**Integration Tests** (11 tests):
- `test_factory.py`: Trainer factory creation, defaults, invalid types
- `test_integration.py`: Cross-trainer compatibility, checkpoint sharing, config preservation

### Inference Tests (17 tests)
- StandardInference, MixedPrecisionInference, AccelerateInference
- Factory tests, backward compatibility, result comparison
- CPU fallback, determinism tests

## Test Execution Time

- **Full suite**: ~90 seconds
- **Network tests**: ~60 seconds
- **Trainer tests**: ~20 seconds
- **Inference tests**: ~10 seconds

## Adding New Tests

### For Network/Architecture Tests
Add test classes or functions to `tests/network/test_architectures.py`

### For New Trainers
1. Create unit test file: `tests/trainers/unit/test_<trainer_name>.py`
2. Add integration tests to `tests/trainers/integration/test_integration.py`
3. Update factory tests in `tests/trainers/integration/test_factory.py`

### For New Inference Strategies
Add test functions to `tests/inference/test_strategies.py`

## Fixtures

Shared pytest fixtures are defined in `conftest.py` at the root and are automatically available to all tests in subdirectories.

Common fixtures:
- `dummy_config`: Minimal configuration dictionary
- `dummy_model`: Small model for testing
- `dummy_dataloader`: Small dataset for quick tests

## Notes

- All tests are CPU-compatible (no GPU required)
- Tests use small models and datasets for speed
- pytest auto-discovers tests in subdirectories
- Use `-v` flag for verbose output
- Use `-k` flag to filter tests by name pattern
