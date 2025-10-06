# Workflow Integration Tests

This directory contains end-to-end integration tests for the complete ML workflows documented in `docs/workflow.md`.

## Purpose

These tests validate that all documented workflows work correctly from start to finish using the actual CLI commands. They are designed to:

1. **Verify documentation accuracy** - ensure documented commands actually work
2. **Catch regressions** - detect when changes break existing workflows  
3. **Validate integrations** - test that all components work together properly
4. **Serve as examples** - demonstrate real usage patterns

## Test Organization

- **`conftest.py`** - Shared fixtures and test utilities
- **`test_quick_start_workflow.py`** - Quick start workflows (split, config, train, inference)
- **`test_training_workflows.py`** - Training variants (resume, mixed precision, early stopping)
- **`test_search_workflow.py`** - Hyperparameter search workflows (requires optuna)
- **`test_visualization_workflows.py`** - Visualization commands (TensorBoard, samples, predictions)
- **`test_cross_validation_workflow.py`** - Cross-validation workflows

## Running Tests

### Run All Workflow Tests
```bash
pytest tests/workflow/
```

### Run Specific Test File
```bash
pytest tests/workflow/test_quick_start_workflow.py
```

### Run Specific Test
```bash
pytest tests/workflow/test_quick_start_workflow.py::test_quick_start_workflow
```

### Skip Slow Tests
```bash
pytest tests/workflow/ -m "not slow"
```

### Skip Optuna Tests (if optuna not installed)
```bash
pytest tests/workflow/ -m "not optuna"
```

### Run Only Quick Start Tests
```bash
pytest tests/workflow/test_quick_start_workflow.py -v
```

### Run with More Output
```bash
pytest tests/workflow/ -v -s
```

## Test Markers

- **`@pytest.mark.workflow`** - Marks test as workflow integration test
- **`@pytest.mark.slow`** - Test takes significant time to run
- **`@pytest.mark.optuna`** - Test requires optuna installation

## Requirements

### Dataset
Tests use the hymenoptera dataset at `data/hymenoptera_data/`. The dataset should have:
- `raw/` directory with class subdirectories
- Images in class-named folders (e.g., `raw/ants/*.jpg`, `raw/bees/*.jpg`)

### Optional Dependencies
Some tests require optional features:
```bash
# For hyperparameter search tests
pip install -e ".[optuna]"

# For differential privacy tests (if added)
pip install -e ".[dp]"
```

## Test Duration

Workflow tests are integration tests and take significant time:

- **Quick Start**: ~5 minutes (includes training 2 epochs)
- **Training Workflows**: ~10-15 minutes (resume, early stopping, etc.)
- **Search Workflows**: ~10-20 minutes (runs multiple trials)
- **Visualization**: ~5 minutes (generates various plots)
- **Cross-Validation**: ~15-20 minutes (trains multiple folds)

**Total**: Approximately 45-70 minutes for all workflow tests

## Test Strategy

1. **Minimal configurations** - Use 2 epochs, small batch sizes for speed
2. **CPU-only by default** - Tests run on CPU unless GPU-specific (mixed precision)
3. **Temporary workspaces** - Each test uses isolated temp directory
4. **Cleanup automatic** - pytest handles cleanup after tests
5. **Timeout protection** - All subprocess calls have timeouts

## What Tests Verify

Each test verifies:
- ✅ Command exits successfully (returncode == 0)
- ✅ Expected output files are created
- ✅ Output files contain expected content
- ✅ Directory structure matches documentation
- ✅ Logs contain expected messages
- ✅ No errors in stderr output

## Debugging Failed Tests

### View Test Output
```bash
pytest tests/workflow/test_name.py::test_func -v -s
```

### Check Temporary Files
Tests create files in `/tmp/pytest-*` directories. You can inspect these during debugging.

### Run With Increased Timeout
If tests timeout on slow machines, modify timeout values in test files.

### Check Dataset
Ensure `data/hymenoptera_data/` exists and has proper structure:
```bash
tree data/hymenoptera_data/raw/
```

## Adding New Workflow Tests

When adding new workflow tests:

1. Use the `temp_workspace` fixture for isolated testing
2. Add appropriate markers (`@pytest.mark.workflow`, `@pytest.mark.slow`)
3. Use subprocess to call CLI commands (mimics real usage)
4. Set reasonable timeouts (consider slow CI environments)
5. Clean up any global state changes
6. Document what the test validates

Example:
```python
@pytest.mark.workflow
@pytest.mark.slow
def test_my_workflow(test_dataset_dir, temp_workspace):
    """Test description of what workflow this validates."""
    # Setup
    config_path = temp_workspace["configs"] / "my_config.yaml"
    
    # Run commands
    result = subprocess.run(
        ["ml-command", "--arg", "value"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    # Verify
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert expected_file.exists()
```

## Continuous Integration

In CI environments:
- Tests run in isolated containers
- Dataset is pre-loaded
- Tests run sequentially to avoid resource contention
- Failed tests include full output in CI logs

## Notes

- Tests are **not** run by default in regular pytest runs (use explicit invocation)
- Tests use **real** CLI commands, not Python API calls
- Tests are **stateful** - they create files and run training
- Tests require **sufficient disk space** (~500MB per test run)
- Tests benefit from **GPU** but work on CPU
