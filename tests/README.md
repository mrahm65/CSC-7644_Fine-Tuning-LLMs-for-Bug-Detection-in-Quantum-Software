# Tests

This folder contains lightweight tests for project utility functions.

## Example tests

| Test file                | What it checks                                            |
|--------------------------|-----------------------------------------------------------|
| `test_data_utils.py`     | Data loading and input-text construction.                 |
| `test_evaluate.py`       | Metric computation (accuracy, macro-F1, ROC-AUC).         |
| `test_schemas.py`        | Backward-compat schema constants and TrainingConfig.      |
| `test_dataset.py`        | Dataset-loader edge cases (legacy module; lightweight).   |

## Running tests

```bash
TMPDIR=/tmp pytest tests/ -q
```

The lightweight tests above run without torch / transformers installed.
The legacy `test_callbacks.py` requires torch and is skipped when those
packages are missing.
