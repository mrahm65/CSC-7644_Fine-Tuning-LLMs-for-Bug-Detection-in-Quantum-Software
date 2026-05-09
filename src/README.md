# `src/` — Reusable Python modules

Six flat modules implementing the data, model, training, evaluation,
and plotting concerns of the classical-vs-quantum bug-classification
pipeline. The notebook
(`notebooks/quantum-vs-classical-bug-detection-Final.ipynb`) and the
runnable scripts (`scripts/run_training.py`, `run_evaluation.py`,
`generate_figures.py`) all import from this package.

## Module map

| File | Purpose |
|------|---------|
| `__init__.py`      | Marks `src/` as a Python package; re-exports the most common public symbols (`load_dataset`, `build_input_text`, `compute_metrics`). |
| `data_utils.py`    | Dataset loading, label filtering, and canonical text construction (`name + description + example_code`). Defines `LABEL_LIST`, `LABEL2ID`, `ID2LABEL`, `NUM_LABELS`, plus `load_dataset()`, `build_input_text()`, and `to_text_label_arrays()`. |
| `model_utils.py`   | HuggingFace tokenizer + sequence-classification model loaders. Provides `load_model_and_tokenizer()` and `get_data_collator()`; heavy `transformers` imports are deferred to call time so the rest of the package imports cheaply. |
| `train_utils.py`   | Training helpers: inverse-frequency `compute_class_weights()`, `oversample_minority_class()`, plus three factory functions that build the custom callbacks / Trainer subclass — `make_manual_early_stopping_callback()`, `make_epoch_log_callback()`, and `make_weighted_trainer_class()`. The manual early-stopping callback is the disk-safe replacement for `transformers.EarlyStoppingCallback`. |
| `evaluate.py`      | Metric primitives: `compute_metrics()` (accuracy, macro-F1, weighted-F1, ROC-AUC), `compute_metrics_for_trainer()` (the HF `Trainer` callback signature), `pooled_metrics()` for cross-fold pooling, and `confusion_breakdown()` for the 2x2 + percentages used by the plot helper. |
| `plotting.py`      | Matplotlib helpers used by the notebook and `scripts/generate_figures.py`: `plot_confusion_matrix()`, `plot_roc_curve()`, `plot_roc_overlay()`, `plot_mean_metrics_bar()`, `plot_dataset_distribution()`, and `plot_methodology_diagram()`. The shared rcParams style is applied via `apply_plot_style()`. |

## Public API

The most common entry points are re-exported at the package level:

```python
from src import build_input_text, load_dataset, compute_metrics
```

For the rest, import the symbol directly from its module:

```python
from src.data_utils import LABEL2ID, to_text_label_arrays
from src.model_utils import load_model_and_tokenizer, get_data_collator
from src.train_utils import (
    compute_class_weights,
    oversample_minority_class,
    make_manual_early_stopping_callback,
    make_epoch_log_callback,
    make_weighted_trainer_class,
)
from src.evaluate import (
    compute_metrics,
    compute_metrics_for_trainer,
    pooled_metrics,
    confusion_breakdown,
)
from src.plotting import (
    apply_plot_style,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_overlay,
    plot_mean_metrics_bar,
    plot_dataset_distribution,
    plot_methodology_diagram,
)
```

## Code-quality conventions

- **PEP 8** formatted (verified with `flake8 --max-line-length=99`,
  zero warnings).
- **Docstrings** on every module, function, method, and class
  (verified by AST scan).
- **Inline comments** on non-obvious logic (e.g., the patience epsilon
  inside `ManualEarlyStoppingCallback`, the device-move in
  `WeightedTrainer.set_class_weights`, the row-normalised percentages
  in the confusion-matrix plot).
- **Snake_case** functions, **CapWords** classes, **UPPER_CASE**
  module-level constants — full PEP 8 naming compliance.

## Running the lightweight tests

The `tests/` package exercises `data_utils.py` and `evaluate.py`
without requiring `torch` or `transformers`, so the test suite runs in
seconds:

```bash
TMPDIR=/tmp pytest tests/ -q
```

For the full training pipeline (which exercises `model_utils.py` and
`train_utils.py` end-to-end), see `scripts/run_training.py` or the
notebook in `notebooks/`.
