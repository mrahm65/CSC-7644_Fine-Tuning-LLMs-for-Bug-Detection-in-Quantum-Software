"""Reusable Python modules for the Study I bug-classification pipeline.

This package contains five flat modules that mirror the major concerns of
the experiment:

* :mod:`src.data_utils`   - data loading, label filtering, text construction
* :mod:`src.model_utils`  - tokenizer + sequence-classification model loading
* :mod:`src.train_utils`  - training helpers (oversampling, class weights,
                             early stopping)
* :mod:`src.evaluate`     - metric computation
* :mod:`src.plotting`     - matplotlib helpers for figures
"""

from src.data_utils import build_input_text, load_dataset
from src.evaluate import compute_metrics

__all__ = [
    "build_input_text",
    "load_dataset",
    "compute_metrics",
]
