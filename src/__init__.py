"""Modular implementation of the classical vs. quantum bug-classification pipeline.

Top-level package for CSC 7644 final project. Re-exports the most commonly
used factories so callers can do, e.g.::

    from src import load_labeled_dataset, run_full_experiment
"""

from src.config import (
    MODEL_REGISTRY,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    TrainingConfig,
    get_default_config,
)
from src.data_loader import load_labeled_dataset, build_text, get_text_label_arrays
from src.training import run_fold, run_full_experiment

__all__ = [
    "MODEL_REGISTRY",
    "LABEL_LIST",
    "LABEL2ID",
    "ID2LABEL",
    "NUM_LABELS",
    "TrainingConfig",
    "get_default_config",
    "load_labeled_dataset",
    "build_text",
    "get_text_label_arrays",
    "run_fold",
    "run_full_experiment",
]
