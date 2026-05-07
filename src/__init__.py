"""Modular implementation of the classical vs. quantum bug-classification pipeline.

Top-level package for CSC 7644 final project.

The lightweight modules (:mod:`src.config`, :mod:`src.data_loader`,
:mod:`src.callbacks`, :mod:`src.evaluation`, :mod:`src.plots`) are
re-exported eagerly. Symbols from :mod:`src.training` are exposed lazily
via :func:`__getattr__` so that simply importing the package does not
require torch / transformers - useful for tests and tooling that only
need to read the configuration.
"""

from __future__ import annotations

from src.config import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    MODEL_REGISTRY,
    NUM_LABELS,
    TrainingConfig,
    get_default_config,
)
from src.data_loader import (
    build_text,
    get_text_label_arrays,
    load_labeled_dataset,
)

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


def __getattr__(name: str):
    """Lazy passthrough for the torch-dependent training symbols."""
    if name in {"run_fold", "run_full_experiment"}:
        from src import training as _training  # local import keeps torch lazy
        return getattr(_training, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
