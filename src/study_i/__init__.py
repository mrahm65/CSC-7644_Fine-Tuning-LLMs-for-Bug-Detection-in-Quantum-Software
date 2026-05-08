"""Study I: classical vs. quantum bug-classification benchmark.

This subpackage contains every module needed to reproduce the three-
backbone comparison (RoBERTa / CodeBERT / GraphCodeBERT) reported in the
final-project quad chart for CSC 7644.

Module map:

* :mod:`src.study_i.schemas`    - configuration dataclass + label space
* :mod:`src.study_i.dataset`    - JSON loading + text-builder
* :mod:`src.study_i.callbacks`  - early stopping, epoch logging, weighted Trainer
* :mod:`src.study_i.training`   - run_fold + run_full_experiment
* :mod:`src.study_i.plotting`   - matplotlib helpers (svg by default)
* :mod:`src.study_i.analysis`   - cross-model comparison + combined JSON
"""

from src.study_i.schemas import (
    ID2LABEL,
    LABEL2ID,
    LABEL_LIST,
    MODEL_REGISTRY,
    NUM_LABELS,
    TrainingConfig,
    get_default_config,
)
from src.study_i.dataset import (
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
        from src.study_i import training as _training
        return getattr(_training, name)
    raise AttributeError(f"module 'src.study_i' has no attribute {name!r}")
