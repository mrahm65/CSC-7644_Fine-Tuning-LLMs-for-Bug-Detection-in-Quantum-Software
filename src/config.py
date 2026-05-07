"""Static configuration for the bug-classification pipeline.

Defines the model registry compared in the experiment, the binary label
mapping, and a :class:`TrainingConfig` dataclass that bundles every
hyperparameter consumed by the training loop.

Environment variables (read at runtime by :func:`get_default_config`):

* ``DATA_PATH``   - JSON file containing the labeled bug-pattern records.
* ``OUTPUT_DIR``  - Directory where figures, JSON results, and CSVs are
                    written.

Both variables fall back to sensible repository-relative defaults so the
package works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
LABEL_LIST: List[str] = ["classical", "quantum"]
LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS: int = len(LABEL_LIST)


# ---------------------------------------------------------------------------
# Model registry: three RoBERTa-architecture encoders (~125M params each)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: List[Dict[str, str]] = [
    {
        "name": "roberta-base",
        "short": "roberta",
        "description": (
            "General-purpose text encoder (English only) - no code "
            "pretraining"
        ),
    },
    {
        "name": "microsoft/codebert-base",
        "short": "codebert",
        "description": "Code + NL bimodal pretraining (CodeSearchNet)",
    },
    {
        "name": "microsoft/graphcodebert-base",
        "short": "graphcodebert",
        "description": "Code + NL + data-flow graph pretraining",
    },
]


# ---------------------------------------------------------------------------
# Repository-relative defaults
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = str(_REPO_ROOT / "data" / "bug_patterns_labeled.json")
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "results")


# ---------------------------------------------------------------------------
# Training-time hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Hyperparameters and protocol settings for a single experiment run.

    Defaults mirror the values used in the original Kaggle notebook so that
    results are reproducible. Override any field via the constructor or
    environment variables in :func:`get_default_config`.
    """

    # Tokenization
    max_len: int = 256

    # Cross-validation
    n_folds: int = 5
    cv_seeds: Tuple[int, ...] = (42, 7, 2024, 99, 123)

    # Optimization
    num_epochs: int = 12
    learning_rate: float = 2e-5
    batch_size: int = 8
    eval_batch_size: int = 64
    weight_decay: float = 0.05
    warmup_ratio: float = 0.15
    dropout: float = 0.2
    val_split: float = 0.10

    # Early stopping
    es_patience: int = 4

    # Loss
    label_smoothing: float = 0.05

    # I/O
    data_path: str = DEFAULT_DATA_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Misc
    use_fp16: bool = True
    logging_steps: int = 500
    label_field: str = "bug_category"
    text_fields: Tuple[str, ...] = ("name", "description", "example_code")

    # Filled in by ``__post_init__`` so each call sees a fresh tuple
    _registry: Tuple[Dict[str, str], ...] = field(default_factory=tuple)

    @property
    def total_folds_per_model(self) -> int:
        """Number of fold-runs per backbone (``n_folds * len(cv_seeds)``)."""
        return self.n_folds * len(self.cv_seeds)


def get_default_config() -> TrainingConfig:
    """Build a :class:`TrainingConfig` honoring ``DATA_PATH`` / ``OUTPUT_DIR``.

    Environment variables override the dataclass defaults; everything else
    keeps the values used in the source notebook.
    """
    return TrainingConfig(
        data_path=os.environ.get("DATA_PATH", DEFAULT_DATA_PATH),
        output_dir=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
