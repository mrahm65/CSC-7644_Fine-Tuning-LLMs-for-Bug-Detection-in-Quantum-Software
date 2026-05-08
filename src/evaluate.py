"""Compute metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and ROC-AUC.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Hard predictions aligned with ``y_true``.
        y_prob: Optional positive-class probabilities. Required for ROC-AUC.

    Returns:
        Dict containing ``accuracy``, ``macro_f1``, ``weighted_f1``, and
        ``roc_auc`` (the last is ``None`` when ``y_prob`` is omitted or
        when only one class is present in ``y_true``).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true_arr, y_pred_arr, average="weighted", zero_division=0
            )
        ),
    }

    metrics["roc_auc"] = None  # type: ignore[assignment]
    if y_prob is not None and len(set(y_true_arr.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob))

    return metrics


def compute_metrics_for_trainer(eval_pred) -> Dict[str, float]:
    """``Trainer.compute_metrics`` callback returning accuracy + F1 variants."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(
            f1_score(labels, preds, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, preds, average="weighted", zero_division=0)
        ),
    }


def pooled_metrics(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    probs_all: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute pooled metrics over the concatenated CV folds."""
    out = {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(
            f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
        ),
    }
    if probs_all is not None and len(set(y_true_all.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true_all, probs_all[:, 1]))
    return out


def confusion_breakdown(y_true: Sequence[int], y_pred: Sequence[int]):
    """Return the 2x2 confusion matrix and its row-normalized percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    return cm, cm_pct
