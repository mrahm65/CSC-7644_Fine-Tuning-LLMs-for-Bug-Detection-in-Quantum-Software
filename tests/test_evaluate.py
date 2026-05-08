"""Tests for src.evaluate."""
from __future__ import annotations

from src.evaluate import compute_metrics, confusion_breakdown, pooled_metrics
import numpy as np


def test_compute_metrics_runs():
    """Check that metric computation runs on a small example."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    y_prob = [0.1, 0.8, 0.4, 0.2]

    metrics = compute_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.75


def test_compute_metrics_handles_single_class():
    """ROC-AUC is None when only one class is present."""
    metrics = compute_metrics([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] is None


def test_confusion_breakdown_shape():
    """confusion_breakdown returns a 2x2 matrix and percentages."""
    cm, cm_pct = confusion_breakdown([0, 1, 1, 0], [0, 1, 0, 0])
    assert cm.shape == (2, 2)
    assert cm_pct.shape == (2, 2)
    np.testing.assert_allclose(cm_pct.sum(axis=1), np.array([100.0, 100.0]))


def test_pooled_metrics_basic():
    """Pooled metrics agree with single-batch metrics on a tiny example."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    probs  = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3]])
    out = pooled_metrics(y_true, y_pred, probs)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["roc_auc"] == 1.0
