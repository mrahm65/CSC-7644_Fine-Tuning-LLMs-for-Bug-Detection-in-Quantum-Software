"""Smoke tests for src.callbacks (no torch / transformers required)."""

from __future__ import annotations

import random
import unittest

from src.callbacks import compute_metrics, oversample_to_balance


class OversampleTests(unittest.TestCase):
    """``oversample_to_balance`` should yield equal class counts."""

    def test_balances_imbalanced_classes(self) -> None:
        texts = [f"t{i}" for i in range(10)]
        labels = [0] * 7 + [1] * 3
        rng = random.Random(0)
        out_t, out_y = oversample_to_balance(texts, labels, rng)
        self.assertEqual(len(out_t), len(out_y))
        self.assertEqual(out_y.count(0), out_y.count(1))

    def test_already_balanced_passthrough(self) -> None:
        texts = [f"t{i}" for i in range(8)]
        labels = [0] * 4 + [1] * 4
        rng = random.Random(0)
        out_t, out_y = oversample_to_balance(texts, labels, rng)
        self.assertEqual(out_y.count(0), 4)
        self.assertEqual(out_y.count(1), 4)


class ComputeMetricsTests(unittest.TestCase):
    """``compute_metrics`` returns the expected metric keys."""

    def test_returns_required_keys(self) -> None:
        import numpy as np

        logits = np.array([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1], [0.1, 2.0]])
        labels = np.array([0, 1, 0, 0])
        out = compute_metrics((logits, labels))
        for key in ("accuracy", "f1_macro", "f1_weighted"):
            self.assertIn(key, out)
            self.assertGreaterEqual(out[key], 0.0)
            self.assertLessEqual(out[key], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
