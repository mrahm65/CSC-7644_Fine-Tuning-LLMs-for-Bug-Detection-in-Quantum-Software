"""Compute accuracy, macro-F1, ROC-AUC, and confusion matrix from saved runs.

Reads the per-model JSON files written by :func:`run_full_experiment` from
``--tables-dir`` (default ``./tables``) and prints a single comparison
table to stdout. Useful for inspecting a completed run without rerunning
training.

Usage::

    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --tables-dir tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.evaluate import compute_metrics  # noqa: E402


def main(argv=None) -> int:
    """Print the saved per-model metrics and the cross-model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir", default="tables",
        help="Directory containing results_<short>.json files.",
    )
    args = parser.parse_args(argv)

    found: List[dict] = []
    for fname in sorted(os.listdir(args.tables_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(args.tables_dir, fname)) as handle:
                found.append(json.load(handle))

    if not found:
        print(f"No results_*.json files in {args.tables_dir!r}.")
        return 1

    print(f"{'Model':<14s}  {'Accuracy':>14s}  {'Macro-F1':>14s}  {'ROC-AUC':>14s}")
    print("-" * 64)
    for s in found:
        print(
            f"{s['model_short']:<14s}  "
            f"{s['mean_accuracy']:.3f} +/- {s['std_accuracy']:.3f}   "
            f"{s['mean_f1_macro']:.3f} +/- {s['std_f1_macro']:.3f}   "
            f"{s['mean_roc_auc']:.3f} +/- {s['std_roc_auc']:.3f}"
        )

    # Show pooled metrics computed via src.evaluate.compute_metrics for
    # the first backbone (sanity check that the helper agrees with the
    # numbers persisted by the trainer).
    s = found[0]
    if "cv_results" in s and s["cv_results"]:
        y_true = np.concatenate([r["y_true"] for r in s["cv_results"]])
        y_pred = np.concatenate([r["y_pred"] for r in s["cv_results"]])
        probs  = np.concatenate([r["probs"]  for r in s["cv_results"]])[:, 1]
        m = compute_metrics(y_true, y_pred, probs)
        print(f"\n[{s['model_short']}] pooled via src.evaluate: {m}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
