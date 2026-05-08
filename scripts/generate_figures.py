"""Create the canonical figures used in the final report.

Produces five PNG figures referenced in ``figures/README.md``:

* ``fig10_dataset_distribution.png``
* ``methodology_figure.png``
* ``fig8_mean_metrics.png``
* ``fig1_confusion_matrix.png``
* ``fig5_roc_curve.png``

If saved per-model JSON files exist under ``--tables-dir``, the cross-
model figures are rebuilt from those real results. Otherwise the script
falls back to the embedded headline numbers from the v11 notebook so the
report-ready PNGs can be regenerated even without the raw fold data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.data_utils import LABEL_LIST  # noqa: E402
from src.plotting import (  # noqa: E402
    apply_plot_style,
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_mean_metrics_bar,
    plot_methodology_diagram,
    plot_roc_curve,
)


# Headline mean +/- std from the v11 notebook (5-fold x 5 seeds = 25 folds).
HEADLINE_NUMBERS = {
    "roberta":       {"mean_accuracy": 0.764, "std_accuracy": 0.061,
                      "mean_f1_macro": 0.754, "std_f1_macro": 0.066,
                      "mean_roc_auc":  0.858, "std_roc_auc":  0.048,
                      "model_short": "roberta"},
    "codebert":      {"mean_accuracy": 0.767, "std_accuracy": 0.057,
                      "mean_f1_macro": 0.763, "std_f1_macro": 0.056,
                      "mean_roc_auc":  0.855, "std_roc_auc":  0.044,
                      "model_short": "codebert"},
    "graphcodebert": {"mean_accuracy": 0.758, "std_accuracy": 0.058,
                      "mean_f1_macro": 0.756, "std_f1_macro": 0.058,
                      "mean_roc_auc":  0.860, "std_roc_auc":  0.047,
                      "model_short": "graphcodebert"},
}


def _try_load_summaries(tables_dir: str) -> List[Dict[str, Any]]:
    """Load per-model results JSON files when available."""
    summaries: List[Dict[str, Any]] = []
    if not os.path.isdir(tables_dir):
        return summaries
    for fname in sorted(os.listdir(tables_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            with open(os.path.join(tables_dir, fname)) as handle:
                summaries.append(json.load(handle))
    return summaries


def _pooled_y(summary: Dict[str, Any]):
    """Concatenate y_true / y_pred / y_prob from cv_results."""
    if "cv_results" not in summary:
        return None
    y_true = np.concatenate([r["y_true"] for r in summary["cv_results"]])
    y_pred = np.concatenate([r["y_pred"] for r in summary["cv_results"]])
    probs  = np.concatenate([np.array(r["probs"])[:, 1] for r in summary["cv_results"]])
    return y_true, y_pred, probs


def main(argv=None) -> int:
    """Generate every required figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/bug_patterns_labeled.json",
                        help="JSON dataset (only used for the class-distribution plot).")
    parser.add_argument("--tables-dir", default="tables",
                        help="Directory containing results_<short>.json files.")
    parser.add_argument("--figures-dir", default="figures",
                        help="Where to write the PNG figures.")
    args = parser.parse_args(argv)

    apply_plot_style()
    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- fig10_dataset_distribution.png --------------------------------
    counts = {"classical": 134, "quantum": 99}  # from the v11 notebook
    if os.path.exists(args.data_path):
        try:
            with open(args.data_path) as handle:
                records = json.load(handle)
            present = [r for r in records if r.get("bug_category") in LABEL_LIST]
            if present:
                counts = {lbl: 0 for lbl in LABEL_LIST}
                for r in present:
                    counts[r["bug_category"]] += 1
        except Exception:
            pass
    plot_dataset_distribution(
        counts, os.path.join(args.figures_dir, "fig10_dataset_distribution.png"),
    )

    # ---- methodology_figure.png ----------------------------------------
    plot_methodology_diagram(
        os.path.join(args.figures_dir, "methodology_figure.png"),
    )

    # ---- fig8_mean_metrics.png -----------------------------------------
    summaries = _try_load_summaries(args.tables_dir)
    if not summaries:
        # Fall back to headline numbers if no result JSONs are present.
        summaries = list(HEADLINE_NUMBERS.values())
    plot_mean_metrics_bar(
        summaries, os.path.join(args.figures_dir, "fig8_mean_metrics.png"),
    )

    # ---- fig1_confusion_matrix.png + fig5_roc_curve.png -----------------
    real_summary = next(
        (s for s in summaries if "cv_results" in s and s["cv_results"]),
        None,
    )
    if real_summary is not None:
        y_true, y_pred, probs = _pooled_y(real_summary)
        plot_confusion_matrix(
            y_true, y_pred,
            os.path.join(args.figures_dir, "fig1_confusion_matrix.png"),
            title=f"Confusion matrix - {real_summary['model_short']} "
                  f"(pooled across {len(real_summary['cv_results'])} folds)",
        )
        plot_roc_curve(
            y_true, probs,
            os.path.join(args.figures_dir, "fig5_roc_curve.png"),
            title=f"ROC curve - {real_summary['model_short']} "
                  f"(pooled across {len(real_summary['cv_results'])} folds)",
        )
    else:
        # Synthetic fallback so the report-ready PNGs always exist.
        rng = np.random.RandomState(42)
        n = 233
        y_true = rng.binomial(1, 99 / 233, size=n)
        probs = np.where(y_true == 1, rng.beta(4, 1.2, n), rng.beta(1.2, 4, n))
        y_pred = (probs >= 0.5).astype(int)
        plot_confusion_matrix(
            y_true, y_pred,
            os.path.join(args.figures_dir, "fig1_confusion_matrix.png"),
            title="Confusion matrix (illustrative, 233 records)",
        )
        plot_roc_curve(
            y_true, probs,
            os.path.join(args.figures_dir, "fig5_roc_curve.png"),
            title="ROC curve (illustrative, 233 records)",
        )

    print(f"All figures written to {args.figures_dir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
