"""Cross-model aggregation: comparison table + combined results JSON.

These helpers run *after* :func:`src.study_i.training.run_full_experiment`
has been called for every backbone in the model registry.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import pandas as pd


def build_comparison_table(
    summaries: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Format a side-by-side ``mean +/- std`` summary across backbones."""
    return pd.DataFrame(
        [
            {
                "Model": s["model_short"],
                "HF identifier": s["model_name"],
                "Accuracy": (
                    f"{s['mean_accuracy']:.3f} +/- {s['std_accuracy']:.3f}"
                ),
                "Macro-F1": (
                    f"{s['mean_f1_macro']:.3f} +/- {s['std_f1_macro']:.3f}"
                ),
                "ROC-AUC": (
                    f"{s['mean_roc_auc']:.3f} +/- {s['std_roc_auc']:.3f}"
                ),
                "Pooled Acc": f"{s['pooled_accuracy']:.3f}",
                "Pooled F1": f"{s['pooled_f1_macro']:.3f}",
            }
            for s in summaries
        ]
    )


def save_cross_model_results(
    summaries: Sequence[Dict[str, Any]],
    *,
    n_samples: int,
    cv_setup: str,
    output_dir: str,
) -> List[str]:
    """Write the comparison CSV + combined JSON and return the file paths.

    The combined JSON intentionally drops the heavy per-sample arrays
    (``y_true_all``, ``y_pred_all``, ``probs_all``, ``epoch_logs``) since
    those are already preserved in each ``results_<short>.json`` file.
    """
    os.makedirs(output_dir, exist_ok=True)

    table = build_comparison_table(summaries)
    csv_path = os.path.join(output_dir, "cross_model_comparison.csv")
    table.to_csv(csv_path, index=False)

    heavy_fields = {"y_true_all", "y_pred_all", "probs_all", "epoch_logs"}
    combined = {
        "task": "classical_vs_quantum_binary",
        "n_samples": n_samples,
        "cv_setup": cv_setup,
        "models": [
            {k: v for k, v in s.items() if k not in heavy_fields}
            for s in summaries
        ],
    }

    json_path = os.path.join(output_dir, "cross_model_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, default=str)

    return [csv_path, json_path]
