"""Populate ``figures/`` and ``tables/`` with reproducible *sample* artifacts.

Real cross-validation training takes ~75-110 minutes on a T4 GPU, which is
not always available. This script synthesizes plausible per-fold results
(drawn from a fixed seed) and runs them through the *real* plotting and
analysis modules so the output folders contain authentic SVG figures and
CSV/JSON tables that exercise every code path in
:mod:`src.study_i.plotting` and :mod:`src.study_i.analysis`.

The artifacts are clearly marked as *demonstration outputs* in the
metadata fields; they should be replaced by running ``main.py`` once the
labeled dataset is available.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.study_i.analysis import (  # noqa: E402
    build_comparison_table,
    save_cross_model_results,
)
from src.study_i.plotting import (  # noqa: E402
    apply_plot_style,
    plot_confusion_matrix,
    plot_cross_model_comparison,
    plot_fold_distribution,
    plot_paired_per_fold,
    plot_roc_curve,
    plot_roc_overlay,
)
from src.study_i.schemas import MODEL_REGISTRY, get_default_config  # noqa: E402


# Per-backbone target performance for the synthetic demo.
DEMO_TARGETS = {
    "roberta":       {"acc": 0.78, "auc": 0.85, "noise": 0.07},
    "codebert":      {"acc": 0.86, "auc": 0.92, "noise": 0.05},
    "graphcodebert": {"acc": 0.88, "auc": 0.94, "noise": 0.05},
}

N_FOLDS = 5
N_SEEDS = 5
SAMPLES_PER_FOLD = 47   # ~ 233 / 5 - matches the real dataset's fold size


def _synth_fold(rng: np.random.RandomState, target_acc: float, target_auc: float):
    """Synthesize one fold's predictions targeting given accuracy / AUC."""
    n = SAMPLES_PER_FOLD
    # Class balance roughly matches the dataset (134 classical : 99 quantum).
    n_q = rng.binomial(n, 99 / 233)
    y_true = np.array([1] * n_q + [0] * (n - n_q))
    rng.shuffle(y_true)

    # Probabilities pulled from class-conditional Beta distributions whose
    # spread is tuned by `target_auc`.
    sep = max(0.0, min(1.0, (target_auc - 0.5) * 2))
    a_pos, b_pos = 1 + 4 * sep, 1
    a_neg, b_neg = 1, 1 + 4 * sep
    probs_q = np.where(
        y_true == 1,
        rng.beta(a_pos, b_pos, n),
        rng.beta(a_neg, b_neg, n),
    )

    # Pick the threshold that hits the target accuracy on this fold.
    sorted_q = np.sort(probs_q)
    target_correct = int(round(target_acc * n))
    # Search threshold in a small window around the target percentile.
    best_thr, best_diff = 0.5, n
    for thr in np.unique(sorted_q):
        y_pred = (probs_q >= thr).astype(int)
        correct = int((y_pred == y_true).sum())
        if abs(correct - target_correct) < best_diff:
            best_diff, best_thr = abs(correct - target_correct), thr
    y_pred = (probs_q >= best_thr).astype(int)
    probs = np.column_stack([1 - probs_q, probs_q])
    return y_true, y_pred, probs


def synthesize_summary(model: Dict[str, str], target: Dict[str, float]) -> Dict:
    """Build a single backbone's full summary dict (matches run_full_experiment)."""
    rng = np.random.RandomState(hash(model["short"]) & 0xFFFFFFFF)

    results_all = []
    fold_accs, fold_f1s, fold_aucs = [], [], []
    y_trues, y_preds, probs_all = [], [], []

    for repeat_idx in range(N_SEEDS):
        for fold_idx in range(N_FOLDS):
            # Jitter the per-fold target slightly so the scatter plot has spread.
            jit_acc = float(np.clip(
                rng.normal(target["acc"], target["noise"]), 0.4, 0.99
            ))
            jit_auc = float(np.clip(
                rng.normal(target["auc"], target["noise"]), 0.5, 0.99
            ))
            y_true, y_pred, probs = _synth_fold(rng, jit_acc, jit_auc)

            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
            }
            results_all.append({
                "repeat": repeat_idx, "fold": fold_idx, "cv_seed": 42 + repeat_idx,
                "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
                "probs": probs.tolist(), **metrics,
            })
            fold_accs.append(metrics["accuracy"])
            fold_f1s.append(metrics["f1_macro"])
            fold_aucs.append(metrics["roc_auc"])
            y_trues.append(y_true); y_preds.append(y_pred); probs_all.append(probs)

    fold_accs_arr = np.array(fold_accs)
    fold_f1s_arr  = np.array(fold_f1s)
    fold_aucs_arr = np.array(fold_aucs)
    y_true_all  = np.concatenate(y_trues)
    y_pred_all  = np.concatenate(y_preds)
    probs_all_arr = np.concatenate(probs_all)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    pooled_acc = float(accuracy_score(y_true_all, y_pred_all))
    pooled_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    pooled_auc = float(roc_auc_score(y_true_all, probs_all_arr[:, 1]))

    return {
        "model_name": model["name"],
        "model_short": model["short"],
        "description": model["description"] + " (DEMO synthetic results)",
        "fold_accs": fold_accs_arr.tolist(),
        "fold_f1s":  fold_f1s_arr.tolist(),
        "fold_aucs": fold_aucs_arr.tolist(),
        "mean_accuracy": float(fold_accs_arr.mean()),
        "std_accuracy": float(fold_accs_arr.std(ddof=1)),
        "mean_f1_macro": float(fold_f1s_arr.mean()),
        "std_f1_macro": float(fold_f1s_arr.std(ddof=1)),
        "mean_roc_auc": float(fold_aucs_arr.mean()),
        "std_roc_auc": float(fold_aucs_arr.std(ddof=1)),
        "pooled_accuracy": pooled_acc,
        "pooled_f1_macro": pooled_f1,
        "pooled_roc_auc": pooled_auc,
        "epoch_logs": [],
        "y_true_all": y_true_all.tolist(),
        "y_pred_all": y_pred_all.tolist(),
        "probs_all":  probs_all_arr.tolist(),
        "_source": "demo-synthetic",
        "results_all": results_all,
    }


def main() -> int:
    """Generate demo SVG figures + CSV/JSON tables in figures/ and tables/."""
    config = get_default_config()
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(config.tables_dir, exist_ok=True)
    apply_plot_style()
    fmt = config.figure_format

    summaries = [
        synthesize_summary(model, DEMO_TARGETS[model["short"]])
        for model in MODEL_REGISTRY
    ]

    # ---- per-model figures + tables -------------------------------------
    for s in summaries:
        plot_confusion_matrix(
            np.array(s["y_true_all"]), np.array(s["y_pred_all"]),
            model_short=s["model_short"], n_folds=N_FOLDS * N_SEEDS,
            pooled_acc=s["pooled_accuracy"],
            pooled_f1=s["pooled_f1_macro"],
            pooled_auc=s["pooled_roc_auc"],
            output_dir=config.figures_dir, fmt=fmt,
        )
        plot_fold_distribution(
            np.array(s["fold_accs"]), np.array(s["fold_f1s"]), np.array(s["fold_aucs"]),
            model_short=s["model_short"], output_dir=config.figures_dir, fmt=fmt,
        )
        plot_roc_curve(
            s["results_all"], model_short=s["model_short"],
            pooled_auc=s["pooled_roc_auc"],
            output_dir=config.figures_dir, fmt=fmt,
        )

        # Per-model JSON / CSV.
        json_path = os.path.join(config.tables_dir, f"results_{s['model_short']}.json")
        csv_path  = os.path.join(config.tables_dir, f"per_fold_{s['model_short']}.csv")
        import pandas as pd
        cv_df = pd.DataFrame([
            {"repeat": r["repeat"] + 1, "fold": r["fold"] + 1,
             "accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
             "f1_weighted": r["f1_weighted"], "roc_auc": r["roc_auc"]}
            for r in s["results_all"]
        ])
        cv_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as handle:
            payload = {
                "task": "classical_vs_quantum_binary",
                "model": s["model_name"],
                "model_short": s["model_short"],
                "description": s["description"],
                "n_samples": 233,
                "cv_setup": f"{N_FOLDS}-fold x {N_SEEDS} seeds = {N_FOLDS*N_SEEDS} folds",
                "mean_accuracy": s["mean_accuracy"],
                "std_accuracy":  s["std_accuracy"],
                "mean_f1_macro": s["mean_f1_macro"],
                "std_f1_macro":  s["std_f1_macro"],
                "mean_roc_auc":  s["mean_roc_auc"],
                "std_roc_auc":   s["std_roc_auc"],
                "pooled_accuracy": s["pooled_accuracy"],
                "pooled_f1_macro": s["pooled_f1_macro"],
                "pooled_roc_auc":  s["pooled_roc_auc"],
                "_source": "demo-synthetic",
                "cv_results": s["results_all"],
            }
            json.dump(payload, handle, indent=2, default=str)

    # ---- cross-model figures + tables -----------------------------------
    plot_cross_model_comparison(summaries, output_dir=config.figures_dir, fmt=fmt)
    plot_roc_overlay(summaries, output_dir=config.figures_dir, fmt=fmt)
    plot_paired_per_fold(summaries, output_dir=config.figures_dir, fmt=fmt)

    save_cross_model_results(
        summaries,
        n_samples=233,
        cv_setup=f"{N_FOLDS}-fold x {N_SEEDS} seeds = {N_FOLDS*N_SEEDS} folds (DEMO)",
        output_dir=config.tables_dir,
    )

    # Drop the heavy fields from each summary before printing the table
    table = build_comparison_table(summaries)
    print("\nDemo cross-model comparison:\n")
    print(table.to_string(index=False))
    print(f"\nFigures saved to: {config.figures_dir}")
    print(f"Tables  saved to: {config.tables_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
