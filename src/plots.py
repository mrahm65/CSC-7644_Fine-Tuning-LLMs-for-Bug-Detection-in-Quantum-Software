"""Matplotlib helpers used by both the per-model and cross-model pipeline.

Each function:

* applies the project's house style on first call (:func:`apply_plot_style`),
* writes its figure to ``output_dir`` with a fixed naming convention,
* honours the active ``fmt`` argument (default ``"svg"``; ``"png"`` also
  supported),
* closes the figure so notebooks/CLIs do not leak memory.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.config import LABEL_LIST


_STYLE_APPLIED = False


def apply_plot_style() -> None:
    """Apply a clean, paper-friendly matplotlib rcParams config (idempotent)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    _STYLE_APPLIED = True


PALETTE = {"classical": "#e74c3c", "quantum": "#2980b9", "mean": "#2c3e50"}
OVERLAY_COLORS = {
    "roberta": "#7f8c8d",
    "codebert": "#2980b9",
    "graphcodebert": "#16a085",
}


def _save(fig, output_dir: str, name: str, fmt: str) -> str:
    """Resolve ``output_dir/<name>.<fmt>``, save, and close the figure."""
    fmt = (fmt or "svg").lower().lstrip(".")
    if fmt not in {"svg", "png", "pdf"}:
        raise ValueError(
            f"Unsupported figure format {fmt!r}; expected svg, png, or pdf."
        )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_short: str,
    n_folds: int,
    pooled_acc: float,
    pooled_f1: float,
    pooled_auc: float,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Render a confusion matrix with cell counts and percentages."""
    apply_plot_style()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=LABEL_LIST,
        yticklabels=LABEL_LIST,
        cbar_kws={"label": "Prediction count"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(
                j + 0.5, i + 0.40, f"{cm[i, j]}",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=color,
            )
            ax.text(
                j + 0.5, i + 0.62, f"({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=11, color=color,
            )
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)
    ax.set_title(
        f"Confusion matrix - {model_short} ({n_folds} CV folds)\n"
        f"Accuracy = {pooled_acc:.1%}   "
        f"Macro-F1 = {pooled_f1:.1%}   "
        f"AUC = {pooled_auc:.3f}",
        fontsize=12, pad=10,
    )
    ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    return _save(fig, output_dir, f"fig1_confusion_matrix_{model_short}", fmt)


def plot_fold_distribution(
    fold_accs: np.ndarray,
    fold_f1s: np.ndarray,
    fold_aucs: np.ndarray,
    *,
    model_short: str,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Plot per-fold accuracy/F1/AUC scatter with mean and +/-1 SD band."""
    apply_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    rng = np.random.RandomState(42)
    panels = [
        (axes[0], fold_accs, "Accuracy", "#e74c3c"),
        (axes[1], fold_f1s, "Macro-F1", "#2980b9"),
        (axes[2], fold_aucs, "ROC-AUC", "#27ae60"),
    ]

    for ax, vals, label, color in panels:
        mean = vals.mean()
        std = vals.std(ddof=1)
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.axhspan(mean - std, mean + std, alpha=0.12, color=color,
                   label=f"+/-1 std ({std:.3f})")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5,
                   label="Random (50%)")
        ax.axhline(mean, color=PALETTE["mean"], linestyle="--",
                   linewidth=2, label=f"Mean = {mean:.3f}")
        ax.scatter(jitter, vals, s=60, color=color, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.2, 1.02)
        ax.set_xticks([])
        ax.set_ylabel(label)
        ax.set_title(f"{label} across {len(vals)} folds", fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")

    fig.suptitle(
        f"Per-fold performance - {model_short}\n5-fold CV x 5 seeds",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    return _save(fig, output_dir, f"fig2_fold_distribution_{model_short}", fmt)


def plot_roc_curve(
    results_all: List[Dict[str, Any]],
    *,
    model_short: str,
    pooled_auc: float,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Plot per-fold ROC curves with the mean ROC and a +/-1 SD band."""
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(6, 6))
    base_fpr = np.linspace(0, 1, 200)
    tprs: List[np.ndarray] = []

    for fold in results_all:
        fpr, tpr, _ = roc_curve(fold["y_true"], np.array(fold["probs"])[:, 1])
        interpolated = np.interp(base_fpr, fpr, tpr)
        tprs.append(interpolated)
        ax.plot(base_fpr, interpolated, color="#2980b9",
                alpha=0.10, linewidth=0.8)

    tprs_arr = np.array(tprs)
    mean_tpr = tprs_arr.mean(axis=0)
    std_tpr = tprs_arr.std(axis=0, ddof=1)
    ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="#2980b9", alpha=0.20, label="+/-1 std band")
    ax.plot(base_fpr, mean_tpr, color="#2980b9", linewidth=2.5,
            label=f"Mean ROC (AUC = {pooled_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        f"ROC curve - {model_short}\n"
        f"Mean AUC = {pooled_auc:.3f} (pooled across {len(results_all)} folds)",
        fontsize=12,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return _save(fig, output_dir, f"fig3_roc_curve_{model_short}", fmt)


def plot_cross_model_comparison(
    summaries: Sequence[Dict[str, Any]],
    *,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Render a 3-panel bar chart (accuracy / F1 / AUC) across backbones."""
    from scipy.stats import ttest_rel  # Lazy import keeps cold-start small.

    apply_plot_style()

    model_shorts = [s["model_short"] for s in summaries]
    n_models = len(summaries)
    metric_keys = [
        ("mean_accuracy", "std_accuracy", "fold_accs", "Accuracy", "#e74c3c"),
        ("mean_f1_macro", "std_f1_macro", "fold_f1s", "Macro-F1", "#2980b9"),
        ("mean_roc_auc", "std_roc_auc", "fold_aucs", "ROC-AUC", "#27ae60"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xpos = np.arange(n_models)

    for ax, (mkey, skey, fkey, label, color) in zip(axes, metric_keys):
        means = [s[mkey] for s in summaries]
        stds = [s[skey] for s in summaries]
        bars = ax.bar(xpos, means, yerr=stds, capsize=6, color=color,
                      alpha=0.80, edgecolor="black", linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5,
                   label="Random (50%)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(model_shorts, rotation=10, ha="right")
        ax.set_ylim(0, 1.10)
        ax.set_ylabel(label)
        ax.set_title(f"{label} across models", fontweight="bold")
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015,
                    f"{m:.3f}", ha="center", fontsize=10, fontweight="bold")

        pairs = [(0, 1), (1, 2), (0, 2)] if n_models >= 3 else []
        for a, b in pairs:
            try:
                t_stat, p_val = ttest_rel(summaries[a][fkey], summaries[b][fkey])
                if p_val < 0.001: sig = "***"
                elif p_val < 0.01: sig = "**"
                elif p_val < 0.05: sig = "*"
                else: sig = "n.s."
                delta = summaries[a][mkey] - summaries[b][mkey]
                print(
                    f"  {label:<10s} | "
                    f"{model_shorts[a]:<14s} vs {model_shorts[b]:<14s}: "
                    f"delta={delta:+.3f}  t={t_stat:+.2f}  "
                    f"p={p_val:.3f}  {sig}"
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"  t-test failed: {exc}")
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Three-Backbone Comparison - Classical vs. Quantum Bug Classification\n"
        "RoBERTa-base . CodeBERT . GraphCodeBERT  "
        "(5-fold CV x 5 seeds = 25 folds each)",
        fontsize=13, y=1.04, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, output_dir, "fig_cross_model_comparison", fmt)


def plot_roc_overlay(
    summaries: Sequence[Dict[str, Any]],
    *,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Plot all backbones' pooled ROC curves on a single axis."""
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(7, 7))
    for summary in summaries:
        y_true = np.array(summary["y_true_all"])
        probs = np.array(summary["probs_all"])
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        auc = roc_auc_score(y_true, probs[:, 1])
        color = OVERLAY_COLORS.get(summary["model_short"], "#34495e")
        ax.plot(fpr, tpr, linewidth=2.2, color=color,
                label=f'{summary["model_short"]}  (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.4, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        "ROC curves - three-backbone comparison\n"
        "Pooled across CV folds per model",
        fontsize=12,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return _save(fig, output_dir, "fig_roc_overlay", fmt)


def plot_paired_per_fold(
    summaries: Sequence[Dict[str, Any]],
    *,
    output_dir: str,
    fmt: str = "svg",
) -> str:
    """Plot per-fold accuracy, paired across backbones (gray connector lines)."""
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    model_shorts = [s["model_short"] for s in summaries]
    n_models = len(summaries)
    n_folds = len(summaries[0]["fold_accs"])
    xpos = np.arange(n_models)

    for fold_i in range(n_folds):
        yvals = [s["fold_accs"][fold_i] for s in summaries]
        ax.plot(xpos, yvals, color="gray", alpha=0.25, linewidth=1.0, zorder=1)

    for k, summary in enumerate(summaries):
        ax.scatter(
            [k] * n_folds, summary["fold_accs"], s=60, alpha=0.75,
            edgecolor="white", linewidth=0.6, zorder=3,
            label=(f'{summary["model_short"]} '
                   f'(mean={summary["mean_accuracy"]:.3f})'),
        )
        ax.scatter([k], [summary["mean_accuracy"]], s=160, marker="_",
                   color="black", linewidth=3, zorder=4)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5,
               label="Random (50%)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(model_shorts)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Per-fold accuracy across backbones (paired by fold)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, output_dir, "fig_paired_per_fold", fmt)
