"""Generate figures."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.data_utils import LABEL_LIST


_STYLE_APPLIED = False
PALETTE = {"classical": "#e74c3c", "quantum": "#2980b9", "mean": "#2c3e50"}
OVERLAY_COLORS = {
    "roberta": "#7f8c8d",
    "codebert": "#2980b9",
    "graphcodebert": "#16a085",
}


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


def _save(fig, output_path: str) -> str:
    """Resolve ``output_path``, save, close. Returns the absolute path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str,
    *,
    title: Optional[str] = None,
) -> str:
    """Generate and save a confusion matrix figure."""
    apply_plot_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=False, cmap="Blues",
        xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
        cbar_kws={"label": "Prediction count"},
        linewidths=0.5, linecolor="white", ax=ax,
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j + 0.5, i + 0.40, f"{cm[i, j]}", ha="center",
                    va="center", fontsize=18, fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.62, f"({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)
    ax.set_title(title or "Confusion matrix (pooled across CV folds)",
                 fontsize=12, pad=10)
    ax.tick_params(axis="both", which="both", length=0)
    plt.tight_layout()
    return _save(fig, output_path)


def plot_roc_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    output_path: str,
    *,
    title: Optional[str] = None,
) -> str:
    """Generate and save an ROC curve figure."""
    apply_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2.5, color="#2980b9",
            label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC curve", fontsize=12)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return _save(fig, output_path)


def plot_roc_overlay(
    summaries: Sequence[Dict[str, Any]],
    output_path: str,
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
    ax.set_title("ROC curves - three-backbone comparison", fontsize=12)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    return _save(fig, output_path)


def plot_mean_metrics_bar(
    summaries: Sequence[Dict[str, Any]],
    output_path: str,
) -> str:
    """Render a 3-panel bar chart (accuracy / macro-F1 / ROC-AUC) across models."""
    apply_plot_style()
    model_shorts = [s["model_short"] for s in summaries]
    metric_keys = [
        ("mean_accuracy", "std_accuracy", "Accuracy", "#e74c3c"),
        ("mean_f1_macro", "std_f1_macro", "Macro-F1", "#2980b9"),
        ("mean_roc_auc",  "std_roc_auc",  "ROC-AUC",  "#27ae60"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xpos = np.arange(len(summaries))
    for ax, (mkey, skey, label, color) in zip(axes, metric_keys):
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
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "Three-Backbone Comparison - Classical vs. Quantum Bug Classification\n"
        "RoBERTa-base . CodeBERT . GraphCodeBERT  "
        "(5-fold CV x 5 seeds = 25 folds each)",
        fontsize=13, y=1.04, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, output_path)


def plot_dataset_distribution(
    counts: Dict[str, int],
    output_path: str,
    *,
    title: Optional[str] = None,
) -> str:
    """Plot the per-class count of records as a labeled bar chart."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    keys = list(counts.keys())
    vals = list(counts.values())
    colors = [PALETTE.get(k, "#34495e") for k in keys]
    bars = ax.bar(keys, vals, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8)
    total = sum(vals)
    for bar, v in zip(bars, vals):
        pct = 100 * v / max(total, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                f"{v}\n({pct:.1f}%)", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of records")
    ax.set_title(title or f"Dataset class distribution (N = {total})",
                 fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    return _save(fig, output_path)


def plot_methodology_diagram(output_path: str) -> str:
    """Render a static block-diagram of the fine-tuning pipeline."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    boxes = [
        ("Bug-pattern\nJSON dataset",      0.05, 0.40, "#bdc3c7"),
        ("Text builder\n(name + desc + code)", 0.22, 0.40, "#f1c40f"),
        ("Tokenizer +\nEncoder backbone\n(RoBERTa / CodeBERT /\nGraphCodeBERT)",
                                              0.42, 0.40, "#3498db"),
        ("Classification head\n+ Weighted CE\n+ Label smoothing", 0.66, 0.40, "#9b59b6"),
        ("5x5 CV evaluation\n(Accuracy, Macro-F1,\nROC-AUC)",      0.85, 0.40, "#27ae60"),
    ]
    for text, x, y, c in boxes:
        ax.add_patch(plt.Rectangle(
            (x - 0.07, y - 0.16), 0.14, 0.32,
            facecolor=c, edgecolor="black", linewidth=1.2, alpha=0.85,
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=11,
                fontweight="bold")

    arrow_y = 0.40
    for x_start, x_end in [(0.13, 0.15), (0.30, 0.35), (0.50, 0.59),
                           (0.74, 0.78)]:
        ax.annotate(
            "",
            xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
            arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        )

    ax.text(0.5, 0.92,
            "Fine-Tuning Pipeline: Classical vs. Quantum Bug Classification",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.06,
            "5-fold stratified CV x 5 seeds = 25 fold-runs per backbone "
            "(75 fold-runs total)",
            ha="center", fontsize=11, style="italic", color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    return _save(fig, output_path)
