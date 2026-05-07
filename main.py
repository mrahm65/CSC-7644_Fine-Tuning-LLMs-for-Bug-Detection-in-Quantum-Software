"""End-to-end CLI entrypoint: run the full three-backbone CV experiment.

Usage::

    python main.py
    python main.py --data-path data/bug_patterns_categorized.json
    python main.py --figures-dir figures --tables-dir tables
    python main.py --models roberta codebert
    python main.py --quick   # smoke-test with 2 folds and 1 seed

Outputs are split across two folders:

* ``--figures-dir`` (default ``./figures``):
    * ``fig1_confusion_matrix_<short>.png``
    * ``fig2_fold_distribution_<short>.png``
    * ``fig3_roc_curve_<short>.png``
    * ``fig_cross_model_comparison.png``
    * ``fig_roc_overlay.png``
    * ``fig_paired_per_fold.png``

* ``--tables-dir`` (default ``./tables``):
    * ``results_<short>.json`` and ``per_fold_<short>.csv``
    * ``cross_model_comparison.csv`` and ``cross_model_results.json``

The ``--output-dir`` (default ``./results``) is reserved for the
HuggingFace Trainer's per-fold scratch directories, which are deleted
at the end of every fold to keep disk usage bounded.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence

import torch

from src.config import MODEL_REGISTRY, TrainingConfig, get_default_config
from src.data_loader import get_text_label_arrays, load_labeled_dataset
from src.evaluation import build_comparison_table, save_cross_model_results
from src.plots import (
    apply_plot_style,
    plot_cross_model_comparison,
    plot_paired_per_fold,
    plot_roc_overlay,
)
from src.training import run_full_experiment


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Three-backbone comparison for classical vs. quantum bug "
            "classification."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "Path to the labeled bug-pattern JSON. Falls back to "
            "$DATA_PATH or data/bug_patterns_labeled.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Trainer scratch directory (per-fold temp dirs live here). "
            "Defaults to $OUTPUT_DIR or ./results."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help=(
            "Where to write PNG figures. Defaults to $FIGURES_DIR or "
            "./figures."
        ),
    )
    parser.add_argument(
        "--tables-dir",
        default=None,
        help=(
            "Where to write CSV/JSON result tables. Defaults to "
            "$TABLES_DIR or ./tables."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=[m["short"] for m in MODEL_REGISTRY],
        help=(
            "Subset of backbones to run. Defaults to all three "
            "(roberta, codebert, graphcodebert)."
        ),
    )
    parser.add_argument(
        "--figure-format",
        default=None,
        choices=["svg", "png", "pdf"],
        help=(
            "Image format for plot outputs. Defaults to $FIGURE_FORMAT or "
            "svg."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke-test: 2 folds, 1 seed, 2 epochs. Useful for verifying "
            "the pipeline locally before launching a full run."
        ),
    )
    return parser.parse_args(argv)


def select_registry(short_names: Optional[List[str]]):
    """Filter ``MODEL_REGISTRY`` down to the requested ``short_names`` (in order)."""
    if not short_names:
        return MODEL_REGISTRY
    keep = set(short_names)
    return [m for m in MODEL_REGISTRY if m["short"] in keep]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the experiment end-to-end and return a process exit code."""
    args = parse_args(argv)

    config: TrainingConfig = get_default_config()
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.figures_dir:
        config.figures_dir = args.figures_dir
    if args.tables_dir:
        config.tables_dir = args.tables_dir
    if args.figure_format:
        config.figure_format = args.figure_format
    if args.quick:
        config.n_folds = 2
        config.cv_seeds = (42,)
        config.num_epochs = 2

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    os.makedirs(config.tables_dir, exist_ok=True)
    apply_plot_style()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------- Load + prep data ------------------------------------------
    labeled = load_labeled_dataset(config)
    texts, labels = get_text_label_arrays(labeled, config)
    print(f"Samples         : {len(texts)}")
    print(
        "Avg text length : "
        f"{sum(len(t) for t in texts) / len(texts):.0f} chars"
    )

    registry = select_registry(args.models)
    print("\nModels to compare:")
    for entry in registry:
        print(f"  - {entry['short']:<14s}  ({entry['name']})")
    print(
        f"CV protocol: {config.n_folds}-fold x {len(config.cv_seeds)} "
        f"seeds = {config.total_folds_per_model} folds per model"
    )
    print(
        f"Total fold-runs: "
        f"{config.total_folds_per_model * len(registry)}"
    )

    # ---------- Per-backbone experiments ----------------------------------
    summaries = []
    for entry in registry:
        summary = run_full_experiment(
            model_name=entry["name"],
            model_short=entry["short"],
            description=entry["description"],
            texts=texts,
            labels=labels,
            config=config,
            n_labeled=len(labeled),
        )
        summaries.append(summary)
        print(
            f"\n[OK] Completed: {entry['short']} "
            f"(acc={summary['mean_accuracy']:.3f}, "
            f"f1={summary['mean_f1_macro']:.3f}, "
            f"auc={summary['mean_roc_auc']:.3f})"
        )

    print("\n" + "=" * 72)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 72)

    # ---------- Cross-model artifacts -------------------------------------
    if len(summaries) >= 2:
        comparison_df = build_comparison_table(summaries)
        print(
            "\nCross-model comparison "
            "(mean +/- std across CV folds):\n"
        )
        print(comparison_df.to_string(index=False))

        plot_cross_model_comparison(
            summaries,
            output_dir=config.figures_dir,
            fmt=config.figure_format,
        )
        plot_roc_overlay(
            summaries,
            output_dir=config.figures_dir,
            fmt=config.figure_format,
        )
        plot_paired_per_fold(
            summaries,
            output_dir=config.figures_dir,
            fmt=config.figure_format,
        )

    save_cross_model_results(
        summaries,
        n_samples=len(labeled),
        cv_setup=(
            f"{config.n_folds}-fold x {len(config.cv_seeds)} "
            f"seeds = {config.total_folds_per_model} folds per model"
        ),
        output_dir=config.tables_dir,
    )

    print(f"\nFigures saved to: {config.figures_dir}")
    print(f"Tables  saved to: {config.tables_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
