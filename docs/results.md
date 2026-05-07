# Results files

Each run of `python main.py` writes the artifacts below.

## Per-model figures (`figures/`)

* `fig1_confusion_matrix_<short>.<fmt>` — pooled confusion matrix with
  cell counts and percentages.
* `fig2_fold_distribution_<short>.<fmt>` — per-fold accuracy / macro-F1 /
  ROC-AUC scatter with mean and ±1 SD band.
* `fig3_roc_curve_<short>.<fmt>` — per-fold ROC curves with the mean and
  a ±1 SD band.

## Cross-model figures (`figures/`)

* `fig_cross_model_comparison.<fmt>` — three-metric bar chart with
  paired-t-test outputs printed to stdout.
* `fig_roc_overlay.<fmt>` — pooled ROC for every backbone, overlaid.
* `fig_paired_per_fold.<fmt>` — per-fold accuracy connected across
  models (paired by fold index).

`<fmt>` is controlled by `--figure-format` / `FIGURE_FORMAT`
(default `svg`).

## Tables (`tables/`)

* `per_fold_<short>.csv` — the per-fold metric table for one backbone.
* `results_<short>.json` — full predictions + summary stats for one
  backbone (heavy file).
* `cross_model_comparison.csv` — side-by-side `mean ± std` comparison
  across all backbones.
* `cross_model_results.json` — combined summary (drops the heavy
  per-sample arrays; those live in the per-model JSON files).

## Sampling without training

If you want to populate the folders without paying the GPU cost, run::

    python scripts/generate_sample_artifacts.py

This drives the plotting and aggregation code with realistic synthetic
predictions so you can see the intended layout. Real training results
will overwrite the sample files when you run `python main.py`.
