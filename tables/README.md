# Tables

This folder contains table files generated from the final experiments.

## Files

| File                          | Description                                                                |
|-------------------------------|----------------------------------------------------------------------------|
| `summary_metrics.csv`         | Mean accuracy, macro-F1, and ROC-AUC for each backbone.                    |
| `statistical_tests.csv`       | Paired statistical test results between every backbone pair.               |
| `model_comparison.csv`        | Model backbone comparison table (mean +/- std + pooled metrics).           |
| `fold_metrics.csv`            | Per-fold (5 seeds x 5 folds) metrics for each backbone.                    |
| `per_fold_<short>.csv`        | Per-fold metrics for a single backbone, written by the training script.   |
| `results_<short>.json`        | Full per-fold predictions + aggregate metrics for a single backbone.       |
| `cross_model_comparison.csv`  | Side-by-side `mean +/- std` summary across backbones.                      |
| `cross_model_results.json`    | Combined results JSON across backbones.                                    |

## Headline numbers

| Model         | Accuracy        | Macro-F1        | ROC-AUC         |
|---------------|-----------------|-----------------|-----------------|
| roberta       | 0.764 +/- 0.061 | 0.754 +/- 0.066 | 0.858 +/- 0.048 |
| codebert      | 0.767 +/- 0.057 | 0.763 +/- 0.056 | 0.855 +/- 0.044 |
| graphcodebert | 0.758 +/- 0.058 | 0.756 +/- 0.058 | 0.860 +/- 0.047 |
