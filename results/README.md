# Results

This folder contains result artifacts from the final experiments.

## Files

| File                              | Description                                            |
|-----------------------------------|--------------------------------------------------------|
| `fold_metrics.csv`                | Per-fold metrics across all seeds and backbones.       |
| `final_summary.json`              | Aggregate metrics (means, stds, pooled).               |
| `predictions_codebert.csv`        | CodeBERT per-sample predictions.                       |
| `predictions_roberta.csv`         | RoBERTa per-sample predictions.                        |
| `predictions_graphcodebert.csv`   | GraphCodeBERT per-sample predictions.                  |

## Headline results

| Model         | Accuracy        | Macro-F1        | ROC-AUC         |
|---------------|-----------------|-----------------|-----------------|
| RoBERTa       | 0.764 +/- 0.061 | 0.754 +/- 0.066 | 0.858 +/- 0.048 |
| CodeBERT      | 0.767 +/- 0.057 | 0.763 +/- 0.056 | 0.855 +/- 0.044 |
| GraphCodeBERT | 0.758 +/- 0.058 | 0.756 +/- 0.058 | 0.860 +/- 0.047 |

Paired Welch's *t*-tests across the matched 25 folds show no
statistically significant difference between any pair on accuracy,
macro-F1, or ROC-AUC (every *p* > 0.3). See
`tables/statistical_tests.csv` for full details.
