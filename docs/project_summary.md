# Project Summary

## Title

Fine-Tuning LLMs for Bug Detection in Quantum Software.

## Course

CSC 7644 - Applied LLM Development, Louisiana State University.

## One-paragraph description

This project fine-tunes three RoBERTa-architecture encoders
(`roberta-base`, `microsoft/codebert-base`, `microsoft/graphcodebert-base`)
to perform binary classification of software-engineering bug reports as
*classical* or *quantum*. The comparison holds architecture and parameter
count constant (~125M) so the only variable is *pretraining data*. We
evaluate with 5-fold stratified cross-validation repeated across 5 random
seeds (25 fold-runs per model) and measure accuracy, macro-F1, and
ROC-AUC. Class imbalance is handled with minority oversampling and
inverse-frequency weighted cross-entropy.

## Headline result

All three backbones converge to ~76% accuracy. Paired Welch's t-tests
across the matched 25 folds show no statistically significant difference
between any pair on accuracy, macro-F1, or ROC-AUC (every p > 0.3). On
233 labeled samples, code-aware and structure-aware pretraining do not
yield a measurable advantage over English-only RoBERTa for this task.

| Model         | Accuracy        | Macro-F1        | ROC-AUC         |
|---------------|-----------------|-----------------|-----------------|
| roberta       | 0.764 +/- 0.061 | 0.754 +/- 0.066 | 0.858 +/- 0.048 |
| codebert      | 0.767 +/- 0.057 | 0.763 +/- 0.056 | 0.855 +/- 0.044 |
| graphcodebert | 0.758 +/- 0.058 | 0.756 +/- 0.058 | 0.860 +/- 0.047 |
