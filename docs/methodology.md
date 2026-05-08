# Methodology

## Task

Binary classification of software-engineering bug reports as `classical`
or `quantum`. Each record is a free-text description with an optional
code snippet; the classifier must decide which label applies.

## Dataset

- 233 labeled bug-pattern records (134 classical, 99 quantum).
- Imbalance ratio ~1.35 : 1, addressed via minority oversampling and
  inverse-frequency weighted cross-entropy.
- Each record contributes a single training string built from
  `name + "\n" + description + "\n" + example_code` (empty fields
  skipped).

## Backbones compared

| Short           | HF identifier                  | Pretraining domain          |
|-----------------|--------------------------------|-----------------------------|
| `roberta`       | `roberta-base`                 | English text only           |
| `codebert`      | `microsoft/codebert-base`      | Code + NL bimodal           |
| `graphcodebert` | `microsoft/graphcodebert-base` | Code + NL + data-flow graph |

All three are 125M-parameter RoBERTa-architecture encoders, so
performance differences isolate the impact of *pretraining data*, not
capacity or architecture.

## Cross-validation protocol

- 5-fold stratified CV repeated across 5 random seeds = 25 fold-runs
  per backbone (75 fold-runs total).
- Each fold:
    - Stratified 90/10 internal val split (used for early stopping).
    - Minority class oversampled to match majority count.
    - Inverse-frequency CE + label smoothing (0.05).
    - Manual early stopping on macro-F1 (patience 4).
- Hyperparameters held constant across all backbones (see
  `src/study_i/schemas.py::TrainingConfig`).

## Evaluation

- Per-fold metrics: accuracy, macro-F1, weighted-F1, ROC-AUC.
- Pooled metrics computed on the concatenated predictions across all 25
  folds.
- Cross-model significance via paired Welch's *t*-tests on the matched
  fold-accuracy / F1 / AUC vectors.

## Reproducibility

- Random seeds drive both the CV split and `transformers.set_seed`.
- Pinned dependencies in `requirements.txt` and `pyproject.toml`.
- `save_strategy="no"` keeps disk usage bounded; trainer scratch dirs
  are deleted at the end of each fold.
