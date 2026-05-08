# Fine-Tuning LLMs for Bug Detection in Quantum Software

## Overview

This is the final project for **CSC 7644: Applied LLM Development**
(Louisiana State University, Spring 2026). The goal is to classify
software-engineering bug reports as **classical** or **quantum-specific**
by fine-tuning three RoBERTa-architecture encoders. Undetected bugs in
quantum applications such as quantum key distribution (QKD) can silently
compromise cryptographic guarantees, so automated detection of
quantum-specific bug patterns is a useful first line of defense for
quantum software developers, researchers, and educators.

The comparison holds architecture and parameter count constant
(~125M parameters per backbone) so any performance difference comes from
**pretraining data**, not capacity or architecture.

## Key Features

- Three-backbone benchmark across `roberta-base`,
  `microsoft/codebert-base`, and `microsoft/graphcodebert-base`.
- 5-fold stratified cross-validation repeated across 5 seeds (25
  fold-runs per backbone, 75 fold-runs total).
- Class-imbalance handling via minority oversampling and
  inverse-frequency weighted cross-entropy with label smoothing.
- Disk-safe training (custom `ManualEarlyStoppingCallback` removes the
  need for HuggingFace checkpoint files).
- Reproducible artifacts: PNG figures in `figures/`, CSV / JSON tables
  in `tables/` and `results/`.
- A standalone Jupyter notebook
  (`notebooks/quantum-vs-classical-bug-prediction-v11.ipynb`) that
  reproduces every result end-to-end.

## Tech Stack

| Concern              | Library / tool                                     |
|----------------------|----------------------------------------------------|
| Deep learning        | PyTorch, HuggingFace `transformers`, `datasets`, `accelerate` |
| Splits / metrics     | scikit-learn                                       |
| Statistical testing  | SciPy (`ttest_rel`)                                |
| Tabular processing   | pandas, numpy                                      |
| Figures              | matplotlib, seaborn                                |
| Notebook environment | Jupyter                                            |
| Testing              | pytest                                             |

## Model Backbones

| Short           | HF identifier                  | Pretraining domain                | Role                  |
|-----------------|--------------------------------|-----------------------------------|-----------------------|
| `roberta`       | `roberta-base`                 | English text only                 | Control               |
| `codebert`      | `microsoft/codebert-base`      | Code + NL bimodal                 | Code-aware encoder    |
| `graphcodebert` | `microsoft/graphcodebert-base` | Code + NL + data-flow graphs      | Structure-aware       |

## Dataset

- 233 labeled bug-pattern records (134 classical, 99 quantum) derived
  from Bugs-QCP and related quantum bug benchmark resources.
- Class imbalance ratio ~1.35 : 1.
- Each record is converted into a single training string via
  `name + "\n" + description + "\n" + example_code` (empty fields
  skipped). See `src/data_utils.py::build_input_text`.
- `data/bug_patterns_labeled.json` ships the full catalog;
  `data/sample_data.json` is a tiny excerpt for quick smoke tests.

## Repository Organization

```text
CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .env.example
|-- LICENSE
|-- main.py
|-- pyproject.toml
|
|-- data/
|-- docs/
|-- figures/
|-- notebooks/
|-- scripts/
|-- src/
|-- tables/
|-- tests/
`-- results/
```

| Folder       | Contents                                                                          |
|--------------|-----------------------------------------------------------------------------------|
| `data/`      | `bug_patterns_labeled.json`, `sample_data.json`, `README.md`.                     |
| `docs/`      | Final report (PDF + LaTeX), `project_summary.md`, `methodology.md`, `architecture.md`. |
| `figures/`   | The five canonical PNG figures plus per-backbone SVG companions.                  |
| `notebooks/` | `quantum-vs-classical-bug-prediction-v11.ipynb` (full end-to-end pipeline).       |
| `scripts/`   | `run_training.py`, `run_evaluation.py`, `generate_figures.py`.                    |
| `src/`       | `data_utils.py`, `model_utils.py`, `train_utils.py`, `evaluate.py`, `plotting.py`. |
| `tables/`    | `summary_metrics.csv`, `statistical_tests.csv`, `model_comparison.csv` + per-fold/per-model files. |
| `results/`   | `predictions_<short>.csv`, `fold_metrics.csv`, `final_summary.json`.              |
| `tests/`     | Pytest suite (`test_data_utils.py`, `test_evaluate.py`, ...).                     |

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11.
- pip (or conda) for installing dependencies.
- An NVIDIA GPU is strongly recommended for training. CPU runs work but
  a full 75-fold sweep takes hours.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA build of PyTorch pick the matching command from
<https://pytorch.org/get-started/locally/> *before* running the line
above. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Configure environment variables

```bash
cp .env.example .env
```

Then fill in `.env` if you need API tokens for downstream tooling.
The pipeline as shipped does **not** require any API keys.

| Variable             | Purpose                                                                    |
|----------------------|----------------------------------------------------------------------------|
| `OPENAI_API_KEY`     | Optional - only needed for downstream LLM-as-judge tooling.                |
| `HUGGINGFACE_TOKEN`  | Optional - only needed for gated models (none used here).                  |

`.gitignore` already excludes `.env`, so secrets stay local.

## Running the Project

### Notebook (full end-to-end pipeline)

```bash
jupyter notebook notebooks/quantum-vs-classical-bug-prediction-v11.ipynb
```

The notebook contains every step from environment setup to figure
generation.

### Scripts

```bash
# Full reproducible training (GPU recommended; ~75-110 min on a T4)
python scripts/run_training.py

# Inspect saved metrics without rerunning training
python scripts/run_evaluation.py --tables-dir tables

# Regenerate the report-ready PNG figures
python scripts/generate_figures.py
```

### Tests

```bash
TMPDIR=/tmp pytest tests/ -q
```

The lightweight tests (`test_data_utils.py`, `test_evaluate.py`,
`test_schemas.py`, `test_dataset.py`) run without torch / transformers.

## Results

| Model         | Accuracy        | Macro-F1        | ROC-AUC         | Pooled Acc | Pooled F1 |
|---------------|-----------------|-----------------|-----------------|-----------:|----------:|
| RoBERTa       | 0.764 +/- 0.061 | 0.754 +/- 0.066 | 0.858 +/- 0.048 |      0.764 |     0.756 |
| CodeBERT      | 0.767 +/- 0.057 | 0.763 +/- 0.056 | 0.855 +/- 0.044 |      0.767 |     0.764 |
| GraphCodeBERT | 0.758 +/- 0.058 | 0.756 +/- 0.058 | 0.860 +/- 0.047 |      0.758 |     0.757 |

Paired Welch's *t*-tests across the 25 matched folds find **no
statistically significant difference** between any pair of backbones on
accuracy, macro-F1, or ROC-AUC (every *p* > 0.3). On 233 labeled
samples, code-aware (CodeBERT) and structure-aware (GraphCodeBERT)
pretraining do not yield a measurable advantage over English-only
RoBERTa for this binary task.

The full per-pair test results are in
[`tables/statistical_tests.csv`](tables/statistical_tests.csv); the
canonical figures are in [`figures/`](figures/).

## Scope Changes from Midterm Proposal

The midterm proposal scoped a fine-tuning + RAG comparison. After
exploration we narrowed the deliverable to a controlled three-backbone
fine-tuning study because:

1. The labeled dataset is small (233 records); a controlled architecture-
   matched comparison gives sharper conclusions than chasing additional
   tooling.
2. Holding architecture and parameter count constant (all three are
   125M-parameter RoBERTa encoders) isolates the *pretraining-data*
   variable, which directly answers the question the proposal raised
   about quantum-specific signal.
3. The final write-up still covers retrieval-augmented generation as
   future work in the discussion section.

## Reproducibility

- Pinned dependency versions in `requirements.txt` and `pyproject.toml`.
- All seeds (CV split, `transformers.set_seed`, oversampling RNG) are
  derived from the per-fold seed, so `python scripts/run_training.py`
  produces bit-identical metrics on the same hardware.
- `save_strategy="no"` avoids writing fold checkpoints; per-fold
  scratch directories under `results/` are deleted at the end of each
  fold so disk usage stays bounded.
- Tests in `tests/` exercise the data-loading and metric paths that are
  hardest to verify by eye.

## Attributions and Citations

- HuggingFace Transformers documentation: <https://huggingface.co/docs/transformers>.
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
  (arXiv:1907.11692, 2019).
- Feng et al., *CodeBERT: A Pre-Trained Model for Programming and
  Natural Languages* (EMNLP 2020).
- Guo et al., *GraphCodeBERT: Pre-training Code Representations with
  Data Flow* (ICLR 2021).
- scikit-learn (<https://scikit-learn.org/>) for cross-validation, paired
  *t*-tests, and metric definitions.
- Bugs-QCP and the broader Qiskit bug-pattern community for the source
  data; see `data/README.md` for redistribution constraints.

No external code was copied verbatim. The `ManualEarlyStoppingCallback`
in `src/train_utils.py` is a stripped-down rewrite of
`transformers.EarlyStoppingCallback` adapted to work with
`save_strategy='no'`.

## Author

**Md Saidur Rahman**
- Email (course): `mrahm65@lsu.edu`
- Email (alt):    `saidur@eub.edu.bd`
- Course:         CSC 7644 - Applied LLM Development, Spring 2026
- Institution:    Louisiana State University
