# Fine-Tuning LLM Models for Bug Detection in Quantum Software

Final project for **CSC 7644: Applied LLM Development**.

## 1. Project Title and Overview

This project fine-tunes three RoBERTa-architecture encoders to perform binary classification of software-engineering bug reports as **classical** or **quantum**. The motivation is concrete: undetected bugs in quantum applications such as quantum key distribution (QKD) can silently compromise cryptographic guarantees, so automated detection of quantum-specific bug patterns is a useful first line of defense for quantum software developers, researchers, and educators.

Because all three backbones share the same architecture and parameter count (~125M), the comparison isolates the effect of *pretraining data* on this downstream task: general-text pretraining vs. code-aware pretraining vs. structure-aware (data-flow) pretraining.

The repository ships the full data, model registry, training loop, evaluation, and plotting code as importable Python modules plus a single CLI entrypoint that reproduces every figure used in the project quad chart.

## 2. Key Features / Capabilities

- **Three-backbone benchmark** - one command fine-tunes and evaluates `roberta-base`, `microsoft/codebert-base`, and `microsoft/graphcodebert-base` under identical hyperparameters.
- **Robust cross-validation** - 5-fold stratified CV repeated across 5 random seeds (25 fold-runs per model) with paired Welch's *t*-tests for cross-backbone significance.
- **Class-imbalance handling** - minority oversampling combined with inverse-frequency weighted cross-entropy and label smoothing.
- **Disk-safe training** - a custom `ManualEarlyStoppingCallback` removes the need for HuggingFace's checkpoint-saving machinery so experiments run end-to-end without filling small Kaggle/Colab disks.
- **Publication-ready figures** - confusion matrix, per-fold scatter, per-model ROC, cross-model bar chart, ROC overlay, and a paired per-fold strip plot are written as 300-dpi PNGs.
- **Reproducible artifacts** - every run also writes per-fold CSVs and a combined JSON so results can be inspected without rerunning training.
- **Clean output layout** - SVG figures land in `figures/`, CSV/JSON tables land in `tables/`, and the HuggingFace Trainer's per-fold scratch lives in `results/` (cleaned automatically at the end of each fold).
- **Sample artifacts shipped** - `scripts/generate_sample_artifacts.py` populates `figures/` and `tables/` with realistic synthetic outputs in seconds, so the layout is visible without paying for a GPU run.

## 3. Tech Stack and Architecture

**Models (HuggingFace Hub):** `roberta-base`, `microsoft/codebert-base`, `microsoft/graphcodebert-base`.

**Frameworks:** PyTorch, HuggingFace `transformers`, `datasets`, `accelerate`; scikit-learn for splits and metrics; pandas / numpy / scipy for aggregation; matplotlib / seaborn for figures.

**High-level component map:**

- `src/config.py` - model registry, label space, and a `TrainingConfig` dataclass that bundles every hyperparameter.
- `src/data_loader.py` - JSON loader and text-builder that concatenates `name`, `description`, and `example_code` per record.
- `src/callbacks.py` - `ManualEarlyStoppingCallback`, `EpochLogCallback`, the `WeightedTrainer` subclass, and the `oversample_to_balance` helper.
- `src/training.py` - `run_fold()` (single fold) and `run_full_experiment()` (full 25-fold CV for one backbone, with per-model figure and JSON output).
- `src/plots.py` - all matplotlib plotting helpers (per-model + cross-model).
- `src/evaluation.py` - cross-model comparison table and combined JSON.
- `main.py` - CLI that wires everything together.
- `notebooks/` - the original Kaggle notebook preserved for reference.

## 4. Setup Instructions

### Prerequisites
- **Python 3.10 or 3.11** (the pinned versions of `transformers==4.44.2` and `scikit-learn==1.5.2` ship wheels for these).
- **OS:** Linux, macOS, or Windows (the project has been tested on Kaggle's Linux T4 environment).
- **Hardware:** an NVIDIA GPU is strongly recommended. CPU runs will complete but a full 75-fold sweep takes several hours.
- `pip` (or `conda`) for installing dependencies.

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch by default. To use a CUDA build pick the matching command from https://pytorch.org/get-started/locally/ *before* running `pip install -r requirements.txt`. For example, for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` to point at your data file and output directory:

| Variable      | Required | Default                              | Purpose                                                                          |
|---------------|----------|--------------------------------------|----------------------------------------------------------------------------------|
| `DATA_PATH`   | no       | `data/bug_patterns_labeled.json`     | Path to the labeled JSON dataset.                                                |
| `OUTPUT_DIR`  | no       | `results`                            | Trainer scratch directory; per-fold temp dirs live here and are auto-cleaned.    |
| `FIGURES_DIR` | no       | `figures`                            | PNG figures (per-model + cross-model).                                           |
| `TABLES_DIR`  | no       | `tables`                             | CSV / JSON result tables (per-model + cross-model).                              |
| `FIGURE_FORMAT` | no     | `svg`                                | Image format for figures (one of `svg`, `png`, `pdf`).                           |
| `HF_HOME`     | no       | system default                       | HuggingFace cache directory (optional).                                          |
| `HF_TOKEN`    | no       | unset                                | Only needed for gated models (none used here).                                   |

No API keys or secrets are required to run the pipeline as shipped. **Never commit `.env` or any secret material** - `.gitignore` already excludes `.env`.

### Dataset format

Each record in the JSON file is expected to have at least:

```json
{
  "name": "CNOT Self-Loop",
  "description": "A CNOT (CX) gate is applied with the same qubit index ...",
  "example_code": "qc.cx(0, 0)  # control == target",
  "bug_category": "quantum"
}
```

Records whose `bug_category` is missing or not in `{"classical", "quantum"}` are filtered out at load time. The repository ships the full uncategorized dataset at `data/bug_patterns_labeled.json`; to run training, point `DATA_PATH` at the categorized version of the dataset (e.g. `data/bug_patterns_categorized.json`) once each record has been labeled `classical` or `quantum`.

## 5. Running the Application

### Full reproducible run (all three backbones, 75 fold-runs)

```bash
python main.py
```

Expected runtime: ~75-110 minutes on a single NVIDIA T4 GPU.

### Quick smoke test (CPU-friendly, ~2-5 minutes)

```bash
python main.py --quick
```

`--quick` reduces the CV protocol to 2 folds x 1 seed and trims epochs to 2 so the wiring can be verified locally before launching a real run.

### Populate `figures/` and `tables/` without training

Sample SVG figures and CSV tables can be generated in seconds without GPUs or any model downloads:

```bash
python scripts/generate_sample_artifacts.py
```

This drives the plotting and aggregation code with realistic synthetic predictions, so reviewers can see the intended layout immediately. Real training results (`python main.py`) overwrite the same filenames, so once a real sweep is finished the sample files are replaced.

### Subset of backbones

```bash
python main.py --models roberta codebert
```

### Custom paths

```bash
python main.py \
    --data-path /path/to/categorized.json \
    --figures-dir /path/to/figures \
    --tables-dir  /path/to/tables \
    --output-dir  /path/to/scratch
```

Each folder serves a different purpose:

- `--figures-dir` / `FIGURES_DIR` (default `figures/`) - all PNG figures.
- `--tables-dir`  / `TABLES_DIR`  (default `tables/`)  - all CSV and JSON result tables.
- `--output-dir`  / `OUTPUT_DIR`  (default `results/`) - trainer scratch only; per-fold temp directories are deleted at the end of every fold.

### Importing the modules

The pipeline pieces are reusable from a notebook:

```python
from src import (
    get_default_config,
    load_labeled_dataset,
    get_text_label_arrays,
    run_full_experiment,
)

config = get_default_config()
labeled = load_labeled_dataset(config)
texts, labels = get_text_label_arrays(labeled, config)
summary = run_full_experiment(
    model_name="microsoft/codebert-base",
    model_short="codebert",
    description="Code + NL bimodal pretraining",
    texts=texts,
    labels=labels,
    config=config,
    n_labeled=len(labeled),
)
```

The original Kaggle notebook is preserved under `notebooks/quantum-vs-classical-bug-prediction-v1.ipynb` for reference and side-by-side comparison with the modular code.

## 6. Repository Organization

```
.
├── main.py                          # CLI entrypoint that runs the full experiment
├── src/
│   ├── __init__.py                  # Re-exports the public API
│   ├── config.py                    # MODEL_REGISTRY, label maps, TrainingConfig
│   ├── data_loader.py               # JSON loading and text construction
│   ├── callbacks.py                 # Early stopping, epoch logging, weighted Trainer
│   ├── training.py                  # run_fold() and run_full_experiment()
│   ├── plots.py                     # Per-model + cross-model matplotlib figures
│   └── evaluation.py                # Comparison table + combined JSON writer
├── data/
│   └── bug_patterns_labeled.json    # Bug-pattern dataset shipped with the repo
├── notebooks/
│   └── quantum-vs-classical-bug-prediction-v1.ipynb   # Original Kaggle notebook
├── scripts/
│   ├── run_experiment.sh            # Convenience wrapper around python main.py
│   └── generate_sample_artifacts.py # Populate figures/ + tables/ without training
├── tests/
│   ├── test_config.py               # Defaults + env-var overrides
│   ├── test_data_loader.py          # JSON loading + text builder
│   └── test_callbacks.py            # Oversampling + metric helpers
├── docs/
│   ├── architecture.md              # Module overview and data flow
│   └── results.md                   # What every output file contains
├── figures/                         # SVG figures produced by main.py
├── tables/                          # CSV / JSON result tables produced by main.py
├── results/                         # Trainer scratch (per-fold temp dirs, gitignored)
├── pyproject.toml                   # Package metadata + dev tooling configuration
├── config.example.yaml              # Annotated reference for every TrainingConfig field
├── requirements.txt                 # Pinned dependencies
├── .env.example                     # Template for local environment variables
├── .gitignore
├── LICENSE
├── Rahman_MdSaidur_quadchart.pdf    # Final-project quad chart
└── README.md                        # This file
```

A grader looking for a specific concern should find it in:

- **Data handling** -> `src/data_loader.py`
- **Model selection / hyperparameters** -> `src/config.py`
- **Training logic / loss** -> `src/training.py` + `src/callbacks.py`
- **Metrics / aggregation** -> `src/training.py` + `src/evaluation.py`
- **Figures** -> `src/plots.py`
- **CLI** -> `main.py`

## 7. Attributions and Citations

- **HuggingFace Transformers** - the `Trainer` API, callback contract, and `DataCollatorWithPadding` follow the official documentation at https://huggingface.co/docs/transformers. The `ManualEarlyStoppingCallback` was written from scratch but is conceptually a stripped-down rewrite of `transformers.EarlyStoppingCallback` adapted to work with `save_strategy='no'`.
- **Pretrained backbones**:
  - Liu, Y. et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692 (2019).
  - Feng, Z. et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.* EMNLP 2020. (`microsoft/codebert-base`)
  - Guo, D. et al. *GraphCodeBERT: Pre-training Code Representations with Data Flow.* ICLR 2021. (`microsoft/graphcodebert-base`)
- **Cross-validation, paired t-tests, and metric definitions** - standard scikit-learn (https://scikit-learn.org/) and SciPy (https://scipy.org/) APIs.
- **Plot styling** - color palette and the per-fold scatter layout are inspired by the quad-chart conventions used in CSC 7644.
- **Course materials** - the project scope, deliverables, and rubric follow the CSC 7644: Applied LLM Development assignment specification.

This codebase was originally developed as a Kaggle notebook (`notebooks/quantum-vs-classical-bug-prediction-v1.ipynb`) and subsequently refactored into the modular package documented above. No external code was copied verbatim; all snippets adapted from documentation or tutorials are cited above.
