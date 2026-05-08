# Fine-Tuning LLMs for Bug Detection in Quantum Software

## Project Title and Overview

This repository contains my final project for **CSC 7644: Applied LLM Development** at Louisiana State University.

The project builds and evaluates a supervised LLM-based classifier for distinguishing between **classical software bugs** and **quantum-specific bugs** in Qiskit-centered bug reports. Quantum software can contain domain-specific defects such as incorrect qubit mappings, gate misuse, measurement-ordering errors, and circuit-construction mistakes. These bugs are difficult to triage manually because they require both software-engineering knowledge and quantum-computing domain expertise.

The goal of this project is to determine whether code-aware pre-training improves bug classification performance. To test this, I fine-tuned and compared three RoBERTa-architecture encoder models:

- `roberta-base`
- `microsoft/codebert-base`
- `microsoft/graphcodebert-base`

The comparison holds architecture and parameter count constant, so the main experimental variable is the model's pre-training corpus rather than model size or architecture.

## Key Features / Capabilities

- Classifies Qiskit bug reports as either **classical** or **quantum-specific**.
- Compares three pre-trained encoder backbones: RoBERTa, CodeBERT, and GraphCodeBERT.
- Uses stratified 5-fold cross-validation repeated across 5 random seeds.
- Produces 25 fold-runs per model and 75 total fold-runs.
- Handles class imbalance using minority oversampling and class-weighted cross-entropy.
- Uses label smoothing and macro-F1 early stopping for more stable training.
- Avoids excessive checkpoint storage with a custom disk-safe early-stopping callback.
- Computes accuracy, macro-F1, ROC-AUC, confusion matrices, and statistical tests.
- Generates reproducible tables and figures for the final report.
- Provides an end-to-end Jupyter notebook for reproducing the full experiment.

## Tech Stack and Architecture (High-Level)

### Tech Stack

| Component | Tools / Libraries |
|---|---|
| Deep learning | PyTorch |
| Transformer models | HuggingFace Transformers |
| Dataset handling | HuggingFace Datasets, pandas, NumPy |
| Metrics and cross-validation | scikit-learn |
| Statistical testing | SciPy |
| Visualization | Matplotlib, Seaborn |
| Notebook environment | Jupyter / Kaggle |
| Testing | pytest |

### Model Backbones

| Short Name | HuggingFace Identifier | Pre-training Domain | Role |
|---|---|---|---|
| RoBERTa | `roberta-base` | Natural language | Control model |
| CodeBERT | `microsoft/codebert-base` | Code + natural language | Code-aware encoder |
| GraphCodeBERT | `microsoft/graphcodebert-base` | Code + natural language + data-flow information | Structure-aware encoder |

### High-Level Architecture

The project follows a supervised fine-tuning and evaluation pipeline:

1. Bug-pattern records are loaded from the `data/` folder.
2. Each record is converted into a canonical text input using fields such as bug name, description, and example code.
3. The input text is tokenized using the selected HuggingFace tokenizer.
4. A transformer encoder is fine-tuned for binary classification.
5. Minority oversampling and class-weighted loss are applied to reduce class imbalance.
6. The model is evaluated using repeated stratified cross-validation.
7. Metrics, predictions, tables, and figures are saved to `results/`, `tables/`, and `figures/`.

The main components are:

- `data/` - input bug-pattern data
- `src/data_utils.py` - data loading and input construction
- `src/model_utils.py` - model and tokenizer loading
- `src/train_utils.py` - training helpers, oversampling, weighted loss, and early stopping
- `src/evaluate.py` - metric computation
- `src/plotting.py` - figure generation
- `notebooks/` - full end-to-end experimental notebook
- `scripts/` - standalone scripts for training, evaluation, and figure generation

## Setup Instructions

### Prerequisites

The project assumes the following:

- Python 3.10 or 3.11
- pip or conda
- Git
- Jupyter Notebook
- An NVIDIA GPU is recommended for full training

The final experiments were run in a GPU environment. CPU execution is possible for lightweight tests, but full cross-validation training may take several hours.

### Clone the Repository

```bash
git clone https://github.com/mrahm65/CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software.git
cd CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software
```

### Create a Virtual Environment

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, follow the official PyTorch installation command for your system:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

The shipped pipeline does **not** require API keys. Optional environment variables are included only for future extensions.

| Variable            | Purpose                                                                      |
| ------------------- | ---------------------------------------------------------------------------- |
| `OPENAI_API_KEY`    | Optional; only needed for downstream LLM-as-judge or prompt-based extensions |
| `HUGGINGFACE_TOKEN` | Optional; only needed for gated HuggingFace models                           |

The `.gitignore` file excludes `.env`, so real secrets should remain local and should not be committed.

## Running the Application

### Option 1: Run the Full Notebook

The main end-to-end implementation is in:

```text
notebooks/quantum-vs-classical-bug-prediction-v11.ipynb
```

Start Jupyter with:

```bash
jupyter notebook notebooks/quantum-vs-classical-bug-prediction-v11.ipynb
```

The notebook performs:

1. Environment setup
2. Data loading
3. Input preprocessing
4. Model/tokenizer loading
5. Training
6. Cross-validation
7. Evaluation
8. Statistical testing
9. Figure generation
10. Final result reporting

### Option 2: Run Scripts

Run full training:

```bash
python scripts/run_training.py
```

Inspect saved evaluation metrics:

```bash
python scripts/run_evaluation.py --tables-dir tables
```

Regenerate report figures:

```bash
python scripts/generate_figures.py
```

### Run Tests

```bash
TMPDIR=/tmp pytest tests/ -q
```

The lightweight tests check data-loading and metric-computation utilities without requiring full model training.

## Repository Organization

The repository is organized so a grader can quickly locate the implementation, data, documentation, results, and evaluation artifacts.

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

### Root Files

| File               | Description                                                                             |
| ------------------ | --------------------------------------------------------------------------------------- |
| `README.md`        | Main project documentation, setup instructions, usage guide, results, and citations     |
| `requirements.txt` | Python dependencies required to run the project                                         |
| `.gitignore`       | Prevents local environments, cache files, checkpoints, and secrets from being committed |
| `.env.example`     | Template for optional environment variables                                             |
| `LICENSE`          | Repository license                                                                      |
| `main.py`          | Lightweight entry point that prints basic project usage information                     |
| `pyproject.toml`   | Project metadata and optional formatting/testing configuration                          |

### Main Directories

| Directory    | Description                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------- |
| `data/`      | Bug-pattern dataset files and sample data for demonstration                                          |
| `docs/`      | Final report, LaTeX source, project summary, methodology notes, and architecture notes               |
| `figures/`   | Report-ready figures such as dataset distribution, workflow diagram, confusion matrix, and ROC curve |
| `notebooks/` | Main Jupyter notebook containing the full end-to-end experimental pipeline                           |
| `scripts/`   | Standalone scripts for training, evaluation, and figure generation                                   |
| `src/`       | Reusable Python modules for data loading, model utilities, training, evaluation, and plotting        |
| `tables/`    | CSV tables containing summary metrics, statistical tests, and model comparisons                      |
| `results/`   | Prediction files, fold-level metrics, and final summary outputs                                      |
| `tests/`     | Lightweight pytest tests for core utility functions                                                  |

### Important Source Files

| File                          | Purpose                                                                               |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| `src/data_utils.py`           | Loads bug-pattern data and builds canonical text inputs                               |
| `src/model_utils.py`          | Loads HuggingFace tokenizers and sequence-classification models                       |
| `src/train_utils.py`          | Implements training helpers, class weighting, oversampling, and manual early stopping |
| `src/evaluate.py`             | Computes accuracy, macro-F1, ROC-AUC, and related evaluation metrics                  |
| `src/plotting.py`             | Generates figures used in the final report                                            |
| `scripts/run_training.py`     | Runs the full training and cross-validation pipeline                                  |
| `scripts/run_evaluation.py`   | Loads saved outputs and computes/prints final evaluation summaries                    |
| `scripts/generate_figures.py` | Regenerates report-ready figures from saved tables/results                            |

## Attributions and Citations

This project uses the following external tools, models, and research resources:

- HuggingFace Transformers: <https://huggingface.co/docs/transformers>
- PyTorch: <https://pytorch.org/>
- scikit-learn: <https://scikit-learn.org/>
- RoBERTa: Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019.
- CodeBERT: Feng et al., *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*, EMNLP 2020.
- GraphCodeBERT: Guo et al., *GraphCodeBERT: Pre-training Code Representations with Data Flow*, ICLR 2021.
- Bugs4Q: Zhao et al., *Bugs4Q: A Benchmark of Real Bugs for Quantum Programs*, ASE 2021.
- QBugs / Bugs-QCP-related resources: Campos and Souto, *QBugs: A Collection of Reproducible Bugs in Quantum Algorithms*, 2021.

No external code was copied verbatim. The custom `ManualEarlyStoppingCallback` in `src/train_utils.py` was implemented as a simplified, disk-safe adaptation inspired by HuggingFace's early-stopping behavior. This was necessary because standard checkpoint-based early stopping exceeded the available disk quota during repeated cross-validation.

## Author

**Md Saidur Rahman**
Department of Computer Science
Louisiana State University
CSC 7644: Applied LLM Development
Email: `mrahm65@lsu.edu`
