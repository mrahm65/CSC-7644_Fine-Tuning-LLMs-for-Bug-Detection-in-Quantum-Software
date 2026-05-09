# Fine-Tuning LLMs for Bug Detection in Quantum Software

## Project Title and Overview

This repository contains my final project for **CSC 7644: Applied LLM Development** at Louisiana State University.

The project builds and evaluates a supervised LLM-based classifier for distinguishing between **classical software bugs** and **quantum-specific bugs** in Qiskit-centered bug reports. Quantum software can contain domain-specific defects such as incorrect qubit mappings, gate misuse, measurement-ordering errors, and circuit-construction mistakes. These bugs are difficult to triage manually because they require both software-engineering knowledge and quantum-computing domain expertise.

The goal of this project is to determine whether code-aware pre-training improves bug classification performance. To test this, I fine-tuned and compared three RoBERTa-architecture encoder models:

- `roberta-base`
- `microsoft/codebert-base`
- `microsoft/graphcodebert-base`

The comparison holds architecture and parameter count constant, so the main experimental variable is the model's pre-training corpus rather than model size or architecture.

---

## Key Features / Capabilities

- Classifies Qiskit bug reports as either **classical** or **quantum-specific**.
- Compares three pre-trained encoder backbones: RoBERTa, CodeBERT, and GraphCodeBERT.
- Uses stratified 5-fold cross-validation repeated across 5 random seeds.
- Produces 25 fold-runs per model and 75 total fold-runs.
- Handles class imbalance using minority oversampling and class-weighted cross-entropy.
- Uses label smoothing and macro-F1 early stopping for more stable training.
- Avoids excessive checkpoint storage with a custom disk-safe early-stopping callback.
- Computes accuracy, macro-F1, ROC-AUC, confusion matrices, and paired statistical tests.
- Generates reproducible tables and figures for the final report.
- Provides an end-to-end Jupyter notebook for reproducing the full experiment.

---

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

```text
Bug-pattern dataset
        |
        v
Canonical text construction
        |
        v
Tokenizer for selected encoder backbone
        |
        v
Transformer sequence-classification model
        |
        v
Repeated stratified cross-validation
        |
        v
Metrics, statistical tests, tables, and figures
```

The main workflow is:

1. Bug-pattern records are loaded from the `data/` folder.
2. Each record is converted into a canonical text input using fields such as bug name, description, and example code.
3. The input text is tokenized using the selected HuggingFace tokenizer.
4. A transformer encoder is fine-tuned for binary classification.
5. Minority oversampling and class-weighted loss are applied to reduce class imbalance.
6. The model is evaluated using repeated stratified cross-validation.
7. Metrics, predictions, tables, and figures are saved to `results/`, `tables/`, and `figures/`.

---

## Key Results

The best overall macro-F1 score was achieved by **CodeBERT**, while **GraphCodeBERT** achieved the highest ROC-AUC.

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| RoBERTa | 0.764 ± 0.061 | 0.754 ± 0.066 | 0.858 ± 0.048 |
| CodeBERT | **0.767 ± 0.057** | **0.763 ± 0.056** | 0.855 ± 0.044 |
| GraphCodeBERT | 0.758 ± 0.058 | 0.756 ± 0.058 | **0.860 ± 0.047** |

These results suggest that code-aware pre-training provides a modest advantage for classical-vs-quantum bug classification, especially when measured by macro-F1.

---

## Setup Instructions

### Prerequisites

The project assumes the following:

- Python 3.10 or 3.11
- pip or conda
- Git
- Jupyter Notebook
- An NVIDIA GPU is recommended for full training

The final experiments were run in a GPU environment. CPU execution is possible for lightweight tests, but full cross-validation training may take several hours.

### 1. Clone the Repository

```bash
git clone https://github.com/mrahm65/CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software.git
cd CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software
```

### 2. Create a Virtual Environment

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

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, follow the official PyTorch installation command for your system:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

The shipped pipeline does **not** require API keys. Optional environment variables are included only for future extensions.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Optional; only needed for downstream LLM-as-judge or prompt-based extensions |
| `HUGGINGFACE_TOKEN` | Optional; only needed for gated HuggingFace models |

The `.gitignore` file excludes `.env`, so real secrets should remain local and should not be committed.

---

## Running the Application

### Option 1: Run the Full Notebook

The main end-to-end implementation is in:

```text
notebooks/quantum-vs-classical-bug-detection-Final.ipynb
```

Start Jupyter with:

```bash
jupyter notebook notebooks/quantum-vs-classical-bug-detection-Final.ipynb
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

Generate sample outputs without full model training:

```bash
python scripts/generate_sample_outputs.py
```

### Run Tests

```bash
TMPDIR=/tmp pytest tests/ -q
```

The lightweight tests check data-loading and metric-computation utilities without requiring full model training.

---

## Repository Organization

The repository is organized so a grader can quickly locate the implementation, data, documentation, results, and evaluation artifacts.

```text
CSC-7644_Fine-Tuning-LLMs-for-Bug-Detection-in-Quantum-Software/
├── README.md                          # Main project documentation, setup, usage, results, and citations
├── requirements.txt                   # Python dependencies required to run the project
├── pyproject.toml                     # Python project metadata and optional tooling configuration
├── main.py                            # Lightweight project entry point
├── config.example.yaml                # Example configuration file for reproducible runs
├── .env.example                       # Template for optional environment variables
├── .gitignore                         # Excludes caches, local environments, checkpoints, and secrets
├── LICENSE                            # MIT license
├── data/                              # Bug-pattern dataset files
│   ├── README.md                      # Description of dataset files and labels
│   ├── bug_patterns.json              # Original/main bug-pattern dataset
│   ├── bug_patterns_categorized.json  # Final labeled dataset used by the notebook
│   ├── bug_patterns_classical.json    # Classical bug-pattern subset
│   ├── bug_patterns_labeled.json      # Labeled/intermediate classical-quantum dataset
│   ├── bug_patterns_quantum.json      # Quantum-specific bug-pattern subset
│   └── sample_data.json               # Small sample dataset for testing/demo use
├── docs/                              # Project documentation and report-related materials
│   ├── README.md                      # Notes describing documentation files
│   ├── Final_Project_Report_CSC_7644.pdf # Final report PDF
│   ├── Rahman_MdSaidur_MidtermProposal.pdf # Midterm proposal PDF
│   ├── Rahman_MdSaidur_quadchart.pdf  # Project quad chart
│   ├── architecture.md                # High-level architecture notes
│   ├── methodology.md                 # Methodology notes
│   ├── project_summary.md             # Short project summary
│   └── results.md                     # Results notes
├── notebooks/                         # End-to-end Jupyter/Kaggle notebook and notebook notes
│   ├── README.md                      # Notes describing the notebook folder
│   └── quantum-vs-classical-bug-detection-Final.ipynb
├── scripts/                           # Standalone scripts for training, evaluation, and figure generation
│   ├── README.md                      # Notes describing available scripts
│   ├── __init__.py                    # Marks scripts as a Python package
│   ├── generate_figures.py            # Regenerates report-ready figures
│   ├── generate_sample_artifacts.py   # Generates sample intermediate/demo artifacts
│   ├── generate_sample_outputs.py     # Creates sample outputs without full model training
│   ├── run_evaluation.py              # Reads saved results and prints evaluation summaries
│   ├── run_experiment.sh              # Shell helper for running experiment workflow
│   ├── run_study_i_codebert.py        # Legacy CodeBERT study script kept for reference
│   └── run_training.py                # Runs the full training and cross-validation pipeline
├── src/                               # Reusable Python source modules
│   ├── __init__.py                    # Marks src as a Python package
│   ├── data_utils.py                  # Data preprocessing and canonical text construction
│   ├── evaluate.py                    # Accuracy, macro-F1, ROC-AUC, and related metric computation
│   ├── model_utils.py                 # HuggingFace tokenizer/model loading utilities
│   ├── plotting.py                    # Figure-generation utilities
│   └── train_utils.py                 # Training helpers, class weighting, oversampling, and early stopping
├── figures/                           # Generated plots and visualizations
├── results/                           # Saved model outputs, fold metrics, and predictions
├── tables/                            # Exported result tables and statistical summaries
└── tests/                             # Unit tests and validation scripts
```

## Code Quality

The Python implementation is organized into reusable modules under `src/` and runnable scripts under `scripts/`. The code follows standard Python/PEP 8-style formatting, uses clear function and variable names, and includes docstrings for core data-loading, preprocessing, training, evaluation, and plotting functions. Inline comments are used for non-obvious implementation choices such as class weighting, minority oversampling, macro-F1 early stopping, and disk-safe checkpoint handling.

---

## Attributions and Citations

This project uses the following external tools, models, and research resources:

- HuggingFace Transformers: <https://huggingface.co/docs/transformers>
- PyTorch: <https://pytorch.org/>
- scikit-learn: <https://scikit-learn.org/>
- SciPy: <https://scipy.org/>
- pandas: <https://pandas.pydata.org/>
- NumPy: <https://numpy.org/>
- Matplotlib: <https://matplotlib.org/>
- Seaborn: <https://seaborn.pydata.org/>
- RoBERTa: Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, 2019.
- CodeBERT: Feng et al., *CodeBERT: A Pre-Trained Model for Programming and Natural Languages*, EMNLP 2020.
- GraphCodeBERT: Guo et al., *GraphCodeBERT: Pre-training Code Representations with Data Flow*, ICLR 2021.
- Bugs4Q: Zhao et al., *Bugs4Q: A Benchmark of Real Bugs for Quantum Programs*, ASE 2021.
- QBugs / Bugs-QCP-related resources: Campos and Souto, *QBugs: A Collection of Reproducible Bugs in Quantum Algorithms*, 2021.

No external code was copied verbatim. The custom `ManualEarlyStoppingCallback` in `src/train_utils.py` was implemented as a simplified, disk-safe adaptation inspired by HuggingFace's early-stopping behavior. This was necessary because standard checkpoint-based early stopping exceeded the available disk quota during repeated cross-validation.

---

## Author

**Md Saidur Rahman**  
Department of Computer Science  
Louisiana State University  
CSC 7644: Applied LLM Development  
Email: `mrahm65@lsu.edu`
