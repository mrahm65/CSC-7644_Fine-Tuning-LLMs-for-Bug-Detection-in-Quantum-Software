# Code Architecture

```text
project/
+- main.py                       # lightweight project entry point
+- scripts/
|   +- run_training.py           # full training + cross-validation pipeline
|   +- run_evaluation.py         # reads saved tables and prints metrics
|   +- generate_figures.py       # regenerates report-ready figures
|   +- generate_sample_outputs.py  # synthesizes demo outputs (no GPU)
+- src/                          # six flat reusable modules
|   +- __init__.py               # public-API re-exports
|   +- data_utils.py             # JSON loader + canonical text builder + label space
|   +- model_utils.py            # HuggingFace tokenizer / model loaders
|   +- train_utils.py            # oversampling, class weights, early stopping, weighted trainer
|   +- evaluate.py               # accuracy, macro-F1, ROC-AUC, confusion-matrix helpers
|   +- plotting.py               # matplotlib helpers for every figure
+- data/                         # bug-pattern dataset(s)
+- figures/                      # PNG outputs (also SVG variants)
+- tables/                       # CSV / JSON outputs
+- results/                      # per-sample predictions + final summary
+- tests/                        # pytest suite (no torch needed)
+- docs/                         # methodology, architecture, results notes
+- notebooks/                    # end-to-end Jupyter / Kaggle notebook
```

## Data flow

1. **Load** `data/bug_patterns_categorized.json` (the labeled file the
   notebook uses) via `src.data_utils.load_dataset()`.
2. **Build text** — per-record `name\ndescription\nexample_code` strings
   via `src.data_utils.build_input_text()`.
3. **Cross-validate** for each backbone:
   - 5-fold × 5 seeds = 25 fold-runs.
   - Per fold: oversample (`src.train_utils.oversample_minority_class`),
     tokenize, fine-tune with weighted CE + label smoothing
     (`make_weighted_trainer_class`), early-stop on macro-F1
     (`make_manual_early_stopping_callback`), predict, score
     (`src.evaluate.compute_metrics`).
4. **Plot** per-model artifacts — confusion matrix
   (`plot_confusion_matrix`), per-fold scatter, per-model ROC
   (`plot_roc_curve`).
5. **Aggregate** across backbones — comparison bar chart
   (`plot_mean_metrics_bar`), ROC overlay (`plot_roc_overlay`), paired
   per-fold strip plot.
6. **Persist** results in `tables/` (CSV + JSON) and `results/`
   (per-sample CSVs + final summary) and figures in `figures/` (PNG by
   default; the notebook also exports SVG).

## Public API

```python
from src import (
    build_input_text,
    load_dataset,
    compute_metrics,
)
from src.data_utils import LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS
from src.model_utils import load_model_and_tokenizer, get_data_collator
from src.train_utils import (
    compute_class_weights,
    oversample_minority_class,
    make_manual_early_stopping_callback,
    make_epoch_log_callback,
    make_weighted_trainer_class,
)
from src.evaluate import compute_metrics, pooled_metrics, confusion_breakdown
from src.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_roc_overlay,
    plot_mean_metrics_bar,
    plot_dataset_distribution,
    plot_methodology_diagram,
)
```

## Adding a new backbone

The model registry now lives in the script that drives the comparison.
To add a fourth backbone, edit `scripts/run_training.py` (or the
`MODEL_REGISTRY` block in the notebook) and append:

```python
{
    "name": "facebook/<model>",
    "short": "<short>",
    "description": "<one-line note about pretraining data>",
}
```

The CV loop, plotting, and cross-model analysis pick up the new entry
automatically — no other code changes needed.
