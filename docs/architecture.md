# Architecture

This project ships a small Python package that fine-tunes three pretrained
encoders for binary bug-report classification (classical vs. quantum) and
compares them under a fixed cross-validation protocol.

## Module overview

| Module                  | Responsibility                                                                |
|-------------------------|-------------------------------------------------------------------------------|
| `src/config.py`         | Model registry, label space, `TrainingConfig` dataclass, env-var overrides.   |
| `src/data_loader.py`    | Loads the labeled JSON, builds the per-record text, and integer-encodes labels. |
| `src/callbacks.py`      | `ManualEarlyStoppingCallback`, `EpochLogCallback`, `WeightedTrainer`, oversample, metrics. |
| `src/training.py`       | `run_fold()` (one CV fold) and `run_full_experiment()` (full sweep for one backbone). |
| `src/plots.py`          | Matplotlib helpers; honours `figure_format` (svg / png / pdf).                |
| `src/evaluation.py`     | Cross-model comparison table + combined JSON.                                 |
| `main.py`               | Argparse-driven CLI that wires it all together.                               |
| `scripts/`              | Convenience wrappers (`run_experiment.sh`, `generate_sample_artifacts.py`).   |
| `tests/`                | Lightweight unit tests covering config, data loader, and helpers.             |

## Data flow

```
data/bug_patterns_labeled.json
        |
        v
load_labeled_dataset  -->  get_text_label_arrays  -->  (texts, labels)
        |
        v
run_full_experiment(model, texts, labels, config)
        |  (per fold)
        v
run_fold  -->  oversample  -->  Trainer (WeightedTrainer)  -->  predictions
        |
        v
plots.* (figures_dir)   tables/*.json + tables/*.csv (tables_dir)
```

## Output layout

* **`figures/`** — every PNG/SVG figure (per-model + cross-model).
* **`tables/`** — every CSV / JSON result table.
* **`results/`** — Trainer scratch only; per-fold temp directories are
  deleted at the end of each fold.

The split keeps version-control noise low: only `tables/` typically needs
to be committed when archiving a run, while `figures/` can be regenerated
from the JSON tables.
