# Code Architecture

```
project/
+- main.py                       # thin wrapper around scripts/run_study_i_codebert.py
+- scripts/
|   +- run_study_i_codebert.py   # CLI entrypoint (full reproduction)
|   +- generate_sample_outputs.py  # synthesize demo SVG/CSV outputs
+- src/
|   +- study_i/                  # package containing the experiment
|       +- schemas.py            # TrainingConfig + label space + MODEL_REGISTRY
|       +- dataset.py            # JSON loader + text builder
|       +- callbacks.py          # ManualEarlyStopping, EpochLog, WeightedTrainer
|       +- training.py           # run_fold() + run_full_experiment()
|       +- plotting.py           # all matplotlib helpers (svg by default)
|       +- analysis.py           # cross-model comparison + combined JSON
+- data/                         # input dataset(s)
+- figures/                      # PNG/SVG outputs
+- tables/                       # CSV/JSON outputs
+- results/                      # trainer scratch (auto-cleaned)
+- tests/                        # pytest suite (no torch needed)
+- docs/                         # methodology + architecture
+- outputs/README.md             # what each output file contains
```

## Data flow

1. **Load** `data/bug_patterns_labeled.json` ->
   `src.study_i.dataset.load_labeled_dataset()`.
2. **Build text** -> per-record `name\ndescription\nexample_code` strings.
3. **Cross-validate** for each backbone via
   `src.study_i.training.run_full_experiment()`:
    - 5-fold x 5 seeds = 25 folds.
    - Per fold: oversample, tokenize, fine-tune, predict, score.
4. **Plot** per-model artifacts (confusion matrix, fold scatter, ROC).
5. **Aggregate** across backbones via `src.study_i.analysis`:
    - Comparison table.
    - Cross-model bar chart, ROC overlay, paired per-fold strip plot.
6. **Persist** results in `tables/` (JSON + CSV) and figures in
   `figures/` (SVG by default).

## Public API

```python
from src.study_i import (
    TrainingConfig, get_default_config,
    load_labeled_dataset, get_text_label_arrays,
    run_full_experiment,
)
```

## Adding a new backbone

Append an entry to `MODEL_REGISTRY` in `src/study_i/schemas.py`:

```python
{
    "name": "facebook/<model>",
    "short": "<short>",
    "description": "<one-line note about pretraining data>",
}
```

No other code changes are needed; the CV loop, plotting, and
cross-model analysis pick up the new backbone automatically.
