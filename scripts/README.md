# Scripts

This folder contains standalone scripts for running parts of the project.

## Scripts

| Script                  | Purpose                                                                  |
|-------------------------|--------------------------------------------------------------------------|
| `run_training.py`       | Trains the selected model backbone(s) end-to-end.                        |
| `run_evaluation.py`     | Reads saved per-model JSONs and prints accuracy / macro-F1 / ROC-AUC.    |
| `generate_figures.py`   | Creates the canonical PNG plots referenced in the final report.          |
| `run_study_i_codebert.py` | Lower-level CLI that powers `run_training.py`.                         |
| `generate_sample_outputs.py` | Synthesizes demo SVG/CSV outputs without running training (no GPU). |

The notebook contains the full end-to-end workflow, while these scripts
separate important reusable steps.

## Quick recipes

```bash
# Full reproducible training (GPU recommended; ~75-110 min on a T4)
python scripts/run_training.py

# Inspect saved metrics without rerunning training
python scripts/run_evaluation.py --tables-dir tables

# Regenerate the report-ready PNG figures
python scripts/generate_figures.py

# Demo SVG/CSV output (no training; good for verifying plumbing)
python scripts/generate_sample_outputs.py
```
