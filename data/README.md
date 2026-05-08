# Data

This folder contains the dataset files used for the quantum-vs-classical bug classification project.

| File | Description |
|---|---|
| `bug_patterns.json` | Original/main bug-pattern dataset. |
| `bug_patterns_labeled.json` | Main labeled dataset used by the training notebook and scripts. |
| `bug_patterns_categorized.json` | Full categorized dataset with the `bug_category` field. |
| `bug_patterns_quantum.json` | Quantum-only subset containing 99 quantum-specific bug records. |
| `bug_patterns_classical.json` | Classical-only subset containing 134 classical bug records. |
| `sample_data.json` | Small sample dataset for quick testing and demonstration. |

The final task is binary classification:

- `quantum`
- `classical`
