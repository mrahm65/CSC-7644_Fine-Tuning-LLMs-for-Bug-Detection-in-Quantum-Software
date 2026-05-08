# Data

This folder contains data resources for the quantum-vs-classical bug
classification project.

The project uses Qiskit-centric bug report data derived from Bugs-QCP and
related quantum bug benchmark resources.

The task is binary classification:

- classical bug
- quantum-specific bug

Each sample is converted into a canonical input string using available
fields such as bug name, description, and example code.

If the full dataset cannot be redistributed, this folder should contain
sample data and instructions for obtaining the original dataset.

## Files

| File                          | Description                                                                |
|-------------------------------|----------------------------------------------------------------------------|
| `bug_patterns_labeled.json`   | Full bug-pattern catalog (233 records) used by every script and notebook.  |
| `sample_data.json`            | A trimmed, anonymous example so the loader can be exercised quickly.       |
