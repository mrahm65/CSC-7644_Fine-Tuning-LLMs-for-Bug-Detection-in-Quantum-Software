"""Data-loading utilities for the bug-pattern JSON dataset.

The expected schema for each record is::

    {
        "name":         "Short pattern title",
        "description":  "Free-text description of the bug",
        "example_code": "Optional code snippet",
        "bug_category": "classical" | "quantum"
    }

Records that are missing ``bug_category`` (or whose value is not one of the
allowed labels) are filtered out and reported on stdout.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from src.config import LABEL2ID, LABEL_LIST, TrainingConfig


def load_labeled_dataset(
    config: TrainingConfig,
    *,
    verbose: bool = True,
) -> List[Dict]:
    """Read the JSON file at ``config.data_path`` and return labeled records.

    Args:
        config: The active :class:`TrainingConfig`.
        verbose: If ``True``, prints a short class-balance summary.

    Returns:
        List of dicts whose ``bug_category`` value is one of
        ``{"classical", "quantum"}``.

    Raises:
        FileNotFoundError: If ``config.data_path`` does not exist.
        ValueError: If the JSON file does not decode to a list of records,
            or contains zero labeled records.
    """
    with open(config.data_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of records, got {type(raw).__name__}"
        )

    label_field = config.label_field
    labeled = [r for r in raw if r.get(label_field) in LABEL_LIST]

    if not labeled:
        raise ValueError(
            f"No records with '{label_field}' in {LABEL_LIST!r} found in "
            f"{config.data_path!r}. Verify the dataset has been categorised."
        )

    if verbose:
        n_total = len(raw)
        n_labeled = len(labeled)
        counts = {lbl: 0 for lbl in LABEL_LIST}
        for record in labeled:
            counts[record[label_field]] += 1

        print(f"Total loaded   : {n_total}")
        print(f"Labeled        : {n_labeled}")
        for lbl in LABEL_LIST:
            pct = 100 * counts[lbl] / n_labeled
            print(f"  {lbl:<11s}: {counts[lbl]}  ({pct:.1f}%)")

        # Imbalance ratio (largest / smallest, defensively avoiding /0).
        max_count = max(counts.values())
        min_count = max(min(counts.values()), 1)
        print(f"Imbalance ratio: {max_count / min_count:.2f}:1")

    return labeled


def build_text(item: Dict, fields: Tuple[str, ...]) -> str:
    """Concatenate ``fields`` of ``item`` into a single newline-joined string.

    Empty / missing fields are silently skipped so the resulting text never
    contains stray newlines.
    """
    parts: List[str] = []
    for key in fields:
        value = (item.get(key) or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def get_text_label_arrays(
    records: List[Dict],
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw records into parallel ``(texts, labels)`` numpy arrays.

    Labels are integer-encoded via :data:`src.config.LABEL2ID`.
    """
    texts = np.array([build_text(r, config.text_fields) for r in records])
    labels = np.array([LABEL2ID[r[config.label_field]] for r in records])
    return texts, labels
