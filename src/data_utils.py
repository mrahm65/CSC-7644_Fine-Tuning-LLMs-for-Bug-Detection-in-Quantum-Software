"""Load data, filter labels, construct canonical input text."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


LABEL_LIST: List[str] = ["classical", "quantum"]
LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS: int = len(LABEL_LIST)

# Default ordered list of bug-record fields used to build the model's
# input string. Empty / missing fields are skipped silently.
DEFAULT_TEXT_FIELDS: Tuple[str, ...] = (
    "name", "description", "example_code",
)


def load_dataset(
    path: str,
    *,
    label_field: str = "bug_category",
    allowed_labels: Iterable[str] = tuple(LABEL_LIST),
    verbose: bool = True,
) -> List[Dict]:
    """Load bug report data from a JSON file.

    Args:
        path: Filesystem path to the dataset (JSON list of records).
        label_field: Name of the per-record field carrying the binary label.
        allowed_labels: Set of label values to keep. Records with a label
            outside this set are dropped.
        verbose: If ``True``, prints a short class-distribution summary.

    Returns:
        List of labeled records.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not decode to a list, or contains no
            records that match ``allowed_labels``.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of records, got {type(raw).__name__}"
        )

    allowed = set(allowed_labels)
    labeled = [r for r in raw if r.get(label_field) in allowed]
    if not labeled:
        raise ValueError(
            f"No records with '{label_field}' in {sorted(allowed)} found "
            f"in {path!r}. Verify the dataset has been categorised."
        )

    if verbose:
        n_total = len(raw)
        n_labeled = len(labeled)
        counts = {lbl: 0 for lbl in allowed}
        for record in labeled:
            counts[record[label_field]] += 1

        print(f"Total loaded   : {n_total}")
        print(f"Labeled        : {n_labeled}")
        for lbl in sorted(allowed):
            pct = 100 * counts[lbl] / n_labeled
            print(f"  {lbl:<11s}: {counts[lbl]}  ({pct:.1f}%)")
        max_count = max(counts.values())
        min_count = max(min(counts.values()), 1)
        print(f"Imbalance ratio: {max_count / min_count:.2f}:1")

    return labeled


def build_input_text(
    example: Dict,
    fields: Tuple[str, ...] = DEFAULT_TEXT_FIELDS,
) -> str:
    """Construct a canonical model input from bug fields.

    Concatenates the ``fields`` of ``example`` with newlines, skipping
    any field whose stripped value is empty so the result has no stray
    blank lines.
    """
    parts: List[str] = []
    for key in fields:
        value = (example.get(key) or "").strip()
        if value:
            parts.append(value)
    return "\n".join(parts)


def to_text_label_arrays(
    records: List[Dict],
    *,
    label_field: str = "bug_category",
    fields: Tuple[str, ...] = DEFAULT_TEXT_FIELDS,
    label2id: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw records into parallel ``(texts, labels)`` numpy arrays."""
    label2id = label2id or LABEL2ID
    texts = np.array([build_input_text(r, fields) for r in records])
    labels = np.array([label2id[r[label_field]] for r in records])
    return texts, labels
