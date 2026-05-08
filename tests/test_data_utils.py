"""Tests for src.data_utils."""
from __future__ import annotations

import json
import pytest

from src.data_utils import (
    DEFAULT_TEXT_FIELDS,
    LABEL2ID,
    LABEL_LIST,
    NUM_LABELS,
    build_input_text,
    load_dataset,
    to_text_label_arrays,
)


def test_label_space_consistency():
    """LABEL_LIST and LABEL2ID stay aligned."""
    assert LABEL_LIST == ["classical", "quantum"]
    assert NUM_LABELS == 2
    assert LABEL2ID == {"classical": 0, "quantum": 1}


def test_build_input_text_skips_empty_fields():
    """Empty / missing fields are dropped silently."""
    item = {"name": "X", "description": "", "example_code": "Z()"}
    assert build_input_text(item, DEFAULT_TEXT_FIELDS) == "X\nZ()"


def test_load_dataset_filters_unlabeled(tmp_path):
    """Records with no bug_category are dropped."""
    records = [
        {"name": "A", "description": "d", "example_code": "c", "bug_category": "classical"},
        {"name": "B", "description": "d", "example_code": "c", "bug_category": "quantum"},
        {"name": "C", "description": "d", "example_code": "c", "bug_category": None},
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(records))
    loaded = load_dataset(str(p), verbose=False)
    assert len(loaded) == 2
    texts, labels = to_text_label_arrays(loaded)
    assert texts.shape == (2,)
    assert set(labels.tolist()) == {0, 1}


def test_load_dataset_raises_on_empty(tmp_path):
    """A file with no labeled records raises ValueError."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([{"name": "X"}]))
    with pytest.raises(ValueError, match="No records with"):
        load_dataset(str(p), verbose=False)
