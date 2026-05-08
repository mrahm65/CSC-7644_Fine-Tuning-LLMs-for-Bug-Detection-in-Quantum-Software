"""Dataset-loader tests using a temporary JSON fixture."""
from __future__ import annotations

import json
import pytest
from src.study_i.dataset import (
    build_text, get_text_label_arrays, load_labeled_dataset,
)
from src.study_i.schemas import TrainingConfig


def test_build_text_skips_empty_fields():
    """build_text() concatenates only non-empty fields with newlines."""
    item = {"name": "X", "description": "", "example_code": "Z()"}
    assert build_text(item, ("name", "description", "example_code")) == "X\nZ()"


def test_load_labeled_dataset_roundtrip(tmp_path):
    """Records with valid bug_category survive the load filter."""
    records = [
        {"name": "A", "description": "d", "example_code": "c", "bug_category": "classical"},
        {"name": "B", "description": "d", "example_code": "c", "bug_category": "quantum"},
        {"name": "C", "description": "d", "example_code": "c", "bug_category": None},
    ]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(records))
    cfg = TrainingConfig(data_path=str(p))
    loaded = load_labeled_dataset(cfg, verbose=False)
    assert len(loaded) == 2
    texts, labels = get_text_label_arrays(loaded, cfg)
    assert texts.shape == (2,)
    assert set(labels.tolist()) == {0, 1}


def test_load_labeled_dataset_raises_on_uncategorized(tmp_path):
    """A file with no labeled records raises ValueError."""
    p = tmp_path / "empty.json"
    p.write_text(json.dumps([{"name": "X"}]))
    cfg = TrainingConfig(data_path=str(p))
    with pytest.raises(ValueError, match="No records with"):
        load_labeled_dataset(cfg, verbose=False)
