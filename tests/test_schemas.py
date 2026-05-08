"""Smoke tests for src.study_i.schemas (no torch / transformers required)."""
from __future__ import annotations

import os
from src.study_i.schemas import (
    ID2LABEL, LABEL2ID, LABEL_LIST, MODEL_REGISTRY, NUM_LABELS,
    TrainingConfig, get_default_config,
)


def test_label_space_consistency():
    """LABEL_LIST, LABEL2ID, ID2LABEL must agree."""
    assert LABEL_LIST == ["classical", "quantum"]
    assert NUM_LABELS == 2
    assert LABEL2ID == {"classical": 0, "quantum": 1}
    assert ID2LABEL == {0: "classical", 1: "quantum"}


def test_model_registry_shape():
    """Registry has three entries, each with name/short/description."""
    assert len(MODEL_REGISTRY) == 3
    expected_shorts = {"roberta", "codebert", "graphcodebert"}
    assert {m["short"] for m in MODEL_REGISTRY} == expected_shorts
    for m in MODEL_REGISTRY:
        assert "name" in m and "description" in m
        assert m["name"]


def test_training_config_defaults():
    """Default TrainingConfig matches the notebook's hyperparameters."""
    cfg = TrainingConfig()
    assert cfg.n_folds == 5
    assert cfg.cv_seeds == (42, 7, 2024, 99, 123)
    assert cfg.num_epochs == 12
    assert cfg.batch_size == 8
    assert cfg.total_folds_per_model == 25
    assert cfg.figure_format == "svg"


def test_get_default_config_env_override(monkeypatch):
    """DATA_PATH / OUTPUT_DIR / FIGURES_DIR / TABLES_DIR / FIGURE_FORMAT respected."""
    monkeypatch.setenv("DATA_PATH", "/tmp/x.json")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/scratch")
    monkeypatch.setenv("FIGURES_DIR", "/tmp/figs")
    monkeypatch.setenv("TABLES_DIR", "/tmp/tabs")
    monkeypatch.setenv("FIGURE_FORMAT", "png")
    cfg = get_default_config()
    assert cfg.data_path == "/tmp/x.json"
    assert cfg.output_dir == "/tmp/scratch"
    assert cfg.figures_dir == "/tmp/figs"
    assert cfg.tables_dir == "/tmp/tabs"
    assert cfg.figure_format == "png"
