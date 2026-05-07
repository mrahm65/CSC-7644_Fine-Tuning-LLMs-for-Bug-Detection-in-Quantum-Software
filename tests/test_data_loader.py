"""Smoke tests for src.data_loader."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.config import get_default_config
from src.data_loader import (
    build_text,
    get_text_label_arrays,
    load_labeled_dataset,
)


class BuildTextTests(unittest.TestCase):
    """The text-builder concatenates only non-empty fields."""

    def test_uses_all_three_fields(self) -> None:
        item = {
            "name": "X",
            "description": "Y",
            "example_code": "Z()",
            "bug_category": "quantum",
        }
        out = build_text(item, ("name", "description", "example_code"))
        self.assertEqual(out, "X\nY\nZ()")

    def test_skips_missing_fields(self) -> None:
        item = {"name": "Only-name", "bug_category": "classical"}
        out = build_text(item, ("name", "description", "example_code"))
        self.assertEqual(out, "Only-name")

    def test_returns_empty_when_all_blank(self) -> None:
        out = build_text({}, ("name", "description"))
        self.assertEqual(out, "")


class LoadLabeledDatasetTests(unittest.TestCase):
    """``load_labeled_dataset`` filters and validates."""

    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        self.config = get_default_config()
        self.config.data_path = self.tmp.name

    def tearDown(self) -> None:
        os.unlink(self.tmp.name)

    def test_filters_unlabeled(self) -> None:
        records = [
            {"name": "A", "bug_category": "classical"},
            {"name": "B", "bug_category": "quantum"},
            {"name": "C"},
            {"name": "D", "bug_category": "other"},
        ]
        json.dump(records, self.tmp)
        self.tmp.close()

        labeled = load_labeled_dataset(self.config, verbose=False)
        self.assertEqual(len(labeled), 2)

        texts, labels = get_text_label_arrays(labeled, self.config)
        self.assertEqual(len(texts), 2)
        self.assertEqual(set(labels.tolist()), {0, 1})

    def test_raises_when_no_labeled(self) -> None:
        json.dump([{"name": "x"}], self.tmp)
        self.tmp.close()
        with self.assertRaises(ValueError):
            load_labeled_dataset(self.config, verbose=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
