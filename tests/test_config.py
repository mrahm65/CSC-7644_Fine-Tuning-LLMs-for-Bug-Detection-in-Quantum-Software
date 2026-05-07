"""Smoke tests for src.config."""

from __future__ import annotations

import os
import unittest

from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TABLES_DIR,
    LABEL2ID,
    LABEL_LIST,
    MODEL_REGISTRY,
    get_default_config,
)


class ConfigDefaultsTests(unittest.TestCase):
    """Defaults are populated and the env-var override works."""

    def test_label_space(self) -> None:
        self.assertEqual(LABEL_LIST, ["classical", "quantum"])
        self.assertEqual(LABEL2ID, {"classical": 0, "quantum": 1})

    def test_three_backbones(self) -> None:
        shorts = [m["short"] for m in MODEL_REGISTRY]
        self.assertIn("roberta", shorts)
        self.assertIn("codebert", shorts)
        self.assertIn("graphcodebert", shorts)

    def test_get_default_config_uses_defaults(self) -> None:
        for var in ("DATA_PATH", "OUTPUT_DIR", "FIGURES_DIR", "TABLES_DIR"):
            os.environ.pop(var, None)
        c = get_default_config()
        self.assertEqual(c.data_path, DEFAULT_DATA_PATH)
        self.assertEqual(c.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(c.figures_dir, DEFAULT_FIGURES_DIR)
        self.assertEqual(c.tables_dir, DEFAULT_TABLES_DIR)
        self.assertEqual(c.figure_format, "svg")
        self.assertEqual(c.total_folds_per_model, 25)

    def test_env_var_overrides(self) -> None:
        os.environ["DATA_PATH"] = "/tmp/custom.json"
        os.environ["FIGURES_DIR"] = "/tmp/figs"
        try:
            c = get_default_config()
            self.assertEqual(c.data_path, "/tmp/custom.json")
            self.assertEqual(c.figures_dir, "/tmp/figs")
        finally:
            del os.environ["DATA_PATH"]
            del os.environ["FIGURES_DIR"]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
