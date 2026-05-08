"""Train the selected model backbone(s) end-to-end.

This is a thin wrapper around the existing reproducible CV pipeline. The
implementation lives in :mod:`src.study_i.training` so the notebook and
this script share exactly the same training code.

Usage::

    python scripts/run_training.py
    python scripts/run_training.py --models codebert
    python scripts/run_training.py --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Delegate to the canonical entry point.
from scripts.run_study_i_codebert import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
