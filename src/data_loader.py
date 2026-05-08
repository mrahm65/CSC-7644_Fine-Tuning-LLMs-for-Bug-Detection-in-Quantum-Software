"""Backward-compatibility shim.

This module re-exports everything from :mod:`src.study_i.dataset`. New
code should import from the study_i subpackage directly.
"""
from src.study_i.dataset import *  # noqa: F401,F403
from src.study_i import dataset as _impl

__all__ = getattr(_impl, "__all__", [n for n in dir(_impl) if not n.startswith("_")])
