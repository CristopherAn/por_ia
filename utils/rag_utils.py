"""Compatibility wrapper.

This module re-exports everything from the legacy top-level `rag_utils.py` so that
new code can import from `utils.rag_utils` without breaking existing scripts.
"""

from rag_utils import *  # noqa: F403,F401
