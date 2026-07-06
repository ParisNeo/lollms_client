"""
conftest.py
===========
Pytest configuration to neutralize environment-specific C-extension crashes
that interfere with test execution.

CRITICAL FIX: The `numexpr` package (optional pandas acceleration backend)
crashes with `AttributeError: _ARRAY_API not found` when compiled against
NumPy 1.x but run under NumPy 2.x. In environments with `pytest-qt` active,
this crash is intercepted by the Qt event loop, producing a misleading
"Exceptions caught in Qt event loop" error that aborts the entire test session.

This conftest blocks `numexpr` from being imported, forcing pandas to fall
back to its pure-Python evaluation path.
"""
import sys
import os
import importlib.abc
import importlib.machinery

# ── 1. Block numexpr to prevent NumPy 2.x binary crash ──
class NumexprBlocker(importlib.abc.MetaPathFinder):
    """Prevents the `numexpr` module from being imported."""
    def find_spec(self, name, path, target=None):
        if name == "numexpr" or name.startswith("numexpr."):
            # Return a dummy spec that points to a non-existent loader
            # This causes ImportError, which pandas handles gracefully
            raise ImportError(
                "numexpr blocked by conftest.py to prevent NumPy 2.x binary crash. "
                "Pandas will fall back to pure-Python evaluation."
            )
        return None

# Install the blocker at the highest priority
sys.meta_path.insert(0, NumexprBlocker())

# ── 2. Environment variables to suppress warnings ──
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
