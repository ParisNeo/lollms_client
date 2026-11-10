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
class _BlockedLoader(importlib.abc.Loader):
    """
    Loader that fails only when a blocked module is actually executed.

    Returning a spec (rather than raising from find_spec) preserves the
    import-machinery contract: find_spec is a non-raising probe used by
    importlib.util.find_spec() and by every `import` statement. Raising
    here would escape those probes and can abort an entire pytest session.
    """

    def __init__(self, name: str):
        self._name = name

    def create_module(self, spec):
        return None  # Use default module creation semantics.

    def exec_module(self, module):
        raise ImportError(
            f"{self._name} blocked by conftest.py to prevent NumPy 2.x binary crash. "
            "Pandas will fall back to pure-Python evaluation."
        )


class NumexprBlocker(importlib.abc.MetaPathFinder):
    """Prevents the `numexpr` module from being imported."""

    _BLOCKED_PREFIX = "numexpr"

    def find_spec(self, name, path, target=None):
        # Only intercept numexpr and its submodules. For every other name,
        # return None so the standard finders continue normally.
        if name == self._BLOCKED_PREFIX or name.startswith(self._BLOCKED_PREFIX + "."):
            return importlib.machinery.ModuleSpec(name, _BlockedLoader(name))
        return None


# Install the blocker at the highest priority.
sys.meta_path.insert(0, NumexprBlocker())

# ── 2. Environment variables to suppress warnings ──
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"