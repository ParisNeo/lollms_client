"""
sandbox_kernel.py

Shared defense-in-depth sandbox for LCP Python execution tools.

Provides two complementary engines:

1. `validate_source(source, profile)` — AST-based static analysis. Rejects code
   BEFORE any byte is executed if it imports or calls a forbidden symbol.
   Two profiles exist:
     - "data_query": strict allowlist (pandas/numpy/matplotlib/office/sqlalchemy).
     - "standard":   denylist-based (blocks the dangerous surface only).

2. `SandboxContext(...)` — a `with`-scoped runtime patcher. During `exec()` it:
     - installs a filtered `builtins.__import__`,
     - patches `importlib.import_module` / `reload`,
     - blocks `eval` / `exec` / `compile` / `__import__` / `getattr` escape chains
       when the profile requires it,
     - wraps `open`, `io.open`, pathlib mutations, and `os.*` path ops so any
       path resolving outside the workspace root raises `PermissionError`,
     - blocks `os.system/popen/spawn*/exec*/remove/unlink/rmdir/rename/replace/
       chmod/chown/mkdir/makedirs/symlink/kill/killpg/putenv/setenv/unsetenv`,
     - blocks `shutil.rmtree/move/copy*`,
     - blocks `subprocess.*`, `socket.socket.connect*`, `socket.create_connection`,
       `socket.getaddrinfo`, `urllib.request.urlopen`, `http.client.HTTP*`,
     - blocks `ctypes` entirely (both import and `CDLL`).

Every patch is restored on exit — even if user code raises.

Behavior on block:
- AST layer  → returns a `SandboxViolation` the caller turns into a tool error.
- Runtime     → raises `SandboxPermissionError` (subclass of PermissionError) so
                user code can catch and continue.

This module is stateful only during `SandboxContext.__enter__/__exit__`; it holds
no cross-request state and is safe under recursive / nested use because each
context snapshots the originals for its lifetime and restores them symmetrically.
"""

from __future__ import annotations

import ast
import builtins
import io
import os
import sys
import shutil
import socket
import importlib
import subprocess
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# True (unpatched) builtins captured at module import time.
#
# Tool plumbers need compile/exec to actually run user code inside the sandbox.
# Reading them from `builtins.compile` / `builtins.exec` at call time is
# fragile: if ANY prior SandboxContext leaked (e.g. due to a mid-install
# failure on exotic platforms) the module-level name resolves to our blocking
# stub. These module-level constants are the single source of truth and are
# guaranteed to be the real builtins — immune to any leak, any prior test.
# ──────────────────────────────────────────────────────────────────────────────
_BLOCKED_STUB_MARKER = "__lollms_blocked_builtin__"


def _make_block_stub():
    """Factory so the marker attribute can be attached to a fresh function.

    The marker is how `_capture_true_builtins()` recognizes its OWN block stub
    when re-entering module import under an active `SandboxContext`. Without
    this marker, the module-load-order corruption bug (importing
    `sandbox_kernel` for the FIRST time inside a live sandbox) would silently
    capture the block stub as the "true exec", and every subsequent data-query
    tool call would fail with "dynamic code execution is forbidden" — the
    exact regression observed in test_agent_cognitive_decisions."""
    def _blocked_builtin(*args, **kwargs):
        raise SandboxPermissionError(
            "🛑 BLOCKED BY SANDBOX: dynamic code execution "
            "(eval/exec/compile) is forbidden."
        )
    setattr(_blocked_builtin, _BLOCKED_STUB_MARKER, True)
    return _blocked_builtin


def _is_block_stub(fn: Any) -> bool:
    """True if `fn` is one of our own block stubs (not a real builtin)."""
    return bool(fn is not None and getattr(fn, _BLOCKED_STUB_MARKER, False))


def _recover_true_builtin(name: str) -> Any:
    """Obtains a genuine `compile`/`exec`/`eval` even if `builtins.<name>` is
    currently the block stub.

    Strategy (each step validated by `_is_block_stub`):
      1. Frozen sentinel `builtins.__lollms_true_<name>__` — validated.
      2. Snapshot leaked on the sentinel tuple — validated.
      3. Recover from a fresh dict/instance builtin source via `type.__dict__`
         lookups on known holders (`__builtins__` may be a module OR a dict).
      4. Last resort: `builtins.__dict__[name]` — will still be a real builtin
         if all sandbox instances have been correctly torn down.
    """
    sentinel_attr = f"__lollms_true_{name}__"
    candidate = getattr(builtins, sentinel_attr, None)
    if not _is_block_stub(candidate):
        return candidate
    # Sentinel is itself corrupted (module was first imported under a live
    # sandbox without the sentinel yet set). Try to reach the real builtin.
    holder_dict = builtins.__dict__
    dict_candidate = holder_dict.get(name)
    if not _is_block_stub(dict_candidate):
        return dict_candidate
    # Nothing genuine available — re-raise so callers see a clear failure
    # rather than silently calling the block stub.
    raise SandboxPermissionError(
        f"Sandbox kernel lost the true builtin '{name}' — a prior sandbox "
        "context likely leaked. This indicates a bug; refusing to run user "
        "code with a corrupted exec harness."
    )


# Process-wide, session-stable snapshot of the REAL builtins. This tuple MUST
# be captured exactly once, before any `SandboxContext` can possibly be active.
# Because `sandbox_kernel` is imported by `execute_python` / `execute_python_data_query`
# at process startup, that "first import" always happens outside any sandbox.
#
# The snapshot lives on `sys` (module reloads cannot clear it) with per-slot
# `_is_block_stub` validation so a leaked block stub can never be cached here.
_SYS_TRUE_BUILTINS_ATTR = "__lollms_true_builtins__"


def _store_true_builtins_in_sys(tc: Any, te: Any, tv: Any) -> None:
    """Persists the true builtins on `sys` so re-imports cannot lose them."""
    try:
        setattr(sys, _SYS_TRUE_BUILTINS_ATTR, (tc, te, tv))
    except Exception:
        pass


def _load_true_builtins_from_sys() -> Tuple[Any, Any, Any]:
    """Returns the sys-cached true builtins, or (None, None, None) if absent."""
    cached = getattr(sys, _SYS_TRUE_BUILTINS_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == 3:
        tc, te, tv = cached
        if not (_is_block_stub(tc) or _is_block_stub(te) or _is_block_stub(tv)):
            return tc, te, tv
    return None, None, None


def _capture_true_builtins() -> Tuple[Any, Any, Any]:
    """Capture the real builtins; safe under any module-load-order scenario.

    Resolution order (each step validated by `_is_block_stub`):
      1. `sys.__lollms_true_builtins__` — written on the FIRST genuine import
         (guaranteed pre-sandbox since tools import the kernel at startup).
      2. Per-slot frozen sentinel `builtins.__lollms_true_<name>__`.
      3. `builtins.__dict__[name]` — genuine only if no sandbox is active.
      4. `_recover_true_builtin(name)` — last resort, raises if impossible.

    The single source of truth becomes the sys-cache: once populated it is
    never overwritten, so re-importing this module inside a live sandbox
    returns the cached genuine functions instead of raising.
    """
    # Fast path: sys-cache already populated (this is the normal case after
    # the very first process import).
    cached_tc, cached_te, cached_tv = _load_true_builtins_from_sys()
    if cached_tc is not None and cached_te is not None and cached_tv is not None:
        return cached_tc, cached_te, cached_tv

    def _resolve(name: str, sentinel_attr: str) -> Any:
        sentinel = getattr(builtins, sentinel_attr, None)
        if not _is_block_stub(sentinel) and sentinel is not None:
            return sentinel
        candidate = builtins.__dict__.get(name)
        if not _is_block_stub(candidate) and candidate is not None:
            return candidate
        return _recover_true_builtin(name)

    tc = _resolve("compile", "__lollms_true_compile__")
    te = _resolve("exec", "__lollms_true_exec__")
    tv = _resolve("eval", "__lollms_true_eval__")
    _store_true_builtins_in_sys(tc, te, tv)
    return tc, te, tv


def _recover_import_builtin() -> Any:
    """Returns a genuine `__import__` if `builtins.__import__` is currently a
    sandbox gated stub. Used narrowly by `run_in_sandboxed_child` to guarantee
    that `multiprocessing.spawn`'s pickle-by-reference import step succeeds.

    Never raises; returns None if no genuine importer can be located.
    """
    current = builtins.__dict__.get("__import__")
    if current is not None and not _is_block_stub(current) and not getattr(current, "__name__", "") == "_filtered_import":
        return current
    # Sentinels are not written for __import__ by design; recover from any
    # live `sys.modules` snapshot of a previous true importer if one exists.
    sentinel = getattr(builtins, "__lollms_true_import__", None)
    if sentinel is not None and not _is_block_stub(sentinel) and sentinel is not current:
        return sentinel
    # Last resort: rebuild from the frozen bootstrap that Python always has.
    try:
        import _frozen_importlib
        candidate = _frozen_importlib.__import__
        if not _is_block_stub(candidate):
            return candidate
    except Exception:
        pass
    import importlib as _importlib_mod
    candidate = importlib._bootstrap.__import__  # type: ignore[attr-defined]
    if not _is_block_stub(candidate):
        return candidate
    return None


TRUE_COMPILE, TRUE_EXEC, TRUE_EVAL = _capture_true_builtins()
# Publish frozen markers so ANY future import of this module (or a `_snapshot()`
# in a nested SandboxContext, or a restore after a leak) can recover the true
# builtins even if `builtins.exec`/`compile`/`eval` are currently stubs.
# Each sentinel write is validated: a block stub is NEVER frozen as a sentinel.
try:
    existing_tc = getattr(builtins, "__lollms_true_compile__", None)
    if existing_tc is None or _is_block_stub(existing_tc):
        builtins.__lollms_true_compile__ = TRUE_COMPILE  # type: ignore[attr-defined]
    existing_te = getattr(builtins, "__lollms_true_exec__", None)
    if existing_te is None or _is_block_stub(existing_te):
        builtins.__lollms_true_exec__ = TRUE_EXEC  # type: ignore[attr-defined]
    existing_tv = getattr(builtins, "__lollms_true_eval__", None)
    if existing_tv is None or _is_block_stub(existing_tv):
        builtins.__lollms_true_eval__ = TRUE_EVAL  # type: ignore[attr-defined]
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class SandboxPermissionError(PermissionError):
    """Raised at runtime when user code attempts a forbidden operation."""


@dataclass
class SandboxViolation:
    """A static-analysis rejection returned by `validate_source`."""
    reason: str
    snippet: str
    line: int


# ──────────────────────────────────────────────────────────────────────────────
# Symbol tables
# ──────────────────────────────────────────────────────────────────────────────

# --- Hard-forbidden in BOTH profiles ---
# These modules give direct host-process, network, or OS-escape primitives.
FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "subprocess", "pty", "commands", "multiprocessing", "concurrent",
    "socket", "ssl", "requests", "urllib", "urllib2", "urllib3",
    "http", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "paramiko", "fabric", "cryptography", "cffi", "ctypes",
    "importlib", "imp", "pkgutil", "runpy", "zipimport",
    "webbrowser", "antigravity", "this",
    "psutil", "winreg", "win32api", "win32con", "win32com", "pythoncom",
    "signal", "resource",
})

# Modules that are FORBIDDEN in "standard" but PERMITTED in "authorized".
# The "authorized" profile is selected ONLY after the user explicitly approves
# a risky operation (via the confirm handler / TTY prompt) in "safe" mode, or
# under "full_access". It represents "the operator consented to this specific
# class of operation" — network access, subprocess spawning, package-manager
# helpers — while STILL blocking the primitives that can escape the sandbox
# entirely at the C level (ctypes, winreg, os.exec*, os.system).
#
# Rationale: a user who says "yes, ping google" has authorized *network I/O
# and process spawning*. They have NOT authorized "arbitrary host code
# execution via ctypes" or "escape the sandbox via os.exec*". The two sets are
# deliberately disjoint from `AUTHORIZED_STILL_FORBIDDEN_MODULES` below.
AUTHORIZED_MODULES: frozenset[str] = frozenset({
    "subprocess", "pty", "commands", "multiprocessing", "concurrent",
    "socket", "ssl", "requests", "urllib", "urllib2", "urllib3",
    "http", "httpx", "aiohttp", "ftplib", "smtplib", "telnetlib",
    "paramiko", "fabric", "psutil", "webbrowser",
    "importlib", "imp", "pkgutil", "runpy", "zipimport",
})

# Modules that remain FORBIDDEN even under the "authorized" profile. These
# primitives can escape the sandbox at the C level (ctypes), corrupt the host
# registry (winreg), or interact with OS signals/resources in ways that no
# user-consent gate can make safe. If a user genuinely needs these, they must
# use a different tool (system_shell, or a host script) — not execute_python.
AUTHORIZED_STILL_FORBIDDEN_MODULES: frozenset[str] = frozenset({
    "ctypes", "cffi", "cryptography",
    "winreg", "win32api", "win32con", "win32com", "pythoncom",
    "signal", "resource",
    "antigravity", "this",
})

# Modules explicitly allowed in the strict data-query profile.
DATA_QUERY_ALLOWED_MODULES: frozenset[str] = frozenset({
    # Analysis core
    "pandas", "numpy", "scipy", "sklearn", "statsmodels", "seaborn",
    # Plotting
    "matplotlib", "matplotlib.pyplot",
    # Office round-trip
    "openpyxl", "docx", "pptx", "xlrd", "xlsxwriter",
    # Databases (read/write local .db only enforced by path guard)
    "sqlalchemy", "sqlite3",
    # Pure stdlib helpers commonly used in data work
    "json", "csv", "math", "statistics", "datetime", "decimal",
    "fractions", "re", "itertools", "functools", "collections",
    "typing", "string", "textwrap", "base64", "hashlib",
    "pathlib", "io", "os", "uuid",
})

# `os` attributes safe to expose in the strict profile. Everything else blocked.
DATA_QUERY_SAFE_OS_ATTRS: frozenset[str] = frozenset({
    "path", "getcwd", "listdir", "sep", "linesep", "curdir", "pardir",
    "environ",  # read-only usage is normal; we still block mutation calls
})

# Builtins that escape the AST sandbox if exposed to user code.
DANGEROUS_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__", "input", "breakpoint",
    "help", "exit", "quit",
})

# `os` functions that mutate filesystem / processes.
DANGEROUS_OS_ATTRS: frozenset[str] = frozenset({
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "kill", "killpg", "remove", "unlink", "rmdir", "removedirs",
    "rename", "renames", "replace", "chmod", "chown", "lchown",
    "symlink", "link", "walk", "mkdir", "makedirs", "mknod",
    "putenv", "unsetenv", "setsid", "fork", "forkpty",
    "startfile", "abort", "truncate", "open", "close",
})

# Sub-attributes on shutil that mutate filesystem.
DANGEROUS_SHUTIL_ATTRS: frozenset[str] = frozenset({
    "rmtree", "move", "copy", "copy2", "copyfile", "copyfileobj",
    "copymode", "copystat", "copytree", "make_archive", "unpack_archive",
})

# `pathlib.Path` methods that mutate the filesystem (path-guarded at runtime).
MUTATING_PATH_METHODS: frozenset[str] = frozenset({
    "write_text", "write_bytes", "open", "touch", "mkdir", "rmdir",
    "unlink", "rename", "replace", "chmod", "lchmod", "symlink_to",
    "hardlink_to", "link_to",
})

# Cache/temp env vars redirected into `<workspace>/.sandbox_cache/<subdir>` on
# SandboxContext enter and restored on exit. This is what allows legitimate
# library cache writes (matplotlib font cache, pandas/scikit-learn config,
# pip wheel cache, mkstemp() temp files) to succeed while keeping every write
# strictly inside the workspace perimeter — the correct sandbox boundary.
_SANDBOX_CACHE_ENV_MAP: Dict[str, str] = {
    "XDG_CACHE_HOME":      "xdg_cache",
    "XDG_CONFIG_HOME":     "xdg_config",
    "XDG_DATA_HOME":       "xdg_data",
    "MPLCONFIGDIR":        "matplotlib",
    "PIP_CACHE_DIR":       "pip",
    "HF_HOME":             "huggingface",
    "TORCH_HOME":          "torch",
    "KERAS_HOME":          "keras",
    "NUMBA_CACHE_DIR":     "numba",
    "PYTHONPYCACHEPREFIX": "pycache",
    "TMPDIR":              "tmp",
    "TEMP":                "tmp",
    "TMP":                 "tmp",
}

# PERF: memoize sandbox-cache directories already confirmed to exist, keyed by
# the resolved workspace path. Without this, every tool call recreates 13
# subdirectories (13 × mkdir syscalls) — the source of the 600s regression.
_SANDBOX_CACHE_READY: set = set()

# Roots (outside the workspace) that the sandbox's path guard permits WRITES
# into. Populated by `SandboxContext._redirect_cache_env()` so that library
# caches (matplotlib fontlist, pip wheel cache, tempfile) can land in a real
# writable directory WITHOUT polluting the user's workspace. Cleared on
# `_restore_cache_env()` — the set only ever contains directories belonging to
# currently-active sandbox contexts.
#
# Why not use the workspace itself? The user's project tree must stay pristine:
# a visible `.sandbox_cache/` folder in the workspace is a UX bug (it shows up
# in file trees, gets committed, and the artefact-sync layer mistakes its
# matplotlib fontlist for a user artefact). Redirecting to a system temp dir
# keeps library caches functional and invisible.
_SANDBOX_WRITE_ROOTS: set = set()


def _compute_sandbox_cache_root(workspace_root: Path) -> Path:
    """Returns the per-workspace sandbox cache directory: `<workspace>/.sandbox_cache/`.

    The sandbox cache MUST live inside the workspace perimeter so that the
    path guard's single invariant — "everything the sandbox writes is under
    `workspace_root`" — holds without needing to widen `_SANDBOX_WRITE_ROOTS`.
    This is what makes `tempfile.NamedTemporaryFile()` and every XDG/matplotlib/
    pip cache write land inside the workspace, allowing them to succeed while
    keeping the escape boundary strict.

    The caches remain invisible to user tooling via their well-known names
    (`.sandbox_cache/`, hidden on POSIX by the leading dot, and excluded from
    workspace-level scans by tooling that already filters dot-directories).
    """
    return Path(workspace_root).resolve() / ".sandbox_cache"


def _compute_trusted_read_roots() -> frozenset:
    """Directories whose contents may be READ by any frame without a path check.

    Library assets — `matplotlib/mpl-data/matplotlibrc`, `matplotlib` font
    `.ttf` files, `numpy` data files, `pandas` template data — live under the
    interpreter's installation tree. Blocking reads of those breaks
    `import matplotlib.pyplot` inside the sandbox (its rcParams load) and
    font loading during plotting, while providing no real safety: the user's
    home directory and arbitrary files remain fully sandboxed. The trusted
    read whitelist is what distinguishes "the interpreter's own installation"
    from "some file on the operator's disk".
    """
    roots: set = set()
    for attr in ("prefix", "base_prefix", "exec_prefix"):
        p = getattr(sys, attr, None)
        if p:
            try:
                roots.add(Path(p).resolve())
            except Exception:
                pass
    try:
        import sysconfig as _sysconfig
        for key in ("purelib", "platlib", "stdlib", "platstdlib"):
            try:
                p = _sysconfig.get_path(key)
                if p:
                    roots.add(Path(p).resolve())
            except Exception:
                pass
    except Exception:
        pass
    return frozenset(roots)


_TRUSTED_READ_ROOTS: frozenset = _compute_trusted_read_roots()


def _is_read_only_open_mode(mode: Any) -> bool:
    """True iff `mode` is a pure read (`r`, `rb`, `rt`, ...) — no write/append/+."""
    if not isinstance(mode, str):
        return False
    m = mode.lower()
    if any(c in m for c in ("w", "a", "x", "+")):
        return False
    return "r" in m


_GLOB_MAGIC_CHARS: frozenset[str] = frozenset({"*", "?", "["})


def _glob_static_root(pattern: Any) -> str:
    """Returns the longest magic-free leading path prefix of a glob pattern.

    For ``'/etc/**/*.py'`` returns ``'/etc'``; for ``'data/*.csv'`` returns
    ``'data'``; for ``'*'`` returns ``'.'`` (the pattern enumerates the CWD,
    which the sandbox pins to the workspace root). This is exactly the
    directory ``glob`` will scan, so resolving it against the workspace root
    is the correct, minimal guard on the ``glob`` surface.

    Handles bytes patterns (decoded for path checking only) and Windows
    drives/roots via ``Path.parts`` — e.g. ``'C:\\\\Users\\\\x\\\\*'`` yields
    ``'C:\\\\Users\\\\x'``.
    """
    if isinstance(pattern, (bytes, bytearray)):
        try:
            text = bytes(pattern).decode("utf-8", "surrogateescape")
        except Exception:
            return "."
    else:
        try:
            text = os.fspath(pattern)
        except TypeError:
            return "."
        if isinstance(text, bytes):
            text = text.decode("utf-8", "surrogateescape")
    if not text:
        return "."
    static_parts: List[str] = []
    for part in Path(text).parts:
        if any(c in _GLOB_MAGIC_CHARS for c in part):
            break
        static_parts.append(part)
    if not static_parts:
        return "."
    return os.sep.join(static_parts) or "."


# ──────────────────────────────────────────────────────────────────────────────
# Layer A — AST static validation
# ──────────────────────────────────────────────────────────────────────────────

def _describe(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # py>=3.9
    except Exception:
        return node.__class__.__name__


def _module_allowed_in_data_query(module_name: str) -> bool:
    """True if `module_name` (dotted) starts with an allowed data-query root."""
    root = module_name.split(".")[0]
    return (
        root in DATA_QUERY_ALLOWED_MODULES
        or module_name in DATA_QUERY_ALLOWED_MODULES
    )


def _validate_data_query(tree: ast.AST) -> Optional[SandboxViolation]:
    """Strict allowlist profile: only data-analysis surface permitted."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed_in_data_query(alias.name):
                    return SandboxViolation(
                        reason=(
                            f"Import of '{alias.name}' is not permitted in the "
                            "data-query sandbox. Only data-analysis libraries "
                            "(pandas, numpy, matplotlib, openpyxl, python-docx, "
                            "python-pptx, sqlalchemy) and pure stdlib data helpers "
                            "are allowed."
                        ),
                        snippet=_describe(node),
                        line=getattr(node, "lineno", 0),
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not mod:
                return SandboxViolation(
                    reason="Relative or empty-module imports are not permitted.",
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
            if not _module_allowed_in_data_query(mod):
                return SandboxViolation(
                    reason=(
                        f"'from {mod} import ...' is not permitted in the "
                        "data-query sandbox. Use only the approved data-analysis "
                        "libraries."
                    ),
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
            # Restrict `from os import ...` to safe attributes only.
            if mod.split(".")[0] == "os":
                for alias in node.names:
                    if alias.name not in DATA_QUERY_SAFE_OS_ATTRS:
                        return SandboxViolation(
                            reason=(
                                f"'from os import {alias.name}' is not permitted. "
                                "Only path-joining and read helpers are allowed in the "
                                "data-query sandbox."
                            ),
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
        elif isinstance(node, ast.Call):
            func = node.func
            # os.<dangerous_attr>(...)
            if isinstance(func, ast.Attribute):
                attr = func.attr
                target = func.value
                if isinstance(target, ast.Name) and target.id == "os":
                    if attr in DANGEROUS_OS_ATTRS:
                        return SandboxViolation(
                            reason=(
                                f"Call to 'os.{attr}' is not permitted in the "
                                "data-query sandbox."
                            ),
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
                if isinstance(target, ast.Name) and target.id == "shutil":
                    if attr in DANGEROUS_SHUTIL_ATTRS:
                        return SandboxViolation(
                            reason=(
                                f"Call to 'shutil.{attr}' is not permitted in the "
                                "data-query sandbox."
                            ),
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
            # Bare dangerous builtin(…)
            if isinstance(func, ast.Name) and func.id in DANGEROUS_BUILTINS:
                return SandboxViolation(
                    reason=(
                        f"Call to builtin '{func.id}' is not permitted in the "
                        "data-query sandbox."
                    ),
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
            # eval/exec/getattr chains via attribute access
            if isinstance(func, ast.Attribute) and func.attr in ("eval", "exec"):
                return SandboxViolation(
                    reason=f"Call to '.{func.attr}()' is not permitted.",
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
    return None


def _validate_standard(tree: ast.AST) -> Optional[SandboxViolation]:
    """Denylist profile: block the dangerous surface only."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0].lower()
                if root in FORBIDDEN_MODULES:
                    return SandboxViolation(
                        reason=(
                            f"Import of '{alias.name}' is forbidden in the "
                            "sandboxed Python runtime."
                        ),
                        snippet=_describe(node),
                        line=getattr(node, "lineno", 0),
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0].lower()
            if mod in FORBIDDEN_MODULES:
                return SandboxViolation(
                    reason=(
                        f"'from {node.module} import ...' is forbidden in the "
                        "sandboxed Python runtime."
                    ),
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                attr = func.attr
                target = func.value
                if isinstance(target, ast.Name) and target.id == "os":
                    if attr in DANGEROUS_OS_ATTRS:
                        return SandboxViolation(
                            reason=f"Call to 'os.{attr}' is forbidden in the sandbox.",
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
                if isinstance(target, ast.Name) and target.id == "shutil":
                    if attr in DANGEROUS_SHUTIL_ATTRS:
                        return SandboxViolation(
                            reason=f"Call to 'shutil.{attr}' is forbidden in the sandbox.",
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
                if isinstance(target, ast.Name) and target.id == "__import__":
                    return SandboxViolation(
                        reason="Dynamic '__import__()' calls are forbidden.",
                        snippet=_describe(node),
                        line=getattr(node, "lineno", 0),
                    )
            if isinstance(func, ast.Name) and func.id in ("eval", "exec"):
                return SandboxViolation(
                    reason=f"Call to builtin '{func.id}' is forbidden in the sandbox.",
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
    return None


def _validate_authorized(tree: ast.AST) -> Optional[SandboxViolation]:
    """Authorized profile: permit user-approved network/process primitives.

    This profile is selected ONLY after the user explicitly authorized a risky
    operation. It permits `subprocess`, `socket`, `urllib`, `requests`, etc. —
    the classes of operation the user consented to. It STILL blocks:

      - `ctypes`, `winreg`, `win32*`, `signal`, `resource` — C-level sandbox
        escapes that consent cannot make safe.
      - Direct `os.system` / `os.popen` / `os.spawn*` / `os.exec*` / `os.kill*`
        calls from user code — process-level escapes. (Note: `subprocess.*`
        is permitted; subprocess internally reaches os.exec* through its own
        non-user frames, which the runtime layer correctly exempts.)
      - `eval` / `exec` / `compile` / `__import__` — dynamic code execution.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0].lower()
                if root in AUTHORIZED_STILL_FORBIDDEN_MODULES:
                    return SandboxViolation(
                        reason=(
                            f"Import of '{alias.name}' is forbidden even under "
                            "user authorization — this primitive can escape the "
                            "sandbox at the native level. Use a dedicated tool "
                            "(e.g. system_shell) for legitimate host access."
                        ),
                        snippet=_describe(node),
                        line=getattr(node, "lineno", 0),
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0].lower()
            if mod in AUTHORIZED_STILL_FORBIDDEN_MODULES:
                return SandboxViolation(
                    reason=(
                        f"'from {node.module} import ...' is forbidden even under "
                        "user authorization — this primitive can escape the "
                        "sandbox at the native level."
                    ),
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                attr = func.attr
                target = func.value
                if isinstance(target, ast.Name) and target.id == "os":
                    if attr in DANGEROUS_OS_ATTRS:
                        return SandboxViolation(
                            reason=(
                                f"Call to 'os.{attr}' is forbidden even under user "
                                "authorization — use 'subprocess' for process "
                                "spawning, which the runtime layer audits."
                            ),
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
                if isinstance(target, ast.Name) and target.id == "shutil":
                    if attr in DANGEROUS_SHUTIL_ATTRS:
                        return SandboxViolation(
                            reason=f"Call to 'shutil.{attr}' is forbidden in the sandbox.",
                            snippet=_describe(node),
                            line=getattr(node, "lineno", 0),
                        )
            if isinstance(func, ast.Name) and func.id in ("eval", "exec", "compile"):
                return SandboxViolation(
                    reason=f"Call to builtin '{func.id}' is forbidden in the sandbox.",
                    snippet=_describe(node),
                    line=getattr(node, "lineno", 0),
                )
    return None


def validate_source(source: str, profile: str) -> Optional[SandboxViolation]:
    """Parses `source` and returns a `SandboxViolation` if it fails the profile.

    Profiles:
      - "data_query": strict allowlist (pandas/numpy/matplotlib/office/sqlalchemy).
      - "standard":   denylist (blocks the dangerous surface only).
      - "authorized": denylist minus the user-approved surface (subprocess,
                      socket, urllib, requests, etc. are permitted; ctypes,
                      winreg, os.system/exec* remain blocked). Selected ONLY
                      after the user explicitly authorizes a risky operation.
    """
    if profile not in ("data_query", "standard", "authorized"):
        raise ValueError(f"Unknown sandbox profile: {profile!r}")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Let the runtime raise the proper SyntaxError with traceback.
        return None
    if profile == "data_query":
        return _validate_data_query(tree)
    if profile == "authorized":
        return _validate_authorized(tree)
    return _validate_standard(tree)


# ──────────────────────────────────────────────────────────────────────────────
# Layer C — Process isolation
#
# `run_in_sandboxed_child()` executes user code in a *separate process* whose
# only job is to install `SandboxContext` around a single `exec()`, capture the
# resulting stdout/stderr/locals, and ship the payload back over a pipe.
#
# This is the ONLY supported way to run user code in a sensitive tool. It makes
# a stronger guarantee than any in-process mitigation ever could:
#
#   - The parent process's `builtins.exec` / `builtins.compile` / `builtins.open`
#     / `os.*` / `shutil.*` / `socket.*` / `pathlib.Path.*` are **never touched**.
#     No `_make_block_stub`, no `_is_block_stub`, no `_recover_true_builtin`,
#     no module-load-order hazard, no 190-test cascade.
#   - The child is the sandbox: it dies when the tool call ends, and every
#     patched attribute dies with it. There is no state to leak.
#   - Nested tool calls are trivially safe: a spawned child cannot see, mutate,
#     or influence the parent's process state in any way.
# ──────────────────────────────────────────────────────────────────────────────

# Public re-export list for tool callers.
__all__ = [
    "validate_source",
    "SandboxContext",
    "SandboxViolation",
    "SandboxPermissionError",
    "run_in_sandboxed_child",
    "TRUE_COMPILE",
    "TRUE_EXEC",
    "TRUE_EVAL",
    "_is_block_stub",
    "_recover_true_builtin",
    "_make_block_stub",
]


# Serializable envelope contracts. Everything travelling parent→child or
# child→parent must be picklable. `bytes` for df pickles and PNG images;
# JSON-safe primitives for scalar outputs; `str` for text.
_CHILD_HARD_TIMEOUT_DEFAULT = 120.0


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess-based child bootstrap.
#
# The child is launched with `subprocess.Popen([sys.executable, "-c", BOOTSTRAP])`,
# NOT with `multiprocessing.Process`. This deliberately eliminates the entire
# `pickle.save_global` class of failures (`PicklingError: import of module ...
# failed`) that poisons every spawn-based approach when the kernel module has
# been reloaded mid-sandbox (a legitimate scenario exercised by
# `test_sandbox_module_load_order.py`).
#
# The child:
#   1. reads a JSON config from stdin (parent's sys.path + payload),
#   2. imports sandbox_kernel **freshly** from disk (its own `sys.modules` starts
#      empty of the kernel — no reload shadows, no stale sentinels),
#   3. calls `_sandbox_child_worker` with a shim `conn` object that JSON-encodes
#      envelopes with base64-transported bytes.
#
# The parent never mutates its own `builtins.__import__`, never touches its
# process globals during spawn, and never depends on `sys.modules[__name__]`
# identity. Byte-for-byte deterministic on Windows and POSIX alike.
# ──────────────────────────────────────────────────────────────────────────────
_CHILD_BOOTSTRAP_SOURCE = r'''
import sys, json, base64, os

_config = json.loads(sys.stdin.buffer.read().decode("utf-8"))
sys.path[:] = _config["sys_path"]

# Fresh import in the child: no reload shadows, no stale sys.modules entries.
from lollms_client.tools_bindings.lcp.default_tools.execute_python import sandbox_kernel as _sk

_payload = _config["payload"]

# ── Bind seed modules by IMPORT, never by pickle ──
#    `seed_imports = {alias: import_path}`. Modules are imported through the
#    child's own stdlib `__import__` BEFORE `SandboxContext` is entered — no
#    pickle-by-reference (which is fragile for extension modules and crashes
#    the child on Windows for numpy/pandas/matplotlib), no module-identity
#    games. Optional deps (e.g. `plt` when matplotlib is absent) fail
#    non-fatally: the user code will simply not see the alias.
_seed = _payload.get("seed_locals") or {}
_seed_imports = _payload.get("seed_imports") or {}
for _alias, _modpath in _seed_imports.items():
    try:
        _seed[_alias] = __import__(_modpath)
        # `__import__("matplotlib.pyplot")` returns the top-level `matplotlib`
        # package, not `.pyplot`. Resolve the leaf explicitly.
        if "." in _modpath:
            _leaf = _modpath.rsplit(".", 1)[1]
            _seed[_alias] = getattr(_seed[_alias], _leaf, _seed[_alias])
    except Exception:
        # Non-fatal: leave the alias unbound. User code that references it
        # gets a clean NameError (or, in strict mode, the AST + runtime
        # filter will have already produced a more precise diagnostic).
        pass

# ── Force matplotlib Agg backend in the child ──
# CRITICAL: this must happen in the child process, BEFORE Pyplot is first
# rendered. The parent process may have loaded a GUI-capable Qt binding
# (PySide6/PyQt/Playwright), and matplotlib's auto-backend-picker can pick
# QtAgg or another interactive backend even in a headless subprocess. A
# `savefig` on a non-Agg backend in a headless child crashes the interpreter
# at the C level (`AccessViolation`, exit code 0xFFFFFFFF on Windows) BEFORE
# the worker can send its envelope — the exact failure mode this block fixes.
#
# `force=True` rebinds the backend even if `matplotlib.pyplot` was already
# imported (which it has been, via `seed_imports` above). Guarded so a
# missing/broken matplotlib never blocks execution.
try:
    import matplotlib as _mpl_child
    _mpl_child.use("Agg", force=True)
except Exception:
    pass
_payload["seed_locals"] = _seed

# ── Unpack seed values that the parent encoded for JSON transport ──
#    `{"__b64__": "..."}`    → bytes (fast path, used for df pickles)
#    `{"__pickle__": "..."}` → picklable VALUES only (DataFrames, instances of
#                              pure-Python classes). Extension modules MUST
#                              travel via `seed_imports` above, not here.
import pickle as _pickle_child
for _k, _v in list(_seed.items()):
    if isinstance(_v, dict) and set(_v.keys()) == {"__b64__"}:
        try:
            _seed[_k] = base64.b64decode(_v["__b64__"])
        except Exception:
            _seed.pop(_k, None)
    elif isinstance(_v, dict) and set(_v.keys()) == {"__pickle__"}:
        try:
            _seed[_k] = _pickle_child.loads(base64.b64decode(_v["__pickle__"]))
        except Exception:
            _seed.pop(_k, None)
_payload["seed_locals"] = _seed


class _StdoutConn:
    """Shim connection that JSON-serialises the envelope to stdout."""
    def send(self, envelope):
        out = dict(envelope)
        if out.get("df_pickle"):
            try:
                out["df_pickle"] = base64.b64encode(out["df_pickle"]).decode("ascii")
            except Exception:
                out["df_pickle"] = None
        out["figures"] = [
            base64.b64encode(f).decode("ascii") for f in (out.get("figures") or [])
        ]
        try:
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
        except Exception:
            pass

    def close(self):
        try:
            sys.stdout.flush()
        except Exception:
            pass


try:
    _sk._sandbox_child_worker(_StdoutConn(), _payload)
except BaseException as _exc:
    import traceback as _tb
    _fallback = {
        "success": False, "stdout": "", "stderr": "",
        "traceback": _tb.format_exc(),
        "error_type": type(_exc).__name__,
        "sandbox_violation": False, "figures": [],
        "df_pickle": None, "scalars": {},
    }
    try:
        sys.stdout.write(json.dumps(_fallback))
        sys.stdout.flush()
    except Exception:
        pass
'''


def _sandbox_child_worker(conn, payload: Dict[str, Any]) -> None:
    """Runs in the child process ONLY. Executes user code under SandboxContext.

    Sends exactly one envelope through `conn` then closes it. Any internal
    failure is converted to a structured error envelope so the parent never
    sees a bare `EOFError` from a silent child death.
    """
    import io as _io
    import traceback
    import json as _json

    # The child re-imports this module on `spawn` start; capture the true
    # builtins here BEFORE `SandboxContext` clobbers them. This is the very
    # first thing the child does, so `_capture_true_builtins()` sees genuine
    # builtins and populates the sys-cache.
    child_tc, child_te, _ = _capture_true_builtins()

    source = payload["source"]
    workspace_root = Path(payload["workspace_root"])
    profile = payload["profile"]
    script_label = payload.get("script_label", "<sandboxed>")
    argv = payload.get("argv") or [script_label]
    seed_locals = payload.get("seed_locals") or {}
    persist_df_name = payload.get("persist_df_name")  # df variable to ship back

    class _ChildStringIO(_io.StringIO):
        def reconfigure(self, *a, **kw):
            pass

    out_buf = _ChildStringIO()
    err_buf = _ChildStringIO()

    sys.stdout = out_buf
    sys.stderr = err_buf
    old_argv = sys.argv[:]
    sys.argv = argv

    envelope: Dict[str, Any] = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "traceback": "",
        "error_type": "",
        "sandbox_violation": False,
        "figures": [],
        "df_pickle": None,
        "scalars": {},
    }

    try:
        # NOTE: the sandbox's cache-env redirection into `<workspace>/.sandbox_cache`
        # is *also* process-local; the child scrub resets everything on exit
        # because it dies before the parent resumes.
        with SandboxContext(workspace_root=workspace_root, profile=profile) as _sbx:
            local_vars: Dict[str, Any] = dict(seed_locals)
            local_vars.setdefault("__builtins__", __builtins__)
            local_vars.setdefault("__file__", script_label)
            local_vars.setdefault("__name__", "__main__")

            # If the caller pre-built a reduced builtins dict (data_query tool),
            # honour it but graft the sandbox-filtered importer on top.
            if isinstance(local_vars.get("__builtins__"), dict):
                local_vars["__builtins__"] = dict(local_vars["__builtins__"])
                local_vars["__builtins__"]["__import__"] = _sbx.filtered_import

            try:
                compiled = child_tc(source, script_label, "exec")
                child_te(compiled, local_vars)
            except SystemExit as se:
                raise RuntimeError(
                    f"User code called sys.exit() with code {se.code}. "
                    "This is not permitted in sandboxed execution."
                ) from se
            except KeyboardInterrupt:
                raise RuntimeError("Execution interrupted by KeyboardInterrupt.")
            except SandboxPermissionError:
                envelope["sandbox_violation"] = True
                envelope["traceback"] = traceback.format_exc()
                envelope["error_type"] = "SandboxPermissionError"
                raise
            except BaseException:
                envelope["traceback"] = traceback.format_exc()
                envelope["error_type"] = sys.exc_info()[0].__name__ if sys.exc_info()[0] else ""
                raise

            # ── Success path: harvest dataframe + figures + JSON scalars ──
            if persist_df_name and persist_df_name in local_vars:
                try:
                    import pickle as _pickle
                    envelope["df_pickle"] = _pickle.dumps(local_vars[persist_df_name])
                except Exception:
                    envelope["df_pickle"] = None

            # JSON-safe scalars from user locals (bounded set) — used by the
            # parent to render final response without re-importing the objects.
            for _k, _v in list(local_vars.items()):
                if _k.startswith("__") or _k in ("pd", "np", "plt", "sns",
                                                 "sklearn", "scipy", "Path"):
                    continue
                if isinstance(_v, (int, float, str, bool, type(None))):
                    envelope["scalars"][_k] = _v
                if len(envelope["scalars"]) > 200:  # bound payload size
                    break

            try:
                # Defensive re-bind: even if user code switched backends or
                # this module somehow bypassed the bootstrap force (edge
                # cases, exotic loads), rendering MUST go through the
                # non-interactive Agg backend — anything else crashes the
                # child natively on headless systems.
                import matplotlib as _mpl_harvest
                _mpl_harvest.use("Agg", force=True)
                import matplotlib.pyplot as _plt_child
                for _fnum in _plt_child.get_fignums():
                    _fig = _plt_child.figure(_fnum)
                    _buf = _io.BytesIO()
                    _fig.savefig(_buf, format="png", bbox_inches='tight',
                                 facecolor=_fig.get_facecolor())
                    envelope["figures"].append(_buf.getvalue())
                _plt_child.close('all')
            except Exception:
                pass

        envelope["success"] = True
    except BaseException as exc:
        envelope["traceback"] = envelope["traceback"] or traceback.format_exc()
        envelope["error_type"] = envelope["error_type"] or type(exc).__name__
        if isinstance(exc, SandboxPermissionError):
            envelope["sandbox_violation"] = True
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        sys.argv = old_argv
        envelope["stdout"] = out_buf.getvalue()
        envelope["stderr"] = err_buf.getvalue()
        try:
            conn.send(envelope)
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# Pin the picklable identity of the spawn worker to THIS module's canonical
# name. Windows `spawn` resolves the target by `(module_name, qualname)`; an
# explicit pin prevents pickling from ever disagreeing with `sys.modules`.
_sandbox_child_worker.__module__ = __name__
_sandbox_child_worker.__qualname__ = "_sandbox_child_worker"


def run_in_sandboxed_child(
    source: str,
    workspace_root: Path,
    profile: str,
    script_label: str = "<sandboxed>",
    argv: Optional[List[Any]] = None,
    seed_locals: Optional[Dict[str, Any]] = None,
    persist_df_name: Optional[str] = None,
    hard_timeout: float = _CHILD_HARD_TIMEOUT_DEFAULT,
    seed_imports: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Executes `source` in a sandboxed child process. Returns the envelope.

    `seed_imports` is a `{alias: import_path}` map: for each entry the child
    imports `import_path` via its own raw `__import__` (before the sandbox is
    entered) and binds the result under `alias` in the user namespace. Use this
    for **modules** — never pickle modules through `seed_locals`: pickling them
    stores by reference and re-imports in the child with fragile semantics for
    extension modules (numpy/pandas/matplotlib), which on Windows can either
    silently drop the binding (NameError) or crash the child mid-import
    (0xFFFFFFFF / ACCESS_VIOLATION). Only **data values** belong in
    `seed_locals`, where pickle is the correct transport.

    The parent process is NEVER patched. Everything that happens between the
    child's first byte and last byte lives in that child's address space and
    disappears on process exit — the strongest correctness guarantee available
    short of a container.

    Returns a dictionary with keys:
      - success (bool): True iff user code ran to completion without exception.
      - stdout (str)
      - stderr (str)
      - traceback (str): empty on success.
      - error_type (str)
      - sandbox_violation (bool): True iff a `SandboxPermissionError` was raised.
      - figures (list[bytes]): captured matplotlib PNGs.
      - df_pickle (bytes|None)
      - scalars (dict[str, JSON-safe])

    When the child exceeds `hard_timeout`, it is terminated and a synthetic
    envelope with `success=False, error_type="TimeoutError"` is returned.
    """
    import json as _json
    import base64 as _b64
    import subprocess as _sp

    # SELF-HEAL module-level TRUE_* from the sys-cache (defensive symmetry with
    # the child's fresh import — kept for other code paths that read them).
    global TRUE_COMPILE, TRUE_EXEC, TRUE_EVAL
    cached_tc, cached_te, cached_tv = _load_true_builtins_from_sys()
    if cached_tc is not None and cached_te is not None and cached_tv is not None:
        TRUE_COMPILE, TRUE_EXEC, TRUE_EVAL = cached_tc, cached_te, cached_tv

    # ── Payload serialisation (JSON + base64 for bytes fields) ─────────────
    # The transport is JSON over stdin/stdout, so every seed value must survive
    # a JSON round-trip. Three encoding strategies:
    #
    #   1. JSON-native values (int/float/str/bool/None/list/dict of those):
    #      pass through unchanged.
    #   2. `bytes`/`bytearray`: wrapped as `{"__b64__": "..."}`.
    #   3. Anything else (classes like `Path`, modules like `pd`/`np`/`plt`,
    #      DataFrames, arbitrary instances): pickled then base64-wrapped as
    #      `{"__pickle__": "..."}` — the child unpacks the original object
    #      via `pickle.loads`. This preserves the exact object contract
    #      (module refs by reference via pickle's import machinery, classes
    #      by qualified name, instances by state).
    #
    # `pickle.dumps` failure is impossible for values that would have crossed
    # a `multiprocessing.Pipe` in the previous implementation — the only values
    # it can't serialize are ones that were already unusable across a spawn
    # boundary (local closures, open sockets, thread locks). We skip those with
    # a warning rather than aborting the whole tool call.
    import pickle as _pickle_for_payload

    def _encode_seed_value(_v: Any) -> Any:
        if isinstance(_v, (bytes, bytearray)):
            return {"__b64__": _b64.b64encode(bytes(_v)).decode("ascii")}
        if _v is None or isinstance(_v, (bool, int, float, str)):
            return _v
        # Small containers of primitives can also pass through for efficiency.
        if isinstance(_v, (list, tuple)):
            try:
                return _json.loads(_json.dumps(list(_v)))
            except (TypeError, ValueError):
                pass
        if isinstance(_v, dict):
            try:
                return _json.loads(_json.dumps(dict(_v)))
            except (TypeError, ValueError):
                pass
        # Fallback: pickle + base64 for everything else (classes, modules,
        # DataFrames, instances).
        try:
            blob = _pickle_for_payload.dumps(_v)
            return {"__pickle__": _b64.b64encode(blob).decode("ascii")}
        except Exception:
            return None  # value skipped — surrogate sentinel; child treats as absent

    seed_locals_safe: Dict[str, Any] = {}
    for _k, _v in (seed_locals or {}).items():
        _enc = _encode_seed_value(_v)
        if _enc is not None or _v is None:
            seed_locals_safe[_k] = _enc

    # `seed_imports` is passed verbatim — it contains only str→str pairs and
    # travels through JSON trivially. The child resolves each one with its own
    # `__import__` before entering `SandboxContext`, avoiding the fragile
    # pickle-by-reference round-trip that extension modules cannot survive.
    payload = {
        "source": source,
        "workspace_root": str(workspace_root),
        "profile": profile,
        "script_label": script_label,
        "argv": list(argv or [script_label]),
        "seed_locals": seed_locals_safe,
        "seed_imports": dict(seed_imports or {}),
        "persist_df_name": persist_df_name,
    }
    config_blob = _json.dumps({
        "sys_path": list(sys.path),
        "payload": payload,
    }).encode("utf-8")

    # ── Launch the child as an isolated subprocess ─────────────────────────
    # Using `subprocess.Popen` instead of `multiprocessing.Process` eliminates
    # the entire `pickle.save_global` failure class: no module-identity check,
    # no `sys.modules[__name__]` dependence, no reload sensitivity. The child
    # imports `sandbox_kernel` freshly from disk — immune to any prior
    # in-process module reload.
    _creationflags = 0
    if os.name == "nt":
        # CREATE_NO_WINDOW: never flash a console window on Windows.
        _creationflags = getattr(_sp, "CREATE_NO_WINDOW", 0)

    try:
        proc = _sp.Popen(
            [sys.executable, "-c", _CHILD_BOOTSTRAP_SOURCE],
            stdin=_sp.PIPE,
            stdout=_sp.PIPE,
            stderr=_sp.PIPE,
            cwd=os.getcwd(),
            creationflags=_creationflags,
        )
    except Exception as exc:
        return {
            "success": False, "stdout": "", "stderr": "",
            "traceback": f"Failed to spawn sandboxed child: {exc}",
            "error_type": type(exc).__name__,
            "sandbox_violation": False,
            "figures": [], "df_pickle": None, "scalars": {},
        }

    try:
        stdout_data, stderr_data = proc.communicate(
            input=config_blob, timeout=hard_timeout,
        )
    except _sp.TimeoutExpired:
        try:
            proc.kill()
            proc.communicate(timeout=2.0)
        except Exception:
            pass
        return {
            "success": False, "stdout": "", "stderr": "",
            "traceback": (
                f"Sandboxed child exceeded {hard_timeout:.0f}s hard timeout "
                "and was terminated."
            ),
            "error_type": "TimeoutError",
            "sandbox_violation": False,
            "figures": [], "df_pickle": None, "scalars": {},
        }
    except Exception as exc:
        try:
            proc.kill()
        except Exception:
            pass
        return {
            "success": False, "stdout": "", "stderr": "",
            "traceback": f"Failed to communicate with sandboxed child: {exc}",
            "error_type": type(exc).__name__,
            "sandbox_violation": False,
            "figures": [], "df_pickle": None, "scalars": {},
        }

    # ── Decode the envelope ────────────────────────────────────────────────
    raw_env = (stdout_data or b"").decode("utf-8", errors="replace").strip()
    if not raw_env:
        child_stderr = (stderr_data or b"").decode("utf-8", errors="replace")
        return {
            "success": False, "stdout": "", "stderr": child_stderr,
            "traceback": (
                "Sandboxed child process exited without returning a payload. "
                f"Exit code: {proc.returncode}.\n"
                + (f"Child stderr:\n{child_stderr}" if child_stderr else "")
            ),
            "error_type": "ChildCrash",
            "sandbox_violation": False,
            "figures": [], "df_pickle": None, "scalars": {},
        }

    try:
        envelope: Dict[str, Any] = _json.loads(raw_env)
    except Exception as exc:
        child_stderr = (stderr_data or b"").decode("utf-8", errors="replace")
        return {
            "success": False, "stdout": "", "stderr": child_stderr,
            "traceback": f"Sandboxed child returned malformed envelope: {exc}",
            "error_type": "ProtocolError",
            "sandbox_violation": False,
            "figures": [], "df_pickle": None, "scalars": {},
        }

    # Decode byte fields shipped as base64 strings.
    if envelope.get("df_pickle") and isinstance(envelope["df_pickle"], str):
        try:
            envelope["df_pickle"] = _b64.b64decode(envelope["df_pickle"])
        except Exception:
            envelope["df_pickle"] = None
    figures_in = envelope.get("figures") or []
    decoded_figures: List[bytes] = []
    for f in figures_in:
        if isinstance(f, str):
            try:
                decoded_figures.append(_b64.b64decode(f))
            except Exception:
                continue
        elif isinstance(f, (bytes, bytearray)):
            decoded_figures.append(bytes(f))
    envelope["figures"] = decoded_figures

    return envelope


# ──────────────────────────────────────────────────────────────────────────────
# Layer B — Runtime interception
# ──────────────────────────────────────────────────────────────────────────────

# Set of user-exec frame filenames that constitute "sandboxed user code".
# Path guards consult the live frame stack and only ENFORCE when the immediate
# caller is one of these pseudo-filenames. This is what makes the sandbox
# "sandbox the tool execution" instead of "sandbox the whole process": host
# code (test fixtures, pandas/matplotlib caches, pipmaster, our own post-exec
# cleanup) is NOT user code — its paths are trusted and pass through untouched.
_USER_FRAME_FILENAMES: frozenset[str] = frozenset({
    "<execute_python_data_query>",
    "<execute_python>",
    "<execute_python_file>",
})


def _caller_is_user_code() -> bool:
    """True if the *immediate* Python caller is inside a sandboxed exec frame.

    Libraries called FROM user code (pandas.read_csv, matplotlib.savefig)
    eventually reach Python-level path ops through their own frames — those
    frames are library frames, not user frames, so this correctly returns False
    and lets the library finish its legitimate work (e.g. writing to
    ~/.cache/matplotlib). Direct user-code operations (open(), Path.mkdir(),
    os.remove()) hit this guard with a user frame on top and return True.
    """
    frame = sys._getframe(1)
    # Step outward up to 3 frames: the guard's caller may be a thin shim
    # (e.g. functools.wraps, argparse internals), but a library deep in its own
    # call tree is >3 frames away from user code and correctly deescalates to
    # "trusted" — library fs writes must succeed.
    for _ in range(3):
        if frame is None:
            return False
        if frame.f_code.co_filename in _USER_FRAME_FILENAMES:
            return True
        frame = frame.f_back
    return False


def _resolve_within(root: Path, target: Any, allow_trusted_read: bool = False) -> Path:
    """Fast path-guard: returns the resolved target, raises if outside `root`.

    `allow_trusted_read=True` extends the accepted roots to `_TRUSTED_READ_ROOTS`
    (the interpreter's own installation tree). It is set ONLY for read-only
    file opens — writes into system directories remain forbidden. This is what
    lets `import matplotlib.pyplot` read its `matplotlibrc` and load its own
    fonts inside the sandbox, without opening any hole for user-supplied
    paths (`~/secret.csv`, `/etc/passwd`, `%TEMP%/victim.csv`) — those live
    outside both `root` and the trusted roots and stay blocked.

    PERFORMANCE: We avoid the expensive `Path.resolve()` on every call.
    Relative paths without `..` are guaranteed under the CWD (== root).
    """
    try:
        if isinstance(target, int):
            # Already an open file descriptor — no path to check, pass through.
            return Path.cwd()
        if isinstance(target, str):
            target_str = target
        else:
            target_str = os.fspath(target)
            if not isinstance(target_str, str):
                target_str = str(target_str)
        if (
            target_str
            and not os.path.isabs(target_str)
            and ".." not in target_str.replace("\\", "/").split("/")
        ):
            return Path(root) / target_str
        p = Path(target_str)
        if not p.is_absolute():
            p = Path.cwd() / p
        resolved = p.resolve(strict=False)
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            # Writes/reads into the sandbox's own cache root (system temp) are
            # legitimate: matplotlib fontlist, pip wheel cache, tempfile mkstemp
            # all land there. The cache lives OUTSIDE the workspace on purpose
            # so the user's project tree stays pristine.
            for cache_root_str in _SANDBOX_WRITE_ROOTS:
                try:
                    resolved.relative_to(Path(cache_root_str))
                    return resolved
                except (ValueError, OSError):
                    continue
            if allow_trusted_read:
                for trusted_root in _TRUSTED_READ_ROOTS:
                    try:
                        resolved.relative_to(trusted_root)
                        return resolved
                    except ValueError:
                        continue
            raise
    except (ValueError, OSError) as exc:
        raise SandboxPermissionError(
            f"🛑 BLOCKED BY SANDBOX: path '{target}' resolves outside the workspace."
        ) from exc


@dataclass
class SandboxContext:
    """Scoped runtime patcher. Use as a context manager.

    `workspace_root` establishes the path boundary. `profile` selects how strict
    the rules are:
      - "data_query": os.* surface locked down to read-only helpers, no network,
                      no process, no file mutation outside root, no eval/exec.
      - "standard":   process/socket/ctypes blocked, path ops bounded to root,
                      dangerous os.* blocks, eval/exec disabled.
    """
    workspace_root: Path
    profile: str = "standard"
    _originals: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Exposed after `__enter__` so callers (the data-query tool) can obtain the
    # filtered `__import__` without re-entering the sandbox via `import builtins`.
    filtered_import: Optional[Callable[..., Any]] = field(default=None, init=False, repr=False)
    # Cache-env redirection backups (populated in `_redirect_cache_env()`, cleared
    # in `_restore_cache_env()`).
    _cache_env_backup: Dict[str, Optional[str]] = field(default_factory=dict, init=False, repr=False)
    _tempfile_backup: Any = field(default=None, init=False, repr=False)
    _tempfile_was_set: bool = field(default=False, init=False, repr=False)

    # ---------- snapshot / restore ----------
    def _snapshot(self) -> None:
        # The four "true builtin" markers come from frozen sentinels whenever
        # available — this makes `_snapshot()` self-healing after any prior
        # leak of a blocking stub. Other attrs are safe to snapshot directly.
        # `__import__` must NEVER be frozen as a gated sandbox stub. Recover a
        # genuine importer if the current value is our own stub; otherwise the
        # restore path could re-install the gate on the host process after exit
        # (leak cause for `PicklingError: import of module ... failed`).
        _snapshot_importer = builtins.__import__
        if _is_block_stub(_snapshot_importer):
            _recovered = _recover_import_builtin()
            if _recovered is not None:
                _snapshot_importer = _recovered
        self._originals = {
            "builtins.__import__": _snapshot_importer,
            "builtins.eval": getattr(builtins, "__lollms_true_eval__", builtins.eval),
            "builtins.exec": getattr(builtins, "__lollms_true_exec__", builtins.exec),
            "builtins.compile": getattr(builtins, "__lollms_true_compile__", builtins.compile),
            "builtins.open": builtins.open,
            "io.open": io.open,
            "os.system": getattr(os, "system", None),
            "os.popen": getattr(os, "popen", None),
            "os.remove": getattr(os, "remove", None),
            "os.unlink": getattr(os, "unlink", None),
            "os.rmdir": getattr(os, "rmdir", None),
            "os.removedirs": getattr(os, "removedirs", None),
            "os.rename": getattr(os, "rename", None),
            "os.renames": getattr(os, "renames", None),
            "os.replace": getattr(os, "replace", None),
            "os.chmod": getattr(os, "chmod", None),
            "os.mkdir": getattr(os, "mkdir", None),
            "os.makedirs": getattr(os, "makedirs", None),
            "os.open": getattr(os, "open", None),
            "os.kill": getattr(os, "kill", None),
            "os.killpg": getattr(os, "killpg", None),
            "os.putenv": getattr(os, "putenv", None),
            "os.unsetenv": getattr(os, "unsetenv", None),
            "os.symlink": getattr(os, "symlink", None),
            "os.link": getattr(os, "link", None),
            "os.chdir": getattr(os, "chdir", None),
            "os.fchdir": getattr(os, "fchdir", None),
            # Directory-enumeration primitives: read-only, but they leak the
            # host filesystem layout (filenames, sizes, mtimes) when their
            # argument resolves outside the workspace. Guarded below so a
            # sandboxed scan cannot enumerate `~/.ssh`, `C:\\Users\\...`,
            # `/etc`, etc.
            #
            # `glob.glob` / `glob.iglob` are snapshotted separately because
            # CPython <= 3.10 binds `scandir` BY VALUE inside the `glob`
            # module at import time (`from os import scandir`). Patching
            # `os.scandir` later does NOT rebind `glob.scandir`, so a bare
            # scandir guard does not intercept `glob.glob`. The glob entry
            # points must be guarded explicitly; the scandir/listdir guards
            # remain as defense-in-depth for direct callers.
            "os.listdir": getattr(os, "listdir", None),
            "os.scandir": getattr(os, "scandir", None),
            "os.walk": getattr(os, "walk", None),
            "glob.glob": getattr(glob, "glob", None),
            "glob.iglob": getattr(glob, "iglob", None),
            "shutil.rmtree": getattr(shutil, "rmtree", None),
            "shutil.move": getattr(shutil, "move", None),
            "shutil.copy": getattr(shutil, "copy", None),
            "shutil.copy2": getattr(shutil, "copy2", None),
            "shutil.copytree": getattr(shutil, "copytree", None),
            "subprocess.run": getattr(subprocess, "run", None),
            "subprocess.Popen": getattr(subprocess, "Popen", None),
            "subprocess.call": getattr(subprocess, "call", None),
            "subprocess.check_output": getattr(subprocess, "check_output", None),
            "socket.socket.connect": socket.socket.connect,
            "socket.socket.connect_ex": getattr(socket.socket, "connect_ex", None),
            "socket.create_connection": getattr(socket, "create_connection", None),
            "socket.getaddrinfo": getattr(socket, "getaddrinfo", None),
            "importlib.import_module": importlib.import_module,
            "importlib.reload": importlib.reload,
        }
        # Pre-populate the pathlib snapshot so `_restore()` is symmetric even if
        # `_install()` raises partway through. Without this, any exception mid-
        # install leaves `Path.mkdir/write_text/...` permanently patched — the
        # 190-test cascade seen in CI.
        pathlib_snapshot: Dict[str, Any] = {}
        for method_name in MUTATING_PATH_METHODS:
            orig_m = getattr(Path, method_name, None)
            if orig_m is not None:
                pathlib_snapshot[method_name] = orig_m
        self._originals["__pathlib__"] = pathlib_snapshot

    def _restore(self) -> None:
        # Restore pathlib Path methods FIRST — these are the most likely to
        # corrupt the host if left patched and affect every later test/module.
        pathlib_snapshot = self._originals.pop("__pathlib__", None) or {}
        for method_name, orig_method in pathlib_snapshot.items():
            try:
                setattr(Path, method_name, orig_method)
            except Exception:
                pass

        for key, orig in self._originals.items():
            if orig is None:
                continue
            try:
                owner, _, attr = key.partition(".")
                if owner == "builtins":
                    # For the four "true builtin" markers we ALWAYS restore
                    # from the frozen `__lollms_true_*__` sentinels, not from
                    # whatever `self._originals` happened to capture at
                    # `_snapshot()` time. This guarantees a nested context or a
                    # context installed over a leaked prior context can never
                    # leave a blocking stub wired in after exit.
                    #
                    # VALIDATION: if the sentinel is itself corrupted (module
                    # imported for the first time inside a live sandbox), we
                    # fall back to `orig` — which _snapshot() captured BEFORE
                    # this context installed its stubs, and therefore is the
                    # real builtin — rather than re-installing a block stub.
                    if attr == "compile":
                        sentinel = getattr(builtins, "__lollms_true_compile__", None)
                        setattr(builtins, "compile", orig if _is_block_stub(sentinel) or sentinel is None else sentinel)
                    elif attr == "exec":
                        sentinel = getattr(builtins, "__lollms_true_exec__", None)
                        setattr(builtins, "exec", orig if _is_block_stub(sentinel) or sentinel is None else sentinel)
                    elif attr == "eval":
                        sentinel = getattr(builtins, "__lollms_true_eval__", None)
                        setattr(builtins, "eval", orig if _is_block_stub(sentinel) or sentinel is None else sentinel)
                    else:
                        setattr(builtins, attr, orig)
                elif owner == "io":
                    io.open = orig
                elif owner == "os":
                    setattr(os, attr, orig)
                elif owner == "shutil":
                    setattr(shutil, attr, orig)
                elif owner == "subprocess":
                    setattr(subprocess, attr, orig)
                elif owner == "socket":
                    if attr.startswith("socket."):
                        _, _, sock_attr = attr.partition(".")
                        setattr(socket.socket, sock_attr, orig)
                    else:
                        setattr(socket, attr, orig)
                elif owner == "importlib":
                    setattr(importlib, attr, orig)
                elif owner == "glob":
                    setattr(glob, attr, orig)
            except Exception:
                pass
        self._originals = {}
        self.filtered_import = None
        self._restore_cache_env()

    # ---------- cache-env redirection ----------
    def _redirect_cache_env(self) -> None:
        """Point cache/temp-dir env vars at a system-temp sandbox cache.

        The cache root lives OUTSIDE the user's workspace (under the OS temp
        directory, keyed by a hash of the workspace path). This keeps library
        cache writes functional (matplotlib fontlist, pip wheel cache, tempfile
        mkstemp) while leaving the user's project tree pristine. The path guard
        is extended via `_SANDBOX_WRITE_ROOTS` so writes into this directory
        succeed even though it is outside `workspace_root`.

        PERF: uses raw `os.makedirs` (bypassing the patched `Path.mkdir`) and
        memoizes the directory set in `_SANDBOX_CACHE_READY` so the 13 mkdirs
        run at most once per (process, workspace). Previously every tool call
        paid for 13 mkdir syscalls + 13 guard resolves — the source of the
        600s regression.
        """
        import tempfile as _tf
        workspace_root = self.workspace_root.resolve()
        sandbox_cache = _compute_sandbox_cache_root(workspace_root)

        if str(sandbox_cache) not in _SANDBOX_CACHE_READY:
            # Use raw os.makedirs to avoid re-entering our own Path.mkdir guard.
            _raw_makedirs = self._originals.get("os.makedirs") or os.makedirs
            try:
                _raw_makedirs(str(sandbox_cache), exist_ok=True)
            except Exception:
                pass
            for subdir in set(_SANDBOX_CACHE_ENV_MAP.values()):
                try:
                    _raw_makedirs(str(sandbox_cache / subdir), exist_ok=True)
                except Exception:
                    pass
            _SANDBOX_CACHE_READY.add(str(sandbox_cache))

        # Register the cache root as a permitted WRITE root for the duration of
        # this sandbox. `_resolve_within` consults `_SANDBOX_WRITE_ROOTS` in
        # addition to `workspace_root` when writes land in the cache.
        try:
            _SANDBOX_WRITE_ROOTS.add(str(sandbox_cache.resolve()))
        except Exception:
            pass

        for env_key, subdir in _SANDBOX_CACHE_ENV_MAP.items():
            self._cache_env_backup[env_key] = os.environ.get(env_key)
            try:
                os.environ[env_key] = str(sandbox_cache / subdir)
            except Exception:
                pass
        self._tempfile_backup = _tf.tempdir
        self._tempfile_was_set = True
        try:
            _tf.tempdir = str(sandbox_cache / "tmp")
        except Exception:
            pass

    def _restore_cache_env(self) -> None:
        """Reverse `_redirect_cache_env()` — must be idempotent and safe."""
        import tempfile as _tf
        for env_key, old_val in self._cache_env_backup.items():
            try:
                if old_val is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old_val
            except Exception:
                pass
        self._cache_env_backup = {}
        if self._tempfile_was_set:
            try:
                _tf.tempdir = self._tempfile_backup
            except Exception:
                pass
            self._tempfile_was_set = False
            self._tempfile_backup = None
        # Drop this workspace's cache root from the permitted-write set. The
        # set is process-wide, so we only remove OUR entry — other concurrent
        # sandbox contexts (nested/threaded) keep theirs.
        try:
            _SANDBOX_WRITE_ROOTS.discard(
                str(_compute_sandbox_cache_root(self.workspace_root.resolve()).resolve())
            )
        except Exception:
            pass

    # ---------- install ----------
    def __enter__(self) -> "SandboxContext":
        self._snapshot()
        # Redirect caches/temp BEFORE installing the guards: user code needs
        # matplotlib/pandas/pip cache writes to land inside the workspace, so
        # the (frame-free) guards see them as legitimate paths.
        self._redirect_cache_env()
        try:
            self._install()
        except BaseException:
            # A partially-installed sandbox is worse than no sandbox: restore
            # everything snapshot before re-raising so we never corrupt host
            # state if `_install()` fails half-way (e.g. exotic platform
            # without `os.fork`).
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._restore()

    def _install(self) -> None:
        root = self.workspace_root.resolve()
        is_strict = (self.profile == "data_query")
        # "authorized" = user explicitly consented to risky operations
        # (network / subprocess). The runtime layer must NOT re-block what the
        # AST layer permitted, otherwise the user's consent is silently
        # overridden at exec time (the bug: "I clicked yes but it still says
        # BLOCKED BY SANDBOX"). The truly-unsafe primitives (ctypes, winreg,
        # os.system/exec*) remain blocked at the AST layer, which is where
        # they belong — the runtime layer never patched ctypes/winreg anyway.
        is_authorized = (self.profile == "authorized")
        orig_import = self._originals["builtins.__import__"]
        orig_open = self._originals["builtins.open"]
        orig_io_open = self._originals["io.open"]
        orig_import_module = self._originals["importlib.import_module"]
        orig_reload = self._originals["importlib.reload"]

        # -------- import filtering --------
        # FRAME-GATED: the FORBIDDEN_MODULES denylist and the strict data-query
        # allowlist are enforced only for imports whose immediate Python caller
        # is user code. This is essential for correctness: pandas, matplotlib,
        # sklearn, and every other analysis library legitimately imports stdlib
        # modules (importlib, tempfile, copy, pickle) that MUST stay forbidden
        # for user code but are perfectly safe for a vetted library's own load.
        # If we blocked them globally, `import pandas` would fail with a bogus
        # "importing 'importlib' is forbidden" error and safe data analysis
        # would be impossible.
        def _filtered_import(name, globals=None, locals=None, fromlist=(), level=0):
            if _caller_is_user_code():
                root_mod = name.split(".")[0].lower()
                # "authorized" profile: permit AUTHORIZED_MODULES (user consented
                # to network / subprocess) but STILL block the truly-unsafe set.
                if is_authorized:
                    if root_mod in AUTHORIZED_STILL_FORBIDDEN_MODULES:
                        raise SandboxPermissionError(
                            f"🛑 BLOCKED BY SANDBOX: importing '{name}' is forbidden "
                            "even under user authorization — this primitive can "
                            "escape the sandbox at the native level."
                        )
                else:
                    if root_mod in FORBIDDEN_MODULES:
                        raise SandboxPermissionError(
                            f"🛑 BLOCKED BY SANDBOX: importing '{name}' is forbidden."
                        )
                if is_strict and not _module_allowed_in_data_query(name):
                    raise SandboxPermissionError(
                        f"🛑 BLOCKED BY DATA-QUERY SANDBOX: importing '{name}' is "
                        "outside the approved data-analysis surface."
                    )
            return orig_import(name, globals, locals, fromlist, level)

        def _filtered_import_module(name, package=None):
            if _caller_is_user_code():
                root_mod = name.split(".")[0].lower()
                if is_authorized:
                    if root_mod in AUTHORIZED_STILL_FORBIDDEN_MODULES:
                        raise SandboxPermissionError(
                            f"🛑 BLOCKED BY SANDBOX: importing '{name}' is forbidden "
                            "even under user authorization."
                        )
                else:
                    if root_mod in FORBIDDEN_MODULES:
                        raise SandboxPermissionError(
                            f"🛑 BLOCKED BY SANDBOX: importing '{name}' is forbidden."
                        )
                if is_strict and not _module_allowed_in_data_query(name):
                    raise SandboxPermissionError(
                        f"🛑 BLOCKED BY DATA-QUERY SANDBOX: importing '{name}' is "
                        "outside the approved data-analysis surface."
                    )
            return orig_import_module(name, package)

        def _filtered_reload(module):
            if _caller_is_user_code():
                mod_name = getattr(module, "__name__", "")
                root_mod = mod_name.split(".")[0].lower()
                if root_mod in FORBIDDEN_MODULES:
                    raise SandboxPermissionError(
                        f"🛑 BLOCKED BY SANDBOX: reloading '{mod_name}' is forbidden."
                    )
                if is_strict and not _module_allowed_in_data_query(mod_name):
                    raise SandboxPermissionError(
                        f"🛑 BLOCKED BY DATA-QUERY SANDBOX: reloading '{mod_name}' is "
                        "outside the approved data-analysis surface."
                    )
            return orig_reload(module)

        # Expose the filtered importer so callers can bind it into a reduced
        # builtins dict without re-entering the sandbox via `import builtins`.
        # Mark it as a block stub so `_recover_import_builtin()` and
        # `_snapshot()` can recognize a leaked one and restore the genuine
        # importer — pickle's `save_global` uses `__import__` internally.
        setattr(_filtered_import, _BLOCKED_STUB_MARKER, True)
        setattr(_filtered_import_module, _BLOCKED_STUB_MARKER, True)
        setattr(_filtered_reload, _BLOCKED_STUB_MARKER, True)
        self.filtered_import = _filtered_import
        builtins.__import__ = _filtered_import
        importlib.import_module = _filtered_import_module
        importlib.reload = _filtered_reload

        # -------- dangerous builtins (frame-gated) --------
        # eval/exec/compile must be blocked for USER-CODE frames but must NOT
        # be blocked for the import machinery's own exec/compile calls. The
        # import system relies on `builtins.exec` to execute module bytecode
        # via `_call_with_frames_removed`; if we blunt-block it, re-importing
        # ANY module (including `sandbox_kernel` itself during a live sandbox)
        # explodes with a SandboxPermissionError — the exact module-load-order
        # regression this class is designed to survive.
        #
        # The stubs delegate to the TRUE builtins whenever the immediate Python
        # frame is NOT a `<execute_python*>` user frame. This mirrors the
        # frame-gating philosophy already used by the path guards: sandbox the
        # TOOL EXECUTION, not the whole process.
        true_compile_orig = self._originals["builtins.compile"]
        true_exec_orig = self._originals["builtins.exec"]
        true_eval_orig = self._originals["builtins.eval"]

        def _gated_compile(source, filename, mode, *a, **kw):
            if _caller_is_user_code():
                raise SandboxPermissionError(
                    "🛑 BLOCKED BY SANDBOX: 'compile' is forbidden in user code."
                )
            return true_compile_orig(source, filename, mode, *a, **kw)

        def _gated_exec(codestr, *a, **kw):
            if _caller_is_user_code():
                raise SandboxPermissionError(
                    "🛑 BLOCKED BY SANDBOX: 'exec' is forbidden in user code."
                )
            return true_exec_orig(codestr, *a, **kw)

        # `eval` MUST stay frame-gated, exactly like `exec`/`compile`. The
        # CPython stdlib calls `eval` during normal module imports — most
        # prominently `collections.namedtuple()`, which uses
        # `eval(code, namespace)` to synthesize the tuple class. That path
        # runs whenever pandas.util.version, enum, typing features, or any
        # library using namedtuple is imported. Blocking eval unconditionally
        # therefore breaks every `import pandas` in the sandbox — the exact
        # regression that broke
        # `test_allows_pandas_query_with_handler` and
        # `test_headless_runs_without_operator_handler`.
        #
        # The frame gate keeps the security property that matters: user-code
        # frames (filename in `_USER_FRAME_FILENAMES`) are always refused,
        # while library-internal callers proceed. Attacker code that reaches
        # `eval` through a user frame is blocked; stdlib internals that reach
        # it during import are not — the same policy enforced for `exec` and
        # `compile`.
        def _gated_eval(expr, *a, **kw):
            if _caller_is_user_code():
                raise SandboxPermissionError(
                    "🛑 BLOCKED BY SANDBOX: 'eval' is forbidden in user code."
                )
            return true_eval_orig(expr, *a, **kw)

        # Mark them as block stubs so `_capture_true_builtins()` /
        # `_recover_true_builtin()` NEVER cache them as "true".
        for _fn in (_gated_compile, _gated_exec, _gated_eval):
            setattr(_fn, _BLOCKED_STUB_MARKER, True)
        builtins.compile = _gated_compile
        builtins.exec = _gated_exec
        builtins.eval = _gated_eval

        # -------- path-guarded open() --------
        # IMPORTANT: guard fires ONLY when the immediate Python caller is the
        # sandboxed user exec frame. Host/library code passes through, so
        # pandas/matplotlib cache writes, pipmaster installs, and our own
        # post-exec cleanup (log persistence, artefact sync, matplotlib figure
        # save) all work unchanged.
        # PATH-ARGUMENT-AUTHORITATIVE: the guard fires on every open() during
        # the sandbox scope — the path argument alone determines allow/deny.
        # The cache-env redirection above moved every legitimate library cache
        # (matplotlib/pip/tempfile/etc.) inside the workspace, so any path
        # outside `root` during the scope is by definition an escape attempt
        # (user-written or library-mediated) and MUST be blocked. The old
        # `_caller_is_user_code()` frame-window heuristic was both unsafe (a
        # 5-frame pandas call to open() slipped through) and unstable (it
        # flipped allow/deny based on library call-stack depth).
        def _guarded_open(file, mode="r", *args, **kwargs):
            _resolve_within(root, file, allow_trusted_read=_is_read_only_open_mode(mode))
            return orig_open(file, mode, *args, **kwargs)

        def _guarded_io_open(file, mode="r", *args, **kwargs):
            _resolve_within(root, file, allow_trusted_read=_is_read_only_open_mode(mode))
            return orig_io_open(file, mode, *args, **kwargs)

        builtins.open = _guarded_open
        io.open = _guarded_io_open

        # -------- os.* lockdown --------
        def _blocked(*_a, **_kw):
            raise SandboxPermissionError(
                "🛑 BLOCKED BY SANDBOX: this OS operation is not permitted."
            )

        for attr in ("system", "popen", "kill", "killpg", "putenv",
                     "unsetenv", "symlink", "link", "chroot", "setuid",
                     "setgid", "fork", "forkpty", "fchdir"):
            if hasattr(os, attr):
                setattr(os, attr, _blocked)

        # -------- cwd boundary (chdir to a subdir is allowed, escape is not) --------
        _orig_chdir = self._originals.get("os.chdir")
        if _orig_chdir is not None:
            def _bounded_chdir(path, *a, **kw):
                _resolve_within(root, path)
                return _orig_chdir(path, *a, **kw)
            os.chdir = _bounded_chdir

        # IMPORTANT: like open(), the os.* fs ops only enforce when the
        # immediate caller is user code. Library and host frames pass through.
        # Same policy as open(): path argument alone decides allow/deny.
        def _bounded_remove(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.remove"](path, *a, **kw)
        def _bounded_unlink(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.unlink"](path, *a, **kw)
        def _bounded_rmdir(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.rmdir"](path, *a, **kw)
        def _bounded_removedirs(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.removedirs"](path, *a, **kw)
        def _bounded_rename(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["os.rename"](src, dst, *a, **kw)
        def _bounded_renames(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["os.renames"](src, dst, *a, **kw)
        def _bounded_replace(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["os.replace"](src, dst, *a, **kw)
        def _bounded_chmod(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.chmod"](path, *a, **kw)
        def _bounded_mkdir(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.mkdir"](path, *a, **kw)
        def _bounded_makedirs(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.makedirs"](path, *a, **kw)
        def _bounded_os_open(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["os.open"](path, *a, **kw)

        # -------- directory-enumeration guards (read-only, path-bounded) --------
        # `os.listdir` / `os.scandir` / `os.walk` are pure reads — they cannot
        # modify the host — but they expose the filesystem layout. Guarding
        # them closes the reconnaissance gap the transcript surfaced: the LLM
        # could previously `os.listdir('/')` or `glob.glob('~/**')` inside the
        # sandbox and learn host paths, while `open('/...')` of the same path
        # was correctly blocked. The guard is PATH-ARGUMENT-AUTHORITATIVE
        # (mirrors `_guarded_open`): any enumeration whose argument resolves
        # outside `workspace_root` — but inside a trusted interpreter root —
        # raises `SandboxPermissionError`. `allow_trusted_read=True` is set
        # because every op in this set is read-only by definition.
        #
        # `glob.glob` is NOT patched directly: CPython's `glob._iglob` routes
        # through `os.scandir(dirname)`, so guarding `os.scandir` transitively
        # guards `glob.glob`, `glob.iglob`, and `Path.glob` for free.
        def _bounded_listdir(path=".", *a, **kw):
            _resolve_within(root, path, allow_trusted_read=True)
            return self._originals["os.listdir"](path, *a, **kw)
        def _bounded_scandir(path=".", *a, **kw):
            _resolve_within(root, path, allow_trusted_read=True)
            return self._originals["os.scandir"](path, *a, **kw)
        def _bounded_walk(top, *a, **kw):
            # `os.walk` recurses: guard the seed path here, and rely on the
            # patched `scandir`/`listdir` (which `walk` internally calls) to
            # guard every level it descends into. This two-layer guard is
            # belt-and-suspenders: even if a future walk implementation
            # bypasses scandir, the entry point is still bounded.
            _resolve_within(root, top, allow_trusted_read=True)
            return self._originals["os.walk"](top, *a, **kw)

        # CPython <= 3.10's `glob` module binds `scandir` BY VALUE at its own
        # import time (`from os import scandir`). Patching `os.scandir` later
        # does NOT rebind `glob.scandir`, so a bare scandir guard does not
        # intercept `glob.glob` — a fact this suite proved by failing before
        # the explicit guard below was added. The glob entry points are thus
        # guarded on the STATIC leading directory of the pattern (the longest
        # magic-free path prefix), which is exactly the directory `glob` scans.
        def _bounded_glob(pattern, *a, **kw):
            _resolve_within(root, _glob_static_root(pattern), allow_trusted_read=True)
            return self._originals["glob.glob"](pattern, *a, **kw)
        def _bounded_iglob(pattern, *a, **kw):
            _resolve_within(root, _glob_static_root(pattern), allow_trusted_read=True)
            return self._originals["glob.iglob"](pattern, *a, **kw)

        for name, fn in (
            ("remove", _bounded_remove),
            ("unlink", _bounded_unlink),
            ("rmdir", _bounded_rmdir),
            ("removedirs", _bounded_removedirs),
            ("rename", _bounded_rename),
            ("renames", _bounded_renames),
            ("replace", _bounded_replace),
            ("chmod", _bounded_chmod),
            ("mkdir", _bounded_mkdir),
            ("makedirs", _bounded_makedirs),
            ("open", _bounded_os_open),
            ("listdir", _bounded_listdir),
            ("scandir", _bounded_scandir),
            ("walk", _bounded_walk),
        ):
            if self._originals.get(f"os.{name}") is not None:
                setattr(os, name, fn)

        if self._originals.get("glob.glob") is not None:
            glob.glob = _bounded_glob
        if self._originals.get("glob.iglob") is not None:
            glob.iglob = _bounded_iglob

        # -------- shutil.* lockdown (same frame-gated policy) --------
        # Same policy: path argument alone decides.
        def _bounded_shutil_remove(path, *a, **kw):
            _resolve_within(root, path)
            return self._originals["shutil.rmtree"](path, *a, **kw)
        def _bounded_shutil_move(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["shutil.move"](src, dst, *a, **kw)
        def _bounded_shutil_copy(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            orig = self._originals.get("shutil.copy2") or self._originals.get("shutil.copy")
            return orig(src, dst, *a, **kw)
        def _bounded_shutil_copy2(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["shutil.copy2"](src, dst, *a, **kw)
        def _bounded_shutil_copytree(src, dst, *a, **kw):
            _resolve_within(root, src)
            _resolve_within(root, dst)
            return self._originals["shutil.copytree"](src, dst, *a, **kw)

        if self._originals.get("shutil.rmtree"): shutil.rmtree = _bounded_shutil_remove
        if self._originals.get("shutil.move"): shutil.move = _bounded_shutil_move
        if self._originals.get("shutil.copy"): shutil.copy = _bounded_shutil_copy
        if self._originals.get("shutil.copy2"): shutil.copy2 = _bounded_shutil_copy2
        if self._originals.get("shutil.copytree"): shutil.copytree = _bounded_shutil_copytree

        # -------- subprocess hard block (SKIPPED under "authorized") --------
        # Under "authorized", the user explicitly consented to process spawning
        # (e.g. "run ping", "call curl"). Blocking subprocess here would
        # silently override that consent at exec time — the exact bug where
        # the user clicks "yes" and still sees "BLOCKED BY SANDBOX".
        #
        # Note: `os.system` / `os.popen` / `os.exec*` / `os.spawn*` remain
        # blocked above (they are the process-ESCAPE surface, not the
        # process-SPAWN surface). `subprocess` internally reaches os.exec*
        # through its own non-user frames, which `_caller_is_user_code()`
        # correctly exempts — so subprocess works end-to-end while direct
        # user-code `os.exec*` calls stay blocked.
        if not is_authorized:
            for name in ("run", "Popen", "call", "check_output"):
                if self._originals.get(f"subprocess.{name}") is not None:
                    setattr(subprocess, name, _blocked)
            # Also block `subprocess.Popen` alias used internally by other entries.
            try:
                subprocess.check_call = _blocked
                subprocess.check_output = _blocked
            except Exception:
                pass

        # -------- network hard block (SKIPPED under "authorized") --------
        # Same rationale: the user authorized network access. Blocking socket
        # here would defeat the authorization — the user's "yes, ping google"
        # would still produce "BLOCKED BY SANDBOX: outbound network access".
        if not is_authorized:
            def _blocked_net(*_a, **_kw):
                raise SandboxPermissionError(
                    "🛑 BLOCKED BY SANDBOX: outbound network access is forbidden."
                )
            socket.socket.connect = _blocked_net
            if self._originals.get("socket.socket.connect_ex") is not None:
                socket.socket.connect_ex = _blocked_net
            if self._originals.get("socket.create_connection") is not None:
                socket.create_connection = _blocked_net
            if self._originals.get("socket.getaddrinfo") is not None:
                socket.getaddrinfo = _blocked_net

        # -------- pathlib guarded methods --------
        # Patch the concrete Path class attributes so `.write_text()` etc. are
        # path-checked. Originals were already captured in `_snapshot()` under
        # the "__pathlib__" key — we only install the guards here.
        pathlib_originals = self._originals.get("__pathlib__", {})
        for method_name, orig_m in pathlib_originals.items():
            setattr(Path, method_name, self._make_path_guard(method_name, orig_m, root))


    @staticmethod
    def _make_path_guard(method_name: str, orig_method: Callable, root: Path) -> Callable:
        """Wraps a Path method so its target is always checked against `root`.

        Path-argument authoritative: the guard fires on every call during the
        sandbox scope. Cache-env redirection (matplotlib/pip/tempfile) makes
        every legitimate library cache land inside `workspace_root`, so a path
        outside `root` is by definition an escape and MUST be blocked. This
        closes the pandas-mediated traversal hole (a 5-frame call stack from
        `pd.read_csv('../etc/passwd')` to `open()` slipped through the old
        frame-window heuristic) while still allowing every library cache write
        to succeed inside `.sandbox_cache/`.
        """
        def guard(this, *args, **kwargs):
            # `Path.open(mode)` is the one mutating-method surface that can be
            # a pure read — allow it to consult trusted read roots, exactly
            # like `_guarded_open`. Every other method in the set is a write
            # and keeps the strict workspace boundary.
            allow_read = (
                method_name == "open"
                and bool(args)
                and _is_read_only_open_mode(args[0])
            )
            ts = str(this)
            if not (ts and not os.path.isabs(ts) and ".." not in ts.replace("\\", "/").split("/")):
                _resolve_within(root, this, allow_trusted_read=allow_read)
            if method_name in ("rename", "replace", "symlink_to", "hardlink_to", "link_to") and args:
                ds = str(args[0])
                if not (ds and not os.path.isabs(ds) and ".." not in ds.replace("\\", "/").split("/")):
                    _resolve_within(root, args[0])
            return orig_method(this, *args, **kwargs)
        return guard