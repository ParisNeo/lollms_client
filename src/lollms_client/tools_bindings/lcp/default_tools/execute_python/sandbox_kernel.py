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


def _capture_true_builtins() -> Tuple[Any, Any, Any]:
    """Capture the real builtins, preferring frozen sentinels if `builtins`
    has already been patched by an earlier sandbox in this process.

    CRITICAL: `_is_block_stub` guards against the module-load-order bug where
    this module is imported for the FIRST time inside an active
    `SandboxContext` — `builtins.exec`/`compile`/`eval` are then already our
    own stub, and without validation we would cache the stub as the "true"
    builtin, permanently breaking every tool that needs exec/compile."""
    tc = getattr(builtins, "__lollms_true_compile__", None)
    if _is_block_stub(tc):
        tc = None
    te = getattr(builtins, "__lollms_true_exec__", None)
    if _is_block_stub(te):
        te = None
    tv = getattr(builtins, "__lollms_true_eval__", None)
    if _is_block_stub(tv):
        tv = None
    if tc is None:
        tc = builtins.__dict__.get("compile")
        if _is_block_stub(tc):
            tc = _recover_true_builtin("compile")
    if te is None:
        te = builtins.__dict__.get("exec")
        if _is_block_stub(te):
            te = _recover_true_builtin("exec")
    if tv is None:
        tv = builtins.__dict__.get("eval")
        if _is_block_stub(tv):
            tv = _recover_true_builtin("eval")
    return tc, te, tv


TRUE_COMPILE, TRUE_EXEC, TRUE_EVAL = _capture_true_builtins()
# Publish frozen markers so ANY future import of this module (or a `_snapshot()`
# in a nested SandboxContext, or a restore after a leak) can recover the true
# builtins even if `builtins.exec`/`compile`/`eval` are currently stubs.
try:
    if not hasattr(builtins, "__lollms_true_compile__"):
        builtins.__lollms_true_compile__ = TRUE_COMPILE  # type: ignore[attr-defined]
    if not hasattr(builtins, "__lollms_true_exec__"):
        builtins.__lollms_true_exec__ = TRUE_EXEC  # type: ignore[attr-defined]
    if not hasattr(builtins, "__lollms_true_eval__"):
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


def validate_source(source: str, profile: str) -> Optional[SandboxViolation]:
    """Parses `source` and returns a `SandboxViolation` if it fails the profile."""
    if profile not in ("data_query", "standard"):
        raise ValueError(f"Unknown sandbox profile: {profile!r}")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Let the runtime raise the proper SyntaxError with traceback.
        return None
    if profile == "data_query":
        return _validate_data_query(tree)
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


def _sandbox_child_worker(conn, payload: Dict[str, Any]) -> None:
    """Runs in the child process ONLY. Executes user code under SandboxContext.

    Sends exactly one envelope through `conn` then closes it. Any internal
    failure is converted to a structured error envelope so the parent never
    sees a bare `EOFError` from a silent child death.
    """
    import io as _io
    import traceback
    import json as _json

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
                compiled = TRUE_COMPILE(source, script_label, "exec")
                TRUE_EXEC(compiled, local_vars)
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


def run_in_sandboxed_child(
    source: str,
    workspace_root: Path,
    profile: str,
    script_label: str = "<sandboxed>",
    argv: Optional[List[Any]] = None,
    seed_locals: Optional[Dict[str, Any]] = None,
    persist_df_name: Optional[str] = None,
    hard_timeout: float = _CHILD_HARD_TIMEOUT_DEFAULT,
) -> Dict[str, Any]:
    """Executes `source` in a sandboxed child process. Returns the envelope.

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
    import multiprocessing as _mp
    import time as _time

    # On Windows the only viable start method is 'spawn' — we always use it,
    # even on POSIX, so behavior is identical across platforms.
    ctx = _mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    payload = {
        "source": source,
        "workspace_root": str(workspace_root),
        "profile": profile,
        "script_label": script_label,
        "argv": argv or [script_label],
        "seed_locals": seed_locals or {},
        "persist_df_name": persist_df_name,
    }

    proc = ctx.Process(
        target=_sandbox_child_worker,
        args=(child_conn, payload),
        daemon=True,
    )
    proc.start()
    child_conn.close()  # parent must close its child-end handle, or poll() blocks

    envelope: Optional[Dict[str, Any]] = None
    deadline = _time.monotonic() + hard_timeout
    try:
        while True:
            remaining = max(0.05, deadline - _time.monotonic())
            if parent_conn.poll(remaining):
                try:
                    envelope = parent_conn.recv()
                except EOFError:
                    envelope = None
                break
            if not proc.is_alive():
                # Child died before sending — try one final non-blocking read.
                if parent_conn.poll(0):
                    try:
                        envelope = parent_conn.recv()
                    except EOFError:
                        envelope = None
                break
            if _time.monotonic() >= deadline:
                proc.terminate()
                proc.join(timeout=2.0)
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "traceback": (
                        f"Sandboxed child exceeded {hard_timeout:.0f}s hard timeout "
                        "and was terminated."
                    ),
                    "error_type": "TimeoutError",
                    "sandbox_violation": False,
                    "figures": [],
                    "df_pickle": None,
                    "scalars": {},
                }
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        try:
            proc.join(timeout=2.0)
        except Exception:
            pass
        if proc.is_alive():
            proc.terminate()
            try:
                proc.join(timeout=2.0)
            except Exception:
                pass

    if envelope is None:
        return {
            "success": False,
            "stdout": "",
            "stderr": "",
            "traceback": (
                "Sandboxed child process exited without returning a payload. "
                f"Exit code: {proc.exitcode}."
            ),
            "error_type": "ChildCrash",
            "sandbox_violation": False,
            "figures": [],
            "df_pickle": None,
            "scalars": {},
        }
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


def _resolve_within(root: Path, target: Any) -> Path:
    """Fast path-guard: returns the resolved target, raises if outside `root`.

    PERFORMANCE: We avoid the expensive `Path.resolve()` on every call.
    `Path.resolve()` performs symlink expansion + multiple `os.stat` syscalls.
    For relative paths that contain no `..` and are not absolute, they are
    guaranteed to be under the CWD already — the caller (sandbox) guarantees
    CWD == root at install time. We only fall back to full resolution for:
      - absolute paths (must be sub-path of root)
      - paths containing `..` (traversal check required)
      - Windows drive-qualified paths
    """
    try:
        if isinstance(target, int):
            # Already an open file descriptor — no path to check, pass through.
            # FDs are only obtainable via os.open() which is already path-guarded.
            return Path.cwd()  # arbitrary in-root placeholder; caller ignores
        # PERF: avoid os.fspath() syscalls for the common `str` case.
        if isinstance(target, str):
            target_str = target
        else:
            target_str = os.fspath(target)
            if not isinstance(target_str, str):
                target_str = str(target_str)
        # Fast path: relative & no traversal segment → inherently inside root.
        if (
            target_str
            and not os.path.isabs(target_str)
            and ".." not in target_str.replace("\\", "/").split("/")
        ):
            return Path(root) / target_str
        # Slow path: absolute or contains traversal → full resolution.
        p = Path(target_str)
        if not p.is_absolute():
            p = Path.cwd() / p
        resolved = p.resolve(strict=False)
        resolved.relative_to(root)
        return resolved
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
        self._originals = {
            "builtins.__import__": builtins.__import__,
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
            except Exception:
                pass
        self._originals = {}
        self.filtered_import = None
        self._restore_cache_env()

    # ---------- cache-env redirection ----------
    def _redirect_cache_env(self) -> None:
        """Point cache/temp-dir env vars at `<workspace>/.sandbox_cache/*`.

        PERF: uses raw `os.makedirs` (bypassing the patched `Path.mkdir`) and
        memoizes the directory set in `_SANDBOX_CACHE_READY` so the 13 mkdirs
        run at most once per (process, workspace). Previously every tool call
        paid for 13 mkdir syscalls + 13 guard resolves — the source of the
        600s regression.
        """
        import tempfile as _tf
        root = self.workspace_root.resolve()
        sandbox_cache = root / ".sandbox_cache"

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
        self.filtered_import = _filtered_import
        builtins.__import__ = _filtered_import
        importlib.import_module = _filtered_import_module
        importlib.reload = _filtered_reload

        # -------- dangerous builtins --------
        # Each of eval/exec/compile gets its OWN marked stub so that a later
        # `_capture_true_builtins()` (nested/re-import scenario) can identify
        # and reject it rather than caching it as the true builtin.
        builtins.eval = _make_block_stub()
        builtins.exec = _make_block_stub()
        builtins.compile = _make_block_stub()

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
            _resolve_within(root, file)
            return orig_open(file, mode, *args, **kwargs)

        def _guarded_io_open(file, mode="r", *args, **kwargs):
            _resolve_within(root, file)
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
        ):
            if self._originals.get(f"os.{name}") is not None:
                setattr(os, name, fn)

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

        # -------- subprocess hard block --------
        for name in ("run", "Popen", "call", "check_output"):
            if self._originals.get(f"subprocess.{name}") is not None:
                setattr(subprocess, name, _blocked)
        # Also block `subprocess.Popen` alias used internally by other entries.
        try:
            subprocess.check_call = _blocked
            subprocess.check_output = _blocked
        except Exception:
            pass

        # -------- network hard block --------
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
            # PERF: sub-millisecond "relative & no .." check is the ONLY thing
            # done for well-behaved paths. `resolve(strict=False).relative_to`
            # is only paid when the argument is absolute OR contains '..'.
            try:
                ts = str(this)
                if ts and not os.path.isabs(ts) and ".." not in ts.replace("\\", "/").split("/"):
                    pass  # Fast path: relative sub-path (sandbox CWD == root).
                else:
                    Path(this).resolve(strict=False).relative_to(root)
            except (ValueError, OSError) as exc:
                raise SandboxPermissionError(
                    f"🛑 BLOCKED BY SANDBOX: Path.{method_name}() target "
                    f"'{this}' resolves outside the workspace."
                ) from exc
            if method_name in ("rename", "replace", "symlink_to", "hardlink_to", "link_to") and args:
                try:
                    ds = str(args[0])
                    if not (ds and not os.path.isabs(ds) and ".." not in ds.replace("\\", "/").split("/")):
                        Path(args[0]).resolve(strict=False).relative_to(root)
                except (ValueError, OSError) as exc:
                    raise SandboxPermissionError(
                        f"🛑 BLOCKED BY SANDBOX: Path.{method_name}() destination "
                        f"'{args[0]}' resolves outside the workspace."
                    ) from exc
            return orig_method(this, *args, **kwargs)
        return guard