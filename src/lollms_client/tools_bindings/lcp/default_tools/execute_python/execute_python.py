"""
execute_python.py
LCP toolset for sandboxed Python execution. Three tools:
  - tool_create_and_execute_python_file: PREFERRED for substantial programs. Writes
                                the given 'code' to a workspace .py file and executes
                                it in ONE call (write + run, no round-trip waste).
  - tool_execute_python_file:   executes an EXISTING workspace .py file with optional
                                argv-style arguments (sys.argv / argparse compatible).
                                Read-only: never saves, creates, or overwrites files.
  - tool_execute_python_code:   executes INLINE Python source (the 'code' string).
                                Nothing is saved to disk by the tool itself.
                                STRICTLY for short, punctual, throwaway snippets.
Interception of matplotlib figures and stdout/stderr capture.

When captured stdout exceeds the inline preview window, the FULL output is
persisted to a timestamped .log file in the workspace and the tool result
embeds a head/tail windowed preview plus an explicit pointer telling the LLM
which inspection tools to use to read the omitted middle.

HUMAN-IN-THE-LOOP AUTHORIZATION:
In safe mode, execution prompts the user with a code preview and authorization choices:
[y]es (run once), [a]lways (auto-approve the CATEGORY of the shown code), [n]o (reject with feedback), [v]iew full code.

"always" is CATEGORY-SCOPED, not session-wide: it authorizes only the risk
category of the code displayed at prompt time (e.g. "network" for socket /
requests; "process" for subprocess). A subsequent snippet touching a DIFFERENT
category will prompt again. This prevents blanket session-wide bypass while
still honoring the operator's intent ("I approve network access, not process
spawning"). Categories: network / process / dynamic / system.

In non-interactive environments without an interactive handler, refusal is cleanly assumed without blocking.
"""

import ast
import os
import re
import sys
import io
import uuid
import base64
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ascii_colors import ASCIIColors

# Shared defense-in-depth sandbox kernel (AST + runtime layers).
#
# Import resolution strategy (robust across all load contexts):
#   1. Relative import  — works when this module is loaded as part of the package
#                          (the normal LCP toolset load path).
#   2. Absolute import  — works when the toolset is loaded by file path via
#                          importlib without a registered parent package, but
#                          `lollms_client` is installed (the runtime case that
#                          previously failed with `No module named 'src'`).
#   3. `src.lollms_client.*` — dev-checkout fallback for repos where the package
#                          is NOT pip-installed and `src/` is on sys.path.
#
# The previous fallback ONLY tried `src.lollms_client.*`, which is not a valid
# module path in any installed environment — producing the observed
# `No module named 'src'` lazy-init failure.
def _import_sandbox_kernel():
    """Resolves `sandbox_kernel` across relative / installed / dev-checkout contexts."""
    errors = []
    # Tier 1: relative (canonical package load).
    try:
        from . import sandbox_kernel as _sk  # type: ignore
        return _sk
    except ImportError as exc:
        errors.append(f"relative: {exc}")
    # Tier 2: canonical installed path.
    try:
        from lollms_client.tools_bindings.lcp.default_tools.execute_python import (  # type: ignore
            sandbox_kernel as _sk,
        )
        return _sk
    except ImportError as exc:
        errors.append(f"lollms_client: {exc}")
    # Tier 3: dev-checkout fallback (repo's `src/` on sys.path).
    try:
        from src.lollms_client.tools_bindings.lcp.default_tools.execute_python import (  # type: ignore
            sandbox_kernel as _sk,
        )
        return _sk
    except ImportError as exc:
        errors.append(f"src.lollms_client: {exc}")
    raise ImportError(
        "Could not import sandbox_kernel from any known location. "
        f"Attempts: {'; '.join(errors)}"
    )


_sandbox_kernel = _import_sandbox_kernel()
_sandbox_validate_source = _sandbox_kernel.validate_source
_SandboxViolation = _sandbox_kernel.SandboxViolation
_SandboxPermissionError = _sandbox_kernel.SandboxPermissionError
_run_in_sandboxed_child = _sandbox_kernel.run_in_sandboxed_child

TOOL_LIBRARY_NAME = "Execute Python"
TOOL_LIBRARY_DESC = (
    "Executes sandboxed Python code. DEFAULT for substantial programs: "
    "tool_create_and_execute_python_file(code=..., file_name=..., args=[...]) — "
    "writes the .py file AND runs it in ONE call (no wasted round-trip). "
    "Use tool_execute_python_file to RE-RUN an already-created workspace .py file. "
    "tool_execute_python_code is strictly reserved for short, punctual inline snippets "
    "(one-liners, quick checks). Long outputs are persisted to a .log file with a "
    "head/tail preview returned inline. Returns stdout, stderr, and generated plots."
)
TOOL_LIBRARY_ICON = "🐍"

AUTONOMY_LEVEL: str = "safe"
# Category-scoped pre-approval set (replaces the old binary `_AUTO_APPROVE_PYTHON`).
# A category (e.g. "network") in this set means: for the rest of the session,
# skip the prompt IF AND ONLY IF the snippet's risk categories are ALL in this
# set. Different categories still prompt. See `_approve_categories()`.
_APPROVED_CATEGORIES: frozenset = frozenset()
# Kept for backward compatibility with external code that reads this name;
# means "at least one category has been pre-approved".
_AUTO_APPROVE_PYTHON: bool = False
_CONFIRM_HANDLER: Optional[Any] = None

_PREVIEW_WINDOW_CHARS = 8000
_PREVIEW_HALF_CHARS = _PREVIEW_WINDOW_CHARS // 2
_STRIP_MARKER = (
    "\n... [stripped for brevity (use read file tools to inspect more)] ...\n"
)


def set_confirm_handler(handler: Optional[Any]) -> None:
    """Sets a custom confirmation handler across all loaded and persistent LCP module instances."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler
    for mod_name, mod in list(sys.modules.items()):
        if "execute_python" in mod_name and mod is not sys.modules.get(__name__):
            try:
                mod._CONFIRM_HANDLER = handler
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# Category-scoped "always allow" registry.
#
# CONTRACT: "always" authorizes ONLY the risk CATEGORY of the code that was
# displayed at prompt time — not the whole session. If the operator approves
# `socket` usage (category "network"), a subsequent `subprocess` snippet
# (category "process") MUST prompt again, because it is a different category.
#
# Categories are a partition of the risky surface exposed by
# `_detect_risky_operations`. When a snippet touches several categories, the
# prompt reports ALL of them (comma-joined) and "always" grants all of them
# together — the operator saw them together when they said "always".
#
# The registry is process-wide (mirrored on `sys`) and propagated to every
# loaded `execute_python` module instance, symmetric with `set_confirm_handler`.
# ──────────────────────────────────────────────────────────────────────────────
_SYS_APPROVED_CATEGORIES_ATTR = "__lollms_execute_python_approved_categories__"


# Category taxonomy — maps a category name to the set of `_detect_risky_operations`
# signal SUBSTRINGS that belong to it. The detection function emits reasons
# like "Attempting to spawn external processes via 'subprocess'" — those are
# what we bucket. Keep the substrings narrow and unambiguous.
_RISK_CATEGORY_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "network": (
        "socket", "ssl", "requests", "urllib", "urllib3", "httpx", "aiohttp",
        "ftplib", "smtplib", "telnetlib", "paramiko", "fabric",
        "Low-level system access",  # generic risky_system_module fallback
    ),
    "process": (
        "spawn external processes", "subprocess", "multiprocessing", "pty",
        "commands", "os.system", "os.popen", "os.spawn", "os.exec",
        "os.kill", "system/process command",
    ),
    "dynamic": (
        "importlib", "runpy", "imp", "pkgutil", "zipimport",
        "Dynamic import", "Dynamic access to risky attribute",
    ),
    "system": (
        "psutil", "webbrowser", "ctypes", "winreg",
    ),
    # Filesystem access OUTSIDE the workspace (reads of site-packages, listing
    # the venv, probing an interpreter install, etc.). This category exists so
    # the operator can say "always allow filesystem reads" and — crucially —
    # have the runtime path guard actually honour it instead of hard-blocking
    # at scandir/open with no prompt. The static and runtime kernels both read
    # the approved-roots registry (`sandbox_kernel.set_extra_read_roots`).
    #
    # A snippet that triggers this category is one the AST layer flagged as
    # touching the host filesystem beyond `workspace_root` — e.g.
    # `os.scandir(r'C:\\...\\site-packages')` from user code, or a literal
    # absolute path outside the workspace appearing in the source.
    "filesystem": (
        "host filesystem", "outside the workspace", "filesystem escape",
        "path outside", "absolute host path",
    ),
    # Dedicated DELETION category — triggered by the AST detector below
    # (`_detect_deletion_ops`) whenever the snippet performs a destructive
    # filesystem operation (os.remove / os.unlink / os.rmdir / shutil.rmtree /
    # Path.unlink / Path.rmdir). The category is intentionally NEVER
    # auto-approved by a previous "always": the operator's intent — "ask me
    # when you would delete files" — must be honoured for EVERY deletion
    # batch, even inside the workspace. Only the current snippet is
    # authorized by a "yes" answer.
    "delete": (
        "deletes files", "deletes directories", "recursive delete",
    ),
}

# AST-detectable destructive filesystem operations. The detector walks every
# call in the snippet and returns the sorted set of operations it found.
# Convenient, because it lets the prompt say *which* operations will run
# ("os.remove, shutil.rmtree") instead of just "destructive work".
_DELETION_OS_ATTRS: frozenset[str] = frozenset({
    "remove", "unlink", "rmdir", "removedirs",
})
_DELETION_SHUTIL_ATTRS: frozenset[str] = frozenset({"rmtree"})
_DELETION_PATH_METHODS: frozenset[str] = frozenset({"unlink", "rmdir"})


def _detect_deletion_ops(source: str) -> List[str]:
    """Returns the sorted list of destructive filesystem calls in `source`.

    Detects the AST surface only — this is a PROMPTING heuristic, not the
    security boundary (the runtime `_bounded_*` guards remain the boundary
    for path bounds). It exists so that:
      * `safe` mode can force a prompt on snippets that delete files, even
        when the snippet touches no other risk category;
      * the prompt can name the exact operations the operator is about to
        authorize ("os.remove, shutil.rmtree"), so consent is informed.

    Never raises; a syntax error yields an empty list (nothing to prompt on
    for a source that cannot compile anyway).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = ""
        if isinstance(func, ast.Attribute):
            attr = func.attr
            target = func.value
            # os.<deletion>
            if isinstance(target, ast.Name) and target.id == "os" and attr in _DELETION_OS_ATTRS:
                found.add(f"os.{attr}")
            # shutil.<deletion>
            elif isinstance(target, ast.Name) and target.id == "shutil" and attr in _DELETION_SHUTIL_ATTRS:
                found.add(f"shutil.{attr}")
            # <anything>.<unlink|rmdir> — matches Path(...).unlink(),
            # open(...).rmdir (impossible but harmless), and the canonical
            # Path form. False positives simply mean an extra confirmation
            # prompt, which is the correct safe default.
            elif attr in _DELETION_PATH_METHODS:
                # Exclude modules we explicitly know are not fs deletion:
                # `tempfile.unlink` (not a real API), `zipfile.rmdir` (not
                # real) — these are non-issues; leaving the broad match is
                # strictly safer than missing a real `Path.unlink()`.
                found.add(f".{attr}()")
        elif isinstance(func, ast.Name):
            func_name = func.id
            # Bare `remove(...)` when `from os import remove` was used.
            if func_name in _DELETION_OS_ATTRS:
                found.add(f"{func_name}()")
    return sorted(found)


def _classify_risk_categories(risky_reason: Optional[str]) -> frozenset:
    """Maps a `_detect_risky_operations` reason string to a set of categories.

    Unknown reasons (defensive: any future detection signal not yet bucketed)
    map to a synthetic category equal to the reason itself, so "always" on such
    a snippet remains scoped to that specific reason and does not leak across
    to wholly different operations.
    """
    if not risky_reason:
        return frozenset()
    lowered = risky_reason.lower()
    matched = set()
    for category, signals in _RISK_CATEGORY_SIGNALS.items():
        if any(sig.lower() in lowered for sig in signals):
            matched.add(category)
    return frozenset(matched) if matched else frozenset({lowered.strip()})


def _get_approved_categories() -> frozenset:
    """Returns the process-wide set of categories the operator pre-approved.

    Reads from `sys` first (survives module reloads and multiple module
    identities), falling back to the local module-global for backward compat.
    """
    sys_set = getattr(sys, _SYS_APPROVED_CATEGORIES_ATTR, None)
    if isinstance(sys_set, frozenset):
        return sys_set
    if isinstance(sys_set, (set, list, tuple)):
        return frozenset(sys_set)
    return frozenset(_APPROVED_CATEGORIES)


def _approve_categories(categories: frozenset) -> None:
    """Adds `categories` to the process-wide pre-approval set.

    Called from the "[a]lways" branch of every authorization prompt, with the
    categories of the code the operator actually saw. Mutates the process-wide
    set exactly once and mirrors it to every loaded module instance.

    The `delete` category is EXCLUDED by invariant: "always" on a snippet that
    deletes files must NOT license future deletions — the operator's intent is
    "confirm every destructive action". Deletions are always re-prompted.
    """
    global _APPROVED_CATEGORIES
    persistent = set(categories) - {"delete"}
    if not persistent:
        return
    new_set = _get_approved_categories() | persistent
    new_frozen = frozenset(new_set)
    _APPROVED_CATEGORIES = new_frozen
    try:
        setattr(sys, _SYS_APPROVED_CATEGORIES_ATTR, new_frozen)
    except Exception:
        pass
    for mod_name, mod in list(sys.modules.items()):
        if "execute_python" in mod_name and mod is not sys.modules.get(__name__):
            try:
                mod._APPROVED_CATEGORIES = new_frozen
            except Exception:
                pass


def _reset_approved_categories() -> None:
    """Clears the pre-approval set. Exposed for operator/host control."""
    global _APPROVED_CATEGORIES
    _APPROVED_CATEGORIES = frozenset()
    try:
        setattr(sys, _SYS_APPROVED_CATEGORIES_ATTR, frozenset())
    except Exception:
        pass
    for mod_name, mod in list(sys.modules.items()):
        if "execute_python" in mod_name and mod is not sys.modules.get(__name__):
            try:
                mod._APPROVED_CATEGORIES = frozenset()
            except Exception:
                pass
    # Symmetric with `_approve_categories`: revoke filesystem roots too.
    _clear_approved_path_roots()


def _sync_approved_path_roots(workspace_root: Optional[Path]) -> None:
    """Publishes the operator-consented extra read/write roots to the kernel.

    The runtime path guard (`sandbox_kernel._resolve_within`) raises when a
    path resolves outside the workspace — BY DESIGN, as a first line of
    defense. But once the operator presses "always" on a snippet whose category
    is `filesystem`, that same guard must stop blocking the operator's
    consented surface. Otherwise the user sees exactly the reported bug:
    "I said always, but it still says BLOCKED BY SANDBOX."

    We compute a conservative set of trusted roots from the source directory
    of this very module and from the interpreter installation. Those are the
    realistic escapes an LLM probes ("is torch installed? what's in
    site-packages?"). Wider roots (e.g. the home directory, `/`) are NEVER
    auto-added — the operator scoped the consent to "filesystem", not
    "everything". This matches the principle: consented category ⇒ unblocked
    for THAT category's realistic surface, not for arbitrary paths.
    """
    roots = set()
    # Interpreter install (site-packages, venv root, stdlib). These are the
    # roots that appear in the reported log (`.../AITIANA_studio/.venv/...`).
    try:
        import sys as _sys_local
        for attr in ("prefix", "base_prefix", "exec_prefix"):
            p = getattr(_sys_local, attr, None)
            if p:
                try:
                    roots.add(str(Path(p).resolve()))
                except Exception:
                    pass
        # The interpreter's Lib/site-packages sits under prefix, but on venvs
        # it's sometimes a separate sysconfig path; resolve it explicitly.
        try:
            import sysconfig as _sysconfig_local
            for key in ("purelib", "platlib", "stdlib", "platstdlib"):
                try:
                    p = _sysconfig_local.get_path(key)
                    if p:
                        roots.add(str(Path(p).resolve()))
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    # Directory containing this tool's module. Legitimate scope: "the agent
    # may inspect its own package tree".
    try:
        roots.add(str(Path(__file__).resolve().parent))
    except Exception:
        pass
    # Workspace root is always allowed — the kernel already enforces it, but
    # including it in the same contract keeps the two layers in sync.
    if workspace_root is not None:
        try:
            roots.add(str(Path(workspace_root).resolve()))
        except Exception:
            pass
    publish_extra_read_roots = getattr(_sandbox_kernel, "set_extra_read_roots", None)
    if callable(publish_extra_read_roots):
        try:
            publish_extra_read_roots(frozenset(roots))
        except Exception:
            pass
    publish_extra_write_roots = getattr(_sandbox_kernel, "set_extra_write_roots", None)
    if callable(publish_extra_write_roots):
        try:
            publish_extra_write_roots(frozenset(roots))
        except Exception:
            pass


def _clear_approved_path_roots() -> None:
    """Revokes the extra roots (used when the operator resets the session)."""
    for name in ("set_extra_read_roots", "set_extra_write_roots"):
        fn = getattr(_sandbox_kernel, name, None)
        if callable(fn):
            try:
                fn(frozenset())
            except Exception:
                pass


def _categories_for(source: str, workspace_root: Optional[Path] = None) -> frozenset:
    """Convenience: runs risk detection + deletion detection, then classifies."""
    categories = set(_classify_risk_categories(
        _detect_risky_operations(source, workspace_root=workspace_root)
    ))
    # Deletion category is added independently of the risky-operation reason
    # so that a snippet whose ONLY risky surface is deletion still prompts in
    # safe mode. The prompt is the security property here: the user's intent
    # is "ask before you delete anything", and this is where we honour it.
    if _detect_deletion_ops(source):
        categories.add("delete")
    return frozenset(categories)


def _is_category_approved(risky_reason: Optional[str], source: Optional[str] = None) -> bool:
    """True iff *every* category of `risky_reason` is already pre-approved.

    Only when ALL present categories are approved may the prompt be skipped.
    A snippet spanning `network` + `process` will prompt even if `network` is
    approved, because `process` has not been. This is the strictly-safe rule:
    the operator is asked about every category they have not consented to.

    When `source` is provided, the snippet is ALSO scanned for deletions. The
    `delete` category is deliberately EXCLUDED from the persistent approval
    set: the operator's intent is "ask me when you would delete files", and
    "always" on a previous snippet must NOT license future deletions. Only a
    "yes" on the CURRENT prompt authorizes the current deletion batch. This is
    the correct contract — every destructive action is confirmed.
    """
    categories = set(_classify_risk_categories(risky_reason))
    if source is not None and _detect_deletion_ops(source):
        categories.add("delete")
    if not categories:
        return True  # benign — no category, no prompt
    # ── Deletion always forces the prompt. ──
    # Not part of `_get_approved_categories()` de facto (it is never added by
    # `_approve_categories` — see the check below), but stated explicitly so
    # the invariant is self-documenting at the dispatch site.
    if "delete" in categories and "delete" not in _get_approved_categories():
        return False
    approved = _get_approved_categories() - {"delete"}
    return (categories - {"delete"}).issubset(approved)


# ──────────────────────────────────────────────────────────────────────────────
# Dependency inventory + trust model for LCP tool initialization.
#
# The LCP `init_tools_library()` hook is the *trusted* place where a toolset
# declares and ensures its dependencies. For first-party (builtin) tools, we
# trust the tool code itself — it ships with the package and its only job here
# is to make the tool usable. The security boundary is the *code the tool
# executes from its parameters*, not the tool's own initialization.
#
# Therefore: pip installation is allowed at init *when the toolset is trusted*,
# and refused at init *when the toolset is user-supplied/untrusted*. Runtime
# code (tool parameters) NEVER auto-installs under any trust level.
#
# NOTE: `pipmaster` is NOT in the inventory. Auto-installing the installer from
# inside a process that lacks the installer is not a legitimate bootstrap —
# when pipmaster is unavailable, `install_missing_dependencies()` returns an
# actionable error and the tool continues with whatever is already present.
# ──────────────────────────────────────────────────────────────────────────────

# Required packages at INIT time (import_name → pip_package_name).
# pandas/numpy/seaborn/sklearn/scipy are resolved lazily at runtime by
# `_ensure_import()` and must NOT gate initialization: `print(2+2)` has no
# business failing because scipy is not installed.
_REQUIRED_PACKAGES: Dict[str, str] = {
    "matplotlib": "matplotlib",
}

TRUSTED_BUILTIN: str = "trusted_builtin"   # shipped with lollms_client
TRUSTED_SIGNED: str = "trusted_signed"     # third-party but cryptographically signed
UNTRUSTED_USER: str = "untrusted_user"     # user-supplied toolset — no auto-install

_TRUSTED_LEVELS: frozenset[str] = frozenset({TRUSTED_BUILTIN, TRUSTED_SIGNED})

# This tool ships with the package — it is a builtin, first-party toolset.
TOOLSET_TRUST_LEVEL: str = TRUSTED_BUILTIN


def _resolve_trust_level(config: Optional[dict]) -> str:
    """Determines the effective trust level for this toolset's init call.

    Precedence: explicit `config["trust_level"]` > module constant >
    trusted default. Unknown values are coerced to UNTRUSTED_USER so a typo
    or hostile caller can never *raise* trust accidentally.
    """
    if isinstance(config, dict):
        declared = config.get("trust_level")
        if isinstance(declared, str) and declared in (TRUSTED_BUILTIN, TRUSTED_SIGNED, UNTRUSTED_USER):
            return declared
        if declared is not None:
            return UNTRUSTED_USER
    return TOOLSET_TRUST_LEVEL


class MissingDependency(RuntimeError):
    """Raised by tool code when user code needs a library that is not installed.

    The message is actionable (names the pip package) and never mentions
    the host's filesystem layout — it's safe to surface verbatim to the LLM.
    """


def check_dependencies() -> Dict[str, List[str]]:
    """Returns {"available": [...import_names], "missing": [...pip_names]}.

    Sandbox-safe: uses `importlib.util.find_spec` only — no module code is
    executed. This can be called at any time, from any context.
    """
    import importlib.util as _ilu
    available: List[str] = []
    missing: List[str] = []
    for import_name, pip_name in _REQUIRED_PACKAGES.items():
        try:
            spec = _ilu.find_spec(import_name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            available.append(import_name)
        else:
            missing.append(pip_name)
    return {"available": available, "missing": missing}


def install_missing_dependencies(packages: Optional[List[str]] = None) -> Dict[str, Any]:
    """Explicit, host-side installation of missing packages.

    Contract:
      - Runs OUTSIDE any sandbox. The host invokes this — either directly or
        via the trusted-toolset path inside `init_tools_library()`.
      - `packages` (optional) selects a subset of pip names; when omitted, all
        packages from `_REQUIRED_PACKAGES` that are currently missing are
        installed.
      - Returns a structured report: {"ok": bool, "installed": [...], "missing": [...], "error": str|None}.
      - If pipmaster is unavailable, an actionable error is returned (never
        raised) so callers can log and continue with degraded functionality.

    The tool's *runtime* surface (what runs in the sandbox) never triggers
    a package install. Only an explicit call (or trusted init) does.
    """
    report: Dict[str, Any] = {"ok": False, "installed": [], "missing": [], "error": None}
    inventory = check_dependencies()
    report["missing"] = list(inventory["missing"])
    if not report["missing"] and packages is None:
        report["ok"] = True
        return report

    target_packages = list(packages) if packages is not None else list(report["missing"])
    normalized: List[str] = []
    for name in target_packages:
        normalized.append(_REQUIRED_PACKAGES.get(name, name))
    report["missing"] = normalized

    try:
        import pipmaster as pm
    except ImportError as exc:
        report["error"] = (
            f"pipmaster is not installed ({exc}). Install it in the host "
            "environment: `pip install pipmaster`. The execute_python tool "
            "itself still works for packages that are already present."
        )
        return report

    try:
        pm.ensure_packages(normalized)
        report["ok"] = True
        report["installed"] = list(normalized)
        report["missing"] = [p for p in check_dependencies()["missing"]]
        return report
    except Exception as exc:
        report["error"] = f"pipmaster failed to install {normalized}: {exc}"
        return report


def _inside_sandbox() -> bool:
    """True iff a `SandboxContext` is currently active in this process.

    Detected by checking the builtins sentinels written by the kernel's
    `_install()` — the sandbox patches `builtins.exec`/`compile` to stubs that
    are tagged with `__lollms_blocked_builtin__`. We never call sandbox
    internals or import them; the sentinel attribute is the stable contract.
    """
    try:
        import builtins as _b
        for name in ("exec", "compile"):
            fn = getattr(_b, name, None)
            if fn is not None and getattr(fn, "__lollms_blocked_builtin__", False):
                return True
    except Exception:
        return False
    return False


def init_tools_library(config: dict = None) -> None:
    """Ensures the tool's dependencies and configures the plotting backend.

    LCP contract: `init_tools_library()` is the *trusted* place where a toolset
    declares and ensures what it needs. For a trusted toolset (builtin first-
    party, or signed third-party), missing dependencies are installed here via
    pipmaster — this is what makes a fresh `pip install lollms-client` just work
    on first tool use. For an untrusted (user-supplied) toolset, we do NOT
    install: missing libraries surface as a per-call `MissingDependency` error
    instead, so the host operator stays in control.

    HARD INVARIANTS:
      1. This function MUST NOT raise, whatever pipmaster/matplotlib does.
      2. This function MUST be idempotent.
      3. This function MUST NOT push a pip install through a live
         `SandboxContext` (see the `_inside_sandbox()` guard below).

    The security boundary is the *code the tool executes from its parameters*,
    not the tool's own initialization. Runtime code never auto-installs at any
    trust level — that separation is enforced by the execution path, whose
    only package-install reference is the pre-existing `_ensure_import()`.

    The autonomy configuration block below is the tool's own decided feature
    (see `_prompt_user_validation`); it is unchanged and runs regardless of
    trust level.
    """
    global AUTONOMY_LEVEL, _CONFIRM_HANDLER

    # ── Autonomy configuration (existing feature, trust-neutral) ──
    if config and isinstance(config, dict):
        autonomy = str(config.get("autonomy_level", "safe")).lower().strip()
        if autonomy in ("strict", "safe", "full_access"):
            AUTONOMY_LEVEL = autonomy
            ASCIIColors.info(f"[execute_python] Configured autonomy level: {AUTONOMY_LEVEL}")
        else:
            AUTONOMY_LEVEL = "safe"
        # Host-controlled pre-approval. Accepts either:
        #   - a bool (legacy `auto_approve: True`)  → approve ALL categories
        #   - an iterable of category names         → approve only those
        #     (e.g. ["network"] authorizes socket/requests but still prompts
        #     for subprocess).
        if "auto_approve" in config:
            raw = config.get("auto_approve")
            if isinstance(raw, bool):
                if raw:
                    _approve_categories(frozenset(_RISK_CATEGORY_SIGNALS.keys()))
                else:
                    _reset_approved_categories()
            elif isinstance(raw, (list, tuple, set, frozenset)):
                _reset_approved_categories()
                valid = {c for c in raw if c in _RISK_CATEGORY_SIGNALS}
                if valid:
                    _approve_categories(frozenset(valid))
            else:
                ASCIIColors.warning(
                    f"[execute_python] Ignoring invalid auto_approve config "
                    f"(expected bool or list of category names, got {type(raw).__name__})."
                )
        if "confirm_handler" in config:
            set_confirm_handler(config.get("confirm_handler"))
    else:
        AUTONOMY_LEVEL = "safe"

    # ── Plotting backend (harmless, always attempted) ──
    try:
        global matplotlib
        import matplotlib
        matplotlib.use("Agg", force=False)
    except Exception:
        pass

    trust_level = _resolve_trust_level(config)

    # ── Refuse install inside a live sandbox (host responsibility) ──
    if _inside_sandbox():
        try:
            inventory = check_dependencies()
            if inventory["missing"]:
                ASCIIColors.warning(
                    "[execute_python] init_tools_library() invoked inside an "
                    "active SandboxContext. Skipping dependency installation "
                    "(that is a host responsibility, not a sandbox one). "
                    f"Missing: {', '.join(inventory['missing'])}. "
                    "Call `install_missing_dependencies()` from host code."
                )
        except Exception:
            pass
        return

    # ── Untrusted toolsets: refuse auto-install, log, defer to runtime ──
    if trust_level not in _TRUSTED_LEVELS:
        try:
            inventory = check_dependencies()
            if inventory["missing"]:
                ASCIIColors.warning(
                    "[execute_python] Untrusted toolset "
                    f"(trust_level={trust_level!r}): missing dependencies "
                    f"{', '.join(inventory['missing'])} were NOT installed. "
                    "A trusted toolset (trusted_builtin/trusted_signed) would "
                    "install them; the operator can install explicitly via "
                    "`install_missing_dependencies([...])`."
                )
        except Exception:
            pass
        return

    # ── Trusted toolsets: install missing deps (never raise) ──
    try:
        report = install_missing_dependencies()
        if not report.get("ok") and report.get("error"):
            ASCIIColors.warning(
                f"[execute_python] Dependency installation failed: "
                f"{report['error']}"
            )
    except Exception as exc:
        ASCIIColors.warning(
            f"[execute_python] init_tools_library() caught an unexpected "
            f"error while ensuring dependencies: {exc}"
        )


def _can_prompt_interactive() -> bool:
    """Checks whether the standard input stream is an interactive terminal."""
    if not sys.stdin:
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _is_headless_environment() -> bool:
    """True ONLY when we are provably inside an automated test/CI context.

    IMPORTANT: this function must NOT use `sys.stdin.isatty()` as a proxy for
    "headless". A non-TTY stdin is extremely common in real interactive
    sessions — GUI apps (NiceGUI, Electron), VS Code's integrated terminal
    wrapper, Jupyter kernels, and Windows services all report
    `isatty() == False` while a human is very much present and expecting a
    prompt. Treating those as headless caused the reported bug: in `safe`
    mode, risky code was silently auto-approved instead of prompting.

    The correct contract:
      - pytest / CI  → headless (no human to prompt; sandbox is the control).
      - everything else → NOT headless. If no TTY is available, the caller
        falls through to the explicit "no interactive channel" rejection
        branch in `_prompt_user_validation()`, which tells the operator how
        to register a `confirm_handler` or switch to `full_access`. That is
        the safe default: refuse rather than silently allow.
    """
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("CI"):
        return True
    return False


def _prompt_user_validation(source: str, script_label: str, argv: Optional[List[Any]] = None) -> Tuple[str, str]:
    global _CONFIRM_HANDLER

    handler = _CONFIRM_HANDLER
    if not handler:
        for mod_name, mod in list(sys.modules.items()):
            if "execute_python" in mod_name and hasattr(mod, "_CONFIRM_HANDLER") and mod._CONFIRM_HANDLER:
                handler = mod._CONFIRM_HANDLER
                break

    # 1. Check custom host/GUI confirmation handler first (NiceGUI, WebUI, etc.)
    if handler is not None and callable(handler):
        try:
            try:
                res = handler(source, script_label, argv)
            except TypeError:
                try:
                    res = handler(source, script_label)
                except TypeError:
                    try:
                        res = handler({
                            "tool_name": "execute_python",
                            "action_type": "python_execution",
                            "source": source,
                            "script_label": script_label,
                            "argv": argv,
                            "label": script_label,
                            "content": source,
                            "metadata": {"argv": argv, "autonomy_level": AUTONOMY_LEVEL}
                        })
                    except TypeError:
                        res = handler(source)

            if isinstance(res, tuple):
                decision = res[0]
                reason = res[1] if len(res) > 1 else ""
            elif isinstance(res, bool):
                decision, reason = ("allow", "") if res else ("reject", "Declined by operator.")
            elif isinstance(res, str):
                decision, reason = res, ""
            else:
                decision, reason = "allow", ""
            return str(decision).lower().strip(), str(reason)
        except Exception as handler_err:
            ASCIIColors.warning(f"[execute_python] Confirm handler failed: {handler_err}")
            return "reject", f"Confirmation handler failed: {handler_err}"

    # 2. Headless (pytest / CI) — the sandbox IS the control.
    # The static AST validator and SandboxContext already block the dangerous
    # surface (subprocess, os.system, socket, ctypes, path escapes). A human
    # prompt cannot add safety where the runtime is authoritative; skipping
    # it here is correct and prevents the input() hang seen in Windows CI.
    #
    # NOTE: this branch fires ONLY for pytest/CI. A non-TTY stdin in a real
    # interactive session (GUI app, VS Code terminal, Jupyter) is NOT headless
    # — it falls through to the explicit rejection below so the operator is
    # told how to register a handler instead of having risky code silently
    # auto-approved.
    if _is_headless_environment():
        return "allow", ""

    # 3. Check interactive TTY for terminal CLI
    if _can_prompt_interactive():
        source_lines = source.splitlines()
        total_lines = len(source_lines)
        max_preview = 25

        preview_lines = []
        for i, line in enumerate(source_lines[:max_preview], 1):
            preview_lines.append(f"[dim]{i:3d} |[/dim] {line}")
        if total_lines > max_preview:
            preview_lines.append(f"[dim]    ... [{total_lines - max_preview} more lines — enter 'v' to view all][/dim]")

        panel_parts = [
            f"[bold cyan]Script / Target:[/bold cyan] [yellow]{script_label}[/yellow]",
            f"[bold cyan]Autonomy Mode:[/bold cyan] [green]{AUTONOMY_LEVEL.upper()}[/green] (Protected Workspace Sandbox)",
        ]
        if argv and len(argv) > 1:
            panel_parts.append(f"[bold cyan]Arguments:[/bold cyan] {argv[1:]}")

        panel_parts.append(f"\n[bold cyan]Code Preview ({total_lines} lines):[/bold cyan]")
        panel_parts.extend(preview_lines)
        panel_parts.append(
            "\n[bold yellow]⚠️  The LLM agent wants to execute this Python code in your workspace.[/bold yellow]"
        )

        ASCIIColors.panel(
            "\n".join(panel_parts),
            title=f"[bold yellow]🛡️ Python Execution Authorization ({AUTONOMY_LEVEL.upper()} Mode)[/bold yellow]",
            border_style="yellow"
        )

        # Make the category scope explicit so the operator knows what they are
        # granting. "always" is bound to the categories shown in the prompt
        # above (derived from the risk reason), NOT to the whole session.
        # We resolve the workspace root here so the filesystem category fires
        # correctly (only for genuine escapes, not for workspace-relative I/O).
        try:
            _ws_for_prompt = _get_workspace_root(None)
        except Exception:
            _ws_for_prompt = None
        risk_categories = set(_classify_risk_categories(
            _detect_risky_operations(source, workspace_root=_ws_for_prompt)
        ))
        # Deletion detection is INDEPENDENT of risk detection: an inside-workspace
        # `os.remove('tmp.txt')` carries no risk category (safe path, no shell),
        # but the operator's directive is "ask me before you delete anything" —
        # so a `delete` category is surfaced here for informational purposes AND
        # to gate the "always" branch (the delete category is never persisted; each
        # deletion batch prompts again).
        deletion_ops = _detect_deletion_ops(source)
        if deletion_ops:
            risk_categories.add("delete")
        cats_hint = ", ".join(sorted(risk_categories)) if risk_categories else "none"
        deletion_note = ""
        if deletion_ops:
            deletion_note = (
                f"\n  ⚠️  This snippet DELETES files/directories: "
                f"{', '.join(deletion_ops)}.\n"
                f"  Deletion confirmation is per-execution: a previous "
                f"'always' does not auto-approve it."
            )
        while True:
            try:
                choice = input(
                    f"  Authorize execution? [y]es / [n]o / "
                    f"[a]lways for category [{cats_hint}] / [v]iew full code "
                    f"(default: y): " + deletion_note
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "reject", "Execution interrupted by operator (Ctrl+C / EOF)."

            if choice in ("", "y", "yes"):
                # Explicit "yes" authorizes THIS execution including any
                # deletions it performs. The next deletion will prompt again
                # because `delete` is never added to the persistent set.
                return "allow", ""
            elif choice in ("a", "always"):
                # "always" persists every category in `risk_categories` EXCEPT
                # `delete` — deletions always prompt, per operator directive.
                return "always", ""
            elif choice in ("n", "no"):
                try:
                    reason = input("  Reason / feedback for the LLM (optional, press Enter to skip): ").strip()
                except (EOFError, KeyboardInterrupt):
                    reason = ""
                return "reject", reason or "Operator declined execution."
            elif choice in ("v", "view"):
                print("\n" + "=" * 80)
                print(f"📄 FULL SOURCE CODE: {script_label} ({total_lines} lines)")
                print("=" * 80)
                for i, line in enumerate(source_lines, 1):
                    print(f"{i:4d} | {line}")
                print("=" * 80 + "\n")
            else:
                ASCIIColors.yellow("  Invalid choice. Please enter 'y', 'n', 'a', or 'v'.")

    # 3. Fallback for non-interactive environments without an attached handler.
    # We cannot block indefinitely without a user channel, so we reject with a
    # clear, actionable message that instructs the operator on how to proceed.
    # This is the SAFE default: refuse rather than silently allow. The previous
    # behavior (silently returning "allow" via an over-eager headless check)
    # was the reported bug — risky code ran without ever asking the operator.
    return "reject", (
        "No interactive channel available (no TTY, no confirm_handler registered). "
        "To authorize this execution, either: "
        "(a) pass a confirm_handler callback to LollmsDiscussion.chat(), or "
        "(b) set python_autonomy_level='full_access' to bypass sandbox prompts."
    )


# Default wall-clock budget before the child is considered "long-running".
# Raised from 180s to 3600s because legitimate analytical workloads (model
# fitting, large-file processing, plot rendering) routinely exceed 3 minutes.
# On expiry the operator is PROMPTED (see `_timeout_decision_prompt`) rather
# than having the child silently killed.
_DEFAULT_EXECUTION_TIMEOUT_SECONDS: float = 3600.0

_IMPORT_CACHE: Dict[str, Any] = {}

def _timeout_decision_prompt(script_label: str, elapsed_seconds: float) -> str:
    """Asks the operator whether to keep waiting on a long-running child.

    Returns "continue" to extend the wait window, or "stop" to terminate.

    Priority order (same policy as `_prompt_user_validation`):
      1. Registered `_CONFIRM_HANDLER` — extended protocol. We pass a dict /
         callable with `action_type="timeout"`. The handler may return the
         canonical "continue"/"stop" strings; legacy booleans are still mapped
         (True→continue, False→stop) so no existing handler regresses.
      2. Headless (pytest/CI) → "stop": a blocking prompt in a headless
         context would hang the entire suite, and the sandbox is the control
         — the operator is not present to consent to a longer wait anyway.
      3. Interactive TTY → prompt the operator directly.
      4. No interactive channel → "stop" (fail safe; do NOT extend silently).

    This function never raises. A broken handler is treated as "stop".
    """
    handler = _CONFIRM_HANDLER
    if not handler:
        for mod_name, mod in list(sys.modules.items()):
            if "execute_python" in mod_name and hasattr(mod, "_CONFIRM_HANDLER") and mod._CONFIRM_HANDLER:
                handler = mod._CONFIRM_HANDLER
                break

    if handler is not None and callable(handler):
        try:
            # Pass a structured payload so the handler can distinguish a
            # timeout decision from a normal authorize/reject prompt.
            try:
                res = handler({
                    "tool_name": "execute_python",
                    "action_type": "timeout",
                    "script_label": script_label,
                    "elapsed_seconds": elapsed_seconds,
                })
            except TypeError:
                res = handler(script_label)
            if isinstance(res, tuple):
                decision = str(res[0]).strip().lower()
            elif isinstance(res, bool):
                decision = "continue" if res else "stop"
            elif isinstance(res, str):
                decision = res.strip().lower()
            else:
                decision = "stop"
            # Legacy handlers return "allow"/"always"/"reject"/"y"/"n"; map
            # them onto the timeout protocol so nothing regresses.
            if decision in ("continue", "wait", "allow", "always", "y", "yes", ""):
                return "continue"
            return "stop"
        except Exception as handler_err:
            ASCIIColors.warning(
                f"[execute_python] Timeout confirm handler failed: {handler_err}"
            )
            return "stop"

    if _is_headless_environment():
        return "stop"

    if _can_prompt_interactive():
        try:
            choice = input(
                f"  ⏱️  '{script_label}' has been running for {elapsed_seconds:.0f}s. "
                f"Keep waiting? [c]ontinue / [s]top (default: continue): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "stop"
        if choice in ("", "c", "continue", "y", "yes"):
            return "continue"
        return "stop"

    return "stop"


def _ensure_import(module_name: str, package_name: str = None):
    """Cached import helper.

    PERFORMANCE: importing pandas repeatedly through `__import__` costs ~150ms
    each call in a cold sys.modules; we cache in `_IMPORT_CACHE` so only the
    first call pays the import cost. Cache is process-wide and immutable
    (None = unavailable, object = loaded module).
    """
    if module_name in _IMPORT_CACHE:
        return _IMPORT_CACHE[module_name]
    try:
        mod = __import__(module_name)
        _IMPORT_CACHE[module_name] = mod
        return mod
    except ImportError:
        import pipmaster as pm
        pkg = package_name or module_name
        ASCIIColors.warning(f"[execute_python] Missing dependency '{pkg}'. Installing automatically...")
        try:
            pm.ensure_packages(pkg)
            mod = __import__(module_name)
            _IMPORT_CACHE[module_name] = mod
            return mod
        except Exception as install_err:
            ASCIIColors.error(f"[execute_python] Failed to auto-install '{pkg}': {install_err}")
            _IMPORT_CACHE[module_name] = None
            return None
    except Exception as import_crash:
        ASCIIColors.warning(
            f"[execute_python] Optional dependency '{module_name}' crashed during import "
            f"({type(import_crash).__name__}: {import_crash}). Continuing without it."
        )
        _IMPORT_CACHE[module_name] = None
        return None


def _sanitize_host_paths(text: str) -> str:
    """
    Strips absolute host filesystem paths from tool outputs to preserve
    sandbox opacity. The LLM must never learn the orchestrator's physical
    location (user folders, install directories).
    """
    if not text:
        return text
    root = str(Path.cwd().resolve())
    sanitized = text.replace(root, ".")
    return re.sub(r'[A-Za-z]:\\(?:Users|home)[\\/][^\s"\']*', '<host-path>', sanitized)


def _annotate_traceback(raw_traceback: str, source_code: str, script_label: str) -> str:
    """Appends the failing source line to a sandboxed-code traceback.

    Matches data_query's contract: when the child traceback names a frame in
    `script_label` at line N, we append `    >>> LINE N: <source>` immediately
    after that line. The LLM can then fix its own code without recounting
    lines. Never raises.
    """
    if not raw_traceback:
        return raw_traceback
    code_lines = source_code.splitlines()
    pattern = re.compile(r'File "' + re.escape(script_label) + r'", line (\d+)')
    annotated: List[str] = []
    for line in raw_traceback.splitlines():
        annotated.append(line)
        match = pattern.search(line)
        if match:
            line_no = int(match.group(1))
            if 1 <= line_no <= len(code_lines):
                annotated.append(f"    >>> LINE {line_no}: {code_lines[line_no - 1]}")
    return "\n".join(annotated)


def _ensure_workspace_subdir(workspace_root: Path, *parts: str) -> Path:
    """Creates (if needed) and returns a subdirectory under the workspace root.

    Centralizes the "don't pollute the root" policy: any tool artifact that is
    not a first-class user file (execution logs, captured plots, screenshots)
    must live in a dedicated subfolder. Never raises for the producer — the
    caller is expected to wrap in its own try/except.
    """
    target = workspace_root.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _window_output(text: str) -> str:
    """
    Builds a head/tail windowed preview of an execution output.

    Short text passes through unchanged. Long text keeps the first and last
    half-windows separated by a stripping marker, preserving both the beginning
    and the end of the log for immediate diagnosis while bounding preview size.
    """
    if not isinstance(text, str) or len(text) <= _PREVIEW_WINDOW_CHARS:
        return text if isinstance(text, str) else str(text)
    head = text[:_PREVIEW_HALF_CHARS]
    tail = text[-_PREVIEW_HALF_CHARS:]
    stripped = len(text) - _PREVIEW_WINDOW_CHARS
    marker = (
        f"\n... [stripped for brevity — {stripped} middle characters omitted "
        f"(use read file tools to inspect more)] ...\n"
    )
    return f"{head}{marker}{tail}"


def _persist_full_output(text: str, script_label: str, tool_ctx: Optional[Any] = None) -> Optional[str]:
    """
    Persists the full captured output to a uniquely named .log file INSIDE
    the workspace's ``logs/`` folder. Returns the workspace-relative path
    (e.g. "logs/exec_output_script_20250101_120000_ab12cd.log") on success,
    None on failure.

    Routing into ``logs/`` keeps the workspace root clean — execution logs
    are ephemeral diagnostic artifacts, not user files.
    """
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(script_label).stem)[:40] or "script"
        log_name = f"exec_output_{safe_label}_{stamp}_{uuid.uuid4().hex[:6]}.log"
        log_dir = _ensure_workspace_subdir(_get_workspace_root(tool_ctx), "logs")
        log_path = log_dir / log_name
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
        return f"logs/{log_name}"
    except Exception as write_err:
        ASCIIColors.warning(f"[execute_python] Failed to persist full output: {write_err}")
        return None


def _detect_risky_operations(
    source: str,
    workspace_root: Optional[Path] = None,
) -> Optional[str]:
    """
    Analyzes Python code AST for potentially risky operations:
      - Spawning external OS processes (subprocess, pty, commands, multiprocessing)
      - Shell execution and process management (os.system, os.popen, os.spawn*, os.exec*, os.kill*)
      - Dynamic system escapes (ctypes, winreg)
      - Filesystem access to absolute paths OUTSIDE the workspace (reads of
        site-packages, venv probing, host tree enumeration). Reported as the
        "filesystem" category so the operator can authorize it explicitly.
    Normal computation, data science (numpy, pandas, scipy, sklearn), document creation
    (python-docx, python-pptx, reportlab, openpyxl), matplotlib/seaborn plotting, and
    workspace-relative I/O return None.

    `workspace_root` (optional): when supplied, any string literal that looks
    like an absolute path and resolves outside it triggers the filesystem
    category. When omitted, only clearly-absolute OS paths (drive letters,
    /home, /etc, C:\\Users, ...) that could NOT be workspace-relative are
    flagged — this keeps inline ``code="print(2+2)"`` snippets free of false
    positives.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    forbidden_process_modules = {"subprocess", "pty", "commands", "multiprocessing"}
    forbidden_os_process_attrs = {
        "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle",
        "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
        "kill", "killpg"
    }
    risky_system_modules = {"ctypes", "winreg", "socket", "ssl", "requests",
                            "urllib", "urllib3", "http", "httpx", "aiohttp",
                            "ftplib", "smtplib", "telnetlib", "paramiko",
                            "psutil", "importlib", "runpy"}

    for node in ast.walk(tree):
        # 1. Direct imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0].lower()
                if root_mod in forbidden_process_modules:
                    return f"Attempting to spawn external processes via '{alias.name}'"
                if root_mod in risky_system_modules:
                    return f"Low-level system access via '{alias.name}'"

        # 2. From imports
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            root_mod = mod.split(".")[0]
            if root_mod in forbidden_process_modules:
                return f"Attempting to spawn external processes from '{mod}'"
            if root_mod in risky_system_modules:
                return f"Low-level system access from '{mod}'"
            if root_mod == "os":
                for alias in node.names:
                    if alias.name.lower() in forbidden_os_process_attrs:
                        return f"Executing system/process command via 'os.{alias.name}'"
            # `from subprocess import run` / `from socket import socket` etc.
            # must be caught even when the module root is not in the process
            # set — the imported NAME itself is the risky primitive.
            if root_mod in ("subprocess", "socket", "urllib", "requests",
                            "http", "httpx", "aiohttp", "ctypes"):
                return f"Importing risky primitive from '{mod}'"

        # 3. Direct call expressions
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr.lower()
                if attr_name in forbidden_os_process_attrs:
                    if isinstance(node.func.value, ast.Name) and node.func.value.id.lower() == "os":
                        return f"Executing system command via 'os.{node.func.attr}'"
            elif isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    mod_name = node.args[0].value.split(".")[0].lower()
                    if mod_name in forbidden_process_modules or mod_name in risky_system_modules:
                        return f"Dynamic import of '{node.args[0].value}'"
            # `getattr(os, 'system')(...)` / `getattr(subprocess, 'run')(...)`
            # — dynamic attribute access to a risky primitive must be flagged
            # so the operator is prompted in safe mode.
            elif isinstance(node.func, ast.Call) and isinstance(node.func.func, ast.Name) and node.func.func.id == "getattr":
                if len(node.func.args) >= 2 and isinstance(node.func.args[1], ast.Constant):
                    attr_name = str(node.func.args[1].value).lower()
                    if attr_name in forbidden_os_process_attrs:
                        return f"Dynamic access to risky attribute '{attr_name}' via getattr"

        # 4. Filesystem access to absolute paths outside the workspace.
        # Detect string literals that look like absolute host paths and resolve
        # outside `workspace_root` (when provided). This is what lets the LLM
        # legitimately probe a venv / site-packages directory after the operator
        # explicitly said "always allow filesystem", instead of being silently
        # blocked at the runtime scandir/open guard with no prompt.
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value
            if _looks_like_absolute_host_path(literal, workspace_root):
                return (
                    f"Filesystem access to a host filesystem path outside the "
                    f"workspace ('{_shorten_path_for_prompt(literal)}')"
                )

    return None


def _looks_like_absolute_host_path(literal: str, workspace_root: Optional[Path]) -> bool:
    """True iff `literal` is an absolute filesystem path that escapes the workspace.

    Conservative heuristic: only treats a string as an absolute path when it
    parses as one (`Path.is_absolute()`), is not a lone drive root (`C:\\`,
    `C:/`), and (when `workspace_root` is known) does NOT resolve inside it.
    Bare words like ``"hello"`` or Windows-less URLs never match.
    """
    if not literal or len(literal) < 3:
        return False
    # Reject strings that carry an obvious scheme (URLs, repo URIs).
    if "://" in literal:
        return False
    # A bare drive root like "C:\\" is directory-listing of a whole volume; we
    # do flag it (it's a broad escape), but we don't treat "C:\\" itself as a
    # "path outside" match unless workspace_root explicitly excludes it.
    try:
        candidate = Path(literal)
    except (ValueError, OSError):
        return False
    if not candidate.is_absolute():
        return False
    if workspace_root is not None:
        try:
            candidate.resolve(strict=False).relative_to(Path(workspace_root).resolve())
            return False  # inside workspace → not a filesystem escape
        except (ValueError, OSError):
            return True  # outside workspace → requires filesystem approval
    # Even without `workspace_root`, an absolute path that resolves under the
    # interpreter's own installation tree is NOT a "filesystem escape" — it is
    # ordinary interpreter introspection (`sys.executable`, `sys.prefix`,
    # `sysconfig.get_paths()`, venv probing). Flagging those was causing a
    # false-positive prompt on EVERY call that printed `sys.executable`. The
    # trusted roots are computed once at kernel import and mirror the same set
    # used by the runtime path guard.
    try:
        resolved_literal = Path(literal).resolve(strict=False)
        # `_sandbox_kernel` was resolved by `_import_sandbox_kernel()` at
        # module load with three import-context fallbacks — reuse it here so
        # this check works identically whether the toolset is imported as a
        # package, by file path, or from a dev checkout.
        for trusted_root in getattr(_sandbox_kernel, "_TRUSTED_READ_ROOTS", ()):
            try:
                resolved_literal.relative_to(trusted_root)
                return False  # interpreter's own tree → legitimate introspection
            except (ValueError, OSError):
                continue
    except Exception:
        pass
    # No workspace_root: flag only unambiguously-host paths (drive roots,
    # POSIX /home /etc /usr /var, Windows C:\\Users\\ ..., C:\\Program ...).
    s = literal.replace("\\", "/").lower()
    host_prefixes = (
        "c:/", "d:/", "e:/", "f:/", "/home/", "/etc/", "/usr/", "/var/",
        "/root/", "/opt/", "/tmp/", "/private/",
    )
    return any(s.startswith(p) for p in host_prefixes)


def _shorten_path_for_prompt(literal: str, limit: int = 70) -> str:
    """Trims a path for display in a prompt / error reason string."""
    if len(literal) <= limit:
        return literal
    return literal[: limit // 2] + "…" + literal[-(limit // 2):]


def _run_python_source(
    source: str,
    script_label: str,
    argv: Optional[List[Any]] = None,
    tool_ctx: Optional[Any] = None,
    pre_authorized: bool = False,
) -> Dict[str, Any]:
    """
    Executes the given Python source string in a sandboxed namespace with
    user validation, stdout/stderr capture, sys.argv override, safety enforcement, and matplotlib interception.

    If tool_ctx is provided, stdout lines are streamed to the UI in real-time
    via tool_ctx.emit_progress() for immediate feedback on long-running scripts.

    `pre_authorized=True` skips the user prompt (the caller — e.g.
    `tool_create_and_execute_python_file` — already solicited authorization
    for the same source). The static profile is then selected from the
    `pre_authorized` state: "authorized" if risky, "standard" otherwise.
    This avoids double-prompting the user for write+execute flows.
    """
    # Emit an initial status so the UI shows a "processing" indicator the
    # instant the tool is invoked — before AST validation, before the child
    # spawns, before any user code runs. Without this, long-running executions
    # leave the UI silent for the whole duration.
    _progress_tc = tool_ctx if (tool_ctx is not None and hasattr(tool_ctx, "emit_progress")) else None
    if _progress_tc is not None:
        try:
            _progress_tc.emit_progress(
                f"⏳ Executing Python ({script_label})…",
                {"stream": "status", "phase": "starting"},
            )
        except Exception:
            pass

    # ── AUTONOMY LEVEL DECISION MATRIX (runs BEFORE static validation) ──
    # Order matters: the user's consent must be solicited BEFORE the static
    # validator decides whether to block. The old order (static-first) meant
    # that in "safe" mode a risky operation was hard-blocked with
    # "BLOCKED BY SANDBOX" and the user was NEVER asked — the prompt code was
    # unreachable. The user's explicit instruction: "in safe mode (not strict
    # mode), I need to get notified and asked to validate or not instead of
    # always blocking."
    #
    # Decision matrix:
    #   1. STRICT:      prompt operator on EVERY execution turn.
    #   2. SAFE (default): auto-approve benign code; when a risky operation is
    #                    detected, PROMPT the user. On approval, run with the
    #                    "authorized" profile (permits the risky surface while
    #                    still blocking C-level escapes like ctypes/winreg).
    #                    On rejection, return a clear blocked error.
    #   3. FULL_ACCESS: no prompt; run with "authorized" profile if risky,
    #                    "standard" otherwise.
    # Determine the workspace root up front so the risk detector can
    # distinguish "inside workspace" from "outside workspace" paths — this is
    # what makes the filesystem category fire ONLY on genuine escapes, so
    # `open('data.csv')` never prompts while `open('C:/.../site-packages')` does.
    _workspace_root_for_detection = _get_workspace_root(tool_ctx)
    # Static risk detection — the reason string drives both the prompt label
    # and the category-scoped approval decision below. Computed once here so
    # the same reason is reused by the prompt and the static profile selector.
    risky_reason = _detect_risky_operations(source, workspace_root=_workspace_root_for_detection)
    # Category-scoped approval: only skip the prompt when EVERY risk category
    # present in this snippet is already pre-approved by the operator. A new
    # category (e.g. operator approved network, now sees subprocess) MUST
    # prompt — this is the core requirement of category-scoped "always".
    #
    # The `source` argument enables deletion detection: a snippet whose ONLY
    # unusual surface is `os.remove('x')` inside the workspace has NO risk
    # category, but MUST still be confirmed — the operator said "ask before
    # deleting". This overrides the auto-approval (see `_is_category_approved`).
    auto_approved = _is_category_approved(risky_reason, source=source)

    requires_user_prompt = False
    prompt_label = script_label
    user_authorized_risky = pre_authorized or auto_approved

    # Deletion detection runs independently: it forces the prompt even when
    # `risky_reason` is None (a purely workspace-internal deletion triggers no
    # risk category — safe path, no shell — but the operator has asked to be
    # consulted before anything is deleted).
    _deletion_ops = _detect_deletion_ops(source)
    _has_deletion = bool(_deletion_ops)

    if not pre_authorized and not auto_approved:
        if AUTONOMY_LEVEL == "strict":
            requires_user_prompt = True
            prompt_label = f"[STRICT] {script_label}"
        elif AUTONOMY_LEVEL == "safe" and (_has_deletion or risky_reason):
            requires_user_prompt = True
            label_parts: List[str] = []
            if risky_reason:
                label_parts.append(f"RISKY: {risky_reason}")
            if _has_deletion:
                label_parts.append(f"DELETES: {', '.join(_deletion_ops)}")
            prompt_label = f"[{' | '.join(label_parts)}] {script_label}"

    if requires_user_prompt:
        decision, rejection_reason = _prompt_user_validation(source, prompt_label, argv)
        if decision == "reject":
            reason_msg = rejection_reason or f"Risky operation detected ({risky_reason or 'Restricted operation'}) and not authorized by operator."
            ASCIIColors.warning(f"[execute_python] ❌ Execution blocked: {reason_msg}")
            return {
                "success": False,
                "error": f"🛑 BLOCKED BY SANDBOX ({AUTONOMY_LEVEL.upper()} MODE): {reason_msg}\n"
                         f"Code execution was blocked because it contains restricted operations ({risky_reason or 'unauthorized actions'}).\n"
                         "In safe mode, process spawning (subprocess, os.system, curl) and network calls require operator approval or 'full_access' mode.",
                "output": "",
                "stderr": f"Execution blocked: {reason_msg}"
            }
        elif decision == "always":
            # Grant ONLY the categories present in THIS snippet. Subsequent
            # code touching a different category will prompt again. The
            # `delete` category is never persisted (see `_approve_categories`):
            # every future deletion prompts the operator again, matching the
            # "ask before you delete" contract.
            approved_now = _classify_risk_categories(risky_reason)
            _approve_categories(approved_now)
            # If the operator authorized filesystem access, widen the runtime
            # path guard to the interpreter's installation roots so a
            # subsequent probe of site-packages / venv succeeds instead of
            # being silently hard-blocked at scandir/open with no prompt —
            # the exact bug reported ("I said always, but it still blocks").
            if "filesystem" in approved_now:
                _sync_approved_path_roots(_workspace_root_for_detection)
            user_authorized_risky = True
            cats_display = ", ".join(sorted(approved_now - {"delete"})) or "none"
            extra = " (deletions will still prompt)" if _has_deletion else ""
            ASCIIColors.success(
                f"[execute_python] 🔓 Auto-approval enabled for category(ies): {cats_display}{extra}."
            )
        elif decision == "allow":
            # HEADLESS AUTO-PROCEED: in pytest/CI there is no operator to
            # consent, so `_prompt_user_validation` returns "allow" to let the
            # static/runtime sandbox be the control. We deliberately do NOT
            # set `user_authorized_risky=True` here — that flag would promote
            # the static profile to "authorized", which PERMITS subprocess /
            # socket, defeating the security tests (`import subprocess` and
            # `import socket` would succeed in headless mode). Under headless
            # auto-proceed the profile stays "standard", so the standard
            # denylist correctly refuses those imports.
            if not _is_headless_environment():
                user_authorized_risky = True

    # ── LAYER 1: static AST validation ──
    # Profile selection:
    #   - "authorized" when the user explicitly approved a risky operation
    #     (permits subprocess/socket/urllib/requests; still blocks ctypes,
    #     winreg, os.system/exec*, eval/exec).
    #   - "authorized" also under full_access when risky (the operator has
    #     opted out of prompts — the same surface is permitted).
    #   - "standard" otherwise (unchanged: best posture for benign code).
    _static_profile = "standard"
    if (user_authorized_risky or AUTONOMY_LEVEL == "full_access") and risky_reason:
        _static_profile = "authorized"

    static_violation = _sandbox_validate_source(source, profile=_static_profile)
    if static_violation is not None:
        ASCIIColors.warning(
            f"[execute_python] Static sandbox rejection ({_static_profile}): "
            f"{static_violation.reason}"
        )
        return {
            "success": False,
            "error": (
                f"🛑 BLOCKED BY SANDBOX (line {static_violation.line}): "
                f"{static_violation.reason}\n"
                f"Offending construct: {static_violation.snippet}\n"
                "This primitive has no safe alternative inside the sandbox — even "
                "with user authorization it cannot be reached from Python code "
                "because it can escape the sandbox at the native level. "
                "Use the system_shell tool if you legitimately need host commands."
            ),
            "output": "",
            "stderr": "static_sandbox_rejection",
        }

    _np = None
    _plt = None

    try:
        import numpy as _np_mod
        _np = _np_mod
    except Exception:
        pass

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt_mod
        _plt = _plt_mod
    except Exception:
        pass

    try:
        pandas_mod = _ensure_import("pandas", "pandas")
    except Exception:
        pandas_mod = None
    try:
        seaborn_mod = _ensure_import("seaborn", "seaborn")
    except Exception:
        seaborn_mod = None
    try:
        sklearn_mod = _ensure_import("sklearn", "scikit-learn")
    except Exception:
        sklearn_mod = None
    try:
        scipy_mod = _ensure_import("scipy", "scipy")
    except Exception:
        scipy_mod = None

    class _NoReconfigureStringIO(io.StringIO):
        """StringIO that silently ignores reconfigure() calls from user code."""
        def reconfigure(self, *args, **kwargs):
            pass

    class _ProgressStringIO(io.StringIO):
        """
        StringIO that buffers output AND streams complete lines to the UI/terminal
        via tool_ctx.emit_progress() for real-time feedback on long-running scripts.
        Guards against recursive re-entry so callback prints go directly to the real terminal.
        """
        def __init__(self, tool_ctx: Optional[Any] = None, stream_name: str = "stdout", real_stream: Optional[Any] = None):
            super().__init__()
            self._tool_context = tool_ctx
            self._stream_name = stream_name
            self._real_stream = real_stream
            self._line_buffer = ""
            self._in_emit = False

        def write(self, text: str) -> int:
            if not text:
                return 0
            if self._in_emit:
                if self._real_stream is not None:
                    try:
                        self._real_stream.write(text)
                        self._real_stream.flush()
                    except Exception:
                        pass
                return len(text)

            written = super().write(text)
            if self._tool_context is not None:
                self._line_buffer += text
                while "\n" in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split("\n", 1)
                    clean_line = line.rstrip('\r\n')
                    if clean_line:
                        self._in_emit = True
                        try:
                            orig_stdout = sys.stdout
                            if self._real_stream is not None:
                                sys.stdout = self._real_stream
                            try:
                                if self._tool_context.emit_progress(clean_line, {"stream": self._stream_name}) is False:
                                    self.silence()
                                    raise KeyboardInterrupt
                            finally:
                                sys.stdout = orig_stdout
                        finally:
                            self._in_emit = False
            return written

        def flush(self):
            super().flush()
            if self._tool_context is not None and self._line_buffer.strip() and not self._in_emit:
                clean_line = self._line_buffer.rstrip('\r\n')
                self._line_buffer = ""
                if clean_line:
                    self._in_emit = True
                    try:
                        orig_stdout = sys.stdout
                        if self._real_stream is not None:
                            sys.stdout = self._real_stream
                        try:
                            if self._tool_context.emit_progress(clean_line, {"stream": self._stream_name}) is False:
                                self.silence()
                                raise KeyboardInterrupt
                        finally:
                            sys.stdout = orig_stdout
                    finally:
                        self._in_emit = False

        def reconfigure(self, *args, **kwargs):
            pass

        def silence(self):
            self._tool_context = None

    cwd_repr = os.path.abspath(os.getcwd())
    if cwd_repr not in sys.path:
        sys.path.insert(0, cwd_repr)

    workspace_root = _get_workspace_root(tool_ctx).resolve()

    # ── PURGE STALE WORKSPACE MODULES FROM sys.modules ──
    # When scripts or modules inside the workspace are edited/patched between rounds,
    # Python's default sys.modules cache holds onto the stale old module definitions.
    # Purge any module located within the workspace root so imports are always fresh from disk.
    import importlib
    # PERFORMANCE: String-based prefix check avoids a `Path.resolve()` syscall
    # per loaded module (hundreds of modules → hundreds of stat syscalls per
    # exec). Normalize both sides to lowercase with '/' for cross-platform.
    _ws_str = str(workspace_root).replace("\\", "/").lower()
    modules_to_purge = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, '__file__', None)
        if mod_file:
            try:
                _mf = str(mod_file).replace("\\", "/").lower()
                if _mf.startswith(_ws_str + "/") or _mf == _ws_str:
                    modules_to_purge.append(mod_name)
            except Exception:
                pass

    for mod_name in modules_to_purge:
        sys.modules.pop(mod_name, None)

    importlib.invalidate_caches()

    # ── SANDBOX EXECUTION — the shared SandboxContext handles everything ──
    # Historically this tool installed its own partial sandbox here (a custom
    # `_sandboxed_import`, `_blocked_subprocess`, `_bounded_remove`, etc).
    # That was redundant with `SandboxContext` and — critically — DOUBLED
    # every patch: imports went through `_sandboxed_import` then
    # `_filtered_import`, os.remove went through `_bounded_remove` (full
    # `resolve()` syscall per call) then through the kernel's guard, and
    # Toplevel `builtins.exec` was captured *mid-patch* by the kernel's
    # `TRUE_EXEC` snapshot → test failures reporting "dynamic code execution
    # is forbidden". All of this is fixed by deferring entirely to the kernel:
    # one patch set, one restore, correct path-only semantics.
    safe_builtins = dict(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__)
    # NOTE: `safe_builtins["__import__"]` is rebound to the sandbox-filtered
    # importer inside the `with SandboxContext(...)` block below. We never
    # install the shape ourselves — the kernel exposes it as
    # `SandboxContext.filtered_import`.

    local_vars = {
        "Path": Path,
        "pd": pandas_mod,
        "np": _np,
        "plt": _plt,
        "sns": seaborn_mod,
        "sklearn": sklearn_mod,
        "scipy": scipy_mod,
        "_ensure_import": _ensure_import,
        "__builtins__": safe_builtins,
        "__file__": script_label,
        "__name__": "__main__",
    }

    # ── PROCESS ISOLATION: run user code in a sandboxed child ──
    # The parent process is NEVER patched. The child installs SandboxContext in
    # its own address space; when it exits, everything dies with it. No state
    # leaks between tool calls, no cascade risk, no module-load-order hazard.
    ASCIIColors.info(
        f"⚡ Executing Python code in isolated sandbox child "
        f"(label: {script_label}, autonomy: {AUTONOMY_LEVEL})"
    )

    # Seed locals that must be available in the child. Modules are pickled by
    # reference (their import is replayed on unpickle), which is correct and
    # cheap. `_ensure_import` is a function — pickling it would fail because it
    # is defined in `execute_python`; the child re-imports that name via the
    # same module which is already loaded by spawn's parent-exec path, so we
    # skip it: user code that needs it can call `import pandas` directly.
    child_seed: Dict[str, Any] = {}
    for k, v in local_vars.items():
        if k in ("__builtins__", "_ensure_import"):
            continue
        child_seed[k] = v

    # `argv` must be a list of strings for sys.argv in the child.
    child_argv = list(argv) if argv is not None else [script_label]

    # The child runtime MUST use the same profile the AST validator used —
    # otherwise the user's authorization is silently overridden at exec time
    # (AST permits subprocess, runtime still blocks it → "BLOCKED BY SANDBOX"
    # after the user clicked "yes"). This was the exact bug reported.
    #
    # TIMEOUT POLICY: the child runs up to `_DEFAULT_EXECUTION_TIMEOUT_SECONDS`
    # (1 hour by default). On expiry, the operator is PROMPTED via
    # `_timeout_decision_prompt` — a "continue" answer extends the wait window
    # by the same slice; a "stop" answer terminates the child and returns a
    # TimeoutError envelope. The operator keeps control of any long-running
    # execution instead of the sandbox silently killing it.
    def _on_exec_timeout(elapsed: float) -> str:
        return _timeout_decision_prompt(script_label, elapsed)

    envelope = _run_in_sandboxed_child(
        source=source,
        workspace_root=workspace_root,
        profile=_static_profile,
        script_label=script_label,
        argv=child_argv,
        seed_locals=child_seed,
        persist_df_name=None,
        hard_timeout=_DEFAULT_EXECUTION_TIMEOUT_SECONDS,
        timeout_decision_cb=_on_exec_timeout,
    )

    progress_tc = tool_ctx if (tool_ctx is not None and hasattr(tool_ctx, "emit_progress")) else None

    raw_output = envelope.get("stdout", "") or ""
    raw_error = envelope.get("stderr", "") or ""
    child_traceback = envelope.get("traceback", "") or ""
    error_type = envelope.get("error_type", "") or ""

    # Replay the child's captured stdout/stderr through the parent's stdout so
    # users who watch the terminal see the same lines they always did.
    try:
        if raw_output:
            sys.stdout.write(raw_output)
            sys.stdout.flush()
        if raw_error:
            sys.stderr.write(raw_error)
            sys.stderr.flush()
    except Exception:
        pass

    # Best-effort streaming: if a tool context is attached, replay complete
    # lines through `emit_progress` after the fact. This gives the same UI
    # contract diff (buffer + emit) even though cross-process live streaming is
    # not possible on win32 spawn.
    if progress_tc is not None and raw_output:
        try:
            for line in raw_output.splitlines():
                if line.strip():
                    progress_tc.emit_progress(line, {"stream": "stdout", "replay": True})
        except Exception:
            pass

    # ── Child reported an error ──
    if not envelope.get("success"):
        annotated_traceback = _annotate_traceback(child_traceback, source, script_label)
        if envelope.get("sandbox_violation"):
            return {
                "success": False,
                "error": f"🛑 SANDBOX BLOCK: {annotated_traceback.strip()}",
                "output": _sanitize_host_paths(_window_output(raw_output)),
                "stderr": _sanitize_host_paths(_window_output(raw_error)),
            }
        if error_type == "TimeoutError":
            return {
                "success": False,
                "error": annotated_traceback.strip() or "Execution timed out.",
                "output": _sanitize_host_paths(_window_output(raw_output)),
                "stderr": _sanitize_host_paths(_window_output(raw_error)),
            }
        ASCIIColors.error(f"❌ Execution Failed (child):\n{child_traceback}")
        return {
            "success": False,
            "error": f"Execution Error:\n{_sanitize_host_paths(annotated_traceback)}",
            "output": _sanitize_host_paths(_window_output(raw_output)),
            "stderr": _sanitize_host_paths(_window_output(raw_error)),
        }

    # ── Success path: write any captured figures to workspace (parent-side) ──
    figures = envelope.get("figures") or []
    if figures:
        ASCIIColors.success(f"[Sandbox] Intercepted {len(figures)} generated plot figure(s)!")
        try:
            plots_dir = _ensure_workspace_subdir(_get_workspace_root(tool_ctx), "logs", "plots")
        except Exception:
            plots_dir = Path(".")  # last-resort fallback: never lose the figure
        for png_bytes in figures:
            plot_filename = f"code_exec_plot_{uuid.uuid4().hex[:6]}.png"
            try:
                (plots_dir / plot_filename).write_bytes(png_bytes)
            except Exception as write_err:
                ASCIIColors.warning(
                    f"[execute_python] Failed to write plot {plot_filename}: {write_err}"
                )

    out_str = raw_output
    err_str = raw_error

    if not out_str.strip():
        out_str = "Code executed successfully (no stdout prints)."

    out_str = out_str.replace("\r\n", "\n").replace("\r", "\n")
    err_str = err_str.replace("\r\n", "\n").replace("\r", "\n")

    if len(out_str) > _PREVIEW_WINDOW_CHARS:
        log_name = _persist_full_output(out_str, script_label, tool_ctx)
        if log_name:
            stripped_chars = len(out_str) - _PREVIEW_WINDOW_CHARS
            out_str = (
                f"{out_str[:_PREVIEW_HALF_CHARS]}"
                f"\n... [stripped for brevity — {stripped_chars} middle characters omitted] ...\n"
                f"[FULL OUTPUT SAVED] The complete output was persisted to '{log_name}' in the workspace. "
                "Use read file tools to inspect the omitted middle.\n"
                f"{out_str[-_PREVIEW_HALF_CHARS:]}"
            )
        else:
            out_str = _window_output(out_str)

    out_str = "".join(ch for ch in out_str if ch.isascii() or ch in "\n\t")

    return {
        "success": True,
        "output": _sanitize_host_paths(out_str),
        "stderr": _sanitize_host_paths(_window_output(err_str)),
    }


def _get_workspace_root(tool_ctx: Optional[Any] = None) -> Path:
    """
    Resolves the workspace root.
    Prefers the host-injected ToolContext workspace (authoritative — it is the
    exact root artifacts are flushed to), then falls back to CWD-based
    resolution for agnostic contexts.
    """
    if tool_ctx is not None:
        ctx_ws = getattr(tool_ctx, "workspace_path", None)
        if ctx_ws:
            try:
                ws = Path(ctx_ws).resolve()
                if ws.exists() and ws.is_dir():
                    return ws
            except Exception:
                pass
    return Path.cwd().resolve()


def _resolve_workspace_path(file_name: str, tool_ctx: Optional[Any] = None) -> Optional[Path]:
    """
    Safely resolves a file path inside the workspace sandbox.
    Blocks path traversal ('..') and absolute paths escaping the root.
    Supports direct resolution under root and CWD as well as under .lollms_code/scripts/.
    """
    if not file_name or not isinstance(file_name, str):
        return None

    clean = file_name.replace("\\", "/").strip().lstrip("/")
    if not clean or ".." in Path(clean).parts:
        return None

    root = _get_workspace_root(tool_ctx)
    cwd = Path.cwd().resolve()

    search_roots = [root]
    if cwd != root:
        search_roots.append(cwd)

    for base in search_roots:
        # 1. Direct path check under base
        try:
            candidate = (base / clean).resolve()
            candidate.relative_to(base)
            if candidate.exists() and candidate.is_file():
                return candidate
        except ValueError:
            pass

        # 2. Resilient check under subdirectories
        sub_candidates = [
            base / ".lollms_code" / "scripts" / clean,
            base / ".lollms_code" / clean,
            base / "scripts" / clean,
            base / ".lollms_code" / "scripts" / Path(clean).name,
            base / "workspace_data" / clean,
            base / "data_workspace" / clean,
        ]
        for sc in sub_candidates:
            try:
                res_sc = sc.resolve()
                res_sc.relative_to(base)
                if res_sc.exists() and res_sc.is_file():
                    return res_sc
            except Exception:
                continue

    return (root / clean).resolve()


def _normalize_argv(script_label: str, args: Optional[List[Any]]) -> List[str]:
    """Builds a safe sys.argv list: [script, arg1, arg2, ...]. All values coerced to str."""
    argv = [script_label]
    for arg in (args or []):
        if isinstance(arg, (dict, list, tuple)):
            import json as _json
            argv.append(_json.dumps(arg, default=str))
        else:
            argv.append(str(arg))
    return argv


_MAX_INLINE_CODE_CHARS = 2000


def _enforce_inline_code_limit(code_str: str) -> Optional[Dict[str, Any]]:
    """
    Hard gate for inline code execution. Returns a failure result when the
    snippet exceeds the inline size limit, or None when it is within bounds.
    """
    if len(code_str.strip()) <= _MAX_INLINE_CODE_CHARS:
        return None
    ASCIIColors.error(
        f"[execute_python] Inline code rejected: {len(code_str.strip())} chars "
        f"exceeds the {_MAX_INLINE_CODE_CHARS}-char limit."
    )
    return {
        "success": False,
        "error": (
            f"BLOCKED: inline code execution is restricted to SHORT, punctual snippets "
            f"(max {_MAX_INLINE_CODE_CHARS} characters). Your snippet was "
            f"{len(code_str.strip())} characters. Do NOT shrink the code by removing "
            f"whitespace or shortening names. Instead you MUST:\n"
            f"1. Emit an <artifact type=\"code\" name=\"your_script.py\"> tag containing "
            f"the COMPLETE program.\n"
            f"2. Wait for the artifact to be persisted to the workspace.\n"
            f"3. Call 'tool_execute_python_file' with file_name=\"your_script.py\".\n"
            f"This workflow makes the program inspectable, patchable via SEARCH/REPLACE, "
            f"and reusable. Inline execution is reserved for quick checks and one-liners."
        ),
        "output": "",
        "stderr": ""
    }


def tool_execute_python_code(code: str = "", env: Optional[Any] = None) -> Dict[str, Any]:
    """
    Executes a SHORT, PUNCTUAL inline Python snippet and returns stdout, stderr, and generated plots.

    STRICTLY RESERVED for short, punctual, throwaway code: quick checks,
    one-liner computations, tiny experiments of a few lines. Nothing is written
    to disk unless your code explicitly saves files. This tool does NOT execute
    files and never saves the code to a file.

    DO NOT use this tool for substantial programs. For anything multi-step,
    reusable, or iterative, the PREFERRED path is: emit an
    <artifact type="code"> tag to persist the .py file, then run it with
    'tool_execute_python_file'.

    SANDBOX INTEGRITY: in safe mode, process execution and shell escape calls
    (subprocess, os.system, os.popen) are blocked. You cannot use this tool
    to bypass shell restrictions.

    Args:
        code (str): The raw Python source code to execute inline. Required.
        env: Optional host-injected runtime context. Its tool_context (when present)
            streams stdout/stderr lines live to the application callback during execution.
    """
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python] Unwrapping nested dictionary parameter.")
        code = code.get("code") or next((v for v in code.values() if isinstance(v, str)), "")

    code_str = code if isinstance(code, str) else ("" if code is None else str(code))

    # Accept a workspace .py file path as `code` for parity with
    # tool_execute_python_data_query. Agents frequently pass a script path
    # to the inline tool (e.g. `str(script)` from a helper helper); the
    # PREVIOUS behavior fed the raw path directly to `exec()`, producing
    # `SyntaxError: unexpected character after line continuation character`
    # on Windows (each backslash in `C:\...\hello.py` is a line
    # continuation). Resolving the path to its CONTENTS up front keeps the
    # rest of the pipeline identical and dials tool behavior correctly.
    stripped_input = code_str.strip() if code_str else ""
    resolved_from_file = False
    if stripped_input and stripped_input.lower().endswith(".py"):
        candidate = Path(stripped_input)
        if candidate.is_file():
            try:
                code_str = candidate.read_text(encoding="utf-8")
            except Exception as read_err:
                return {
                    "success": False,
                    "error": f"Failed to read '{stripped_input}': {read_err}",
                    "output": "",
                    "stderr": "",
                }


    # The inline-size limit steers agents toward artifacts for SUBSTANTIAL
    # code. A file that already lives on disk has been persisted — the limit
    # does not apply; we are simply running it.
    if not resolved_from_file:
        gate_result = _enforce_inline_code_limit(code_str)
        if gate_result is not None:
            return gate_result
    if not code_str.strip():
        return {
            "success": False,
            "error": (
                "The 'code' parameter is empty. Provide the raw Python source string "
                "to execute inline. If you meant to run an existing workspace .py "
                "file, call 'tool_execute_python_file' with its 'file_name' instead."
            ),
            "output": "",
            "stderr": ""
        }

    tool_ctx = getattr(env, "tool_context", None) if env else None
    return _run_python_source(code_str, "python_code", tool_ctx=tool_ctx)


def _sanitize_python_file_name(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validates and normalizes a target .py file path for the combined write+run tool.

    Returns (clean_rel_path, None) on success, or (None, error_message) on rejection.

    The returned value is a **workspace-relative** path and MAY contain
    subdirectory components (e.g. ``"scripts/build_presentation.py"``). This is
    the correct contract: the workspace root is the containment boundary, and
    the final ``target_path.relative_to(workspace_root)`` check in
    ``tool_create_and_execute_python_file`` is what actually enforces it —
    traversal below the workspace is a legitimate organizing pattern, not an
    escape.

    Enforces:
      * non-empty;
      * no parent-traversal (``..``) segments — this is the STRING-level escape
        check that runs before `Path.resolve()`;
      * no absolute paths (leading ``/``, drive letters, ``~`` expansion) — the
        tool only ever writes INSIDE the workspace, so absolute targets are
        rejected at parse time;
      * ``.py`` extension appended when missing;
      * the *basename* does not collide with the tool's own ``exec_output_*``
        log naming scheme.

    The authoritative containment check remains the post-resolve
    ``relative_to(workspace_root)`` guard in the caller — that is what catches
    symlink escapes and (on Windows) alternate-drive tricks the string scan
    cannot see. This helper deliberately leaves that check to the caller so
    the resolve cost is paid once, against the real workspace root.
    """
    if not isinstance(file_name, str) or not file_name.strip():
        return None, "The 'file_name' parameter is empty. Provide a target .py file path (e.g. 'script.py' or 'scripts/script.py')."

    raw = file_name.strip().replace("\\", "/").strip()

    # -- Reject absolute paths and tilde. Both are ALWAYS escapes: the tool
    #    writes only inside the workspace, so an absolute target has no
    #    legitimate meaning here.
    if raw.startswith("~"):
        return None, (
            f"Invalid file name: '{file_name}'. '~' expansion is not allowed; "
            "provide a path relative to the workspace root."
        )
    if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
        return None, (
            f"Invalid file name: '{file_name}'. Absolute paths are not allowed; "
            "provide a path relative to the workspace root (e.g. "
            "'scripts/build_presentation.py')."
        )

    # -- Reject '..' segments explicitly. This is the string-level traversal
    #    guard; the caller's `relative_to(workspace_root)` after `resolve()`
    #    is the authoritative one (catches symlink-based escapes too).
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts:
        return None, f"Invalid file name: '{file_name}'."
    if any(p == ".." for p in parts):
        return None, (
            f"Invalid file name: '{file_name}'. Parent traversal ('..') is "
            "not allowed; provide a path relative to the workspace root."
        )

    # --- Extension & reserved-prefix checks on the BASENAME only: nothing
    #     represents the log naming scheme beyond the final component, and a
    #     subdirectory named `exec_output_scripts/` should remain usable.
    basename = parts[-1]
    if not basename.lower().endswith(".py"):
        basename = f"{basename}.py"
        parts[-1] = basename
    if basename.lower().startswith("exec_output_"):
        return None, (
            f"Reserved file name: '{basename}'. Names starting with "
            "'exec_output_' are used for persisted execution logs. Choose a "
            "different name."
        )

    return "/".join(parts), None


def tool_create_and_execute_python_file(
    code: str = "",
    file_name: str = "",
    args: Optional[List[Any]] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    PREFERRED tool for substantial programs: writes 'code' to a workspace .py file
    and executes it in ONE call, returning stdout, stderr, and generated plots.

    Collapses the wasteful two-call pattern (inline run, then run-file) into a
    single round-trip. The file is written to the workspace root and executed
    through the exact same sandboxed pipeline as 'tool_execute_python_file'
    (matplotlib Agg, stdout/stderr capture, .log persistence, user validation).

    The file is written ONLY AFTER user authorization succeeds, so a rejected
    prompt leaves no dangling file behind. If the file already exists it is
    overwritten (intended for iterative fixes).

    SANDBOX INTEGRITY: in safe mode, process execution and shell escape calls
    (subprocess, os.system, os.popen) are blocked.

    Args:
        code (str): The complete Python source to write and execute. Required.
        file_name (str): Workspace-RELATIVE target .py file path. It MAY include
            subdirectories (e.g. ``'scripts/build_presentation.py'``); they are
            created on demand. Absolute paths, ``~`` expansion, and ``..``
            traversal are rejected. Required.
        args (list): Optional command-line style arguments passed via sys.argv[1:].
        env: Optional host-injected runtime context; its tool_context streams stdout live.
    """
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python] Unwrapping nested dictionary parameter.")
        code = code.get("code") or next((v for v in code.values() if isinstance(v, str)), "")

    code_str = code if isinstance(code, str) else ("" if code is None else str(code))
    if not code_str.strip():
        return {
            "success": False,
            "error": (
                "The 'code' parameter is empty. Provide the complete Python source "
                "to write and execute. To run an existing workspace .py file instead, "
                "call 'tool_execute_python_file'."
            ),
            "output": "",
            "stderr": ""
        }

    clean_name, name_error = _sanitize_python_file_name(file_name)
    if name_error:
        return {
            "success": False,
            "error": name_error,
            "output": "",
            "stderr": ""
        }

    tool_ctx = getattr(env, "tool_context", None) if env else None
    workspace_root = _get_workspace_root(tool_ctx)
    # `clean_name` is already normalized to forward slashes and is guaranteed
    # to have no '..' segments, no leading slash, and no drive letter (see
    # `_sanitize_python_file_name`). The `resolve()` + `relative_to()` pair is
    # the AUTHORITATIVE containment check: it catches symlink-based escapes
    # that the string scan cannot detect, and — because resolve() is a no-op
    # for a path that doesn't yet exist — it also protects the write that
    # follows.
    target_path = (workspace_root / clean_name).resolve()
    try:
        target_path.relative_to(workspace_root.resolve())
    except ValueError:
        return {
            "success": False,
            "error": (
                f"Invalid or unsafe file path: '{file_name}'. The resolved "
                "target escapes the workspace root (symlink or mount point)."
            ),
            "output": "",
            "stderr": ""
        }

    # Authorize BEFORE writing so a rejection leaves no dangling file behind.
    # Same decision matrix as `_run_python_source` — the user must be prompted
    # for risky code in safe mode, and their approval must select the
    # "authorized" profile so the runtime doesn't re-block what they approved.
    risky_reason = _detect_risky_operations(code_str, workspace_root=workspace_root)
    requires_user_prompt = False
    prompt_label = clean_name

    # Category-scoped approval: skip the prompt only when ALL categories of
    # this snippet are pre-approved. A different category must prompt again.
    # Deletion detection adds an independent prompt trigger — the same
    # "ask before deleting" contract as the inline tool.
    _deletion_ops = _detect_deletion_ops(code_str)
    _has_deletion = bool(_deletion_ops)
    if not _is_category_approved(risky_reason, source=code_str):
        if AUTONOMY_LEVEL == "strict":
            requires_user_prompt = True
            prompt_label = f"[STRICT] {clean_name}"
        elif AUTONOMY_LEVEL == "safe" and (_has_deletion or risky_reason):
            requires_user_prompt = True
            label_parts: List[str] = []
            if risky_reason:
                label_parts.append(f"RISKY: {risky_reason}")
            if _has_deletion:
                label_parts.append(f"DELETES: {', '.join(_deletion_ops)}")
            prompt_label = f"[{' | '.join(label_parts)}] {clean_name}"

    if requires_user_prompt:
        decision, rejection_reason = _prompt_user_validation(code_str, prompt_label, args)
        if decision == "reject":
            reason_msg = rejection_reason or (
                f"Risky operation detected ({risky_reason or 'Restricted operation'}) "
                "and not authorized by operator."
            )
            ASCIIColors.warning(f"[execute_python] ❌ Write+execute blocked: {reason_msg}")
            return {
                "success": False,
                "error": (
                    f"🛑 BLOCKED BY SANDBOX ({AUTONOMY_LEVEL.upper()} MODE): {reason_msg}\n"
                    "The file was NOT written and the code was NOT executed."
                ),
                "output": "",
                "stderr": f"Execution rejected: {reason_msg}"
            }
        if decision == "always":
            # Persist every category EXCEPT `delete` (see `_approve_categories`):
            # deletions always prompt again. The alias maps every risky
            # category from the classifier, plus the `filesystem` widening
            # that unlocks interpreter-tree reads.
            approved_now = _classify_risk_categories(risky_reason)
            _approve_categories(approved_now)
            if "filesystem" in approved_now:
                _sync_approved_path_roots(workspace_root)
            cats_display = ", ".join(sorted(approved_now - {"delete"})) or "none"
            extra = " (deletions will still prompt)" if _has_deletion else ""
            ASCIIColors.success(
                f"[execute_python] 🔓 Auto-approval enabled for category(ies): {cats_display}{extra}."
            )

    try:
        # Create any subdirectory components (e.g. `scripts/`) required by the
        # target path. `exist_ok=True` makes repeated calls idempotent; the
        # parent directory is spawned ONLY after the containment check above,
        # so it is guaranteed to be inside the workspace.
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(code_str, encoding="utf-8")
    except Exception as write_err:
        return {
            "success": False,
            "error": f"Failed to write file '{clean_name}': {write_err}",
            "output": "",
            "stderr": ""
        }

    argv = _normalize_argv(clean_name, args)
    # `pre_authorized=True`: this function already solicited authorization
    # above. Passing it through avoids a second prompt for the same source
    # (which would be confusing UX and, worse, would let a user authorize on
    # the write prompt then get re-blocked by the exec prompt's static check).
    result = _run_python_source(
        code_str, clean_name, argv=argv, tool_ctx=tool_ctx, pre_authorized=True,
    )

    if isinstance(result, dict):
        result["file_path"] = clean_name
        if result.get("success"):
            result["message"] = (
                f"Wrote '{clean_name}' to the workspace and executed it in one call."
            )
    return result


def tool_execute_python_file(
    file_name: str = "",
    args: Optional[List[Any]] = None,
    env: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Preferred execution path: executes an existing Python file from the workspace and returns stdout, stderr, and generated plots.

    Reads the .py file that ALREADY EXISTS on disk in the sandboxed workspace and runs it.
    The script is invoked exactly like `python <file_name> <arg1> <arg2> ...`: arguments are
    exposed via sys.argv[1:] or argparse. This tool ONLY READS existing files — it never
    saves, creates, or overwrites files.

    SANDBOX INTEGRITY: in safe mode, process execution and shell escape calls
    (subprocess, os.system, os.popen) are blocked.

    Args:
        file_name (str): Name of an existing .py file in the workspace to execute. Required. Path traversal is blocked.
        args (list): Optional command-line style arguments passed to the script via sys.argv[1:]. All values are stringified.
        env: Optional host-injected runtime context. Its tool_context (when present)
            streams stdout/stderr lines live to the application callback during execution.
    """
    file_name_str = file_name if isinstance(file_name, str) else ("" if file_name is None else str(file_name))
    if not file_name_str.strip():
        return {
            "success": False,
            "error": (
                "The 'file_name' parameter is empty. Provide the name of an EXISTING "
                ".py file in the workspace. To run inline Python code instead, call "
                "'tool_execute_python_code' with the 'code' parameter."
            ),
            "output": "",
            "stderr": ""
        }

    tool_ctx = getattr(env, "tool_context", None) if env else None
    resolved = _resolve_workspace_path(file_name_str, tool_ctx)
    if resolved is None:
        return {
            "success": False,
            "error": f"Invalid or unsafe file path: '{file_name_str}'. Path traversal is blocked.",
            "output": "",
            "stderr": ""
        }
    if not resolved.exists():
        return {
            "success": False,
            "error": f"File '{file_name_str}' not found in workspace. Write it first (e.g. via an <artifact type=\"code\"> tag), or verify the exact file name in the workspace tree.",
            "output": "",
            "stderr": ""
        }
    if resolved.suffix.lower() != ".py":
        return {
            "success": False,
            "error": f"File '{file_name_str}' is not a Python file (only .py files can be executed).",
            "output": "",
            "stderr": ""
        }

    try:
        source = resolved.read_text(encoding="utf-8")
    except Exception as read_err:
        return {
            "success": False,
            "error": f"Failed to read file '{file_name_str}': {read_err}",
            "output": "",
            "stderr": ""
        }

    argv = _normalize_argv(file_name_str, args)
    return _run_python_source(source, file_name_str, argv=argv, tool_ctx=tool_ctx)