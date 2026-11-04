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
[y]es (run once), [a]lways (auto-approve for this session), [n]o (reject with feedback), [v]iew full code.
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
try:
    from .sandbox_kernel import (
        validate_source as _sandbox_validate_source,
        SandboxViolation as _SandboxViolation,
        SandboxPermissionError as _SandboxPermissionError,
        run_in_sandboxed_child as _run_in_sandboxed_child,
    )
except ImportError:  # pragma: no cover
    from src.lollms_client.tools_bindings.lcp.default_tools.execute_python.sandbox_kernel import (  # type: ignore
        validate_source as _sandbox_validate_source,
        SandboxViolation as _SandboxViolation,
        SandboxPermissionError as _SandboxPermissionError,
        run_in_sandboxed_child as _run_in_sandboxed_child,
    )

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
    global AUTONOMY_LEVEL, _AUTO_APPROVE_PYTHON, _CONFIRM_HANDLER

    # ── Autonomy configuration (existing feature, trust-neutral) ──
    if config and isinstance(config, dict):
        autonomy = str(config.get("autonomy_level", "safe")).lower().strip()
        if autonomy in ("strict", "safe", "full_access"):
            AUTONOMY_LEVEL = autonomy
            ASCIIColors.info(f"[execute_python] Configured autonomy level: {AUTONOMY_LEVEL}")
        else:
            AUTONOMY_LEVEL = "safe"
        if "auto_approve" in config:
            _AUTO_APPROVE_PYTHON = bool(config.get("auto_approve"))
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
    """True when `input()` must never be called.

    Windows CI frequently reports `sys.stdin.isatty() == True` even when stdin
    is a pipe with no reader — a raw `input()` then blocks indefinitely and
    hangs the entire test suite (this was the source of the ~500s extra
    runtime in the previous CI run). We treat pytest, CI, and unwrapped stdin
    as headless and skip the interactive prompt entirely. When the caller
    genuinely wants a prompt, they must register a `confirm_handler`; that is
    the sole supported interactive channel and it always wins.
    """
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("CI"):
        return True
    if not sys.stdin:
        return True
    try:
        return not sys.stdin.isatty()
    except Exception:
        return True


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

    # 2. Headless (pytest / CI / no reader) — the sandbox IS the control.
    # The static AST validator and SandboxContext already block the dangerous
    # surface (subprocess, os.system, socket, ctypes, path escapes). A human
    # prompt cannot add safety where the runtime is authoritative; skipping
    # it here is correct and prevents the input() hang seen in Windows CI.
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

        while True:
            try:
                choice = input("  Authorize execution? [y]es / [n]o / [a]lways for session / [v]iew full code (default: y): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "reject", "Execution interrupted by operator (Ctrl+C / EOF)."

            if choice in ("", "y", "yes"):
                return "allow", ""
            elif choice in ("a", "always"):
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
    # clear, actionable message that instructs the LLM on how to proceed.
    return "reject", (
        "No interactive channel available (no TTY, no confirm_handler registered). "
        "To authorize this execution, either: "
        "(a) pass a confirm_handler callback to LollmsDiscussion.chat(), or "
        "(b) set python_autonomy_level='full_access' to bypass sandbox prompts."
    )


_IMPORT_CACHE: Dict[str, Any] = {}

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
    Persists the full captured output to a uniquely named .log file in the
    workspace root. Returns the file name on success, None on failure.
    """
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(script_label).stem)[:40] or "script"
        log_name = f"exec_output_{safe_label}_{stamp}_{uuid.uuid4().hex[:6]}.log"
        log_path = _get_workspace_root(tool_ctx) / log_name
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
        return log_name
    except Exception as write_err:
        ASCIIColors.warning(f"[execute_python] Failed to persist full output: {write_err}")
        return None


def _detect_risky_operations(source: str) -> Optional[str]:
    """
    Analyzes Python code AST for potentially risky operations:
      - Spawning external OS processes (subprocess, pty, commands, multiprocessing)
      - Shell execution and process management (os.system, os.popen, os.spawn*, os.exec*, os.kill*)
      - Dynamic system escapes (ctypes, winreg)
    Normal computation, data science (numpy, pandas, scipy, sklearn), document creation
    (python-docx, python-pptx, reportlab, openpyxl), matplotlib/seaborn plotting, and workspace I/O return None.
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

    return None


def _run_python_source(source: str, script_label: str, argv: Optional[List[Any]] = None, tool_ctx: Optional[Any] = None) -> Dict[str, Any]:
    """
    Executes the given Python source string in a sandboxed namespace with
    user validation, stdout/stderr capture, sys.argv override, safety enforcement, and matplotlib interception.
    
    If tool_ctx is provided, stdout lines are streamed to the UI in real-time
    via tool_ctx.emit_progress() for immediate feedback on long-running scripts.
    """
    global _AUTO_APPROVE_PYTHON

    # ── LAYER 1: static AST validation (denylist profile) ──
    # Rejects forbidden imports / calls BEFORE any byte runs. Runs under every
    # autonomy level; there is no bypass even under full_access because these
    # primitives (ctypes, subprocess, socket) can escape the runtime layer.
    static_violation = _sandbox_validate_source(source, profile="standard")
    if static_violation is not None:
        ASCIIColors.warning(
            f"[execute_python] Static sandbox rejection: {static_violation.reason}"
        )
        return {
            "success": False,
            "error": (
                f"🛑 BLOCKED BY SANDBOX (line {static_violation.line}): "
                f"{static_violation.reason}\n"
                f"Offending construct: {static_violation.snippet}\n"
                "This primitive has no safe alternative inside the sandbox — even "
                "in 'full_access' mode it cannot be reached from Python code. "
                "Use the system_shell tool if you legitimately need host commands."
            ),
            "output": "",
            "stderr": "static_sandbox_rejection",
        }

    # ── AUTONOMY LEVEL DECISION MATRIX ──
    # 1. STRICT: Prompts operator on EVERY execution turn.
    # 2. SAFE (Default): Auto-approves benign code (algorithms, data science, docx/pptx/pdf/xlsx, matplotlib/seaborn).
    #    Prompts user / blocks when risky operations (process spawning, subprocess/os.system, network calls) are detected.
    # 3. FULL ACCESS: Executes everything without confirmation prompts.
    risky_reason = _detect_risky_operations(source)

    requires_user_prompt = False
    prompt_label = script_label
    user_authorized_risky = False

    if not _AUTO_APPROVE_PYTHON:
        if AUTONOMY_LEVEL == "strict":
            requires_user_prompt = True
            prompt_label = f"[STRICT] {script_label}"
        elif AUTONOMY_LEVEL == "safe" and risky_reason:
            requires_user_prompt = True
            prompt_label = f"[RISKY: {risky_reason}] {script_label}"

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
            _AUTO_APPROVE_PYTHON = True
            user_authorized_risky = True
            ASCIIColors.success("[execute_python] 🔓 Auto-approval enabled for this session.")
        elif decision == "allow":
            user_authorized_risky = True

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

    envelope = _run_in_sandboxed_child(
        source=source,
        workspace_root=workspace_root,
        profile="standard",
        script_label=script_label,
        argv=child_argv,
        seed_locals=child_seed,
        persist_df_name=None,
        hard_timeout=180.0,
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
        for png_bytes in figures:
            plot_filename = f"code_exec_plot_{uuid.uuid4().hex[:6]}.png"
            try:
                (Path(".") / plot_filename).write_bytes(png_bytes)
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
    Validates and normalizes a target .py file name for the combined write+run tool.

    Returns (clean_name, None) on success, or (None, error_message) on rejection.
    Enforces: non-empty, no directory components, no traversal, .py extension,
    and rejects names that collide with the tool's own log naming scheme.
    """
    if not isinstance(file_name, str) or not file_name.strip():
        return None, "The 'file_name' parameter is empty. Provide a target .py file name (e.g. 'script.py')."

    raw = file_name.strip()
    if ".." in raw or "/" in raw or "\\" in raw or raw.startswith("~"):
        return None, (
            f"Invalid or unsafe file name: '{file_name}'. Directory components, "
            "path traversal ('..'), and '~' are blocked. Provide a bare file name "
            "such as 'script.py'."
        )

    name = Path(raw).name
    if not name or name in (".", ".."):
        return None, f"Invalid file name: '{file_name}'."

    if not name.lower().endswith(".py"):
        name = f"{name}.py"

    if name.lower().startswith("exec_output_"):
        return None, (
            f"Reserved file name: '{name}'. Names starting with 'exec_output_' are "
            "used for persisted execution logs. Choose a different name."
        )

    return name, None


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
        file_name (str): Target .py file name in the workspace (bare name, no
            directories). Required. Path traversal is blocked.
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
    target_path = (workspace_root / clean_name).resolve()
    try:
        target_path.relative_to(workspace_root.resolve())
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid or unsafe file path: '{file_name}'. Path traversal is blocked.",
            "output": "",
            "stderr": ""
        }

    # Authorize BEFORE writing so a rejection leaves no dangling file behind.
    global _AUTO_APPROVE_PYTHON
    risky_reason = _detect_risky_operations(code_str)
    requires_user_prompt = False
    prompt_label = clean_name

    if not _AUTO_APPROVE_PYTHON:
        if AUTONOMY_LEVEL == "strict":
            requires_user_prompt = True
            prompt_label = f"[STRICT] {clean_name}"
        elif AUTONOMY_LEVEL == "safe" and risky_reason:
            requires_user_prompt = True
            prompt_label = f"[RISKY: {risky_reason}] {clean_name}"

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
            _AUTO_APPROVE_PYTHON = True
            ASCIIColors.success("[execute_python] 🔓 Auto-approval enabled for this session.")

    try:
        target_path.write_text(code_str, encoding="utf-8")
    except Exception as write_err:
        return {
            "success": False,
            "error": f"Failed to write file '{clean_name}': {write_err}",
            "output": "",
            "stderr": ""
        }

    argv = _normalize_argv(clean_name, args)
    result = _run_python_source(code_str, clean_name, argv=argv, tool_ctx=tool_ctx)

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