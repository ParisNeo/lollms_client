"""
execute_python.py
LCP toolset for sandboxed Python execution. Two single-purpose tools:
  - tool_execute_python_code:   executes INLINE Python source (the 'code' string).
                                Nothing is saved to disk by the tool itself.
  - tool_execute_python_file:   executes an EXISTING workspace .py file with optional
                                argv-style arguments (sys.argv / argparse compatible).
                                Read-only: never saves, creates, or overwrites files.
Interception of matplotlib figures and stdout/stderr capture.

When captured stdout exceeds the inline preview window, the FULL output is
persisted to a timestamped .log file in the workspace and the tool result
embeds a head/tail windowed preview plus an explicit pointer telling the LLM
which inspection tools to use to read the omitted middle.

HUMAN-IN-THE-LOOP AUTHORIZATION:
In safe mode, execution prompts the user with a code preview and authorization choices:
[y]es (run once), [a]lways (auto-approve for this session), [n]o (reject with feedback), [v]iew full code.
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

TOOL_LIBRARY_NAME = "Execute Python"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code. PREFERRED path: persist code as a .py artifact and run it with tool_execute_python_file. tool_execute_python_code is strictly reserved for short, punctual inline snippets. Long outputs are persisted to a .log file with a head/tail preview returned inline. Returns stdout, stderr, and generated plots."
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
    """Sets a custom confirmation handler (used by GUI or host app)."""
    global _CONFIRM_HANDLER
    _CONFIRM_HANDLER = handler


def init_tools_library(config: dict = None) -> None:
    global AUTONOMY_LEVEL, _AUTO_APPROVE_PYTHON, _CONFIRM_HANDLER
    if config and isinstance(config, dict):
        autonomy = config.get("autonomy_level", "safe").lower()
        if autonomy in ("safe", "full_access"):
            AUTONOMY_LEVEL = autonomy
            ASCIIColors.info(f"[execute_python] Configured autonomy level: {AUTONOMY_LEVEL}")
        else:
            AUTONOMY_LEVEL = "safe"
        if "auto_approve" in config:
            _AUTO_APPROVE_PYTHON = bool(config.get("auto_approve"))
        if "confirm_handler" in config:
            _CONFIRM_HANDLER = config.get("confirm_handler")
    else:
        AUTONOMY_LEVEL = "safe"

    try:
        import pipmaster as pm
        pm.ensure_packages(["matplotlib", "pipmaster"])
        global matplotlib
        import matplotlib
        matplotlib.use('Agg')
    except Exception as e:
        import ascii_colors
        ascii_colors.ASCIIColors.warning(f"[execute_python] Failed to ensure dependencies: {e}")


def _can_prompt_interactive() -> bool:
    """Checks whether the standard input stream is an interactive terminal."""
    if not sys.stdin:
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _prompt_user_validation(source: str, script_label: str, argv: Optional[List[Any]] = None) -> Tuple[str, str]:
    """
    Prompts the user for validation when executing Python code in safe mode.
    Returns:
        (decision, reason)
        where decision is one of: "allow", "always", "reject"
    """
    # 1. Check custom host/GUI confirmation handler first
    if _CONFIRM_HANDLER is not None and callable(_CONFIRM_HANDLER):
        try:
            decision, reason = _CONFIRM_HANDLER(source, script_label, argv)
            return decision, reason
        except Exception as handler_err:
            ASCIIColors.warning(f"[execute_python] Confirm handler failed: {handler_err}")
            return "allow", ""

    # 2. Check interactive TTY for terminal CLI
    if not _can_prompt_interactive():
        return "allow", ""

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
        f"[bold cyan]Autonomy Mode:[/bold cyan] [green]SAFE[/green] (Protected Workspace Sandbox)",
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
        title="[bold yellow]🛡️ Python Execution Authorization (Safe Mode)[/bold yellow]",
        border_style="yellow"
    )

    while True:
        try:
            choice = input("  Authorize execution? [y]es / [n]o / [a]lways for session / [v]iew full code (default: y): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return "reject", "Execution interrupted by user (Ctrl+C / EOF)."

        if choice in ("", "y", "yes"):
            return "allow", ""
        elif choice in ("a", "always"):
            return "always", ""
        elif choice in ("n", "no"):
            try:
                reason = input("  Reason / feedback for the LLM (optional, press Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            return "reject", reason
        elif choice in ("v", "view"):
            print("\n" + "=" * 80)
            print(f"📄 FULL SOURCE CODE: {script_label} ({total_lines} lines)")
            print("=" * 80)
            for i, line in enumerate(source_lines, 1):
                print(f"{i:4d} | {line}")
            print("=" * 80 + "\n")
        else:
            ASCIIColors.yellow("  Invalid choice. Please enter 'y', 'n', 'a', or 'v'.")


def _ensure_import(module_name: str, package_name: str = None):
    try:
        return __import__(module_name)
    except ImportError:
        import pipmaster as pm
        pkg = package_name or module_name
        ASCIIColors.warning(f"[execute_python] Missing dependency '{pkg}'. Installing automatically...")
        try:
            pm.ensure_packages(pkg)
            return __import__(module_name)
        except Exception as install_err:
            ASCIIColors.error(f"[execute_python] Failed to auto-install '{pkg}': {install_err}")
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


def _persist_full_output(text: str, script_label: str) -> Optional[str]:
    """
    Persists the full captured output to a uniquely named .log file in the
    workspace root. Returns the file name on success, None on failure.
    """
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(script_label).stem)[:40] or "script"
        log_name = f"exec_output_{safe_label}_{stamp}_{uuid.uuid4().hex[:6]}.log"
        log_path = _get_workspace_root() / log_name
        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(text)
        return log_name
    except Exception as write_err:
        ASCIIColors.warning(f"[execute_python] Failed to persist full output: {write_err}")
        return None


def _check_python_code_safety(source: str) -> Optional[str]:
    """
    Enforces sandbox integrity in safe mode by analyzing code AST before execution.
    Rejects process execution escapes (subprocess, os.system, os.popen, pty) and
    prevents using Python to circumvent shell command restrictions.
    """
    if AUTONOMY_LEVEL != "safe":
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    forbidden_modules = {"subprocess", "pty", "commands"}
    forbidden_os_attrs = {
        "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
        "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle",
        "execlp", "execlpe", "execv", "execve", "execvp", "execvpe"
    }

    for node in ast.walk(tree):
        # 1. Direct imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0].lower()
                if root_mod in forbidden_modules:
                    return (
                        f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Importing '{alias.name}' is prohibited in safe mode.\n"
                        "Spawning system processes and executing shell commands from Python is blocked by the sandbox.\n"
                        "You cannot bypass shell restrictions using Python. "
                        "If this task genuinely requires elevated privileges, ask the user to enable 'full_access' mode via `/shell`."
                    )

        # 2. From imports
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").lower()
            root_mod = mod.split(".")[0]
            if root_mod in forbidden_modules:
                return (
                    f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Importing from '{mod}' is prohibited in safe mode.\n"
                    "Spawning system processes and executing shell commands from Python is blocked by the sandbox.\n"
                    "If this task genuinely requires elevated privileges, ask the user to enable 'full_access' mode via `/shell`."
                )
            if root_mod == "os":
                for alias in node.names:
                    if alias.name.lower() in forbidden_os_attrs:
                        return (
                            f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Importing 'os.{alias.name}' is prohibited in safe mode.\n"
                            "Executing shell commands via Python is blocked by the sandbox.\n"
                            "If this task genuinely requires elevated privileges, ask the user to enable 'full_access' mode via `/shell`."
                        )

        # 3. Call expressions on os.system, os.popen, etc.
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr.lower()
                if attr_name in forbidden_os_attrs:
                    if isinstance(node.func.value, ast.Name) and node.func.value.id.lower() == "os":
                        return (
                            f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Calling 'os.{node.func.attr}' is prohibited in safe mode.\n"
                            "Executing arbitrary shell commands via Python is blocked by the sandbox.\n"
                            "If this task genuinely requires elevated privileges, ask the user to enable 'full_access' mode via `/shell`."
                        )
            elif isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    mod_name = node.args[0].value.split(".")[0].lower()
                    if mod_name in forbidden_modules:
                        return f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Dynamic import of '{node.args[0].value}' is prohibited in safe mode."

    return None


def _run_python_source(source: str, script_label: str, argv: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Executes the given Python source string in a sandboxed namespace with
    user validation, stdout/stderr capture, sys.argv override, safety enforcement, and matplotlib interception.
    """
    global _AUTO_APPROVE_PYTHON

    safety_violation = _check_python_code_safety(source)
    if safety_violation:
        ASCIIColors.error(f"[execute_python] {safety_violation}")
        return {
            "success": False,
            "error": safety_violation,
            "output": "",
            "stderr": safety_violation
        }

    # ── SYSTEMIC HUMAN VALIDATION IN SAFE MODE ──
    if AUTONOMY_LEVEL == "safe" and not _AUTO_APPROVE_PYTHON:
        decision, rejection_reason = _prompt_user_validation(source, script_label, argv)
        if decision == "reject":
            reason_msg = rejection_reason or "The user reviewed the code and declined permission to execute it."
            ASCIIColors.warning(f"[execute_python] ❌ Execution rejected by user: {reason_msg}")
            return {
                "success": False,
                "error": f"🛑 EXECUTION REJECTED BY USER: {reason_msg}\nThe user inspected your Python code and denied execution authorization. Revise your approach, ask the user for clarification, or modify the code based on their feedback.",
                "output": "",
                "stderr": f"Execution rejected by user: {reason_msg}"
            }
        elif decision == "always":
            _AUTO_APPROVE_PYTHON = True
            ASCIIColors.success("[execute_python] 🔓 Auto-approval enabled for this session. Python execution will run autonomously without prompts.")
        elif decision == "allow":
            pass

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

    pandas_mod = _ensure_import("pandas", "pandas")
    seaborn_mod = _ensure_import("seaborn", "seaborn")
    sklearn_mod = _ensure_import("sklearn", "scikit-learn")
    scipy_mod = _ensure_import("scipy", "scipy")

    class _NoReconfigureStringIO(io.StringIO):
        """StringIO that silently ignores reconfigure() calls from user code."""
        def reconfigure(self, *args, **kwargs):
            pass

    cwd_repr = os.path.abspath(os.getcwd())
    if cwd_repr not in sys.path:
        sys.path.insert(0, cwd_repr)

    workspace_root = _get_workspace_root().resolve()

    # ── PURGE STALE WORKSPACE MODULES FROM sys.modules ──
    # When scripts or modules inside the workspace are edited/patched between rounds,
    # Python's default sys.modules cache holds onto the stale old module definitions.
    # Purge any module located within the workspace root so imports are always fresh from disk.
    import importlib
    modules_to_purge = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, '__file__', None)
        if mod_file:
            try:
                mod_path = Path(mod_file).resolve()
                if mod_path.is_relative_to(workspace_root):
                    modules_to_purge.append(mod_name)
            except (ValueError, Exception):
                pass

    for mod_name in modules_to_purge:
        sys.modules.pop(mod_name, None)

    importlib.invalidate_caches()

    # ── DEFENSE-IN-DEPTH: RUNTIME SANDBOX WRAPPERS (SAFE MODE) ──
    safe_builtins = dict(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__)
    orig_import = safe_builtins.get("__import__", __import__)

    def _sandboxed_import(name, *args, **kwargs):
        if AUTONOMY_LEVEL == "safe":
            root_mod = name.split(".")[0].lower()
            if root_mod in ("subprocess", "pty", "commands"):
                raise PermissionError(
                    f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Importing '{name}' is forbidden in safe mode. "
                    "Process execution and shell escapes via Python are blocked. Ask user to enable 'full_access' mode if required."
                )
        return orig_import(name, *args, **kwargs)

    safe_builtins["__import__"] = _sandboxed_import

    orig_os_system = getattr(os, "system", None)
    orig_os_popen = getattr(os, "popen", None)
    orig_os_remove = getattr(os, "remove", None)
    orig_os_unlink = getattr(os, "unlink", None)
    orig_os_rmdir = getattr(os, "rmdir", None)
    orig_shutil_rmtree = getattr(shutil, "rmtree", None)

    if AUTONOMY_LEVEL == "safe":
        def _blocked_system(*args, **kwargs):
            raise PermissionError("🛑 BLOCKED BY SANDBOX (SAFE MODE): 'os.system' cannot be used to run shell commands in safe mode.")

        def _blocked_popen(*args, **kwargs):
            raise PermissionError("🛑 BLOCKED BY SANDBOX (SAFE MODE): 'os.popen' cannot be used to run shell commands in safe mode.")

        def _bounded_remove(path, *args, **kwargs):
            p = Path(path).resolve()
            try:
                p.relative_to(workspace_root)
            except ValueError:
                raise PermissionError(f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Deleting file '{path}' outside workspace is forbidden.")
            return orig_os_remove(path, *args, **kwargs)

        def _bounded_rmdir(path, *args, **kwargs):
            p = Path(path).resolve()
            try:
                p.relative_to(workspace_root)
            except ValueError:
                raise PermissionError(f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Deleting folder '{path}' outside workspace is forbidden.")
            return orig_os_rmdir(path, *args, **kwargs)

        def _bounded_rmtree(path, *args, **kwargs):
            p = Path(path).resolve()
            try:
                p.relative_to(workspace_root)
            except ValueError:
                raise PermissionError(f"🛑 BLOCKED BY SANDBOX (SAFE MODE): Deleting directory tree '{path}' outside workspace is forbidden.")
            return orig_shutil_rmtree(path, *args, **kwargs)

        os.system = _blocked_system
        os.popen = _blocked_popen
        if orig_os_remove:
            os.remove = _bounded_remove
        if orig_os_unlink:
            os.unlink = _bounded_remove
        if orig_os_rmdir:
            os.rmdir = _bounded_rmdir
        if orig_shutil_rmtree:
            shutil.rmtree = _bounded_rmtree

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

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv[:]
    old_dunder_stdout = getattr(sys, "__stdout__", None)
    old_dunder_stderr = getattr(sys, "__stderr__", None)
    redirected_output = _NoReconfigureStringIO()
    redirected_error = _NoReconfigureStringIO()
    exec_stdout_hijacked = False
    exec_stderr_hijacked = False

    sys.stdout = redirected_output
    sys.stderr = redirected_error
    sys.__stdout__ = redirected_output
    sys.__stderr__ = redirected_error
    if argv is not None:
        sys.argv = argv

    try:
        ASCIIColors.info(f"⚡ Executing Python code (label: {script_label}, autonomy: {AUTONOMY_LEVEL})")
        if _plt is not None:
            _plt.clf()
            _plt.close('all')

        try:
            exec(compile(source, script_label, "exec"), local_vars)
        except SystemExit as se:
            raise RuntimeError(f"User code called sys.exit() with code {se.code}. This is not permitted in sandboxed execution.") from se
        except KeyboardInterrupt:
            raise RuntimeError("Execution interrupted by KeyboardInterrupt.")
        except Exception:
            import traceback
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            raw_output = redirected_output.getvalue()
            raw_error = redirected_error.getvalue()
            raw_traceback = traceback.format_exc()
            ASCIIColors.error(f"❌ Execution Failed:\n{raw_traceback}")
            return {
                "success": False,
                "error": f"Execution Error:\n{_sanitize_host_paths(raw_traceback)}",
                "output": _sanitize_host_paths(_window_output(raw_output)),
                "stderr": _sanitize_host_paths(_window_output(raw_error))
            }

        exec_stdout_hijacked = sys.stdout is not redirected_output
        exec_stderr_hijacked = sys.stderr is not redirected_error
        if exec_stdout_hijacked and hasattr(sys.stdout, "getvalue"):
            try:
                swapped_stdout = sys.stdout.getvalue()
                if isinstance(swapped_stdout, str) and swapped_stdout.strip():
                    redirected_output.write(swapped_stdout)
            except Exception:
                pass
        if exec_stderr_hijacked and hasattr(sys.stderr, "getvalue"):
            try:
                swapped_stderr = sys.stderr.getvalue()
                if isinstance(swapped_stderr, str) and swapped_stderr.strip():
                    redirected_error.write(swapped_stderr)
            except Exception:
                pass

        fig_nums = _plt.get_fignums() if _plt is not None else []
        if fig_nums:
            ASCIIColors.success(f"[Sandbox] Intercepted {len(fig_nums)} generated plot figure(s)!")
            for idx, f_num in enumerate(fig_nums):
                buf = io.BytesIO()
                fig = _plt.figure(f_num)
                fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                _ = base64.b64encode(buf.getvalue()).decode('utf-8')

                plot_filename = f"code_exec_plot_{uuid.uuid4().hex[:6]}.png"
                plot_path = Path(".") / plot_filename
                fig.savefig(str(plot_path), bbox_inches='tight', facecolor=fig.get_facecolor())

            _plt.close('all')

    except BaseException as outer_err:
        import traceback
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        raw_output = redirected_output.getvalue()
        raw_error = redirected_error.getvalue()
        raw_traceback = traceback.format_exc()
        ASCIIColors.error(f"❌ Unexpected execution failure:\n{raw_traceback}")
        return {
            "success": False,
            "error": f"Unexpected execution failure:\n{_sanitize_host_paths(raw_traceback)}",
            "output": _sanitize_host_paths(_window_output(raw_output)),
            "stderr": _sanitize_host_paths(_window_output(raw_error))
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.__stdout__ = old_dunder_stdout
        sys.__stderr__ = old_dunder_stderr
        sys.argv = old_argv

        # Restore system functions if modified
        if AUTONOMY_LEVEL == "safe":
            if orig_os_system:
                os.system = orig_os_system
            if orig_os_popen:
                os.popen = orig_os_popen
            if orig_os_remove:
                os.remove = orig_os_remove
            if orig_os_unlink:
                os.unlink = orig_os_unlink
            if orig_os_rmdir:
                os.rmdir = orig_os_rmdir
            if orig_shutil_rmtree:
                shutil.rmtree = orig_shutil_rmtree

    out_str = redirected_output.getvalue()
    err_str = redirected_error.getvalue()

    if not out_str.strip():
        if exec_stdout_hijacked:
            out_str = (
                "Code executed successfully, but NO stdout was captured: the script "
                "reassigned sys.stdout to a stream the sandbox cannot read (e.g. a file "
                "or an os-level descriptor), so its print output bypassed capture. In "
                "sandboxed scripts, use plain print() and NEVER reassign sys.stdout."
            )
        else:
            out_str = "Code executed successfully (no stdout prints)."

    out_str = out_str.replace("\r\n", "\n").replace("\r", "\n")
    err_str = err_str.replace("\r\n", "\n").replace("\r", "\n")

    if len(out_str) > _PREVIEW_WINDOW_CHARS:
        log_name = _persist_full_output(out_str, script_label)
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
        "stderr": _sanitize_host_paths(_window_output(err_str))
    }


def _get_workspace_root() -> Path:
    """
    Resolves the workspace root.
    The orchestrator chdirs into the sandboxed workspace before invoking this tool,
    so process CWD is authoritative.
    """
    cwd = Path.cwd()
    if cwd.name in ("workspace_data", "data_workspace") or (cwd / "data_workspace").exists():
        return cwd
    return Path("./data_workspace").resolve()


def _resolve_workspace_path(file_name: str) -> Optional[Path]:
    """
    Safely resolves a file path inside the workspace sandbox.
    Blocks path traversal ('..') and absolute paths escaping the root.
    Returns None when the path is unsafe.
    """
    if not file_name or not isinstance(file_name, str):
        return None

    root = _get_workspace_root()
    clean = file_name.replace("\\", "/").lstrip("/")

    if not clean or ".." in Path(clean).parts:
        return None

    candidate = (root / clean).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


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


def tool_execute_python_code(code: str = "") -> Dict[str, Any]:
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

    return _run_python_source(code_str, "python_code")


def tool_execute_python_file(
    file_name: str = "",
    args: Optional[List[Any]] = None,
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

    resolved = _resolve_workspace_path(file_name_str)
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
    return _run_python_source(source, file_name_str, argv=argv)