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

TOOL SELECTION DOCTRINE:
    Privilege tool_execute_python_file. Only fall back to
    tool_execute_python_code when the code is short, punctual, and disposable.
    Multi-step logic, algorithms, classes, or anything worth inspecting, fixing,
    iterating on, or reusing belongs in a persisted .py artifact.
"""

import os
import re
import sys
import io
import uuid
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code. PREFERRED path: persist code as a .py artifact and run it with tool_execute_python_file. tool_execute_python_code is strictly reserved for short, punctual inline snippets. Long outputs are persisted to a .log file with a head/tail preview returned inline. Returns stdout, stderr, and generated plots."
TOOL_LIBRARY_ICON = "🐍"

_PREVIEW_WINDOW_CHARS = 8000
_PREVIEW_HALF_CHARS = _PREVIEW_WINDOW_CHARS // 2
_STRIP_MARKER = (
    "\n... [stripped for brevity (use read file tools to inspect more)] ...\n"
)


def init_tools_library(config: dict = None) -> None:
    try:
        import pipmaster as pm
        pm.ensure_packages(["matplotlib", "pipmaster"])
        global matplotlib
        import matplotlib
        matplotlib.use('Agg')
    except Exception as e:
        import ascii_colors
        ascii_colors.ASCIIColors.warning(f"[execute_python] Failed to ensure dependencies: {e}")


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


def _run_python_source(source: str, script_label: str, argv: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Executes the given Python source string in a sandboxed namespace with
    stdout/stderr capture, sys.argv override, and matplotlib figure interception.
    """
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

    local_vars = {
        "Path": Path,
        "pd": pandas_mod,
        "np": _np,
        "plt": _plt,
        "sns": seaborn_mod,
        "sklearn": sklearn_mod,
        "scipy": scipy_mod,
        "_ensure_import": _ensure_import,
        "__builtins__": __builtins__,
        "__file__": script_label,
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv[:]
    redirected_output = _NoReconfigureStringIO()
    redirected_error = _NoReconfigureStringIO()

    sys.stdout = redirected_output
    sys.stderr = redirected_error
    if argv is not None:
        sys.argv = argv

    try:
        ASCIIColors.info(f"⚡ Executing arbitrary Python code (label: {script_label})")
        sibling_note = (
            "[sandbox] CWD = workspace root. "
            "Artifact .py files are importable as siblings: use 'from <module> import ...' directly; "
            "do NOT prepend 'workspace/' or manipulate sys.path."
        )
        print(sibling_note)
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
        sys.argv = old_argv

    out_str = redirected_output.getvalue()
    err_str = redirected_error.getvalue()

    if not out_str.strip():
        out_str = "Code executed successfully (no stdout prints)."

    workspace_contract = (
        "\n\n[WORKSPACE NOTE] The sandbox CWD is the workspace root. "
        "Files created via <artifact> tags are siblings of your code: import them directly "
        "(e.g. 'from rlc_filter import RLCFilter') without path prefixes or sys.path manipulation."
    )
    if script_label == "python_code":
        code_len = len(source.strip())
        doctrine_note = (
            " [TOOL SELECTION DOCTRINE] This tool is STRICTLY for short, punctual snippets. "
            "For substantial or reusable code, emit an <artifact type=\"code\"> tag to persist "
            "the .py file, then run it with tool_execute_python_file."
        )
        if code_len > 800:
            doctrine_note += (
                f" NOTE: your inline snippet was {code_len} chars, which is substantial code. "
                "Persist it as a .py artifact and use tool_execute_python_file next time."
            )
        workspace_contract += doctrine_note
    else:
        workspace_contract += (
            " To run a short, punctual inline snippet instead, use tool_execute_python_code."
        )

    out_str = out_str + workspace_contract

    if len(out_str) > _PREVIEW_WINDOW_CHARS:
        log_name = _persist_full_output(out_str, script_label)
        if log_name:
            stripped_chars = len(out_str) - _PREVIEW_WINDOW_CHARS
            out_str = (
                f"{out_str[:_PREVIEW_HALF_CHARS]}"
                f"\n... [stripped for brevity — {stripped_chars} middle characters omitted] ...\n"
                f"[FULL OUTPUT SAVED] The complete output was persisted to '{log_name}' in the workspace. "
                "Use read file tools (e.g. unlock the .log file, or run a short Python snippet that reads "
                f"and prints a slice of '{log_name}') to inspect the omitted middle.\n"
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
    The orchestrator (ChatMixin) chdirs into the sandboxed workspace before
    invoking this tool, so the process CWD is authoritative. We only fall back
    to ./data_workspace when running completely standalone (e.g., manual tests).
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

    OUTPUT WINDOWING: when stdout exceeds the inline preview window, the FULL
    output is saved to a .log file in the workspace and the result contains a
    head/tail windowed preview (beginning + end) plus a pointer to the .log
    file. Use that pointer with read file tools to inspect the omitted middle.

    The execution environment automatically provides common aliases:
    - pd (pandas), np (numpy), plt (matplotlib.pyplot)
    - sns (seaborn), sklearn (scikit-learn), scipy
    If any of these libraries are missing, they will be automatically installed.

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

    WORKFLOW (mandatory for substantial code): FIRST emit an <artifact type="code">
    tool to create the .py file, THEN call this tool with its file name. Persisting
    scripts makes them inspectable, patchable via SEARCH/REPLACE, and reusable.
    'tool_execute_python_code' is strictly reserved for short, punctual snippets.

    OUTPUT WINDOWING: when stdout exceeds the inline preview window, the FULL
    output is saved to a .log file in the workspace and the result contains a
    head/tail windowed preview (beginning + end) plus a pointer to the .log
    file. Use that pointer with read file tools to inspect the omitted middle.

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