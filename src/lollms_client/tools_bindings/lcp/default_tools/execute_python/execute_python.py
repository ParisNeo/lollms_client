"""
execute_python.py
=================
LCP toolset for sandboxed Python execution. Two single-purpose tools:
  - tool_execute_python_code:   executes INLINE Python source (the 'code' string).
                                Nothing is saved to disk by the tool itself.
  - tool_execute_python_file:   executes an EXISTING workspace .py file with optional
                                argv-style arguments (sys.argv / argparse compatible).
                                Read-only: never saves, creates, or overwrites files.
Interception of matplotlib figures and stdout/stderr capture.
"""

import os
import sys
import io
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python"
TOOL_LIBRARY_DESC = "Executes arbitrary sandboxed Python code (inline string or existing workspace .py file) and returns stdout, stderr, and generated plots."
TOOL_LIBRARY_ICON = "🐍"


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
        ASCIIColors.info(f"⚡ Executing arbitrary Python code (label: {script_label}, CWD: {os.getcwd()})")
        sibling_note = (
            f"[sandbox] CWD = workspace root ({os.getcwd()}). "
            f"Artifact .py files are importable as siblings: use 'from <module> import ...' directly; "
            f"do NOT prepend 'workspace/' or manipulate sys.path."
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
                "error": f"Execution Error:\n{raw_traceback}",
                "output": raw_output,
                "stderr": raw_error
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
            "error": f"Unexpected execution failure:\n{raw_traceback}",
            "output": raw_output,
            "stderr": raw_error
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
        "(e.g. 'from rlc_filter import RLCFilter') without path prefixes or sys.path manipulation. "
        "To execute an existing script as a program, use tool_execute_python_file instead."
    )
    out_str = out_str + workspace_contract

    return {
        "success": True,
        "output": out_str,
        "stderr": err_str
    }


def _get_workspace_root() -> Path:
    """Resolves the workspace root: honors orchestrator CWD, falls back to ./data_workspace."""
    cwd = Path.cwd()
    if (cwd / "data_workspace").exists() or cwd.name == "data_workspace":
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


def tool_execute_python_code(code: str = "") -> Dict[str, Any]:
    """
    Executes inline Python code and returns stdout, stderr, and generated matplotlib plots.

    Runs the raw Python source provided in the 'code' parameter inside a sandboxed
    workspace. Nothing is written to disk unless your code explicitly saves files.
    This tool does NOT execute files and never saves the code to a file; to run an
    existing workspace .py script, use 'tool_execute_python_file' instead.

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
    Executes an existing Python file from the workspace and returns stdout, stderr, and generated plots.

    Reads the .py file that ALREADY EXISTS on disk in the sandboxed workspace and runs it.
    The script is invoked exactly like `python <file_name> <arg1> <arg2> ...`: arguments are
    exposed via sys.argv[1:] or argparse. This tool ONLY READS existing files — it never
    saves, creates, or overwrites files. To write a new script to disk, emit an
    <artifact type="code"> tag first, then call this tool with its file name. To run
    inline code without saving anything, use 'tool_execute_python_code' instead.

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