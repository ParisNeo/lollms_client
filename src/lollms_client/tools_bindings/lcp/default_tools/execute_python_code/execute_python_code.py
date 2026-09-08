"""
execute_python_code.py
======================
LCP toolset for sandboxed Python execution. Supports:
- Arbitrary inline code execution (string payload).
- Running a workspace file with command-line style arguments (sys.argv).
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

TOOL_LIBRARY_NAME = "Execute Python Code"
TOOL_LIBRARY_DESC = "Executes arbitrary sandboxed Python code and returns stdout, stderr, and generated plots. Can also run a workspace .py file with arguments via sys.argv."
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
        ascii_colors.ASCIIColors.warning(f"[execute_python_code] Failed to ensure dependencies: {e}")


def _ensure_import(module_name: str, package_name: str = None):
    try:
        return __import__(module_name)
    except ImportError:
        import pipmaster as pm
        pkg = package_name or module_name
        ASCIIColors.warning(f"[execute_python_code] Missing dependency '{pkg}'. Installing automatically...")
        try:
            pm.ensure_packages(pkg)
            return __import__(module_name)
        except Exception as install_err:
            ASCIIColors.error(f"[execute_python_code] Failed to auto-install '{pkg}': {install_err}")
            return None


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


def _unify_code_payload(
    code: str,
    file_name: str,
    args: Optional[List[Any]],
) -> Dict[str, Any]:
    """
    Resolves the final execution payload:
    - If file_name is provided: reads the workspace .py file (verified safe & existing).
      Execution mode is 'file' with sys.argv = [file_name, *args].
    - Else: uses the inline code string. Execution mode is 'inline'.
    """
    if file_name:
        resolved = _resolve_workspace_path(file_name)
        if resolved is None:
            return {
                "success": False,
                "error": f"Invalid or unsafe file path: '{file_name}'. Path traversal is blocked.",
                "output": "",
                "stderr": ""
            }
        if not resolved.exists():
            return {
                "success": False,
                "error": f"File '{file_name}' not found in workspace.",
                "output": "",
                "stderr": ""
            }
        if resolved.suffix.lower() != ".py":
            return {
                "success": False,
                "error": f"File '{file_name}' is not a Python file (only .py files can be executed).",
                "output": "",
                "stderr": ""
            }
        try:
            source = resolved.read_text(encoding="utf-8")
        except Exception as read_err:
            return {
                "success": False,
                "error": f"Failed to read file '{file_name}': {read_err}",
                "output": "",
                "stderr": ""
            }
        return {"success": True, "mode": "file", "source": source, "script_label": file_name}

    normalized_code = str(code).strip()
    if not normalized_code:
        return {
            "success": False,
            "error": "No code provided for execution.",
            "output": "",
            "stderr": ""
        }
    return {"success": True, "mode": "inline", "source": normalized_code, "script_label": "python_code"}


def tool_execute_python_code(
    code: str = "",
    file_name: str = "",
    args: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Execute arbitrary sandboxed Python code or a workspace Python file with arguments.

    Two mutually exclusive modes:
    1. **Inline mode**: pass `code` (raw Python string). `args` is ignored for inline execution.
    2. **File mode**: pass `file_name` (a .py file in the workspace) and optionally `args`,
       a list of command-line style arguments. The script is executed exactly as if invoked
       via `python <file_name> <arg1> <arg2> ...`: it can read them through `sys.argv[1:]`
       or `argparse`.

    The execution environment automatically provides common aliases:
    - pd (pandas), np (numpy), plt (matplotlib.pyplot)
    - sns (seaborn), sklearn (scikit-learn), scipy
    If any of these libraries are missing, they will be automatically installed.

    Args:
        code (str): The raw Python code string to execute (inline mode).
        file_name (str): Name of a .py file in the workspace to execute (file mode).
        args (list): Command-line style arguments to expose to the script via sys.argv (file mode).
    """
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python_code] Unwrapping nested dictionary parameter.")
        code = code.get("code") or next((v for v in code.values() if isinstance(v, str)), "")

    payload = _unify_code_payload(code, file_name, args)
    if not payload.get("success"):
        return payload

    source = payload["source"]
    mode = payload["mode"]
    script_label = payload["script_label"]

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
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_argv = sys.argv[:]
    redirected_output = _NoReconfigureStringIO()
    redirected_error = _NoReconfigureStringIO()

    sys.stdout = redirected_output
    sys.stderr = redirected_error

    if mode == "file":
        sys.argv = _normalize_argv(script_label, args)

    try:
        ASCIIColors.info(f"⚡ Executing arbitrary Python code (mode: {mode}, CWD: {os.getcwd()})")
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
                plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

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

    return {
        "success": True,
        "output": out_str,
        "stderr": err_str
    }