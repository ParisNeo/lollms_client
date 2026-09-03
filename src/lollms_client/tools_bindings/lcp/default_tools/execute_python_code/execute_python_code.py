import os
import sys
import io
import uuid
import base64
from pathlib import Path
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python Code"
TOOL_LIBRARY_DESC = "Executes arbitrary sandboxed Python code and returns stdout, stderr, and generated plots."
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

def tool_execute_python_code(
    code: str = ""
) -> Dict[str, Any]:
    """
    Execute arbitrary sandboxed Python code directly from a string.
    Useful for quick calculations, data transformations, or ad-hoc scripting.
    
    The execution environment automatically provides common aliases:
    - pd (pandas), np (numpy), plt (matplotlib.pyplot)
    - sns (seaborn), sklearn (scikit-learn), scipy
    If any of these libraries are missing, they will be automatically installed.

    Args:
        code (str): The raw Python code string to execute.
    """
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python_code] Unwrapping nested dictionary parameter.")
        code = code.get("code") or next((v for v in code.values() if isinstance(v, str)), "")

    code = str(code).strip()
    if not code:
        return {
            "success": False,
            "error": "No code provided for execution.",
            "output": "",
            "stderr": ""
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
    redirected_output = _NoReconfigureStringIO()
    redirected_error = _NoReconfigureStringIO()
    
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    try:
        ASCIIColors.info(f"⚡ Executing arbitrary Python code (CWD: {os.getcwd()})")
        if _plt is not None:
            _plt.clf()
            _plt.close('all')

        try:
            exec(code, local_vars)
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

    out_str = redirected_output.getvalue()
    err_str = redirected_error.getvalue()

    if not out_str.strip():
        out_str = "Code executed successfully (no stdout prints)."

    return {
        "success": True,
        "output": out_str,
        "stderr": err_str
    }