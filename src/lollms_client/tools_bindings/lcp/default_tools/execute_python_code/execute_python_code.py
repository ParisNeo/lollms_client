import os
import sys
import io
import uuid
import base64
from pathlib import Path
from typing import Any, Dict
from ascii_colors import ASCIIColors

import matplotlib
matplotlib.use('Agg')

TOOL_LIBRARY_NAME = "Execute Python Code"
TOOL_LIBRARY_DESC = "Executes arbitrary sandboxed Python code and returns stdout, stderr, and generated plots."
TOOL_LIBRARY_ICON = "🐍"

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl"])

def tool_execute_python_code(
    code: str = ""
) -> Dict[str, Any]:
    """
    Execute arbitrary sandboxed Python code directly from a string.
    Useful for quick calculations, data transformations, or ad-hoc scripting.
    
    The code runs in the current workspace directory. Any plots generated via matplotlib
    are automatically saved to disk and registered as image artifacts.

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
            "error": "No code provided for execution."
        }

    import numpy as np
    import matplotlib.pyplot as plt

    local_vars = {
        "Path": Path,
        "pd": __import__("pandas"),
        "np": np,
        "plt": plt,
        "__builtins__": __builtins__
    }

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    try:
        ASCIIColors.info(f"⚡ Executing arbitrary Python code (CWD: {os.getcwd()})")
        plt.clf()
        plt.close('all')

        exec(code, local_vars)

        fig_nums = plt.get_fignums()
        if fig_nums:
            ASCIIColors.success(f"[Sandbox] Intercepted {len(fig_nums)} generated plot figure(s)!")
            for idx, f_num in enumerate(fig_nums):
                buf = io.BytesIO()
                fig = plt.figure(f_num)
                fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                plot_filename = f"code_exec_plot_{uuid.uuid4().hex[:6]}.png"
                plot_path = Path(".") / plot_filename
                fig.savefig(str(plot_path), bbox_inches='tight', facecolor=fig.get_facecolor())

            plt.close('all')

    except Exception as e:
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
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    out_str = redirected_output.getvalue()
    err_str = redirected_error.getvalue()

    return {
        "success": True,
        "output": out_str or "Code executed successfully (no stdout prints).",
        "stderr": err_str
    }
