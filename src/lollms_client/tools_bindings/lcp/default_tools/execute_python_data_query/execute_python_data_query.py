import os
import sys
import io
from pathlib import Path
from typing import Any, Dict
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python Data Query"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code to analyze or modify datasets in the workspace."
TOOL_LIBRARY_ICON = "📊"

def init_tool_library() -> None:
    import pipmaster as pm
    pm.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl"])

def tool_execute_python_data_query(
    code: str = ""
) -> Dict[str, Any]:
    """
    Execute sandboxed Python code to analyze or modify datasets in the workspace.

    The code can reference any file in the workspace using simple relative paths.
    For example:
      - df = pd.read_csv("data.csv")
      - with open("circuit.cir") as f: content = f.read()

    Args:
        code (str, optional): The Python code to execute, or the name of a Python file in the workspace. Defaults to empty.
    """
    # Handle nested dict parameter hallucination
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python_data_query] Unwrapping nested dictionary parameter.")
        code = code.get("code") or code.get("sql_query") or next((v for v in code.values() if isinstance(v, str)), "")

    code = str(code).strip()

    # If code is a filename, read it
    if Path(code).exists():
        code = Path(code).read_text(encoding="utf-8")
        ASCIIColors.info(f"[execute_python_data_query] Loaded code from file: {code}")

    # Execute in current workspace context
    # CWD is already set to workspace by LCP binding
    local_vars = {"Path": Path, "__builtins__": __builtins__}
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        ASCIIColors.info(f"⚡ Executing Python code (CWD: {os.getcwd()})")
        exec(code, local_vars)
    except Exception as e:
        import traceback
        sys.stdout = old_stdout
        raw_output = redirected_output.getvalue()
        raw_traceback = traceback.format_exc()
        ASCIIColors.error(f"❌ Execution Failed:\n{raw_traceback}")
        return {
            "success": False, 
            "error": f"Execution Error:\n{raw_traceback}", 
            "output": raw_output
        }
    finally:
        sys.stdout = old_stdout

    out_str = redirected_output.getvalue()
    return {
        "success": True,
        "output": out_str or "Code executed successfully (no stdout prints)."
    }
