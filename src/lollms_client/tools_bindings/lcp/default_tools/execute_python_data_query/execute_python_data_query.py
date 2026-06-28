import os
import sys
import io
from pathlib import Path
from typing import Any, Dict, Optional 
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python Data Query"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code to analyze or modify datasets in the workspace."
TOOL_LIBRARY_ICON = "📊"

def init_tool_library() -> None:
    import pipmaster as pm
    pm.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl"])

def tool_execute_python_data_query(
    code: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute sandboxed Python code to analyze or modify datasets in the workspace.

    The code can reference any file in the workspace using simple relative paths.
    For example:
      - df = pd.read_csv("data.csv")
      - with open("circuit.cir") as f: content = f.read()

    Args:
        code (str, optional): The Python code to execute, or the name of a Python file in the workspace. Defaults to empty.
        discussion_instance (Any, optional): Active discussion session instance. Defaults to None.
        lollms_client_instance (Any, optional): Active client instance. Defaults to None.
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

    # Set up headless matplotlib and standard modules inside execution context
    import base64
    import numpy as np
    import matplotlib
    matplotlib.use('Agg') # Safe headless execution
    import matplotlib.pyplot as plt

    local_vars = {
        "Path": Path, 
        "pd": __import__("pandas"),
        "np": np,
        "plt": plt,
        "__builtins__": __builtins__
    }

    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    try:
        ASCIIColors.info(f"⚡ Executing Python code (CWD: {os.getcwd()})")
        plt.clf()
        plt.close('all')

        exec(code, local_vars)

        # ── INTERCEPT MATPLOTLIB PLOTS ON THE FLY ──
        fig_nums = plt.get_fignums()
        if fig_nums and discussion_instance:
            ASCIIColors.success(f"[Sandbox] Intercepted {len(fig_nums)} generated plot figure(s)!")
            for idx, f_num in enumerate(fig_nums):
                buf = io.BytesIO()
                fig = plt.figure(f_num)
                fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                # Register the plot image as a first-class active workspace artifact
                plot_filename = f"custom_analysis_plot_{uuid.uuid4().hex[:6]}.png"
                art = discussion_instance.artefacts.add(
                    title=plot_filename,
                    artefact_type="image",
                    content=f"### Custom Analysis Plot: {plot_filename}\n\n<artefact_image id=\"{plot_filename}::0\" />",
                    images=[plot_b64],
                    image_media_types=["image/png"],
                    active=True
                )
                # Cache in discussion instance for post-turn visual injection
                if hasattr(discussion_instance, "_affected_artefacts_this_turn"):
                    discussion_instance._affected_artefacts_this_turn.append(art)

            plt.close('all')

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
