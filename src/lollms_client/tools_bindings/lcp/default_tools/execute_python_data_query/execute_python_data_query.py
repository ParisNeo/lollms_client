import os
import sys
import io
import re
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python Data Query"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code to analyze or modify datasets in the workspace."
TOOL_LIBRARY_ICON = "📊"

def init_tools_library(config: dict = None) -> None:
    try:
        import pipmaster as pm
        pm.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl", "sqlalchemy"])
        global matplotlib
        import matplotlib
        matplotlib.use('Agg')
    except Exception as e:
        import ascii_colors
        ascii_colors.ASCIIColors.warning(f"[execute_python_data_query] Failed to ensure all dependencies: {e}")

def _lazy_import_libs():
    _np = None
    _pd = None
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
        import pandas as _pd_mod
        _pd = _pd_mod
    except Exception:
        ASCIIColors.warning("[execute_python_data_query] pandas unavailable — df auto-load and CSV persistence disabled.")
    return _np, _pd, _plt

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
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python_data_query] Unwrapping nested dictionary parameter.")
        code = code.get("code") or code.get("sql_query") or next((v for v in code.values() if isinstance(v, str)), "")

    code = str(code).strip()

    if not code:
        return {
            "success": False,
            "error": "No code provided for execution.",
            "output": ""
        }

    if code.endswith(".py") and Path(code).exists():
        filename = code
        code = Path(code).read_text(encoding="utf-8")
        ASCIIColors.info(f"[execute_python_data_query] Loaded code from file: {filename}")

    import base64

    _np, _pd, _plt = _lazy_import_libs()

    class _NoReconfigureStringIO(io.StringIO):
        """StringIO that silently ignores reconfigure() calls from user code."""
        def reconfigure(self, *args, **kwargs):
            pass

    local_vars = {
        "Path": Path,
        "pd": _pd,
        "np": _np,
        "plt": _plt,
        "__builtins__": __builtins__,
    }

    uses_df = re.search(r'\bdf\b', code) is not None
    defines_df = re.search(r'\bdf\s*=\s*', code) is not None
    auto_loaded_df_file = None
    auto_loaded_sep = ","

    if uses_df and not defines_df and _pd is not None:
        data_files = list(Path(".").glob("*.csv")) + list(Path(".").glob("*.xlsx"))
        if data_files:
            first_file = data_files[0]
            auto_loaded_df_file = first_file
            ASCIIColors.info(f"[execute_python_data_query] Auto-loading df from {first_file.name}")
            try:
                if first_file.suffix.lower() == ".csv":
                    first_line = first_file.read_text(encoding="utf-8", errors="ignore").splitlines()[0] if first_file.stat().st_size > 0 else ""
                    auto_loaded_sep = ";" if ";" in first_line else ","
                    local_vars["df"] = _pd.read_csv(first_file, sep=auto_loaded_sep, encoding="utf-8-sig")
                elif first_file.suffix.lower() in (".xlsx", ".xls"):
                    local_vars["df"] = _pd.read_excel(first_file)
            except Exception as load_err:
                ASCIIColors.warning(f"[execute_python_data_query] Failed to auto-load df: {load_err}")
    elif uses_df and not defines_df and _pd is None:
        ASCIIColors.warning("[execute_python_data_query] Code references 'df' but pandas is unavailable. User must load data manually.")

    uses_conn = re.search(r'\bconn\b', code) is not None
    defines_conn = re.search(r'\bconn\s*=\s*', code) is not None
    auto_loaded_engine = None

    if uses_conn and not defines_conn:
        sqlconn_files = list(Path(".").glob("*.sqlconn"))
        if sqlconn_files:
            first_sqlconn = sqlconn_files[0]
            ASCIIColors.info(f"[execute_python_data_query] Auto-loading conn from {first_sqlconn.name}")
            try:
                from sqlalchemy import create_engine
                conn_info = json.loads(first_sqlconn.read_text(encoding="utf-8"))
                dialect = conn_info.get("dialect", "sqlite").lower()
                connection_url = conn_info.get("url", "")
                if not connection_url:
                    if dialect == "sqlite":
                        db_path = conn_info.get('database', '')
                        db_path = db_path.replace("\\", "/")
                        connection_url = f"sqlite:///{db_path}"
                    else:
                        host = conn_info.get("host", "localhost")
                        port = conn_info.get("port", "")
                        username = conn_info.get("username", "")
                        password = conn_info.get("password", "")
                        database = conn_info.get("database", "")
                        port_str = f":{port}" if port else ""
                        if dialect == "mysql":
                            connection_url = f"mysql+pymysql://{username}:{password}@{host}{port_str}/{database}"
                        elif dialect == "postgresql":
                            connection_url = f"postgresql+psycopg2://{username}:{password}@{host}{port_str}/{database}"

                auto_loaded_engine = create_engine(connection_url)
                local_vars["conn"] = auto_loaded_engine.raw_connection()
            except Exception as load_err:
                ASCIIColors.warning(f"[execute_python_data_query] Failed to auto-load conn: {load_err}")

    old_stdout = sys.stdout
    redirected_output = _NoReconfigureStringIO()
    sys.stdout = redirected_output

    try:
        ASCIIColors.info(f"⚡ Executing Python code (CWD: {os.getcwd()})")
        if _plt is not None:
            _plt.clf()
            _plt.close('all')

        try:
            import builtins
            native_compile = getattr(builtins, 'compile', None)
            if native_compile and native_compile.__module__ == 'builtins':
                compiled_code = native_compile(code, "<execute_python_data_query>", "exec")
                exec(compiled_code, local_vars)
            else:
                exec(code, local_vars)
        except SystemExit as se:
            raise RuntimeError(f"User code called sys.exit() with code {se.code}. This is not permitted in sandboxed execution.") from se
        except KeyboardInterrupt:
            raise RuntimeError("Execution interrupted by KeyboardInterrupt.")
        except Exception:
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

        if auto_loaded_df_file and "df" in local_vars and _pd is not None and isinstance(local_vars["df"], _pd.DataFrame):
            try:
                ASCIIColors.info(f"[execute_python_data_query] Persisting modified df back to {auto_loaded_df_file.name}")
                if auto_loaded_df_file.suffix.lower() == ".csv":
                    local_vars["df"].to_csv(auto_loaded_df_file, sep=auto_loaded_sep, index=False, encoding="utf-8-sig")
                elif auto_loaded_df_file.suffix.lower() in (".xlsx", ".xls"):
                    local_vars["df"].to_excel(auto_loaded_df_file, index=False)

                if discussion_instance and hasattr(discussion_instance, "artefacts"):
                    art_title = auto_loaded_df_file.name
                    existing_art = discussion_instance.artefacts.get(art_title)
                    if not existing_art and auto_loaded_df_file.stem != auto_loaded_df_file.name:
                        existing_art = discussion_instance.artefacts.get(auto_loaded_df_file.stem)
                        if existing_art:
                            art_title = auto_loaded_df_file.stem

                    if existing_art:
                        discussion_instance.artefacts.update(
                            title=art_title,
                            new_content=local_vars["df"].to_csv(index=False, sep=auto_loaded_sep if auto_loaded_df_file.suffix.lower() == ".csv" else ","),
                            bump_version=True,
                            active=True
                        )
                        discussion_instance.commit()
            except Exception as save_err:
                ASCIIColors.warning(f"[execute_python_data_query] Failed to persist df: {save_err}")

        fig_nums = _plt.get_fignums() if _plt is not None else []
        if fig_nums:
            ASCIIColors.success(f"[Sandbox] Intercepted {len(fig_nums)} generated plot figure(s)!")
            for idx, f_num in enumerate(fig_nums):
                buf = io.BytesIO()
                fig = _plt.figure(f_num)
                fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                plot_filename = f"custom_analysis_plot_{uuid.uuid4().hex[:6]}.png"
                plot_path = Path(".") / plot_filename
                fig.savefig(str(plot_path), bbox_inches='tight', facecolor=fig.get_facecolor())

                if discussion_instance and hasattr(discussion_instance, "artefacts"):
                    try:
                        art_title = plot_filename
                        comp_title = f"{art_title}::images"
                        existing_comp = discussion_instance.artefacts.get(comp_title)

                        if existing_comp:
                            existing_imgs = existing_comp.get("images", [])
                            existing_mtypes = existing_comp.get("image_media_types", [])
                            existing_imgs.append(plot_b64)
                            existing_mtypes.append("image/png")
                            discussion_instance.artefacts.update(
                                title=comp_title,
                                new_images=existing_imgs,
                                new_image_media_types=existing_mtypes,
                                active=True
                            )
                        else:
                            discussion_instance.artefacts.add(
                                title=art_title,
                                artefact_type="image",
                                content=f"![{art_title}]({art_title})",
                                images=[plot_b64],
                                image_media_types=["image/png"],
                                active=True,
                                file_ext=".png"
                            )
                        discussion_instance.commit()
                        ASCIIColors.info(f"[execute_python_data_query] Synced plot '{art_title}' to artefact system.")
                    except Exception as sync_err:
                        ASCIIColors.warning(f"[execute_python_data_query] Failed to sync plot to artefact system: {sync_err}")

            _plt.close('all')

    except BaseException as outer_err:
        import traceback
        sys.stdout = old_stdout
        raw_output = redirected_output.getvalue()
        raw_traceback = traceback.format_exc()
        ASCIIColors.error(f"❌ Unexpected execution failure:\n{raw_traceback}")
        return {
            "success": False,
            "error": f"Unexpected execution failure:\n{raw_traceback}",
            "output": raw_output
        }
    finally:
        sys.stdout = old_stdout
        if auto_loaded_engine is not None:
            try:
                if "conn" in local_vars and hasattr(local_vars["conn"], "close"):
                    local_vars["conn"].close()
                auto_loaded_engine.dispose()
            except Exception:
                pass

    out_str = redirected_output.getvalue()

    if not out_str.strip():
        out_str = "Code executed successfully (no stdout prints)."

    return {
        "success": True,
        "output": out_str
    }