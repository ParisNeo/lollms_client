import json
import os
import sys
import io
import shutil
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Union
from ascii_colors import ASCIIColors, trace_exception

TOOL_LIBRARY_NAME = "Execute Python Data Query"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code to analyze or modify active datasets."
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
    MANDATORY FOR DATA ANALYSIS/MODIFICATION: Execute sandboxed Python code using pandas or sqlite3 to analyze active datasets.
    
    To completely avoid JSON escaping errors (e.g., newlines or quotes issues), you can write your Python code inside a separate code artifact first, e.g., using:
      <artifact name="query.py" type="code" language="python">...</artifact>
    And then simply pass the artifact's exact name as the 'code' parameter, e.g.:
      {"code": "query.py"}
    The engine will automatically resolve and execute the code from that artifact!

    Args:
        code (str, optional): The Python code to execute, or the name of an active code artifact containing the code (e.g. 'query.py'). Defaults to empty.
    """
    if not discussion_instance:
        return {"success": False, "error": "No discussion instance provided."}

    # ── 0. Defensive Parameter Unwrapping Guard ──
    if isinstance(code, dict):
        ASCIIColors.warning("[execute_python_data_query] Unwrapping nested dictionary parameter hallucination.")
        code = code.get("code") or code.get("sql_query") or next((v for v in code.values() if isinstance(v, str)), "")

    code = str(code).strip()
    sep = ","  # Define early for all blocks

    # ── 0.5. Subconscious Artifact Resolver (Lazy Call Safeguard) ──
    if not code:
        # Scan for the most recently updated Python/code artifact in the session
        code_arts = [
            a for a in discussion_instance.artefacts.list()
            if a.get("type") in ("code", "document") and (a.get("title", "").endswith(".py") or "query" in a.get("title", "").lower())
        ]
        if code_arts:
            code_arts.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            code = code_arts[0]["title"]
            ASCIIColors.success(f"[execute_python_data_query] Parameter 'code' was omitted. Auto-resolved to latest session artifact: '{code}'")
        else:
            return {"success": False, "error": "No Python code provided and no active code artifacts found to resolve."}

    # ── 1. Resolve Code from Artifacts (Dynamic Resolution Paradigm) ──
    input_query_stripped = code.strip()
    artifact = discussion_instance.artefacts.get(input_query_stripped)
    if not artifact:
        # Support reference markers like @query.py or artifact:query.py
        clean_title = re.sub(r'^[@#\s]+|[\s]+$', '', input_query_stripped)
        clean_title = clean_title.replace("artifact:", "").strip()
        artifact = discussion_instance.artefacts.get(clean_title)

    if artifact:
        ASCIIColors.success(f"[execute_python_data_query] Resolving and loading code from active artifact: '{artifact['title']}'")
        art_content = artifact.get("content", "").strip()
        
        # Robust local extractor to completely avoid import dependencies
        match_py = re.search(r'```python\s*\n([\s\S]*?)\n```', art_content, re.I)
        if match_py:
            code = match_py.group(1).strip()
        else:
            code = re.sub(r'^```python\s*|\s*```$', '', art_content, flags=re.I).strip()

    latest_data = [
        a for a in discussion_instance.artefacts.list(active_only=True) 
        if a.get("type") == "data" or any(a.get("title", "").endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"))
    ]
    if not latest_data:
        err_msg = "No active data artifact found in this session."
        ASCIIColors.error(f"❌ {err_msg}")
        return {"success": False, "error": err_msg}

    title = latest_data[0]["title"]
    ext = latest_data[0].get("file_ext") or Path(title).suffix or ".csv"
    current_version = latest_data[0].get("version", 1)
    is_read_only = latest_data[0].get("read_only", False)
    plot_b64 = None

    workspace_dir = Path("./data_workspace")
    try:
        from lollms_client.app.server import get_discussion_workspace
        workspace_dir = get_discussion_workspace()
    except ImportError:
        pass

    workspace_dir.mkdir(exist_ok=True)

    if is_read_only:
        new_version = current_version
        source_file_path = workspace_dir / f"{title}{ext}"
        new_file_path = source_file_path
    else:
        new_version = current_version + 1
        source_file_path = workspace_dir / "versions" / f"{title}_v{current_version}{ext}"
        new_file_path = workspace_dir / "versions" / f"{title}_v{new_version}{ext}"

    ASCIIColors.info(f"--- [execute_python_data_query] Ingestion started for: '{title}' (v{current_version}{ext}) ---")

    if not source_file_path.exists():
        found_path = None
        scan_dirs = [Path("./data_workspace")]
        try:
            from lollms_client.app.server import APP_DIR
            if APP_DIR and APP_DIR.exists():
                ws_dir = APP_DIR / "workspaces"
                if ws_dir.exists():
                    for d in ws_dir.iterdir():
                        if d.is_dir():
                            scan_dirs.append(d / "data_workspace")
        except Exception:
            pass

        for sd in scan_dirs:
            cand = sd / f"{title}_v{current_version}{ext}"
            if cand.exists():
                found_path = cand
                break
            cand = sd / f"{title}{ext}"
            if cand.exists():
                found_path = cand
                break

        if found_path:
            try:
                workspace_dir.mkdir(parents=True, exist_ok=True)
                if found_path.resolve() != source_file_path.resolve():
                    shutil.copy(str(found_path), str(source_file_path))
                unversioned_dest = workspace_dir / f"{title}{ext}"
                try:
                    if found_path.resolve() != unversioned_dest.resolve():
                        shutil.copy(str(found_path), str(unversioned_dest))
                except Exception:
                    pass
                ASCIIColors.success(f"✓ Recovered missing data file from '{found_path.parent.parent.name or 'global'}' workspace!")
            except Exception as copy_err:
                ASCIIColors.error(f"Failed to copy recovered file: {copy_err}")

    if not source_file_path.exists():
        # Fallback: check if the data exists inside the SQLite artifact version record first
        # before creating a dummy fallback dataset!
        db_recovered = False
        if discussion_instance:
            art_rec = discussion_instance.artefacts.get(title)
            if art_rec and art_rec.get("content"):
                try:
                    workspace_dir.mkdir(parents=True, exist_ok=True)
                    source_file_path.parent.mkdir(parents=True, exist_ok=True)
                    source_file_path.write_text(art_rec["content"], encoding="utf-8")

                    unversioned_dest = workspace_dir / f"{title}{ext}"
                    unversioned_dest.write_text(art_rec["content"], encoding="utf-8")

                    db_recovered = True
                    ASCIIColors.success(f"✓ [Self-Healing Sandbox] Successfully recovered missing '{source_file_path.name}' from SQLite database!")
                except Exception as r_err:
                    ASCIIColors.warning(f"Database recovery failed: {r_err}")

        if not db_recovered:
            # Strictly let the file opening operation fail naturally to preserve data integrity and prevent any invented content
            err_msg = f"FileNotFoundError: The physical data file '{source_file_path.name}' is missing and has no saved record in the database."
            ASCIIColors.error(f"❌ {err_msg}")
            return {"success": False, "error": err_msg}

    if not source_file_path.exists():
        err_msg = f"Raw data file '{title}_v{current_version}{ext}' is missing from workspace."
        ASCIIColors.error(f"❌ {err_msg}")
        return {"success": False, "error": err_msg}

    import pandas as pd
    import numpy as np
    import base64
    import io
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Robust pandas read_csv wrapper ──
    _original_read_csv = pd.read_csv

    def _robust_read_csv(filepath_or_buffer, *args, **kwargs):
        f_path = Path(filepath_or_buffer)
        if not f_path.is_absolute():
            f_path = workspace_dir / f_path

        if f_path.exists() and f_path.is_file():
            try:
                text_content = f_path.read_text(encoding="utf-8", errors="ignore")
                first_line = text_content.splitlines()[0] if text_content.splitlines() else ""
                if "sep" not in kwargs and "delimiter" not in kwargs:
                    kwargs["sep"] = ";" if ";" in first_line and "," not in first_line else ","
                if "encoding" not in kwargs:
                    kwargs["encoding"] = "utf-8-sig"
            except Exception:
                pass
        return _original_read_csv(filepath_or_buffer, *args, **kwargs)

    pd.read_csv = _robust_read_csv

    local_vars = {
        "pd": pd,
        "plt": plt,
        "np": np,
        "Path": Path
    }

    try:
        if ext == ".sqlconn":
            from lollms_client.lollms_discussion._data_files import _get_sqlalchemy_engine_from_file
            engine, dialect = _get_sqlalchemy_engine_from_file(source_file_path)
            local_vars["engine"] = engine
            local_vars["dialect"] = dialect
            # Also provide a raw connection for compatibility and robust pandas execution
            conn = engine.raw_connection()
            local_vars["conn"] = conn
            local_vars["cursor"] = conn
        elif ext in (".db", ".sqlite", ".sqlite3"):
            if source_file_path.resolve() != new_file_path.resolve():
                shutil.copy(str(source_file_path), str(new_file_path))
            conn = sqlite3.connect(str(new_file_path))
            local_vars["conn"] = conn
            local_vars["cursor"] = conn.cursor()
        elif ext in (".xlsx", ".xls"):
            xl = pd.ExcelFile(str(source_file_path))
            dfs = {sheet: pd.read_excel(str(source_file_path), sheet_name=sheet) for sheet in xl.sheet_names}
            local_vars["dfs"] = dfs
            if len(dfs) == 1:
                local_vars["df"] = list(dfs.values())[0]
        else:
            sep = ";" if ext == ".csv" and ";" in source_file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0] else ","
            local_vars["df"] = pd.read_csv(str(source_file_path), sep=sep)
        ASCIIColors.success(f"✓ Dataset loaded into memory successfully.")
    except Exception as e:
        ASCIIColors.error(f"❌ Failed to load dataset: {e}")
        return {"success": False, "error": f"Failed to load dataset: {e}"}

    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    old_cwd = os.getcwd()

    active_file = workspace_dir / f"{title}{ext}"
    active_file_mtime_before = active_file.stat().st_mtime if active_file.exists() else 0

    # Pre-execution Path Sanitization: Convert accidental absolute paths to clean relative ones
    code_sanitized = re.sub(r'["\']/?(?:workspace|data_workspace)/([^"\']+)["\']', r"'\1'", code)

    def sanitize_output_paths(text: str, ws_path: Path) -> str:
        if not text:
            return ""

        # Standardize slashes to avoid OS-specific leaks
        text_clean = text.replace("\\", "/")

        # Mask the workspace directory absolute path
        abs_path_str = str(ws_path.resolve()).replace("\\", "/")
        text_clean = text_clean.replace(abs_path_str, "./")

        # Mask any user directory leaks (e.g. C:/Users/name -> ~)
        try:
            user_home_str = str(Path.home().resolve()).replace("\\", "/")
            text_clean = text_clean.replace(user_home_str, "~")

            # Mask potential parent C:/Users/ or /home/ references
            text_clean = re.sub(r'(?:[a-zA-Z]:/Users|/home)/[^/]+', '~', text_clean, flags=re.I)
        except Exception:
            pass

        # Normalize virtual workspace references
        text_clean = text_clean.replace("/workspace/", "./")
        return text_clean

    # ── 🛡️ SECURE EXECUTION SANDBOX (RCE & Path Traversal Prevention) ──
    def _is_safe_path(target_path: Union[str, Path]) -> bool:
        try:
            resolved_base = workspace_dir.resolve()
            resolved_target = Path(target_path).resolve()
            # Must resolve strictly inside the isolated workspace directory
            return resolved_target.parts[:len(resolved_base.parts)] == resolved_base.parts
        except Exception:
            return False

    # Safe restricted open wrapper
    _original_open = open
    def _restricted_open(file, mode='r', *args, **kwargs):
        resolved_file = Path(file)
        if not resolved_file.is_absolute():
            resolved_file = workspace_dir / resolved_file

        if not _is_safe_path(resolved_file):
            raise PermissionError(f"Access Denied: Path '{file}' is outside the authorized sandbox folder.")
        return _original_open(resolved_file, mode, *args, **kwargs)

    # Safe restricted import wrapper blocking system commands
    _original_import = __import__
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        forbidden_modules = {
            "os", "sys", "subprocess", "shutil", "socket", "ctypes", "pty", "platform",
            "requests", "urllib", "http", "multiprocessing", "threading"
        }
        if name in forbidden_modules or any(mod in name for mod in forbidden_modules):
            raise ImportError(f"Security Block: Import of dangerous module '{name}' is forbidden in the sandbox.")
        return _original_import(name, globals, locals, fromlist, level)

    # Inject secure builtins overrides in a 100% bulletproof way
    import builtins
    sandbox_builtins = {}
    for name in dir(builtins):
        sandbox_builtins[name] = getattr(builtins, name)

    sandbox_builtins["open"] = _restricted_open
    sandbox_builtins["__import__"] = _restricted_import
    # Disable dangerous builtins completely
    for dangerous_builtin in ["eval", "exec", "compile", "globals", "locals"]:
        sandbox_builtins.pop(dangerous_builtin, None)

    local_vars["__builtins__"] = sandbox_builtins

    # Capture the list of files BEFORE code execution to accurately detect newly written files
    files_before = set(os.listdir(str(workspace_dir))) if workspace_dir.exists() else set()

    try:
        if workspace_dir and workspace_dir.exists():
            os.chdir(str(workspace_dir))

        plt.clf()
        plt.close('all')
        ASCIIColors.info(f"⚡ Executing Sandboxed Python code:\n{code_sanitized}")
        exec(code_sanitized, local_vars)
    except Exception as e:
        import traceback
        sys.stdout = old_stdout
        raw_output = redirected_output.getvalue()

        # Capture the complete traceback details for high-fidelity debugging
        raw_traceback = traceback.format_exc()
        sanitized_tb = sanitize_output_paths(raw_traceback, workspace_dir)
        clean_output = sanitize_output_paths(raw_output, workspace_dir)

        ASCIIColors.error(f"❌ Sandbox Execution Failed:\n{sanitized_tb}")
        return {
            "success": False, 
            "error": f"Execution Error:\n{sanitized_tb}", 
            "output": clean_output
        }

    try:
        files_after = set(os.listdir(str(workspace_dir))) if workspace_dir.exists() else set()
        new_or_modified_files = []
        for f in files_after:
            if f.endswith((".csv", ".xlsx", ".xls")) and not "_v" in f:
                new_or_modified_files.append(f)

        ai_message = discussion_instance.get_message(discussion_instance.active_branch_id)
        if new_or_modified_files and ai_message:
            current_meta = dict(ai_message.metadata or {})
            ui_views = current_meta.get("ui_data_views", [])
            for f in new_or_modified_files:
                if f not in ui_views:
                    ui_views.append(f)
            current_meta["ui_data_views"] = ui_views
            ai_message.metadata = current_meta

        fig_nums = plt.get_fignums()
        if fig_nums:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            plot_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            ASCIIColors.success("✓ Matplotlib visualization generated inside sandbox.")

            plot_title = f"{title}_plot"
            existing_plot = discussion_instance.artefacts.get(plot_title)
            if existing_plot is None:
                discussion_instance.artefacts.add(
                    title=plot_title,
                    artefact_type="image",
                    content=f"### Matplotlib Visualization: {plot_title}\n\n<artefact_image id=\"{plot_title}::0\" />",
                    images=[plot_b64],
                    image_media_types=["image/png"],
                    active=True
                )
            else:
                discussion_instance.artefacts.update(
                    title=plot_title,
                    new_content=f"### Matplotlib Visualization (Version {existing_plot.get('version', 1) + 1}): {plot_title}\n\n<artefact_image id=\"{plot_title}::{existing_plot.get('version', 1)}\" />",
                    new_images=existing_plot.get("images", []) + [plot_b64],
                    new_image_media_types=existing_plot.get("image_media_types", []) + ["image/png"],
                    bump_version=True,
                    active=True
                )
            discussion_instance.commit()
            ASCIIColors.success(f"✓ Registered/updated Matplotlib plot as a persistent image artifact: '{plot_title}'")

            # Append interactive inline image rendering to the assistant response
            img_index = len(existing_plot.get('images', [])) if existing_plot else 0
            image_tag = f'\n\n<artefact_image id="{plot_title}::{img_index}" />\n'
            if ai_message:
                ai_message.content += image_tag

            # Append notice to output
            out_prefix = f"\n\n[SYSTEM: A matplotlib visualization was generated. Use <artefact_image id=\"{plot_title}::{img_index}\" /> to view it.]"
            redirected_output.write(out_prefix)

        # 4.5. New Workspace Images Auto-Ingestion scanner
        files_after = set(os.listdir(str(workspace_dir))) if workspace_dir.exists() else set()
        new_images = [f_name for f_name in (files_after - files_before) if f_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]

        for img_file in new_images:
            try:
                img_path = workspace_dir / img_file
                img_bytes = img_path.read_bytes()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                img_title = img_file.replace(".", "_")

                existing_img = discussion_instance.artefacts.get(img_title)
                if existing_img is None:
                    discussion_instance.artefacts.add(
                        title=img_title,
                        artefact_type="image",
                        content=f"### Image: {img_title}\n\n<artefact_image id=\"{img_title}::0\" />",
                        images=[img_b64],
                        image_media_types=[f"image/{Path(img_path).suffix[1:]}"],
                        active=True
                    )
                else:
                    discussion_instance.artefacts.update(
                        title=img_title,
                        new_content=f"### Image (Version {existing_img.get('version', 1) + 1}): {img_title}\n\n<artefact_image id=\"{img_title}::{existing_img.get('version', 1)}\" />",
                        new_images=existing_img.get("images", []) + [img_b64],
                        new_image_media_types=existing_img.get("image_media_types", []) + [f"image/{Path(img_path).suffix[1:]}"],
                        bump_version=True,
                        active=True
                    )
                discussion_instance.commit()

                # Append interactive inline image rendering to the assistant response
                img_index = len(existing_img.get('images', [])) if existing_img else 0
                image_tag = f'\n\n<artefact_image id="{img_title}::{img_index}" />\n'
                if ai_message:
                    ai_message.content += image_tag

                out_prefix = f"\n\n[SYSTEM: Image '{img_file}' detected. Use <artefact_image id=\"{img_title}::{img_index}\" /> to view it.]"
                redirected_output.write(out_prefix)
                ASCIIColors.success(f"✓ Auto-ingested workspace image: '{img_file}' as artifact '{img_title}'")
            except Exception as img_ing_err:
                ASCIIColors.warning(f"Failed to auto-ingest image '{img_file}': {img_ing_err}")

        # 4.6. New/Modified Datasets Auto-Ingestion scanner
        new_datasets = [f_name for f_name in (files_after - files_before) if f_name.lower().endswith((".csv", ".tsv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"))]
        for data_file in new_datasets:
            try:
                data_path = workspace_dir / data_file
                data_title = data_path.stem
                data_ext = data_path.suffix.lower()

                # Parse and extract structural schema using core parser
                from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
                new_schema, _ = _parse_data_file(data_path, data_title, version=1, progress_cb=None)

                existing_data = discussion_instance.artefacts.get(data_title)
                if existing_data is None:
                    discussion_instance.artefacts.add(
                        title=data_title,
                        artefact_type="data",
                        content=new_schema,
                        file_ext=data_ext,
                        active=True,
                        read_only=True
                    )
                else:
                    discussion_instance.artefacts.update(
                        title=data_title,
                        new_content=new_schema,
                        new_type="data",
                        file_ext=data_ext,
                        active=True,
                        read_only=True
                    )
                discussion_instance.commit()
                out_prefix = f"\n\n[SYSTEM: New dataset '{data_file}' detected and auto-ingested successfully as an active session artifact.]"
                redirected_output.write(out_prefix)
                ASCIIColors.success(f"✓ Auto-ingested workspace dataset: '{data_file}' as artifact '{data_title}'")
            except Exception as data_ing_err:
                ASCIIColors.warning(f"Failed to auto-ingest dataset '{data_file}': {data_ing_err}")

        if not is_read_only:
            active_file_mtime_after = active_file.stat().st_mtime if active_file.exists() else 0
            file_written_by_script = (active_file_mtime_after > active_file_mtime_before)

            if file_written_by_script:
                if active_file.resolve() != new_file_path.resolve():
                    shutil.copy(str(active_file), str(new_file_path))
            else:
                # If SQLite, commit and close
                if ext in (".db", ".sqlite", ".sqlite3") and "conn" in local_vars:
                    local_vars["conn"].commit()
                    local_vars["conn"].close()
                elif ext == ".sqlconn" and "conn" in local_vars:
                    try:
                        local_vars["conn"].close()
                        local_vars["engine"].dispose()
                    except Exception:
                        pass
                # If CSV/Excel, write DataFrame back to the new versioned file
                elif ext in (".xlsx", ".xls") and "dfs" in local_vars:
                    with pd.ExcelWriter(new_file_path, engine="openpyxl") as writer:
                        for sheet_name, sheet_df in local_vars["dfs"].items():
                            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                elif "df" in local_vars:
                    local_vars["df"].to_csv(new_file_path, index=False, sep=sep)

            # Parse new schema and dynamically update/version the artifact
            from lollms_client.lollms_discussion._mixin_file_import import _parse_data_file
            if ext == ".sqlconn":
                # Re-parse connection in place to reflect any new tables or schema changes
                new_schema, _ = _parse_data_file(source_file_path, title, version=current_version, progress_cb=None)
                discussion_instance.artefacts.update(
                    title=title,
                    new_content=new_schema,
                    new_type="data",
                    active=True,
                    file_ext=ext,
                    version=current_version
                )
            else:
                new_schema, _ = _parse_data_file(new_file_path, title, version=new_version, progress_cb=None)
                discussion_instance.artefacts.update(
                    title=title,
                    new_content=new_schema,
                    new_type="data",
                    active=True,
                    file_ext=ext,
                    version=new_version
                )
            ASCIIColors.success(f"✓ Code executed and data version incremented to v{new_version} successfully.")
        else:
            if ext in (".db", ".sqlite", ".sqlite3") and "conn" in local_vars:
                local_vars["conn"].close()
            ASCIIColors.success("✓ Code executed successfully (Read-Only mode: no files written).")
    except Exception as e:
        sys.stdout = old_stdout
        err_msg = f"System Error in database updater: {e}"
        ASCIIColors.error(f"❌ {err_msg}")
        if new_file_path.exists() and ext not in (".db", ".sqlite", ".sqlite3", ".sqlconn"):
            if new_file_path.resolve() != source_file_path.resolve():
                new_file_path.unlink()
        return {"success": False, "error": err_msg, "output": redirected_output.getvalue()}
    finally:
        sys.stdout = old_stdout
        try:
            os.chdir(old_cwd)
        except Exception:
            pass

    out_str = redirected_output.getvalue()
    clean_output = sanitize_output_paths(out_str, workspace_dir)
    result = {
        "success": True,
        "output": clean_output or "Code executed successfully (no stdout prints)."
    }
    return result
