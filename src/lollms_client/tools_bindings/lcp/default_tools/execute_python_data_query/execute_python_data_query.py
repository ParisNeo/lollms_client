import os
import sys
import io
import re
import json
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Execute Python Data Query"
TOOL_LIBRARY_DESC = "Executes sandboxed Python code to analyze or modify datasets in the workspace."
TOOL_LIBRARY_ICON = "📊"

# --- Office OOXML transactional guard: format registry ---
# All Microsoft Office content shares the same ZIP + XML container structure;
# only the required root parts and the Layer-3 re-opener library differ per family.
OOXML_VALIDATORS = {
    ".xlsx": {"required_parts": ("[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"), "reopener": "openpyxl"},
    ".xlsm": {"required_parts": ("[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"), "reopener": "openpyxl"},
    ".docx": {"required_parts": ("[Content_Types].xml", "word/document.xml", "word/_rels/document.xml.rels"), "reopener": "python-docx"},
    ".docm": {"required_parts": ("[Content_Types].xml", "word/document.xml", "word/_rels/document.xml.rels"), "reopener": "python-docx"},
    ".pptx": {"required_parts": ("[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"), "reopener": "python-pptx"},
    ".pptm": {"required_parts": ("[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"), "reopener": "python-pptx"},
}
GUARDED_EXTENSIONS = tuple(OOXML_VALIDATORS.keys())

OFFICE_FAILURE_CODES = {
    ".xlsx": "xlsx_failed_excel_validation",
    ".xlsm": "xlsx_failed_excel_validation",
    ".docx": "docx_failed_word_validation",
    ".docm": "docx_failed_word_validation",
    ".pptx": "pptx_failed_powerpoint_validation",
    ".pptm": "pptx_failed_powerpoint_validation",
}
OFFICE_GUARD_FAILURE_CODE = "office_file_failed_validation"

OFFICE_SNAPSHOT_MAX_BYTES = 50 * 1024 * 1024

OFFICE_GUARD_GUIDANCE = (
    "Likely causes: chart anchored over data/helper ranges, chart width/height mutated after add_chart(), "
    "dropped native parts (richData/VBA/slicers) when re-saving a native workbook, formulas written "
    "without cached values, or python-docx/python-pptx re-saves dropping parts desktop Word/PowerPoint "
    "requires (styles.xml, numbering rels, theme rels). Do NOT re-issue identical code. Rebuild the "
    "content: set chart width/height BEFORE add_chart(), anchor charts away from data/helper ranges, "
    "precompute aggregation values in Python (write literals) instead of SUMIF formulas feeding the "
    "chart, and prefer building fresh documents over round-tripping desktop-authored files."
)

# --- Backward-compatibility aliases (existing tests import these names) ---
XLSX_REQUIRED_PARTS = OOXML_VALIDATORS[".xlsx"]["required_parts"]
XLSX_SNAPSHOT_MAX_BYTES = OFFICE_SNAPSHOT_MAX_BYTES
XLSX_GUARD_FAILURE_CODE = OFFICE_FAILURE_CODES[".xlsx"]
XLSX_GUARD_GUIDANCE = OFFICE_GUARD_GUIDANCE

def init_tools_library(config: dict = None) -> None:
    try:
        import pipmaster as pm
        pm.ensure_packages(["pandas", "numpy", "matplotlib", "openpyxl", "python-docx", "python-pptx", "sqlalchemy"])
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

def _annotate_traceback(raw_traceback: str, source_code: str) -> str:
    """Appends the exact failing source line(s) to a traceback raised by sandboxed code.

    Frame file names matching "<execute_python_data_query>" carry line numbers relative
    to the submitted snippet; the model otherwise has to re-count lines in its own code
    to locate the failure. This shows the line directly so fixes stay surgical."""
    code_lines = source_code.splitlines()
    annotated_lines = []
    for line in raw_traceback.splitlines():
        annotated_lines.append(line)
        match = re.search(
            r'File "<execute_python_data_query>", line (\d+)', line
        )
        if match:
            failing_line_no = int(match.group(1))
            if 1 <= failing_line_no <= len(code_lines):
                annotated_lines.append(
                    f"    >>> LINE {failing_line_no}: {code_lines[failing_line_no - 1]}"
                )
    return "\n".join(annotated_lines)


def _reopen_office_file(path: Path, suffix: str) -> Tuple[bool, str]:
    """Layer-3 format-aware re-open. Returns (ok, reason). Never raises."""
    try:
        if suffix in (".xlsx", ".xlsm"):
            from openpyxl import load_workbook
            if suffix == ".xlsm":
                load_workbook(path, keep_vba=True)
            else:
                load_workbook(path)
        elif suffix in (".docx", ".docm"):
            from docx import Document
            Document(path)
        elif suffix in (".pptx", ".pptm"):
            from pptx import Presentation
            Presentation(path)
        else:
            return True, "ok (layers 1-2 only; no reopener for this format)"
        return True, "ok"
    except ImportError:
        reopener_name = OOXML_VALIDATORS.get(suffix, {}).get("reopener", "reopener")
        return True, f"ok (layers 1-2 only; {reopener_name} unavailable)"
    except Exception as reopen_err:
        return False, f"re-open failed: {reopen_err}"


def _validate_office_file(path) -> Tuple[bool, str]:
    """Three-layer Office OOXML integrity validation. Never raises.

    Layer 1: ZIP container integrity + presence of the format's required OOXML parts.
    Layer 2: strict XML parse of every .xml/.rels member.
    Layer 3: format-aware re-open (openpyxl / python-docx / python-pptx) — catches
             orphaned chart/style/theme references the lenient ZIP+XML layers miss.

    Desktop Office acceptance is the goal, not merely a library re-read: these
    libraries are lenient and will happily re-read OOXML that Word/Excel/PowerPoint's
    strict schema validators reject ("unreadable content" repair prompt).
    """
    office_path = Path(path)
    try:
        if not office_path.is_file():
            return False, "file does not exist"
        suffix = office_path.suffix.lower()
        validator = OOXML_VALIDATORS.get(suffix, {})
        required_parts = validator.get("required_parts", ("[Content_Types].xml",))
        with zipfile.ZipFile(office_path) as z:
            bad_member = z.testzip()
            if bad_member:
                return False, f"corrupt zip member: {bad_member}"
            names = z.namelist()
            for required in required_parts:
                if required not in names:
                    return False, f"missing required part: {required}"
            for member_name in names:
                if member_name.endswith((".xml", ".rels")):
                    try:
                        ET.fromstring(z.read(member_name))
                    except ET.ParseError as xml_err:
                        return False, f"malformed XML in part '{member_name}': {xml_err}"
        reopen_ok, reopen_reason = _reopen_office_file(office_path, suffix)
        if not reopen_ok:
            return False, reopen_reason
        return True, reopen_reason
    except zipfile.BadZipFile:
        return False, "not a valid zip container (BadZipFile)"
    except Exception as e:
        return False, f"validation error: {e}"


def _snapshot_office_files() -> Dict[Path, bytes]:
    """Captures in-memory byte snapshots of every guarded Office file in the CWD.

    Files larger than OFFICE_SNAPSHOT_MAX_BYTES are skipped (with a warning) to
    avoid memory blowups; they remain unguarded by the transactional wrapper.
    """
    snapshot: Dict[Path, bytes] = {}
    for entry in Path(".").iterdir():
        try:
            if not entry.is_file() or entry.suffix.lower() not in GUARDED_EXTENSIONS:
                continue
            resolved = entry.resolve()
            if resolved in snapshot:
                continue
            if entry.stat().st_size > OFFICE_SNAPSHOT_MAX_BYTES:
                ASCIIColors.warning(
                    f"[Office Guard] {entry.name} exceeds {OFFICE_SNAPSHOT_MAX_BYTES // (1024 * 1024)}MB — skipped from snapshot (unguarded)."
                )
                continue
            snapshot[resolved] = entry.read_bytes()
        except Exception as snap_err:
            ASCIIColors.warning(f"[Office Guard] Failed to snapshot {entry.name}: {snap_err}")
    return snapshot


def _build_office_guard_failure(file_name: str, reason: str, restoration_note: str, captured_output: str) -> Dict[str, Any]:
    """Builds the structured failure payload returned when Office file validation fails."""
    failure_code = OFFICE_FAILURE_CODES.get(Path(file_name).suffix.lower(), OFFICE_GUARD_FAILURE_CODE)
    error_text = (
        f"Office Document Integrity Failure [{failure_code}]: the updated document "
        f"failed strict validation and was rolled back to its previous version.\n"
        f"File: {file_name}\n"
        f"Reason: {reason}\n"
        f"{restoration_note}\n"
        f"{OFFICE_GUARD_GUIDANCE}"
    )
    guard_note = f"\n[Office Guard] Rolled back {file_name} ({reason})."
    output_text = (captured_output + guard_note) if captured_output else guard_note.lstrip("\n")
    return {
        "success": False,
        "error": error_text,
        "output": output_text,
    }


def _validate_and_maybe_rollback(snapshot: Dict[Path, bytes], captured_output: str = "") -> Optional[Dict[str, Any]]:
    """Validates mutated/new Office documents; restores or deletes corrupt files.

    Returns None when every guarded document on disk is structurally valid, otherwise
    returns the structured failure payload after restoring all corrupted files:
    - Snapshot files that were mutated and fail validation → byte-identical restore.
    - Files legitimately deleted by user code are respected (not resurrected).
    - New guarded files (absent from the snapshot) that fail validation → deleted.
    """
    failures: List[Tuple[Path, str, str]] = []

    for snap_path, snap_bytes in snapshot.items():
        try:
            if not snap_path.is_file():
                continue
            if snap_path.read_bytes() == snap_bytes:
                continue
            ok, reason = _validate_office_file(snap_path)
            if not ok:
                snap_path.write_bytes(snap_bytes)
                failures.append((snap_path, reason, f"Restored from pre-execution in-memory snapshot."))
                ASCIIColors.error(f"[Office Guard] {snap_path.name} failed validation ({reason}) — restored from snapshot.")
        except Exception as guard_err:
            ASCIIColors.warning(f"[Office Guard] Could not validate/restore {snap_path.name}: {guard_err}")

    snapshot_keys = set(snapshot.keys())
    for entry in Path(".").iterdir():
        try:
            if entry.suffix.lower() not in GUARDED_EXTENSIONS or not entry.is_file():
                continue
            resolved = entry.resolve()
            if resolved in snapshot_keys:
                continue
            ok, reason = _validate_office_file(entry)
            if not ok:
                entry.unlink(missing_ok=True)
                failures.append((entry, reason, "New file deleted (no previous version to restore)."))
                ASCIIColors.error(f"[Office Guard] New file {entry.name} failed validation ({reason}) — deleted.")
        except Exception as guard_err:
            ASCIIColors.warning(f"[Office Guard] Could not validate new file {entry.name}: {guard_err}")

    if not failures:
        return None

    primary_path, primary_reason, primary_note = failures[0]
    failure = _build_office_guard_failure(primary_path.name, primary_reason, primary_note, captured_output)
    if len(failures) > 1:
        extra = "; ".join(f"{p.name} ({r})" for p, r, _ in failures[1:])
        failure["error"] += f"\nAdditional rolled-back files: {extra}"
        failure["output"] += f"\n[Office Guard] Additional rolled-back files: {extra}"
    return failure


# --- Backward-compatibility aliases (existing tests import these names) ---
_validate_xlsx = _validate_office_file
_snapshot_xlsx_files = _snapshot_office_files
_build_xlsx_guard_failure = _build_office_guard_failure


def tool_execute_python_data_query(
    code: str = "",
    env: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Execute sandboxed Python code to analyze or modify datasets in the workspace.

    The code can reference any file in the workspace using simple relative paths.
    For example:
      - df = pd.read_csv("data.csv")
      - with open("circuit.cir") as f: content = f.read()

    Office document safety: every .xlsx/.xlsm/.docx/.docm/.pptx/.pptm file in the
    workspace is snapshotted before execution. After execution, any mutated or
    newly created Office document must pass strict three-layer validation (ZIP
    integrity, strict XML parse of every part, format-aware re-open via
    openpyxl/python-docx/python-pptx). Files that would be rejected by
    Word/Excel/PowerPoint are rolled back to their pre-execution state (or
    deleted if newly created) and the tool returns a structured failure instead
    of committing corruption to the workspace.

    Args:
        code (str, optional): The Python code to execute, or the name of a Python file in the workspace. Defaults to empty.
        env (Any, optional): Host-injected unified runtime context (env.discussion, env.client). Defaults to None.
    """
    discussion_instance = getattr(env, "discussion", None) if env else None
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

    _saved_utf8_env = {}
    for _env_key in ("PYTHONIOENCODING", "PYTHONUTF8"):
        _saved_utf8_env[_env_key] = os.environ.get(_env_key)
        os.environ[_env_key] = "utf-8" if _env_key == "PYTHONIOENCODING" else "1"

    if code.endswith(".py") and Path(code).exists():
        filename = code
        code = Path(code).read_text(encoding="utf-8")
        ASCIIColors.info(f"[execute_python_data_query] Loaded code from file: {filename}")

    office_snapshot = _snapshot_office_files()
    if office_snapshot:
        ASCIIColors.info(f"[Office Guard] Snapshot of {len(office_snapshot)} document(s) taken.")

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
            annotated_traceback = _annotate_traceback(raw_traceback, code)
            ASCIIColors.error(f"❌ Execution Failed:\n{raw_traceback}")
            rollback_failure = _validate_and_maybe_rollback(office_snapshot, raw_output)
            if rollback_failure is not None:
                annotated_traceback += (
                    f"\n[Office Guard] {OFFICE_GUARD_FAILURE_CODE}: corrupted documents were rolled back "
                    f"to their pre-execution state. {rollback_failure['output']}"
                )
            return {
                "success": False,
                "error": f"Execution Error:\n{annotated_traceback}",
                "output": raw_output
            }

        guard_failure = _validate_and_maybe_rollback(office_snapshot, redirected_output.getvalue())
        if guard_failure is not None:
            return guard_failure

        if auto_loaded_df_file and "df" in local_vars and _pd is not None and isinstance(local_vars["df"], _pd.DataFrame):
            try:
                ASCIIColors.info(f"[execute_python_data_query] Persisting modified df back to {auto_loaded_df_file.name}")
                if auto_loaded_df_file.suffix.lower() == ".csv":
                    local_vars["df"].to_csv(auto_loaded_df_file, sep=auto_loaded_sep, index=False, encoding="utf-8-sig")
                elif auto_loaded_df_file.suffix.lower() in (".xlsx", ".xls"):
                    local_vars["df"].to_excel(auto_loaded_df_file, index=False)

                if auto_loaded_df_file.suffix.lower() in (".xlsx", ".xls"):
                    df_ok, df_reason = _validate_office_file(auto_loaded_df_file)
                    if not df_ok:
                        resolved_df_path = auto_loaded_df_file.resolve()
                        if resolved_df_path in office_snapshot:
                            auto_loaded_df_file.write_bytes(office_snapshot[resolved_df_path])
                            restoration_note = "Restored from pre-execution in-memory snapshot."
                        else:
                            auto_loaded_df_file.unlink(missing_ok=True)
                            restoration_note = "New file deleted (no previous version to restore)."
                        ASCIIColors.error(
                            f"[Office Guard] df persistence produced an invalid spreadsheet ({df_reason}) — rolled back."
                        )
                        return _build_office_guard_failure(
                            auto_loaded_df_file.name, df_reason, restoration_note, redirected_output.getvalue()
                        )

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
        rollback_failure = _validate_and_maybe_rollback(office_snapshot, raw_output)
        if rollback_failure is not None:
            raw_traceback += (
                f"\n[Office Guard] {OFFICE_GUARD_FAILURE_CODE}: corrupted documents were rolled back "
                f"to their pre-execution state. {rollback_failure['output']}"
            )
        return {
            "success": False,
            "error": f"Unexpected execution failure:\n{raw_traceback}",
            "output": raw_output
        }
    finally:
        sys.stdout = old_stdout
        for _env_key, _saved_val in _saved_utf8_env.items():
            if _saved_val is None:
                os.environ.pop(_env_key, None)
            else:
                os.environ[_env_key] = _saved_val
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