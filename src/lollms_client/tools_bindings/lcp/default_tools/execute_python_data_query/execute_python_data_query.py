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


def _sanitize_host_paths(text: str, workspace_root: Path) -> str:
    """Strips absolute host filesystem paths from tool outputs.

    The operator's physical layout (user folders, install dirs, temp dirs)
    must never leak to the LLM. The workspace root is rendered as '.';
    anything else under the user's profile / system dirs becomes '<host-path>'.
    """
    if not text:
        return text
    sanitized = text.replace(str(workspace_root), ".")
    return re.sub(
        r'[A-Za-z]:\\(?:Users|home|Program|Windows|AppData)[\\/][^\s"\'\n]*',
        '<host-path>',
        sanitized,
    )

# Import the shared sandbox kernel. Fall back to a localised import path when
# the tool is loaded outside the package tree.
try:
    from .sandbox_kernel import (
        validate_source,
        SandboxViolation,
        run_in_sandboxed_child,
    )
except ImportError:  # pragma: no cover
    from ..execute_python.sandbox_kernel import (  # type: ignore
        validate_source,
        SandboxViolation,
        run_in_sandboxed_child,
    )

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


def _has_registered_confirm_handler(env: Optional[Any] = None) -> bool:
    """True iff THIS tool call has a confirmation handler available.

    Only considers (a) `env.confirm_handler` (the standard host injection path)
    and (b) a handler registered directly on this module. We deliberately do
    NOT scan sibling modules: another test, GUI component, or unrelated LCP
    toolset in the same process registering a handler is NOT the same as "the
    operator wants to be prompted for THIS call". The previous `sys.modules`
    scan caused a session-wide cascade where one test's handler forced every
    subsequent data-query call into the "operator present" branch — which
    refuses in headless CI and broke 16+ unrelated tests.
    """
    if env is not None and callable(getattr(env, "confirm_handler", None)):
        return True
    own_mod = sys.modules.get(__name__)
    if own_mod is not None and callable(getattr(own_mod, "_CONFIRM_HANDLER", None)):
        return True
    return False


def _confirm_data_query_execution(source: str, env: Optional[Any] = None) -> Tuple[str, str]:
    """Prompts the operator before running code in the strict data-query sandbox.

    Honors `confirm_handler` set on the parent `execute_python` tool module if
    the host has registered one. Returns ("allow"|"reject", reason).
    """
    handler = None
    try:
        import sys as _sys
        for mod_name, mod in list(_sys.modules.items()):
            if "execute_python" in mod_name and mod.__name__ != __name__:
                handler = getattr(mod, "_CONFIRM_HANDLER", None)
                if handler:
                    break
        if handler is None and env is not None:
            handler = getattr(env, "confirm_handler", None)
    except Exception:
        handler = None

    if handler is not None and callable(handler):
        try:
            res = handler({
                "tool_name": "execute_python_data_query",
                "action_type": "data_query_execution",
                "source": source,
                "content": source,
                "label": "data_query",
                "metadata": {"profile": "strict-data"},
            })
            if isinstance(res, tuple):
                return str(res[0]).lower(), str(res[1]) if len(res) > 1 else ""
            if isinstance(res, bool):
                return ("allow", "") if res else ("reject", "Declined by operator.")
            return str(res).lower(), ""
        except Exception as exc:
            return "reject", f"Confirmation handler failed: {exc}"

    if sys.stdin and sys.stdin.isatty():
        ASCIIColors.panel(
            source[:4000] + ("\n…" if len(source) > 4000 else ""),
            title="🛡️ Execute Python Data Query Authorization (strict data sandbox)",
            border_style="yellow",
        )
        try:
            choice = input("  Authorize execution? [y]es / [n]o (default: y): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "reject", "Execution interrupted."
        if choice in ("", "y", "yes"):
            return "allow", ""
        return "reject", "Operator declined execution."

    # Non-interactive fallback: refuse. There is no way to safely auto-authorize
    # analytical code in a headless context.
    return "reject", (
        "No interactive channel is available. The strict data-query sandbox "
        "requires an explicit operator approval or a registered confirm_handler. "
        "Register a handler via LollmsDiscussion.chat(confirm_handler=...) before "
        "using this tool."
    )


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
    env: Optional[Any] = None,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
    **kwargs: Any,
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
    disc = discussion_instance or (getattr(env, "discussion", None) if env else None)
    client = lollms_client_instance or (getattr(env, "client", None) if env else None)  # parity; may be unused
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

    # ── LAYER 1: static AST validation (strict data-query allowlist) ──
    violation = validate_source(code, profile="data_query")
    if violation is not None:
        ASCIIColors.error(f"[execute_python_data_query] Static sandbox rejection: {violation.reason}")
        return {
            "success": False,
            "error": (
                f"🛑 BLOCKED BY DATA-QUERY SANDBOX (line {violation.line}): "
                f"{violation.reason}\n"
                f"Offending construct: {violation.snippet}\n"
                "This sandbox only permits pandas/numpy/matplotlib/openpyxl/"
                "python-docx/python-pptx/sqlalchemy and pure stdlib data helpers. "
                "No subprocess, no os.system, no network, no eval/exec, no imports "
                "outside the approved list."
            ),
            "output": "",
        }

    # ── LAYER 0 (OPTIONAL): operator authorization ──
    # The strict static + runtime sandbox is the authoritative guarantee: source
    # that survives `validate_source(profile="data_query")` AND the runtime
    # `SandboxContext` can only touch approved data-analysis libraries and the
    # workspace perimeter. Human authorization is therefore an *opt-in* UX layer:
    # we only prompt when the host has registered a confirm_handler. Headless
    # environments (CI, servers, notebooks) run without a prompt because the
    # sandbox itself is the control — not the operator.
    _handler_present = _has_registered_confirm_handler(env)
    if _handler_present:
        decision, reason = _confirm_data_query_execution(code, env=env)
        if decision not in ("allow", "always"):
            return {
                "success": False,
                "error": (
                    "🛑 BLOCKED BY OPERATOR: the data-query execution was not "
                    f"authorized ({reason or 'rejected'})."
                ),
                "output": "",
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

    # Prefer the host-injected workspace (parity with execute_python); fall
    # back to CWD. The sandbox boundary is exactly this path.
    workspace_root = Path.cwd().resolve()
    if env is not None:
        _ctx_ws = getattr(getattr(env, "tool_context", None), "workspace_path", None)
        if _ctx_ws:
            try:
                _ws = Path(_ctx_ws).resolve()
                if _ws.exists() and _ws.is_dir():
                    workspace_root = _ws
            except Exception:
                pass
    import base64

    _np, _pd, _plt = _lazy_import_libs()

    class _NoReconfigureStringIO(io.StringIO):
        """StringIO that silently ignores reconfigure() calls from user code."""
        def reconfigure(self, *args, **kwargs):
            pass

    # Defense-in-depth: install the runtime sandbox patch BEFORE exposing any
    # builtins to user code. User code only sees an explicitly reduced surface.
    safe_builtins = {
        # Essential pure operations
        "abs": abs, "all": all, "any": any, "bool": bool, "bytearray": bytearray,
        "bytes": bytes, "callable": callable, "chr": chr, "complex": complex,
        "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "frozenset": frozenset,
        "getattr": getattr, "hasattr": hasattr, "hash": hash, "id": id,
        "int": int, "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list, "map": map,
        "max": max, "min": min, "next": next, "object": object,
        "oct": oct, "ord": ord, "pow": pow, "range": range,
        "repr": repr, "reversed": reversed, "round": round, "set": set,
        "setattr": setattr, "slice": slice, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "type": type, "zip": zip,
        # Exceptions
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
        "RuntimeError": RuntimeError, "StopIteration": StopIteration,
        "ZeroDivisionError": ZeroDivisionError, "FileNotFoundError": FileNotFoundError,
        "PermissionError": PermissionError, "IOError": OSError, "OSError": OSError,
        # Print only (no stdout hijack helpers)
        "print": print,
        # Import is installed LAST below, after the SandboxContext patches it.
        "__import__": None,
        "__build_class__": __build_class__,
    }

    local_vars = {
        "Path": Path,
        "pd": _pd,
        "np": _np,
        "plt": _plt,
        "__builtins__": safe_builtins,
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

    # ── PROCESS ISOLATION: run user code in a sandboxed child ──
    # The parent process state is NEVER patched. All sandbox behavior — the
    # filtered importer, path guards, blocked builtins — lives inside the child
    # and dies with it. This is the strongest guarantee the sandbox can make:
    # no state leaks between tool calls, no cascade risk, no module-load-order
    # hazard.
    ASCIIColors.info(f"⚡ Executing Python code in isolated sandbox child (CWD: {os.getcwd()})")

    # Seed the child's local namespace with the parent-side auto-loaded df and
    # the guaranteed-available libraries. The child cannot see parent objects,
    # so we ship df as a pickled payload via seed_locals — the child unpickles
    # it into its own namespace under the name "df".
    child_seed: Dict[str, Any] = {}
    df_present_at_start = "df" in local_vars
    persist_df = (auto_loaded_df_file is not None) or defines_df
    persist_df_target = auto_loaded_df_file

    # Note: `seed_locals` must be picklable. `pd`/`np`/`plt` are modules, and
    # modules are picklable by reference — the child's fresh interpreter imports
    # them again on unpickle (guaranteed available since we required them).
    # The initial `df` (if auto-loaded) is pickled for transport.
    if "df" in local_vars:
        try:
            import pickle as _pickle
            child_seed["__df_pickle__"] = _pickle.dumps(local_vars["df"])
        except Exception:
            pass

    envelope = run_in_sandboxed_child(
        source=code,
        workspace_root=workspace_root,
        profile="data_query",
        script_label="<execute_python_data_query>",
        argv=["<execute_python_data_query>"],
        seed_locals=child_seed,
        persist_df_name="df" if persist_df else None,
        hard_timeout=120.0,
    )

    # Restore the parent-side UTF-8 env (the child had its own copies).
    for _env_key, _saved_val in _saved_utf8_env.items():
        if _saved_val is None:
            os.environ.pop(_env_key, None)
        else:
            os.environ[_env_key] = _saved_val

    if auto_loaded_engine is not None:
        try:
            auto_loaded_engine.dispose()
        except Exception:
            pass

    raw_output = envelope.get("stdout", "") or ""
    raw_stderr = envelope.get("stderr", "") or ""
    child_traceback = envelope.get("traceback", "") or ""
    error_type = envelope.get("error_type", "") or ""

    # ── Child reported an error: shape it like before, run Office rollback ──
    if not envelope.get("success"):
        if envelope.get("sandbox_violation"):
            rollback_failure = _validate_and_maybe_rollback(office_snapshot, raw_output)
            annotated_traceback = (
                f"SandboxViolation: {child_traceback.strip()}\n"
                f"(This operation is not permitted in the strict data-query sandbox.)"
            )
            if rollback_failure is not None:
                annotated_traceback += (
                    f"\n[Office Guard] {OFFICE_GUARD_FAILURE_CODE}: corrupted documents were rolled back "
                    f"to their pre-execution state. {rollback_failure['output']}"
                )
            return {
                "success": False,
                "error": f"🛑 SANDBOX BLOCK: {annotated_traceback}",
                "output": raw_output,
            }
        rollback_failure = _validate_and_maybe_rollback(office_snapshot, raw_output)
        annotated_traceback = _annotate_traceback(child_traceback, code)
        ASCIIColors.error(f"❌ Execution Failed (child):\n{child_traceback}")
        if rollback_failure is not None:
            annotated_traceback += (
                f"\n[Office Guard] {OFFICE_GUARD_FAILURE_CODE}: corrupted documents were rolled back "
                f"to their pre-execution state. {rollback_failure['output']}"
            )
        return {
            "success": False,
            "error": f"Execution Error:\n{annotated_traceback}",
            "output": raw_output,
        }

    # ── Success path: parent-side post-execution processing ──
    guard_failure = _validate_and_maybe_rollback(office_snapshot, raw_output)
    if guard_failure is not None:
        return guard_failure

    # Rehydrate the child's df (if user modified it) for parent-side persistence.
    df_from_child = None
    if envelope.get("df_pickle") and _pd is not None:
        try:
            import pickle as _pickle
            df_from_child = _pickle.loads(envelope["df_pickle"])
        except Exception as reload_err:
            ASCIIColors.warning(f"[execute_python_data_query] Failed to rehydrate df from child: {reload_err}")

    if persist_df_target and df_from_child is not None and isinstance(df_from_child, _pd.DataFrame):
        try:
            ASCIIColors.info(f"[execute_python_data_query] Persisting modified df back to {persist_df_target.name}")
            if persist_df_target.suffix.lower() == ".csv":
                df_from_child.to_csv(persist_df_target, sep=auto_loaded_sep, index=False, encoding="utf-8-sig")
            elif persist_df_target.suffix.lower() in (".xlsx", ".xls"):
                df_from_child.to_excel(persist_df_target, index=False)

            if persist_df_target.suffix.lower() in (".xlsx", ".xls"):
                df_ok, df_reason = _validate_office_file(persist_df_target)
                if not df_ok:
                    resolved_df_path = persist_df_target.resolve()
                    if resolved_df_path in office_snapshot:
                        persist_df_target.write_bytes(office_snapshot[resolved_df_path])
                        restoration_note = "Restored from pre-execution in-memory snapshot."
                    else:
                        persist_df_target.unlink(missing_ok=True)
                        restoration_note = "New file deleted (no previous version to restore)."
                    ASCIIColors.error(
                        f"[Office Guard] df persistence produced an invalid spreadsheet ({df_reason}) — rolled back."
                    )
                    return _build_office_guard_failure(
                        persist_df_target.name, df_reason, restoration_note, raw_output
                    )

            if disc and hasattr(disc, "artefacts"):
                art_title = persist_df_target.name
                existing_art = disc.artefacts.get(art_title)
                if not existing_art and persist_df_target.stem != persist_df_target.name:
                    existing_art = disc.artefacts.get(persist_df_target.stem)
                    if existing_art:
                        art_title = persist_df_target.stem

                if existing_art:
                    disc.artefacts.update(
                        title=art_title,
                        new_content=df_from_child.to_csv(
                            index=False,
                            sep=auto_loaded_sep if persist_df_target.suffix.lower() == ".csv" else ",",
                        ),
                        bump_version=True,
                        active=True,
                    )
                    disc.commit()
        except Exception as save_err:
            ASCIIColors.warning(f"[execute_python_data_query] Failed to persist df: {save_err}")

    # Rehydrate plot figures — the child captured them as PNG bytes. The parent
    # writes them to the workspace and syncs artefacts; no matplotlib state
    # cross-process.
    figures = envelope.get("figures") or []
    if figures:
        ASCIIColors.success(f"[Sandbox] Intercepted {len(figures)} generated plot figure(s)!")
        for png_bytes in figures:
            plot_b64 = base64.b64encode(png_bytes).decode("utf-8")
            plot_filename = f"custom_analysis_plot_{uuid.uuid4().hex[:6]}.png"
            plot_path = Path(".") / plot_filename
            try:
                plot_path.write_bytes(png_bytes)
            except Exception as write_err:
                ASCIIColors.warning(f"[execute_python_data_query] Failed to write plot {plot_filename}: {write_err}")
                continue

            if disc and hasattr(disc, "artefacts"):
                try:
                    art_title = plot_filename
                    comp_title = f"{art_title}::images"
                    existing_comp = disc.artefacts.get(comp_title)

                    if existing_comp:
                        existing_imgs = existing_comp.get("images", [])
                        existing_mtypes = existing_comp.get("image_media_types", [])
                        existing_imgs.append(plot_b64)
                        existing_mtypes.append("image/png")
                        disc.artefacts.update(
                            title=comp_title,
                            new_images=existing_imgs,
                            new_image_media_types=existing_mtypes,
                            active=True,
                        )
                    else:
                        disc.artefacts.add(
                            title=art_title,
                            artefact_type="image",
                            content=f"![{art_title}]({art_title})",
                            images=[plot_b64],
                            image_media_types=["image/png"],
                            active=True,
                            file_ext=".png",
                        )
                    disc.commit()
                    ASCIIColors.info(f"[execute_python_data_query] Synced plot '{art_title}' to artefact system.")
                except Exception as sync_err:
                    ASCIIColors.warning(f"[execute_python_data_query] Failed to sync plot to artefact system: {sync_err}")

    out_str = raw_output
    if not out_str.strip():
        out_str = "Code executed successfully (no stdout prints)."

    # Preserve host-path opacity: the LLM must only ever see the workspace
    # root represented as '.', never absolute paths under the operator's profile.
    out_str = _sanitize_host_paths(out_str, workspace_root)

    return {
        "success": True,
        "output": out_str,
    }