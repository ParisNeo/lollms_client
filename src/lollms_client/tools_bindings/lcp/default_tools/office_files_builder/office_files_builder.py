import re
import textwrap
import traceback
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Office Files Builder"
TOOL_LIBRARY_DESC = "Read, create and edit every kind of Microsoft Office file (.pptx, .docx, .xlsx, .xlsm, .docm, .pptm) via sandboxed recipe code. Pre-imported design system: python-pptx, python-docx, openpyxl symbols. Recipes are plain Python with NO import statements; violations are rejected before execution. Every save is verified with a strict three-layer OOXML integrity check before being committed to the workspace."
TOOL_LIBRARY_ICON = "🧰"

# --- Recipe sandbox limits (mirrors pptx_builder) ---
_MAX_RECIPE_CHARS = 60_000
_FORBIDDEN_TOKENS = (
    "import ", "__import__", "eval(", "exec(", "compile(", "open(",
    "os.", "sys.", "subprocess", "shutil", "pathlib", "__",
    "globals(", "locals(", "vars(", "getattr(", "setattr(", "delattr(",
)

# --- OOXML integrity registry (mirrors execute_python_data_query) ---
OOXML_VALIDATORS = {
    ".xlsx": {"required_parts": ("[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"), "reopener": "openpyxl"},
    ".xlsm": {"required_parts": ("[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"), "reopener": "openpyxl"},
    ".docx": {"required_parts": ("[Content_Types].xml", "word/document.xml", "word/_rels/document.xml.rels"), "reopener": "python-docx"},
    ".docm": {"required_parts": ("[Content_Types].xml", "word/document.xml", "word/_rels/document.xml.rels"), "reopener": "python-docx"},
    ".pptx": {"required_parts": ("[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"), "reopener": "python-pptx"},
    ".pptm": {"required_parts": ("[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"), "reopener": "python-pptx"},
}
GUARDED_EXTENSIONS = tuple(OOXML_VALIDATORS.keys())

OFFICE_GUARD_FAILURE_CODE = "office_file_failed_validation"

OFFICE_GUARD_GUIDANCE = (
    "Likely causes: chart anchored over data/helper ranges, chart width/height mutated after add_chart(), "
    "dropped native parts (richData/VBA/slicers) when re-saving a native workbook, formulas written "
    "without cached values, or python-docx/python-pptx re-saves dropping parts desktop Word/PowerPoint "
    "requires (styles.xml, numbering rels, theme rels). Do NOT re-issue identical code. Rebuild the "
    "content: set chart width/height BEFORE add_chart(), anchor charts away from data/helper ranges, "
    "precompute aggregation values in Python (write literals) instead of SUMIF formulas feeding the "
    "chart, and prefer building fresh documents over round-tripping desktop-authored files."
)


def init_tools_library(config: dict = None) -> None:
    try:
        import pipmaster as pm
        pm.ensure_packages(["python-pptx", "python-docx", "openpyxl"])
    except Exception as e:
        ASCIIColors.warning(f"[office_files_builder] Failed to ensure all dependencies: {e}")


def _check_recipe(recipe: str) -> None:
    """Rejects recipes containing forbidden tokens before execution."""
    for token in _FORBIDDEN_TOKENS:
        if token in recipe:
            raise ValueError(
                f"Forbidden token '{token}' in recipe. Recipes are plain Python with NO imports, "
                "no file/system access, and no dunder access. Use only the pre-imported design symbols."
            )


def _annotate_traceback(raw_traceback: str, source_code: str, source_name: str) -> str:
    """Appends the exact failing recipe line after each frame referencing source_name."""
    code_lines = source_code.splitlines()
    annotated_lines = []
    for line in raw_traceback.splitlines():
        annotated_lines.append(line)
        match = re.search(
            re.escape(f'File "{source_name}", line ') + r"(\d+)", line
        )
        if match:
            failing_line_no = int(match.group(1))
            if 1 <= failing_line_no <= len(code_lines):
                annotated_lines.append(
                    f"    >>> LINE {failing_line_no}: {code_lines[failing_line_no - 1]}"
                )
    return "\n".join(annotated_lines)


def _build_recipe_namespace() -> Dict[str, Any]:
    """Pre-populates the recipe namespace with per-format design symbols (lazy imports)."""
    namespace: Dict[str, Any] = {}

    # python-pptx symbols (canonical names, as in pptx_builder)
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.oxml.ns import qn
        namespace.update({
            "Presentation": Presentation,
            "Inches": Inches,
            "Pt": Pt,
            "Emu": Emu,
            "RGBColor": RGBColor,
            "MSO_SHAPE": MSO_SHAPE,
            "MSO_CONNECTOR": MSO_CONNECTOR,
            "MSO_ANCHOR": MSO_ANCHOR,
            "PP_ALIGN": PP_ALIGN,
            "qn": qn,
        })
    except ImportError:
        ASCIIColors.warning("[office_files_builder] python-pptx unavailable — pptx recipes disabled.")

    # python-docx symbols (prefixed to avoid collisions with pptx Inches/Pt/RGBColor)
    try:
        from docx import Document as DocxDocument
        from docx.shared import Inches as DocxInches, Pt as DocxPt, RGBColor as DocxRGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        namespace.update({
            "DocxDocument": DocxDocument,
            "DocxInches": DocxInches,
            "DocxPt": DocxPt,
            "DocxRGBColor": DocxRGBColor,
            "WD_ALIGN_PARAGRAPH": WD_ALIGN_PARAGRAPH,
        })
    except ImportError:
        ASCIIColors.warning("[office_files_builder] python-docx unavailable — docx recipes disabled.")

    # openpyxl symbols
    try:
        from openpyxl import Workbook, load_workbook
        from openpyxl.chart import PieChart, LineChart, BarChart, Reference
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        namespace.update({
            "Workbook": Workbook,
            "load_workbook": load_workbook,
            "PieChart": PieChart,
            "LineChart": LineChart,
            "BarChart": BarChart,
            "Reference": Reference,
            "Font": Font,
            "PatternFill": PatternFill,
            "Alignment": Alignment,
            "Border": Border,
            "Side": Side,
            "get_column_letter": get_column_letter,
        })
    except ImportError:
        ASCIIColors.warning("[office_files_builder] openpyxl unavailable — xlsx recipes disabled.")

    return namespace


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
    Layer 3: format-aware re-open — catches orphaned chart/style/theme references.
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


def _save_and_verify(obj: Any, save_path: Path) -> Dict[str, Any]:
    """Transactional save: snapshot → save → validate → rollback on failure.

    Guarantees that no Office-rejected file is ever committed to the workspace:
    on validation failure the pre-save bytes are restored (or the new file deleted).
    """
    save_path = Path(save_path)
    pre_existing = save_path.is_file()
    snapshot_bytes = save_path.read_bytes() if pre_existing else None
    try:
        obj.save(str(save_path))
    except Exception as save_err:
        if pre_existing:
            save_path.write_bytes(snapshot_bytes)
        else:
            save_path.unlink(missing_ok=True)
        return {
            "success": False,
            "error": (
                f"Save failed before validation could run: {save_err}\n"
                "The file was restored to its pre-save state. Fix the recipe and retry."
            ),
        }

    ok, reason = _validate_office_file(save_path)
    if ok:
        return {
            "success": True,
            "saved_path": str(save_path),
            "file_name": save_path.name,
        }

    if pre_existing:
        save_path.write_bytes(snapshot_bytes)
        restoration_note = "The file was restored to its pre-save state (pre-save in-memory snapshot)."
    else:
        save_path.unlink(missing_ok=True)
        restoration_note = "The new file was deleted (no previous version to restore)."

    ASCIIColors.error(f"[Office Guard] {save_path.name} failed validation ({reason}) — rolled back.")
    return {
        "success": False,
        "error": (
            f"Office Document Integrity Failure [{OFFICE_GUARD_FAILURE_CODE}]: the saved document "
            f"failed strict validation and was rolled back.\n"
            f"File: {save_path.name}\n"
            f"Reason: {reason}\n"
            f"{restoration_note}\n"
            f"{OFFICE_GUARD_GUIDANCE}"
        ),
    }


def _exec_recipe(recipe: str, namespace: Dict[str, Any], source_name: str) -> Optional[Dict[str, Any]]:
    """Validates and executes a recipe in the sandboxed namespace.

    Returns None on success, or a failure dict with an annotated traceback.
    """
    if not recipe or not recipe.strip():
        return None
    if len(recipe) > _MAX_RECIPE_CHARS:
        return {
            "success": False,
            "error": f"Recipe exceeds the maximum length of {_MAX_RECIPE_CHARS} characters.",
        }
    try:
        _check_recipe(recipe)
    except ValueError as forbidden_err:
        return {"success": False, "error": str(forbidden_err)}

    dedented = textwrap.dedent(recipe)
    try:
        exec(compile(dedented, source_name, "exec"), namespace)
        return None
    except Exception:
        annotated = _annotate_traceback(traceback.format_exc(), dedented, source_name)
        return {"success": False, "error": f"Recipe execution error:\n{annotated}"}


def _resolve_working_dir(discussion_instance: Optional[Any]) -> Path:
    """Resolves the directory files are saved to / read from (discussion workspace or CWD)."""
    if discussion_instance is not None:
        workspace = getattr(discussion_instance, "workspace_data_path", None)
        if workspace:
            return Path(workspace)
    return Path(".")


def _sanitize_file_name(file: str, default: str) -> str:
    """Normalizes a file name, strips path components, and applies the default when empty."""
    name = (file or "").strip()
    if not name:
        return default
    name = Path(name).name
    if not name or name in (".", ".."):
        return default
    return name


def _create_root_object(suffix: str, namespace: Dict[str, Any]) -> Any:
    """Creates a fresh root object for the requested format and binds it into the namespace."""
    if suffix in (".pptx", ".pptm"):
        prs = namespace["Presentation"]()
        prs.slide_width = namespace["Inches"](13.333)
        prs.slide_height = namespace["Inches"](7.5)
        namespace["prs"] = prs
        namespace["presentation"] = prs
        return prs
    if suffix in (".docx", ".docm"):
        doc = namespace["DocxDocument"]()
        namespace["doc"] = doc
        return doc
    if suffix in (".xlsx", ".xlsm"):
        wb = namespace["Workbook"]()
        ws = wb.active
        namespace["wb"] = wb
        namespace["ws"] = ws
        return wb
    raise ValueError(f"Unsupported office file extension: {suffix}")


def _open_existing(path: Path, namespace: Dict[str, Any]) -> Any:
    """Opens an existing office file by suffix and binds it into the namespace."""
    suffix = path.suffix.lower()
    if suffix in (".pptx", ".pptm"):
        prs = namespace["Presentation"](str(path))
        namespace["prs"] = prs
        namespace["presentation"] = prs
        return prs
    if suffix in (".docx", ".docm"):
        doc = namespace["DocxDocument"](str(path))
        namespace["doc"] = doc
        return doc
    if suffix in (".xlsx", ".xlsm"):
        wb = namespace["load_workbook"](str(path), keep_vba=(suffix == ".xlsm"))
        namespace["wb"] = wb
        namespace["ws"] = wb.active
        return wb
    raise ValueError(f"Unsupported office file extension: {suffix}")


def _build_result(save_result: Dict[str, Any], save_path: Path) -> Dict[str, Any]:
    """Shapes a successful save result into the pptx_builder result contract."""
    return {
        "success": True,
        "workspace_file": str(save_path.resolve()),
        "context_files": [str(save_path.resolve())],
        "file_name": save_path.name,
    }


def tool_office_build(
    file: str = "",
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates a fresh Microsoft Office file (.pptx/.docx/.xlsx/.xlsm/.docm/.pptm) from a recipe.

    The root object is pre-bound into the recipe namespace under documented aliases:
    - pptx: `prs` / `presentation`
    - docx: `doc`
    - xlsx: `wb` (workbook) and `ws` (active sheet)

    Args:
        file: Target file name (default "document.docx"). The extension selects the format.
        recipe: Plain Python recipe code (no imports) mutating the root object.
        discussion_instance: Active discussion session instance.
        lollms_client_instance: Active client instance.
    """
    try:
        file_name = _sanitize_file_name(file, "document.docx")
        suffix = Path(file_name).suffix.lower()
        if suffix not in GUARDED_EXTENSIONS:
            return {
                "success": False,
                "error": (
                    f"Unsupported office file extension '{suffix}'. "
                    f"Supported: {', '.join(GUARDED_EXTENSIONS)}."
                ),
            }

        namespace = _build_recipe_namespace()
        try:
            root_obj = _create_root_object(suffix, namespace)
        except KeyError as missing_symbol:
            return {
                "success": False,
                "error": (
                    f"The library required for '{suffix}' files is not installed "
                    f"(missing symbol: {missing_symbol}). Ask the user to install it and retry."
                ),
            }

        exec_failure = _exec_recipe(recipe, namespace, "<office_recipe>")
        if exec_failure is not None:
            return exec_failure

        save_path = _resolve_working_dir(discussion_instance) / file_name
        save_result = _save_and_verify(root_obj, save_path)
        if not save_result["success"]:
            return save_result
        return _build_result(save_result, save_path)
    except Exception as unexpected_err:
        return {
            "success": False,
            "error": f"Unexpected failure building office file: {unexpected_err}",
        }


def tool_office_edit(
    file: str,
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Edits an existing Microsoft Office file in place via a recipe.

    The opened object is pre-bound into the recipe namespace under the same aliases
    as tool_office_build (`prs`/`presentation`, `doc`, `wb`/`ws`). The save is
    transactional: if the edited file fails validation, the pre-edit bytes are restored.
    """
    try:
        if not file or not str(file).strip():
            return {"success": False, "error": "No file provided for editing."}

        working_dir = _resolve_working_dir(discussion_instance)
        candidate = Path(str(file).strip())
        file_path = candidate if candidate.is_absolute() else working_dir / candidate
        if not file_path.is_file():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        suffix = file_path.suffix.lower()
        if suffix not in GUARDED_EXTENSIONS:
            return {
                "success": False,
                "error": (
                    f"Unsupported office file extension '{suffix}'. "
                    f"Supported: {', '.join(GUARDED_EXTENSIONS)}."
                ),
            }

        namespace = _build_recipe_namespace()
        try:
            root_obj = _open_existing(file_path, namespace)
        except KeyError as missing_symbol:
            return {
                "success": False,
                "error": (
                    f"The library required for '{suffix}' files is not installed "
                    f"(missing symbol: {missing_symbol}). Ask the user to install it and retry."
                ),
            }

        exec_failure = _exec_recipe(recipe, namespace, "<office_recipe>")
        if exec_failure is not None:
            return exec_failure

        save_result = _save_and_verify(root_obj, file_path)
        if not save_result["success"]:
            return save_result
        return _build_result(save_result, file_path)
    except Exception as unexpected_err:
        return {
            "success": False,
            "error": f"Unexpected failure editing office file: {unexpected_err}",
        }


def _read_pptx(path: Path) -> str:
    """Builds a markdown inventory of a presentation (slides, shapes, texts, tables)."""
    from pptx import Presentation

    prs = Presentation(str(path))
    lines = [f"# Presentation: {path.name}", f"Slides: {len(prs.slides)}", ""]
    for slide_idx, slide in enumerate(prs.slides, start=1):
        lines.append(f"## Slide {slide_idx}")
        shape_summaries = []
        for shape in slide.shapes:
            kind = shape.shape_type
            summary = f"- shape {shape.shape_id}: {kind}"
            if shape.has_text_frame:
                texts = [p.text for p in shape.text_frame.paragraphs if p.text.strip()]
                if texts:
                    summary += f" | text: {' / '.join(texts)}"
            if getattr(shape, "has_table", False):
                table = shape.table
                summary += f" | table {len(table.rows)}x{len(table.columns)}"
            shape_summaries.append(summary)
        lines.extend(shape_summaries if shape_summaries else ["- (empty slide)"])
        lines.append("")
    return "\n".join(lines)


def _read_docx(path: Path) -> str:
    """Builds a markdown inventory of a document (paragraphs and tables)."""
    from docx import Document

    doc = Document(str(path))
    lines = [f"# Document: {path.name}", ""]
    paragraph_lines = []
    for para_idx, para in enumerate(doc.paragraphs):
        if para.text.strip():
            paragraph_lines.append(f"- [{para_idx}] ({para.style.name}): {para.text}")
    lines.append("## Paragraphs")
    lines.extend(paragraph_lines if paragraph_lines else ["- (no paragraphs)"])
    lines.append("")
    lines.append(f"## Tables: {len(doc.tables)}")
    for table_idx, table in enumerate(doc.tables, start=1):
        lines.append(f"- table {table_idx}: {len(table.rows)} rows x {len(table.columns)} cols")
    return "\n".join(lines)


def _read_xlsx(path: Path) -> str:
    """Builds a markdown inventory of a workbook (sheets, dimensions, cell previews, formulas)."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), data_only=False)
    lines = [f"# Workbook: {path.name}", f"Sheets: {len(wb.sheetnames)}", ""]
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"## Sheet: {sheet_name} ({ws.max_row} rows x {ws.max_column} cols)")
        max_preview_rows = min(ws.max_row, 20)
        max_preview_cols = min(ws.max_column, 10)
        formula_cells = []
        for row in ws.iter_rows(min_row=1, max_row=max_preview_rows, max_col=max_preview_cols):
            for cell in row:
                if cell.value is None:
                    continue
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    formula_cells.append(f"{cell.coordinate}: {cell.value}")
        if formula_cells:
            lines.append("Formulas (first rows):")
            lines.extend(f"- {entry}" for entry in formula_cells[:20])
        lines.append("Preview (values, first 20 rows x 10 cols):")
        lines.append("```")
        for row in ws.iter_rows(min_row=1, max_row=max_preview_rows, max_col=max_preview_cols, values_only=True):
            lines.append(" | ".join("" if v is None else str(v) for v in row))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def tool_office_read(
    file: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reads an existing Microsoft Office file and returns a markdown inventory.

    The file is validated first; a corrupt file returns a structured failure
    instead of a crash. No recipe is executed and nothing is saved.
    """
    try:
        if not file or not str(file).strip():
            return {"success": False, "error": "No file provided for reading."}

        working_dir = _resolve_working_dir(discussion_instance)
        candidate = Path(str(file).strip())
        file_path = candidate if candidate.is_absolute() else working_dir / candidate
        if not file_path.is_file():
            return {"success": False, "error": f"File not found: {file_path}"}

        ok, reason = _validate_office_file(file_path)
        if not ok:
            return {
                "success": False,
                "error": (
                    f"Office Document Integrity Failure [{OFFICE_GUARD_FAILURE_CODE}]: "
                    f"'{file_path.name}' is corrupt and cannot be read safely.\n"
                    f"Reason: {reason}"
                ),
            }

        suffix = file_path.suffix.lower()
        if suffix in (".pptx", ".pptm"):
            inventory = _read_pptx(file_path)
        elif suffix in (".docx", ".docm"):
            inventory = _read_docx(file_path)
        elif suffix in (".xlsx", ".xlsm"):
            inventory = _read_xlsx(file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported office file extension '{suffix}'.",
            }

        return {
            "success": True,
            "file_name": file_path.name,
            "workspace_file": str(file_path.resolve()),
            "inventory": inventory,
        }
    except Exception as unexpected_err:
        return {
            "success": False,
            "error": f"Unexpected failure reading office file: {unexpected_err}",
        }