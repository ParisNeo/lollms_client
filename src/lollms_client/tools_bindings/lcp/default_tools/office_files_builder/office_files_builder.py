import re
import textwrap
import traceback
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ascii_colors import ASCIIColors

TOOL_LIBRARY_NAME = "Office Files Builder"
TOOL_LIBRARY_DESC = "Read, create and edit every kind of Microsoft Office file (.pptx, .docx, .xlsx, .xlsm, .docm, .pptm) via sandboxed recipe code. Pre-imported design system: python-pptx, python-docx, openpyxl symbols. Recipes are plain Python. Imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. Every save is verified with a strict three-layer OOXML integrity check before being committed to the workspace."
TOOL_LIBRARY_ICON = "🧰"

# --- Recipe sandbox limits (mirrors pptx_builder) ---
_MAX_RECIPE_CHARS = 60_000
_FORBIDDEN_TOKENS = (
    "eval(", "exec(", "compile(", "open(",
    "os.", "sys.", "subprocess", "shutil", "pathlib", "__",
    "globals(", "locals(", "vars(", "getattr(", "setattr(", "delattr(",
)

# --- Import whitelist (imports are intercepted at __import__, not token-banned) ---
_IMPORT_PREFIX_WHITELIST = ("pptx.", "docx.", "openpyxl.")
_SAFE_STDLIB_MODULES = frozenset({
    "math", "datetime", "random", "json", "statistics", "itertools",
    "functools", "re", "uuid", "calendar", "decimal", "fractions",
    "textwrap", "base64", "hashlib", "string", "collections",
})

# --- Workspace media support (read-only, workspace-contained) ---
_MEDIA_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tiff", ".ico",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv",
)
_MAX_MEDIA_SCAN = 5000
_MAX_MEDIA_LIST = 2000

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


def _safe_import(name: Any, *args: Any, **kwargs: Any) -> Any:
    """Whitelist gate for recipe imports (installed as __import__ in the sandbox).

    Allows pptx.*, docx.*, openpyxl.* derivatives and curated safe stdlib
    modules; anything else raises an ImportError naming the allowed list.
    Accepts the full __import__(name, globals, locals, fromlist, level) signature.
    """
    module_name = str(name)
    root_mod = module_name.split(".")[0]
    if root_mod in _SAFE_STDLIB_MODULES or any(
        module_name == prefix.rstrip(".") or module_name.startswith(prefix)
        for prefix in _IMPORT_PREFIX_WHITELIST
    ):
        import importlib
        return importlib.import_module(module_name)
    raise ImportError(
        f"Import '{module_name}' is not allowed in recipes. Allowed imports: pptx.*, docx.*, "
        "openpyxl.* and safe stdlib modules (math, datetime, random, json, statistics, "
        "itertools, functools, re, uuid, calendar, decimal, fractions, textwrap, base64, "
        "hashlib, string, collections). All design symbols are also pre-imported."
    )


def _build_safe_builtins() -> Dict[str, Any]:
    """Curated builtins for the recipe sandbox: no file/system/eval access,
    no reflective helpers, and __import__ replaced by the whitelist gate."""
    import builtins
    base = {k: v for k, v in vars(builtins).items() if not k.startswith("_")}
    for dangerous in ("open", "eval", "exec", "compile", "breakpoint", "input",
                      "exit", "quit", "help", "license", "globals", "locals",
                      "vars", "getattr", "setattr", "delattr"):
        base.pop(dangerous, None)
    base["__import__"] = _safe_import
    return base


def _check_recipe(recipe: str) -> None:
    """Rejects recipes containing forbidden tokens before execution."""
    for token in _FORBIDDEN_TOKENS:
        if token in recipe:
            raise ValueError(
                f"Forbidden token '{token}' in recipe. Recipes are plain Python: imports are "
                "allowed only from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib "
                "modules), with no file/system access and no dunder access. Prefer the "
                "pre-imported design symbols."
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


def _resolve_media_path(ref: str, working_dir: Path) -> Path:
    """Resolves a workspace-relative media reference to an absolute path.

    Search order: exact path under the workspace → recursive search by file
    name (subfolders included). Raises ValueError when the reference is
    absolute, escapes the workspace, or cannot be found. Read-only: never
    writes, never returns a path outside `working_dir`.
    """
    raw = str(ref or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("media() received an empty reference.")

    if "\\" in raw:
        raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]

    if Path(raw).is_absolute() or raw.startswith("/"):
        raise ValueError(
            f"Media reference '{ref}' must be workspace-relative, not absolute."
        )
    if ".." in Path(raw).parts:
        raise ValueError(
            f"Media reference '{ref}' must not contain '..' (workspace escape rejected)."
        )

    root = Path(working_dir).resolve()
    candidate = (root / raw).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(
            f"Media reference '{ref}' resolves outside the workspace and was rejected."
        )
    if candidate.is_file():
        return candidate

    # Recursive fallback: search subfolders by file name.
    target_name = Path(raw).name
    scanned = 0
    for hit in root.rglob(target_name):
        scanned += 1
        if scanned > _MAX_MEDIA_SCAN:
            break
        hit_resolved = hit.resolve()
        if hit_resolved.is_relative_to(root) and hit_resolved.is_file():
            return hit_resolved

    raise ValueError(
        f"Media file '{ref}' not found in workspace (searched recursively, "
        f"subfolders included). Use list_media() to see available files."
    )


def _list_media_files(working_dir: Path) -> List[str]:
    """Enumerates media files under the workspace (subfolders included).

    Returns workspace-relative posix-style paths, sorted, capped at
    _MAX_MEDIA_LIST entries. Never raises.
    """
    root = Path(working_dir).resolve()
    if not root.is_dir():
        return []
    found: List[str] = []
    scanned = 0
    for entry in root.rglob("*"):
        scanned += 1
        if scanned > _MAX_MEDIA_SCAN:
            break
        if entry.suffix.lower() in _MEDIA_EXTENSIONS:
            try:
                rel = entry.resolve().relative_to(root)
            except ValueError:
                continue
            found.append(rel.as_posix())
            if len(found) >= _MAX_MEDIA_LIST:
                break
    return sorted(found)


def _build_recipe_namespace(working_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Pre-populates the recipe namespace with per-format design symbols (lazy imports).

    When `working_dir` is provided, also injects the read-only workspace media
    helpers `media(ref) -> str` and `list_media() -> List[str]`.
    """
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

    # Read-only workspace media helpers (images/videos usable in recipes).
    if working_dir is not None:
        namespace["media"] = lambda ref: str(_resolve_media_path(ref, working_dir))
        namespace["list_media"] = lambda: _list_media_files(working_dir)

    return namespace


def _reopen_office_file(path: Path, suffix: str) -> Tuple[bool, str]:
    """Layer-3 format-aware re-open. Returns (ok, reason). Never raises.

    Beyond loading, verifies the document is non-degenerate: a pptx must
    contain at least one slide, a docx at least one paragraph or table.
    """
    try:
        if suffix in (".xlsx", ".xlsm"):
            from openpyxl import load_workbook
            if suffix == ".xlsm":
                load_workbook(path, keep_vba=True)
            else:
                load_workbook(path)
        elif suffix in (".docx", ".docm"):
            from docx import Document
            doc = Document(path)
            if len(doc.paragraphs) == 0 and len(doc.tables) == 0:
                return False, "re-open succeeded but the document is empty (no paragraphs, no tables)"
        elif suffix in (".pptx", ".pptm"):
            from pptx import Presentation
            prs = Presentation(path)
            if len(prs.slides) == 0:
                return False, "re-open succeeded but the presentation has zero slides"
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
    namespace["__builtins__"] = _build_safe_builtins()
    try:
        exec(compile(dedented, source_name, "exec"), namespace)
        return None
    except Exception:
        annotated = _annotate_traceback(traceback.format_exc(), dedented, source_name)
        if "save(" in annotated and "missing" in annotated:
            annotated += (
                "\nHint: the tool saves the file automatically after your recipe — "
                "remove the save() call."
            )
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


def _pick_effective_root(namespace: Dict[str, Any], suffix: str, original_root: Any) -> Any:
    """Returns the recipe-rebound root alias when the recipe replaced the
    pre-bound root object (e.g. `wb = Workbook()`), else the original root.

    This is what makes recipes that build their own document save THEIR
    object instead of the stale pre-bound one (the empty-excel defect).
    """
    aliases = {
        ".pptx": ("prs", "presentation"), ".pptm": ("prs", "presentation"),
        ".docx": ("doc",), ".docm": ("doc",),
        ".xlsx": ("wb",), ".xlsm": ("wb",),
    }
    for alias in aliases.get(suffix, ()):
        candidate = namespace.get(alias)
        if candidate is not None and candidate is not original_root:
            return candidate
    return original_root


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

    Recipes are plain Python. Imports are allowed ONLY from the whitelist
    (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design
    symbols are also pre-imported. The tool saves automatically; calling
    prs.save()/doc.save()/wb.save() inside the recipe is a no-op — remove it.

    Args:
        file: Target file name (default "document.docx"). The extension selects the format.
        recipe: Plain Python recipe code mutating the root object.
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

        working_dir = _resolve_working_dir(discussion_instance)
        namespace = _build_recipe_namespace(working_dir=working_dir)
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

        try:
            root_obj.save = lambda *args, **kwargs: None  # the tool performs the real validated save
        except Exception:
            pass

        exec_failure = _exec_recipe(recipe, namespace, "<office_recipe>")
        if exec_failure is not None:
            return exec_failure

        root_obj = _pick_effective_root(namespace, suffix, root_obj)

        save_path = working_dir / file_name
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
    as tool_office_build (`prs`/`presentation`, `doc`, `wb`/`ws`). Recipes are plain
    Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.*
    and safe stdlib modules); all core design symbols are also pre-imported. The
    tool saves automatically; calling prs.save()/doc.save()/wb.save() inside the
    recipe is a no-op — remove it. The save is transactional: if the edited file
    fails validation, the pre-edit bytes are restored.
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

        namespace = _build_recipe_namespace(working_dir=working_dir)
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

        try:
            root_obj.save = lambda *args, **kwargs: None  # the tool performs the real validated save
        except Exception:
            pass

        exec_failure = _exec_recipe(recipe, namespace, "<office_recipe>")
        if exec_failure is not None:
            return exec_failure

        root_obj = _pick_effective_root(namespace, suffix, root_obj)

        save_result = _save_and_verify(root_obj, file_path)
        if not save_result["success"]:
            return save_result
        return _build_result(save_result, file_path)
    except Exception as unexpected_err:
        return {
            "success": False,
            "error": f"Unexpected failure editing office file: {unexpected_err}",
        }


def build_office_presentation(
    file: str,
    slides: List[Dict[str, Any]],
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Builds a real .pptx from a structured slide specification.

    This is the structured entry point used by the chat dispatcher when an
    LLM emits an <artifact type="presentation"> tag: the markdown body is
    parsed into slide specs upstream, and this helper renders them through
    the same sandboxed recipe + integrity-verified save path as
    tool_office_build. Guarantees a valid OOXML file with real slides.
    """
    try:
        file_name = _sanitize_file_name(file, "presentation.pptx")
        if Path(file_name).suffix.lower() not in (".pptx", ".pptm"):
            file_name = f"{Path(file_name).stem}.pptx"

        working_dir = _resolve_working_dir(discussion_instance)
        namespace = _build_recipe_namespace(working_dir=working_dir)
        try:
            root_obj = _create_root_object(Path(file_name).suffix.lower(), namespace)
        except KeyError as missing_symbol:
            return {
                "success": False,
                "error": (
                    f"python-pptx is not installed (missing symbol: {missing_symbol}). "
                    "Ask the user to install it and retry."
                ),
            }

        prs = root_obj
        blank_layout = prs.slide_layouts[6]

        for spec in slides or []:
            slide = prs.slides.add_slide(blank_layout)
            title = str(spec.get("title", "")).strip()
            bullets = [str(b).strip() for b in spec.get("bullets", []) if str(b).strip()]
            body_text = str(spec.get("body", "")).strip()

            if title:
                title_box = slide.shapes.add_textbox(
                    namespace["Inches"](0.5), namespace["Inches"](0.4),
                    namespace["Inches"](12.3), namespace["Inches"](1.2),
                )
                title_frame = title_box.text_frame
                title_frame.text = title
                title_run = title_frame.paragraphs[0].runs[0]
                title_run.font.size = namespace["Pt"](36)
                title_run.font.bold = True

            if bullets or body_text:
                content_box = slide.shapes.add_textbox(
                    namespace["Inches"](0.8), namespace["Inches"](1.9),
                    namespace["Inches"](11.7), namespace["Inches"](5.0),
                )
                frame = content_box.text_frame
                frame.word_wrap = True
                first = True
                for bullet in bullets:
                    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
                    first = False
                    run = paragraph.add_run()
                    run.text = bullet
                    run.font.size = namespace["Pt"](20)
                if body_text and first:
                    paragraph = frame.paragraphs[0]
                    run = paragraph.add_run()
                    run.text = body_text
                    run.font.size = namespace["Pt"](20)

        save_path = working_dir / file_name
        save_result = _save_and_verify(root_obj, save_path)
        if not save_result["success"]:
            return save_result
        return _build_result(save_result, save_path)
    except Exception as unexpected_err:
        return {
            "success": False,
            "error": f"Unexpected failure building presentation: {unexpected_err}",
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


# ── Per-format tool aliases (thin wrappers over the generic office tools) ──
# The generate_slides incident showed the model routes far better to
# explicitly-named, format-specific tools than to one generic
# extension-dispatching tool. Each wrapper validates the file suffix matches
# its format family, then delegates to the single internal path — all
# sandbox/validation/rollback logic stays in tool_office_* (zero duplication).

_PPTX_SUFFIXES = (".pptx", ".pptm")
_DOCX_SUFFIXES = (".docx", ".docm")
_XLSX_SUFFIXES = (".xlsx", ".xlsm")


def _require_suffix(
    file: str,
    allowed_suffixes: Tuple[str, ...],
    tool_name: str,
) -> Optional[Dict[str, Any]]:
    """Returns a structured failure dict when the file suffix is outside the
    tool's format family, else None (proceed to delegate)."""
    actual = Path(str(file or "")).suffix.lower()
    if actual not in allowed_suffixes:
        return {
            "success": False,
            "error": (
                f"{tool_name} requires a {'/'.join(allowed_suffixes)} file, "
                f"got '{actual or '(no extension)'}'. Use the tool matching "
                f"this file's format instead."
            ),
            "code": "wrong_file_type_for_tool",
        }
    return None


def tool_build_powerpoint(
    file: str = "presentation.pptx",
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates a new PowerPoint presentation (.pptx/.pptm) from a sandboxed python-pptx recipe.

    The root object is pre-bound as `prs` (alias `presentation`). Available
    symbols: Presentation, Inches, Pt, Emu, RGBColor, MSO_SHAPE, MSO_CONNECTOR,
    MSO_ANCHOR, PP_ALIGN, qn. Recipes are plain Python: imports are allowed ONLY
    from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all
    core design symbols are also pre-imported. The tool saves automatically;
    calling prs.save() inside the recipe is a no-op — remove it.
    """
    suffix_failure = _require_suffix(file, _PPTX_SUFFIXES, "tool_build_powerpoint")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_build(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_edit_powerpoint(
    file: str,
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Edits an existing PowerPoint presentation (.pptx/.pptm) in place via a sandboxed python-pptx recipe.

    The opened presentation is pre-bound as `prs` (alias `presentation`). Recipes
    are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*,
    openpyxl.* and safe stdlib modules); all core design symbols are also
    pre-imported. The tool saves automatically; calling prs.save() inside the
    recipe is a no-op — remove it. The save is transactional: on validation
    failure the pre-edit bytes are restored.
    """
    suffix_failure = _require_suffix(file, _PPTX_SUFFIXES, "tool_edit_powerpoint")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_edit(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_read_powerpoint(
    file: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reads an existing PowerPoint presentation (.pptx/.pptm) and returns a markdown inventory (slides, shapes, texts, tables)."""
    suffix_failure = _require_suffix(file, _PPTX_SUFFIXES, "tool_read_powerpoint")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_read(
        file=file,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_build_word(
    file: str = "document.docx",
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates a new Word document (.docx/.docm) from a sandboxed python-docx recipe.

    The root object is pre-bound as `doc`. Available symbols: DocxDocument,
    DocxInches, DocxPt, DocxRGBColor, WD_ALIGN_PARAGRAPH. Recipes are plain
    Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*,
    openpyxl.* and safe stdlib modules); all core design symbols are also
    pre-imported. The tool saves automatically; calling doc.save() inside the
    recipe is a no-op — remove it.
    """
    suffix_failure = _require_suffix(file, _DOCX_SUFFIXES, "tool_build_word")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_build(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_edit_word(
    file: str,
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Edits an existing Word document (.docx/.docm) in place via a sandboxed python-docx recipe.

    The opened document is pre-bound as `doc`. Recipes are plain Python: imports
    are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe
    stdlib modules); all core design symbols are also pre-imported. The tool
    saves automatically; calling doc.save() inside the recipe is a no-op —
    remove it. The save is transactional: on validation failure the pre-edit
    bytes are restored.
    """
    suffix_failure = _require_suffix(file, _DOCX_SUFFIXES, "tool_edit_word")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_edit(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_read_word(
    file: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reads an existing Word document (.docx/.docm) and returns a markdown inventory (paragraphs and tables)."""
    suffix_failure = _require_suffix(file, _DOCX_SUFFIXES, "tool_read_word")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_read(
        file=file,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_build_excel(
    file: str = "workbook.xlsx",
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Creates a new Excel workbook (.xlsx/.xlsm) from a sandboxed openpyxl recipe.

    The root objects are pre-bound as `wb` (workbook) and `ws` (active sheet).
    Available symbols: Workbook, load_workbook, PieChart, LineChart, BarChart,
    Reference, Font, PatternFill, Alignment, Border, Side, get_column_letter.
    Recipes are plain Python: imports are allowed ONLY from the whitelist
    (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design
    symbols are also pre-imported. The tool saves automatically; calling
    wb.save() inside the recipe is a no-op — remove it.
    """
    suffix_failure = _require_suffix(file, _XLSX_SUFFIXES, "tool_build_excel")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_build(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_edit_excel(
    file: str,
    recipe: str = "",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Edits an existing Excel workbook (.xlsx/.xlsm) in place via a sandboxed openpyxl recipe.

    The opened workbook is pre-bound as `wb` and its active sheet as `ws`
    (.xlsm files are opened with keep_vba=True). Recipes are plain Python:
    imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.*
    and safe stdlib modules); all core design symbols are also pre-imported.
    The tool saves automatically; calling wb.save() inside the recipe is a
    no-op — remove it. The save is transactional: on validation failure the
    pre-edit bytes are restored.
    """
    suffix_failure = _require_suffix(file, _XLSX_SUFFIXES, "tool_edit_excel")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_edit(
        file=file, recipe=recipe,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def tool_read_excel(
    file: str,
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reads an existing Excel workbook (.xlsx/.xlsm) and returns a markdown inventory (sheets, dimensions, cell previews, formulas)."""
    suffix_failure = _require_suffix(file, _XLSX_SUFFIXES, "tool_read_excel")
    if suffix_failure is not None:
        return suffix_failure
    return tool_office_read(
        file=file,
        discussion_instance=discussion_instance,
        lollms_client_instance=lollms_client_instance,
    )


def get_tools() -> List[Dict[str, Any]]:
    """
    Returns the tool specifications for the office files builder toolset.
    Each spec includes name, description, parameters, output, and callable.
    """
    return [
        {
            "name": "tool_office_build",
            "description": "Creates a fresh Microsoft Office file (.pptx/.docx/.xlsx/.xlsm/.docm/.pptm) from a recipe. Recipes are plain Python. Imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules: math, datetime, random, json, statistics, itertools, functools, re, uuid, calendar, decimal, fractions, textwrap, base64, hashlib, string, collections); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. The sandbox namespace also provides media('relative/path.ext') to resolve workspace images/videos (recursive, subfolders included) and list_media() to enumerate them. Saved files are validated against strict OOXML rules and rolled back on failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Target file name (default 'document.docx'). The extension selects the format.",
                    "optional": True,
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the root object (prs/presentation for pptx, doc for docx, wb/ws for xlsx). Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_office_build,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_office_edit",
            "description": "Edits an existing Microsoft Office file in place via a recipe. The opened object is pre-bound into the recipe namespace under the same aliases as tool_office_build. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. The namespace also provides media('relative/path.ext') to resolve workspace images/videos (recursive, subfolders included) and list_media() to enumerate them. The save is transactional: if the edited file fails validation, the pre-edit bytes are restored.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "File to edit (relative to workspace or absolute path).",
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the opened object. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_office_edit,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_office_read",
            "description": "Reads an existing Microsoft Office file and returns a markdown inventory. The file is validated first; a corrupt file returns a structured failure instead of a crash. No recipe is executed and nothing is saved.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "File to read (relative to workspace or absolute path).",
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "inventory", "type": "str", "description": "Markdown inventory of the file contents."},
                {"name": "file_name", "type": "str", "description": "Name of the file read."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_office_read,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "build_office_presentation",
            "description": "Internal structured entry point: renders a slide specification list into a real .pptx via the sandboxed python-pptx recipe path. Not exposed to the LLM tool grammar; invoked by the chat dispatcher for <artifact type=\"presentation\"> tags.",
            "parameters": [],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": build_office_presentation,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_build_powerpoint",
            "description": "Creates a new PowerPoint presentation (.pptx/.pptm) from a sandboxed python-pptx recipe. The root object is pre-bound as `prs` (alias `presentation`). Symbols: Presentation, Inches, Pt, Emu, RGBColor, MSO_SHAPE, MSO_CONNECTOR, MSO_ANCHOR, PP_ALIGN, qn, plus media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Target .pptx/.pptm file name (default 'presentation.pptx').",
                    "optional": True,
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the pre-bound `prs` presentation object. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_build_powerpoint,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_edit_powerpoint",
            "description": "Edits an existing PowerPoint presentation (.pptx/.pptm) in place via a sandboxed python-pptx recipe. The opened presentation is pre-bound as `prs`. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. The namespace also provides media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Transactional save: pre-edit bytes restored on validation failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .pptx/.pptm file to edit (relative to workspace or absolute path).",
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the opened `prs` presentation object. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_edit_powerpoint,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_read_powerpoint",
            "description": "Reads an existing PowerPoint presentation (.pptx/.pptm) and returns a markdown inventory (slides, shapes, texts, tables). Validates the file first; corrupt files return a structured failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .pptx/.pptm file to read (relative to workspace or absolute path).",
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "inventory", "type": "str", "description": "Markdown inventory of the presentation contents."},
                {"name": "file_name", "type": "str", "description": "Name of the file read."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_read_powerpoint,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_build_word",
            "description": "Creates a new Word document (.docx/.docm) from a sandboxed python-docx recipe. The root object is pre-bound as `doc`. Symbols: DocxDocument, DocxInches, DocxPt, DocxRGBColor, WD_ALIGN_PARAGRAPH, plus media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Target .docx/.docm file name (default 'document.docx').",
                    "optional": True,
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the pre-bound `doc` document object. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_build_word,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_edit_word",
            "description": "Edits an existing Word document (.docx/.docm) in place via a sandboxed python-docx recipe. The opened document is pre-bound as `doc`. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. The namespace also provides media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Transactional save: pre-edit bytes restored on validation failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .docx/.docm file to edit (relative to workspace or absolute path).",
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the opened `doc` document object. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_edit_word,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_read_word",
            "description": "Reads an existing Word document (.docx/.docm) and returns a markdown inventory (paragraphs and tables). Validates the file first; corrupt files return a structured failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .docx/.docm file to read (relative to workspace or absolute path).",
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "inventory", "type": "str", "description": "Markdown inventory of the document contents."},
                {"name": "file_name", "type": "str", "description": "Name of the file read."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_read_word,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_build_excel",
            "description": "Creates a new Excel workbook (.xlsx/.xlsm) from a sandboxed openpyxl recipe. The root objects are pre-bound as `wb` (workbook) and `ws` (active sheet). Symbols: Workbook, load_workbook, PieChart, LineChart, BarChart, Reference, Font, PatternFill, Alignment, Border, Side, get_column_letter, plus media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Target .xlsx/.xlsm file name (default 'workbook.xlsx').",
                    "optional": True,
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the pre-bound `wb`/`ws` objects. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_build_excel,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_edit_excel",
            "description": "Edits an existing Excel workbook (.xlsx/.xlsm) in place via a sandboxed openpyxl recipe. The opened workbook is pre-bound as `wb` and its active sheet as `ws` (.xlsm opened with keep_vba=True). Recipes are plain Python: imports are allowed ONLY from the whitelist (pptx.*, docx.*, openpyxl.* and safe stdlib modules); all core design symbols are pre-imported. Do NOT call save() — the tool saves and validates automatically. The namespace also provides media('relative/path.ext') to resolve workspace images/videos and list_media() to enumerate them. Transactional save: pre-edit bytes restored on validation failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .xlsx/.xlsm file to edit (relative to workspace or absolute path).",
                },
                {
                    "name": "recipe",
                    "type": "str",
                    "description": "Plain Python recipe code mutating the opened `wb`/`ws` objects. Imports allowed only from the whitelist (pptx.*, docx.*, openpyxl.*, safe stdlib). Do NOT call save() — the tool saves automatically.",
                    "optional": True,
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the saved file."},
                {"name": "context_files", "type": "list", "description": "List of context files (the saved file)."},
                {"name": "file_name", "type": "str", "description": "Name of the saved file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_edit_excel,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
        {
            "name": "tool_read_excel",
            "description": "Reads an existing Excel workbook (.xlsx/.xlsm) and returns a markdown inventory (sheets, dimensions, cell previews, formulas). Validates the file first; corrupt files return a structured failure.",
            "parameters": [
                {
                    "name": "file",
                    "type": "str",
                    "description": "Existing .xlsx/.xlsm file to read (relative to workspace or absolute path).",
                },
            ],
            "output": [
                {"name": "success", "type": "bool", "description": "Whether the operation succeeded."},
                {"name": "inventory", "type": "str", "description": "Markdown inventory of the workbook contents."},
                {"name": "file_name", "type": "str", "description": "Name of the file read."},
                {"name": "workspace_file", "type": "str", "description": "Absolute path to the file."},
                {"name": "error", "type": "str", "description": "Error message if failed."},
            ],
            "callable": tool_read_excel,
            "_source": "office_files_builder",
            "_source_file": __file__,
        },
    ]