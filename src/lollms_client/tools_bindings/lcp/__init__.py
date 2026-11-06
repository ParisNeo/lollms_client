import sys
import os
import io
import types
import uuid
import importlib.util
import ast
import difflib
import re
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable, Tuple

from builtins import compile as _native_compile

from lollms_client.lollms_tools_binding import LollmsToolBinding
from ascii_colors import ASCIIColors, trace_exception

BindingName = "LCPBinding"

# ── OFFICE-FAMILY LIBRARY GATE (fail-closed) ──
# Per-software toolsets: without their explicit session flag they are EXCLUDED
# from discovery/mounting. This prevents office toolsets from auto-appearing in
# every session (the as_is leak class).
_OFFICE_LIBRARY_STEMS = frozenset({
    "pptx_builder", "docx_mastery", "pptx_mastery", "xlsx_mastery",
    "powerpoint_tools", "word_tools", "excel_tools",
    # PDF toolset — same explicit-flag gating contract as the OOXML toolsets.
    # Registration in both this set and _PER_SOFTWARE_GATES means the mount
    # is refused unless enable_pdf is delivered via set_session_flags().
    "pdf_tools",
})
# Per-software stems gate on their OWN flag only (fail-closed when absent).
_PER_SOFTWARE_GATES = {
    "powerpoint_tools": ("enable_powerpoint",),
    "word_tools": ("enable_word",),
    "excel_tools": ("enable_excel",),
    # PDF toolkit — same fail-closed per-software contract as the OOXML toolsets.
    # The chat layer drives enable_pdf from the explicit chat flag AND from
    # file-type auto-detection (.pdf found in workspace/artefacts).
    "pdf_tools": ("enable_pdf",),
}

# Per-software stems also auto-activate when a matching file type is present
# in the workspace. This lets the LLM handle existing office files "as is"
# even when the corresponding chat flag is off. When no matching file exists,
# the per-software flag remains the sole gate (fail-closed).
_PER_SOFTWARE_FILE_TYPES = {
    "word_tools": (".docx", ".docm"),
    "excel_tools": (".xlsx", ".xlsm"),
    "powerpoint_tools": (".pptx", ".pptm"),
    "pdf_tools": (".pdf",),
}

# ── DOCUMENT-FAMILY LIBRARY GATE (file-type aware) ──
# Document-family toolsets (as_is_document_tools, document_editor) are admitted
# based on which file types are ACTUALLY PRESENT in the workspace. Dropping a
# PDF into the workspace mounts the PDF-capable document toolset automatically —
# no enable_office_files flag required. The library declares the extensions it
# handles via a module-level attribute:
#
#     TOOL_LIBRARY_FILE_TYPES = [".pdf", ".docx", ".xlsx", ".pptx"]
#
# When a stem in this family declares file types AND at least one of those
# extensions is found in the workspace, it is admitted. When a stem in this
# family does NOT declare file types, it falls back to the legacy office gate
# (enable_office_files), preserving backward compatibility.
_DOCUMENT_LIBRARY_STEMS = frozenset({
    "as_is_document_tools", "document_editor",
    # `pdf_tools` shares the file-type gate: declares TOOL_LIBRARY_FILE_TYPES
    # = [".pdf"] in step 2 and is admitted only when a PDF is present (or
    # enable_pdf / enable_office_files are set).
    "pdf_tools",
    # `classic_document_tools` (step 2) declares the classic text extensions
    # (.txt/.md/.py/.js/... etc.) and auto-mounts when any is present. No flag
    # is required — file presence is the entire admission criterion.
    "classic_document_tools",
    # `markdown_tools` declares TOOL_LIBRARY_FILE_TYPES = [".md", ".markdown"]
    # and auto-mounts whenever a Markdown file is present in the workspace.
    "markdown_tools",
})

# Directory names skipped while scanning the workspace for file-type detection.
_SKIP_SCAN_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv",
    "venv", ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
})
# Ceiling on how deep the workspace scan goes (safety for very large trees).
_WORKSPACE_SCAN_MAX_DEPTH = 4


def _office_library_admitted(
    stem: str,
    session_flags: Optional[Dict[str, Any]],
    workspace_file_types: Optional[Any] = None,
) -> bool:
    """Resolves office-family admission for a per-software library stem.

    Per-software stems (powerpoint_tools/word_tools/excel_tools) are admitted
    when EITHER:
      1. Their own session flag is truthy (explicit opt-in), OR
      2. A file with a matching extension is present in the workspace
         (auto-activation so existing office files can be handled "as is").

    When neither condition holds, the stem is fail-closed (excluded).
    """
    if stem not in _OFFICE_LIBRARY_STEMS:
        return True
    flags = session_flags if isinstance(session_flags, dict) else {}
    own_gates = _PER_SOFTWARE_GATES.get(stem)
    if own_gates is not None:
        if any(bool(flags.get(g)) for g in own_gates):
            return True
        # File-presence auto-activation: a matching workspace file admits the
        # toolset even when the chat flag is off.
        declared = _PER_SOFTWARE_FILE_TYPES.get(stem, ())
        if declared:
            present = {str(t).lower() for t in (workspace_file_types or set())}
            if any(ext.lower() in present for ext in declared):
                return True
        return False
    office_enabled = bool(flags.get("enable_office_files")) or any(
        bool(flags.get(g)) for gates in _PER_SOFTWARE_GATES.values() for g in gates
    )
    return office_enabled


def _document_library_admitted(
    stem: str,
    session_flags: Optional[Dict[str, Any]],
    workspace_file_types: Optional[Any],
    declared_file_types: Optional[List[str]],
) -> bool:
    """Resolves document-family admission for a library stem.

    Admission rules (any one suffices):
      1. `enable_office_files` is truthy (backward-compat escape hatch).
      2. The library declares `TOOL_LIBRARY_FILE_TYPES` AND at least one of
         those extensions is present in the workspace.

    When the library does not declare any file types, this falls back to the
    legacy office gate so existing behaviour is preserved.
    """
    if stem not in _DOCUMENT_LIBRARY_STEMS:
        return True

    flags = session_flags if isinstance(session_flags, dict) else {}
    if bool(flags.get("enable_office_files")):
        return True

    declared = {
        f".{str(t).lower().lstrip('.')}"
        for t in (declared_file_types or [])
        if t
    }
    if not declared:
        return _office_library_admitted(stem, session_flags, workspace_file_types)

    present = {str(t).lower() for t in (workspace_file_types or set())}
    return bool(declared & present)


class LCPBinding(LollmsToolBinding):
    """
    Local LollmsCommunicationProtocol (LCP) Binding.
    
    PHILOSOPHY:
    - Tools are agnostic Python scripts.
    - Tools receive the unified host runtime via the reserved `env` parameter — a
never-None façade (`env.llm`, `env.tti`, `*_model_profiles`, `*_switch`,
`env.client`, `env.discussion`, `env.tool_context`, `env.custom`) whose
sub-attributes must be feature-checked for None.
    - Tools operate on files in the current working directory.
    - The Binding handles environment setup (CWD, sync) transparently.
    - Host-Configurable parameters are invisible to the LLM and enforced by the host app.
    """

    # ── Sandbox gated-task registry (single source of truth) ──
    # Every confirmation prompt a tool may raise MUST declare one of these
    # categories. "Always allow" answers are recorded per (discussion,
    # category) so a user accepting process execution in discussion A never
    # silently unlocks it in discussion B.
    GATED_TASK_CATEGORIES: Tuple[str, ...] = (
        "process_execution",   # running shells / subprocesses (system_shell, python kernel)
        "internet_access",     # HTTP(S) fetches, websearch, internet_import
        "file_write",          # writing/deleting files outside the safe workspace
        "filesystem_delete",   # destructive filesystem ops (rm, rmtree)
        "external_api",        # third-party authenticated API calls
        "computer_use",        # desktop/UI automation (clicks, keystrokes)
        "code_execution",      # executing LLM-authored code (execute_python)
    )

    def __init__(self, host_tool_configs: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs: Any):
        super().__init__(binding_name="LCP")

        # Host-provided configurations for tools (e.g., {"system_shell": {"autonomy_level": "safe"}})
        self.host_tool_configs: Dict[str, Dict[str, Any]] = host_tool_configs or {}
        self.confirm_handler: Optional[Callable] = kwargs.get("confirm_handler")
        self.user_interaction_handler: Optional[Callable] = kwargs.get("user_interaction_handler")

        # Per-discussion "always allow" gate store.
        #   key   -> discussion id (hashable), or "__host__" when no discussion
        #           object is available (personality chat / bare tool calls)
        #   value -> set of GATED_TASK_CATEGORIES the user has permanently
        #           accepted for that discussion
        # This structure NEVER crosses discussion boundaries: a new discussion
        # starts with an empty set, and reset_discussion_gates() clears it.
        self._discussion_gates: Dict[Any, set] = {}
        self._no_discussion_scope: str = "__host__"

        # Resolve Direct Tool Files (must resolve first: explicit tool_files implies an empty folder scan)
        self.tool_files: List[Path] = []
        files_input = kwargs.get("tool_files")
        if files_input:
            if isinstance(files_input, (str, Path)):
                self.tool_files.append(Path(files_input))
            elif isinstance(files_input, list):
                for f in files_input:
                    self.tool_files.append(Path(f))

        # Resolve Multi-Folder Config
        self.tools_folders: List[Path] = []
        folders_input = kwargs.get("tools_folders") or kwargs.get("tools_folder_path")
        if folders_input:
            if isinstance(folders_input, (str, Path)):
                self.tools_folders.append(Path(folders_input))
            elif isinstance(folders_input, list):
                for f in folders_input:
                    self.tools_folders.append(Path(f))
        elif not self.tool_files:
            self.tools_folders.append(Path(__file__).parent / "default_tools")

        self._load_flag_kwargs: Dict[str, Any] = kwargs

        # Session flags (set by the host chat layer via set_session_flags).
        # Office-family and document-family libraries are gated against these;
        # absent flags are fail-closed.
        self.session_flags: Dict[str, Any] = {}

        # Workspace folders used for file-type detection. Document-family
        # toolsets mount themselves when a matching file is found (e.g.
        # as_is_document_tools auto-loads when a .pdf exists). Defaults to CWD,
        # overridable via `workspace_paths` / `workspace_path` kwargs.
        self.workspace_paths: List[Path] = self._resolve_workspace_paths(kwargs)

        # Cached set of lowercase extensions (".pdf", ".docx", ...) detected
        # in the workspace. Populated eagerly here and refreshed on every
        # set_session_flags() call.
        self._workspace_file_types: set = set()
        self._refresh_workspace_file_types()

        self.discovered_tools: List[Dict[str, Any]] = []
        self._dynamic_tool_modules: Dict[str, types.ModuleType] = {}
        self._discover_local_tools()

    def set_session_flags(self, flags: Optional[Dict[str, Any]]) -> None:
        """Registers session feature flags and re-runs discovery so office- and
        document-family libraries are admitted/excluded per the new flag state.

        The workspace is re-scanned first because document-family admission is
        driven by file presence (e.g. a .pdf in the workspace auto-mounts the
        PDF-capable document toolset).
        """
        self.session_flags = dict(flags) if isinstance(flags, dict) else {}
        self._refresh_workspace_file_types()
        self._discover_local_tools()

    def refresh_workspace_file_types(self) -> None:
        """Public hook: re-scans the workspace and re-runs discovery so newly
        added/removed files trigger mount/unmount of document-family toolsets
        without requiring a flag change."""
        self._refresh_workspace_file_types()
        self._discover_local_tools()

    @property
    def tools_folder_path(self) -> Optional[Path]:
        return self.tools_folders[0] if self.tools_folders else None

    def _parse_tool_via_ast(self, py_file_path: Path) -> List[Dict[str, Any]]:
        try:
            code_text = py_file_path.read_text(encoding="utf-8")
            tree = ast.parse(code_text)

            def _iter_functions_ordered(node):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        yield child
                    else:
                        yield from _iter_functions_ordered(child)

            tools = []
            for node in _iter_functions_ordered(tree):
                if node.name.startswith("tool_"):
                    tool_def = self._extract_single_tool_schema(node, py_file_path.stem)
                    if tool_def:
                        tools.append(tool_def)

            return tools if tools else None
        except Exception as e:
            ASCIIColors.warning(f"AST parse failed for '{py_file_path.name}': {e}. Attempting regex fallback.")
            return self._parse_tool_via_regex(py_file_path)

    def _parse_tool_via_regex(self, py_file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Lightweight regex-based fallback when AST parsing fails (e.g., partially
        syntactically invalid files). Extracts 'tool_' function names, docstrings,
        and simple type-annotated parameters. Never raises.
        """
        try:
            code_text = py_file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as read_err:
            ASCIIColors.error(f"[LCP] Cannot read tool file '{py_file_path.name}': {read_err}")
            return None

        func_pattern = re.compile(
            r'^[ \t]*def\s+(tool_\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:',
            re.MULTILINE
        )
        tools: List[Dict[str, Any]] = []

        for func_match in func_pattern.finditer(code_text):
            tool_name = func_match.group(1)
            params_str = func_match.group(2)

            body_start = func_match.end()
            docstring = ""
            doc_match = re.search(
                r'"""(.*?)"""', code_text[body_start:body_start + 4000], re.DOTALL
            )
            if doc_match:
                docstring = doc_match.group(1).strip()
            description = docstring.split("\n\n")[0].strip() if docstring else f"Executes {tool_name}."

            properties: Dict[str, Any] = {}
            required: List[str] = []

            for param_raw in params_str.split(","):
                param = param_raw.strip()
                if not param or param.startswith("*") or ":" not in param:
                    if param and param.split("=")[0].strip() and param.split("=")[0].strip() not in ("args", "kwargs", "discussion_instance", "lollms_client_instance", "tool_context", "user_interaction_handler", "env"):
                        p_name = param.split("=")[0].strip()
                        properties[p_name] = {"type": "string", "description": f"Parameter '{p_name}'"}
                        if "=" not in param:
                            required.append(p_name)
                    continue

                p_name, p_type_raw = param.split(":", 1)
                p_name = p_name.strip()
                if not p_name or p_name in ("args", "kwargs", "discussion_instance", "lollms_client_instance", "tool_context", "user_interaction_handler", "env"):
                    continue

                p_type_raw = p_type_raw.strip().lower()
                p_type = "string"
                if "int" in p_type_raw:
                    p_type = "integer"
                elif "float" in p_type_raw or "number" in p_type_raw:
                    p_type = "number"
                elif "bool" in p_type_raw:
                    p_type = "boolean"
                elif "list" in p_type_raw or "array" in p_type_raw:
                    p_type = "array"
                elif "dict" in p_type_raw or "object" in p_type_raw:
                    p_type = "object"

                is_optional = "=" in param
                entry = {"type": p_type, "description": f"Parameter '{p_name}'"}
                if is_optional:
                    default_match = re.search(r'=\s*(.+)$', param)
                    if default_match:
                        entry["default"] = default_match.group(1).strip().strip("'\"")
                else:
                    required.append(p_name)
                properties[p_name] = entry

            tools.append({
                "name": tool_name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                },
                "_python_file_path": str(py_file_path.resolve()),
                "_parsed_via_regex_fallback": True
            })
            ASCIIColors.success(f"[LCP] Regex fallback extracted tool '{tool_name}' from '{py_file_path.name}'.")

        return tools if tools else None

    def _extract_single_tool_schema(self, func_node: ast.FunctionDef, file_stem: str) -> Optional[Dict[str, Any]]:
        tool_name = func_node.name
        docstring = ast.get_docstring(func_node) or ""
        description = docstring.strip().split("\n\n")[0].strip() if docstring else "No description provided."

        doc_params = {}
        if docstring:
            for line in docstring.splitlines():
                m = re.match(r'^(?:[-\*\d\.]+\s*)?([a-zA-Z0-9_]+)\s*(?:\(([^)]+)\))?\s*[:\-]\s*(.+)', line.strip())
                if m:
                    doc_params[m.group(1).strip()] = m.group(3).strip()

        properties = {}
        required = []
        args_list = func_node.args.args
        defaults_list = func_node.args.defaults
        defaults_offset = len(args_list) - len(defaults_list) if defaults_list else len(args_list)

        for idx, arg in enumerate(args_list):
            arg_name = arg.arg
            if arg.arg in ("args", "kwargs", "discussion_instance", "lollms_client_instance", "tool_context", "user_interaction_handler", "env"):
                continue

            arg_type = "string"
            if arg.annotation:
                anno_str = ast.unparse(arg.annotation).strip().lower()
                if "int" in anno_str: arg_type = "integer"
                elif "float" in anno_str or "number" in anno_str: arg_type = "number"
                elif "bool" in anno_str: arg_type = "boolean"
                elif "list" in anno_str or "array" in anno_str: arg_type = "array"
                elif "dict" in anno_str or "object" in anno_str: arg_type = "object"

            has_default = idx >= defaults_offset
            default_val = None
            if has_default and defaults_list:
                default_node = defaults_list[idx - defaults_offset]
                try:
                    default_val = ast.literal_eval(default_node)
                except:
                    default_val = ast.unparse(default_node).strip("'\"")

            desc = doc_params.get(arg_name, f"Parameter '{arg_name}'")
            properties[arg_name] = {"type": arg_type, "description": desc}
            if has_default:
                properties[arg_name]["default"] = default_val
            else:
                required.append(arg_name)

        if not properties and docstring:
            for line in docstring.splitlines():
                m = re.match(r'^(?:[-\*\d\.]+\s*)?([a-zA-Z0-9_]+)\s*(?:\(([^)]+)\))?\s*[:\-]\s*(.+)', line.strip())
                if m:
                    p_name = m.group(1).strip()
                    p_type_raw = (m.group(2) or "string").lower().strip()
                    p_desc = m.group(3).strip()
                    if p_name.lower() in ("args", "parameters", "returns", "example"): continue

                    p_type = "string"
                    if "int" in p_type_raw: p_type = "integer"
                    elif "float" in p_type_raw: p_type = "number"
                    elif "bool" in p_type_raw: p_type = "boolean"

                    properties[p_name] = {"type": p_type, "description": p_desc}
                    if "optional" not in p_type_raw and "default" not in p_type_raw:
                        required.append(p_name)

        return {
            "name": tool_name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "_python_file_path": str(Path(__file__).parent / "default_tools" / file_stem / f"{file_stem}.py")
        }

    def _library_load_flag(self, py_file: Path) -> Optional[str]:
        """
        Returns the TOOL_LIBRARY_LOAD_FLAG name declared by a library file,
        or None when the library is not gated.
        """
        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            return None

        match = re.search(
            r'^TOOL_LIBRARY_LOAD_FLAG\s*=\s*["\']([A-Za-z0-9_]+)["\']',
            code_text,
            re.MULTILINE,
        )
        return match.group(1) if match else None

    def _read_library_file_types(self, py_file: Path) -> List[str]:
        """Returns the extensions a library declares via TOOL_LIBRARY_FILE_TYPES.

        Example declaration inside a library module:

            TOOL_LIBRARY_FILE_TYPES = [".pdf", ".docx"]

        Returns an empty list when the library is not file-type-gated.
        """
        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            return []
        match = re.search(
            r'^TOOL_LIBRARY_FILE_TYPES\s*=\s*([\[\(].*?[\]\)])',
            code_text,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            return []
        return [ext.lower() for ext in re.findall(r'["\']([^"\']+)["\']', match.group(1))]

    def _resolve_workspace_paths(self, kwargs: Dict[str, Any]) -> List[Path]:
        """Resolves the folders to scan for workspace file-type detection."""
        raw = kwargs.get("workspace_paths") or kwargs.get("workspace_path")
        candidates: List[Path] = []
        if isinstance(raw, (str, Path)):
            candidates.append(Path(raw))
        elif isinstance(raw, (list, tuple)):
            for entry in raw:
                try:
                    candidates.append(Path(entry))
                except Exception:
                    continue
        if not candidates:
            candidates.append(Path.cwd())
        return [p.resolve() for p in candidates if p.exists() and p.is_dir()]

    def _refresh_workspace_file_types(self) -> None:
        """(Re-)populates `_workspace_file_types` by scanning workspace folders.

        Bounded by depth and VCS/cache-directory exclusions so huge trees do
        not slow down discovery.
        """
        detected: set = set()
        for root in self.workspace_paths:
            try:
                self._scan_workspace_for_file_types(root, depth=0, detected=detected)
            except Exception:
                continue
        self._workspace_file_types = detected

    def _scan_workspace_for_file_types(self, folder: Path, depth: int, detected: set) -> None:
        """Recursive helper for `_refresh_workspace_file_types`. Never raises."""
        if depth > _WORKSPACE_SCAN_MAX_DEPTH:
            return
        try:
            for entry in folder.iterdir():
                try:
                    if entry.is_dir():
                        if entry.name in _SKIP_SCAN_DIRS or entry.name.startswith("."):
                            continue
                        self._scan_workspace_for_file_types(entry, depth + 1, detected)
                    elif entry.is_file() and entry.suffix:
                        detected.add(entry.suffix.lower())
                except OSError:
                    continue
        except OSError:
            return

    def _is_library_admitted(self, py_file: Path) -> bool:
        """Combines office- and document-family admission for a single library
        file. Returns True when the library may be loaded/mounted."""
        stem = py_file.stem

        if not _office_library_admitted(
            stem, self.session_flags, self._workspace_file_types
        ):
            if stem in _OFFICE_LIBRARY_STEMS:
                ASCIIColors.debug(
                    f"[LCP] Library '{stem}' skipped: office session flag not "
                    f"enabled and no matching workspace file "
                    f"(detected={sorted(self._workspace_file_types)})."
                )
            return False

        if stem in _DOCUMENT_LIBRARY_STEMS:
            declared = self._read_library_file_types(py_file)
            if not _document_library_admitted(
                stem, self.session_flags, self._workspace_file_types, declared
            ):
                ASCIIColors.debug(
                    f"[LCP] Library '{stem}' skipped: no matching workspace "
                    f"file type (declared={declared}, "
                    f"detected={sorted(self._workspace_file_types)})."
                )
                return False
        return True

    @staticmethod
    def _normalize_name(raw: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", (raw or "").strip().lower())
        if normalized.startswith("tool_"):
            normalized = normalized[5:]
        return normalized

    def _library_aliases(self, py_file: Path) -> List[str]:
        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            return []
        match = re.search(
            r'^TOOL_LIBRARY_ALIASES\s*=\s*[\[(](.*?)[\])]',
            code_text,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            return []
        return [
            self._normalize_name(alias)
            for alias in re.findall(r'["\']([^"\']+)["\']', match.group(1))
        ]

    def _library_display_name(self, py_file: Path) -> str:
        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            return ""
        match = re.search(
            r'^TOOL_LIBRARY_NAME\s*=\s*["\']([^"\']+)["\']',
            code_text,
            re.MULTILINE,
        )
        return match.group(1) if match else ""

    def _iter_library_files(self):
        search_dirs = list(self.tools_folders)
        default_dir = Path(__file__).parent / "default_tools"
        if default_dir not in search_dirs:
            search_dirs.append(default_dir)
        for folder in search_dirs:
            if not folder.exists() or not folder.is_dir():
                continue
            for item in folder.iterdir():
                if item.is_dir():
                    primary = item / f"{item.name}.py"
                    if primary.exists():
                        yield item.name, primary
                    else:
                        for fallback in item.glob("*.py"):
                            if fallback.stem != "__init__":
                                yield item.name, fallback
                elif item.suffix == ".py" and item.stem != "__init__":
                    yield item.stem, item

    def _resolve_library_name(self, raw: str) -> Optional[str]:
        """Resolves a user/LLM-provided toolkit name to a library stem,
        tolerating casing, separators, display names, declared aliases,
        and close misspellings."""
        clean = (raw or "").strip()
        if not clean:
            return None

        direct_dir = self._find_library_dir(clean)
        if direct_dir:
            return direct_dir.name

        normalized = self._normalize_name(clean)
        candidates: Dict[str, List[str]] = {}
        for stem, py_file in self._iter_library_files():
            variants = [self._normalize_name(stem)]
            variants.append(self._normalize_name(self._library_display_name(py_file)))
            variants.extend(self._library_aliases(py_file))
            candidates[stem] = [v for v in variants if v]
            if normalized in candidates[stem]:
                return stem

        all_variants = {v for variants in candidates.values() for v in variants}
        close = difflib.get_close_matches(normalized, all_variants, n=1, cutoff=0.75)
        if close:
            for stem, variants in candidates.items():
                if close[0] in variants:
                    return stem
        return None

    def _is_library_permitted(self, py_file: Path) -> bool:
        """
        Resolves a library's TOOL_LIBRARY_LOAD_FLAG declaration against host
        configuration. Returns True when the library has no gating flag or the
        flag is explicitly enabled, False when the flag exists and is disabled.
        """
        flag_name = self._library_load_flag(py_file)
        if not flag_name:
            return True
        host_cfg = self.host_tool_configs.get(py_file.stem, {})
        raw_value = host_cfg.get(flag_name, self._load_flag_kwargs.get(flag_name, False))

        if isinstance(raw_value, str):
            return raw_value.strip().lower() in ("true", "1", "yes", "on")
        return bool(raw_value)

    def _load_tool_file(self, py_file: Path) -> int:
        file_stem = py_file.stem

        # ── OFFICE + DOCUMENT SESSION GATE (fail-closed) ──
        # Office per-software toolsets require their own flag. Document-family
        # toolsets are admitted when their declared file types are present in
        # the workspace (see _document_library_admitted).
        if not self._is_library_admitted(py_file):
            return 0

        if not self._is_library_permitted(py_file):
            ASCIIColors.debug(
                f"[LCP] Library '{file_stem}' skipped: TOOL_LIBRARY_LOAD_FLAG "
                "not enabled in host configuration."
            )
            return 0

        tool_defs = self._parse_tool_via_ast(py_file)
        if not tool_defs:
            return 0

        try:
            code_text = py_file.read_text(encoding="utf-8")
            file_tree = ast.parse(code_text)
        except Exception:
            file_tree = None

        count = 0
        for tool_def in tool_defs:
            tool_name = tool_def.get("name")
            tool_def['_python_file_path'] = str(py_file.resolve())

            if any(t.get("name") == tool_name for t in self.discovered_tools):
                continue

            base_tool_name = tool_name[5:] if tool_name.startswith("tool_") else tool_name
            dynamic_prompt_fn_name = f"tool_{base_tool_name}_prompt"
            has_dynamic_prompt = False
            if file_tree:
                has_dynamic_prompt = any(node.name == dynamic_prompt_fn_name for node in ast.walk(file_tree) if isinstance(node, ast.FunctionDef))

            if has_dynamic_prompt:
                tool_def["_has_dynamic_prompt"] = True
                tool_def["_dynamic_prompt_fn_name"] = dynamic_prompt_fn_name
                tool_def["_python_file_path"] = str(py_file.resolve())

            self.discovered_tools.append(tool_def)
            count += 1

        return count

    def _find_library_dir(self, library_name: str) -> Optional[Path]:
        """Resolves the directory path for a library across all configured tool folders."""
        clean_name = library_name.strip()
        search_dirs = list(self.tools_folders)
        default_dir = Path(__file__).parent / "default_tools"
        if default_dir not in search_dirs:
            search_dirs.append(default_dir)

        for folder in search_dirs:
            if not folder.exists() or not folder.is_dir():
                continue
            candidate = folder / clean_name
            if candidate.exists() and candidate.is_dir():
                return candidate
            if clean_name.startswith("tool_"):
                candidate_alt = folder / clean_name[5:]
                if candidate_alt.exists() and candidate_alt.is_dir():
                    return candidate_alt
        return None

    def find_library_for_tool(self, tool_name: str) -> Optional[str]:
        """Finds which library directory or file provides the specified tool or library name."""
        clean_name = tool_name.strip()

        # 1. Check if clean_name is directly an available library folder name
        direct_dir = self._find_library_dir(clean_name)
        if direct_dir:
            return direct_dir.name

        # 2. Check already discovered tools
        for t in self.discovered_tools:
            if t.get("name") == clean_name and t.get("_python_file_path"):
                p = Path(t["_python_file_path"])
                return p.parent.name if p.parent.name not in ("default_tools", "tools") else p.stem

        # 3. Check AST function names inside all library files across all tools folders
        search_dirs = list(self.tools_folders)
        default_dir = Path(__file__).parent / "default_tools"
        if default_dir not in search_dirs:
            search_dirs.append(default_dir)

        for folder in search_dirs:
            if not folder.exists() or not folder.is_dir():
                continue
            for item in folder.iterdir():
                py_files = []
                if item.is_dir():
                    py_files = [f for f in item.glob("*.py") if f.stem != "__init__"]
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_files = [item]

                for py_f in py_files:
                    try:
                        extracted = self._extract_tool_names_from_file(py_f)
                        if clean_name in extracted and self._is_library_permitted(py_f):
                            return item.name if item.is_dir() else py_f.stem
                    except Exception:
                        pass
        return None

    def is_tool_available(self, tool_name: str) -> bool:
        """Checks if a tool is either already discovered or can be dynamically mounted."""
        clean_name = tool_name.strip()
        if any(t.get("name") == clean_name for t in self.discovered_tools):
            return True
        return self.find_library_for_tool(clean_name) is not None

    def load_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Mounts the library containing tool_name if needed and returns its specification."""
        clean_name = tool_name.strip()
        match = next((t for t in self.discovered_tools if t.get("name") == clean_name), None)
        if match:
            return match

        lib_name = self.find_library_for_tool(clean_name) or self._resolve_library_name(clean_name)
        if lib_name:
            self.mount_tool_library(lib_name)
            tool_match = next((t for t in self.discovered_tools if t.get("name") == clean_name), None)
            if tool_match:
                return tool_match
            for t in self.discovered_tools:
                py_p = t.get("_python_file_path", "")
                if lib_name in py_p:
                    return t

        return None

    def is_library_mounted(self, library_name: str) -> bool:
        """Checks if a tool library is already fully mounted by verifying that
        all expected tool names from the library are present in discovered_tools."""
        lib_path = self._find_library_dir(library_name)
        if not lib_path or not lib_path.exists() or not lib_path.is_dir():
            return False

        expected_tool_names: set = set()
        for item in lib_path.iterdir():
            py_file = None
            sibling_py_files: List[Path] = []
            if item.is_dir():
                py_file = item / f"{item.name}.py"
                if not py_file.exists():
                    sub_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]
                    for fallback_py in sub_py_files:
                        expected_tool_names.update(self._extract_tool_names_from_file(fallback_py))
                    continue
            elif item.suffix == ".py" and item.stem != "__init__":
                py_file = item

            if py_file and py_file.exists():
                expected_tool_names.update(self._extract_tool_names_from_file(py_file))
            for sibling_py in sibling_py_files:
                expected_tool_names.update(self._extract_tool_names_from_file(sibling_py))

        if not expected_tool_names:
            return False

        registered_names = {t.get("name") for t in self.discovered_tools}
        return expected_tool_names.issubset(registered_names)

    def mount_tool_library_if_absent(self, library_name: str) -> bool:
        """Mounts a tool library only if it is not already fully mounted.
        Prevents idempotent mount warnings on repeated calls."""
        if self.is_library_mounted(library_name):
            return True
        return self.mount_tool_library(library_name)

    def mount_tool_library(self, library_name: str) -> bool:
        resolved_name = self._resolve_library_name(library_name)
        _gate_dir = self._find_library_dir(resolved_name or library_name)
        if _gate_dir and _gate_dir.is_dir():
            _gate_py = _gate_dir / f"{_gate_dir.name}.py"
            if _gate_py.exists() and not self._is_library_admitted(_gate_py):
                ASCIIColors.warning(
                    f"[LCP Mount] Library '{library_name}' is gated and its "
                    "session flag/file-type trigger is not satisfied. "
                    "Mount refused (fail-closed)."
                )
                return False
        if resolved_name and resolved_name != library_name.strip():
            ASCIIColors.info(f"[LCP Mount] Resolving toolkit '{library_name}' -> '{resolved_name}'.")
        lib_path = self._find_library_dir(resolved_name or library_name)
        if not lib_path or not lib_path.exists() or not lib_path.is_dir():
            ASCIIColors.warning(f"[LCP Mount] Library '{library_name}' not found in search paths.")
            return False

        primary_py = lib_path / f"{lib_path.name}.py"
        if primary_py.exists() and not self._is_library_permitted(primary_py):
            flag_name = self._library_load_flag(primary_py)
            ASCIIColors.warning(
                f"[LCP Mount] Library '{library_name}' is gated by "
                f"TOOL_LIBRARY_LOAD_FLAG '{flag_name}' which is not enabled. "
                f"Enable it with tools_binding_config['{flag_name}']=True or "
                f"host_tool_configs['{primary_py.stem}']['{flag_name}']=True."
            )
            return False

        if self.is_library_mounted(library_name):
            return True

        expected_tool_names: set = set()
        for item in lib_path.iterdir():
            py_file = None
            if item.is_dir():
                py_file = item / f"{item.name}.py"
                if not py_file.exists():
                    sub_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]
                    for fallback_py in sub_py_files:
                        expected_tool_names.update(self._extract_tool_names_from_file(fallback_py))
                    continue
            elif item.suffix == ".py" and item.stem != "__init__":
                py_file = item

            if py_file and py_file.exists():
                expected_tool_names.update(self._extract_tool_names_from_file(py_file))

        if not expected_tool_names:
            ASCIIColors.warning(f"[LCP Mount] ⚠️ Library '{library_name}' contains no tool definitions.")
            return False

        already_registered = {
            t.get("name") for t in self.discovered_tools
            if t.get("name") in expected_tool_names
        }

        if lib_path not in self.tools_folders:
            self.tools_folders.append(lib_path)

        if len(already_registered) == len(expected_tool_names):
            ASCIIColors.info(f"[LCP Mount] ✅ Library '{library_name}' already registered ({len(already_registered)} tools). Idempotent mount.")
            return True

        initial_tool_count = len(self.discovered_tools)

        for item in lib_path.iterdir():
            py_file = None
            sibling_py_files: List[Path] = []
            if item.is_dir():
                py_file = item / f"{item.name}.py"
                sibling_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__" and f != py_file]
            elif item.suffix == ".py" and item.stem != "__init__":
                py_file = item

            if py_file and py_file.exists():
                self._load_tool_file(py_file)
            for sibling_py in sibling_py_files:
                self._load_tool_file(sibling_py)

        new_tool_count = len(self.discovered_tools) - initial_tool_count

        if new_tool_count > 0:
            ASCIIColors.success(f"[LCP Mount] ✅ Successfully mounted '{library_name}': {new_tool_count} tools registered (lazy init).")
            return True
        elif len(already_registered) > 0:
            ASCIIColors.info(f"[LCP Mount] ✅ Library '{library_name}' already registered ({len(already_registered)} tools). Idempotent mount.")
            return True
        else:
            ASCIIColors.warning(f"[LCP Mount] ⚠️ Library '{library_name}' mounted but no tools discovered.")
            return False

    def _extract_tool_names_from_file(self, py_file: Path) -> set:
        names: set = set()
        try:
            code_text = py_file.read_text(encoding="utf-8")
            tree = ast.parse(code_text)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_"):
                    names.add(node.name)
        except Exception:
            pass
        return names

    def _discover_local_tools(self):
        self.discovered_tools = []

        for folder in self.tools_folders:
            if not folder or not folder.is_dir():
                continue

            for item in folder.iterdir():
                py_file = None
                sibling_py_files: List[Path] = []
                if item.is_dir():
                    py_file = item / f"{item.name}.py"
                    sibling_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__" and f != py_file]
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_file = item

                if py_file and py_file.exists():
                    self._load_tool_file(py_file)
                for sibling_py in sibling_py_files:
                    self._load_tool_file(sibling_py)

        for py_file in self.tool_files:
            if py_file and py_file.exists() and py_file.suffix == ".py":
                self._load_tool_file(py_file)

    # ── Sandbox gating helpers (per category, per discussion) ──

    def _discussion_scope(self, env: Any = None, discussion_instance: Any = None) -> Any:
        """Returns the current discussion's gate-scope key.

        Resolution order:
            1. `env.discussion` (unified Env passed by the dispatch seam).
            2. Explicit `discussion_instance` argument.
            3. Fallback to the host-wide scope (never crosses into a durable
               per-discussion store).
        """
        candidate = None
        if env is not None:
            candidate = getattr(env, "discussion", None)
        if candidate is None:
            candidate = discussion_instance
        if candidate is None:
            return self._no_discussion_scope
        # Prefer stable ids over object identity (discussion may be reloaded).
        for attr in ("id", "discussion_id", "uid"):
            val = getattr(candidate, attr, None)
            if val is not None:
                return f"discussion:{val}"
        return f"discussion_obj:{id(candidate)}"

    def _is_gate_open(self, scope: Any, category: Optional[str]) -> bool:
        """True when `category` is already allow-listed for `scope`."""
        if not category:
            return False
        return category in self._discussion_gates.get(scope, set())

    def _open_gate(self, scope: Any, category: Optional[str]) -> None:
        """Mark `category` as "always allowed" for `scope` only."""
        if not category:
            return
        self._discussion_gates.setdefault(scope, set()).add(category)

    def revoke_gate(self, discussion_id: Any, category: str) -> bool:
        """Clears one category for one discussion. Returns True if revoked."""
        scope = f"discussion:{discussion_id}"
        cats = self._discussion_gates.get(scope)
        if not cats or category not in cats:
            return False
        cats.discard(category)
        if not cats:
            self._discussion_gates.pop(scope, None)
        return True

    def reset_discussion_gates(self, discussion_id: Any) -> None:
        """Drops the entire gate set for a discussion (call on discussion end)."""
        self._discussion_gates.pop(f"discussion:{discussion_id}", None)

    def list_open_gates(self, discussion_id: Any) -> List[str]:
        """Read-only snapshot of the currently open gates for a discussion."""
        return sorted(self._discussion_gates.get(f"discussion:{discussion_id}", set()))

    @staticmethod
    def _confirm_answer_is_always(answer: Any) -> bool:
        """Interprets a host confirm-handler answer to detect an `always` decision.

        The host UI contract is intentionally permissive to accept every
        shape already in use (bool, str, tuple, dict). Anything unrecognized
        is treated as a *once-only* approval — never as "always".
        """
        if answer is True:
            return False  # plain True means this-time-only
        if isinstance(answer, str):
            return answer.strip().lower() in ("always", "all", "permanent", "forever")
        if isinstance(answer, (tuple, list)) and answer:
            head = str(answer[0]).strip().lower()
            tail = answer[1] if len(answer) > 1 else None
            if head in ("always", "allow_always") and tail is not False:
                return True
            return False
        if isinstance(answer, dict):
            return bool(answer.get("always") or answer.get("permanent"))
        return False

    def _gated_confirm(
        self,
        message: str,
        task_category: Optional[str] = None,
        env: Any = None,
        discussion_instance: Any = None,
        **extra: Any,
    ) -> bool:
        """Sandbox-aware confirmation entry point.

        Order of operations:
        1. If the category is already open for the current discussion,
           short-circuit to True WITHOUT calling the host UI.
        2. Otherwise delegate to the host `confirm_handler`.
        3. If the host answered "always", persist the category in the
           per-discussion gate set ONLY (never globally).
        """
        if task_category and task_category not in self.GATED_TASK_CATEGORIES:
            # Unknown categories are still routed through the host UI but
            # never permanently persisted — prevents typo-driven leaks.
            task_category = None

        scope = self._discussion_scope(env=env, discussion_instance=discussion_instance)

        if task_category and self._is_gate_open(scope, task_category):
            ASCIIColors.debug(
                f"[LCP Gate] Auto-approved '{task_category}' for '{scope}' "
                "(gate open for this discussion)."
            )
            return True

        if self.confirm_handler is None:
            # Fail-closed: without a host handler, only pre-approved gates pass.
            return False

        try:
            answer = self.confirm_handler(message, **extra)
        except TypeError:
            # Some host handlers accept only the message positional arg.
            try:
                answer = self.confirm_handler(message)
            except Exception as exc:
                ASCIIColors.error(f"[LCP Gate] Confirm handler failed: {exc}")
                return False
        except Exception as exc:
            ASCIIColors.error(f"[LCP Gate] Confirm handler failed: {exc}")
            return False

        approved = bool(answer) and answer is not False or self._confirm_answer_is_always(answer)
        if not approved:
            return False

        if task_category and self._confirm_answer_is_always(answer):
            self._open_gate(scope, task_category)
            ASCIIColors.info(
                f"[LCP Gate] User accepted '{task_category}' ALWAYS for '{scope}' "
                "(discussion-scoped)."
            )

        return True

    def set_confirm_handler(self, handler: Optional[Callable]) -> None:
        """Registers a confirmation callback on LCPBinding and propagates it to all loaded tool modules."""
        self.confirm_handler = handler
        for lib_cfg in self.host_tool_configs.values():
            if isinstance(lib_cfg, dict):
                lib_cfg["confirm_handler"] = handler
        for mod_name, mod in list(sys.modules.items()):
            if "execute_python" in mod_name or "system_shell" in mod_name:
                if hasattr(mod, "set_confirm_handler") and callable(mod.set_confirm_handler):
                    try:
                        mod.set_confirm_handler(handler)
                    except Exception:
                        pass
                if hasattr(mod, "_CONFIRM_HANDLER"):
                    try:
                        mod._CONFIRM_HANDLER = handler
                    except Exception:
                        pass

    def set_user_interaction_handler(self, handler: Optional[Callable]) -> None:
        """Registers a user-interaction callback and propagates it to all loaded tool modules."""
        self.user_interaction_handler = handler
        for lib_cfg in self.host_tool_configs.values():
            if isinstance(lib_cfg, dict):
                lib_cfg["user_interaction_handler"] = handler
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.startswith("lollms_client.tools_bindings.lcp.persistent_"):
                if hasattr(mod, "set_user_interaction_handler") and callable(mod.set_user_interaction_handler):
                    try:
                        mod.set_user_interaction_handler(handler)
                    except Exception:
                        pass

    def _get_or_load_tool_module(self, python_file_path: Path) -> Optional[types.ModuleType]:
        module_name = f"lollms_client.tools_bindings.lcp.persistent_{python_file_path.stem}"
        if module_name in sys.modules:
            tool_mod = sys.modules[module_name]
            if self.confirm_handler and hasattr(tool_mod, "set_confirm_handler") and callable(tool_mod.set_confirm_handler):
                try:
                    tool_mod.set_confirm_handler(self.confirm_handler)
                except Exception:
                    pass
            if self.user_interaction_handler and hasattr(tool_mod, "set_user_interaction_handler") and callable(tool_mod.set_user_interaction_handler):
                try:
                    tool_mod.set_user_interaction_handler(self.user_interaction_handler)
                except Exception:
                    pass
            return tool_mod

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(python_file_path.resolve()))
            if not spec or not spec.loader:
                return None

            tool_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = tool_module
            spec.loader.exec_module(tool_module)

            # ── 🛡️ HOST CONFIGURATION & CONFIRMATION INJECTION ──
            library_name = python_file_path.stem
            host_config = self.host_tool_configs.get(library_name, {})
            if self.confirm_handler and "confirm_handler" not in host_config:
                host_config["confirm_handler"] = self.confirm_handler
            if self.user_interaction_handler and "user_interaction_handler" not in host_config:
                host_config["user_interaction_handler"] = self.user_interaction_handler
            if self.user_interaction_handler and "user_interaction_handler" not in host_config:
                host_config["user_interaction_handler"] = self.user_interaction_handler

            if hasattr(tool_module, "set_confirm_handler") and callable(tool_module.set_confirm_handler):
                try:
                    tool_module.set_confirm_handler(host_config.get("confirm_handler", self.confirm_handler))
                except Exception:
                    pass

            if hasattr(tool_module, "set_user_interaction_handler") and callable(tool_module.set_user_interaction_handler):
                try:
                    tool_module.set_user_interaction_handler(host_config.get("user_interaction_handler", self.user_interaction_handler))
                except Exception:
                    pass

            if hasattr(tool_module, "set_user_interaction_handler") and callable(tool_module.set_user_interaction_handler):
                try:
                    tool_module.set_user_interaction_handler(host_config.get("user_interaction_handler", self.user_interaction_handler))
                except Exception:
                    pass

            if hasattr(tool_module, "init_tools_library") and callable(tool_module.init_tools_library):

                import inspect as _lcp_inspect
                _init_sig = _lcp_inspect.signature(tool_module.init_tools_library)
                _init_params = _init_sig.parameters

                _accepts_positional = any(
                    p.kind in (_lcp_inspect.Parameter.POSITIONAL_ONLY, _lcp_inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for p in _init_params.values()
                )
                _accepts_var_positional = any(
                    p.kind == _lcp_inspect.Parameter.VAR_POSITIONAL
                    for p in _init_params.values()
                )

                if _accepts_positional or _accepts_var_positional:
                    tool_module.init_tools_library(host_config)
                else:
                    tool_module.init_tools_library()

                if getattr(ASCIIColors, '_force_debug', False):
                    ASCIIColors.success(f"[LCP Lazy Init] ✅ Initialized library for '{library_name}' with host configs.")
                else:
                    ASCIIColors.debug(f"[LCP Lazy Init] ✅ Initialized library for '{library_name}' with host configs.")
            return tool_module
        except Exception as init_ex:
            ASCIIColors.error(f"[LCP] ❌ Toolset '{python_file_path.stem}' FAILED lazy init: {init_ex}")
            if module_name in sys.modules:
                del sys.modules[module_name]
            return None

    def get_dynamic_tool_description(self, tool_def: Dict[str, Any]) -> Optional[str]:
        """
        Executes the dynamic prompt function if defined for this tool,
        returning the dynamic description string.
        """
        if not tool_def.get("_has_dynamic_prompt"):
            return None
        fn_name = tool_def.get("_dynamic_prompt_fn_name")
        if not fn_name:
            return None

        python_file_path = Path(tool_def["_python_file_path"]) if tool_def.get("_python_file_path") else None
        if python_file_path and python_file_path.exists():
            tool_module = self._get_or_load_tool_module(python_file_path)
        else:
            tool_module = tool_def.get("_dynamic_module")

        if tool_module and hasattr(tool_module, fn_name):
            try:
                fn = getattr(tool_module, fn_name)
                if callable(fn):
                    return str(fn())
            except Exception as e:
                ASCIIColors.warning(f"[LCP] Failed to execute dynamic prompt function '{fn_name}': {e}")
        return None

    def to_chat_tool_specs(self, **discover_kwargs) -> Dict[str, Dict[str, Any]]:
        specs = super().to_chat_tool_specs(**discover_kwargs)
        for tool_name, spec in specs.items():
            tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)
            if tool_def and tool_def.get("_has_dynamic_prompt"):
                dynamic_desc = self.get_dynamic_tool_description(tool_def)
                if dynamic_desc:
                    spec["description"] = dynamic_desc
                    tool_def["description"] = dynamic_desc
        return specs

    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        if kwargs.get("force_refresh", False) or not self.discovered_tools:
             self._discover_local_tools()
        tools = self.discovered_tools
        if specific_tool_names:
            tools = [t for t in tools if t.get("name") in specific_tool_names]
        for t in tools:
            if t.get("_has_dynamic_prompt"):
                dyn_desc = self.get_dynamic_tool_description(t)
                if dyn_desc:
                    t["description"] = dyn_desc
        return tools

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        return self.discover_tools(**kwargs)

    def register_tool_from_code(self, tool_name_prefix: str, code: str) -> bool:
        try:
            module_name = f"dynamic_tool_{tool_name_prefix}_{uuid.uuid4().hex[:8]}"
            module = types.ModuleType(module_name)
            exec(_native_compile(code, "<dynamic_tool>", "exec"), module.__dict__)
            tree = ast.parse(code)
            registered_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_"):
                    tool_name = node.name
                    self.discovered_tools = [t for t in self.discovered_tools if t.get("name") != tool_name]
                    tool_def = self._extract_single_tool_schema(node, tool_name_prefix)
                    if tool_def:
                        tool_def["_dynamic_module"] = module
                        tool_def["_python_file_path"] = None
                        self.discovered_tools.append(tool_def)
                        registered_count += 1
                        ASCIIColors.success(f"[LCP Dynamic] Registered tool '{tool_name}' from artefact '{tool_name_prefix}'")
            return registered_count > 0
        except Exception as e:
            ASCIIColors.error(f"[LCP Dynamic] Failed to register tool from code: {e}")
            trace_exception(e)
            return False

    def _resolve_context_files(self, context_files: Any) -> List[str]:
        """Coerces a tool's context_files payload into absolute, existing file paths."""
        if isinstance(context_files, (str, Path)):
            entries: List[Any] = [context_files]
        elif isinstance(context_files, (list, tuple)):
            entries = list(context_files)
        else:
            return []

        resolved: List[str] = []
        for entry in entries:
            try:
                entry_path = Path(str(entry))
                if not entry_path.is_absolute():
                    entry_path = Path.cwd() / entry_path
                entry_path = entry_path.resolve()
                if entry_path.is_file():
                    resolved.append(str(entry_path))
                else:
                    ASCIIColors.debug(f"[LCP execute_tool] Dropped missing context file: {entry}")
            except Exception:
                continue
        return resolved

    def unregister_tools_by_prefix(self, tool_name_prefix: str) -> int:
        initial_count = len(self.discovered_tools)
        self.discovered_tools = [
            t for t in self.discovered_tools 
            if not (t.get("name", "").startswith(f"tool_{tool_name_prefix}") or t.get("name", "") == tool_name_prefix)
        ]
        removed = initial_count - len(self.discovered_tools)
        if removed > 0:
            ASCIIColors.info(f"[LCP Dynamic] Unregistered {removed} tool(s) for prefix '{tool_name_prefix}'")
        return removed

    def execute_tool(self, tool_name: str, params: Dict[str, Any], discussion_instance=None, **kwargs) -> Dict[str, Any]:
        ASCIIColors.debug(f"[LCP execute_tool] Calling: '{tool_name}' with params: {params}")

        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)
        if not tool_def:
            return {"error": f"Tool '{tool_name}' not found.", "status_code": 404}

        python_file_path = Path(tool_def.get('_python_file_path')) if tool_def.get('_python_file_path') else None

        # Resolve the discussion scope once so the sandbox gate wrapper below
        # sees the exact same key the call-shortcut path will consult.
        _scope_env = kwargs.get("env")
        _discussion_scope_kw = discussion_instance
        if _scope_env is not None and getattr(_scope_env, "discussion", None) is not None:
            _discussion_scope_kw = getattr(_scope_env, "discussion")

        input_schema = tool_def.get("input_schema", {})
        for prop_name, prop_info in input_schema.get("properties", {}).items():
            if prop_name not in params and isinstance(prop_info, dict) and "default" in prop_info:
                params[prop_name] = prop_info["default"]

        try:
            if python_file_path:
                tool_module = self._get_or_load_tool_module(python_file_path)
                if not tool_module:
                    return {"error": f"Tool initialization failed for '{python_file_path.stem}'.", "status_code": 500}
                if self.confirm_handler and hasattr(tool_module, "set_confirm_handler") and callable(tool_module.set_confirm_handler):
                    try:
                        tool_module.set_confirm_handler(self.confirm_handler)
                    except Exception:
                        pass
                if self.user_interaction_handler and hasattr(tool_module, "set_user_interaction_handler") and callable(tool_module.set_user_interaction_handler):
                    try:
                        tool_module.set_user_interaction_handler(self.user_interaction_handler)
                    except Exception:
                        pass
                if self.user_interaction_handler and hasattr(tool_module, "set_user_interaction_handler") and callable(tool_module.set_user_interaction_handler):
                    try:
                        tool_module.set_user_interaction_handler(self.user_interaction_handler)
                    except Exception:
                        pass
            else:
                tool_module = tool_def.get("_dynamic_module")
                if not tool_module:
                    return {"error": f"Dynamic module missing for '{tool_name}'.", "status_code": 500}

            if not hasattr(tool_module, tool_name):
                return {"error": f"Function '{tool_name}' not found in module.", "status_code": 500}

            execute_function = getattr(tool_module, tool_name)

            import inspect
            sig = inspect.signature(execute_function)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            clean_params = {}
            for k, v in params.items():
                if k in sig.parameters or has_var_kw:
                    clean_params[k] = v

            env = kwargs.get("env") or params.get("env")
            clean_params.pop("env", None)
            if env is not None and ("env" in sig.parameters or has_var_kw):
                clean_params["env"] = env

            host_tool_context = getattr(env, "tool_context", None) if env is not None else None
            clean_params.pop("tool_context", None)
            if host_tool_context is not None and ("tool_context" in sig.parameters or has_var_kw):
                clean_params["tool_context"] = host_tool_context

            host_client = getattr(env, "client", None) if env is not None else None
            clean_params.pop("lollms_client_instance", None)
            if host_client is not None and ("lollms_client_instance" in sig.parameters or has_var_kw):
                clean_params["lollms_client_instance"] = host_client

            if self.user_interaction_handler is not None and ("user_interaction_handler" in sig.parameters or has_var_kw):
                clean_params["user_interaction_handler"] = self.user_interaction_handler

            clean_params.pop("discussion_instance", None)
            host_discussion = getattr(env, "discussion", None) if env is not None else discussion_instance
            if host_discussion is not None and ("discussion_instance" in sig.parameters or has_var_kw):
                clean_params["discussion_instance"] = host_discussion

            # ── SANDBOX CATEGORY PROPAGATION ──
            # If a tool declares a ``GATED_TASK_CATEGORY`` module constant, we
            # surface it on the invocation env so that any ``confirm_handler``
            # call the tool performs can be matched against the current
            # discussion's gate set. This makes "always allow" discussion-scoped
            # without every toolset reimplementing the logic.
            tool_category: Optional[str] = None
            try:
                if python_file_path and getattr(tool_module, "GATED_TASK_CATEGORY", None):
                    declared = getattr(tool_module, "GATED_TASK_CATEGORY")
                    if isinstance(declared, str) and declared in self.GATED_TASK_CATEGORIES:
                        tool_category = declared
            except Exception:
                tool_category = None

            # Wrap the tool's confirm handler for the duration of the call so
            # short-circuited categories never reach the host UI.
            if tool_category and hasattr(tool_module, "set_confirm_handler"):
                def _scoped_confirm(message, _cat=tool_category, _scope_env=env, _disc=_discussion_scope_kw, **_kw):
                    return self._gated_confirm(
                        message,
                        task_category=_cat,
                        env=_scope_env,
                        discussion_instance=_disc,
                        **_kw,
                    )
                try:
                    tool_module.set_confirm_handler(_scoped_confirm)
                except Exception:
                    pass

            try:
                result = execute_function(**clean_params)
            except BaseException as exec_bex:
                import traceback as _exec_tb
                tb_str = _exec_tb.format_exc()
                ASCIIColors.error(f"[LCP execute_tool] ❌ Tool '{tool_name}' crashed with {type(exec_bex).__name__}: {exec_bex}\n{tb_str}")
                return {
                    "output": {"success": False, "error": f"Tool '{tool_name}' crashed: {exec_bex}"},
                    "error": f"Tool '{tool_name}' crashed: {exec_bex}",
                    "traceback": tb_str,
                    "success": False,
                    "status_code": 500
                }

            ASCIIColors.debug(f"[LCP execute_tool] Tool '{tool_name}' returned: {str(result)[:500] if result is not None else 'None'}")

            # ── UNIFIED RESULT NORMALIZATION ──
            # The harness must be flexible to different tool conventions.
            # We define a result as "Failed" if any of these are true:
            # 1. Explicit success=False
            # 2. Explicit status="error" or "failure"
            # 3. return_code is not None and != 0
            # 4. An "error" key exists with a truthy value
            # Otherwise, it is "Success". We ensure "output" is always present.
            
            is_failure = False
            error_msg = None
            
            if not isinstance(result, dict):
                # Primitive or List: Always success, wrap in output
                result = {"success": True, "output": result}
            else:
                # Check Failure Signals
                if result.get("success") is False:
                    is_failure = True
                    error_msg = result.get("error") or result.get("message") or "Tool reported failure."
                
                status_val = str(result.get("status", "")).lower()
                if status_val in ("error", "failure", "failed"):
                    is_failure = True
                    error_msg = result.get("error") or result.get("message") or f"Tool status was '{status_val}'."
                
                rc = result.get("return_code")
                if rc is not None and rc != 0:
                    is_failure = True
                    error_msg = result.get("error") or result.get("stderr") or f"Non-zero return code: {rc}"
                
                if "error" in result and result["error"]:
                    if not is_failure:
                        # If we haven't failed yet, but there's an error key, it might be a soft error
                        # But usually "error" implies failure. Let's treat presence of truthy error as failure 
                        # unless success is explicitly True (which is a conflict, but success=True is stronger).
                        if result.get("success") is not True:
                            is_failure = True
                            error_msg = result["error"]

                # Ensure "output" exists for the LLM
                if "output" not in result:
                    # Look for common payload keys
                    payload = None
                    for key in ("result", "data", "response", "papers", "content", "text"):
                        if key in result:
                            payload = result[key]
                            break
                    if payload is None:
                        # If no specific payload key, dump the whole dict minus meta keys
                        meta_keys = {"success", "status", "error", "message", "return_code", "stderr"}
                        clean_dict = {k: v for k, v in result.items() if k not in meta_keys}
                        if clean_dict:
                            payload = clean_dict
                        else:
                            payload = result.get("output", "Tool executed successfully.")
                    result["output"] = payload

            if is_failure:
                # Standardize failure output
                if not result.get("output"):
                    result["output"] = result.get("error", "Tool failed.")
                
                ASCIIColors.error(f"[LCP Error Tracking] Tool '{tool_name}' reported failure: {error_msg}")
                return {
                    "output": result.get("output"),
                    "error": error_msg,
                    "success": False,
                    "status_code": 500
                }
            
            # Success Path
            if not result.get("output"):
                result["output"] = "Tool executed successfully (no output)."

            final_result: Dict[str, Any] = {
                "output": result.get("output"),
                "status_code": 200,
                "success": True
            }

            for passthrough_key in (
                "image_filename",
                "image_b64",
                "images",
                "image_media_types",
                "prompt_injection",
            ):
                passthrough_value = result.get(passthrough_key)
                if passthrough_value is not None:
                    final_result[passthrough_key] = passthrough_value

            if result.get("context_files") is not None:
                final_result["context_files"] = self._resolve_context_files(result.get("context_files"))

            return final_result

        except Exception as e:
            tb_str = traceback.format_exc()
            trace_exception(e)
            ASCIIColors.error(f"[LCP Error Tracking] Unexpected crash executing '{tool_name}':\n{tb_str}")
            return {
                "error": f"Error executing '{tool_name}': {str(e)}",
                "traceback": tb_str,
                "status_code": 500
            }