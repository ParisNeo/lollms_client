import sys
import os
import io
import importlib.util
import ast
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

from lollms_client.lollms_tools_binding import LollmsToolBinding
from ascii_colors import ASCIIColors, trace_exception

BindingName = "LCPBinding"

class LCPBinding(LollmsToolBinding):
    """
    Local LollmsCommunicationProtocol (LCP) Binding.
    
    PHILOSOPHY:
    - Tools are agnostic Python scripts.
    - Tools do NOT know about discussions, artifacts, or clients.
    - Tools operate on files in the current working directory.
    - The Binding handles environment setup (CWD, sync) transparently.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(binding_name="LCP")
        
        # Resolve Multi-Folder Config
        self.tools_folders: List[Path] = []
        folders_input = kwargs.get("tools_folders") or kwargs.get("tools_folder_path")
        if folders_input:
            if isinstance(folders_input, (str, Path)):
                self.tools_folders.append(Path(folders_input))
            elif isinstance(folders_input, list):
                for f in folders_input:
                    self.tools_folders.append(Path(f))
        else:
            self.tools_folders.append(Path(__file__).parent / "default_tools")

        # Resolve Direct Tool Files
        self.tool_files: List[Path] = []
        files_input = kwargs.get("tool_files")
        if files_input:
            if isinstance(files_input, (str, Path)):
                self.tool_files.append(Path(files_input))
            elif isinstance(files_input, list):
                for f in files_input:
                    self.tool_files.append(Path(f))

        self.discovered_tools: List[Dict[str, Any]] = []
        self._discover_local_tools()

    @property
    def tools_folder_path(self) -> Optional[Path]:
        return self.tools_folders[0] if self.tools_folders else None

    def _parse_tool_via_ast(self, py_file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts name, description, and schema from ALL tool_ functions in a file using AST.
        Returns a LIST of tool definitions to support Multi-Tool Files.
        """
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
            ASCIIColors.warning(f"AST parse failed for '{py_file_path.name}': {e}")
            return None

    def _extract_single_tool_schema(self, func_node: ast.FunctionDef, file_stem: str) -> Optional[Dict[str, Any]]:
        """Helper to extract schema for a single function."""
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
            if arg.arg in ("args", "kwargs", "discussion_instance", "lollms_client_instance"):
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

        # Fallback to docstring if no annotations
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
            "_python_file_path": str(Path(__file__).parent / "default_tools" / file_stem / f"{file_stem}.py") # Store path for execution
        }

    def _load_tool_file(self, py_file: Path) -> bool:
        file_stem = py_file.stem
        ASCIIColors.info(f"[LCP Discovery] Scanning file for tools: {py_file.name}")

        tool_defs = self._parse_tool_via_ast(py_file)
        if not tool_defs:
            ASCIIColors.red(f"[LCP Discovery] ❌ No tool_ functions found in '{file_stem}'")
            return False

        count = 0
        for tool_def in tool_defs:
            tool_name = tool_def.get("name")
            # Ensure path is absolute and correct
            tool_def['_python_file_path'] = str(py_file.resolve())

            if any(t.get("name") == tool_name for t in self.discovered_tools):
                ASCIIColors.yellow(f"[LCP Discovery] Skipping '{tool_name}' (already registered)")
                continue

            self.discovered_tools.append(tool_def)
            ASCIIColors.green(f"[LCP Discovery] ✅ Registered tool: {tool_name}")
            count += 1

        if count > 0:
            ASCIIColors.success(f"[LCP Discovery] 📦 Loaded {count} tools from '{file_stem}'")
            return True
        return False

    def mount_tool_library(self, library_name: str) -> bool:
        """
        Dynamically mounts a tool library from the default_tools directory at runtime.
        This allows the system to auto-load specialized tools (e.g., semantic_data_engineer)
        when context conditions are met (e.g., data files detected).

        Args:
            library_name: The name of the folder inside default_tools (e.g., 'semantic_data_engineer').

        Returns:
            True if successfully mounted and discovered, False otherwise.
        """
        base_dir = Path(__file__).parent / "default_tools"
        lib_path = base_dir / library_name

        if not lib_path.exists() or not lib_path.is_dir():
            ASCIIColors.warning(f"[LCP Mount] Library '{library_name}' not found at {lib_path}")
            return False

        if lib_path in self.tools_folders:
            ASCIIColors.info(f"[LCP Mount] Library '{library_name}' already mounted.")
            return True

        ASCIIColors.cyan(f"[LCP Mount] Mounting library '{library_name}' from {lib_path}...")
        self.tools_folders.append(lib_path)

        # Trigger discovery for this specific new folder only (optimization)
        # But for safety and simplicity, we re-scan all (fast enough for <50 tools)
        self._discover_local_tools()

        # Verify if tools were actually added
        # FIX: Use string comparison or .is_relative_to() instead of 'in' with Path objects
        lib_path_str = str(lib_path.resolve())
        new_tools = []
        for t in self.discovered_tools:
            file_path_str = str(Path(t.get('_python_file_path', '')).resolve())
            if file_path_str.startswith(lib_path_str):
                new_tools.append(t)

        if new_tools:
            ASCIIColors.success(f"[LCP Mount] ✅ Successfully mounted '{library_name}': {len(new_tools)} tools registered.")
            return True
        else:
            ASCIIColors.warning(f"[LCP Mount] ⚠️ Library '{library_name}' mounted but no tools discovered.")
            return False

    def _discover_local_tools(self):
        self.discovered_tools = []
        ASCIIColors.info(f"[LCP Discovery] Starting discovery scan...")
        ASCIIColors.info(f"[LCP Discovery] Configured Tools Folders: {self.tools_folders}")
        ASCIIColors.info(f"[LCP Discovery] Configured Tool Files: {self.tool_files}")

        for folder in self.tools_folders:
            if not folder or not folder.is_dir(): 
                ASCIIColors.warning(f"[LCP Discovery] Folder does not exist: {folder}")
                continue
            ASCIIColors.info(f"[LCP Discovery] 📂 Scanning directory: {folder}")

            for item in folder.iterdir():
                py_file = None
                if item.is_dir():
                    py_file = item / f"{item.name}.py"
                    ASCIIColors.info(f"[LCP Discovery]   Checking folder: {item.name} -> Expected: {py_file.name}")
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_file = item
                    ASCIIColors.info(f"[LCP Discovery]   Checking file: {item.name}")

                if py_file and py_file.exists():
                    self._load_tool_file(py_file)
                elif py_file:
                    ASCIIColors.warning(f"[LCP Discovery]   File missing: {py_file}")

        for py_file in self.tool_files:
            if py_file and py_file.exists() and py_file.suffix == ".py":
                self._load_tool_file(py_file)
            elif py_file:
                ASCIIColors.warning(f"[LCP Discovery] Explicit tool file missing: {py_file}")

        ASCIIColors.info(f"[LCP Discovery] 🏁 Discovery complete. Total tools found: {len(self.discovered_tools)}")
        ASCIIColors.info(f"[LCP Discovery] Tool List: {[t['name'] for t in self.discovered_tools]}")

    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        if kwargs.get("force_refresh", False) or not self.discovered_tools:
             self._discover_local_tools()
        if specific_tool_names:
            return [t for t in self.discovered_tools if t.get("name") in specific_tool_names]
        return self.discovered_tools

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        return self.discover_tools(**kwargs)

    def execute_tool(self, tool_name: str, params: Dict[str, Any], discussion_instance=None, **kwargs) -> Dict[str, Any]:
        """
        Executes a specific tool function from a potentially multi-tool file.
        """
        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)
        if not tool_def:
            return {"error": f"Tool '{tool_name}' not found.", "status_code": 404}

        python_file_path = Path(tool_def.get('_python_file_path'))

        # Ingest Schema Defaults
        input_schema = tool_def.get("input_schema", {})
        for prop_name, prop_info in input_schema.get("properties", {}).items():
            if prop_name not in params and isinstance(prop_info, dict) and "default" in prop_info:
                params[prop_name] = prop_info["default"]

        try:
            module_name = f"lollms_client.tools_bindings.lcp.{python_file_path.stem}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, str(python_file_path))
            if not spec or not spec.loader:
                return {"error": f"Spec creation failed for '{tool_name}'.", "status_code": 500}

            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)

            # CRITICAL FIX: Execute the EXACT function name requested (e.g., tool_get_table_schema)
            # Do not search for generic entry points.
            if not hasattr(tool_module, tool_name):
                return {"error": f"Function '{tool_name}' not found in module '{module_name}'.", "status_code": 500}

            execute_function = getattr(tool_module, tool_name)

            # ── ENVIRONMENT PREPARATION (Discussion-Isolated Workspace) ──
            # ARCHITECTURAL RULE: CWD is managed by the caller (ChatMixin).
            # Tools operate on files in the CURRENT WORKING DIRECTORY.
            # We trust that the caller has already set CWD to the discussion workspace.

            # Verify we're in the correct workspace (defensive check)
            import os
            current_cwd = Path(os.getcwd()).resolve()
            ASCIIColors.info(f"[LCP Tool '{tool_name}'] Executing in CWD: {current_cwd}")

            # Snapshot workspace files BEFORE execution (Store content hash to detect changes reliably)
            files_before = {}
            if current_cwd.exists():
                for f in current_cwd.rglob("*"):
                    if f.is_file():
                        rel_path = f.relative_to(current_cwd)
                        try:
                            # Store content hash and mtime
                            content = f.read_text(encoding="utf-8", errors="ignore")
                            files_before[rel_path] = {
                                "hash": hash(content),
                                "mtime": f.stat().st_mtime,
                                "path": f,
                                "content": content
                            }
                        except Exception:
                            # Skip binary files or unreadable files
                            files_before[rel_path] = {
                                "hash": None,
                                "mtime": f.stat().st_mtime,
                                "path": f,
                                "content": None
                            }

            ASCIIColors.info(f"[LCP Tool '{tool_name}'] Snapshot taken: {len(files_before)} files before execution")

            # Add workspace to sys.path and Environment
            current_cwd_str = str(current_cwd)
            if current_cwd_str not in sys.path:
                sys.path.insert(0, current_cwd_str)
            old_pythonpath = os.environ.get("PYTHONPATH", "")
            old_path = os.environ.get("PATH", "")
            if current_cwd_str not in old_pythonpath:
                os.environ["PYTHONPATH"] = f"{current_cwd_str}{os.pathsep}{old_pythonpath}" if old_pythonpath else current_cwd_str
            if current_cwd_str not in old_path:
                os.environ["PATH"] = f"{current_cwd_str}{os.pathsep}{old_path}"

                # 6. Capture stdout/stderr
                captured_stdout = io.StringIO()
                captured_stderr = io.StringIO()
                old_sys_stdout = sys.stdout
                old_sys_stderr = sys.stderr

                try:
                    sys.stdout = captured_stdout
                    sys.stderr = captured_stderr

                    # Execute with CLEAN params
                    # Tools are agnostic. They operate on files in CWD.
                    # No discussion_instance or lollms_client_instance is passed.
                    result = execute_function(**params)
                finally:
                    sys.stdout = old_sys_stdout
                    sys.stderr = old_sys_stderr
                    # Forward prints
                    if captured_stdout.getvalue():
                        old_sys_stdout.write(f"[LCP Tool '{tool_name}' STDOUT]:\n{captured_stdout.getvalue()}")
                        old_sys_stdout.flush()
                    if captured_stderr.getvalue():
                        old_sys_stdout.write(f"[LCP Tool '{tool_name}' STDERR]:\n{captured_stderr.getvalue()}")
                        old_sys_stdout.flush()

                # ── POST-EXECUTION: AUTOMATIC ARTIFACT SYNC ──
                # 7. Snapshot workspace files AFTER execution
                files_after = {}
                if current_cwd.exists():
                    for f in current_cwd.rglob("*"):
                        if f.is_file():
                            rel_path = f.relative_to(current_cwd)
                            try:
                                content = f.read_text(encoding="utf-8", errors="ignore")
                                files_after[rel_path] = {
                                    "hash": hash(content),
                                    "mtime": f.stat().st_mtime,
                                    "path": f,
                                    "content": content
                                }
                            except Exception:
                                files_after[rel_path] = {
                                    "hash": None,
                                    "mtime": f.stat().st_mtime,
                                    "path": f,
                                    "content": None
                                }

                if not discussion_instance:
                    ASCIIColors.warning("[LCP Tool] No discussion_instance provided. Skipping artifact sync.")
                else:
                    # 8. Detect NEW files
                    new_files = set(files_after.keys()) - set(files_before.keys())
                    ASCIIColors.info(f"[LCP Tool '{tool_name}'] Detected {len(new_files)} NEW files: {[str(f) for f in new_files]}")

                    for rel_path in new_files:
                        file_info = files_after[rel_path]
                        file_name = rel_path.name
                        file_ext = rel_path.suffix.lower()
                        file_path = file_info["path"]
                        file_size = file_path.stat().st_size

                        # 1. Determine Artifact Type based on Extension
                        atype = "document"
                        if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir", ".net", ".op"):
                            atype = "code"
                        elif file_ext in (".csv", ".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet"):
                            atype = "data"
                        elif file_ext in (".md", ".txt", ".log", ".out", ".trace", ".asc", ".raw", ".json", ".yaml", ".yml", ".xml", ".ttl"):
                            atype = "document"
                        elif file_ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"):
                            atype = "image"
                        elif file_ext in (".pdf", ".docx", ".zip", ".tar", ".gz"):
                            atype = "document"

                        # 2. Decide if we should read content or create a placeholder
                        EXPLICIT_BINARY_EXTS = {".db", ".sqlite", ".sqlite3", ".xlsx", ".xls", ".parquet", 
                                                ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", 
                                                ".zip", ".tar", ".gz", ".pdf", ".docx"}

                        should_read_content = True
                        content_placeholder = None

                        if file_ext in EXPLICIT_BINARY_EXTS:
                            should_read_content = False
                            content_placeholder = (
                                f"### Data File Generated: `{file_name}`\n\n"
                                f"This file was created by the tool `{tool_name}`.\n"
                                f"- **Type**: {file_ext.upper()} (Binary/Structured Data)\n"
                                f"- **Size**: {file_size:,} bytes\n"
                                f"- **Location**: `{file_path.resolve()}`\n\n"
                                f"> **Action**: You can download this file from the Workspace Artifacts panel or reference it in SQL/Python tools."
                            )
                        elif file_info["content"] is None:
                            try:
                                with open(file_path, 'rb') as f:
                                    chunk = f.read(1024)
                                    if b'\x00' in chunk:
                                        should_read_content = False
                                        content_placeholder = (
                                            f"### Binary File Detected: `{file_name}`\n\n"
                                            f"This file appears to be binary (contains null bytes).\n"
                                            f"- **Type**: {file_ext.upper()} (Unknown Binary)\n"
                                            f"- **Size**: {file_size:,} bytes\n"
                                            f"- **Location**: `{file_path.resolve()}`\n\n"
                                            f"> **Action**: Download from Workspace Artifacts panel."
                                        )
                                    else:
                                        ASCIIColors.warning(f"[LCP Tool] File '{file_name}' read failed initially, forcing text read with encoding ignore.")
                                        forced_content = file_path.read_text(encoding='utf-8', errors='ignore')
                                        file_info["content"] = forced_content
                                        should_read_content = True
                            except Exception as e:
                                ASCIIColors.error(f"[LCP Tool] Unable to inspect file '{file_name}': {e}")
                                should_read_content = False
                                content_placeholder = f"### File Error: `{file_name}`\n\nFailed to read or inspect file: {e}"

                        # 3. Register Artifact
                        if not should_read_content and content_placeholder:
                            existing_art = discussion_instance.artefacts.get(file_name)
                            if existing_art:
                                discussion_instance.artefacts.update(
                                    title=file_name,
                                    new_content=content_placeholder,
                                    new_type=atype,
                                    active=True,
                                    commit_message=f"Updated binary file reference by tool '{tool_name}'"
                                )
                            else:
                                discussion_instance.artefacts.add(
                                    title=file_name,
                                    artefact_type=atype,
                                    content=content_placeholder,
                                    active=True,
                                    commit_message=f"Created by tool '{tool_name}'"
                                )
                            ASCIIColors.success(f"[LCP Tool '{tool_name}'] ✨ Registered file (placeholder): '{file_name}'")
                            continue

                        # Handle Text/Readable Files
                        existing_art = discussion_instance.artefacts.get(file_name)
                        if existing_art:
                            ASCIIColors.info(f"[LCP Tool] File '{file_name}' reappeared on disk. Updating artifact.")
                            discussion_instance.artefacts.update(
                                title=file_name,
                                new_content=file_info["content"],
                                new_type=atype,
                                active=True,
                                commit_message=f"Restored by tool '{tool_name}'"
                            )
                        else:
                            ASCIIColors.success(f"[LCP Tool '{tool_name}'] ✨ Creating NEW artifact from file: '{file_name}'")
                            discussion_instance.artefacts.add(
                                title=file_name,
                                artefact_type=atype,
                                content=file_info["content"],
                                active=True,
                                commit_message=f"Created by tool '{tool_name}'"
                            )
                            ASCIIColors.success(f"[LCP Tool '{tool_name}'] ✨ Created NEW artifact: '{file_name}'")

                    # 9. Detect MODIFIED files
                    common_files = set(files_after.keys()) & set(files_before.keys())
                    for rel_path in common_files:
                        before_info = files_before[rel_path]
                        after_info = files_after[rel_path]
                        file_name = rel_path.name
                        file_ext = rel_path.suffix.lower()
                        file_path = after_info["path"]

                        content_changed = before_info.get("hash") != after_info.get("hash")
                        became_binary = before_info.get("content") is not None and after_info.get("content") is None
                        became_text = before_info.get("content") is None and after_info.get("content") is not None

                        if content_changed or became_binary or became_text:
                            if after_info["content"] is None:
                                ASCIIColors.info(f"[LCP Tool] Detected modified binary file '{file_name}'. Updating placeholder artifact.")
                                content_placeholder = (
                                    f"### Binary File Modified: `{file_name}`\n\n"
                                    f"This file was updated by the tool `{tool_name}`.\n"
                                    f"- **Type**: {file_ext.upper()} Binary/Data\n"
                                    f"- **Location**: `{file_path.resolve()}`\n"
                                    f"- **Size**: {file_path.stat().st_size:,} bytes\n\n"
                                    f"You can download or view this file directly from the Workspace Artifacts panel."
                                )

                                existing_art = discussion_instance.artefacts.get(file_name)
                                if existing_art:
                                    discussion_instance.artefacts.update(
                                        title=file_name,
                                        new_content=content_placeholder,
                                        new_type=atype,
                                        active=True,
                                        commit_message=f"Modified binary file by tool '{tool_name}'"
                                    )
                                else:
                                    discussion_instance.artefacts.add(
                                        title=file_name,
                                        artefact_type=atype,
                                        content=content_placeholder,
                                        active=True,
                                        commit_message=f"Synced binary file by tool '{tool_name}'"
                                    )
                                ASCIIColors.success(f"[LCP Tool '{tool_name}'] 🔄 Updated binary file reference: '{file_name}'")
                                continue

                            # Handle Text/Readable Modified Files
                            existing_art = discussion_instance.artefacts.get(file_name)
                            if existing_art:
                                atype = existing_art.get("type", "document")
                                if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir"):
                                    atype = "code"
                                elif file_ext in (".csv", ".db", ".sqlite", ".xlsx", ".xls"):
                                    atype = "data"

                                discussion_instance.artefacts.update(
                                    title=file_name,
                                    new_content=after_info["content"],
                                    new_type=atype,
                                    active=True,
                                    commit_message=f"Modified by tool '{tool_name}'"
                                )
                                ASCIIColors.success(f"[LCP Tool '{tool_name}'] 🔄 Updated artifact '{file_name}' (v{existing_art.get('version', 1)+1})")
                            else:
                                ASCIIColors.warning(f"[LCP Tool] File '{file_name}' modified on disk but no artifact found. Creating one.")
                                atype = "code" if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir") else "document"
                                discussion_instance.artefacts.add(
                                    title=file_name,
                                    artefact_type=atype,
                                    content=after_info["content"],
                                    active=True,
                                    commit_message=f"Synced by tool '{tool_name}'"
                                )

                return {"output": result, "status_code": 200}

        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing '{tool_name}': {str(e)}", "status_code": 500}
