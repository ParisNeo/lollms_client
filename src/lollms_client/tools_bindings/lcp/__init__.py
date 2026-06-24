import sys
import os
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

    def _parse_tool_via_ast(self, py_file_path: Path) -> Optional[Dict[str, Any]]:
        """Extracts name, description, and schema from tool_ functions using AST."""
        try:
            code_text = py_file_path.read_text(encoding="utf-8")
            tree = ast.parse(code_text)
            
            def _iter_functions_ordered(node):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        yield child
                    else:
                        yield from _iter_functions_ordered(child)

            entry_fn = None
            for node in _iter_functions_ordered(tree):
                if node.name.startswith("tool_"):
                    entry_fn = node
                    break

            if not entry_fn:
                return None

            docstring = ast.get_docstring(entry_fn) or ""
            description = docstring.strip().split("\n\n")[0].strip() if docstring else "No description provided."

            doc_params = {}
            if docstring:
                for line in docstring.splitlines():
                    m = re.match(r'^(?:[-\*\d\.]+\s*)?([a-zA-Z0-9_]+)\s*(?:\(([^)]+)\))?\s*[:\-]\s*(.+)', line.strip())
                    if m:
                        doc_params[m.group(1).strip()] = m.group(3).strip()

            properties = {}
            required = []
            args_list = entry_fn.args.args
            defaults_list = entry_fn.args.defaults
            defaults_offset = len(args_list) - len(defaults_list) if defaults_list else len(args_list)

            for idx, arg in enumerate(args_list):
                arg_name = arg.arg
                # NO EXCLUSIONS: We trust the tool builder to define clean signatures
                # Only exclude *args/**kwargs
                if arg.arg in ("args", "kwargs"):
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
                "name": py_file_path.stem,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        except Exception as e:
            ASCIIColors.warning(f"AST parse failed for '{py_file_path.name}': {e}")
            return None

    def _load_tool_file(self, py_file: Path) -> bool:
        tool_name = py_file.stem
        if any(t.get("name") == tool_name for t in self.discovered_tools):
            return False

        tool_def = self._parse_tool_via_ast(py_file)
        if tool_def:
            tool_def['_python_file_path'] = str(py_file.resolve())
            self.discovered_tools.append(tool_def)
            ASCIIColors.green(f"Discovered LCP tool: {tool_name} ✓")
            return True
        return False

    def _discover_local_tools(self):
        self.discovered_tools = []
        for folder in self.tools_folders:
            if not folder or not folder.is_dir(): continue
            ASCIIColors.info(f"Discovering LCP tools in: {folder}")
            for item in folder.iterdir():
                py_file = None
                if item.is_dir():
                    py_file = item / f"{item.name}.py"
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_file = item
                if py_file and py_file.exists():
                    self._load_tool_file(py_file)

        for py_file in self.tool_files:
            if py_file and py_file.exists() and py_file.suffix == ".py":
                self._load_tool_file(py_file)
        
        ASCIIColors.info(f"Discovery complete. Found {len(self.discovered_tools)} tools.")

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
        Executes a tool in an isolated environment where:
        1. CWD = Workspace Directory (so artifacts appear as local files)
        2. No discussion/client instances are passed to the tool
        3. Tools operate on files using simple relative paths (e.g., "file.cir")
        4. AFTER execution: Sync any file changes back to artifacts automatically
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
            module_name = f"lollms_client.tools_bindings.lcp.tools.{tool_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, str(python_file_path))
            if not spec or not spec.loader:
                return {"error": f"Spec creation failed for '{tool_name}'.", "status_code": 500}

            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)

            execute_function = None
            for name in [f"tool_{tool_name}", "execute"]:
                if hasattr(tool_module, name):
                    execute_function = getattr(tool_module, name)
                    break

            if not execute_function:
                for attr_name in dir(tool_module):
                    if attr_name.startswith("tool_") and callable(getattr(tool_module, attr_name)):
                        execute_function = getattr(tool_module, attr_name)
                        break

            if not execute_function:
                return {"error": f"No entry point found in '{tool_name}'.", "status_code": 500}

            # ── ENVIRONMENT PREPARATION (Agnostic Workspace Injection) ──
            # 1. Get Workspace Directory
            workspace_dir = Path("./data_workspace")
            try:
                from lollms_client.app.server import get_discussion_workspace
                workspace_dir = get_discussion_workspace()
            except ImportError:
                pass

            workspace_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir_str = str(workspace_dir.resolve())

            # 2. Save Current CWD and Environment
            old_cwd = os.getcwd()
            old_pythonpath = os.environ.get("PYTHONPATH", "")
            old_path = os.environ.get("PATH", "")

            # 3. Snapshot workspace files BEFORE execution
            files_before = set()
            if workspace_dir.exists():
                for f in workspace_dir.rglob("*"):
                    if f.is_file():
                        files_before.add(f.relative_to(workspace_dir))

            # CRITICAL: Log the CWD BEFORE changing it
            ASCIIColors.info(f"[LCP Tool '{tool_name}'] CWD BEFORE change: {os.getcwd()}")
            ASCIIColors.info(f"[LCP Tool '{tool_name}'] Target workspace: {workspace_dir_str}")

            try:
                # 4. Change CWD to Workspace
                # This ensures tools see artifacts as simple files in "." or "./subfolder"
                os.chdir(workspace_dir_str)

                # CRITICAL: Verify CWD changed successfully
                current_cwd = os.getcwd()
                ASCIIColors.info(f"[LCP Tool '{tool_name}'] CWD AFTER change: {current_cwd}")

                if current_cwd != workspace_dir_str:
                    ASCIIColors.error(f"[LCP Tool '{tool_name}'] CRITICAL: CWD change FAILED! Expected '{workspace_dir_str}' but got '{current_cwd}'")
                    return {"error": f"Working directory change failed. Expected {workspace_dir_str} but got {current_cwd}", "status_code": 500}

                ASCIIColors.success(f"[LCP Tool '{tool_name}'] ✓ CWD successfully set to: {os.getcwd()}")

                # List files in current directory to verify artifact visibility
                current_files = os.listdir(".")
                ASCIIColors.info(f"[LCP Tool '{tool_name}'] Files visible in CWD: {current_files}")

                # 5. Add workspace to sys.path for imports
                if workspace_dir_str not in sys.path:
                    sys.path.insert(0, workspace_dir_str)

                # 6. Add workspace to PYTHONPATH environment variable
                if workspace_dir_str not in old_pythonpath:
                    new_pythonpath = f"{workspace_dir_str}{os.pathsep}{old_pythonpath}" if old_pythonpath else workspace_dir_str
                    os.environ["PYTHONPATH"] = new_pythonpath

                # 7. Add workspace to PATH environment variable for executables
                if workspace_dir_str not in old_path:
                    new_path = f"{workspace_dir_str}{os.pathsep}{old_path}"
                    os.environ["PATH"] = new_path

                ASCIIColors.info(f"[LCP Tool '{tool_name}'] Environment configured for agnostic file access")
                ASCIIColors.info(f"[LCP Tool '{tool_name}'] PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

                # 8. Execute with CLEAN params (NO discussion_instance, NO client)
                # Tools can now access files using simple paths like "file.cir" or "./subdir/file.txt"
                # We strictly pass only the parameters defined in the tool's signature.
                result = execute_function(**params)

                # ── POST-EXECUTION: Sync File Changes to Artifacts ──
                # 9. Snapshot workspace files AFTER execution
                files_after = set()
                if workspace_dir.exists():
                    for f in workspace_dir.rglob("*"):
                        if f.is_file():
                            files_after.add(f.relative_to(workspace_dir))

                # 10. Detect NEW files
                new_files = files_after - files_before
                for rel_path in new_files:
                    file_path = workspace_dir / rel_path
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        file_name = rel_path.name
                        file_ext = file_path.suffix.lower()

                        # Determine artifact type based on extension
                        atype = "document"
                        if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir"):
                            atype = "code"
                        elif file_ext in (".csv", ".db", ".sqlite", ".xlsx", ".xls"):
                            atype = "data"
                        elif file_ext in (".md", ".txt"):
                            atype = "document"
                        elif file_ext in (".json", ".yaml", ".yml"):
                            atype = "document"

                        # Check if artifact already exists
                        existing_art = discussion_instance.artefacts.get(file_name) if discussion_instance else None

                        if existing_art:
                            # Update existing artifact version
                            discussion_instance.artefacts.update(
                                title=file_name,
                                new_content=content,
                                new_type=atype,
                                active=True,
                                commit_message=f"Updated by tool '{tool_name}'"
                            )
                            ASCIIColors.success(f"[LCP Tool '{tool_name}'] Updated artifact '{file_name}' to new version")
                        else:
                            # Create new artifact
                            discussion_instance.artefacts.add(
                                title=file_name,
                                artefact_type=atype,
                                content=content,
                                active=True,
                                commit_message=f"Created by tool '{tool_name}'"
                            )
                            ASCIIColors.success(f"[LCP Tool '{tool_name}'] Created new artifact '{file_name}'")
                    except Exception as sync_err:
                        ASCIIColors.warning(f"[LCP Tool '{tool_name}'] Failed to sync file '{rel_path}' to artifact: {sync_err}")

                # 11. Detect MODIFIED files (check mtimes)
                modified_files = files_after & files_before
                for rel_path in modified_files:
                    file_path = workspace_dir / rel_path
                    try:
                        # Check if file was modified by comparing mtime
                        import time
                        mtime = file_path.stat().st_mtime
                        file_name = rel_path.name

                        existing_art = discussion_instance.artefacts.get(file_name) if discussion_instance else None
                        if existing_art:
                            # Read content and check if it changed
                            content = file_path.read_text(encoding="utf-8", errors="ignore")
                            if content != existing_art.get("content", ""):
                                # Content changed, update artifact
                                file_ext = file_path.suffix.lower()
                                atype = existing_art.get("type", "document")
                                if file_ext in (".py", ".js", ".ts", ".html", ".css", ".sql", ".cir"):
                                    atype = "code"
                                elif file_ext in (".csv", ".db", ".sqlite", ".xlsx", ".xls"):
                                    atype = "data"

                                discussion_instance.artefacts.update(
                                    title=file_name,
                                    new_content=content,
                                    new_type=atype,
                                    active=True,
                                    commit_message=f"Modified by tool '{tool_name}'"
                                )
                                ASCIIColors.success(f"[LCP Tool '{tool_name}'] Modified artifact '{file_name}' detected and updated")
                    except Exception as sync_err:
                        ASCIIColors.warning(f"[LCP Tool '{tool_name}'] Failed to check modified file '{rel_path}': {sync_err}")

                return {"output": result, "status_code": 200}

            finally:
                # 12. Restore CWD and Environment
                try:
                    os.chdir(old_cwd)
                    ASCIIColors.info(f"[LCP Tool '{tool_name}'] CWD restored to: {os.getcwd()}")
                except Exception:
                    pass

                try:
                    if old_pythonpath:
                        os.environ["PYTHONPATH"] = old_pythonpath
                    elif "PYTHONPATH" in os.environ:
                        del os.environ["PYTHONPATH"]
                except Exception:
                    pass

                try:
                    if old_path:
                        os.environ["PATH"] = old_path
                    elif "PATH" in os.environ:
                        del os.environ["PATH"]
                except Exception:
                    pass

        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing '{tool_name}': {str(e)}", "status_code": 500}
