import sys
import os
import io
import types
import uuid
import importlib.util
import ast
import re
import traceback
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
        self._dynamic_tool_modules: Dict[str, types.ModuleType] = {}
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

        tool_defs = self._parse_tool_via_ast(py_file)
        if not tool_defs:
            ASCIIColors.red(f"[LCP Discovery] ❌ No tool_ functions found in '{file_stem}'")
            return False

        # ── 🛡️ EARLY INITIALIZATION & HEALTH GATE ──
        # Load the module once to check for and execute init_tools_library().
        # If initialization fails, we reject the entire toolset to prevent
        # the LLM from hallucinating calls to broken tools.
        module_name = f"lollms_client.tools_bindings.lcp.persistent_{file_stem}"
        try:
            if module_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(module_name, str(py_file.resolve()))
                if not spec or not spec.loader:
                    ASCIIColors.warning(f"[LCP Discovery] ⚠️ Failed to create module spec for '{file_stem}'. Skipping.")
                    return False
                tool_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = tool_module
                spec.loader.exec_module(tool_module)
            else:
                tool_module = sys.modules[module_name]

            if hasattr(tool_module, "init_tools_library") and callable(tool_module.init_tools_library):
                try:
                    tool_module.init_tools_library()
                    ASCIIColors.success(f"[LCP Init] ✅ Initialized library for '{file_stem}'")
                except Exception as init_ex:
                    ASCIIColors.error(f"[LCP Discovery] ❌ Toolset '{file_stem}' REJECTED: init_tools_library() failed: {init_ex}")
                    # Clean up the broken module from cache
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    return False

        except Exception as load_ex:
            ASCIIColors.error(f"[LCP Discovery] ❌ Toolset '{file_stem}' REJECTED: Failed to load module: {load_ex}")
            if module_name in sys.modules:
                del sys.modules[module_name]
            return False

        count = 0
        for tool_def in tool_defs:
            tool_name = tool_def.get("name")
            tool_def['_python_file_path'] = str(py_file.resolve())

            if any(t.get("name") == tool_name for t in self.discovered_tools):
                ASCIIColors.yellow(f"[LCP Discovery] Skipping '{tool_name}' (already registered)")
                continue

            self.discovered_tools.append(tool_def)
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

        self.tools_folders.append(lib_path)

        initial_tool_count = len(self.discovered_tools)

        # Scan the specific library directory for tool files
        for item in lib_path.iterdir():
            py_file = None
            if item.is_dir():
                py_file = item / f"{item.name}.py"
                if not py_file.exists():
                    sub_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]
                    if sub_py_files:
                        for fallback_py in sub_py_files:
                            self._load_tool_file(fallback_py)
                        continue
                    else:
                        continue
            elif item.suffix == ".py" and item.stem != "__init__":
                py_file = item

            if py_file and py_file.exists():
                self._load_tool_file(py_file)

        new_tool_count = len(self.discovered_tools) - initial_tool_count

        if new_tool_count > 0:
            ASCIIColors.success(f"[LCP Mount] ✅ Successfully mounted '{library_name}': {new_tool_count} tools registered.")
            return True
        else:
            ASCIIColors.warning(f"[LCP Mount] ⚠️ Library '{library_name}' mounted but no tools discovered.")
            return False

    def _discover_local_tools(self):
        self.discovered_tools = []

        for folder in self.tools_folders:
            if not folder or not folder.is_dir(): 
                ASCIIColors.warning(f"[LCP Discovery] Folder does not exist or is not a directory: {folder}")
                continue

            for item in folder.iterdir():
                py_file = None
                if item.is_dir():
                    # Standard convention: directory name matches filename (e.g., my_lib/my_lib.py)
                    py_file = item / f"{item.name}.py"
                    if not py_file.exists():
                        # Fallback: Scan for any .py file inside the directory (excluding __init__.py)
                        # This ensures tools are discovered even if the strict naming convention isn't met.
                        sub_py_files = [f for f in item.iterdir() if f.is_file() and f.suffix == ".py" and f.stem != "__init__"]
                        if sub_py_files:
                            ASCIIColors.info(f"[LCP Discovery] Standard file '{py_file.name}' not found in '{item.name}'. Falling back to scan: {[f.name for f in sub_py_files]}")
                            for fallback_py in sub_py_files:
                                self._load_tool_file(fallback_py)
                            continue
                        else:
                            ASCIIColors.warning(f"[LCP Discovery]   Directory '{item.name}' has no matching .py tool file.")
                            continue
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_file = item

                if py_file and py_file.exists():
                    self._load_tool_file(py_file)
                elif py_file:
                    ASCIIColors.warning(f"[LCP Discovery]   File missing: {py_file}")

        for py_file in self.tool_files:
            if py_file and py_file.exists() and py_file.suffix == ".py":
                self._load_tool_file(py_file)
            elif py_file:
                ASCIIColors.warning(f"[LCP Discovery] Explicit tool file missing: {py_file}")

    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        if kwargs.get("force_refresh", False) or not self.discovered_tools:
             self._discover_local_tools()
        if specific_tool_names:
            return [t for t in self.discovered_tools if t.get("name") in specific_tool_names]
        return self.discovered_tools

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        return self.discover_tools(**kwargs)

    def register_tool_from_code(self, tool_name_prefix: str, code: str) -> bool:
        """
        Dynamically registers a tool from raw Python code in memory.
        Used for LLM-generated tool artifacts.
        """
        try:
            module_name = f"dynamic_tool_{tool_name_prefix}_{uuid.uuid4().hex[:8]}"
            module = types.ModuleType(module_name)
            
            # Execute code in the module's namespace
            exec(compile(code, "<dynamic_tool>", "exec"), module.__dict__)
            
            # Find tool_ functions
            tree = ast.parse(code)
            registered_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("tool_"):
                    tool_name = node.name
                    
                    # Remove old version if exists
                    self.discovered_tools = [t for t in self.discovered_tools if t.get("name") != tool_name]
                    
                    # Extract schema
                    tool_def = self._extract_single_tool_schema(node, tool_name_prefix)
                    if tool_def:
                        # Store the module and function reference for execution
                        tool_def["_dynamic_module"] = module
                        tool_def["_python_file_path"] = None  # Mark as dynamic
                        self.discovered_tools.append(tool_def)
                        registered_count += 1
                        ASCIIColors.success(f"[LCP Dynamic] Registered tool '{tool_name}' from artefact '{tool_name_prefix}'")
            
            return registered_count > 0
            
        except Exception as e:
            ASCIIColors.error(f"[LCP Dynamic] Failed to register tool from code: {e}")
            trace_exception(e)
            return False

    def unregister_tools_by_prefix(self, tool_name_prefix: str) -> int:
        """
        Removes dynamically registered tools matching a prefix.
        Returns the count of removed tools.
        """
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
        """
        Executes a specific tool function from a potentially multi-tool file.
        """
        ASCIIColors.info(f"[LCP execute_tool] Calling: '{tool_name}'")

        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)
        if not tool_def:
            return {"error": f"Tool '{tool_name}' not found.", "status_code": 404}

        python_file_path = Path(tool_def.get('_python_file_path')) if tool_def.get('_python_file_path') else None

        # Ingest Schema Defaults
        input_schema = tool_def.get("input_schema", {})
        for prop_name, prop_info in input_schema.get("properties", {}).items():
            if prop_name not in params and isinstance(prop_info, dict) and "default" in prop_info:
                params[prop_name] = prop_info["default"]

        try:
            if python_file_path:
                module_name = f"lollms_client.tools_bindings.lcp.persistent_{python_file_path.stem}"

                # 🛑 CRITICAL ARCHITECTURAL RULE: Trust the Early Initialization.
                # The module MUST already exist in sys.modules because _load_tool_file()
                # loaded it and ran init_tools_library() at construction time.
                # If it's missing here, the tool was rejected or the registry is corrupted.
                if module_name not in sys.modules:
                    ASCIIColors.error(f"[LCP execute_tool] CRITICAL: Module '{module_name}' for tool '{tool_name}' is missing from cache!")
                    return {"error": f"Module for '{tool_name}' not initialized. The tool may have been rejected at startup.", "status_code": 500}

                tool_module = sys.modules[module_name]
            else:
                # Use the dynamically loaded module
                tool_module = tool_def.get("_dynamic_module")
                if not tool_module:
                    return {"error": f"Dynamic module missing for '{tool_name}'.", "status_code": 500}

            # CRITICAL FIX: Execute the EXACT function name requested (e.g., tool_get_table_schema)
            if not hasattr(tool_module, tool_name):
                return {"error": f"Function '{tool_name}' not found in module.", "status_code": 500}

            execute_function = getattr(tool_module, tool_name)

            # Execute with CLEAN params
            # Only pass parameters that are explicitly accepted by the tool signature, OR are generic *args/**kwargs
            import inspect
            sig = inspect.signature(execute_function)
            clean_params = {}
            for k, v in params.items():
                if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    clean_params[k] = v

            # 🚀 PERFORMANCE FIX: Execute instantly. 
            # CWD switching and artifact syncing are handled by the ChatMixin orchestrator.
            result = execute_function(**clean_params)

            # ── 🛡️ ERROR TRACKING: Enrich tool-returned errors with tracebacks ──
            if isinstance(result, dict) and result.get("success") is False and result.get("error"):
                tb_str = "".join(traceback.format_stack())
                result["traceback"] = f"Explicit tool failure captured during execution:\n{tb_str}"
                ASCIIColors.error(f"[LCP Error Tracking] Tool '{tool_name}' reported failure: {result['error']}")

            return {"output": result, "status_code": 200}

        except Exception as e:
            # ── 🛡️ ERROR TRACKING: Capture full traceback for unexpected crashes ──
            tb_str = traceback.format_exc()
            trace_exception(e)
            ASCIIColors.error(f"[LCP Error Tracking] Unexpected crash executing '{tool_name}':\n{tb_str}")
            return {
                "error": f"Error executing '{tool_name}': {str(e)}",
                "traceback": tb_str,
                "status_code": 500
            }
