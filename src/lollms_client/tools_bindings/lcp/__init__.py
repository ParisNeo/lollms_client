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

from builtins import compile as _native_compile

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
    - Host-Configurable parameters are invisible to the LLM and enforced by the host app.
    """

    def __init__(self, host_tool_configs: Optional[Dict[str, Dict[str, Any]]] = None, **kwargs: Any):
        super().__init__(binding_name="LCP")
        
        # Host-provided configurations for tools (e.g., {"system_shell": {"autonomy_level": "safe"}})
        self.host_tool_configs: Dict[str, Dict[str, Any]] = host_tool_configs or {}
        
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

    def _load_tool_file(self, py_file: Path) -> int:
        file_stem = py_file.stem

        tool_defs = self._parse_tool_via_ast(py_file)
        if not tool_defs:
            return 0

        count = 0
        for tool_def in tool_defs:
            tool_name = tool_def.get("name")
            tool_def['_python_file_path'] = str(py_file.resolve())

            if any(t.get("name") == tool_name for t in self.discovered_tools):
                continue

            self.discovered_tools.append(tool_def)
            count += 1

        return count

    def mount_tool_library(self, library_name: str) -> bool:
        base_dir = Path(__file__).parent / "default_tools"
        lib_path = base_dir / library_name

        if not lib_path.exists() or not lib_path.is_dir():
            ASCIIColors.warning(f"[LCP Mount] Library '{library_name}' not found at {lib_path}")
            return False

        if lib_path in self.tools_folders:
            for t in self.discovered_tools:
                py_path = t.get('_python_file_path')
                if py_path and lib_path in Path(py_path).resolve().parents:
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

        for py_file in self.tool_files:
            if py_file and py_file.exists() and py_file.suffix == ".py":
                self._load_tool_file(py_file)

    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        if kwargs.get("force_refresh", False) or not self.discovered_tools:
             self._discover_local_tools()
        if specific_tool_names:
            return [t for t in self.discovered_tools if t.get("name") in specific_tool_names]
        return self.discovered_tools

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
        ASCIIColors.info(f"[LCP execute_tool] Calling: '{tool_name}'")

        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)
        if not tool_def:
            return {"error": f"Tool '{tool_name}' not found.", "status_code": 404}

        python_file_path = Path(tool_def.get('_python_file_path')) if tool_def.get('_python_file_path') else None

        input_schema = tool_def.get("input_schema", {})
        for prop_name, prop_info in input_schema.get("properties", {}).items():
            if prop_name not in params and isinstance(prop_info, dict) and "default" in prop_info:
                params[prop_name] = prop_info["default"]

        try:
            if python_file_path:
                module_name = f"lollms_client.tools_bindings.lcp.persistent_{python_file_path.stem}"

                if module_name not in sys.modules:
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, str(python_file_path.resolve()))
                        if not spec or not spec.loader:
                            return {"error": f"Failed to create module spec for '{python_file_path.stem}'.", "status_code": 500}

                        tool_module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = tool_module
                        spec.loader.exec_module(tool_module)

                        # ── 🛡️ HOST CONFIGURATION INJECTION ──
                        if hasattr(tool_module, "init_tools_library") and callable(tool_module.init_tools_library):
                            library_name = python_file_path.stem
                            host_config = self.host_tool_configs.get(library_name, {})

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

                            ASCIIColors.success(f"[LCP Lazy Init] ✅ Initialized library for '{library_name}' with host configs.")
                    except Exception as init_ex:
                        ASCIIColors.error(f"[LCP execute_tool] ❌ Toolset '{python_file_path.stem}' FAILED lazy init: {init_ex}")
                        if module_name in sys.modules:
                            del sys.modules[module_name]
                        return {"error": f"Tool initialization failed: {init_ex}", "status_code": 500}
                else:
                    tool_module = sys.modules[module_name]
            else:
                tool_module = tool_def.get("_dynamic_module")
                if not tool_module:
                    return {"error": f"Dynamic module missing for '{tool_name}'.", "status_code": 500}

            if not hasattr(tool_module, tool_name):
                return {"error": f"Function '{tool_name}' not found in module.", "status_code": 500}

            execute_function = getattr(tool_module, tool_name)

            import inspect
            sig = inspect.signature(execute_function)
            clean_params = {}
            for k, v in params.items():
                if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    clean_params[k] = v

            result = execute_function(**clean_params)

            if isinstance(result, dict) and result.get("success") is False and result.get("error"):
                if "traceback" not in result:
                    result["traceback"] = None
                ASCIIColors.error(f"[LCP Error Tracking] Tool '{tool_name}' reported failure: {result['error']}")

            return {"output": result, "status_code": 200}

        except Exception as e:
            tb_str = traceback.format_exc()
            trace_exception(e)
            ASCIIColors.error(f"[LCP Error Tracking] Unexpected crash executing '{tool_name}':\n{tb_str}")
            return {
                "error": f"Error executing '{tool_name}': {str(e)}",
                "traceback": tb_str,
                "status_code": 500
            }