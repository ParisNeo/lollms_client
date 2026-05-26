# lollms_client/tools_bindings/lcp/__init__.py
import sys
import importlib.util
import ast
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

from lollms_client.lollms_tools_binding import LollmsToolBinding
from ascii_colors import ASCIIColors, trace_exception

# This variable is used by the LollmsTOOLBindingManager to identify the binding class.
BindingName = "LCPBinding"

class LCPBinding(LollmsToolBinding):
    """
    Local LollmsCommunicationProtocol (LCP) Binding.

    This binding discovers and executes tools defined locally across multiple folders
    and direct Python file paths. It uses Python's standard Abstract Syntax Tree (AST) 
    to dynamically parse function annotations and docstrings, removing the need for 
    separate JSON schemas.
    """

    def __init__(self,
                 **kwargs: Any
                 ):
        """
        Initialize the LCPBinding.

        Args:
            tools_folders (List[str|Path], optional): List of folders containing tools.
            tool_files (List[str|Path], optional): List of direct Python tool files.
            tools_folder_path (str|Path, optional): Legacy fallback single folder.
        """
        super().__init__(binding_name="LCP")
        
        # 1. Resolve Multi-Folder Config (with legacy fallback)
        self.tools_folders: List[Path] = []
        folders_input = kwargs.get("tools_folders") or kwargs.get("tools_folder_path")
        if folders_input:
            if isinstance(folders_input, (str, Path)):
                self.tools_folders.append(Path(folders_input))
            elif isinstance(folders_input, list):
                for f in folders_input:
                    self.tools_folders.append(Path(f))
        else:
            # Default fallback to default_tools subdirectory
            self.tools_folders.append(Path(__file__).parent / "default_tools")

        # 2. Resolve Direct Tool Files
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
        """Backward-compatibility shim. Returns the primary tools folder."""
        return self.tools_folders[0] if self.tools_folders else None

    def _parse_tool_via_ast(self, py_file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Inspects a Python tool file and extracts its name, description, and input_schema
        directly from its function signature and docstrings using Abstract Syntax Trees.
        """
        try:
            code_text = py_file_path.read_text(encoding="utf-8")
            tree = ast.parse(code_text)
            
            # Find the main entry point function in deterministic source order
            def _iter_functions_ordered(node):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        yield child
                    else:
                        yield from _iter_functions_ordered(child)

            entry_fn = None
            for node in _iter_functions_ordered(tree):
                if node.name.startswith("tool_") or node.name == "execute":
                    entry_fn = node
                    break

            if not entry_fn:
                return None
                
            # Extract Description from Docstring
            docstring = ast.get_docstring(entry_fn) or ""
            description = docstring.strip().split("\n\n")[0].strip() if docstring else "No description provided."
            
            # Extract Parameters from Function Arguments
            properties = {}
            required = []
            
            args_list = entry_fn.args.args
            defaults_list = entry_fn.args.defaults
            
            # Align defaults with arguments
            defaults_offset = len(args_list) - len(defaults_list) if defaults_list else len(args_list)
            
            for idx, arg in enumerate(args_list):
                arg_name = arg.arg
                # Exclude standard context parameters
                if arg_name in ("lollms_client_instance", "client", "discussion_instance", "discussion", "args", "params"):
                    continue
                    
                # Extract Type from Annotation
                arg_type = "string" # default
                if arg.annotation:
                    anno_str = ast.unparse(arg.annotation).strip().lower()
                    if "int" in anno_str:
                        arg_type = "integer"
                    elif "float" in anno_str or "number" in anno_str:
                        arg_type = "number"
                    elif "bool" in anno_str:
                        arg_type = "boolean"
                    elif "list" in anno_str or "array" in anno_str:
                        arg_type = "array"
                    elif "dict" in anno_str or "object" in anno_str:
                        arg_type = "object"
                        
                # Extract Default value if present
                has_default = idx >= defaults_offset
                default_val = None
                if has_default and defaults_list:
                    default_node = defaults_list[idx - defaults_offset]
                    try:
                        default_val = ast.literal_eval(default_node)
                    except:
                        default_val = ast.unparse(default_node).strip("'\"")
                        
                properties[arg_name] = {
                    "type": arg_type,
                    "description": f"Parameter '{arg_name}'"
                }
                
                if has_default:
                    properties[arg_name]["default"] = default_val
                else:
                    required.append(arg_name)
                    
            # Docstring Fallback: If the function takes a generic dict (args/params), parse parameters from the docstring
            if not properties and docstring:
                lines = docstring.splitlines()
                for line in lines:
                    line_stripped = line.strip()
                    m = re.match(r'^(?:[-\*\d\.]+\s*)?([a-zA-Z0-9_]+)\s*(?:\(([^)]+)\))?\s*[:\-]\s*(.+)', line_stripped)
                    if m:
                        p_name = m.group(1).strip()
                        p_type_raw = (m.group(2) or "string").lower().strip()
                        p_desc = m.group(3).strip()
                        
                        # Skip keywords or headings
                        if p_name.lower() in ("args", "parameters", "returns", "example", "usage", "note", "class", "def", "raises"):
                            continue
                            
                        p_type = "string"
                        if "int" in p_type_raw: p_type = "integer"
                        elif "float" in p_type_raw or "number" in p_type_raw: p_type = "number"
                        elif "bool" in p_type_raw: p_type = "boolean"
                        elif "list" in p_type_raw or "array" in p_type_raw: p_type = "array"
                        elif "dict" in p_type_raw or "object" in p_type_raw: p_type = "object"
                        
                        properties[p_name] = {
                            "type": p_type,
                            "description": p_desc
                        }
                        if "optional" not in p_type_raw and "default" not in p_type_raw:
                            required.append(p_name)
                            
            tool_name = py_file_path.stem
            return {
                "name": tool_name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        except Exception as e:
            ASCIIColors.warning(f"Failed to AST-parse tool '{py_file_path.name}': {e}")
            return None

    def _load_tool_file(self, py_file: Path) -> bool:
        """Helper to parse a Python tool file and append its definition."""
        tool_name = py_file.stem
        
        # Check if already discovered to avoid duplicates across multi-folder scans
        if any(t.get("name") == tool_name for t in self.discovered_tools):
            return False

        tool_def = None

        # Attempt AST-based dynamic schema extraction
        try:
            tool_def = self._parse_tool_via_ast(py_file)
        except Exception as e:
            ASCIIColors.warning(f"AST-parsing failed for '{tool_name}': {e}")

        if tool_def:
            tool_def['_python_file_path'] = str(py_file.resolve())
            self.discovered_tools.append(tool_def)
            ASCIIColors.green(f"Discovered LCP tool: {tool_name} ✓")
            return True
        else:
            ASCIIColors.warning(f"Skipping tool '{tool_name}': Could not compile a valid schema from AST.")
            return False

    def _discover_local_tools(self):
        """Scans the configured tools folders and direct files for valid Python tools using AST parsing."""
        self.discovered_tools = []
        
        # 1. Scan configured directories
        for folder in self.tools_folders:
            if not folder or not folder.is_dir():
                continue
            ASCIIColors.info(f"Discovering local LCP tools in directory: {folder}")
            for item in folder.iterdir():
                py_file = None
                if item.is_dir():
                    py_file = item / f"{item.name}.py"
                elif item.suffix == ".py" and item.stem != "__init__":
                    py_file = item

                if py_file and py_file.exists():
                    self._load_tool_file(py_file)

        # 2. Load direct tool files
        if self.tool_files:
            ASCIIColors.info(f"Loading direct LCP tool files: {len(self.tool_files)} file(s)")
            for py_file in self.tool_files:
                if py_file and py_file.exists() and py_file.suffix == ".py":
                    self._load_tool_file(py_file)

        ASCIIColors.info(f"Discovery complete. Found {len(self.discovered_tools)} local LCP tools.")


    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover available local tools.

        Args:
            specific_tool_names (Optional[List[str]]): If provided, filter discovery
                                                       to only these tool names.
            **kwargs: Ignored by this binding.

        Returns:
            List[Dict[str, Any]]: A list of discovered tool definitions.
        """
        if not self.tools_folders and not self.tool_files:
            return []
            
        # Re-scan if needed, or if discovery hasn't happened
        if not self.discovered_tools:
             self._discover_local_tools()

        if specific_tool_names:
            return [tool for tool in self.discovered_tools if tool.get("name") in specific_tool_names]
        return self.discovered_tools

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Return a list of tools formatted for consumption by the discussion chat module.
        This implementation simply forwards to ``discover_tools``.
        """
        return self.discover_tools(**kwargs)

    def execute_tool(self,
                     tool_name: str,
                     params: Dict[str, Any],
                     lollms_client_instance: Optional[Any] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Execute a locally defined Python tool.

        Args:
            tool_name (str): The name of the tool to execute.
            params (Dict[str, Any]): Parameters for the tool.
            lollms_client_instance (Optional[Any]): The LollmsClient instance, if available.
            **kwargs: Can include 'discussion' or 'discussion_instance'
        """
        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)

        if not tool_def:
            return {"error": f"Local tool '{tool_name}' not found or not discovered.", "status_code": 404}
        
        python_file_path_str = tool_def.get('_python_file_path')
        if not python_file_path_str:
            return {"error": f"Python implementation path missing for tool '{tool_name}'.", "status_code": 500}
        
        python_file_path = Path(python_file_path_str)

        # Ingest Schema Defaults for Omitted Parameters
        input_schema = tool_def.get("input_schema", {})
        properties = input_schema.get("properties", {})
        for prop_name, prop_info in properties.items():
            if prop_name not in params and isinstance(prop_info, dict) and "default" in prop_info:
                params[prop_name] = prop_info["default"]

        try:
            module_name = f"lollms_client.tools_bindings.lcp.tools.{tool_name}" 
            
            # Unregister module from sys.cache to force recompilation and load latest disk changes
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, str(python_file_path))
            
            if not spec or not spec.loader:
                return {"error": f"Could not create module spec for tool '{tool_name}'.", "status_code": 500}

            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)

            # Adaptive Entry Point Locator
            # Supports tool_[name], execute, and legacy formats
            execute_function = None
            possible_names = [f"tool_{tool_name}", "execute", f"tool_{tool_name.replace('-', '_')}"]
            for name in possible_names:
                if hasattr(tool_module, name):
                    execute_function = getattr(tool_module, name)
                    break
            
            if not execute_function:
                # Fallback: scan for any callable starting with "tool_"
                for attr_name in dir(tool_module):
                    if attr_name.startswith("tool_") and callable(getattr(tool_module, attr_name)):
                        execute_function = getattr(tool_module, attr_name)
                        break

            if not execute_function:
                return {
                    "error": f"Tool '{tool_name}' Python file does not have a valid entry point (expected 'tool_{tool_name}' or 'execute').",
                    "status_code": 500
                }
            
            # Inspect signature to align arguments dynamically
            import inspect
            sig = inspect.signature(execute_function)
            
            exec_params = {}
            if 'params' in sig.parameters:
                exec_params['params'] = params
            elif 'args' in sig.parameters:
                exec_params['args'] = params
            elif 'args' not in sig.parameters and 'params' not in sig.parameters:
                # Flat mapping for functions taking explicit parameters
                exec_params = params.copy()
            
            # Pass client context
            if 'lollms_client_instance' in sig.parameters:
                exec_params['lollms_client_instance'] = lollms_client_instance
            elif 'client' in sig.parameters:
                exec_params['client'] = lollms_client_instance

            # Pass discussion context
            disc = kwargs.get("discussion") or kwargs.get("discussion_instance")
            if 'discussion_instance' in sig.parameters:
                exec_params['discussion_instance'] = disc
            elif 'discussion' in sig.parameters:
                exec_params['discussion'] = disc
            
            ASCIIColors.info(f"Executing local tool '{tool_name}' with arguments: {list(exec_params.keys())}")
            result = execute_function(**exec_params)
            
            return {"output": result, "status_code": 200}

        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing tool '{tool_name}': {str(e)}", "status_code": 500}


# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    ASCIIColors.magenta("--- LCPBinding Test ---")

    # Create a temporary tools directory for testing
    test_tools_base_dir = Path(__file__).parent.parent.parent / "temp_mcp_tools_for_test" # Place it outside the package
    test_tools_base_dir.mkdir(parents=True, exist_ok=True)

    # Define a sample tool: get_weather
    tool1_dir = test_tools_base_dir / "get_weather"
    tool1_dir.mkdir(exist_ok=True)

    # Python code for get_weather
    get_weather_py = """
import random
def tool_get_weather(city: str, unit: str = "celsius") -> dict:
    \"\"\"
    Fetches the current weather for a given city.
    
    Args:
        city (str): The city name.
        unit (str, optional): Temperature unit (celsius or fahrenheit). Defaults to 'celsius'.
    \"\"\"
    if not city:
        return {"error": "City not provided"}
        
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temp = random.randint(-10 if unit == "celsius" else 14, 35 if unit == "celsius" else 95)
    
    return {
        "temperature": temp,
        "condition": random.choice(conditions),
        "unit": unit
    }
"""
    with open(tool1_dir / "get_weather.py", "w") as f:
        f.write(get_weather_py)

    local_mcp_binding = LCPBinding(binding_name="lcp_test", tools_folders=[test_tools_base_dir])

    ASCIIColors.cyan("\n1. Discovering all tools...")
    all_tools = local_mcp_binding.discover_tools()
    if all_tools:
        ASCIIColors.green(f"Discovered {len(all_tools)} tools:")
        for tool in all_tools:
            print(f"  - Name: {tool.get('name')}, Description: {tool.get('description')}")
            print(f"    Schema properties: {list(tool.get('input_schema', {}).get('properties', {}).keys())}")
            assert "_python_file_path" in tool # Internal check
    else:
        ASCIIColors.warning("No tools discovered.")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(test_tools_base_dir)
        ASCIIColors.info(f"Cleaned up temporary tools directory: {test_tools_base_dir}")
    except Exception as e_clean:
        ASCIIColors.error(f"Could not clean up temporary tools directory: {e_clean}")

    ASCIIColors.magenta("\n--- LCPBinding Test Finished ---")
