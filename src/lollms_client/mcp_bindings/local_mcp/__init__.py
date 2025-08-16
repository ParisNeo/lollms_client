# lollms_client/mcp_bindings/local_mcp/__init__.py
import json
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from lollms_client.lollms_mcp_binding import LollmsMCPBinding
from ascii_colors import ASCIIColors, trace_exception

# This variable is used by the LollmsMCPBindingManager to identify the binding class.
BindingName = "LocalMCPBinding"

class LocalMCPBinding(LollmsMCPBinding):
    """
    Local Model Context Protocol (MCP) Binding.

    This binding discovers and executes tools defined locally in a specified folder.
    Each tool is expected to have:
    1. A Python file (`<tool_name>.py`) containing an `execute(params: Dict[str, Any]) -> Dict[str, Any]` function.
    2. A JSON file (`<tool_name>.mcp.json`) defining the tool's MCP metadata (name, description, input_schema, output_schema).
    """

    def __init__(self,
                 **kwargs: Any
                 ):
        """
        Initialize the LocalMCPBinding.

        Args:
            tools_folder_path (str|Path) a folder where to find tools
        """
        super().__init__(binding_name="LocalMCP")
        tools_folder_path = kwargs.get("tools_folder_path")
        if tools_folder_path:
            try:
                self.tools_folder_path: Optional[Path] = Path(tools_folder_path)
            except:
                self.tools_folder_path = None
        else:
            self.tools_folder_path = None
        self.discovered_tools: List[Dict[str, Any]] = []
        if not self.tools_folder_path:
            self.tools_folder_path = Path(__file__).parent/"default_tools"
        self._discover_local_tools()

    def _discover_local_tools(self):
        """Scans the tools_folder_path for valid tool definitions."""
        if not self.tools_folder_path or not self.tools_folder_path.is_dir():
            return

        self.discovered_tools = []
        ASCIIColors.info(f"Discovering local MCP tools in: {self.tools_folder_path}")

        for item in self.tools_folder_path.iterdir():
            if item.is_dir(): # Each tool in its own subdirectory
                tool_name = item.name
                mcp_json_file = item / f"{tool_name}.mcp.json"
                python_file = item / f"{tool_name}.py"

                if mcp_json_file.exists() and python_file.exists():
                    try:
                        with open(mcp_json_file, 'r', encoding='utf-8') as f:
                            tool_def = json.load(f)
                        
                        # Basic validation of MCP definition
                        if not all(k in tool_def for k in ["name", "description", "input_schema"]):
                            ASCIIColors.warning(f"Tool '{tool_name}' MCP definition is missing required fields (name, description, input_schema). Skipping.")
                            continue
                        if tool_def["name"] != tool_name:
                             ASCIIColors.warning(f"Tool name in MCP JSON ('{tool_def['name']}') does not match folder/file name ('{tool_name}'). Using folder name. Consider aligning them.")
                             tool_def["name"] = tool_name # Standardize to folder name

                        # Store the full definition and path to python file for execution
                        tool_def['_python_file_path'] = str(python_file.resolve())
                        self.discovered_tools.append(tool_def)
                        ASCIIColors.green(f"Discovered local tool: {tool_name}")
                    except json.JSONDecodeError:
                        ASCIIColors.warning(f"Could not parse MCP JSON for tool '{tool_name}'. Skipping.")
                    except Exception as e:
                        ASCIIColors.warning(f"Error loading tool '{tool_name}': {e}")
                        trace_exception(e)
                else:
                    if not mcp_json_file.exists():
                        ASCIIColors.debug(f"Tool '{tool_name}' missing MCP JSON definition ({mcp_json_file.name}). Skipping.")
                    if not python_file.exists():
                         ASCIIColors.debug(f"Tool '{tool_name}' missing Python implementation ({python_file.name}). Skipping.")
        ASCIIColors.info(f"Discovery complete. Found {len(self.discovered_tools)} local tools.")


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
        if not self.tools_folder_path:
            return []
            
        # Re-scan if needed, or if discovery hasn't happened
        if not self.discovered_tools and self.tools_folder_path:
             self._discover_local_tools()

        if specific_tool_names:
            return [tool for tool in self.discovered_tools if tool.get("name") in specific_tool_names]
        return self.discovered_tools

    def execute_tool(self,
                     tool_name: str,
                     params: Dict[str, Any],
                     lollms_client_instance: Optional[Any] = None, # Added lollms_client_instance
                     **kwargs) -> Dict[str, Any]:
        """
        Execute a locally defined Python tool.

        Args:
            tool_name (str): The name of the tool to execute.
            params (Dict[str, Any]): Parameters for the tool.
            lollms_client_instance (Optional[Any]): The LollmsClient instance, if available.
            **kwargs: Ignored by this binding.

        Returns:
            Dict[str, Any]: The result from the tool's execute function, or an error dictionary.
        """
        tool_def = next((t for t in self.discovered_tools if t.get("name") == tool_name), None)

        if not tool_def:
            return {"error": f"Local tool '{tool_name}' not found or not discovered.", "status_code": 404}
        
        python_file_path_str = tool_def.get('_python_file_path')
        if not python_file_path_str:
            return {"error": f"Python implementation path missing for tool '{tool_name}'.", "status_code": 500}
        
        python_file_path = Path(python_file_path_str)

        try:
            module_name = f"lollms_client.mcp_bindings.local_mcp.tools.{tool_name}" 
            spec = importlib.util.spec_from_file_location(module_name, str(python_file_path))
            
            if not spec or not spec.loader:
                return {"error": f"Could not create module spec for tool '{tool_name}'.", "status_code": 500}

            tool_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_module)

            if not hasattr(tool_module, 'execute'):
                return {"error": f"Tool '{tool_name}' Python file does not have an 'execute' function.", "status_code": 500}
            
            execute_function = getattr(tool_module, 'execute')
            
            # Inspect the execute function's signature
            import inspect
            sig = inspect.signature(execute_function)
            
            exec_params = {}
            if 'params' in sig.parameters: # Always pass params if expected
                exec_params['params'] = params
            
            # Conditionally pass lollms_client_instance
            if 'lollms_client_instance' in sig.parameters and lollms_client_instance is not None:
                exec_params['lollms_client_instance'] = lollms_client_instance
            elif 'lollms_client_instance' in sig.parameters and lollms_client_instance is None:
                ASCIIColors.warning(f"Tool '{tool_name}' expects 'lollms_client_instance', but it was not provided to execute_tool.")
            
            ASCIIColors.info(f"Executing local tool '{tool_name}' with effective params for its execute(): {exec_params.keys()}")
            result = execute_function(**exec_params) # Pass parameters accordingly
            
            return {"output": result, "status_code": 200}

        except Exception as e:
            trace_exception(e)
            return {"error": f"Error executing tool '{tool_name}': {str(e)}", "status_code": 500}


# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    ASCIIColors.magenta("--- LocalMCPBinding Test ---")

    # Create a temporary tools directory for testing
    test_tools_base_dir = Path(__file__).parent.parent.parent / "temp_mcp_tools_for_test" # Place it outside the package
    test_tools_base_dir.mkdir(parents=True, exist_ok=True)

    # Define a sample tool: get_weather
    tool1_dir = test_tools_base_dir / "get_weather"
    tool1_dir.mkdir(exist_ok=True)

    # MCP JSON for get_weather
    get_weather_mcp = {
        "name": "get_weather",
        "description": "Fetches the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["city"]
        },
        "output_schema": { # Optional, but good practice
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "condition": {"type": "string"},
                "unit": {"type": "string"}
            }
        }
    }
    with open(tool1_dir / "get_weather.mcp.json", "w") as f:
        json.dump(get_weather_mcp, f, indent=2)

    # Python code for get_weather
    get_weather_py = """
import random
def execute(params: dict) -> dict:
    city = params.get("city")
    unit = params.get("unit", "celsius")
    
    if not city:
        return {"error": "City not provided"}
        
    # Simulate weather fetching
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

    # Define another sample tool: sum_numbers
    tool2_dir = test_tools_base_dir / "sum_numbers"
    tool2_dir.mkdir(exist_ok=True)
    sum_numbers_mcp = {
        "name": "sum_numbers",
        "description": "Calculates the sum of a list of numbers.",
        "input_schema": {
            "type": "object",
            "properties": {"numbers": {"type": "array", "items": {"type": "number"}}},
            "required": ["numbers"]
        },
        "output_schema": {"type": "object", "properties": {"sum": {"type": "number"}}}
    }
    with open(tool2_dir / "sum_numbers.mcp.json", "w") as f:
        json.dump(sum_numbers_mcp, f, indent=2)
    sum_numbers_py = """
def execute(params: dict) -> dict:
    numbers = params.get("numbers", [])
    if not isinstance(numbers, list) or not all(isinstance(n, (int, float)) for n in numbers):
        return {"error": "Invalid input: 'numbers' must be a list of numbers."}
    return {"sum": sum(numbers)}
"""
    with open(tool2_dir / "sum_numbers.py", "w") as f:
        f.write(sum_numbers_py)


    local_mcp_binding = LocalMCPBinding(binding_name="local_mcp_test", tools_folder_path=test_tools_base_dir)

    ASCIIColors.cyan("\n1. Discovering all tools...")
    all_tools = local_mcp_binding.discover_tools()
    if all_tools:
        ASCIIColors.green(f"Discovered {len(all_tools)} tools:")
        for tool in all_tools:
            print(f"  - Name: {tool.get('name')}, Description: {tool.get('description')}")
            assert "_python_file_path" in tool # Internal check
    else:
        ASCIIColors.warning("No tools discovered.")

    ASCIIColors.cyan("\n2. Executing 'get_weather' tool...")
    weather_params = {"city": "London", "unit": "celsius"}
    weather_result = local_mcp_binding.execute_tool("get_weather", weather_params)
    ASCIIColors.green(f"Weather result: {weather_result}")
    assert "error" not in weather_result.get("output", {}), f"Weather tool execution failed: {weather_result}"
    assert weather_result.get("status_code") == 200

    ASCIIColors.cyan("\n3. Executing 'sum_numbers' tool...")
    sum_params = {"numbers": [10, 2.5, 7]}
    sum_result = local_mcp_binding.execute_tool("sum_numbers", sum_params)
    ASCIIColors.green(f"Sum result: {sum_result}")
    assert sum_result.get("output", {}).get("sum") == 19.5, f"Sum tool execution incorrect: {sum_result}"
    assert sum_result.get("status_code") == 200

    ASCIIColors.cyan("\n4. Executing non-existent tool...")
    non_existent_result = local_mcp_binding.execute_tool("do_magic", {"spell": "abracadabra"})
    ASCIIColors.warning(f"Non-existent tool result: {non_existent_result}")
    assert non_existent_result.get("status_code") == 404

    ASCIIColors.cyan("\n5. Discovering a specific tool ('sum_numbers')...")
    specific_tools = local_mcp_binding.discover_tools(specific_tool_names=["sum_numbers"])
    if specific_tools and len(specific_tools) == 1 and specific_tools[0].get("name") == "sum_numbers":
        ASCIIColors.green("Successfully discovered specific tool 'sum_numbers'.")
    else:
        ASCIIColors.error(f"Failed to discover specific tool. Found: {specific_tools}")


    # Cleanup: Remove the temporary tools directory
    import shutil
    try:
        shutil.rmtree(test_tools_base_dir)
        ASCIIColors.info(f"Cleaned up temporary tools directory: {test_tools_base_dir}")
    except Exception as e_clean:
        ASCIIColors.error(f"Could not clean up temporary tools directory: {e_clean}")

    ASCIIColors.magenta("\n--- LocalMCPBinding Test Finished ---")