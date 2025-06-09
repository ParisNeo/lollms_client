## DOCS FOR: `lollms_client/lollms_mcp_binding.py` (MCP Bindings)

**Purpose:**
This module defines the abstract base class (`LollmsMCPBinding`) for Model Context Protocol (MCP) bindings and a manager class (`LollmsMCPBindingManager`) for their discovery and instantiation. MCP bindings enable `LollmsClient` to facilitate function calling or tool use by LLMs, allowing them to interact with external systems or local Python tools.

---
### `LollmsMCPBinding` (Abstract Base Class)

**Purpose:**
`LollmsMCPBinding` is the contract for all MCP binding implementations. It ensures that each binding can discover available tools and execute them based on parameters provided, typically by an LLM's decision.

**Key Attributes/Properties:**

*   `binding_name` (str): The unique name of this MCP binding (e.g., "local_mcp", "standard_mcp").

**Abstract Methods (must be implemented by subclasses):**

*   **`discover_tools(**kwargs) -> List[Dict[str, Any]]`**:
    *   **Purpose**: Discovers available tools that the binding can interact with. Each tool's definition should conform to the MCP standard.
    *   **MCP Tool Definition Standard Fields**:
        *   `name` (str): Unique name of the tool.
        *   `description` (str): Natural language description of the tool's purpose and capabilities.
        *   `input_schema` (dict): A JSON schema defining the expected input parameters for the tool.
        *   `output_schema` (dict, optional but recommended): A JSON schema defining the structure of the tool's output.
    *   **Parameters**: `**kwargs` for binding-specific discovery options (e.g., `force_refresh`, `timeout_per_server`).
    *   **Returns**: A list of tool definition dictionaries. Returns an empty list if no tools are found or an error occurs.

*   **`execute_tool(tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]`**:
    *   **Purpose**: Executes a specified tool with the given parameters.
    *   **Parameters**:
        *   `tool_name` (str): The name of the tool to execute (may include a server alias prefix for bindings like `StandardMCPBinding`).
        *   `params` (Dict[str, Any]): A dictionary of parameters for the tool, conforming to its `input_schema`.
        *   `**kwargs`: Additional binding-specific execution options (e.g., `timeout`, `lollms_client_instance` for `LocalMCPBinding`).
    *   **Returns**: A dictionary representing the tool's output. On success, this typically includes an `"output"` key with the actual result (conforming to `output_schema`) and a `"status_code"` (e.g., 200). On failure, it should include an `"error"` key with a descriptive message and a relevant `"status_code"`.
        *   Example Success: `{"output": {"temperature": 22, "condition": "sunny"}, "status_code": 200}`
        *   Example Error: `{"error": "Tool execution failed: API timeout", "status_code": 504}`

**Concrete Methods (provided by the base class):**

*   **`get_binding_config() -> Dict[str, Any]`**:
    *   **Purpose**: Returns the configuration dictionary that was used to initialize the binding instance.
    *   **Returns**: `Dict[str, Any]`.

**Usage Example (Conceptual - Subclassing):**
```python
from lollms_client.lollms_mcp_binding import LollmsMCPBinding
from typing import List, Dict, Any

class MyCustomMCPBinding(LollmsMCPBinding):
    def __init__(self, api_endpoint: str):
        super().__init__(binding_name="my_custom_mcp")
        self.api_endpoint = api_endpoint
        # ... further initialization ...

    def discover_tools(self, **kwargs) -> List[Dict[str, Any]]:
        # Implementation to discover tools from self.api_endpoint
        # For example, fetch a /tools endpoint
        # discovered_tools = requests.get(f"{self.api_endpoint}/tools").json()
        # return discovered_tools
        return [{"name": "example_remote_tool", "description": "Does something remotely.", "input_schema": {"type": "object", "properties": {"data": {"type": "string"}}}}]

    def execute_tool(self, tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]]:
        # Implementation to call the remote tool
        # response = requests.post(f"{self.api_endpoint}/execute/{tool_name}", json=params)
        # if response.ok:
        #     return {"output": response.json(), "status_code": 200}
        # else:
        #     return {"error": response.text, "status_code": response.status_code}
        if tool_name == "example_remote_tool":
            return {"output": {"result": f"Executed with {params.get('data')}"}, "status_code": 200}
        return {"error": "Tool not found", "status_code": 404}

    def get_binding_config(self) -> Dict[str, Any]:
        return {"api_endpoint": self.api_endpoint}

```

---
### `LollmsMCPBindingManager`

**Purpose:**
The `LollmsMCPBindingManager` is responsible for discovering available MCP binding implementations within a specified directory and instantiating them. This allows `LollmsClient` to support different ways of interacting with tools (local execution, remote servers, etc.).

**Key Attributes/Properties:**

*   `mcp_bindings_dir` (Path): The directory where MCP binding implementations are stored (e.g., `lollms_client/mcp_bindings/`).
*   `available_bindings` (Dict[str, type[LollmsMCPBinding]]): A dictionary caching loaded MCP binding classes.

**Methods:**

*   **`__init__(mcp_bindings_dir: Union[str, Path] = Path(__file__).parent / "mcp_bindings")`**:
    *   **Purpose**: Initializes the manager.
    *   **Parameters**: `mcp_bindings_dir` (Union[str, Path]): Path to the directory containing MCP binding modules.

*   **`_load_binding_class(binding_name: str) -> Optional[type[LollmsMCPBinding]]`**: (Internal method)
    *   **Purpose**: Dynamically imports the Python module for the specified `binding_name` from the `mcp_bindings_dir`, retrieves the binding class (identified by a `BindingName` variable within the module's `__init__.py`), and caches it.
    *   **Parameters**: `binding_name` (str).

*   **`create_binding(binding_name: str, **kwargs) -> Optional[LollmsMCPBinding]`**:
    *   **Purpose**: Creates and returns an instance of the specified MCP binding.
    *   **Parameters**:
        *   `binding_name` (str): The name of the MCP binding to instantiate (e.g., "local_mcp", "standard_mcp").
        *   `**kwargs`: Keyword arguments to be passed to the constructor of the binding class (e.g., `tools_folder_path` for `LocalMCPBinding`, `initial_servers` for `StandardMCPBinding`).
    *   **Returns**: An instance of the requested `LollmsMCPBinding` subclass, or `None` if failed.

*   **`get_available_bindings() -> List[str]`**:
    *   **Purpose**: Scans the `mcp_bindings_dir` for valid MCP binding modules and returns their names.
    *   **Returns**: A list of strings, each being a discoverable MCP binding name.

---
### Concrete MCP Binding Implementations:

#### 1. `LocalMCPBinding`
   **Module**: `lollms_client.mcp_bindings.local_mcp`
   *   **Purpose**: Discovers and executes tools defined as local Python scripts. Each tool requires a `<tool_name>.py` file with an `execute` function and a corresponding `<tool_name>.mcp.json` file for its MCP metadata.
   *   **Key `__init__` Parameters**:
        *   `tools_folder_path` (Optional[str|Path]): Path to the directory containing tool subdirectories. If `None`, it defaults to a `default_tools` folder within the `local_mcp` binding directory (containing `file_writer`, `internet_search`, `python_interpreter`, `generate_image_from_prompt`).
   *   **`discover_tools`**: Scans the `tools_folder_path`.
   *   **`execute_tool`**: Dynamically imports and runs the `execute` function from the tool's Python file.
        *   Special `kwargs` for `execute_tool`:
            *   `lollms_client_instance` (Optional[LollmsClient]): If the local Python tool's `execute` function accepts an `lollms_client_instance` parameter, the `LollmsClient` instance calling this MCP binding will be passed. This allows tools to leverage other `LollmsClient` functionalities (e.g., TTI for an image generation tool).
   *   **Default Tools (if `tools_folder_path` is not provided)**:
        *   **`file_writer`**: Writes or appends content to a file.
            *   Inputs: `file_path`, `content`, `mode` ("overwrite" or "append").
            *   Outputs: `status`, `message`, `file_path`.
        *   **`generate_image_from_prompt`**: Generates an image using the `LollmsClient`'s active TTI binding.
            *   Inputs: `prompt`, `negative_prompt` (optional), `width` (optional), `height` (optional), `output_filename_suggestion` (optional).
            *   Outputs: `status`, `message`, `image_path` (relative), `image_url` (file URL).
            *   *Requires `lollms_client_instance` to be passed to `execute_tool` with an active TTI binding.*
        *   **`internet_search`**: Performs an internet search using DuckDuckGo.
            *   Inputs: `query`, `num_results` (default 5).
            *   Outputs: `search_results` (list of title, link, snippet), `error` (optional).
            *   *Requires `duckduckgo_search` package to be installed.*
        *   **`python_interpreter`**: Executes a Python code snippet in a restricted environment (using `RestrictedPython`).
            *   Inputs: `code`, `timeout_seconds` (default 10, currently a hint as `RestrictedPython` doesn't directly support hard timeouts).
            *   Outputs: `stdout`, `stderr`, `returned_value` (typically `None` for `exec`), `execution_status` ("success", "timeout", "error").
            *   *Requires `RestrictedPython` package.*

#### 2. `StandardMCPBinding`
   **Module**: `lollms_client.mcp_bindings.standard_mcp`
   *   **Purpose**: Connects to one or more external MCP-compliant tool servers that communicate over `stdio`. Each server is launched as a subprocess.
   *   **Key `__init__` Parameters**:
        *   `initial_servers` (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are server aliases and values are server configuration dictionaries.
            *   Server Config: `{"command": List[str], "cwd": Optional[str], "env": Optional[Dict[str, str]]}`.
   *   **`discover_tools`**: Sends `ListTools` requests to all configured and initialized servers and aggregates the results. Tool names are prefixed with `server_alias::`.
   *   **`execute_tool`**: Parses the server alias from the `tool_name` (e.g., "my_calc_server::add_numbers"), then sends a `CallTool` request to the corresponding server.
   *   **Methods**: `add_server(alias, command, ...)` and `remove_server(alias)` allow dynamic management of server connections.
   *   **Dependencies**: Requires the `mcp` library (Python SDK for MCP).

#### 3. `RemoteMCPBinding`
   **Module**: `lollms_client.mcp_bindings.remote_mcp`
   *   **Purpose**: (Conceptual) Connects to a single remote MCP-compliant tool server over HTTP (or other network transport supported by the `mcp` Python SDK's `streamablehttp_client` or similar).
   *   **Key `__init__` Parameters**:
        *   `server_url` (str): The base URL of the remote MCP server (e.g., "http://my-mcp-server.com/mcp").
        *   `alias` (str): An alias for this remote server connection (default "remote_server").
        *   `auth_config` (Optional[Dict]): Configuration for authentication (e.g., API key, OAuth details).
   *   **`discover_tools`**: Sends `ListTools` request to the configured `server_url`.
   *   **`execute_tool`**: Sends `CallTool` request to the `server_url`.
   *   **Dependencies**: Requires the `mcp` library, particularly its client components for network communication.
