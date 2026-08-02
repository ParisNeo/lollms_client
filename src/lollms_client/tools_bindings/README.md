# Lollms Tool Bindings

The `tools_bindings` directory contains the implementations for the Lollms tool execution framework. It provides a unified abstraction layer that allows the LLM to discover, register, and execute external tools and utilities seamlessly.

## 🏗️ Architecture

All tool bindings inherit from the `LollmsToolBinding` base class. This base class defines the standard contract for tool discovery (`discover_tools`, `list_tools`) and execution (`execute_tool`). 

The system is designed to be completely agnostic of the underlying transport mechanism. Whether a tool runs locally in the same Python process or remotely over a network protocol, the LLM interacts with it using the exact same interface.

## 📦 Available Bindings

### 1. LCP (LollmsCommunicationProtocol)
**Location**: `lcp/`

LCP is a lightweight, zero-dependency local tool execution framework. It allows the LLM to discover and execute custom Python scripts directly in-process, without needing to run or maintain external servers.

- **AST-Based Schema Ingestion**: Automatically extracts tool names, parameters, type annotations, and docstrings using Python's `ast` module. No duplicate JSON schemas are required.
- **Multi-Tool Files**: A single Python file can expose multiple `tool_*` functions, which are all registered as independent, callable tools.
- **Dynamic Library Mounting**: Specialized tool libraries (e.g., `semantic_data_engineer`) can be automatically mounted at runtime based on workspace context (e.g., when data files are detected).
- **Dynamic Tool Generation**: Supports compiling and registering LLM-authored Python code as active tools on the fly.

See `lcp/README.md` for detailed usage and tool creation guidelines.

### 2. MCP (Model Context Protocol)
**Location**: `mcp/`

MCP is a unified binding that connects LollmsClient to external MCP-compliant tool servers. It supports both local subprocess servers communicating over standard I/O (`stdio`) and remote servers communicating over Streamable HTTP (`http`).

- **Dual Transport**: Transparently manages `stdio` and `streamable_http` MCP transports.
- **Multi-Server Aggregation**: Connect to multiple MCP servers simultaneously. Tools are automatically namespaced using the `alias::tool_name` convention to prevent collisions.
- **Authentication Support**: Full support for remote server authentication including API keys (custom headers), Bearer tokens, and OAuth2 introspection.
- **Thread-Safe Async**: Runs a dedicated background `asyncio` event loop to ensure non-blocking IO operations while maintaining a synchronous interface for the agent loop.

See `mcp/README.md` for detailed configuration and authentication guidelines.

## 🛠️ Creating a New Tool Binding

To create a new tool binding, create a new directory under `tools_bindings/` and implement a class that inherits from `LollmsToolBinding` (located in `src/lollms_client/lollms_tools_binding.py`).

Your binding must implement the following methods:

```python
from typing import List, Dict, Any, Optional
from lollms_client.lollms_tools_binding import LollmsToolBinding

class MyCustomBinding(LollmsToolBinding):
    def __init__(self, **kwargs: Any):
        super().__init__(binding_name="my_custom_binding")
        # Initialize your connection or scan your tool directory here

    def discover_tools(self, specific_tool_names: Optional[List[str]] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Returns a list of tool dictionaries. 
        Each dictionary must contain:
        - "name": The unique tool name.
        - "description": A string describing what the tool does.
        - "input_schema": A JSON Schema dictionary defining the tool's parameters.
        """
        pass

    def list_tools(self, **kwargs) -> List[Dict[str, Any]]:
        """Alias for discover_tools."""
        return self.discover_tools(**kwargs)

    def execute_tool(self, tool_name: str, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Executes the specified tool with the given parameters.
        Must return a dictionary containing:
        - "output": The result of the tool execution.
        - "status_code": An integer (e.g., 200 for success, 500 for failure).
        """
        pass
```

### Configuration
To use your custom binding, pass its name and configuration to the `LollmsClient`:

```python
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gemma4:e2b"},
    tools_binding_name="my_custom_binding",
    tools_binding_config={
        "custom_param": "value"
    }
)
```