# MCP Binding (Unified)

The `MCPBinding` class provides a unified interface for connecting LollmsClient to Model Context Protocol (MCP) servers. It natively supports both **local subprocess servers** communicating over standard I/O (`stdio`) and **remote servers** communicating over Streamable HTTP (`http`).

## Features

- **Dual Transport**: Transparently manages `stdio` and `streamable_http` MCP transports.
- **Multi-Server Aggregation**: Connect to multiple MCP servers simultaneously. Tools are automatically namespaced using the `alias::tool_name` convention to prevent collisions.
- **Authentication Support**: Full support for remote server authentication including API keys (custom headers), Bearer tokens, and OAuth2 introspection via `lollms_mcp_security.py`.
- **Thread-Safe Async**: Runs a dedicated background `asyncio` event loop to ensure non-blocking IO operations while maintaining a synchronous interface for the LollmsClient agent loop.
- **Dynamic Configuration**: Add or remove servers at runtime using `add_server()` and `remove_server()`.

## Configuration

The binding is initialized with a `servers` list in the keyword arguments. Each server defines its `type` (`stdio` or `http`), an `alias`, and transport-specific configurations.

### Example: Initialization

```python
from lollms_client.tools_bindings.mcp import MCPBinding

servers_config = [
    {
        "alias": "local_fs",
        "type": "stdio",
        "command": ["python", "-m", "mcp_server_filesystem", "/path/to/allowed/dir"]
    },
    {
        "alias": "remote_api",
        "type": "http",
        "url": "https://api.example.com/mcp",
        "auth_config": {
            "type": "bearer",
            "token": "your_oauth_token_here"
        },
        "timeout": 5.0
    }
]

mcp_binding = MCPBinding(servers=servers_config)
```

### HTTP Authentication (`auth_config`)

For `http` type servers, you can pass an `auth_config` dictionary:

- **API Key**: `{"type": "api_key", "key": "xxx", "header_name": "X-API-Key"}`
- **Bearer Token**: `{"type": "bearer", "token": "xxx"}`

For advanced OAuth flows using an introspection endpoint, refer to `lollms_mcp_security.py` to configure your FastMCP server, and simply pass the resulting Bearer token to the `auth_config` here.

## Usage

Once initialized, use the standard `LollmsToolBinding` interface:

```python
# Discover all tools from all connected servers
tools = mcp_binding.discover_tools()

# Execute a specific tool
# Tool names must be prefixed with the server alias
result = mcp_binding.execute_tool(
    tool_name="local_fs::read_file",
    params={"path": "/path/to/allowed/dir/test.txt"}
)
print(result)
```