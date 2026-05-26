# LCP: LollmsCommunicationProtocol Tool Binding

LCP is a lightweight, zero-dependency local tool execution framework for Lollms. It allows the LLM to discover and execute custom Python scripts directly in-process, without needing to run or maintain external servers.

## 🚀 Core Features
- **AST-Based Schema Ingestion**: No need for duplicate `.mcp.json` schema files! LCP uses Python's `ast` (Abstract Syntax Tree) module to automatically extract tool names, parameters, type annotations, defaults, and descriptions directly from your Python function and its docstring on-the-fly.
- **Multi-Source Discovery**:
  - **`tools_folders`**: Scan multiple local directories simultaneously to keep your tools organized.
  - **`tool_files`**: Import standalone, isolated Python tool files directly from anywhere on disk.
- **Multiple Entry Point Formats**: Supports standard Lollms conventions (`tool_[tool_name]`), general execution triggers (`execute`), or any function starting with `tool_`.
- **Adaptive Parameter Mapping**: Inspects the function signature dynamically to resolve and bind inputs whether your code expects `args: dict`, `params: dict`, or explicit keyword parameters (e.g. `city: str, unit: str = "celsius"`).
- **Context Awareness**: Option to receive the active `LollmsClient` instance and `LollmsDiscussion` session state directly in your tool during execution.

## 🛠️ How to Write an LCP Tool

To create a tool, simply write a Python file and place it inside your LCP tools directory (by default, `src/lollms_client/tools_bindings/lcp/default_tools/`). 
You can organize tools as flat files (e.g., `get_weather.py`) or group them in subdirectories.

### Example 1: Explicit Parameter Binding (Highly Recommended)
```python
import random

def tool_get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Fetches the current weather for a given city.

    Args:
        city (str): The city name.
        unit (str, optional): Temperature unit (celsius or fahrenheit). Defaults to 'celsius'.
    """
    if not city:
        return {"error": "City not provided"}

    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temp = random.randint(-10 if unit == "celsius" else 14, 35 if unit == "celsius" else 95)

    return {
        "temperature": temp,
        "condition": random.choice(conditions),
        "unit": unit
    }
```

### Example 2: Scoped Context Injection (With LollmsClient & Discussion)
If your tool needs to access active session details, query other models, or write user artifacts, declare the context parameters in the function signature:

```python
from typing import Optional, Any
import json

def tool_file_analyzer(
    file_name: str,
    lollms_client_instance: Optional[Any] = None,
    discussion_instance: Optional[Any] = None
) -> dict:
    """
    Analyzes a file and logs details back to the active conversation.

    Args:
        file_name (str): Path or name of the file to inspect.
    """
    # lollms_client_instance and discussion_instance are automatically injected on execution
    if discussion_instance:
        discussion_instance.add_message(
            sender="system",
            content=f"Starting analysis on file: {file_name}"
        )
    return {"status": "Analysis logged."}
```

## 🔧 Configuration
Configure the LCP binding inside your client parameters by providing multiple scan directories and direct script mappings:

```python
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gemma4:e2b"},
    tools_binding_name="lcp",
    tools_binding_config={
        # Scan multiple local folders
        "tools_folders": [
            "./my_custom_tools_directory",
            "C:/shared_network_tools/lcp_library"
        ],
        # Or map standalone files directly from anywhere on disk
        "tool_files": [
            "C:/projects/utilities/matter_lock_controller.py"
        ]
    }
)
```