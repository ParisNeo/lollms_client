## DOCS FOR: `lollms_client/__init__.py` (Package Interface)

**Purpose:**
The `lollms_client/__init__.py` file serves as the main entry point for the `lollms_client` package. It controls what classes, functions, and constants are directly available to users when they import from the `lollms_client` namespace.

**Key Exports:**

*   **`LollmsClient` (from `lollms_client.lollms_core`)**:
    *   The primary class for interacting with LLMs and modality services. This is the main object users will instantiate and use.

*   **`ELF_COMPLETION_FORMAT` (from `lollms_client.lollms_core` via `lollms_client.lollms_types`)**:
    *   An enumeration (`Enum`) used to specify the desired completion format (e.g., `Instruct` or `Chat`) for certain LLM backends.

*   **`MSG_TYPE` (from `lollms_client.lollms_types`)**:
    *   An enumeration (`Enum`) defining various message types used in streaming callbacks and other communication, indicating the nature of the data being sent (e.g., chunk, full message, error, step start/end).

*   **`LollmsDiscussion` (from `lollms_client.lollms_discussion`)**:
    *   A class for managing a sequence of messages in a conversation, useful for building chat applications or maintaining context for LLMs.

*   **`LollmsMessage` (from `lollms_client.lollms_discussion`)**:
    *   A data class representing a single message within a `LollmsDiscussion`.

*   **`PromptReshaper` (from `lollms_client.lollms_utilities`)**:
    *   A utility class for formatting and potentially truncating prompts based on a template and token limits.

*   **`LollmsMCPBinding` (from `lollms_client.lollms_mcp_binding`)**:
    *   The abstract base class for Model Context Protocol (MCP) bindings. Exported for type hinting or if developers want to create custom MCP bindings that integrate with the manager system.

*   **`LollmsMCPBindingManager` (from `lollms_client.lollms_mcp_binding`)**:
    *   The manager class responsible for discovering and instantiating MCP binding implementations. Exported for advanced use cases or introspection.

*   **`__version__` (str)**:
    *   The current version of the `lollms_client` package (e.g., "0.20.0").

**Usage Example (How users import from the package):**

```python
from lollms_client import (
    LollmsClient,
    MSG_TYPE,
    ELF_COMPLETION_FORMAT,
    LollmsDiscussion,
    LollmsMessage,
    PromptReshaper,
    LollmsMCPBinding, # For type hinting or custom binding development
    LollmsMCPBindingManager # For advanced binding management
)

# Get package version
from lollms_client import __version__
print(f"Using lollms_client version: {__version__}")

# Instantiate the client
client = LollmsClient(binding_name="ollama", model_name="mistral")

# Use other exported classes
discussion = LollmsDiscussion(client)
discussion.add_message("user", "Hello!")

if client.binding:
    print(f"Default completion format for binding: {client.binding.default_completion_format}")

# Define a callback using MSG_TYPE
def my_callback(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None):
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="")
    return True

client.generate_text("Tell me a joke.", streaming_callback=my_callback, stream=True)
```

**Developer Notes:**
*   The `__all__` list explicitly defines which names are exported when a user does `from lollms_client import *`. It's good practice to maintain this list if more symbols are intended for direct public use.
*   Sub-modules (like individual bindings in `llm_bindings/`, `tts_bindings/`, etc.) are generally *not* directly exported at the top level. Users interact with them through the `LollmsClient` and its associated managers.