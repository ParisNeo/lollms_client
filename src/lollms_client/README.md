# 📚 LollmsClient: Binding Discovery & Initialization Guide

This guide covers how to programmatically discover available LLM bindings, inspect their configuration requirements (`description.yaml`), instantiate a `LollmsClient` instance, and execute text generation.

---

## 1. Listing Available LLM Bindings

The `LollmsLLMBindingManager` automatically scans the `llm_bindings/` directory for subfolders containing an `__init__.py` file. You can retrieve the list of available binding names programmatically.

```python
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager

manager = LollmsLLMBindingManager()

# Get a list of all available binding names (e.g., ['ollama', 'openai', 'llama_cpp_server', ...])
available_bindings = manager.get_available_bindings()

print("Available LLM Bindings:")
for binding_name in available_bindings:
    print(f"  - {binding_name}")
```

---

## 2. Inspecting Binding Descriptions (`description.yaml`)

Every binding ships with a `description.yaml` file that defines its metadata, supported features, and required configuration parameters. You can load this YAML file and export it as a JSON-serializable dictionary.

```python
import json
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager

manager = LollmsLLMBindingManager()

# Select a specific binding to inspect
target_binding = "ollama"

# Retrieve the parsed description.yaml content as a dictionary
description = manager.get_binding_description(target_binding)

if description:
    # Print as formatted JSON content
    print(f"--- {target_binding} description.json ---")
    print(json.dumps(description, indent=2))
else:
    print(f"No description found for {target_binding}")
```

**Example JSON Output:**
```json
{
  "name": "ollama",
  "author": "Lollms Team",
  "version": "1.0",
  "description": "Binding for the Ollama local LLM server.",
  "supported_features": ["text_generation", "streaming", "vision"],
  "config_requirements": [
    {
      "name": "host_address",
      "type": "str",
      "required": false,
      "default": "http://localhost:11434"
    }
  ]
}
```

---

## 3. Populating a `LollmsClient` Instance

The `LollmsClient` is the sovereign entry point for all LLM operations. You initialize it by specifying the `llm_binding_name` and passing a dictionary of configuration parameters (`llm_binding_config`) that match the requirements found in the `description.yaml`.

```python
from lollms_client import LollmsClient

# 1. Define the configuration for the binding
#    Keys must match the parameters expected by the specific binding's __init__.py
llm_config = {
    "model_name": "gpt-4o",          # The specific model to use
    "host_address": "http://localhost:11434", # Required by ollama/llama_cpp_server
    # "api_key": "sk-...",           # Required by openai, mistral, groq, etc.
}

# 2. Instantiate the LollmsClient
#    This automatically initializes the LLM binding, TTI, TTS, and Tool bindings if specified.
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config=llm_config,
    debug=True
)

print(f"Successfully initialized LollmsClient with binding: {client.llm.binding_name}")
print(f"Active model: {client.llm.model_name}")
```

### Initialization Parameters
*   `llm_binding_name` (`str`): The exact name of the binding folder (e.g., `"openai"`, `"ollama"`, `"llama_cpp_server"`).
*   `llm_binding_config` (`dict`): A dictionary of keyword arguments passed directly to the binding's constructor.
*   `tools_binding_name` (`Optional[str]`): Name of the tools/MCP binding to mount (e.g., `"lcp"`).
*   `tools_binding_config` (`Optional[dict]`): Configuration for the tools binding.
*   `debug` (`bool`): Enables verbose ASCII-color logging.

---

## 4. Using the Client for Text Generation

Once the `LollmsClient` is instantiated, you can access the underlying binding via `client.llm` to generate text, either as a direct string or via a streaming callback.

### A. Simple Text Generation

```python
prompt = "Explain the concept of sovereignty in software architecture in one paragraph."

# generate_text is a standard method exposed by all LLM bindings
response = client.generate_text(
    prompt=prompt,
    temperature=0.7,
    max_size=4096
)

print("\n=== LLM Response ===")
print(response)
```

### B. Streaming Generation (Token by Token)

For UI integration or real-time feedback, use a streaming callback function.

```python
from lollms_client.lollms_types import MSG_TYPE

def stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict):
    # Print each chunk without a newline for continuous streaming
    print(chunk, end="", flush=True)

print("\n=== Streaming LLM Response ===")
client.generate_text(
    prompt="Write a haiku about neural networks.",
    streaming_callback=stream_callback,
    temperature=0.5
)
print("\n")
```

### C. Chat-Formatted Generation

If the binding supports chat templates (most modern bindings do), you can use `generate_from_messages`.

```python
messages = [
    {"role": "system", "content": "You are a strict and concise technical auditor."},
    {"role": "user", "content": "What are the three most critical vulnerabilities in legacy PHP applications?"}
]

response = client.generate_from_messages(
    messages=messages,
    temperature=0.2
)

print("\n=== Chat Response ===")
print(response)
```
