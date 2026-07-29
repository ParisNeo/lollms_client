# 📚 LollmsClient: Core Architecture & Text Processing Guide

The `lollms_client` library provides a unified, sovereign interface for interacting with Large Language Models (LLMs) and various modality bindings (TTS, TTI, STT, etc.). This guide covers the fundamental architecture of the `LollmsClient`, the powerful text processing utilities, image generation via TTI bindings, and how to properly initialize and discover bindings.

---

## 1. Architecture Overview

The library is structured around three primary components:
1.  **`LollmsClient` (`lollms_core.py`)**: The main orchestrator. It manages bindings (LLM, TTS, TTI, etc.), handles cooperative VRAM management, and delegates high-level text/image operations.
2.  **`LollmsTextProcessor` (`lollms_text_processing.py`)**: A comprehensive text and code processing layer that sits on top of the LLM binding. It handles context chunking, code generation, structured JSON generation, and tag-based extraction.
3.  **`LollmsTTIBinding` (`lollms_tti_binding.py`)**: The abstract interface for Text-to-Image and Omni (text+image) generation bindings, covering classic diffusion models as well as modern omni-modality models served via engines like vLLM-Omni.

---

## 2. Binding Discovery & Initialization

Before using the client, you need to know what bindings are available and how to configure them. Every modality (`llm`, `tti`, `tts`, `stt`, `ttv`, `ttm`) has its own binding manager that automatically scans the corresponding `*_bindings/` directory.

### Listing Available Bindings

```python
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBindingManager

llm_manager = LollmsLLMBindingManager()
print("Available LLM Bindings:", llm_manager.get_available_bindings())

tti_manager = LollmsTTIBindingManager()
print("Available TTI Bindings:", tti_manager.get_available_bindings())
```

### Inspecting Binding Requirements

Every binding ships with a `description.yaml` file defining its configuration parameters. This works uniformly for LLM and TTI bindings.

```python
import json
from lollms_client.lollms_llm_binding import get_binding_desc

description = get_binding_desc("ollama", binding_type="llm")
print(json.dumps(description, indent=2))

description = get_binding_desc("vllm_omni", binding_type="tti")
print(json.dumps(description, indent=2))
```

### Populating a `LollmsClient` Instance

You initialize the `LollmsClient` by specifying binding names and configuration dictionaries per modality, including `tti_binding_name` / `tti_binding_config`.

```python
from lollms_client import LollmsClient

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "gpt-4o", "host_address": "http://localhost:11434"},
    tti_binding_name="vllm_omni",
    tti_binding_config={
        "host_address": "http://localhost:8091",
        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    },
    debug=True
)

print(f"Active LLM model: {client.llm.model_name}")
print(f"Active TTI binding: {client.tti.binding_name}")
```

You can also mount multiple TTI bindings simultaneously using `extra_ttis` and switch between them at runtime with `mount_tti`:

```python
client = LollmsClient(
    llm_binding_name="ollama",
    tti_binding_name="diffusers",
    tti_binding_config={"model_name": "stabilityai/sdxl-turbo"},
    extra_ttis={
        "omni": {
            "binding_name": "vllm_omni",
            "binding_config": {"host_address": "http://localhost:8091", "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct"}
        }
    }
)

client.mount_tti("omni")  # switches the active TTI binding to the omni engine
```

---

## 3. Fundamental LLM Operations

Once initialized, the `LollmsClient` provides direct access to the LLM binding via `client.llm` or through high-level wrapper methods.

### Simple Text Generation

```python
response = client.generate_text(
    prompt="Explain the concept of sovereignty in software architecture.",
    temperature=0.7,
    max_size=4096
)
print(response)
```

### Streaming Generation

For UI integration or real-time feedback, use a streaming callback function.

```python
from lollms_client.lollms_types import MSG_TYPE

def stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict):
    print(chunk, end="", flush=True)

client.generate_text(
    prompt="Write a haiku about neural networks.",
    streaming_callback=stream_callback,
    temperature=0.5
)
```

### Chat-Formatted Generation

```python
messages = [
    {"role": "system", "content": "You are a strict technical auditor."},
    {"role": "user", "content": "What are critical vulnerabilities in legacy PHP?"}
]

response = client.generate_from_messages(
    messages=messages,
    temperature=0.2
)
```

---

## 4. Image Generation with TTI Bindings

The `LollmsTTIBinding` interface supports two usage modes: the **classic mode** (retrocompatible, returns raw image bytes only) and the **omni mode** (returns images *and* text together for modern multimodal engines like vLLM-Omni).

### Classic Mode: `generate_image` / `edit_image`

This is the original interface, supported by every TTI binding (`diffusers`, `dalle`, `vllm_omni`, etc.) and returns raw `bytes`.

```python
image_bytes = client.generate_image(
    prompt="A cyberpunk cat riding a motorcycle, neon lights, 8k",
    negative_prompt="blurry, low quality",
    width=1024,
    height=1024
)

with open("output/cat.png", "wb") as f:
    f.write(image_bytes)
```

Editing an existing image (image-to-image or inpainting) works the same way:

```python
edited_bytes = client.edit_image(
    images="input/photo.png",
    prompt="Turn the sky into a sunset",
    mask="input/mask.png"
)
```

### Omni Mode: `generate_omni`

Modern models served via engines like [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/) can return descriptive text alongside the generated image in a single call — useful for models that narrate, caption, or reason about what they produced. Use `generate_omni` to access this richer result:

```python
result = client.generate_omni(
    prompt="Generate an image of a whale swimming through clouds above a city, and describe the mood.",
    width=1024,
    height=1024,
    modalities=["text", "image"]
)

print("Narration:", result.text)

for idx, img_bytes in enumerate(result.images):
    with open(f"output/whale_{idx}.png", "wb") as f:
        f.write(img_bytes)
```

`generate_omni` always returns a `TTIGenerationResult` object (a dict-like structure), even for classic non-omni bindings — in that case `result.text` is simply `None`, so calling code can safely check for it:

```python
result = client.generate_omni(prompt="A watercolor painting of a lighthouse")
if result.text:
    print("Model commentary:", result.text)
image_bytes = result.first_image_bytes()
```

| Method | Returns | Works with all bindings? | Supports text output? |
|---|---|---|---|
| `generate_image` / `edit_image` | `bytes` | Yes (always) | No |
| `generate_omni` | `TTIGenerationResult` (`.images`, `.text`) | Yes (falls back gracefully) | Yes, when binding supports it |

### Checking Omni Capability

Before requesting text output, you can check whether the active binding actually supports it:

```python
if getattr(client.tti, "supports_omni", False):
    result = client.generate_omni(prompt="...", modalities=["text", "image"])
else:
    image_bytes = client.generate_image(prompt="...")
```

### Listing TTI Models and Settings

```python
models = client.tti.list_models()
settings = client.tti.get_settings()
print(models, settings)
```

---

## 5. Advanced Text Processing & Extraction

The `LollmsTextProcessor` (accessible via `client.llm.tp` or through `client` wrapper methods) provides robust utilities for handling LLM outputs.

### Tag-Based Extraction: `generate_with_tag`

When you need the LLM to generate a specific block of content (like an SQL query, an HTML snippet, or a report) but want to allow the LLM to "think out loud" or add comments before/after the content, use `generate_with_tag`.

This method instructs the LLM to wrap the final answer in `<tag>...</tag>` and then extracts **only** the content inside the tag.

**Example: Extracting an SQL Query**

```python
prompt = """
Given the following database schema:
Table users (id, name, email, created_at)
Table orders (id, user_id, amount, status)

Write a query to find the top 5 users by total order amount.
"""

# The LLM might output reasoning like "I need to join tables..." 
# but we only get the SQL inside <sql_query>
sql_query = client.generate_with_tag(
    prompt=prompt,
    tag="sql_query",
    temperature=0.1
)

print("Extracted SQL:")
print(sql_query)
```

**Guaranteed Clean Extraction**: The text processor ensures that content inside `<thinking>` or `<think>` tags is never accidentally extracted. The `remove_thinking_blocks` logic runs prior to tag extraction.

### Multi-Output Extraction: `generate_with_tags`

For complex tasks like generating multiple files, structuring a document with distinct sections, or returning data alongside an explanation, use `generate_with_tags`.

This method expects the LLM to output multiple tags with `name` attributes:
```xml
<tag name="main.py">
print("Hello")
</tag>
<tag name="utils.py">
def helper(): pass
</tag>
```

It returns a dictionary mapping the names to the content.

**Example: Multi-File Code Generation**

```python
prompt = "Create a simple Python REST API using Flask with a main file and a utils file."

files_dict = client.generate_with_tags(
    prompt=prompt,
    temperature=0.2
)

# files_dict is now:
# {
#     "main.py": "from flask import Flask...",
#     "utils.py": "def format_response..."
# }

for filename, code in files_dict.items():
    print(f"--- {filename} ---")
    print(code)
    print()
```

**Example: Structured Data & Explanation**

```python
prompt = "Analyze the benefits of Rust over C++ for systems programming."

result = client.generate_with_tags(
    prompt=prompt,
    temperature=0.4
)

# result could be:
# {
#     "summary": "Rust offers memory safety without garbage collection...",
#     "comparison_table": "| Feature | Rust | C++ |\n|---|---|---|..."
# }

print("Summary:", result.get("summary"))
```

### Long Context Processing

When dealing with text that exceeds the LLM's context window, `long_context_processing` automatically chunks, processes, and synthesizes the information.

```python
long_text = open("large_document.txt").read()

summary = client.long_context_processing(
    text_to_process=long_text,
    contextual_prompt="Summarize the key legal risks mentioned in this document.",
    processing_type="text",
    chunk_size_ratio=0.5,
    overlap_ratio=0.1
)
```

### Structured JSON Generation

To force the LLM to return valid JSON conforming to a schema, use `generate_structured_content`.

```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"}
    },
    "required": ["sentiment", "confidence"]
}

result = client.generate_structured_content(
    prompt="Analyze the sentiment of: 'I love this new feature, it works great!'",
    schema=schema,
    max_retries=3
)

# result: {"sentiment": "positive", "confidence": 0.95}
```

### Code Editing

Instead of regenerating entire files, `edit_code` uses a structured diff approach to efficiently patch existing code.

```python
original_code = "def greet(name):\n    print(f'Hello {name}')"
instruction = "Add type hints and a docstring."

result = client.llm.tp.edit_code(
    original_code=original_code,
    edit_instruction=instruction,
    language="python"
)

if result["success"]:
    print(result["content"])
```

---

## 6. Writing a New TTI Binding

Every TTI binding lives under `tti_bindings/<binding_name>/` and must expose:

- An `__init__.py` defining `BindingName = "YourBindingClass"` and the class itself, subclassing `LollmsTTIBinding`.
- A `description.yaml` (or `binding_config.py` with `get_binding_desc()`) declaring `global_input_parameters` and `model_input_parameters` so the UI can auto-generate settings forms.
- Implementations for the abstract methods `generate_image`, `edit_image`, `list_services`, `get_settings`, `set_settings`, and `list_models`.
- Optionally, an override of `generate()` for bindings that support returning text alongside images (set `supports_omni=True` in `__init__`).

```python
# tti_bindings/my_binding/__init__.py
from lollms_client.lollms_tti_binding import LollmsTTIBinding

BindingName = "MyBinding"

class MyBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="my_binding", supports_omni=False, **kwargs)

    def generate_image(self, prompt, negative_prompt="", width=512, height=512, **kwargs) -> bytes:
        ...

    def edit_image(self, images, prompt, negative_prompt="", mask=None, width=None, height=None, **kwargs) -> bytes:
        ...

    def list_services(self, **kwargs): ...
    def get_settings(self, **kwargs): ...
    def set_settings(self, settings, **kwargs): ...
    def list_models(self): ...
```

---

## 7. Helper Methods

The client also provides quick wrappers for common tasks:

*   `client.yes_no(question, context)`: Returns a boolean.
*   `client.multichoice_question(question, possible_answers)`: Returns the index of the best answer.
*   `client.extract_keywords(text, num_keywords=5)`: Returns a list of keywords.
*   `client.mount_tti(alias)` / `client.mount_llm(alias)`: Switches the active binding for a modality among mounted aliases.
*   `client.list_models()`: Aggregates model lists across all active bindings (LLM, TTI, TTS, STT).
