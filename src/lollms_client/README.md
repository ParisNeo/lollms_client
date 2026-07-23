# 📚 LollmsClient: Core Architecture & Text Processing Guide

The `lollms_client` library provides a unified, sovereign interface for interacting with Large Language Models (LLMs) and various modality bindings (TTS, TTI, STT, etc.). This guide covers the fundamental architecture of the `LollmsClient`, the powerful text processing utilities, and how to properly initialize and discover bindings.

---

## 1. Architecture Overview

The library is structured around two primary components:
1.  **`LollmsClient` (`lollms_core.py`)**: The main orchestrator. It manages bindings (LLM, TTS, TTI, etc.), handles cooperative VRAM management, and delegates high-level text operations to the text processor.
2.  **`LollmsTextProcessor` (`lollms_text_processing.py`)**: A comprehensive text and code processing layer that sits on top of the LLM binding. It handles context chunking, code generation, structured JSON generation, and tag-based extraction.

---

## 2. Binding Discovery & Initialization

Before using the client, you need to know what bindings are available and how to configure them. The `LollmsLLMBindingManager` automatically scans the `llm_bindings/` directory.

### Listing Available Bindings

```python
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager

manager = LollmsLLMBindingManager()
available_bindings = manager.get_available_bindings()

print("Available LLM Bindings:")
for binding_name in available_bindings:
    print(f"  - {binding_name}")
```

### Inspecting Binding Requirements

Every binding ships with a `description.yaml` file defining its configuration parameters.

```python
import json
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager

manager = LollmsLLMBindingManager()
description = manager.get_binding_description("ollama")

if description:
    print(f"--- ollama description.json ---")
    print(json.dumps(description, indent=2))
```

### Populating a `LollmsClient` Instance

You initialize the `LollmsClient` by specifying the `llm_binding_name` and passing a dictionary of configuration parameters (`llm_binding_config`).

```python
from lollms_client import LollmsClient

llm_config = {
    "model_name": "gpt-4o",
    "host_address": "http://localhost:11434",
}

client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config=llm_config,
    debug=True
)

print(f"Active model: {client.llm.model_name}")
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

## 4. Advanced Text Processing & Extraction

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

## 5. Helper Methods

The client also provides quick wrappers for common tasks:

*   `client.yes_no(question, context)`: Returns a boolean.
*   `client.multichoice_question(question, possible_answers)`: Returns the index of the best answer.
*   `client.extract_keywords(text, num_keywords=5)`: Returns a list of keywords.
