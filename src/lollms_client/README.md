# 📚 LollmsClient: Universal Profiles, Smart Routing & Text Processing Guide

The `lollms_client` library provides a unified, sovereign interface for interacting with Large Language Models (LLMs) and various modality bindings (TTS, TTI, STT, TTM, TTV, and Tools). This guide covers the **Universal Two-Tier Profile Architecture**, **Smart Router** auto-routing, comprehensive text processing utilities, and practical examples for all helper functions.

---

## 1. Universal Two-Tier Profile Architecture

The library implements a strict, decoupled architecture separating **Connection Layer** (bindings/servers) from **Execution Layer** (models). This enables lazy loading, efficient VRAM management, and seamless multi-model ecosystems.

### Core Concepts

1.  **Binding Profiles** (`*_binding_profiles`): Define connection parameters to backend engines (host, API keys, binding type). Think of these as "server connections."
2.  **Model Profiles** (`*_model_profiles`): Define specific models, routing metadata, and execution parameters. These reference a binding profile via `binding_profile_name`.

Only profiles marked `is_default=True` are instantiated at startup. All others are **lazy-loaded** on-demand when switched to via `switch_model()`, `switch_tti()`, etc.

### Basic Profile Setup

```python
from lollms_client import LollmsClient

client = LollmsClient(
    # 1. Connection Layer: Define your servers/engines
    llm_binding_profiles={
        "local_ollama": {
            "binding_name": "ollama",
            "binding_config": {"host_address": "http://localhost:11434"},
            "is_default": True
        },
        "openai_cloud": {
            "binding_name": "openai",
            "binding_config": {"service_key": "sk-...", "host_address": "https://api.openai.com/v1"}
        },
        "vllm_omni_server": {
            "binding_name": "vllm_omni",
            "binding_config": {"host_address": "http://localhost:8091"}
        }
    },
    
    # 2. Execution Layer: Define your models
    llm_model_profiles={
        "fast_local": {
            "binding_profile_name": "local_ollama",
            "model_name": "llama3.2:3b",
            "is_default": True,  # Eagerly loaded at startup
            "vision_enabled": False,
            "forced_context_size": 8192
        },
        "gpt4o": {
            "binding_profile_name": "openai_cloud",
            "model_name": "gpt-4o",
            "vision_enabled": True
        },
        "omni_reasoner": {
            "binding_profile_name": "vllm_omni_server",
            "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "vision_enabled": True,
            "routing_config": {
                "description": "complex reasoning, math, coding, architecture",
                "complexity_tier": 3,
                "cost_per_1k_tokens": 0.0,
                "avg_latency_ms": 1500
            }
        }
    },
    
    # Other modalities follow the same pattern
    tti_binding_profiles={
        "local_diffusers": {
            "binding_name": "diffusers",
            "binding_config": {"model_name": "stabilityai/sdxl-turbo"}
        }
    },
    tti_model_profiles={
        "sdxl_turbo": {
            "binding_profile_name": "local_diffusers",
            "is_default": True
        }
    }
)
```

### Switching Models at Runtime

```python
# Switch to a different LLM profile (lazy-loads if not already instantiated)
client.switch_model("gpt4o")

# Check current active profile
print(f"Active LLM: {client.llm.model_name}")

# Switch TTI binding
client.switch_tti("sdxl_turbo")
```

### Legacy Compatibility

The client automatically maps legacy parameters to the new profile system:

```python
# Legacy style (auto-converted to "master" profiles)
client = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={"model_name": "llama3", "host_address": "http://localhost:11434"}
)
# Equivalent to creating binding_profile "master" + model_profile "master"
```

---

## 2. Smart Router (Auto-Routing)

The `smart_router` binding is a meta-binding that routes generation requests to child bindings based on **TF-IDF subject matching**, **complexity heuristics**, and **cost/latency weights**.

### Configuration

```python
client = LollmsClient(
    llm_binding_profiles={
        "smart_router_engine": {
            "binding_name": "smart_router",
            "binding_config": {
                "routing_strategy": "balanced",  # or "cost_optimized", "quality_optimized"
                "model_profiles": {
                    "fast_cheap": {
                        "binding_name": "ollama",
                        "binding_config": {"model_name": "llama3.2:3b", "host_address": "http://localhost:11434"},
                        "routing_profile": {
                            "description": "simple chat, quick answers, casual conversation",
                            "complexity_tier": 1,
                            "cost_per_1k_tokens": 0.0,
                            "avg_latency_ms": 50,
                            "priority": 1
                        }
                    },
                    "balanced": {
                        "binding_name": "ollama",
                        "binding_config": {"model_name": "llama3.1:8b"},
                        "routing_profile": {
                            "description": "general purpose, coding, writing, analysis",
                            "complexity_tier": 2,
                            "cost_per_1k_tokens": 0.0,
                            "avg_latency_ms": 200,
                            "priority": 2
                        }
                    },
                    "reasoning": {
                        "binding_name": "vllm_omni",
                        "binding_config": {"model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
                        "vision_enabled": True,
                        "routing_profile": {
                            "description": "complex reasoning, math, architecture, optimization, proof",
                            "complexity_tier": 3,
                            "cost_per_1k_tokens": 0.0,
                            "avg_latency_ms": 1500,
                            "priority": 3
                        }
                    }
                }
            }
        }
    },
    llm_model_profiles={
        "auto": {
            "binding_profile_name": "smart_router_engine",
            "is_default": True
        }
    }
)
```

### Routing Strategies

- **`balanced`**: Equal weight to subject match, complexity, cost, and latency (default: `{"subject": 0.35, "complexity": 0.35, "cost": 0.15, "latency": 0.15}`)
- **`cost_optimized`**: Prioritizes low-cost models (`{"subject": 0.2, "complexity": 0.2, "cost": 0.5, "latency": 0.1}`)
- **`quality_optimized`**: Prioritizes capability and low latency (`{"subject": 0.4, "complexity": 0.4, "cost": 0.0, "latency": 0.2}`)

### How Routing Works

1.  **Hard Filters**: If images are present, non-vision bindings are excluded.
2.  **Subject Matching**: TF-IDF cosine similarity between prompt and `routing_profile.description`.
3.  **Complexity Matching**: Prompt complexity (1-3) is compared to `complexity_tier` (1-3).
4.  **Scoring**: Weighted sum of subject match, complexity match, negative normalized cost, and latency penalty.
5.  **Tie-Breaking**: Higher `priority` wins if scores are equal.
6.  **Graceful Degradation**: If all models are filtered out, falls back to highest priority text model.

---

## 3. Fundamental LLM Operations

### Simple Text Generation

```python
response = client.generate_text(
    prompt="Explain the concept of sovereignty in software architecture.",
    temperature=0.7,
    n_predict=4096
)
```

### Streaming Generation

```python
from lollms_client.lollms_types import MSG_TYPE

def stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict):
    print(chunk, end="", flush=True)
    return True  # Return False to abort generation

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

### Cancellation

```python
client.llm.cancel()  # Signals the binding to abort generation immediately
```

---

## 4. Context Size Management & Measurement

Understanding and managing context windows is critical for preventing truncation and optimizing performance.

### 4.1 Retrieving Available Context Size

The `get_ctx_size()` method returns the maximum context window for the active model, respecting the following priority hierarchy:

**Priority Order:**
1. **Forced Override**: If `forced_context_size` is set in the model profile or via `set_forced_ctx_size()`, this value is returned immediately.
2. **Binding-Specific**: Delegates to the binding's internal `_get_ctx_size()` implementation (e.g., querying Ollama's API or reading GGUF metadata).
3. **Known Models Database**: Looks up the model in `assets/models_ctx_sizes.json` (exact match, then alias, then prefix matching).
4. **Default Fallback**: Returns `default_ctx_size` if configured, otherwise 4096.

```python
# Get context size for active model
ctx_size = client.get_ctx_size()
print(f"Active model context: {ctx_size} tokens")

# Get context size for a specific model (without switching)
ctx_size = client.get_ctx_size(model_name="llama3.1:8b")

# Force a specific context size at runtime (overrides all detection)
client.llm.set_forced_ctx_size(32768)

# Clear forced override to resume auto-detection
client.llm.set_forced_ctx_size(None)
```

**Configuration via Model Profile:**
```python
llm_model_profiles={
    "custom_model": {
        "binding_profile_name": "local_ollama",
        "model_name": "custom-model",
        "forced_context_size": 16384,  # Hard limit override
        "is_default": True
    }
}
```

### 4.2 Measuring Actual Token Usage

To measure the **actual** number of tokens in a text payload (prompt, discussion history, or generated content), use `count_tokens()`. This performs real tokenization using the active model's tokenizer, not estimation.

```python
# Measure a simple string
prompt = "Explain quantum computing in detail."
token_count = client.count_tokens(prompt)
print(f"Prompt uses {token_count} tokens")

# Measure conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]
total_tokens = sum(client.count_tokens(m["content"]) for m in messages)
print(f"Conversation history: {total_tokens} tokens")

# Measure image tokens (multimodal models)
image_path = "path/to/image.jpg"
image_tokens = client.count_image_tokens(image_path)
print(f"Image consumes {image_tokens} tokens")
```

### 4.3 Practical Context Management

Combine measurement with limits to implement safety checks:

```python
def safe_generate(client, prompt, system_prompt="", safety_margin=512):
    """Generate text with automatic context overflow protection."""

    # 1. Get limit
    max_ctx = client.get_ctx_size()
    if max_ctx is None:
        max_ctx = 4096  # Conservative fallback

    # 2. Measure actual usage
    prompt_tokens = client.count_tokens(prompt)
    system_tokens = client.count_tokens(system_prompt) if system_prompt else 0
    used_tokens = prompt_tokens + system_tokens

    # 3. Calculate available space for generation
    available = max_ctx - used_tokens - safety_margin

    if available <= 0:
        raise ValueError(
            f"Context overflow: {used_tokens} tokens used, "
            f"exceeds limit of {max_ctx} with safety margin."
        )

    # 4. Set generation limit
    return client.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        n_predict=min(4096, available)  # Cap at available space
    )

# Usage
response = safe_generate(client, long_document_prompt)
```

### 4.4 Token Counting Caching

The client implements an **MD5-based in-memory cache** for `count_tokens()` to prevent redundant backend calls:

```python
# First call: hits the LLM backend (slow)
count1 = client.count_tokens("Very long text...")

# Second call with identical text: returns cached result (instant)
count2 = client.count_tokens("Very long text...")

assert count1 == count2  # Same result, zero network overhead
```

**Note:** The cache is in-memory only and resets when the client restarts. For persistent caching across sessions, implement your own disk-based wrapper.

---

## 5. Image Generation with TTI Bindings

### Classic Mode: Raw Bytes

```python
# Generate image
image_bytes = client.generate_image(
    prompt="A cyberpunk cat riding a motorcycle, neon lights, 8k",
    negative_prompt="blurry, low quality",
    width=1024,
    height=1024
)

with open("output/cat.png", "wb") as f:
    f.write(image_bytes)

# Edit image
edited_bytes = client.edit_image(
    images="input/photo.png",
    prompt="Turn the sky into a sunset",
    mask="input/mask.png"
)
```

### Omni Mode: Text + Images

Modern engines like vLLM-Omni can return descriptive text alongside images:

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

---

## 5. Agentic Tool Calling

Enable autonomous tool use with `generate_with_tools`:

```python
def get_weather(location: str) -> dict:
    """Fetches the weather for a given location."""
    return {"location": location, "temperature": "22C", "condition": "Sunny"}

weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "parameters": [
        {"name": "location", "type": "str", "description": "The city and state", "optional": False}
    ],
    "callable": get_weather
}

result = client.generate_with_tools(
    prompt="What should I wear today in Paris?",
    tools=[weather_tool],
    max_tool_rounds=5,
    temperature=0.7
)

print("Final Answer:", result["response"])
print("Tool Calls:", result["tool_calls"])
print("Tool Results:", result["tool_results"])
```

---

## 6. Advanced Text Processing & Helper Functions

The `LollmsTextProcessor` (accessible via `client.llm.tp` or client wrapper methods) provides robust utilities for code generation, structured output, and content extraction.

### Code Editing (`edit_code`)

Efficiently patches existing code using structured diffs rather than full regeneration:

```python
original_code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
"""

instruction = "Add type hints, a docstring, and handle the case where items is empty."

result = client.llm.tp.edit_code(
    original_code=original_code,
    edit_instruction=instruction,
    language="python",
    temperature=0.1
)

if result["success"]:
    print("Patched Code:")
    print(result["content"])
    print(f"Lines changed: {result.get('lines_changed', 'N/A')}")
else:
    print("Edit failed:", result.get("error"))
```

### Multi-File Code Generation (`generate_codes`)

Generate multiple code files in one pass:

```python
files_spec = [
    {"name": "main.py", "description": "Entry point with Flask app"},
    {"name": "models.py", "description": "SQLAlchemy database models"},
    {"name": "utils.py", "description": "Helper functions for validation"}
]

result = client.generate_codes(
    prompt="Create a simple REST API for a todo list",
    files_spec=files_spec,
    temperature=0.2
)

# result is a dict: {"main.py": "code...", "models.py": "code...", ...}
for filename, code in result.items():
    print(f"--- {filename} ---")
    print(code)
```

### Single Code Generation (`generate_code`)

```python
code = client.generate_code(
    prompt="Write a Python function to calculate Fibonacci numbers using memoization",
    language="python",
    temperature=0.1
)
print(code)
```

### Tag-Based Extraction (`generate_with_tag`)

Extract specific content blocks while allowing the LLM to "think out loud":

```python
prompt = """
Given the following database schema:
Table users (id, name, email, created_at)
Table orders (id, user_id, amount, status)

Write a query to find the top 5 users by total order amount.
"""

sql_query = client.generate_with_tag(
    prompt=prompt,
    tag="sql_query",
    temperature=0.1
)

print("Extracted SQL:")
print(sql_query)
```

### Multi-Tag Extraction (`generate_with_tags`)

Generate structured documents with multiple sections:

```python
prompt = "Create a simple Python REST API using Flask with a main file and a utils file."

files_dict = client.generate_with_tags(
    prompt=prompt,
    temperature=0.2
)

# Returns: {"main.py": "...", "utils.py": "..."}
for filename, code in files_dict.items():
    print(f"--- {filename} ---")
    print(code)
```

### Structured JSON Generation (`generate_structured_content`)

Force valid JSON conforming to a schema:

```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sentiment", "confidence"]
}

result = client.generate_structured_content(
    prompt="Analyze: 'I love this new feature, it works great!'",
    schema=schema,
    max_retries=3
)
# Returns: {"sentiment": "positive", "confidence": 0.95, "keywords": ["love", "great"]}
```

### Pydantic Model Generation (`generate_structured_content_pydantic`)

```python
from pydantic import BaseModel
from typing import List

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: List[str]

result = client.generate_structured_content_pydantic(
    prompt="Analyze the sentiment of: 'I love this new feature!'",
    model=SentimentResult
)
# Returns a validated SentimentResult instance
```

### Long Context Processing

Handle documents exceeding the context window via automatic chunking and synthesis:

```python
long_text = open("large_document.txt").read()

summary = client.long_context_processing(
    text_to_process=long_text,
    contextual_prompt="Summarize the key legal risks mentioned in this document.",
    processing_type="text",
    chunk_size_ratio=0.5,  # Use 50% of context window per chunk
    overlap_ratio=0.1      # 10% overlap between chunks
)
```

### Question Answering Helpers

```python
# Boolean yes/no
is_true = client.yes_no(
    question="Is Python a compiled language?",
    context="Python is an interpreted language."
)

# Multiple choice
answers = ["Paris", "London", "Berlin", "Madrid"]
index = client.multichoice_question(
    question="What is the capital of France?",
    possible_answers=answers
)
print(f"Answer: {answers[index]}")

# Ranking
ranked_indices = client.multichoice_ranking(
    question="Rank these programming languages by performance (fastest first):",
    possible_answers=["Python", "C++", "JavaScript", "Rust"]
)
```

### Code Block Extraction

```python
text_with_code = """
Here's the solution:
```python
def hello():
    print("world")
```
And some explanation.
"""

blocks = client.extract_code_blocks(text_with_code)
# Returns: [{"language": "python", "code": "def hello():\n    print(\"world\")"}]
```

### Dynamic Skill & Note Generation

Agents can learn continuously by creating persistent skills and notes during conversation turns:

```python
# Streaming callback with secondary channel handling
from lollms_client.lollms_types import MSG_TYPE

def my_stream_listener(chunk: str, msg_type: MSG_TYPE, meta: dict):
    if msg_type == MSG_TYPE.MSG_TYPE_SKILL_CHUNK:
        print(f"[Skill Streaming: {meta.get('title')}] {chunk}", end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_SKILL_DONE:
        print(f"\n[Skill Complete] {meta.get('title')} saved.")
    elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

# If the personality is loaded from a Handbag, skills are saved to handbag/skills/
# If manual/stateless, skills are saved as discussion workspace artefacts.
response = discussion.chat(
    user_message="Teach yourself how to format YouTube Shorts and save it as a skill.",
    streaming_callback=my_stream_listener
)
```

### Thinking Block Extraction

```python
# Extract reasoning traces from models that support it
thinking, response = client.extract_thinking_blocks(raw_llm_output)

# Remove thinking blocks from final output
clean_response = client.remove_thinking_blocks(raw_llm_output)
```

---

## 7. VLM as a Tool (Vision Fallback)

When your primary model lacks vision capabilities, the `vlm_query` tool enables on-demand image analysis:

```python
# The LollmsDiscussion chat loop automatically mounts vlm_query if:
# 1. enable_vlm_query=True is passed to chat()
# 2. Active binding does NOT support vision
# 3. Another binding in lc.llms DOES support vision

result = client.chat(
    discussion,
    enable_vlm_query=True,
    images=["path/to/image.jpg"]
)
```

The LLM can then call `tool_vlm_query(image_index=0, query="What objects are in this image?")` to delegate vision tasks to a secondary VLM binding.

---

## 8. Binding Discovery & Inspection

### List Available Bindings

```python
from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBindingManager

llm_manager = LollmsLLMBindingManager()
print("Available LLM Bindings:", llm_manager.get_available_bindings())

tti_manager = LollmsTTIBindingManager()
print("Available TTI Bindings:", tti_manager.get_available_bindings())
```

### Inspect Binding Requirements

```python
from lollms_client.lollms_bindings_utils import get_binding_desc
import json

description = get_binding_desc("ollama", binding_type="llm")
print(json.dumps(description, indent=2))
```

---

## 9. Cooperative VRAM Management

For systems with limited VRAM, enable cooperative unloading to automatically free memory when switching modalities:

```python
client = LollmsClient(
    llm_binding_name="ollama",
    tti_binding_name="diffusers",
    cooperative_vram_management=True  # Unloads TTI when using LLM, and vice versa
)
```

---

## 10. Writing a New TTI Binding

Every TTI binding lives under `tti_bindings/<binding_name>/` and must expose:

- An `__init__.py` defining `BindingName = "YourBindingClass"` and the class itself, subclassing `LollmsTTIBinding`.
- A `description.yaml` declaring `global_input_parameters` and `model_input_parameters`.
- Implementations for `generate_image`, `edit_image`, `list_services`, `get_settings`, `set_settings`, and `list_models`.
- Optionally, override `generate()` for omni-support (set `supports_omni=True` in `__init__`).

```python
# tti_bindings/my_binding/__init__.py
from lollms_client.lollms_tti_binding import LollmsTTIBinding

BindingName = "MyBinding"

class MyBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        super().__init__(binding_name="my_binding", supports_omni=False, **kwargs)

    def generate_image(self, prompt, negative_prompt="", width=512, height=512, **kwargs) -> bytes:
        # Implementation here
        pass

    def edit_image(self, images, prompt, negative_prompt="", mask=None, width=None, height=None, **kwargs) -> bytes:
        pass

    def list_services(self, **kwargs): ...
    def get_settings(self, **kwargs): ...
    def set_settings(self, settings, **kwargs): ...
    def list_models(self): ...
```