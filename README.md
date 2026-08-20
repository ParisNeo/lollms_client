# LoLLMs Client Library

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/lollms_client.svg)](https://badge.fury.io/py/lollms_client)
[![Python Versions](https://img.shields.io/pypi/pyversions/lollms_client.svg)](https://pypi.org/project/lollms-client/)
[![Downloads](https://static.pepy.tech/personalized-badge/lollms-client?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/lollms-client)
[![Documentation - Usage](https://img.shields.io/badge/docs-Usage%20Guide-brightgreen)](DOC_USE.md)
[![Documentation - Developer](https://img.shields.io/badge/docs-Developer%20Guide-blue)](DOC_DEV.md)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/lollms_client.svg?style=social&label=Star&maxAge=2592000)](https://github.com/ParisNeo/lollms_client/stargazers/)
[![GitHub issues](https://img.shields.io/github/issues/ParisNeo/lollms_client.svg)](https://github.com/ParisNeo/lollms_client/issues)

<img width="941" height="1672" alt="lolms rchitecture" src="https://github.com/user-attachments/assets/bebb5958-5037-4e7c-b167-a198530c1438" />



**`lollms_client`** is a powerful and flexible Python library designed to simplify interactions with the **LoLLMs (Lord of Large Language Models)** ecosystem and various other Large Language Model (LLM) backends. It provides a unified API for text generation, multimodal operations (text-to-image, text-to-speech, etc.), and robust function calling through the Model Context Protocol (MCP).

Whether you're connecting to a remote LoLLMs server, an Ollama instance, the OpenAI API, or running models locally using GGUF (via `llama-cpp-python` or a managed `llama.cpp` server), Hugging Face Transformers, or vLLM, `lollms-client` offers a consistent and developer-friendly experience.

## ⚡ Why LoLLMs Client? Key Competitive Advantages

`lollms_client` is not just another API wrapper. It is a highly optimized, production-grade coordination engine built to grant Large Language Models true local and hybrid autonomy.

### 🧠 Biological-Inspired 5-Tier Memory System (Memory Level 0-4)
*   **Persistent Multi-Level Storage**: Features an advanced, cognitive hierarchical storage system consisting of **Volatile Scratchpad** (single-turn intermediate reasoning), **Working Memory** (directly injected into prompt space), **Deep Memory** (stubbed as handles to prevent context bloating), **Archived Memory** (historical backup), and **Episodic Memory** (immutable step-by-step trace of interactions).
*   **Memory Decay & Consolidation**: Memories decay logarithmically over time. Frequently referenced concepts are automatically reinforced.
*   **AI-Assisted Dreaming (`dream()`)**: During idle cycles, an automated "dream consolidation pass" cleans up old data. Important rules and architecture patterns are maintained, while low-importance noise is forgotten.

### 🤖 Sovereign Multi-Step Agency & MCP Integration
*   **Deterministic State Control**: Uses a robust **Observe-Think-Act-Verify** state machine. If the model generates a thought process but fails to act, the parser detects the omission, restricts reasoning, and injects precise structural corrections to guide it back on track.
*   **Model Context Protocol (MCP)**: Native integration of local and remote MCP tool registries (e.g., File I/O, Web Search, Sandboxed Code Execution) giving agents direct hands-on power.
*   **Real-Time Performance Metrics**: Tracks exact performance statistics per spinoff agent turn, capturing **Time to First Token (TTFT)**, **Average Generation Speed (TPS)**, and total token usage stored directly in the discussion database.

### 💻 Aider-Style Structural Code Patching & Text Processor (`tp`)
*   **Non-Destructive SEARCH/REPLACE Edits**: Features the **Lollms Text Processor** layer (`lc.llm.tp`). Instead of full document rewrites, it supports structural, aider-style Search/Replace code patches.
*   **Schema & Pydantic Enforcement**: Easily output structured data with built-in schema validation and truncation-recovery algorithms. If the model's output gets cut off, the Text Processor reconstructs the JSON tree and repairs the output.
*   **Yes/No & Multi-Choice Helpers**: Built-in helper primitives to perform discrete evaluations, ranking, and classification.

### 🖼️ Multimodal Context Isolation & Multi-Image Fusion
*   **Fine-Grained Vision Controls**: Multi-image inputs can be selectively toggled active or inactive on each message turn without purging original databases—significantly reducing vision model token costs.
*   **Qwen Multi-Image Fusion**: Diffusers integration supports cutting-edge local image-to-image engines capable of single-image semantic edits and advanced **multi-image fusion, character swaps, pose transfers, and background transplants**.

### 🔌 Standardized, Multi-Provider Architecture
*   **Unified Configuration**: Run local GGUFs (via llama.cpp/python), local Ollama instances, or scale to OpenAI, Anthropic, Gemini, Groq, and OpenRouter using a single, unified `llm_binding_config` block.
*   **Automatic Context Compression**: Dynamically monitors context token sizes, summarizes old turns using targeted AI-synthesizers, and collapses long historical sequences into a clean, single-turn **Project State Synopsis** to keep models sharp and conversational context pristine.

### 🧠 Universal Lazy Profiles & Multi-Model Routing
*   **Memory-Efficient Multi-Binding**: Define declarative registries of `LollmsBindingProfile` configurations for *any* modality (LLM, TTI, TTS, STT, TTV, TTM). Only the binding marked as `is_default` is eagerly loaded at startup. Other bindings are instantiated lazily *on-demand* when you switch to them, saving massive amounts of RAM and VRAM.
*   **Dynamic Switching**: Seamlessly switch between a local coding model, a cloud vision model, and a fast text model at runtime using `client.switch_model("alias")` or `client.switch_tti("alias")`. Instantiated bindings are cached for zero-latency re-activation.
*   **100% Backward Compatible**: If you use the legacy `llm_binding_name` or `extra_llms` parameters, they are automatically registered as the `"master"` profiles and eagerly instantiated, ensuring older code runs without modification.

## Installation

You can install `lollms_client` directly from PyPI:

```bash
pip install lollms-client
```

This will install the core library. Some bindings may require additional dependencies (e.g., `llama-cpp-python`, `torch`, `transformers`, `ollama`, `vllm`, `Pillow` for image utilities, `docling` for document parsing). The library attempts to manage these using `pipmaster`, but for complex dependencies (especially those requiring compilation like `llama-cpp-python` with GPU support), manual installation might be preferred.

## Core Generation Methods

The `LollmsClient` provides several methods for generating text, catering to different use cases.

### Basic Text Generation (`generate_text`)

This is the most straightforward method for generating a response based on a simple prompt.

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors
import os

# Callback for streaming output
def simple_streaming_callback(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nStreaming Error: {chunk}")
    return True # True to continue streaming

try:
    # Initialize client to connect to a LoLLMs server.
    # All binding-specific parameters now go into the 'llm_binding_config' dictionary.
    lc = LollmsClient(
        llm_binding_name="lollms", # This is the default binding
        llm_binding_config={
            "host_address": "http://localhost:9642", # Default port for LoLLMs server
            # "service_key": "your_lollms_api_key_here" # Get key from LoLLMs UI -> User Settings if security is enabled
            # "verify_ssl_certificate": True #if false the ssl certifcate verification will be ignored (only used when using https in lollms service address)
        }
    )

    prompt = "Tell me a fun fact about space."
    ASCIIColors.yellow(f"Prompt: {prompt}")

    # Generate text with streaming
    ASCIIColors.green("Streaming Response:")
    response_text = lc.generate_text(
        prompt,
        n_predict=100,
        stream=True,
        streaming_callback=simple_streaming_callback
    )
    print("\n--- End of Stream ---")

    # The 'response_text' variable will contain the full concatenated text
    # if streaming_callback returns True throughout.
    if isinstance(response_text, str):
        ASCIIColors.cyan(f"\nFull streamed text collected: {response_text[:100]}...")
    elif isinstance(response_text, dict) and "error" in response_text:
        ASCIIColors.error(f"Error during generation: {response_text['error']}")

except ValueError as ve:
    ASCIIColors.error(f"Initialization Error: {ve}")
    ASCIIColors.info("Ensure a LoLLMs server is running or configure another binding.")
except ConnectionRefusedError:
    ASCIIColors.error("Connection refused. Is the LoLLMs server running at http://localhost:9642?")
except Exception as e:
    ASCIIColors.error(f"An unexpected error occurred: {e}")

```

### Generating from Message Lists (`generate_from_messages`)

For more complex conversational interactions, you can provide the LLM with a list of messages, similar to the OpenAI Chat Completion API. This allows you to define roles (system, user, assistant) and build multi-turn conversations programmatically.

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors
import os

def streaming_callback_for_messages(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

try:
    # Example for an Ollama binding
    # Ensure you have Ollama installed and model 'llama3' pulled (e.g., ollama pull llama3)
    lc = LollmsClient(
        llm_binding_name="ollama", 
        llm_binding_config={
            "model_name": "llama3",
            "host_address": "http://localhost:11434" # Default Ollama address
        }
    )

    # Define the conversation history as a list of messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that specializes in programming."},
        {"role": "user", "content": "Hello, what's your name?"},
        {"role": "assistant", "content": "I am an AI assistant created by Google."},
        {"role": "user", "content": "Can you explain recursion in Python?"}
    ]

    ASCIIColors.yellow("\nGenerating response from messages:")
    response_text = lc.generate_from_messages(
        messages=messages,
        n_predict=200,
        stream=True,
        streaming_callback=streaming_callback_for_messages
    )
    print("\n--- End of Message Stream ---")
    ASCIIColors.cyan(f"\nFull collected response: {response_text[:150]}...")

except Exception as e:
    ASCIIColors.error(f"Error during message generation: {e}")

```

### Agentic Tool-Enabled Generation (`generate_with_tools`)

The `generate_with_tools` method enables LLMs to act as agents that can discover, call, and chain external tools. This is the foundation for building autonomous AI assistants that can search the web, query databases, execute code, or interact with APIs.

**Key Features:**
- **File-based tools**: Load tools from lollms-format Python scripts (`tool_*.py` with docstring-described arguments)
- **Inline tools**: Pass tool dicts directly with `{"name": ..., "callable": ..., "parameters": [...]}`
- **Automatic execution**: The agentic loop parses `<tool>` tags, executes tools, and feeds results back
- **Multi-step reasoning**: The model can chain multiple tool calls across rounds to solve complex tasks

**Tool Format (lollms scripts):**
A tool script is a Python file containing:
- `TOOL_LIBRARY_NAME`, `TOOL_LIBRARY_DESC`, `TOOL_LIBRARY_ICON` metadata
- An optional `init_tools_library()` for dependency setup
- One or more `tool_*` functions with docstring-described arguments

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
from pathlib import Path

# Create a simple calculator tool file
tool_content = '''
TOOL_LIBRARY_NAME = 'Calculator'
TOOL_LIBRARY_DESC = 'Basic arithmetic operations'
TOOL_LIBRARY_ICON = '🧮'

def tool_calculate(args: dict):
    """
    Perform arithmetic calculations.

    Args:
        args: dict with keys:
            - expression (str): Mathematical expression to evaluate (e.g., "2 + 2 * 5")
    """
    try:
        expression = args.get('expression', '')
        # Safe evaluation using limited operators
        allowed = {"__builtins__": {}}
        allowed.update({k: v for k, v in __import__('math').__dict__.items()})
        result = eval(expression, allowed, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
'''

tool_path = Path.home() / ".lollms_hub" / "tools" / "calculator.py"
tool_path.parent.mkdir(parents=True, exist_ok=True)
tool_path.write_text(tool_content, encoding="utf-8")

try:
    lc = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config={
            "models_path": "data/models/llama_cpp_models",
            "ctx_size": 4096,
            "n_gpu_layers": -1,
        }
    )
    
    # Load a model first...
    # lc.llm.load_model("your-model.gguf")

    result = lc.generate_with_tools(
        prompt="What is the square root of 144 plus 25?",
        tools=[str(tool_path)],  # Pass file path(s) or inline dicts
        system_prompt="You are a helpful math assistant. Use the calculator tool when needed.",
        temperature=0.7,
        n_predict=1024,
        max_tool_rounds=5,
        auto_execute=True,
    )

    ASCIIColors.green(f"\nFinal Answer: {result['response']}")
    ASCIIColors.cyan(f"Tool calls made: {len(result['tool_calls'])}")
    for tc in result['tool_calls']:
        print(f"  - {tc['name']}: {tc['parameters']}")

except Exception as e:
    ASCIIColors.error(f"Error during tool generation: {e}")
```

**Return Value:**
The method returns a comprehensive result dict:

```python
{
    "response": str,        # Final text answer from the model
    "tool_calls": [          # All tool calls made during the session
        {"round": int, "name": str, "parameters": dict, "raw": str}
    ],
    "tool_results": [        # All tool execution results
        {"round": int, "name": str, "result": dict}
    ],
    "rounds": int,           # Number of agentic rounds executed
    "pending_tool": dict,     # Present only if auto_execute=False (manual mode)
}
```

### Advanced Structured Content Generation (`generate_structured_content`)

The `generate_structured_content` method is a powerful utility for forcing an LLM's output into a specific JSON format. It's ideal for extracting information, getting consistent tool parameters, or any task requiring reliable, machine-readable output.

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import json
import os

try:
    # Using Ollama as an example binding
    lc = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llama3"})

    text_block = "John Doe is a 34-year-old software engineer from New York. He loves hiking and Python programming."

    # Define the exact JSON structure you want
    output_template = {
        "full_name": "string",
        "age": "integer",
        "profession": "string",
        "city": "string",
        "hobbies": ["list", "of", "strings"] # Example of a list in schema
    }

    ASCIIColors.yellow(f"\nExtracting structured data from: '{text_block}'")
    ASCIIColors.yellow(f"Using schema: {json.dumps(output_template)}")

    # Generate the structured data
    extracted_data = lc.generate_structured_content(
        prompt=f"Extract the relevant information from the following text:\n\n{text_block}",
        schema=output_template, # Note: parameter is 'schema'
        temperature=0.0 # Use low temperature for deterministic structured output
    )

    if extracted_data:
        ASCIIColors.green("\nExtracted Data (JSON):")
        print(json.dumps(extracted_data, indent=2))
    else:
        ASCIIColors.error("\nFailed to extract structured data.")

except Exception as e:
    ASCIIColors.error(f"An error occurred during structured content generation: {e}")
```

---

## 🧠 Lollms Text Processor

The **Lollms Text Processor** is a high-level utility designed to turn raw LLM generations into **production-ready workflows**. It handles long documents, structured outputs, robust code generation, intelligent editing, and reliable parsing.

It is directly accessible via:

```python
lc.llm.tp
```

### 🔧 Initialization

```python
from lollms_client import LollmsClient

lc = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={
        "model_name": "llama3",
        "host_address": "http://localhost:9642",
        "service_key": "the service key"
    }
)

llm = lc.llm
tp = lc.llm.tp
```

* `llm` provides low-level text generation primitives
* `tp` is the **Text Processor**, ready to use out of the box

### 📚 1. Long Context Processing

The Text Processor automatically handles documents that exceed the model’s context window by chunking, synthesizing intermediate results, and producing a final consolidated output.

#### Text generation from a very long document

```python
summary = tp.long_context_processing(
    text_to_process=long_document,
    contextual_prompt="Summarize the main findings about climate change",
    processing_type="text"
)
```

#### Structured extraction from long context

```python
result = tp.long_context_processing(
    text_to_process=long_document,
    contextual_prompt="Extract all people mentioned with their roles",
    processing_type="structured",
    schema={
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"}
                    }
                }
            }
        }
    }
)
```

#### Yes / No question over long documents

```python
answer = tp.long_context_processing(
    text_to_process=long_document,
    contextual_prompt="Does this document mention Marie Curie?",
    processing_type="yes_no",
    return_explanation=True
)
```

### 💻 2. Code Generation and Editing

#### Single-file code generation

```python
code = tp.generate_code(
    prompt="Create a binary search function",
    language="python"
)
```

#### Multi-file project generation

```python
files = tp.generate_codes(
    prompt="Create a Flask web app with an HTML frontend"
)
```

#### Efficient code editing (non-destructive)

```python
updated_code = tp.edit_code(
    original_code=existing_code,
    edit_instruction="Add error handling and logging",
    language="python"
)
```

Unlike naïve prompting, edits are **structural**, not full rewrites.

### 🧩 3. Structured Content Generation

#### Using JSON Schema

```python
data = tp.generate_structured_content(
    prompt="Create a presentation about AI",
    schema={
        "type": "object",
        "properties": {
            "slides": {
                "type": "array",
                "items": {"type": "object"}
            }
        }
    }
)
```

#### Using Pydantic models

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

person = tp.generate_structured_content_pydantic(
    prompt="Create a person named Alice, age 30",
    pydantic_model=Person
)
```

✔ Automatic validation
✔ Truncation recovery
✔ Agent-friendly outputs

### 🧠 4. LLM Helper Utilities

#### Yes / No questions

```python
answer = tp.yes_no(
    question="Is Marie Curie a scientist?",
    context="Marie Curie was a physicist...",
    return_explanation=True
)
```

#### Multiple-choice questions

```python
choice = tp.multichoice_question(
    question="What field did Marie Curie work in?",
    possible_answers=["Biology", "Physics", "Chemistry"]
)
```

#### Text summarization

```python
summary = tp.summerize_text(text="Long article...")
```

#### Keyword extraction

```python
keywords = tp.extract_keywords(
    text="Long article...",
    num_keywords=5
)
```

### 🧪 5. Response Parsing and Cleanup

#### Extract reasoning / thinking blocks

```python
thoughts = tp.extract_thinking_blocks(llm_response)
```

#### Remove reasoning blocks

```python
clean_text = tp.remove_thinking_blocks(llm_response)
```

#### Extract code blocks (legacy support)

```python
blocks = tp.extract_code_blocks(
    text=llm_response,
    format="markdown"
)
```

### ✨ Key Features

* ✅ Automatic **long-context handling**
* ✅ XML-based code generation (no fragile backticks)
* ✅ Truncation recovery for JSON and code
* ✅ Non-destructive, structured code editing
* ✅ JSON Schema & Pydantic support
* ✅ Decision helpers (yes/no, multichoice, ranking)
* ✅ Graceful fallback strategies

---

## 🧠 Universal Lazy Profiles & Multi-Model Routing

`lollms_client` features a powerful, memory-efficient multi-binding architecture via **Universal Lazy Profiles**. Instead of eagerly instantiating all models/engines at startup (which wastes RAM and VRAM), you define declarative registries of `LollmsBindingProfile` configurations for *any* modality (LLM, TTI, TTS, STT, TTV, TTM). 

Only the binding marked as `is_default` is loaded at startup. Other bindings are instantiated lazily *on-demand* when you switch to them. This system is 100% backward compatible. If you use the legacy `llm_binding_name` or `tti_binding_name` parameters, they are automatically registered as the `"master"` profiles and marked as defaults.

### 1. Multi-Model LLM Routing
Define multiple LLM profiles to handle different domains or complexity tiers.

```python
from lollms_client import LollmsClient, LollmsBindingProfile

# 1. Define LLM profiles declaratively (None of these are instantiated yet)
llm_profiles = {
    "cloud_vision": LollmsBindingProfile(
        name="cloud_vision",
        binding_name="openai",
        binding_config={"model_name": "gpt-4o", "service_key": "your-key"},
        vision_enabled=True,
        is_default=True  # This one will be loaded at startup
    ),
    "local_coder": LollmsBindingProfile(
        name="local_coder",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "qwen2.5-coder:7b"},
        forced_context_size=32768
    ),
    "local_fast": LollmsBindingProfile(
        name="local_fast",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "llama3.2:3b"}
    )
}

# 2. Initialize the client (Only "cloud_vision" is instantiated)
lc = LollmsClient(llm_profiles=llm_profiles)

# 3. Use the default model
response1 = lc.generate_text("Explain quantum physics.")

# 4. Dynamically switch to the local coder (Instantiated on-the-fly and cached)
lc.switch_model("local_coder")
response2 = lc.generate_text("Write a Python script to sort a list.")

# 5. Switch back to the default cloud vision model (Retrieved from cache, no re-instantiation)
lc.switch_model("cloud_vision")
```

### 2. Multi-Engine TTI (Text-to-Image) Routing
The profile system is universal. You can define profiles for image generation engines and switch between a local Stable Diffusion model and a cloud DALL-E API seamlessly.

```python
# Define TTI profiles
tti_profiles = {
    "local_sd": LollmsBindingProfile(
        name="local_sd",
        binding_name="diffusers",
        binding_config={"model_name": "stable-diffusion-v1-5"},
        is_default=True
    ),
    "cloud_dalle": LollmsBindingProfile(
        name="cloud_dalle",
        binding_name="openai",
        binding_config={"model_name": "dall-e-3", "service_key": "your-key"}
    )
}

lc = LollmsClient(llm_binding_name="ollama", tti_profiles=tti_profiles)

# Generate with default local SD
img_bytes = lc.generate_image("A cyberpunk cat")

# Switch to DALL-E 3 for a specific prompt
lc.switch_tti("cloud_dalle")
img_bytes = lc.generate_image("A hyperrealistic oil painting of a dog")
```

### 3. The Smart Router Binding
For automated routing, the `smart_router` meta-binding evaluates incoming prompts using TF-IDF (subject matching), complexity heuristics, and weighted constraints (cost/latency) to delegate generation to the optimal child model automatically.

```python
from lollms_client import LollmsClient

router_config = {
    "routing_strategy": "cost_optimized",
    "model_profiles": {
        "cheap_fast": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5:3b"},
            "routing_profile": {"description": "fast simple tasks", "complexity_tier": 1}
        },
        "smart_coder": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5-coder:7b"},
            "routing_profile": {"description": "python code debugging", "complexity_tier": 2}
        }
    }
}

lc = LollmsClient(llm_binding_name="smart_router", llm_binding_config=router_config)

# The router automatically selects "cheap_fast" for simple prompts
response = lc.generate_text("What is 2+2?")

# The router automatically selects "smart_coder" for coding prompts
response = lc.generate_text("Write a Python script to sort a list.")
```

### 4. VLM as a Tool (`vlm_query`)
When using a text-only LLM (or Smart Router without a VLM child), you can enable VLM collaboration via the `vlm_query` LCP tool. This allows the text LLM to delegate visual analysis to a secondary VLM on-demand.

```python
# Client with a text-only LLM and a separate VLM profile
lc = LollmsClient(
    llm_profiles={
        "text_llm": LollmsBindingProfile(name="text_llm", binding_name="ollama", binding_config={"model_name": "llama3"}, is_default=True),
        "vlm": LollmsBindingProfile(name="vlm", binding_name="ollama", binding_config={"model_name": "llava"}, vision_enabled=True)
    }
)

# The user sends an image and asks a question
discussion.chat(
    user_message="Look at the diagram in the image. What are the main components?",
    images=["base64_encoded_image_data..."],
    enable_vlm_query=True, # CRITICAL: Explicitly enable the fallback tool
    max_reasoning_steps=10
)
# The LLM (llama3) will realize it needs vision, call `tool_vlm_query(0, "Identify the main components.")`,
# receive the text description from the VLM (llava), and synthesize the final answer.
```

---

## 👜 The Unified LollmsPersonality & Handbag System

In `lollms_client`, the **`LollmsPersonality`** is the sovereign unit of execution. It replaces the legacy bifurcation between stateless "prompt wrappers" and stateful "autonomous agents". A single `LollmsPersonality` scales progressively from a simple 5-line `SOUL.md` to a fully-armed, stateful, multi-persona **Crew Handbag** with tools, memory, skills, and multimodal assets.

### 1. The Two Tiers of Personality

#### Tier 1: The Simple Personality (Stateless)
A simple personality is just a system prompt and an optional RAG data source. It has no memory, no workspace, and no tools. It is 100% backward compatible with all legacy `LollmsDiscussion.chat(personality=...)` calls.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_personality import LollmsPersonality

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llama3"})
db_manager = LollmsDataManager("sqlite:///discussion.db")
discussion = LollmsDiscussion.create_new(lollms_client=client, db_manager=db_manager)

# A simple, stateless personality
simple_pers = LollmsPersonality(
    name="MathTutor",
    author="user",
    category="general",
    description="A helpful math tutor.",
    system_prompt="You are a helpful math tutor. Explain concepts simply."
)

# Drops directly into a discussion
discussion.chat(user_message="Explain Pythagoras' theorem.", personality=simple_pers)
```

#### Tier 2: The Handbag Personality (Stateful)
A Handbag Personality is created by pointing a `LollmsPersonality` to a **Handbag** folder. This lazily instantiates a `SkillsManager`, a `MemoryManager`, and a `ToolRegistry`. It can be dropped into a `LollmsDiscussion` just like a simple personality, but it brings state and capabilities.

```python
from lollms_client.lollms_personality import LollmsPersonality

# A complex, stateful personality loaded from a Handbag folder
# Memory, Skills, and Tools are lazily instantiated upon loading.
autonomous_pers = LollmsPersonality.from_handbag("./my_research_handbag")

# It still drops directly into a discussion!
# The LollmsDiscussion loop will automatically detect the stateful components
# and inject the memory, skills, and tools into the active turn.
discussion.chat(user_message="Refactor the utils file.", personality=autonomous_pers)
```

### 2. The Handbag (Portable Resource Folder)

The **Handbag** is a self-contained, portable folder that carries ALL of a personality's resources. It allows you to package multiple personas, tools, skills, RAG knowledge, memory, and multimodal assets (like 3D models or voice samples) into a single directory.

#### Folder Structure

```text
my_handbag/
├── SOUL.md                   # Primary personality (YAML frontmatter + Markdown body)
├── handbag.yaml              # Optional manifest (name, default_personality, skills_mode)
├── coworkers/                # Multi-persona support (Crew Handbag)
│   ├── researcher/
│   │   └── SOUL.md
│   └── coder/
│       └── SOUL.md
├── tools/                    # Shared LCP tools (.py files or subdirs)
│   ├── my_custom_tool.py
│   └── another_tool/
│       └── another_tool.py
├── skills/                   # SKILL.md files (personality creates/updates these)
│   ├── python_patterns/
│   │   └── SKILL.md
│   └── ...
├── rag/                      # RAG documents (text files for retrieval)
│   ├── doc1.txt
│   └── doc2.md
├── memory/                   # Memory database
│   └── memory.db
└── assets/                   # Multimodal assets (icons, voice samples, 3D models)
    ├── logo.png
    └── voice.wav
```

#### Creating a Handbag

You can automatically scaffold a new, empty Handbag folder using the `Handbag.create_structure()` helper.

```python
from lollms_client.lollms_personality.handbag import Handbag

# Scaffolds the complete folder structure
hb_path = Handbag.create_structure("./my_research_handbag", name="Research Handbag")
```

#### Populating the Handbag

*   **Personalities (`coworkers/`)**: Create a subdirectory for each persona. Inside each, place a `SOUL.md` file. This forms your **Crew**.
*   **Tools (`tools/`)**: Place LCP tool scripts (`.py` files) here. All personas in the Handbag share this tool pool.
*   **Skills (`skills/`)**: Create subdirectories for each skill, each containing a `SKILL.md` file.
*   **RAG (`rag/`)**: Place text files here. The Handbag automatically indexes them.
*   **Memory (`memory/`)**: A `memory.db` SQLite file will be automatically created here.
*   **Assets (`assets/`)**: Store custom graphics, audio, or 3D models for NPC-style applications.

You can optionally configure the `handbag.yaml` manifest:

```yaml
name: Research Handbag
version: '1.0'
description: A handbag for research tasks.
default_personality: researcher  # Name of the folder in personalities/
skills_mode: always_visible      # "always_visible", "loadable", or "mixed"
```

### 3. The Crew Handbag (Multi-Persona Routing)

A single Handbag can contain multiple `SOUL.md` files (e.g., `coworkers/coder/` and `coworkers/researcher/`). When loaded, the primary `LollmsPersonality` object acts as a router.

All crewmates share the **same** Handbag tools, memory, and assets, but they have different system prompts and RAG data sources.

```python
from lollms_client.lollms_personality import LollmsPersonality

# Load the Crew Handbag
crew_pers = LollmsPersonality.from_handbag("./my_crew_handbag")

# The primary personality is determined by handbag.yaml (e.g., "researcher")
print(f"Active persona: {crew_pers.name}")

# List all available personas in the crew
print("Available crewmates:", crew_pers.list_crewmates())

# Dynamically switch the active persona (tools and memory remain shared)
crew_pers.switch_crewmate("coder")

# Use it in a discussion
discussion.chat(user_message="Write a Python script.", personality=crew_pers)
```

### 4. Workspace Sovereignty

The host application (via `LollmsDiscussion` or direct instantiation) dictates the working directory. The `Handbag` provides the *resources* (tools, memory, skills), but **does not force its own workspace path**. 

If you use a Handbag-backed Personality inside a `LollmsDiscussion`, the Discussion's `workspace_path` is used for file operations. If you use the Handbag's standalone workspace, you must explicitly pass it to the Discussion.

```python
# The Discussion dictates the workspace
discussion = LollmsDiscussion.create_new(
    lollms_client=client, 
    db_manager=db_manager,
    workspace_path="./my_project_workspace" # Explicit workspace
)

# The personality brings tools and memory, but uses the discussion's workspace
discussion.chat(
    user_message="Analyze the data.", 
    personality=autonomous_pers
)
```

---

## 🧠 LollmsDiscussion: Cognitive Sessions & Artefacts

`LollmsDiscussion` is a stateful, thread-safe conversational engine that bridges transient LLM tokens and permanent, versioned knowledge storage. It implements the **Agentic State Machine** and the **Dual-Stream Artefact System**. It is composed of nine orthogonal mixins:

1.  **`CoreMixin`**: Lifecycle, ORM proxy, message CRUD, and thread-safe DB commits.
2.  **`ChatMixin`**: The agentic reasoning loop, tool execution orchestration, and stream parsing.
3.  **`UtilsMixin`**: Branch management, export normalization, and context token auditing.
4.  **`PromptMixin`**: System prompt construction and XML tag post-processing.
5.  **`MemoryMixin`**: Integration with `LollmsMemoryManager` for tiered persistent memory, episodic memory saving, and graph relationship traversal.
6.  **`FileImportMixin`**: Multi-modal ingestion (PDF, DOCX, Data) and Dual-Stream storage.
7.  **`InternetImportMixin`**: Web content extraction and quality scoring for internet-based RAG.
8.  **`ExportMixin`**: Standalone Artefact Archive (`.laa`) and Linked Artefact Bundle (`.lab`) export/import protocols.
9.  **`BranchMixin`**: Directed Acyclic Graph (DAG) branch discovery, navigation, forking, and merging.

### The Dual-Stream Artefact System (.lam Protocol)

To solve the "Context vs. Tool" paradox, artefacts are split into two streams:
1.  **Physical Twin**: Raw bytes on disk (CSV, SQLite, PNG). Executable by local tools.
2.  **Logical Twin (`.lam`)**: High-density text schemas. Injected into the LLM context.

**Visibility Tiers**:
*   `[C]` FULL: Verbatim content injected.
*   `[U]` TREE_UNLOCKABLE: Listed in directory, excluded from context. The LLM can unlock this via `<add_files_to_context>`.
*   `[L]` TREE_LOCKED: Excluded from context. Cannot be unlocked by the LLM.

### The Agentic Loop & `<done/>` Protocol

The `chat()` method is an Agentic State Machine. The LLM is given explicit control via the `<done/>` tag.
1.  **Round 1 Short-Circuit**: If the LLM generates pure text without functional tags, the loop breaks immediately.
2.  **Action Continuation**: If the LLM emits `<tool>` or `<artifact>`, the action is executed, and a mandate is injected.
3.  **Explicit Termination**: The loop only breaks if the LLM emits `<done/>` on a new line, or if `max_reasoning_steps` is reached.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager

client = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llama3"})
db_manager = LollmsDataManager("sqlite:///discussion.db")
discussion = LollmsDiscussion.create_new(lollms_client=client, db_manager=db_manager)

response = discussion.chat(
    user_message="Analyze data.csv and build a plot.",
    enable_artefacts=True,
    enable_code_execution=True,
    max_reasoning_steps=20
)
```

### The `chat()` Method API

The `chat()` method is the primary entry point for the `LollmsDiscussion` session. It orchestrates the entire agentic loop, including pre-hydration, multi-step reasoning, tool execution, and self-healing file restoration.

```python
def chat(
    self,
    user_message: str,
    personality=None,
    branch_tip_id=None,
    tools=None,
    add_user_message: bool = True,
    images=None,
    debug: bool = False,
    remove_thinking_blocks: bool = True,
    enable_image_generation: bool = True,
    enable_image_editing:    bool = True,
    auto_activate_artefacts: bool = True,
    enable_inline_widgets:        bool = False,
    enable_notes:                 bool = True,
    enable_skills:                bool = True,
    enable_forms:                 bool = True,
    enable_books:                 bool = False,
    enable_presentations:         bool = False,
    memory_manager=None,
    enable_artefacts:             bool = True,
    enable_memory:                bool = True,
    enable_auto_dream:            bool = True,
    enable_deep_memory_pulling:   bool = True,
    prehydrate_rag:               bool = True,
    max_reasoning_steps:          int = 20,
    enable_in_message_status:     bool = False,
    enable_sub_agents:            bool = False,
    forward_artefact_chunks:      bool = False,
    fast_artefact_replicas:       Optional[List[str]] = None,
    tolerance_level:              Optional[str] = "strict",
    allow_dynamic_tools:          bool = False,
    enable_data_tools:            bool = True,
    enable_code_execution:        bool = False,
    suppress_images:              bool = False,
    debug_export:                 bool = False,
    **kwargs
) -> Dict[str, Any]:
```

**Key Parameters:**

*   **Core Conversation**: `user_message`, `personality`, `branch_tip_id`, `add_user_message`, `images`, `suppress_images` (set `True` for non-vision LLMs).
*   **Feature Flags**: `enable_artefacts`, `enable_inline_widgets`, `enable_notes`, `enable_skills`, `enable_forms`, `enable_books`, `enable_presentations`, `enable_image_generation`, `enable_image_editing`.
*   **Security Gates**: `allow_dynamic_tools` (LLM writes its own tools), `enable_code_execution` (arbitrary Python string execution), `enable_data_tools` (auto-mounts `semantic_data_engineer` if data files exist).
*   **Memory & RAG**: `memory_manager`, `enable_memory`, `enable_deep_memory_pulling`, `enable_auto_dream`, `prehydrate_rag`.
*   **Debugging & UI**: `debug`, `debug_export`, `enable_in_message_status`, `remove_thinking_blocks`, `event_mode` (see Event Modes below).

**Return Value:**
Returns a dictionary containing the complete result of the conversational turn:

```python
{
    "user_message": LollmsMessage,  # The user message object
    "ai_message": LollmsMessage,    # The final AI message object
    "sources": List[Dict],          # RAG sources retrieved
    "artefacts": List[Dict],        # Artifacts created/modified this turn
    "memory_report": Dict,          # Memory operations report
    "dream_report": Optional[Dict], # Auto-dream consolidation report
    "was_cancelled": bool           # Cancellation status
}
```

### Event Modes & Streaming Protocol

The `chat()` method supports multiple event reporting strategies via the `event_mode` parameter (using the `EventMode` enum from `lollms_client.lollms_types`). This allows host applications to choose between parsing raw text tags or consuming structured callback events.

| Mode | Behavior | Use Case |
| :--- | :--- | :--- |
| **`EventMode.PROCESSING_TAG_MODE`** (Default) | Injects `<processing>` tags into the `MSG_TYPE_CHUNK` stream. | Simple text-based UIs, CLIs. |
| **`EventMode.FULL_CALLBACK_MODE`** | Emits specific `MSG_TYPE_*` events via the callback with structured metadata. No `<processing>` tags in text. | Rich UI applications that render dedicated panels. |
| **`EventMode.MIXED_MODE`** | Emits both `<processing>` tags and structured `MSG_TYPE_*` events. | Debugging or transitioning applications. |
| **`EventMode.SILENT_MODE`** | Suppresses all event reporting. Only final text is streamed. | Background tasks. |

**Structured Events in `FULL_CALLBACK_MODE`:**
*   `MSG_TYPE_TOOL_START`: Meta `{"tool_name": str, "parameters": dict}`
*   `MSG_TYPE_TOOL_END`: Meta `{"tool_name": str, "success": bool, "output": str, "error": str|None}`
*   `MSG_TYPE_ARTEFACT_BUILD_START`: Meta `{"title": str, "art_type": str, "language": str|None, "is_patch": bool}`
*   `MSG_TYPE_ARTEFACT_BUILD_END`: Meta `{"title": str, "art_type": str, "version": int, "success": bool, "error": str|None}`
*   `MSG_TYPE_CONTEXT_UPDATE`: Meta `{"action": str, "files": list[str], "status": str}`

### Multi-Source Tool Orchestration (LCP)

The `chat()` method enforces a **Strict Sovereign Opt-In Doctrine**. Tools are ONLY activated if explicitly requested:
1.  **Personality Handbag Tools**: Auto-mounted if the personality contains a `tools` attribute.
2.  **Explicit `tools` parameter**: Accepts a dict of callables OR a list of string names matching the LCP registry.
3.  **Auto-Mounted Data Tools**: `semantic_data_engineer` is auto-mounted IF `enable_data_tools=True` AND data files exist.

```python
# Explicitly activate default LCP tools by name
discussion.chat(
    user_message="Search the internet and run python code.",
    tools=["tool_internet_search", "tool_execute_python_code"],
    enable_code_execution=True
)
```

**LCP Tool Agnosticism Doctrine:**
LCP tools are strictly agnostic by default. They must **NEVER** accept `discussion_instance` or `lollms_client_instance` as input parameters (unless explicitly required for advanced patterns like recursive sub-agent spawning). The LCP AST parser filters these internal parameters out when building the JSON schema for the LLM. Tools must rely on CWD for file resolution and communicate results back solely via their return dictionary.

### Cognitive Checkpoint System

To prevent context window bloat and preserve the LLM's cognitive state across turns, the `ChatMixin` implements three core mechanisms:

1.  **Smart Tool Output Offloading**: When a tool returns >1500 tokens, the output is intercepted before entering context. Structured data is replaced with compact markers; unstructured text is saved to a `.log` file and registered as a `TREE_UNLOCKABLE` artifact.
2.  **Unfinished Intent Interceptor**: If the LLM stops generating before emitting a functional tag but its text matches an intent pattern (e.g., "Let me query..."), the system intercepts and forces a correction round.
3.  **Cognitive Scratchpad Protocol**: The LLM maintains a `scratchpad.md` artifact for intermediate hypotheses during multi-step analysis, preventing contradictory context accumulation.

### Cancellation & Interrupt Protocol

The `chat()` method implements a **Thread-Safe Cancellation Protocol** using a boolean flag, ensuring long-running loops can be interrupted without database corruption.

```python
# Start generation in a background thread
import threading
def run_chat():
    response = discussion.chat(user_message="Analyze this 1GB CSV...")
    print(response["was_cancelled"])

thread = threading.Thread(target=run_chat)
thread.start()

# User clicks "Stop"
discussion.cancel_generation()

# Check status
if discussion.is_generation_cancelled():
    print("Stopping...")

# The cancel state is automatically reset after chat() returns.
```

The loop checks the flag at four critical safe points: start of reasoning round, during streaming, post-generation, and tool cleanup. Partial messages are saved with a `"[Generation cancelled by user]"` marker.

### Tool Failure Visibility & Anti-Loop Protocol

The system ensures the LLM **always sees error details** when a tool fails, implementing a three-pronged defense against infinite retry loops:

1.  **Raw Dict Failure Detection**: Checks for `success: False` or non-200 `status_code` before examining sanitized text.
2.  **Error-Aware Sanitization**: Includes the actual `error` message in the text fed back to the LLM.
3.  **`FailureMemory` Loop Guard**: A signature-based interceptor (`tool_name::params`). The first failure is recorded and the LLM is allowed to retry. A second identical call is **blocked** and the loop breaks, preventing token waste.

### File Import Modes & Conflict Resolution

The `import_file` method supports 6 ingestion modes: `text`, `text_images`, `images_only`, `ocr`, `data` (Dual-Stream), and `data_bundle` (Schema Fusion).

When importing, 4 conflict resolution strategies are available via `on_conflict`:
*   `suffix` (default): Renames new file (e.g., `README_1.md`).
*   `version`: Updates existing artifact and bumps version.
*   `overwrite`: Replaces content without version bump.
*   `replace`: Purges all history and creates fresh `v1`.

### Decoupled Artefact Protocols (`.laa` & `.lab`)

Artefacts are fully decoupled from discussions via two standalone archive formats:
*   **`.laa` (Standalone Artefact Archive)**: Exports a single artefact with its entire version history.
*   **`.lab` (Linked Artefact Bundle)**: Exports multiple artefacts preserving relative folder structure.

```python
# Export/Import single artefact
discussion.artefacts.export_artefact_to_archive("main.py", "main.laa")
discussion.artefacts.import_artefact_from_archive("main.laa")

# Export/Import bundle
discussion.artefacts.export_artefact_bundle(["main.py", "index.html"], "app.lab")
discussion.artefacts.import_artefact_bundle("app.lab")
```

A **Global Artefact Library** (`data_workspace/standalone_artefacts/`) exists for cross-discussion sharing.

### External Workspace Access

If your host application needs to execute an artifact directly without the LLM loop, use the built-in path getters and sync method:

```python
import subprocess

script_path = discussion.get_active_file_path("my_script.py")
ws_data_path = discussion.get_workspace_data_path()

if script_path:
    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True,
        cwd=ws_data_path  # 🛡️ MANDATORY: Set CWD so relative paths resolve!
    )
    # Sync new files created by the script back to the artefact system
    sync_report = discussion.sync_workspace_to_artefacts()
```

### Interactive Widgets (`<lollms_inline>`)

Widgets are **ephemeral, in-context, interactive educational demonstrations** rendered inside the chat bubble. They are strictly constrained to **600x400px** and are for teaching/visualizing concepts only (not for building apps). The backend passes raw HTML/CSS/JS to the UI via streaming events (`MSG_TYPE_WIDGET_CHUNK`, `MSG_TYPE_WIDGET_DONE`). The host application must render them inside a sandboxed `<iframe>` using the Blob URL protocol.

### Interactive Forms (`<lollms_form>`)

Forms allow the LLM to request structured data from the user mid-generation. When the LLM emits a `<lollms_form>` tag, the system parses it, fires `MSG_TYPE_FORM_READY`, and **pauses the generation loop**. The host application renders the form, collects answers, and calls `discussion.submit_form_response(form_id, answers)` to resume. Supported field types: `text`, `textarea`, `number`, `range`, `select`, `radio`, `checkbox`, `rating`, `section`.

---

## 👜 Personality Bundles & RAG

Personalities are packaged using the **Bundle Format**. A bundle is a directory containing a `SOUL.md` file and optional resource folders (`tools/`, `skills/`, `knowledge/`).

```python
from lollms_client.lollms_personality import PersonalityBundle

# Import a personality from a folder
personality = PersonalityBundle.import_bundle(
    bundle_path="./personalities/my_cinema_agent",
    lollms_client=client
)

# Use it in a discussion
discussion.chat(user_message="Pitch a sci-fi movie.", personality=personality)
```

---

## Advanced Discussion Management

The `LollmsDiscussion` class is a core component for managing conversational state, including message history, long-term memory, and various context zones.

### Basic Chat with `LollmsDiscussion`

For general conversational agents that need to maintain context across turns, `LollmsDiscussion` simplifies the process. It automatically handles message formatting, history management, and context window limitations.

```python
from lollms_client import LollmsClient, LollmsDiscussion, MSG_TYPE, LollmsDataManager
from ascii_colors import ASCIIColors
import os
import tempfile

# Initialize LollmsClient
try:
    lc = LollmsClient(
        llm_binding_name="ollama", 
        llm_binding_config={
            "model_name": "llama3",
            "host_address": "http://localhost:11434"
        }
    )
except Exception as e:
    ASCIIColors.error(f"Failed to initialize LollmsClient for discussion: {e}")
    exit()

# Create a new discussion. For persistent discussions, pass a db_manager.
# Using a temporary directory for the database for this example's simplicity
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "discussion_db.sqlite"
    db_manager = LollmsDataManager(f"sqlite:///{db_path}")

    discussion_id = "basic_chat_example"
    discussion = db_manager.get_discussion(lc, discussion_id)
    if not discussion:
        ASCIIColors.yellow(f"\nCreating new discussion '{discussion_id}'...")
        discussion = LollmsDiscussion.create_new(
            lollms_client=lc,
            db_manager=db_manager,
            id=discussion_id,
            autosave=True # Important for persistence
        )
        discussion.system_prompt = "You are a friendly and helpful AI."
        discussion.commit()
    else:
        ASCIIColors.green(f"\nLoaded existing discussion '{discussion_id}'.")


    # Define a simple callback for streaming
    def chat_callback(chunk: str, msg_type: MSG_TYPE, **kwargs) -> bool:
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(chunk, end="", flush=True)
        return True

    try:
        ASCIIColors.cyan("> User: Hello, how are you today?")
        response = discussion.chat(
            user_message="Hello, how are you today?",
            streaming_callback=chat_callback
        )
        print("\n") # Newline after stream finishes

        ai_message = response['ai_message']
        user_message = response['user_message']

        ASCIIColors.green(f"< Assistant (Full): {ai_message.content[:100]}...")

        # Now, continue the conversation
        ASCIIColors.cyan("\n> User: Can you recommend a good book?")
        response = discussion.chat(
            user_message="Can you recommend a good book?",
            streaming_callback=chat_callback
        )
        print("\n")

        # You can inspect the full message history
        ASCIIColors.magenta("\n--- Discussion History (last 3 messages) ---")
        for msg in discussion.get_messages()[-3:]:
            print(f"[{msg.sender.capitalize()}]: {msg.content[:50]}...")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during discussion chat: {e}")
```

### Building Stateful Agents with Memory and Data Zones

The `LollmsDiscussion` class provides a sophisticated system for creating stateful agents that can remember information across conversations. This is achieved through a layered system of "context zones" that are automatically combined into the AI's system prompt.

#### Understanding the Context Zones

The AI's context is more than just chat history. It's built from several distinct components, each with a specific purpose:

*   **`system_prompt`**: The foundational layer defining the AI's core identity, persona, and primary instructions.
*   **`memory`**: The AI's long-term, persistent memory. It stores key facts about the user or topics, built up over time using the `memorize()` method.
*   **`user_data_zone`**: Holds session-specific information about the user's current state or goals (e.g., "User is currently working on 'file.py'").
*   **`discussion_data_zone`**: Contains state or meta-information about the current conversational task (e.g., "Step 1 of the plan is complete").
*   **`personality_data_zone`**: A knowledge base or set of rules automatically injected from a `LollmsPersonality`'s `data_source`.
*   **`pruning_summary`**: An automatic, AI-generated summary of the oldest messages in a very long chat, used to conserve tokens without losing the gist of the early conversation.

The `get_context_status()` method is your window into this system, showing you exactly how these zones are combined and how many tokens they consume.

Let's see this in action with a "Personal Assistant" agent that learns about the user over time.

```python
from lollms_client import LollmsClient, LollmsDataManager, LollmsDiscussion, MSG_TYPE
from ascii_colors import ASCIIColors
import json
import tempfile
import os

# --- 1. Setup a persistent database for our discussion ---
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "my_assistant.db"
    db_manager = LollmsDataManager(f"sqlite:///{db_path}")

    try:
        lc = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llama3"})
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient for stateful agent: {e}")
        exit()

    # Try to load an existing discussion or create a new one
    discussion_id = "user_assistant_chat_1"
    discussion = db_manager.get_discussion(lc, discussion_id)
    if not discussion:
        ASCIIColors.yellow("Creating a new discussion for stateful agent...")
        discussion = LollmsDiscussion.create_new(
            lollms_client=lc,
            db_manager=db_manager,
            id=discussion_id,
            autosave=True # Important for persistence
        )
        # Let's preset some data in different zones
        discussion.system_prompt = "You are a helpful Personal Assistant."
        discussion.user_data_zone = "User's Name: Alex\nUser's Goal: Learn about AI development."
        discussion.commit()
    else:
        ASCIIColors.green("Loaded existing discussion for stateful agent.")


    def run_chat_turn(prompt: str):
        """Helper function to run a single chat turn and print details."""
        ASCIIColors.cyan(f"\n> User: {prompt}")

        # --- A. Check context status BEFORE the turn using get_context_status() ---
        ASCIIColors.magenta("\n--- Context Status (Before Generation) ---")
        status = discussion.get_context_status()
        print(f"Max Tokens: {status.get('max_tokens')}, Current Tokens: {status.get('current_tokens')}")
        
        # Print the system context details
        if 'system_context' in status['zones']:
            sys_ctx = status['zones']['system_context']
            print(f"  - System Context Tokens: {sys_ctx['tokens']}")
            # The 'breakdown' shows the individual zones that were combined
            for name, content in sys_ctx.get('breakdown', {}).items():
                # For brevity, show only first line of content
                print(f"    -> Contains '{name}': {content.split(os.linesep)}...")

        # Print the message history details
        if 'message_history' in status['zones']:
            msg_hist = status['zones']['message_history']
            print(f"  - Message History Tokens: {msg_hist['tokens']} ({msg_hist['message_count']} messages)")

        print("------------------------------------------")

        # --- B. Run the chat ---
        ASCIIColors.green("\n< Assistant:")
        response = discussion.chat(
            user_message=prompt,
            streaming_callback=lambda chunk, type, **k: print(chunk, end="", flush=True) if type==MSG_TYPE.MSG_TYPE_CHUNK else None
        )
        print() # Newline after stream

        # --- C. Trigger memorization to update the 'memory' zone ---
        ASCIIColors.yellow("\nTriggering memorization process...")
        discussion.memorize()
        discussion.commit() # Save the new memory to the DB
        ASCIIColors.yellow("Memorization complete.")

    # --- Run a few turns ---
    run_chat_turn("Hi there! Can you recommend a good Python library for building web APIs?")
    run_chat_turn("That sounds great. By the way, my favorite programming language is Rust, I find its safety features amazing.")
    run_chat_turn("What was my favorite programming language again?")

    # --- Final Inspection of Memory ---
    ASCIIColors.magenta("\n--- Final Context Status ---")
    status = discussion.get_context_status()
    print(f"Max Tokens: {status.get('max_tokens')}, Current Tokens: {status.get('current_tokens')}")
    if 'system_context' in status['zones']:
        sys_ctx = status['zones']['system_context']
        print(f"  - System Context Tokens: {sys_ctx['tokens']}")
        for name, content in sys_ctx.get('breakdown', {}).items():
            # Print the full content of the memory zone to verify it was updated
            if name == 'memory':
                ASCIIColors.yellow(f"    -> Full '{name}' content:\n{content}")
            else:
                print(f"    -> Contains '{name}': {content.split(os.linesep)}...")
    print("------------------------------------------")

```

#### How it Works:

1.  **Persistence & Initialization:** The `LollmsDataManager` saves and loads the discussion. We initialize the `system_prompt` and `user_data_zone` to provide initial context.
2.  **`get_context_status()`:** Before each generation, we call this method. The output shows a `system_context` block with a token count for all combined zones and a `breakdown` field that lets us see the content of each individual zone that contributed to it.
3.  **`memorize()`:** After the user mentions their favorite language, `memorize()` is called. The LLM analyzes the last turn, identifies this new, important fact, and appends it to the `discussion.memory` zone.
4.  **Recall:** In the final turn, when asked to recall the favorite language, the AI has access to the updated `memory` content within its system context and can correctly answer "Rust". This demonstrates true long-term, stateful memory.

### Dynamic Context Size Resolution (4-Layer Protocol)

When working with models, accurately determining the context window size is critical for token budgeting and preventing overflow. `lollms_client` employs a sophisticated 4-layer resolution cascade when `get_ctx_size()` is called on an LLM binding:

1.  **Forced Context Size (`forced_ctx_size`)**: If explicitly set (either via `kwargs` during initialization or dynamically via `set_forced_ctx_size()`), this value is returned immediately. It acts as the absolute source of truth, overriding all automatic detection.
2.  **Binding-Specific Detection (`_get_ctx_size`)**: If the binding instance implements a `_get_ctx_size()` method (e.g., querying an API for the model's specific limit), it is called. If it returns a valid integer, that is used.
3.  **Hardcoded List (`assets/models_ctx_sizes.json`)**: The library maintains a local JSON file mapping known model names/aliases to their context sizes. If the `model_name` matches an entry, the hardcoded value is used.
4.  **Default Fallback (`default_ctx_size`)**: If the model is completely unknown and no binding-specific method exists, it falls back to the `default_ctx_size` provided during initialization.

This protocol ensures that you can always force a specific context window for testing or constrained environments, while still benefiting from automatic detection for known models.

```python
from lollms_client import LollmsClient

# Initialize with a forced context size (Layer 1)
lc = LollmsClient(
    llm_binding_name="ollama",
    llm_binding_config={
        "model_name": "llama3",
        "forced_ctx_size": 4096 # Forces get_ctx_size() to always return 4096
    }
)

print(f"Forced Context Size: {lc.get_ctx_size()}") # Output: 4096

# You can also dynamically update it at runtime
lc.llm.set_forced_ctx_size(8192)
print(f"Updated Context Size: {lc.get_ctx_size()}") # Output: 8192

# Clear the forced override to fall back to automatic detection (Layers 2-4)
lc.llm.set_forced_ctx_size(None)
print(f"Auto-detected Context Size: {lc.get_ctx_size()}") # Output: 8192 (from hardcoded list for llama3)
```

### Human-Inspired Multi-Level Memory System

`LollmsDiscussion` incorporates a biological-inspired persistent memory system (`LollmsMemoryManager`) consisting of five hierarchical layers (Levels 0-4):
- **Level 0 — Volatile Scratchpad**: Appended before the last user prompt for single-turn intermediate reasoning. Cleared after the turn.
- **Level 1 — Working Memory**: Active, high-importance facts currently in focus. Injected directly into the conversation context. Capped by a token budget; excess memories are automatically demoted to Deep Memory.
- **Level 2 — Deep Memory**: Long-term memories that have faded due to lack of use. Not injected in full. Instead, compact *handles* (stubs) are displayed in the context so the LLM knows they exist and can call `<mem_load id="XXXXXXXX" />` to load them back to active Working Memory.
- **Level 3 — Archived Memory**: Extremely old or low-importance memories. Never loaded automatically. Subject to automatic pruning or re-activation during the periodic dream consolidation pass.
- **Level 4 — Episodic Memory**: Chronological, highly preserved event and conversation interaction logs of past turns. These provide permanent historical context of past conversations and are stable against decay.

#### Memory XML Commands

The memory manager intercepts several XML tags emitted by the model during a chat turn to manipulate its database state:
- `<mem_new importance="0.9">Your fact here</mem_new>`: Store a new long-term memory.
- `<mem_tag id="UUID" />`: Retrieve or reference a memory (boosts importance).
- `<mem_update id="UUID">New updated content</mem_update>`: Actively update a memory's content.
- `<mem_delete id="UUID" />`: Permanently delete a memory.
- `<mem_load id="UUID" />`: Load a deep memory handle back to active Working Memory.

#### The Dream Consolidation Cycle (`dream()`)

To manage and organize its memories over time, the AI periodically executes a "dream cycle". You can trigger this at the end of chat turns or in a background worker:
- **Reinforcement**: Memories that are frequently tagged/retrieved have their importance boosted logarithmically.
- **Decay**: Unused memories decay over time and are automatically demoted (Working → Deep → Archived).
- **LLM-Assisted Selective Forgetting**: Archive memories falling below the `forget_threshold` (e.g., 0.02) are evaluated by the subconscious "Dreamer" LLM. If the dreamer decides they contain critical architectural/preference rules, they are restored to active status; otherwise, they are permanently pruned.

#### Logical vs. Physical Scoping

The system supports both shared and isolated scopes across multiple conversations using a single parameters layout:
- **Logical User Scoping (Cross-Discussion Propagation)**: Point your `LollmsMemoryManager` to a shared database path and assign `owner_id` to the unique User ID (e.g., `owner_id="user_ParisNeo"`). Memories learned in Chat A will propagate automatically to Chat B.
- **Discussion Isolation**: Set `owner_id` to the specific Discussion Thread ID (e.g. `owner_id=discussion.id`) to confine learned facts to that specific context.
- **Physical Partitioning**: Provide user-specific file paths (e.g. `sqlite:///app_data/users/{user_id}/long_term_memory.db`) for full physical filesystem data isolation.

### Managing Multimodal Context: Activating and Deactivating Images

When working with multimodal models, you can now control which images in a message are active and sent to the model. This is useful for focusing the AI's attention, saving tokens on expensive vision models, or allowing a user to correct which images are relevant.

This is managed at the `LollmsMessage` level using the `toggle_image_activation()` method.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager, MSG_TYPE
from ascii_colors import ASCIIColors
import base64
from pathlib import Path
import os
import tempfile

# Helper to create a dummy image b64 string
def create_dummy_image(text, output_dir):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        ASCIIColors.warning("Pillow not installed. Skipping image example.")
        return None
    
    # Try to find a common font, otherwise use default
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") # Common Linux path
    if not font_path.exists():
        font_path = Path("/Library/Fonts/Arial.ttf") # Common macOS path
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/arial.ttf") # Common Windows path
    
    try:
        font = ImageFont.truetype(str(font_path), 15)
    except (IOError, OSError):
        font = ImageFont.load_default() # Fallback to default if font not found

    img = Image.new('RGB', (200, 50), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), text, fill=(255,255,0), font=font)
    
    temp_file = Path(output_dir) / f"temp_img_{text.replace(' ', '_')}.png"
    img.save(temp_file, "PNG")
    b64 = base64.b64encode(temp_file.read_bytes()).decode('utf-8')
    temp_file.unlink() # Clean up temporary file
    return b64

# --- 1. Setup ---
try:
    # Llava is a good multi-modal model for Ollama
    # Ensure Ollama is running and 'llava' model is pulled (e.g., ollama pull llava)
    lc = LollmsClient(llm_binding_name="ollama", llm_binding_config={"model_name": "llava"})
except Exception as e:
    ASCIIColors.warning(f"Failed to initialize LollmsClient for image example: {e}")
    ASCIIColors.warning("Skipping image activation example. Ensure Ollama is running and 'llava' model is pulled.")
    exit()

with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "image_discussion_db.sqlite"
    db_manager = LollmsDataManager(f"sqlite:///{db_path}")
    discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db_manager)

    # --- 2. Add a message with multiple images ---
    # Ensure Pillow is installed: pip install Pillow
    img1_b64 = create_dummy_image("Image 1: Apple", tmpdir)
    img2_b64 = create_dummy_image("Image 2: Cat", tmpdir)
    img3_b64 = create_dummy_image("Image 3: Dog", tmpdir)

    if not img1_b64 or not img2_b64 or not img3_b64:
        ASCIIColors.warning("Skipping image activation example due to image creation failure (likely missing Pillow or font).")
        exit()

    discussion.add_message(
        sender="user", 
        content="What is in the second image?", 
        images=[img1_b64, img2_b64, img3_b64]
    )
    user_message = discussion.get_messages()[-1]

    # --- 3. Check the initial state ---
    ASCIIColors.magenta("--- Initial State (All 3 Images Active) ---")
    status_before = discussion.get_context_status()
    # The 'content' field for message history will indicate the number of images if present
    print(f"Message History Text (showing active images):\n{status_before['zones']['message_history']['content']}")

    # --- 4. Deactivate irrelevant images ---
    ASCIIColors.magenta("\n--- Deactivating images 1 and 3 ---")
    user_message.toggle_image_activation(index=0, active=False) # Deactivate first image (Apple)
    user_message.toggle_image_activation(index=2, active=False) # Deactivate third image (Dog)
    discussion.commit() # Save changes to the message

    # --- 5. Check the new state ---
    ASCIIColors.magenta("\n--- New State (Only Image 2 is Active) ---")
    status_after = discussion.get_context_status()
    print(f"Message History Text (showing active images):\n{status_after['zones']['message_history']['content']}")

    ASCIIColors.green("\nNotice the message now says '(1 image(s) attached)' instead of 3, and only the active image will be sent to the multimodal LLM.")
    ASCIIColors.green("To confirm, let's ask the model what it sees:")

    # This will send only the activated image
    response = discussion.chat(
        user_message="What do you see in the image(s) attached to my last message?",
        # Use a streaming callback to see the response
        streaming_callback=lambda chunk, type, **k: print(chunk, end="", flush=True) if type==MSG_TYPE.MSG_TYPE_CHUNK else None
    )
    print("\n")
    ASCIIColors.green(f"Assistant's response after toggling images: {response['ai_message'].content}")

```
**Note:** The image generation helper in the example requires `Pillow` (`pip install Pillow`). It also attempts to find common system fonts; if issues persist, you might need to install `matplotlib` for better font handling or provide a specific font path.

### Putting It All Together: An Advanced Agentic Example

Let's create a **Python Coder Agent**. This agent will use a set of coding rules from a local file as its knowledge base and will be equipped with a tool to execute the code it writes. This demonstrates the synergy between `LollmsPersonality` (with `data_source` and `active_mcps`), `LollmsDiscussion`, and the MCP system.

#### Step 1: Create the Knowledge Base (`coding_rules.txt`)

Create a simple text file with the rules our agent must follow.

```text
# File: coding_rules.txt

1.  All Python functions must include a Google-style docstring.
2.  Use type hints for all function parameters and return values.
3.  The main execution block should be protected by `if __name__ == "__main__":`.
4.  After defining a function, add a simple example of its usage inside the main block.
5.  Print the output of the example usage to the console.
```

#### Step 2: The Main Script (`agent_example.py`)

This script will define the personality, initialize the client, and run the agent.

```python
from pathlib import Path
from lollms_client import LollmsClient, LollmsPersonality, LollmsDiscussion, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
import json
import tempfile
import os

# A detailed callback to visualize the agent's process
def agent_callback(chunk: str, msg_type: MSG_TYPE, params: dict = None, **kwargs) -> bool:
    if not params: params = {}
    
    if msg_type == MSG_TYPE.MSG_TYPE_STEP:
        ASCIIColors.yellow(f"\n>> Agent Step: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        ASCIIColors.yellow(f"\n>> Agent Step Start: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        result = params.get('result', '')
        # Only print a snippet of result to avoid overwhelming console for large outputs
        if isinstance(result, dict):
            result_str = json.dumps(result)[:150] + ("..." if len(json.dumps(result)) > 150 else "")
        else:
            result_str = str(result)[:150] + ("..." if len(str(result)) > 150 else "")
        ASCIIColors.green(f"<< Agent Step End: {chunk} -> Result: {result_str}")
    elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CONTENT:
        ASCIIColors.magenta(f"🤔 Agent Thought: {chunk}")
    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
        tool_name = params.get('name', 'unknown_tool')
        tool_params = params.get('parameters', {})
        ASCIIColors.blue(f"🛠️  Agent Action: Called '{tool_name}' with {tool_params}")
    elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
        ASCIIColors.cyan(f"👀 Agent Observation (Tool Output): {params.get('result', 'No result')}")
    elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True) # Final answer stream
    return True

# Create a temporary directory for the discussion DB and coding rules file
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = Path(tmpdir) / "agent_discussion.db"
    
    # Create the coding rules file
    rules_path = Path(tmpdir) / "coding_rules.txt"
    rules_content = """
1.  All Python functions must include a Google-style docstring.
2.  Use type hints for all function parameters and return values.
3.  The main execution block should be protected by `if __name__ == "__main__":`.
4.  After defining a function, add a simple example of its usage inside the main block.
5.  Print the output of the example usage to the console.
"""
    rules_path.write_text(rules_content.strip())
    ASCIIColors.yellow(f"Created temporary coding rules file at: {rules_path}")

    try:
        # --- 1. Load the knowledge base from the file ---
        coding_rules = rules_path.read_text()

        # --- 2. Define the Coder Agent Personality ---
        coder_personality = LollmsPersonality(
            name="Python Coder Agent",
            author="lollms-client",
            category="Coding",
            description="An agent that writes and executes Python code according to specific rules.",
            system_prompt=(
                "You are an expert Python programmer. Your task is to write clean, executable Python code based on the user's request. "
                "You MUST strictly follow all rules provided in the 'Personality Static Data' section. "
                "First, think about the plan. Then, use the `python_code_interpreter` tool to write and execute the code. "
                "Finally, present the code and its output to the user."
            ),
            # A) Attach the static knowledge base
            data_source=coding_rules,
            # B) Equip the agent with a code execution tool
            active_mcps=["python_code_interpreter"]
        )

        # --- 3. Initialize the Client and Discussion ---
        # A code-specialized model is recommended (e.g., codellama, deepseek-coder)
        # Ensure Ollama is running and 'codellama' model is pulled (e.g., ollama pull codellama)
        lc = LollmsClient(
            llm_binding_name="ollama",          
            llm_binding_config={
                "model_name": "codellama",
                "host_address": "http://localhost:11434"
            },
            tools_binding_name="local_mcp"    # Enable the local tool execution engine
        )
        # For agentic workflows, it's often good to have a persistent discussion
        db_manager = LollmsDataManager(f"sqlite:///{db_path}")
        discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db_manager)
        
        # --- 4. The User's Request ---
        user_prompt = "Write a Python function that takes two numbers and returns their sum."

        ASCIIColors.yellow(f"User Prompt: {user_prompt}")
        print("\n" + "="*50 + "\nAgent is now running...\n" + "="*50)

        # --- 5. Run the Agentic Chat Turn ---
        response = discussion.chat(
            user_message=user_prompt,
            personality=coder_personality,
            streaming_callback=agent_callback,
            max_llm_iterations=5, # Limit iterations for faster demo
            tool_call_decision_temperature=0.0 # Make decision more deterministic
        )

        print("\n\n" + "="*50 + "\nAgent finished.\n" + "="*50)
        
        # --- 6. Inspect the results ---
        ai_message = response['ai_message']
        ASCIIColors.green("\n--- Final Answer from Agent ---")
        print(ai_message.content)
        
        ASCIIColors.magenta("\n--- Tool Calls Made (from metadata) ---")
        if "tool_calls" in ai_message.metadata:
            print(json.dumps(ai_message.metadata["tool_calls"], indent=2))
        else:
            print("No tool calls recorded in message metadata.")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during agent execution: {e}")
        ASCIIColors.warning("Please ensure Ollama is running, 'codellama' model is pulled, and 'local_mcp' binding is available.")
        trace_exception(e) # Provide detailed traceback
```

#### Step 3: What Happens Under the Hood

When you run `agent_example.py`, a sophisticated process unfolds:

1.  **Initialization:** The `LollmsDiscussion.chat()` method is called with the `coder_personality`.
2.  **Knowledge Injection:** The `chat` method sees that `personality.data_source` is a string. It automatically takes the content of `coding_rules.txt` and injects it into the discussion's data zones.
3.  **Tool Activation:** The method also sees `personality.active_mcps`. It enables the `python_code_interpreter` tool for this turn.
4.  **Context Assembly:** The `LollmsClient` assembles a rich prompt for the LLM that includes:
    *   The personality's `system_prompt`.
    *   The content of `coding_rules.txt` (from the data zones).
    *   The list of available tools (including `python_code_interpreter`).
    *   The user's request ("Write a function...").
5.  **Reason and Act:** The LLM, now fully briefed, reasons that it needs to use the `python_code_interpreter` tool. It formulate the Python code *according to the rules it was given*.
6.  **Tool Execution:** The `local_mcp` binding receives the code and executes it in a secure local environment. It captures any output (`stdout`, `stderr`) and results.
7.  **Observation:** The execution results are sent back to the LLM as an "observation."
8.  **Final Synthesis:** The LLM now has the user's request, the rules, the code it wrote, and the code's output. It synthesizes all of this into a final, comprehensive answer for the user.

This example showcases how `lollms-client` allows you to build powerful, knowledgeable, and capable agents by simply composing personalities with data and tools.

## Agentic Workflows with Personality and Tools

The `Agent` class combines `LollmsClient`, `LollmsPersonality`, and tool execution into a single, powerful unit for building autonomous agents. This enables multi-step reasoning where the agent can chain tool calls, reflect on results, and synthesize comprehensive answers.

### Building a Research Agent with Multi-Step Reasoning

This example demonstrates a **Research Agent** that:
1. Uses a custom personality with research-focused system prompts
2. Loads multiple tools (arXiv search + Wikipedia search)
3. Performs multi-step reasoning: searches arXiv → searches Wikipedia → synthesizes findings
4. Uses `Agent.generate_with_tools()` for the full agentic loop

```python
#!/usr/bin/env python3
"""
research_agent_example.py
=========================
A full agentic workflow demonstrating:
- Custom personality definition
- Multiple file-based tools
- Multi-step reasoning with tool chaining
- Agent.generate_with_tools() with rich metadata
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_agent import Agent, ToolsManager, AgentRole
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE


# ── Tool Definitions ─────────────────────────────────────────────────────────

ARXIV_TOOL = '''TOOL_LIBRARY_NAME = 'ArXiv Explorer'
TOOL_LIBRARY_DESC = 'Search scientific papers on ArXiv.'
TOOL_LIBRARY_ICON = '🔬'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'arxiv': '>=2.1.0'})

def tool_search_papers(args: dict):
    """
    Search for scientific papers on ArXiv.

    Args:
        args: dict with keys:
            - query (str): Scientific keywords
            - count (int, optional): Number of papers (default: 3)
            - year_start (int, optional): Start year filter
            - year_end (int, optional): End year filter
    """
    import arxiv
    try:
        query = args.get('query', '')
        count = args.get('count', 3)
        search = arxiv.Search(query=query, max_results=100)
        client = arxiv.Client()
        results = []
        for res in client.results(search):
            authors = ', '.join(a.name for a in res.authors)
            date = res.published.strftime('%Y-%m-%d') if res.published else "Unknown"
            results.append(
                f"[{res.entry_id}] {res.title}\\n"
                f"Authors: {authors} | Published: {date}\\n"
                f"Abstract: {res.summary[:400]}..."
            )
            if len(results) >= count:
                break
        return "\\n\\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error: {str(e)}"
'''

WIKI_TOOL = '''TOOL_LIBRARY_NAME = 'Wikipedia Search'
TOOL_LIBRARY_DESC = 'Search and retrieve article summaries from Wikipedia.'
TOOL_LIBRARY_ICON = '📖'

def init_tools_library() -> None:
    import pipmaster as pm
    pm.ensure_packages({'wikipedia': '>=1.4.0'})

def tool_search_wikipedia(args: dict):
    """
    Search Wikipedia for articles.

    Args:
        args: dict with keys:
            - query (str): Search term
            - max_results (int, optional): Max results (default: 3)
    """
    import wikipedia
    try:
        query = args.get('query', '')
        limit = args.get('max_results', 3)
        search_results = wikipedia.search(query)
        output = []
        for title in search_results[:limit]:
            try:
                page = wikipedia.summary(title, sentences=5)
                output.append(f"--- {title} ---\\n{page}")
            except: 
                continue
        return "\\n\\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error: {str(e)}"
'''


def setup_tools():
    """Create tool files in the lollms hub directory."""
    tools_dir = Path.home() / ".lollms_hub" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    arxiv_path = tools_dir / "arxiv_search.py"
    wiki_path = tools_dir / "wikipedia_search.py"
    
    if not arxiv_path.exists():
        arxiv_path.write_text(ARXIV_TOOL, encoding="utf-8")
    if not wiki_path.exists():
        wiki_path.write_text(WIKI_TOOL, encoding="utf-8")
    
    return str(arxiv_path), str(wiki_path)


def main():
    print("=" * 70)
    print("🔬 Research Agent — Multi-Step Reasoning Demo")
    print("=" * 70)

    # ── 1. Setup tools ──────────────────────────────────────────────────
    arxiv_path, wiki_path = setup_tools()
    print(f"📁 Tools ready: arxiv_search.py, wikipedia_search.py")

    # ── 2. Create LollmsClient ──────────────────────────────────────────
    client = LollmsClient(
        llm_binding_name="llama_cpp_server",
        llm_binding_config={
            "models_path": "data/models/llama_cpp_models",
            "binaries_path": "data/bin/llm/llama_cpp_server",
            "ctx_size": 8192,
            "n_gpu_layers": -1,
            "n_threads": 4,
            "idle_timeout": 300,
        },
    )

    # Download/load model (Ministral 3B for this demo)
    zoo = client.llm.get_zoo()
    model_idx = 1  # Ministral-3-3B-Instruct-2512
    chosen = zoo[model_idx]
    model_file = chosen["filename"]

    model_path = Path("data/models/llama_cpp_models") / model_file
    if not model_path.exists():
        print(f"\n⬇️  Downloading {chosen['name']}...")
        client.llm.download_from_zoo(model_idx)
    print(f"\n🔌 Loading {model_file}...")
    client.llm.load_model(model_file)

    # ── 3. Create Personality ───────────────────────────────────────────
    personality = LollmsPersonality(
        name="ResearchAgent",
        system_prompt=(
            "You are an expert research assistant with deep knowledge of "
            "computer science and artificial intelligence. Your workflow:\n"
            "1. Search arXiv for the latest academic papers on the topic\n"
            "2. Search Wikipedia for foundational concepts and background\n"
            "3. Synthesize findings into a comprehensive, well-structured report\n"
            "4. Cite sources clearly and highlight key insights\n\n"
            "Always use tools when available — never rely solely on training data."
        ),
    )

    # ── 4. Create Agent ────────────────────────────────────────────────
    agent = Agent(
        lc=client,
        personality=personality,
        name="ResearchAgent",
        role=AgentRole.DOMAIN_EXPERT,
        model_params={"temperature": 0.7, "n_predict": 2048},
        max_tokens_per_turn=4096,
    )
    print(f"\n🤖 Agent created: {agent}")

    # ── 5. Multi-step research query ──────────────────────────────────
    query = (
        "I want to understand the current state of reasoning in large language models. "
        "Find recent papers from 2024-2025, then look up background on chain-of-thought "
        "reasoning, and finally synthesize a comprehensive overview with citations."
    )

    print("\n" + "-" * 70)
    print("📝 RESEARCH QUERY:")
    print("-" * 70)
    print(query)
    print("-" * 70)

    # ── 6. Execute agentic generation ─────────────────────────────────
    print("\n🔍 Starting multi-step research (this may take several rounds)...\n")

    result = agent.generate_with_tools(
        prompt=query,
        tools=[arxiv_path, wiki_path],  # Both tools available
        system_prompt=personality.system_prompt,
        temperature=0.7,
        n_predict=4096,
        max_tool_rounds=10,  # Allow multiple tool chains
        auto_execute=True,
    )

    # ── 7. Display results ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 EXECUTION METADATA")
    print("=" * 70)
    print(f"Total rounds:     {result['rounds']}")
    print(f"Tool calls made:  {len(result['tool_calls'])}")
    
    for i, tc in enumerate(result['tool_calls'], 1):
        print(f"\n  Round {tc['round']} — {tc['name']}")
        print(f"    Parameters: {json.dumps(tc['parameters'], indent=2)}")
        # Show result summary
        tr = next((r for r in result['tool_results'] if r['round'] == tc['round']), None)
        if tr:
            res = tr['result']
            status = "✅" if res.get('success') else "❌"
            output = str(res.get('output', res))[:200]
            print(f"    Result: {status} {output}...")

    print("\n" + "=" * 70)
    print("📝 FINAL SYNTHESIZED REPORT")
    print("=" * 70)
    print(result['response'])

    # ── 8. Cleanup ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("🧹 Cleanup")
    print("=" * 70)
    client.llm.unload_model()
    print("✅ Done!")


if __name__ == "__main__":
    main()
```

**How Multi-Step Reasoning Works:**

1. **Round 1**: The agent receives the query and decides to search arXiv for recent papers on LLM reasoning. It emits a `<tool>` for `tool_search_papers`.

2. **Tool Execution**: The arXiv tool executes and returns 3 recent papers with abstracts.

3. **Round 2**: The agent sees the arXiv results and decides it needs background on chain-of-thought reasoning. It emits a `<tool>` for `tool_search_wikipedia`.

4. **Tool Execution**: The Wikipedia tool returns foundational concepts and explanations.

5. **Round 3**: With both academic and encyclopedic sources in context, the agent synthesizes a comprehensive report with proper citations and key insights.

6. **Final Answer**: The agent produces a structured response combining all gathered information.

**Key Agent Configuration Options:**

| Parameter | Description |
|-----------|-------------|
| `tools` | List of file paths (`.py`) or inline tool dicts |
| `max_tool_rounds` | Maximum agentic loops (default: 10) |
| `auto_execute` | If `False`, returns pending tool for manual execution |
| `system_prompt` | Override personality's system prompt for this call |
| `temperature` | Sampling temperature for generation |
| `n_predict` | Max tokens per generation step |

**Using `generate_with_tools_sync()`:**

For simple fire-and-forget usage, the sync wrapper returns only the final text:

```python
answer = agent.generate_with_tools_sync(
    prompt="What are the latest papers on quantum computing?",
    tools=[arxiv_path],
)
print(answer)  # Just the final response string
```

### Universal Lazy Profiles & Smart Routing

`lollms_client` features a powerful, memory-efficient multi-binding architecture via **Universal Lazy Profiles**. Instead of eagerly instantiating all models/engines at startup (which wastes RAM and VRAM), you define declarative registries of `LollmsBindingProfile` configurations for *any* modality (LLM, TTI, TTS, STT, TTV, TTM). 

Only the binding marked as `is_default` is loaded at startup. Other bindings are instantiated lazily *on-demand* when you switch to them. This system is 100% backward compatible. If you use the legacy `llm_binding_name` or `tti_binding_name` parameters, they are automatically registered as the `"master"` profiles and marked as defaults.

#### 1. Multi-Model LLM Routing
Define multiple LLM profiles to handle different domains or complexity tiers.

```python
from lollms_client import LollmsClient, LollmsBindingProfile

# 1. Define LLM profiles declaratively (None of these are instantiated yet)
llm_profiles = {
    "cloud_vision": LollmsBindingProfile(
        name="cloud_vision",
        binding_name="openai",
        binding_config={"model_name": "gpt-4o", "service_key": "your-key"},
        vision_enabled=True,
        is_default=True  # This one will be loaded at startup
    ),
    "local_coder": LollmsBindingProfile(
        name="local_coder",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "qwen2.5-coder:7b"},
        forced_context_size=32768
    ),
    "local_fast": LollmsBindingProfile(
        name="local_fast",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434", "model_name": "llama3.2:3b"}
    )
}

# 2. Initialize the client (Only "cloud_vision" is instantiated)
lc = LollmsClient(llm_profiles=llm_profiles)

# 3. Use the default model
response1 = lc.generate_text("Explain quantum physics.")

# 4. Dynamically switch to the local coder (Instantiated on-the-fly and cached)
lc.switch_model("local_coder")
response2 = lc.generate_text("Write a Python script to sort a list.")

# 5. Switch back to the default cloud vision model (Retrieved from cache, no re-instantiation)
lc.switch_model("cloud_vision")
```

#### 2. Multi-Engine TTI (Text-to-Image) Routing
The profile system is universal. You can define profiles for image generation engines and switch between a local Stable Diffusion model and a cloud DALL-E API seamlessly.

```python
# Define TTI profiles
tti_profiles = {
    "local_sd": LollmsBindingProfile(
        name="local_sd",
        binding_name="diffusers",
        binding_config={"model_name": "stable-diffusion-v1-5"},
        is_default=True
    ),
    "cloud_dalle": LollmsBindingProfile(
        name="cloud_dalle",
        binding_name="openai",
        binding_config={"model_name": "dall-e-3", "service_key": "your-key"}
    )
}

lc = LollmsClient(llm_binding_name="ollama", tti_profiles=tti_profiles)

# Generate with default local SD
img_bytes = lc.generate_image("A cyberpunk cat")

# Switch to DALL-E 3 for a specific prompt
lc.switch_tti("cloud_dalle")
img_bytes = lc.generate_image("A hyperrealistic oil painting of a dog")
```

#### 3. The Smart Router Binding
For automated routing, the `smart_router` meta-binding evaluates incoming prompts using TF-IDF (subject matching), complexity heuristics, and weighted constraints (cost/latency) to delegate generation to the optimal child model automatically.

```python
from lollms_client import LollmsClient

router_config = {
    "routing_strategy": "cost_optimized",
    "model_profiles": {
        "cheap_fast": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5:3b"},
            "routing_profile": {"description": "fast simple tasks", "complexity_tier": 1}
        },
        "smart_coder": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5-coder:7b"},
            "routing_profile": {"description": "python code debugging", "complexity_tier": 2}
        }
    }
}

lc = LollmsClient(llm_binding_name="smart_router", llm_binding_config=router_config)

# The router automatically selects "cheap_fast" for simple prompts
response = lc.generate_text("What is 2+2?")

# The router automatically selects "smart_coder" for coding prompts
response = lc.generate_text("Write a Python script to sort a list.")
```

#### 4. VLM as a Tool (`vlm_query`)
When using a text-only LLM (or Smart Router without a VLM child), you can enable VLM collaboration via the `vlm_query` LCP tool. This allows the text LLM to delegate visual analysis to a secondary VLM on-demand.

```python
# Client with a text-only LLM and a separate VLM profile
lc = LollmsClient(
    llm_profiles={
        "text_llm": LollmsBindingProfile(name="text_llm", binding_name="ollama", binding_config={"model_name": "llama3"}, is_default=True),
        "vlm": LollmsBindingProfile(name="vlm", binding_name="ollama", binding_config={"model_name": "llava"}, vision_enabled=True)
    }
)

# The user sends an image and asks a question
discussion.chat(
    user_message="Look at the diagram in the image. What are the main components?",
    images=["base64_encoded_image_data..."],
    enable_vlm_query=True, # CRITICAL: Explicitly enable the fallback tool
    max_reasoning_steps=10
)
# The LLM (llama3) will realize it needs vision, call `tool_vlm_query(0, "Identify the main components.")`,
# receive the text description from the VLM (llava), and synthesize the final answer.
```

### Smart Routing & VLM Collaboration

`lollms_client` introduces a powerful **Smart Router** binding and a dynamic **VLM Query Tool** to enable cost-effective, multi-model ecosystems.

#### 1. The Smart Router Binding (`smart_router`)

The `smart_router` is a meta-binding that masquerades as a standard LLM binding. When selected, it accepts a dictionary of `model_profiles`. It evaluates incoming prompts using **TF-IDF (subject matching)**, **complexity heuristics**, and **weighted constraints (cost/latency)** to delegate generation to the optimal child model.

```python
from lollms_client import LollmsClient

# 1. Define the routing configuration
router_config = {
    "routing_strategy": "cost_optimized",  # or "balanced", "quality_optimized"
    "model_profiles": {
        "cheap_fast": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5:3b"},
            "routing_profile": {
                "description": "fast simple tasks math formatting",
                "cost_per_1k_tokens": 0.0,
                "avg_latency_ms": 20,
                "complexity_tier": 1
            }
        },
        "smart_coder": {
            "binding_name": "ollama",
            "binding_config": {"model_name": "qwen2.5-coder:7b"},
            "routing_profile": {
                "description": "python javascript rust code debugging algorithms",
                "cost_per_1k_tokens": 0.0,
                "avg_latency_ms": 40,
                "complexity_tier": 2
            }
        }
    }
}

# 2. Initialize the client with the Smart Router as the primary binding
lc = LollmsClient(
    llm_binding_name="smart_router",
    llm_binding_config=router_config
)

# The router automatically selects "cheap_fast" for simple prompts
response = lc.generate_text("What is 2+2?")

# The router automatically selects "smart_coder" for coding prompts
response = lc.generate_text("Write a Python script to sort a list.")
```

#### 2. VLM as a Tool (`vlm_query`)

For VLM+LLM collaboration, the routing binding is stateless by design. Instead of forcing a VLM to process every image, `lollms_client` uses a dynamic LCP tool called `vlm_query`. 

The `LollmsDiscussion` chat loop conditionally mounts this tool as a fallback mechanism. It is only mounted if the active binding **lacks** vision capabilities, another instantiated binding **supports** vision, and the user explicitly passes `enable_vlm_query=True` to `discussion.chat()`. The LLM can then call `tool_vlm_query(image_index, query)` on-demand to ask a VLM specific questions about images, preserving context and saving compute.

```python
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager

# Client with a Smart Router that includes a VLM profile
lc = LollmsClient(
    llm_binding_name="smart_router",
    llm_binding_config={
        "model_profiles": {
            "text_llm": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llama3"},
                "vision_enabled": False
            },
            "vlm": {
                "binding_name": "ollama",
                "binding_config": {"model_name": "llava"},
                "vision_enabled": True
            }
        }
    }
)

db_manager = LollmsDataManager("sqlite:///discussion.db")
discussion = LollmsDiscussion.create_new(lollms_client=lc, db_manager=db_manager)

# The user sends an image and asks a question
discussion.chat(
    user_message="Look at the diagram in the image. What are the main components?",
    images=["base64_encoded_image_data..."],
    max_reasoning_steps=10
)

# The LLM (llama3) will realize it needs vision, call `tool_vlm_query(0, "Identify the main components in this diagram.")`,
# receive the text description from the VLM (llava), and synthesize the final answer.

# CRITICAL: The tool is ONLY mounted if enable_vlm_query=True is passed.
# Since the active model (text_llm) lacks vision, we MUST enable it explicitly.
```

## Using LoLLMs Client with Different Bindings

`lollms-client` supports a wide range of LLM backends through its binding system. This section provides practical examples of how to initialize `LollmsClient` for each of the major supported bindings.

### A New Configuration Model

Configuration for all bindings has been unified. Instead of passing parameters like `host_address` or `model_name` directly to the `LollmsClient` constructor, you now pass them inside a single dictionary: `llm_binding_config`.

This approach provides a clean, consistent, and extensible way to manage settings for any backend. Each binding defines its own set of required and optional parameters (e.g., `host_address`, `model_name`, `service_key`, `n_gpu_layers`).

```python
# General configuration pattern
from lollms_client import LollmsClient
# ... other imports as needed

# lc = LollmsClient(
#     llm_binding_name="your_binding_name",
#     llm_binding_config={
#         "parameter_1_for_this_binding": "value_1",
#         "parameter_2_for_this_binding": "value_2",
#         # ... and so on
#     }
# )
```

---

### 1. Core and Local Server Bindings

These bindings connect to servers running on your local network, including the core LoLLMs server itself.

#### **LoLLMs (Default Binding)**

This connects to a running LoLLMs service, which acts as a powerful backend providing access to models, personalities, and tools. This is the default and most feature-rich way to use `lollms-client`.

**Prerequisites:**
*   A LoLLMs server instance installed and running (e.g., `lollms-webui`).
*   An API key can be generated from the LoLLMs web UI (under User Settings -> Security) if security is enabled.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

try:
    # The default port for a LoLLMs server is 9642 (a nod to The Hitchhiker's Guide to the Galaxy).
    # The API key can also be set via the LOLLMS_API_KEY environment variable.
    config = {
        "host_address": "http://localhost:9642",
        # "service_key": "your_lollms_api_key_here" # Uncomment and replace if security is enabled
        # "verify_ssl_certificate": True #if false the ssl certifcate verification will be ignored (only used when using https in lollms service address)
    }

    lc = LollmsClient(
        llm_binding_name="lollms", # This is the default, so specifying it is optional
        llm_binding_config=config
    )

    response = lc.generate_text("What is the answer to life, the universe, and everything?")
    ASCIIColors.green(f"\nResponse from LoLLMs: {response}")

except ConnectionRefusedError:
    ASCIIColors.error("Connection refused. Is the LoLLMs server running at http://localhost:9642?")
except ValueError as ve:
    ASCIIColors.error(f"Initialization Error: {ve}")
except Exception as e:
    ASCIIColors.error(f"An unexpected error occurred: {e}")
```

#### **Ollama**

The `ollama` binding connects to a running Ollama server instance on your machine or network.

**Prerequisites:**
*   [Ollama installed and running](https://ollama.com/).
*   Models pulled, e.g., `ollama pull llama3`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

try:
    # Configuration for a local Ollama server
    lc = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={
            "model_name": "llama3",  # Or any other model you have pulled
            "host_address": "http://localhost:11434" # Default Ollama address
        }
    )

    # Now you can use lc.generate_text(), lc.chat(), etc.
    response = lc.generate_text("Why is the sky blue?")
    ASCIIColors.green(f"\nResponse from Ollama: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing Ollama binding: {e}")
    ASCIIColors.info("Please ensure Ollama is installed, running, and the specified model is pulled.")
```

#### **PythonLlamaCpp (Local GGUF Models)**

The `pythonllamacpp` binding loads and runs GGUF model files directly using the powerful `llama-cpp-python` library. This is ideal for high-performance, local inference on CPU or GPU.

**Prerequisites:**
*   A GGUF model file downloaded to your machine.
*   `llama-cpp-python` installed. For GPU support, it must be compiled with the correct flags (e.g., `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`).

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os
from pathlib import Path

# Path to your GGUF model file
# IMPORTANT: Replace this with the actual path to your model file
# Example: MODEL_PATH = Path.home() / "models" / "your_model_name.gguf"
MODEL_PATH = Path("./path/to/your/model.gguf") 

# Binding-specific configuration
config = {
    "model_path": str(MODEL_PATH), # The path to the GGUF file
    "n_gpu_layers": -1,       # -1 for all layers to GPU, 0 for CPU
    "n_ctx": 4096,            # Context size
    "seed": -1,               # -1 for random seed
    "chat_format": "chatml"   # Or another format like 'llama-2' or 'mistral'
}

if not MODEL_PATH.exists():
    ASCIIColors.warning(f"Model file not found at: {MODEL_PATH}")
    ASCIIColors.warning("Skipping PythonLlamaCpp example. Please download a GGUF model and update MODEL_PATH.")
else:
    try:
        lc = LollmsClient(
            llm_binding_name="pythonllamacpp",
            llm_binding_config=config
        )

        response = lc.generate_text("Write a recipe for a great day.")
        ASCIIColors.green(f"\nResponse from PythonLlamaCpp: {response}")

    except ImportError:
        ASCIIColors.error("`llama-cpp-python` not installed. Please install it (`pip install llama-cpp-python`) to run this example.")
    except Exception as e:
        ASCIIColors.error(f"Error initializing PythonLlamaCpp binding: {e}")
        ASCIIColors.info("Please ensure the model path is correct and `llama-cpp-python` is correctly installed (with GPU support if desired).")

```

---

### 2. Cloud Service Bindings

These bindings connect to hosted LLM APIs from major providers.

#### **OpenAI**

Connects to the official OpenAI API to use models like GPT-4o, GPT-4, and GPT-3.5.

**Prerequisites:**
*   An OpenAI API key (starts with `sk-...`). It's recommended to set this as an environment variable `OPENAI_API_KEY`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

try:
    if "OPENAI_API_KEY" not in os.environ and "your_openai_api_key_here" in "your_openai_api_key_here":
        ASCIIColors.warning("OPENAI_API_KEY not set in environment or hardcoded. Skipping OpenAI example.")
    else:
        lc = LollmsClient(
            llm_binding_name="openai",
            llm_binding_config={
                "model_name": "gpt-4o", # Or "gpt-3.5-turbo"
                "service_key": os.environ.get("OPENAI_API_KEY", "your_openai_api_key_here") 
                # ^ service_key is optional if OPENAI_API_KEY env var is set
            }
        )

        response = lc.generate_text("What is the difference between AI and machine learning?")
        ASCIIColors.green(f"\nResponse from OpenAI: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing OpenAI binding: {e}")
    ASCIIColors.info("Please ensure your OpenAI API key is correctly set and you have access to the specified model.")
```

#### **Google Gemini**

Connects to Google's Gemini family of models via the Google AI Studio API.

**Prerequisites:**
*   A Google AI Studio API key. It's recommended to set this as an environment variable `GEMINI_API_KEY`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["GEMINI_API_KEY"] = "your_google_api_key_here"

try:
    if "GEMINI_API_KEY" not in os.environ and "your_google_api_key_here" in "your_google_api_key_here":
        ASCIIColors.warning("GEMINI_API_KEY not set in environment or hardcoded. Skipping Gemini example.")
    else:
        lc = LollmsClient(
            llm_binding_name="gemini",
            llm_binding_config={
                "model_name": "gemini-1.5-pro-latest",
                "service_key": os.environ.get("GEMINI_API_KEY", "your_google_api_key_here")
            }
        )

        response = lc.generate_text("Summarize the plot of 'Dune' in three sentences.")
        ASCIIColors.green(f"\nResponse from Gemini: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing Gemini binding: {e}")
    ASCIIColors.info("Please ensure your Google AI Studio API key is correctly set and you have access to the specified model.")
```

#### **Anthropic Claude**

Connects to Anthropic's API to use the Claude family of models, including Claude 3.5 Sonnet, Opus, and Haiku.

**Prerequisites:**
*   An Anthropic API key. It's recommended to set this as an environment variable `ANTHROPIC_API_KEY`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key_here"

try:
    if "ANTHROPIC_API_KEY" not in os.environ and "your_anthropic_api_key_here" in "your_anthropic_api_key_here":
        ASCIIColors.warning("ANTHROPIC_API_KEY not set in environment or hardcoded. Skipping Claude example.")
    else:
        lc = LollmsClient(
            llm_binding_name="claude",
            llm_binding_config={
                "model_name": "claude-3-5-sonnet-20240620",
                "service_key": os.environ.get("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
            }
        )

        response = lc.generate_text("What are the core principles of constitutional AI?")
        ASCIIColors.green(f"\nResponse from Claude: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing Claude binding: {e}")
    ASCIIColors.info("Please ensure your Anthropic API key is correctly set and you have access to the specified model.")
```

---

### 3. API Aggregator Bindings

These bindings connect to services that provide access to many different models through a single API.

#### **OpenRouter**

OpenRouter provides a unified, OpenAI-compatible interface to access models from dozens of providers (Google, Anthropic, Mistral, Groq, etc.) with one API key.

**Prerequisites:**
*   An OpenRouter API key (starts with `sk-or-...`). It's recommended to set this as an environment variable `OPENROUTER_API_KEY`.

**Usage:**
Model names must be specified in the format `provider/model-name`.

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["OPENROUTER_API_KEY"] = "your_openrouter_api_key_here"

try:
    if "OPENROUTER_API_KEY" not in os.environ and "your_openrouter_api_key_here" in "your_openrouter_api_key_here":
        ASCIIColors.warning("OPENROUTER_API_KEY not set in environment or hardcoded. Skipping OpenRouter example.")
    else:
        lc = LollmsClient(
            llm_binding_name="open_router",
            llm_binding_config={
                "model_name": "anthropic/claude-3-haiku-20240307",
                # "open_router_api_key": os.environ.get("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
            }
        )

        response = lc.generate_text("Explain what an API aggregator is, as if to a beginner.")
        ASCIIColors.green(f"\nResponse from OpenRouter: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing OpenRouter binding: {e}")
    ASCIIColors.info("Please ensure your OpenRouter API key is correctly set and you have access to the specified model.")
```

#### **Groq**

While Groq is a direct provider, it's famous as an aggregator of speed. It runs open-source models on custom LPU hardware for exceptionally fast inference.

**Prerequisites:**
*   A Groq API key. It's recommended to set this as an environment variable `GROQ_API_KEY`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

try:
    if "GROQ_API_KEY" not in os.environ and "your_groq_api_key_here" in "your_groq_api_key_here":
        ASCIIColors.warning("GROQ_API_KEY not set in environment or hardcoded. Skipping Groq example.")
    else:
        lc = LollmsClient(
            llm_binding_name="groq",
            llm_binding_config={
                "model_name": "llama3-8b-8192", # Or "mixtral-8x7b-32768"
                # "groq_api_key": os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
            }
        )

        response = lc.generate_text("Write a 3-line poem about incredible speed.")
        ASCIIColors.green(f"\nResponse from Groq: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing Groq binding: {e}")
    ASCIIColors.info("Please ensure your Groq API key is correctly set and you have access to the specified model.")
```

#### **Hugging Face Inference API**

This connects to the serverless Hugging Face Inference API, allowing experimentation with thousands of open-source models without local hardware.

**Note:** This API can have "cold starts," so the first request might be slow.

**Prerequisites:**
*   A Hugging Face User Access Token (starts with `hf_...`). It's recommended to set this as an environment variable `HF_API_KEY`.

**Usage:**

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

# Set your API key as an environment variable or directly in the config
# os.environ["HF_API_KEY"] = "your_hugging_face_token_here"

try:
    if "HF_API_KEY" not in os.environ and "your_hugging_face_token_here" in "your_hugging_face_token_here":
        ASCIIColors.warning("HF_API_KEY not set in environment or hardcoded. Skipping Hugging Face Inference API example.")
    else:
        lc = LollmsClient(
            llm_binding_name="hugging_face_inference_api",
            llm_binding_config={
                "model_name": "google/gemma-1.1-7b-it", # Or other suitable models from HF
                # "hf_api_key": os.environ.get("HF_API_KEY", "your_hugging_face_token_here")
            }
        )

        response = lc.generate_text("Write a short story about a robot who discovers music.")
        ASCIIColors.green(f"\nResponse from Hugging Face: {response}")

except Exception as e:
    ASCIIColors.error(f"Error initializing Hugging Face Inference API binding: {e}")
    ASCIIColors.info("Please ensure your Hugging Face API token is correctly set and you have access to the specified model.")```
```

---

### 4. Local Multimodal and Advanced Bindings

#### **Diffusers (Local Text-to-Image Generation and Editing)**

The `diffusers` binding leverages the Hugging Face `diffusers` library to run a vast array of text-to-image models locally on your own hardware (CPU or GPU). It supports models from Hugging Face and Civitai, providing everything from basic image generation to advanced, state-of-the-art image editing.

**Prerequisites:**
*   `torch` and `torchvision` must be installed. For GPU acceleration, it's critical to install the version that matches your CUDA toolkit.
*   The binding will attempt to auto-install other requirements like `diffusers`, `transformers`, and `safetensors`.

**Usage:**

**Example 1: Basic Text-to-Image Generation**
This example shows how to generate an image from a simple text prompt using a classic Stable Diffusion model.

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
from pathlib import Path

try:
    # Initialize the client with the diffusers TTI binding
    # Let's use a classic Stable Diffusion model for this example
    lc = LollmsClient(
        tti_binding_name="diffusers",
        tti_binding_config={
            "model_name": "runwayml/stable-diffusion-v1-5",
            # Other options: "device", "torch_dtype_str", "enable_xformers"
        }
    )

    prompt = "A high-quality photograph of an astronaut riding a horse on Mars."
    ASCIIColors.yellow(f"Generating image for prompt: '{prompt}'")

    # Generate the image. The result is returned as bytes.
    image_bytes = lc.generate_image(prompt, width=512, height=512)

    if image_bytes:
        output_path = Path("./astronaut_on_mars.png")
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        ASCIIColors.green(f"Image saved successfully to: {output_path.resolve()}")
    else:
        ASCIIColors.error("Image generation failed.")

except Exception as e:
    ASCIIColors.error(f"An error occurred with the Diffusers binding: {e}")
    ASCIIColors.info("Please ensure torch is installed correctly for your hardware (CPU/GPU).")
```

**Example 2: Advanced Multi-Image Fusion with Qwen-Image-Edit-2509**
This example demonstrates a cutting-edge capability: using a specialized model to fuse elements from multiple input images based on a text prompt. Here, we'll ask the model to take a person from one image and place them in the background of another.

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
from pathlib import Path

# --- IMPORTANT ---
# Replace these with actual paths to your local images
path_to_person_image = "./path/to/your/person.jpg"
path_to_background_image = "./path/to/your/background.jpg"

if not Path(path_to_person_image).exists() or not Path(path_to_background_image).exists():
    ASCIIColors.warning("Input images not found. Skipping multi-image fusion example.")
    ASCIIColors.warning(f"Please update 'path_to_person_image' and 'path_to_background_image'.")
else:
    try:
        # Initialize with the advanced Qwen multi-image editing model
        lc = LollmsClient(
            tti_binding_name="diffusers",
            tti_binding_config={
                "model_name": "Qwen/Qwen-Image-Edit-2509",
                "torch_dtype_str": "bfloat16" # Recommended for this model
            }
        )

        # The prompt guides how the images are combined
        prompt = "Place the person from the first image into the scenic background of the second image."
        ASCIIColors.yellow(f"Fusing images with prompt: '{prompt}'")

        # The edit_image method can accept a list of image paths for fusion
        fused_image_bytes = lc.edit_image(
            images=[path_to_person_image, path_to_background_image],
            prompt=prompt,
            num_inference_steps=50
        )

        if fused_image_bytes:
            output_path = Path("./fused_image_result.png")
            with open(output_path, "wb") as f:
                f.write(fused_image_bytes)
            ASCIIColors.green(f"Fused image saved successfully to: {output_path.resolve()}")
        else:
            ASCIIColors.error("Multi-image editing failed.")

    except Exception as e:
        ASCIIColors.error(f"An error occurred during multi-image fusion: {e}")
```

This powerful feature allows for complex creative tasks like character swapping, background replacement, and style transfer directly through the `lollms_client` library.

## 🔌 Tool Bindings: LCP & MCP

The library supports two primary tool execution frameworks:

### 1. LCP (LollmsCommunicationProtocol)
A lightweight, zero-dependency local tool execution framework. LCP uses AST parsing to automatically extract tool schemas from Python docstrings.

**Writing an LCP Tool:**
```python
# my_tool.py
def tool_analyze_data(file_name: str) -> dict:
    """
    Analyzes a file and returns statistics.

    Args:
        file_name (str): Name of the file to inspect in the workspace.
    """
    from pathlib import Path
    path = Path(file_name)
    if not path.exists():
        return {"success": False, "error": f"File '{file_name}' not found."}

    return {
        "success": True, 
        "output": f"File contains {len(path.read_text())} characters.",
        "prompt_injection": "Tell the user the analysis is complete."
    }
```

**Security Gates:**
*   `enable_code_execution=True`: Registers `tool_execute_python_code` (arbitrary Python string execution).
*   `allow_dynamic_tools=True`: Allows the LLM to write and execute its own `.py` tools on the fly via `type="tool"` artifacts.

### 2. MCP (Model Context Protocol)
Connects to external MCP-compliant servers (local `stdio` or remote `http`). Tools are namespaced using `alias::tool_name`.

```python
from lollms_client.tools_bindings.mcp import MCPBinding

mcp_binding = MCPBinding(servers=[
    {
        "alias": "local_fs",
        "type": "stdio",
        "command": ["python", "-m", "mcp_server_filesystem", "/path/to/allowed/dir"]
    }
])

result = mcp_binding.execute_tool(
    tool_name="local_fs::read_file",
    params={"path": "/path/to/allowed/dir/test.txt"}
)
```

---

## 🧠 Cognitive Memory Architecture (`lollms_memory`)

The memory system is a stateful, human-brain-inspired cognitive graph that persists across sessions. It combines mathematical decay with semantic graph traversal to manage context efficiently.

### Multi-Level Memory Tiers

| Tier | Type | Scope | Context Behavior |
|---|---|---|---|
| **Level 0** | Volatile Scratchpad | Single Turn | Appended before the last user prompt. Cleared after turn. |
| **Level 1** | Working Memory | Active Session | Rendered verbatim in the system prompt context. |
| **Level 2** | Deep Memory | Inactive / Latent | Injected as lightweight ID handles only. The LLM must use `<mem_load>` to promote them. |
| **Level 3** | Archived Memory | Deep Storage | Completely excluded from context. Evaluated during the Dream Cycle for permanent deletion or restoration. |
| **Level 4** | Episodic Memory | Permanent History | Chronological, immutable logs of past interactions. Used for retrospective queries. |

### Petroff's Power-Law Decay & Spreading Activation

Memories do not expire linearly. Their activation energy ($B_i$) decays logarithmically based on retrieval history:
$$B_i = \ln \left( \sum_{j} (t - t_j)^{-d} \right)$$
If $B_i$ drops below the `demotion_threshold`, the memory is demoted from Level 1 to Level 2. 

When a memory is retrieved, energy is spread **multiplicatively** to its semantically linked neighbors (defined via `RELATED_TO`, `SUPPORTS`, `CONTRADICTS` graph edges). This pre-warms related concepts in Deep Memory without injecting them into the context.

### The Dream Cycle (`dream()`)

An asynchronous consolidation pass that runs periodically or on-demand:
1.  **Soft-Delete Purge**: Permanently deletes nodes with `0.0` importance.
2.  **Centrality Auditing**: Computes PageRank-like centrality to identify keystone memories.
3.  **Synaptic Fusion**: Merges redundant memories sharing identical tags.
4.  **LLM-Assisted Forgetting**: Evaluates faded memories and decides whether to restore or permanently purge them.

### Memory XML Commands

The LLM interacts with the memory graph using stream tags:
*   `<mem_new tags="..." subject="..." predicate="..." object="..." importance="0.9">Fact</mem_new>`
*   `<mem_update id="UUID">Updated Fact</mem_update>`
*   `<mem_tag id="UUID" />` (Boosts importance)
*   `<mem_load id="UUID" />` (Promotes Deep Memory to Working Memory)
*   `<mem_rel source="UUID" target="UUID" type="SUPPORTS" weight="1.0" />`

### Dual-Database Architecture & Graph Traversal

The system utilizes a **Dual-Database Attachment** paradigm over SQLite:
1.  **Private Local Database (`main.memories`)**: Bound strictly to the current discussion session.
2.  **Shared Semantic Database (`shared_mem_db.memories`)**: Shared across all discussions inside a given project workspace.

When a `shared_db_path` is provided, the manager executes `ATTACH DATABASE` and constructs a cross-schema `UNION ALL` query layer, allowing the application to query both local and shared schemas transparently as a single unified graph.

**Graph Traversal API (`MemoryMixin`)**:
*   `add_memory_relationship(source_id, target_id, relationship_type, weight)`: Create an explicit graph edge.
*   `traverse_memory_graph(start_id, max_depth, relationship_types)`: Perform a Breadth-First Search (BFS) traversal to discover distant connections.
*   `get_high_centrality_memories(top_k, level)`: Retrieve the most connected/important memories based on graph centrality.

---

## 👁️ Multi-Tier Artefact Visibility & Context Budget (`lollms_artefact`)

To prevent context window exhaustion, the Artefact System enforces a strict visibility state machine. **All newly registered or tool-generated files default to `TREE_UNLOCKABLE` (`[U]`)** to prevent automatic context pollution.

### Visibility Tiers

| Tier | Symbol | Context Behavior |
|---|---|---|
| **FULL** | `[C]` | Content (or `.lam` schema) fully injected into the prompt. |
| **METADATA** | `[M]` | Only basic metadata (filename, size, type) is injected. |
| **TREE_UNLOCKABLE** | `[U]` | Listed in the directory index, but excluded from context. **The default state.** |
| **TREE_LOCKED** | `[L]` | Excluded from context. The LLM **cannot** unlock this. |
| **HIDDEN** | — | Completely excluded from both context and the directory tree. |

### State Transition Matrix

The system enforces a strict state machine for artifact visibility. The LLM can trigger transitions using XML tags, while the host application or system orchestrator can trigger transitions via API calls or background processes.

| Current State | Target State | Trigger / Mechanism | Description |
| :--- | :--- | :--- | :--- |
| **`[U]` TREE_UNLOCKABLE** | **`[C]` FULL** | `<unlock_file>` tag | LLM requests to load file content into context. |
| **`[U]` TREE_UNLOCKABLE** | **`[L]` TREE_LOCKED** | `<lock_file>` tag | LLM requests to lock the file. |
| **`[U]` TREE_UNLOCKABLE** | **`HIDDEN`** | `<hide_file>` tag | LLM removes file from awareness. |
| **`[C]` FULL** | **`[L]` TREE_LOCKED** | `<lock_file>` tag | LLM requests to lock the file. |
| **`[C]` FULL** | **`HIDDEN`** | `<hide_file>` tag | LLM removes file from awareness. |
| **`[L]` TREE_LOCKED** | **`HIDDEN`** | `<hide_file>` tag | LLM removes file from awareness. |
| **`[M]` METADATA** | **`[C]` FULL** | `<unlock_file>` tag | LLM promotes from metadata to full content. |
| **`[M]` METADATA** | **`[L]` TREE_LOCKED** | `<lock_file>` tag | LLM requests to lock the file. |
| **`[M]` METADATA** | **`HIDDEN`** | `<hide_file>` tag | LLM removes file from awareness. |
| **`HIDDEN`** | **`[C]` FULL** | Host Application API | User or host app explicitly activates the artifact. |
| **`HIDDEN`** | **`[U]` TREE_UNLOCKABLE** | System Auto-Sync | New file detected on disk or external tool modifies hidden file. |
| **`[L]` TREE_LOCKED** | **`[C]` FULL** | Host Application API | User or host app explicitly unlocks the file. |
| **`[C]` FULL** | **`[U]` TREE_UNLOCKABLE** | System Auto-Prune | Context management demotes file to save tokens. |
| **`[C]` FULL** | **`[M]` METADATA** | System Auto-Prune | Context management demotes to metadata-only. |

> **Note**: The LLM **cannot** unlock a `[L]` (Locked) file. Once locked, it remains locked unless the host application intervenes.

### LLM Control Tags

The LLM can dynamically manage its context budget by emitting these tags:
```xml
<unlock_file>
filename.ext
</unlock_file>

<lock_file>
filename.ext
</lock_file>

<hide_file>
filename.ext
</hide_file>
```
The `ChatMixin` intercepts these tags, applies the visibility changes via `ArtefactManager.set_visibility()`, and forces a continuation round so the LLM immediately utilizes the newly available (or freed) context.

### The Context Budget Guard

The system actively blocks the LLM from unlocking files >50,000 tokens, instructing it to use tools (SQL, grep) instead. Tool-generated files >100KB are automatically registered as `[U]` to protect the context window.

---

## 🧬 Advanced Agentic Loop Mechanics (`lollms_discussion`)

The `LollmsDiscussion.chat()` method is an **Agentic State Machine** that goes far beyond simple tool execution.

### The `<done/>` Termination Protocol

The loop does not rely on heuristics to decide when to stop.
1.  **Round 1 Short-Circuit**: If the LLM generates pure text without `<tool>` or `<artifact>` tags on the first round, the loop breaks immediately.
2.  **Action Continuation**: If the LLM emits a functional tag, the action is executed. A mandate is injected: "When you think you finished your task, issue a final conversational text and end it with a `<done/>` tag."
3.  **Explicit Termination**: The loop only breaks if the LLM emits `<done/>` on a new line, or if `max_reasoning_steps` is reached. The tag is stripped from the final UI output.

### The Context Diet Protocol (Three-View Protocol)

To prevent context bloat during long autonomous sessions, historical messages are sanitized using a three-view protocol:
1.  **Recent View (Last 2 Functional Actions)**: Raw content is preserved **verbatim**, including `<tool>` tags and `<processing>` logs. This maintains perfect KV-cache alignment for multi-turn tool chaining.
2.  **Reduced View (Older Turns)**: Functional XML is replaced with opaque placeholders (e.g., `[🔒SYSTEM_ARTIFACT_ANCHOR:main.py]`). Execution logs are scrubbed completely.
3.  **User View (Always Verbatim)**: User messages are **never** sanitized.

### Dual-Copy Virtual History Persistence

To solve **Multi-Turn Context Amnesia**, the system stores two copies of assistant messages involving multi-step tool calls:
*   **UI Content**: The sanitized, user-facing text.
*   **Virtual History**: The raw, unsanitized alternation of `<tool>` tags and `<tool_result>` payloads.

When building the context for the *next* turn, the sanitized UI content is discarded and replaced with the raw Virtual History. This ensures the LLM sees its exact previous execution path, preventing it from repeating tools or halting prematurely.

---

## 🏗️ Agent Application Build Guide

To build a robust autonomous application using `lollms_client`, follow this architectural pattern:

1.  **Initialize the Client**: Configure your LLM and Tool bindings using the unified `llm_binding_config` dictionary.
2.  **Define the Personality**: Use `LollmsPersonality` or import a bundle via `PersonalityBundle.import_bundle()`. Include explicit autonomous instructions in the `system_prompt` (e.g., "NEVER ask the user for help", "Emit `<done/>` when finished").
3.  **Set Capability Flags**: Use `CapabilityFlags` to strictly gate dangerous capabilities (`enable_code_execution=True`, `enable_sub_agents=True`).
4.  **Configure the Workspace**: Point the `Agent` or `LollmsDiscussion` to an isolated `workspace_path`.
5.  **Execute the Loop**: Call `agent.chat()` with a high `max_reasoning_steps` (80-150 for build→test→fix loops) and low `temperature` (0.1-0.3 for code).

### Listing Available Models

You can query the active LLM binding to get a list of models it supports or has available. The exact information returned depends on the binding (e.g., Ollama lists local models, OpenAI lists all its API models).

```python
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors
import os

try:
    # Initialize client for Ollama (or any other binding)
    lc = LollmsClient(
        llm_binding_name="ollama",
        llm_binding_config={
            "host_address": "http://localhost:11434"
            # model_name is not needed just to list models
        }
    )

    ASCIIColors.yellow("\nListing available models for the current binding:")
    available_models = lc.list_models()

    if isinstance(available_models, list):
        for model in available_models:
            # Model structure varies by binding, common fields are 'name'
            model_name = model.get('name', 'N/A')
            model_size = model.get('size', 'N/A') # Common for Ollama
            print(f"- {model_name} (Size: {model_size})")
    elif isinstance(available_models, dict) and "error" in available_models:
        ASCIIColors.error(f"Error listing models: {available_models['error']}")
    else:
        print("Could not retrieve model list or unexpected format.")

except Exception as e:
    ASCIIColors.error(f"An error occurred: {e}")

```