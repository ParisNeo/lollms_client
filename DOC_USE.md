# LoLLMs Client - User Documentation

**Version:** (Based on files up to 2025-06-04)
**GitHub:** [https://github.com/ParisNeo/simplified_lollms](https://github.com/ParisNeo/simplified_lollms)
**PyPI:** [https://pypi.org/project/lollms-client/](https://pypi.org/project/lollms-client/)
**License:** Apache 2.0

## 1. Introduction

Welcome to `lollms_client`! This Python library provides a simple and powerful way to interact with a wide range of Large Language Models (LLMs), other generative AI services (Text-to-Speech, Text-to-Image, etc.), and enable robust function calling capabilities for your LLMs.

Whether you want to integrate advanced AI capabilities into your Python applications, experiment with different models, or build custom AI-powered tools that can interact with the real world, `lollms_client` aims to make the process straightforward.

**Key Features:**

*   **Unified Interface:** Interact with diverse AI models and services using a consistent API.
*   **Extensible Bindings:** Easily switch between different AI backends (e.g., Ollama, OpenAI, local Llama.cpp, Hugging Face Transformers, Bark TTS, Piper TTS, XTTS) by changing configuration parameters.
*   **Function Calling with MCP:** Empower LLMs to use external tools and functions through the Model Context Protocol (MCP), with built-in support for local Python tool execution via the `local_mcp` binding.
*   **Local & Cloud Support:** Works with both locally hosted models and cloud-based AI services.
*   **Multimodal Ready:** Designed to support text, speech, image, and potentially video/music generation and processing.
*   **Helper Utilities:** Includes tools for prompt engineering and discussion management.
*   **High-Level Operations:** Direct methods on `LollmsClient` for tasks like summarization, code generation, and structured Q&A.

This guide will walk you through installing `lollms_client` and using its core features.

## 2. Installation

### 2.1. Basic Installation

You can install `lollms_client` directly from PyPI using pip:

```bash
pip install lollms-client
```

This will install the core client library. Dependencies for specific AI bindings (like `ollama`, `openai-whisper`, `TTS`, `piper-tts`, `audiocraft`, `transformers`, `llama-cpp-python`, etc.) are **not** installed by default to keep the core installation lightweight. They will be attempted to be installed automatically by `pipmaster` (a utility included in `lollms-client`) when you first try to use a binding that requires them, or you can pre-install them.

### 2.2. Installing Dependencies for Specific Bindings

When you initialize `LollmsClient` with a specific binding, the library will attempt to install any required Python packages for that binding if they are missing.

**Common Binding Dependencies (you might need to pre-install or let `pipmaster` handle them):**

*   **Ollama:** `pip install ollama`
*   **OpenAI:** `pip install openai tiktoken`
*   **LlamaCppServer (GGUF server):** `pip install requests pillow llama-cpp-binaries` (or `llama-cpp-python[server]`)
    *   `llama-cpp-binaries` attempts to provide pre-compiled wheels. If this fails, you might need to compile `llama-cpp-python` from source with specific hardware flags (e.g., for CUDA or Metal).
*   **PythonLlamaCpp (GGUF local):** `pip install llama-cpp-python pillow tiktoken`
    *   Similar to LlamaCppServer, `llama-cpp-python` might require compilation for optimal performance.
*   **Transformers (Hugging Face local):** `pip install torch transformers accelerate bitsandbytes sentence_transformers pillow`
    *   `bitsandbytes` is primarily for 4-bit/8-bit quantization on CUDA.
*   **vLLM (High-performance local LLM serving):** `pip install vllm torch transformers huggingface_hub pillow`
    *   vLLM is Linux-only and typically requires a compatible NVIDIA GPU.
*   **Bark TTS/TTM:** `pip install transformers accelerate sentencepiece torch scipy numpy`
*   **Piper TTS:** `pip install piper-tts onnxruntime`
*   **XTTS (Coqui TTS):** `pip install TTS torch torchaudio scipy numpy soundfile`
*   **AudioCraft (MusicGen TTM):** `pip install audiocraft torch torchaudio scipy numpy`
*   **Whisper STT (OpenAI):** `pip install openai-whisper torch`
*   **WhisperCpp STT (Local GGUF):** Requires a compiled `whisper.cpp` executable and `ffmpeg`.
*   **Local MCP Binding (`local_mcp`):**
    *   `duckduckgo_search` (for the default `internet_search` tool)
    *   `RestrictedPython` (for the default `python_interpreter` tool)

**System-Level Dependencies:**

*   **`ffmpeg`:** Required by many audio/video processing bindings (Whisper, WhisperCpp, AudioCraft for MP3, XTTS if it processes audio). Install it system-wide:
    *   **Linux:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS:** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.
*   **`whisper.cpp` executable:** For the `whispercpp` STT binding, you need to compile and install the `whisper.cpp` command-line tool from [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp).

## 3. Core Usage: `LollmsClient`

The primary way to use the library is through the `LollmsClient` class.

### 3.1. Initialization

```python
from lollms_client import LollmsClient

# Example 1: LLM with Ollama, TTS with Bark, and MCP with local_mcp for default tools
try:
    client = LollmsClient(
        # LLM Configuration
        binding_name="ollama",
        model_name="mistral", # LLM model
        host_address="http://localhost:11434", # For Ollama

        # TTS Configuration
        tts_binding_name="bark",
        tts_binding_config={
            "model_name": "suno/bark-small",
            "default_voice": "v2/en_speaker_3",
            "device": "cuda" # Optional for Bark: "cpu", "cuda", "mps"
        },

        # MCP (Function Calling) Configuration
        mcp_binding_name="local_mcp", # Use the local MCP binding
        # mcp_binding_config can be used to point to a custom tools folder:
        # mcp_binding_config={"tools_folder_path": "path/to/my/custom_mcp_tools"},
        # If mcp_binding_config is None or tools_folder_path is not set,
        # it uses the packaged default_tools (internet_search, file_writer, etc.).

        # STT Configuration (Example with Whisper)
        # stt_binding_name="whisper",
        # stt_binding_config={ "model_name": "base", "device": "cuda" }

        # TTI Configuration (Example with DALL-E, requires OPENAI_API_KEY env var)
        # tti_binding_name="dalle",
        # tti_binding_config={"model_name": "dall-e-3"} # Or "dall-e-2"
    )
    print("LollmsClient initialized with LLM, TTS, and MCP (local_mcp for default tools).")

except Exception as e:
    print(f"Error initializing LollmsClient: {e}")

```

**Key `LollmsClient` Initialization Parameters:**

*   `binding_name` (str): The name of the LLM binding to use (e.g., "ollama", "openai", "llamacpp", "pythonllamacpp", "transformers", "lollms").
*   `model_name` (str): The identifier for the LLM model.
*   `host_address` (Optional[str]): For server-based bindings.
*   `models_path` (Optional[str]): For local file-based LLM bindings.
*   `service_key` (Optional[str]): API key for services like OpenAI.
*   `llm_binding_config` (Optional[Dict]): Additional parameters for the LLM binding.
*   `tts_binding_name`, `stt_binding_name`, `tti_binding_name`, `ttm_binding_name`, `ttv_binding_name`, `mcp_binding_name`: Names of bindings for different modalities.
*   `tts_binding_config`, `stt_binding_config`, etc.: Dictionaries of configuration parameters for specific modality bindings.
*   `verify_ssl_certificate` (bool): Defaults to `True`.

### 3.2. Generating Text

```python
from lollms_client import LollmsClient, MSG_TYPE

# Assume client is initialized as shown in 3.1
# For this example, let's re-init simply for text gen:
client = LollmsClient(binding_name="ollama", model_name="mistral")

# --- Simple Text Generation ---
prompt = "What is the capital of France?"
response_text = client.generate_text(prompt, n_predict=50)
print(f"Q: {prompt}\nA: {response_text}")

# --- Streaming Text Generation ---
def my_streaming_callback(chunk: str, message_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True

print("\nStreaming response:")
full_streamed_response = client.generate_text(
    prompt="Tell me a short story about a robot.",
    n_predict=200,
    stream=True,
    streaming_callback=my_streaming_callback,
    temperature=0.8,
)
print("\n--- End of Stream ---")

# --- Multimodal Text Generation (if LLM binding supports it) ---
# image_paths = ["path/to/your/image1.jpg"]
# multimodal_prompt = "Describe this image."
# try:
#     multimodal_response = client.generate_text(multimodal_prompt, images=image_paths)
#     print(f"Multimodal Response: {multimodal_response}")
# except Exception as e:
#     print(f"Multimodal generation failed: {e}")
```

**Common `generate_text` Parameters:**
*   `prompt` (str), `images` (Optional[List[str]]), `system_prompt` (str).
*   `n_predict`, `stream`, `streaming_callback`.
*   `temperature`, `top_k`, `top_p`, `repeat_penalty`, `seed`, `ctx_size`.

### 3.3. Tokenization & Embeddings

```python
# client = LollmsClient(...)
text = "Hello, world! This is a test."
tokens = client.tokenize(text)
print(f"Tokens for '{text}': {tokens}")
detokenized_text = client.detokenize(tokens)
print(f"Detokenized: {detokenized_text}")
token_count = client.count_tokens(text)
print(f"Token count: {token_count}")

# Embeddings (if binding supports it)
# try:
#     embedding_vector = client.embed("This is text to embed.")
#     print(f"Embedding (first 5 dims): {embedding_vector[:5]}...")
# except NotImplementedError:
#     print("The current LLM binding does not support embeddings.")
```

### 3.4. Listing and Switching LLM Models

```python
# client = LollmsClient(...)
available_models = client.list_models()
print(f"Available models for current binding ({client.binding.binding_name}): {available_models}")

# Example with Ollama
# if client.binding.binding_name == "ollama":
#     try:
#         client.switch_model("codellama") # Switch to another Ollama model
#         print("Switched to codellama model.")
#     except Exception as e:
#         print(f"Failed to switch model: {e}")
```
Behavior of `list_models()` and `switch_model()` is binding-dependent.

### 3.5. Using Other Modalities (TTS, STT, TTI, TTM, TTV)

Access modality bindings via attributes like `client.tts`, `client.stt`, etc., if they were configured during initialization.

**Text-to-Speech (TTS) Example (using Bark, as set up in 3.1):**
```python
# client = LollmsClient(...) # Initialized with tts_binding_name="bark"
if client.tts:
    try:
        text_to_say = "Welcome to AI-powered audio!"
        audio_bytes = client.tts.generate_audio(text_to_say, voice="v2/en_speaker_1") # Override default
        if audio_bytes:
            with open("bark_speech.wav", "wb") as f: f.write(audio_bytes)
            print("Bark speech saved to bark_speech.wav")
        available_voices = client.tts.list_voices()
        print(f"Available Bark voice presets (examples): {available_voices[:5]}")
    except Exception as e: print(f"Bark TTS error: {e}")
else: print("TTS binding not configured.")
```
Refer to specific binding documentation or its `__init__.py` test block for details.

## 4. Advanced Operations (Built-in)

`LollmsClient` provides several built-in methods for more complex tasks, previously part of `TasksLibrary`.

### 4.1. Code Generation

```python
# client = LollmsClient(...)
code_prompt = "Write a Python function to calculate Fibonacci numbers."
# Generates a single code block
python_code = client.generate_code(code_prompt, language="python")
if python_code:
    print(f"Generated Python Code:\n{python_code}")

# For multiple code blocks or specific templates:
# codes_list = client.generate_codes(prompt="Show me factorial in Python and JS.", ...)
```

### 4.2. Structured Q&A

```python
# client = LollmsClient(...)
# Yes/No Question
context_for_yes_no = "The sky is blue during the day. At night, it appears dark."
is_sky_green = client.yes_no(
    question="Is the sky green during the day?",
    context=context_for_yes_no,
    return_explanation=True
)
print(f"Is sky green? Answer: {is_sky_green['answer']}, Explanation: {is_sky_green['explanation']}")

# Multichoice Question
mc_question = "What is the capital city of France?"
mc_answers = ["Paris", "Berlin", "London", "Madrid"]
selected_option = client.multichoice_question(mc_question, mc_answers, return_explanation=False)
print(f"Multichoice - Selected Index: {selected_option}, Answer: {mc_answers[selected_option]}")
```

### 4.3. Summarization and Analysis

```python
# client = LollmsClient(...)
long_text = "..." # Your long document text here
# summary = client.sequential_summarize(
#     long_text,
#     chunk_processing_prompt="Extract key points from this chunk.",
#     final_memory_processing_prompt="Combine all key points into a concise summary."
# )
# print(f"Sequential Summary: {summary}")

# query = "What are the main differences between model A and model B discussed in the text?"
# analysis_result = client.deep_analyze(query, text=long_text)
# print(f"Deep Analysis for '{query}': {analysis_result}")
```
These methods (`sequential_summarize`, `deep_analyze`) are powerful for processing large texts by breaking them into manageable chunks.

## 5. Function Calling with MCP (Model Context Protocol)

`lollms-client` facilitates LLM interaction with external tools or functions using the Model Context Protocol (MCP). The `LollmsClient.generate_with_mcp()` method orchestrates this.

### 5.1. MCP Concept

MCP allows an LLM to:
1.  Be aware of available tools (functions with defined inputs/outputs).
2.  Decide to call a tool with specific parameters based on the user's prompt and conversation history.
3.  Receive the tool's output and use it to formulate a final answer or decide on further actions (like calling another tool).

### 5.2. `local_mcp` Binding

`lollms-client` includes a built-in MCP binding called `local_mcp`.
*   **Tool Discovery:** It discovers tools from a specified `tools_folder_path`. If no path is given, it uses a set of `default_tools` packaged with the client.
*   **Default Tools:**
    *   `internet_search`: Performs a web search (uses `duckduckgo_search`).
    *   `file_writer`: Writes or appends content to files.
    *   `python_interpreter`: Executes Python code snippets in a restricted environment.
    *   `generate_image_from_prompt`: Generates an image by calling the `LollmsClient`'s active TTI binding.
*   **Custom Tools:** You can create your own tools by placing a `<tool_name>.py` (with an `execute` function) and a `<tool_name>.mcp.json` (defining metadata like name, description, input/output schemas) in a folder and pointing `mcp_binding_config={"tools_folder_path": "your/tools/dir"}` to it during `LollmsClient` initialization.

### 5.3. Using `generate_with_mcp()`

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors
import json # For pretty printing results

# Callback for MCP streaming
def mcp_stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, turn_history: list = None) -> bool:
    """Handles various stages of MCP interaction."""
    # Simplified callback for brevity. See examples/ for more detailed one.
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: ASCIIColors.success(chunk, end="", flush=True) # LLM output
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START: ASCIIColors.info(f"\n>> MCP Step Start: {metadata.get('tool_name', chunk)}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END: ASCIIColors.info(f"\n<< MCP Step End. Result snippet: {str(metadata.get('result', ''))[:50]}...", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_INFO and metadata and metadata.get("type") == "tool_call_request": ASCIIColors.info(f"\nAI requests tool: {metadata.get('name')}({metadata.get('params')})", flush=True)
    return True

try:
    client_mcp = LollmsClient(
        binding_name="ollama", model_name="mistral", # Choose a capable LLM
        mcp_binding_name="local_mcp" # Activate local_mcp with its default tools
    )

    user_prompt_mcp = "Search the web for 'latest Python version' and then write the result into a file named 'python_version_info.txt'."
    ASCIIColors.blue(f"User: {user_prompt_mcp}")
    ASCIIColors.yellow("AI processing with MCP (streaming):")

    mcp_output = client_mcp.generate_with_mcp(
        prompt=user_prompt_mcp,
        streaming_callback=mcp_stream_callback,
        # interactive_tool_execution=True, # Uncomment to confirm each tool call
    )
    print("\n--- End of MCP Interaction ---")

    if mcp_output.get("error"):
        ASCIIColors.error(f"MCP Error: {mcp_output['error']}")
    else:
        ASCIIColors.cyan(f"\nFinal AI Answer: {mcp_output.get('final_answer', 'N/A')}")
        ASCIIColors.magenta("\nTool Calls Summary:")
        for tc in mcp_output.get("tool_calls", []):
            print(f"  - Tool: {tc.get('name')}, Params: {tc.get('params')}, Result (brief): {str(tc.get('result'))[:70]}...")

except Exception as e:
    print(f"Error in MCP example: {e}")
    # from ascii_colors import trace_exception; trace_exception(e) # For full trace

```

**Key `generate_with_mcp` Parameters:**
*   `prompt` (str): User's request.
*   `discussion_history` (Optional): For conversational context.
*   `images` (Optional): If the initial prompt includes images.
*   `tools` (Optional): Override discovered tools with a specific list for this call.
*   `max_tool_calls` (int): Limit on distinct tools executed per turn.
*   `max_llm_iterations` (int): Safety limit on LLM re-decisions to call tools.
*   `streaming_callback` (Optional): Receives updates about LLM thoughts, tool call requests, execution steps, and final answer chunks.
*   `interactive_tool_execution` (bool): If `True`, prompts the user in the console to confirm before each tool execution.

Refer to `examples/function_calling_with_local_custom_mcp.py` (for custom tools) and `examples/local_mcp.py` (for default tools) for more detailed examples.

## 6. `LollmsDiscussion`

Manages conversation history.

```python
from lollms_client import LollmsClient, LollmsDiscussion

client = LollmsClient(binding_name="ollama", model_name="mistral:latest")
discussion = LollmsDiscussion(lollmsClient=client) # Pass client for tokenization

discussion.add_message(sender="user", content="Hello, AI!")
# Format discussion for LLM, respecting context limits
formatted_prompt = discussion.format_discussion(max_allowed_tokens=1000) + f"\n{client.ai_full_header}"
ai_reply = client.generate_text(formatted_prompt, n_predict=50)
discussion.add_message(sender="assistant", content=ai_reply)

print("\n--- Full Discussion ---")
for msg in discussion.messages:
    print(f"{msg.sender}: {msg.content}")
```

## 6. `LollmsDiscussion` and the `chat()` Method

The `LollmsDiscussion` class is a powerful tool for managing conversation histories. The recommended way to generate text from a discussion is by using the `LollmsClient.chat()` method, which operates directly on a `LollmsDiscussion` object.

This approach is cleaner and more reliable than manually formatting prompts with `format_discussion()`, as it leverages the structured data within the discussion to build the perfect, API-specific payload for the backend model, including roles, images, and system prompts.

### 6.1. A Typical Chat Workflow

Here is a complete example demonstrating the standard conversational workflow.

```python
from lollms_client import LollmsClient, LollmsDiscussion

# 1. Initialize the Client and Discussion
# We pass the client to the discussion for tokenization capabilities.
client = LollmsClient(binding_name="ollama", model_name="llava")
discussion = LollmsDiscussion(lollmsClient=client)

# 2. Configure the Discussion (Optional but Recommended)
discussion.set_participants({
    "Explorer": "user",
    "Guide": "assistant"
})
discussion.set_system_prompt("You are a helpful 'Guide' for an 'Explorer'. Be concise and encouraging.")

# 3. Add the User's First Message
discussion.add_message(
    sender="Explorer",
    content="I've found a strange glowing mushroom. Is it safe?",
    images=[{"type": "url", "data": "https://example.com/glowing_mushroom.jpg"}]
)

# 4. Use client.chat() to Generate a Response
print("Guide is thinking...")
# The chat method handles all formatting internally.
# We can use a streaming callback to get the response token by token.
full_response = ""
def my_callback(token, message_type):
    global full_response
    print(token, end="", flush=True)
    full_response += token

# The chat() method takes the discussion and any generation parameters.
client.chat(
    discussion=discussion,
    stream=True,
    streaming_callback=my_callback,
    n_predict=256
)

# 5. Add the AI's Full Response Back to the Discussion
# This keeps the context updated for the next turn.
discussion.add_message(sender="Guide", content=full_response)

print("\n\n--- Full Discussion History ---")
# You can now view the complete, structured conversation
for msg in discussion.messages:
    print(f"<{msg.sender}>: {msg.content}" + (f" [{len(msg.images)} image(s)]" if msg.images else ""))
```

### 6.2. Benefits of Using `chat()`

*   **Simplicity**: No need to manually format prompts with special tokens (`!@>user:`).
*   **Robustness**: Automatically handles system prompts, participant roles, and multi-modal data (images).
*   **API-Agnostic**: Your code remains the same whether you're talking to an OpenAI, Ollama, or other compatible backend. The `LollmsBinding` handles the specific formatting.
*   **Branching Support**: You can easily generate a response from an alternative conversation path by passing the `branch_tip_id` to the `chat()` method.


## 7. Available Bindings (Summary)

As of this documentation, `lollms_client` includes or is designed for bindings such as:

*   **LLM:** `lollms`, `ollama`, `openai`, `llamacpp`, `pythonllamacpp`, `transformers`, `vllm`, `openllm`.
*   **TTS (Text-to-Speech):** `lollms`, `bark`, `xtts`, `piper`.
*   **STT (Speech-to-Text):** `lollms`, `whisper`, `whispercpp`.
*   **TTI (Text-to-Image):** `lollms`, `dalle`, `diffusers`.
*   **TTM (Text-to-Music/Sound):** `lollms`, `audiocraft`, `bark`.
*   **TTV (Text-to-Video):** `lollms`.
*   **MCP (Model Context Protocol):**
    *   `local_mcp`: Executes local Python tools. Comes with default tools for file I/O, internet search, Python execution, and image generation (delegates to active TTI binding).

To find configuration options for a specific binding, consult its `__init__.py` file within the `lollms_client/<modality>_bindings/<binding_name>/` directory and its `if __name__ == "__main__":` test block.

## 8. Examples

The `examples/` directory in the repository contains various scripts demonstrating how to use `lollms_client` for different tasks. Explore these examples to see `lollms-client` in action.

## 9. Troubleshooting

*   **Dependency Issues:** If a binding fails to load, check its specific Python package dependencies. `pipmaster` will attempt to install them. System-level tools like `ffmpeg` must be installed separately.
*   **Model Not Found:** For server-based bindings, ensure the server is running and the model name is correct. For local file-based bindings, ensure `models_path` and `model_name` are correctly set.
*   **CUDA/GPU Issues:** Ensure compatible drivers and PyTorch versions. Check `n_gpu_layers` (GGUF) or `device` parameters.
*   **`ffmpeg` Not Found:** Many audio bindings require `ffmpeg` in the system PATH.
*   **API Keys:** For services like OpenAI, ensure your API key (`service_key`) is correctly set.

## 10. Further Help

*   **GitHub Issues:** For bugs or feature requests: [main `simplified_lollms` GitHub repository](https://github.com/ParisNeo/simplified_lollms/issues).
*   **Examples:** Study the scripts in the `examples/` directory.
*   **Binding Code:** For advanced configuration, look into `__init__.py` files within `lollms_client/<modality>_bindings/<binding_name>/`.

We hope you enjoy using `lollms_client`!