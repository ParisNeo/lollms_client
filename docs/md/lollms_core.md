# LollmsCore Documentation

The `LollmsClient` class, found within `lollms_client/lollms_core.py`, is the central entry point for interacting with various Large Language Model (LLM) and multimodal services provided by the LOLLMS ecosystem. It acts as a unified interface to manage and leverage different binding implementations for text generation, text-to-speech (TTS), text-to-image (TTI), speech-to-text (STT), text-to-video (TTV), text-to-music (TTM), and Multi-tool Code Processors (MCP).

This documentation provides a comprehensive guide to initializing the client, interacting with different modalities, and utilizing advanced agentic features like RAG and MCP-driven workflows.

## `LollmsClient` Class

### Overview

The `LollmsClient` simplifies access to powerful AI capabilities by abstracting away the complexities of direct API or local model interactions. It supports a plug-and-play architecture for different "bindings" (e.g., Llama.cpp, Ollama, OpenAI, Bark, Diffusers, etc.), allowing users to seamlessly switch between providers and models without changing their core application logic.

### Initialization (`__init__`)

The `LollmsClient` constructor is highly configurable, allowing you to specify which bindings to use for each modality and provide specific configurations for them.

```python
from lollms_client import LollmsClient
from pathlib import Path
import os

# Example 1: Basic LLM-only client using the default 'lollms' binding
# Assumes lollms-webui is running on localhost:9600 and has a model loaded
client = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={
        "host_address": "http://localhost:9600",
        "model_name": "lollms_vllm_4bit", # Or whatever model is loaded in your webui
        "ctx_size": 4096,
        "n_predict": 1024,
        "temperature": 0.7,
        "repeat_penalty": 1.1,
    }
)
print(f"LLM client initialized with model: {client.get_model_name()}")

# Example 2: Client with LLM, TTS (Bark), and TTI (Diffusers) bindings
tts_config = {
    "output_path": "./speech_output",
    "voice": "v2/en_speaker_9", # Example voice
    "device": "cpu"
}
tti_config = {
    "output_path": "./image_output",
    "model_name": "stabilityai/stable-diffusion-v1-5",
    "device": "cpu"
}
mcp_config = {
    "host_address": "http://localhost:9600", # Or the host for your MCP
}

multimodal_client = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={"host_address": "http://localhost:9600", "model_name": "lollms_vllm_4bit"},
    tts_binding_name="bark",
    tts_binding_config=tts_config,
    tti_binding_name="diffusers",
    tti_binding_config=tti_config,
    mcp_binding_name="local_mcp", # Use the local MCP binding
    mcp_binding_config=mcp_config,
    user_name="John",
    ai_name="Lolly"
)
print(f"Multimodal client initialized. LLM: {multimodal_client.get_model_name()}, TTS: {multimodal_client.tts.binding_name if multimodal_client.tts else 'None'}, TTI: {multimodal_client.tti.binding_name if multimodal_client.tti else 'None'}")

# Ensure output directories exist for examples
os.makedirs(tts_config["output_path"], exist_ok=True)
os.makedirs(tti_config["output_path"], exist_ok=True)
```

**Parameters:**

*   `llm_binding_name` (str, default: "lollms"): The name of the LLM binding to use (e.g., "lollms", "ollama", "openai").
*   `tts_binding_name` (Optional[str]): The name of the Text-to-Speech binding (e.g., "bark", "piper_tts").
*   `tti_binding_name` (Optional[str]): The name of the Text-to-Image binding (e.g., "dalle", "diffusers").
*   `stt_binding_name` (Optional[str]): The name of the Speech-to-Text binding (e.g., "whisper", "whispercpp").
*   `ttv_binding_name` (Optional[str]): The name of the Text-to-Video binding.
*   `ttm_binding_name` (Optional[str]): The name of the Text-to-Music binding (e.g., "audiocraft", "bark").
*   `mcp_binding_name` (Optional[str]): The name of the Multi-tool Code Processor binding (e.g., "local_mcp", "remote_mcp").
*   `llm_bindings_dir`, `tts_bindings_dir`, `tti_bindings_dir`, etc. (Path): Directories where the respective binding implementations are located. Defaults to subdirectories within `lollms_client/`.
*   `llm_binding_config`, `tts_binding_config`, etc. (Optional[Dict[str, Any]]): Dictionaries containing binding-specific configurations (e.g., `host_address`, `model_name`, `api_key`, `device`, `output_path`).
*   `user_name` (str, default: "user"): The name used for the user in prompt formatting.
*   `ai_name` (str, default: "assistant"): The name used for the AI in prompt formatting.
*   `**kwargs`: Additional keyword arguments passed directly to the LLM binding's initialization. These can include LLM generation parameters like `ctx_size`, `n_predict`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads`, `stream`, `streaming_callback`, `model_name`. Note that these are primarily for setting *default* values for generation.

**Raises:**
*   `ValueError`: If the primary `llm_binding_name` cannot be created.
*   `Warning`: If any optional modality binding fails to initialize.

### Prompt Formatting Properties

`LollmsClient` provides properties to easily access formatted headers for building prompts, ensuring consistency with the chosen prompt template (which is internally managed by the LLM binding).

*   `system_full_header`: Returns the full system header (e.g., `!@>system:`).
*   `system_custom_header(ai_name)`: Returns a custom system header (e.g., `!@>MyAI:`).
*   `user_full_header`: Returns the full user header (e.g., `!@>user:`).
*   `user_custom_header(user_name)`: Returns a custom user header (e.g., `!@>John:`).
*   `ai_full_header`: Returns the full AI header (e.g., `!@>assistant:`).
*   `ai_custom_header(ai_name)`: Returns a custom AI header (e.g., `!@>Lolly:`).

**Example:**
```python
print(f"System Header: {client.system_full_header}")
print(f"User Header: {client.user_full_header}")
print(f"AI Header: {client.ai_full_header}")
print(f"Custom AI Header: {multimodal_client.ai_custom_header('Lolly')}")
```

### Binding Management

You can dynamically update the active binding for any modality after initialization.

*   `update_llm_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active LLM binding.
*   `update_tts_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTS binding.
*   `update_tti_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTI binding.
*   `update_stt_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active STT binding.
*   `update_ttv_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTV binding.
*   `update_ttm_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTM binding.
*   `update_mcp_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active MCP binding.

**Utility Methods:**

*   `get_ctx_size(model_name: Optional[str] = None) -> Optional[int]`: Returns the context size of the currently loaded model or a specified model.
*   `get_model_name() -> Optional[str]`: Returns the name of the currently loaded LLM model.
*   `set_model_name(model_name: str) -> bool`: Sets the model name for the current LLM binding. Note: This typically triggers a model load operation within the binding.

**Example:**
```python
# Assuming 'client' from Example 1 is active
print(f"Current LLM model: {client.get_model_name()}")
print(f"Current context size: {client.get_ctx_size()}")

# Example of updating LLM binding (e.g., switching to Ollama)
# try:
#     client.update_llm_binding(
#         binding_name="ollama",
#         config={"host_address": "http://localhost:11434", "model_name": "llama3"}
#     )
#     print(f"LLM binding updated to: {client.binding.binding_name}, model: {client.get_model_name()}")
# except ValueError as e:
#     print(f"Error updating LLM binding: {e}")
```

## Core LLM Interaction Methods

These methods provide the primary interface for text generation and related LLM functionalities.

### Tokenization and Model Information

*   `tokenize(text: str) -> list`: Tokenizes the input text using the active LLM binding's tokenizer.
*   `detokenize(tokens: list) -> str`: Detokenizes a list of tokens back into text.
*   `count_tokens(text: str) -> int`: Counts the number of tokens in a given text.
*   `count_image_tokens(image: str) -> int`: Estimates the number of tokens an image would consume in a multi-modal context (e.g., base64 encoded image).
*   `get_model_details() -> dict`: Retrieves detailed information about the currently loaded LLM model.
*   `listModels() -> List[Dict]`: Lists all models available to the active LLM binding.
*   `switch_model(model_name: str) -> bool`: Attempts to load a different model within the active LLM binding.

**Example:**
```python
text_to_process = "Hello, world! This is a test."
tokens = client.tokenize(text_to_process)
print(f"Text: '{text_to_process}'")
print(f"Tokens: {tokens}")
print(f"Detokenized: '{client.detokenize(tokens)}'")
print(f"Token count: {client.count_tokens(text_to_process)}")

# print(f"Available LLM models for '{client.binding.binding_name}': {client.listModels()}")
# client.switch_model("another_model_name") # Uncomment to test model switching
```

### Text Generation

#### `generate_text`

```python
generate_text(
    prompt: str,
    images: Optional[List[str]] = None,
    system_prompt: str = "",
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
    split: Optional[bool] = False,
    user_keyword: Optional[str] = "!@>user:",
    ai_keyword: Optional[str] = "!@>assistant:",
    **kwargs
) -> Union[str, dict]
```

Generates text based on a given prompt. This is a fundamental method that allows for flexible text generation, including multi-modal inputs (images) and control over various generation parameters. Parameters not explicitly provided will fall back to the defaults set during `LollmsClient` initialization or by the binding itself.

**Parameters:**
*   `prompt` (str): The main input prompt for the LLM.
*   `images` (Optional[List[str]]): A list of image paths (or base64 strings) for multi-modal generation.
*   `system_prompt` (str): An initial instruction or context for the AI.
*   `n_predict` (Optional[int]): Maximum number of tokens to generate.
*   `stream` (Optional[bool]): If `True`, the output will be streamed token by token to the `streaming_callback`.
*   `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads`: Standard LLM generation parameters.
*   `ctx_size` (Optional[int]): Overrides the default context size for this specific generation.
*   `streaming_callback` (Optional[Callable[[str, MSG_TYPE], None]]): A function to call with each new token if `stream` is `True`.
*   `split` (Optional[bool]): If `True`, the prompt will be split into messages based on `user_keyword` and `ai_keyword` (useful for discussion-like prompts).
*   `user_keyword`, `ai_keyword` (Optional[str]): Keywords used for splitting the prompt when `split` is `True`.
*   `**kwargs`: Additional arguments passed directly to the underlying binding's generation method.

**Returns:**
*   `str`: The generated text if successful.
*   `dict`: An error dictionary if the generation fails.

**Example: Basic Text Generation**
```python
generated_text = client.generate_text(
    prompt="Tell me a short story about a brave knight.",
    n_predict=100,
    temperature=0.7
)
print("\n--- Basic Text Generation ---")
print(generated_text)
```

**Example: Streaming Text Generation**
```python
from lollms_client.lollms_types import MSG_TYPE

def stream_callback(token: str, message_type: MSG_TYPE):
    print(token, end="", flush=True)

print("\n--- Streaming Text Generation ---")
client.generate_text(
    prompt="Describe the benefits of using a local LLM binding.",
    n_predict=200,
    stream=True,
    streaming_callback=stream_callback
)
print("\n--- End Streaming ---")
```

#### `generate_from_messages`

```python
generate_from_messages(
    messages: List[Dict],
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
    **kwargs
) -> Union[str, dict]
```

Generates text based on an OpenAI-compatible list of messages. This is ideal for structured conversational interactions.

**Parameters:**
*   `messages` (List[Dict]): A list of message dictionaries, each with a "role" ("system", "user", "assistant") and "content" field. Content can be a string or a list of content blocks for multimodal inputs.
*   Other parameters are similar to `generate_text`.

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.generate_from_messages(messages)
print("\n--- Generate from Messages ---")
print(response)

# Example with a follow-up
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "And what about Germany?"})
response_germany = client.generate_from_messages(messages)
print("\n--- Generate from Messages (Follow-up) ---")
print(response_germany)
```

### Conversational AI (`chat`)

```python
chat(
    discussion: LollmsDiscussion,
    branch_tip_id: Optional[str] = None,
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE, Dict], bool]] = None,
    **kwargs
) -> Union[str, dict]
```

This is the recommended high-level method for managing multi-turn conversations. It uses the `LollmsDiscussion` object (from `lollms_client.lollms_discussion`) to build the conversation context, automatically handling roles, message history, and branching.

**Parameters:**
*   `discussion` (`LollmsDiscussion`): An instance of `LollmsDiscussion` containing the conversation history.
*   `branch_tip_id` (Optional[str]): The ID of the message to use as the tip of the conversation branch. If `None`, the active branch is used.
*   Other parameters are similar to `generate_text` but apply to the chat context.

**Returns:**
*   `str`: The generated AI response message.
*   `dict`: An error dictionary if generation fails.

**Example (Requires `LollmsDiscussion`):**
```python
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_types import MSG_TYPE
import tempfile
import os

# Create a temporary database for the discussion
with tempfile.TemporaryDirectory() as tmpdir:
    discussion_db_path = Path(tmpdir) / "temp_discussion.db"
    discussion = LollmsDiscussion(db_path=discussion_db_path)
    discussion.create_new_discussion("My Test Discussion")

    def chat_stream_callback(token: str, message_type: MSG_TYPE, metadata: dict):
        if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(token, end="", flush=True)

    print("\n--- Chat with LollmsDiscussion ---")
    discussion.add_message("user", "Can you explain quantum entanglement simply?")
    print(f"{multimodal_client.user_full_header} Can you explain quantum entanglement simply?")
    
    response = multimodal_client.chat(discussion=discussion, streaming_callback=chat_stream_callback)
    discussion.add_message("assistant", response)
    
    print(f"\n{multimodal_client.ai_full_header} {response}") # Print final response if not streaming
    print("\n--- End Chat Turn 1 ---")

    discussion.add_message("user", "What are its practical implications?")
    print(f"{multimodal_client.user_full_header} What are its practical implications?")
    response_2 = multimodal_client.chat(discussion=discussion, streaming_callback=chat_stream_callback)
    discussion.add_message("assistant", response_2)

    print(f"\n{multimodal_client.ai_full_header} {response_2}") # Print final response if not streaming
    print("\n--- End Chat Turn 2 ---")
```

### Embeddings (`embed`)

```python
embed(text: Union[str, List[str]], **kwargs) -> List[float]
```

Generates vector embeddings for the input text using the LLM binding's embedding capabilities.

**Parameters:**
*   `text` (Union[str, List[str]]): The text or list of texts to embed.
*   `**kwargs`: Additional arguments specific to the binding's embed method.

**Returns:**
*   `List[float]`: A list of floats representing the embedding vector.

**Example:**
```python
embedding = client.embed("The quick brown fox jumps over the lazy dog.")
print("\n--- Embedding Example ---")
print(f"Embedding length: {len(embedding)}")
print(f"First 5 dimensions: {embedding[:5]}")
```

### `listMountedPersonalities`

```python
listMountedPersonalities() -> Union[List[Dict], Dict]
```

Specific to the `lollms` LLM binding. Lists personalities currently mounted in the LOLLMS server.

**Example:**
```python
# Assuming 'client' is initialized with llm_binding_name="lollms"
# mounted_personalities = client.listMountedPersonalities()
# print("\n--- Mounted Personalities (if using lollms binding) ---")
# print(mounted_personalities)
```

## Modality Bindings (TTS, TTI, STT, TTV, TTM)

`LollmsClient` provides properties for each initialized modality binding (e.g., `client.tts`, `client.tti`). Once a binding is successfully initialized, you can directly call its methods. The available methods and their parameters will depend on the specific binding loaded.

**General Usage Pattern:**

```python
# Check if a specific binding is available
if multimodal_client.tts:
    print(f"\nTTS Binding: {multimodal_client.tts.binding_name}")
    # Call methods available on the TTS binding object
    # For example, a Bark TTS binding might have:
    # audio_data = multimodal_client.tts.generate_audio(
    #     prompt="Hello, this is a test from LollmsClient.",
    #     voice="v2/en_speaker_9",
    #     filename="output.wav",
    #     output_path="./speech_output"
    # )
    # print(f"Generated audio: {audio_data}")
else:
    print("\nTTS binding not initialized.")

if multimodal_client.tti:
    print(f"\nTTI Binding: {multimodal_client.tti.binding_name}")
    # For example, a Diffusers TTI binding might have:
    # image_data = multimodal_client.tti.generate_image(
    #     prompt="A futuristic city skyline at sunset, cyberpunk style, highly detailed",
    #     width=512,
    #     height=512,
    #     filename="city_image.png",
    #     output_path="./image_output"
    # )
    # print(f"Generated image: {image_data}")
else:
    print("\nTTI binding not initialized.")

if multimodal_client.stt:
    print(f"\nSTT Binding: {multimodal_client.stt.binding_name}")
    # transcript = multimodal_client.stt.transcribe_audio("path/to/audio.wav")
    # print(f"Audio transcript: {transcript}")
else:
    print("\nSTT binding not initialized.")

# Similar patterns apply to TTV and TTM bindings.
```

To see the specific methods available for each binding, refer to the `__init__.py` file within their respective directories (e.g., `lollms_client/tts_bindings/bark/__init__.py`). Each binding adheres to its respective Abstract Base Class (ABC) defined in `lollms_client/lollms_tts_binding.py`, `lollms_client/lollms_tti_binding.py`, etc.

## Agentic Workflows

`LollmsClient` includes powerful methods for building sophisticated agentic systems that can reason, plan, and use tools.

### `generate_with_mcp` (Multi-tool Code Processor)

```python
generate_with_mcp(
    prompt: str,
    system_prompt: Optional[str] = None,
    objective_extraction_system_prompt: str = "Build a plan",
    images: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tool_calls: int = 5,
    max_llm_iterations: int = 10,
    ctx_size: Optional[int] = None,
    max_json_retries: int = 1,
    tool_call_decision_temperature: float = 0.0,
    final_answer_temperature: Optional[float] = None,
    streaming_callback: Optional[Callable[[str, int, Optional[Dict], Optional[List]], bool]] = None,
    debug: bool = False,
    **llm_generation_kwargs
) -> Dict[str, Any]
```

This method orchestrates a dynamic "observe-think-act" loop, enabling the AI to break down complex tasks, select and execute appropriate tools, and synthesize findings into a coherent final answer. It employs robust error handling and state management to prevent loops and ensure progress.

**Key Features:**
*   **Tool Discovery:** Automatically discovers tools from the active MCP binding.
*   **Context-Aware Asset Ingestion:** Detects code/images in the `context` and registers them as internal assets (with UUIDs) to avoid large data in prompts and prevent JSON errors.
*   **Tool Filtering:** Hides tools that directly consume code from the LLM's view, forcing the use of a safer `local_tools::generate_and_call` meta-tool for code execution.
*   **Knowledge Scratchpad:** Maintains a cumulative knowledge base (`knowledge_scratchpad`) updated after each tool call.
*   **State-Driven Execution:** The agent checks its `knowledge_scratchpad` against its `overall plan` to avoid redundant work.
*   **Internal Tools:** Provides built-in tools like `local_tools::generate_and_call` (for generating and executing code), `local_tools::refactor_scratchpad`, `local_tools::request_clarification`, and `local_tools::final_answer`.

**Parameters:**
*   `prompt` (str): The user's request for the agent to fulfill.
*   `system_prompt` (Optional[str]): System prompt for the final answer generation.
*   `objective_extraction_system_prompt` (str): System prompt used for initial plan building.
*   `images` (Optional[List[str]]): List of image base64 strings or paths relevant to the initial prompt.
*   `tools` (Optional[List[Dict[str, Any]]]): A pre-defined list of tools to use. If `None`, tools are discovered from the MCP binding.
*   `max_tool_calls` (int): Maximum number of tool calls allowed within a single `generate_with_mcp` execution.
*   `max_llm_iterations` (int): Maximum number of reasoning cycles (observe-think-act steps).
*   `ctx_size` (Optional[int]): Context size for LLM calls within the agent loop.
*   `max_json_retries` (int): How many times to retry JSON parsing if the LLM's output is invalid.
*   `tool_call_decision_temperature` (float): Temperature for the LLM's tool-calling decision-making.
*   `final_answer_temperature` (Optional[float]): Temperature for synthesizing the final answer.
*   `streaming_callback` (Optional[Callable[[str, int, Optional[Dict], Optional[List]], bool]]): Callback for real-time updates on agent steps, thoughts, and tool outputs.
*   `debug` (bool): If `True`, enables extensive logging of internal prompts and states.
*   `**llm_generation_kwargs`: Additional keyword arguments passed to internal LLM generation calls.

**Returns:**
*   `Dict[str, Any]`: A dictionary containing:
    *   `final_answer` (str): The comprehensive answer generated by the agent.
    *   `tool_calls` (List[Dict]): A list of all tool calls made during the process, including their names, parameters, and results.
    *   `sources` (List[Dict]): Not fully implemented in current version, but intended for data sources.
    *   `clarification_required` (bool): True if the agent decided it needed more information.
    *   `error` (Optional[str]): An error message if the process failed.

**Example (Requires `local_mcp` binding and default tools):**
To run this example, ensure you have the `local_mcp` binding enabled in your `LollmsClient` and that `internet_search` tool is available.

```python
# Assuming 'multimodal_client' is initialized with mcp_binding_name="local_mcp"
if multimodal_client.mcp:
    print("\n--- Generate with MCP (Agentic Workflow) ---")
    
    def mcp_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None, turn_history: Optional[List] = None):
        if message_type == MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n[MCP Step Start]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"[MCP Step End]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_INFO:
            print(f"[MCP Info]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_THOUGHT_CONTENT:
            print(f"[MCP Thought]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"[MCP Tool Call]: Tool '{metadata.get('name')}' with params {metadata.get('parameters')}")
        elif message_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"[MCP Tool Output]: {metadata}")
        elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(message, end="", flush=True)

    try:
        # Example: Use the internet_search tool via the agent
        # For this to work, the 'internet_search' tool must be active in your local_mcp binding
        # See lollms_client/mcp_bindings/local_mcp/default_tools/internet_search/internet_search.py
        
        # A simple query that might trigger the internet search tool
        agent_result = multimodal_client.generate_with_mcp(
            prompt="What is the current population of the United States?",
            streaming_callback=mcp_stream_callback,
            max_llm_iterations=5,
            debug=False # Set to True for verbose internal logging
        )

        print("\n--- MCP Agent Final Result ---")
        if agent_result.get("error"):
            print(f"Error: {agent_result['error']}")
        elif agent_result.get("clarification"):
            print(f"Clarification Required: {agent_result['final_answer']}")
        else:
            print(f"Final Answer: {agent_result['final_answer']}")
            print(f"Tool Calls Made: {len(agent_result['tool_calls'])}")
            for tc in agent_result['tool_calls']:
                print(f"  - {tc['name']}({tc['params']}) -> {tc['result'].get('status', 'N/A')}")

    except Exception as e:
        print(f"An error occurred during MCP generation: {e}")
else:
    print("\nMCP binding not initialized. Skipping generate_with_mcp example.")
```

### `generate_text_with_rag` (Retrieval Augmented Generation)

```python
generate_text_with_rag(
    prompt: str,
    rag_query_function: Callable[[str, Optional[str], int, float], List[Dict[str, Any]]],
    system_prompt: str = "",
    objective_extraction_system_prompt: str = "Extract objectives",
    rag_query_text: Optional[str] = None,
    rag_vectorizer_name: Optional[str] = None,
    rag_top_k: int = 5,
    rag_min_similarity_percent: float = 70.0,
    max_rag_hops: int = 3,
    images: Optional[List[str]] = None,
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    extract_objectives: bool = True,
    streaming_callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict], Optional[List]], bool]] = None,
    max_rag_context_characters: int = 32000,
    **llm_generation_kwargs
) -> Dict[str, Any]
```

This method facilitates multi-hop Retrieval Augmented Generation (RAG). It enables the AI to dynamically formulate queries, retrieve relevant information from a provided knowledge base (via `rag_query_function`), synthesize new knowledge into a scratchpad, and refine its objectives iteratively until it can provide a comprehensive answer.

**Parameters:**
*   `prompt` (str): The initial user query.
*   `rag_query_function` (Callable): A required function that the `LollmsClient` calls to retrieve documents. This function must accept `(query: str, vectorizer_name: Optional[str], top_k: int, min_similarity_percent: float)` and return `List[Dict[str, Any]]` where each dict contains at least `"chunk_text"`, `"file_path"`, and `"similarity_percent"`. This function connects to *your* external RAG system.
*   `system_prompt` (str): System prompt for the final answer synthesis.
*   `objective_extraction_system_prompt` (str): System prompt for extracting initial objectives.
*   `rag_query_text` (Optional[str]): If provided, this specific query is used for the *first* RAG hop, overriding the LLM's initial query generation.
*   `rag_vectorizer_name` (Optional[str]): Name of the vectorizer to use if your `rag_query_function` supports multiple.
*   `rag_top_k` (int): Number of top-N documents to retrieve per query.
*   `rag_min_similarity_percent` (float): Minimum similarity threshold for retrieved documents.
*   `max_rag_hops` (int): Maximum number of iterative query-retrieval-synthesis cycles.
*   `images` (Optional[List[str]]): Images to include in the final answer synthesis prompt.
*   `ctx_size`: Context size override for LLM calls during RAG.
*   `extract_objectives` (bool): If `True`, the LLM will first extract research objectives from the user prompt.
*   `streaming_callback` (Optional[Callable]): Callback for real-time updates on RAG steps, queries, and findings.
*   `max_rag_context_characters` (int): Maximum character length of raw retrieved context to include in the final prompt.
*   `**llm_generation_kwargs`: Additional arguments passed to internal LLM generation calls.

**Returns:**
*   `Dict[str, Any]`: A dictionary containing:
    *   `final_answer` (str): The synthesized answer.
    *   `rag_hops_history` (List[Dict]): Details of each RAG hop (query, retrieved chunks).
    *   `all_retrieved_sources` (List[Dict]): A flattened list of all unique chunks retrieved across all hops.
    *   `error` (Optional[str]): An error message if the process failed.

**Example (Conceptual - requires a RAG function implementation):**
```python
# THIS IS A MOCK RAG QUERY FUNCTION FOR DEMONSTRATION PURPOSES.
# In a real application, this would query a vector database (e.g., Chroma, FAISS, Milvus).
def mock_rag_query_function(query: str, vectorizer_name: Optional[str], top_k: int, min_similarity_percent: float) -> List[Dict[str, Any]]:
    print(f"Mock RAG: Querying for '{query}' (top {top_k}, min sim {min_similarity_percent}%)")
    # Simulate retrieving some relevant chunks
    if "python" in query.lower():
        return [
            {"chunk_text": "Python is a high-level, interpreted, general-purpose programming language.", "file_path": "docs/python_intro.txt", "similarity_percent": 95.0},
            {"chunk_text": "It supports multiple programming paradigms, including structured, object-oriented, and functional programming.", "file_path": "docs/python_features.txt", "similarity_percent": 88.0},
        ]
    elif "javascript" in query.lower():
        return [
            {"chunk_text": "JavaScript is a programming language that is one of the core technologies of the World Wide Web.", "file_path": "docs/js_intro.txt", "similarity_percent": 92.0},
            {"chunk_text": "It is often used for client-side web development to make interactive web pages.", "file_path": "docs/js_web.txt", "similarity_percent": 85.0},
        ]
    else:
        return []

# Assuming 'client' is initialized
# Set debug=True for verbose output of RAG steps
# Define a streaming callback if you want real-time updates
def rag_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None, turn_history: Optional[List] = None):
    if message_type in [MSG_TYPE.MSG_TYPE_STEP_START, MSG_TYPE.MSG_TYPE_STEP_END, MSG_TYPE.MSG_TYPE_INFO]:
        print(f"\n[RAG {message_type.name.split('_')[-1]}]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(message, end="", flush=True)

print("\n--- Generate Text with RAG ---")
try:
    rag_result = client.generate_text_with_rag(
        prompt="Tell me about the key features of Python programming language and its common uses.",
        rag_query_function=mock_rag_query_function,
        max_rag_hops=2,
        streaming_callback=rag_stream_callback,
        debug=False # Set to True to see detailed RAG prompts
    )

    print("\n--- RAG Final Result ---")
    if rag_result.get("error"):
        print(f"Error: {rag_result['error']}")
    else:
        print(f"Final Answer:\n{rag_result['final_answer']}")
        print(f"\nTotal Retrieved Sources: {len(rag_result['all_retrieved_sources'])}")
        # for source in rag_result['all_retrieved_sources']:
        #     print(f"  - {source.get('file_path')} (Sim: {source.get('similarity'):.1f}%)")

except Exception as e:
    print(f"An error occurred during RAG generation: {e}")
```

## Code & Structured Content Generation/Extraction

These methods are designed to facilitate AI interaction with code and structured data formats like JSON.

### `generate_code`

```python
generate_code(
    prompt: str,
    images: List[str] = [],
    system_prompt: Optional[str] = None,
    template: Optional[str] = None,
    language: str = "json",
    code_tag_format: str = "markdown", # or "html"
    n_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    callback: Optional[Callable] = None,
    debug: bool = False
) -> Optional[str]
```

Generates a single code block. It's particularly useful for enforcing specific code formats or structures using a `template`. It also attempts to handle cases where the model might generate an incomplete code block by requesting continuation.

**Parameters:**
*   `prompt` (str): The prompt instructing the LLM to generate code.
*   `images` (List[str]): Optional list of image paths for multimodal context.
*   `system_prompt` (Optional[str]): Additional system instructions for code generation.
*   `template` (Optional[str]): A string representing the desired structure or template for the code (e.g., a JSON schema, a Python function signature).
*   `language` (str): The programming language or format of the code (e.g., "python", "json", "javascript").
*   `code_tag_format` (str): The format of the code block tags ("markdown" or "html").
*   `n_predict`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`: LLM generation parameters.
*   `callback` (Optional[Callable]): Streaming callback function.
*   `debug` (bool): Enables verbose debugging output.

**Returns:**
*   `Optional[str]`: The content of the generated code block, or `None` if generation fails.

**Example:**
```python
print("\n--- Generate Python Code ---")
python_code_prompt = "Write a Python function to calculate the factorial of a number."
generated_python_code = client.generate_code(
    prompt=python_code_prompt,
    language="python",
    template="def factorial(n):\n    # implementation\n    pass",
    n_predict=150
)
if generated_python_code:
    print(generated_python_code)
else:
    print("Failed to generate Python code.")
```

### `generate_codes`

```python
generate_codes(
    prompt,
    images=[],
    template=None,
    language="json",
    code_tag_format="markdown", # or "html"
    n_predict = None,
    temperature = None,
    top_k = None,
    top_p=None,
    repeat_penalty=None,
    repeat_last_n=None,
    callback=None,
    debug=False
) -> List[Dict]
```

Generates one or more code blocks based on a prompt. This is an older method; for single, structured code blocks, `generate_code` is often preferred as it handles continuation.

**Parameters:** Same as `generate_code`.

**Returns:**
*   `List[Dict]`: A list of dictionaries, where each dictionary represents a code block with keys like `content`, `type`, `file_name`, `is_complete`.

**Example:**
```python
print("\n--- Generate Multiple Code Blocks (basic) ---")
multi_code_prompt = "Give me a simple HTML page and a simple CSS stylesheet for it."
generated_blocks = client.generate_codes(
    prompt=multi_code_prompt,
    language="html", # This applies to the first language specifier
    n_predict=200
)
for block in generated_blocks:
    print(f"--- Code Block (Type: {block['type']}) ---")
    print(block['content'])
    print("---------------------------------------")
```

### `generate_structured_content`

```python
generate_structured_content(
    prompt: str,
    images: List[str] = [],
    schema: Union[Dict, str] = {},
    system_prompt: Optional[str] = None,
    **kwargs
) -> Optional[Dict]
```

Generates structured data (specifically a Python dictionary) by guiding the LLM to output JSON that conforms to a provided schema. This method is a high-level wrapper around `generate_code` tailored for JSON output.

**Parameters:**
*   `prompt` (str): The prompt for the LLM, instructing it to extract/generate information.
*   `images` (List[str]): Optional list of image paths for multimodal context.
*   `schema` (Union[Dict, str]): A Python dictionary or JSON string defining the desired output structure.
*   `system_prompt` (Optional[str]): Additional system instructions.
*   `**kwargs`: Additional keyword arguments passed directly to `generate_code` (e.g., `temperature`, `n_predict`, `debug`).

**Returns:**
*   `Optional[Dict]`: The parsed JSON data as a Python dictionary, or `None` if generation or parsing fails.

**Example:**
```python
print("\n--- Generate Structured Content (JSON) ---")
person_prompt = "Extract the name, age, and profession from the following text: 'Alice is a 30-year-old software engineer.'"
person_schema = {
    "name": "string",
    "age": "integer",
    "type": "string" # Using type instead of profession here, just for schema matching
}

extracted_data = client.generate_structured_content(
    prompt=person_prompt,
    schema=person_schema,
    n_predict=100,
    temperature=0.0 # Use low temperature for structured output
)
if extracted_data:
    print(extracted_data)
else:
    print("Failed to extract structured data.")

# Example with a list of objects
companies_prompt = "List 3 major tech companies, their founding year, and their primary product type."
companies_schema = {
    "companies": [
        {"name": "string", "founding_year": "integer", "product_type": "string"}
    ]
}

companies_data = client.generate_structured_content(
    prompt=companies_prompt,
    schema=companies_schema,
    n_predict=200,
    temperature=0.0
)
if companies_data:
    print(companies_data)
else:
    print("Failed to generate companies data.")
```

### Content Extraction Helpers

These utility methods help in parsing common AI output formats.

*   `extract_code_blocks(text: str, format: str = "markdown") -> List[dict]`:
    Extracts code blocks from a given text, supporting Markdown (triple backticks) and HTML (`<code>` tags). Each block is returned as a dictionary with `content`, `type` (language), `file_name`, and `is_complete` status.

*   `extract_thinking_blocks(text: str) -> List[str]`:
    Extracts content enclosed within `<thinking>...</thinking>` or `<think>...</think>` tags (case-insensitive) from a text. Useful for separating AI's internal monologue from its main response.

*   `remove_thinking_blocks(text: str) -> str`:
    Removes all occurrences of `<thinking>...</thinking>` or `<think>...</think>` blocks and their tags from a text.

**Example:**
```python
output_with_thinking = """
<thinking>
I need to first understand the user's intent. Then I will generate a simple Python script.
</thinking>
Hello! Here is your Python script:
```python
print("Hello from Python!")
```
And here's some more text.
<think>
This is another thought block.
</think>
"""

print("\n--- Content Extraction ---")
print(f"Original text:\n{output_with_thinking}")

thinking_parts = client.extract_thinking_blocks(output_with_thinking)
print(f"\nExtracted thinking parts: {thinking_parts}")

cleaned_text = client.remove_thinking_blocks(output_with_thinking)
print(f"\nText after removing thinking blocks:\n{cleaned_text}")

code_blocks = client.extract_code_blocks(output_with_thinking, format="markdown")
print(f"\nExtracted code blocks: {code_blocks}")
if code_blocks:
    print(f"Code content:\n{code_blocks[0]['content']}")
    print(f"Code type: {code_blocks[0]['type']}")
```

## Long Context Processing & Question Answering Utilities

### `long_context_processing`

```python
long_context_processing(
    text_to_process: str,
    contextual_prompt: Optional[str] = None,
    chunk_size_tokens: Optional[int] = None,
    overlap_tokens: int = 0,
    streaming_callback: Optional[Callable] = None,
    **kwargs
) -> str
```

Processes very long texts that exceed the LLM's context window. It breaks the text into overlapping chunks, processes (summarizes/extracts from) each chunk sequentially into a "scratchpad" memory, and then synthesizes a final comprehensive output from the accumulated scratchpad.

**Parameters:**
*   `text_to_process` (str): The long text to be processed.
*   `contextual_prompt` (Optional[str]): A specific instruction to guide the processing focus (e.g., "Summarize the text focusing on financial implications.").
*   `chunk_size_tokens` (Optional[int]): Target size of each chunk in tokens. Defaults to `ctx_size // 2`.
*   `overlap_tokens` (int): Number of tokens to overlap between chunks to maintain context. Defaults to 0.
*   `streaming_callback` (Optional[Callable]): Callback for real-time progress updates.
*   `**kwargs`: Additional LLM generation parameters.

**Returns:**
*   `str`: The final processed output (e.g., comprehensive summary).

**Example:**
```python
long_text = """
Chapter 1: The Beginning. In a small village nestled by the Whispering Woods, Elara, a young apprentice alchemist, began her journey. She dreamed of discovering the legendary Sunstone, a relic said to bring eternal prosperity. Her master, a wise old man named Kael, taught her the ancient arts, but warned her of the dangers that lurked in the deeper parts of the woods.

Chapter 2: The First Clue. One day, while foraging for rare herbs, Elara stumbled upon an ancient scroll. It detailed a riddle, hinting at the Sunstone's location being near 'the tears of the mountain'. Kael, upon inspecting the scroll, recognized the calligraphy as belonging to the forgotten Silver Elves, a race known for their hidden treasures. He cautioned Elara that such a quest would require immense courage and knowledge of the arcane.

Chapter 3: Trials of the Forest. Elara ventured deeper. The Whispering Woods lived up to its name, with strange sounds and shifting paths. She faced trials of cunning and bravery, outsmarting mischievous sprites and navigating treacherous bogs. Her alchemical knowledge proved invaluable, as she concocted potions to ward off illusions and reveal hidden paths. She learned that courage wasn't just about fighting, but about facing fears.

Chapter 4: The Mountain's Tears. Following the scroll, Elara arrived at a towering peak where a waterfall cascaded down, forming a glistening pool. These were the 'tears of the mountain'. At the base, a shimmering aura pulsed from behind the waterfall. It was here, within a hidden grotto, that she found the Sunstone, radiating a gentle warmth.

Chapter 5: Return and Reflection. With the Sunstone in hand, Elara returned to her village, greeted as a hero. The Sunstone's warmth filled the village, bringing bountiful harvests and dispelling shadows. But for Elara, the true treasure wasn't just the Sunstone, but the knowledge, courage, and self-discovery she gained on her perilous journey. She understood that prosperity wasn't just about wealth, but about growth and resilience.
"""
print("\n--- Long Context Processing (Summarization) ---")

def lcp_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None):
    if message_type == MSG_TYPE.MSG_TYPE_STEP_START:
        print(f"\n[LCP Step Start]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_STEP_END:
        print(f"[LCP Step End]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(message, end="", flush=True) # Print summarized chunks

summarized_text = client.long_context_processing(
    text_to_process=long_text,
    contextual_prompt="Summarize the entire story, focusing on Elara's journey and her discoveries.",
    chunk_size_tokens=250, # Adjust based on your model's ctx_size and target output
    overlap_tokens=50,
    streaming_callback=lcp_stream_callback
)
print("\n--- Final Long Context Summary ---")
print(summarized_text)
```

### `chunk_text`

```python
chunk_text(
    text: str,
    tokenizer: Callable,
    detokenizer: Callable,
    chunk_size: int,
    overlap: int,
    use_separators: bool = True
) -> List[str]
```

A helper function (defined outside the class but used internally by some methods) that divides a given text into token-based chunks with optional overlap and attempts to align chunk breaks with natural language separators (paragraphs, sentences).

**Parameters:**
*   `text` (str): The text to be chunked.
*   `tokenizer` (Callable): A function (like `client.tokenize`) to convert text to tokens.
*   `detokenizer` (Callable): A function (like `client.detokenize`) to convert tokens back to text.
*   `chunk_size` (int): The desired number of tokens per chunk.
*   `overlap` (int): The number of tokens to overlap between consecutive chunks.
*   `use_separators` (bool): If `True`, the function tries to break chunks at natural separators (e.g., paragraph breaks).

**Returns:**
*   `List[str]`: A list of text strings, each representing a chunk.

**Example (Direct use):**
```python
sample_text = "This is the first sentence. This is the second sentence, and it's a bit longer. Finally, the third sentence. This is the last bit of text."
# Use the client's tokenizer/detokenizer
chunks = chunk_text(
    text=sample_text,
    tokenizer=client.tokenize,
    detokenizer=client.detokenize,
    chunk_size=10, # Very small chunk size for demonstration
    overlap=2,
    use_separators=True
)
print("\n--- Chunked Text Example ---")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (Tokens: {client.count_tokens(chunk)}): {chunk}")
```

### `deep_analyze`

```python
deep_analyze(
    query: str,
    text: str = None,
    files: Optional[List[Union[str, Path]]] = None,
    aggregation_prompt: str = "Aggregate the findings from the memory into a coherent answer to the original query.",
    output_format: str = "markdown",
    ctx_size: int = None,
    chunk_size: int = None,
    overlap: int = None,
    bootstrap_chunk_size: int = None,
    bootstrap_steps: int = None,
    callback=None,
    debug: bool = False
) -> str
```

Performs an in-depth analysis or search for information related to a `query` across long `text` or multiple `files`. It processes the content chunk by chunk, building and updating a cumulative "memory" of findings, and then aggregates this memory into a final answer. It leverages the `docling` library for file conversion if `files` are provided.

**Parameters:**
*   `query` (str): The specific question or task for the analysis.
*   `text` (str): A single long string to analyze.
*   `files` (Optional[List[Union[str, Path]]]): A list of file paths (e.g., .txt, .pdf, .docx) to analyze. Requires `docling` to be installed.
*   `aggregation_prompt` (str): Instructions for the final aggregation step.
*   `output_format` (str): Desired format for the final output (e.g., "markdown").
*   `ctx_size`, `chunk_size`, `overlap`, `bootstrap_chunk_size`, `bootstrap_steps`: Parameters controlling chunking and processing.
*   `callback` (Optional[Callable]): Streaming callback for real-time updates.
*   `debug` (bool): Enables verbose debugging.

**Returns:**
*   `str`: The final aggregated answer.

**Example (requires `docling` to process files or provide a long string):**
```python
# Create a dummy text file for demonstration
dummy_file_path = Path("./temp_report.txt")
dummy_file_path.write_text("""
This is a comprehensive report on the economic impact of renewable energy.
Section 1: Solar Power. Solar energy has seen rapid growth due to technological advancements.
It contributes significantly to job creation in manufacturing and installation.
Section 2: Wind Energy. Wind farms are becoming more efficient.
Environmental benefits include reduced carbon emissions.
Economic benefits include local investments and energy independence.
Section 3: Hydropower. Hydropower is a stable source but has environmental considerations.
The overall conclusion is that renewables are crucial for future sustainability.
""")

print("\n--- Deep Analyze (from file) ---")

def deep_analyze_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None):
    if message_type in [MSG_TYPE.MSG_TYPE_STEP_START, MSG_TYPE.MSG_TYPE_STEP_END, MSG_TYPE.MSG_TYPE_INFO]:
        print(f"\n[Deep Analyze {message_type.name.split('_')[-1]}]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(message, end="", flush=True) # Print chunk processing updates
    elif message_type == MSG_TYPE.MSG_TYPE_STEP_PROGRESS:
        print(f"\r[Progress]: {metadata.get('progress'):.0f}% ", end="", flush=True)

try:
    analysis_result = client.deep_analyze(
        query="What are the economic benefits of renewable energy according to the document?",
        files=[dummy_file_path],
        chunk_size=100, # Small chunk for quick demo
        overlap=20,
        streaming_callback=deep_analyze_stream_callback,
        debug=False # Set to True for verbose prompts
    )
    print("\n--- Final Deep Analysis Output ---")
    print(analysis_result)

except ImportError:
    print("\nSkipping deep_analyze example: `docling` library not found. Install with `pip install docling`.")
except Exception as e:
    print(f"\nAn error occurred during deep_analyze: {e}")
finally:
    if dummy_file_path.exists():
        dummy_file_path.unlink()
```

### `sequential_summarize`

```python
sequential_summarize(
    text: str,
    chunk_processing_prompt: str = "Extract relevant information from the current text chunk and update the memory if needed.",
    chunk_processing_output_format: str = "markdown",
    final_memory_processing_prompt: str = "Create final summary using this memory.",
    final_output_format: str = "markdown",
    ctx_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    bootstrap_chunk_size: Optional[int] = None,
    bootstrap_steps: Optional[int] = None,
    callback: Optional[Callable] = None,
    debug: bool = False
) -> str
```

Processes text in chunks sequentially, updating a running "memory" at each step, and then produces a final summary from the accumulated memory. This is suitable for comprehensive summarization of very long documents, similar to `long_context_processing` but with a more explicit "memory" updating prompt.

**Parameters:**
*   `text` (str): The long text to summarize.
*   `chunk_processing_prompt` (str): Instructions for processing each chunk.
*   `chunk_processing_output_format` (str): Format for the memory output (e.g., "markdown").
*   `final_memory_processing_prompt` (str): Instructions for the final summarization from the memory.
*   `final_output_format` (str): Desired format for the final summary.
*   `ctx_size`, `chunk_size`, `overlap`, `bootstrap_chunk_size`, `bootstrap_steps`: Parameters controlling chunking and processing.
*   `callback` (Optional[Callable]): Streaming callback.
*   `debug` (bool): Enables verbose debugging.

**Returns:**
*   `str`: The final generated summary.

**Example:**
```python
# Using the same `long_text` from `long_context_processing` example
print("\n--- Sequential Summarize ---")

def seq_summarize_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None):
    if message_type == MSG_TYPE.MSG_TYPE_STEP:
        print(f"\n[Seq Summarize Step]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(message, end="", flush=True)

summarized_seq_text = client.sequential_summarize(
    text=long_text,
    chunk_processing_prompt="Extract and summarize key plot points and character developments from this chunk, adding them to the memory.",
    final_memory_processing_prompt="Generate a detailed story synopsis from the accumulated memory.",
    chunk_size=200, # Small chunks for quick demo
    overlap=40,
    streaming_callback=seq_summarize_stream_callback,
    debug=False
)
print("\n--- Final Sequential Summary ---")
print(summarized_seq_text)
```

### Question Answering Utilities

These methods provide structured ways to get specific types of answers from the LLM.

#### `yes_no`

```python
yes_no(
    question: str,
    context: str = "",
    max_answer_length: Optional[int] = None,
    conditionning: str = "",
    return_explanation: bool = False,
    callback: Optional[Callable] = None
) -> Union[bool, Dict]
```

Answers a yes/no question based on a given context, ensuring a boolean output and optionally an explanation.

**Parameters:**
*   `question` (str): The question to answer.
*   `context` (str): Relevant context to consider for the answer.
*   `max_answer_length` (Optional[int]): Maximum tokens for the LLM's response.
*   `conditionning` (str): Additional system-level instructions.
*   `return_explanation` (bool): If `True`, returns a dictionary `{"answer": bool, "explanation": str}`. Otherwise, returns just the boolean.
*   `callback` (Optional[Callable]): Streaming callback.

**Returns:**
*   `bool`: The boolean answer (`True` for yes, `False` for no).
*   `Dict`: `{"answer": bool, "explanation": str}` if `return_explanation` is `True`.

**Example:**
```python
print("\n--- Yes/No Question ---")
answer_with_explanation = client.yes_no(
    question="Is the sky blue?",
    context="The sky is visibly azure during clear weather.",
    return_explanation=True
)
print(answer_with_explanation)

answer_no_explanation = client.yes_no(
    question="Is a cat a type of bird?",
    context="Cats are furry mammals known for purring.",
    return_explanation=False
)
print(f"Is a cat a bird? {answer_no_explanation}")
```

#### `multichoice_question`

```python
multichoice_question(
    question: str,
    possible_answers: list,
    context: str = "",
    max_answer_length: Optional[int] = None,
    conditionning: str = "",
    return_explanation: bool = False,
    callback: Optional[Callable] = None
) -> Union[int, Dict]
```

Answers a multiple-choice question by selecting the best option from a list, returning the index of the chosen answer and optionally an explanation.

**Parameters:**
*   `question` (str): The multiple-choice question.
*   `possible_answers` (list): A list of strings, each representing a possible answer choice.
*   Other parameters are similar to `yes_no`.

**Returns:**
*   `int`: The 0-based index of the chosen answer. Returns -1 on failure.
*   `Dict`: `{"index": int, "explanation": str}` if `return_explanation` is `True`.

**Example:**
```python
print("\n--- Multichoice Question ---")
mc_question = "Which of these is a programming language?"
choices = ["Gold", "Python", "Jupiter", "Tea"]
selected_index = client.multichoice_question(
    question=mc_question,
    possible_answers=choices,
    context="Python is widely used in AI.",
    return_explanation=True
)
print(f"Question: {mc_question}")
print(f"Choices: {choices}")
print(f"Selected: {choices[selected_index['index']] if selected_index['index'] != -1 else 'N/A'} (Index: {selected_index['index']})")
print(f"Explanation: {selected_index.get('explanation', 'None')}")
```

#### `multichoice_ranking`

```python
multichoice_ranking(
    question: str,
    possible_answers: list,
    context: str = "",
    max_answer_length: Optional[int] = None,
    conditionning: str = "",
    return_explanation: bool = False,
    callback: Optional[Callable] = None
) -> Dict
```

Ranks a list of possible answers for a given question from best to worst, returning a list of indices in ranked order and optionally explanations for the ranking.

**Parameters:**
*   `question` (str): The question for which to rank answers.
*   `possible_answers` (list): A list of strings, each a candidate answer to be ranked.
*   Other parameters are similar to `yes_no`.

**Returns:**
*   `Dict`: `{"ranking": List[int], "explanations": List[str]}`. `ranking` is a list of 0-based indices representing the ranked order. `explanations` is an optional list of strings explaining each ranking.

**Example:**
```python
print("\n--- Multichoice Ranking ---")
ranking_question = "Rank the following fruits by their common sweetness, from sweetest to least sweet."
fruit_choices = ["Lemon", "Apple", "Banana", "Strawberry"]
ranked_fruits_info = client.multichoice_ranking(
    question=ranking_question,
    possible_answers=fruit_choices,
    return_explanation=True
)
print(f"Question: {ranking_question}")
print(f"Choices: {fruit_choices}")
print(f"Ranked Indices: {ranked_fruits_info['ranking']}")
ranked_names = [fruit_choices[idx] for idx in ranked_fruits_info['ranking']]
print(f"Ranked Fruits: {ranked_names}")
print(f"Explanations: {ranked_fruits_info.get('explanations', 'N/A')}")
```

## Further Examples and Resources

For more detailed and diverse examples, please refer to the `examples/` directory in the `lollms_client` repository. It contains scripts demonstrating various use cases, including:

*   `examples/console_discussion/console_app.py`: A console-based chat application.
*   `examples/deep_analyze/deep_analyse.py`: Demonstrates deep analysis of a single file.
*   `examples/generate_and_speak/generate_and_speak.py`: Combines text generation with speech output.
*   `examples/lollms_chat/test_openai_compatible_with_lollms_chat.py`: Shows `generate_from_messages` with an OpenAI-compatible interface.
*   `examples/internet_search_with_rag.py`: An example of RAG using an internet search tool.
*   `examples/simple_text_gen_test.py`: Basic text generation.
*   `examples/text_2_audio.py`, `text_2_image.py`, etc.: Examples of using modality bindings.
```

---

**Concluding Summary:**

The `LollmsCore` module, encapsulated by the `LollmsClient` class, provides a robust and flexible framework for interacting with various AI modalities within the LOLLMS ecosystem. This documentation details its initialization, core LLM functionalities (text generation, chat, embeddings), and advanced agentic capabilities like Multi-tool Code Processing (MCP) and Retrieval Augmented Generation (RAG). It also covers utilities for structured content generation, text analysis, and question answering.

The provided examples demonstrate how to leverage `LollmsClient` for diverse AI tasks. Users should ensure their desired LLM and modality bindings are correctly installed and configured (e.g., via their `description.yaml` files and any necessary environment variables or API keys) for the `LollmsClient` to function correctly. Some examples require additional Python libraries (e.g., `docling` for file processing), which can be installed via `pip`.```python
# docs/md/lollms_core.md
# LollmsCore Documentation

The `LollmsClient` class, found within `lollms_client/lollms_core.py`, is the central entry point for interacting with various Large Language Model (LLM) and multimodal services provided by the LOLLMS ecosystem. It acts as a unified interface to manage and leverage different binding implementations for text generation, text-to-speech (TTS), text-to-image (TTI), speech-to-text (STT), text-to-video (TTV), text-to-music (TTM), and Multi-tool Code Processors (MCP).

This documentation provides a comprehensive guide to initializing the client, interacting with different modalities, and utilizing advanced agentic features like RAG and MCP-driven workflows.

## `LollmsClient` Class

### Overview

The `LollmsClient` simplifies access to powerful AI capabilities by abstracting away the complexities of direct API or local model interactions. It supports a plug-and-play architecture for different "bindings" (e.g., Llama.cpp, Ollama, OpenAI, Bark, Diffusers, etc.), allowing users to seamlessly switch between providers and models without changing their core application logic.

### Initialization (`__init__`)

The `LollmsClient` constructor is highly configurable, allowing you to specify which bindings to use for each modality and provide specific configurations for them.

```python
from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE
from pathlib import Path
import os
import tempfile
import json
import re

# Example 1: Basic LLM-only client using the default 'lollms' binding
# Assumes lollms-webui is running on localhost:9600 and has a model loaded
client = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={
        "host_address": "http://localhost:9600",
        "model_name": "lollms_vllm_4bit", # Or whatever model is loaded in your webui
        "ctx_size": 4096,
        "n_predict": 1024,
        "temperature": 0.7,
        "repeat_penalty": 1.1,
    }
)
print(f"LLM client initialized with model: {client.get_model_name()}")

# Example 2: Client with LLM, TTS (Bark), and TTI (Diffusers) bindings
# Ensure you have 'bark' and 'diffusers' bindings available in their respective folders
# and potentially their dependencies installed (e.g., pip install transformers diffusers accelerate)
tts_config = {
    "output_path": "./speech_output",
    "voice": "v2/en_speaker_9", # Example voice
    # "device": "cpu" # Uncomment for CPU, default is often cuda if available
}
tti_config = {
    "output_path": "./image_output",
    "model_name": "stabilityai/stable-diffusion-v1-5",
    # "device": "cpu" # Uncomment for CPU, default is often cuda if available
}
mcp_config = {
    "host_address": "http://localhost:9600", # Or the host for your MCP
}

# Ensure output directories exist for examples
os.makedirs(tts_config["output_path"], exist_ok=True)
os.makedirs(tti_config["output_path"], exist_ok=True)

multimodal_client = LollmsClient(
    llm_binding_name="lollms",
    llm_binding_config={"host_address": "http://localhost:9600", "model_name": "lollms_vllm_4bit"},
    tts_binding_name="bark",
    tts_binding_config=tts_config,
    tti_binding_name="diffusers",
    tti_binding_config=tti_config,
    mcp_binding_name="local_mcp", # Use the local MCP binding
    mcp_binding_config=mcp_config,
    user_name="John",
    ai_name="Lolly"
)
print(f"Multimodal client initialized. LLM: {multimodal_client.get_model_name()}, TTS: {multimodal_client.tts.binding_name if multimodal_client.tts else 'None'}, TTI: {multimodal_client.tti.binding_name if multimodal_client.tti else 'None'}")
```

**Parameters:**

*   `llm_binding_name` (str, default: "lollms"): The name of the LLM binding to use (e.g., "lollms", "ollama", "openai").
*   `tts_binding_name` (Optional[str]): The name of the Text-to-Speech binding (e.g., "bark", "piper_tts").
*   `tti_binding_name` (Optional[str]): The name of the Text-to-Image binding (e.g., "dalle", "diffusers").
*   `stt_binding_name` (Optional[str]): The name of the Speech-to-Text binding (e.g., "whisper", "whispercpp").
*   `ttv_binding_name` (Optional[str]): The name of the Text-to-Video binding.
*   `ttm_binding_name` (Optional[str]): The name of the Text-to-Music binding (e.g., "audiocraft", "bark").
*   `mcp_binding_name` (Optional[str]): The name of the Multi-tool Code Processor binding (e.g., "local_mcp", "remote_mcp").
*   `llm_bindings_dir`, `tts_bindings_dir`, `tti_bindings_dir`, etc. (Path): Directories where the respective binding implementations are located. Defaults to subdirectories within `lollms_client/`.
*   `llm_binding_config`, `tts_binding_config`, etc. (Optional[Dict[str, Any]]): Dictionaries containing binding-specific configurations (e.g., `host_address`, `model_name`, `api_key`, `device`, `output_path`).
*   `user_name` (str, default: "user"): The name used for the user in prompt formatting.
*   `ai_name` (str, default: "assistant"): The name used for the AI in prompt formatting.
*   `**kwargs`: Additional keyword arguments passed directly to the LLM binding's initialization. These can include LLM generation parameters like `ctx_size`, `n_predict`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads`, `stream`, `streaming_callback`, `model_name`. Note that these are primarily for setting *default* values for generation.

**Raises:**
*   `ValueError`: If the primary `llm_binding_name` cannot be created.
*   `Warning`: If any optional modality binding fails to initialize.

### Prompt Formatting Properties

`LollmsClient` provides properties to easily access formatted headers for building prompts, ensuring consistency with the chosen prompt template (which is internally managed by the LLM binding).

*   `system_full_header`: Returns the full system header (e.g., `!@>system:`).
*   `system_custom_header(ai_name)`: Returns a custom system header (e.g., `!@>MyAI:`).
*   `user_full_header`: Returns the full user header (e.g., `!@>user:`).
*   `user_custom_header(user_name)`: Returns a custom user header (e.g., `!@>John:`).
*   `ai_full_header`: Returns the full AI header (e.g., `!@>assistant:`).
*   `ai_custom_header(ai_name)`: Returns a custom AI header (e.g., `!@>Lolly:`).

**Example:**
```python
print(f"System Header: {client.system_full_header}")
print(f"User Header: {client.user_full_header}")
print(f"AI Header: {client.ai_full_header}")
print(f"Custom AI Header: {multimodal_client.ai_custom_header('Lolly')}")
```

### Binding Management

You can dynamically update the active binding for any modality after initialization.

*   `update_llm_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active LLM binding.
*   `update_tts_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTS binding.
*   `update_tti_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTI binding.
*   `update_stt_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active STT binding.
*   `update_ttv_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTV binding.
*   `update_ttm_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active TTM binding.
*   `update_mcp_binding(binding_name: str, config: Optional[Dict[str, Any]] = None)`: Updates the active MCP binding.

**Utility Methods:**

*   `get_ctx_size(model_name: Optional[str] = None) -> Optional[int]`: Returns the context size of the currently loaded model or a specified model.
*   `get_model_name() -> Optional[str]`: Returns the name of the currently loaded LLM model.
*   `set_model_name(model_name: str) -> bool`: Sets the model name for the current LLM binding. Note: This typically triggers a model load operation within the binding.

**Example:**
```python
# Assuming 'client' from Example 1 is active
print(f"Current LLM model: {client.get_model_name()}")
print(f"Current context size: {client.get_ctx_size()}")

# Example of updating LLM binding (e.g., switching to Ollama)
# try:
#     client.update_llm_binding(
#         binding_name="ollama",
#         config={"host_address": "http://localhost:11434", "model_name": "llama3"}
#     )
#     print(f"LLM binding updated to: {client.binding.binding_name}, model: {client.get_model_name()}")
# except ValueError as e:
#     print(f"Error updating LLM binding: {e}")
```

## Core LLM Interaction Methods

These methods provide the primary interface for text generation and related LLM functionalities.

### Tokenization and Model Information

*   `tokenize(text: str) -> list`: Tokenizes the input text using the active LLM binding's tokenizer.
*   `detokenize(tokens: list) -> str`: Detokenizes a list of tokens back into text.
*   `count_tokens(text: str) -> int`: Counts the number of tokens in a given text.
*   `count_image_tokens(image: str) -> int`: Estimates the number of tokens an image would consume in a multi-modal context (e.g., base64 encoded image).
*   `get_model_details() -> dict`: Retrieves detailed information about the currently loaded LLM model.
*   `listModels() -> List[Dict]`: Lists all models available to the active LLM binding.
*   `switch_model(model_name: str) -> bool`: Attempts to load a different model within the active LLM binding.

**Example:**
```python
text_to_process = "Hello, world! This is a test."
tokens = client.tokenize(text_to_process)
print(f"Text: '{text_to_process}'")
print(f"Tokens: {tokens}")
print(f"Detokenized: '{client.detokenize(tokens)}'")
print(f"Token count: {client.count_tokens(text_to_process)}")

# print(f"Available LLM models for '{client.binding.binding_name}': {client.listModels()}")
# client.switch_model("another_model_name") # Uncomment to test model switching
```

### Text Generation

#### `generate_text`

```python
generate_text(
    prompt: str,
    images: Optional[List[str]] = None,
    system_prompt: str = "",
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
    split: Optional[bool] = False,
    user_keyword: Optional[str] = "!@>user:",
    ai_keyword: Optional[str] = "!@>assistant:",
    **kwargs
) -> Union[str, dict]
```

Generates text based on a given prompt. This is a fundamental method that allows for flexible text generation, including multi-modal inputs (images) and control over various generation parameters. Parameters not explicitly provided will fall back to the defaults set during `LollmsClient` initialization or by the binding itself.

**Parameters:**
*   `prompt` (str): The main input prompt for the LLM.
*   `images` (Optional[List[str]]): A list of image paths (or base64 strings) for multi-modal generation.
*   `system_prompt` (str): An initial instruction or context for the AI.
*   `n_predict` (Optional[int]): Maximum number of tokens to generate.
*   `stream` (Optional[bool]): If `True`, the output will be streamed token by token to the `streaming_callback`.
*   `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `n_threads`: Standard LLM generation parameters.
*   `ctx_size` (Optional[int]): Overrides the default context size for this specific generation.
*   `streaming_callback` (Optional[Callable[[str, MSG_TYPE], None]]): A function to call with each new token if `stream` is `True`.
*   `split` (Optional[bool]): If `True`, the prompt will be split into messages based on `user_keyword` and `ai_keyword` (useful for discussion-like prompts).
*   `user_keyword`, `ai_keyword` (Optional[str]): Keywords used for splitting the prompt when `split` is `True`.
*   `**kwargs`: Additional arguments passed directly to the underlying binding's generation method.

**Returns:**
*   `str`: The generated text if successful.
*   `dict`: An error dictionary if the generation fails.

**Example: Basic Text Generation**
```python
generated_text = client.generate_text(
    prompt="Tell me a short story about a brave knight.",
    n_predict=100,
    temperature=0.7
)
print("\n--- Basic Text Generation ---")
print(generated_text)
```

**Example: Streaming Text Generation**
```python
def stream_callback(token: str, message_type: MSG_TYPE):
    if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(token, end="", flush=True)

print("\n--- Streaming Text Generation ---")
client.generate_text(
    prompt="Describe the benefits of using a local LLM binding.",
    n_predict=200,
    stream=True,
    streaming_callback=stream_callback
)
print("\n--- End Streaming ---")
```

#### `generate_from_messages`

```python
generate_from_messages(
    messages: List[Dict],
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
    **kwargs
) -> Union[str, dict]
```

Generates text based on an OpenAI-compatible list of messages. This is ideal for structured conversational interactions.

**Parameters:**
*   `messages` (List[Dict]): A list of message dictionaries, each with a "role" ("system", "user", "assistant") and "content" field. Content can be a string or a list of content blocks for multimodal inputs.
*   Other parameters are similar to `generate_text`.

**Returns:**
*   `str`: The generated text.
*   `dict`: An error dictionary if the generation fails.

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.generate_from_messages(messages)
print("\n--- Generate from Messages ---")
print(response)

# Example with a follow-up
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "And what about Germany?"})
response_germany = client.generate_from_messages(messages)
print("\n--- Generate from Messages (Follow-up) ---")
print(response_germany)
```

### Conversational AI (`chat`)

```python
chat(
    discussion: LollmsDiscussion,
    branch_tip_id: Optional[str] = None,
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    streaming_callback: Optional[Callable[[str, MSG_TYPE, Dict], bool]] = None,
    **kwargs
) -> Union[str, dict]
```

This is the recommended high-level method for managing multi-turn conversations. It uses the `LollmsDiscussion` object (from `lollms_client.lollms_discussion`) to build the conversation context, automatically handling roles, message history, and branching.

**Parameters:**
*   `discussion` (`LollmsDiscussion`): An instance of `LollmsDiscussion` containing the conversation history.
*   `branch_tip_id` (Optional[str]): The ID of the message to use as the tip of the conversation branch. If `None`, the active branch is used.
*   Other parameters are similar to `generate_text` but apply to the chat context.

**Returns:**
*   `str`: The generated AI response message.
*   `dict`: An error dictionary if generation fails.

**Example (Requires `LollmsDiscussion`):**
```python
from lollms_client.lollms_discussion import LollmsDiscussion

# Create a temporary database for the discussion
with tempfile.TemporaryDirectory() as tmpdir:
    discussion_db_path = Path(tmpdir) / "temp_discussion.db"
    discussion = LollmsDiscussion(db_path=discussion_db_path)
    discussion.create_new_discussion("My Test Discussion")

    def chat_stream_callback(token: str, message_type: MSG_TYPE, metadata: dict):
        if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(token, end="", flush=True)

    print("\n--- Chat with LollmsDiscussion ---")
    discussion.add_message("user", "Can you explain quantum entanglement simply?")
    print(f"{multimodal_client.user_full_header} Can you explain quantum entanglement simply?")
    
    response = multimodal_client.chat(discussion=discussion, streaming_callback=chat_stream_callback)
    discussion.add_message("assistant", response)
    
    print(f"\n{multimodal_client.ai_full_header} {response}") # Print final response if not streaming
    print("\n--- End Chat Turn 1 ---")

    discussion.add_message("user", "What are its practical implications?")
    print(f"{multimodal_client.user_full_header} What are its practical implications?")
    response_2 = multimodal_client.chat(discussion=discussion, streaming_callback=chat_stream_callback)
    discussion.add_message("assistant", response_2)

    print(f"\n{multimodal_client.ai_full_header} {response_2}") # Print final response if not streaming
    print("\n--- End Chat Turn 2 ---")
```

### Embeddings (`embed`)

```python
embed(text: Union[str, List[str]], **kwargs) -> List[float]
```

Generates vector embeddings for the input text using the LLM binding's embedding capabilities.

**Parameters:**
*   `text` (Union[str, List[str]]): The text or list of texts to embed.
*   `**kwargs`: Additional arguments specific to the binding's embed method.

**Returns:**
*   `List[float]`: A list of floats representing the embedding vector.

**Example:**
```python
embedding = client.embed("The quick brown fox jumps over the lazy dog.")
print("\n--- Embedding Example ---")
print(f"Embedding length: {len(embedding)}")
print(f"First 5 dimensions: {embedding[:5]}")
```

### `listMountedPersonalities`

```python
listMountedPersonalities() -> Union[List[Dict], Dict]
```

Specific to the `lollms` LLM binding. Lists personalities currently mounted in the LOLLMS server.

**Example:**
```python
# Assuming 'client' is initialized with llm_binding_name="lollms"
# mounted_personalities = client.listMountedPersonalities()
# print("\n--- Mounted Personalities (if using lollms binding) ---")
# print(mounted_personalities)
```

## Modality Bindings (TTS, TTI, STT, TTV, TTM)

`LollmsClient` provides properties for each initialized modality binding (e.g., `client.tts`, `client.tti`). Once a binding is successfully initialized, you can directly call its methods. The available methods and their parameters will depend on the specific binding loaded.

**General Usage Pattern:**

```python
# Check if a specific binding is available
if multimodal_client.tts:
    print(f"\nTTS Binding: {multimodal_client.tts.binding_name}")
    # Call methods available on the TTS binding object
    # For example, a Bark TTS binding might have:
    try:
        audio_data_path = multimodal_client.tts.generate_audio(
            prompt="Hello, this is a test from LollmsClient.",
            voice="v2/en_speaker_9",
            filename="output.wav",
            output_path="./speech_output"
        )
        print(f"Generated audio saved to: {audio_data_path}")
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
else:
    print("\nTTS binding not initialized.")

if multimodal_client.tti:
    print(f"\nTTI Binding: {multimodal_client.tti.binding_name}")
    # For example, a Diffusers TTI binding might have:
    try:
        image_data_path = multimodal_client.tti.generate_image(
            prompt="A futuristic city skyline at sunset, cyberpunk style, highly detailed",
            width=512,
            height=512,
            filename="city_image.png",
            output_path="./image_output"
        )
        print(f"Generated image saved to: {image_data_path}")
    except Exception as e:
        print(f"Error generating TTI image: {e}")
else:
    print("\nTTI binding not initialized.")

if multimodal_client.stt:
    print(f"\nSTT Binding: {multimodal_client.stt.binding_name}")
    # transcript = multimodal_client.stt.transcribe_audio("path/to/audio.wav")
    # print(f"Audio transcript: {transcript}")
else:
    print("\nSTT binding not initialized.")

# Similar patterns apply to TTV and TTM bindings.
```

To see the specific methods available for each binding, refer to the `__init__.py` file within their respective directories (e.g., `lollms_client/tts_bindings/bark/__init__.py`). Each binding adheres to its respective Abstract Base Class (ABC) defined in `lollms_client/lollms_tts_binding.py`, `lollms_client/lollms_tti_binding.py`, etc.

## Agentic Workflows

`LollmsClient` includes powerful methods for building sophisticated agentic systems that can reason, plan, and use tools.

### `generate_with_mcp` (Multi-tool Code Processor)

```python
generate_with_mcp(
    prompt: str,
    system_prompt: Optional[str] = None,
    objective_extraction_system_prompt: str = "Build a plan",
    images: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tool_calls: int = 5,
    max_llm_iterations: int = 10,
    ctx_size: Optional[int] = None,
    max_json_retries: int = 1,
    tool_call_decision_temperature: float = 0.0,
    final_answer_temperature: Optional[float] = None,
    streaming_callback: Optional[Callable[[str, int, Optional[Dict], Optional[List]], bool]] = None,
    debug: bool = False,
    **llm_generation_kwargs
) -> Dict[str, Any]
```

This method orchestrates a dynamic "observe-think-act" loop, enabling the AI to break down complex tasks, select and execute appropriate tools, and synthesize findings into a coherent final answer. It employs robust error handling and state management to prevent loops and ensure progress.

**Key Features:**
*   **Tool Discovery:** Automatically discovers tools from the active MCP binding.
*   **Context-Aware Asset Ingestion:** Detects code/images in the `context` and registers them as internal assets (with UUIDs) to avoid large data in prompts and prevent JSON errors.
*   **Tool Filtering:** Hides tools that directly consume code from the LLM's view, forcing the use of a safer `local_tools::generate_and_call` meta-tool for code execution.
*   **Knowledge Scratchpad:** Maintains a cumulative knowledge base (`knowledge_scratchpad`) updated after each tool call.
*   **State-Driven Execution:** The agent checks its `knowledge_scratchpad` against its `overall plan` to avoid redundant work.
*   **Internal Tools:** Provides built-in tools like `local_tools::generate_and_call` (for generating and executing code), `local_tools::refactor_scratchpad`, `local_tools::request_clarification`, and `local_tools::final_answer`.

**Parameters:**
*   `prompt` (str): The user's request for the agent to fulfill.
*   `system_prompt` (Optional[str]): System prompt for the final answer generation.
*   `objective_extraction_system_prompt` (str): System prompt used for initial plan building.
*   `images` (Optional[List[str]]): List of image base64 strings or paths relevant to the initial prompt.
*   `tools` (Optional[List[Dict[str, Any]]]): A pre-defined list of tools to use. If `None`, tools are discovered from the MCP binding.
*   `max_tool_calls` (int): Maximum number of tool calls allowed within a single `generate_with_mcp` execution.
*   `max_llm_iterations` (int): Maximum number of reasoning cycles (observe-think-act steps).
*   `ctx_size` (Optional[int]): Context size for LLM calls within the agent loop.
*   `max_json_retries` (int): How many times to retry JSON parsing if the LLM's output is invalid.
*   `tool_call_decision_temperature` (float): Temperature for the LLM's tool-calling decision-making.
*   `final_answer_temperature` (Optional[float]): Temperature for synthesizing the final answer.
*   `streaming_callback` (Optional[Callable[[str, int, Optional[Dict], Optional[List]], bool]]): Callback for real-time updates on agent steps, thoughts, and tool outputs.
*   `debug` (bool): If `True`, enables extensive logging of internal prompts and states.
*   `**llm_generation_kwargs`: Additional keyword arguments passed to internal LLM generation calls.

**Returns:**
*   `Dict[str, Any]`: A dictionary containing:
    *   `final_answer` (str): The comprehensive answer generated by the agent.
    *   `tool_calls` (List[Dict]): A list of all tool calls made during the process, including their names, parameters, and results.
    *   `sources` (List[Dict]): Not fully implemented in current version, but intended for data sources.
    *   `clarification_required` (bool): True if the agent decided it needed more information.
    *   `error` (Optional[str]): An error message if the process failed.

**Example (Requires `local_mcp` binding and default tools):**
To run this example, ensure you have the `local_mcp` binding enabled in your `LollmsClient` and that `internet_search` tool is available.

```python
# Assuming 'multimodal_client' is initialized with mcp_binding_name="local_mcp"
if multimodal_client.mcp:
    print("\n--- Generate with MCP (Agentic Workflow) ---")
    
    def mcp_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None, turn_history: Optional[List] = None):
        if message_type == MSG_TYPE.MSG_TYPE_STEP_START:
            print(f"\n[MCP Step Start]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_STEP_END:
            print(f"[MCP Step End]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_INFO:
            print(f"[MCP Info]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_THOUGHT_CONTENT:
            print(f"[MCP Thought]: {message}")
        elif message_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
            print(f"[MCP Tool Call]: Tool '{metadata.get('name')}' with params {metadata.get('parameters')}")
        elif message_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
            print(f"[MCP Tool Output]: {metadata}")
        elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(message, end="", flush=True)

    try:
        # Example: Use the internet_search tool via the agent
        # For this to work, the 'internet_search' tool must be active in your local_mcp binding
        # See lollms_client/mcp_bindings/local_mcp/default_tools/internet_search/internet_search.py
        
        # A simple query that might trigger the internet search tool
        agent_result = multimodal_client.generate_with_mcp(
            prompt="What is the current population of the United States?",
            streaming_callback=mcp_stream_callback,
            max_llm_iterations=5,
            debug=False # Set to True for verbose internal logging
        )

        print("\n--- MCP Agent Final Result ---")
        if agent_result.get("error"):
            print(f"Error: {agent_result['error']}")
        elif agent_result.get("clarification"):
            print(f"Clarification Required: {agent_result['final_answer']}")
        else:
            print(f"Final Answer: {agent_result['final_answer']}")
            print(f"Tool Calls Made: {len(agent_result['tool_calls'])}")
            for tc in agent_result['tool_calls']:
                print(f"  - {tc['name']}({tc['params']}) -> {tc['result'].get('status', 'N/A')}")

    except Exception as e:
        print(f"An error occurred during MCP generation: {e}")
else:
    print("\nMCP binding not initialized. Skipping generate_with_mcp example.")
```

### `generate_text_with_rag` (Retrieval Augmented Generation)

```python
generate_text_with_rag(
    prompt: str,
    rag_query_function: Callable[[str, Optional[str], int, float], List[Dict[str, Any]]],
    system_prompt: str = "",
    objective_extraction_system_prompt: str = "Extract objectives",
    rag_query_text: Optional[str] = None,
    rag_vectorizer_name: Optional[str] = None,
    rag_top_k: int = 5,
    rag_min_similarity_percent: float = 70.0,
    max_rag_hops: int = 3,
    images: Optional[List[str]] = None,
    n_predict: Optional[int] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    n_threads: Optional[int] = None,
    ctx_size: Optional[int] = None,
    extract_objectives: bool = True,
    streaming_callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict], Optional[List]], bool]] = None,
    max_rag_context_characters: int = 32000,
    **llm_generation_kwargs
) -> Dict[str, Any]
```

This method facilitates multi-hop Retrieval Augmented Generation (RAG). It enables the AI to dynamically formulate queries, retrieve relevant information from a provided knowledge base (via `rag_query_function`), synthesize new knowledge into a scratchpad, and refine its objectives iteratively until it can provide a comprehensive answer.

**Parameters:**
*   `prompt` (str): The initial user query.
*   `rag_query_function` (Callable): A required function that the `LollmsClient` calls to retrieve documents. This function must accept `(query: str, vectorizer_name: Optional[str], top_k: int, min_similarity_percent: float)` and return `List[Dict[str, Any]]` where each dict contains at least `"chunk_text"`, `"file_path"`, and `"similarity_percent"`. This function connects to *your* external RAG system.
*   `system_prompt` (str): System prompt for the final answer synthesis.
*   `objective_extraction_system_prompt` (str): System prompt for extracting initial objectives.
*   `rag_query_text` (Optional[str]): If provided, this specific query is used for the *first* RAG hop, overriding the LLM's initial query generation.
*   `rag_vectorizer_name` (Optional[str]): Name of the vectorizer to use if your `rag_query_function` supports multiple.
*   `rag_top_k` (int): Number of top-N documents to retrieve per query.
*   `rag_min_similarity_percent` (float): Minimum similarity threshold for retrieved documents.
*   `max_rag_hops` (int): Maximum number of iterative query-retrieval-synthesis cycles.
*   `images` (Optional[List[str]]): Images to include in the final answer synthesis prompt.
*   `ctx_size`: Context size override for LLM calls during RAG.
*   `extract_objectives` (bool): If `True`, the LLM will first extract research objectives from the user prompt.
*   `streaming_callback` (Optional[Callable]): Callback for real-time updates on RAG steps, queries, and findings.
*   `max_rag_context_characters` (int): Maximum character length of raw retrieved context to include in the final prompt.
*   `**llm_generation_kwargs`: Additional arguments passed to internal LLM generation calls.

**Returns:**
*   `Dict[str, Any]`: A dictionary containing:
    *   `final_answer` (str): The synthesized answer.
    *   `rag_hops_history` (List[Dict]): Details of each RAG hop (query, retrieved chunks).
    *   `all_retrieved_sources` (List[Dict]): A flattened list of all unique chunks retrieved across all hops.
    *   `error` (Optional[str]): An error message if the process failed.

**Example (Conceptual - requires a RAG function implementation):**
```python
# THIS IS A MOCK RAG QUERY FUNCTION FOR DEMONSTRATION PURPOSES.
# In a real application, this would query a vector database (e.g., Chroma, FAISS, Milvus).
def mock_rag_query_function(query: str, vectorizer_name: Optional[str], top_k: int, min_similarity_percent: float) -> List[Dict[str, Any]]:
    print(f"Mock RAG: Querying for '{query}' (top {top_k}, min sim {min_similarity_percent}%)")
    # Simulate retrieving some relevant chunks
    if "python" in query.lower():
        return [
            {"chunk_text": "Python is a high-level, interpreted, general-purpose programming language. It is known for its readability and large standard library.", "file_path": "docs/python_intro.txt", "similarity_percent": 95.0},
            {"chunk_text": "It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Popular frameworks include Django and Flask.", "file_path": "docs/python_features.txt", "similarity_percent": 88.0},
        ]
    elif "javascript" in query.lower():
        return [
            {"chunk_text": "JavaScript is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS.", "file_path": "docs/js_intro.txt", "similarity_percent": 92.0},
            {"chunk_text": "It is often used for client-side web development to make interactive web pages, and increasingly for server-side with Node.js.", "file_path": "docs/js_web.txt", "similarity_percent": 85.0},
        ]
    else:
        return []

# Assuming 'client' is initialized
# Set debug=True for verbose output of RAG steps
# Define a streaming callback if you want real-time updates
def rag_stream_callback(message: str, message_type: MSG_TYPE, metadata: Optional[Dict] = None, turn_history: Optional[List] = None):
    if message_type in [MSG_TYPE.MSG_TYPE_STEP_START, MSG_TYPE.MSG_TYPE_STEP_END, MSG_TYPE.MSG_TYPE_INFO]:
        print(f"\n[RAG {message_type.name.split('_')[-1]}]: {message}")
    elif message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(message, end="", flush=True)

print("\n--- Generate Text with RAG ---")
try:
    rag_result = client.generate_text_with_rag(
        prompt="Tell me about the key features of Python programming language and its common uses.",
        rag_query_function=mock_rag_query_function,
        max_rag_hops=2,
        streaming_callback=rag_stream_callback,
        debug=False # Set to True to see detailed RAG prompts
    )

    print("\n--- RAG Final Result ---")
    if rag_result.get("error"):
        print(f"Error: {rag_result['error']}")
    else:
        print(f"Final Answer:\n{rag_result['final_answer']}")
        print(f"\nTotal Retrieved Sources: {len(rag_result['all_retrieved_sources'])}")
        # for source in rag_result['all_retrieved_sources']:
        #     print(f"  - {source.get('file_path')} (Sim: {source.get('similarity'):.1f}%)")

except Exception as e:
    print(f"An error occurred during RAG generation: {e}")
```

## Code & Structured Content Generation/Extraction

These methods are designed to facilitate AI interaction with code and structured data formats like JSON.

### `generate_code`

```python
generate_code(
    prompt: str,
    images: List[str] = [],
    system_prompt: Optional[str] = None,
    template: Optional[str] = None,
    language: str = "json",
    code_tag_format: str = "markdown", # or "html"
    n_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    callback: Optional[Callable] = None,
    debug: bool = False
) -> Optional[str]
```

Generates a single code block. It's particularly useful for enforcing specific code formats or structures using a `template`. It also attempts to handle cases where the model might generate an incomplete code block by requesting continuation.

**Parameters:**
*   `prompt` (str): The prompt instructing the LLM to generate code.
*   `images` (List[str]): Optional list of image paths for multimodal context.
*   `system_prompt` (Optional[str]): Additional system instructions for code generation.
*   `template` (Optional[str]): A string representing the desired structure or template for the code (e.g., a JSON schema, a Python function signature).
*   `language` (str): The programming language or format of the code (e.g., "python", "json", "javascript").
*   `code_tag_format` (str): The format of the code block tags ("markdown" or "html").
*   `n_predict`, `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`: LLM generation parameters.
*   `callback` (Optional[Callable]): Streaming callback function.
*   `debug` (bool): Enables verbose debugging output.

**Returns:**
*   `Optional[str]`: The content of the generated code block, or `None` if generation fails.

**Example:**
```python
print("\n--- Generate Python Code ---")
python_code_prompt = "Write a Python function to calculate the factorial of a number."
generated_python_code = client.generate_code(
    prompt=python_code_prompt,
    language="python",
    template="def factorial(n):\n    # implementation\n    pass",
    n_predict=150
)
if generated_python_code:
    print(generated_python_code)
else:
    print("Failed to generate Python code.")
```

### `generate_codes`

```python
generate_codes(
    prompt,
    images=[],
    template=None,
    language="json",
    code_tag_format="markdown", # or "html"
    n_predict = None,
    temperature = None,
    top_k = None,
    top_p=None,
    repeat_penalty=None,
    repeat_last_n=None,
    callback=None,
    debug=False
) -> List[Dict]
```

Generates one or more code blocks based on a prompt. This is an older method; for single, structured code blocks, `generate_code` is often preferred as it handles continuation.

**Parameters:** Same as `generate_code`.

**Returns:**