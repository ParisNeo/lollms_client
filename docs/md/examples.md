## DOCS FOR: `examples/` (Example Scripts)

**Purpose:**
The `examples/` directory provides a collection of Python scripts that demonstrate how to use the `lollms_client` library for various tasks. These examples serve as practical guides and starting points for developers looking to integrate `lollms_client` into their applications.

---
### General Structure of Examples:

Most examples typically involve:
1.  Importing `LollmsClient` and other necessary components (e.g., `MSG_TYPE`, `ASCIIColors`).
2.  Initializing `LollmsClient` with a specific LLM binding and/or modality bindings.
3.  Defining prompts or input data.
4.  Calling methods on the `LollmsClient` instance (e.g., `generate_text`, `generate_with_mcp`, `sequential_summarize`).
5.  Handling the output, often including a streaming callback for real-time display.
6.  Using `ascii_colors` for enhanced console output.

---
### Overview of Key Example Scripts:

*   **`simple_text_gen_test.py`**:
    *   **Purpose**: Demonstrates basic text generation using `LollmsClient` with configurable LLM bindings (LoLLMs server, Ollama, OpenAI).
    *   **Features Shown**:
        *   Initializing `LollmsClient` for different backends.
        *   Non-streaming text generation (`lc.generate_text(stream=False)`).
        *   Streaming text generation (`lc.generate_text(stream=True)`) with a callback.
        *   Listing available models (`lc.list_models()`).
        *   Switching or setting the model for generation.
        *   Token counting (`lc.count_tokens()`) and embedding (`lc.embed()`).

*   **`simple_text_gen_with_image_test.py`**:
    *   **Purpose**: Shows how to perform text generation with image inputs (multimodal generation).
    *   **Features Shown**:
        *   Using `lc.generate_text()` with the `images` parameter.
        *   Highlights compatibility with vision-capable models (e.g., LLaVA with Ollama or `llamacpp`).
        *   Includes non-streaming and streaming examples with image context.

*   **`function_calling_with_local_custom_mcp.py`**:
    *   **Purpose**: A comprehensive example of function calling using the `local_mcp` binding with custom-defined local tools.
    *   **Features Shown**:
        *   Defining simple local tools (`get_weather`, `sum_numbers`) with `.py` execution logic and `.mcp.json` metadata files.
        *   Initializing `LollmsClient` with `mcp_binding_name="local_mcp"` and `mcp_binding_config={"tools_folder_path": ...}`.
        *   Using `lc.generate_with_mcp()` for complex interactions involving tool discovery, LLM decision-making, tool execution, and final answer generation.
        *   Streaming callback for observing MCP interaction steps.
        *   Multi-step tool use.

*   **`local_mcp.py`**:
    *   **Purpose**: Demonstrates using the `local_mcp` binding with its *default packaged tools* (internet search, file writer, image generation, python interpreter).
    *   **Features Shown**:
        *   Initializing `LollmsClient` with `mcp_binding_name="local_mcp"` (without specifying `tools_folder_path` to use defaults).
        *   Invoking default tools like `internet_search` and `generate_image_from_prompt` (which uses the client's TTI binding, e.g., DALL-E).
        *   Handling tool outputs, including saving generated images.

*   **`run_standard_mcp_example.py`**:
    *   **Purpose**: Illustrates how to use the `standard_mcp` binding to connect to external MCP tool servers launched as subprocesses and communicating via `stdio`.
    *   **Features Shown**:
        *   Defining dummy MCP server scripts (`time_server.py`, `calculator_server.py`) using `mcp.server.fastmcp`.
        *   Configuring `initial_servers` for `StandardMCPBinding` with command-line instructions to launch these servers.
        *   Using `lc.generate_with_mcp()` to interact with these external tool servers.

*   **`external_mcp.py`**:
    *   **Purpose**: Extends the `standard_mcp` example to include an externally managed MCP server, specifically demonstrating integration with an ElevenLabs TTS MCP server (if available via `uvx`).
    *   **Features Shown**:
        *   Conditional configuration of an external MCP server (ElevenLabs) based on environment variables (`ELEVENLABS_API_KEY`) and `uvx` availability.
        *   Interacting with tools from both locally launched dummy servers and the external ElevenLabs server within the same `LollmsClient` session.
        *   Using `dotenv` to load API keys.

*   **`openai_mcp.py`**:
    *   **Purpose**: Showcases connecting `LollmsClient` (via `standard_mcp`) to a Python-based MCP server that wraps OpenAI's DALL-E (image generation) and TTS services.
    *   **Features Shown**:
        *   Launching a custom OpenAI MCP server (assumed to be in a project like `ParisNeoMCPServers/openai-mcp-server`) using `uv run`.
        *   Prompting the LLM to use tools provided by this OpenAI MCP server (e.g., `my_openai_server::generate_tts`, `my_openai_server::generate_image_dalle`).
        *   Handling base64 encoded audio/image data returned by the MCP tools.

*   **`generate_text_with_multihop_rag_example.py`**:
    *   **Purpose**: Demonstrates Retrieval Augmented Generation (RAG) with multi-hop capabilities using `lc.generate_text_with_rag()`.
    *   **Features Shown**:
        *   Using a mock RAG query function.
        *   Classic RAG (0 hops) vs. multi-hop RAG (LLM refines queries or decides if enough info).
        *   Streaming callback to observe RAG steps (query generation, retrieval, LLM decision).

*   **`internet_search_with_rag.py`**:
    *   **Purpose**: Implements RAG using actual internet search results (via DuckDuckGo) as the knowledge source.
    *   **Features Shown**:
        *   A `perform_internet_search_rag` function that uses `duckduckgo_search`.
        *   Using this search function with `lc.generate_text_with_rag()` for multi-hop internet-grounded answers.

*   **`article_summary/article_summary.py`**:
    *   **Purpose**: Uses `lc.sequential_summarize()` to summarize a web article (e.g., an arXiv paper).
    *   **Features Shown**:
        *   Fetching and converting a document from a URL using `docling`.
        *   Detailed custom prompts for chunk processing and final summary generation within `sequential_summarize`.

*   **`deep_analyze/deep_analyse.py` & `deep_analyze_multiple_files.py`**:
    *   **Purpose**: Utilizes `lc.deep_analyze()` to find information related to a specific query within a long document or multiple files.
    *   **Features Shown**:
        *   Analyzing a single document from a URL.
        *   Analyzing multiple local files (PDF, TXT, MD, DOCX, etc.) found in a directory.

*   **`generate_and_speak/generate_and_speak.py`**:
    *   **Purpose**: Combines LLM text generation with Text-to-Speech (TTS) output using various TTS bindings.
    *   **Features Shown**:
        *   Initializing `LollmsClient` with both LLM and TTS bindings (e.g., Ollama + Bark, Ollama + XTTS, LoLLMs server + Piper).
        *   Generating text and then synthesizing it into speech.
        *   Using `pygame` for audio playback.
        *   Command-line arguments for selecting bindings and models.
        *   Helper function to download a default Piper voice for demo purposes.

*   **`generate_game_sfx/generate_game_fx.py`**:
    *   **Purpose**: Demonstrates generating game sound effects (SFX) using Text-to-Music (TTM) bindings like AudioCraft or Bark.
    *   **Features Shown**:
        *   Initializing `LollmsClient` with TTM bindings.
        *   Using `lc.ttm.generate_music()` with prompts tailored for SFX.
        *   Specific parameter handling for AudioCraft (duration, temperature) vs. Bark (voice_preset for SFX).
        *   Playing generated SFX using `pygame`.

*   **Text-to-Image Examples**:
    *   **`text_2_image.py`**: Uses the `lollms` TTI binding (connecting to a LoLLMs WebUI server's TTI service). Demonstrates listing services, getting/setting TTI settings, and generating an image.
    *   **`text_2_image_diffusers.py`**: Focuses on the `diffusers` TTI binding for local Stable Diffusion inference. Shows setup for a Diffusers environment, model download, and image generation. Also includes a DALL-E binding test section.

*   **Personality Tests (`personality_test/`)**: (These examples seem to interact with a LoLLMs server that supports personalities, rather than client-side personality logic directly within `lollms-client` itself, which `lollms-client` doesn't have yet. The client-side `LollmsPersonality` class in these examples seems to be a custom wrapper not part of the core `lollms-client` library shown elsewhere.)
    *   `chat_test.py`: Basic chat interaction with a custom inline "personality".
    *   `chat_with_aristotle.py`: Chat interaction with an "Aristotle" personality.
    *   `tesks_test.py`: (Likely "tasks\_test.py") Demonstrates using `TasksLibrary` (which uses `LollmsClient`) for Q&A and code extraction.
        *   *Note: The `LollmsPersonality` class used in `chat_test.py` and `chat_with_aristotle.py` seems to be a user-defined class in those examples, not directly from the `lollms_client` library's core exports shown in `lollms_client/__init__.py`. The `TasksLibrary` is also likely user-defined or from another related project, as it's not in the `lollms_client` core.*

*   **Other Specific Tests**:
    *   `test_local_models/local_chat.py`: A simple chat example using the `transformers` binding with a local model.
    *   `text_2_audio.py`: Focuses on TTS using the `lollms` TTS binding.
    *   `text_and_image_2_audio.py`: (Relies on `cv2` and a local webcam) Captures an image, gets a description using `lc.generate_with_images()`, then converts the description to audio using the `lollms` TTS binding (via `LollmsTTS` helper, which isn't standard in client).
    *   `text_gen.py`: Basic text generation tests with different bindings (LoLLMs, Ollama, LlamaCpp).
    *   `text_gen_system_prompt.py`: Shows using `system_prompt` with `lc.generate_text()`.
    *   `generate_a_benchmark_for_safe_store.py`: Uses `LollmsClient` to generate paraphrases, related sentences, etc., from a Hugging Face dataset to create a benchmark, saving results to JSON.

**How to Use Examples:**
1.  Ensure `lollms_client` is installed.
2.  Install any additional dependencies required by a specific example (e.g., `pygame` for audio playback, `docling` for document conversion, `cv2` for webcam, specific LLM/TTS libraries for local bindings). The examples often use `pipmaster` to try and install these.
3.  Configure the example script:
    *   Update model names, host addresses, API keys (often via `.env` files or directly in the script for testing).
    *   Provide paths to local models or data files if needed.
4.  Run the Python script from your terminal (e.g., `python examples/simple_text_gen_test.py`).
