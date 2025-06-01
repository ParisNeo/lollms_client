# LoLLMS Client - User Documentation

**Version:** (Based on files up to 2025-06-01)
**GitHub:** [https://github.com/ParisNeo/simplified_lollms](https://github.com/ParisNeo/simplified_lollms)
**PyPI:** [https://pypi.org/project/lollms-client/](https://pypi.org/project/lollms-client/) (Link may need verification if package name differs)
**License:** Apache 2.0

## 1. Introduction

Welcome to `lollms_client`! This Python library provides a simple and powerful way to interact with a wide range of Large Language Models (LLMs) and other generative AI services like Text-to-Speech (TTS), Speech-to-Text (STT), and more.

Whether you want to integrate advanced AI capabilities into your Python applications, experiment with different models, or build custom AI-powered tools, `lollms_client` aims to make the process straightforward.

**Key Features:**

*   **Unified Interface:** Interact with diverse AI models and services using a consistent API.
*   **Extensible Bindings:** Easily switch between different AI backends (e.g., Ollama, OpenAI, local Llama.cpp, Hugging Face Transformers, Bark TTS, Piper TTS, XTTS) by changing a single configuration parameter.
*   **Local & Cloud Support:** Works with both locally hosted models and cloud-based AI services.
*   **Multimodal Ready:** Designed to support text, speech, image, and potentially video/music generation and processing.
*   **Helper Utilities:** Includes tools for prompt engineering, discussion management, and task automation.

This guide will walk you through installing `lollms_client` and using its core features.

## 2. Installation

### 2.1. Basic Installation

You can install `lollms_client` directly from PyPI using pip:

```bash
pip install lollms-client
```

This will install the core client library. Dependencies for specific AI bindings (like `ollama`, `openai-whisper`, `TTS`, `piper-tts`, `audiocraft`, `transformers`, `llama-cpp-python`, etc.) are **not** installed by default to keep the core installation lightweight. They will be installed automatically by `pipmaster` (a utility included in `lollms-client`) when you first try to use a binding that requires them, or you can pre-install them.

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

**System-Level Dependencies:**

*   **`ffmpeg`:** Required by many audio/video processing bindings (Whisper, WhisperCpp, AudioCraft for MP3, XTTS if it processes audio). Install it system-wide:
    *   **Linux:** `sudo apt update && sudo apt install ffmpeg`
    *   **macOS:** `brew install ffmpeg`
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.
*   **`whisper.cpp` executable:** For the `whispercpp` STT binding, you need to compile and install the `whisper.cpp` command-line tool from [ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp).

## 3. Core Usage: `LollmsClient`

The primary way to use the library is through the `LollmsClient` class.

### 3.1. Basic Initialization (LLM Only)

```python
from lollms_client import LollmsClient

# Example 1: Using Ollama (ensure Ollama server is running)
# Assumes 'mistral' model is pulled: `ollama pull mistral`
try:
    client = LollmsClient(
        binding_name="ollama",
        model_name="mistral" # Or any model you have in Ollama
    )
    print("Ollama client initialized successfully!")
except Exception as e:
    print(f"Error initializing Ollama client: {e}")

# Example 2: Using OpenAI (requires API key)
# Set OPENAI_API_KEY environment variable or pass as service_key
# try:
#     client_openai = LollmsClient(
#         binding_name="openai",
#         model_name="gpt-3.5-turbo",
#         service_key="sk-YOUR_OPENAI_API_KEY"
#     )
#     print("OpenAI client initialized successfully!")
# except Exception as e:
#     print(f"Error initializing OpenAI client: {e}")

# Example 3: Using a local GGUF model with LlamaCppServer binding
# Requires llama.cpp server binary (e.g., from llama-cpp-python[server] or llama-cpp-binaries)
# And a GGUF model file.
# try:
#     client_llamacpp_server = LollmsClient(
#         binding_name="llamacpp", # This is for LlamaCppServerBinding
#         models_path="/path/to/your/gguf_models_folder/",
#         model_name="your_model.gguf",
#         llm_binding_config={ # Optional: pass config to the binding
#             "n_gpu_layers": 20, # Example: offload 20 layers to GPU
#             "n_ctx": 4096
#         }
#     )
#     print("LlamaCppServer client initialized!")
# except Exception as e:
#     print(f"Error initializing LlamaCppServer client: {e}")
```

**Key `LollmsClient` Initialization Parameters for LLMs:**

*   `binding_name` (str): The name of the LLM binding to use (e.g., "ollama", "openai", "llamacpp", "pythonllamacpp", "transformers", "lollms").
*   `model_name` (str): The identifier for the model (e.g., "mistral" for Ollama, "gpt-3.5-turbo" for OpenAI, filename for local GGUF models).
*   `host_address` (Optional[str]): For server-based bindings like "ollama", "lollms", or a custom OpenAI endpoint. Defaults are usually "http://localhost:11434" for Ollama and "http://localhost:9600" for a local LOLLMS server.
*   `models_path` (Optional[str]): For local file-based bindings like "llamacpp", "pythonllamacpp", "transformers", "vllm". This is the directory where your model files (e.g., GGUF files, Hugging Face model directories) are stored.
*   `service_key` (Optional[str]): API key or other authentication token for services like OpenAI.
*   `llm_binding_config` (Optional[Dict]): A dictionary of additional parameters to pass directly to the chosen LLM binding's `__init__` method (e.g., `{"n_gpu_layers": -1}` for `PythonLlamaCppBinding` or `LlamaCppServerBinding` to offload all layers to GPU).
*   `verify_ssl_certificate` (bool): Defaults to `True`. Set to `False` for local servers with self-signed certificates.

### 3.2. Initializing with Multiple Modalities (LLM, TTS, STT, etc.)

You can initialize `LollmsClient` to use bindings for other modalities simultaneously.

```python
from lollms_client import LollmsClient

# Example: LLM with Ollama, TTS with Bark
try:
    client = LollmsClient(
        # LLM Configuration
        binding_name="ollama",
        model_name="mistral", # LLM model
        host_address="http://localhost:11434",

        # TTS Configuration
        tts_binding_name="bark",
        tts_binding_config={
            "model_name": "suno/bark-small", # Bark model ID
            "default_voice": "v2/en_speaker_3", # Default Bark voice preset
            "device": "cuda" # Optional: "cpu", "cuda", "mps" for Bark
        },

        # STT Configuration (Example with Whisper)
        # stt_binding_name="whisper",
        # stt_binding_config={
        #     "model_name": "base", # Whisper model size
        #     "device": "cuda"
        # }
        # Add other modalities (tti_binding_name, ttv_binding_name, ttm_binding_name) similarly
    )
    print("LollmsClient initialized with LLM and TTS (Bark).")

    # --- Generate text ---
    text_to_speak = client.generate_text("Tell me a fun fact.", n_predict=50)
    print(f"\nGenerated Text: {text_to_speak}")

    # --- Generate speech (if TTS is configured) ---
    if client.tts and text_to_speak:
        print("\nSynthesizing speech...")
        audio_bytes = client.tts.generate_audio(text_to_speak) # Uses default voice from init
        # audio_bytes = client.tts.generate_audio(text_to_speak, voice="v2/en_speaker_1") # Override voice
        
        if audio_bytes:
            output_path = "generated_speech.wav"
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Speech saved to {output_path}")
            # You can now play this file using any audio player or a library like pygame
        else:
            print("Speech synthesis failed or returned no data.")

except Exception as e:
    print(f"Error initializing LollmsClient or during operation: {e}")

```

**Key Parameters for Other Modalities:**

*   `tts_binding_name`, `stt_binding_name`, `tti_binding_name`, `ttm_binding_name`, `ttv_binding_name`: String names of the desired bindings (e.g., "bark", "xtts", "piper" for TTS; "whisper", "whispercpp" for STT; "audiocraft", "bark" for TTM).
*   `tts_binding_config`, `stt_binding_config`, etc.: Dictionaries of configuration parameters specific to the chosen binding for that modality.
*   `tts_host_address`, `stt_host_address`, etc.: Host address if the modality binding is server-based (e.g., "lollms" TTS).

### 3.3. Generating Text

```python
from lollms_client import LollmsClient, MSG_TYPE

# Assume client is initialized as shown above
client = LollmsClient(binding_name="ollama", model_name="mistral") # Example

# --- Simple Text Generation ---
prompt = "What is the capital of France?"
response_text = client.generate_text(prompt, n_predict=50) # n_predict limits token output
print(f"Q: {prompt}\nA: {response_text}")

# --- Streaming Text Generation ---
def my_streaming_callback(chunk: str, message_type: MSG_TYPE) -> bool:
    """
    Callback function to process streamed text chunks.
    message_type helps understand the nature of the chunk (see lollms_types.py).
    Return False to stop streaming.
    """
    if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    return True # Continue streaming

print("\nStreaming response:")
full_streamed_response = client.generate_text(
    prompt="Tell me a short story about a robot.",
    n_predict=200,
    stream=True,
    streaming_callback=my_streaming_callback,
    temperature=0.8, # Adjust creativity
    # Other parameters: top_k, top_p, repeat_penalty, seed, etc.
)
print("\n--- End of Stream ---")
# full_streamed_response will contain the complete text after streaming

# --- Multimodal Text Generation (if LLM binding supports it, e.g., LlamaCppServer with LLaVA) ---
# Ensure your LLM binding (e.g., LlamaCppServer or Transformers with a LLaVA model)
# is configured to handle images.
# image_paths = ["path/to/your/image1.jpg", "path/to/your/image2.png"]
# multimodal_prompt = "Describe these images."
# try:
#     multimodal_response = client.generate_text(multimodal_prompt, images=image_paths)
#     print(f"Multimodal Response: {multimodal_response}")
# except Exception as e:
#     print(f"Multimodal generation failed (binding might not support it or images are invalid): {e}")

```

**Common `generate_text` Parameters:**

*   `prompt` (str): The main text input to the LLM.
*   `images` (Optional[List[str]]): A list of file paths to images if the LLM binding supports multimodal input (e.g., LLaVA models).
*   `system_prompt` (str): An optional instruction to guide the LLM's persona or behavior, placed before the main prompt.
*   `n_predict` (Optional[int]): Maximum number of new tokens to generate.
*   `stream` (bool): If `True`, enables streaming output via the `streaming_callback`.
*   `streaming_callback` (Optional[Callable]): A function called with each new chunk of text when `stream=True`.
*   `temperature`, `top_k`, `top_p`, `repeat_penalty`, `seed`: Standard LLM sampling parameters to control output randomness and style.
*   `ctx_size` (Optional[int]): Override the context window size for this specific generation.

### 3.4. Tokenization

```python
# client = LollmsClient(...)
text = "Hello, world! This is a test."
tokens = client.tokenize(text)
print(f"Tokens for '{text}': {tokens}")

detokenized_text = client.detokenize(tokens)
print(f"Detokenized: {detokenized_text}")

token_count = client.count_tokens(text)
print(f"Token count: {token_count}")
```
Tokenization behavior depends on the active LLM binding.

### 3.5. Embeddings

If the active LLM binding supports generating embeddings:
```python
# client = LollmsClient(...) # Ensure binding supports embeddings (e.g., Ollama with an embed model, Transformers)
try:
    embedding_vector = client.embed("This is text to embed.")
    print(f"Embedding (first 5 dims): {embedding_vector[:5]}...")
except NotImplementedError:
    print("The current LLM binding does not support embeddings.")
except Exception as e:
    print(f"Embedding generation failed: {e}")
```

### 3.6. Listing and Switching LLM Models

```python
# client = LollmsClient(...)
available_models = client.listModels() # Behavior depends on the binding
print(f"Available models for current binding ({client.binding.binding_name}): {available_models}")

# if client.binding.binding_name == "ollama":
#     try:
#         client.switch_model("codellama") # Switch to another Ollama model
#         print("Switched to codellama model.")
#         print(client.generate_text("def fibonacci(n):", n_predict=50))
#     except Exception as e:
#         print(f"Failed to switch model or generate: {e}")
```
The `listModels()` and `switch_model()` behavior is highly dependent on the active LLM binding.
*   Server-based bindings (Ollama, OpenAI, LOLLMS server) will list models available on that server.
*   Local file-based bindings (LlamaCpp, PythonLlamaCpp, Transformers, vLLM) might list models in their configured `models_path` or describe the currently loaded model.

### 3.7. Using Other Modalities (TTS, STT, TTI, TTM, TTV)

If you initialized `LollmsClient` with bindings for other modalities, you can access them via attributes like `client.tts`, `client.stt`, etc.

**Text-to-Speech (TTS) Example (using Bark, as set up in 3.2):**
```python
# client = LollmsClient(...) # Initialized with tts_binding_name="bark"
if client.tts:
    try:
        text_to_say = "Welcome to the world of AI-powered audio!"
        # The 'voice' parameter corresponds to Bark's voice_preset
        audio_bytes = client.tts.generate_audio(text_to_say, voice="v2/en_speaker_1")
        
        if audio_bytes:
            with open("bark_speech.wav", "wb") as f:
                f.write(audio_bytes)
            print("Bark speech saved to bark_speech.wav")
            # You would then use a library like pygame or simpleaudio to play it
        else:
            print("Bark TTS returned no audio data.")
            
        available_voices = client.tts.list_voices() # For Bark, lists voice presets
        print(f"Available Bark voice presets (examples): {available_voices[:5]}")

    except Exception as e:
        print(f"Bark TTS error: {e}")
else:
    print("TTS binding not configured in LollmsClient.")
```

**Speech-to-Text (STT) Example (Conceptual - assuming a Whisper binding):**
```python
# client = LollmsClient(stt_binding_name="whisper", stt_binding_config={"model_name":"base"})
if client.stt:
    try:
        # Ensure you have an audio file (e.g., my_audio.wav)
        transcribed_text = client.stt.transcribe_audio("path/to/your/my_audio.wav")
        print(f"Transcribed text: {transcribed_text}")
        
        available_stt_models = client.stt.list_models() # For Whisper, lists model sizes
        print(f"Available Whisper STT models: {available_stt_models}")
    except FileNotFoundError:
        print("Audio file for STT not found.")
    except Exception as e:
        print(f"STT error: {e}")
else:
    print("STT binding not configured.")
```

**Text-to-Image (TTI) Example (Conceptual - assuming a suitable binding):**
```python
# client = LollmsClient(tti_binding_name="your_tti_binding_name", ...)
if client.tti:
    try:
        image_prompt = "A futuristic cityscape at sunset, cyberpunk style."
        # TTI bindings might have parameters like width, height, negative_prompt, seed, steps etc.
        image_bytes = client.tti.generate_image(image_prompt, width=1024, height=768)
        
        if image_bytes:
            with open("generated_image.png", "wb") as f: # Or .jpg depending on binding
                f.write(image_bytes)
            print("Image saved to generated_image.png")
    except Exception as e:
        print(f"TTI error: {e}")
else:
    print("TTI binding not configured.")
```

Refer to the specific binding's documentation or test block (`if __name__ == "__main__":`) within its `__init__.py` file for details on its unique configuration parameters and capabilities.

## 4. Advanced Features

### 4.1. `TasksLibrary` (`lollms_tasks.py`)

For more complex operations that might involve structured prompting or multiple LLM calls, `TasksLibrary` offers pre-built solutions.

```python
from lollms_client import LollmsClient, TasksLibrary

client = LollmsClient(binding_name="ollama", model_name="mistral") # Example
tasks = TasksLibrary(lollms=client)

# Example: Yes/No Question
context_for_yes_no = "The sky is blue during the day. At night, it appears dark."
is_sky_green = tasks.yes_no(
    question="Is the sky green during the day?",
    context=context_for_yes_no,
    return_explanation=True # Get explanation as well
)
print(f"Is sky green? Answer: {is_sky_green['answer']}, Explanation: {is_sky_green['explanation']}")

# Example: Summarization (simplified call)
long_text = "..." # Your long text here
# summary = tasks.sequential_summarize(long_text, chunk_processing_prompt="Extract key points.")
# print(f"Summary: {summary}")
```
`TasksLibrary` includes methods for:
*   Code generation (`generate_code`, `generate_codes`)
*   Structured Q&A (`yes_no`, `multichoice_question`)
*   Summarization and analysis (`sequential_summarize`, `deep_analyze`)
*   Function calling orchestration (`generate_with_function_calls`)

### 4.2. `FunctionCalling_Library` (`lollms_functions.py`)

Facilitates interactions where the LLM can request the execution of predefined Python functions.

```python
from lollms_client import LollmsClient, TasksLibrary, FunctionCalling_Library

client = LollmsClient(binding_name="ollama", model_name="mistral") # Example
tasks = TasksLibrary(lollms=client)
fn_library = FunctionCalling_Library(tasks_library=tasks)

# 1. Define your Python functions
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Gets the current weather for a given location."""
    if location.lower() == "paris":
        return f"The weather in Paris is sunny, 25 degrees {unit}."
    elif location.lower() == "london":
        return f"The weather in London is cloudy, 18 degrees {unit}."
    else:
        return f"Sorry, I don't have weather information for {location}."

# 2. Register them with the library
fn_library.register_function(
    function_name="get_current_weather",
    function_callable=get_current_weather,
    function_description="Useful for getting the current weather in a specific city.",
    function_parameters=[ # Describe parameters for the LLM
        {"name": "location", "type": "string", "description": "The city and state, e.g. San Francisco, CA"},
        {"name": "unit", "type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit for temperature"}
    ]
)

# 3. Generate with function calling capability
user_prompt = "What's the weather like in Paris today in fahrenheit?"
print(f"\nUser: {user_prompt}")

# This call internally uses TasksLibrary to prompt the LLM with function descriptions
ai_response_text, function_calls_requested = fn_library.generate_with_functions(user_prompt)

print(f"AI Initial Response: {ai_response_text}")

if function_calls_requested:
    print(f"LLM requested function calls: {function_calls_requested}")
    # 4. Execute the functions
    results = fn_library.execute_function_calls(function_calls_requested)
    print(f"Function Results: {results}")

    # 5. (Optional) Send results back to LLM for a final answer
    # This would involve constructing a new prompt with the function results.
    # For brevity, this step is omitted here.
    # final_prompt = f"{user_prompt}\n{ai_response_text}\n" # Original interaction
    # for call, res in zip(function_calls_requested, results):
    #     final_prompt += f"Function call {call['function_name']}({call['function_parameters']}) executed and returned: {res}\n"
    # final_prompt += "Now, please provide the final answer to the user."
    # final_answer = client.generate_text(final_prompt)
    # print(f"AI Final Answer (after function execution): {final_answer}")
else:
    print("LLM did not request any function calls.")

```

### 4.3. `LollmsDiscussion` (`lollms_discussion.py`)

Manages conversation history.

```python
from lollms_client import LollmsClient, LollmsDiscussion

client = LollmsClient(binding_name="ollama", model_name="mistral") # Example
discussion = LollmsDiscussion(lollmsClient=client)

discussion.add_message(sender="user", content="Hello, AI!")
ai_reply = client.generate_text(
    discussion.format_discussion(max_allowed_tokens=1000) + "\n!@>assistant:\n", # Format for LLM
    n_predict=50
)
discussion.add_message(sender="assistant", content=ai_reply)

discussion.add_message(sender="user", content="What did I just say?")
ai_contextual_reply = client.generate_text(
    discussion.format_discussion(max_allowed_tokens=1000) + "\n!@>assistant:\n",
    n_predict=50
)
discussion.add_message(sender="assistant", content=ai_contextual_reply)

print("\n--- Full Discussion ---")
for msg in discussion.messages:
    print(f"{msg.sender}: {msg.content}")

# discussion.save_to_disk("my_conversation.yaml")
```

## 5. Available Bindings (Summary)

As of this documentation, `lollms_client` includes or is designed for bindings such as:

*   **LLM:**
    *   `lollms`: Connects to a full LOLLMS server instance.
    *   `ollama`: Connects to an Ollama server.
    *   `openai`: Connects to OpenAI API or compatible endpoints.
    *   `llamacpp`: Manages `llama.cpp` server subprocesses for GGUF models.
    *   `pythonllamacpp`: Uses `llama-cpp-python` library directly for GGUF models.
    *   `transformers`: Uses Hugging Face `transformers` library for local models.
    *   `vllm`: Uses vLLM library for high-throughput local model serving (Linux/NVIDIA).
    *   `openllm`: Connects to an OpenLLM server.
*   **TTS (Text-to-Speech):**
    *   `lollms`: Uses TTS service from a LOLLMS server.
    *   `bark`: Uses Suno AI's Bark model locally via `transformers`.
    *   `xtts`: Uses Coqui AI's XTTSv2 model locally via `TTS` library.
    *   `piper`: Uses Piper TTS voices locally.
*   **STT (Speech-to-Text):**
    *   `lollms`: Uses STT service from a LOLLMS server.
    *   `whisper`: Uses OpenAI's Whisper model locally via `openai-whisper`.
    *   `whispercpp`: Uses `whisper.cpp` command-line tool locally.
*   **TTI (Text-to-Image):**
    *   `lollms`: Uses TTI service from a LOLLMS server.
    *   *(More local TTI bindings can be added, e.g., for Stable Diffusion via `diffusers`)*
*   **TTM (Text-to-Music/Sound):**
    *   `lollms`: Uses TTM service from a LOLLMS server.
    *   `audiocraft`: Uses Meta's AudioCraft (MusicGen) locally.
    *   `bark`: Can also generate simple sound effects and non-speech audio.
*   **TTV (Text-to-Video):**
    *   `lollms`: Uses TTV service from a LOLLMS server.
    *   *(More local TTV bindings can be added)*

To find the exact configuration options for a specific binding, it's often best to look at its `__init__.py` file within the `lollms_client/<modality>_bindings/<binding_name>/` directory and its `if __name__ == "__main__":` test block.

## 6. Examples

The `examples/` directory in the repository contains various scripts demonstrating how to use `lollms_client` for different tasks:
*   Simple text generation.
*   Text generation with image input.
*   Function calling.
*   Article summarization.
*   Deep analysis of text.
*   Generating game sound effects.
*   Text generation and speech synthesis combined.

Explore these examples to see `lollms_client` in action.

## 7. Troubleshooting

*   **Dependency Issues:** If a binding fails to load, check its specific Python package dependencies. `pipmaster` will attempt to install them, but sometimes manual intervention or specific versions are needed. System-level tools like `ffmpeg` must be installed separately.
*   **Model Not Found:**
    *   For server-based bindings (Ollama, OpenAI, LOLLMS server), ensure the server is running and the model name is correct and available on the server.
    *   For local file-based bindings, ensure `models_path` is correctly set and the model files (GGUF, HF directory) are present at the specified `model_name` location within that path.
*   **CUDA/GPU Issues:** For bindings that use GPUs (LlamaCpp, PythonLlamaCpp, Transformers, vLLM, Bark, XTTS, etc.):
    *   Ensure you have compatible NVIDIA drivers (for CUDA) or macOS setup (for MPS).
    *   PyTorch must be installed with the correct CUDA version if you intend to use CUDA. Bindings like `audiocraft` and `bark` in `lollms_client` attempt to install a CUDA-enabled PyTorch if appropriate.
    *   Check `n_gpu_layers` (for GGUF models) or `device` parameters.
*   **`ffmpeg` Not Found:** Many audio bindings (Whisper, AudioCraft for MP3, XTTS for some processing) require `ffmpeg`. Install it and ensure it's in your system's PATH.
*   **API Keys:** For services like OpenAI, ensure your API key (`service_key`) is correctly set.

## 8. Further Help

*   **GitHub Issues:** If you encounter bugs or have feature requests, please open an issue on the [main `simplified_lollms` GitHub repository](https://github.com/ParisNeo/simplified_lollms/issues).
*   **Examples:** Study the scripts in the `examples/` directory.
*   **Binding Code:** For advanced configuration or understanding of a specific binding, looking into its `__init__.py` file within `lollms_client/<modality>_bindings/<binding_name>/` can be very insightful, especially its `__init__` method and test block.

We hope you enjoy using `lollms_client`!