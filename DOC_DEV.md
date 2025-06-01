# LoLLMS Client - Developer Documentation

Welcome to the developer documentation for `lollms_client`! This guide is intended for developers who want to understand the project's architecture, contribute new features, fix bugs, or create new bindings.

**Project Links:**
*   **GitHub Repository:** [https://github.com/ParisNeo/simplified_lollms](https://github.com/ParisNeo/simplified_lollms) (Note: The client is part of this larger ecosystem, this doc focuses on the `lollms_client` library itself)
*   **PyPI Package:** [https://pypi.org/project/lollms-client/](https://pypi.org/project/lollms-client/)
*   **License:** Apache 2.0

## Table of Contents

1.  [Introduction](#1-introduction)
    *   [Project Goal](#project-goal)
    *   [Key Features](#key-features)
2.  [Getting Started for Developers](#2-getting-started-for-developers)
    *   [Prerequisites](#prerequisites)
    *   [Cloning the Repository](#cloning-the-repository)
    *   [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
    *   [Editable Install](#editable-install)
    *   [Installing Development Dependencies](#installing-development-dependencies)
3.  [Project Structure](#3-project-structure)
4.  [Core Concepts & Architecture](#4-core-concepts--architecture)
    *   [LollmsClient (The Orchestrator)](#lollmsclient-the-orchestrator)
    *   [Bindings (The Backends)](#bindings-the-backends)
        *   [Abstract Base Classes (ABCs)](#abstract-base-classes-abcs)
        *   [Binding Managers](#binding-managers)
    *   [Supported Modalities and Bindings](#supported-modalities-and-bindings)
        *   [LLM (Large Language Model) Bindings](#llm-large-language-model-bindings)
        *   [TTS (Text-to-Speech) Bindings](#tts-text-to-speech-bindings)
        *   [TTI (Text-to-Image) Bindings](#tti-text-to-image-bindings)
        *   [STT (Speech-to-Text) Bindings](#stt-speech-to-text-bindings)
        *   [TTM (Text-to-Music/Sound) Bindings](#ttm-text-to-musicsound-bindings)
        *   [TTV (Text-to-Video) Bindings](#ttv-text-to-video-bindings)
    *   [TasksLibrary (High-Level Operations)](#taskslibrary-high-level-operations)
    *   [LollmsDiscussion & LollmsMessage](#lollmsdiscussion--lollmsmessage)
    *   [Configuration (lollms_config.py)](#configuration-lollms_configpy)
    *   [Utilities & Types](#utilities--types)
5.  [Adding New Bindings (A Practical Guide)](#5-adding-new-bindings-a-practical-guide)
    *   [Step 1: Create the Binding Directory](#step-1-create-the-binding-directory)
    *   [Step 2: Create `__init__.py`](#step-2-create-__init__py)
    *   [Step 3: Define `BindingName`](#step-3-define-bindingname)
    *   [Step 4: Implement the Binding Class](#step-4-implement-the-binding-class)
    *   [Step 5: Handle Dependencies](#step-5-handle-dependencies)
    *   [Step 6: Add a Test Block](#step-6-add-a-test-block)
6.  [Running Examples & Tests](#6-running-examples--tests)
7.  [Coding Standards & Conventions](#7-coding-standards--conventions)
8.  [Contribution Guidelines](#8-contribution-guidelines)
9.  [Reporting Issues](#9-reporting-issues)
10. [Roadmap & Future Ideas](#10-roadmap--future-ideas)
11. [Community & Contact](#11-community--contact)

---

## 1. Introduction

### Project Goal

`lollms_client` is a simplified Python client library designed to provide a unified and easy-to-use interface for interacting with various AI model backends and services. It aims to abstract the complexities of different APIs and local model execution, allowing developers to seamlessly switch between different AI modalities (LLM, TTS, TTI, STT, TTM, TTV) and their underlying implementations (bindings).

The "LoLLMS" ecosystem (Lord of Large Language and Multimodal Systems) aims to be a versatile platform for AI interaction, and this client is a key component for Python-based applications and integrations.

### Key Features

*   **Modular Binding System:** Easily add support for new AI backends or libraries.
*   **Multi-Modality:** Supports Large Language Models (LLM), Text-to-Speech (TTS), Text-to-Image (TTI), Speech-to-Text (STT), Text-to-Music/Sound (TTM), and Text-to-Video (TTV).
*   **Unified API:** Provides a consistent `LollmsClient` interface for common operations across different bindings.
*   **Local & Remote Backends:** Supports both local model execution (e.g., LlamaCpp, Transformers, Piper, Bark) and remote services (e.g., Ollama, OpenAI, LOLLMS server).
*   **Helper Utilities:** Includes tools for prompt management, task execution (`TasksLibrary`), and discussion handling.
*   **Dependency Management:** Uses `pipmaster` within bindings to attempt to ensure necessary Python packages are installed.

## 2. Getting Started for Developers

### Prerequisites

*   Python 3.8 or higher
*   Git
*   `pip` and `venv` (recommended for virtual environments)

### Cloning the Repository

The `lollms_client` is part of the main `simplified_lollms` repository.
```bash
git clone https://github.com/ParisNeo/simplified_lollms.git
cd simplified_lollms/lollms_client
```
This documentation will refer to the `lollms_client` subdirectory as the project root for development of the client library.

### Setting up a Virtual Environment

It's highly recommended to use a virtual environment for development:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Editable Install

To make your local changes to `lollms_client` immediately available for testing, install it in editable mode:
```bash
pip install -e .
```
This command should be run from the `lollms_client` directory (where `pyproject.toml` or `setup.py` is located).

### Installing Development Dependencies

While `lollms_client` itself has its core dependencies listed in `requirements.txt` or `pyproject.toml`, for development, you might want tools like:
```bash
pip install pytest black flake8 # Example tools
```
Refer to the main project's contribution guidelines for specific linting or testing tools if they are enforced.

## 3. Project Structure

Here's an overview of the `lollms_client` directory structure:

```
📁 lollms_client/
├─ 📁 ai_documentation/         # Markdown files generated by AI documentation tools (like lollms_python_analyzer.py)
├─ 📁 dist/                     # Build artifacts (not typically version controlled)
├─ 📁 examples/                 # Example scripts demonstrating client usage
│  ├─ 📁 article_summary/
│  ├─ 📁 deep_analyze/
│  ├─ 📁 function_call/
│  ├─ 📁 game_sfx_generation/  # Example for TTM
│  ├─ 📁 personality_test/
│  ├─ 📁 test_and_speech_demo/ # Example for LLM + TTS
│  └─ 📁 test_local_models/
├─ 📁 lollms_client/            # The core library source code
│  ├─ 📁 llm_bindings/           # LLM-specific binding implementations
│  │  ├─ 📁 llamacpp/
│  │  ├─ 📁 lollms/
│  │  ├─ 📁 ollama/
│  │  ├─ 📁 openai/
│  │  ├─ 📁 openllm/
│  │  ├─ 📁 pythonllamacpp/
│  │  ├─ 📁 tensor_rt/          # (TensorRT-LLM via vLLM-like structure)
│  │  ├─ 📁 transformers/
│  │  └─ 📁 vllm/
│  ├─ 📁 stt_bindings/           # Speech-to-Text binding implementations
│  │  ├─ 📁 lollms/
│  │  ├─ 📁 whisper/
│  │  └─ 📁 whispercpp/
│  ├─ 📁 tti_bindings/           # Text-to-Image binding implementations
│  │  └─ 📁 lollms/
│  ├─ 📁 ttm_bindings/           # Text-to-Music/Sound binding implementations
│  │  ├─ 📁 audiocraft/
│  │  ├─ 📁 bark/
│  │  └─ 📁 lollms/
│  ├─ 📁 tts_bindings/           # Text-to-Speech binding implementations
│  │  ├─ 📁 bark/
│  │  ├─ 📁 lollms/
│  │  ├─ 📁 piper/
│  │  └─ 📁 xtts/
│  ├─ 📁 ttv_bindings/           # Text-to-Video binding implementations
│  │  └─ 📁 lollms/
│  ├─ 📄 __init__.py             # Makes lollms_client a package, exports key classes
│  ├─ 📄 lollms_config.py        # Configuration classes (BaseConfig, ConfigTemplate)
│  ├─ 📄 lollms_core.py          # LollmsClient class, main orchestrator
│  ├─ 📄 lollms_discussion.py    # LollmsDiscussion and LollmsMessage classes
│  ├─ 📄 lollms_functions.py     # FunctionCalling_Library for LLM function calls
│  ├─ 📄 lollms_js_analyzer.py   # Utility for analyzing JS code (AI documentation helper)
│  ├─ 📄 lollms_llm_binding.py   # ABC for LLM bindings and LollmsLLMBindingManager
│  ├─ 📄 lollms_python_analyzer.py # Utility for analyzing Python code (AI documentation helper)
│  ├─ 📄 lollms_stt_binding.py   # ABC for STT bindings and LollmsSTTBindingManager
│  ├─ 📄 lollms_tasks.py         # TasksLibrary for high-level operations
│  ├─ 📄 lollms_tti_binding.py   # ABC for TTI bindings and LollmsTTIBindingManager
│  ├─ 📄 lollms_ttm_binding.py   # ABC for TTM bindings and LollmsTTMBindingManager
│  ├─ 📄 lollms_tts_binding.py   # ABC for TTS bindings and LollmsTTSBindingManager
│  ├─ 📄 lollms_ttv_binding.py   # ABC for TTV bindings and LollmsTTVBindingManager
│  ├─ 📄 lollms_types.py         # Enums (MSG_TYPE, ELF_COMPLETION_FORMAT, etc.)
│  └─ 📄 lollms_utilities.py     # Helper functions (e.g., PromptReshaper, encode_image)
├─ 📁 lollms_client.egg-info/   # Packaging metadata (generated during build/install)
├─ 📄 CHANGELOG.md              # Log of changes across versions
├─ 📄 DOC_DEV.md                # This developer documentation file
├─ 📄 DOC_USE.md                # User-focused documentation
├─ 📄 log.log                   # General log file (can be ignored by git)
├─ 📄 pyproject.toml            # Modern Python packaging configuration
├─ 📄 README.md                 # Project overview for users
└─ 📄 requirements.txt          # Core dependencies
```

## 4. Core Concepts & Architecture

### `LollmsClient` (The Orchestrator)

Located in `lollms_client/lollms_core.py`, the `LollmsClient` class is the main entry point for users of the library. Its responsibilities include:
*   Initializing and managing different types of bindings (LLM, TTS, TTI, STT, TTM, TTV).
*   Providing a unified API for common operations like text generation (`generate_text`), tokenization, speech synthesis (`tts.generate_audio`), etc., by delegating calls to the active binding for that modality.
*   Storing default generation parameters that can be overridden on a per-call basis.
*   Handling basic prompt formatting attributes (though more complex templating might be in `TasksLibrary` or specific bindings).

### Bindings (The Backends)

Bindings are the heart of `lollms_client`'s modularity. Each binding provides an implementation for a specific AI backend or library for a particular modality.

#### Abstract Base Classes (ABCs)
For each modality, there's an Abstract Base Class (ABC) defining the interface that concrete bindings must implement. These are found in:
*   `lollms_llm_binding.py` (for `LollmsLLMBinding`)
*   `lollms_tts_binding.py` (for `LollmsTTSBinding`)
*   `lollms_tti_binding.py` (for `LollmsTTIBinding`)
*   `lollms_stt_binding.py` (for `LollmsSTTBinding`)
*   `lollms_ttm_binding.py` (for `LollmsTTMBinding`)
*   `lollms_ttv_binding.py` (for `LollmsTTVBinding`)

These ABCs ensure that all bindings for a given modality offer a consistent set of methods (e.g., all LLM bindings must have `generate_text`, `tokenize`, `detokenize`).

#### Binding Managers
Associated with each ABC is a Binding Manager class (e.g., `LollmsLLMBindingManager`). Their roles are:
*   **Discovery:** Dynamically find available binding implementations within their respective subdirectories (e.g., `llm_bindings/`).
*   **Instantiation:** Create instances of specific bindings when requested by `LollmsClient`.

This design allows new bindings to be added simply by creating a new subdirectory and implementing the required class, without modifying the core client or manager code.

### Supported Modalities and Bindings

#### LLM (Large Language Model) Bindings
*   **Directory:** `lollms_client/llm_bindings/`
*   **ABC:** `LollmsLLMBinding`
*   **Manager:** `LollmsLLMBindingManager`
*   **Implemented Bindings (Examples):**
    *   `lollms`: Connects to a running LOLLMS WebUI server.
    *   `ollama`: Connects to a local Ollama server.
    *   `openai`: Uses the OpenAI API (or compatible endpoints).
    *   `llamacpp` (server): Interacts with a `llama.cpp` server instance.
    *   `pythonllamacpp`: Uses the `llama-cpp-python` library for local GGUF model execution.
    *   `transformers`: Uses the Hugging Face `transformers` library for local model execution.
    *   `vllm`: Uses the vLLM library for optimized local inference (primarily NVIDIA GPUs).
    *   `tensor_rt`: (Placeholder, likely also using vLLM or TensorRT-LLM backend)

#### TTS (Text-to-Speech) Bindings
*   **Directory:** `lollms_client/tts_bindings/`
*   **ABC:** `LollmsTTSBinding`
*   **Manager:** `LollmsTTSBindingManager`
*   **Implemented Bindings:**
    *   `lollms`: Connects to a LOLLMS server's TTS service.
    *   `bark`: Uses the `suno/bark` model via `transformers` for local TTS.
    *   `xtts`: Uses Coqui AI's XTTSv2 model via the `TTS` library for local, high-quality voice cloning TTS.
    *   `piper`: Uses Piper TTS (`piper-tts` library) for fast, local, lightweight TTS.

#### TTI (Text-to-Image) Bindings
*   **Directory:** `lollms_client/tti_bindings/`
*   **ABC:** `LollmsTTIBinding`
*   **Manager:** `LollmsTTIBindingManager`
*   **Implemented Bindings:**
    *   `lollms`: Connects to a LOLLMS server's TTI service. (Others can be added, e.g., for Automatic1111, ComfyUI, local Diffusers).

#### STT (Speech-to-Text) Bindings
*   **Directory:** `lollms_client/stt_bindings/`
*   **ABC:** `LollmsSTTBinding`
*   **Manager:** `LollmsSTTBindingManager`
*   **Implemented Bindings:**
    *   `lollms`: Connects to a LOLLMS server's STT service.
    *   `whisper`: Uses OpenAI's Whisper model locally via `openai-whisper` library.
    *   `whispercpp`: Uses `whisper.cpp` command-line tool for local STT.

#### TTM (Text-to-Music/Sound) Bindings
*   **Directory:** `lollms_client/ttm_bindings/`
*   **ABC:** `LollmsTTMBinding`
*   **Manager:** `LollmsTTMBindingManager`
*   **Implemented Bindings:**
    *   `lollms`: Connects to a LOLLMS server's TTM service (placeholder).
    *   `audiocraft`: Uses Meta's AudioCraft (MusicGen) locally via `audiocraft` library.
    *   `bark`: (Can also be used here if prompted for music/sound, though its TTS binding is more common).

#### TTV (Text-to-Video) Bindings
*   **Directory:** `lollms_client/ttv_bindings/`
*   **ABC:** `LollmsTTVBinding`
*   **Manager:** `LollmsTTVBindingManager`
*   **Implemented Bindings:**
    *   `lollms`: Connects to a LOLLMS server's TTV service (placeholder).

### `TasksLibrary` (High-Level Operations)

Located in `lollms_client/lollms_tasks.py`, the `TasksLibrary` provides higher-level functions that often combine multiple calls to the `LollmsClient` or its bindings. Examples include:
*   `generate_code`, `generate_codes`: Structured code generation with template support.
*   `yes_no`, `multichoice_question`: Interpreting user responses or making decisions.
*   `sequential_summarize`, `deep_analyze`: Complex text processing tasks.
*   `generate_with_function_calls`: LLM function calling orchestration.

It relies on the `LollmsClient` instance passed to it for the actual AI operations.

### `LollmsDiscussion` & `LollmsMessage`

Found in `lollms_client/lollms_discussion.py`, these classes manage chat history:
*   `LollmsMessage`: Represents a single message with sender, content, and ID.
*   `LollmsDiscussion`: Holds a list of `LollmsMessage` objects, can format the discussion for an LLM prompt (respecting context limits), and save/load discussions.

### Configuration (`lollms_config.py`)

This file defines `BaseConfig` and `ConfigTemplate` (and `TypedConfig`), which are used by bindings or applications to manage their settings. They provide a structured way to define, load, and save configurations, often from YAML files.

### Utilities & Types

*   `lollms_utilities.py`: Contains miscellaneous helper functions like `PromptReshaper` for dynamic prompt construction, `encode_image` for preparing images for multimodal models, and text processing utilities.
*   `lollms_types.py`: Defines important enumerations like `MSG_TYPE` (for streaming callbacks), `ELF_COMPLETION_FORMAT`, `SENDER_TYPES`, etc., ensuring consistency across the library.

## 5. Adding New Bindings (A Practical Guide)

Contributing a new binding is a great way to enhance `lollms_client`. Here's a general process, taking a new hypothetical TTS binding named `mycooltts` as an example:

### Step 1: Create the Binding Directory

Create a new subdirectory within the appropriate modality's binding directory.
For our TTS example: `lollms_client/tts_bindings/mycooltts/`

### Step 2: Create `__init__.py`

Inside the new directory (`mycooltts/`), create an `__init__.py` file. This file will contain your binding's implementation.

### Step 3: Define `BindingName`

At the top of your `mycooltts/__init__.py`, define a `BindingName` variable. This string is used by the Binding Manager to identify your binding's main class.
```python
# lollms_client/tts_bindings/mycooltts/__init__.py
BindingName = "MyCoolTTSBinding" # Must match your class name
```

### Step 4: Implement the Binding Class

Create a class that inherits from the relevant ABC (e.g., `LollmsTTSBinding`).
```python
from lollms_client.lollms_tts_binding import LollmsTTSBinding
from typing import Optional, List, Union, Path
from ascii_colors import ASCIIColors, trace_exception

# (Import necessary libraries for your MyCoolTTS API/SDK)
# import my_cool_tts_sdk

BindingName = "MyCoolTTSBinding"

class MyCoolTTSBinding(LollmsTTSBinding):
    def __init__(self,
                 # Common parameters (may not all be used by every binding)
                 host_address: Optional[str] = None,
                 model_name: Optional[str] = None, # e.g., a default voice ID for MyCoolTTS
                 service_key: Optional[str] = None, # API key for MyCoolTTS
                 verify_ssl_certificate: bool = True,
                 # MyCoolTTS specific parameters
                 custom_mycooltts_param: str = "default_value",
                 **kwargs): # Catch-all

        super().__init__(binding_name="mycooltts") # Pass your binding's internal name

        ASCIIColors.info(f"Initializing MyCoolTTSBinding...")
        self.api_key = service_key
        self.mycooltts_param = custom_mycooltts_param
        self.model_name = model_name # Store default voice/model

        # Example: Initialize the SDK client
        # try:
        #     self.client = my_cool_tts_sdk.Client(api_key=self.api_key)
        #     ASCIIColors.green("MyCoolTTS SDK initialized successfully.")
        # except Exception as e:
        #     ASCIIColors.error(f"Failed to initialize MyCoolTTS SDK: {e}")
        #     trace_exception(e)
        #     raise RuntimeError("MyCoolTTS SDK initialization failed") from e
        ASCIIColors.info("MyCoolTTSBinding (Placeholder): Initialization complete.")


    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        if not self.api_key: # Example check
            raise ValueError("API key is required for MyCoolTTS.")

        effective_voice = voice if voice is not None else self.model_name
        if not effective_voice:
            raise ValueError("A voice/model must be specified for MyCoolTTS.")

        ASCIIColors.info(f"MyCoolTTS: Generating audio for '{text[:30]}...' using voice '{effective_voice}'")
        try:
            # --- YOUR MyCoolTTS API/SDK CALL GOES HERE ---
            # Example placeholder:
            # raw_audio_data = self.client.synthesize(
            #     text=text,
            #     voice_id=effective_voice,
            #     output_format="wav_bytes", # Assuming SDK can return bytes
            #     sample_rate=kwargs.get("sample_rate", 22050)
            # )
            # if not raw_audio_data:
            #    raise RuntimeError("MyCoolTTS returned no audio data.")
            # return raw_audio_data
            # --- END OF YOUR API/SDK CALL ---
            ASCIIColors.warning("MyCoolTTSBinding generate_audio is a placeholder.")
            return f"Spoken by MyCoolTTS ({effective_voice}): {text}".encode('utf-8') # Placeholder
        except Exception as e:
            ASCIIColors.error(f"MyCoolTTS audio generation failed: {e}")
            trace_exception(e)
            raise RuntimeError(f"MyCoolTTS audio generation error: {e}") from e

    def list_voices(self, **kwargs) -> List[str]:
        ASCIIColors.info("MyCoolTTS: Listing available voices...")
        try:
            # --- YOUR MyCoolTTS API/SDK CALL TO LIST VOICES ---
            # Example placeholder:
            # available_voices = self.client.list_available_voices() # Returns list of voice IDs/names
            # return [v.id for v in available_voices]
            # --- END OF YOUR API/SDK CALL ---
            ASCIIColors.warning("MyCoolTTSBinding list_voices is a placeholder.")
            return ["mycool_voice_1", "mycool_voice_2", self.model_name or "default_placeholder_voice"] # Placeholder
        except Exception as e:
            ASCIIColors.error(f"Failed to list MyCoolTTS voices: {e}")
            trace_exception(e)
            return []
```

**Implement Abstract Methods:** You *must* implement all `@abstractmethod`s defined in the corresponding ABC (e.g., `generate_audio`, `list_voices` for `LollmsTTSBinding`).

### Step 5: Handle Dependencies

If your binding requires specific Python packages:
*   At the top of your `__init__.py`, use `pipmaster` to ensure they are installed. This is the preferred way within `lollms_client`.
    ```python
    _mycooltts_sdk_installed = False
    _install_error = ""
    try:
        import pipmaster as pm
        pm.ensure_packages(["my-cool-tts-sdk", "another-dependency"])
        import my_cool_tts_sdk # Now try to import
        _mycooltts_sdk_installed = True
    except Exception as e:
        _install_error = str(e)
        # Set imported modules to None if import fails after attempt
        my_cool_tts_sdk = None

    # ... later in __init__
    if not _mycooltts_sdk_installed:
        raise ImportError(f"MyCoolTTS binding dependencies not met. Error: {_install_error}")
    ```
If your binding relies on external command-line tools (like `whisper.cpp` or `ffmpeg`), check for their existence using `shutil.which()` in your `__init__` method and raise an informative error or warning if not found.

### Step 6: Add a Test Block

Include an `if __name__ == "__main__":` block at the end of your `mycooltts/__init__.py` for standalone testing of your binding. This helps in development and debugging.
```python
if __name__ == "__main__":
    ASCIIColors.yellow("--- MyCoolTTSBinding Test ---")
    # Configure parameters for your test
    # Ensure API keys or model paths are set correctly for testing,
    # possibly via environment variables or dummy files.
    try:
        # Example config (replace with actual needed params for MyCoolTTS)
        # config = {
        #     "service_key": os.environ.get("MYCOOLTTS_API_KEY", "YOUR_FALLBACK_KEY_FOR_TESTING"),
        #     "model_name": "default_voice_for_test",
        #     "custom_mycooltts_param": "test_value"
        # }
        # binding = MyCoolTTSBinding(**config)
        
        # Placeholder test if no real config
        binding = MyCoolTTSBinding(service_key="test_key", model_name="test_voice")

        ASCIIColors.cyan("\n--- Listing Voices ---")
        voices = binding.list_voices()
        print(f"Available voices: {voices}")

        if voices:
            text_to_speak = "Hello from the My Cool TTS binding!"
            ASCIIColors.cyan(f"\n--- Generating Audio for: '{text_to_speak}' (Voice: {voices[0]}) ---")
            audio_bytes = binding.generate_audio(text_to_speak, voice=voices[0])

            if audio_bytes:
                output_path = Path("./test_mycooltts_output.wav") # Or other format
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)
                ASCIIColors.green(f"Test audio saved to: {output_path} ({len(audio_bytes)/1024:.2f} KB)")
            else:
                ASCIIColors.error("Audio generation returned empty bytes.")
        else:
            ASCIIColors.warning("No voices listed, skipping audio generation test.")

    except ImportError as e:
        ASCIIColors.error(f"Test failed due to import error: {e}")
    except RuntimeError as e:
        ASCIIColors.error(f"Test failed due to runtime error: {e}")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred during test: {e}")
        trace_exception(e)

    ASCIIColors.yellow("\n--- MyCoolTTSBinding Test Finished ---")
```

Once these steps are done, your new binding should be discoverable by `LollmsClient` when it initializes the corresponding Binding Manager.

## 6. Running Examples & Tests

*   **Examples:** The `examples/` directory contains various scripts demonstrating how to use `LollmsClient` and its features. To run an example:
    ```bash
    python examples/some_example_folder/example_script.py --help # To see its options
    python examples/some_example_folder/example_script.py [options]
    ```
*   **Binding Self-Tests:** Most binding `__init__.py` files (e.g., `llm_bindings/ollama/__init__.py`) have an `if __name__ == "__main__":` block. You can run these directly to test that specific binding in isolation:
    ```bash
    python lollms_client/llm_bindings/ollama/__init__.py
    python lollms_client/tts_bindings/bark/__init__.py
    ```
    You might need to configure model paths or API keys within these test blocks or via environment variables for them to run successfully.
*   **Adding Formal Tests:** Contributions of unit tests (e.g., using `pytest`) are highly welcome to improve code reliability. Place tests in a `tests/` directory at the root of the `lollms_client` library.

## 7. Coding Standards & Conventions

*   **PEP 8:** Follow PEP 8 guidelines for Python code style. Use a linter like `flake8` and a formatter like `black`.
*   **Type Hinting:** Use type hints for function signatures and variables where appropriate to improve code readability and maintainability.
*   **Docstrings:** Write clear and concise docstrings for all public classes, methods, and functions. Google style or NumPy style are good choices.
*   **Console Output:** Use `ascii_colors.ASCIIColors` for colored console output (e.g., `ASCIIColors.green()`, `ASCIIColors.error()`) to provide clear feedback to the user.
*   **Error Handling:** Use `try-except` blocks for operations that might fail (e.g., API calls, file I/O). Use `ascii_colors.trace_exception(e)` to print detailed error information during development.
*   **Dependency Management:** For Python package dependencies within a binding, use `pipmaster.ensure_packages()` at the beginning of the binding's `__init__.py`. For external tools, check with `shutil.which()` and provide informative messages.
*   **Logging:** For more persistent logging beyond console output, consider using the standard `logging` module.

## 8. Contribution Guidelines

We welcome contributions to `lollms_client`!

1.  **Fork the Repository:** Fork the `ParisNeo/simplified_lollms` repository on GitHub.
2.  **Clone Your Fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/simplified_lollms.git
    cd simplified_lollms/lollms_client
    ```
3.  **Create a Feature Branch:**
    ```bash
    git checkout -b my-new-feature-or-fix
    ```
4.  **Make Your Changes:** Implement your feature, fix the bug, or add your new binding.
5.  **Test Your Changes:** Run relevant examples, binding self-tests, and add new tests if applicable.
6.  **Commit Your Changes:** Write clear and concise commit messages.
    ```bash
    git add .
    git commit -m "feat: Add MyCoolTTS binding"
    # or "fix: Resolve issue in Ollama binding tokenization"
    ```
7.  **Push to Your Fork:**
    ```bash
    git push origin my-new-feature-or-fix
    ```
8.  **Create a Pull Request (PR):** Open a PR from your feature branch on your fork to the `main` branch of `ParisNeo/simplified_lollms`.
    *   Provide a clear description of your changes in the PR.
    *   Link to any relevant issues.
9.  **Code Review:** Your PR will be reviewed, and feedback may be provided. Make any necessary changes.
10. **Merge:** Once approved, your PR will be merged.

## 9. Reporting Issues

*   If you find a bug or have a feature request, please create an issue on the [GitHub Issues page](https://github.com/ParisNeo/simplified_lollms/issues).
*   Provide as much detail as possible:
    *   `lollms_client` version.
    *   Python version.
    *   Operating System.
    *   Steps to reproduce the issue.
    *   Relevant logs or error messages (use `trace_exception` for detailed tracebacks).
    *   Expected behavior vs. actual behavior.

## 10. Roadmap & Future Ideas

(This section can be expanded by the project maintainers)

*   **More Bindings:** Continuously add support for new and popular LLMs, TTS, TTI, STT, TTM, and TTV services and local libraries.
*   **Enhanced `TasksLibrary`:** Add more sophisticated pre-built tasks.
*   **Asynchronous Operations:** Explore `async/await` for non-blocking calls, especially for I/O-bound operations like API requests.
*   **Improved Error Handling & Resilience:** More robust error handling and retries in bindings.
*   **Standardized Configuration:** Further standardize configuration schemas across bindings.
*   **Comprehensive Testing Framework:** Implement more thorough unit and integration tests.
*   **Plugin System for Bindings:** Potentially make bindings even more pluggable, perhaps loadable from external packages.
*   **Documentation Generation:** Automate more of the AI-generated documentation within `ai_documentation/`.

## 11. Community & Contact

*   **GitHub Discussions:** For questions, ideas, and general discussion, use the [Discussions tab](https://github.com/ParisNeo/simplified_lollms/discussions) on the repository.
*   **Issues:** For bug reports and specific feature requests.
*   (Add links to Discord, forums, or other community channels if they exist).

Thank you for your interest in contributing to `lollms_client`! We look forward to your contributions.