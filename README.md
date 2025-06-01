# LoLLMs Client Library

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/lollms_client.svg)](https://badge.fury.io/py/lollms_client)
[![Python Versions](https://img.shields.io/pypi/pyversions/lollms_client.svg)](https://pypi.org/project/lollms_client/)
[![Downloads](https://static.pepy.tech/personalized-badge/lollms-client?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/lollms-client)
[![Documentation - Usage](https://img.shields.io/badge/docs-Usage%20Guide-brightgreen)](DOC_USE.md)
[![Documentation - Developer](https://img.shields.io/badge/docs-Developer%20Guide-blue)](DOC_DEV.md)
[![GitHub stars](https://img.shields.io/github/stars/ParisNeo/lollms_client.svg?style=social&label=Star&maxAge=2592000)](https://github.com/ParisNeo/lollms_client/stargazers/)
[![GitHub issues](https://img.shields.io/github/issues/ParisNeo/lollms_client.svg)](https://github.com/ParisNeo/lollms_client/issues)

**`lollms_client`** is a powerful and flexible Python library designed to simplify interactions with the **LoLLMs (Lord of Large Language Models)** ecosystem and various other Large Language Model (LLM) backends. It provides a unified API for text generation, multimodal operations (text-to-image, text-to-speech, etc.), function calling, and advanced AI-driven tasks.

Whether you're connecting to a remote LoLLMs server, an Ollama instance, the OpenAI API, or running models locally using GGUF (via `llama-cpp-python` or a managed `llama.cpp` server), Hugging Face Transformers, or vLLM, `lollms-client` offers a consistent and developer-friendly experience.

## Key Features

*   🔌 **Versatile Binding System:** Seamlessly switch between different LLM backends (LoLLMs, Ollama, OpenAI, Llama.cpp, Transformers, vLLM, OpenLLM) without major code changes.
*   🗣️ **Multimodal Support:** Interact with models capable of processing images and generate various outputs like speech (TTS) and images (TTI).
*   🚀 **Streaming & Callbacks:** Efficiently handle real-time text generation with customizable callback functions.
*   🛠️ **Task-Oriented Library:** High-level `TasksLibrary` for common operations like summarization, Q&A, code generation, and structured data extraction.
*   📞 **Function Calling:** Enable LLMs to invoke your custom Python functions, bridging the gap between language models and external tools or data sources.
*   💬 **Discussion Management:** Utilities to easily manage and format conversation histories for chat applications.
*   ⚙️ **Configuration Management:** Flexible ways to configure bindings and generation parameters.
*   🧩 **Extensible:** Designed to easily incorporate new LLM backends and modality services.

## Installation

You can install `lollms_client` directly from PyPI:

```bash
pip install lollms-client
```

This will install the core library. Some bindings may require additional dependencies (e.g., `llama-cpp-python`, `torch`, `transformers`, `ollama`, `vllm`). The library attempts to manage these using `pipmaster`, but for complex dependencies (especially those requiring compilation like `llama-cpp-python` with GPU support), manual installation might be preferred.

## Quick Start

Here's a very basic example of how to use `LollmsClient` to generate text with a LoLLMs server (ensure one is running at `http://localhost:9600`):

```python
from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors

# Callback for streaming output
def simple_streaming_callback(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nStreaming Error: {chunk}")
    return True # True to continue streaming

try:
    # Initialize client to connect to a LoLLMs server
    # For other backends, change 'binding_name' and provide necessary parameters.
    # See DOC_USE.md for detailed initialization examples.
    lc = LollmsClient(
        binding_name="lollms",
        host_address="http://localhost:9600"
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
    ASCIIColors.error("Connection refused. Is the LoLLMs server running at http://localhost:9600?")
except Exception as e:
    ASCIIColors.error(f"An unexpected error occurred: {e}")

```

## Documentation

For more in-depth information, please refer to:

*   **[Usage Guide (DOC_USE.md)](DOC_USE.md):** Learn how to use `LollmsClient`, different bindings, modality features, `TasksLibrary`, and `FunctionCalling_Library` with comprehensive examples.
*   **[Developer Guide (DOC_DEV.md)](DOC_DEV.md):** Understand the architecture, how to create new bindings, and contribute to the library.

## Core Concepts

```mermaid
graph LR
    A[Your Application] --> LC[LollmsClient];

    subgraph LollmsClient_Core
        LC -- Manages --> LLB[LLM Binding];
        LC -- Provides Access To --> TL[TasksLibrary];
        LC -- Provides Access To --> FCL[FunctionCalling_Library];
        LC -- Provides Access To --> DM[DiscussionManager];
        LC -- Provides Access To --> ModalityBindings[TTS, TTI, STT etc.];
    end

    subgraph LLM_Backends
        LLB --> LollmsServer[LoLLMs Server];
        LLB --> OllamaServer[Ollama];
        LLB --> OpenAPIServer[OpenAI API];
        LLB --> LocalGGUF[Local GGUF<br>(pythonllamacpp / llamacpp server)];
        LLB --> LocalHF[Local HuggingFace<br>(transformers / vLLM)];
    end

    ModalityBindings --> ModalityServices[Modality Services<br>(e.g., LoLLMs Server TTS/TTI)];
```

*   **`LollmsClient`**: The central class for all interactions. It holds the currently active LLM binding and provides access to modality bindings and helper libraries.
*   **LLM Bindings**: These are plugins that allow `LollmsClient` to communicate with different LLM backends. You choose a binding (e.g., `"ollama"`, `"lollms"`, `"pythonllamacpp"`) when you initialize `LollmsClient`.
*   **Modality Bindings**: Similar to LLM bindings, but for services like Text-to-Speech (`tts`), Text-to-Image (`tti`), etc.
*   **`TasksLibrary`**: Offers high-level functions for common AI tasks (summarization, Q&A) built on top of `LollmsClient`.
*   **`FunctionCalling_Library`**: Enables you to define Python functions that the LLM can request to execute, allowing for tool usage.
*   **`LollmsDiscussion`**: Helps manage and format conversation histories for chat applications.

## Examples

The `examples/` directory in this repository contains a rich set of scripts demonstrating various features:
*   Basic text generation with different bindings.
*   Streaming and non-streaming examples.
*   Multimodal generation (text with images).
*   Using `TasksLibrary` for summarization and Q&A.
*   Implementing and using function calls.
*   Text-to-Speech and Text-to-Image generation.

Explore these examples to see `lollms-client` in action!

## Contributing

Contributions are welcome! Whether it's bug reports, feature suggestions, documentation improvements, or new bindings, please feel free to open an issue or submit a pull request on our [GitHub repository](https://github.com/ParisNeo/lollms_client).

## License

This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details (assuming you have a LICENSE file, if not, state "Apache 2.0 License").

## Changelog

For a list of changes and updates, please refer to the [CHANGELOG.md](CHANGELOG.md) file.