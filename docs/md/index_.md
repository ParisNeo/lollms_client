## DOCS FOR: `INDEX.md` (Project Documentation Index)

```markdown
# LoLLMs Client Library - Documentation Index

Welcome to the documentation for the `lollms_client` library. This index provides links to various parts of the documentation to help you find the information you need.

## I. Overview & Guides

*   **[Project README](./README.md)**: Main overview, features, quick start, and contribution guidelines.
*   **[User Guide (Full)](./ai_documentation/README.md)**: A comprehensive guide covering installation, core concepts, key methods, and usage examples.
    *   _Alternatively, if `ai_documentation/README.md` is intended to be the main project README, this link might point elsewhere or be merged._
*   **[Developer Guide](./DOC_DEV.md)**: Information on the architecture, creating new bindings, and contributing to development.
*   **[Changelog](./CHANGELOG.md)**: History of changes and new features.

## II. Core Components API Reference

*   **Core Client & Types:**
    *   **[`LollmsClient`](#lollmsclient-class-lollms_clientlollms_corepy)**: The main class for all interactions.
        *   [Initialization & Configuration](#lollmsclient-initialization--configuration)
        *   [LLM Text Generation Methods](#lollmsclient-llm-text-generation-methods)
        *   [Function Calling (MCP) Methods](#lollmsclient-function-calling-mcp-methods)
        *   [High-Level Task Methods](#lollmsclient-high-level-task-methods)
        *   [Tokenization & Embedding Methods](#lollmsclient-tokenization--embedding-methods)
        *   [Utility & Management Methods](#lollmsclient-utility--management-methods)
    *   **[`LollmsDiscussion` & `LollmsMessage`](#lollmsdiscussion--lollmsmessage-classes-lollms_clientlollms_discussionpy)**: For managing conversation context.
    *   **[Core Types (`MSG_TYPE`, `ELF_COMPLETION_FORMAT`)](./#core-types-lollms_clientlollms_typespy)**: Key enumerations.
    *   **[Configuration Utilities (`ConfigTemplate`, `BaseConfig`, `TypedConfig`)](./#configuration-utilities-lollms_clientlollms_configpy)**: For managing settings.
    *   **[General Utilities (`PromptReshaper`, `encode_image`, etc.)](./#general-utilities-lollms_clientlollms_utilitiespy)**

*   **Binding Architecture:**
    *   **LLM Bindings (`LollmsLLMBinding`, `LollmsLLMBindingManager`)**:
        *   [Abstract Base Class & Manager](#llm-bindings-lollms_clientlollms_llm_bindingpy)
        *   Implementations:
            *   [`lollms`](#lollmsllmbinding_impl-binding-lollms_clientllm_bindingslollms) (LoLLMs Server)
            *   [`ollama`](#ollamabinding-lollms_clientllm_bindingsollama)
            *   [`openai`](#openaibinding-lollms_clientllm_bindingsopenai)
            *   [`pythonllamacpp`](#pythonllamacppbinding-lollms_clientllm_bindingspythonllamacpp) (Local GGUF via `llama-cpp-python`)
            *   [`llamacpp`](#llamacppserverbinding-lollms_clientllm_bindingsllamacpp) (Managed `llama.cpp` server)
            *   [`transformers`](#huggingfacehubbinding-lollms_clientllm_bindingstransformers) (Local Hugging Face models)
            *   [`vllm`](#vllmbinding-lollms_clientllm_bindingsvllm) (Local models via vLLM)
            *   [`openllm`](#openllmbinding-lollms_clientllm_bindingsopenllm) (OpenLLM server)
            *   [`tensor_rt`](#vllmbinding-lollms_clientllm_bindingstensor_rt) (Local models via vLLM TensorRT - *Note: shares VLLMBinding structure*)
    *   **Modality Bindings**:
        *   Text-to-Speech (TTS): [`LollmsTTSBinding`](#text-to-speech-tts-bindings), `LollmsTTSBindingManager`
            *   Implementations: `lollms` (Server), `bark`, `xtts`, `piper_tts`.
        *   Text-to-Image (TTI): [`LollmsTTIBinding`](#text-to-image-tti-bindings), `LollmsTTIBindingManager`
            *   Implementations: `lollms` (Server), `dalle`, `diffusers`.
        *   Speech-to-Text (STT): [`LollmsSTTBinding`](#speech-to-text-stt-bindings), `LollmsSTTBindingManager`
            *   Implementations: `lollms` (Server), `whisper`, `whispercpp`.
        *   Text-to-Music (TTM): [`LollmsTTMBinding`](#text-to-music-ttm-bindings), `LollmsTTMBindingManager`
            *   Implementations: `lollms` (Server), `audiocraft`, `bark` (for SFX).
        *   Text-to-Video (TTV): [`LollmsTTVBinding`](#text-to-video-ttv-bindings), `LollmsTTVBindingManager`
            *   Implementations: `lollms` (Server - Placeholder).
    *   **MCP Bindings (`LollmsMCPBinding`, `LollmsMCPBindingManager`)**:
        *   [Abstract Base Class & Manager](#mcp-bindings-lollms_clientlollms_mcp_bindingpy)
        *   Implementations:
            *   [`local_mcp`](#localmcpbinding-lollms_clientmcp_bindingslocal_mcp) (Local Python tools)
                *   [Default Tools for `local_mcp`](#default-tools-for-localmcp)
            *   [`standard_mcp`](#standardmcpbinding-lollms_clientmcp_bindingsstandard_mcp) (External `stdio` MCP servers)
            *   [`remote_mcp`](#remotemcpbinding-lollms_clientmcp_bindingsremote_mcp) (Remote HTTP MCP servers - Conceptual)

## III. Code Examples

*   **[Index of Examples](./examples/README.md)**: A guide to the practical examples demonstrating various functionalities.
    *   Basic Text Generation: `simple_text_gen_test.py`, `simple_text_gen_with_image_test.py`
    *   Function Calling (MCP): `function_calling_with_local_custom_mcp.py`, `local_mcp.py`, `run_standard_mcp_example.py`, `external_mcp.py`, `openai_mcp.py`
    *   RAG: `generate_text_with_multihop_rag_example.py`, `internet_search_with_rag.py`
    *   Summarization & Analysis: `article_summary/`, `deep_analyze/`
    *   Multimodal Output: `generate_and_speak/`, `generate_game_sfx/`, `text_2_image.py`, `text_2_image_diffusers.py`
    *   Specific Binding Tests: `test_local_models/local_chat.py`, `text_gen.py`

## IV. AI-Assisted Documentation

*   **[LollmsClient Class Structure (Raw)](./ai_documentation/lollms_core_classes.md)**: AI-extracted class and method signatures for `LollmsClient`.
*   **(User Guide within `ai_documentation` can be linked here if separate from main project README)**

## V. Project Files

*   **[pyproject.toml](./pyproject.toml)**: Project build and metadata configuration.
*   **[requirements.txt](./requirements.txt)**: Core dependencies.
