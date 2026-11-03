"""
tti_bindings — LOLLMS Text-to-Image (TTI) Bindings.

Hosts concrete TTI binding implementations. Each sub-package exposes a
`LollmsTTIBinding` subclass that the `LollmsTTIBindingManager` can instantiate
by name (e.g. `"diffusers"`, `"openai"`, `"stability_ai"`).

Available bindings:
    - diffusers      : Local Hugging Face Diffusers pipelines (FLUX, SDXL, SD3)
                       served through a shared singleton FastAPI daemon.
    - gemini         : Google Gemini image generation API.
    - gguf_diffusion : Local GGUF-quantized diffusion (FLUX, Qwen-Image).
    - leonardo_ai    : Leonardo.ai cloud API.
    - lollms         : Remote LoLLMS server TTI endpoint.
    - novita_ai      : Novita.ai Ming-Image generation & layer decoupling.
    - open_router    : OpenRouter multimodal chat-completion image models.
    - openai         : OpenAI gpt-image / DALL-E family (also OpenAI-compatible
                       servers: Ollama, LM Studio, vLLM, LocalAI, …).
    - stability_ai   : Stability AI Stable Diffusion 3 / SDXL API.
    - vllm_omni      : vLLM-Omni image generation models.

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the TTI binding manager.
"""