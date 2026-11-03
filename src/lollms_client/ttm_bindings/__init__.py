"""
ttm_bindings — LOLLMS Text-to-Music (TTM) Bindings.

Hosts concrete TTM binding implementations. Each sub-package exposes a
`LollmsTTMBinding` subclass that the `LollmsTTMBindingManager` can instantiate
by name (e.g. `"diffusers"`, `"suno_ttm"`, `"replicate"`).

Available bindings:
    - diffusers    : Local Hugging Face Diffusers music pipelines served through
                     a shared singleton FastAPI daemon.
    - lollms       : Remote LoLLMS server TTM endpoint (placeholder).
    - replicate    : Replicate-hosted music models (MusicGen, Bark, etc.).
    - stability_ai : Stability AI Stable Audio API.
    - suno_ttm     : Suno AI browser-session cookie binding.
    - topmediai    : TopMediai music generation API.
"""