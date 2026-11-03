"""
ttm_bindings — LOLLMS Text-to-Music (TTM) Bindings.

Hosts concrete TTM binding implementations. Each sub-package exposes a
`LollmsTTMBinding` subclass that the `LollmsTTMBindingManager` can instantiate
by name (e.g. `"diffusers"`, `"suno_ttm"`, `"replicate"`).

Available bindings:
    - audiocraft   : Meta AudioCraft MusicGen (local, PyTorch).
    - beatoven_ai  : Beatoven.ai cloud music API.
    - diffusers    : Local Hugging Face Diffusers music pipelines (MiniMax
                     Music 3, Stable Audio Open, AudioLDM 2, MusicGen) served
                     through a shared singleton FastAPI daemon.
    - lollms       : Remote LoLLMS server TTM endpoint (placeholder).
    - replicate    : Replicate-hosted music models (MusicGen, Bark, etc.).
    - stability_ai : Stability AI Stable Audio API.
    - suno_ttm     : Suno AI browser-session cookie binding.
    - topmediai    : TopMediai music generation API.

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the TTM binding manager.
"""