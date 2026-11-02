"""
ttv_bindings — LOLLMS Text-to-Video (TTV) Bindings.

Hosts concrete TTV binding implementations. Each sub-package exposes a
`LollmsTTVBinding` subclass that the `LollmsTTVBindingManager` can instantiate
by name (e.g. `"diffusers"`, `"lollms"`).

Available bindings:
    - diffusers : Local Hugging Face Diffusers pipelines (LTX-Video, CogVideoX,
                  AnimateDiff, generic DiffusionPipeline) served through a
                  shared singleton FastAPI daemon.
    - lollms    : Remote LoLLMS server TTV endpoint (placeholder).
"""