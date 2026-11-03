"""
ttv_bindings — LOLLMS Text-to-Video (TTV) Bindings.

Hosts concrete TTV binding implementations. Each sub-package exposes a
`LollmsTTVBinding` subclass that the `LollmsTTVBindingManager` can instantiate
by name (e.g. `"diffusers"`, `"minimaxh3"`, `"novita_ai_ttv"`).

Available bindings:
    - diffusers        : Local Hugging Face Diffusers video pipelines
                         (LTX-Video, CogVideoX, AnimateDiff, ModelScope T2V)
                         served through a shared singleton FastAPI daemon.
    - lollms           : Remote LoLLMS server TTV endpoint (placeholder).
    - lollms_minimax_h3: Self-managed vLLM-Omni MiniMax-H3 (T2V, I2V, FL2V).
    - minimaxh3        : Shared-server MiniMax-H3 with VRAM tier auto-selection
                         (ComfyUI GGUF or Diffusers FP8 backends).
    - novita_ai_ttv    : Novita.ai cloud TTV API (Wan, Kling, PixVerse, Hunyuan,
                         Vidu, Seedance, MiniMax Hailuo).

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the TTV binding manager.
"""