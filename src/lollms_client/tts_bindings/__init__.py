"""
tts_bindings — LOLLMS Text-to-Speech (TTS) Bindings.

Hosts concrete TTS binding implementations. Each sub-package exposes a
`LollmsTTSBinding` subclass that the `LollmsTTSBindingManager` can instantiate
by name (e.g. `"xtts"`, `"piper_tts"`, `"bark"`, `"vllm_omni"`).

Available bindings:
    - bark          : Suno Bark generative TTS (shared daemon).
    - canvas_tts    : Browser-native Canvas + TTS orchestration.
    - FishSpeech    : Fish Speech (OpenAudio S1) TTS.
    - lollms        : Remote LoLLMS server TTS endpoint.
    - piper_tts     : Piper TTS (shared daemon).
    - pocket_tts    : Kyutai Pocket TTS (100M CPU-optimized).
    - vibevoice     : VibeVoice real-time TTS.
    - vllm_omni     : vLLM-Omni TTS (Voxtral, Qwen3-TTS, Fish Speech, etc.).
    - voxtral       : Voxtral local TTS via vLLM-Omni.
    - voxtral_cloud : Mistral Voxtral TTS cloud API.
    - xtts          : Coqui XTTS v2 (shared daemon).

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the TTS binding manager.
"""