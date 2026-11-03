"""
stt_bindings — LOLLMS Speech-to-Text (STT) Bindings.

Hosts concrete STT binding implementations. Each sub-package exposes a
`LollmsSTTBinding` subclass that the `LollmsSTTBindingManager` can instantiate
by name (e.g. `"whisper"`, `"whispercpp"`, `"lollms"`).

Available bindings:
    - lollms     : Remote LoLLMS server STT endpoint.
    - whisper    : OpenAI Whisper (local, via a shared FastAPI daemon).
    - whispercpp : whisper.cpp C++ executable (local, GGUF models).

This package intentionally does not eagerly import any submodule; bindings are
loaded on demand by the STT binding manager.
"""