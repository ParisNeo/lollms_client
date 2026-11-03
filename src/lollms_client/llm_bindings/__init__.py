# lollms_client.llm_bindings package
#
# This package intentionally does NOT eagerly import any concrete binding
# submodule. Each binding is loaded on demand by the LollmsLLMBindingManager
# via importlib, which keeps `import lollms_client` fast and dependency-free.
#
# Concrete bindings live in sub-packages (e.g. `openai`, `ollama`, `vllm`, ...)
# and each exposes a `BindingName` constant plus a `LollmsLLMBinding` subclass.