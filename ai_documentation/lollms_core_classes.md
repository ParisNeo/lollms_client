# Class Information for lollms_core.py

File: `E:\test2_lollms\lollms_client\lollms_client\lollms_core.py`

## Classes

### ELF_GENERATION_FORMAT

```python
class ELF_GENERATION_FORMAT:
```

### ELF_COMPLETION_FORMAT

```python
class ELF_COMPLETION_FORMAT:
```

### LollmsClient

```python
class LollmsClient:
    def __init__(self, host_address = None, model_name = None, ctx_size = 4096, personality = -1, n_predict = 1024, min_n_predict = 512, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', tokenizer = None, default_generation_mode = ELF_GENERATION_FORMAT.LOLLMS) -> None
    def tokenize(self, prompt: str) -> None
    def detokenize(self, tokens_list: list) -> None
    def generate_with_images(self, prompt, images, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', streaming_callback = None) -> None
    def generate(self, prompt, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', streaming_callback = None) -> None
    def generate_text(self, prompt, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', streaming_callback = None) -> None
    def lollms_generate(self, prompt, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', streaming_callback = None) -> None
    def lollms_generate_with_images(self, prompt, images, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, service_key: str = '', streaming_callback = None) -> None
    def openai_generate(self, prompt, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Instruct, service_key: str = '', streaming_callback = None) -> None
    def ollama_generate(self, prompt, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Instruct, service_key: str = '', streaming_callback = None) -> None
    def litellm_generate(self, prompt, host_address = None, model_name = None, personality = None, n_predict = None, stream = False, temperature = 0.7, top_k = 50, top_p = 0.95, repeat_penalty = 0.8, repeat_last_n = 40, seed = None, n_threads = 8, completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Instruct, service_key: str = '', streaming_callback = None) -> None
    def listMountedPersonalities(self, host_address: str = None) -> None
    def list_models(self, host_address: str = None) -> None
```

