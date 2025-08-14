## DOCS FOR: `lollms_client/lollms_llm_binding.py` (LLM Bindings)

### `lollms_client.lollms_llm_binding`

**Purpose:**
This module defines the abstract base class (`LollmsLLMBinding`) for all Large Language Model (LLM) bindings and a manager class (`LollmsLLMBindingManager`) to discover and instantiate these bindings. LLM bindings provide a standardized interface for `LollmsClient` to interact with various LLM backends, whether they are remote services or locally run models.

---
### `LollmsLLMBinding` (Abstract Base Class)

**Purpose:**
`LollmsLLMBinding` serves as the contract for all concrete LLM binding implementations. It ensures that each binding provides a consistent set of methods for text generation, tokenization, model information retrieval, and model loading/switching.

**Key Attributes/Properties:**

*   `binding_name` (Optional[str]): The unique name of the binding (e.g., "ollama", "openai", "pythonllamacpp").
*   `model_name` (Optional[str]): The identifier (name, path, or ID) of the currently active or default model for this binding instance. This attribute *must* be set by concrete implementations, often during `__init__` or `load_model`.

**Abstract Methods (must be implemented by subclasses):**

*   **`generate_text(...) -> Union[str, dict]`**:
    *   **Purpose**: Generates text based on a prompt, optionally with images and system instructions.
    *   **Parameters**:
        *   `prompt` (str): The main text prompt.
        *   `images` (Optional[List[str]]): List of image file paths for multimodal input.
        *   `system_prompt` (str): An optional system-level instruction.
        *   `n_predict` (Optional[int]): Maximum number of tokens to generate.
        *   `stream` (Optional[bool]): Whether to stream the output.
        *   `temperature` (Optional[float]): Sampling temperature.
        *   `top_k` (Optional[int]): Top-k sampling parameter.
        *   `top_p` (Optional[float]): Top-p sampling parameter.
        *   `repeat_penalty` (Optional[float]): Penalty for repeated tokens.
        *   `repeat_last_n` (Optional[int]): Number of previous tokens for repeat penalty.
        *   `seed` (Optional[int]): Random seed.
        *   `n_threads` (Optional[int]): Number of threads (if applicable to backend).
        *   `ctx_size` (Optional[int]): Context size for this generation.
        *   `streaming_callback` (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streamed output chunks.
        *   `split` (Optional[bool]): If `True`, treats the prompt as a structured discussion for splitting.
        *   `user_keyword` (Optional[str]), `ai_keyword` (Optional[str]): Keywords for splitting discussion prompts.
    *   **Returns**: Generated text as a string, or a dictionary with an "error" key if generation failed.

*   **`tokenize(text: str) -> list`**:
    *   **Purpose**: Converts input text into a list of tokens specific to the model used by the binding.
    *   **Parameters**: `text` (str).
    *   **Returns**: List of tokens (typically integers).

*   **`detokenize(tokens: list) -> str`**:
    *   **Purpose**: Converts a list of tokens back into human-readable text.
    *   **Parameters**: `tokens` (list).
    *   **Returns**: Detokenized string.

*   **`count_tokens(text: str) -> int`**:
    *   **Purpose**: Counts the number of tokens the input text would produce.
    *   **Parameters**: `text` (str).
    *   **Returns**: Integer count of tokens.

*   **`embed(text: str, **kwargs) -> list`**:
    *   **Purpose**: Generates an embedding vector for the input text.
    *   **Parameters**: `text` (str or List[str]), `**kwargs` for additional binding-specific options.
    *   **Returns**: List of floats representing the embedding.

*   **`get_model_info() -> dict`**:
    *   **Purpose**: Retrieves information about the currently loaded model or the binding's configuration.
    *   **Returns**: A dictionary containing details such as model name, path, loaded status, context size, supported features (e.g., vision, structured output), and any relevant configuration parameters.

*   **`listModels() -> list`**:
    *   **Purpose**: Lists models available to this binding. For server-based bindings, this queries the server. For local file-based bindings, it might scan a configured model directory.
    *   **Returns**: A list of dictionaries, where each dictionary provides information about an available model (e.g., `{'model_name': 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'path_hint': '...', 'size_gb': '...'}`).

*   **`load_model(model_name: str) -> bool`**:
    *   **Purpose**: Loads or switches to the specified model within the binding.
    *   **Parameters**: `model_name` (str): The identifier (name, path, or ID) of the model to load.
    *   **Returns**: `True` if the model was loaded successfully, `False` otherwise.

**Concrete Methods (provided by the base class):**

*   **`split_discussion(lollms_prompt_string: str, system_keyword="!@>system:", user_keyword="!@>user:", ai_keyword="!@>assistant:") -> list`**:
    *   **Purpose**: Utility method to split a LoLLMs-formatted discussion string (using `!@>role:` prefixes) into a list of OpenAI-style message dictionaries (e.g., `[{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]`).
    *   **Parameters**:
        *   `lollms_prompt_string` (str): The discussion string.
        *   `system_keyword`, `user_keyword`, `ai_keyword` (str): Customizable prefixes for roles.
    *   **Returns**: A list of message dictionaries.

**Usage Example (Conceptual - Subclassing):**
```python
from lollms_client.lollms_llm_binding import LollmsLLMBinding, MSG_TYPE
from typing import Optional, Callable, List, Union

class MyCustomLLMBinding(LollmsLLMBinding):
    def __init__(self, model_identifier: str, some_config_param: str):
        super().__init__(binding_name="my_custom_binding")
        self.model_name = model_identifier # Important: Set model_name
        self.config_param = some_config_param
        # ... further initialization for MyCustomLLMBinding ...
        if not self.load_model(model_identifier):
            raise RuntimeError(f"Failed to load model {model_identifier}")

    def load_model(self, model_name: str) -> bool:
        # Implementation to load the model (e.g., from a file or connect to a service)
        print(f"Loading model: {model_name} with config: {self.config_param}")
        self.model_name = model_name # Update current model name
        # ... actual loading logic ...
        return True # or False if failed

    def generate_text(self, prompt: str, ..., streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None) -> Union[str, dict]:
        # Implementation for text generation
        # If streaming_callback is provided and stream=True:
        #   for chunk in generated_output:
        #       if not streaming_callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
        #           break # Stop if callback returns False
        #   return full_concatenated_text
        # Else (not streaming or no callback):
        #   return full_generated_text
        pass # Replace with actual implementation

    # ... Implement other abstract methods (tokenize, detokenize, count_tokens, embed, get_model_info, listModels) ...
```

---
### `LollmsLLMBindingManager`

**Purpose:**
The `LollmsLLMBindingManager` is responsible for discovering available LLM binding implementations within a specified directory and instantiating them on demand. This allows `LollmsClient` to dynamically support various LLM backends without hardcoding them.

**Key Attributes/Properties:**

*   `llm_bindings_dir` (Path): The directory where binding implementations are stored (e.g., `lollms_client/llm_bindings/`).
*   `available_bindings` (Dict[str, type[LollmsLLMBinding]]): A dictionary caching loaded binding classes, mapping binding names to their respective classes.

**Methods:**

*   **`__init__(llm_bindings_dir: str = "llm_bindings")`**:
    *   **Purpose**: Initializes the manager.
    *   **Parameters**: `llm_bindings_dir` (str): Path to the directory containing binding modules.

*   **`_load_binding(binding_name: str)`**: (Internal method)
    *   **Purpose**: Dynamically imports the Python module for the specified `binding_name` from the `llm_bindings_dir`, retrieves the binding class (identified by a `BindingName` variable within the module's `__init__.py`), and caches it in `available_bindings`.
    *   **Parameters**: `binding_name` (str).

*   **`create_binding(binding_name: str, **kwargs) -> Optional[LollmsLLMBinding]`**:
    *   **Purpose**: Creates and returns an instance of the specified LLM binding. If the binding class hasn't been loaded yet, it calls `_load_binding` first.
    *   **Parameters**:
        *   `binding_name` (str): The name of the binding to instantiate (e.g., "ollama", "openai").
        *   `**kwargs`: Keyword arguments to be passed to the constructor of the binding class (e.g., `host_address`, `model_name`, `service_key`).
    *   **Returns**: An instance of the requested `LollmsLLMBinding` subclass, or `None` if the binding cannot be found or instantiated.

*   **`get_available_bindings() -> List[str]`**:
    *   **Purpose**: Scans the `llm_bindings_dir` for subdirectories that represent valid binding modules (i.e., contain an `__init__.py`) and returns a list of their names.
    *   **Returns**: A list of strings, where each string is a discoverable binding name.

**Usage Example (Internal to `LollmsClient`):**
```python
# Inside LollmsClient.__init__ typically:
# from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
#
# llm_bindings_directory = Path(__file__).parent / "llm_bindings"
# self.llm_binding_manager = LollmsLLMBindingManager(llm_bindings_directory)
#
# # To create a specific binding:
# binding_config = {"host_address": "http://localhost:11434", "model_name": "mistral"}
# self.llm = self.llm_binding_manager.create_binding(
#     binding_name="ollama",
#     **binding_config
# )
# if self.llm is None:
#     raise ValueError("Failed to create Ollama binding.")
```

**Developer Notes for Creating New Bindings:**
1.  Create a new subdirectory under `lollms_client/llm_bindings/` (e.g., `lollms_client/llm_bindings/my_new_binding/`).
2.  Inside this new directory, create an `__init__.py` file.
3.  In `__init__.py`, define a variable `BindingName = "MyNewBindingClassName"`.
4.  Implement your binding class (e.g., `MyNewBindingClassName`) in `__init__.py` or import it from another file within the same subdirectory. This class *must* inherit from `LollmsLLMBinding` and implement all its abstract methods.
5.  The `LollmsLLMBindingManager` will then be able to discover and instantiate `MyNewBindingClassName` using the name "my\_new\_binding".
