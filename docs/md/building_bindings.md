## Documentation: Building LoLLMS Bindings

### 1. What is a LoLLMS Binding?

A LoLLMS binding is a Python module that acts as a standardized adapter or "bridge" between the LoLLMS client and a specific AI service or local model library. Its purpose is to translate LoLLMS's universal API calls into the specific format required by the target AI service, and then return the results in a standardized way.

This modular approach allows LoLLMS to support a vast and growing number of AI models and services without changing its core logic. By creating a new binding, you can integrate any AI service you want into the LoLLMS ecosystem.

There are six main types of bindings:
*   **LLM (Large Language Model):** For text generation, chat, tokenization, and embeddings.
*   **TTI (Text-to-Image):** For generating and editing images from text prompts.
*   **STT (Speech-to-Text):** For transcribing audio files into text.
*   **TTS (Text-to-Speech):** For synthesizing speech from text.
*   **TTV (Text-to-Video):** For generating video clips from text prompts.
*   **TTM (Text-to-Music):** For generating music tracks from text prompts.

### 2. Core Concepts and File Structure

Every binding, regardless of its type, follows the same fundamental structure. It is a Python package (a folder containing an `__init__.py` file) with a corresponding metadata file.

```
lollms_bindings/
└── my_new_binding/
    ├── __init__.py      # The core logic of your binding.
    └── description.yaml # Metadata for the UI and configuration.
```

#### `__init__.py` - The Logic Core

This file contains the main Python class that implements the binding.

*   **Inheritance:** Your class must inherit from the appropriate abstract base class (e.g., `LollmsLLMBinding`, `LollmsTTIBinding`, etc.).
*   **`BindingName` Constant:** You must define a module-level constant named `BindingName`. This string tells the LoLLMS binding manager which class within the file is the main binding class.
    ```python
    # In my_new_binding/__init__.py
    BindingName = "MyNewBindingClass"

    class MyNewBindingClass(LollmsLLMBinding):
        # ... implementation ...
    ```
*   **Dependency Management:** It is best practice to use the `pipmaster` utility to ensure any required packages are installed for the user.
    ```python
    import pipmaster as pm
    pm.ensure_packages(["requests", "Pillow"])
    ```

#### `description.yaml` - The Configuration Manifest

This YAML file describes your binding to the LoLLMS user interface. It defines the title, author, a user-friendly description, and the configuration parameters required to initialize and run your binding.

*   **`input_parameters`**: This section defines the settings for your binding's `__init__` method. These are global settings like API keys, model paths, or default model names.
*   **Method-Specific Parameters**: You can also define parameters for specific methods, which will appear in the UI when that action is used. The convention is `<method_name>_parameters` (e.g., `generate_image_parameters`, `generate_audio_parameters`).

Each parameter in any of these sections should have:
*   `name`: The programmatic name (this will be a key in the `kwargs` of the corresponding method).
*   `type`: The data type (`str`, `int`, `float`, `bool`).
*   `description`: A helpful explanation for the user.
*   `default`: A default value.
*   `options` (optional): A list of valid choices for the parameter.

### 3. How to Build an LLM Binding

**Base Class:** `lollms_client.lollms_llm_binding.LollmsLLMBinding`

*   **`__init__(self, **kwargs)`**: Initializes the client, stores API keys, model names, etc., from `input_parameters`.
*   **`chat(self, discussion: LollmsDiscussion, ...)`**: The main conversational method. Convert the `LollmsDiscussion` object into the API's required format and handle the generation.
*   **`generate_text(self, prompt: str, ...)`**: For single-shot text generation. Can often be implemented by wrapping the prompt in a minimal chat structure and calling the `chat` logic.
*   **`tokenize(self, text)` / `detokenize(self, tokens)` / `count_tokens(self, text)`**: Implement text-to-token conversions. Use the service's official tokenizer library (e.g., `tiktoken`) if available.
*   **`listModels(self)`**: Return a list of available text generation models.
*   **`embed(self, text)`**: Implement if the service has an embedding API, otherwise raise `NotImplementedError`.

### 4. How to Build a TTI (Text-to-Image) Binding

**Base Class:** `lollms_client.lollms_tti_binding.LollmsTTIBinding`

*   **`__init__(self, **kwargs)`**: Initializes the image generation client using parameters from `description.yaml`.
*   **`generate_image(self, prompt, ...)`**: Takes a prompt and image dimensions. Your job is to call the API and return the final image as `bytes`.
*   **`edit_image(self, images, prompt, ...)`**: Implements image-to-image and inpainting. Check if a `mask` is provided to determine which API endpoint to call. Return the edited image as `bytes`.
*   **`listModels(self)`**: Return a list of available image generation models.
*   **`set_settings(self, settings)`**: Allows the UI to dynamically change configuration, such as switching the active model.

### 5. How to Build an STT (Speech-to-Text) Binding

**Base Class:** `lollms_client.lollms_stt_binding.LollmsSTTBinding`

*   **`__init__(self, **kwargs)`**: Set up the STT service client, including any necessary authentication.
*   **`transcribe_audio(self, audio_path, ...)`**: This is the core method.
    *   It receives a path to an audio file (`audio_path`).
    *   You must read the audio file and send its binary data to the transcription service.
    *   The method **must** return the transcribed text as a `str`.
*   **`list_models(self)`**: Return a list of available transcription models or language options (e.g., `["whisper-1", "whisper-large-v3"]`).

### 6. How to Build a TTS (Text-to-Speech) Binding

**Base Class:** `lollms_client.lollms_tts_binding.LollmsTTSBinding`

*   **`__init__(self, **kwargs)`**: Configure the client for the speech synthesis API.
*   **`generate_audio(self, text, voice, ...)`**: The main synthesis method.
    *   It takes a `text` string and an optional `voice` identifier.
    *   You need to call the API with this text and the selected voice.
    *   The method **must** return the generated audio data (e.g., MP3, WAV) as `bytes`.
*   **`list_voices(self)`**: Return a list of available voice identifiers that the user can choose from.
*   **`list_models(self)`**: Return a list of available synthesis models (e.g., `["tts-1", "tts-1-hd"]`).

### 7. How to Build a TTV (Text-to-Video) Binding

**Base Class:** `lollms_client.lollms_ttv_binding.LollmsTTVBinding`

*   **`__init__(self, **kwargs)`**: Initialize the client for the video generation service.
*   **`generate_video(self, prompt, ...)`**: The core method for video generation.
    *   It takes a `prompt` string describing the desired video.
    *   You will call the service's API, which may be asynchronous (requiring you to poll for results).
    *   The method **must** return the final video file as `bytes` (e.g., the content of an MP4 file).
*   **`list_models(self)`**: Return a list of available video generation models or styles.

### 8. How to Build a TTM (Text-to-Music) Binding

**Base Class:** `lollms_client.lollms_ttm_binding.LollmsTTMBinding`

*   **`__init__(self, **kwargs)`**: Configure the client for the music generation API.
*   **`generate_music(self, prompt, ...)`**: The main music generation method.
    *   It takes a `prompt` describing the mood, genre, instruments, or style of music.
    *   Call the API to generate the audio.
    *   The method **must** return the generated music data as `bytes` (e.g., WAV or MP3).
*   **`list_models(self)`**: Return a list of available music generation models.

### 9. Best Practices and Final Tips

*   **API Keys:** **Never** hardcode API keys. Always load them from the configuration (`kwargs` in `__init__`) or fall back to environment variables (`os.getenv("SOME_API_KEY")`). Mark them as `is_secret: true` in `description.yaml`.
*   **Error Handling:** Wrap all API calls in `try...except` blocks. Use `ascii_colors.trace_exception(e)` to log detailed errors. For generation methods, return a structured error dictionary (e.g., `{"status": "error", "message": str(e)}`) if a call fails.
*   **Logging:** Use `ascii_colors.ASCIIColors` to print informative messages to the console (e.g., `ASCIIColors.info("Loading model...")`, `ASCIIColors.green("Generation complete.")`). This is invaluable for debugging.
*   **Testing:** Include a test block at the end of your `__init__.py` file under `if __name__ == '__main__':`. This allows you to run the file directly to test its core functionality without needing the full LoLLMS client.