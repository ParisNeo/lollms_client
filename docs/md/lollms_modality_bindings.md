## DOCS FOR: Modality Bindings (TTS, TTI, STT, TTM, TTV)

**Purpose:**
This section covers the abstract base classes and manager classes for various modality bindings supported by `lollms_client`. These bindings allow interaction with services or local models for Text-to-Speech (TTS), Text-to-Image (TTI), Speech-to-Text (STT), Text-to-Music (TTM), and Text-to-Video (TTV) operations. Each modality follows a similar pattern: an abstract base class defining the interface and a manager class for discovering and instantiating concrete binding implementations.

---
### 1. Text-to-Speech (TTS) Bindings

**Module:** `lollms_client.lollms_tts_binding`

#### `LollmsTTSBinding` (Abstract Base Class)

*   **Purpose**: Defines the standard interface for TTS bindings.
*   **Key Attributes**:
    *   `binding_name` (str): Name of the TTS binding.
*   **Abstract Methods**:
    *   `generate_audio(text: str, voice: Optional[str] = None, **kwargs) -> bytes`:
        *   **Purpose**: Converts text to audio.
        *   **Parameters**:
            *   `text` (str): The text to synthesize.
            *   `voice` (Optional[str]): Identifier for the voice or model to use (binding-specific).
            *   `**kwargs`: Additional binding-specific parameters.
        *   **Returns**: `bytes` representing the audio data (e.g., WAV, MP3).
    *   `list_voices(**kwargs) -> List[str]`:
        *   **Purpose**: Lists available voices or TTS models for the binding.
        *   **Parameters**: `**kwargs` for binding-specific options.
        *   **Returns**: `List[str]` of voice/model identifiers.

#### `LollmsTTSBindingManager`

*   **Purpose**: Discovers and instantiates `LollmsTTSBinding` implementations.
*   **Key Attributes**:
    *   `tts_bindings_dir` (Path): Directory where TTS binding modules are located (e.g., `lollms_client/tts_bindings/`).
*   **Methods**:
    *   `__init__(tts_bindings_dir: Union[str, Path])`
    *   `create_binding(binding_name: str, **kwargs) -> Optional[LollmsTTSBinding]`: Creates a TTS binding instance.
    *   `get_available_bindings() -> List[str]`: Lists discoverable TTS binding names.

#### Concrete TTS Implementations (Examples):

*   **`lollms_client.tts_bindings.lollms.LollmsTTSBinding_Impl`**:
    *   **Purpose**: Interacts with a TTS service endpoint on a LoLLMs server.
    *   **Key `__init__` Parameters**: `host_address`, `model_name` (default voice), `service_key` (client ID).
    *   **`generate_audio` Notes**: Sends text and voice preference to the server; server typically saves the file and returns status. Client binding returns `b""`.
    *   **`list_voices` Notes**: Queries the LoLLMs server for available voices.

*   **`lollms_client.tts_bindings.bark.BarkTTSBinding`**:
    *   **Purpose**: Uses the `suno/bark` model locally via `transformers`.
    *   **Key `__init__` Parameters**: `model_name` (Bark model ID, e.g., "suno/bark-small"), `default_voice` (Bark voice preset, e.g., "v2/en_speaker_6"), `device`.
    *   **`generate_audio` Notes**: `voice` parameter corresponds to Bark's `voice_preset`.
    *   **`list_voices` Notes**: Returns a list of known Bark voice presets.

*   **`lollms_client.tts_bindings.xtts.XTTSBinding`**:
    *   **Purpose**: Uses Coqui XTTS models locally via the `TTS` library.
    *   **Key `__init__` Parameters**: `model_name` (Coqui TTS model ID, e.g., "tts_models/multilingual/multi-dataset/xtts_v2"), `default_speaker_wav` (path to reference audio for default voice cloning), `default_language`, `device`.
    *   **`generate_audio` Notes**: `voice` parameter is the path to a speaker WAV file for voice cloning. `language` parameter specifies the language.
    *   **`list_voices` Notes**: Indicates that voices are dynamic based on `speaker_wav`.

*   **`lollms_client.tts_bindings.piper_tts.PiperTTSBinding`**:
    *   **Purpose**: Uses Piper TTS models (.onnx) locally via the `piper-tts` library.
    *   **Key `__init__` Parameters**: `default_voice_model_path` (path to default .onnx voice file), `piper_voices_dir` (directory to scan for `.onnx` voice files).
    *   **`generate_audio` Notes**: `voice` parameter is the filename of the `.onnx` voice model (e.g., "en_US-lessac-medium.onnx") to be found in `piper_voices_dir` or as an absolute path.
    *   **`list_voices` Notes**: Scans `piper_voices_dir` for available `.onnx` voice files.

---
### 2. Text-to-Image (TTI) Bindings

**Module:** `lollms_client.lollms_tti_binding`

#### `LollmsTTIBinding` (Abstract Base Class)

*   **Purpose**: Defines the standard interface for TTI bindings.
*   **Key Attributes**:
    *   `binding_name` (str): Name of the TTI binding.
*   **Abstract Methods**:
    *   `generate_image(prompt: str, negative_prompt: Optional[str] = "", width: int = 512, height: int = 512, **kwargs) -> bytes`:
        *   **Purpose**: Generates an image from text.
        *   **Parameters**: `prompt` (str), `negative_prompt` (Optional[str]), `width` (int), `height` (int), `**kwargs` for binding-specific options (e.g., seed, steps).
        *   **Returns**: `bytes` representing the image data (e.g., PNG, JPEG).
    *   `list_services(**kwargs) -> List[Dict[str, str]]`:
        *   **Purpose**: Lists available TTI models or services.
        *   **Parameters**: `**kwargs` (e.g., `client_id` for LoLLMs server TTI).
        *   **Returns**: `List[Dict[str, str]]` of service descriptions.
    *   `get_settings(**kwargs) -> Optional[Dict[str, Any]]`:
        *   **Purpose**: Retrieves current settings for the active TTI service.
        *   **Parameters**: `**kwargs`.
        *   **Returns**: `Optional[Dict[str, Any]]` representing settings, often in a `ConfigTemplate`-like list format.
    *   `set_settings(settings: Dict[str, Any], **kwargs) -> bool`:
        *   **Purpose**: Applies new settings to the TTI service.
        *   **Parameters**: `settings` (Dict), `**kwargs`.
        *   **Returns**: `True` if successful.

#### `LollmsTTIBindingManager`

*   **Purpose**: Discovers and instantiates `LollmsTTIBinding` implementations.
*   **Key Attributes**:
    *   `tti_bindings_dir` (Path): Directory for TTI binding modules (e.g., `lollms_client/tti_bindings/`).
*   **Methods**:
    *   `__init__(tti_bindings_dir: Union[str, Path])`
    *   `create_binding(binding_name: str, **kwargs) -> Optional[LollmsTTIBinding]`: Creates a TTI binding instance.
    *   `get_available_bindings() -> List[str]`: Lists discoverable TTI binding names.

#### Concrete TTI Implementations (Examples):

*   **`lollms_client.tti_bindings.lollms.LollmsTTIBinding_Impl`**:
    *   **Purpose**: Interacts with TTI service endpoints on a LoLLMs server.
    *   **Key `__init__` Parameters**: `host_address`, `service_key` (client ID).
    *   **Methods Notes**: `list_services`, `get_settings`, `set_settings` require `client_id` to be passed via `kwargs` or set via `service_key` at init.

*   **`lollms_client.tti_bindings.dalle.DalleTTIBinding_Impl`**:
    *   **Purpose**: Uses OpenAI's DALL-E API.
    *   **Key `__init__` Parameters**: `api_key` (OpenAI API key), `model_name` ("dall-e-2" or "dall-e-3"), `default_size`, `default_quality`, `default_style`.
    *   **`generate_image` Notes**: Handles DALL-E 2 vs DALL-E 3 differences in prompt/negative prompt handling.
    *   **`list_services` Notes**: Returns predefined DALL-E model options.
    *   `get_settings`/`set_settings`: Manage instance defaults for size, quality, style.

*   **`lollms_client.tti_bindings.diffusers.DiffusersTTIBinding_Impl`**:
    *   **Purpose**: Uses Hugging Face Diffusers library for local model inference.
    *   **Key `__init__` Parameters**: `config` (dict for various Diffusers settings like `model_id_or_path`, `device`, `torch_dtype_str`, `scheduler_name`), `lollms_paths` (for model/cache dirs).
    *   **`generate_image` Notes**: Constructs parameters for the Diffusers pipeline.
    *   **`list_services` Notes**: Describes the currently loaded Diffusers model.
    *   `get_settings`/`set_settings`: Manage detailed Diffusers pipeline configurations. Some settings may trigger a model reload.

---
### 3. Speech-to-Text (STT) Bindings

**Module:** `lollms_client.lollms_stt_binding`

#### `LollmsSTTBinding` (Abstract Base Class)

*   **Purpose**: Defines the standard interface for STT bindings.
*   **Key Attributes**:
    *   `binding_name` (str): Name of the STT binding.
*   **Abstract Methods**:
    *   `transcribe_audio(audio_path: Union[str, Path], model: Optional[str] = None, **kwargs) -> str`:
        *   **Purpose**: Transcribes an audio file to text.
        *   **Parameters**: `audio_path` (Union[str, Path]), `model` (Optional[str] STT model identifier), `**kwargs` (e.g., language hint).
        *   **Returns**: Transcribed text as a string.
    *   `list_models(**kwargs) -> List[str]`:
        *   **Purpose**: Lists available STT models.
        *   **Parameters**: `**kwargs`.
        *   **Returns**: `List[str]` of model identifiers.

#### `LollmsSTTBindingManager`

*   **Purpose**: Discovers and instantiates `LollmsSTTBinding` implementations.
*   **Key Attributes**:
    *   `stt_bindings_dir` (Path): Directory for STT binding modules.
*   **Methods**:
    *   `__init__(stt_bindings_dir: Union[str, Path])`
    *   `create_binding(binding_name: str, **kwargs) -> Optional[LollmsSTTBinding]`: Creates an STT binding instance.
    *   `get_available_bindings() -> List[str]`: Lists discoverable STT binding names.

#### Concrete STT Implementations (Examples):

*   **`lollms_client.stt_bindings.lollms.LollmsSTTBinding_Impl`**:
    *   **Purpose**: Interacts with an STT service endpoint on a LoLLMs server.
    *   **Key `__init__` Parameters**: `host_address`, `model_name`, `service_key`.
    *   **`transcribe_audio` Notes**: Sends audio data (base64 encoded) to the server.

*   **`lollms_client.stt_bindings.whisper.WhisperSTTBinding`**:
    *   **Purpose**: Uses OpenAI's Whisper model locally via the `openai-whisper` library.
    *   **Key `__init__` Parameters**: `model_name` (Whisper model size, e.g., "base"), `device`.
    *   **`transcribe_audio` Notes**: Requires `ffmpeg` to be installed for audio processing.
    *   **`list_models` Notes**: Returns standard Whisper model sizes.

*   **`lollms_client.stt_bindings.whispercpp.WhisperCppSTTBinding`**:
    *   **Purpose**: Uses GGUF-quantized Whisper models locally via the `whisper.cpp` C++ executable.
    *   **Key `__init__` Parameters**: `model_path` (path to GGUF Whisper model), `whispercpp_exe_path` (path to `whisper.cpp` executable), `ffmpeg_path`, `models_search_path`, `default_language`, `n_threads`.
    *   **`transcribe_audio` Notes**: Converts audio to 16kHz mono WAV using `ffmpeg` then processes with `whisper.cpp` executable.
    *   **`list_models` Notes**: Lists the default model and scans `models_search_path` for GGUF files.

---
### 4. Text-to-Music (TTM) Bindings

**Module:** `lollms_client.lollms_ttm_binding`

#### `LollmsTTMBinding` (Abstract Base Class)

*   **Purpose**: Defines the standard interface for TTM bindings.
*   **Key Attributes**:
    *   `binding_name` (str): Name of the TTM binding.
*   **Abstract Methods**:
    *   `generate_music(prompt: str, **kwargs) -> bytes`:
        *   **Purpose**: Generates music from a text prompt.
        *   **Parameters**: `prompt` (str), `**kwargs` (e.g., duration, style, seed).
        *   **Returns**: `bytes` representing audio data (e.g., WAV, MP3).
    *   `list_models(**kwargs) -> List[str]`:
        *   **Purpose**: Lists available TTM models or services.
        *   **Parameters**: `**kwargs`.
        *   **Returns**: `List[str]` of model/service identifiers.

#### `LollmsTTMBindingManager`

*   **Purpose**: Discovers and instantiates `LollmsTTMBinding` implementations.
*   **Key Attributes**:
    *   `ttm_bindings_dir` (Path): Directory for TTM binding modules.
*   **Methods**:
    *   `__init__(ttm_bindings_dir: Union[str, Path])`
    *   `create_binding(binding_name: str, **kwargs) -> Optional[LollmsTTMBinding]`: Creates a TTM binding instance.
    *   `get_available_bindings() -> List[str]`: Lists discoverable TTM binding names.

#### Concrete TTM Implementations (Examples):

*   **`lollms_client.ttm_bindings.lollms.LollmsTTMBinding_Impl`**:
    *   **Purpose**: (Placeholder) Intended for a LoLLMs server TTM endpoint. Currently raises `NotImplementedError`.

*   **`lollms_client.ttm_bindings.audiocraft.AudioCraftTTMBinding`**:
    *   **Purpose**: Uses Meta's AudioCraft (MusicGen) models locally via the `audiocraft` library.
    *   **Key `__init__` Parameters**: `model_name` (MusicGen model ID, e.g., "facebook/musicgen-small"), `device`, `output_format`.
    *   **`generate_music` Notes**: `kwargs` can include `duration`, `temperature`, `cfg_coef`, etc.
    *   **`list_models` Notes**: Returns a list of default MusicGen model IDs.

*   **`lollms_client.ttm_bindings.bark.BarkTTMBinding` ( repurposed for SFX/short audio )**:
    *   **Purpose**: Uses the `suno/bark` model locally, often for sound effects (SFX) or short audio snippets rather than long-form music.
    *   **Key `__init__` Parameters**: `model_name` (Bark model ID), `device`, `default_voice_preset` (Bark voice presets can influence SFX characteristics).
    *   **`generate_music` Notes**: The `prompt` can include SFX cues like `[SFX: explosion]`. `voice_preset` can be used.
    *   **`list_models` Notes**: Returns known Bark model IDs. `list_voice_presets` lists Bark voice presets.

---
### 5. Text-to-Video (TTV) Bindings

**Module:** `lollms_client.lollms_ttv_binding`

#### `LollmsTTVBinding` (Abstract Base Class)

*   **Purpose**: Defines the standard interface for TTV bindings.
*   **Key Attributes**:
    *   `binding_name` (str): Name of the TTV binding.
*   **Abstract Methods**:
    *   `generate_video(prompt: str, **kwargs) -> bytes`:
        *   **Purpose**: Generates video from a text prompt.
        *   **Parameters**: `prompt` (str), `**kwargs` (e.g., duration, fps, style).
        *   **Returns**: `bytes` representing video data (e.g., MP4).
    *   `list_models(**kwargs) -> List[str]`:
        *   **Purpose**: Lists available TTV models or services.
        *   **Parameters**: `**kwargs`.
        *   **Returns**: `List[str]` of model/service identifiers.

#### `LollmsTTVBindingManager`

*   **Purpose**: Discovers and instantiates `LollmsTTVBinding` implementations.
*   **Key Attributes**:
    *   `ttv_bindings_dir` (Path): Directory for TTV binding modules.
*   **Methods**:
    *   `__init__(ttv_bindings_dir: Union[str, Path])`
    *   `create_binding(binding_name: str, **kwargs) -> Optional[LollmsTTVBinding]`: Creates a TTV binding instance.
    *   `get_available_bindings() -> List[str]`: Lists discoverable TTV binding names.

#### Concrete TTV Implementations (Examples):

*   **`lollms_client.ttv_bindings.lollms.LollmsTTVBinding_Impl`**:
    *   **Purpose**: (Placeholder) Intended for a LoLLMs server TTV endpoint. Currently raises `NotImplementedError`.

**Usage Example (Conceptual for a Modality Binding - e.g., TTS):**
```python
from lollms_client import LollmsClient

# Initialize LollmsClient with a TTS binding (e.g., Bark)
# Ensure LLM binding is also specified, even if not primary for this task.
try:
    lc = LollmsClient(
        binding_name="lollms", # Dummy LLM for this example
        tts_binding_name="bark",
        tts_binding_config={"model_name": "suno/bark-small"}
    )

    if lc.tts:
        print(f"Available TTS voices/presets: {lc.tts.list_voices()[:5]}")
        audio_bytes = lc.tts.generate_audio("Hello from LOLLMS Client using Bark!", voice="v2/en_speaker_1")
        with open("bark_output.wav", "wb") as f:
            f.write(audio_bytes)
        print("Generated bark_output.wav")
    else:
        print("TTS binding not available.")

except Exception as e:
    print(f"An error occurred: {e}")
```
