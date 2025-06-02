# lollms_client/stt_bindings/whisper/__init__.py
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from ascii_colors import trace_exception, ASCIIColors

# --- Package Management and Conditional Imports ---
_whisper_installed = False
_whisper_installation_error = ""

try:
    import pipmaster as pm
    import platform # For OS detection for torch index

    # Determine initial device preference to guide torch installation
    preferred_torch_device_for_install = "cpu" # Default assumption
    
    # Tentatively set preference based on OS, assuming user might want GPU if available
    if platform.system() == "Linux" or platform.system() == "Windows":
        # On Linux/Windows, CUDA is the primary GPU acceleration for PyTorch.
        # We will try to install a CUDA version of PyTorch.
        preferred_torch_device_for_install = "cuda"
    elif platform.system() == "Darwin":
        # On macOS, MPS is the acceleration. Standard torch install usually handles this.
        preferred_torch_device_for_install = "mps" # or keep cpu if mps detection is later

    torch_pkgs = ["torch", "torchaudio","xformers"]
    audiocraft_core_pkgs = ["openai-whisper"]

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        # Specify a common CUDA version index. Pip should resolve the correct torch version.
        # As of late 2023/early 2024, cu118 or cu121 are common. Let's use cu126.
        # Users with different CUDA setups might need to pre-install torch manually.
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url})")
        # Install torch and torchaudio first from the specific index
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        # Then install audiocraft and other dependencies; pip should use the already installed torch
        pm.ensure_packages(audiocraft_core_pkgs)
    else:
        # For CPU, MPS, or if no specific CUDA preference was determined for install
        ASCIIColors.info("Ensuring PyTorch, AudioCraft, and dependencies using default PyPI index.")
        pm.ensure_packages(torch_pkgs + audiocraft_core_pkgs)

    import whisper
    import torch
    _whisper_installed = True
except Exception as e:
    _whisper_installation_error = str(e)
    whisper = None
    torch = None


# --- End Package Management ---

from lollms_client.lollms_stt_binding import LollmsSTTBinding

# Defines the binding name for the manager
BindingName = "WhisperSTTBinding" # Changed to avoid conflict with class name

class WhisperSTTBinding(LollmsSTTBinding):
    """
    LollmsSTTBinding implementation for OpenAI's Whisper model.
    This binding runs Whisper locally.
    Requires `ffmpeg` to be installed on the system.
    """

    # Standard Whisper model sizes
    WHISPER_MODEL_SIZES = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3"]

    def __init__(self,
                 model_name: str = "base", # Default Whisper model size
                 device: Optional[str] = None, # "cpu", "cuda", "mps", or None for auto
                 **kwargs # To catch any other LollmsSTTBinding standard args
                 ):
        """
        Initialize the Whisper STT binding.

        Args:
            model_name (str): The Whisper model size to use (e.g., "tiny", "base", "small", "medium", "large", "large-v2", "large-v3").
                              Defaults to "base".
            device (Optional[str]): The device to run the model on ("cpu", "cuda", "mps").
                                    If None, `torch` will attempt to auto-detect. Defaults to None.
        """
        super().__init__(binding_name="whisper") # Not applicable

        if not _whisper_installed:
            raise ImportError(f"Whisper STT binding dependencies not met. Please ensure 'openai-whisper' and 'torch' are installed. Error: {_whisper_installation_error}")

        self.device = device
        if self.device is None: # Auto-detect if not specified
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
                self.device = "mps"
            else:
                self.device = "cpu"
        
        ASCIIColors.info(f"WhisperSTTBinding: Using device '{self.device}'.")
        
        self.loaded_model_name = None
        self.model = None
        self._load_whisper_model(model_name)


    def _load_whisper_model(self, model_name_to_load: str):
        """Loads or reloads the Whisper model."""
        if model_name_to_load not in self.WHISPER_MODEL_SIZES:
            ASCIIColors.warning(f"'{model_name_to_load}' is not a standard Whisper model size. Attempting to load anyway. Known sizes: {self.WHISPER_MODEL_SIZES}")

        if self.model is not None and self.loaded_model_name == model_name_to_load:
            ASCIIColors.info(f"Whisper model '{model_name_to_load}' already loaded.")
            return

        ASCIIColors.info(f"Loading Whisper model: '{model_name_to_load}' on device '{self.device}'...")
        try:
            # Whisper's load_model might download the model if not already cached.
            # Cache is typically in ~/.cache/whisper
            self.model = whisper.load_model(model_name_to_load, device=self.device)
            self.loaded_model_name = model_name_to_load
            self.model_name = model_name_to_load # Update the binding's current model_name
            ASCIIColors.green(f"Whisper model '{model_name_to_load}' loaded successfully.")
        except Exception as e:
            self.model = None
            self.loaded_model_name = None
            ASCIIColors.error(f"Failed to load Whisper model '{model_name_to_load}': {e}")
            trace_exception(e)
            # Re-raise critical error for initialization or model switching
            raise RuntimeError(f"Failed to load Whisper model '{model_name_to_load}'") from e


    def transcribe_audio(self, audio_path: Union[str, Path], model: Optional[str] = None, **kwargs) -> str:
        """
        Transcribes the audio file at the given path using Whisper.

        Args:
            audio_path (Union[str, Path]): The path to the audio file to transcribe.
            model (Optional[str]): The specific Whisper model size to use.
                                  If None, uses the model loaded during initialization.
            **kwargs: Additional parameters for Whisper's transcribe method, e.g.:
                      `language` (str): Language code (e.g., "en", "fr"). If None, Whisper auto-detects.
                      `fp16` (bool): Whether to use fp16, defaults to True if CUDA available.
                      `task` (str): "transcribe" or "translate".

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If the Whisper model is not loaded or transcription fails.
            Exception: For other errors during transcription.
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        if model and model != self.loaded_model_name:
            ASCIIColors.info(f"Switching Whisper model to '{model}' for this transcription.")
            try:
                self._load_whisper_model(model) # Attempt to load the new model
            except RuntimeError as e:
                 # If switching fails, keep using the old model if available, or raise if none loaded
                if self.model is None:
                    raise RuntimeError(f"Failed to switch to Whisper model '{model}' and no model currently loaded.") from e
                else:
                    ASCIIColors.warning(f"Failed to switch to Whisper model '{model}'. Using previously loaded model '{self.loaded_model_name}'. Error: {e}")


        if self.model is None:
            raise RuntimeError("Whisper model is not loaded. Cannot transcribe.")

        # Prepare Whisper-specific options from kwargs
        whisper_options = {}
        if "language" in kwargs:
            whisper_options["language"] = kwargs["language"]
        if "fp16" in kwargs: # Typically handled by device selection, but allow override
            whisper_options["fp16"] = kwargs["fp16"]
        else: # Default fp16 based on device
            whisper_options["fp16"] = (self.device == "cuda")
        if "task" in kwargs: # "transcribe" or "translate"
            whisper_options["task"] = kwargs["task"]


        ASCIIColors.info(f"Transcribing '{audio_file.name}' with Whisper model '{self.loaded_model_name}' (options: {whisper_options})...")
        try:
            result = self.model.transcribe(str(audio_file), **whisper_options)
            transcribed_text = result.get("text", "")
            ASCIIColors.green("Transcription successful.")
            return transcribed_text.strip()
        except Exception as e:
            ASCIIColors.error(f"Whisper transcription failed for '{audio_file.name}': {e}")
            trace_exception(e)
            raise Exception(f"Whisper transcription error: {e}") from e


    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available standard Whisper model sizes.

        Args:
            **kwargs: Additional parameters (currently unused).

        Returns:
            List[str]: A list of available Whisper model size identifiers.
        """
        return self.WHISPER_MODEL_SIZES.copy() # Return a copy

    def __del__(self):
        """Clean up: Unload the model to free resources."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            ASCIIColors.info(f"WhisperSTTBinding for model '{self.loaded_model_name}' destroyed and resources released.")


# --- Main Test Block (Example Usage) ---
if __name__ == '__main__':
    if not _whisper_installed:
        print(f"{ASCIIColors.RED}Whisper dependencies not met. Skipping tests. Error: {_whisper_installation_error}{ASCIIColors.RESET}")
        print(f"{ASCIIColors.YELLOW}Please ensure 'openai-whisper' and 'torch' are installed, and 'ffmpeg' is in your system PATH.{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- WhisperSTTBinding Test ---")
    
    # --- Prerequisites for testing ---
    # 1. Create a dummy WAV file for testing, or provide a path to a real one.
    #    You'll need `scipy` to create a dummy WAV easily, or use an external tool.
    #    Let's assume a simple way to signal a missing test file.
    test_audio_file = Path("test_audio_for_whisper.wav")
    
    # Try to create a dummy file if it doesn't exist (requires scipy)
    if not test_audio_file.exists():
        try:
            import numpy as np
            from scipy.io.wavfile import write as write_wav
            samplerate = 44100; fs = 100
            t = np.linspace(0., 1., samplerate)
            amplitude = np.iinfo(np.int16).max
            data = amplitude * np.sin(2. * np.pi * fs * t)
            write_wav(test_audio_file, samplerate, data.astype(np.int16))
            ASCIIColors.green(f"Created dummy audio file: {test_audio_file}")
        except ImportError:
            ASCIIColors.warning(f"SciPy not installed. Cannot create dummy audio file.")
            ASCIIColors.warning(f"Please place a '{test_audio_file.name}' in the current directory or modify the path.")
        except Exception as e_dummy_audio:
            ASCIIColors.error(f"Could not create dummy audio file: {e_dummy_audio}")


    if not test_audio_file.exists():
        ASCIIColors.error(f"Test audio file '{test_audio_file}' not found. Skipping transcription test.")
    else:
        try:
            ASCIIColors.cyan("\n--- Initializing WhisperSTTBinding (model: 'tiny') ---")
            # Using 'tiny' model for faster testing. Change to 'base' or 'small' for better quality.
            stt_binding = WhisperSTTBinding(model_name="tiny")

            ASCIIColors.cyan("\n--- Listing available Whisper models ---")
            models = stt_binding.list_models()
            print(f"Available models: {models}")

            ASCIIColors.cyan(f"\n--- Transcribing '{test_audio_file.name}' with 'tiny' model ---")
            transcription = stt_binding.transcribe_audio(test_audio_file)
            print(f"Transcription (tiny): '{transcription}'")

            # Test with a specific language hint (if your audio is not English or for robustness)
            # ASCIIColors.cyan(f"\n--- Transcribing '{test_audio_file.name}' with 'tiny' model and language hint 'en' ---")
            # transcription_lang_hint = stt_binding.transcribe_audio(test_audio_file, language="en")
            # print(f"Transcription (tiny, lang='en'): '{transcription_lang_hint}'")

            # Test switching model dynamically (optional, will re-download/load if different)
            # ASCIIColors.cyan(f"\n--- Transcribing '{test_audio_file.name}' by switching to 'base' model ---")
            # transcription_base = stt_binding.transcribe_audio(test_audio_file, model="base")
            # print(f"Transcription (base): '{transcription_base}'")


        except ImportError as e_imp:
            ASCIIColors.error(f"Import error during test: {e_imp}")
            ASCIIColors.info("This might be due to `openai-whisper` or `torch` not being installed correctly.")
        except FileNotFoundError as e_fnf:
            ASCIIColors.error(f"File not found during test: {e_fnf}")
        except RuntimeError as e_rt:
            ASCIIColors.error(f"Runtime error during test (often model load or ffmpeg issue): {e_rt}")
            if "ffmpeg" in str(e_rt).lower():
                ASCIIColors.yellow("This error often means 'ffmpeg' is not installed or not found in your system's PATH.")
                ASCIIColors.yellow("Please install ffmpeg: https://ffmpeg.org/download.html")
        except Exception as e:
            ASCIIColors.error(f"An unexpected error occurred during testing: {e}")
            trace_exception(e)
        finally:
            # Clean up dummy audio file if we created it for this test
            # (Be careful if you are using a real test_audio_file you want to keep)
            # if "samplerate" in locals() and test_audio_file.exists(): # Simple check if we likely created it
            #     try:
            #         os.remove(test_audio_file)
            #         ASCIIColors.info(f"Removed dummy audio file: {test_audio_file}")
            #     except Exception as e_del:
            #         ASCIIColors.warning(f"Could not remove dummy audio file {test_audio_file}: {e_del}")
            pass # For this example, let's not auto-delete. User can manage it.


    ASCIIColors.yellow("\n--- WhisperSTTBinding Test Finished ---")