# lollms_client/ttm_bindings/audiocraft/__init__.py
import io
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from ascii_colors import trace_exception, ASCIIColors

# --- Package Management and Conditional Imports ---
_audiocraft_installed_with_correct_torch = False
_audiocraft_installation_error = ""

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
    audiocraft_core_pkgs = ["audiocraft"]
    other_deps = ["scipy", "numpy"]

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        # Specify a common CUDA version index. Pip should resolve the correct torch version.
        # As of late 2023/early 2024, cu118 or cu121 are common. Let's use cu121.
        # Users with different CUDA setups might need to pre-install torch manually.
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url})")
        # Install torch and torchaudio first from the specific index
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        # Then install audiocraft and other dependencies; pip should use the already installed torch
        pm.ensure_packages(audiocraft_core_pkgs + other_deps)
    else:
        # For CPU, MPS, or if no specific CUDA preference was determined for install
        ASCIIColors.info("Ensuring PyTorch, AudioCraft, and dependencies using default PyPI index.")
        pm.ensure_packages(torch_pkgs + audiocraft_core_pkgs + other_deps)

    # Now, perform the actual imports
    import torch, torchaudio
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write # For saving to bytes
    import numpy as np
    import scipy.io.wavfile # For direct WAV manipulation if needed, though audio_write is preferred

    _audiocraft_installed_with_correct_torch = True # If imports succeed after ensure_packages
except Exception as e:
    _audiocraft_installation_error = str(e)
    # Set placeholders if imports fail
    MusicGen, torch, audio_write, np, scipy = None, None, None, None, None
# --- End Package Management ---

from lollms_client.lollms_ttm_binding import LollmsTTMBinding

BindingName = "AudioCraftTTMBinding"

# Common MusicGen model IDs from Hugging Face
DEFAULT_AUDIOCRAFT_MODELS = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-melody", # Can be conditioned on a melody audio file too
    "facebook/musicgen-large",
]


class AudioCraftTTMBinding(LollmsTTMBinding):
    def __init__(self,
                 model_name: str = "facebook/musicgen-small", # HF ID or local path
                 device: Optional[str] = None, # "cpu", "cuda", "mps", or None for auto
                 output_format: str = "wav", # 'wav', 'mp3' (mp3 needs ffmpeg via audiocraft)
                 # Catch LollmsTTMBinding standard args
                 host_address: Optional[str] = None, # Not used by local binding
                 service_key: Optional[str] = None,  # Not used by local binding
                 verify_ssl_certificate: bool = True,# Not used by local binding
                 **kwargs): # Catch-all for future compatibility or specific audiocraft params

        super().__init__(binding_name="audiocraft")

        if not _audiocraft_installed_with_correct_torch:
            raise ImportError(f"AudioCraft TTM binding dependencies not met. Please ensure 'audiocraft', 'torch', 'torchaudio', 'scipy', 'numpy' are installed. Error: {_audiocraft_installation_error}")

        self.device = device
        if self.device is None: # Auto-detect if not specified by user
            if torch.cuda.is_available():
                self.device = "cuda"
                ASCIIColors.info("CUDA device detected by PyTorch.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
                self.device = "mps"
                ASCIIColors.info("MPS device detected by PyTorch for Apple Silicon.")
            else:
                self.device = "cpu"
                ASCIIColors.info("No GPU (CUDA/MPS) detected by PyTorch, using CPU.")
        elif self.device == "cuda" and not torch.cuda.is_available():
            ASCIIColors.warning("CUDA device requested, but torch.cuda.is_available() is False. Falling back to CPU.")
            self.device = "cpu"
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            ASCIIColors.warning("MPS device requested, but not available. Falling back to CPU.")
            self.device = "cpu"


        ASCIIColors.info(f"AudioCraftTTMBinding: Using device '{self.device}'.")

        self.loaded_model_name = None
        self.model: Optional[MusicGen] = None
        self.output_format = output_format.lower()
        if self.output_format not in ["wav", "mp3"]:
            ASCIIColors.warning(f"Unsupported output_format '{self.output_format}'. Defaulting to 'wav'.")
            self.output_format = "wav"
        
        self._load_audiocraft_model(model_name)

    def _load_audiocraft_model(self, model_name_to_load: str):
        if self.model is not None and self.loaded_model_name == model_name_to_load:
            ASCIIColors.info(f"AudioCraft model '{model_name_to_load}' already loaded.")
            return

        ASCIIColors.info(f"Loading AudioCraft (MusicGen) model: '{model_name_to_load}' on device '{self.device}'...")
        try:
            self.model = MusicGen.get_pretrained(model_name_to_load, device=self.device)
            self.loaded_model_name = model_name_to_load
            # self.model_name is part of LollmsBinding base, but audiocraft uses loaded_model_name for its own logic.
            # We can assign it for consistency if needed by LollmsClient core, though it's not directly used by this binding's logic post-load.
            # self.model_name = model_name_to_load 

            ASCIIColors.green(f"AudioCraft model '{model_name_to_load}' loaded successfully.")
        except Exception as e:
            self.model = None
            self.loaded_model_name = None
            ASCIIColors.error(f"Failed to load AudioCraft model '{model_name_to_load}': {e}")
            trace_exception(e)
            raise RuntimeError(f"Failed to load AudioCraft model '{model_name_to_load}'") from e

    def generate_music(self,
                       prompt: str,
                       duration: int = 8, 
                       temperature: float = 1.0,
                       top_k: int = 250,
                       top_p: float = 0.0, 
                       cfg_coef: float = 3.0, 
                       progress: bool = True, 
                       **kwargs) -> bytes:
        if self.model is None:
            raise RuntimeError("AudioCraft model is not loaded. Cannot generate music.")

        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
            **kwargs 
        )

        ASCIIColors.info(f"Generating music for prompt: '{prompt[:50]}...' (Duration: {duration}s, Temp: {temperature}, TopK: {top_k}, TopP: {top_p}, CFG: {cfg_coef})")
        try:
            wav_tensor = self.model.generate(descriptions=[prompt], progress=progress)

            if wav_tensor is None or wav_tensor.numel() == 0:
                raise RuntimeError("MusicGen returned empty audio data.")

            if wav_tensor.ndim == 3 and wav_tensor.shape[0] == 1:
                wav_tensor_single = wav_tensor.squeeze(0)
            elif wav_tensor.ndim == 2:
                wav_tensor_single = wav_tensor
            else:
                raise ValueError(f"Unexpected tensor shape from MusicGen: {wav_tensor.shape}")

            buffer = io.BytesIO()
            dummy_filename = f"musicgen_output.{self.output_format}" # For audiocraft's format detection
            
            # audio_write needs tensor on CPU
            torchaudio.save(
                buffer,
                wav_tensor_single.cpu(),
                self.model.sample_rate,
                format="wav" # Explicitly WAV
            )
            
            audio_bytes = buffer.getvalue()
            buffer.close()

            ASCIIColors.green("Music generation successful.")
            return audio_bytes

        except Exception as e:
            ASCIIColors.error(f"AudioCraft music generation failed: {e}")
            trace_exception(e)
            # Provide more specific feedback for common issues
            if "out of memory" in str(e).lower() and self.device == "cuda":
                 ASCIIColors.yellow("CUDA out of memory. Consider using a smaller model (e.g., 'facebook/musicgen-small'), a shorter duration, or ensure your GPU has sufficient VRAM (medium models might need ~10-12GB, large ~16GB+).")
            elif "ffmpeg" in str(e).lower() and self.output_format == "mp3":
                 ASCIIColors.yellow("An FFmpeg error occurred. Ensure FFmpeg is installed and accessible in your system's PATH if you are generating MP3s.")
            raise RuntimeError(f"AudioCraft music generation error: {e}") from e

    def list_models(self, **kwargs) -> List[str]:
        return DEFAULT_AUDIOCRAFT_MODELS.copy()

    def __del__(self):
        if hasattr(self, 'model') and self.model is not None: # Check if model attribute exists
            del self.model
            self.model = None
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(self, 'loaded_model_name') and self.loaded_model_name: # Check if loaded_model_name exists
                ASCIIColors.info(f"AudioCraftTTMBinding for model '{self.loaded_model_name}' destroyed and resources released.")
            else:
                ASCIIColors.info("AudioCraftTTMBinding destroyed (no model was fully loaded or name not set).")


# --- Main Test Block (Example Usage) ---
if __name__ == '__main__':
    if not _audiocraft_installed_with_correct_torch:
        print(f"{ASCIIColors.RED}AudioCraft dependencies not met or import failed. Skipping tests. Error: {_audiocraft_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- AudioCraftTTMBinding Test ---")
    test_model_id = "facebook/musicgen-small" # Smallest model for quicker testing
    test_output_dir = Path("./test_audiocraft_output")
    test_output_dir.mkdir(exist_ok=True)
    ttm_binding = None

    try:
        ASCIIColors.cyan(f"\n--- Initializing AudioCraftTTMBinding (model: '{test_model_id}') ---")
        # Explicitly set device to CPU for basic test if no GPU, or let it auto-detect
        # device_for_test = "cpu" if not (torch and torch.cuda.is_available()) else None 
        ttm_binding = AudioCraftTTMBinding(model_name=test_model_id, output_format="wav") # device=device_for_test

        ASCIIColors.cyan("\n--- Listing common MusicGen models ---")
        models = ttm_binding.list_models()
        print(f"Common MusicGen models: {models}")

        test_prompt_1 = "A lo-fi hip hop beat with a chill piano melody and soft drums, perfect for studying."
        test_prompt_2 = "Epic orchestral score for a fantasy battle scene, with choirs and horns."

        prompts_to_test = [
            ("lofi_chill", test_prompt_1),
            ("epic_battle", test_prompt_2),
        ]

        for name, prompt in prompts_to_test:
            ASCIIColors.cyan(f"\n--- Generating music for: '{name}' (duration 3s) ---")
            print(f"Prompt: {prompt}")
            try:
                music_bytes = ttm_binding.generate_music(prompt, duration=3, progress=True) 

                if music_bytes:
                    output_filename = f"test_{name}_{test_model_id.split('/')[-1]}.{ttm_binding.output_format}"
                    output_path = test_output_dir / output_filename
                    with open(output_path, "wb") as f:
                        f.write(music_bytes)
                    ASCIIColors.green(f"Music for '{name}' saved to: {output_path} ({len(music_bytes) / 1024:.2f} KB)")
                else:
                    ASCIIColors.error(f"Music generation for '{name}' returned empty bytes.")
            except Exception as e_gen:
                ASCIIColors.error(f"Failed to generate music for '{name}': {e_gen}")
                # Error details already printed by generate_music method

    except ImportError as e_imp:
        ASCIIColors.error(f"Import error during test setup: {e_imp}")
    except RuntimeError as e_rt: # Catch runtime errors from init or generate
        ASCIIColors.error(f"Runtime error during test: {e_rt}")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred during testing: {e}")
        trace_exception(e)
    finally:
        if ttm_binding:
            del ttm_binding 
        ASCIIColors.info(f"Test artifacts (if any) are in: {test_output_dir.resolve()}")
        print(f"{ASCIIColors.YELLOW}Remember to check the audio files in '{test_output_dir.resolve()}'!{ASCIIColors.RESET}")

    ASCIIColors.yellow("\n--- AudioCraftTTMBinding Test Finished ---")