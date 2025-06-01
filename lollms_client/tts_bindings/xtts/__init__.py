# lollms_client/tts_bindings/xtts/__init__.py
import io
import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from ascii_colors import trace_exception, ASCIIColors

# --- Package Management and Conditional Imports ---
_xtts_deps_installed_with_correct_torch = False
_xtts_installation_error = ""
try:
    import pipmaster as pm
    import platform

    preferred_torch_device_for_install = "cpu"
    if platform.system() == "Linux" or platform.system() == "Windows":
        preferred_torch_device_for_install = "cuda"
    elif platform.system() == "Darwin":
        preferred_torch_device_for_install = "mps"

    torch_pkgs = ["torch", "torchaudio"] # TTS often needs torchaudio
    # Coqui-TTS has specific version requirements sometimes, ensure_packages handles this
    xtts_core_pkgs = ["TTS"]
    other_deps = ["scipy", "numpy", "soundfile"] # soundfile is often a TTS dependency

    torch_index_url = None
    if preferred_torch_device_for_install == "cuda":
        torch_index_url = "https://download.pytorch.org/whl/cu126"
        ASCIIColors.info(f"Attempting to ensure PyTorch with CUDA support (target index: {torch_index_url}) for XTTS binding.")
        pm.ensure_packages(torch_pkgs, index_url=torch_index_url)
        pm.ensure_packages(xtts_core_pkgs + other_deps)
    else:
        ASCIIColors.info("Ensuring PyTorch, Coqui-TTS, and dependencies using default PyPI index for XTTS binding.")
        pm.ensure_packages(torch_pkgs + xtts_core_pkgs + other_deps)

    import torch
    from TTS.api import TTS # Main Coqui TTS class
    import scipy.io.wavfile
    import numpy as np
    import soundfile as sf # For reading speaker_wav if not in standard wav

    _xtts_deps_installed_with_correct_torch = True
except ImportError as e_imp: # Catch ImportError specifically if TTS itself fails
    _xtts_installation_error = f"ImportError: {e_imp}. Coqui TTS (TTS lib) might not be installed correctly or has missing dependencies."
    TTS, torch, scipy, np, sf = None, None, None, None, None
except Exception as e:
    _xtts_installation_error = str(e)
    TTS, torch, scipy, np, sf = None, None, None, None, None
# --- End Package Management ---

from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "XTTSBinding"

# Common XTTS model IDs from Coqui on Hugging Face
# The primary one is usually "coqui/XTTS-v2" or similar official releases.
# Users might also point to fine-tuned versions or local paths.
XTTS_MODELS = [
    "tts_models/multilingual/multi-dataset/xtts_v2", # Standard XTTS v2 model string for Coqui TTS lib
    # "coqui/XTTS-v2" # This is the HF repo ID, TTS lib might map it or expect the above format
]

# Supported languages by XTTS v2 (example, check latest Coqui docs)
XTTS_SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

class XTTSBinding(LollmsTTSBinding):
    def __init__(self,
                 model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", # Coqui TTS model identifier
                 default_speaker_wav: Optional[Union[str, Path]] = None, # Path to a reference WAV for default voice
                 default_language: str = "en",
                 device: Optional[str] = None,
                 # Standard LollmsTTSBinding args
                 host_address: Optional[str] = None,
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True,
                 **kwargs): # Catch-all for future TTS API changes or specific params

        super().__init__(binding_name="xtts")

        if not _xtts_deps_installed_with_correct_torch:
            raise ImportError(f"XTTS binding dependencies not met. Error: {_xtts_installation_error}")

        self.device = device
        if self.device is None:
            if torch.cuda.is_available(): self.device = "cuda"; ASCIIColors.info("CUDA device detected by PyTorch for XTTS.")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): self.device = "mps"; ASCIIColors.info("MPS device detected for XTTS.")
            else: self.device = "cpu"; ASCIIColors.info("No GPU (CUDA/MPS) by PyTorch, using CPU for XTTS.")
        elif self.device == "cuda" and not torch.cuda.is_available(): self.device = "cpu"; ASCIIColors.warning("CUDA req, not avail. CPU for XTTS.")
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()): self.device = "cpu"; ASCIIColors.warning("MPS req, not avail. CPU for XTTS.")

        ASCIIColors.info(f"XTTSBinding: Using device '{self.device}'.")

        self.xtts_model_id_or_path = model_name # Store the model identifier passed by user
        self.loaded_xtts_model_id = None
        self.tts_model: Optional[TTS] = None
        self.default_speaker_wav = str(default_speaker_wav) if default_speaker_wav else None
        self.default_language = default_language
        
        if self.default_speaker_wav and not Path(self.default_speaker_wav).exists():
            ASCIIColors.warning(f"Default speaker WAV not found: {self.default_speaker_wav}. Voice cloning will require a speaker_wav per call.")
            self.default_speaker_wav = None # Invalidate if not found

        self._load_xtts_model(self.xtts_model_id_or_path)

    def _load_xtts_model(self, model_id_to_load: str):
        if self.tts_model is not None and self.loaded_xtts_model_id == model_id_to_load:
            ASCIIColors.info(f"XTTS model '{model_id_to_load}' already loaded.")
            return

        ASCIIColors.info(f"Loading XTTS model: '{model_id_to_load}' on device '{self.device}'...")
        try:
            # TTS class handles model downloading from Hugging Face or loading from local path.
            # It also manages moving to the specified device.
            self.tts_model = TTS(model_name=model_id_to_load, progress_bar=True).to(self.device)
            self.loaded_xtts_model_id = model_id_to_load
            ASCIIColors.green(f"XTTS model '{model_id_to_load}' loaded successfully.")
        except Exception as e:
            self.tts_model = None; self.loaded_xtts_model_id = None
            ASCIIColors.error(f"Failed to load XTTS model '{model_id_to_load}': {e}"); trace_exception(e)
            raise RuntimeError(f"Failed to load XTTS model '{model_id_to_load}'") from e

    def generate_audio(self,
                       text: str,
                       voice: Optional[Union[str, Path]] = None, # Path to speaker WAV for XTTS
                       language: Optional[str] = None,
                       # XTTS specific parameters (can be passed via kwargs)
                       # speed: float = 1.0, # Not directly in XTTS v2 tts() method's main signature
                       **kwargs) -> bytes:
        if self.tts_model is None:
            raise RuntimeError("XTTS model not loaded.")

        speaker_wav_path = voice if voice is not None else self.default_speaker_wav
        effective_language = language if language is not None else self.default_language

        if not speaker_wav_path:
            raise ValueError("XTTS requires a 'speaker_wav' path for voice cloning. Provide it in the 'voice' argument or set 'default_speaker_wav' during initialization.")
        
        speaker_wav_p = Path(speaker_wav_path)
        if not speaker_wav_p.exists():
            raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav_path}")
        
        if effective_language not in XTTS_SUPPORTED_LANGUAGES:
            ASCIIColors.warning(f"Language '{effective_language}' might not be officially supported by XTTS v2. "
                                f"Known supported: {XTTS_SUPPORTED_LANGUAGES}. Attempting anyway.")

        ASCIIColors.info(f"Generating speech with XTTS: '{text[:60]}...' (Speaker: {speaker_wav_p.name}, Lang: {effective_language})")
        
        try:
            # The tts() method returns a NumPy array (waveform)
            # It expects speaker_wav and language as direct arguments.
            # Other TTS generation parameters might be available via model's config or specific methods.
            # For XTTS, common ones like speed are handled internally or via config.
            # We can pass other kwargs if the TTS library might pick them up for specific models.
            
            # XTTS's tts() returns list of ints (scaled PCM), not float numpy array directly
            wav_array_int_list = self.tts_model.tts(
                text=text,
                speaker_wav=str(speaker_wav_path), # Must be a string path
                language=effective_language,
                # split_sentences=True, # Default True, good for longer texts
                **kwargs # Pass other potential TTS lib args
            )

            if not wav_array_int_list: # Check if list is empty
                raise RuntimeError("XTTS model returned empty audio data (list of ints was empty).")

            # Convert list of ints to a NumPy array of int16
            # The TTS library usually returns samples scaled appropriately for int16.
            audio_array_np = np.array(wav_array_int_list, dtype=np.int16)


            if audio_array_np.ndim == 0 or audio_array_np.size == 0: # Double check after conversion
                raise RuntimeError("XTTS model resulted in empty NumPy audio array.")


            buffer = io.BytesIO()
            # Get sample rate from the loaded TTS model's config
            sample_rate = self.tts_model.synthesizer.output_sample_rate if hasattr(self.tts_model, 'synthesizer') and hasattr(self.tts_model.synthesizer, 'output_sample_rate') else 24000 # XTTS v2 default is 24kHz
            
            scipy.io.wavfile.write(buffer, rate=sample_rate, data=audio_array_np)
            audio_bytes = buffer.getvalue()
            buffer.close()

            ASCIIColors.green("XTTS audio generation successful.")
            return audio_bytes
        except Exception as e:
            ASCIIColors.error(f"XTTS audio generation failed: {e}"); trace_exception(e)
            if "out of memory" in str(e).lower() and self.device == "cuda":
                 ASCIIColors.yellow("CUDA out of memory. Ensure GPU has sufficient VRAM for XTTS (can be several GB).")
            raise RuntimeError(f"XTTS audio generation error: {e}") from e

    def list_voices(self, **kwargs) -> List[str]:
        """
        For XTTS, voices are determined by the `speaker_wav` file.
        This method returns a message or an empty list, as there are no predefined voices.
        Optionally, one could implement scanning a user-defined directory of speaker WAVs.
        """
        # return ["Dynamic (provide 'speaker_wav' path to generate_audio)"]
        ASCIIColors.info("XTTS voices are dynamic and determined by the 'speaker_wav' file provided during generation.")
        ASCIIColors.info("You can provide a path to any reference WAV file for voice cloning.")
        return [] # Or provide a helper message as above in a different way

    def get_xtts_model_ids(self) -> List[str]:
        """Helper to list known XTTS model identifiers for Coqui TTS library."""
        return XTTS_MODELS.copy()
    
    def get_supported_languages(self) -> List[str]:
        """Helper to list known supported languages for XTTS v2."""
        return XTTS_SUPPORTED_LANGUAGES.copy()


    def __del__(self):
        if hasattr(self, 'tts_model') and self.tts_model is not None:
            del self.tts_model; self.tts_model = None
            if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            loaded_name = getattr(self, 'loaded_xtts_model_id', None)
            msg = f"XTTSBinding for model '{loaded_name}' destroyed." if loaded_name else "XTTSBinding destroyed."
            ASCIIColors.info(msg)

# --- Main Test Block ---
if __name__ == '__main__':
    if not _xtts_deps_installed_with_correct_torch:
        print(f"{ASCIIColors.RED}XTTS binding dependencies not met. Skipping tests. Error: {_xtts_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- XTTSBinding Test ---")
    # For XTTS, model_name is the Coqui TTS model string or HF repo ID if supported by TTS lib directly
    test_xtts_model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
    test_output_dir = Path("./test_xtts_output")
    test_output_dir.mkdir(exist_ok=True)
    
    # --- IMPORTANT: Create or provide a speaker reference WAV file ---
    # For this test to work, you need a short (~5-15 seconds) clean audio file of a voice.
    # Name it 'speaker_ref.wav' and place it in the same directory as this script,
    # or update the path below.
    default_speaker_wav_path = Path(__file__).parent / "speaker_ref.wav" # Assumes it's next to this __init__.py

    if not default_speaker_wav_path.exists():
        ASCIIColors.warning(f"Reference speaker WAV file not found: {default_speaker_wav_path}")
        ASCIIColors.warning("Please create/place a 'speaker_ref.wav' (clean, ~5-15s audio) in the "
                            f"'{default_speaker_wav_path.parent}' directory for the test to run properly.")
        # Attempt to create a very basic dummy if scipy available, NOT suitable for good cloning
        try:
            import numpy as np; import scipy.io.wavfile
            samplerate = 22050; duration = 2; frequency = 440
            t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
            data = (np.iinfo(np.int16).max * 0.1 * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
            scipy.io.wavfile.write(default_speaker_wav_path, samplerate, data)
            ASCIIColors.info(f"Created a VERY BASIC dummy 'speaker_ref.wav'. Replace with a real voice sample for good results.")
        except Exception as e_dummy_spk:
            ASCIIColors.error(f"Could not create dummy speaker_ref.wav: {e_dummy_spk}. Test will likely fail or use no speaker.")
            default_speaker_wav_path = None # Ensure it's None if creation failed

    tts_binding = None
    try:
        ASCIIColors.cyan(f"\n--- Initializing XTTSBinding (XTTS Model: '{test_xtts_model_id}') ---")
        tts_binding = XTTSBinding(
            model_name=test_xtts_model_id,
            default_speaker_wav=str(default_speaker_wav_path) if default_speaker_wav_path else None,
            default_language="en"
        )

        ASCIIColors.cyan("\n--- Listing XTTS 'voices' (dynamic, requires speaker_wav) ---")
        voices = tts_binding.list_voices(); # This will print an informational message
        
        ASCIIColors.cyan("\n--- Listing known XTTS model IDs for Coqui TTS library ---")
        xtts_models = tts_binding.get_xtts_model_ids(); print(f"Known XTTS model IDs: {xtts_models}")
        ASCIIColors.cyan("\n--- Listing known XTTS supported languages ---")
        langs = tts_binding.get_supported_languages(); print(f"Supported languages (example): {langs[:5]}...")


        texts_to_synthesize = [
            ("english_greeting", "Hello, this is a test of the XTTS voice synthesis system. I hope you like my voice!", "en"),
            ("spanish_question", "¿Cómo estás hoy? Espero que tengas un día maravilloso.", "es"),
            # ("short_custom_voice", "This voice should sound like your reference audio.", "en", "path/to/your/custom_speaker.wav"), # Example for custom
        ]
        if not default_speaker_wav_path: # If no default speaker, we can't run text loop as is
            ASCIIColors.error("No default_speaker_wav available. Skipping synthesis loop.")
            texts_to_synthesize = []


        for name, text, lang, *speaker_override_list in texts_to_synthesize:
            speaker_to_use = speaker_override_list[0] if speaker_override_list else None # Uses binding default if None
            
            ASCIIColors.cyan(f"\n--- Synthesizing TTS for: '{name}' (Lang: {lang}, Speaker: {speaker_to_use or tts_binding.default_speaker_wav}) ---")
            print(f"Text: {text}")
            try:
                # XTTS tts() doesn't have as many direct generation params as Bark's generate()
                # Control is more via the model config or specific methods if available.
                audio_bytes = tts_binding.generate_audio(text, voice=speaker_to_use, language=lang)
                if audio_bytes:
                    output_filename = f"tts_{name}_{tts_binding.loaded_xtts_model_id.replace('/','_')}.wav"
                    output_path = test_output_dir / output_filename
                    with open(output_path, "wb") as f: f.write(audio_bytes)
                    ASCIIColors.green(f"TTS for '{name}' saved to: {output_path} ({len(audio_bytes) / 1024:.2f} KB)")
                else: ASCIIColors.error(f"TTS generation for '{name}' returned empty bytes.")
            except Exception as e_gen: ASCIIColors.error(f"Failed to generate TTS for '{name}': {e_gen}")

    except ImportError as e_imp: ASCIIColors.error(f"Import error: {e_imp}")
    except RuntimeError as e_rt: ASCIIColors.error(f"Runtime error: {e_rt}")
    except Exception as e: ASCIIColors.error(f"Unexpected error: {e}"); trace_exception(e)
    finally:
        if tts_binding: del tts_binding
        ASCIIColors.info(f"Test TTS audio (if any) are in: {test_output_dir.resolve()}")
        print(f"{ASCIIColors.YELLOW}Check the audio files in '{test_output_dir.resolve()}'!{ASCIIColors.RESET}")
        # Clean up dummy speaker_ref.wav if we created it
        if "samplerate" in locals() and default_speaker_wav_path and default_speaker_wav_path.name == "speaker_ref.wav" and "dummy" in str(default_speaker_wav_path).lower():
            if default_speaker_wav_path.exists():
                try: default_speaker_wav_path.unlink(); ASCIIColors.info("Removed dummy speaker_ref.wav")
                except: pass


    ASCIIColors.yellow("\n--- XTTSBinding Test Finished ---")