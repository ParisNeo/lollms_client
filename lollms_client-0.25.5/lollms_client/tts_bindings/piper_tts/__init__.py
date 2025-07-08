# lollms_client/tts_bindings/piper/__init__.py
import io
import os
import wave # Standard Python library for WAV files
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

from ascii_colors import trace_exception, ASCIIColors

# --- Package Management and Conditional Imports ---
_piper_tts_installed = False
_piper_tts_installation_error = ""
try:
    import pipmaster as pm
    # piper-tts should handle onnxruntime, but ensure it's there if needed
    # We might need specific onnxruntime for CUDA/DirectML later if we extend device support
    pm.ensure_packages(["piper-tts", "onnxruntime"])

    from piper import PiperVoice
    import numpy as np # For converting audio samples if needed

    _piper_tts_installed = True
except Exception as e:
    _piper_tts_installation_error = str(e)
    PiperVoice = None
    np = None # Piper often returns bytes, but numpy can be handy for sample rate conversion if needed
# --- End Package Management ---

from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "PiperTTSBinding"

# Example of a known good voice URL prefix from rhasspy.github.io/piper-voices/
PIPER_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"


class PiperTTSBinding(LollmsTTSBinding):
    def __init__(self,
                 default_voice_model_path: Optional[Union[str, Path]] = None, # Path to .onnx file
                 piper_voices_dir: Optional[Union[str, Path]] = None, # Directory to scan for voices
                 # Standard LollmsTTSBinding args (host, service_key, verify_ssl are not used for local Piper)
                 host_address: Optional[str] = None,
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True,
                 **kwargs): # Catch-all for future params or Piper-specific init options

        super().__init__(binding_name="piper")

        if not _piper_tts_installed:
            raise ImportError(f"Piper TTS binding dependencies not met. Error: {_piper_tts_installation_error}")

        self.piper_voices_dir = Path(piper_voices_dir).resolve() if piper_voices_dir else None
        if self.piper_voices_dir and not self.piper_voices_dir.is_dir():
            ASCIIColors.warning(f"Piper voices directory does not exist: {self.piper_voices_dir}. Voice listing will be limited.")
            self.piper_voices_dir = None

        self.current_voice_model_path: Optional[Path] = None
        self.piper_voice: Optional[PiperVoice] = None
        self.voice_config: Optional[Dict] = None # To store sample rate, channels etc.

        if default_voice_model_path:
            self._load_piper_voice(default_voice_model_path)
        else:
            ASCIIColors.info("No default_voice_model_path provided for Piper. Load a voice via generate_audio or ensure piper_voices_dir is set.")


    def _load_piper_voice(self, voice_model_identifier: Union[str, Path]):
        """
        Loads a Piper voice model.
        identifier can be a full path to .onnx or a filename to be found in piper_voices_dir.
        """
        voice_model_path_onnx: Optional[Path] = None
        voice_model_path_json: Optional[Path] = None

        potential_path = Path(voice_model_identifier)

        if potential_path.is_absolute() and potential_path.suffix == ".onnx" and potential_path.exists():
            voice_model_path_onnx = potential_path
            voice_model_path_json = potential_path.with_suffix(".onnx.json")
        elif self.piper_voices_dir and (self.piper_voices_dir / voice_model_identifier).exists():
            # Assume voice_model_identifier is a filename like "en_US-ryan-medium.onnx"
            p = self.piper_voices_dir / voice_model_identifier
            if p.suffix == ".onnx":
                voice_model_path_onnx = p
                voice_model_path_json = p.with_suffix(".onnx.json")
        elif potential_path.suffix == ".onnx" and potential_path.exists(): # Relative path
             voice_model_path_onnx = potential_path.resolve()
             voice_model_path_json = voice_model_path_onnx.with_suffix(".onnx.json")


        if not voice_model_path_onnx or not voice_model_path_onnx.exists():
            raise FileNotFoundError(f"Piper ONNX voice model not found: {voice_model_identifier}")
        if not voice_model_path_json or not voice_model_path_json.exists():
            raise FileNotFoundError(f"Piper voice JSON config not found for {voice_model_path_onnx} (expected: {voice_model_path_json})")

        if self.piper_voice and self.current_voice_model_path == voice_model_path_onnx:
            ASCIIColors.info(f"Piper voice '{voice_model_path_onnx.name}' already loaded.")
            return

        ASCIIColors.info(f"Loading Piper voice: {voice_model_path_onnx.name}...")
        try:
            # Piper documentation often shows use_cuda=True for GPU with onnxruntime-gpu.
            # For simplicity and Piper's primary CPU strength, we'll omit it for now.
            # onnxruntime will use CPU by default.
            # To enable GPU: user needs onnxruntime-gpu and then `PiperVoice.from_files(..., use_cuda=True)`
            self.piper_voice = PiperVoice.from_files(
                onnx_path=str(voice_model_path_onnx),
                config_path=str(voice_model_path_json)
                # use_cuda=True # if onnxruntime-gpu is installed and desired
            )
            with open(voice_model_path_json, 'r', encoding='utf-8') as f:
                self.voice_config = json.load(f)

            self.current_voice_model_path = voice_model_path_onnx
            ASCIIColors.green(f"Piper voice '{voice_model_path_onnx.name}' loaded successfully.")
        except Exception as e:
            self.piper_voice = None
            self.current_voice_model_path = None
            self.voice_config = None
            ASCIIColors.error(f"Failed to load Piper voice '{voice_model_path_onnx.name}': {e}"); trace_exception(e)
            raise RuntimeError(f"Failed to load Piper voice '{voice_model_path_onnx.name}'") from e

    def generate_audio(self,
                       text: str,
                       voice: Optional[Union[str, Path]] = None, # Filename or path to .onnx
                       **kwargs) -> bytes: # kwargs can include Piper synthesis options
        if voice:
            try:
                self._load_piper_voice(voice) # Attempt to switch voice
            except Exception as e_load:
                ASCIIColors.error(f"Failed to switch to Piper voice '{voice}': {e_load}. Using previously loaded voice if available.")
                if not self.piper_voice: # If no voice was previously loaded either
                    raise RuntimeError("No Piper voice loaded and failed to switch.") from e_load

        if not self.piper_voice or not self.voice_config:
            raise RuntimeError("Piper voice model not loaded. Cannot generate audio.")

        ASCIIColors.info(f"Generating speech with Piper voice '{self.current_voice_model_path.name}': '{text[:60]}...'")
        
        try:
            # Piper's synthesize returns raw audio bytes (PCM s16le)
            # Piper can also stream with synthesize_stream_raw if needed for very long texts
            # For simplicity, using synthesize which returns all bytes at once.
            
            # synthesis_kwargs: length_scale, noise_scale, noise_w
            piper_synthesis_kwargs = {}
            if 'length_scale' in kwargs: piper_synthesis_kwargs['length_scale'] = float(kwargs['length_scale'])
            if 'noise_scale' in kwargs: piper_synthesis_kwargs['noise_scale'] = float(kwargs['noise_scale'])
            if 'noise_w' in kwargs: piper_synthesis_kwargs['noise_w'] = float(kwargs['noise_w'])


            audio_bytes_iterable = self.piper_voice.synthesize_stream_raw(text, **piper_synthesis_kwargs)
            
            # Accumulate bytes from the stream
            pcm_s16le_data = b"".join(audio_bytes_iterable)

            if not pcm_s16le_data:
                raise RuntimeError("Piper synthesize_stream_raw returned empty audio data.")

            # Now package these raw PCM bytes into a WAV container
            buffer = io.BytesIO()
            sample_rate = self.voice_config.get("audio", {}).get("sample_rate", 22050) # Default if not in config
            num_channels = 1 # Piper voices are typically mono
            sample_width = 2 # 16-bit audio means 2 bytes per sample

            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_s16le_data)
            
            wav_bytes = buffer.getvalue()
            buffer.close()

            ASCIIColors.green("Piper TTS audio generation successful.")
            return wav_bytes
        except Exception as e:
            ASCIIColors.error(f"Piper TTS audio generation failed: {e}"); trace_exception(e)
            raise RuntimeError(f"Piper TTS audio generation error: {e}") from e

    def list_voices(self, **kwargs) -> List[str]:
        """
        Lists available Piper voice models found in the piper_voices_dir.
        Returns a list of .onnx filenames.
        """
        voices = []
        if self.piper_voices_dir and self.piper_voices_dir.is_dir():
            for item in self.piper_voices_dir.iterdir():
                if item.is_file() and item.suffix == ".onnx":
                    json_config_path = item.with_suffix(".onnx.json")
                    if json_config_path.exists():
                        voices.append(item.name) # Return just the filename
        
        if not voices and not self.current_voice_model_path:
             ASCIIColors.warning("No voices found in piper_voices_dir and no default voice loaded.")
             ASCIIColors.info(f"Download Piper voices (e.g., from {PIPER_VOICES_BASE_URL} or https://rhasspy.github.io/piper-voices/) "
                              "and place the .onnx and .onnx.json files into your voices directory.")
        elif not voices and self.current_voice_model_path:
            voices.append(self.current_voice_model_path.name) # Add the default loaded one if dir is empty

        return sorted(list(set(voices))) # Ensure unique and sorted

    def __del__(self):
        # PiperVoice objects don't have an explicit close/del, Python's GC should handle C extensions
        if hasattr(self, 'piper_voice') and self.piper_voice is not None:
            del self.piper_voice
            self.piper_voice = None
            ASCIIColors.info(f"PiperTTSBinding voice '{getattr(self, 'current_voice_model_path', 'N/A')}' resources released.")

# --- Main Test Block ---
if __name__ == '__main__':
    if not _piper_tts_installed:
        print(f"{ASCIIColors.RED}Piper TTS binding dependencies not met. Skipping tests. Error: {_piper_tts_installation_error}{ASCIIColors.RESET}")
        exit()

    ASCIIColors.yellow("--- PiperTTSBinding Test ---")

    # --- USER CONFIGURATION FOR TEST ---
    # 1. Create a directory to store Piper voices, e.g., "./test_piper_voices"
    TEST_PIPER_VOICES_DIR = Path("./test_piper_voices")
    TEST_PIPER_VOICES_DIR.mkdir(exist_ok=True)

    # 2. Download at least one voice model (ONNX + JSON files) into that directory.
    #    From: https://rhasspy.github.io/piper-voices/
    #    Example: Download en_US-lessac-medium.onnx and en_US-lessac-medium.onnx.json
    #             and place them in TEST_PIPER_VOICES_DIR
    #    Or find direct links on Hugging Face: e.g., from https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium
    #    Let's pick a common English voice for testing.
    DEFAULT_TEST_VOICE_FILENAME = "en_US-lessac-medium.onnx" # Ensure this (and .json) is in TEST_PIPER_VOICES_DIR
    DEFAULT_TEST_VOICE_ONNX_URL = f"{PIPER_VOICES_BASE_URL}en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    DEFAULT_TEST_VOICE_JSON_URL = f"{PIPER_VOICES_BASE_URL}en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

    # Function to download test voice if missing
    def ensure_test_voice(voices_dir: Path, voice_filename: str, onnx_url: str, json_url: str):
        onnx_path = voices_dir / voice_filename
        json_path = voices_dir / f"{voice_filename}.json"
        if not onnx_path.exists() or not json_path.exists():
            ASCIIColors.info(f"Test voice '{voice_filename}' not found. Attempting to download...")
            try:
                import requests
                # Download ONNX
                if not onnx_path.exists():
                    ASCIIColors.info(f"Downloading {onnx_url} to {onnx_path}")
                    r_onnx = requests.get(onnx_url, stream=True)
                    r_onnx.raise_for_status()
                    with open(onnx_path, 'wb') as f:
                        for chunk in r_onnx.iter_content(chunk_size=8192): f.write(chunk)
                # Download JSON
                if not json_path.exists():
                    ASCIIColors.info(f"Downloading {json_url} to {json_path}")
                    r_json = requests.get(json_url)
                    r_json.raise_for_status()
                    with open(json_path, 'w', encoding='utf-8') as f: f.write(r_json.text)
                ASCIIColors.green(f"Test voice '{voice_filename}' downloaded successfully.")
            except Exception as e_download:
                ASCIIColors.error(f"Failed to download test voice '{voice_filename}': {e_download}")
                ASCIIColors.warning(f"Please manually download '{voice_filename}' and '{voice_filename}.json' "
                                    f"from {PIPER_VOICES_BASE_URL} (or rhasspy.github.io/piper-voices/) "
                                    f"and place them in {voices_dir.resolve()}")
                return False
        return True

    if not ensure_test_voice(TEST_PIPER_VOICES_DIR, DEFAULT_TEST_VOICE_FILENAME, DEFAULT_TEST_VOICE_ONNX_URL, DEFAULT_TEST_VOICE_JSON_URL):
        ASCIIColors.error("Cannot proceed with test without a default voice model.")
        exit(1)

    # Optional: Download a second voice for testing voice switching
    SECOND_TEST_VOICE_FILENAME = "de_DE-thorsten-medium.onnx" # Example German voice
    SECOND_TEST_VOICE_ONNX_URL = f"{PIPER_VOICES_BASE_URL}de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
    SECOND_TEST_VOICE_JSON_URL = f"{PIPER_VOICES_BASE_URL}de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
    ensure_test_voice(TEST_PIPER_VOICES_DIR, SECOND_TEST_VOICE_FILENAME, SECOND_TEST_VOICE_ONNX_URL, SECOND_TEST_VOICE_JSON_URL)


    test_output_dir = Path("./test_piper_tts_output")
    test_output_dir.mkdir(exist_ok=True)
    tts_binding = None
    # --- END USER CONFIGURATION FOR TEST ---

    try:
        ASCIIColors.cyan(f"\n--- Initializing PiperTTSBinding ---")
        # Initialize with the path to the ONNX file for the default voice
        tts_binding = PiperTTSBinding(
            default_voice_model_path = TEST_PIPER_VOICES_DIR / DEFAULT_TEST_VOICE_FILENAME,
            piper_voices_dir = TEST_PIPER_VOICES_DIR
        )

        ASCIIColors.cyan("\n--- Listing available Piper voices ---")
        voices = tts_binding.list_voices();
        if voices: print(f"Available voices in '{TEST_PIPER_VOICES_DIR}': {voices}")
        else: ASCIIColors.warning(f"No voices found in {TEST_PIPER_VOICES_DIR}. Check paths and ensure .onnx/.json pairs exist.")


        texts_to_synthesize = [
            ("english_hello", "Hello world, this is a test of the Piper text to speech binding."),
            ("english_question", "Can you generate speech quickly and efficiently? Let's find out!"),
        ]
        if (TEST_PIPER_VOICES_DIR / SECOND_TEST_VOICE_FILENAME).exists():
             texts_to_synthesize.append(
                 ("german_greeting", "Hallo Welt, wie geht es Ihnen heute?", SECOND_TEST_VOICE_FILENAME)
             )


        for name, text, *voice_file_arg in texts_to_synthesize:
            voice_to_use_filename = voice_file_arg[0] if voice_file_arg else None # Filename like "en_US-lessac-medium.onnx"
            
            ASCIIColors.cyan(f"\n--- Synthesizing TTS for: '{name}' (Voice file: {voice_to_use_filename or DEFAULT_TEST_VOICE_FILENAME}) ---")
            print(f"Text: {text}")
            try:
                # Example of passing Piper-specific synthesis parameters
                synthesis_kwargs = {"length_scale": 1.0} # Default is 1.0. Smaller is faster, larger is slower.
                if "question" in name:
                    synthesis_kwargs["length_scale"] = 0.9 # Slightly faster for questions

                audio_bytes = tts_binding.generate_audio(text, voice=voice_to_use_filename, **synthesis_kwargs)
                if audio_bytes:
                    output_filename = f"tts_piper_{name}.wav"
                    output_path = test_output_dir / output_filename
                    with open(output_path, "wb") as f: f.write(audio_bytes)
                    ASCIIColors.green(f"TTS for '{name}' saved to: {output_path} ({len(audio_bytes) / 1024:.2f} KB)")
                else: ASCIIColors.error(f"TTS generation for '{name}' returned empty bytes.")
            except Exception as e_gen: ASCIIColors.error(f"Failed to generate TTS for '{name}': {e_gen}")

    except ImportError as e_imp: ASCIIColors.error(f"Import error: {e_imp}")
    except FileNotFoundError as e_fnf: ASCIIColors.error(f"File not found error during init/load: {e_fnf}")
    except RuntimeError as e_rt: ASCIIColors.error(f"Runtime error: {e_rt}")
    except Exception as e: ASCIIColors.error(f"Unexpected error: {e}"); trace_exception(e)
    finally:
        if tts_binding: del tts_binding
        ASCIIColors.info(f"Test TTS audio (if any) are in: {test_output_dir.resolve()}")
        print(f"{ASCIIColors.YELLOW}Check the audio files in '{test_output_dir.resolve()}'!{ASCIIColors.RESET}")
        # Optional: Clean up downloaded test voices
        # if input("Clean up downloaded test voices? (y/N): ").lower() == 'y':
        #     for f_name in [DEFAULT_TEST_VOICE_FILENAME, SECOND_TEST_VOICE_FILENAME]:
        #         onnx_p = TEST_PIPER_VOICES_DIR / f_name
        #         json_p = TEST_PIPER_VOICES_DIR / f"{f_name}.json"
        #         if onnx_p.exists(): onnx_p.unlink()
        #         if json_p.exists(): json_p.unlink()
        #     if not any(TEST_PIPER_VOICES_DIR.iterdir()): TEST_PIPER_VOICES_DIR.rmdir()
        #     ASCIIColors.info("Cleaned up test voices.")


    ASCIIColors.yellow("\n--- PiperTTSBinding Test Finished ---")