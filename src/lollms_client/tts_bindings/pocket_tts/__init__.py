from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import io
import os
import re
import shutil
import wave
import numpy as np
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_tts_binding import LollmsTTSBinding

BindingName = "PocketTTSBinding"

class PocketTTSBinding(LollmsTTSBinding):
    """
    Binding for Kyutai Labs' Pocket TTS model (100M parameter CPU-optimized TTS with zero-shot voice cloning).
    """

    PREMADE_VOICES = [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
        "estelle",
        "giovanni",
        "lola",
        "juergen",
        "rafael",
        "anna",
        "bill_boerst",
        "caro_davy",
        "charles",
        "eve",
        "george",
        "jane",
        "mary",
        "michael",
        "paul",
        "peter_yearsley",
        "stuart_bell",
        "vera"
    ]

    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".safetensors"}

    MODEL_ZOO = [
        {
            "name": "Kyutai Pocket TTS (Multilingual + Voice Cloning)",
            "model_name": "kyutai/pocket-tts",
            "description": "Standard 100M parameter model with zero-shot voice cloning. Supports English, French, German, Spanish, Portuguese, Italian.",
            "size": "200 MB",
            "type": "CPU/GPU",
            "link": "https://huggingface.co/kyutai/pocket-tts"
        },
        {
            "name": "Kyutai Pocket TTS (Standard Voices Only)",
            "model_name": "kyutai/pocket-tts-without-voice-cloning",
            "description": "Ultra-lightweight variant without zero-shot voice cloning network for lowest memory overhead.",
            "size": "160 MB",
            "type": "CPU/GPU",
            "link": "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning"
        }
    ]

    def __init__(self, **kwargs):
        super().__init__(binding_name="pocket_tts", **kwargs)
        self.model_name = kwargs.get("model_name", "kyutai/pocket-tts")
        self.device = kwargs.get("device", "cpu")
        self.default_voice = kwargs.get("voice", "alba")
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.auto_install = kwargs.get("auto_install", True)
        
        # Resolve configurable paths with safe defaults
        raw_voices_dir = kwargs.get("voices_dir", "data/tts/pocket-tts/voices")
        self.voices_dir = Path(raw_voices_dir).resolve()
        self.voices_dir.mkdir(exist_ok=True, parents=True)

        raw_models_dir = kwargs.get("models_dir", "data/tts/pocket-tts/models")
        self.models_dir = Path(raw_models_dir).resolve()
        self.models_dir.mkdir(exist_ok=True, parents=True)

        self.tts_model = None
        self._voice_cache: Dict[str, Any] = {}
        self._discovered_custom_voices: Dict[str, Path] = {}
        self._scan_custom_voices()

    def _scan_custom_voices(self) -> Dict[str, Path]:
        """Scans voices_dir for user-uploaded audio files."""
        self._discovered_custom_voices.clear()
        if not self.voices_dir.exists():
            return self._discovered_custom_voices

        for file_path in self.voices_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.ALLOWED_EXTENSIONS:
                voice_name = file_path.stem
                self._discovered_custom_voices[voice_name] = file_path
        return self._discovered_custom_voices

    def get_zoo(self) -> List[Dict[str, Any]]:
        """Returns the list of available models from the Pocket TTS Zoo."""
        return self.MODEL_ZOO

    def download_from_zoo(self, index: int, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        """Downloads or pre-fetches a model from the model zoo by index."""
        if index < 0 or index >= len(self.MODEL_ZOO):
            return {"status": False, "message": f"Index {index} out of range (0-{len(self.MODEL_ZOO)-1})."}

        selected = self.MODEL_ZOO[index]
        model_id = selected["model_name"]

        try:
            if progress_callback:
                progress_callback({"status": "downloading", "progress": 10, "message": f"Downloading {model_id}..."})

            self._install_dependencies_if_needed()
            from pocket_tts import TTSModel

            if progress_callback:
                progress_callback({"status": "loading", "progress": 50, "message": f"Loading model {model_id} into cache..."})

            os.environ["HF_HUB_CACHE"] = str(self.models_dir)
            model = TTSModel.load_model(model_id)
            self.model_name = model_id
            self.tts_model = model
            if hasattr(self.tts_model, "to") and self.device:
                try:
                    self.tts_model.to(self.device)
                except Exception:
                    pass

            if progress_callback:
                progress_callback({"status": "complete", "progress": 100, "message": f"Model {model_id} downloaded and loaded successfully."})

            return {"status": True, "message": f"Model {model_id} downloaded and loaded successfully.", "model_name": model_id}
        except Exception as e:
            ASCIIColors.error(f"Failed to download model from zoo: {e}")
            trace_exception(e)
            return {"status": False, "message": str(e)}

    def _install_dependencies_if_needed(self):
        """Ensures pocket-tts is available in the current environment."""
        try:
            import pocket_tts  # noqa: F401
        except ImportError:
            if self.auto_install:
                ASCIIColors.warning("pocket_tts not found. Attempting installation via pipmaster...")
                try:
                    import pipmaster as pm
                    pm.install("pocket-tts")
                except Exception as e:
                    ASCIIColors.error(f"Failed to install pocket-tts: {e}")
                    raise ImportError("pocket-tts is required. Please run `pip install pocket-tts`.") from e
            else:
                raise ImportError("pocket-tts is required. Please run `pip install pocket-tts`.")

    def _ensure_model_loaded(self):
        """Lazy loader for Pocket TTS model."""
        if self.tts_model is not None:
            return

        self._install_dependencies_if_needed()
        from pocket_tts import TTSModel

        try:
            ASCIIColors.info(f"Loading Pocket TTS model: {self.model_name}...")
            os.environ["HF_HUB_CACHE"] = str(self.models_dir)
            self.tts_model = TTSModel.load_model(self.model_name)
            if hasattr(self.tts_model, "to") and self.device:
                try:
                    self.tts_model.to(self.device)
                except Exception as dev_err:
                    ASCIIColors.warning(f"Could not move Pocket TTS model to {self.device}: {dev_err}")
            ASCIIColors.green("Pocket TTS model loaded successfully.")
        except Exception as e:
            ASCIIColors.error(f"Error loading Pocket TTS model: {e}")
            trace_exception(e)
            raise

    # ── Commands ─────────────────────────────────────────────────────────────

    def upload_voice(self, voice_path: str, voice_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Uploads and registers a new reference voice file into the configured voices directory.
        """
        source_path = Path(voice_path)
        if not source_path.exists() or not source_path.is_file():
            return {
                "status": False,
                "voice_name": "",
                "message": f"Source audio file not found: {voice_path}"
            }

        suffix = source_path.suffix.lower()
        if suffix not in self.ALLOWED_EXTENSIONS:
            return {
                "status": False,
                "voice_name": "",
                "message": f"Invalid audio extension '{suffix}'. Allowed: {sorted(list(self.ALLOWED_EXTENSIONS))}"
            }

        raw_name = voice_name.strip() if voice_name else source_path.stem
        safe_name = re.sub(r"[^\w\-_]", "_", raw_name).strip("_") or "custom_voice"
        dest_file = self.voices_dir / f"{safe_name}{suffix}"

        try:
            shutil.copy2(str(source_path.resolve()), str(dest_file.resolve()))
            self._discovered_custom_voices[safe_name] = dest_file

            try:
                self._ensure_model_loaded()
                state = self.tts_model.get_state_for_audio_prompt(str(dest_file.resolve()))
                self._voice_cache[safe_name] = state
            except Exception as model_err:
                ASCIIColors.warning(f"Voice copied, conditioning pre-computation deferred: {model_err}")

            ASCIIColors.green(f"Pocket TTS: Registered new voice '{safe_name}' at {dest_file}")
            return {
                "status": True,
                "voice_name": safe_name,
                "message": f"Voice '{safe_name}' uploaded and registered successfully."
            }
        except Exception as e:
            ASCIIColors.error(f"Failed to upload voice: {e}")
            trace_exception(e)
            return {
                "status": False,
                "voice_name": safe_name,
                "message": f"Failed to upload voice: {str(e)}"
            }

    def delete_voice(self, voice_name: str, **kwargs) -> Dict[str, Any]:
        """Deletes a custom uploaded voice from disk and cache."""
        self._scan_custom_voices()
        if voice_name in self.PREMADE_VOICES:
            return {
                "status": False,
                "message": f"Cannot delete built-in Kyutai voice '{voice_name}'."
            }

        file_path = self._discovered_custom_voices.get(voice_name)
        if not file_path or not file_path.exists():
            return {
                "status": False,
                "message": f"Custom voice '{voice_name}' not found."
            }

        try:
            file_path.unlink(missing_ok=True)
            self._discovered_custom_voices.pop(voice_name, None)
            self._voice_cache.pop(voice_name, None)
            ASCIIColors.info(f"Deleted custom voice: {voice_name}")
            return {
                "status": True,
                "message": f"Custom voice '{voice_name}' was deleted successfully."
            }
        except Exception as e:
            ASCIIColors.error(f"Failed to delete voice '{voice_name}': {e}")
            return {
                "status": False,
                "message": f"Failed to delete voice: {str(e)}"
            }

    def list_custom_voices(self, **kwargs) -> Dict[str, Any]:
        """Lists all custom uploaded voices and their disk information."""
        self._scan_custom_voices()
        custom_voices = []
        for name, path in self._discovered_custom_voices.items():
            if path.exists():
                size_kb = round(path.stat().st_size / 1024, 2)
                custom_voices.append({
                    "name": name,
                    "file_name": path.name,
                    "path": str(path),
                    "size_kb": size_kb,
                    "format": path.suffix.lstrip(".")
                })
        return {
            "status": True,
            "voices": custom_voices
        }

    def test_voice(self, voice: str = "alba", sample_text: str = "Hello, this is a test of Pocket TTS speech synthesis.", **kwargs) -> Dict[str, Any]:
        """Generates a test phrase to verify voice synthesis functionality."""
        try:
            audio_bytes = self.generate_audio(text=sample_text, voice=voice)
            if audio_bytes and len(audio_bytes) > 0:
                return {
                    "status": True,
                    "message": f"Successfully generated {len(audio_bytes)} bytes of audio for voice '{voice}'."
                }
            return {
                "status": False,
                "message": f"Generation produced empty audio for voice '{voice}'."
            }
        except Exception as e:
            return {
                "status": False,
                "message": f"Test failed: {str(e)}"
            }

    def select_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """Selects and loads a new model by name."""
        try:
            self.unload_model()
            self.model_name = model_name
            self._ensure_model_loaded()
            return {
                "status": True,
                "message": f"Model '{model_name}' loaded successfully."
            }
        except Exception as e:
            ASCIIColors.error(f"Failed to select model {model_name}: {e}")
            return {
                "status": False,
                "message": f"Failed to load model '{model_name}': {str(e)}"
            }

    # ── Core TTS Operations ──────────────────────────────────────────────────

    def _get_voice_state(self, voice_identifier: str):
        """Retrieves or creates a cached voice conditioning state."""
        self._ensure_model_loaded()
        
        if voice_identifier in self._voice_cache:
            return self._voice_cache[voice_identifier]

        if voice_identifier in self._discovered_custom_voices:
            target_path = self._discovered_custom_voices[voice_identifier]
            if target_path.exists():
                prompt_input = str(target_path.resolve())
            else:
                prompt_input = voice_identifier
        else:
            vpath = Path(voice_identifier)
            if vpath.exists() and vpath.is_file():
                prompt_input = str(vpath.resolve())
            else:
                prompt_input = voice_identifier

        try:
            state = self.tts_model.get_state_for_audio_prompt(prompt_input)
            self._voice_cache[voice_identifier] = state
            return state
        except Exception as e:
            ASCIIColors.warning(f"Could not load voice prompt '{voice_identifier}': {e}. Falling back to 'alba'.")
            if "alba" not in self._voice_cache:
                self._voice_cache["alba"] = self.tts_model.get_state_for_audio_prompt("alba")
            return self._voice_cache["alba"]

    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """
        Synthesizes text into 16-bit WAV PCM audio bytes using Pocket TTS.
        """
        self._ensure_model_loaded()
        selected_voice = voice or self.default_voice
        voice_state = self._get_voice_state(selected_voice)

        try:
            audio_tensor = self.tts_model.generate_audio(voice_state, text)
            
            if hasattr(audio_tensor, "detach"):
                audio_np = audio_tensor.detach().cpu().numpy()
            else:
                audio_np = np.asarray(audio_tensor)

            audio_np = audio_np.squeeze().astype(np.float32)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            sample_rate = getattr(self.tts_model, "sample_rate", 24000)

            pcm_data = (audio_np * 32767.0).astype(np.int16)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data.tobytes())

            return buffer.getvalue()
        except Exception as e:
            ASCIIColors.error(f"Pocket TTS generation failed: {e}")
            trace_exception(e)
            return b""

    def list_voices(self, **kwargs) -> List[str]:
        """Returns list of premade Pocket TTS voices and discovered custom uploaded voices."""
        self._scan_custom_voices()
        all_voices = set(self.PREMADE_VOICES)
        all_voices.update(self._discovered_custom_voices.keys())
        all_voices.update(self._voice_cache.keys())
        return sorted(list(all_voices))

    def list_models(self, **kwargs) -> List[str]:
        """Returns supported Pocket TTS model identifiers."""
        return [
            "kyutai/pocket-tts",
            "kyutai/pocket-tts-without-voice-cloning"
        ]

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        """Frees model tensors and voice state caches from memory."""
        self.tts_model = None
        self._voice_cache.clear()
        import gc
        gc.collect()
        ASCIIColors.info("Pocket TTS model and voice cache unloaded successfully.")
        return True