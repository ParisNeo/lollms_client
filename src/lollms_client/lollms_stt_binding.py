# lollms_client/lollms_stt_binding.py
from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception
import yaml
from lollms_client.lollms_base_binding import LollmsBaseBinding

class LollmsSTTBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS Speech-to-Text bindings."""

    def __init__(self,
                 binding_name:str="unknown",
                 debug:Optional[bool] = False,
                 **kwargs):
        """
        Initialize the LollmsSTTBinding base class.
        """
        super().__init__(binding_name=binding_name, debug=debug, **kwargs)

    @abstractmethod
    def transcribe_audio(self, audio_source: Union[str, Path, bytes], model: Optional[str] = None, **kwargs) -> str:
        """
        Transcribes the audio source into text.

        Args:
            audio_source (Union[str, Path, bytes]): Path to the audio file or raw audio bytes.
            model (Optional[str]): The model to use for transcription.
            **kwargs: Additional parameters.

        Returns:
            str: The transcribed text.
        """
        pass

    def transcribe_audio_with_diarization(
        self,
        audio_source: Union[str, Path, bytes],
        participants: Optional[List[str]] = None,
        voice_samples: Optional[Dict[str, Union[str, Path, bytes]]] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribes the audio source with speaker diarization, identifying distinct
        speakers and mapping them to participant names or known voice samples.

        Args:
            audio_source (Union[str, Path, bytes]): Audio path or raw bytes.
            participants (Optional[List[str]]): List of speaker names in order of appearance.
                If fewer names are provided than detected voices (or if omitted),
                remaining voices are named "Speaker 1", "Speaker 2", etc.
            voice_samples (Optional[Dict[str, Union[str, Path, bytes]]]): Reference voice
                samples mapping speaker names to audio clips for biometric matching.
            model (Optional[str]): The model to use.
            **kwargs: Additional arguments.

        Returns:
            List[Dict[str, Any]]: List of dialogue turns, e.g.:
                [
                    {
                        "speaker": "Alice",
                        "start": 0.0,
                        "end": 4.5,
                        "text": "Hello everyone."
                    },
                    ...
                ]
        """
        # Default fallback for bindings without native diarization
        text = self.transcribe_audio(audio_source, model=model, **kwargs)
        default_speaker = participants[0] if (participants and len(participants) > 0) else "Speaker 1"
        if not text:
            return []
        return [{
            "speaker": default_speaker,
            "start": 0.0,
            "end": 0.0,
            "text": text.strip()
        }]

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available STT models supported by the binding.
        """
        pass

    def get_zoo(self) -> List[Dict[str, Any]]:
        """
        Returns a list of models available for download.
        each entry is a dict with:
        name, description, size, type, link
        """
        return []

    def download_from_zoo(self, index: int, progress_callback: Callable[[dict], None] = None) -> dict:
        """
        Downloads a model from the zoo using its index.
        """
        return {"status": False, "message": "Not implemented"}

class LollmsSTTBindingManager:
    """Manages STT binding discovery and instantiation."""

    def __init__(self, stt_bindings_dir: Union[str, Path] = Path(__file__).parent / "stt_bindings"):
        self.stt_bindings_dir = Path(stt_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific STT binding implementation."""
        binding_dir = self.stt_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.stt_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load STT binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      host_address: Optional[str] = None,
                      model_name: Optional[str] = None,
                      service_key: Optional[str] = None,
                      verify_ssl_certificate: bool = True,
                      **kwargs) -> Optional[LollmsSTTBinding]:
        """
        Create an instance of a specific STT binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                # Merge specific args into kwargs to match unified init
                kwargs.update({
                    "host_address": host_address,
                    "model_name": model_name,
                    "service_key": service_key,
                    "verify_ssl_certificate": verify_ssl_certificate
                })
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate STT binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available STT binding names based on subdirectories.
        """
        return [binding_dir.name for binding_dir in self.stt_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]


    @staticmethod
    def _get_fallback_description(binding_name: str) -> Dict:
        return {
            "binding_name": binding_name,
            "title": binding_name.replace("_", " ").title(),
            "author": "Unknown",
            "version": "N/A",
            "description": f"A binding for {binding_name}. No description.yaml file was found.",
            "input_parameters": [
                 {
                    "name": "model_name",
                    "type": "str",
                    "description": "The model name or ID to be used.",
                    "mandatory": False,
                    "default": ""
                }
            ],
            "generate_audio_parameters": []
        }
    @staticmethod
    def get_bindings_list(stt_bindings_dir: Union[str, Path]) -> List[Dict]:
        bindings_dir = Path(stt_bindings_dir)
        if not bindings_dir.is_dir():
            return []

        bindings_list = []
        for binding_folder in bindings_dir.iterdir():
            if binding_folder.is_dir() and (binding_folder / "__init__.py").exists():
                binding_name = binding_folder.name
                description_file = binding_folder / "description.yaml"
                
                binding_info = {}
                if description_file.exists():
                    try:
                        with open(description_file, 'r', encoding='utf-8') as f:
                            binding_info = yaml.safe_load(f)
                        binding_info['binding_name'] = binding_name
                    except Exception as e:
                        print(f"Error loading description.yaml for {binding_name}: {e}")
                        binding_info = LollmsSTTBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsSTTBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))
    

def get_available_bindings(stt_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    if stt_bindings_dir is None:
        stt_bindings_dir = Path(__file__).resolve().parent / "stt_bindings"
    return LollmsSTTBindingManager.get_bindings_list(stt_bindings_dir)

def list_binding_models(stt_binding_name: str, stt_binding_config: Optional[Dict[str, any]]|None = None, stt_bindings_dir: str|Path = Path(__file__).parent / "stt_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsSTTBindingManager(stt_bindings_dir).create_binding(
        binding_name=stt_binding_name,
        **{
            k: v
            for k, v in (stt_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
