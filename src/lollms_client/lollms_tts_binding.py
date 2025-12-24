from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
import yaml
from ascii_colors import trace_exception
from lollms_client.lollms_base_binding import LollmsBaseBinding

class LollmsTTSBinding(LollmsBaseBinding):
    def __init__(self,
                 binding_name: str = "unknown",
                 **kwargs):
        super().__init__(binding_name=binding_name, **kwargs)
        self.settings = kwargs

    @abstractmethod
    def generate_audio(self,
                       text: str,
                       voice: Optional[str] = None,
                       **kwargs) -> bytes:
        pass

    @abstractmethod
    def list_voices(self, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
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

    def get_settings(self, **kwargs) -> Dict[str, Any]:
        return self.settings

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.settings.update(settings)
        return True

class LollmsTTSBindingManager:
    def __init__(self, tts_bindings_dir: Union[str, Path] = Path(__file__).parent / "tts_bindings"):
        self.tts_bindings_dir = Path(tts_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        binding_dir = self.tts_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.tts_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTS binding {binding_name}: {str(e)}")
    def create_binding(self, 
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTSBinding]:
        """
        Create an instance of a specific binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        return None


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
    def get_bindings_list(tts_bindings_dir: Union[str, Path]) -> List[Dict]:
        bindings_dir = Path(tts_bindings_dir)
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
                        binding_info = LollmsTTSBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsTTSBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))

    def get_available_bindings(self) -> list[str]:
        return [binding_dir.name for binding_dir in self.tts_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]

def get_available_bindings(tts_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    if tts_bindings_dir is None:
        tts_bindings_dir = Path(__file__).resolve().parent / "tts_bindings"
    return LollmsTTSBindingManager.get_bindings_list(tts_bindings_dir)

def list_binding_models(stt_binding_name: str, stt_binding_config: Optional[Dict[str, any]]|None = None, stt_bindings_dir: str|Path = Path(__file__).parent / "stt_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsTTSBindingManager(stt_bindings_dir).create_binding(
        binding_name=stt_binding_name,
        **{
            k: v
            for k, v in (stt_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
