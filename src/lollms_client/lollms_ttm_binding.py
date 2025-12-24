# lollms_client/lollms_ttm_binding.py
from abc import abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception
from lollms_client.lollms_base_binding import LollmsBaseBinding

class LollmsTTMBinding(LollmsBaseBinding):
    """Abstract base class for all LOLLMS Text-to-Music bindings."""

    def __init__(self,
                 binding_name:str="unknown",
                 **kwargs):
        """
        Initialize the LollmsTTMBinding base class.
        """
        super().__init__(binding_name=binding_name, **kwargs)


    @abstractmethod
    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music data from the provided text prompt.
        """
        pass

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available TTM models or services supported by the binding.
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

class LollmsTTMBindingManager:
    """Manages TTM binding discovery and instantiation."""

    def __init__(self, ttm_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "ttm_bindings"):
        self.ttm_bindings_dir = Path(ttm_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific TTM binding implementation."""
        binding_dir = self.ttm_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.ttm_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTM binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTMBinding]:
        """
        Create an instance of a specific TTM binding.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate TTM binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTM binding names based on subdirectories.
        """
        return [binding_dir.name for binding_dir in self.ttm_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]


def list_binding_models(ttm_binding_name: str, ttm_binding_config: Optional[Dict[str, any]]|None = None, ttm_bindings_dir: str|Path = Path(__file__).parent / "ttm_bindings") -> List[Dict]:
    """
    Lists all available models for a specific binding.
    """
    binding = LollmsTTMBindingManager(ttm_bindings_dir).create_binding(
        binding_name=ttm_binding_name,
        **{
            k: v
            for k, v in (ttm_binding_config or {}).items()
            if k != "binding_name"
        }
    )

    return binding.list_models() if binding else []
