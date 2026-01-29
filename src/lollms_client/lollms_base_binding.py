from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from pathlib import Path
import yaml
import inspect
from ascii_colors import ASCIIColors

class LollmsBaseBinding(ABC):
    """
    Base class for all LOLLMS bindings (LLM, TTI, TTS, STT, TTM, TTV, MCP).
    Enforces a unified initialization and common methods.
    """
    def __init__(self, binding_name: str, debug:Optional[bool]=False, **kwargs):
        """
        Initialize the binding.
        
        Args:
            binding_name (str): The name of the binding.
            **kwargs: Configuration parameters passed from the manager/app.
        """
        self.binding_name = binding_name
        self.debug = debug
        self.config = kwargs
        self.binding_dir = self._get_binding_dir()
        self.description = self._load_description()
        
    def _get_binding_dir(self) -> Path:
        """
        Locates the directory of the concrete binding class.
        """
        try:
            return Path(inspect.getfile(self.__class__)).parent
        except Exception:
            return Path(".")

    def _load_description(self) -> Dict:
        """
        Loads the description.yaml file from the binding directory.
        """
        desc_file = self.binding_dir / "description.yaml"
        if desc_file.exists():
            try:
                with open(desc_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                ASCIIColors.error(f"Failed to load description.yaml for {self.binding_name}: {e}")
        return {}

    @abstractmethod
    def list_models(self) -> List[Any]:
        """
        List available models or resources provided by this binding.
        Must be implemented by all bindings.
        """
        pass

    def ps(self) -> List[Any]:
        """
        Verify resources or processes associated with this binding.
        Returns a list of status information.
        """
        return []
