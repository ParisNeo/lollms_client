# lollms_client/lollms_ttm_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception

class LollmsTTMBinding(ABC):
    """Abstract base class for all LOLLMS Text-to-Music bindings."""

    def __init__(self,
                 binding_name:str="unknown"):
        """
        Initialize the LollmsTTMBinding base class.

        Args:
            binding_name (Optional[str]): The binding name
        """
        self.binding_name = binding_name


    @abstractmethod
    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music data from the provided text prompt.

        Args:
            prompt (str): The text prompt describing the desired music (e.g., genre, mood, instruments).
            **kwargs: Additional binding-specific parameters (e.g., duration, style, seed).

        Returns:
            bytes: The generated music data (e.g., in WAV, MP3, or MIDI format).

        Raises:
            Exception: If music generation fails.
        """
        pass

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available TTM models or services supported by the binding.

        Args:
            **kwargs: Additional binding-specific parameters.

        Returns:
            List[str]: A list of available model/service identifiers.
        """
        pass

class LollmsTTMBindingManager:
    """Manages TTM binding discovery and instantiation."""

    def __init__(self, ttm_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "ttm_bindings"):
        """
        Initialize the LollmsTTMBindingManager.

        Args:
            ttm_bindings_dir (Union[str, Path]): Directory containing TTM binding implementations.
                                                 Defaults to the "ttm_bindings" subdirectory.
        """
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

        Args:
            binding_name (str): Name of the TTM binding to create.
            **kwargs: Additional parameters specific to the binding's __init__.

        Returns:
            Optional[LollmsTTMBinding]: Binding instance or None if creation failed.
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

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.ttm_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]