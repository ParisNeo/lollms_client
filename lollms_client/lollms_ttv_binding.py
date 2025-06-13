# lollms_client/lollms_ttv_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception

class LollmsTTVBinding(ABC):
    """Abstract base class for all LOLLMS Text-to-Video bindings."""

    def __init__(self,
                 binding_name:str="unknown"):
        """
        Initialize the LollmsTTVBinding base class.

        Args:
            binding_name (Optional[str]): The binding name
        """
        self.binding_name = binding_name

    @abstractmethod
    def generate_video(self, prompt: str, **kwargs) -> bytes:
        """
        Generates video data from the provided text prompt.

        Args:
            prompt (str): The text prompt describing the desired video content.
            **kwargs: Additional binding-specific parameters (e.g., duration, fps, style, seed).

        Returns:
            bytes: The generated video data (e.g., in MP4 format).

        Raises:
            Exception: If video generation fails.
        """
        pass

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available TTV models or services supported by the binding.

        Args:
            **kwargs: Additional binding-specific parameters.

        Returns:
            List[str]: A list of available model/service identifiers.
        """
        pass

class LollmsTTVBindingManager:
    """Manages TTV binding discovery and instantiation."""

    def __init__(self, ttv_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "ttv_bindings"):
        """
        Initialize the LollmsTTVBindingManager.

        Args:
            ttv_bindings_dir (Union[str, Path]): Directory containing TTV binding implementations.
                                                 Defaults to the "ttv_bindings" subdirectory.
        """
        self.ttv_bindings_dir = Path(ttv_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific TTV binding implementation."""
        binding_dir = self.ttv_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.ttv_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTV binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTVBinding]:
        """
        Create an instance of a specific TTV binding.

        Args:
            binding_name (str): Name of the TTV binding to create.
            **kwargs: Additional parameters specific to the binding's __init__.

        Returns:
            Optional[LollmsTTVBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate TTV binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTV binding names based on subdirectories.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.ttv_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]