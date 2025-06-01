# lollms_client/lollms_tti_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception

class LollmsTTIBinding(ABC):
    """Abstract base class for all LOLLMS Text-to-Image bindings."""

    def __init__(self,
                 binding_name:str="unknown"):
        """
        Initialize the LollmsTTIBinding base class.

        Args:
            binding_name (Optional[str]): The binding name
        """
        self.binding_name = binding_name

    @abstractmethod
    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 512,
                       height: int = 512,
                       **kwargs) -> bytes:
        """
        Generates image data from the provided text prompt.

        Args:
            prompt (str): The positive text prompt describing the desired image.
            negative_prompt (Optional[str]): Text prompt describing elements to avoid.
            width (int): The desired width of the image.
            height (int): The desired height of the image.
            **kwargs: Additional binding-specific parameters (e.g., seed, steps, cfg_scale).

        Returns:
            bytes: The generated image data (e.g., in PNG or JPEG format).

        Raises:
            Exception: If image generation fails.
        """
        pass

    @abstractmethod
    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """
        Lists the available TTI services or models supported by the binding.
        This might require authentication depending on the implementation.

        Args:
            **kwargs: Additional binding-specific parameters (e.g., client_id).

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each describing a service
                                  (e.g., {"name": "...", "caption": "...", "help": "..."}).
        """
        pass

    @abstractmethod
    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current settings for the active TTI service/model.
        This might require authentication depending on the implementation.

        Args:
            **kwargs: Additional binding-specific parameters (e.g., client_id).

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the settings structure
                                     (often a list matching ConfigTemplate format) or None if not supported/failed.
        """
        pass

    @abstractmethod
    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        """
        Applies new settings to the active TTI service/model.
        This might require authentication depending on the implementation.

        Args:
            settings (Dict[str, Any]): The new settings to apply (structure depends on the service).
            **kwargs: Additional binding-specific parameters (e.g., client_id).

        Returns:
            bool: True if settings were applied successfully, False otherwise.
        """
        pass

class LollmsTTIBindingManager:
    """Manages TTI binding discovery and instantiation."""

    def __init__(self, tti_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "tti_bindings"):
        """
        Initialize the LollmsTTIBindingManager.

        Args:
            tti_bindings_dir (Union[str, Path]): Directory containing TTI binding implementations.
                                                 Defaults to the "tti_bindings" subdirectory.
        """
        self.tti_bindings_dir = Path(tti_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific TTI binding implementation."""
        binding_dir = self.tti_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.tti_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTI binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTIBinding]:
        """
        Create an instance of a specific TTI binding.

        Args:
            binding_name (str): Name of the TTI binding to create.
            **kwargs: Additional parameters specific to the binding's __init__.

        Returns:
            Optional[LollmsTTIBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate TTI binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTI binding names based on subdirectories.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.tti_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]