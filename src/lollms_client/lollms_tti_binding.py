# lollms_client/lollms_tti_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception
import yaml
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
    def listModels(self) -> list:
        """Lists models"""
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
    def _get_fallback_description(binding_name: str) -> Dict:
        """
        Generates a default description dictionary for a binding without a description.yaml file.
        """
        return {
            "binding_name": binding_name,
            "title": binding_name.replace("_", " ").title(),
            "author": "Unknown",
            "creation_date": "N/A",
            "last_update_date": "N/A",
            "description": f"A binding for {binding_name}. No description.yaml file was found, so common parameters are shown as a fallback.",
            "input_parameters": [
                {
                    "name": "model_name",
                    "type": "str",
                    "description": "The model name, ID, or filename to be used.",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "host_address",
                    "type": "str",
                    "description": "The host address of the service (for API-based bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "models_path",
                    "type": "str",
                    "description": "The path to the models directory (for local bindings).",
                    "mandatory": False,
                    "default": ""
                },
                {
                    "name": "service_key",
                    "type": "str",
                    "description": "The API key or service key for authentication (if applicable).",
                    "mandatory": False,
                    "default": ""
                }
            ]
        }

    @staticmethod
    def get_bindings_list(llm_bindings_dir: Union[str, Path]) -> List[Dict]:
        """
        Lists all available LLM bindings by scanning a directory, loading their
        description.yaml file if present, or providing a default description.

        Args:
            llm_bindings_dir (Union[str, Path]): The path to the directory containing LLM binding folders.

        Returns:
            List[Dict]: A list of dictionaries, each describing a binding.
        """
        bindings_dir = Path(llm_bindings_dir)
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
                        binding_info = LollmsTTIBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsTTIBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))
    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTI binding names based on subdirectories.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.tti_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]

def get_available_bindings(tti_bindings_dir: Union[str, Path] = None) -> List[Dict]:
    """
    Lists all available LLM bindings with their detailed descriptions.

    This function serves as a primary entry point for discovering what bindings
    are available and how to configure them.

    Args:
        llm_bindings_dir (Union[str, Path], optional): 
            The path to the LLM bindings directory. If None, it defaults to the
            'llm_bindings' subdirectory relative to this file. 
            Defaults to None.

    Returns:
        List[Dict]: A list of dictionaries, each describing a binding.
    """
    if tti_bindings_dir is None:
        tti_bindings_dir = Path(__file__).parent / "tti_bindings"
    return LollmsTTIBindingManager.get_bindings_list(tti_bindings_dir)