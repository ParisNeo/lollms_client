# lollms_client/lollms_stt_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception

class LollmsSTTBinding(ABC):
    """Abstract base class for all LOLLMS Speech-to-Text bindings."""

    def __init__(self,
                 host_address: Optional[str] = None,
                 model_name: Optional[str] = None, # Can represent a default model
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True):
        """
        Initialize the LollmsSTTBinding base class.

        Args:
            host_address (Optional[str]): The host address for the STT service.
            model_name (Optional[str]): A default identifier for the STT model.
            service_key (Optional[str]): Authentication key for the service.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
        """
        if host_address is not None:
            self.host_address = host_address.rstrip('/')
        else:
            self.host_address = None
        self.model_name = model_name
        self.service_key = service_key
        self.verify_ssl_certificate = verify_ssl_certificate

    @abstractmethod
    def transcribe_audio(self, audio_path: Union[str, Path], model: Optional[str] = None, **kwargs) -> str:
        """
        Transcribes the audio file at the given path into text.

        Args:
            audio_path (Union[str, Path]): The path to the audio file to transcribe.
            model (Optional[str]): The specific STT model to use (if supported by the binding).
                                  If None, a default model might be used.
            **kwargs: Additional binding-specific parameters (e.g., language hint).

        Returns:
            str: The transcribed text.

        Raises:
            Exception: If audio transcription fails.
        """
        pass

    @abstractmethod
    def list_models(self, **kwargs) -> List[str]:
        """
        Lists the available STT models supported by the binding.

        Args:
            **kwargs: Additional binding-specific parameters.

        Returns:
            List[str]: A list of available STT model identifiers.
        """
        pass

class LollmsSTTBindingManager:
    """Manages STT binding discovery and instantiation."""

    def __init__(self, stt_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "stt_bindings"):
        """
        Initialize the LollmsSTTBindingManager.

        Args:
            stt_bindings_dir (Union[str, Path]): Directory containing STT binding implementations.
                                                 Defaults to the "stt_bindings" subdirectory.
        """
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

        Args:
            binding_name (str): Name of the STT binding to create.
            host_address (Optional[str]): Host address for the service.
            model_name (Optional[str]): Default model identifier.
            service_key (Optional[str]): Authentication key for the service.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
            **kwargs: Additional parameters specific to the binding's __init__.

        Returns:
            Optional[LollmsSTTBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(host_address=host_address,
                                     model_name=model_name,
                                     service_key=service_key,
                                     verify_ssl_certificate=verify_ssl_certificate,
                                     **kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate STT binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available STT binding names based on subdirectories.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.stt_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]