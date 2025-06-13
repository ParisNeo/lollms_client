# lollms_client/lollms_tts_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from ascii_colors import trace_exception

class LollmsTTSBinding(ABC):
    """Abstract base class for all LOLLMS Text-to-Speech bindings."""

    def __init__(self,
                 binding_name:str="unknown"):
        """
        Initialize the LollmsTTSBinding base class.

        Args:
            binding_name (Optional[str]): The binding name
        """
        self.binding_name = binding_name

    @abstractmethod
    def generate_audio(self, text: str, voice: Optional[str] = None, **kwargs) -> bytes:
        """
        Generates audio data from the provided text.

        Args:
            text (str): The text content to synthesize.
            voice (Optional[str]): The specific voice or model to use (if supported by the binding).
                                   If None, a default voice might be used.
            **kwargs: Additional binding-specific parameters.

        Returns:
            bytes: The generated audio data (e.g., in WAV or MP3 format).

        Raises:
            Exception: If audio generation fails.
        """
        pass

    @abstractmethod
    def list_voices(self, **kwargs) -> List[str]:
        """
        Lists the available voices or TTS models supported by the binding.

        Args:
            **kwargs: Additional binding-specific parameters.

        Returns:
            List[str]: A list of available voice/model identifiers.
        """
        pass

class LollmsTTSBindingManager:
    """Manages TTS binding discovery and instantiation."""

    def __init__(self, tts_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "tts_bindings"):
        """
        Initialize the LollmsTTSBindingManager.

        Args:
            tts_bindings_dir (Union[str, Path]): Directory containing TTS binding implementations.
                                                 Defaults to the "tts_bindings" subdirectory.
        """
        self.tts_bindings_dir = Path(tts_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific TTS binding implementation."""
        binding_dir = self.tts_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                # Adjust module path for dynamic loading
                module = importlib.import_module(f"lollms_client.tts_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName) # Assumes BindingName is defined
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load TTS binding {binding_name}: {str(e)}")

    def create_binding(self,
                      binding_name: str,
                      **kwargs) -> Optional[LollmsTTSBinding]:
        """
        Create an instance of a specific TTS binding.

        Args:
            binding_name (str): Name of the TTS binding to create.
            **kwargs: Additional parameters specific to the binding's __init__.

        Returns:
            Optional[LollmsTTSBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)

        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            try:
                return binding_class(**kwargs)
            except Exception as e:
                trace_exception(e)
                print(f"Failed to instantiate TTS binding {binding_name}: {str(e)}")
                return None
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available TTS binding names based on subdirectories.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.tts_bindings_dir.iterdir()
                if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]