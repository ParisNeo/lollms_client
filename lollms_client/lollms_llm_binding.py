# lollms_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, Callable, List
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
import importlib
from pathlib import Path
from typing import Optional
from ascii_colors import trace_exception

class LollmsLLMBinding(ABC):
    """Abstract base class for all LOLLMS LLM bindings"""
    
    def __init__(self, 
                 host_address: Optional[str] = None,
                 model_name: str = "",
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True,
                 default_completion_format: ELF_COMPLETION_FORMAT = ELF_COMPLETION_FORMAT.Chat
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            host_address (Optional[str]): The host address for the service. Defaults to None.
            model_name (str): The name of the model to use. Defaults to empty string.
            service_key (Optional[str]): Authentication key for the service. Defaults to None.
            verify_ssl_certificate (bool): Whether to verify SSL certificates. Defaults to True.
            default_completion_format (ELF_COMPLETION_FORMAT): The completion format (Chat or Instruct)
        """
        if host_address is not None:
            self.host_address = host_address[:-1] if host_address.endswith("/") else host_address
        else:
            self.host_address = None
        self.model_name = model_name
        self.service_key = service_key
        self.verify_ssl_certificate = verify_ssl_certificate
        self.default_completion_format = default_completion_format
    
    @abstractmethod
    def generate_text(self, 
                     prompt: str,
                     images: Optional[List[str]] = None,
                     n_predict: Optional[int] = None,
                     stream: bool = False,
                     temperature: float = 0.1,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     repeat_penalty: float = 0.8,
                     repeat_last_n: int = 40,
                     seed: Optional[int] = None,
                     n_threads: int = 8,
                     streaming_callback: Optional[Callable[[str, str], None]] = None) -> str:
        """
        Generate text based on the provided prompt and parameters.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): Sampling temperature. Defaults to 0.1.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p sampling parameter. Defaults to 0.95.
            repeat_penalty (float): Penalty for repeated tokens. Defaults to 0.8.
            repeat_last_n (int): Number of previous tokens to consider for repeat penalty. Defaults to 40.
            seed (Optional[int]): Random seed for generation.
            n_threads (int): Number of threads to use. Defaults to 8.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).

        Returns:
            str: Generated text or error dictionary if failed.
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        pass
    
    @abstractmethod
    def detokenize(self, tokens: list) -> str:
        """
        Convert a list of tokens back to text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> list:
        """
        Get embeddings for the input text using Ollama API
        
        Args:
            text (str or List[str]): Input text to embed
            **kwargs: Additional arguments like model, truncate, options, keep_alive
        
        Returns:
            dict: Response containing embeddings
        """
        pass    
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Return information about the current model.

        Returns:
            dict: Model information dictionary.
        """
        pass

    @abstractmethod
    def listModels(self) -> list:
        """Lists models"""
        pass
    
    
    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        pass


class LollmsLLMBindingManager:
    """Manages binding discovery and instantiation"""

    def __init__(self, llm_bindings_dir: str = "llm_bindings"):
        """
        Initialize the LollmsLLMBindingManager.

        Args:
            llm_bindings_dir (str): Directory containing binding implementations. Defaults to "llm_bindings".
        """
        self.llm_bindings_dir = Path(llm_bindings_dir)
        self.available_bindings = {}

    def _load_binding(self, binding_name: str):
        """Dynamically load a specific binding implementation from the llm bindings directory."""
        binding_dir = self.llm_bindings_dir / binding_name
        if binding_dir.is_dir() and (binding_dir / "__init__.py").exists():
            try:
                module = importlib.import_module(f"lollms_client.llm_bindings.{binding_name}")
                binding_class = getattr(module, module.BindingName)
                self.available_bindings[binding_name] = binding_class
            except Exception as e:
                trace_exception(e)
                print(f"Failed to load binding {binding_name}: {str(e)}")

    def create_binding(self, 
                      binding_name: str,
                      host_address: Optional[str] = None,
                      model_name: str = "",
                      service_key: Optional[str] = None,
                      verify_ssl_certificate: bool = True,
                      personality: Optional[int] = None) -> Optional[LollmsLLMBinding]:
        """
        Create an instance of a specific binding.

        Args:
            binding_name (str): Name of the binding to create.
            host_address (Optional[str]): Host address for the service.
            model_name (str): Name of the model to use.
            service_key (Optional[str]): Authentication key for the service.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
            personality (Optional[int]): Personality ID for LOLLMS binding.

        Returns:
            Optional[LollmsLLMBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(host_address, model_name, service_key, verify_ssl_certificate, personality)
        return None

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available binding names.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.llm_bindings_dir.iterdir() if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]
