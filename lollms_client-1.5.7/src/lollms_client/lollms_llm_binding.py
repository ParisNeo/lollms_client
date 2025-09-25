# lollms_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, Callable, List, Union
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
import importlib
from pathlib import Path
from typing import Optional, Dict, List
from ascii_colors import trace_exception, ASCIIColors
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_utilities import ImageTokenizer
import re
import yaml

from pathlib import Path
import json

def load_known_contexts():
    """
    Loads the known_contexts data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the known_contexts data, or None if an error occurs.
    """
    try:
        file_path = Path(__file__).parent / "assets" / "models_ctx_sizes.json"
        with open(file_path, "r") as f:
            known_contexts = json.load(f)
        return known_contexts
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

class LollmsLLMBinding(ABC):
    """Abstract base class for all LOLLMS LLM bindings"""
    
    def __init__(self, 
                 binding_name: Optional[str] ="unknown",
                 **kwargs
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            binding_name (Optional[str]): The name of the bindingto be used
        """
        self.binding_name=binding_name
        self.model_name = None #Must be set by the instance
        self.default_ctx_size = kwargs.get("ctx_size") 
        self.default_n_predict = kwargs.get("n_predict")
        self.default_stream = kwargs.get("stream")
        self.default_temperature = kwargs.get("temperature")
        self.default_top_k = kwargs.get("top_k")
        self.default_top_p = kwargs.get("top_p")
        self.default_repeat_penalty = kwargs.get("repeat_penalty")
        self.default_repeat_last_n = kwargs.get("repeat_last_n")
        self.default_seed = kwargs.get("seed")
        self.default_n_threads = kwargs.get("n_threads")
        self.default_streaming_callback = kwargs.get("streaming_callback")

    
    @abstractmethod
    def generate_text(self,
                     prompt: str,
                     images: Optional[List[str]] = None,
                     system_prompt: str = "",
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repeat_penalty: Optional[float] = None,
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     split:Optional[bool]=False, # put to true if the prompt is a discussion
                     user_keyword:Optional[str]="!@>user:",
                     ai_keyword:Optional[str]="!@>assistant:",
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            prompt (str): The input prompt for text generation.
            images (Optional[List[str]]): List of image file paths for multimodal generation.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, str], None]]): Callback function for streaming output.
                - First parameter (str): The chunk of text received.
                - Second parameter (str): The message type (e.g., MSG_TYPE.MSG_TYPE_CHUNK).
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        pass

    def generate_from_messages(self,
                     messages: List[Dict],
                     n_predict: Optional[int] = None,
                     stream: Optional[bool] = None,
                     temperature: Optional[float] = None,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repeat_penalty: Optional[float] = None,
                     repeat_last_n: Optional[int] = None,
                     seed: Optional[int] = None,
                     n_threads: Optional[int] = None,
                     ctx_size: int | None = None,
                     streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                     **kwargs
                     ) -> Union[str, dict]:
        """
        Generate text using the active LLM binding, using instance defaults if parameters are not provided.

        Args:
            messages (List[Dict]): A openai compatible list of messages
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (int | None): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        ASCIIColors.red("This binding does not support generate_from_messages")


    @abstractmethod
    def chat(self,
             discussion: LollmsDiscussion,
             branch_tip_id: Optional[str] = None,
             n_predict: Optional[int] = None,
             stream: Optional[bool] = None,
             temperature: Optional[float] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             repeat_penalty: Optional[float] = None,
             repeat_last_n: Optional[int] = None,
             seed: Optional[int] = None,
             n_threads: Optional[int] = None,
             ctx_size: Optional[int] = None,
             streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None
             ) -> Union[str, dict]:
        """
        A method to conduct a chat session with the model using a LollmsDiscussion object.
        This method is responsible for formatting the discussion into the specific
        format required by the model's API and then calling the generation endpoint.

        Args:
            discussion (LollmsDiscussion): The discussion object containing the conversation history.
            branch_tip_id (Optional[str]): The ID of the message to use as the tip of the conversation branch. Defaults to the active branch.
            n_predict (Optional[int]): Maximum number of tokens to generate.
            stream (Optional[bool]): Whether to stream the output.
            temperature (Optional[float]): Sampling temperature.
            top_k (Optional[int]): Top-k sampling parameter.
            top_p (Optional[float]): Top-p sampling parameter.
            repeat_penalty (Optional[float]): Penalty for repeated tokens.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty.
            seed (Optional[int]): Random seed for generation.
            n_threads (Optional[int]): Number of threads to use.
            ctx_size (Optional[int]): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: The generated text or an error dictionary.
        """
        pass

    def get_ctx_size(self, model_name: Optional[str|None] = None) -> Optional[int]:
        """
        Retrieves context size for a model from a hardcoded list.

        This method checks if the model name contains a known base model identifier
        (e.g., 'llama3.1', 'gemma2') to determine its context length. It's intended
        as a failsafe when the context size cannot be retrieved directly from the
        Ollama API.
        """
        if model_name is None:
            model_name = self.model_name

        # Hardcoded context sizes for popular models. More specific names (e.g., 'llama3.1')
        # should appear, as they will be checked first due to the sorting logic below.
        known_contexts = load_known_contexts()

        normalized_model_name = model_name.lower().strip()

        # Sort keys by length in descending order. This ensures that a more specific
        # name like 'llama3.1' is checked before a less specific name like 'llama3'.
        sorted_base_models = sorted(known_contexts.keys(), key=len, reverse=True)

        for base_name in sorted_base_models:
            if base_name in normalized_model_name:
                context_size = known_contexts[base_name]
                ASCIIColors.warning(
                    f"Using hardcoded context size for model '{model_name}' "
                    f"based on base name '{base_name}': {context_size}"
                )
                return context_size

        ASCIIColors.warning(f"Context size not found for model '{model_name}' in the hardcoded list.")
        return None


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
    def count_tokens(self, text: str) -> int:
        """
        Count tokens from a text.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            int: Number of tokens in text.
        """        
        pass

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.

        Args:
            image (str): Image to count tokens from. Either base64 string, path to image file, or URL.

        Returns:
            int: Estimated number of tokens for the image. Returns -1 on error.
        """
        try:
            # Delegate token counting to ImageTokenizer
            return ImageTokenizer(self.model_name).count_image_tokens(image)
        except Exception as e:
            ASCIIColors.warning(f"Could not estimate image tokens: {e}")
            return -1
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


    def split_discussion(self, lollms_prompt_string: str, system_keyword="!@>system:", user_keyword="!@>user:", ai_keyword="!@>assistant:") -> list:
        """
        Splits a LoLLMs prompt into a list of OpenAI-style messages.
        If the very first chunk has no prefix, it's assigned to "system".
        """
        # Regex to split on any of the three prefixes (lookahead)
        pattern = r"(?={}|{}|{})".format(
            re.escape(system_keyword),
            re.escape(user_keyword),
            re.escape(ai_keyword)
        )
        parts = re.split(pattern, lollms_prompt_string)
        messages = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Determine role and strip prefix if present
            if part.startswith(system_keyword):
                role = "system"
                content = part[len(system_keyword):].strip()
            elif part.startswith(user_keyword):
                role = "user"
                content = part[len(user_keyword):].strip()
            elif part.startswith(ai_keyword):
                role = "assistant"
                content = part[len(ai_keyword):].strip()
            else:
                # No prefix: if it's the first valid chunk, treat as system
                if not messages:
                    role = "system"
                    content = part
                else:
                    # otherwise skip unrecognized segments
                    continue

            messages.append({"role": role, "content": content})
            if messages[-1]["content"]=="":
                del messages[-1]
        return messages
    def ps(self):
        return []



class LollmsLLMBindingManager:
    """Manages binding discovery and instantiation"""

    def __init__(self, llm_bindings_dir: Union[str, Path] = Path(__file__).parent.parent / "llm_bindings"):
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
                      **kwargs) -> Optional[LollmsLLMBinding]:
        """
        Create an instance of a specific binding.

        Args:
            binding_name (str): Name of the binding to create.
            kwargs: binding specific arguments

        Returns:
            Optional[LollmsLLMBinding]: Binding instance or None if creation failed.
        """
        if binding_name not in self.available_bindings:
            self._load_binding(binding_name)
        
        binding_class = self.available_bindings.get(binding_name)
        if binding_class:
            return binding_class(**kwargs)
        return None
    @staticmethod
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
                        binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                else:
                    binding_info = LollmsLLMBindingManager._get_fallback_description(binding_name)
                
                bindings_list.append(binding_info)

        return sorted(bindings_list, key=lambda b: b.get('title', b['binding_name']))
    
    def get_available_bindings(self) -> List[Dict]:
        """
        Retrieves a list of all available LLM bindings with their full descriptions.

        This method scans the configured `llm_bindings_dir`, parsing the `description.yaml`
        file for each valid binding. If a `description.yaml` is missing, a fallback
        description with common parameters is generated. This is the primary method
        for discovering available bindings and their configuration requirements.

        Returns:
            List[Dict]: 
                A list of dictionaries, where each dictionary represents the
                full description of an available binding.

                Each dictionary contains the following keys:
                - ``binding_name`` (str): The programmatic name of the binding (its folder name).
                - ``title`` (str): A user-friendly title for the binding.
                - ``author`` (str): The creator of the binding.
                - ``creation_date`` (str): The date the binding was created.
                - ``last_update_date`` (str): The date of the last major update.
                - ``description`` (str): A detailed explanation of the binding's purpose.
                - ``input_parameters`` (List[Dict]): A list of parameters required to
                  configure the binding. Each parameter is a dictionary with:
                    - ``name`` (str): The parameter's name (e.g., 'model_name').
                    - ``type`` (str): The expected data type ('str', 'int', 'float', 'bool').
                    - ``description`` (str): A user-friendly description of the parameter.
                    - ``mandatory`` (bool): True if the parameter must be provided.
                    - ``default``: The default value for the parameter.

        Example of a returned dictionary in the list:
        .. code-block:: python

            {
                "binding_name": "ollama",
                "title": "Ollama",
                "author": "ParisNeo",
                ...
                "input_parameters": [
                    {
                        "name": "host_address",
                        "type": "str",
                        "description": "The URL of the Ollama server.",
                        "mandatory": True,
                        "default": "http://localhost:11434"
                    },
                    ...
                ]
            }
        """
        return LollmsLLMBindingManager.get_bindings_list(self.llm_bindings_dir)

def get_available_bindings(llm_bindings_dir: Union[str, Path] = None) -> List[Dict]:
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
    if llm_bindings_dir is None:
        llm_bindings_dir = Path(__file__).parent / "llm_bindings"
    return LollmsLLMBindingManager.get_bindings_list(llm_bindings_dir)