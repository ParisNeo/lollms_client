# lollms_binding.py
from abc import ABC, abstractmethod
import importlib
from pathlib import Path
from typing import Optional, Callable, List, Union
from lollms_client.lollms_types import ELF_COMPLETION_FORMAT
import importlib
from pathlib import Path
from typing import Optional
from ascii_colors import trace_exception
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import LollmsDiscussion
import re
class LollmsLLMBinding(ABC):
    """Abstract base class for all LOLLMS LLM bindings"""
    
    def __init__(self, 
                 binding_name: Optional[str] ="unknown"
        ):
        """
        Initialize the LollmsLLMBinding base class.

        Args:
            binding_name (Optional[str]): The name of the bindingto be used
        """
        self.binding_name=binding_name
        self.model_name = None #Must be set by the instance
    
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

    def get_available_bindings(self) -> list[str]:
        """
        Return list of available binding names.

        Returns:
            list[str]: List of binding names.
        """
        return [binding_dir.name for binding_dir in self.llm_bindings_dir.iterdir() if binding_dir.is_dir() and (binding_dir / "__init__.py").exists()]
