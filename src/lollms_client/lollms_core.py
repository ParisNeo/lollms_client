# lollms_client/lollms_core.py
import requests
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import robust_json_parser # Keep utilities needed by core
from lollms_client.lollms_llm_binding import LollmsLLMBinding, LollmsLLMBindingManager
# Import new Abstract Base Classes and Managers
from lollms_client.lollms_tts_binding import LollmsTTSBinding, LollmsTTSBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBinding, LollmsTTIBindingManager
from lollms_client.lollms_stt_binding import LollmsSTTBinding, LollmsSTTBindingManager
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, LollmsTTVBindingManager
from lollms_client.lollms_ttm_binding import LollmsTTMBinding, LollmsTTMBindingManager
from lollms_client.lollms_mcp_binding import LollmsMCPBinding, LollmsMCPBindingManager

from lollms_client.lollms_discussion import LollmsDiscussion

from lollms_client.lollms_agentic import TaskStatus, TaskPlanner, MemoryManager, UncertaintyManager, ToolPerformanceTracker

from lollms_client.lollms_utilities import build_image_dicts, dict_to_markdown
import json, re
from enum import Enum
import base64
import requests
from typing import List, Optional, Callable, Union, Dict, Any
import numpy as np
from pathlib import Path
import uuid
import hashlib
import time
class LollmsClient():
    """
    Core client class for interacting with LOLLMS services, including LLM, TTS, TTI, STT, TTV, and TTM.
    Provides a unified interface to manage and use different bindings for various modalities.
    """
    def __init__(self,

                 # Optional Modality Binding Names
                 llm_binding_name: Optional[str] = None,
                 tts_binding_name: Optional[str] = None,
                 tti_binding_name: Optional[str] = None,
                 stt_binding_name: Optional[str] = None,
                 ttv_binding_name: Optional[str] = None,
                 ttm_binding_name: Optional[str] = None,
                 mcp_binding_name: Optional[str] = None,

                 # Modality Binding Directories
                 llm_bindings_dir: Path = Path(__file__).parent / "llm_bindings",
                 tts_bindings_dir: Path = Path(__file__).parent / "tts_bindings",
                 tti_bindings_dir: Path = Path(__file__).parent / "tti_bindings",
                 stt_bindings_dir: Path = Path(__file__).parent / "stt_bindings",
                 ttv_bindings_dir: Path = Path(__file__).parent / "ttv_bindings",
                 ttm_bindings_dir: Path = Path(__file__).parent / "ttm_bindings",
                 mcp_bindings_dir: Path = Path(__file__).parent / "mcp_bindings",

                 # Configurations
                 llm_binding_config: Optional[Dict[str, any]] = None,
                 tts_binding_config: Optional[Dict[str, any]] = None, 
                 tti_binding_config: Optional[Dict[str, any]] = None, 
                 stt_binding_config: Optional[Dict[str, any]] = None, 
                 ttv_binding_config: Optional[Dict[str, any]] = None, 
                 ttm_binding_config: Optional[Dict[str, any]] = None, 
                 mcp_binding_config: Optional[Dict[str, any]] = None,
                 user_name ="user",
                 ai_name = "assistant",
                 **kwargs
                 ):
        """
        Initialize the LollmsClient with LLM and optional modality bindings.

        Args:
            binding_name (str): Name of the primary LLM binding (e.g., "lollms", "ollama").
            host_address (Optional[str]): Default host address for all services. Overridden by binding defaults if None.
            models_path (Optional[str]): Default models folder path. Overridden by binding defaults if None.
            model_name (str): Default model name for the LLM binding.
            llm_bindings_dir (Path): Directory for LLM binding implementations.
            llm_binding_config (Optional[Dict]): Additional config for the LLM binding.
            tts_binding_name (Optional[str]): Name of the TTS binding to use (e.g., "lollms").
            tti_binding_name (Optional[str]): Name of the TTI binding to use (e.g., "lollms").
            stt_binding_name (Optional[str]): Name of the STT binding to use (e.g., "lollms").
            ttv_binding_name (Optional[str]): Name of the TTV binding to use (e.g., "lollms").
            ttm_binding_name (Optional[str]): Name of the TTM binding to use (e.g., "lollms").
            tts_bindings_dir (Path): Directory for TTS bindings.
            tti_bindings_dir (Path): Directory for TTI bindings.
            stt_bindings_dir (Path): Directory for STT bindings.
            ttv_bindings_dir (Path): Directory for TTV bindings.
            ttm_bindings_dir (Path): Directory for TTM bindings.
            tts_binding_config (Optional[Dict]): Additional config for the TTS binding.
            tti_binding_config (Optional[Dict]): Additional config for the TTI binding.
            stt_binding_config (Optional[Dict]): Additional config for the STT binding.
            ttv_binding_config (Optional[Dict]): Additional config for the TTV binding.
            ttm_binding_config (Optional[Dict]): Additional config for the TTM binding.
            service_key (Optional[str]): Shared authentication key or client_id.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
            ctx_size (Optional[int]): Default context size for LLM.
            n_predict (Optional[int]): Default max tokens for LLM.
            stream (bool): Default streaming mode for LLM.
            temperature (float): Default temperature for LLM.
            top_k (int): Default top_k for LLM.
            top_p (float): Default top_p for LLM.
            repeat_penalty (float): Default repeat penalty for LLM.
            repeat_last_n (int): Default repeat last n for LLM.
            seed (Optional[int]): Default seed for LLM.
            n_threads (int): Default threads for LLM.
            streaming_callback (Optional[Callable]): Default streaming callback for LLM.
            user_name (str): Default user name for prompts.
            ai_name (str): Default AI name for prompts.

        Raises:
            ValueError: If the primary LLM binding cannot be created.
        """
        # --- LLM Binding Setup ---
        # --- Modality Binding Setup ---
        self.llm_binding_manager = LollmsLLMBindingManager(llm_bindings_dir)
        self.tts_binding_manager = LollmsTTSBindingManager(tts_bindings_dir)
        self.tti_binding_manager = LollmsTTIBindingManager(tti_bindings_dir)
        self.stt_binding_manager = LollmsSTTBindingManager(stt_bindings_dir)
        self.ttv_binding_manager = LollmsTTVBindingManager(ttv_bindings_dir)
        self.ttm_binding_manager = LollmsTTMBindingManager(ttm_bindings_dir)
        self.mcp_binding_manager = LollmsMCPBindingManager(mcp_bindings_dir)


        self.llm: Optional[LollmsLLMBinding] = None
        self.tts: Optional[LollmsTTSBinding] = None
        self.tti: Optional[LollmsTTIBinding] = None
        self.stt: Optional[LollmsSTTBinding] = None
        self.ttv: Optional[LollmsTTVBinding] = None
        self.ttm: Optional[LollmsTTMBinding] = None
        self.mcp: Optional[LollmsMCPBinding] = None


        if llm_binding_name:
            self.llm = self.llm_binding_manager.create_binding(
                binding_name=llm_binding_name,
                **{
                    k: v
                    for k, v in (llm_binding_config or {}).items()
                    if k != "binding_name"
                }
            )

            if self.llm is None:
                available = self.llm_binding_manager.get_available_bindings()
                ASCIIColors.warning(f"Failed to create LLM binding: {llm_binding_name}. Available: {available}")

        if tts_binding_name:
            self.tts = self.tts_binding_manager.create_binding(
                binding_name=tts_binding_name,
                **{
                    k: v
                    for k, v in (tts_binding_config or {}).items()
                    if k != "binding_name"
                }
            )
            if self.tts is None:
                ASCIIColors.warning(f"Failed to create TTS binding: {tts_binding_name}. Available: {self.tts_binding_manager.get_available_bindings()}")

        if tti_binding_name:
            if tti_binding_config:
                self.tti = self.tti_binding_manager.create_binding(
                    binding_name=tti_binding_name,
                    **{
                        k: v
                        for k, v in (tti_binding_config or {}).items()
                        if k != "binding_name"
                    }
                )
            else:
                self.tti = self.tti_binding_manager.create_binding(
                    binding_name=tti_binding_name
                )
            if self.tti is None:
                ASCIIColors.warning(f"Failed to create TTI binding: {tti_binding_name}. Available: {self.tti_binding_manager.get_available_bindings()}")

        if stt_binding_name:
            if stt_binding_config:
                self.stt = self.stt_binding_manager.create_binding(
                    binding_name=stt_binding_name,
                    **{
                        k: v
                        for k, v in (stt_binding_config or {}).items()
                        if k != "binding_name"
                    }
                )

            else:
                self.stt = self.stt_binding_manager.create_binding(
                    binding_name=stt_binding_name,
                )
            if self.stt is None:
                ASCIIColors.warning(f"Failed to create STT binding: {stt_binding_name}. Available: {self.stt_binding_manager.get_available_bindings()}")
        if ttv_binding_name:
            if ttv_binding_config:
                self.ttv = self.ttv_binding_manager.create_binding(
                    binding_name=ttv_binding_name,
                    **{
                        k: v
                        for k, v in ttv_binding_config.items()
                        if k != "binding_name"
                    }
                )

            else:
                self.ttv = self.ttv_binding_manager.create_binding(
                    binding_name=ttv_binding_name
                )
            if self.ttv is None:
                ASCIIColors.warning(f"Failed to create TTV binding: {ttv_binding_name}. Available: {self.ttv_binding_manager.get_available_bindings()}")

        if ttm_binding_name:
            if ttm_binding_config:
                self.ttm = self.ttm_binding_manager.create_binding(
                    binding_name=ttm_binding_name,
                    **{
                        k: v
                        for k, v in (ttm_binding_config or {}).items()
                        if k != "binding_name"
                    }
                )
            else:
                self.ttm = self.ttm_binding_manager.create_binding(
                    binding_name=ttm_binding_name
                )
            if self.ttm is None:
                ASCIIColors.warning(f"Failed to create TTM binding: {ttm_binding_name}. Available: {self.ttm_binding_manager.get_available_bindings()}")

        if mcp_binding_name:
            if mcp_binding_config:
                self.mcp = self.mcp_binding_manager.create_binding(
                    binding_name=mcp_binding_name,
                    **{
                        k: v
                        for k, v in (mcp_binding_config or {}).items()
                        if k != "binding_name"
                    }
                )
            else:
                self.mcp = self.mcp_binding_manager.create_binding(
                    mcp_binding_name
                )
            if self.mcp is None:
                ASCIIColors.warning(f"Failed to create MCP binding: {mcp_binding_name}. Available: {self.mcp_binding_manager.get_available_bindings()}")

        # --- Store Default Generation Parameters ---

        # --- Prompt Formatting Attributes ---
        self.user_name = user_name
        self.ai_name = ai_name
        self.start_header_id_template ="!@>"
        self.end_header_id_template =": "
        self.system_message_template ="system"
        self.separator_template ="!@>"
        self.start_user_header_id_template ="!@>"
        self.end_user_header_id_template =": "
        self.end_user_message_id_template =""
        self.start_ai_header_id_template ="!@>"
        self.end_ai_header_id_template =": "
        self.end_ai_message_id_template =""

    # 
    def update_llm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the LLM binding with a new configuration."""
        self.llm = self.llm_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.llm is None:
            available = self.llm_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update LLM binding: {binding_name}. Available: {available}")

    def get_ctx_size(self, model_name:str|None=None):
        if self.llm:
            ctx_size = self.llm.get_ctx_size(model_name)
            return ctx_size if ctx_size else self.llm.default_ctx_size
        else:
            return None

    def get_model_name(self):
        if self.llm:
            return self.llm.model_name
        else:
            return None

    def set_model_name(self, model_name)->bool:
        if self.llm:
            self.llm.model_name = model_name
            return True
        else:
            return False

    def update_tts_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the TTS binding with a new configuration."""
        self.tts = self.tts_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.tts is None:
            available = self.tts_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update TTS binding: {binding_name}. Available: {available}")

    def update_tti_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the TTI binding with a new configuration."""
        self.tti = self.tti_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.tti is None:
            available = self.tti_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update TTI binding: {binding_name}. Available: {available}")

    def update_stt_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the STT binding with a new configuration."""
        self.stt = self.stt_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.stt is None:
            available = self.stt_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update STT binding: {binding_name}. Available: {available}")

    def update_ttv_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the TTV binding with a new configuration."""
        self.ttv = self.ttv_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.ttv is None:
            available = self.ttv_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update TTV binding: {binding_name}. Available: {available}")

    def update_ttm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the TTM binding with a new configuration."""
        self.ttm = self.ttm_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.ttm is None:
            available = self.ttm_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update TTM binding: {binding_name}. Available: {available}")

    def update_mcp_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        """Update the MCP binding with a new configuration."""
        self.mcp = self.mcp_binding_manager.create_binding(
            binding_name=binding_name,
            **(config or {})
        )
        if self.mcp is None:
            available = self.mcp_binding_manager.get_available_bindings()
            raise ValueError(f"Failed to update MCP binding: {binding_name}. Available: {available}")

    # --- Prompt Formatting Properties ---
    @property
    def system_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_header_id_template}{self.system_message_template}{self.end_header_id_template}"

    def system_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_header_id_template}{ai_name}{self.end_header_id_template}"

    @property
    def user_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{self.user_name}{self.end_user_header_id_template}"

    def user_custom_header(self, user_name="user") -> str:
        """Get the start_header_id_template."""
        return f"{self.start_user_header_id_template}{user_name}{self.end_user_header_id_template}"

    @property
    def ai_full_header(self) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_ai_header_id_template}{self.ai_name}{self.end_ai_header_id_template}"

    def ai_custom_header(self, ai_name) -> str:
        """Get the start_header_id_template."""
        return f"{self.start_ai_header_id_template}{ai_name}{self.end_ai_header_id_template}"

    def sink(self, s=None,i=None,d=None):
        """Placeholder sink method."""
        pass

    # --- Core LLM Binding Methods ---
    def tokenize(self, text: str) -> list:
        """
        Tokenize text using the active LLM binding.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: List of tokens.
        """
        if self.llm:
            return self.llm.tokenize(text)
        raise RuntimeError("LLM binding not initialized.")

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize tokens using the active LLM binding.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        if self.llm:
            return self.llm.detokenize(tokens)
        raise RuntimeError("LLM binding not initialized.")
    def count_tokens(self, text: str) -> int:
        """
        Counts how many tokens are there in the text using the active LLM binding.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Number of tokens.
        """
        if self.llm:
            return self.llm.count_tokens(text)
        raise RuntimeError("LLM binding not initialized.")

    def count_image_tokens(self, image: str) -> int:
        """
        Estimate the number of tokens for an image using ImageTokenizer based on self.model_name.

        Args:
            image (str): Image to count tokens from. Either base64 string, path to image file, or URL.

        Returns:
            int: Estimated number of tokens for the image. Returns -1 on error.
        """
        if self.llm:
            return self.llm.count_image_tokens(image)
        raise RuntimeError("LLM binding not initialized.")

    def get_model_details(self) -> dict:
        """
        Get model information from the active LLM binding.

        Returns:
            dict: Model information dictionary.
        """
        if self.llm:
            return self.llm.get_model_info()
        raise RuntimeError("LLM binding not initialized.")

    def switch_model(self, model_name: str) -> bool:
        """
        Load a new model in the active LLM binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if self.llm:
            return self.llm.load_model(model_name)
        raise RuntimeError("LLM binding not initialized.")

    def get_available_llm_bindings(self) -> List[str]: 
        """
        Get list of available LLM binding names.

        Returns:
            List[str]: List of binding names that can be used for LLMs.
        """
        return self.llm_binding_manager.get_available_bindings()

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
                     **kwargs
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
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        if self.llm:
            
            ctx_size = ctx_size if ctx_size is not None else self.llm.default_ctx_size if self.llm.default_ctx_size else None
            if ctx_size is None:
                ctx_size = self.llm.get_ctx_size()
                if ctx_size is None:
                    ctx_size = 1024*8 # 1028*8= 8192 tokens, a common default for many models
            nb_input_tokens = self.count_tokens(prompt)+ (sum([self.count_image_tokens(image) for image in images]) if images else 0)
            if kwargs.get("debug", False):
                ASCIIColors.magenta(f"Generating text using these parameters:")
                ASCIIColors.magenta(f"ctx_size : {ctx_size}")
                ASCIIColors.magenta(f"nb_input_tokens : {nb_input_tokens}")
            
            return self.llm.generate_text(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
                n_predict=n_predict if n_predict else self.llm.default_n_predict if self.llm.default_n_predict else ctx_size - nb_input_tokens,
                stream=stream if stream is not None else self.llm.default_stream,
                temperature=temperature if temperature is not None else self.llm.default_temperature,
                top_k=top_k if top_k is not None else self.llm.default_top_k,
                top_p=top_p if top_p is not None else self.llm.default_top_p,
                repeat_penalty=repeat_penalty if repeat_penalty is not None else self.llm.default_repeat_penalty,
                repeat_last_n=repeat_last_n if repeat_last_n is not None else self.llm.default_repeat_last_n,
                seed=seed if seed is not None else self.llm.default_seed,
                n_threads=n_threads if n_threads is not None else self.llm.default_n_threads,
                ctx_size = ctx_size if ctx_size is not None else self.llm.default_ctx_size,
                streaming_callback=streaming_callback if streaming_callback is not None else self.llm.default_streaming_callback,
                split= split,
                user_keyword=user_keyword,
                ai_keyword=ai_keyword
            )
        raise RuntimeError("LLM binding not initialized.")

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
        if self.llm:
            return self.llm.generate_from_messages(
                messages=messages,
                n_predict=n_predict if n_predict is not None else self.llm.default_n_predict,
                stream=stream if stream is not None else self.llm.default_stream,
                temperature=temperature if temperature is not None else self.llm.default_temperature,
                top_k=top_k if top_k is not None else self.llm.default_top_k,
                top_p=top_p if top_p is not None else self.llm.default_top_p,
                repeat_penalty=repeat_penalty if repeat_penalty is not None else self.llm.default_repeat_penalty,
                repeat_last_n=repeat_last_n if repeat_last_n is not None else self.llm.default_repeat_last_n,
                seed=seed if seed is not None else self.llm.default_seed,
                n_threads=n_threads if n_threads is not None else self.llm.default_n_threads,
                ctx_size = ctx_size if ctx_size is not None else self.llm.default_ctx_size,
                streaming_callback=streaming_callback if streaming_callback is not None else self.llm.default_streaming_callback,
            )
        raise RuntimeError("LLM binding not initialized.")

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
             streaming_callback: Optional[Callable[[str, MSG_TYPE, Dict], bool]] = None,
             **kwargs
             ) -> Union[str, dict]:
        """
        High-level method to perform a chat generation using a LollmsDiscussion object.

        This is the recommended method for conversational interactions. It uses the
        discussion object to correctly format the context for the model, including
        system prompts, roles, and multi-modal content.

        Args:
            discussion (LollmsDiscussion): The discussion object to use for context.
            branch_tip_id (Optional[str]): The ID of the message to use as the end of the conversation branch. If None, the active branch is used.
            n_predict (Optional[int]): Maximum number of tokens to generate. Uses instance default if None.
            stream (Optional[bool]): Whether to stream the output. Uses instance default if None.
            temperature (Optional[float]): Sampling temperature. Uses instance default if None.
            top_k (Optional[int]): Top-k sampling parameter. Uses instance default if None.
            top_p (Optional[float]): Top-p sampling parameter. Uses instance default if None.
            repeat_penalty (Optional[float]): Penalty for repeated tokens. Uses instance default if None.
            repeat_last_n (Optional[int]): Number of previous tokens to consider for repeat penalty. Uses instance default if None.
            seed (Optional[int]): Random seed for generation. Uses instance default if None.
            n_threads (Optional[int]): Number of threads to use. Uses instance default if None.
            ctx_size (Optional[int]): Context size override for this generation.
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.

        Returns:
            Union[str, dict]: Generated text or an error dictionary if failed.
        """
        if self.llm:
            return self.llm.chat(
                discussion=discussion,
                branch_tip_id=branch_tip_id,
                n_predict=n_predict if n_predict is not None else self.llm.default_n_predict,
                stream=stream if stream is not None else True if streaming_callback is not None else self.llm.default_stream,
                temperature=temperature if temperature is not None else self.llm.default_temperature,
                top_k=top_k if top_k is not None else self.llm.default_top_k,
                top_p=top_p if top_p is not None else self.llm.default_top_p,
                repeat_penalty=repeat_penalty if repeat_penalty is not None else self.llm.default_repeat_penalty,
                repeat_last_n=repeat_last_n if repeat_last_n is not None else self.llm.default_repeat_last_n,
                seed=seed if seed is not None else self.llm.default_seed,
                n_threads=n_threads if n_threads is not None else self.llm.default_n_threads,
                ctx_size = ctx_size if ctx_size is not None else self.llm.default_ctx_size,
                streaming_callback=streaming_callback if streaming_callback is not None else self.llm.default_streaming_callback
            )
        raise RuntimeError("LLM binding not initialized.")

    def embed(self, text, **kwargs):
        """
        Generate embeddings for the input text using the active LLM binding.

        Args:
            text (str or List[str]): Input text to embed.
            **kwargs: Additional arguments specific to the binding's embed method.

        Returns:
            list: List of embeddings.
        """
        if self.llm:
            return self.llm.embed(text, **kwargs)
        raise RuntimeError("LLM binding not initialized.")


    def listModels(self):
        """Lists models available to the current LLM binding."""
        if self.llm:
            return self.llm.listModels()
        raise RuntimeError("LLM binding not initialized.")

    # --- Convenience Methods for Lollms LLM Binding Features ---
    def listMountedPersonalities(self) -> Union[List[Dict], Dict]:
        """
        Lists mounted personalities *if* the active LLM binding is 'lollms'.

        Returns:
            Union[List[Dict], Dict]: List of personality dicts or error dict.
        """
        if self.llm and hasattr(self.llm, 'lollms_listMountedPersonalities'):
            return self.llm.lollms_listMountedPersonalities()
        else:
            ASCIIColors.warning("listMountedPersonalities is only available for the 'lollms' LLM binding.")
            return {"status": False, "error": "Functionality not available for the current binding"}

    # --- Code Generation / Extraction Helpers (These might be moved to TasksLibrary later) ---
    def generate_codes(
                        self,
                        prompt,
                        images=[],
                        template=None,
                        language="json",
                        code_tag_format="markdown", # or "html"
                        n_predict = None,
                        temperature = None,
                        top_k = None,
                        top_p=None,
                        repeat_penalty=None,
                        repeat_last_n=None,
                        callback=None,
                        debug=False
                        ):
        """
        Generates multiple code blocks based on a prompt.
        Uses the underlying LLM binding via `generate_text`.
        """
        response_full = ""
        system_prompt = f"""Act as a code generation assistant that generates code from user prompt."""

        if template:
            system_prompt += "Here is a template of the answer:\n"
            if code_tag_format=="markdown":
                system_prompt += f"""You must answer with the code placed inside the markdown code tag like this:
```{language}
{template}
```
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a markdown code tag. This is mandatory.
"""
            elif code_tag_format=="html":
                system_prompt +=f"""You must answer with the code placed inside the html code tag like this:
<code language="{language}">
{template}
</code>
{"Make sure you fill all fields and to use the exact same keys as the template." if language in ["json","yaml","xml"] else ""}
The code tag is mandatory.
Don't forget encapsulate the code inside a html code tag. This is mandatory.
"""
        system_prompt += f"""Do not split the code in multiple tags."""

        # Use generate_text which handles images internally
        response = self.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback # Assuming generate_text handles streaming callback
            )

        if isinstance(response, dict) and not response.get("status", True): # Check for error dict
             ASCIIColors.error(f"Code generation failed: {response.get('error')}")
             return []

        response_full += response
        codes = self.extract_code_blocks(response, format=code_tag_format)
        return codes
    
    def _synthesize_knowledge(
        self,
        previous_scratchpad: str,
        tool_name: str,
        tool_params: dict,
        tool_result: dict
    ) -> str:
        """
        A dedicated LLM call to interpret a tool's output and update the knowledge scratchpad.
        """
        synthesis_prompt = (
            "You are a data analyst assistant. Your sole job is to interpret the output of a tool and integrate it into the existing research summary (knowledge scratchpad).\n\n"
            "--- PREVIOUS KNOWLEDGE SCRATCHPAD ---\n"
            f"{previous_scratchpad}\n\n"
            "--- ACTION JUST TAKEN ---\n"
            f"Tool Called: `{tool_name}`\n"
            f"Parameters: {json.dumps(tool_params)}\n\n"
            "--- RAW TOOL OUTPUT ---\n"
            f"```json\n{json.dumps(tool_result, indent=2)}\n```\n\n"
            "--- YOUR TASK ---\n"
            "Read the 'RAW TOOL OUTPUT' and explain what it means in plain language. Then, integrate this new information with the 'PREVIOUS KNOWLEDGE SCRATCHPAD' to create a new, complete, and self-contained summary.\n"
            "Your output should be ONLY the text of the new scratchpad, with no extra commentary or formatting.\n\n"
            "--- NEW KNOWLEDGE SCRATCHPAD ---\n"
        )
        new_scratchpad_text = self.generate_text(prompt=synthesis_prompt, n_predict=1024, temperature=0.0)
        return self.remove_thinking_blocks(new_scratchpad_text).strip()

    def _build_final_decision_prompt(
        self,
        formatted_tools_list: str,
        formatted_conversation_history: str,
        current_plan: str,
        knowledge_scratchpad: str,
        agent_work_history_str: str,
        ctx_size: Optional[int],
    ) -> str:
        """
        Builds the decision prompt with explicit state-checking instructions to prevent loops.
        """
        final_agent_history = agent_work_history_str

        if ctx_size:
            get_token_count = len
            static_parts_text = (
                "You are a task-oriented AI assistant. Your goal is to execute a plan step-by-step without repeating work.\n\n"
                "--- AVAILABLE TOOLS ---\n"
                f"{formatted_tools_list}\n\n"
                "--- CONVERSATION HISTORY ---\n"
                f"{formatted_conversation_history}\n\n"
                "--- CUMULATIVE KNOWLEDGE (What you know so far) ---\n"
                f"{knowledge_scratchpad}\n\n"
                "--- THE OVERALL PLAN ---\n"
                f"{current_plan}\n\n"
                "--- ACTIONS TAKEN THIS TURN ---\n"
                "\n\n" # Empty history for size calculation
                "--- YOUR TASK: STATE-DRIVEN EXECUTION ---\n"
                "1.  **Identify the next step:** Look at 'THE OVERALL PLAN' and identify the very next incomplete step.\n"
                "2.  **Check your knowledge:** Look at the 'CUMULATIVE KNOWLEDGE'. Have you already performed this step and recorded the result? For example, if the step is 'search for papers', check if the search results are already in the knowledge base.\n"
                "3.  **Decide your action:**\n"
                "    -   **If the step is NOT DONE:** Your action is `call_tool` to execute it.\n"
                "    -   **If the step IS ALREADY DONE:** Your job is to update the plan by removing the completed step. Then, re-evaluate from step 1 with the *new, shorter plan*.\n"
                "    -   **If ALL steps are done:** Your action is `final_answer`.\n"
                "    -   **If you are blocked:** Your action is `clarify`.\n\n"
                "--- OUTPUT FORMAT ---\n"
                "Respond with a single JSON object inside a ```json markdown tag.\n"
                "```json\n{\n"
                '    "thought": "My explicit reasoning. First, I will state the next step from the plan. Second, I will check the cumulative knowledge to see if this step is already complete. Third, I will state my conclusion and chosen action based on that comparison.",\n'
                '    "updated_plan": "The new, remaining plan. It is CRITICAL that you remove any step that you have confirmed is complete in your thought process.",\n'
                '    "action": "The chosen action: \'call_tool\', \'clarify\', or \'final_answer\'.",\n'
                '    "action_details": {\n'
                '        "tool_name": "(Required if action is \'call_tool\') The tool for the CURRENT incomplete step.",\n'
                '        "tool_params": {},\n'
                '        "clarification_request": "(Required if action is \'clarify\') Your specific question to the user."\n'
                "    }\n}\n```"
            )
            fixed_parts_size = get_token_count(static_parts_text)
            available_space_for_history = ctx_size - fixed_parts_size - 100
            if get_token_count(agent_work_history_str) > available_space_for_history:
                if available_space_for_history > 0:
                    truncation_point = len(agent_work_history_str) - available_space_for_history
                    final_agent_history = ("[...history truncated due to context size...]\n" + agent_work_history_str[truncation_point:])
                    ASCIIColors.warning("Agent history was truncated to fit the context window.")
                else:
                    final_agent_history = "[...history truncated due to context size...]"
        
        return (
            "You are a task-oriented AI assistant. Your goal is to execute a plan step-by-step without repeating work.\n\n"
            "--- AVAILABLE TOOLS ---\n"
            f"{formatted_tools_list}\n\n"
            "--- CONVERSATION HISTORY ---\n"
            f"{formatted_conversation_history}\n\n"
            "--- CUMULATIVE KNOWLEDGE (What you know so far) ---\n"
            f"{knowledge_scratchpad}\n\n"
            "--- THE OVERALL PLAN ---\n"
            f"{current_plan}\n\n"
            "--- ACTIONS TAKEN THIS TURN ---\n"
            f"{final_agent_history}\n\n"
            "--- YOUR TASK: STATE-DRIVEN EXECUTION ---\n"
            "1.  **Identify the next step:** Look at 'THE OVERALL PLAN' and identify the very next incomplete step.\n"
            "2.  **Check your knowledge:** Look at the 'CUMULATIVE KNOWLEDGE'. Have you already performed this step and recorded the result? For example, if the step is 'search for papers', check if the search results are already in the knowledge base.\n"
            "3.  **Decide your action:**\n"
            "    -   **If the step is NOT DONE:** Your action is `call_tool` to execute it.\n"
            "    -   **If the step IS ALREADY DONE:** Your job is to update the plan by removing the completed step. Then, re-evaluate from step 1 with the *new, shorter plan*.\n"
            "    -   **If ALL steps are done:** Your action is `final_answer`.\n"
            "    -   **If you are blocked:** Your action is `clarify`.\n\n"
            "--- OUTPUT FORMAT ---\n"
            "Respond with a single JSON object inside a ```json markdown tag.\n"
            "```json\n"
            "{\n"
            '    "thought": "My explicit reasoning. First, I will state the next step from the plan. Second, I will check the cumulative knowledge to see if this step is already complete. Third, I will state my conclusion and chosen action based on that comparison.",\n'
            '    "updated_plan": "The new, remaining plan. It is CRITICAL that you remove any step that you have confirmed is complete in your thought process.",\n'
            '    "action": "The chosen action: \'call_tool\', \'clarify\', or \'final_answer\'.",\n'
            '    "action_details": {\n'
            '        "tool_name": "(Required if action is \'call_tool\') The tool for the CURRENT incomplete step.",\n'
            '        "tool_params": {},\n'
            '        "clarification_request": "(Required if action is \'clarify\') Your specific question to the user."\n'
            "    }\n"
            "}\n"
            "```"
        )


    def generate_with_mcp(
        self,
        prompt: str,
        system_prompt:str = None,
        objective_extraction_system_prompt="Build a plan",
        images: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tool_calls: int = 5,
        max_llm_iterations: int = 10,
        ctx_size: Optional[int] = None,
        max_json_retries: int = 1,
        tool_call_decision_temperature: float = 0.0,
        final_answer_temperature: float = None,
        streaming_callback: Optional[Callable[[str, int, Optional[Dict], Optional[List]], bool]] = None,
        **llm_generation_kwargs
    ) -> Dict[str, Any]:
        if not self.llm or not self.mcp:
            return {"final_answer": "", "tool_calls": [], "error": "LLM or MCP binding not initialized."}

        turn_history: List[Dict[str, Any]] = []
        conversation_context = prompt 

        if tools is None:
            try:
                tools = self.mcp.discover_tools(force_refresh=True)
                if not tools: ASCIIColors.warning("No MCP tools discovered.")
            except Exception as e_disc:
                return {"final_answer": "", "tool_calls": [], "error": f"Failed to discover MCP tools: {e_disc}"}

        if not tools:
            final_answer_text = self.generate_text(prompt=prompt, system_prompt=system_prompt, stream=streaming_callback is not None, streaming_callback=streaming_callback)
            return {"final_answer": self.remove_thinking_blocks(final_answer_text), "tool_calls": [], "error": None}

        knowledge_scratchpad = "No information gathered yet."
        agent_work_history = [] 
        formatted_tools_list = "\n".join([f"- Tool: {t.get('name')}\n  Description: {t.get('description')}\n  Schema: {json.dumps(t.get('input_schema'))}" for t in tools])
        
        if streaming_callback:
            streaming_callback("Building/Revising plan...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "plan_extraction"}, turn_history = turn_history)
        
        obj_prompt = (
            "You are an Intelligent Workflow Planner. Your mission is to create the most efficient plan possible by analyzing the user's request within the context of the full conversation.\n\n"
            "Your Guiding Principle: **Always choose the path of least resistance.**\n\n"
            "**Your Logical Process:**\n"
            "1.  **Analyze the Entire Conversation:** Understand the user's ultimate goal based on all interaction so far.\n"
            "2.  **Check for a Single-Step Solution:** Scrutinize the available tools. Can a single tool call directly achieve the user's current goal? \n"
            "3.  **Formulate a Plan:** Based on your analysis, create a concise, numbered list of steps to achieve the goal. If the goal is simple, this may be only one step. If it is complex or multi-turn, it may be several steps.\n\n"
            "**CRITICAL RULES:**\n"
            "*   **MANDATORY: Be helpful, curious and creative.\n"
            "*   **Focus on the Goal:** Your plan should directly address the user's request as it stands now in the conversation.\n\n"
            "---\n"
            "**Available Tools:**\n"
            f"{formatted_tools_list}\n\n"
            "**Full Conversation History:**\n"
            f'"{conversation_context}"'
        )
        initial_plan_gen = self.generate_text(prompt=obj_prompt, system_prompt=objective_extraction_system_prompt, temperature=0.0, stream=False)
        if type(initial_plan_gen)!=str:
            if "error" in initial_plan_gen:
                ASCIIColors.error(initial_plan_gen["error"])
                raise Exception(initial_plan_gen["error"])
            else:
                raise Exception("generate text failed. Make sure you are connected to the binding server if you are using remote one")
        current_plan = self.remove_thinking_blocks(initial_plan_gen).strip()

        if streaming_callback:
            streaming_callback("Building initial plan...", MSG_TYPE.MSG_TYPE_STEP_END, {"id": "plan_extraction"}, turn_history = turn_history)
            streaming_callback(f"Current plan:\n{current_plan}", MSG_TYPE.MSG_TYPE_STEP, {"id": "plan"}, turn_history = turn_history)
        turn_history.append({"type": "initial_plan", "content": current_plan})
        
        tool_calls_made_this_turn = []
        llm_iterations = 0

        while llm_iterations < max_llm_iterations:
            llm_iterations += 1
            if streaming_callback: streaming_callback(f"LLM reasoning step (iteration {llm_iterations})...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": f"planning_step_{llm_iterations}"}, turn_history = turn_history)
            
            formatted_agent_history = "No actions taken yet in this turn."
            if agent_work_history:
                history_parts = [ f"### Step {i+1}:\n**Thought:**\n{entry['thought']}\n**Action:** Called tool `{entry['tool_name']}` with parameters `{json.dumps(entry['tool_params'])}`\n**Observation (Tool Output):**\n```json\n{json.dumps(entry['tool_result'], indent=2)}\n```" for i, entry in enumerate(agent_work_history)]
                formatted_agent_history = "\n\n".join(history_parts)

            llm_decision = None
            current_decision_prompt = self._build_final_decision_prompt(
                formatted_tools_list=formatted_tools_list, formatted_conversation_history=conversation_context,
                current_plan=current_plan, knowledge_scratchpad=knowledge_scratchpad,
                agent_work_history_str=formatted_agent_history, ctx_size=ctx_size
            )

            for i in range(max_json_retries + 1):
                raw_llm_decision_json = self.generate_text(prompt=current_decision_prompt, n_predict=2048, temperature=tool_call_decision_temperature)
                try:
                    llm_decision = robust_json_parser(raw_llm_decision_json)
                    if "action" not in llm_decision or "action_details" not in llm_decision or "updated_plan" not in llm_decision:
                        raise KeyError("The JSON is missing required keys: 'action', 'action_details', or 'updated_plan'.")
                    break 
                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    error_message = f"JSON parsing failed (Attempt {i+1}/{max_json_retries+1}). Error: {e}"
                    ASCIIColors.warning(error_message)
                    if streaming_callback: streaming_callback(error_message, MSG_TYPE.MSG_TYPE_WARNING, None, turn_history = turn_history)
                    turn_history.append({"type": "error", "content": f"Invalid JSON response: {raw_llm_decision_json}"})
                    if i >= max_json_retries:
                        ASCIIColors.error("Max JSON retries reached. Aborting agent loop.")
                        llm_decision = None
                        break
                    current_decision_prompt = (
                        "You previously failed to generate a valid JSON object. Review the error and your last output, then try again, adhering strictly to the required schema.\n\n"
                        "--- ERROR ---\n"
                        f"{str(e)}\n\n"
                        "--- YOUR PREVIOUS (INVALID) OUTPUT ---\n"
                        f"{raw_llm_decision_json}\n\n"
                        "--- REQUIRED SCHEMA REMINDER ---\n"
                        "Your response MUST be a single JSON object inside a ```json markdown tag. It must contain 'action', 'action_details', and 'updated_plan' keys.\n\n"
                        "Now, please re-generate the JSON response correctly."
                    )
            if not llm_decision: break

            turn_history.append({"type": "llm_decision", "content": llm_decision})
            current_plan = llm_decision.get("updated_plan", current_plan)
            action = llm_decision.get("action")
            action_details = llm_decision.get("action_details", {})
            if streaming_callback: streaming_callback(f"LLM thought: {llm_decision.get('thought', 'N/A')}", MSG_TYPE.MSG_TYPE_INFO, {"id": "llm_thought"}, turn_history = turn_history)

            if action == "call_tool":
                if len(tool_calls_made_this_turn) >= max_tool_calls:
                    ASCIIColors.warning("Max tool calls reached. Forcing final answer.")
                    break
                tool_name = action_details.get("tool_name")
                tool_params = action_details.get("tool_params", {})
                if not tool_name or not isinstance(tool_params, dict):
                    ASCIIColors.error(f"Invalid tool call from LLM: name={tool_name}, params={tool_params}")
                    break
                
                if streaming_callback: streaming_callback(f"Executing tool: {tool_name}...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": f"tool_exec_{llm_iterations}"}, turn_history = turn_history)
                tool_result = self.mcp.execute_tool(tool_name, tool_params, lollms_client_instance=self)
                if streaming_callback:
                    streaming_callback(f"Tool {tool_name} finished.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"tool_exec_{llm_iterations}"}, turn_history = turn_history)
                    streaming_callback(json.dumps(tool_result, indent=2), MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, tool_result, turn_history = turn_history)

                if streaming_callback: streaming_callback("Synthesizing new knowledge...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": f"synthesis_step_{llm_iterations}"}, turn_history = turn_history)
                new_scratchpad = self._synthesize_knowledge(previous_scratchpad=knowledge_scratchpad, tool_name=tool_name, tool_params=tool_params, tool_result=tool_result)
                knowledge_scratchpad = new_scratchpad
                if streaming_callback:
                    streaming_callback(f"Knowledge scratchpad updated.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"synthesis_step_{llm_iterations}"}, turn_history = turn_history)
                    streaming_callback(f"New Scratchpad:\n{knowledge_scratchpad}", MSG_TYPE.MSG_TYPE_INFO, {"id": "scratchpad_update"}, turn_history = turn_history)
                
                work_entry = { "thought": llm_decision.get("thought", "N/A"), "tool_name": tool_name, "tool_params": tool_params, "tool_result": tool_result, "synthesized_knowledge": knowledge_scratchpad }
                agent_work_history.append(work_entry)
                tool_calls_made_this_turn.append({"name": tool_name, "params": tool_params, "result": tool_result})
            
            elif action == "clarify":
                clarification_request = action_details.get("clarification_request", "I need more information to proceed. Could you please clarify?")
                return { "final_answer": clarification_request, "tool_calls": tool_calls_made_this_turn, "error": None, "clarification": True }
            
            elif action == "final_answer":
                ASCIIColors.info("LLM decided to formulate a final answer.")
                break
            
            else:
                ASCIIColors.warning(f"LLM returned unknown or missing action: '{action}'. Forcing final answer.")
                break
            
            if streaming_callback: 
                streaming_callback(f"LLM reasoning step (iteration {llm_iterations}) complete.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"planning_step_{llm_iterations}"}, turn_history = turn_history)
       
        if streaming_callback: 
            streaming_callback(f"LLM reasoning step (iteration {llm_iterations}) complete.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"planning_step_{llm_iterations}"}, turn_history = turn_history)
        if streaming_callback: 
            streaming_callback("Synthesizing final answer...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "final_answer_synthesis"}, turn_history = turn_history)
        
        final_answer_prompt = (
            "You are an AI assistant tasked with providing a final, comprehensive answer to the user based on the research performed.\n\n"
            "--- FULL CONVERSATION CONTEXT ---\n"
            f"{conversation_context}\n\n"
            "--- SUMMARY OF FINDINGS (Your Knowledge Scratchpad) ---\n"
            f"{knowledge_scratchpad}\n\n"
            "--- INSTRUCTIONS ---\n"
            "- Synthesize a clear and complete answer for the user based ONLY on the information in the 'Summary of Findings'.\n"
            "- Address the user directly and answer their latest query, considering the full conversation.\n"
            "- Do not make up information. If the findings are insufficient to fully answer the request, state what you found and what remains unanswered.\n"
            "- Format your response clearly using markdown where appropriate.\n"
        )
        final_answer_text = self.generate_text(prompt=final_answer_prompt, system_prompt=system_prompt, images=images, stream=streaming_callback is not None, streaming_callback=streaming_callback, temperature=final_answer_temperature if final_answer_temperature is not None else self.llm.default_temperature, **(llm_generation_kwargs or {}))
        
        if streaming_callback: 
            streaming_callback("Final answer generation complete.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": "final_answer_synthesis"}, turn_history = turn_history)

        final_answer = self.remove_thinking_blocks(final_answer_text)
        turn_history.append({"type":"final_answer_generated", "content": final_answer})
        
        return {"final_answer": final_answer, "tool_calls": tool_calls_made_this_turn, "error": None}


    def generate_text_with_rag(
        self,
        prompt: str,
        rag_query_function: Callable[[str, Optional[str], int, float], List[Dict[str, Any]]],
        system_prompt: str = "",
        objective_extraction_system_prompt="Extract objectives",
        rag_query_text: Optional[str] = None,
        rag_vectorizer_name: Optional[str] = None,
        rag_top_k: int = 5,
        rag_min_similarity_percent: float = 70.0,
        max_rag_hops: int = 3,
        images: Optional[List[str]] = None,
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
        extract_objectives: bool = True,
        streaming_callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict], Optional[List]], bool]] = None,
        max_rag_context_characters: int = 32000,
        **llm_generation_kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced RAG with dynamic objective refinement and a knowledge scratchpad.
        """
        if not self.llm:
            return {"final_answer": "", "rag_hops_history": [], "all_retrieved_sources": [], "error": "LLM binding not initialized."}

        effective_ctx_size = ctx_size or getattr(self, "default_ctx_size", 20000)

        turn_rag_history_for_callback: List[Dict[str, Any]] = []
        rag_hops_details_list: List[Dict[str, Any]] = []
        all_unique_retrieved_chunks_map: Dict[str, Dict[str, Any]] = {}

        original_user_prompt = prompt
        
        knowledge_scratchpad = "No information gathered yet."
        current_objectives = ""

        if extract_objectives:
            if streaming_callback:
                streaming_callback("Extracting initial objectives...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "objectives_extraction"}, turn_rag_history_for_callback)
            
            obj_prompt = (
                "You are an expert analyst. Your task is to extract and structure the key research objectives from the user's request below. "
                "These objectives will guide a research process. Frame them as questions or tasks. "
                "Output a bulleted list of objectives only without a comment.\n\n"
                f"User request:\n\"{original_user_prompt}\""
            )
            initial_objectives_gen = self.generate_text(prompt=obj_prompt, system_prompt=objective_extraction_system_prompt, temperature=0.0, stream=False)
            current_objectives = self.remove_thinking_blocks(initial_objectives_gen).strip()
            
            if streaming_callback:
                streaming_callback(f"Initial Objectives:\n{current_objectives}", MSG_TYPE.MSG_TYPE_STEP_END, {"id": "objectives_extraction"}, turn_rag_history_for_callback)
        else:
            current_objectives = f"Answer the user's request: '{original_user_prompt}'"

        if streaming_callback:
            streaming_callback("Generating initial search query...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "initial_query_generation"}, turn_rag_history_for_callback)
        if not rag_query_text:
            initial_query_gen_prompt = f"""
You are a research assistant. Your task is to formulate the first search query for a vector database based on an initial user request and research objectives. The query should be concise and target the most crucial information needed to start.

--- User's Request ---
{original_user_prompt}

--- Initial Research Objectives ---
{current_objectives}

--- INSTRUCTIONS ---
Generate a single, effective search query.

--- OUTPUT FORMAT ---
Provide your response as a single JSON object with one key, "query".
```json
{{
    "query": "Your generated search query here."
}}
```
"""
            try:
                raw_initial_query_response = self.generate_code(initial_query_gen_prompt, system_prompt="You are a query generation expert.", temperature=0.0)
                initial_plan = robust_json_parser(raw_initial_query_response)
                current_query_for_rag = initial_plan.get("query")
                if not current_query_for_rag:
                    raise ValueError("LLM returned an empty initial query.")
                if streaming_callback:
                    streaming_callback(f"Initial query generated:\n'{current_query_for_rag}'", MSG_TYPE.MSG_TYPE_STEP_END, {"id": "initial_query_generation"}, turn_rag_history_for_callback)
            except Exception as e:
                trace_exception(e)
                current_query_for_rag = original_user_prompt
                if streaming_callback:
                    streaming_callback(f"Failed to generate initial query, falling back to user prompt. Error: {e}", MSG_TYPE.MSG_TYPE_WARNING, {"id": "initial_query_failure"}, turn_rag_history_for_callback)
        else:
            current_query_for_rag=rag_query_text
            
        previous_queries = []

        for hop_count in range(max_rag_hops):
            if streaming_callback:
                streaming_callback(f"Starting RAG Hop {hop_count + 1}", MSG_TYPE.MSG_TYPE_STEP_START, {"id": f"rag_hop_{hop_count + 1}"}, turn_rag_history_for_callback)
            
            if streaming_callback:
                streaming_callback(f"Executing Query:\n{current_query_for_rag}", MSG_TYPE.MSG_TYPE_STEP, {"id": f"query_exec_{hop_count + 1}"}, turn_rag_history_for_callback)
            
            try:
                retrieved_chunks = rag_query_function(current_query_for_rag, rag_vectorizer_name, rag_top_k, rag_min_similarity_percent)
            except Exception as e:
                trace_exception(e)
                return {"final_answer": "", "rag_hops_history": rag_hops_details_list, "all_retrieved_sources": list(all_unique_retrieved_chunks_map.values()), "error": str(e)}

            hop_details = {"query": current_query_for_rag, "retrieved_chunks_details": [], "status": ""}
            previous_queries.append(current_query_for_rag)
            
            newly_retrieved_text = ""
            new_chunks_count = 0
            if retrieved_chunks:
                for chunk in retrieved_chunks:
                    doc = chunk.get("file_path", "Unknown")
                    content = str(chunk.get("chunk_text", ""))
                    sim = float(chunk.get("similarity_percent", 0.0))
                    detail = {"document": doc, "similarity": sim, "content": content, "retrieved_in_hop": hop_count + 1, "query_used": current_query_for_rag}
                    hop_details["retrieved_chunks_details"].append(detail)
                    
                    key = f"{doc}::{content[:100]}"
                    if key not in all_unique_retrieved_chunks_map:
                        all_unique_retrieved_chunks_map[key] = detail
                        newly_retrieved_text += f"--- Document: {doc} (Similarity: {sim:.1f}%)\n{content}\n---\n"
                        new_chunks_count += 1
            
            hop_details["status"] = f"Completed, found {len(retrieved_chunks)} chunks ({new_chunks_count} new)."
            rag_hops_details_list.append(hop_details)

            if streaming_callback:
                streaming_callback(f"Retrieved {len(retrieved_chunks)} chunks ({new_chunks_count} new).", MSG_TYPE.MSG_TYPE_STEP, {"id": f"retrieval_info_{hop_count + 1}"}, turn_rag_history_for_callback)

            if new_chunks_count == 0 and hop_count > 0:
                 if streaming_callback:
                    streaming_callback("No new unique information found, stopping RAG hops.", MSG_TYPE.MSG_TYPE_INFO, {"id": "rag_stop_no_new_info"}, turn_rag_history_for_callback)
                 break

            if streaming_callback:
                streaming_callback("Analyzing findings and refining plan...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": f"planning_step_{hop_count + 1}"}, turn_rag_history_for_callback)
                
            planning_system_prompt = (
                "You are a strategic research agent via multiple hops. Your task is to analyze new information, update your "
                "understanding, refine your research objectives, and decide on the next best action."
            )

            planning_prompt = f"""
--- Original User Request ---
{original_user_prompt}

--- Objectives you have formulated ---
{current_objectives}

--- Existing Knowledge Scratchpad (Summary of previous findings) ---
{knowledge_scratchpad}

--- Newly Retrieved Documents for this Hop ---
{newly_retrieved_text if newly_retrieved_text else "No new documents were found with the last query."}

--- Previous Queries (for reference, do not repeat) ---
- {"- ".join(previous_queries)}

--- INSTRUCTIONS ---
1.  **Analyze & Update Knowledge:** Read the 'Newly Retrieved Documents'. Summarize the most important new facts and insights into a few bullet points for the 'new_notes_for_scratchpad'.
2.  **Refine Objectives:** Review the 'Current Research Objectives'. Do the new documents answer any objectives? Do they reveal that some objectives need to be changed or made more specific? Rewrite the complete, updated list of objectives.
3.  **Decide & Plan Next Query:** Based on your updated objectives and knowledge, decide if you have enough information to form a final answer.
    - If YES, set `decision` to `false`.
    - If NO, set `decision` to `true` and formulate a new, focused `query` to address the most critical remaining gap in your knowledge. The query must be different from previous ones.

--- OUTPUT FORMAT ---
Provide your response as a single JSON object inside a JSON markdown tag. Use this exact schema:
```json
{{
    "updated_objectives": "(string) A bulleted list of the new, refined objectives based on the latest information.",
    "new_notes_for_scratchpad": "(string) A concise summary in bullet points of key findings from the new documents.",
    "decision": "boolean (true if you need to query again, false if you are done).",
    "query": "(string, optional) The next query for the vector database if decision is true."
}}
```
"""
            raw_planning_response = self.generate_code(planning_prompt, system_prompt=planning_system_prompt, temperature=0.0)
            
            try:
                plan = robust_json_parser(raw_planning_response)
                
                raw_notes = plan.get("new_notes_for_scratchpad")
                if isinstance(raw_notes, list):
                    notes_from_hop = "\n".join(str(item) for item in raw_notes if item).strip()
                elif isinstance(raw_notes, str):
                    notes_from_hop = raw_notes.strip()
                else:
                    notes_from_hop = ""
                
                if notes_from_hop:
                    if knowledge_scratchpad == "No information gathered yet.":
                        knowledge_scratchpad = f"Findings from Hop {hop_count + 1}:\n{notes_from_hop}"
                    else:
                        knowledge_scratchpad += f"\n\nFindings from Hop {hop_count + 1}:\n{notes_from_hop}"
                
                raw_objectives = plan.get("updated_objectives")
                if isinstance(raw_objectives, list):
                    current_objectives = "\n".join(str(item) for item in raw_objectives if item).strip()
                elif isinstance(raw_objectives, str) and raw_objectives.strip():
                     current_objectives = raw_objectives.strip()

                if streaming_callback:
                    streaming_callback(f"Refined Objectives:\n{current_objectives}\n\nNew Learnings:\n{notes_from_hop}", MSG_TYPE.MSG_TYPE_STEP, {"id": f"planning_output_{hop_count + 1}"}, turn_rag_history_for_callback)

                if not plan.get("decision", False):
                    if streaming_callback:
                        streaming_callback("LLM decided it has enough information.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"rag_hop_{hop_count + 1}"}, turn_rag_history_for_callback)
                    break
                else:
                    next_query = plan.get("query")
                    if not next_query:
                        if streaming_callback:
                            streaming_callback("LLM decided to continue but provided no query. Stopping.", MSG_TYPE.MSG_TYPE_WARNING, {"id": "rag_stop_no_query"}, turn_rag_history_for_callback)
                        break
                    current_query_for_rag = next_query
            
            except Exception as ex:
                trace_exception(ex)
                if streaming_callback:
                    streaming_callback(f"Error processing planning step: {ex}. Stopping RAG.", MSG_TYPE.MSG_TYPE_EXCEPTION, {"id": f"planning_error_{hop_count + 1}"}, turn_rag_history_for_callback)
                break

            if streaming_callback:
                streaming_callback(f"RAG Hop {hop_count + 1} done", MSG_TYPE.MSG_TYPE_STEP_END, {"id": f"rag_hop_{hop_count + 1}"}, turn_rag_history_for_callback)

        sorted_chunks = sorted(all_unique_retrieved_chunks_map.values(), key=lambda c: c["similarity"], reverse=True)
        context_lines = []
        total_chars = 0
        for c in sorted_chunks:
            snippet = (f"Source: {c['document']} (Sim: {c['similarity']:.1f}%)\n{c['content']}\n---\n")
            if total_chars + len(snippet) > max_rag_context_characters: break
            context_lines.append(snippet)
            total_chars += len(snippet)
        accumulated_context = "".join(context_lines)

        if self.count_tokens(accumulated_context) > effective_ctx_size:
            pass

        if streaming_callback:
            streaming_callback("Compiling final answer from all findings...", MSG_TYPE.MSG_TYPE_STEP_START, {"id": "final_answer_generation"}, turn_rag_history_for_callback)

        final_prompt_parts = [
            f"**User's Original Request:**\n{original_user_prompt}\n",
            f"**Final Research Objectives:**\n{current_objectives}\n",
            f"**Knowledge Scratchpad (Summary of Findings):**\n{knowledge_scratchpad}\n",
        ]
        if accumulated_context:
            final_prompt_parts.append(
                "**Supporting Raw Context from Retrieved Documents:**\n---\n"
                f"{accumulated_context}\n---\n"
            )
        else:
            final_prompt_parts.append("**Supporting Raw Context:**\n(No relevant documents were retrieved.)\n")
        
        final_prompt_parts.append(
            "**Final Instruction:**\nSynthesize a comprehensive answer to the user's original request. "
            "Use the 'Knowledge Scratchpad' as your primary source of information and the 'Supporting Raw Context' for specific details and quotes. "
            "Adhere strictly to the information provided. If the information is insufficient to fully answer, state what is missing based on your 'Final Research Objectives'."
        )
        final_prompt_parts.append(self.ai_full_header)

        final_answer = self.generate_text(
            prompt="\n".join(final_prompt_parts),
            images=images, system_prompt=system_prompt, n_predict=n_predict, stream=stream,
            temperature=temperature, top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n, seed=seed, n_threads=n_threads, ctx_size=ctx_size,
            streaming_callback=streaming_callback if stream else None,
            **llm_generation_kwargs
        )
        answer_text = self.remove_thinking_blocks(final_answer) if isinstance(final_answer, str) else final_answer
        
        if streaming_callback:
            streaming_callback("Final answer generated.", MSG_TYPE.MSG_TYPE_STEP_END, {"id": "final_answer_generation"}, turn_rag_history_for_callback)

        return {
            "final_answer": answer_text,
            "rag_hops_history": rag_hops_details_list,
            "all_retrieved_sources": list(all_unique_retrieved_chunks_map.values()),
            "error": None
        }
    
    # --- Start of modified/added methods ---
    def _synthesize_knowledge(
        self,
        previous_scratchpad: str,
        tool_name: str,
        tool_params: dict,
        tool_result: dict
    ) -> str:
        """
        A dedicated LLM call to interpret a tool's output and update the knowledge scratchpad.
        """
        # Sanitize tool_result for LLM to avoid sending large binary/base64 data
        sanitized_result = tool_result.copy()
        if 'image_path' in sanitized_result:
            sanitized_result['summary'] = f"An image was successfully generated and saved to '{sanitized_result['image_path']}'."
            # Remove keys that might contain large data if they exist
            sanitized_result.pop('image_base64', None)
        elif 'file_path' in sanitized_result and 'content' in sanitized_result:
             sanitized_result['summary'] = f"Content was successfully written to '{sanitized_result['file_path']}'."
             sanitized_result.pop('content', None)


        synthesis_prompt = (
            "You are a data analyst assistant. Your sole job is to interpret the output of a tool and integrate it into the existing research summary (knowledge scratchpad).\n\n"
            "--- PREVIOUS KNOWLEDGE SCRATCHPAD ---\n"
            f"{previous_scratchpad}\n\n"
            "--- ACTION JUST TAKEN ---\n"
            f"Tool Called: `{tool_name}`\n"
            f"Parameters: {json.dumps(tool_params)}\n\n"
            "--- RAW TOOL OUTPUT ---\n"
            f"```json\n{json.dumps(sanitized_result, indent=2)}\n```\n\n"
            "--- YOUR TASK ---\n"
            "Read the 'RAW TOOL OUTPUT' and explain what it means in plain language. Then, integrate this new information with the 'PREVIOUS KNOWLEDGE SCRATCHPAD' to create a new, complete, and self-contained summary.\n"
            "Your output should be ONLY the text of the new scratchpad, with no extra commentary or formatting.\n\n"
            "--- NEW KNOWLEDGE SCRATCHPAD ---\n"
        )
        new_scratchpad_text = self.generate_text(prompt=synthesis_prompt, n_predict=1024, temperature=0.0)
        return self.remove_thinking_blocks(new_scratchpad_text).strip()
    def _synthesize_knowledge(
        self,
        previous_scratchpad: str,
        tool_name: str,
        tool_params: dict,
        tool_result: dict
    ) -> str:
        """
        A dedicated LLM call to interpret a tool's output and update the knowledge scratchpad.
        """
        # Sanitize tool_result for LLM to avoid sending large binary/base64 data
        sanitized_result = tool_result.copy()
        if 'image_path' in sanitized_result:
            sanitized_result['summary'] = f"An image was successfully generated and saved to '{sanitized_result['image_path']}'."
            # Remove keys that might contain large data if they exist
            sanitized_result.pop('image_base64', None)
        elif 'file_path' in sanitized_result and 'content' in sanitized_result:
             sanitized_result['summary'] = f"Content was successfully written to '{sanitized_result['file_path']}'."
             sanitized_result.pop('content', None)


        synthesis_prompt = (
            "You are a data analyst assistant. Your sole job is to interpret the output of a tool and integrate it into the existing research summary (knowledge scratchpad).\n\n"
            "--- PREVIOUS KNOWLEDGE SCRATCHPAD ---\n"
            f"{previous_scratchpad}\n\n"
            "--- ACTION JUST TAKEN ---\n"
            f"Tool Called: `{tool_name}`\n"
            f"Parameters: {json.dumps(tool_params)}\n\n"
            "--- RAW TOOL OUTPUT ---\n"
            f"```json\n{json.dumps(sanitized_result, indent=2)}\n```\n\n"
            "--- YOUR TASK ---\n"
            "Read the 'RAW TOOL OUTPUT' and explain what it means in plain language. Then, integrate this new information with the 'PREVIOUS KNOWLEDGE SCRATCHPAD' to create a new, complete, and self-contained summary.\n"
            "Your output should be ONLY the text of the new scratchpad, with no extra commentary or formatting.\n\n"
            "--- NEW KNOWLEDGE SCRATCHPAD ---\n"
        )
        new_scratchpad_text = self.generate_text(prompt=synthesis_prompt, n_predict=1024, temperature=0.0)
        return self.remove_thinking_blocks(new_scratchpad_text).strip()
    

    def _get_friendly_action_description(self, tool_name: str, requires_code: bool, requires_image: bool) -> str:
            """Convert technical tool names to user-friendly descriptions for logging."""
            # Handle specific, high-priority built-in tools first
            if tool_name == "local_tools::final_answer":
                return "📋 Ready to provide your final answer"
            elif tool_name == "local_tools::request_clarification":
                return "❓ Asking for more information to proceed"
            elif tool_name == "local_tools::generate_image":
                return "🎨 Creating an image based on your request"
            
            # Handle RAG (data store) tools by their pattern
            elif "research::" in tool_name:
                # Extract the friendly name of the data source
                source_name = tool_name.split("::")[-1].replace("_", " ").title()
                return f"🔍 Searching {source_name} for relevant information"
            
            # Handle generic actions based on their input requirements
            elif requires_code:
                return "💻 Working on a coding solution"
            elif requires_image:
                return "🖼️ Analyzing the provided image(s)"
            
            # Default fallback for any other tool
            else:
                # Clean up the technical tool name for a more readable display
                clean_name = tool_name.replace("_", " ").replace("::", " - ").title()
                return f"🔧 Using the {clean_name} tool"
    def generate_with_mcp_rag(
            self,
            prompt: str,
            context: Optional[str] = None,
            use_mcps: Union[None, bool, List[str]] = None,
            use_data_store: Union[None, Dict[str, Callable]] = None,
            system_prompt: str|None = None,
            reasoning_system_prompt: str = "You are a logical AI assistant. Your task is to achieve the user's goal by thinking step-by-step and using the available tools.",
            images: Optional[List[str]] = None,
            max_reasoning_steps: int|None = None,
            decision_temperature: float = 0.5,
            final_answer_temperature: float = 0.7,
            streaming_callback: Optional[Callable[[str, 'MSG_TYPE', Optional[Dict], Optional[List]], bool]] = None,
            rag_top_k: int = 5,
            rag_min_similarity_percent: float = 50.0,
            output_summarization_threshold: int = 500,
            force_mcp_use: bool = False,
            debug: bool = False,
            enable_parallel_execution: bool = True,
            enable_self_reflection: bool = True,
            **llm_generation_kwargs
        ) -> Dict[str, Any]:
        
        if not self.llm:
            return {"final_answer": "", "tool_calls": [], "sources": [], "error": "LLM binding not initialized."}
        if max_reasoning_steps is None:
            max_reasoning_steps=15

        def log_event(desc, event_type=MSG_TYPE.MSG_TYPE_CHUNK, meta=None, event_id=None) -> Optional[str]:
            if not streaming_callback: return None
            is_start = event_type == MSG_TYPE.MSG_TYPE_STEP_START
            event_id = str(uuid.uuid4()) if is_start and not event_id else event_id
            params = {"type": event_type, "description": desc, **(meta or {})}
            if event_id: params["id"] = event_id
            streaming_callback(desc, event_type, params)
            return event_id

        def log_prompt(title: str, prompt_text: str):
            if not debug: return
            ASCIIColors.cyan(f"** DEBUG: {title} **")
            ASCIIColors.magenta(prompt_text[-15000:])
            prompt_size = self.count_tokens(prompt_text)
            ASCIIColors.red(f"Prompt size:{prompt_size}/{self.llm.default_ctx_size}")
            ASCIIColors.cyan(f"** DEBUG: DONE **")

        discovery_step_id = log_event("🔧 Setting up capabilities...", MSG_TYPE.MSG_TYPE_STEP_START)
        all_discovered_tools, visible_tools, rag_registry, rag_tool_specs = [], [], {}, {}
        if use_mcps and hasattr(self, 'mcp'):
            mcp_tools = self.mcp.discover_tools(force_refresh=True)
            if isinstance(use_mcps, list): all_discovered_tools.extend([t for t in mcp_tools if t["name"] in use_mcps])
            elif use_mcps is True: all_discovered_tools.extend(mcp_tools)
        if use_data_store:
            for name, info in use_data_store.items():
                tool_name, description, call_fn = f"research::{name}", f"Queries the '{name}' knowledge base.", None
                if callable(info): call_fn = info
                elif isinstance(info, dict):
                    if "callable" in info and callable(info["callable"]): call_fn = info["callable"]
                    description = info.get("description", description)
                if call_fn:
                    visible_tools.append({"name": tool_name, "description": description, "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}})
                    rag_registry[tool_name] = call_fn
                    rag_tool_specs[tool_name] = {"default_top_k": rag_top_k, "default_min_sim": rag_min_similarity_percent}
        visible_tools.extend(all_discovered_tools)
        built_in_tools = [{"name": "local_tools::final_answer", "description": "Provide the final answer directly to the user.", "input_schema": {}}]
        if getattr(self, "tti", None): built_in_tools.append({"name": "local_tools::generate_image", "description": "Generate an image from a text description.", "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}})
        all_visible_tools = visible_tools + built_in_tools
        tool_summary = "\n".join([f"- {t['name']}: {t['description']}" for t in all_visible_tools[:15]])
        log_event(f"✅ Ready with {len(all_visible_tools)} capabilities", MSG_TYPE.MSG_TYPE_STEP_END, event_id=discovery_step_id)

        triage_step_id = log_event("🤔 Analyzing the best approach...", MSG_TYPE.MSG_TYPE_STEP_START)
        strategy = "COMPLEX_PLAN"
        strategy_data = {}
        try:
            triage_prompt = f"""Analyze the user's request and determine the most efficient strategy.
USER REQUEST: "{prompt}"
AVAILABLE TOOLS:\n{tool_summary}
Choose a strategy:
- "DIRECT_ANSWER": For greetings or simple questions that need no tools.
- "REQUEST_CLARIFICATION": If the request is ambiguous and you need more information from the user.
- "SINGLE_TOOL": If the request can be resolved with one tool call.
- "COMPLEX_PLAN": For multi-step requests requiring multiple tools or complex reasoning.

Provide your decision as JSON: {{"thought": "...", "strategy": "...", "text_output": "Your direct answer or clarification question.", "required_tool_name": "..."}}"""
            
            triage_schema = {
                "thought": "string", "strategy": "string", 
                "text_output": "string", "required_tool_name": "string"
            }
            strategy_data = self.generate_structured_content(prompt=triage_prompt, schema=triage_schema, temperature=0.1, **llm_generation_kwargs)
            strategy = strategy_data.get("strategy") if strategy_data else "COMPLEX_PLAN"
        except Exception as e:
            log_event(f"Triage failed, defaulting to complex plan. Error: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, event_id=triage_step_id)

        if force_mcp_use and strategy == "DIRECT_ANSWER":
            strategy = "COMPLEX_PLAN"
        log_event(f"✅ Approach decided: {strategy.replace('_', ' ').title()}", MSG_TYPE.MSG_TYPE_STEP_END, event_id=triage_step_id)

        if strategy == "DIRECT_ANSWER":
            final_answer = strategy_data.get("text_output", "I can help with that.")
            if streaming_callback: streaming_callback(final_answer, MSG_TYPE.MSG_TYPE_CONTENT, {})
            return {"final_answer": final_answer, "tool_calls": [], "sources": [], "error": None, "clarification_required": False, "final_scratchpad": f"Strategy: DIRECT_ANSWER\nThought: {strategy_data.get('thought')}"}

        if strategy == "REQUEST_CLARIFICATION":
            clarification_question = strategy_data.get("text_output", "Could you please provide more details?")
            return {"final_answer": clarification_question, "tool_calls": [], "sources": [], "error": None, "clarification_required": True, "final_scratchpad": f"Strategy: REQUEST_CLARIFICATION\nThought: {strategy_data.get('thought')}"}

        if strategy == "SINGLE_TOOL":
            synthesis_id = log_event("⚡ Taking a direct approach...", MSG_TYPE.MSG_TYPE_STEP_START)
            try:
                tool_name = strategy_data.get("required_tool_name")
                tool_spec = next((t for t in all_visible_tools if t['name'] == tool_name), None)
                if not tool_spec:
                    raise ValueError(f"LLM chose an unavailable tool: '{tool_name}'")
                
                param_prompt = f"""Given the user request, generate the correct parameters for the selected tool.
USER REQUEST: "{prompt}"
SELECTED TOOL: {json.dumps(tool_spec, indent=2)}
Output ONLY the JSON for the tool's parameters: {{"tool_params": {{...}}}}"""
                param_data = self.generate_structured_content(prompt=param_prompt, schema={"tool_params": "object"}, temperature=0.1, **llm_generation_kwargs)
                tool_params = param_data.get("tool_params", {}) if param_data else {}
                
                start_time, sources, tool_result = time.time(), [], {}
                if tool_name in rag_registry:
                    query = tool_params.get("query", prompt)
                    rag_fn = rag_registry[tool_name]
                    raw_results = rag_fn(query=query, rag_top_k=rag_top_k, rag_min_similarity_percent=rag_min_similarity_percent)
                    docs = [d for d in (raw_results.get("results", []) if isinstance(raw_results, dict) else raw_results or [])]
                    tool_result = {"status": "success", "results": docs}
                    sources = [{"source": tool_name, "metadata": d.get("metadata", {}), "score": d.get("score", 0.0)} for d in docs]
                elif hasattr(self, "mcp") and "local_tools" not in tool_name:
                    tool_result = self.mcp.execute_tool(tool_name, tool_params, lollms_client_instance=self)
                else:
                    tool_result = {"status": "failure", "error": f"Tool '{tool_name}' could not be executed in single-step mode."}
                
                if tool_result.get("status","success") != "success" or "error" in tool_result:
                    error_detail = tool_result.get("error", "Unknown tool error in single-step mode.")
                    raise RuntimeError(error_detail)

                response_time = time.time() - start_time
                tool_calls_this_turn = [{"name": tool_name, "params": tool_params, "result": tool_result, "response_time": response_time}]
                
                synthesis_prompt = f"""The user asked: "{prompt}"
I used the tool '{tool_name}' and got this result: {json.dumps(tool_result, indent=2)}
Synthesize a direct, user-friendly final answer."""
                final_answer = self.generate_text(prompt=synthesis_prompt, system_prompt=system_prompt, stream=streaming_callback is not None, streaming_callback=streaming_callback, temperature=final_answer_temperature, **llm_generation_kwargs)
                final_answer = self.remove_thinking_blocks(final_answer)
                
                log_event("✅ Direct answer ready!", MSG_TYPE.MSG_TYPE_STEP_END, event_id=synthesis_id)
                return {"final_answer": final_answer, "tool_calls": tool_calls_this_turn, "sources": sources, "error": None, "clarification_required": False, "final_scratchpad": f"Strategy: SINGLE_TOOL\nTool: {tool_name}\nResult: {json.dumps(tool_result)}"}

            except Exception as e:
                log_event(f"Direct approach failed: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, event_id=synthesis_id)
                log_event("Escalating to a more detailed plan.", MSG_TYPE.MSG_TYPE_INFO)

        return self._execute_complex_reasoning_loop(
            prompt=prompt, context=context, system_prompt=system_prompt,
            reasoning_system_prompt=reasoning_system_prompt, images=images,
            max_reasoning_steps=max_reasoning_steps, decision_temperature=decision_temperature,
            final_answer_temperature=final_answer_temperature, streaming_callback=streaming_callback,
            debug=debug, enable_self_reflection=enable_self_reflection,
            all_visible_tools=all_visible_tools, rag_registry=rag_registry, rag_tool_specs=rag_tool_specs,
            log_event_fn=log_event, log_prompt_fn=log_prompt,
            **llm_generation_kwargs
        )

    def _execute_complex_reasoning_loop(
        self, prompt, context, system_prompt, reasoning_system_prompt, images,
        max_reasoning_steps, decision_temperature, final_answer_temperature,
        streaming_callback, debug, enable_self_reflection, all_visible_tools,
        rag_registry, rag_tool_specs, log_event_fn, log_prompt_fn, **llm_generation_kwargs
    ) -> Dict[str, Any]:
        
        planner, memory_manager, performance_tracker = TaskPlanner(self), MemoryManager(), ToolPerformanceTracker()
        
        def _get_friendly_action_description(tool_name, requires_code, requires_image):
            if tool_name == "local_tools::final_answer": return "📋 Ready to provide your answer"
            if tool_name == "local_tools::request_clarification": return "❓ Need to ask for clarification"
            if tool_name == "local_tools::generate_image": return "🎨 Creating an image for you"
            if "research::" in tool_name: return f"🔍 Searching {tool_name.split('::')[-1]} for information"
            if requires_code: return "💻 Working on a coding solution"
            if requires_image: return "🖼️ Analyzing the provided images"
            return f"🔧 Using {tool_name.replace('_', ' ').replace('::', ' - ').title()}"

        original_user_prompt, tool_calls_this_turn, sources_this_turn = prompt, [], []
        asset_store: Dict[str, Dict] = {}
        
        planning_step_id = log_event_fn("📋 Creating a detailed plan...", MSG_TYPE.MSG_TYPE_STEP_START)
        execution_plan = planner.decompose_task(original_user_prompt, context or "")
        log_event_fn(f"✅ Plan ready ({len(execution_plan.tasks)} steps)", MSG_TYPE.MSG_TYPE_STEP_END, event_id=planning_step_id)
        
        initial_state_parts = [f"### Execution Plan\n- Total tasks: {len(execution_plan.tasks)}"]
        for i, task in enumerate(execution_plan.tasks): initial_state_parts.append(f"  {i+1}. {task.description}")
        if images:
            for img_b64 in images:
                img_uuid = str(uuid.uuid4())
                asset_store[img_uuid] = {"type": "image", "content": img_b64, "source": "user"}
                initial_state_parts.append(f"- User provided image, asset ID: {img_uuid}")
        current_scratchpad = "\n".join(initial_state_parts)

        formatted_tools_list = "\n".join([f"**{t['name']}**: {t['description']}" for t in all_visible_tools])
        completed_tasks, current_task_index = set(), 0
        
        for i in range(max_reasoning_steps):
            step_desc = f"🤔 Working on: {execution_plan.tasks[current_task_index].description}" if current_task_index < len(execution_plan.tasks) else f"🤔 Analyzing next steps... ({i+1}/{max_reasoning_steps})"
            reasoning_step_id = log_event_fn(step_desc, MSG_TYPE.MSG_TYPE_STEP_START)
            
            try:
                if len(current_scratchpad) > 12000:
                    current_scratchpad = memory_manager.compress_scratchpad(current_scratchpad, original_user_prompt, 8000)
                
                reasoning_prompt = f"""--- AVAILABLE ACTIONS ---\n{formatted_tools_list}\n--- YOUR INTERNAL SCRATCHPAD ---\n{current_scratchpad}\n--- END SCRATCHPAD ---\n
INSTRUCTIONS: Observe, think, and then act. Choose the single best next action to achieve: "{original_user_prompt}".
Produce ONLY this JSON: {{"thought": "short reasoning", "action": {{"tool_name": "...", "requires_code_input": false, "requires_image_input": false}}}}"""
                decision_data = self.generate_structured_content(prompt=reasoning_prompt, schema={"thought": "string", "action": "object"}, system_prompt=reasoning_system_prompt, temperature=decision_temperature, **llm_generation_kwargs)
                
                if not (decision_data and isinstance(decision_data.get("action"), dict)):
                    log_event_fn("LLM failed to produce a valid action JSON.", MSG_TYPE.MSG_TYPE_WARNING, event_id=reasoning_step_id)
                    current_scratchpad += "\n\n### Step Failure\n- Error: Invalid decision JSON from LLM."
                    continue

                action = decision_data.get("action", {})
                tool_name, requires_code, requires_image = action.get("tool_name"), action.get("requires_code_input", False), action.get("requires_image_input", False)
                current_scratchpad += f"\n\n### Step {i+1}: Thought\n{decision_data.get('thought', '')}"
                
                log_event_fn(_get_friendly_action_description(tool_name, requires_code, requires_image), MSG_TYPE.MSG_TYPE_STEP)
                if tool_name == "local_tools::final_answer": break
                if tool_name == "local_tools::request_clarification":
                    clarification_prompt = f"Based on your thought process, what is the single question you need to ask the user?\n\nSCRATCHPAD:\n{current_scratchpad}\n\nQUESTION:"
                    question = self.generate_text(clarification_prompt)
                    return {"final_answer": self.remove_thinking_blocks(question), "clarification_required": True, "final_scratchpad": current_scratchpad, "tool_calls": tool_calls_this_turn, "sources": sources_this_turn, "error": None}

                param_assets = {}
                if requires_code:
                    code_prompt = f"Generate only the raw code required for the current step.\n\nSCRATCHPAD:\n{current_scratchpad}\n\nCODE:"
                    code_content = self.generate_code(prompt=code_prompt, **llm_generation_kwargs)
                    code_uuid = f"code_asset_{uuid.uuid4()}"
                    asset_store[code_uuid] = {"type": "code", "content": code_content}
                    param_assets['code_asset_id'] = code_uuid
                    log_event_fn("Code asset generated.", MSG_TYPE.MSG_TYPE_STEP)
                if requires_image:
                    image_assets = [asset_id for asset_id, asset in asset_store.items() if asset['type'] == 'image' and asset.get('source') == 'user']
                    if image_assets:
                        param_assets['image_asset_id'] = image_assets[0]

                param_prompt = f"""Fill the parameters for the tool: '{tool_name}'. Available assets: {json.dumps(param_assets)}.
SCRATCHPAD:\n{current_scratchpad}\n
Output only: {{"tool_params": {{...}}}}"""
                param_data = self.generate_structured_content(prompt=param_prompt, schema={"tool_params": "object"}, temperature=decision_temperature, **llm_generation_kwargs)
                tool_params = param_data.get("tool_params", {}) if param_data else {}
                
                def _hydrate(data: Any, store: Dict) -> Any:
                    if isinstance(data, dict): return {k: _hydrate(v, store) for k, v in data.items()}
                    if isinstance(data, list): return [_hydrate(item, store) for item in data]
                    if isinstance(data, str) and "asset_" in data and data in store: return store[data].get("content", data)
                    return data
                hydrated_params = _hydrate(tool_params, asset_store)
                
                start_time, tool_result = time.time(), {"status": "failure", "error": f"Tool '{tool_name}' failed to execute."}
                try:
                    if tool_name in rag_registry:
                        query = hydrated_params.get("query", "")
                        top_k, min_sim = rag_tool_specs[tool_name]["default_top_k"], rag_tool_specs[tool_name]["default_min_sim"]
                        raw_results = rag_registry[tool_name](query=query, rag_top_k=top_k)
                        raw_iter = raw_results["results"] if isinstance(raw_results, dict) and "results" in raw_results else raw_results
                        docs = [{"text": d.get("text", str(d)), "score": d.get("score", 0)*100, "metadata": d.get("metadata", {})} for d in raw_iter or []]
                        kept = [x for x in docs if x['score'] >= min_sim]
                        tool_result = {"status": "success", "results": kept, "dropped": len(docs) - len(kept)}
                        sources_this_turn.extend([{"source": tool_name, "metadata": x["metadata"], "score": x["score"]} for x in kept])
                    elif hasattr(self, "mcp"):
                        tool_result = self.mcp.execute_tool(tool_name, hydrated_params, lollms_client_instance=self)
                except Exception as e:
                    error_msg = f"Exception during '{tool_name}' execution: {e}"
                    log_event_fn(error_msg, MSG_TYPE.MSG_TYPE_EXCEPTION)
                    tool_result = {"status": "failure", "error": error_msg}

                response_time = time.time() - start_time
                success = tool_result.get("status") == "success"
                performance_tracker.record_tool_usage(tool_name, success, 0.8, response_time, tool_result.get("error"))
                
                if success and current_task_index < len(execution_plan.tasks):
                    execution_plan.tasks[current_task_index].status = TaskStatus.COMPLETED
                    current_task_index += 1
                
                observation_text = f"```json\n{json.dumps(tool_result, indent=2)}\n```"
                tool_calls_this_turn.append({"name": tool_name, "params": tool_params, "result": tool_result, "response_time": response_time})
                current_scratchpad += f"\n\n### Step {i+1}: Observation\n- Action: `{tool_name}`\n- Result:\n{observation_text}"
                
                if success:
                    log_event_fn(f"✅ Step completed successfully", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id)
                else:
                    error_detail = tool_result.get("error", "No error detail provided.")
                    log_event_fn(f"Tool reported failure: {error_detail}", MSG_TYPE.MSG_TYPE_WARNING)
                    log_event_fn(f"⚠️ Step completed with issues", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id, meta={"error": error_detail})
                
                if len(completed_tasks) == len(execution_plan.tasks): break
                    
            except Exception as ex:
                log_event_fn(f"An unexpected error occurred in reasoning loop: {ex}", MSG_TYPE.MSG_TYPE_EXCEPTION, event_id=reasoning_step_id)
                trace_exception(ex)
                log_event_fn("⚠️ Encountered an issue, adjusting approach...", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id)

        if enable_self_reflection and len(tool_calls_this_turn) > 1:
            reflection_id = log_event_fn("🤔 Reviewing my work...", MSG_TYPE.MSG_TYPE_STEP_START)
            try:
                reflection_prompt = f"""Review the user request and your work. Was the goal achieved effectively?
REQUEST: "{original_user_prompt}"
SCRATCHPAD:\n{current_scratchpad}\n
JSON assessment: {{"goal_achieved": true, "effectiveness_score": 0.8, "summary": "..."}}"""
                reflection_data = self.generate_structured_content(prompt=reflection_prompt, schema={"goal_achieved": "boolean", "effectiveness_score": "number", "summary": "string"}, temperature=0.3, **llm_generation_kwargs)
                if reflection_data: current_scratchpad += f"\n\n### Self-Reflection\n- Goal Achieved: {reflection_data.get('goal_achieved')}\n- Effectiveness: {reflection_data.get('effectiveness_score')}"
                log_event_fn("✅ Quality check completed", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reflection_id)
            except Exception as e:
                log_event_fn(f"Self-review failed: {e}", MSG_TYPE.MSG_TYPE_WARNING, event_id=reflection_id)

        synthesis_id = log_event_fn("📝 Preparing your complete answer...", MSG_TYPE.MSG_TYPE_STEP_START)
        final_answer_prompt = f"""Synthesize a comprehensive, user-friendly final answer based on your complete analysis.
USER REQUEST: "{original_user_prompt}"
FULL SCRATCHPAD:\n{current_scratchpad}\n---
FINAL ANSWER:"""
        
        final_answer_text = self.generate_text(prompt=final_answer_prompt, system_prompt=system_prompt, stream=streaming_callback is not None, streaming_callback=streaming_callback, temperature=final_answer_temperature, **llm_generation_kwargs)
        if isinstance(final_answer_text, dict) and "error" in final_answer_text: 
            return {"final_answer": "", "error": final_answer_text["error"], "final_scratchpad": current_scratchpad}
        
        final_answer = self.remove_thinking_blocks(final_answer_text)
        log_event_fn("✅ Answer ready!", MSG_TYPE.MSG_TYPE_STEP_END, event_id=synthesis_id)

        overall_confidence = sum(c.get('confidence', 0.5) for c in tool_calls_this_turn) / max(len(tool_calls_this_turn), 1)
        return {
            "final_answer": final_answer, "final_scratchpad": current_scratchpad,
            "tool_calls": tool_calls_this_turn, "sources": sources_this_turn,
            "performance_stats": {"total_steps": len(tool_calls_this_turn), "average_confidence": overall_confidence},
            "clarification_required": False, "overall_confidence": overall_confidence, "error": None
        }


    def generate_code(
                        self,
                        prompt:str,
                        images=[],
                        system_prompt:str|None=None,
                        template:str|None=None,
                        language="json",
                        code_tag_format="markdown", # or "html"
                        n_predict:int|None = None,
                        temperature:float|None = None,
                        top_k:int|None= None,
                        top_p:float|None=None,
                        repeat_penalty:float|None=None,
                        repeat_last_n:int|None=None,
                        callback=None,
                        debug:bool=False ):
        """
        Generates a single code block based on a prompt.
        Uses the underlying LLM binding via `generate_text`.
        Handles potential continuation if the code block is incomplete.
        """
        if not  system_prompt:
            system_prompt = f"""Act as a code generation assistant that generates code from user prompt."""

        if template and template !="{}":
            if language in ["json","yaml","xml"]:
                system_prompt += f"\nMake sure the generated context follows the following schema:\n```{language}\n{template}\n```\n"
            else:
                system_prompt += f"\nHere is a template of the answer:\n```{language}\n{template}\n```\n"
            
            if code_tag_format=="markdown":
                system_prompt += f"""You must answer with the code placed inside the markdown code tag:
```{language}
```
"""
            elif code_tag_format=="html":
                system_prompt +=f"""You must answer with the code placed inside the html code tag:
<code language="{language}">
</code>
"""
        system_prompt += f"""You must return a single code tag.
Do not split the code in multiple tags.
{self.ai_full_header}"""

        response = self.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            streaming_callback=callback
            )

        if isinstance(response, dict) and not response.get("status", True):
             ASCIIColors.error(f"Code generation failed: {response.get('error')}")
             return None

        codes = self.extract_code_blocks(response, format=code_tag_format)
        code_content = None

        if codes:
            last_code = codes[-1]
            code_content = last_code["content"]

            # Handle incomplete code block continuation (simple approach)
            max_retries = 3 # Limit continuation attempts
            retries = 0
            while not last_code["is_complete"] and retries < max_retries:
                retries += 1
                ASCIIColors.info(f"Code block seems incomplete. Attempting continuation ({retries}/{max_retries})...")
                continuation_prompt = f"{prompt}\n\nAssistant:\n{code_content}\n\n{self.user_full_header}The previous code block was incomplete. Continue the code exactly from where it left off. Do not repeat the previous part. Only provide the continuation inside a single {code_tag_format} code tag.\n{self.ai_full_header}"

                continuation_response = self.generate_text(
                    continuation_prompt,
                    images=images, # Resend images if needed for context
                    n_predict=n_predict, # Allow space for continuation
                    temperature=temperature, # Use same parameters
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    repeat_last_n=repeat_last_n,
                    streaming_callback=callback
                )

                if isinstance(continuation_response, dict) and not continuation_response.get("status", True):
                    ASCIIColors.warning(f"Continuation attempt failed: {continuation_response.get('error')}")
                    break # Stop trying if generation fails

                continuation_codes = self.extract_code_blocks(continuation_response, format=code_tag_format)

                if continuation_codes:
                    new_code_part = continuation_codes[0]["content"]
                    code_content += "\n" + new_code_part # Append continuation
                    last_code["is_complete"] = continuation_codes[0]["is_complete"] # Update completeness
                    if last_code["is_complete"]:
                        ASCIIColors.info("Code block continuation successful.")
                        break # Exit loop if complete
                else:
                     ASCIIColors.warning("Continuation response contained no code block.")
                     break # Stop if no code block found in continuation

            if not last_code["is_complete"]:
                ASCIIColors.warning("Code block remained incomplete after multiple attempts.")

        return code_content # Return the (potentially completed) code content or None

    def generate_structured_content(
        self,
        prompt,
        images=None,
        schema=None,
        system_prompt=None,
        max_retries=1,
        **kwargs
    ):
        import json
        images = [] if images is None else images
        schema = {} if schema is None else schema
        try:
            from jsonschema import validate
            has_validator = True
        except ImportError:
            has_validator = False

        if isinstance(schema, dict):
            schema_obj = schema
        elif isinstance(schema, str):
            try:
                schema_obj = json.loads(schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"The provided schema string is not valid JSON: {e}")
        else:
            raise TypeError("schema must be a dict or a JSON string.")

        # --- FIX STARTS HERE ---
        # Heuristic to detect if the schema is a properties-only dictionary
        # and needs to be wrapped in a root object to be a valid schema.
        # This handles cases where the user provides `{"field1": {...}, "field2": {...}}`
        # instead of `{"type": "object", "properties": {"field1": ...}}`.
        if "type" not in schema_obj and "properties" not in schema_obj and all(isinstance(v, dict) for v in schema_obj.values()):
            if kwargs.get("debug"):
                ASCIIColors.info("Schema appears to be a properties-only dictionary; wrapping it in a root object.")
            schema_obj = {
                "type": "object",
                "properties": schema_obj,
                # Assume all top-level keys are required when wrapping
                "required": list(schema_obj.keys())
            }
        # --- FIX ENDS HERE ---

        def _instance_skeleton(s):
            if not isinstance(s, dict):
                return {}
            if "const" in s:
                return s["const"]
            if "enum" in s and isinstance(s["enum"], list) and s["enum"]:
                return s["enum"][0]
            
            # Handle default values
            if "default" in s:
                return s["default"]

            t = s.get("type")
            if t == "string":
                return ""
            if t == "integer":
                return 0
            if t == "number":
                return 0.0
            if t == "boolean":
                return False
            if t == "array":
                # Generate one minimal item if schema is provided
                items = s.get("items", {})
                min_items = s.get("minItems", 0)
                # Let's generate at least one item for the example if possible
                num_items = max(min_items, 1) if items and not min_items == 0 else min_items
                return [_instance_skeleton(items) for _ in range(num_items)]
            if t == "object":
                props = s.get("properties", {})
                # Use required fields, otherwise fall back to all properties for the skeleton
                req = s.get("required", list(props.keys()))
                out = {}
                for k in req:
                    if k in props:
                        out[k] = _instance_skeleton(props[k])
                    else:
                        out[k] = None # Should not happen if schema is well-formed
                return out
            if "oneOf" in s and isinstance(s["oneOf"], list) and s["oneOf"]:
                return _instance_skeleton(s["oneOf"][0])
            if "anyOf" in s and isinstance(s["anyOf"], list) and s["anyOf"]:
                return _instance_skeleton(s["anyOf"][0])
            if "allOf" in s and isinstance(s["allOf"], list) and s["allOf"]:
                merged = {}
                for sub in s["allOf"]:
                    val = _instance_skeleton(sub)
                    if isinstance(val, dict):
                        merged.update(val)
                return merged if merged else {}
            return {}

        # Now derive strings from the (potentially corrected) schema_obj
        schema_str = json.dumps(schema_obj, indent=2, ensure_ascii=False)
        example_obj = _instance_skeleton(schema_obj)
        example_str = json.dumps(example_obj, indent=2, ensure_ascii=False)

        base_system = (
            "Your objective is to generate a JSON object that satisfies the user's request and conforms to the provided schema.\n"
            "Rules:\n"
            "1) The schema is reference ONLY. Do not include the schema in the output.\n"
            "2) Output exactly ONE valid JSON object.\n"
            "3) Wrap the JSON object inside a single ```json code block.\n"
            "4) Do not output explanations or text outside the JSON.\n"
            "5) Use 2 spaces for indentation. Do not use tabs.\n"
            "6) Only include fields allowed by the schema and ensure all required fields are present.\n"
            "7) For enums, choose a valid value from the list.\n\n"
            "Schema (reference only):\n"
            f"```json\n{schema_str}\n```\n\n"
            "Correct example of output format (structure only, values are illustrative):\n"
            f"```json\n{example_str}\n```"
        )
        full_system_prompt = f"{system_prompt}\n\n{base_system}" if system_prompt else base_system

        if kwargs.get("debug"):
            ASCIIColors.info("Generating structured content...")

        last_error = None
        for attempt in range(max_retries + 1):
            json_string = self.generate_code(
                prompt=prompt,
                images=images,
                system_prompt=full_system_prompt if attempt == 0 else f"{full_system_prompt}\n\nPrevious attempt failed validation: {last_error}\nReturn a corrected JSON instance that strictly satisfies the schema.",
                template=example_str,
                language="json",
                code_tag_format="markdown",
                **kwargs
            )
            if not json_string:
                last_error = "LLM returned an empty response."
                if kwargs.get("debug"): ASCIIColors.warning(last_error)
                continue

            if kwargs.get("debug"):
                ASCIIColors.info("Parsing generated JSON string...")
                print(f"--- Raw JSON String ---\n{json_string}\n-----------------------")

            try:
                parsed_json = robust_json_parser(json_string)
                if parsed_json is None:
                    last_error = "Failed to robustly parse the generated string into JSON."
                    if kwargs.get("debug"): ASCIIColors.warning(last_error)
                    continue
                
                if has_validator:
                    try:
                        validate(instance=parsed_json, schema=schema_obj)
                        return parsed_json
                    except Exception as ve:
                        last_error = f"JSON Schema Validation Error: {ve}"
                        if kwargs.get("debug"): ASCIIColors.warning(last_error)
                        if attempt < max_retries:
                            continue
                        # Return the invalid object after last retry if validation fails
                        return parsed_json
                return parsed_json
            except Exception as e:
                trace_exception(e)
                ASCIIColors.error(f"Unexpected error during JSON processing: {e}")
                last_error = f"An unexpected error occurred: {e}"
                # Do not retry on unexpected errors, break the loop
                break
        
        ASCIIColors.error(f"Failed to generate valid structured content after {max_retries + 1} attempts. Last error: {last_error}")
        return None


    def extract_code_blocks(self, text: str, format: str = "markdown") -> List[dict]:
        """
        Extracts code blocks from text in Markdown or HTML format.
        (Implementation remains the same as provided before)
        """
        # ... (Keep the existing implementation from the previous file) ...
        code_blocks = []
        remaining = text
        first_index = 0
        indices = []

        if format.lower() == "markdown":
            # Markdown: Find triple backtick positions
            while remaining:
                try:
                    index = remaining.index("```")
                    indices.append(index + first_index)
                    remaining = remaining[index + 3:]
                    first_index += index + 3
                except ValueError:
                    if len(indices) % 2 == 1:  # Odd number of delimiters means the last block is open
                        indices.append(first_index + len(remaining)) # Mark end of text as end of block
                    break

        elif format.lower() == "html":
            # HTML: Find <code> and </code> positions, handling nested tags
            cursor = 0
            while cursor < len(text):
                try:
                    # Look for opening <code tag
                    start_index = text.index("<code", cursor)
                    try:
                        end_of_opening = text.index(">", start_index)
                    except ValueError:
                        break # Invalid opening tag

                    indices.append(start_index)
                    opening_tag_end = end_of_opening + 1
                    cursor = opening_tag_end

                    # Look for matching </code>, accounting for nested <code>
                    nest_level = 0
                    temp_cursor = cursor
                    found_closing = False
                    while temp_cursor < len(text):
                        if text[temp_cursor:].startswith("<code"):
                            nest_level += 1
                            try:
                                temp_cursor = text.index(">", temp_cursor) + 1
                            except ValueError:
                                break # Invalid nested opening tag
                        elif text[temp_cursor:].startswith("</code>"):
                            if nest_level == 0:
                                indices.append(temp_cursor)
                                cursor = temp_cursor + len("</code>")
                                found_closing = True
                                break
                            nest_level -= 1
                            temp_cursor += len("</code>")
                        else:
                            temp_cursor += 1

                    if not found_closing: # If no closing tag found until the end
                        indices.append(len(text))
                        break # Stop searching

                except ValueError:
                    break # No more opening tags found

        else:
            raise ValueError("Format must be 'markdown' or 'html'")

        # Process indices to extract blocks
        for i in range(0, len(indices), 2):
            block_infos = {
                'index': i // 2,
                'file_name': "",
                'content': "",
                'type': 'language-specific', # Default type
                'is_complete': False
            }

            start_pos = indices[i]
            # --- Extract preceding text for potential file name hints ---
            # Look backwards from start_pos for common patterns
            search_area_start = max(0, start_pos - 200) # Limit search area
            preceding_text_segment = text[search_area_start:start_pos]
            lines = preceding_text_segment.strip().splitlines()
            if lines:
                last_line = lines[-1].strip()
                # Example patterns (adjust as needed)
                if last_line.startswith("<file_name>") and last_line.endswith("</file_name>"):
                    block_infos['file_name'] = last_line[len("<file_name>"):-len("</file_name>")].strip()
                elif last_line.lower().startswith("file:") or last_line.lower().startswith("filename:"):
                    block_infos['file_name'] = last_line.split(":", 1)[1].strip()
            # --- End file name extraction ---

            # Extract content and type based on format
            if format.lower() == "markdown":
                content_start = start_pos + 3 # After ```
                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content_raw = text[content_start:end_pos]
                    block_infos['is_complete'] = True
                else: # Last block is open
                    content_raw = text[content_start:]
                    block_infos['is_complete'] = False

                # Check for language specifier on the first line
                first_line_end = content_raw.find('\n')
                if first_line_end != -1:
                    first_line = content_raw[:first_line_end].strip()
                    if first_line and not first_line.isspace() and ' ' not in first_line: # Basic check for language specifier
                        block_infos['type'] = first_line
                        content = content_raw[first_line_end + 1:].strip()
                    else:
                        content = content_raw.strip()
                else: # Single line code block or no language specifier
                    content = content_raw.strip()
                    # If content itself looks like a language specifier, clear it
                    if content and not content.isspace() and ' ' not in content and len(content)<20:
                         block_infos['type'] = content
                         content = ""


            elif format.lower() == "html":
                # Find end of opening tag to get content start
                try:
                    opening_tag_end = text.index(">", start_pos) + 1
                except ValueError:
                    continue # Should not happen if indices are correct

                opening_tag = text[start_pos:opening_tag_end]

                if i + 1 < len(indices):
                    end_pos = indices[i + 1]
                    content = text[opening_tag_end:end_pos].strip()
                    block_infos['is_complete'] = True
                else: # Last block is open
                    content = text[opening_tag_end:].strip()
                    block_infos['is_complete'] = False


                # Extract language from class attribute (more robust)
                import re
                match = re.search(r'class\s*=\s*["\']([^"\']*)["\']', opening_tag)
                if match:
                    classes = match.group(1).split()
                    for cls in classes:
                        if cls.startswith("language-"):
                            block_infos['type'] = cls[len("language-"):]
                            break # Take the first language- class found

            block_infos['content'] = content
            if block_infos['content'] or block_infos['is_complete']: # Add block if it has content or is closed
                code_blocks.append(block_infos)

        return code_blocks


    def extract_thinking_blocks(self, text: str) -> List[str]:
        """
        Extracts content between <thinking> or <think> tags from a given text.
        (Implementation remains the same as provided before)
        """
        import re
        pattern = r'<(thinking|think)>(.*?)</\1>'
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE) # Added IGNORECASE
        thinking_blocks = [match.group(2).strip() for match in matches]
        return thinking_blocks

    def remove_thinking_blocks(self, text: str) -> str:
        """
        Removes thinking blocks (either <thinking> or <think>) from text including the tags.
        (Implementation remains the same as provided before)
        """
        import re
        pattern = r'<(thinking|think)>.*?</\1>\s*' # Added \s* to remove potential trailing whitespace/newlines
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE) # Added IGNORECASE
        # Further cleanup might be needed depending on desired newline handling
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip() # Collapse excess newlines
        return cleaned_text

    # --- Task-oriented methods (Candidates for moving to TasksLibrary) ---
    # Keeping them here for now, but they primarily use generate_code/generate_text

    def yes_no(
            self,
            question: str,
            context: str = "",
            max_answer_length: int = None,
            conditionning: str = "",
            return_explanation: bool = False,
            callback = None
        ) -> bool | dict:
        """
        Answers a yes/no question using LLM JSON generation.
        (Implementation requires self.generate_code which uses self.generate_text)
        """
        # ... (Implementation as provided before, relies on self.generate_code) ...
        if not callback:
            callback=self.sink

        prompt = f"{self.system_full_header}{conditionning}\n{self.user_full_header}Based on the context, answer the question with only 'true' or 'false' and provide a brief explanation.\nContext:\n{context}\nQuestion: {question}\n{self.ai_full_header}"

        template = """{
    "answer": true | false, // boolean required
    "explanation": "A brief explanation for the answer"
}"""

        # Assuming generate_code exists and works as intended
        response_json_str = self.generate_code(
            prompt=prompt,
            language="json",
            template=template,
            code_tag_format="markdown",
            n_predict=max_answer_length,
            callback=callback
        )

        if response_json_str is None:
            ASCIIColors.error("LLM failed to generate JSON for yes/no question.")
            return {"answer": False, "explanation": "Generation failed"} if return_explanation else False

        try:
            # Attempt to repair minor JSON issues before parsing
            import json
            import re
            # Remove potential comments, trailing commas etc.
            response_json_str = re.sub(r"//.*", "", response_json_str)
            response_json_str = re.sub(r",\s*}", "}", response_json_str)
            response_json_str = re.sub(r",\s*]", "]", response_json_str)

            parsed_response = robust_json_parser(response_json_str)
            answer = parsed_response.get("answer")
            explanation = parsed_response.get("explanation", "")

            # Validate boolean type
            if not isinstance(answer, bool):
                # Attempt to coerce common string representations
                if isinstance(answer, str):
                    answer_lower = answer.lower()
                    if answer_lower == 'true':
                        answer = True
                    elif answer_lower == 'false':
                        answer = False
                    else:
                        raise ValueError("Answer is not a valid boolean representation.")
                else:
                     raise ValueError("Answer is not a boolean.")


            if return_explanation:
                return {"answer": answer, "explanation": explanation}
            else:
                return answer
        except (json.JSONDecodeError, ValueError) as e:
            ASCIIColors.error(f"Failed to parse or validate JSON response for yes/no: {e}")
            ASCIIColors.error(f"Received: {response_json_str}")
            # Fallback: try simple string check in the raw LLM output (less reliable)
            if "true" in response_json_str.lower():
                answer_fallback = True
            elif "false" in response_json_str.lower():
                answer_fallback = False
            else:
                answer_fallback = False # Default to false on ambiguity

            if return_explanation:
                return {"answer": answer_fallback, "explanation": f"Parsing failed ({e}). Fallback used."}
            else:
                return answer_fallback


    def multichoice_question(
            self,
            question: str,
            possible_answers: list,
            context: str = "",
            max_answer_length: int = None,
            conditionning: str = "",
            return_explanation: bool = False,
            callback = None
        ) -> int | dict: # Corrected return type hint
        """
        Interprets a multi-choice question using LLM JSON generation.
        (Implementation requires self.generate_code which uses self.generate_text)
        """
        # ... (Implementation as provided before, relies on self.generate_code) ...
        if not callback:
            callback=self.sink

        choices_text = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])

        prompt = f"{self.system_full_header}{conditionning}\n"
        prompt += f"{self.user_full_header}Answer the following multiple-choice question based on the context. Respond with a JSON object containing the index of the single best answer and an optional explanation.\n"
        if context:
             prompt += f"Context:\n{context}\n"
        prompt += f"Question:\n{question}\n"
        prompt += f"Possible Answers:\n{choices_text}\n"
        prompt += f"{self.ai_full_header}"

        template = """{
    "index": 0, // integer index required
    "explanation": "Optional explanation for the choice"
}"""

        response_json_str = self.generate_code(
            prompt=prompt,
            template=template,
            language="json",
            code_tag_format="markdown",
            n_predict=max_answer_length,
            callback=callback
        )

        if response_json_str is None:
            ASCIIColors.error("LLM failed to generate JSON for multichoice question.")
            return {"index": -1, "explanation": "Generation failed"} if return_explanation else -1

        try:
            # Attempt to repair minor JSON issues before parsing
            import json
            import re
            response_json_str = re.sub(r"//.*", "", response_json_str)
            response_json_str = re.sub(r",\s*}", "}", response_json_str)
            response_json_str = re.sub(r",\s*]", "]", response_json_str)

            result = robust_json_parser(response_json_str)
            index = result.get("index")
            explanation = result.get("explanation", "")

            if not isinstance(index, int) or not (0 <= index < len(possible_answers)):
                raise ValueError(f"Invalid index returned: {index}")

            if return_explanation:
                return {"index": index, "explanation": explanation}
            else:
                return index
        except (json.JSONDecodeError, ValueError) as e:
            ASCIIColors.error(f"Failed to parse or validate JSON response for multichoice: {e}")
            ASCIIColors.error(f"Received: {response_json_str}")
             # Fallback logic could be added here (e.g., regex for index) but is less reliable
            return {"index": -1, "explanation": f"Parsing failed ({e})."} if return_explanation else -1


    def multichoice_ranking(
            self,
            question: str,
            possible_answers: list,
            context: str = "",
            max_answer_length: int = None,
            conditionning: str = "",
            return_explanation: bool = False,
            callback = None
        ) -> dict:
        """
        Ranks answers for a question from best to worst using LLM JSON generation.
        (Implementation requires self.generate_code which uses self.generate_text)
        """
        if not callback:
            callback = self.sink

        choices_text = "\n".join([f"{i}. {ans}" for i, ans in enumerate(possible_answers)])

        prompt = f"{self.system_full_header}{conditionning}\n"
        prompt += f"{self.user_full_header}Rank the following answers to the question from best to worst based on the context. Respond with a JSON object containing a list of indices in ranked order and an optional list of explanations.\n"
        if context:
             prompt += f"Context:\n{context}\n"
        prompt += f"Question:\n{question}\n"
        prompt += f"Possible Answers to Rank:\n{choices_text}\n"
        prompt += f"{self.ai_full_header}"

        template = """{
    "ranking": [0, 1, 2], // list of integer indices required, length must match number of answers
    "explanations": ["Optional explanation 1", "Optional explanation 2", "Optional explanation 3"] // Optional list of strings
}"""

        response_json_str = self.generate_code(
            prompt=prompt,
            template=template,
            language="json",
            code_tag_format="markdown",
            n_predict=max_answer_length,
            callback=callback
        )

        default_return = {"ranking": [], "explanations": []} if return_explanation else {"ranking": []}

        if response_json_str is None:
            ASCIIColors.error("LLM failed to generate JSON for ranking.")
            return default_return

        try:
            # Attempt to repair minor JSON issues before parsing
            import json
            import re
            response_json_str = re.sub(r"//.*", "", response_json_str)
            response_json_str = re.sub(r",\s*}", "}", response_json_str)
            response_json_str = re.sub(r",\s*]", "]", response_json_str)

            result = robust_json_parser(response_json_str)
            ranking = result.get("ranking")
            explanations = result.get("explanations", []) if return_explanation else None

            # Validation
            if not isinstance(ranking, list) or len(ranking) != len(possible_answers):
                 raise ValueError("Ranking is not a list or has incorrect length.")
            if not all(isinstance(idx, int) and 0 <= idx < len(possible_answers) for idx in ranking):
                 raise ValueError("Ranking contains invalid indices.")
            if len(set(ranking)) != len(possible_answers):
                 raise ValueError("Ranking contains duplicate indices.")
            if return_explanation and not isinstance(explanations, list):
                 ASCIIColors.warning("Explanations format is invalid, returning ranking only.")
                 explanations = None # Ignore invalid explanations


            if return_explanation:
                return {"ranking": ranking, "explanations": explanations or [""] * len(ranking)} # Provide empty strings if explanations were invalid/missing
            else:
                return {"ranking": ranking}

        except (json.JSONDecodeError, ValueError) as e:
            ASCIIColors.error(f"Failed to parse or validate JSON response for ranking: {e}")
            ASCIIColors.error(f"Received: {response_json_str}")
            return default_return

    # --- Summarization / Analysis Methods (Candidates for TasksLibrary) ---
    # These use generate_text and tokenization/detokenization

    def sequential_summarize(
                                self,
                                text:str,
                                chunk_processing_prompt:str="Extract relevant information from the current text chunk and update the memory if needed.",
                                chunk_processing_output_format="markdown",
                                final_memory_processing_prompt="Create final summary using this memory.",
                                final_output_format="markdown",
                                ctx_size:int=None,
                                chunk_size:int=None,
                                overlap:int=None, # Added overlap
                                bootstrap_chunk_size:int=None,
                                bootstrap_steps:int=None,
                                callback = None,
                                debug:bool= False):
        """
        Processes text in chunks sequentially, updating a memory at each step.
        (Implementation requires self.tokenize, self.detokenize, self.generate_text, self.extract_code_blocks)
        """
        # ... (Implementation as provided before, relies on core methods) ...
        if not callback:
            callback = self.sink

        if ctx_size is None:
            ctx_size = self.llm.default_ctx_size or 8192 # Provide a fallback default
        if chunk_size is None:
            chunk_size = ctx_size // 4
        if overlap is None:
             overlap = chunk_size // 10 # Default overlap
        if bootstrap_chunk_size is None:
             bootstrap_chunk_size = chunk_size // 2 # Smaller initial chunks
        if bootstrap_steps is None:
             bootstrap_steps = 2 # Process first few chunks smaller

        # Tokenize entire text
        try:
            all_tokens = self.tokenize(text)
        except RuntimeError as e:
            ASCIIColors.error(f"Tokenization failed: {e}")
            return "Error: Could not tokenize input text."
        total_tokens = len(all_tokens)

        # Initialize memory and chunk index
        memory = ""
        start_token_idx = 0
        chunk_id = 0

        # Create static prompt template
        static_prompt_template = f"""{self.system_full_header}
You are a structured sequential text summary assistant that processes documents chunk by chunk, updating a memory of previously generated information at each step.

Your goal is to extract and combine relevant information from each text chunk with the existing memory, ensuring no key details are omitted or invented.

If requested, infer metadata like titles or authors from the content.

{self.user_full_header}
Update the memory by merging previous information with new details from this text chunk.
Only add information explicitly present in the chunk. Retain all relevant prior memory unless clarified or updated by the current chunk.

----
Text chunk (Chunk number: {{chunk_id}}):
```markdown
{{chunk}}
```

{{custom_prompt}}

Before updating, verify each requested detail:
1. Does the chunk explicitly mention the information?
2. Should prior memory be retained, updated, or clarified?

Include only confirmed details in the output.
Rewrite the full memory including the updates and keeping relevant data.
Do not discuss the information inside the memory, just put the relevant information without comments.
The output memory must be put inside a {chunk_processing_output_format} markdown code block.
----
Current document analysis memory:
```{chunk_processing_output_format}
{{memory}}
```
{self.ai_full_header}
```{chunk_processing_output_format}
""" # Added start of code block for AI

        # Calculate static prompt tokens (with estimated placeholders)
        example_prompt = static_prompt_template.format(
            custom_prompt=chunk_processing_prompt,
            memory="<est_memory>",
            chunk="<est_chunk>",
            chunk_id=0
            )
        try:
             static_tokens = len(self.tokenize(example_prompt)) - len(self.tokenize("<est_memory>")) - len(self.tokenize("<est_chunk>"))
        except RuntimeError as e:
             ASCIIColors.error(f"Tokenization failed during setup: {e}")
             return "Error: Could not calculate prompt size."

        # Process text in chunks
        while start_token_idx < total_tokens:
            # Calculate available tokens for chunk + memory
            available_tokens_for_dynamic_content = ctx_size - static_tokens - (self.llm.default_n_predict or 1024) # Reserve space for output
            if available_tokens_for_dynamic_content <= 100: # Need some minimum space
                ASCIIColors.error("Context size too small for summarization with current settings.")
                return "Error: Context size too small."

            # Estimate token split between memory and chunk (e.g., 50/50)
            max_memory_tokens = available_tokens_for_dynamic_content // 2
            max_chunk_tokens = available_tokens_for_dynamic_content - max_memory_tokens

            # Truncate memory if needed
            current_memory_tokens = self.tokenize(memory)
            if len(current_memory_tokens) > max_memory_tokens:
                 memory = self.detokenize(current_memory_tokens[-max_memory_tokens:]) # Keep recent memory
                 if debug: ASCIIColors.yellow(f"Memory truncated to {max_memory_tokens} tokens.")

            # Determine actual chunk size based on remaining space and settings
            current_chunk_size = bootstrap_chunk_size if chunk_id < bootstrap_steps else chunk_size
            current_chunk_size = min(current_chunk_size, max_chunk_tokens) # Adjust chunk size based on available space

            end_token_idx = min(start_token_idx + current_chunk_size, total_tokens)
            chunk_tokens = all_tokens[start_token_idx:end_token_idx]
            chunk = self.detokenize(chunk_tokens)

            chunk_id += 1
            callback(f"Processing chunk {chunk_id}...", MSG_TYPE.MSG_TYPE_STEP)

            # Generate memory update
            prompt = static_prompt_template.format(
                custom_prompt=chunk_processing_prompt,
                memory=memory,
                chunk=chunk,
                chunk_id=chunk_id
                )
            if debug:
                ASCIIColors.magenta(f"--- Chunk {chunk_id} Prompt ---")
                ASCIIColors.cyan(prompt)

            response = self.generate_text(prompt, n_predict=(self.llm.default_n_predict or 1024), streaming_callback=callback)

            if isinstance(response, dict): # Handle generation error
                 ASCIIColors.error(f"Chunk {chunk_id} processing failed: {response.get('error')}")
                 # Option: skip chunk or stop? Let's skip for now.
                 start_token_idx = end_token_idx # Move to next chunk index
                 continue

            memory_code_blocks = self.extract_code_blocks(response, format=chunk_processing_output_format)
            if memory_code_blocks:
                memory = memory_code_blocks[0]["content"] # Assume first block is the memory
            else:
                # Fallback: Try to extract from the end if the AI added text after the block
                end_tag = f"```{chunk_processing_output_format}"
                last_occurrence = response.rfind(end_tag)
                if last_occurrence != -1:
                    # Extract content between the start and end tags
                    start_tag_len = len(f"```{chunk_processing_output_format}\n") # Approx
                    potential_memory = response[last_occurrence + start_tag_len:].strip()
                    if potential_memory.endswith("```"):
                         potential_memory = potential_memory[:-3].strip()
                    if potential_memory: # Use if non-empty
                         memory = potential_memory
                    else: # If extraction failed, keep old memory or use raw response? Use raw response for now.
                         ASCIIColors.warning(f"Could not extract memory block for chunk {chunk_id}. Using raw response.")
                         memory = response.strip().rstrip('```') # Basic cleanup
                else:
                     ASCIIColors.warning(f"Could not extract memory block for chunk {chunk_id}. Using raw response.")
                     memory = response.strip().rstrip('```')


            if debug:
                ASCIIColors.magenta(f"--- Chunk {chunk_id} Updated Memory ---")
                ASCIIColors.green(memory)
                ASCIIColors.magenta("----------------------------")

            # Move to next chunk start, considering overlap
            start_token_idx = max(start_token_idx, end_token_idx - overlap) if overlap>0 and end_token_idx < total_tokens else end_token_idx


        # --- Final Aggregation Step ---
        callback("Aggregating final summary...", MSG_TYPE.MSG_TYPE_STEP)
        final_prompt_template = f"""{self.system_full_header}
You are a memory summarizer assistant.
{final_memory_processing_prompt}.
{self.user_full_header}
Here is the document analysis memory:
```{chunk_processing_output_format}
{{memory}}
```
The final output must be put inside a {final_output_format} markdown tag.
{self.ai_full_header}
```{final_output_format}
"""

        # Truncate memory if needed for the final prompt
        final_example_prompt = final_prompt_template.format(memory="<final_memory>")
        try:
            final_static_tokens = len(self.tokenize(final_example_prompt)) - len(self.tokenize("<final_memory>"))
            available_final_tokens = ctx_size - final_static_tokens - (self.llm.default_n_predict or 1024) # Reserve space for output
        except RuntimeError as e:
             ASCIIColors.error(f"Tokenization failed during final setup: {e}")
             return "Error: Could not calculate final prompt size."


        memory_tokens = self.tokenize(memory)
        if len(memory_tokens) > available_final_tokens:
            memory = self.detokenize(memory_tokens[-available_final_tokens:]) # Keep most recent info
            if debug: ASCIIColors.yellow(f"Final memory truncated to {available_final_tokens} tokens.")

        # Generate final summary
        final_prompt = final_prompt_template.format(memory=memory)
        if debug:
            ASCIIColors.magenta("--- Final Aggregation Prompt ---")
            ASCIIColors.cyan(final_prompt)

        final_summary_raw = self.generate_text(final_prompt, n_predict=(self.llm.default_n_predict or 1024), streaming_callback=callback)

        if isinstance(final_summary_raw, dict):
             ASCIIColors.error(f"Final aggregation failed: {final_summary_raw.get('error')}")
             return "Error: Final aggregation failed."

        final_code_blocks = self.extract_code_blocks(final_summary_raw, format=final_output_format)
        if final_code_blocks:
             final_summary = final_code_blocks[0]["content"]
        else:
             # Fallback similar to chunk processing
             end_tag = f"```{final_output_format}"
             last_occurrence = final_summary_raw.rfind(end_tag)
             if last_occurrence != -1:
                 start_tag_len = len(f"```{final_output_format}\n") # Approx
                 potential_summary = final_summary_raw[last_occurrence + start_tag_len:].strip()
                 if potential_summary.endswith("```"):
                      potential_summary = potential_summary[:-3].strip()
                 final_summary = potential_summary if potential_summary else final_summary_raw.strip().rstrip('```')
             else:
                  final_summary = final_summary_raw.strip().rstrip('```')
             ASCIIColors.warning("Could not extract final summary block. Using raw response.")

        if debug:
            ASCIIColors.magenta("--- Final Summary ---")
            ASCIIColors.green(final_summary)
            ASCIIColors.magenta("-------------------")

        return final_summary


    def deep_analyze(
        self,
        query: str,
        text: str = None,
        files: Optional[List[Union[str, Path]]] = None,
        aggregation_prompt: str = "Aggregate the findings from the memory into a coherent answer to the original query.",
        output_format: str = "markdown",
        ctx_size: int = None,
        chunk_size: int = None,
        overlap: int = None, # Added overlap
        bootstrap_chunk_size: int = None,
        bootstrap_steps: int = None,
        callback=None,
        debug: bool = False
    ):
        """
        Searches for information related to a query in long text or files, processing chunk by chunk.
        (Implementation requires self.tokenize, self.detokenize, self.generate_text, self.extract_code_blocks)
        """
        # ... (Implementation mostly similar to previous version, but needs updates) ...
        if not callback:
            callback=self.sink

        # Set defaults and validate input
        if ctx_size is None:
            ctx_size = self.llm.default_ctx_size or 8192
        if chunk_size is None:
            chunk_size = ctx_size // 4
        if overlap is None:
             overlap = chunk_size // 10
        if bootstrap_chunk_size is None:
            bootstrap_chunk_size = chunk_size // 2
        if bootstrap_steps is None:
            bootstrap_steps = 2

        if not text and not files:
            raise ValueError("Either 'text' or 'files' must be provided.")
        if text and files:
             ASCIIColors.warning("Both 'text' and 'files' provided. Processing 'files' only.")
             text = None # Prioritize files if both are given

        # Prepare input texts from files or the single text string
        all_texts = []
        if files:
            from docling import DocumentConverter # Lazy import
            converter = DocumentConverter()
            callback("Loading and converting files...", MSG_TYPE.MSG_TYPE_STEP)
            for i, file_path in enumerate(files):
                 file_p = Path(file_path)
                 callback(f"Processing file {i+1}/{len(files)}: {file_p.name}", MSG_TYPE.MSG_TYPE_STEP_PROGRESS, {"progress":(i+1)/len(files)*100})
                 try:
                     if file_p.exists():
                          file_content_result = converter.convert(file_p)
                          if file_content_result and file_content_result.document:
                              # Exporting to markdown for consistent processing
                              all_texts.append((str(file_path), file_content_result.document.export_to_markdown()))
                          else:
                              ASCIIColors.error(f"Could not convert file: {file_path}")
                     else:
                          ASCIIColors.error(f"File not found: {file_path}")
                 except Exception as e:
                     ASCIIColors.error(f"Error processing file {file_path}: {e}")
                     trace_exception(e)
            callback("File processing complete.", MSG_TYPE.MSG_TYPE_STEP_END)

        elif text:
            all_texts = [("input_text", text)]

        if not all_texts:
             return "Error: No valid text content found to analyze."

        # Initialize memory and counters
        memory = ""
        global_chunk_id = 0

        # Define prompts (can be customized)
        def update_memory_prompt_template(file_name, file_chunk_id, global_chunk_id, chunk, memory, query):
            system_header = self.system_full_header
            user_header = self.user_full_header
            ai_header = self.ai_full_header
            mem_header = "Initial memory template:" if not memory else "Current findings memory (cumulative):"

            return f"""{system_header}
You are a search assistant processing document chunks to find information relevant to a user query. Update the markdown memory with findings from the current chunk.

----
File: {file_name}
Chunk in File: {file_chunk_id}
Global Chunk: {global_chunk_id}
Text Chunk:
```markdown
{chunk}
```
{mem_header}
```markdown
"""+memory or '# Findings\\n## Key Information\\nDetails relevant to the query...\\n## Context\\nSupporting context...'+f"""
```
{user_header}
Query: '{query}'
Task: Update the markdown memory by adding new information from this chunk relevant to the query. Retain prior findings unless contradicted. Only include explicitly relevant details. Return the *entire updated* markdown memory inside a markdown code block.
{ai_header}
```markdown
""" # Start AI response with code block

        # Estimate static prompt size (approximate)
        example_prompt = update_memory_prompt_template("f.txt", 0, 0, "<chunk>", "<memory>", query)
        try:
            static_tokens = len(self.tokenize(example_prompt)) - len(self.tokenize("<chunk>")) - len(self.tokenize("<memory>"))
        except RuntimeError as e:
            ASCIIColors.error(f"Tokenization failed during setup: {e}")
            return "Error: Could not calculate prompt size."

        # Process each text (from file or input)
        callback("Starting deep analysis...", MSG_TYPE.MSG_TYPE_STEP_START)
        for file_path_str, file_text_content in all_texts:
            file_name = Path(file_path_str).name
            callback(f"Analyzing: {file_name}", MSG_TYPE.MSG_TYPE_STEP)
            try:
                file_tokens = self.tokenize(file_text_content)
            except RuntimeError as e:
                 ASCIIColors.error(f"Tokenization failed for {file_name}: {e}")
                 continue # Skip this file

            start_token_idx = 0
            file_chunk_id = 0

            while start_token_idx < len(file_tokens):
                # Calculate available space dynamically
                available_tokens_for_dynamic_content = ctx_size - static_tokens - (self.llm.default_n_predict or 1024)
                if available_tokens_for_dynamic_content <= 100:
                     ASCIIColors.error(f"Context window too small during analysis of {file_name}.")
                     # Option: try truncating memory drastically or break
                     break # Stop processing this file if context is too full

                max_memory_tokens = available_tokens_for_dynamic_content // 2
                max_chunk_tokens = available_tokens_for_dynamic_content - max_memory_tokens

                # Truncate memory if needed
                current_memory_tokens = self.tokenize(memory)
                if len(current_memory_tokens) > max_memory_tokens:
                    memory = self.detokenize(current_memory_tokens[-max_memory_tokens:])
                    if debug: ASCIIColors.yellow(f"Memory truncated (File: {file_name}, Chunk: {file_chunk_id})")

                # Determine chunk size
                current_chunk_size = bootstrap_chunk_size if global_chunk_id < bootstrap_steps else chunk_size
                current_chunk_size = min(current_chunk_size, max_chunk_tokens)

                end_token_idx = min(start_token_idx + current_chunk_size, len(file_tokens))
                chunk_tokens = file_tokens[start_token_idx:end_token_idx]
                chunk = self.detokenize(chunk_tokens)

                file_chunk_id += 1
                global_chunk_id += 1
                callback(f"Processing chunk {file_chunk_id} (Global {global_chunk_id}) of {file_name}", MSG_TYPE.MSG_TYPE_STEP_PROGRESS, {"progress": end_token_idx/len(file_tokens)*100})

                # Generate updated memory
                prompt = update_memory_prompt_template(
                    file_name=file_name,
                    file_chunk_id=file_chunk_id,
                    global_chunk_id=global_chunk_id,
                    chunk=chunk,
                    memory=memory,
                    query=query
                )
                if debug:
                    ASCIIColors.magenta(f"--- Deep Analysis Prompt (Global Chunk {global_chunk_id}) ---")
                    ASCIIColors.cyan(prompt)

                response = self.generate_text(prompt, n_predict=(self.llm.default_n_predict or 1024), streaming_callback=callback) # Use main callback for streaming output

                if isinstance(response, dict): # Handle error
                     ASCIIColors.error(f"Chunk processing failed (Global {global_chunk_id}): {response.get('error')}")
                     start_token_idx = end_token_idx # Skip to next chunk index
                     continue

                memory_code_blocks = self.extract_code_blocks(response, format="markdown")
                if memory_code_blocks:
                     memory = memory_code_blocks[0]["content"]
                else:
                     # Fallback logic (same as sequential_summarize)
                     end_tag = "```markdown"
                     last_occurrence = response.rfind(end_tag)
                     if last_occurrence != -1:
                         start_tag_len = len("```markdown\n")
                         potential_memory = response[last_occurrence + start_tag_len:].strip()
                         if potential_memory.endswith("```"):
                             potential_memory = potential_memory[:-3].strip()
                         memory = potential_memory if potential_memory else response.strip().rstrip('```')
                     else:
                         memory = response.strip().rstrip('```')
                     ASCIIColors.warning(f"Could not extract memory block for chunk {global_chunk_id}. Using raw response.")


                if debug:
                    ASCIIColors.magenta(f"--- Updated Memory (After Global Chunk {global_chunk_id}) ---")
                    ASCIIColors.green(memory)
                    ASCIIColors.magenta("-----------------------------------")

                # Move to next chunk start index with overlap
                start_token_idx = max(start_token_idx, end_token_idx - overlap) if overlap > 0 and end_token_idx < len(file_tokens) else end_token_idx

            callback(f"Finished analyzing: {file_name}", MSG_TYPE.MSG_TYPE_STEP_END)


        # --- Final Aggregation ---
        callback("Aggregating final answer...", MSG_TYPE.MSG_TYPE_STEP_START)
        final_prompt = f"""{self.system_full_header}
You are a search results aggregator.
{self.user_full_header}
{aggregation_prompt}
Collected findings (across all sources):
```markdown
{memory}
```
Provide the final aggregated answer in {output_format} format, directly addressing the original query: '{query}'. The final answer must be put inside a {output_format} markdown tag.
{self.ai_full_header}
```{output_format}
""" # Start AI response

        # Truncate memory if needed for final prompt (similar logic to sequential_summarize)
        final_example_prompt = final_prompt.replace("{memory}", "<final_memory>")
        try:
             final_static_tokens = len(self.tokenize(final_example_prompt)) - len(self.tokenize("<final_memory>"))
             available_final_tokens = ctx_size - final_static_tokens - (self.llm.default_n_predict or 1024)
        except RuntimeError as e:
              ASCIIColors.error(f"Tokenization failed during final setup: {e}")
              return "Error: Could not calculate final prompt size."

        memory_tokens = self.tokenize(memory)
        if len(memory_tokens) > available_final_tokens:
             memory = self.detokenize(memory_tokens[-available_final_tokens:])
             if debug: ASCIIColors.yellow(f"Final memory truncated for aggregation.")

        final_prompt = final_prompt.format(memory=memory) # Format with potentially truncated memory

        if debug:
            ASCIIColors.magenta("--- Final Aggregation Prompt ---")
            ASCIIColors.cyan(final_prompt)

        final_output_raw = self.generate_text(final_prompt, n_predict=(self.llm.default_n_predict or 1024), streaming_callback=callback) # Use main callback

        if isinstance(final_output_raw, dict):
             ASCIIColors.error(f"Final aggregation failed: {final_output_raw.get('error')}")
             callback("Aggregation failed.", MSG_TYPE.MSG_TYPE_STEP_END, {'status':False})
             return "Error: Final aggregation failed."

        final_code_blocks = self.extract_code_blocks(final_output_raw, format=output_format)
        if final_code_blocks:
             final_output = final_code_blocks[0]["content"]
        else:
             # Fallback logic
             end_tag = f"```{output_format}"
             last_occurrence = final_output_raw.rfind(end_tag)
             if last_occurrence != -1:
                 start_tag_len = len(f"```{output_format}\n")
                 potential_output = final_output_raw[last_occurrence + start_tag_len:].strip()
                 if potential_output.endswith("```"):
                     potential_output = potential_output[:-3].strip()
                 final_output = potential_output if potential_output else final_output_raw.strip().rstrip('```')
             else:
                 final_output = final_output_raw.strip().rstrip('```')
             ASCIIColors.warning("Could not extract final output block. Using raw response.")


        if debug:
            ASCIIColors.magenta("--- Final Aggregated Output ---")
            ASCIIColors.green(final_output)
            ASCIIColors.magenta("-----------------------------")

        callback("Deep analysis complete.", MSG_TYPE.MSG_TYPE_STEP_END)
        return final_output

    def long_context_processing(
        self,
        text_to_process: str,
        contextual_prompt: Optional[str] = None,
        chunk_size_tokens: int|None = None,
        overlap_tokens: int = 0,
        streaming_callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Summarizes a long text that may not fit into the model's context window.

        This method works in two stages:
        1.  **Chunk & Summarize:** It breaks the text into overlapping chunks and summarizes each one individually.
        2.  **Synthesize:** It then takes all the chunk summaries and performs a final summarization pass to create a single, coherent, and comprehensive summary.

        Args:
            text_to_process (str): The long text content to be summarized.
            contextual_prompt (Optional[str], optional): A specific instruction to guide the summary's focus. 
                                                       For example, "Summarize the text focusing on the financial implications."
                                                       Defaults to None.
            chunk_size_tokens (int, optional): The number of tokens in each text chunk. This should be well
                                             within the model's context limit to allow space for prompts.
                                             Defaults to 1500.
            overlap_tokens (int, optional): The number of tokens to overlap between chunks to ensure context
                                          is not lost at the boundaries. Defaults to 250.
            streaming_callback (Optional[Callable], optional): A callback function to receive real-time updates
                                                             on the process (e.g., which chunk is being processed).
                                                             It receives a message, a message type, and optional metadata.
                                                             Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the generation method (e.g., temperature, top_p).

        Returns:
            str: The final, comprehensive summary of the text.
        """
        if not text_to_process and len(kwargs.get("images",[]))==0:
            return ""
        if not text_to_process:
            text_to_process=""
            tokens = []
        else:
            # Use the binding's tokenizer for accurate chunking
            tokens = self.llm.tokenize(text_to_process)
        if chunk_size_tokens is None:
            chunk_size_tokens = self.llm.default_ctx_size//2
        
        if len(tokens) <= chunk_size_tokens:
            if streaming_callback:
                streaming_callback("Text is short enough for a single pass.", MSG_TYPE.MSG_TYPE_STEP, {"progress": 0})
            system_prompt = ("You are a content processor expert.\n"
                            "You perform tasks on the content as requested by the user.\n\n"
                            "--- Content ---\n"
                            f"{text_to_process}\n\n"
                            "** Important **\n"
                            "Strictly adhere to the user prompt.\n"
                            "Do not add comments unless asked to do so.\n"
                            )
            if "system_prompt" in kwargs:
                system_prompt += "-- Extra instructions --\n"+ kwargs["system_prompt"] +"\n"
                del kwargs["system_prompt"]            
            prompt_objective = contextual_prompt or "Provide a comprehensive summary of the content."
            final_prompt = f"{prompt_objective}"
            
            processed_output = self.generate_text(final_prompt, system_prompt=system_prompt, **kwargs)
            
            if streaming_callback:
                streaming_callback("Content processed.", MSG_TYPE.MSG_TYPE_STEP, {"progress": 100})
            
            return processed_output

        # --- Stage 1: Chunking and Independent Summarization ---
        chunks = []
        step = chunk_size_tokens - overlap_tokens
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size_tokens]
            chunk_text = self.llm.detokenize(chunk_tokens)
            chunks.append(chunk_text)

        chunk_summaries = []
        
        # Total steps include each chunk plus the final synthesis step
        total_steps = len(chunks) + 1
        
        # Define the prompt for summarizing each chunk
        summarization_objective = contextual_prompt or "Summarize the key points of the following text excerpt."
        system_prompt = ("You are a sequential document processing agent.\n"
                         "The process is done in two phases:\n"
                         "** Phase1 : **\n"
                         "Sequencially extracting information from the text chunks and adding them to the scratchpad.\n"
                         "** Phase2: **\n"
                         "Synthesizing a comprehensive Response using the scratchpad content given the objective formatting instructions if applicable.\n"
                         "We are now performing ** Phase 1 **, and we are processing chunk number {{chunk_id}}.\n"
                         "Your job is to extract information from the current chunk given previous chunks extracted information placed in scratchpad as well as the current chunk content.\n"
                         "Add the information to the scratchpad while strictly adhering to the Global objective extraction instructions:\n"
                         "-- Sequencial Scratchpad --\n"
                         "{{scratchpad}}\n"
                         "** Important **\n"
                         "Respond only with the extracted information from the current chunk without repeating things that are already in the scratchpad.\n"
                         "Strictly adhere to the Global objective content for the extraction phase.\n"
                         "Do not add comments.\n"
                        )
        if "system_prompt" in kwargs:
            system_prompt += "-- Extra instructions --\n"+ kwargs["system_prompt"] +"\n"
            del kwargs["system_prompt"]
        chunk_summary_prompt_template = f"--- Global objective ---\n{summarization_objective}\n\n--- Text Excerpt ---\n{{chunk_text}}"

        for i, chunk in enumerate(chunks):
            progress_before = (i / total_steps) * 100
            if streaming_callback:
                streaming_callback(
                    f"Processing chunk {i + 1} of {len(chunks)}...", 
                    MSG_TYPE.MSG_TYPE_STEP_START, 
                    {"id": f"chunk_{i+1}", "progress": progress_before}
                )

            prompt = chunk_summary_prompt_template.format(chunk_text=chunk)
            processed_system_prompt = system_prompt.format(chunk_id=i,scratchpad="\n\n---\n\n".join(chunk_summaries))
            try:
                # Generate summary for the current chunk
                chunk_summary = self.generate_text(prompt, system_prompt=processed_system_prompt, **kwargs)
                chunk_summaries.append(chunk_summary)
                
                progress_after = ((i + 1) / total_steps) * 100
                if streaming_callback:
                    streaming_callback(
                        f"Chunk {i + 1} processed. Progress: {progress_after:.0f}%", 
                        MSG_TYPE.MSG_TYPE_STEP_END, 
                        {"id": f"chunk_{i+1}", "output_snippet": chunk_summary[:100], "progress": progress_after}
                    )
            except Exception as e:
                trace_exception(e)
                if streaming_callback:
                    streaming_callback(f"Failed to process chunk {i+1}: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION)
                # Still add a placeholder to not break the chain
                chunk_summaries.append(f"[Error processing chunk {i+1}]")

        # --- Stage 2: Final Synthesis of All Chunk Summaries ---
        progress_before_synthesis = (len(chunks) / total_steps) * 100
        if streaming_callback:
            streaming_callback(
                "Processing the scratchpad content into a final version...", 
                MSG_TYPE.MSG_TYPE_STEP_START, 
                {"id": "final_synthesis", "progress": progress_before_synthesis}
            )

        combined_summaries = "\n\n---\n\n".join(chunk_summaries)
        
        # Define the prompt for the final synthesis
        synthesis_objective = contextual_prompt or "Create a single, final, coherent, and comprehensive summary."
        system_prompt = ("You are a sequential document processing agent.\n"
                         "The process is done in two phases:\n"
                         "** Phase1 : **\n"
                         "Sequencially extracting information from the text chunks and adding them to the scratchpad.\n"
                         "** Phase2: **\n"
                         "Synthesizing a comprehensive Response using the scratchpad content given the objective formatting instructions if applicable.\n"
                         "\n"
                         "We are now performing ** Phase 2 **.\n"
                         "Your job is to use the extracted information to fulfill the user prompt objectives.\n"
                         "Make sure you respect the user formatting if provided and if not, then use markdown output format."
                         "-- Sequencial Scratchpad --\n"
                         f"{combined_summaries}\n"
                         "** Important **\n"
                         "Respond only with the requested task without extra comments unless told to.\n"
                         "Strictly adhere to the Global objective content for the extraction phase.\n"
                         "Do not add comments.\n"
                        )
        final_synthesis_prompt = (
            f"--- Global objective ---\n{synthesis_objective}\n\n"
            "--- Final Response ---"
        )

        final_answer = self.generate_text(final_synthesis_prompt, system_prompt=system_prompt, **kwargs)
        
        if streaming_callback:
            streaming_callback(
                "Final summary synthesized.", 
                MSG_TYPE.MSG_TYPE_STEP_END, 
                {"id": "final_synthesis", "progress": 100}
            )

        return final_answer.strip()

def chunk_text(text, tokenizer, detokenizer, chunk_size, overlap, use_separators=True):
    """
    Chunks text based on token count.

    Args:
        text (str): The text to chunk.
        tokenizer (callable): Function to tokenize text.
        detokenizer (callable): Function to detokenize tokens.
        chunk_size (int): The desired number of tokens per chunk.
        overlap (int): The number of tokens to overlap between chunks.
        use_separators (bool): If True, tries to chunk at natural separators (paragraphs, sentences).

    Returns:
        List[str]: A list of text chunks.
    """
    tokens = tokenizer(text)
    chunks = []
    start_idx = 0

    if not use_separators:
        while start_idx < len(tokens):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunks.append(detokenizer(tokens[start_idx:end_idx]))
            start_idx += chunk_size - overlap
            if start_idx >= len(tokens): # Ensure last chunk is added correctly
                 break
            start_idx = max(0, start_idx) # Prevent negative index
    else:
        # Find potential separator positions (more robust implementation needed)
        # This is a basic example using paragraphs first, then sentences.
        import re
        separators = ["\n\n", "\n", ". ", "? ", "! "] # Order matters
        
        current_pos = 0
        while current_pos < len(text):
            # Determine target end position based on tokens
            target_end_token = min(start_idx + chunk_size, len(tokens))
            target_end_char_approx = len(detokenizer(tokens[:target_end_token])) # Approximate char position

            best_sep_pos = -1
            # Try finding a good separator near the target end
            for sep in separators:
                # Search backwards from the approximate character position
                search_start = max(current_pos, target_end_char_approx - chunk_size // 2) # Search in a reasonable window
                sep_pos = text.rfind(sep, search_start, target_end_char_approx + len(sep))
                if sep_pos > current_pos: # Found a separator after the current start
                    best_sep_pos = max(best_sep_pos, sep_pos + len(sep)) # Take the latest separator found

            # If no good separator found, just cut at token limit
            if best_sep_pos == -1 or best_sep_pos <= current_pos:
                end_idx = target_end_token
                end_char = len(detokenizer(tokens[:end_idx])) if end_idx < len(tokens) else len(text)
            else:
                end_char = best_sep_pos
                end_idx = len(tokenizer(text[:end_char])) # Re-tokenize to find token index


            chunk_text_str = text[current_pos:end_char]
            chunks.append(chunk_text_str)

            # Move to next chunk start, considering overlap in characters
            overlap_char_approx = len(detokenizer(tokens[:overlap])) # Approx overlap chars
            next_start_char = max(current_pos, end_char - overlap_char_approx)

            # Try to align next start with a separator too for cleaner breaks
            best_next_start_sep = next_start_char
            for sep in separators:
                 sep_pos = text.find(sep, next_start_char)
                 if sep_pos != -1:
                     best_next_start_sep = min(best_next_start_sep, sep_pos+len(sep)) if best_next_start_sep!=next_start_char else sep_pos+len(sep) # Find earliest separator after overlap point

            current_pos = best_next_start_sep if best_next_start_sep > next_start_char else next_start_char
            start_idx = len(tokenizer(text[:current_pos])) # Update token index for next iteration


            if current_pos >= len(text):
                break

    return chunks
