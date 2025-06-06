# lollms_client/lollms_core.py
import requests
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import encode_image # Keep utilities needed by core
from lollms_client.lollms_llm_binding import LollmsLLMBinding, LollmsLLMBindingManager
# Import new Abstract Base Classes and Managers
from lollms_client.lollms_tts_binding import LollmsTTSBinding, LollmsTTSBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBinding, LollmsTTIBindingManager
from lollms_client.lollms_stt_binding import LollmsSTTBinding, LollmsSTTBindingManager
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, LollmsTTVBindingManager
from lollms_client.lollms_ttm_binding import LollmsTTMBinding, LollmsTTMBindingManager
from lollms_client.lollms_mcp_binding import LollmsMCPBinding, LollmsMCPBindingManager

import json, re
from enum import Enum
import base64
import requests
from typing import List, Optional, Callable, Union, Dict, Any
import numpy as np
from pathlib import Path
import os

class LollmsClient():
    """
    Core client class for interacting with LOLLMS services, including LLM, TTS, TTI, STT, TTV, and TTM.
    Provides a unified interface to manage and use different bindings for various modalities.
    """
    def __init__(self,
                 # LLM Binding Parameters
                 binding_name: str = "lollms",
                 host_address: Optional[str] = None, # Shared host address (for service based bindings) default for all bindings if not specified
                 models_path: Optional[str] = None, # Shared models folder path (for local file based bindings) default for all bindings if not specified
                 model_name: str = "",
                 llm_bindings_dir: Path = Path(__file__).parent / "llm_bindings",
                 llm_binding_config: Optional[Dict[str, any]] = None, 

                 # Optional Modality Binding Names
                 tts_binding_name: Optional[str] = None,
                 tti_binding_name: Optional[str] = None,
                 stt_binding_name: Optional[str] = None,
                 ttv_binding_name: Optional[str] = None,
                 ttm_binding_name: Optional[str] = None,
                 mcp_binding_name: Optional[str] = None,

                 # Modality Binding Directories
                 tts_bindings_dir: Path = Path(__file__).parent / "tts_bindings",
                 tti_bindings_dir: Path = Path(__file__).parent / "tti_bindings",
                 stt_bindings_dir: Path = Path(__file__).parent / "stt_bindings",
                 ttv_bindings_dir: Path = Path(__file__).parent / "ttv_bindings",
                 ttm_bindings_dir: Path = Path(__file__).parent / "ttm_bindings",
                 mcp_bindings_dir: Path = Path(__file__).parent / "mcp_bindings",

                 # Configurations
                 tts_binding_config: Optional[Dict[str, any]] = None, 
                 tti_binding_config: Optional[Dict[str, any]] = None, 
                 stt_binding_config: Optional[Dict[str, any]] = None, 
                 ttv_binding_config: Optional[Dict[str, any]] = None, 
                 ttm_binding_config: Optional[Dict[str, any]] = None, 
                 mcp_binding_config: Optional[Dict[str, any]] = None,

                 # General Parameters (mostly defaults for LLM generation)
                 service_key: Optional[str] = None, # Shared service key/client_id
                 verify_ssl_certificate: bool = True,
                 ctx_size: Optional[int] = 8192,
                 n_predict: Optional[int] = 4096,
                 stream: bool = False,
                 temperature: float = 0.7, # Ollama default is 0.8, common default 0.7
                 top_k: int = 40,          # Ollama default is 40
                 top_p: float = 0.9,       # Ollama default is 0.9
                 repeat_penalty: float = 1.1, # Ollama default is 1.1
                 repeat_last_n: int = 64,  # Ollama default is 64

                 seed: Optional[int] = None,
                 n_threads: int = 8,
                 streaming_callback: Optional[Callable[[str, MSG_TYPE], None]] = None,
                 user_name ="user",
                 ai_name = "assistant"):
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
        self.host_address = host_address # Store initial preference
        self.models_path = models_path
        self.service_key = service_key
        self.verify_ssl_certificate = verify_ssl_certificate

        # --- LLM Binding Setup ---
        self.binding_manager = LollmsLLMBindingManager(llm_bindings_dir)
        self.binding = self.binding_manager.create_binding(
            binding_name=binding_name,
            host_address=host_address, # Pass initial host preference
            models_path=models_path,
            model_name=model_name,
            service_key=service_key,
            verify_ssl_certificate=verify_ssl_certificate,
            # Pass LLM specific config if needed
            **(llm_binding_config or {})
        )

        if self.binding is None:
            available = self.binding_manager.get_available_bindings()
            raise ValueError(f"Failed to create LLM binding: {binding_name}. Available: {available}")

        # Determine the effective host address (use LLM binding's if initial was None)
        effective_host_address = self.host_address

        # --- Modality Binding Setup ---
        self.tts_binding_manager = LollmsTTSBindingManager(tts_bindings_dir)
        self.tti_binding_manager = LollmsTTIBindingManager(tti_bindings_dir)
        self.stt_binding_manager = LollmsSTTBindingManager(stt_bindings_dir)
        self.ttv_binding_manager = LollmsTTVBindingManager(ttv_bindings_dir)
        self.ttm_binding_manager = LollmsTTMBindingManager(ttm_bindings_dir)
        self.mcp_binding_manager = LollmsMCPBindingManager(mcp_bindings_dir)

        self.tts: Optional[LollmsTTSBinding] = None
        self.tti: Optional[LollmsTTIBinding] = None
        self.stt: Optional[LollmsSTTBinding] = None
        self.ttv: Optional[LollmsTTVBinding] = None
        self.ttm: Optional[LollmsTTMBinding] = None
        self.mcp: Optional[LollmsMCPBinding] = None

        if tts_binding_name:
            self.tts = self.tts_binding_manager.create_binding(
                binding_name=tts_binding_name,
                **tts_binding_config
            )
            if self.tts is None:
                ASCIIColors.warning(f"Failed to create TTS binding: {tts_binding_name}. Available: {self.tts_binding_manager.get_available_bindings()}")

        if tti_binding_name:
            if tti_binding_config:
                self.tti = self.tti_binding_manager.create_binding(
                    binding_name=tti_binding_name,
                    **tti_binding_config
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
                    **stt_binding_config
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
                    **ttv_binding_config
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
                    **ttm_binding_config
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
                    mcp_binding_name,
                    **mcp_binding_config
                )
            else:
                self.mcp = self.mcp_binding_manager.create_binding(
                    mcp_binding_name
                )
            if self.mcp is None:
                ASCIIColors.warning(f"Failed to create MCP binding: {mcp_binding_name}. Available: {self.mcp_binding_manager.get_available_bindings()}")

        # --- Store Default Generation Parameters ---
        self.default_ctx_size = ctx_size
        self.default_n_predict = n_predict
        self.default_stream = stream
        self.default_temperature = temperature
        self.default_top_k = top_k
        self.default_top_p = top_p
        self.default_repeat_penalty = repeat_penalty
        self.default_repeat_last_n = repeat_last_n
        self.default_seed = seed
        self.default_n_threads = n_threads
        self.default_streaming_callback = streaming_callback

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
        if self.binding:
            return self.binding.tokenize(text)
        raise RuntimeError("LLM binding not initialized.")

    def detokenize(self, tokens: list) -> str:
        """
        Detokenize tokens using the active LLM binding.

        Args:
            tokens (list): List of tokens to detokenize.

        Returns:
            str: Detokenized text.
        """
        if self.binding:
            return self.binding.detokenize(tokens)
        raise RuntimeError("LLM binding not initialized.")
    def count_tokens(self, text: str) -> int:
        """
        Counts how many tokens are there in the text using the active LLM binding.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: Number of tokens.
        """
        if self.binding:
            return self.binding.count_tokens(text)
        raise RuntimeError("LLM binding not initialized.")
    
    def get_model_details(self) -> dict:
        """
        Get model information from the active LLM binding.

        Returns:
            dict: Model information dictionary.
        """
        if self.binding:
            return self.binding.get_model_info()
        raise RuntimeError("LLM binding not initialized.")

    def switch_model(self, model_name: str) -> bool:
        """
        Load a new model in the active LLM binding.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        if self.binding:
            return self.binding.load_model(model_name)
        raise RuntimeError("LLM binding not initialized.")

    def get_available_llm_bindings(self) -> List[str]: 
        """
        Get list of available LLM binding names.

        Returns:
            List[str]: List of binding names that can be used for LLMs.
        """
        return self.binding_manager.get_available_bindings()

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
            streaming_callback (Optional[Callable[[str, MSG_TYPE], None]]): Callback for streaming output.
            split:Optional[bool]: put to true if the prompt is a discussion
            user_keyword:Optional[str]: when splitting we use this to extract user prompt 
            ai_keyword:Optional[str]": when splitting we use this to extract ai prompt

        Returns:
            Union[str, dict]: Generated text or error dictionary if failed.
        """
        if self.binding:
            return self.binding.generate_text(
                prompt=prompt,
                images=images,
                system_prompt=system_prompt,
                n_predict=n_predict if n_predict is not None else self.default_n_predict,
                stream=stream if stream is not None else self.default_stream,
                temperature=temperature if temperature is not None else self.default_temperature,
                top_k=top_k if top_k is not None else self.default_top_k,
                top_p=top_p if top_p is not None else self.default_top_p,
                repeat_penalty=repeat_penalty if repeat_penalty is not None else self.default_repeat_penalty,
                repeat_last_n=repeat_last_n if repeat_last_n is not None else self.default_repeat_last_n,
                seed=seed if seed is not None else self.default_seed,
                n_threads=n_threads if n_threads is not None else self.default_n_threads,
                ctx_size = ctx_size if ctx_size is not None else self.default_ctx_size,
                streaming_callback=streaming_callback if streaming_callback is not None else self.default_streaming_callback,
                split= split,
                user_keyword=user_keyword,
                ai_keyword=ai_keyword
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
        if self.binding:
            return self.binding.embed(text, **kwargs)
        raise RuntimeError("LLM binding not initialized.")


    def listModels(self):
        """Lists models available to the current LLM binding."""
        if self.binding:
            return self.binding.listModels()
        raise RuntimeError("LLM binding not initialized.")

    # --- Convenience Methods for Lollms LLM Binding Features ---
    def listMountedPersonalities(self) -> Union[List[Dict], Dict]:
        """
        Lists mounted personalities *if* the active LLM binding is 'lollms'.

        Returns:
            Union[List[Dict], Dict]: List of personality dicts or error dict.
        """
        if self.binding and hasattr(self.binding, 'lollms_listMountedPersonalities'):
            return self.binding.lollms_listMountedPersonalities()
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
                        max_size = None,
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
            n_predict=max_size,
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

    # --- Function Calling with MCP ---
    def generate_with_mcp(
        self,
        prompt: str,
        discussion_history: Optional[List[Dict[str, str]]] = None, # e.g. [{"role":"user", "content":"..."}, {"role":"assistant", "content":"..."}]
        images: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None, # List of MCP tool definitions
        max_tool_calls: int = 5,
        max_llm_iterations: int = 10, # Safety break for LLM deciding to call tools repeatedly
        tool_call_decision_temperature: float = 0.1, # Lower temp for more deterministic decision making
        final_answer_temperature: float = None, # Use instance default if None
        streaming_callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict], Optional[List]], bool]] = None,
        interactive_tool_execution: bool = False, # If true, prompts user before executing a tool
        **llm_generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generates a response that may involve calling one or more tools via MCP.

        Args:
            prompt (str): The user's initial prompt.
            discussion_history (Optional[List[Dict[str, str]]]): Previous turns of conversation.
            images (Optional[List[str]]): Images provided with the current user prompt.
            tools (Optional[List[Dict[str, Any]]]): A list of MCP tool definitions available for this call.
                                                     If None, tools will be discovered from the MCP binding.
            max_tool_calls (int): Maximum number of distinct tool calls allowed in one interaction turn.
            max_llm_iterations (int): Maximum number of times the LLM can decide to call a tool
                                      before being forced to generate a final answer.
            tool_call_decision_temperature (float): Temperature for LLM when deciding on tool calls.
            final_answer_temperature (float): Temperature for LLM when generating the final answer.
            streaming_callback (Optional[Callable]): Callback for streaming LLM responses (tool decisions/final answer).
                                                     Signature: (chunk_str, msg_type, metadata_dict, history_list_of_dicts_for_this_turn) -> bool
            interactive_tool_execution (bool): If True, ask user for confirmation before executing each tool.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "final_answer" (str): The LLM's final textual answer.
                - "tool_calls" (List[Dict]): A list of tools called, their params, and results.
                - "error" (Optional[str]): Error message if something went wrong.
        """
        if not self.binding:
            return {"final_answer": "", "tool_calls": [], "error": "LLM binding not initialized."}
        if not self.mcp:
            return {"final_answer": "", "tool_calls": [], "error": "MCP binding not initialized."}

        turn_history: List[Dict[str, Any]] = [] # Tracks this specific turn's interactions (LLM thoughts, tool calls, tool results)
        
        # 1. Discover tools if not provided
        if tools is None:
            try:
                tools = self.mcp.discover_tools()
                if not tools:
                    ASCIIColors.warning("No MCP tools discovered by the binding.")
            except Exception as e_disc:
                return {"final_answer": "", "tool_calls": [], "error": f"Failed to discover MCP tools: {e_disc}"}
        
        if not tools: # If still no tools after discovery attempt
            ASCIIColors.info("No tools available for function calling. Generating direct response.")
            final_answer = self.remove_thinking_blocks(self.generate_text(
                prompt=prompt,
                system_prompt= (discussion_history[0]['content'] if discussion_history and discussion_history[0]['role'] == 'system' else "") + "\nYou are a helpful assistant.", # Basic system prompt
                images=images,
                stream=streaming_callback is not None, # stream if callback is provided
                streaming_callback=lambda chunk, msg_type: streaming_callback(chunk, msg_type, None, turn_history) if streaming_callback else None, # Adapt callback
                temperature=final_answer_temperature if final_answer_temperature is not None else self.default_temperature,
                **(llm_generation_kwargs or {})
            ))
            if isinstance(final_answer, dict) and "error" in final_answer: # Handle generation error
                return {"final_answer": "", "tool_calls": [], "error": final_answer["error"]}
            return {"final_answer": final_answer, "tool_calls": [], "error": None}


        formatted_tools_list = "\n".join([
            f"- Name: {t.get('name')}\n  Description: {t.get('description')}\n  Input Schema: {json.dumps(t.get('input_schema'))}"
            for t in tools
        ])

        current_conversation: List[Dict[str, str]] = []
        if discussion_history:
            current_conversation.extend(discussion_history)
        current_conversation.append({"role": "user", "content": prompt})
        if images: # Add image representations to the last user message if supported by LLM and chat format
            # This part is highly dependent on how the specific LLM binding handles images in chat.
            # For simplicity, we'll assume if images are passed, the underlying generate_text handles it.
            # A more robust solution would modify current_conversation[-1]['content'] structure.
            ASCIIColors.info("Images provided. Ensure LLM binding's generate_text handles them with chat history.")


        tool_calls_made_this_turn = []
        llm_iterations = 0

        while llm_iterations < max_llm_iterations:
            llm_iterations += 1
            
            # 2. Construct prompt for LLM to decide on tool call or direct answer
            # We need to convert current_conversation into a single string prompt for `generate_code`
            # or adapt `generate_code` to take a message list if underlying LLM supports chat for structured output.
            # For now, let's assume `generate_code` takes a flat prompt.
            
            # Create a string representation of the conversation history
            history_str = ""
            for msg in current_conversation:
                role_prefix = self.user_custom_header(msg["role"]) if msg["role"]=="user" else self.ai_custom_header(msg["role"]) if msg["role"]=="assistant" else self.system_custom_header(msg["role"]) if msg["role"]=="system" else "!@>unknown:"
                history_str += f"{role_prefix}{msg['content']}\n"
            
            # Add tool execution results from previous iterations in this turn to the history string
            for tc_info in tool_calls_made_this_turn:
                if tc_info.get("result"): # Only add if there's a result (successful or error)
                    history_str += f"{self.ai_full_header}(Executed tool '{tc_info['name']}' with params {tc_info['params']}. Result: {json.dumps(tc_info['result'])})\n"


            decision_prompt_template = f"""You are an AI assistant that can use tools to answer user requests.
Available tools:
{formatted_tools_list}

Current conversation:
{history_str}

Based on the available tools and the current conversation, decide the next step.
Respond with a JSON object containing ONE of the following structures:
1. If you need to use a tool:
   {{"action": "call_tool", "tool_name": "<name_of_tool_to_call>", "tool_params": {{<parameters_for_tool_as_json_object>}}}}
2. If you can answer directly without using a tool OR if you have sufficient information from previous tool calls:
   {{"action": "final_answer"}}
3. If the user's request is unclear or you need more information before deciding:
   {{"action": "clarify", "clarification_request": "<your_question_to_the_user>"}}
""" # No {self.ai_full_header} here, generate_code will get raw JSON

            if streaming_callback:
                streaming_callback(f"LLM deciding next step (iteration {llm_iterations})...", MSG_TYPE.MSG_TYPE_STEP_START, {"type": "decision_making"}, turn_history)

            # Use generate_code to get structured JSON output from LLM
            # Note: generate_code itself uses generate_text. We are asking for JSON here.
            raw_llm_decision_json = self.generate_text(
                prompt=decision_prompt_template, # This is the full prompt for the LLM
                n_predict=512, # Reasonable size for decision JSON
                temperature=tool_call_decision_temperature,
                images=images
                # `images` are part of the history_str if relevant to the binding
                # streaming_callback=None, # Decisions are usually not streamed chunk by chunk
            )
            if streaming_callback:
                streaming_callback(f"LLM decision received.", MSG_TYPE.MSG_TYPE_STEP_END, {"type": "decision_making"}, turn_history)


            if not raw_llm_decision_json:
                ASCIIColors.error("LLM failed to provide a decision JSON.")
                turn_history.append({"type": "error", "content": "LLM failed to provide a decision."})
                return {"final_answer": "I'm sorry, I encountered an issue trying to process your request.", "tool_calls": tool_calls_made_this_turn, "error": "LLM decision JSON was empty."}
            
            processed_raw_json = raw_llm_decision_json.strip() # Strip whitespace first
            try:
                llm_decision = json.loads(processed_raw_json)
                turn_history.append({"type": "llm_decision", "content": llm_decision})
            except json.JSONDecodeError:
                ASCIIColors.error(f"Failed to parse LLM decision JSON: {raw_llm_decision_json}")
                try:
                    decoder = json.JSONDecoder()
                    # Try to decode the first JSON object from the (stripped) string
                    llm_decision, end_index = decoder.raw_decode(processed_raw_json)
                    turn_history.append({"type": "llm_decision_extracted", "content": llm_decision, "raw_trimmed": processed_raw_json[:end_index]})
                    
                    remaining_text = processed_raw_json[end_index:].strip()
                    if remaining_text:
                        ASCIIColors.warning(f"LLM output contained additional text after the first JSON object: '{remaining_text}'. Processing only the first object.")
                        turn_history.append({"type": "llm_extra_output_ignored", "content": remaining_text})
                except json.JSONDecodeError as e_inner:
                    ASCIIColors.error(f"Failed to parse LLM decision JSON even after attempting to extract first object: {raw_llm_decision_json}. Error: {e_inner}")
                    turn_history.append({"type": "error", "content": "Failed to parse LLM decision JSON.", "raw_json": raw_llm_decision_json, "error_details": str(e_inner)})
                    # Provide a generic error message, as the LLM's output was malformed.
                    # Adding the raw output or a snippet to the conversation history might help the LLM recover or inform the user.
                    current_conversation.append({
                        "role": "assistant", 
                        "content": "(I encountered an internal error trying to understand my next step. I will try to answer directly based on what I have so far.)"
                    })
                    break # Break to generate final answer with current info

            if llm_decision is None: # If parsing failed and couldn't recover
                return {"final_answer": "I'm sorry, I had trouble understanding the next step due to a formatting issue.", "tool_calls": tool_calls_made_this_turn, "error": "Invalid JSON from LLM for decision."}

            action = llm_decision.get("action")

            if action == "call_tool":
                if len(tool_calls_made_this_turn) >= max_tool_calls:
                    ASCIIColors.warning("Maximum tool calls reached for this turn. Forcing final answer.")
                    current_conversation.append({"role":"assistant", "content":"(Max tool calls reached. I will now try to formulate an answer based on available information.)"})
                    break # Exit loop to generate final answer

                tool_name = llm_decision.get("tool_name")
                tool_params = llm_decision.get("tool_params", {})

                if not tool_name:
                    ASCIIColors.warning("LLM decided to call a tool but didn't specify tool_name.")
                    current_conversation.append({"role":"assistant", "content":"(I decided to use a tool, but I'm unsure which one. Could you clarify?)"})
                    break # Or ask LLM to try again without this faulty decision in history

                tool_call_info = {"type": "tool_call_request", "name": tool_name, "params": tool_params}
                turn_history.append(tool_call_info)
                if streaming_callback:
                     streaming_callback(f"LLM requests to call tool: {tool_name} with params: {tool_params}", MSG_TYPE.MSG_TYPE_INFO, tool_call_info, turn_history)
                
                # Interactive execution if enabled
                if interactive_tool_execution:
                    try:
                        user_confirmation = input(f"AI wants to execute tool '{tool_name}' with params {tool_params}. Allow? (yes/no/details): ").lower()
                        if user_confirmation == "details":
                            tool_def_for_details = next((t for t in tools if t.get("name") == tool_name), None)
                            print(f"Tool details: {json.dumps(tool_def_for_details, indent=2)}")
                            user_confirmation = input(f"Allow execution of '{tool_name}'? (yes/no): ").lower()
                        
                        if user_confirmation != "yes":
                            ASCIIColors.info("Tool execution cancelled by user.")
                            tool_result = {"error": "Tool execution cancelled by user."}
                            # Add this info to conversation for LLM
                            current_conversation.append({"role": "assistant", "content": f"(Tool '{tool_name}' execution was cancelled by the user. What should I do next?)"})
                            tool_call_info["result"] = tool_result # Record cancellation
                            tool_calls_made_this_turn.append(tool_call_info)
                            continue # Back to LLM for next decision
                    except Exception as e_input: # Catch issues with input() e.g. in non-interactive env
                        ASCIIColors.warning(f"Error during interactive confirmation: {e_input}. Proceeding without confirmation.")


                if streaming_callback:
                    streaming_callback(f"Executing tool: {tool_name}...", MSG_TYPE.MSG_TYPE_STEP_START, {"type": "tool_execution", "tool_name": tool_name}, turn_history)
                
                tool_result = self.mcp.execute_tool(tool_name, tool_params, lollms_client_instance=self)
                
                tool_call_info["result"] = tool_result # Add result to this call's info
                tool_calls_made_this_turn.append(tool_call_info) # Log the completed call

                if streaming_callback:
                    streaming_callback(f"Tool {tool_name} execution finished. Result: {json.dumps(tool_result)}", MSG_TYPE.MSG_TYPE_STEP_END, {"type": "tool_execution", "tool_name": tool_name, "result": tool_result}, turn_history)

                # Add tool execution result to conversation for the LLM
                # The format of this message can influence how the LLM uses the tool output.
                # current_conversation.append({"role": "tool_result", "tool_name": tool_name, "content": json.dumps(tool_result)}) # More structured
                current_conversation.append({"role": "assistant", "content": f"(Tool '{tool_name}' executed. Result: {json.dumps(tool_result)})"})


            elif action == "clarify":
                clarification_request = llm_decision.get("clarification_request", "I need more information. Could you please clarify?")
                if streaming_callback:
                    streaming_callback(clarification_request, MSG_TYPE.MSG_TYPE_FULL, {"type": "clarification_request"}, turn_history)
                turn_history.append({"type":"clarification_request_sent", "content": clarification_request})
                return {"final_answer": clarification_request, "tool_calls": tool_calls_made_this_turn, "error": None}

            elif action == "final_answer":
                ASCIIColors.info("LLM decided to formulate a final answer.")
                current_conversation.append({"role":"assistant", "content":"(I will now formulate the final answer based on the information gathered.)"}) # Inform LLM's "thought process"
                break # Exit loop to generate final answer
            
            else:
                ASCIIColors.warning(f"LLM returned unknown action: {action}")
                current_conversation.append({"role":"assistant", "content":f"(Received an unexpected decision: {action}. I will try to answer directly.)"})
                break # Exit loop
            
            # Safety break if too many iterations without reaching final answer or max_tool_calls
            if llm_iterations >= max_llm_iterations:
                ASCIIColors.warning("Max LLM iterations reached. Forcing final answer.")
                current_conversation.append({"role":"assistant", "content":"(Max iterations reached. I will now try to formulate an answer.)"})
                break
        
        # 3. Generate final answer if LLM decided to, or if loop broke
        if streaming_callback:
            streaming_callback("LLM generating final answer...", MSG_TYPE.MSG_TYPE_STEP_START, {"type": "final_answer_generation"}, turn_history)

        # Construct the final prompt string for generate_text from current_conversation
        final_prompt_str = ""
        final_system_prompt = ""
        
        # Consolidate system messages if any
        interim_history_for_final_answer = []
        for msg in current_conversation:
            if msg["role"] == "system":
                final_system_prompt += msg["content"] + "\n"
            else:
                interim_history_for_final_answer.append(msg)
        
        if not any(msg['role'] == 'user' for msg in interim_history_for_final_answer): # Ensure there's a user turn if only system + tool calls
            interim_history_for_final_answer.append({'role':'user', 'content': prompt}) # Add original prompt if lost


        # The generate_text method needs a single prompt and an optional system_prompt.
        # We need to format the interim_history_for_final_answer into a single prompt string,
        # or modify generate_text to accept a list of messages.
        # For now, flatten to string:
        current_prompt_for_final_answer = ""
        for i, msg in enumerate(interim_history_for_final_answer):
            role_prefix = self.user_custom_header(msg["role"]) if msg["role"]=="user" else self.ai_custom_header(msg["role"]) if msg["role"]=="assistant" else f"!@>{msg['role']}:"
            current_prompt_for_final_answer += f"{role_prefix}{msg['content']}"
            if i < len(interim_history_for_final_answer) -1 : # Add newline separator except for last
                 current_prompt_for_final_answer += "\n"
        # Add AI header to prompt AI to speak
        current_prompt_for_final_answer += f"\n{self.ai_full_header}"


        final_answer_text = self.generate_text(
            prompt=current_prompt_for_final_answer, # Pass the conversation history as the prompt
            system_prompt=final_system_prompt.strip(),
            images=images if not tool_calls_made_this_turn else None, # Only pass initial images if no tool calls happened (context might be lost)
            stream=streaming_callback is not None,
            streaming_callback=lambda chunk, msg_type: streaming_callback(chunk, msg_type, {"type":"final_answer_chunk"}, turn_history) if streaming_callback else None,
            temperature=final_answer_temperature if final_answer_temperature is not None else self.default_temperature,
            **(llm_generation_kwargs or {})
        )

        if streaming_callback:
            streaming_callback("Final answer generation complete.", MSG_TYPE.MSG_TYPE_STEP_END, {"type": "final_answer_generation"}, turn_history)

        if isinstance(final_answer_text, dict) and "error" in final_answer_text: # Handle generation error
            turn_history.append({"type":"error", "content":f"LLM failed to generate final answer: {final_answer_text['error']}"})
            return {"final_answer": "", "tool_calls": tool_calls_made_this_turn, "error": final_answer_text["error"]}

        turn_history.append({"type":"final_answer_generated", "content":final_answer_text})
        return {"final_answer": final_answer_text, "tool_calls": tool_calls_made_this_turn, "error": None}

    def generate_text_with_rag(
        self,
        prompt: str,
        rag_query_function: Callable[[str, Optional[str], int, float], List[Dict[str, Any]]],
        rag_query_text: Optional[str] = None,
        rag_vectorizer_name: Optional[str] = None,
        rag_top_k: int = 5,
        rag_min_similarity_percent: float = 70.0,
        max_rag_hops: int = 0,
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
        ctx_size: Optional[int] = None,
        extract_objectives: bool = True,
        streaming_callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict], Optional[List]], bool]] = None,
        max_rag_context_characters: int = 32000,
        **llm_generation_kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced RAG with optional initial objective extraction and automatic intermediate summaries
        when context grows beyond ctx_size or self.default_ctx_size.
        """
        if not self.binding:
            return {"final_answer": "", "rag_hops_history": [], "all_retrieved_sources": [], "error": "LLM binding not initialized."}

        # Determine effective context size limit
        effective_ctx_size = ctx_size or getattr(self, "default_ctx_size", 20000)

        turn_rag_history_for_callback: List[Dict[str, Any]] = []
        rag_hops_details_list: List[Dict[str, Any]] = []
        all_unique_retrieved_chunks_map: Dict[str, Dict[str, Any]] = {}

        original_user_prompt = prompt
        objectives_text = ""
        # 0. Optional Objectives Extraction Step
        if extract_objectives:
            if streaming_callback:
                streaming_callback("Extracting and structuring objectives...", MSG_TYPE.MSG_TYPE_STEP, {"type": "objectives_extraction"}, turn_rag_history_for_callback)
            obj_prompt = (
                "You are an expert analyst. "
                "Your task is to extract and structure the key objectives from the user's request below. "
                "Output a bullet list of objectives only.\n\n"
                f"User request:\n\"{original_user_prompt}\""
            )
            obj_gen = self.generate_text(
                prompt=obj_prompt,
                system_prompt="Extract objectives",
                temperature=0.0,
                n_predict=200,
                stream=False
            )
            objectives_text = self.remove_thinking_blocks(obj_gen).strip()
            if streaming_callback:
                streaming_callback(f"Objectives extracted:\n{objectives_text}", MSG_TYPE.MSG_TYPE_STEP_END, {"type": "objectives_extracted"}, turn_rag_history_for_callback)

        current_query_for_rag = rag_query_text or None
        previous_queries=[]
        # 1. RAG Hops
        for hop_count in range(max_rag_hops + 1):
            if streaming_callback:
                streaming_callback(f"Starting RAG Hop {hop_count + 1}", MSG_TYPE.MSG_TYPE_STEP, {"type": "rag_hop_start", "hop": hop_count + 1}, turn_rag_history_for_callback)

            txt_previous_queries = f"Previous queries:\n"+'\n'.join(previous_queries)+"\n\n" if len(previous_queries)>0 else ""
            txt_informations = f"Information:\n"+'\n'.join([f"(from {chunk['document']}):{chunk['content']}" for _, chunk in all_unique_retrieved_chunks_map.items()]) if len(all_unique_retrieved_chunks_map)>0 else "This is the first request. No data received yet. Build a new query."
            txt_sp = "Your objective is to analyze the provided chunks of information, then decise if they are sufficient to reach the objective. If you need more information, formulate a new query to extract more data."
            txt_formatting = """The output format must be in form of json placed inside a json markdown tag. Here is the schema to use:
```json
{
    "decision": A boolean depicting your decision (true: more data is needed, false: there is enough data to reach objective),
    "query": (optional, only if decision is true). A new query to recover more information from the data source (do not use previous queries as they have already been used)
}
```
"""
            p = f"Objective:\n{objectives_text}\n\n{txt_previous_queries}\n\n{txt_informations}\n\n{txt_formatting}\n\n"
            response = self.generate_code(p,system_prompt=txt_sp)
            try:
                answer = json.loads(response)
                decision = answer["decision"]
                if not decision:
                    break
                else:
                    current_query_for_rag = answer["query"]
            except Exception as ex:
                trace_exception(ex)

            # Retrieve chunks
            try:
                retrieved = rag_query_function(current_query_for_rag, rag_vectorizer_name, rag_top_k, rag_min_similarity_percent)
            except Exception as e:
                trace_exception(e)
                return {"final_answer": "", "rag_hops_history": rag_hops_details_list, "all_retrieved_sources": list(all_unique_retrieved_chunks_map.values()), "error": str(e)}

            hop_details = {"query": current_query_for_rag, "retrieved_chunks_details": [], "status": ""}
            previous_queries.append(current_query_for_rag)
            new_unique = 0
            for chunk in retrieved:
                doc = chunk.get("file_path", "Unknown")
                content = str(chunk.get("chunk_text", ""))
                sim = float(chunk.get("similarity_percent", 0.0))
                detail = {"document": doc, "similarity": sim, "content": content,
                          "retrieved_in_hop": hop_count + 1, "query_used": current_query_for_rag}
                hop_details["retrieved_chunks_details"].append(detail)
                key = f"{doc}::{content[:100]}"
                if key not in all_unique_retrieved_chunks_map:
                    all_unique_retrieved_chunks_map[key] = detail
                    new_unique += 1
            hop_details["status"] = "Completed" if retrieved else "No chunks retrieved"
            if hop_count > 0 and new_unique == 0:
                hop_details["status"] = "No *new* unique chunks retrieved"
            rag_hops_details_list.append(hop_details)



        # 2. Prepare & Summarize Context
        sorted_chunks = sorted(all_unique_retrieved_chunks_map.values(),
                               key=lambda c: c["similarity"], reverse=True)
        context_lines = []
        total_chars = 0
        for c in sorted_chunks:
            snippet = (
                f"Source: {c['document']} (Sim: {c['similarity']:.1f}%, "
                f"Hop: {c['retrieved_in_hop']}, Query: '{c['query_used']}')\n"
                f"{c['content']}\n---\n"
            )
            if total_chars + len(snippet) > max_rag_context_characters:
                break
            context_lines.append(snippet)
            total_chars += len(snippet)

        accumulated_context = "".join(context_lines)

        # If context exceeds our effective limit, summarize it
        if self.count_tokens(accumulated_context) > effective_ctx_size:
            if streaming_callback:
                streaming_callback("Context too large, performing intermediate summary...", MSG_TYPE.MSG_TYPE_STEP, {"type": "intermediate_summary"}, turn_rag_history_for_callback)
            summary_prompt = (
                "Summarize the following gathered context into a concise form "
                "that preserves all key facts and sources needed to answer the user's request:\n\n"
                f"{accumulated_context}"
            )
            summary = self.generate_text(
                prompt=summary_prompt,
                system_prompt="Intermediate summary",
                temperature=0.0,
                n_predict= n_predict or 512,
                stream=False
            )
            accumulated_context = self.remove_thinking_blocks(summary).strip()
            if streaming_callback:
                streaming_callback("Intermediate summary complete.", MSG_TYPE.MSG_TYPE_STEP_END, {"type": "intermediate_summary"}, turn_rag_history_for_callback)

        # 3. Final Answer Generation
        final_prompt = [
            f"Original request: {original_user_prompt}"
        ]
        if objectives_text:
            final_prompt.insert(1, f"Structured Objectives:\n{objectives_text}\n")
        if accumulated_context:
            final_prompt.append(
                "\nBased on the gathered context:\n---\n"
                f"{accumulated_context}\n---"
            )
        else:
            final_prompt.append("\n(No relevant context retrieved.)")
        final_prompt.append(
            "\nProvide a comprehensive answer using ONLY the above context. "
            "If context is insufficient, state so clearly."
        )
        final_prompt.append(self.ai_full_header)

        final_answer = self.generate_text(
            prompt="\n".join(final_prompt),
            images=images,
            system_prompt=system_prompt,
            n_predict=n_predict,
            stream=stream,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            seed=seed,
            n_threads=n_threads,
            ctx_size=ctx_size,
            streaming_callback=streaming_callback if stream else None,
            **llm_generation_kwargs
        )
        answer_text = self.remove_thinking_blocks(final_answer) if isinstance(final_answer, str) else final_answer

        return {
            "final_answer": answer_text,
            "rag_hops_history": rag_hops_details_list,
            "all_retrieved_sources": list(all_unique_retrieved_chunks_map.values()),
            "error": None
        }


    def generate_code(
                        self,
                        prompt,
                        images=[],
                        system_prompt=None,
                        template=None,
                        language="json",
                        code_tag_format="markdown", # or "html"
                        max_size = None,
                        temperature = None,
                        top_k = None,
                        top_p=None,
                        repeat_penalty=None,
                        repeat_last_n=None,
                        callback=None,
                        debug=False ):
        """
        Generates a single code block based on a prompt.
        Uses the underlying LLM binding via `generate_text`.
        Handles potential continuation if the code block is incomplete.
        """
        if not  system_prompt:
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
        system_prompt += f"""You must return a single code tag.
Do not split the code in multiple tags.
{self.ai_full_header}"""

        response = self.generate_text(
            prompt,
            images=images,
            system_prompt=system_prompt,
            n_predict=max_size,
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
                continuation_prompt = f"{full_prompt}{code_content}\n\n{self.user_full_header}The previous code block was incomplete. Continue the code exactly from where it left off. Do not repeat the previous part. Only provide the continuation inside a single {code_tag_format} code tag.\n{self.ai_full_header}"

                continuation_response = self.generate_text(
                    continuation_prompt,
                    images=images, # Resend images if needed for context
                    n_predict=max_size, # Allow space for continuation
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
            max_size=max_answer_length,
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

            parsed_response = json.loads(response_json_str)
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
            max_size=max_answer_length,
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

            result = json.loads(response_json_str)
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
            max_size=max_answer_length,
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

            result = json.loads(response_json_str)
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
            ctx_size = self.default_ctx_size or 8192 # Provide a fallback default
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
            available_tokens_for_dynamic_content = ctx_size - static_tokens - (self.default_n_predict or 1024) # Reserve space for output
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

            response = self.generate_text(prompt, n_predict=(self.default_n_predict or 1024), streaming_callback=callback)

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
            available_final_tokens = ctx_size - final_static_tokens - (self.default_n_predict or 1024) # Reserve space for output
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

        final_summary_raw = self.generate_text(final_prompt, n_predict=(self.default_n_predict or 1024), streaming_callback=callback)

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
            ctx_size = self.default_ctx_size or 8192
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
                available_tokens_for_dynamic_content = ctx_size - static_tokens - (self.default_n_predict or 1024)
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

                response = self.generate_text(prompt, n_predict=(self.default_n_predict or 1024), streaming_callback=callback) # Use main callback for streaming output

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
             available_final_tokens = ctx_size - final_static_tokens - (self.default_n_predict or 1024)
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

        final_output_raw = self.generate_text(final_prompt, n_predict=(self.default_n_predict or 1024), streaming_callback=callback) # Use main callback

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


