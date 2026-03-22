# lollms_client/lollms_core.py
# author: ParisNeo
# description: LollmsClient definition file
import requests
import json
import re
import base64
import numpy as np
import uuid
import hashlib
import time
from pathlib import Path
from enum import Enum
from typing import List, Optional, Callable, Union, Dict, Any

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import robust_json_parser, build_image_dicts, dict_to_markdown
from lollms_client.lollms_llm_binding import LollmsLLMBinding, LollmsLLMBindingManager
from lollms_client.lollms_tts_binding import LollmsTTSBinding, LollmsTTSBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBinding, LollmsTTIBindingManager
from lollms_client.lollms_stt_binding import LollmsSTTBinding, LollmsSTTBindingManager
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, LollmsTTVBindingManager
from lollms_client.lollms_ttm_binding import LollmsTTMBinding, LollmsTTMBindingManager
from lollms_client.lollms_tools_binding import LollmsToolBinding, LollmsTOOLBindingManager

from lollms_client.lollms_discussion import LollmsDiscussion



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
                 tools_binding_name: Optional[str] = None,

                 # Modality Binding Directories
                 llm_bindings_dir: Path = Path(__file__).parent / "llm_bindings",
                 tts_bindings_dir: Path = Path(__file__).parent / "tts_bindings",
                 tti_bindings_dir: Path = Path(__file__).parent / "tti_bindings",
                 stt_bindings_dir: Path = Path(__file__).parent / "stt_bindings",
                 ttv_bindings_dir: Path = Path(__file__).parent / "ttv_bindings",
                 ttm_bindings_dir: Path = Path(__file__).parent / "ttm_bindings",
                 tools_bindings_dir: Path = Path(__file__).parent / "tools_bindings",

                 # Configurations
                 llm_binding_config: Optional[Dict[str, any]] = None,
                 tts_binding_config: Optional[Dict[str, any]] = None, 
                 tti_binding_config: Optional[Dict[str, any]] = None, 
                 stt_binding_config: Optional[Dict[str, any]] = None, 
                 ttv_binding_config: Optional[Dict[str, any]] = None, 
                 ttm_binding_config: Optional[Dict[str, any]] = None, 
                 tools_binding_config: Optional[Dict[str, any]] = None,
                 user_name ="user",
                 ai_name = "assistant",
                 callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict]], bool]] = None,

                 debug: Optional[bool] = False,
                 **kwargs
                 ):
        """
        Initialize the LollmsClient with LLM and optional modality bindings.
        """

        self.debug = debug
        if callback: callback("🚀 Initializing **Lollms Client**...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
        
        self.llm_binding_manager = LollmsLLMBindingManager(llm_bindings_dir)
        self.tts_binding_manager = LollmsTTSBindingManager(tts_bindings_dir)
        self.tti_binding_manager = LollmsTTIBindingManager(tti_bindings_dir)
        self.stt_binding_manager = LollmsSTTBindingManager(stt_bindings_dir)
        self.ttv_binding_manager = LollmsTTVBindingManager(ttv_bindings_dir)
        self.ttm_binding_manager = LollmsTTMBindingManager(ttm_bindings_dir)
        self.tools_binding_manager = LollmsTOOLBindingManager(tools_bindings_dir)

        self.llm: Optional[LollmsLLMBinding] = None
        self.tts: Optional[LollmsTTSBinding] = None
        self.tti: Optional[LollmsTTIBinding] = None
        self.stt: Optional[LollmsSTTBinding] = None
        self.ttv: Optional[LollmsTTVBinding] = None
        self.ttm: Optional[LollmsTTMBinding] = None
        self.tools: Optional[LollmsToolBinding] = None

        # User and AI names are important for prompt construction
        self.user_name = user_name
        self.ai_name = ai_name

        if llm_binding_name:
            if callback: callback(f"🤖 Initializing **LLM** binding: `{llm_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            config = llm_binding_config or {}
            config['user_name'] = self.user_name
            config['ai_name'] = self.ai_name
            self.llm = self.llm_binding_manager.create_binding(
                binding_name=llm_binding_name,
                **{k: v for k, v in config.items() if k != "binding_name"}
            )
            if self.llm is None:
                msg = f"Failed to create LLM binding: {llm_binding_name}."
                if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                ASCIIColors.warning(msg)
            elif callback:
                callback(f"✅ **LLM** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})

        if tts_binding_name:
            if callback: callback(f"🗣️ Initializing **TTS** binding: `{tts_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.tts = self.tts_binding_manager.create_binding(binding_name=tts_binding_name, **(tts_binding_config or {}))
                if self.tts is None: 
                    msg = f"Failed to create TTS binding: {tts_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **TTS** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.tts = None
                if callback: callback(f"❌ Error initializing TTS: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})

        if tti_binding_name:
            if callback: callback(f"🎨 Initializing **TTI** binding: `{tti_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.tti = self.tti_binding_manager.create_binding(binding_name=tti_binding_name, **(tti_binding_config or {}))
                if self.tti is None: 
                    msg = f"Failed to create TTI binding: {tti_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **TTI** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.tti = None
                if callback: callback(f"❌ Error initializing TTI: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})
                
        if stt_binding_name:
            if callback: callback(f"👂 Initializing **STT** binding: `{stt_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.stt = self.stt_binding_manager.create_binding(binding_name=stt_binding_name, **(stt_binding_config or {}))
                if self.stt is None: 
                    msg = f"Failed to create STT binding: {stt_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **STT** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.stt = None
                if callback: callback(f"❌ Error initializing STT: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})
                
        if ttv_binding_name:
            if callback: callback(f"🎬 Initializing **TTV** binding: `{ttv_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.ttv = self.ttv_binding_manager.create_binding(binding_name=ttv_binding_name, **(ttv_binding_config or {}))
                if self.ttv is None: 
                    msg = f"Failed to create TTV binding: {ttv_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **TTV** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.ttv = None
                if callback: callback(f"❌ Error initializing TTV: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})

        if ttm_binding_name:
            if callback: callback(f"🎵 Initializing **TTM** binding: `{ttm_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.ttm = self.ttm_binding_manager.create_binding(binding_name=ttm_binding_name, **(ttm_binding_config or {}))
                if self.ttm is None: 
                    msg = f"Failed to create TTM binding: {ttm_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **TTM** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.ttm = None
                if callback: callback(f"❌ Error initializing TTM: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})

        if tools_binding_name:
            if callback: callback(f"🔌 Initializing **MCP** binding: `{tools_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.tools = self.tools_binding_manager.create_binding(binding_name=tools_binding_name, **(tools_binding_config or {}))
                if self.tools is None: 
                    msg = f"Failed to create MCP binding: {tools_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **MCP** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.tools = None  
                if callback: callback(f"❌ Error initializing MCP: {e}", MSG_TYPE.MSG_TYPE_ERROR, {})   

        if callback: callback("✨ **Lollms Client** Initialization Complete.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})       

    # --- Properties delegating to LLM ---
    @property
    def start_header_id_template(self): return self.llm.start_header_id_template if self.llm else "!@>"
    @property
    def end_header_id_template(self): return self.llm.end_header_id_template if self.llm else ": "
    @property
    def system_message_template(self): return self.llm.system_message_template if self.llm else "system"
    @property
    def system_full_header(self): return self.llm.system_full_header if self.llm else f"!@>system: "
    @property
    def user_full_header(self): return self.llm.user_full_header if self.llm else f"!@>{self.user_name}: "
    @property
    def ai_full_header(self): return self.llm.ai_full_header if self.llm else f"!@>{self.ai_name}: "

    def sink(self, s=None,i=None,d=None): pass

    # --- Binding Updates ---
    def update_llm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['user_name'] = self.user_name
        config['ai_name'] = self.ai_name
        self.llm = self.llm_binding_manager.create_binding(binding_name=binding_name, **config)
        if self.llm is None: raise ValueError(f"Failed to update LLM binding: {binding_name}")

    def update_tts_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.tts = self.tts_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.tts is None: raise ValueError(f"Failed to update TTS binding: {binding_name}")

    def update_tti_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.tti = self.tti_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.tti is None: raise ValueError(f"Failed to update TTI binding: {binding_name}")

    def update_stt_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.stt = self.stt_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.stt is None: raise ValueError(f"Failed to update STT binding: {binding_name}")

    def update_ttv_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.ttv = self.ttv_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.ttv is None: raise ValueError(f"Failed to update TTV binding: {binding_name}")

    def update_ttm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.ttm = self.ttm_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.ttm is None: raise ValueError(f"Failed to update TTM binding: {binding_name}")

    def update_tools_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.tools = self.tools_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.tools is None: raise ValueError(f"Failed to update MCP binding: {binding_name}")

    # --- Core LLM Methods (Delegated) ---
    def tokenize(self, text: str) -> list:
        if text is None:
            text = ""
        if self.llm: return self.llm.tokenize(text)
        raise RuntimeError("LLM binding not initialized.")

    def detokenize(self, tokens: list) -> str:
        if self.llm: return self.llm.detokenize(tokens)
        raise RuntimeError("LLM binding not initialized.")

    def count_tokens(self, text: str) -> int:
        if text is None:
            text = ""
        if self.llm: return self.llm.count_tokens(text)
        raise RuntimeError("LLM binding not initialized.")

    def count_image_tokens(self, image: str) -> int:
        if self.llm: return self.llm.count_image_tokens(image)
        raise RuntimeError("LLM binding not initialized.")

    def get_model_details(self) -> dict:
        if self.llm: return self.llm.get_model_info()
        raise RuntimeError("LLM binding not initialized.")

    def switch_model(self, model_name: str) -> bool:
        if self.llm: return self.llm.load_model(model_name)
        raise RuntimeError("LLM binding not initialized.")

    def get_available_llm_bindings(self) -> List[str]: 
        return self.llm_binding_manager.get_available_bindings()

    def generate_text(self, *args, **kwargs) -> Union[str, dict]:
        if self.llm: return self.llm.generate_text(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
    
    def generate(self, *args, **kwargs) -> Union[str, dict]:
        if self.llm: return self.llm.generate_text(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def generate_from_messages(self, *args, **kwargs) -> Union[str, dict]:
        if self.llm: return self.llm.generate_from_messages(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def chat(self, *args, **kwargs) -> Union[str, dict]:
        if self.llm: return self.llm.chat(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def embed(self, *args, **kwargs):
        if self.llm: return self.llm.embed(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
    def get_ctx_size(self, *args, **kwargs) -> Optional[int]:
        """
        Retrieves context size for a model from a hardcoded list.
        """
        if self.llm: return self.llm.get_ctx_size(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def list_models(self):
        if self.llm: return self.llm.list_models()
        raise RuntimeError("LLM binding not initialized.")

    def listMountedPersonalities(self) -> Union[List[Dict], Dict]:
        if self.llm and hasattr(self.llm, 'lollms_listMountedPersonalities'):
            return self.llm.lollms_listMountedPersonalities()
        return {"status": False, "error": "Functionality not available for the current binding"}

    # --- High Level Text Operations (Delegated to LLM Binding) ---
    def generate_codes(self, *args, **kwargs):
        if self.llm: return self.llm.tp.generate_codes(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def generate_code(self, *args, **kwargs):
        if self.llm: return self.llm.tp.generate_code(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def update_code(self, *args, **kwargs):
        if self.llm: return self.llm.tp.update_code(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def generate_structured_content(self, *args, **kwargs):
        if self.llm: return self.llm.tp.generate_structured_content(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def generate_structured_content_pydantic(self, *args, **kwargs):
        if self.llm: return self.llm.tp.generate_structured_content_pydantic(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def yes_no(self, *args, **kwargs):
        if self.llm: return self.llm.tp.yes_no(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def multichoice_question(self, *args, **kwargs):
        if self.llm: return self.llm.tp.multichoice_question(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def multichoice_ranking(self, *args, **kwargs):
        if self.llm: return self.llm.tp.multichoice_ranking(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def extract_code_blocks(self, *args, **kwargs):
        if self.llm: return self.llm.tp.extract_code_blocks(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def extract_thinking_blocks(self, *args, **kwargs):
        if self.llm: return self.llm.tp.extract_thinking_blocks(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def remove_thinking_blocks(self, *args, **kwargs):
        if self.llm: return self.llm.tp.remove_thinking_blocks(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    # --- Wrappers for other Modality Bindings ---
    def generate_image(self, *args, **kwargs):
        if self.tti: return self.tti.generate_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def edit_image(self, *args, **kwargs):
        if self.tti: return self.tti.edit_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def generate_audio(self, *args, **kwargs):
        if self.tts: return self.tts.generate_audio(*args, **kwargs)
        raise RuntimeError("TTS binding not initialized.")

    def transcribe_audio(self, *args, **kwargs):
        if self.stt: return self.stt.transcribe_audio(*args, **kwargs)
        raise RuntimeError("STT binding not initialized.")

    def generate_video(self, *args, **kwargs):
        if self.ttv: return self.ttv.generate_video(*args, **kwargs)
        raise RuntimeError("TTV binding not initialized.")

    def generate_music(self, *args, **kwargs):
        if self.ttm: return self.ttm.generate_music(*args, **kwargs)
        raise RuntimeError("TTM binding not initialized.")

    def long_context_processing(self, text_to_process: str, contextual_prompt: str, **kwargs) -> str:
        if self.llm:
            return self.llm.tp.long_context_processing(text_to_process, contextual_prompt, **kwargs)

def chunk_text(text, tokenizer, detokenizer, chunk_size, overlap, use_separators=True):
    tokens = tokenizer(text)
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunks.append(detokenizer(tokens[start_idx:end_idx]))
        start_idx += chunk_size - overlap
        if start_idx >= len(tokens): break
        start_idx = max(0, start_idx)
    return chunks
