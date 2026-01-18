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
from lollms_client.lollms_mcp_binding import LollmsMCPBinding, LollmsMCPBindingManager

from lollms_client.lollms_discussion import LollmsDiscussion
from lollms_client.lollms_agentic import (
    TaskStatus, TaskPlanner, MemoryManager, 
    UncertaintyManager, ToolPerformanceTracker, 
    SubTask, ExecutionPlan
)


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
                 callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict]], bool]] = None,
                 **kwargs
                 ):
        """
        Initialize the LollmsClient with LLM and optional modality bindings.
        """
        if callback: callback("🚀 Initializing **Lollms Client**...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
        
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

        if mcp_binding_name:
            if callback: callback(f"🔌 Initializing **MCP** binding: `{mcp_binding_name}`...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            try:
                self.mcp = self.mcp_binding_manager.create_binding(binding_name=mcp_binding_name, **(mcp_binding_config or {}))
                if self.mcp is None: 
                    msg = f"Failed to create MCP binding: {mcp_binding_name}"
                    if callback: callback(f"❌ {msg}", MSG_TYPE.MSG_TYPE_ERROR, {})
                    ASCIIColors.warning(msg)
                elif callback:
                    callback(f"✅ **MCP** binding ready.", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
            except Exception as e:
                trace_exception(e)
                self.mcp = None  
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

    def update_mcp_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        self.mcp = self.mcp_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.mcp is None: raise ValueError(f"Failed to update MCP binding: {binding_name}")

    # --- Core LLM Methods (Delegated) ---
    def tokenize(self, text: str) -> list:
        if self.llm: return self.llm.tokenize(text)
        raise RuntimeError("LLM binding not initialized.")

    def detokenize(self, tokens: list) -> str:
        if self.llm: return self.llm.detokenize(tokens)
        raise RuntimeError("LLM binding not initialized.")

    def count_tokens(self, text: str) -> int:
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

    # --- Agentic & Complex workflows ---
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
            max_scratchpad_size: int = 20000,
            **llm_generation_kwargs
        ) -> Dict[str, Any]:
        
        if not self.llm:
            return {"final_answer": "", "tool_calls": [], "sources": [], "error": "LLM binding not initialized."}
        if max_reasoning_steps is None:
            max_reasoning_steps=15
        if rag_min_similarity_percent is None:
            rag_min_similarity_percent=50.0
        if final_answer_temperature is None:
            final_answer_temperature=0.7
        if rag_top_k is None:
            rag_top_k=5
            
        tools_infos = []
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
            ASCIIColors.red(f"Prompt size:{prompt_size}/{self.llm.default_ctx_size if self.llm.default_ctx_size else 'Unknown'}")
            ASCIIColors.cyan(f"** DEBUG: DONE **")

        # Enhanced discovery phase with more detailed logging
        discovery_step_id = log_event("🔧 Discovering and configuring available capabilities...", MSG_TYPE.MSG_TYPE_STEP_START)
        all_discovered_tools, visible_tools, rag_registry, rag_tool_specs = [], [], {}, {}
        
        if use_mcps and hasattr(self, 'mcp'):
            log_event("  📡 Connecting to MCP services...", MSG_TYPE.MSG_TYPE_INFO)
            mcp_tools = self.mcp.discover_tools(force_refresh=True)
            if isinstance(use_mcps, list): 
                filtered_tools = [t for t in mcp_tools if t["name"] in use_mcps]
                tools_infos+=[f"    🛠️{f['name']}" for f in filtered_tools]
                all_discovered_tools.extend(filtered_tools)
                log_event(f"  ✅ Loaded {len(filtered_tools)} specific MCP tools: {', '.join(use_mcps)}", MSG_TYPE.MSG_TYPE_INFO)
            elif use_mcps is True: 
                tools_infos+=[f"    🛠️{f['name']}" for f in mcp_tools]
                all_discovered_tools.extend(mcp_tools)
                log_event(f"  ✅ Loaded {len(mcp_tools)} MCP tools", MSG_TYPE.MSG_TYPE_INFO)
        
        if use_data_store:
            log_event(f"  📚 Setting up {len(use_data_store)} knowledge bases...", MSG_TYPE.MSG_TYPE_INFO)
            for name, info in use_data_store.items():
                tool_name, description, call_fn = f"rag::{name}", f"Queries the '{name}' knowledge base.", None
                if callable(info): call_fn = info
                elif isinstance(info, dict):
                    if "callable" in info and callable(info["callable"]): call_fn = info["callable"]
                    description = info.get("description", "This is a datastore with the following description: \n" + description)
                if call_fn:
                    visible_tools.append({"name": tool_name, "description": description, "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}})
                    rag_registry[tool_name] = call_fn
                    rag_tool_specs[tool_name] = {"default_top_k": rag_top_k, "default_min_sim": rag_min_similarity_percent}
                    tools_infos.append(f"    📖 {name}")
        visible_tools.extend(all_discovered_tools)
        built_in_tools = [
            {"name": "local_tools::final_answer", "description": "Provide the final answer directly to the user.", "input_schema": {}},
            {"name": "local_tools::request_clarification", "description": "Ask the user for more specific information when the request is ambiguous.", "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
            {"name": "local_tools::revise_plan", "description": "Update the execution plan based on new discoveries or changing requirements.", "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}, "new_plan": {"type": "array"}}, "required": ["reason", "new_plan"]}}
        ]
        tools_infos+=[f"    🔨 final_answer","    🔨 request_clarification","    🔨 revise_plan"]
        
        
        if getattr(self, "tti", None): 
            built_in_tools.append({"name": "local_tools::generate_image", "description": "Generate an image from a text description.", "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}})
        
        all_visible_tools = visible_tools + built_in_tools
        tool_summary = "\n".join([f"- **{t['name']}**: {t['description']}" for t in all_visible_tools[:20]])
        
        log_event("\n".join(tools_infos), MSG_TYPE.MSG_TYPE_INFO)
        log_event(f"✅ Ready with {len(all_visible_tools)} total capabilities", MSG_TYPE.MSG_TYPE_STEP_END, event_id=discovery_step_id, meta={"tool_count": len(all_visible_tools), "mcp_tools": len(all_discovered_tools), "rag_tools": len(rag_registry)})

        # Enhanced triage with better prompting
        triage_step_id = log_event("🤔 Analyzing request complexity and optimal approach...", MSG_TYPE.MSG_TYPE_STEP_START)
        strategy = "COMPLEX_PLAN"
        strategy_data = {}
        try:
            triage_prompt = f"""Analyze this user request to determine the most efficient execution strategy.

DISCUSSION and Final Prompt:
{prompt}
CONTEXT: {context or "No additional context provided"}
IMAGES PROVIDED: {"Yes" if images else "No"}

AVAILABLE CAPABILITIES:
{tool_summary}

Based on the request complexity and available tools, choose the optimal strategy:

1. **DIRECT_ANSWER**: For simple greetings, basic questions, or requests that don't require any tools
   - Use when: The request can be fully answered with your existing knowledge with confidence, and no tool seems to add any significant value to the answer
   - Example: "Hello", "What is Python?", "Explain quantum physics"

2. **REQUEST_CLARIFICATION**: When the request is too vague or ambiguous
   - Use when: The request lacks essential details needed to proceed
   - Example: "Help me with my code" (what code? what issue?)

3. **SINGLE_TOOL**: For straightforward requests that need exactly one tool
   - Use when: The request clearly maps to a single, specific tool operation
   - Example: "Search for information about X", "Generate an image of Y"

4. **COMPLEX_PLAN**: For multi-step requests requiring coordination of multiple tools
   - Use when: The request involves multiple steps, data analysis, or complex reasoning
   - Example: "Research X, then create a report comparing it to Y"

Provide your analysis in JSON format:
{{"thought": "Detailed reasoning about the request complexity and requirements", "strategy": "ONE_OF_THE_FOUR_OPTIONS", "confidence": percentage float value, eg 80, "text_output": "Direct answer or clarification question if applicable", "required_tool_name": "specific tool name if SINGLE_TOOL strategy", "estimated_steps": 3}}"""
            
            triage_schema = {
                "thought": "string", "strategy": "string", "confidence": "number",
                "text_output": "string", "required_tool_name": "string", "estimated_steps": "number"
            }
            strategy_data = self.generate_structured_content(prompt=triage_prompt, schema=triage_schema, temperature=0.1, system_prompt=system_prompt, **llm_generation_kwargs)
            strategy = strategy_data.get("strategy") if strategy_data else "COMPLEX_PLAN"
            
            log_event(f"Strategy analysis complete.\n**confidence**: {strategy_data.get('confidence', 0.5)}\n**reasoning**: {strategy_data.get('thought', 'None')}", MSG_TYPE.MSG_TYPE_INFO, meta={
                "strategy": strategy,
                "confidence": strategy_data.get("confidence", 50),
                "estimated_steps": strategy_data.get("estimated_steps", 1),
                "reasoning": strategy_data.get("thought", "")
            })
            
        except Exception as e:
            log_event(f"Triage analysis failed: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, event_id=triage_step_id)
            log_event("Defaulting to complex planning approach", MSG_TYPE.MSG_TYPE_WARNING)

        if force_mcp_use and strategy == "DIRECT_ANSWER":
            strategy = "COMPLEX_PLAN"
            log_event("Forcing tool usage - switching to complex planning", MSG_TYPE.MSG_TYPE_INFO)
            
        log_event(f"✅ Strategy selected: {strategy.replace('_', ' ').title()}", MSG_TYPE.MSG_TYPE_STEP_END, event_id=triage_step_id, meta={"final_strategy": strategy})

        # Handle simple strategies
        if strategy == "DIRECT_ANSWER":
            final_answer = strategy_data.get("text_output", "I can help with that.")
            log_event("Providing direct response", MSG_TYPE.MSG_TYPE_INFO)
            if streaming_callback: streaming_callback(final_answer, MSG_TYPE.MSG_TYPE_CONTENT, {})
            return {"final_answer": final_answer, "tool_calls": [], "sources": [], "error": None, "clarification_required": False, "final_scratchpad": f"Strategy: DIRECT_ANSWER\nConfidence: {strategy_data.get('confidence', 0.9)}\nReasoning: {strategy_data.get('thought')}"}

        if strategy == "REQUEST_CLARIFICATION":
            clarification_question = strategy_data.get("text_output", "Could you please provide more details about what specifically you'd like me to help with?")
            log_event("Requesting clarification from user", MSG_TYPE.MSG_TYPE_INFO)
            return {"final_answer": clarification_question, "tool_calls": [], "sources": [], "error": None, "clarification_required": True, "final_scratchpad": f"Strategy: REQUEST_CLARIFICATION\nConfidence: {strategy_data.get('confidence', 0.8)}\nReasoning: {strategy_data.get('thought')}"}

        # Enhanced single tool execution
        if strategy == "SINGLE_TOOL":
            synthesis_id = log_event("⚡ Executing single-tool strategy...", MSG_TYPE.MSG_TYPE_STEP_START)
            try:
                tool_name = strategy_data.get("required_tool_name")
                tool_spec = next((t for t in all_visible_tools if t['name'] == tool_name), None)
                if not tool_spec:
                    raise ValueError(f"Strategy analysis selected unavailable tool: '{tool_name}'")
                
                log_event(f"Selected tool: {tool_name}", MSG_TYPE.MSG_TYPE_INFO)
                
                # Enhanced parameter generation prompt
                param_prompt = f"""Generate the optimal parameters for the selected tool to fulfill the user's request.

FULL discussion and USER REQUEST: 
{prompt}
SELECTED TOOL: {json.dumps(tool_spec, indent=2)}
CONTEXT: {context or "None"}

Analyze the user's request carefully and provide the most appropriate parameters.
If the request has implicit requirements, infer them intelligently.

Output the parameters as JSON: {{"tool_params": {{...}}}}"""
                
                log_prompt("Parameter Generation Prompt", param_prompt)
                param_data = self.generate_structured_content(prompt=param_prompt, schema={"tool_params": "object"}, temperature=0.1, **llm_generation_kwargs)
                tool_params = param_data.get("tool_params", {}) if param_data else {}
                
                log_event(f"Generated parameters: {json.dumps(tool_params)}", MSG_TYPE.MSG_TYPE_INFO)
                
                start_time, sources, tool_result = time.time(), [], {}
                if tool_name in rag_registry:
                    query = tool_params.get("query", prompt)
                    log_event(f"Searching knowledge base with query: '{query}'", MSG_TYPE.MSG_TYPE_INFO)
                    rag_fn = rag_registry[tool_name]
                    raw_results = rag_fn(query=query, rag_top_k=rag_top_k, rag_min_similarity_percent=rag_min_similarity_percent)
                    docs = [d for d in (raw_results.get("results", []) if isinstance(raw_results, dict) else raw_results or [])]
                    tool_result = {"status": "success", "results": docs}
                    sources = [
                                {
                                    "title":d["title"], 
                                    "content":d["content"], 
                                    "source": tool_name, 
                                    "metadata": d.get("metadata", {}), 
                                    "score": d.get("score", 0.0)
                                } 
                                for d in docs]
                    log_event(sources, MSG_TYPE.MSG_TYPE_SOURCES_LIST)
                    log_event(f"Retrieved {len(docs)} relevant documents", MSG_TYPE.MSG_TYPE_INFO)
                elif hasattr(self, "mcp") and "local_tools" not in tool_name:
                    log_event(f"Executing MCP tool: {tool_name}", MSG_TYPE.MSG_TYPE_TOOL_CALL, meta={"tool_name": tool_name, "params": tool_params})
                    tool_result = self.mcp.execute_tool(tool_name, tool_params, lollms_client_instance=self)
                    log_event(f"Tool execution completed", MSG_TYPE.MSG_TYPE_TOOL_OUTPUT, meta={"result_status": tool_result.get("status", "unknown")})
                else:
                    tool_result = {"status": "failure", "error": f"Tool '{tool_name}' could not be executed in single-step mode."}
                
                if tool_result.get("status","success") != "success" or "error" in tool_result:
                    error_detail = tool_result.get("error", "Unknown tool error in single-step mode.")
                    raise RuntimeError(error_detail)

                response_time = time.time() - start_time
                tool_calls_this_turn = [{"name": tool_name, "params": tool_params, "result": tool_result, "response_time": response_time}]
                
                # Enhanced synthesis prompt
                synthesis_prompt = f"""Create a comprehensive and user-friendly response based on the tool execution results.

FULL DISCUSSON and USER REQUEST:
{prompt}
TOOL USED: {tool_name}
TOOL RESULT: {json.dumps(tool_result, indent=2)}

Guidelines for your response:
1. Be direct and helpful
2. Synthesize the information clearly
3. Address the user's specific needs
4. If the tool provided data, present it in an organized way
5. If relevant, mention any limitations or additional context

RESPONSE:"""
                
                log_event("Synthesizing final response", MSG_TYPE.MSG_TYPE_INFO)
                final_answer = self.generate_text(prompt=synthesis_prompt, system_prompt=system_prompt, stream=streaming_callback is not None, streaming_callback=streaming_callback, temperature=final_answer_temperature, **llm_generation_kwargs)
                final_answer = self.remove_thinking_blocks(final_answer)
                
                log_event("✅ Single-tool execution completed successfully", MSG_TYPE.MSG_TYPE_STEP_END, event_id=synthesis_id)
                return {"final_answer": final_answer, "tool_calls": tool_calls_this_turn, "sources": sources, "error": None, "clarification_required": False, "final_scratchpad": f"Strategy: SINGLE_TOOL\nTool: {tool_name}\nResult: Success\nResponse Time: {response_time:.2f}s"}

            except Exception as e:
                trace_exception(e)
                log_event(f"Single-tool execution failed: {e}", MSG_TYPE.MSG_TYPE_EXCEPTION, event_id=synthesis_id)
                log_event("Escalating to complex planning approach", MSG_TYPE.MSG_TYPE_INFO)

        # Execute complex reasoning with enhanced capabilities
        return self._execute_complex_reasoning_loop(
            prompt=prompt, context=context, system_prompt=system_prompt,
            reasoning_system_prompt=reasoning_system_prompt, images=images,
            max_reasoning_steps=max_reasoning_steps, decision_temperature=decision_temperature,
            final_answer_temperature=final_answer_temperature, streaming_callback=streaming_callback,
            debug=debug, enable_self_reflection=enable_self_reflection,
            all_visible_tools=all_visible_tools, rag_registry=rag_registry, rag_tool_specs=rag_tool_specs,
            log_event_fn=log_event, log_prompt_fn=log_prompt, max_scratchpad_size=max_scratchpad_size,
            **llm_generation_kwargs
        )

    def _execute_complex_reasoning_loop(
        self, prompt, context="", system_prompt="", reasoning_system_prompt="", images=None,
        max_reasoning_steps=15, decision_temperature=0.5, final_answer_temperature=0.7,
        streaming_callback=None, debug=False, enable_self_reflection=True, all_visible_tools=None,
        rag_registry=None, rag_tool_specs=None, log_event_fn=None, log_prompt_fn=None, 
        max_scratchpad_size=20000, **llm_generation_kwargs
    ) -> Dict[str, Any]:
        """Sophisticated agentic loop for solving complex tasks through multi-step reasoning and tool use."""
        
        # Ensure registries exist
        if all_visible_tools is None: all_visible_tools = []
        if rag_registry is None: rag_registry = {}
        if rag_tool_specs is None: rag_tool_specs = {}
        if log_event_fn is None: log_event_fn = self.sink
        if log_prompt_fn is None: log_prompt_fn = self.sink

        planner, memory_manager, performance_tracker = TaskPlanner(self), MemoryManager(), ToolPerformanceTracker()
        
        def _get_friendly_action_description(tool_name, requires_code, requires_image):
            descriptions = {
                "local_tools::final_answer": "📋 Preparing final answer",
                "local_tools::request_clarification": "❓ Requesting clarification",
                "local_tools::generate_image": "🎨 Creating image",
                "local_tools::revise_plan": "📝 Revising execution plan"
            }
            if tool_name in descriptions:
                return descriptions[tool_name]
            if "rag::" in tool_name: 
                return f"🔍 Searching {tool_name.split('::')[-1]} knowledge base"
            if requires_code: 
                return "💻 Processing code"
            if requires_image: 
                return "🖼️ Analyzing images"
            return f"🔧 Using {tool_name.replace('_', ' ').replace('::', ' - ').title()}"

        def _compress_scratchpad_intelligently(scratchpad: str, original_request: str, target_size: int) -> str:
            """Enhanced scratchpad compression that preserves key decisions and recent context"""
            if len(scratchpad) <= target_size:
                return scratchpad
            
            log_event_fn("📝 Compressing scratchpad to maintain focus...", MSG_TYPE.MSG_TYPE_INFO)
            
            # Extract key components
            lines = scratchpad.split('\n')
            plan_section = []
            decisions = []
            recent_observations = []
            
            current_section = None
            for i, line in enumerate(lines):
                if "### Execution Plan" in line or "### Updated Plan" in line:
                    current_section = "plan"
                elif "### Step" in line and ("Thought" in line or "Decision" in line):
                    current_section = "decision"
                elif "### Step" in line and "Observation" in line:
                    current_section = "observation"
                elif line.startswith("###"):
                    current_section = None
                
                if current_section == "plan" and line.strip():
                    plan_section.append(line)
                elif current_section == "decision" and line.strip():
                    decisions.append((i, line))
                elif current_section == "observation" and line.strip():
                    recent_observations.append((i, line))
            
            # Keep most recent items and important decisions
            recent_decisions = decisions[-3:] if len(decisions) > 3 else decisions
            recent_obs = recent_observations[-5:] if len(recent_observations) > 5 else recent_observations
            
            compressed_parts = [
                f"### Original Request\n{original_request}",
                f"### Current Plan\n" + '\n'.join(plan_section[-10:]),
                f"### Recent Key Decisions"
            ]
            
            for _, decision in recent_decisions:
                compressed_parts.append(decision)
            
            compressed_parts.append("### Recent Observations")
            for _, obs in recent_obs:
                compressed_parts.append(obs)
            
            compressed = '\n'.join(compressed_parts)
            if len(compressed) > target_size:
                # Final trim if still too long
                compressed = compressed[:target_size-100] + "\n...[content compressed for focus]"
            
            return compressed

        original_user_prompt, tool_calls_this_turn, sources_this_turn = prompt, [], []
        asset_store: Dict[str, Dict] = {}
        decision_history = []  # Track all decisions made
        
        # Enhanced planning phase
        planning_step_id = log_event_fn("📋 Creating adaptive execution plan...", MSG_TYPE.MSG_TYPE_STEP_START)
        execution_plan = planner.decompose_task(original_user_prompt, context or "", "\n".join([f"{tool['name']}:{tool['description']}" for tool in all_visible_tools]))
        current_plan_version = 1
        
        log_event_fn(f"Initial plan created with {len(execution_plan.tasks)} tasks", MSG_TYPE.MSG_TYPE_INFO, meta={
            "plan_version": current_plan_version,
            "total_tasks": len(execution_plan.tasks),
            "estimated_complexity": "medium" if len(execution_plan.tasks) <= 5 else "high"
        })
        
        for i, task in enumerate(execution_plan.tasks):
            log_event_fn(f"Task {i+1}: {task.description}", MSG_TYPE.MSG_TYPE_INFO)
            
        log_event_fn("✅ Adaptive plan ready", MSG_TYPE.MSG_TYPE_STEP_END, event_id=planning_step_id)
        
        # Enhanced initial state
        initial_state_parts = [
            f"### Original User Request\n{original_user_prompt}",
            f"### Context\n{context or 'No additional context provided'}",
            f"### Execution Plan (Version {current_plan_version})\n- Total tasks: {len(execution_plan.tasks)}",
            f"- Estimated complexity: {'High' if len(execution_plan.tasks) > 5 else 'Medium'}"
        ]
        
        for i, task in enumerate(execution_plan.tasks): 
            initial_state_parts.append(f"  {i+1}. {task.description} [Status: {task.status.value}]")
        
        if images:
            initial_state_parts.append(f"### Provided Assets")
            for img_b64 in images:
                img_uuid = str(uuid.uuid4())
                asset_store[img_uuid] = {"type": "image", "content": img_b64, "source": "user"}
                initial_state_parts.append(f"- Image asset: {img_uuid}")
        
        current_scratchpad = "\n".join(initial_state_parts)
        log_event_fn("Initial analysis complete", MSG_TYPE.MSG_TYPE_SCRATCHPAD, meta={"scratchpad_size": len(current_scratchpad)})

        formatted_tools_list = "\n".join([f"**{t['name']}**: {t['description']}" for t in all_visible_tools])
        completed_tasks, current_task_index = set(), 0
        plan_revision_count = 0
        
        # Main reasoning loop with enhanced decision tracking
        for i in range(max_reasoning_steps):
            current_task_desc = execution_plan.tasks[current_task_index].description if current_task_index < len(execution_plan.tasks) else "Finalizing analysis"
            step_desc = f"🤔 Step {i+1}: {current_task_desc}"
            reasoning_step_id = log_event_fn(step_desc, MSG_TYPE.MSG_TYPE_STEP_START)
            
            try:
                # Enhanced scratchpad management
                if len(current_scratchpad) > max_scratchpad_size:
                    log_event_fn(f"Scratchpad size ({len(current_scratchpad)}) exceeds limit, compressing...", MSG_TYPE.MSG_TYPE_INFO)
                    current_scratchpad = _compress_scratchpad_intelligently(current_scratchpad, original_user_prompt, max_scratchpad_size // 2)
                
                # Enhanced reasoning prompt with better decision tracking
                reasoning_prompt = f"""You are working on: "{original_user_prompt}"

=== AVAILABLE ACTIONS ===
{formatted_tools_list}

=== YOUR COMPLETE ANALYSIS HISTORY ===
{current_scratchpad}
=== END ANALYSIS HISTORY ===

=== DECISION GUIDELINES ===
1. **Review your progress**: Look at what you've already discovered and accomplished
2. **Consider your current task**: Focus on the next logical step in your plan
3. **Remember your decisions**: If you previously decided to use a tool, follow through unless you have a good reason to change
4. **Be adaptive**: If you discover new information that changes the situation, consider revising your plan
5. **Stay focused**: Each action should clearly advance toward the final goal

=== YOUR NEXT DECISION ===
Choose the single most appropriate action to take right now. Consider:
- What specific step are you currently working on?
- What information do you still need?
- What would be most helpful for the user?

Provide your decision as JSON:
{{
    "reasoning": "Explain your current thinking and why this action makes sense now",
    "action": {{
        "tool_name": "exact_tool_name",
        "requires_code_input": false,
        "requires_image_input": false,
        "confidence": 80
    }},
    "plan_status": "on_track" // or "needs_revision" if you want to change the plan
}}"""

                log_prompt_fn(f"Reasoning Prompt Step {i+1}", reasoning_prompt)
                decision_data = self.generate_structured_content(
                    prompt=reasoning_prompt, 
                    schema={
                        "reasoning": "string", 
                        "action": "object",
                        "plan_status": "string"
                    }, 
                    system_prompt=reasoning_system_prompt, 
                    temperature=decision_temperature, 
                    **llm_generation_kwargs
                )
                
                if not (decision_data and isinstance(decision_data.get("action"), dict)):
                    log_event_fn("⚠️ Invalid decision format from AI", MSG_TYPE.MSG_TYPE_WARNING, event_id=reasoning_step_id)
                    current_scratchpad += f"\n\n### Step {i+1}: Decision Error\n- Error: AI produced invalid decision JSON\n- Continuing with fallback approach"
                    continue

                action = decision_data.get("action", {})
                reasoning = decision_data.get("reasoning", "No reasoning provided")
                plan_status = decision_data.get("plan_status", "on_track")
                tool_name = action.get("tool_name")
                requires_code = action.get("requires_code_input", False)
                requires_image = action.get("requires_image_input", False)
                confidence = action.get("confidence", 50)
                
                # Track the decision
                decision_history.append({
                    "step": i+1,
                    "tool_name": tool_name,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "plan_status": plan_status
                })
                
                current_scratchpad += f"\n\n### Step {i+1}: Decision & Reasoning\n**Reasoning**: {reasoning}\n**Chosen Action**: {tool_name}\n**Confidence**: {confidence}\n**Plan Status**: {plan_status}"
                
                log_event_fn(_get_friendly_action_description(tool_name, requires_code, requires_image), MSG_TYPE.MSG_TYPE_STEP, meta={
                    "tool_name": tool_name,
                    "confidence": confidence,
                    "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                })
                
                # Handle plan revision
                if plan_status == "needs_revision" and tool_name != "local_tools::revise_plan":
                    log_event_fn("🔄 AI indicates plan needs revision", MSG_TYPE.MSG_TYPE_INFO)
                    tool_name = "local_tools::revise_plan"  # Force plan revision
                
                # Handle final answer
                if tool_name == "local_tools::final_answer": 
                    log_event_fn("🎯 Ready to provide final answer", MSG_TYPE.MSG_TYPE_INFO)
                    break
                
                # Handle clarification request
                if tool_name == "local_tools::request_clarification":
                    clarification_prompt = f"""Based on your analysis, what specific information do you need from the user?

CURRENT ANALYSIS:
{current_scratchpad}

Generate a clear, specific question that will help you proceed effectively:"""
                    
                    question = self.generate_text(clarification_prompt, temperature=0.3)
                    question = self.remove_thinking_blocks(question)
                    
                    log_event_fn("❓ Clarification needed from user", MSG_TYPE.MSG_TYPE_INFO)
                    return {
                        "final_answer": question, 
                        "clarification_required": True, 
                        "final_scratchpad": current_scratchpad,
                        "tool_calls": tool_calls_this_turn, 
                        "sources": sources_this_turn, 
                        "error": None,
                        "decision_history": decision_history
                    }

                # Handle plan revision logic
                if tool_name == "local_tools::revise_plan":
                    plan_revision_count += 1
                    revision_id = log_event_fn(f"📝 Revising execution plan (revision #{plan_revision_count})", MSG_TYPE.MSG_TYPE_STEP_START)
                    
                    try:
                        revision_prompt = f"""Based on your current analysis and discoveries, create an updated execution plan.

ORIGINAL REQUEST: "{original_user_prompt}"
CURRENT ANALYSIS:
{current_scratchpad}

REASON FOR REVISION: {reasoning}

Create a new plan that reflects your current understanding. Consider:
1. What have you already accomplished?
2. What new information have you discovered?
3. What steps are still needed?
4. How can you be more efficient?

Provide your revision as JSON:
{{
    "revision_reason": "Clear explanation of why the plan needed to change",
    "new_plan": [
        {{"step": 1, "description": "First revised step", "status": "pending"}},
        {{"step": 2, "description": "Second revised step", "status": "pending"}}
    ],
    "confidence": 0.8
}}"""

                        revision_data = self.generate_structured_content(
                            prompt=revision_prompt,
                            schema={
                                "revision_reason": "string",
                                "new_plan": "array", 
                                "confidence": "number"
                            },
                            temperature=0.3,
                            **llm_generation_kwargs
                        )
                        
                        if revision_data and revision_data.get("new_plan"):
                            # Update the plan
                            current_plan_version += 1
                            new_tasks = []
                            for task_data in revision_data["new_plan"]:
                                task = SubTask(
                                    id=str(uuid.uuid4()),
                                    description=task_data.get("description", "Undefined step"),
                                    status=TaskStatus.PENDING
                                )
                                new_tasks.append(task)
                            
                            execution_plan.tasks = new_tasks
                            current_task_index = 0  # Reset to beginning
                            
                            # Update scratchpad with new plan
                            current_scratchpad += f"\n\n### Updated Plan (Version {current_plan_version})\n"
                            current_scratchpad += f"**Revision Reason**: {revision_data.get('revision_reason', 'Plan needed updating')}\n"
                            current_scratchpad += f"**New Tasks**:\n"
                            for i, task in enumerate(execution_plan.tasks):
                                current_scratchpad += f"  {i+1}. {task.description}\n"
                            
                            log_event_fn(f"✅ Plan revised with {len(execution_plan.tasks)} updated tasks", MSG_TYPE.MSG_TYPE_STEP_END, event_id=revision_id, meta={
                                "plan_version": current_plan_version,
                                "new_task_count": len(execution_plan.tasks),
                                "revision_reason": revision_data.get("revision_reason", "")
                            })
                            continue
                        else:
                            raise ValueError("Failed to generate valid plan revision")
                            
                    except Exception as e:
                        log_event_fn(f"Plan revision failed: {e}", MSG_TYPE.MSG_TYPE_WARNING, event_id=revision_id)
                        current_scratchpad += f"\n**Plan Revision Failed**: {str(e)}\nContinuing with original plan."

                # Prepare parameters for tool execution
                param_assets = {}
                if requires_code:
                    log_event_fn("💻 Generating code for task", MSG_TYPE.MSG_TYPE_INFO)
                    code_prompt = f"""Generate the specific code needed for the current step.

CURRENT CONTEXT:
{current_scratchpad}

CURRENT TASK: {tool_name}
USER REQUEST: "{original_user_prompt}"

Generate clean, functional code that addresses the specific requirements.

CODE:"""

                    code_content = self.generate_code(prompt=code_prompt, **llm_generation_kwargs)
                    code_uuid = f"code_asset_{uuid.uuid4()}"
                    asset_store[code_uuid] = {"type": "code", "content": code_content}
                    param_assets['code_asset_id'] = code_uuid
                    log_event_fn(f"Code asset created: {code_uuid[:8]}...", MSG_TYPE.MSG_TYPE_INFO)
                    
                if requires_image:
                    image_assets = [asset_id for asset_id, asset in asset_store.items() if asset['type'] == 'image' and asset.get('source') == 'user']
                    if image_assets:
                        param_assets['image_asset_id'] = image_assets[0]
                        log_event_fn(f"Using image asset: {image_assets[0][:8]}...", MSG_TYPE.MSG_TYPE_INFO)
                    else:
                        log_event_fn("⚠️ Image required but none available", MSG_TYPE.MSG_TYPE_WARNING)

                # Enhanced parameter generation
                param_prompt = f"""Generate the optimal parameters for this tool execution.

TOOL: {tool_name}
CURRENT CONTEXT: {current_scratchpad}
CURRENT REASONING: {reasoning}
AVAILABLE ASSETS: {json.dumps(param_assets) if param_assets else "None"}

Based on your analysis and the current step you're working on, provide the most appropriate parameters.
Output format: {{"tool_params": {{...}}}}"""

                log_prompt_fn(f"Parameter Generation Step {i+1}", param_prompt)
                param_data = self.generate_structured_content(
                    prompt=param_prompt, 
                    schema={"tool_params": "object"}, 
                    temperature=decision_temperature, 
                    **llm_generation_kwargs
                )
                tool_params = param_data.get("tool_params", {}) if param_data else {}
                
                # Hydrate parameters with assets
                def _hydrate(data: Any, store: Dict) -> Any:
                    if isinstance(data, dict): return {k: _hydrate(v, store) for k, v in data.items()}
                    if isinstance(data, list): return [_hydrate(item, store) for item in data]
                    if isinstance(data, str) and data.startswith("asset_") and data in store: return store[data].get("content", data)
                    return data
                    
                hydrated_params = _hydrate(tool_params, asset_store)
                
                # Execute tool
                start_time = time.time()
                tool_result = {"status": "failure", "error": f"Tool '{tool_name}' failed to execute."}
                
                try:
                    if tool_name in rag_registry:
                        query = hydrated_params.get("query", original_user_prompt)
                        log_event_fn(f"🔍 Searching knowledge base: '{query[:50]}...'", MSG_TYPE.MSG_TYPE_INFO)
                        
                        top_k = rag_tool_specs.get(tool_name, {}).get("default_top_k", 5)
                        min_sim = rag_tool_specs.get(tool_name, {}).get("default_min_sim", 50.0)
                        
                        raw_results = rag_registry[tool_name](query=query, rag_top_k=top_k)
                        raw_iter = raw_results["results"] if isinstance(raw_results, dict) and "results" in raw_results else raw_results
                        
                        docs = []
                        for d in raw_iter or []:
                            doc_data = {"text": d.get("text", str(d)), "score": d.get("score", 0) * 100, "metadata": d.get("metadata", {}), "title": d.get("title", "Unknown Source")}
                            docs.append(doc_data)
                        
                        kept = [x for x in docs if x['score'] >= min_sim]
                        tool_result = {"status": "success", "results": kept, "query_used": query}
                        
                        sources_this_turn.extend([{"source": tool_name, "metadata": x["metadata"], "score": x["score"], "title": x["title"]} for x in kept])
                        
                    elif hasattr(self, "mcp") and "local_tools" not in tool_name:
                        log_event_fn(f"🔧 Executing MCP tool: {tool_name}", MSG_TYPE.MSG_TYPE_TOOL_CALL, meta={"tool_name": tool_name})
                        tool_result = self.mcp.execute_tool(tool_name, hydrated_params, lollms_client_instance=self)
                        
                    elif tool_name == "local_tools::generate_image" and hasattr(self, "tti"):
                        image_prompt = hydrated_params.get("prompt", "")
                        image_result = self.tti.generate_image(image_prompt)
                        if image_result:
                            image_uuid = f"generated_image_{uuid.uuid4()}"
                            asset_store[image_uuid] = {"type": "image", "content": image_result, "source": "generated"}
                            tool_result = {"status": "success", "image_id": image_uuid}
                        else:
                            tool_result = {"status": "failure", "error": "Image generation failed"}
                except Exception as e:
                    tool_result = {"status": "failure", "error": str(e)}

                response_time = time.time() - start_time
                success = tool_result.get("status") == "success"
                
                # Update task status
                if success and current_task_index < len(execution_plan.tasks):
                    execution_plan.tasks[current_task_index].status = TaskStatus.COMPLETED
                    completed_tasks.add(current_task_index)
                    current_task_index += 1
                
                current_scratchpad += f"\n\n### Step {i+1}: Observation\n**Tool**: {tool_name}\n**Success**: {success}\n**Result**:\n```json\n{json.dumps(tool_result, indent=2)[:2000]}\n```"
                
                tool_calls_this_turn.append({"name": tool_name, "params": tool_params, "result": tool_result, "response_time": response_time})
                
                if success:
                    log_event_fn(f"✅ Step {i+1} completed", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id)
                else:
                    log_event_fn(f"⚠️ Step {i+1} issues: {tool_result.get('error')}", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id)
                
                if len(completed_tasks) >= len(execution_plan.tasks): break
                    
            except Exception as ex:
                log_event_fn(f"💥 Error in reasoning: {str(ex)}", MSG_TYPE.MSG_TYPE_ERROR, event_id=reasoning_step_id)
                current_scratchpad += f"\n\n### Step {i+1}: Error\n{str(ex)}"
                log_event_fn("🔄 Continuing...", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reasoning_step_id)

        # Enhanced self-reflection
        if enable_self_reflection and len(tool_calls_this_turn) > 0:
            reflection_id = log_event_fn("🤔 Self-reflection assessment...", MSG_TYPE.MSG_TYPE_STEP_START)
            try:
                reflection_prompt = f"Review your work for: \"{original_user_prompt}\"\nANALYSIS:\n{current_scratchpad}\nEvaluate effectiveness and goal achievement in JSON format."
                reflection_data = self.generate_structured_content(prompt=reflection_prompt, schema={"goal_achieved": "boolean", "summary": "string"}, temperature=0.3, **llm_generation_kwargs)
                if reflection_data:
                    current_scratchpad += f"\n\n### Self-Reflection\n**Achieved**: {reflection_data.get('goal_achieved')}\n**Summary**: {reflection_data.get('summary')}"
                log_event_fn("✅ Reflection complete", MSG_TYPE.MSG_TYPE_STEP_END, event_id=reflection_id)
            except:
                log_event_fn("Reflection failed", MSG_TYPE.MSG_TYPE_WARNING, event_id=reflection_id)

        # Final synthesis
        synthesis_id = log_event_fn("📝 Final synthesis...", MSG_TYPE.MSG_TYPE_STEP_START)
        final_answer_prompt = f"User Request: \"{original_user_prompt}\"\nWork History:\n{current_scratchpad}\nSynthesize final answer for the user."
        final_answer_text = self.generate_text(prompt=final_answer_prompt, system_prompt=system_prompt, stream=streaming_callback is not None, streaming_callback=streaming_callback, temperature=final_answer_temperature, **llm_generation_kwargs)
        final_answer = self.remove_thinking_blocks(final_answer_text)
        log_event_fn("✅ Response ready", MSG_TYPE.MSG_TYPE_STEP_END, event_id=synthesis_id)

        return {
            "final_answer": final_answer,
            "final_scratchpad": current_scratchpad,
            "tool_calls": tool_calls_this_turn,
            "sources": sources_this_turn,
            "decision_history": decision_history,
            "clarification_required": False,
            "error": None
        }

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
