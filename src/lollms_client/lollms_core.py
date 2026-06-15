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

        debug: Optional[bool] = True,
        cooperative_vram_management: Optional[bool] = False,
        **kwargs
        ):
        """
        Initialize the LollmsClient with LLM and optional modality bindings.
        """

        self.debug = debug
        if not self.debug:
            import logging
            logging.getLogger("ASCIIColors").setLevel(logging.WARNING)
            from ascii_colors import ASCIIColors
            for method_name in ["info", "success", "cyan", "blue", "green", "panel"]:
                if hasattr(ASCIIColors, method_name):
                    setattr(ASCIIColors, method_name, lambda *args, **kwargs: None)

        self.cooperative_vram_management = cooperative_vram_management
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
            config['debug'] = self.debug
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
                tti_config = (tti_binding_config or {}).copy()
                tti_config['debug'] = self.debug
                self.tti = self.tti_binding_manager.create_binding(binding_name=tti_binding_name, **tti_config)
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

        # In-memory MD5 token-count caching to prevent redundant backend server floods
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
        if not hasattr(self, "_token_count_cache"):
            self._token_count_cache = {}

        if text_hash in self._token_count_cache:
            return self._token_count_cache[text_hash]

        if self.llm: 
            count = self.llm.count_tokens(text)
            self._token_count_cache[text_hash] = count
            return count
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

    def _cooperative_unload_except(self, active_modality: str):
        if not getattr(self, "cooperative_vram_management", False):
            return

        modalities = {
            "llm": self.llm,
            "tts": self.tts,
            "tti": self.tti,
            "stt": self.stt,
            "ttv": self.ttv,
            "ttm": self.ttm,
        }

        for name, binding in modalities.items():
            if name != active_modality and binding:
                ASCIIColors.info(f"[Cooperative VRAM] Unloading {name.upper()} model to free VRAM for {active_modality.upper()}...")
                try:
                    binding.unload_model()
                except Exception as e:
                    ASCIIColors.warning(f"Failed to unload {name.upper()} model: {e}")

    def _cooperative_unload_tti(self):
        self._cooperative_unload_except("llm")

    def _cooperative_unload_llm(self):
        self._cooperative_unload_except("tti")

    def generate_text(self, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_except("llm")
        if not self.llm:
            raise RuntimeError("LLM binding not initialized.")

        # Default think to False if not explicitly provided as True
        if "think" not in kwargs:
            kwargs["think"] = False
        else:
            kwargs["think"] = kwargs["think"] is True

        return self.llm.generate_text(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Union[str, dict]:
        return self.generate_text(*args, **kwargs)

    def generate_from_messages(self, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_except("llm")
        if not self.llm:
            raise RuntimeError("LLM binding not initialized.")

        # Default think to False if not explicitly provided as True
        if "think" not in kwargs:
            kwargs["think"] = False
        else:
            kwargs["think"] = kwargs["think"] is True

        return self.llm.generate_from_messages(*args, **kwargs)

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: int = 4096,
        max_tool_rounds: int = 10,
        streaming_callback: Optional[Callable] = None,
        auto_execute: bool = True,
        **extra,
    ) -> Dict[str, Any]:
        """
        Generate a response with access to tools (file-based or inline).

        Parameters
        ----------
        prompt : str
            The user prompt / task description.
        tools : list
            Mixed list of:
              • ``str`` or ``Path`` — file path to a lollms-format tool script
              • ``dict`` — inline tool spec with ``{"name": ..., "callable": ..., ...}``
        system_prompt : str
            Optional system prompt override.
        temperature : float
            Sampling temperature.
        n_predict : int
            Max tokens per generation.
        max_tool_rounds : int
            Maximum agentic tool-call loops before forcing final answer.
        streaming_callback : callable
            Optional streaming callback ``(chunk, msg_type, meta) -> bool``.
        auto_execute : bool
            If True, automatically execute tool calls and feed results back.

        Returns
        -------
        dict
            {
                "response": str,           # Final text response
                "tool_calls": list,        # All tool calls made
                "tool_results": list,      # All tool execution results
                "rounds": int,             # Number of agentic rounds
            }
        """
        from ascii_colors import ASCIIColors
        from lollms_client.lollms_agent import ToolsManager

        if self.llm is None:
            raise RuntimeError("LLM binding not initialized.")

        # ── 1. Build unified tool registry ──────────────────────────────
        tools_mgr = ToolsManager()
        inline_tools = tools_mgr.build_inline_tools_dict(tools)

        if not inline_tools:
            # No valid tools — fall back to plain generation
            return {
                "response": self.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    n_predict=n_predict,
                    streaming_callback=streaming_callback,
                    **extra,
                ),
                "tool_calls": [],
                "tool_results": [],
                "rounds": 0,
            }

        # ── 2. Build tool descriptions for the system prompt ──────────────
        tool_descriptions: List[str] = []
        for name, spec in inline_tools.items():
            params = spec.get("parameters", [])
            param_str = ", ".join(
                f"{p['name']}: {p['type']}" + (" (optional)" if p.get("optional") else "")
                for p in params
            )
            desc = spec.get("description", f"Execute {name}")
            tool_descriptions.append(f"- {name}({param_str}): {desc}")

        tool_header = (
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  TOOL USE — MANDATORY FORMAT                                     ║\n"
            "╠══════════════════════════════════════════════════════════════════╣\n"
            "║  You have external tools. To use one you MUST use EXACTLY this  ║\n"
            "║  format — copy the pattern below character-for-character:        ║\n"
            "║                                                                  ║\n"
            "║    <tool_call>{\"name\": \"tool_name\",                              ║\n"
            "║                \"parameters\": {\"key\": \"value\"}}</tool_call>       ║\n"
            "║                                                                  ║\n"
            "║  CRITICAL:                                                       ║\n"
            "║    • The ENTIRE tool call must be wrapped in <tool_call> tags    ║\n"
            "║    • NO markdown code fences (no ```json)                        ║\n"
            "║    • NO raw JSON without the XML wrapper                         ║\n"
            "║    • NO explanations before or after the tool call               ║\n"
            "║    • ONLY the <tool_call> line when calling a tool               ║\n"
            "║                                                                  ║\n"
            "║  Rules:                                                          ║\n"
            "║    • One tool call per response turn.                            ║\n"
            "║    • After calling ALL needed tools, write your final answer.    ║\n"
            "║    • If the user explicitly asks you to use a tool, USE IT.      ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n\n"
            "TOOLS AVAILABLE:\n"
        )

        tool_block = tool_header + "\n".join(tool_descriptions)

        # ── 3. Prepare conversation state ─────────────────────────────────
        full_system = system_prompt.rstrip()
        if full_system:
            full_system += "\n\n"
        full_system += tool_block

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt},
        ]

        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        rounds = 0

        # ── 4. Agentic loop ───────────────────────────────────────────────
        while rounds < max_tool_rounds:
            rounds += 1

            # Generate response
            gen_kwargs: Dict[str, Any] = {
                "temperature": temperature,
                "n_predict": n_predict,
                **extra,
            }
            if streaming_callback:
                gen_kwargs["streaming_callback"] = streaming_callback

            try:
                raw_response = self.generate_from_messages(
                    messages=conversation,
                    **gen_kwargs,
                )
            except Exception as e:
                if self.debug:
                    trace_exception(e)
                ASCIIColors.error(f"generate_with_tools: generation failed: {e}")
                return {
                    "response": f"[Error during generation: {e}]",
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "rounds": rounds,
                }

            if not isinstance(raw_response, str):
                raw_response = str(raw_response) if raw_response is not None else ""

            # ── 5. Parse tool calls ─────────────────────────────────────────
            # Primary: XML-wrapped tool calls <tool_call>...</tool_call>
            tool_call_pattern = re.compile(
                r'<tool_call>(.*?)</tool_call>',
                re.DOTALL | re.IGNORECASE,
            )
            matches = list(tool_call_pattern.finditer(raw_response))

            # Fallback: detect raw JSON tool calls (models sometimes omit XML tags)
            tool_json_str = None
            visible_response = raw_response.strip()

            if matches:
                # Extract the first tool call (one per turn)
                match = matches[0]
                tool_json_str = match.group(1).strip()
                visible_response = raw_response[:match.start()].strip()
            else:
                # Try to detect raw JSON that looks like a tool call
                # Pattern: {"name": "tool_...", "parameters": {...}}
                json_obj_pattern = re.compile(
                    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\}',
                    re.DOTALL,
                )

                json_match = json_obj_pattern.search(raw_response)
                if json_match:
                    tool_json_str = json_match.group(0).strip()
                    # Determine visible response (text before the JSON object)
                    json_start = json_match.start()
                    visible_response = raw_response[:json_start].strip()
                    ASCIIColors.warning(
                        f"Model emitted raw JSON tool call (missing <tool_call> tags). "
                        f"Tool: {json_match.group(1)}"
                    )

            if not tool_json_str:
                # No tool call — this is the final answer
                cleaned = tool_call_pattern.sub('', raw_response).strip()
                return {
                    "response": cleaned,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "rounds": rounds,
                }

            # ALWAYS add assistant message to maintain strict user/assistant
            # alternation required by llama.cpp Jinja chat templates.
            # Even if visible_response is empty, the assistant "spoke" (the tool call).
            conversation.append({"role": "assistant", "content": visible_response})

            # Parse tool call JSON
            try:
                call_data = json.loads(tool_json_str)
            except json.JSONDecodeError as e:
                ASCIIColors.warning(f"Failed to parse tool call JSON: {e}")
                conversation.append({
                    "role": "user",
                    "content": f"Error: Invalid tool call JSON. {e}",
                })
                continue

            tool_name = call_data.get("name", "")
            tool_params = call_data.get("parameters", {})

            call_record = {
                "round": rounds,
                "name": tool_name,
                "parameters": tool_params,
                "raw": tool_json_str,
            }
            all_tool_calls.append(call_record)

            if not auto_execute:
                # Manual mode: return the tool call for external handling
                return {
                    "response": visible_response,
                    "tool_calls": all_tool_calls,
                    "tool_results": all_tool_results,
                    "pending_tool": call_record,
                    "rounds": rounds,
                }

            # ── 6. Execute tool ─────────────────────────────────────────────
            if tool_name not in inline_tools:
                error_msg = f"Error: Tool '{tool_name}' not found in registry."
                ASCIIColors.warning(error_msg)
                result = {"error": error_msg, "success": False}
            else:
                tool_spec = inline_tools[tool_name]
                fn = tool_spec.get("callable")
                if not callable(fn):
                    error_msg = f"Error: Tool '{tool_name}' has no callable."
                    ASCIIColors.warning(error_msg)
                    result = {"error": error_msg, "success": False}
                else:
                    try:
                        # Normalize parameters: lollms-format tools use `args: dict`
                        # but some inline tools may use kwargs. Try kwargs first,
                        # fall back to single dict arg if signature mismatch.
                        try:
                            result = fn(**tool_params)
                        except TypeError as te:
                            if "unexpected keyword argument" in str(te):
                                result = fn(tool_params)
                            else:
                                raise

                        # Normalize result to dict if it's a plain string
                        if isinstance(result, str):
                            result = {"output": result, "success": True}
                        elif not isinstance(result, dict):
                            result = {"output": str(result), "success": True}

                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {e}"
                        if self.debug:
                            trace_exception(e)
                            ASCIIColors.warning(error_msg)
                        result = {"error": error_msg, "success": False}

            result_record = {
                "round": rounds,
                "name": tool_name,
                "result": result,
            }
            all_tool_results.append(result_record)

            # Format result for LLM context
            if isinstance(result, dict) and result.get("success"):
                result_text = result.get("output", json.dumps(result, indent=2))
            else:
                result_text = json.dumps(result, indent=2, ensure_ascii=False)

            # Truncate very large results
            max_result_len = 4000
            if len(result_text) > max_result_len:
                result_text = result_text[:max_result_len] + f"\n... [{len(result_text) - max_result_len} chars truncated]"

            # Add tool result to conversation
            conversation.append({
                "role": "user",
                "content": (
                    f'<tool_result name="{tool_name}">\n'
                    f"{result_text}\n"
                    f"</tool_result>"
                ),
            })

        # ── 7. Max rounds exceeded — force final answer ───────────────────
        ASCIIColors.warning(f"generate_with_tools: max rounds ({max_tool_rounds}) exceeded")
        conversation.append({
            "role": "user",
            "content": (
                "[SYSTEM] Maximum tool rounds reached. "
                "Provide your final answer now without calling any more tools."
            ),
        })

        try:
            final_response = self.generate_from_messages(
                messages=conversation,
                temperature=temperature,
                n_predict=n_predict,
                **{k: v for k, v in extra.items() if k not in ("temperature", "n_predict")},
            )
        except Exception as e:
            final_response = f"[Error generating final answer: {e}]"

        cleaned = tool_call_pattern.sub('', str(final_response)).strip()
        return {
            "response": cleaned,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "rounds": rounds,
        }

    def chat(self, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_tti()
        if self.llm:
            # Log image payload status at core client layer
            images = kwargs.get("images")
            if images is not None:
                ASCIIColors.info(f"[LollmsClient.chat] Forwarding 'images' to binding: count={len(images)}, types={[type(img).__name__ for img in images[:5]]}")
            else:
                ASCIIColors.warning("[LollmsClient.chat] No 'images' parameter found in kwargs")
            return self.llm.chat(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def embed(self, *args, **kwargs):
        if self.llm: return self.llm.embed(*args, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
    def get_ctx_size(self, model_name: Optional[str] = None) -> Optional[int]:
        """
        Retrieves the context size for the active model.
        Delegates directly to the active LLM binding.
        """
        if self.llm:
            return self.llm.get_ctx_size(model_name)
        return 4096

    def list_models(self):
        models = []
        if self.llm: models += self.llm.list_models()
        if self.tti: models +=  self.tti.list_models()
        if self.tts: models +=  self.tts.list_models()
        if self.stt: models +=  self.stt.list_models()
        return models

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
        self._cooperative_unload_except("tti")
        if self.tti: return self.tti.generate_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def edit_image(self, *args, **kwargs):
        self._cooperative_unload_except("tti")
        if self.tti: return self.tti.edit_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def generate_audio(self, *args, **kwargs):
        self._cooperative_unload_except("tts")
        if self.tts: return self.tts.generate_audio(*args, **kwargs)
        raise RuntimeError("TTS binding not initialized.")

    def transcribe_audio(self, *args, **kwargs):
        self._cooperative_unload_except("stt")
        if self.stt: return self.stt.transcribe_audio(*args, **kwargs)
        raise RuntimeError("STT binding not initialized.")

    def generate_video(self, *args, **kwargs):
        self._cooperative_unload_except("ttv")
        if self.ttv: return self.ttv.generate_video(*args, **kwargs)
        raise RuntimeError("TTV binding not initialized.")

    def generate_music(self, *args, **kwargs):
        self._cooperative_unload_except("ttm")
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
