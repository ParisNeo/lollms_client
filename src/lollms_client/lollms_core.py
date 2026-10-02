# lollms_client/lollms_core.py
# author: ParisNeo
# description: LollmsClient definition file
import requests
import json
import re
import base64
import os
import numpy as np
import uuid
import hashlib
import time
import warnings
from pathlib import Path
from enum import Enum
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
from dataclasses import dataclass, field
import urllib3
import ascii_colors as logging
from ascii_colors import ASCIIColors, trace_exception

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request is being made")
logging.getLogger("urllib3").setLevel(logging.ERROR)
from lollms_client.lollms_types import MSG_TYPE, ELF_COMPLETION_FORMAT
from lollms_client.lollms_utilities import robust_json_parser, build_image_dicts, dict_to_markdown
from lollms_client.lollms_llm_binding import LollmsLLMBinding, LollmsLLMBindingManager
from lollms_client.lollms_tts_binding import LollmsTTSBinding, LollmsTTSBindingManager
from lollms_client.lollms_tti_binding import LollmsTTIBinding, LollmsTTIBindingManager
from lollms_client.lollms_stt_binding import LollmsSTTBinding, LollmsSTTBindingManager
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, LollmsTTVBindingManager
from lollms_client.lollms_ttm_binding import LollmsTTMBinding, LollmsTTMBindingManager
from lollms_client.lollms_tools_binding import LollmsToolBinding, LollmsTOOLBindingManager
from lollms_client.lollms_connection_binding import LollmsConnectionBinding, LollmsConnectionBindingManager
try:
    from lollms_client.lollms_rag_binding import LollmsRAGBinding, LollmsRAGBindingManager
except ImportError:
    LollmsRAGBinding = None
    LollmsRAGBindingManager = None
from lollms_client.lollms_personality.lollms_personality import ToolsManager
from lollms_client.lollms_base_binding import LollmsBaseBinding

from lollms_client.lollms_discussion import LollmsDiscussion

@dataclass
class LollmsBindingProfile:
    """
    Declarative profile for a modality binding engine/server (Connection Layer).
    Defines the backend connection (e.g., host, api_key, binding library).
    """
    name: str
    binding_name: str
    binding_config: Dict[str, Any] = field(default_factory=dict)
    is_default: bool = False

@dataclass
class LollmsModelProfile:
    """
    Declarative profile for a specific model (Execution Layer).
    References a binding profile and defines model-specific execution parameters.
    """
    name: str
    binding_profile_name: str
    model_name: Optional[str] = None
    is_default: bool = False
    vision_enabled: bool = False
    forced_context_size: Optional[int] = None
    routing_config: Optional[Dict[str, Any]] = None
    glm_image_embedding: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None
    video_enabled: bool = False
    supported_reasoning_efforts: Optional[List[str]] = None



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
        connection_binding_name: Optional[str] = None,
        rag_binding_name: Optional[str] = None,

        # Modality Binding Directories
        llm_bindings_dir: Path = Path(__file__).parent / "llm_bindings",
        tts_bindings_dir: Path = Path(__file__).parent / "tts_bindings",
        tti_bindings_dir: Path = Path(__file__).parent / "tti_bindings",
        stt_bindings_dir: Path = Path(__file__).parent / "stt_bindings",
        ttv_bindings_dir: Path = Path(__file__).parent / "ttv_bindings",
        ttm_bindings_dir: Path = Path(__file__).parent / "ttm_bindings",
        tools_bindings_dir: Path = Path(__file__).parent / "tools_bindings",
        connection_bindings_dir: Path = Path(__file__).parent / "connection_bindings",
        rag_bindings_dir: Path = Path(__file__).parent / "rag_bindings",

        # Configurations
        llm_binding_config: Optional[Dict[str, any]] = None,
        tts_binding_config: Optional[Dict[str, any]] = None, 
        tti_binding_config: Optional[Dict[str, any]] = None, 
        stt_binding_config: Optional[Dict[str, any]] = None, 
        ttv_binding_config: Optional[Dict[str, any]] = None, 
        ttm_binding_config: Optional[Dict[str, any]] = None, 
        tools_binding_config: Optional[Dict[str, any]] = None,
        connection_binding_config: Optional[Dict[str, any]] = None,
        rag_binding_config: Optional[Dict[str, any]] = None,
        user_name ="user",
        ai_name = "assistant",
        callback: Optional[Callable[[str, MSG_TYPE, Optional[Dict]], bool]] = None,

        debug: Optional[bool] = True,
        cooperative_vram_management: Optional[bool] = False,

        # 🧠 Modern Lazy Profiles (Universal across all modalities)
        llm_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        tti_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        tts_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        stt_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        ttv_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        ttm_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        connection_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        rag_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,

        llm_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        tti_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        tts_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        stt_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        ttv_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        ttm_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        connection_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        rag_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,

        **kwargs
        ):
        """
        Initialize the LollmsClient with LLM and optional modality bindings.

        This client implements a strict, decoupled, lazy-loaded Two-Tier Profile Architecture:

        1. Connection Layer (`*_binding_profiles`): Defines the backend engine/server configuration 
           (e.g., host address, API keys, binding library name). Declared once per server.
        2. Execution Layer (`*_model_profiles`): Defines specific models, routing rules, and vision flags.
           References a binding profile via `binding_profile_name`.

        Only the model profile marked as `is_default=True` is eagerly instantiated at startup. 
        All other models are instantiated lazily on-demand when switched to, preserving RAM and VRAM.

        Backward Compatibility:
        - Legacy parameters like `llm_binding_name` and `llm_binding_config` are automatically 
          registered as the `"master"` binding and model profiles.
        - The legacy `extra_llms` kwarg is mapped into the new two-tier system automatically.

        Args:
            llm_binding_name (Optional[str]): Legacy direct LLM binding name.
            tts_binding_name (Optional[str]): Legacy direct TTS binding name.
            tti_binding_name (Optional[str]): Legacy direct TTI binding name.
            stt_binding_name (Optional[str]): Legacy direct STT binding name.
            ttv_binding_name (Optional[str]): Legacy direct TTV binding name.
            ttm_binding_name (Optional[str]): Legacy direct TTM binding name.
            tools_binding_name (Optional[str]): Legacy direct MCP/Tools binding name.

            llm_bindings_dir (Path): Directory containing LLM bindings.
            tts_bindings_dir (Path): Directory containing TTS bindings.
            tti_bindings_dir (Path): Directory containing TTI bindings.
            stt_bindings_dir (Path): Directory containing STT bindings.
            ttv_bindings_dir (Path): Directory containing TTV bindings.
            ttm_bindings_dir (Path): Directory containing TTM bindings.
            tools_bindings_dir (Path): Directory containing Tools bindings.

            llm_binding_config (Optional[Dict]): Legacy direct LLM configuration.
            tts_binding_config (Optional[Dict]): Legacy direct TTS configuration.
            tti_binding_config (Optional[Dict]): Legacy direct TTI configuration.
            stt_binding_config (Optional[Dict]): Legacy direct STT configuration.
            ttv_binding_config (Optional[Dict]): Legacy direct TTV configuration.
            ttm_binding_config (Optional[Dict]): Legacy direct TTM configuration.
            tools_binding_config (Optional[Dict]): Legacy direct Tools configuration.

            user_name (str): Name used for the user in prompt headers.
            ai_name (str): Name used for the AI in prompt headers.
            callback (Optional[Callable]): Initialization progress callback.
            debug (Optional[bool]): Enable debug logging.
            cooperative_vram_management (Optional[bool]): Unload inactive modality models to free VRAM.

            llm_binding_profiles (Optional[Dict]): Connection profiles for LLM engines.
            tti_binding_profiles (Optional[Dict]): Connection profiles for TTI engines.
            tts_binding_profiles (Optional[Dict]): Connection profiles for TTS engines.
            stt_binding_profiles (Optional[Dict]): Connection profiles for STT engines.
            ttv_binding_profiles (Optional[Dict]): Connection profiles for TTV engines.
            ttm_binding_profiles (Optional[Dict]): Connection profiles for TTM engines.

            llm_model_profiles (Optional[Dict]): Execution profiles for LLM models.
            tti_model_profiles (Optional[Dict]): Execution profiles for TTI models.
            tts_model_profiles (Optional[Dict]): Execution profiles for TTS models.
            stt_model_profiles (Optional[Dict]): Execution profiles for STT models.
            ttv_model_profiles (Optional[Dict]): Execution profiles for TTV models.
            ttm_model_profiles (Optional[Dict]): Execution profiles for TTM models.

            **kwargs: Catch-all for legacy parameters like `extra_llms`.
        """

        self.debug = debug

        system_dir_arg = kwargs.get("system_dir") or kwargs.get("lollms_system_path") or kwargs.get("cwd")
        if system_dir_arg:
            self.system_dir = Path(system_dir_arg).expanduser().resolve()
            self.system_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.system_dir = None

        self.cooperative_vram_management = cooperative_vram_management
        if callback: callback("🚀 Initializing **Lollms Client**...", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})
        
        self.llm_binding_manager = LollmsLLMBindingManager(llm_bindings_dir)
        self.tts_binding_manager = LollmsTTSBindingManager(tts_bindings_dir)
        self.tti_binding_manager = LollmsTTIBindingManager(tti_bindings_dir)
        self.stt_binding_manager = LollmsSTTBindingManager(stt_bindings_dir)
        self.ttv_binding_manager = LollmsTTVBindingManager(ttv_bindings_dir)
        self.ttm_binding_manager = LollmsTTMBindingManager(ttm_bindings_dir)
        self.tools_binding_manager = LollmsTOOLBindingManager(tools_bindings_dir)
        self.connection_binding_manager = LollmsConnectionBindingManager(connection_bindings_dir)
        self.rag_binding_manager = LollmsRAGBindingManager(rag_bindings_dir) if LollmsRAGBindingManager else None

        self.llm: Optional[LollmsLLMBinding] = None
        self.tts: Optional[LollmsTTSBinding] = None
        self.tti: Optional[LollmsTTIBinding] = None
        self.stt: Optional[LollmsSTTBinding] = None
        self.ttv: Optional[LollmsTTVBinding] = None
        self.ttm: Optional[LollmsTTMBinding] = None
        self.tools: Optional[LollmsToolBinding] = None
        self.connection: Optional[LollmsConnectionBinding] = None
        self.rag: Optional[LollmsRAGBinding] = None

        # Multi-Binding Registries (Instantiated Models)
        self.llms: Dict[str, LollmsLLMBinding] = {}
        self.ttis: Dict[str, LollmsTTIBinding] = {}
        self.tts_bindings: Dict[str, LollmsTTSBinding] = {}
        self.stts: Dict[str, LollmsSTTBinding] = {}
        self.ttvs: Dict[str, LollmsTTVBinding] = {}
        self.ttms: Dict[str, LollmsTTMBinding] = {}
        self.connections: Dict[str, LollmsConnectionBinding] = {}
        self.rags: Dict[str, LollmsRAGBinding] = {}

        self._active_llm_alias: Optional[str] = None
        self._active_tti_alias: Optional[str] = None
        self._active_tts_alias: Optional[str] = None
        self._active_stt_alias: Optional[str] = None
        self._active_ttv_alias: Optional[str] = None
        self._active_ttm_alias: Optional[str] = None
        self._active_connection_alias: Optional[str] = None
        self._active_rag_alias: Optional[str] = None

        # 🖼️ VLM Image Description Cache (Image Hash -> Text Description)
        self._image_description_cache: Dict[str, str] = {}

        # ⚡ Fast token estimation (CLI mode): bypass remote tokenizer round-trips
        self.use_fast_token_estimate: bool = False
        self._fast_token_coefficient: float = self._parse_fast_token_coefficient()

        # 🧠 Profile Registries (Declarative Configs - Universal Two-Tier Architecture)
        self.llm_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.tti_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.tts_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.stt_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.ttv_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.ttm_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.connection_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.rag_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}

        self.llm_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.tti_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.tts_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.stt_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.ttv_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.ttm_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.connection_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.rag_model_profiles_registry: Dict[str, LollmsModelProfile] = {}

        # Backward compatibility: Map legacy extra_llms to llm_model_profiles
        legacy_extra_llms = kwargs.pop("extra_llms", None)
        if legacy_extra_llms:
            if llm_model_profiles is None:
                llm_model_profiles = {}
            for alias, profile_data in legacy_extra_llms.items():
                if alias not in llm_model_profiles:
                    # Create a master binding profile if it doesn't exist
                    if "legacy_master_binding" not in (llm_binding_profiles or {}):
                        if llm_binding_profiles is None: llm_binding_profiles = {}
                        llm_binding_profiles["legacy_master_binding"] = LollmsBindingProfile(
                            name="legacy_master_binding",
                            binding_name=profile_data.get("binding_name"),
                            binding_config=profile_data.get("binding_config", {})
                        )
                    llm_model_profiles[alias] = {
                        "binding_profile_name": "legacy_master_binding",
                        "model_name": profile_data.get("binding_config", {}).get("model_name")
                    }

        # User and AI names are important for prompt construction
        self.user_name = user_name
        self.ai_name = ai_name

        # 1. Register Connection Layer Profiles (Bindings) — including CONNECTION & RAG
        self._register_binding_profiles(llm_binding_profiles, self.llm_binding_profiles_registry, "LLM", llm_binding_name, llm_binding_config)
        self._register_binding_profiles(tts_binding_profiles, self.tts_binding_profiles_registry, "TTS", tts_binding_name, tts_binding_config)
        self._register_binding_profiles(tti_binding_profiles, self.tti_binding_profiles_registry, "TTI", tti_binding_name, tti_binding_config)
        self._register_binding_profiles(stt_binding_profiles, self.stt_binding_profiles_registry, "STT", stt_binding_name, stt_binding_config)
        self._register_binding_profiles(ttv_binding_profiles, self.ttv_binding_profiles_registry, "TTV", ttv_binding_name, ttv_binding_config)
        self._register_binding_profiles(ttm_binding_profiles, self.ttm_binding_profiles_registry, "TTM", ttm_binding_name, ttm_binding_config)
        self._register_binding_profiles(connection_binding_profiles, self.connection_binding_profiles_registry, "CONNECTION", connection_binding_name, connection_binding_config)
        self._register_binding_profiles(rag_binding_profiles, self.rag_binding_profiles_registry, "RAG", rag_binding_name, rag_binding_config)

        # 2. Register Execution Layer Profiles (Models / Stores) — including CONNECTION & RAG
        self._register_model_profiles(llm_model_profiles, self.llm_model_profiles_registry, "LLM", self.llm_binding_profiles_registry)
        self._register_model_profiles(tts_model_profiles, self.tts_model_profiles_registry, "TTS", self.tts_binding_profiles_registry)
        self._register_model_profiles(tti_model_profiles, self.tti_model_profiles_registry, "TTI", self.tti_binding_profiles_registry)
        self._register_model_profiles(stt_model_profiles, self.stt_model_profiles_registry, "STT", self.stt_binding_profiles_registry)
        self._register_model_profiles(ttv_model_profiles, self.ttv_model_profiles_registry, "TTV", self.ttv_binding_profiles_registry)
        self._register_model_profiles(ttm_model_profiles, self.ttm_model_profiles_registry, "TTM", self.ttm_binding_profiles_registry)
        self._register_model_profiles(connection_model_profiles, self.connection_model_profiles_registry, "CONNECTION", self.connection_binding_profiles_registry)
        self._register_model_profiles(rag_model_profiles, self.rag_model_profiles_registry, "RAG", self.rag_binding_profiles_registry)

        # 3. Tools binding remains direct (not part of the two-tier profile system yet)
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

        # 4. Eagerly instantiate ONLY the default models for all modalities
        resolved_defaults_to_save: Dict[str, Tuple[str, List[str]]] = {}

        def _eagerly_instantiate_default(model_registry: dict, switch_method: Callable, modality_name: str):
            default_aliases = [a for a, p in model_registry.items() if p.is_default]
            if len(default_aliases) > 1:
                chosen_default = default_aliases[0]
                demoted = default_aliases[1:]
                ASCIIColors.info(
                    f"[LollmsClient] Multiple default profiles detected for {modality_name}: {default_aliases}. "
                    f"Selected '{chosen_default}' as default and demoted {demoted}."
                )
                for alias in demoted:
                    model_registry[alias].is_default = False
                resolved_defaults_to_save[modality_name.lower()] = (chosen_default, demoted)

            default_alias = next((a for a, p in model_registry.items() if p.is_default), None)
            if default_alias:
                switch_method(default_alias, callback=callback)
            elif "master" in model_registry:
                switch_method("master", callback=callback)
            elif model_registry:
                first_alias = next(iter(model_registry))
                model_registry[first_alias].is_default = True
                resolved_defaults_to_save[modality_name.lower()] = (first_alias, [])
                switch_method(first_alias, callback=callback)

        missing_model_names = [
            alias for alias, prof in self.llm_model_profiles_registry.items()
            if not prof.model_name and alias != "master"
        ]
        if missing_model_names:
            ASCIIColors.warning(
                f"[LollmsClient] LLM model profiles without a model_name: {missing_model_names}. "
                "These profiles will fall back to the server's default model."
            )

        _eagerly_instantiate_default(self.llm_model_profiles_registry, self.switch_model, "LLM")
        _eagerly_instantiate_default(self.tts_model_profiles_registry, self.switch_tts, "TTS")
        _eagerly_instantiate_default(self.tti_model_profiles_registry, self.switch_tti, "TTI")
        _eagerly_instantiate_default(self.stt_model_profiles_registry, self.switch_stt, "STT")
        _eagerly_instantiate_default(self.ttv_model_profiles_registry, self.switch_ttv, "TTV")
        _eagerly_instantiate_default(self.ttm_model_profiles_registry, self.switch_ttm, "TTM")
        _eagerly_instantiate_default(self.connection_model_profiles_registry, self.switch_connection, "CONNECTION")
        _eagerly_instantiate_default(self.rag_model_profiles_registry, self.switch_rag, "RAG")

        # Automatically persist resolved single defaults to disk so warnings never repeat
        if resolved_defaults_to_save:
            self._save_single_default_profiles_to_disk(resolved_defaults_to_save)

    def _register_binding_profiles(self, profiles_dict: Optional[Dict], registry: Dict[str, LollmsBindingProfile], modality_name: str, legacy_binding_name: Optional[str] = None, legacy_binding_config: Optional[Dict] = None):
        """Registers connection layer profiles (binding engines/servers)."""
        if profiles_dict:
            for alias, p_data in profiles_dict.items():
                if isinstance(p_data, LollmsBindingProfile):
                    profile = p_data
                else:
                    hoisted_config = dict(p_data.get("binding_config", {}) or {})
                    for ssl_key in ("verify_ssl_certificate", "certificate_file_path"):
                        if ssl_key in p_data and ssl_key not in hoisted_config:
                            hoisted_config[ssl_key] = p_data[ssl_key]
                    profile = LollmsBindingProfile(
                        name=alias,
                        binding_name=p_data.get("binding_name"),
                        binding_config=hoisted_config,
                        is_default=p_data.get("is_default", False)
                    )
                registry[alias] = profile

        # Backward compatibility: auto-register legacy direct binding params as "master"
        if legacy_binding_name and "master" not in registry:
            registry["master"] = LollmsBindingProfile(
                name="master",
                binding_name=legacy_binding_name,
                binding_config=legacy_binding_config or {},
                is_default=not bool(registry)
            )

    def _register_model_profiles(self, profiles_dict: Optional[Dict], registry: Dict[str, LollmsModelProfile], modality_name: str, binding_registry: Dict[str, LollmsBindingProfile]):
        """Registers execution layer profiles (models). Auto-creates a master profile only for legacy bindings."""
        found_default = False
        if profiles_dict:
            for alias, p_data in profiles_dict.items():
                if isinstance(p_data, LollmsModelProfile):
                    profile = p_data
                    if profile.is_default:
                        if not found_default:
                            found_default = True
                        else:
                            profile.is_default = False
                else:
                    raw_efforts = p_data.get("supported_reasoning_efforts") or p_data.get("reasoning_efforts")
                    if isinstance(raw_efforts, str) and raw_efforts.strip():
                        parsed_efforts = [s.strip() for s in raw_efforts.split(",") if s.strip()]
                    elif isinstance(raw_efforts, list):
                        parsed_efforts = raw_efforts
                    else:
                        parsed_efforts = None

                    is_def = p_data.get("is_default", False)
                    if is_def:
                        if not found_default:
                            found_default = True
                        else:
                            is_def = False

                    profile = LollmsModelProfile(
                        name=alias,
                        binding_profile_name=p_data.get("binding_profile_name") or p_data.get("binding_alias") or "master",
                        model_name=p_data.get("model_name"),
                        is_default=is_def,
                        vision_enabled=p_data.get("vision_enabled", False),
                        forced_context_size=p_data.get("forced_context_size"),
                        routing_config=p_data.get("routing_config") or p_data.get("routing_profile"),
                        glm_image_embedding=p_data.get("glm_image_embedding", False),
                        supported_reasoning_efforts=parsed_efforts,
                        video_enabled=p_data.get("video_enabled", False)
                    )
                registry[alias] = profile

        # Backward compatibility: Create a master model profile if a master binding exists and master is not in registry
        if "master" in binding_registry and "master" not in registry:
            model_name = binding_registry["master"].binding_config.get("model_name")
            has_default = any(p.is_default for p in registry.values())
            registry["master"] = LollmsModelProfile(
                name="master",
                binding_profile_name="master",
                model_name=model_name,
                is_default=not has_default
            )

    def _save_single_default_profiles_to_disk(self, resolved_defaults: Dict[str, Tuple[str, List[str]]]) -> None:
        """
        Persists resolved single default profiles to configuration files on disk
        so multiple-default warnings do not recur on subsequent app launches.
        """
        if not resolved_defaults:
            return

        candidate_yamls = [
            Path.home() / ".lollms_client" / "config.yaml",
            Path.cwd() / ".lollms_code" / "config.yaml",
        ]
        if self.system_dir:
            candidate_yamls.insert(0, Path(self.system_dir) / "config.yaml")

        candidate_envs = [
            Path.home() / ".lollms_client" / ".env",
            Path.cwd() / ".lollms_code" / ".env",
        ]
        if self.system_dir:
            candidate_envs.insert(0, Path(self.system_dir) / ".env")

        # 1. Update YAML configuration files
        for yaml_path in candidate_yamls:
            if not yaml_path.exists():
                continue
            try:
                import yaml as _yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    cfg = _yaml.safe_load(f) or {}

                if not isinstance(cfg, dict):
                    continue

                changed = False
                for mod_name, (chosen_default, demoted_list) in resolved_defaults.items():
                    mod_data = cfg.get(mod_name)
                    if not isinstance(mod_data, dict):
                        continue
                    profiles_data = mod_data.get("profiles")
                    if not isinstance(profiles_data, dict):
                        continue

                    # If a phantom 'master' profile was saved in yaml without model_name, purge it
                    if "master" in profiles_data and not profiles_data["master"].get("model_name"):
                        profiles_data.pop("master", None)
                        changed = True

                    for p_alias, p_info in profiles_data.items():
                        if not isinstance(p_info, dict):
                            continue
                        if p_alias == chosen_default:
                            if not p_info.get("is_default", False):
                                p_info["is_default"] = True
                                changed = True
                        elif p_alias in demoted_list or p_info.get("is_default", False):
                            p_info["is_default"] = False
                            changed = True

                if changed:
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                    ASCIIColors.success(f"[LollmsClient] Saved single default profiles to: {yaml_path}")
            except Exception as e:
                ASCIIColors.warning(f"[LollmsClient] Could not update {yaml_path}: {e}")

        # 2. Update .env configuration files
        for env_path in candidate_envs:
            if not env_path.exists():
                continue
            try:
                lines = env_path.read_text(encoding="utf-8").splitlines()
                new_lines = []
                changed = False

                for line in lines:
                    stripped = line.strip()
                    updated_line = line
                    for mod_name, (chosen_default, demoted_list) in resolved_defaults.items():
                        mod_prefix = f"{mod_name.upper()}_PROFILES_"
                        for demoted_alias in demoted_list:
                            clean_alias = re.sub(r"[^A-Za-z0-9_]", "_", demoted_alias.upper())
                            target_key = f"{mod_prefix}{clean_alias}_IS_DEFAULT"
                            if stripped.upper().startswith(f"{target_key}=") and "=TRUE" in stripped.upper():
                                updated_line = f"{target_key}=false"
                                changed = True
                    new_lines.append(updated_line)

                if changed:
                    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    ASCIIColors.success(f"[LollmsClient] Saved single default profiles to: {env_path}")
            except Exception as e:
                ASCIIColors.warning(f"[LollmsClient] Could not update {env_path}: {e}")

    def _instantiate_binding_from_profile(self, alias: str, model_profile: LollmsModelProfile, manager: Any, modality: str, callback=None) -> Optional[Any]:
        """Instantiates a binding from a model profile, merging the referenced binding profile config."""
        binding_registry = getattr(self, f"{modality}_binding_profiles_registry")
        binding_profile = binding_registry.get(model_profile.binding_profile_name)

        if not binding_profile:
            ASCIIColors.error(f"{modality.upper()} binding profile '{model_profile.binding_profile_name}' not found for model '{alias}'.")
            return None

        # Base connection config from the binding profile
        b_config = binding_profile.binding_config.copy() if binding_profile.binding_config else {}

        _ssl_debug = os.getenv("LOLLMS_DEBUG_SSL", "").lower() in ("1", "true", "yes")
        if _ssl_debug:
            ASCIIColors.yellow(f"[LollmsClient][SSL-DEBUG] instantiating {modality.upper()} '{alias}' from binding profile '{binding_profile.name}' (binding_name='{binding_profile.binding_name}')")
            ASCIIColors.yellow(f"[LollmsClient][SSL-DEBUG] b_config BEFORE sanitize: {b_config}")

        if "verify_ssl_certificate" in b_config and isinstance(b_config["verify_ssl_certificate"], str):
            b_config["verify_ssl_certificate"] = b_config["verify_ssl_certificate"].lower().strip() in ("true", "1", "yes", "y", "on")

        # For secondary non-LLM modalities, ensure server spawning does not block the caller's main thread
        if modality != "llm":
            b_config.setdefault("wait_for_server", False)

        if _ssl_debug:
            ASCIIColors.yellow(f"[LollmsClient][SSL-DEBUG] b_config AFTER sanitize: {b_config}")
            
        # Inject model_name if specified at the model profile level        if "verify_ssl_certificate" in b_config and isinstance(b_config["verify_ssl_certificate"], str):
            b_config["verify_ssl_certificate"] = b_config["verify_ssl_certificate"].lower().strip() in ("true", "1", "yes", "y", "on")

        # For secondary non-LLM modalities, ensure server spawning does not block the caller's main thread
        if modality != "llm":
            b_config.setdefault("wait_for_server", False)

        if _ssl_debug:
            ASCIIColors.yellow(f"[LollmsClient][SSL-DEBUG] b_config AFTER sanitize: {b_config}")
            
        # Inject model_name if specified at the model profile level        if "verify_ssl_certificate" in b_config and isinstance(b_config["verify_ssl_certificate"], str):
            b_config["verify_ssl_certificate"] = b_config["verify_ssl_certificate"].lower().strip() in ("true", "1", "yes", "y", "on")

        # For secondary non-LLM modalities, ensure server spawning does not block the caller's main thread
        if modality != "llm":
            b_config.setdefault("wait_for_server", False)

        if _ssl_debug:
            ASCIIColors.yellow(f"[LollmsClient][SSL-DEBUG] b_config AFTER sanitize: {b_config}")
            
        # Inject model_name if specified at the model profile level
        if model_profile.model_name:
            b_config['model_name'] = model_profile.model_name
            if modality == "rag":
                b_config.setdefault('db_path', model_profile.model_name)
                b_config.setdefault('store_name', model_profile.model_name)

        # Inject LLM-specific configs if applicable
        if modality == "llm":
            b_config['user_name'] = self.user_name
            b_config['ai_name'] = self.ai_name
        elif modality == "rag":
            b_config['lollms_client'] = self

        b_config['debug'] = self.debug
        if self.system_dir:
            b_config.setdefault('system_dir', str(self.system_dir))
            b_config.setdefault('cwd', str(self.system_dir))

        try:
            binding = manager.create_binding(
                binding_name=binding_profile.binding_name,
                **{k: v for k, v in b_config.items() if k != "binding_name"}
            )
            if binding:
                binding.vision_enabled = model_profile.vision_enabled or getattr(binding, "vision_enabled", False)
                binding.video_enabled = model_profile.video_enabled or getattr(binding, "video_enabled", False)
                if hasattr(binding, "forced_context_size"):
                    binding.forced_context_size = model_profile.forced_context_size
                if hasattr(binding, "routing_config"):
                    binding.routing_config = model_profile.routing_config
                if hasattr(binding, "glm_image_embedding"):
                    binding.glm_image_embedding = model_profile.glm_image_embedding or getattr(binding, "glm_image_embedding", False)
                else:
                    setattr(binding, "glm_image_embedding", model_profile.glm_image_embedding or getattr(binding, "glm_image_embedding", False))
                if hasattr(binding, "supported_reasoning_efforts"):
                    if model_profile.supported_reasoning_efforts is not None:
                        binding.supported_reasoning_efforts = model_profile.supported_reasoning_efforts
                else:
                    if model_profile.supported_reasoning_efforts is not None:
                        setattr(binding, "supported_reasoning_efforts", model_profile.supported_reasoning_efforts)
                return binding
        except Exception as e:
            trace_exception(e)
            if callback: callback(f"❌ Failed to instantiate {modality.upper()} '{alias}': {e}", MSG_TYPE.MSG_TYPE_ERROR, {})
        return None

    def _switch_modality(self, alias: str, model_registry: dict, binding_registry: dict, instance_cache: dict, manager: Any, modality: str, attr_name: str, active_alias_attr: str, callback=None) -> bool:
        """Generic switch method for all modalities, operating on the two-tier profile system."""
        if alias not in model_registry:
            ASCIIColors.error(f"{modality.upper()} model profile '{alias}' not found. Available: {list(model_registry.keys())}")
            return False

        if alias in instance_cache:
            current_binding = instance_cache[alias]
            object.__setattr__(self, attr_name, current_binding)
        else:
            model_profile = model_registry[alias]
            current_binding = self._instantiate_binding_from_profile(alias, model_profile, manager, modality, callback)
            if not current_binding: return False

            instance_cache[alias] = current_binding
            object.__setattr__(self, attr_name, current_binding)

            if callback: callback(f"✅ Instantiated & mounted {modality.upper()}: `{alias}`", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})

        if modality == "llm":
            self._remote_tokenizer_healthy = True
            if hasattr(self, "_ctx_size_cache"):
                self._ctx_size_cache = {}

        # Ensure model is ready (triggering resource reclamation if local and needed)
        if getattr(current_binding, "is_local", lambda: False)():
            model_name = getattr(model_registry.get(alias), "model_name", None)
            self.ensure_model_loaded(current_binding, model_name)

        object.__setattr__(self, active_alias_attr, alias)
        ASCIIColors.info(f"[LollmsClient] Active {modality.upper()} switched to '{alias}'.")
        return True

    def switch_model(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.llm_model_profiles_registry, self.llm_binding_profiles_registry, self.llms, self.llm_binding_manager, "llm", "llm", "_active_llm_alias", callback)

    def switch_tti(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.tti_model_profiles_registry, self.tti_binding_profiles_registry, self.ttis, self.tti_binding_manager, "tti", "tti", "_active_tti_alias", callback)

    def switch_tts(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.tts_model_profiles_registry, self.tts_binding_profiles_registry, self.tts_bindings, self.tts_binding_manager, "tts", "tts", "_active_tts_alias", callback)

    def switch_stt(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.stt_model_profiles_registry, self.stt_binding_profiles_registry, self.stts, self.stt_binding_manager, "stt", "stt", "_active_stt_alias", callback)

    def switch_ttv(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.ttv_model_profiles_registry, self.ttv_binding_profiles_registry, self.ttvs, self.ttv_binding_manager, "ttv", "ttv", "_active_ttv_alias", callback)

    def switch_ttm(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.ttm_model_profiles_registry, self.ttm_binding_profiles_registry, self.ttms, self.ttm_binding_manager, "ttm", "ttm", "_active_ttm_alias", callback)

    def switch_connection(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.connection_model_profiles_registry, self.connection_binding_profiles_registry, self.connections, self.connection_binding_manager, "connection", "connection", "_active_connection_alias", callback)

    def switch_rag(self, alias: str, callback=None) -> bool:
        return self._switch_modality(alias, self.rag_model_profiles_registry, self.rag_binding_profiles_registry, self.rags, self.rag_binding_manager, "rag", "rag", "_active_rag_alias", callback)

    # Legacy aliases
    def mount_llm(self, alias: str) -> bool: return self.switch_model(alias)
    def mount_rag(self, alias: str) -> bool: return self.switch_rag(alias)
    def mount_tti(self, alias: str) -> bool: return self.switch_tti(alias)
    def mount_tts(self, alias: str) -> bool: return self.switch_tts(alias)
    def mount_stt(self, alias: str) -> bool: return self.switch_stt(alias)
    def mount_ttv(self, alias: str) -> bool: return self.switch_ttv(alias)
    def mount_ttm(self, alias: str) -> bool: return self.switch_ttm(alias)
    def mount_connection(self, alias: str) -> bool: return self.switch_connection(alias)

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
    def _update_binding(self, binding_name: str, config: Optional[Dict[str, Any]], binding_registry: dict, model_registry: dict, instance_cache: dict, switch_method: Callable, modality: str):
        """Generic update method for all modality bindings, respecting the two-tier architecture."""
        config = config or {}
        binding_registry["master"] = LollmsBindingProfile(
            name="master",
            binding_name=binding_name,
            binding_config=config,
            is_default=True
        )
        # Ensure a default model profile exists that points to the master binding
        if "master" not in model_registry:
            model_registry["master"] = LollmsModelProfile(
                name="master",
                binding_profile_name="master",
                is_default=True
            )
        if "master" in instance_cache:
            del instance_cache["master"]
        return switch_method("master")

    def update_llm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.llm_binding_profiles_registry, self.llm_model_profiles_registry, self.llms, self.switch_model, "LLM")

    def update_tts_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.tts_binding_profiles_registry, self.tts_model_profiles_registry, self.tts_bindings, self.switch_tts, "TTS")

    def update_tti_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.tti_binding_profiles_registry, self.tti_model_profiles_registry, self.ttis, self.switch_tti, "TTI")

    def update_stt_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.stt_binding_profiles_registry, self.stt_model_profiles_registry, self.stts, self.switch_stt, "STT")

    def update_ttv_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.ttv_binding_profiles_registry, self.ttv_model_profiles_registry, self.ttvs, self.switch_ttv, "TTV")

    def update_ttm_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.ttm_binding_profiles_registry, self.ttm_model_profiles_registry, self.ttms, self.switch_ttm, "TTM")

    def update_tools_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        # Tools binding does not use the profile system yet, fallback to direct instantiation
        self.tools = self.tools_binding_manager.create_binding(binding_name=binding_name, **(config or {}))
        if self.tools is None: raise ValueError(f"Failed to update MCP binding: {binding_name}")

    def update_connection_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.connection_binding_profiles_registry, self.connection_model_profiles_registry, self.connections, self.switch_connection, "CONNECTION")

    def update_rag_binding(self, binding_name: str, config: Optional[Dict[str, Any]] = None):
        return self._update_binding(binding_name, config, self.rag_binding_profiles_registry, self.rag_model_profiles_registry, self.rags, self.switch_rag, "RAG")

    # --- Core LLM Methods (Delegated) ---
    def tokenize(self, text: str) -> list:
        if text is None:
            text = ""
        if self.llm: return self.llm.tokenize(text)
        raise RuntimeError("LLM binding not initialized.")

    def detokenize(self, tokens: list) -> str:
        if self.llm: return self.llm.detokenize(tokens)
        raise RuntimeError("LLM binding not initialized.")

    def enable_fast_token_estimate(self, coefficient: Optional[float] = None) -> None:
        """
        Activates local heuristic token estimation (no remote tokenizer calls).

        Used by latency-sensitive entry points (lollms-code CLI) where a rough
        estimate is acceptable and remote tokenization blocks generation start.
        """
        self.use_fast_token_estimate = True
        if coefficient is not None:
            self._fast_token_coefficient = self._sanitize_coefficient(coefficient)
        ASCIIColors.info(
            f"[LollmsClient] ⚡ Fast token estimation enabled "
            f"(coefficient: {self._fast_token_coefficient})."
        )

    def disable_fast_token_estimate(self) -> None:
        """Restores exact remote tokenization for all subsequent calls."""
        self.use_fast_token_estimate = False
        ASCIIColors.info("[LollmsClient] Exact remote tokenization restored.")

    @staticmethod
    def _sanitize_coefficient(value: float) -> float:
        """Clamps a coefficient into a safe range to guard against bad inputs."""
        return min(max(value, 0.5), 5.0)

    @staticmethod
    def _parse_fast_token_coefficient() -> float:
        """
        Reads LOLLMS_FAST_TOKEN_COEFFICIENT from the environment with strict
        sanitization. Falls back to 1.0 on any malformed or out-of-range value.
        """
        raw = os.getenv("LOLLMS_FAST_TOKEN_COEFFICIENT")
        if raw is None:
            return 1.0
        try:
            parsed = float(raw)
        except (TypeError, ValueError):
            ASCIIColors.warning(
                f"[LollmsClient] Invalid LOLLMS_FAST_TOKEN_COEFFICIENT '{raw}'. Using 1.0."
            )
            return 1.0
        if not 0.5 <= parsed <= 5.0:
            ASCIIColors.warning(
                f"[LollmsClient] LOLLMS_FAST_TOKEN_COEFFICIENT '{parsed}' out of range [0.5, 5.0]. Using 1.0."
            )
            return 1.0
        return parsed

    @staticmethod
    def _estimate_tokens_locally(text: str, coefficient: float) -> int:
        """
        Heuristic estimator: tokens ≈ (words + indentation_chars / 4) × coefficient.

        The indentation term accounts for 4-space indents (one token per level),
        matching observed tokenizer behavior on formatted source code.
        """
        if not text:
            return 0
        word_count = len(text.split())
        indent_chars = len(re.findall(r"[ \t]+", text))
        return int((word_count + indent_chars / 4.0) * coefficient)

    def count_tokens(self, text: str) -> int:
        if text is None:
            text = ""

        if getattr(self, "use_fast_token_estimate", False):
            return self._estimate_tokens_locally(text, self._fast_token_coefficient)

        # Fast in-memory token count caching to prevent redundant tokenizer round-trips
        cache_key = f"{len(text)}:{hash(text)}"
        if not hasattr(self, "_token_count_cache"):
            self._token_count_cache = {}

        cached = self._token_count_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.llm:
            if not getattr(self, "_remote_tokenizer_healthy", True):
                count = len(text) // 4
                self._token_count_cache[cache_key] = count
                return count
            try:
                count = self.llm.count_tokens(text)
                self._remote_tokenizer_healthy = True
                self._token_count_cache[cache_key] = count
                return count
            except Exception:
                self._remote_tokenizer_healthy = False
                count = len(text) // 4
                self._token_count_cache[cache_key] = count
                return count
        raise RuntimeError("LLM binding not initialized.")

    def count_image_tokens(self, image: str) -> int:
        if self.llm: return self.llm.count_image_tokens(image)
        raise RuntimeError("LLM binding not initialized.")

    def get_model_details(self) -> dict:
        if self.llm: return self.llm.get_model_info()
        raise RuntimeError("LLM binding not initialized.")

    def switch_active_model(self, model_name: str) -> bool:
        if self.llm: return self.llm.load_model(model_name)
        raise RuntimeError("LLM binding not initialized.")

    def get_available_llm_bindings(self) -> List[str]: 
        return self.llm_binding_manager.get_available_bindings()

    def free_local_binding_resources(self, except_binding: Optional[Any] = None) -> bool:
        """
        Audits all instantiated bindings across all modalities. If any local binding
        (other than except_binding) is holding resources (active daemon/models in RAM/VRAM),
        asks it to unload its models.
        Returns True if any resources were successfully liberated.
        """
        freed = False
        all_instances = (
            list(self.llms.values()) + list(self.ttis.values()) +
            list(self.ttms.values()) + list(self.tts_bindings.values()) +
            list(self.stts.values()) + list(self.ttvs.values())
        )

        for b in all_instances:
            if b is None or b == except_binding:
                continue
            is_loc = getattr(b, "is_local", lambda: False)()
            has_res = getattr(b, "has_active_resources", lambda: False)()
            if is_loc and has_res:
                loaded = getattr(b, "get_loaded_models", lambda: [])()
                b_name = getattr(b, "binding_name", "unknown")
                ASCIIColors.warning(f"[Resource Manager] Asking {b_name} to unload models {loaded} to liberate local VRAM/RAM...")
                try:
                    if b.unload_model():
                        freed = True
                        ASCIIColors.green(f"[Resource Manager] Successfully liberated resources from {b_name}.")
                except Exception as ex:
                    ASCIIColors.warning(f"[Resource Manager] Failed unloading models from {b_name}: {ex}")

        if freed:
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return freed

    def ensure_model_loaded(self, binding: Any, model_name: Optional[str] = None) -> bool:
        """
        Verifies if a model is loaded in the specified binding.
        If it is not loaded, attempts to load it. If loading fails (due to VRAM/RAM),
        coordinates with other local bindings to unload their resources, then retries.
        Remote bindings (is_local() == False) pass through immediately.
        """
        if not binding:
            return False

        if not getattr(binding, "is_local", lambda: False)():
            return True

        target_model = model_name or getattr(binding, "model_name", None)
        if not target_model:
            return True

        if getattr(binding, "is_model_loaded", lambda m=None: False)(target_model):
            return True

        b_name = getattr(binding, "binding_name", "binding")
        ASCIIColors.info(f"[{b_name}] Model '{target_model}' not loaded. Attempting to load...")
        if binding.load_model(target_model):
            return True

        # First attempt failed. Check if any other local bindings are consuming resources
        last_err = getattr(binding, "get_last_error", lambda: "")() or ""
        ASCIIColors.warning(f"[{b_name}] Initial load of '{target_model}' failed ({last_err}). Initiating resource reclamation...")

        freed = self.free_local_binding_resources(except_binding=binding)
        if freed:
            ASCIIColors.info(f"[{b_name}] Resources liberated. Retrying model load for '{target_model}'...")
            time.sleep(1.0)
            if binding.load_model(target_model):
                ASCIIColors.success(f"[{b_name}] Model '{target_model}' loaded successfully after resource reclamation.")
                return True

        ASCIIColors.error(f"[{b_name}] Could not load model '{target_model}'.")
        return False

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
        active_b = modalities.get(active_modality)
        self.free_local_binding_resources(except_binding=active_b)

    def _cooperative_unload_tti(self):
        self._cooperative_unload_except("llm")

    def _cooperative_unload_llm(self):
        self._cooperative_unload_except("tti")

    def has_vision_capability(self, binding: Optional[Any] = None) -> bool:
        """
        Checks if the specified binding (or the currently active LLM) has vision capabilities.
        """
        target_binding = binding or self.llm
        if not target_binding:
            return False

        if getattr(target_binding, "vision_enabled", False) is True:
            return True
        if getattr(target_binding, "supports_vision", False) is True:
            return True
        if getattr(target_binding, "video_enabled", False) is True:
            return True
        if getattr(target_binding, "glm_image_embedding", False) is True:
            return True

        models_dir = getattr(target_binding, "models_dir", None)
        find_mmproj_fn = getattr(target_binding, "_find_mmproj", None)
        model_name = getattr(target_binding, "model_name", None)
        if (
            not hasattr(target_binding, "_mock_return_value")
            and isinstance(models_dir, Path)
            and isinstance(model_name, str)
            and callable(find_mmproj_fn)
        ):
            try:
                model_p = models_dir / model_name
                if model_p.exists() and find_mmproj_fn(model_p) is not None:
                    target_binding.vision_enabled = True
                    return True
            except Exception:
                pass

        if hasattr(target_binding, "child_bindings") and isinstance(target_binding.child_bindings, dict):
            return any(getattr(child, "vision_enabled", False) or getattr(child, "supports_vision", False)
                       for child in target_binding.child_bindings.values())

        if hasattr(self, "_active_llm_alias") and self._active_llm_alias in self.llm_model_profiles_registry:
            active_profile = self.llm_model_profiles_registry[self._active_llm_alias]
            if getattr(active_profile, "vision_enabled", False):
                return True

        return False

    def has_video_capability(self, binding: Optional[Any] = None) -> bool:
        """
        Checks if the specified binding (or the currently active LLM) has video input capabilities.
        """
        target_binding = binding or self.llm
        if not target_binding:
            return False

        if getattr(target_binding, "video_enabled", False) is True:
            return True
        if getattr(target_binding, "supports_video", False) is True:
            return True

        if hasattr(target_binding, "child_bindings") and isinstance(target_binding.child_bindings, dict):
            return any(getattr(child, "video_enabled", False) or getattr(child, "supports_video", False)
                       for child in target_binding.child_bindings.values())

        if hasattr(self, "_active_llm_alias") and self._active_llm_alias in self.llm_model_profiles_registry:
            active_profile = self.llm_model_profiles_registry[self._active_llm_alias]
            if getattr(active_profile, "video_enabled", False):
                return True

        return False

    def find_available_vlm(self) -> Optional[Any]:
        """
        Discovers a vision-capable LLM binding from the instantiated registry,
        model profiles, or router children.
        """
        # 1. Check active LLM router child bindings
        if self.llm and hasattr(self.llm, "child_bindings") and isinstance(self.llm.child_bindings, dict):
            for child in self.llm.child_bindings.values():
                if getattr(child, "vision_enabled", False) or getattr(child, "supports_vision", False):
                    return child

        # 2. Check instantiated models cache
        for alias, binding in self.llms.items():
            if alias != self._active_llm_alias:
                if getattr(binding, "vision_enabled", False) or getattr(binding, "supports_vision", False):
                    return binding

        # 3. Check declared model profiles and instantiate VLM lazily if available
        for alias, profile in self.llm_model_profiles_registry.items():
            if alias != self._active_llm_alias and getattr(profile, "vision_enabled", False):
                vlm_binding = self._instantiate_binding_from_profile(alias, profile, self.llm_binding_manager, "llm")
                if vlm_binding:
                    self.llms[alias] = vlm_binding
                    return vlm_binding

        return None

    def get_or_generate_image_description(self, image: Union[str, Dict[str, Any], bytes]) -> str:
        """
        Generates and caches a textual description of an image using an available VLM in the bundle.
        If no VLM is available, returns a clean summary indicator.
        """
        raw_str = ""
        if isinstance(image, dict):
            raw_str = image.get("data") or image.get("url") or str(image)
        elif isinstance(image, bytes):
            raw_str = base64.b64encode(image).decode("utf-8")
        else:
            raw_str = str(image)

        image_hash = hashlib.sha256(raw_str.encode("utf-8", errors="ignore")).hexdigest()

        if not hasattr(self, "_image_description_cache"):
            self._image_description_cache = {}

        if image_hash in self._image_description_cache:
            return self._image_description_cache[image_hash]

        vlm = self.find_available_vlm()
        if not vlm:
            desc = "[Image attached (non-vision model)]"
            self._image_description_cache[image_hash] = desc
            return desc

        try:
            ASCIIColors.info("[LollmsClient] Non-vision active model detected. Generating image description using VLM...")
            prompt = "Describe this image in detail, including all visible text, layout, objects, and relationships. Be concise, objective, and precise."

            # Format image parameter appropriately for VLM
            img_payload = [raw_str]
            description = ""
            if hasattr(vlm, "generate_text"):
                res = vlm.generate_text(prompt=prompt, images=img_payload, temperature=0.1, n_predict=512)
                description = str(res).strip()
            elif hasattr(vlm, "generate_from_messages"):
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{raw_str}" if not raw_str.startswith("http") else raw_str}}]}
                ]
                res = vlm.generate_from_messages(messages=messages, temperature=0.1, n_predict=512)
                description = str(res).strip()

            if not description:
                description = "[Image attached]"

            formatted_desc = f"[Image Description: {description}]"
            self._image_description_cache[image_hash] = formatted_desc
            return formatted_desc
        except Exception as e:
            ASCIIColors.warning(f"[LollmsClient] VLM image description generation failed: {e}")
            desc = "[Image attached]"
            self._image_description_cache[image_hash] = desc
            return desc

    def _sanitize_images_for_active_llm(self, text: str, images: Optional[List[Any]]) -> Tuple[str, Optional[List[Any]]]:
        """
        If the active LLM lacks vision, strips all raw images, substitutes descriptions from VLM,
        and appends descriptions to the prompt text.
        """
        if not images:
            return text, None

        if self.has_vision_capability():
            return text, images

        descriptions = []
        for img in images:
            desc = self.get_or_generate_image_description(img)
            descriptions.append(desc)

        desc_block = "\n\n".join(descriptions)
        updated_text = f"{text}\n\n{desc_block}" if text else desc_block
        return updated_text, None

    def _sanitize_messages_for_active_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If the active LLM lacks vision, strips image blocks and replaces them with cached VLM descriptions.
        """
        if not messages:
            return messages

        if self.has_vision_capability():
            return messages

        sanitized_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content", "")

            # 1. Handle structured content list (OpenAI-style text and image_url blocks)
            if isinstance(content, list):
                new_content_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img_info = part.get("image_url", {})
                        img_data = img_info.get("url", "")
                        if img_data.startswith("data:image"):
                            _, b64 = img_data.split(",", 1)
                            img_data = b64
                        desc = self.get_or_generate_image_description(img_data)
                        new_content_parts.append({"type": "text", "text": desc})
                    else:
                        new_content_parts.append(part)
                msg_copy["content"] = new_content_parts

            # 2. Handle direct images field attached to the message
            if "images" in msg_copy and msg_copy["images"]:
                raw_images = msg_copy.pop("images", [])
                descriptions = [self.get_or_generate_image_description(img) for img in raw_images]
                desc_text = "\n\n".join(descriptions)
                if isinstance(msg_copy["content"], str):
                    msg_copy["content"] = f"{msg_copy['content']}\n\n{desc_text}".strip()
                elif isinstance(msg_copy["content"], list):
                    msg_copy["content"].append({"type": "text", "text": desc_text})

            if "active_images" in msg_copy:
                msg_copy.pop("active_images", None)

            if not self.has_video_capability():
                if "videos" in msg_copy and msg_copy["videos"]:
                    msg_copy.pop("videos", None)
                    vid_note = "[Video attached (non-video model)]"
                    if isinstance(msg_copy["content"], str):
                        msg_copy["content"] = f"{msg_copy['content']}\n\n{vid_note}".strip()
                    elif isinstance(msg_copy["content"], list):
                        msg_copy["content"].append({"type": "text", "text": vid_note})

            sanitized_messages.append(msg_copy)

        return sanitized_messages

    def generate_text(self, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_except("llm")
        if not self.llm:
            raise RuntimeError("LLM binding not initialized. Cannot use generate_text.")

        if getattr(self.llm, "is_local", lambda: False)():
            self.ensure_model_loaded(self.llm)

        think_arg = kwargs.get("think")
        effort_arg = kwargs.get("reasoning_effort")

        if self.llm:
            reasoning_effort = self.llm.get_effective_reasoning_effort(
                think=think_arg,
                reasoning_effort=effort_arg
            )
        else:
            reasoning_effort = LollmsLLMBinding.normalize_reasoning_effort(
                think_arg, effort_arg
            )

        kwargs.pop("reasoning_summary", None)

        is_thinking_deactivated = (
            think_arg is not True
            or reasoning_effort is None
            or str(reasoning_effort).strip().lower() in ("none", "off", "disabled", "false", "0", "")
        )

        if is_thinking_deactivated:
            kwargs["think"] = False
            kwargs["reasoning_effort"] = None
            m_name = getattr(getattr(self, "llm", None), "model_name", "") or ""
            extra_body = kwargs.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                extra_body.setdefault("chat_template_kwargs", {})["thinking"] = False
                if "glm" in m_name.lower():
                    extra_body["thinking"] = {"type": "disabled"}
                else:
                    extra_body["thinking"] = False
        else:
            kwargs["think"] = True
            kwargs["reasoning_effort"] = reasoning_effort

        ASCIIColors.info(
            f"[LollmsClient.generate_text] think={kwargs.get('think')} (input: {think_arg}), "
            f"reasoning_effort={kwargs.get('reasoning_effort')} (input: {effort_arg})"
        )

        # Non-vision model image stripping and VLM substitution
        prompt = kwargs.get("prompt", args[0] if len(args) > 0 else "")
        images = kwargs.get("images")
        if images is not None or not self.has_vision_capability():
            new_prompt, new_images = self._sanitize_images_for_active_llm(prompt, images)
            if "prompt" in kwargs:
                kwargs["prompt"] = new_prompt
            elif len(args) > 0:
                args = (new_prompt,) + args[1:]
            if "images" in kwargs:
                kwargs["images"] = new_images

        return self.llm.generate_text(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Union[str, dict]:
        return self.generate_text(*args, **kwargs)

    def generate_from_messages(self, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_except("llm")
        if not self.llm:
            raise RuntimeError("LLM binding not initialized. Cannot use generate_from_messages.")

        if getattr(self.llm, "is_local", lambda: False)():
            self.ensure_model_loaded(self.llm)

        think_arg = kwargs.get("think")
        effort_arg = kwargs.get("reasoning_effort")

        if self.llm:
            reasoning_effort = self.llm.get_effective_reasoning_effort(
                think=think_arg,
                reasoning_effort=effort_arg
            )
        else:
            reasoning_effort = LollmsLLMBinding.normalize_reasoning_effort(
                think_arg, effort_arg
            )

        kwargs.pop("reasoning_summary", None)

        is_thinking_deactivated = (
            think_arg is not True
            or reasoning_effort is None
            or str(reasoning_effort).strip().lower() in ("none", "off", "disabled", "false", "0", "")
        )

        if is_thinking_deactivated:
            kwargs["think"] = False
            kwargs["reasoning_effort"] = None
            m_name = getattr(getattr(self, "llm", None), "model_name", "") or ""
            extra_body = kwargs.setdefault("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                extra_body.setdefault("chat_template_kwargs", {})["thinking"] = False
                if "glm" in m_name.lower():
                    extra_body["thinking"] = {"type": "disabled"}
                else:
                    extra_body["thinking"] = False
        else:
            kwargs["think"] = True
            kwargs["reasoning_effort"] = reasoning_effort

        ASCIIColors.debug(
            f"[LollmsClient.generate_from_messages] think={kwargs.get('think')} (input: {think_arg}), "
            f"reasoning_effort={kwargs.get('reasoning_effort')} (input: {effort_arg})"
        )

        # Non-vision model message sanitization and VLM substitution
        messages = kwargs.get("messages", args[0] if len(args) > 0 else [])
        if messages:
            sanitized_messages = self._sanitize_messages_for_active_llm(messages)
            if "messages" in kwargs:
                kwargs["messages"] = sanitized_messages
            elif len(args) > 0:
                args = (sanitized_messages,) + args[1:]

        images = kwargs.get("images")
        if images is not None:
            _, new_images = self._sanitize_images_for_active_llm("", images)
            kwargs["images"] = new_images

        return self.llm.generate_from_messages(*args, **kwargs)

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Union[str, Path, Dict[str, Any]]],
        system_prompt: str = "",
        temperature: float = 0.7,
        n_predict: Optional[int] = None,
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
            param_sig = ", ".join([f"{p.get('name', 'param')}: {p.get('type', 'any')}" for p in params]) if params else ""
            param_details = []
            for p in params:
                opt = " (optional)" if p.get("optional") else ""
                param_details.append(f"`{p.get('name', 'param')}: {p.get('type', 'any')}`{opt}")
            param_desc = ", ".join(param_details) if param_details else "none"
            desc = (spec.get("description") or f"Execute {name}").strip()

            tool_entry = (
                f"#### 🛠️ **`{name}`**\n"
                f"- **Signature**: `{name}({param_sig})`\n"
                f"- **Parameters**: {param_desc}\n"
                f"- **Description**:\n  {desc}\n"
            )
            tool_descriptions.append(tool_entry)

        tool_header = (
            "=== TOOL USE — MANDATORY FORMAT ===\n"
            "You have external tools. To use one you MUST use EXACTLY this format:\n"
            "<tool>{\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}</tool>\n\n"
            "CRITICAL RULES:\n"
            "1. The ENTIRE tool call must be wrapped in <tool> tags.\n"
            "2. NO markdown code fences (no ```json).\n"
            "3. NO raw JSON without the XML wrapper.\n"
            "4. NO explanations before or after the tool call.\n"
            "5. ONLY output the <tool> line when calling a tool.\n"
            "6. One tool call per response turn.\n"
            "7. After calling ALL needed tools, write your final answer.\n"
            "8. If the user explicitly asks you to use a tool, USE IT.\n"
            "=== END TOOL USE RULES ===\n\n"
            "### Available Tools:\n\n"
        )

        tool_block = tool_header + "\n".join(tool_descriptions) + "\n=== END TOOLS AVAILABLE ===\n"

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
            # Primary: XML-wrapped tool calls <tool>...</tool>
            tool_call_pattern = re.compile(
                r'<tool>(.*?)</tool>',
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
                        f"Model emitted raw JSON tool call (missing <tool> tags). "
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

    def chat(self, discussion, *args, **kwargs) -> Union[str, dict]:
        self._cooperative_unload_tti()
        if discussion:
            # Log image payload status at core client layer
            images = kwargs.get("images")
            if images is not None:
                ASCIIColors.info(f"[LollmsClient.chat] Forwarding 'images' to binding: count={len(images)}, types={[type(img).__name__ for img in images[:5]]}")
            else:
                ASCIIColors.warning("[LollmsClient.chat] No 'images' parameter found in kwargs")
            return discussion.chat(*args, **kwargs)
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
            active_model = model_name or getattr(self.llm, "model_name", "default")
            cache_key = f"ctx_size_{active_model}"

            if not hasattr(self, "_ctx_size_cache"):
                self._ctx_size_cache = {}

            if cache_key in self._ctx_size_cache:
                return self._ctx_size_cache[cache_key]

            try:
                ctx_size = self.llm.get_ctx_size(model_name)
                if ctx_size and ctx_size > 0:
                    self._ctx_size_cache[cache_key] = ctx_size
                    return ctx_size
                self._ctx_size_cache[cache_key] = 32000
                return 32000
            except Exception:
                if cache_key not in self._ctx_size_cache:
                    self._ctx_size_cache[cache_key] = 32000
                    return 32000
                return self._ctx_size_cache[cache_key]
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

    def generate_structured_content(self, *args, **kwargs)->dict:
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
    def install_model(self, model_name: str, modality: str = "tti", **kwargs) -> dict:
        """
        Installs/downloads a model from Hugging Face on the specified modality binding.

        Args:
            model_name (str): The Hugging Face repo ID or model name to install.
            modality (str): Target modality ('tti', 'ttm', 'ttv', 'llm', etc.).
        """
        binding = getattr(self, modality, None)
        if binding and hasattr(binding, "install_model"):
            return binding.install_model(model_name, **kwargs)
        elif binding and hasattr(binding, "pull_model"):
            return binding.pull_model(model_name, **kwargs)
        raise RuntimeError(f"Modality '{modality}' binding not initialized or does not support install_model.")

    def pull_model(self, model_name: str, modality: str = "tti", **kwargs) -> dict:
        """Alias for install_model."""
        return self.install_model(model_name, modality=modality, **kwargs)

    def generate_image(self, *args, **kwargs):
        self._cooperative_unload_except("tti")
        if self.tti: return self.tti.generate_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def edit_image(self, *args, **kwargs):
        self._cooperative_unload_except("tti")
        if self.tti: return self.tti.edit_image(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def generate_omni(self, *args, **kwargs):
        """
        Unified TTI/Omni generation. Returns a TTIGenerationResult
        (images list + optional text) instead of raw bytes.
        Falls back cleanly for legacy bindings since the base class
        provides a default generate() wrapper.
        """
        self.cooperative_unload_except("tti")
        if self.tti:
            return self.tti.generate(*args, **kwargs)
        raise RuntimeError("TTI binding not initialized.")

    def send_connection_message(self, content: str, channel_alias: Optional[str] = None, sender_name: Optional[str] = None, **kwargs):
        """
        Send a message via a connection binding.

        Args:
            content: The message text to send.
            channel_alias: The connection profile alias to use. If None, uses the active/default connection.
            sender_name: Optional display name override.
            **kwargs: Platform-specific extra parameters.

        Returns:
            ConnectionSendResult dict with sent status and metadata.
        """
        if channel_alias and channel_alias != self._active_connection_alias:
            self.switch_connection(channel_alias)

        if self.connection:
            return self.connection.send_message(content, sender_name=sender_name, **kwargs)
        raise RuntimeError(
            "Connection binding not initialized. Configure connection_binding_name or connection_model_profiles."
        )

    def list_connection_channels(self) -> list:
        """List available channels on the active connection binding."""
        if self.connection:
            return self.connection.list_channels()
        return []

    # --- Delegated RAG Operations ---

    def query_rag(
        self,
        query: str,
        top_k: int = 5,
        store_alias: Optional[str] = None,
        hybrid: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Executes a query against the active or specified RAG data store.
        """
        if store_alias and store_alias != self._active_rag_alias:
            self.switch_rag(store_alias)

        if not self.rag:
            raise RuntimeError("RAG binding not initialized. Configure rag_binding_name or rag_binding_profiles.")

        if hybrid and hasattr(self.rag, "hybrid_query"):
            return self.rag.hybrid_query(query, top_k=top_k, **kwargs)
        return self.rag.query(query, top_k=top_k, **kwargs)

    def add_document_to_rag(
        self,
        file_path: Union[str, Path],
        store_alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Ingests a document file into the active or specified RAG data store.
        """
        if store_alias and store_alias != self._active_rag_alias:
            self.switch_rag(store_alias)

        if not self.rag:
            raise RuntimeError("RAG binding not initialized. Configure rag_binding_name or rag_binding_profiles.")

        return self.rag.add_document(file_path, metadata=metadata, **kwargs)

    def add_text_to_rag(
        self,
        text: str,
        unique_id: Optional[str] = None,
        store_alias: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Ingests text into the active or specified RAG data store.
        """
        if store_alias and store_alias != self._active_rag_alias:
            self.switch_rag(store_alias)

        if not self.rag:
            raise RuntimeError("RAG binding not initialized. Configure rag_binding_name or rag_binding_profiles.")

        return self.rag.add_text(text, unique_id=unique_id, metadata=metadata, **kwargs)

    def query_sparql(
        self,
        sparql_query: str,
        store_alias: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Executes a SPARQL 1.1 query on the active or specified RAG knowledge graph.
        """
        if store_alias and store_alias != self._active_rag_alias:
            self.switch_rag(store_alias)

        if not self.rag:
            raise RuntimeError("RAG binding not initialized. Configure rag_binding_name or rag_binding_profiles.")

        return self.rag.query_sparql(sparql_query, **kwargs)

    def get_rag_info(self, store_alias: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves database diagnostic statistics and metadata from the active RAG store.
        """
        if store_alias and store_alias != self._active_rag_alias:
            self.switch_rag(store_alias)

        if not self.rag:
            return {"error": "RAG binding not initialized."}

        return self.rag.get_database_info()

    def generate_audio(self, *args, **kwargs):
        self._cooperative_unload_except("tts")
        if self.tts: return self.tts.generate_audio(*args, **kwargs)
        raise RuntimeError("TTS binding not initialized.")

    def transcribe_audio(self, *args, **kwargs):
        self._cooperative_unload_except("stt")
        if self.stt: return self.stt.transcribe_audio(*args, **kwargs)
        raise RuntimeError("STT binding not initialized.")

    def transcribe_audio_with_diarization(self, *args, **kwargs):
        """
        Transcribes audio with speaker diarization, identifying distinct speakers,
        mapping them to ordered participants or biometric voice samples, and
        returning a list of dialogue turns.
        """
        self._cooperative_unload_except("stt")
        if self.stt:
            return self.stt.transcribe_audio_with_diarization(*args, **kwargs)
        raise RuntimeError("STT binding not initialized.")

    def generate_video(self, *args, **kwargs):
        self._cooperative_unload_except("ttv")
        if self.ttv: return self.ttv.generate_video(*args, **kwargs)
        raise RuntimeError("TTV binding not initialized.")

    def generate_music(self, *args, **kwargs):
        self._cooperative_unload_except("ttm")
        if self.ttm: return self.ttm.generate_music(*args, **kwargs)
        raise RuntimeError("TTM binding not initialized.")

    def generate_song(self, *args, **kwargs):
        self._cooperative_unload_except("ttm")
        if self.ttm: return self.ttm.generate_song(*args, **kwargs)
        raise RuntimeError("TTM binding not initialized.")

    def generate_song_from_lyrics(self, *args, **kwargs):
        self._cooperative_unload_except("ttm")
        if self.ttm: return self.ttm.generate_song_from_lyrics(*args, **kwargs)
        raise RuntimeError("TTM binding not initialized.")

    def long_context_processing(self, text_to_process: str, contextual_prompt: str, **kwargs) -> str:
        if self.llm:
            return self.llm.tp.long_context_processing(text_to_process, contextual_prompt, **kwargs)

    def generate_with_tag(self, prompt:str, tag:str, **kwargs):
        if self.llm:
            return self.llm.tp.generate_with_tag(prompt, tag, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
            
    def generate_with_tags(self, prompt:str, **kwargs):
        if self.llm:
            return self.llm.tp.generate_with_tags(prompt, **kwargs)
        raise RuntimeError("LLM binding not initialized.")

    def cancel(self) -> None:
        """Signal the active LLM binding to abort generation immediately."""
        if self.llm and hasattr(self.llm, "cancel"):
            self.llm.cancel()


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