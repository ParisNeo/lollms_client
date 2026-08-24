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
import warnings
from pathlib import Path
from enum import Enum
from typing import List, Optional, Callable, Union, Dict, Any
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
from lollms_client.lollms_personality.lollms_personality import ToolsManager

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
        
        # 🧠 Modern Lazy Profiles (Universal across all modalities)
        llm_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        tti_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        tts_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        stt_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        ttv_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,
        ttm_binding_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsBindingProfile']]] = None,

        llm_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        tti_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        tts_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        stt_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        ttv_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,
        ttm_model_profiles: Optional[Dict[str, Union[Dict[str, Any], 'LollmsModelProfile']]] = None,

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

        # Multi-Binding Registries (Instantiated Models)
        self.llms: Dict[str, LollmsLLMBinding] = {}
        self.ttis: Dict[str, LollmsTTIBinding] = {}
        self.tts_bindings: Dict[str, LollmsTTSBinding] = {}
        self.stts: Dict[str, LollmsSTTBinding] = {}
        self.ttvs: Dict[str, LollmsTTVBinding] = {}
        self.ttms: Dict[str, LollmsTTMBinding] = {}

        self._active_llm_alias: Optional[str] = None
        self._active_tti_alias: Optional[str] = None
        self._active_tts_alias: Optional[str] = None
        self._active_stt_alias: Optional[str] = None
        self._active_ttv_alias: Optional[str] = None
        self._active_ttm_alias: Optional[str] = None

        # 🧠 Profile Registries (Declarative Configs - Universal Two-Tier Architecture)
        self.llm_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.tti_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.tts_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.stt_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.ttv_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}
        self.ttm_binding_profiles_registry: Dict[str, LollmsBindingProfile] = {}

        self.llm_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.tti_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.tts_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.stt_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.ttv_model_profiles_registry: Dict[str, LollmsModelProfile] = {}
        self.ttm_model_profiles_registry: Dict[str, LollmsModelProfile] = {}

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

        # 1. Register Connection Layer Profiles (Bindings)
        self._register_binding_profiles(llm_binding_profiles, self.llm_binding_profiles_registry, "LLM", llm_binding_name, llm_binding_config)
        self._register_binding_profiles(tts_binding_profiles, self.tts_binding_profiles_registry, "TTS", tts_binding_name, tts_binding_config)
        self._register_binding_profiles(tti_binding_profiles, self.tti_binding_profiles_registry, "TTI", tti_binding_name, tti_binding_config)
        self._register_binding_profiles(stt_binding_profiles, self.stt_binding_profiles_registry, "STT", stt_binding_name, stt_binding_config)
        self._register_binding_profiles(ttv_binding_profiles, self.ttv_binding_profiles_registry, "TTV", ttv_binding_name, ttv_binding_config)
        self._register_binding_profiles(ttm_binding_profiles, self.ttm_binding_profiles_registry, "TTM", ttm_binding_name, ttm_binding_config)

        # 2. Register Execution Layer Profiles (Models)
        self._register_model_profiles(llm_model_profiles, self.llm_model_profiles_registry, "LLM", self.llm_binding_profiles_registry)
        self._register_model_profiles(tts_model_profiles, self.tts_model_profiles_registry, "TTS", self.tts_binding_profiles_registry)
        self._register_model_profiles(tti_model_profiles, self.tti_model_profiles_registry, "TTI", self.tti_binding_profiles_registry)
        self._register_model_profiles(stt_model_profiles, self.stt_model_profiles_registry, "STT", self.stt_binding_profiles_registry)
        self._register_model_profiles(ttv_model_profiles, self.ttv_model_profiles_registry, "TTV", self.ttv_binding_profiles_registry)
        self._register_model_profiles(ttm_model_profiles, self.ttm_model_profiles_registry, "TTM", self.ttm_binding_profiles_registry)

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
        def _eagerly_instantiate_default(model_registry: dict, switch_method: Callable, modality_name: str):
            default_alias = next((a for a, p in model_registry.items() if p.is_default), None)
            if default_alias:
                switch_method(default_alias, callback=callback)
            elif "master" in model_registry:
                switch_method("master", callback=callback)

        _eagerly_instantiate_default(self.llm_model_profiles_registry, self.switch_model, "LLM")
        _eagerly_instantiate_default(self.tts_model_profiles_registry, self.switch_tts, "TTS")
        _eagerly_instantiate_default(self.tti_model_profiles_registry, self.switch_tti, "TTI")
        _eagerly_instantiate_default(self.stt_model_profiles_registry, self.switch_stt, "STT")
        _eagerly_instantiate_default(self.ttv_model_profiles_registry, self.switch_ttv, "TTV")
        _eagerly_instantiate_default(self.ttm_model_profiles_registry, self.switch_ttm, "TTM")

    def _register_binding_profiles(self, profiles_dict: Optional[Dict], registry: Dict[str, LollmsBindingProfile], modality_name: str, legacy_binding_name: Optional[str] = None, legacy_binding_config: Optional[Dict] = None):
        """Registers connection layer profiles (binding engines/servers)."""
        if profiles_dict:
            for alias, p_data in profiles_dict.items():
                if isinstance(p_data, LollmsBindingProfile):
                    profile = p_data
                else:
                    profile = LollmsBindingProfile(
                        name=alias,
                        binding_name=p_data.get("binding_name"),
                        binding_config=p_data.get("binding_config", {}) or {},
                        is_default=p_data.get("is_default", False)
                    )
                registry[alias] = profile

        # Backward compatibility: auto-register legacy direct binding params as "master"
        if legacy_binding_name and "master" not in registry:
            registry["master"] = LollmsBindingProfile(
                name="master",
                binding_name=legacy_binding_name,
                binding_config=legacy_binding_config or {},
                is_default=True
            )

    def _register_model_profiles(self, profiles_dict: Optional[Dict], registry: Dict[str, LollmsModelProfile], modality_name: str, binding_registry: Dict[str, LollmsBindingProfile]):
        """Registers execution layer profiles (models). Auto-creates a master profile for legacy bindings."""
        if profiles_dict:
            for alias, p_data in profiles_dict.items():
                if isinstance(p_data, LollmsModelProfile):
                    profile = p_data
                else:
                    profile = LollmsModelProfile(
                        name=alias,
                        binding_profile_name=p_data.get("binding_profile_name", "master"),
                        model_name=p_data.get("model_name"),
                        is_default=p_data.get("is_default", False),
                        vision_enabled=p_data.get("vision_enabled", False),
                        forced_context_size=p_data.get("forced_context_size"),
                        routing_config=p_data.get("routing_config") or p_data.get("routing_profile")
                    )
                registry[alias] = profile

        # Backward compatibility: If we registered a master binding but no master model profile exists, create one.
        if "master" in binding_registry and "master" not in registry:
            registry["master"] = LollmsModelProfile(
                name="master",
                binding_profile_name="master",
                is_default=True
            )

    def _instantiate_binding_from_profile(self, alias: str, model_profile: LollmsModelProfile, manager: Any, modality: str, callback=None) -> Optional[Any]:
        """Instantiates a binding from a model profile, merging the referenced binding profile config."""
        binding_registry = getattr(self, f"{modality}_binding_profiles_registry")
        binding_profile = binding_registry.get(model_profile.binding_profile_name)

        if not binding_profile:
            ASCIIColors.error(f"{modality.upper()} binding profile '{model_profile.binding_profile_name}' not found for model '{alias}'.")
            return None

        # Base connection config from the binding profile
        b_config = binding_profile.binding_config.copy() if binding_profile.binding_config else {}

        # Inject model_name if specified at the model profile level
        if model_profile.model_name:
            b_config['model_name'] = model_profile.model_name

        # Inject LLM-specific configs if applicable
        if modality == "llm":
            b_config['user_name'] = self.user_name
            b_config['ai_name'] = self.ai_name
        b_config['debug'] = self.debug

        try:
            binding = manager.create_binding(
                binding_name=binding_profile.binding_name,
                **{k: v for k, v in b_config.items() if k != "binding_name"}
            )
            if binding:
                binding.vision_enabled = model_profile.vision_enabled
                if hasattr(binding, "forced_context_size"):
                    binding.forced_context_size = model_profile.forced_context_size
                if hasattr(binding, "routing_config"):
                    binding.routing_config = model_profile.routing_config
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
            object.__setattr__(self, attr_name, instance_cache[alias])
        else:
            model_profile = model_registry[alias]
            new_binding = self._instantiate_binding_from_profile(alias, model_profile, manager, modality, callback)
            if not new_binding: return False

            instance_cache[alias] = new_binding
            object.__setattr__(self, attr_name, new_binding)

            if callback: callback(f"✅ Instantiated & mounted {modality.upper()}: `{alias}`", MSG_TYPE.MSG_TYPE_INIT_PROGRESS, {})

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

    # Legacy aliases
    def mount_llm(self, alias: str) -> bool: return self.switch_model(alias)
    def mount_tti(self, alias: str) -> bool: return self.switch_tti(alias)
    def mount_tts(self, alias: str) -> bool: return self.switch_tts(alias)
    def mount_stt(self, alias: str) -> bool: return self.switch_stt(alias)
    def mount_ttv(self, alias: str) -> bool: return self.switch_ttv(alias)
    def mount_ttm(self, alias: str) -> bool: return self.switch_ttm(alias)

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
            try:
                # Attempt to get exact token count from active LLM binding
                count = self.llm.count_tokens(text)
                self._token_count_cache[text_hash] = count
                return count
            except Exception:
                # Fast offline fallback: estimate tokens to prevent external API flooding on connection errors
                count = len(text) // 4
                self._token_count_cache[text_hash] = count
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
            raise RuntimeError("LLM binding not initialized. Cannot use generate_text.")

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
            raise RuntimeError("LLM binding not initialized. Cannot use generate_from_messages.")

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

    def generate_with_tag(self, prompt:str, tag:str, **kwargs):
        if self.llm:
            return self.llm.tp.generate_with_tag(prompt, tag, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
            
    def generate_with_tags(self, prompt:str, **kwargs):
        if self.llm:
            return self.llm.tp.generate_with_tags(prompt, **kwargs)
        raise RuntimeError("LLM binding not initialized.")
            

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