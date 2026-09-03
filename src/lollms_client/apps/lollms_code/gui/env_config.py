"""
env_config.py — GUI-side wrapper around lollms_client's real configuration
wizard (lollms_config_cli_env.py).

Deliberately reuses that module's functions instead of reimplementing them,
so the GUI and CLI wizard can never drift out of sync on the .env format,
binding/profile alias scheme, or parameter schemas:

    {TYPE}_BINDINGS_{ALIAS}_{KEY}   e.g. LLM_BINDINGS_MASTER_HOST_ADDRESS
    {TYPE}_PROFILES_{ALIAS}_{KEY}   e.g. LLM_PROFILES_MASTER_MODEL_NAME

If lollms_client isn't importable yet, everything degrades to safe no-ops
so the GUI still boots and shows a clear "not connected" state instead of
crashing on launch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODALITIES = ["llm", "tti", "tts", "stt", "ttm", "ttv"]
MODALITY_LABELS = {
    "llm": "LLM (text)", "tti": "TTI (image)", "tts": "TTS (speech)",
    "stt": "STT (transcription)", "ttm": "TTM (music)", "ttv": "TTV (video)",
}

_IMPORT_ERROR: Optional[str] = None
try:
    from lollms_client.lollms_config_cli_env import (
        resolve_env_file,
        load_env_file,
        _list_bindings_by_type as list_bindings_by_type,
        _get_binding_description as get_binding_description,
        _fetch_available_models as fetch_available_models,
        _format_env_value as format_env_value,
        _convert_value as convert_value,
        _get_configured_aliases as get_configured_aliases,
        _get_binding_keys as get_binding_keys_raw,
        _get_profile_keys as get_profile_keys_raw,
        get_client_from_env,
        _extract_bindings_from_env,
        _extract_profiles_from_env,
    )
except ImportError as e:
    _IMPORT_ERROR = str(e)

    def _extract_bindings_from_env(prefix, env_data):
        return {}

    def _extract_profiles_from_env(prefix, bindings, env_data):
        return {}

    def resolve_env_file(cli_env_path=None):
        home_env = Path.home() / ".lollms-client" / ".env"
        return (home_env, False) if home_env.exists() else (None, True)

    def load_env_file(env_path):
        pass

    def list_bindings_by_type(binding_type: str) -> List[str]:
        return []

    def get_binding_description(binding_name, binding_type):
        return None

    def fetch_available_models(binding_type, binding_name, config_map):
        return []

    def format_env_value(value: Any) -> str:
        return "true" if value is True else "false" if value is False else str(value)

    def convert_value(raw: str, ptype: str) -> Any:
        return raw

    def get_configured_aliases(binding_type, config_map, category="BINDINGS") -> List[str]:
        prefix = f"{binding_type.upper()}_{category}_"
        aliases = set()
        for k in config_map:
            if k.startswith(prefix):
                parts = k[len(prefix):].split("_", 1)
                if len(parts) == 2:
                    aliases.add(parts[0])
        return sorted(aliases)

    def get_binding_keys_raw(binding_type, alias, config_map) -> Dict[str, str]:
        prefix = f"{binding_type.upper()}_BINDINGS_{alias}_"
        return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}

    def get_profile_keys_raw(binding_type, alias, config_map) -> Dict[str, str]:
        prefix = f"{binding_type.upper()}_PROFILES_{alias}_"
        return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}

    def get_client_from_env(**kwargs):
        raise RuntimeError("lollms_client is not importable in this environment.")


class EnvStore:
    """In-memory config_map + .env persistence, mirroring run_wizard_and_save()."""

    def __init__(self):
        self.config_map: Dict[str, str] = {}
        self.env_path: Optional[Path] = None
        self.import_error = _IMPORT_ERROR
        self.load()

    # ---------- load / persist ----------

    def load(self) -> None:
        path, _needs_wizard = resolve_env_file()
        self.env_path = path
        self.config_map = {}
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            self.config_map[k.strip()] = v.strip().strip("'\"")
            except Exception:
                pass

    def is_configured(self, require_llm: bool = True, require_tti: bool = False, require_tts: bool = False, require_stt: bool = False, require_ttm: bool = False, require_ttv: bool = False) -> bool:
        """Validates configuration based on required modalities using the Two-Tier Profile System."""
        required_modalities = {
            "llm": require_llm,
            "tti": require_tti,
            "tts": require_tts,
            "stt": require_stt,
            "ttm": require_ttm,
            "ttv": require_ttv
        }

        for modality, required in required_modalities.items():
            if required:
                has_binding = bool(get_configured_aliases(modality, self.config_map, "BINDINGS"))
                has_profile = bool(get_configured_aliases(modality, self.config_map, "PROFILES"))
                if not (has_binding and has_profile):
                    return False
        return True

    def save(self) -> Path:
        target_dir = Path.home() / ".lollms-client"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / ".env"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("# Lollms Client Configuration\n# Written by lollms_code GUI settings\n\n")
            for k, v in self.config_map.items():
                if v:
                    f.write(f"{k}={v}\n")
        self.env_path = target_file
        load_env_file(target_file)  # refresh os.environ for this process too
        return target_file

    def validate(self) -> Tuple[bool, str]:
        """Try to actually build a LollmsClient from the current MASTER-ish
        config, same approach as the wizard's _save_and_validate()."""
        try:
            from lollms_client import LollmsClient

            def to_bool(v):
                return v.lower().strip() in ("true", "1", "yes", "y") if isinstance(v, str) else bool(v)

            kwargs: Dict[str, Any] = {}
            found_any = False
            for b_type in MODALITIES:
                aliases = get_configured_aliases(b_type, self.config_map, "BINDINGS")
                if not aliases:
                    continue
                alias = "MASTER" if "MASTER" in aliases else aliases[0]
                prefix = f"{b_type.upper()}_BINDINGS_{alias}_"
                b_name = self.config_map.get(prefix + "BINDING_NAME")
                if not b_name:
                    continue
                b_config = {}
                for k, v in self.config_map.items():
                    if k.startswith(prefix) and k != prefix + "BINDING_NAME":
                        key_lower = k[len(prefix):].lower()
                        b_config[key_lower] = to_bool(v) if key_lower == "verify_ssl_certificate" else v
                kwargs[f"{b_type}_binding_name"] = b_name
                kwargs[f"{b_type}_binding_config"] = b_config
                found_any = True

            if not found_any:
                return False, "No bindings configured yet."

            LollmsClient(**kwargs)
            return True, "Connection validated successfully."
        except Exception as e:
            return False, str(e)

    # ---------- bindings ----------

    def available_bindings(self, binding_type: str) -> List[str]:
        return list_bindings_by_type(binding_type)

    def binding_param_schema(self, binding_type: str, binding_name: str) -> List[Dict[str, Any]]:
        desc = get_binding_description(binding_name, binding_type)
        if not desc:
            return []
        params = (desc.get("global_input_parameters") or desc.get("input_parameters") or []) + \
                 (desc.get("model_input_parameters") or [])
        return [p for p in params if p.get("name") and p.get("name") != "model_name"]

    def configured_binding_aliases(self, binding_type: str) -> List[str]:
        return get_configured_aliases(binding_type, self.config_map, "BINDINGS")

    def binding_keys(self, binding_type: str, alias: str) -> Dict[str, str]:
        return get_binding_keys_raw(binding_type, alias, self.config_map)

    def save_binding(self, binding_type: str, binding_name: str, alias: str, params: Dict[str, Any]) -> None:
        alias = alias.strip().upper()
        prefix = f"{binding_type.upper()}_BINDINGS_{alias}_"
        self.config_map[prefix + "BINDING_NAME"] = binding_name
        for pname, value in params.items():
            self.config_map[prefix + pname.upper()] = format_env_value(value)

    def delete_binding(self, binding_type: str, alias: str) -> None:
        prefix = f"{binding_type.upper()}_BINDINGS_{alias}_"
        for k in list(self.config_map.keys()):
            if k.startswith(prefix):
                del self.config_map[k]

    def set_binding_key(self, binding_type: str, alias: str, key: str, value: str) -> None:
        self.config_map[f"{binding_type.upper()}_BINDINGS_{alias}_{key}"] = value

    # ---------- profiles ----------

    def configured_profile_aliases(self, binding_type: str) -> List[str]:
        return get_configured_aliases(binding_type, self.config_map, "PROFILES")

    def profile_keys(self, binding_type: str, alias: str) -> Dict[str, str]:
        return get_profile_keys_raw(binding_type, alias, self.config_map)

    def fetch_models(self, binding_type: str, binding_alias: str) -> List[str]:
        keys = self.binding_keys(binding_type, binding_alias)
        binding_name = keys.get("BINDING_NAME")
        if not binding_name:
            return []
        prefix = f"{binding_type.upper()}_"
        sub_map = {f"{prefix}{k}": v for k, v in keys.items()}
        return fetch_available_models(binding_type, binding_name, sub_map)

    def save_profile(
        self, binding_type: str, alias: str, binding_alias: str, model_name: str,
        is_default: bool = False, vision_enabled: bool = False,
        forced_context_size: str = "", routing: Optional[Dict[str, str]] = None,
    ) -> None:
        alias = alias.strip().upper()
        prefix = f"{binding_type.upper()}_PROFILES_{alias}_"

        if is_default:
            # Only one default per modality — clear any existing one first.
            for other in self.configured_profile_aliases(binding_type):
                self.config_map.pop(f"{binding_type.upper()}_PROFILES_{other}_IS_DEFAULT", None)

        self.config_map[prefix + "BINDING_ALIAS"] = binding_alias.strip().upper()
        if model_name:
            self.config_map[prefix + "MODEL_NAME"] = model_name
        if is_default:
            self.config_map[prefix + "IS_DEFAULT"] = "true"
        else:
            self.config_map.pop(prefix + "IS_DEFAULT", None)

        if binding_type == "llm":
            if vision_enabled:
                self.config_map[prefix + "VISION_ENABLED"] = "true"
            else:
                self.config_map.pop(prefix + "VISION_ENABLED", None)
            if forced_context_size.strip():
                self.config_map[prefix + "FORCED_CONTEXT_SIZE"] = forced_context_size.strip()
            else:
                self.config_map.pop(prefix + "FORCED_CONTEXT_SIZE", None)
            if routing:
                for k, v in routing.items():
                    if v:
                        self.config_map[prefix + f"ROUTING_{k.upper()}"] = str(v)

    def delete_profile(self, binding_type: str, alias: str) -> None:
        prefix = f"{binding_type.upper()}_PROFILES_{alias}_"
        for k in list(self.config_map.keys()):
            if k.startswith(prefix):
                del self.config_map[k]

    def set_profile_key(self, binding_type: str, alias: str, key: str, value: str) -> None:
        self.config_map[f"{binding_type.upper()}_PROFILES_{alias}_{key}"] = value

    # ---------- resolution for actual client construction ----------

    def resolve_default_connection(self, binding_type: str = "llm") -> Dict[str, Any]:
        """Mirrors CodeAgentConfig.load()'s default-profile resolution from the
        original CLI: find the profile flagged IS_DEFAULT, fall back to the
        first configured binding if no profile is marked default."""
        prefix = binding_type.upper()
        default_alias = None
        for k, v in self.config_map.items():
            if k.startswith(f"{prefix}_PROFILES_") and k.endswith("_IS_DEFAULT") and v.lower() in ("true", "1", "yes"):
                default_alias = k[len(f"{prefix}_PROFILES_"):-len("_IS_DEFAULT")]
                break

        if default_alias:
            binding_alias = self.config_map.get(f"{prefix}_PROFILES_{default_alias}_BINDING_ALIAS", default_alias)
            model_name = self.config_map.get(f"{prefix}_PROFILES_{default_alias}_MODEL_NAME")
        else:
            aliases = self.configured_binding_aliases(binding_type)
            if not aliases:
                return {}
            binding_alias = aliases[0]
            model_name = None
            for p_alias in self.configured_profile_aliases(binding_type):
                if self.config_map.get(f"{prefix}_PROFILES_{p_alias}_BINDING_ALIAS") == binding_alias:
                    model_name = self.config_map.get(f"{prefix}_PROFILES_{p_alias}_MODEL_NAME")
                    break

        return dict(
            binding_name=self.config_map.get(f"{prefix}_BINDINGS_{binding_alias}_BINDING_NAME"),
            model_name=model_name,
            host_address=self.config_map.get(f"{prefix}_BINDINGS_{binding_alias}_HOST_ADDRESS", ""),
            api_key=self.config_map.get(f"{prefix}_BINDINGS_{binding_alias}_SERVICE_KEY", ""),
            verify_ssl=self.config_map.get(f"{prefix}_BINDINGS_{binding_alias}_VERIFY_SSL_CERTIFICATE", "false"),
        )
