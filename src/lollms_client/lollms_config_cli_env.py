"""
lollms_config_cli_env.py
Interactive configuration wizard and unified configuration resolver for Lollms Client.
Supports Multi-Source Ingestion (env, json, yaml, ini) and the Two-Tier Profile System.
"""
import os
import re
import sys
import json
import functools
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from lollms_client import LollmsClient

from ascii_colors import ASCIIColors, Menu  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None

try:
    import configparser
except ImportError:
    configparser = None

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_default_config_file() -> Path:
    """Returns the active configuration file according to the shared path policy.

    Honors (in order): an explicit path (not used here), the permanent
    ``$LOLLMS_CONFIG_FILE`` env override, and falls back to
    ``~/.lollms/config.yaml`` — see ``lollms_paths.get_config_file``.
    """
    try:
        from lollms_client.lollms_paths import get_config_file
        return get_config_file()
    except Exception:
        return (Path.home() / ".lollms" / "config.yaml").resolve()


def resolve_env_file(cli_env_path: Optional[Union[str, Path]] = None) -> Tuple[Optional[Path], bool]:
    """Resolves the active configuration file path (config.yaml only)."""
    if cli_env_path:
        p = Path(cli_env_path)
        if p.exists():
            return p, False

    default_yaml = _get_default_config_file()
    if default_yaml.exists():
        return default_yaml, False

    return None, True


def load_env_file(env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Parses a classic .env file into a flat KEY=VALUE string map.

    Legacy compatibility shim: the configuration system is YAML-first, but
    callers (e.g. lollms_code CLI) still consume flat .env maps. Supports
    'KEY=VALUE' lines, '#' comments, optional 'export ' prefixes, and strips
    surrounding quotes from values. Missing or unreadable files yield an
    empty dict (soft-fail, consistent with the module's philosophy).
    """
    if not env_path:
        return {}

    env_map: Dict[str, str] = {}
    try:
        env_file = Path(env_path).expanduser()
        if not env_file.is_file():
            return {}

        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export ") or line.startswith("set "):
                line = line.split(None, 1)[1].strip()
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                env_map[key] = value
    except Exception as e:
        ASCIIColors.warning(f"Failed to parse env file '{env_path}': {e}")
        return {}

    return env_map

_SERIALIZATION_PROFILE_SUFFIXES = (
    "_BINDING_ALIAS", "_BINDING_NAME", "_MODEL_NAME", "_IS_DEFAULT",
    "_VISION_ENABLED", "_VIDEO_ENABLED", "_IMAGE_TOKEN_COST",
    "_THINKING_ENABLED", "_TOOL_CALLING_ENABLED",
    "_SUPPORTED_REASONING_EFFORTS", "_DEFAULT_REASONING_EFFORT",
    "_FORCED_CONTEXT_SIZE", "_VERIFY_SSL_CERTIFICATE",
    "_INSTANCE_NAME",
)


def _serialize_config_map_to_yaml(config_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Reconstructs a structured dictionary adhering to the Two-Tier profile schema:
    <modality>:
      bindings:
        <alias>:
          <param_name>: <value>
      profiles:
        <alias>:
          <param_name>: <value>
    Pre-discovers configured aliases so aliases containing underscores
    (e.g. local_ollama, qwen_2_5) are preserved without truncation.
    """
    yaml_data: Dict[str, Any] = {}

    def _convert_scalar(v: Any) -> Any:
        if isinstance(v, bool):
            return v
        v_str = str(v).strip()
        if v_str.lower() in ("true", "yes", "1", "on"):
            return True
        if v_str.lower() in ("false", "no", "0", "off"):
            return False
        try:
            return int(v_str)
        except ValueError:
            try:
                return float(v_str)
            except ValueError:
                return v_str

    known_aliases_by_modality: Dict[str, Dict[str, List[str]]] = {}
    for mod in ("llm", "tti", "tts", "stt", "ttm", "ttv", "connection", "rag"):
        known_aliases_by_modality[mod] = {
            "bindings": _get_configured_aliases(mod, config_map, "BINDINGS"),
            "profiles": _get_configured_aliases(mod, config_map, "PROFILES"),
            "commands": _get_configured_command_aliases(mod, config_map),
        }

    for k, v in config_map.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue

        k_upper = k.upper().strip()

        # 📂 Top-level client-wide data folder: persist as a first-class key,
        # never parsed as a modality-namespaced entry.
        if k_upper == "LOLLMS_DATA_FOLDER":
            yaml_data["lollms_data_folder"] = str(v).strip()
            continue

        parts = k_upper.split("_", 2)
        if len(parts) < 3:
            continue

        modality = parts[0].lower()
        category = parts[1].lower()
        if category not in ("bindings", "profiles", "commands"):
            continue
        remainder = parts[2]

        known_aliases = known_aliases_by_modality.get(modality, {}).get(category, [])
        alias = None
        param_name = None

        for known_alias in sorted(known_aliases, key=len, reverse=True):
            marker = f"{known_alias.upper()}_"
            if remainder.startswith(marker):
                alias = known_alias
                param_name = remainder[len(marker):].lower()
                break

        if not alias:
            if category == "profiles":
                for suffix in _SERIALIZATION_PROFILE_SUFFIXES:
                    if remainder.endswith(suffix):
                        alias = remainder[: -len(suffix)]
                        param_name = suffix.lstrip("_").lower()
                        break
                if param_name is None and "_ROUTING_" in remainder:
                    idx = remainder.find("_ROUTING_")
                    alias = remainder[:idx]
                    param_name = remainder[idx + 1:].lower()
            elif category == "commands":
                # Command keys look like: <binding_alias>_<command_alias>_<param>.
                # The command alias is user-chosen and used verbatim in the map,
                # so we keep the entire remainder before NAME/TITLE/DESCRIPTION
                # as-is. Longest-suffix-first matching avoids stranding name chars.
                for suffix in ("_DESCRIPTION", "_TITLE", "_NAME"):
                    if remainder.endswith(suffix):
                        alias = remainder[: -len(suffix)]
                        param_name = suffix.lstrip("_").lower()
                        break
            else:
                if remainder.endswith("_BINDING_NAME"):
                    alias = remainder[:-len("_BINDING_NAME")]
                    param_name = "binding_name"

        if not alias or not param_name:
            continue

        alias = _sanitize_alias(alias)
        yaml_data.setdefault(modality, {}).setdefault(category, {}).setdefault(alias, {})[param_name] = _convert_scalar(v)

    return yaml_data

def load_json_file(file_path: Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    if not yaml:
        raise ImportError("PyYAML is required to parse YAML configurations.")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_ini_file(file_path: Path, entry: Optional[str] = None) -> Dict[str, Any]:
    if not configparser:
        raise ImportError("configparser is required to parse INI configurations.")
    config = configparser.ConfigParser()
    config.read(file_path)

    if entry:
        if entry in config:
            return dict(config[entry])
        return {}

    data = {}
    for section in config.sections():
        for key, val in config.items(section):
            data[f"{section.upper()}_{key.upper()}"] = val
    return data

def _descend_into_entry(data: Dict[str, Any], entry: Optional[str]) -> Dict[str, Any]:
    """Safely descends into nested dictionary keys (e.g., 'app.llms')."""
    if not entry:
        return data
    current = data
    for part in entry.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return {}
    return current if isinstance(current, dict) else {}

def _flatten_dict_to_env(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, str]:
    """Flattens nested dicts into environment-style keys (e.g. A_B_C = val).
    Keys are normalized to UPPERCASE so every downstream consumer can rely on
    a single canonical casing regardless of the source yaml/json casing."""
    items = []
    for k, v in d.items():
        base_key = f"{parent_key}{sep}{str(k).upper()}" if parent_key else str(k).upper()
        new_key = base_key
        if isinstance(v, dict):
            items.extend(_flatten_dict_to_env(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict_to_env(item, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", str(item)))
        else:
            items.append((new_key, str(v)))
    return dict(items)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Binding/Profile Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _convert_to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        return val.lower().strip() in ("true", "1", "yes", "y", "on")
    return False


def _find_first_upper_param_boundary(remainder: str) -> int:
    """
    Given a string like "GENERAL_TIMEOUT" or "MY_ALIAS_TIMEOUT",
    find the position where the alias ends and the parameter key begins.

    Convention: The alias may contain any chars (including underscores).
    The parameter key starts at the first substring that matches
    a known binding_config key pattern (all-uppercase word starting
    with a standard key like HOST_ADDRESS, SERVICE_KEY, MODEL_NAME,
    TIMEOUT, VERIFY_SSL_CERTIFICATE, etc.).

    Simple heuristic: find the first "_" followed by a substring that
    looks like a parameter key (starts with a known prefix or is a
    common connection/LLM config key).
    """
    # Common keys we look for as potential parameter starts
    KNOWN_KEYS = {
        "HOST_ADDRESS", "SERVICE_KEY", "TIMEOUT", "MODEL_NAME",
        "INSTANCE_NAME", "VERIFY_SSL_CERTIFICATE", "CERTIFICATE_FILE_PATH",
        "BINDING_NAME", "N_THREADS", "VERIFY", "SSL", "MODEL",
        "INSTANCE", "TOKEN", "KEY", "URL", "PORT", "HOST",
    }
    # We look for "_" followed by a word that is a known key start
    parts = remainder.split("_")
    for i in range(1, len(parts)):
        candidate_key = "_".join(parts[i:])
        if candidate_key.upper() in KNOWN_KEYS or any(candidate_key.upper().startswith(k) for k in KNOWN_KEYS):
            # Reconstruct position: find the index of this "_" in remainder
            pos = 0
            for _ in range(i):
                pos = remainder.find("_", pos)
                pos += 1
            return pos - 1  # Return index of the "_" before the key
    # Fallback: first underscore splits alias from key
    return remainder.find("_")



def _sanitize_alias(alias: str) -> str:
    """Normalizes a profile/binding alias into an env-safe, case-insensitive registry key."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(alias).strip().lower())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"a_{cleaned}"
    return cleaned


def _extract_bindings_from_env(prefix: str, env_data: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    clean_prefix = prefix.rstrip("_").upper()
    binding_prefix = f"{clean_prefix}_BINDINGS_"

    # Step 1: Discover all genuine binding aliases defined by BINDING_NAME
    known_aliases: Dict[str, str] = {}
    for k, v in env_data.items():
        k_upper = k.upper()
        if k_upper.startswith(binding_prefix) and k_upper.endswith("_BINDING_NAME"):
            raw_alias = k_upper[len(binding_prefix):-len("_BINDING_NAME")]
            if raw_alias:
                known_aliases[raw_alias] = _sanitize_alias(raw_alias)

    if not known_aliases:
        return {}

    bindings: Dict[str, Dict[str, Any]] = {}
    for raw_alias, sanitized in known_aliases.items():
        bindings[sanitized] = {}

    sorted_aliases = sorted(known_aliases.keys(), key=len, reverse=True)

    # Step 2: Match parameters to verified aliases
    for k, v in env_data.items():
        k_upper = k.upper()
        if not k_upper.startswith(binding_prefix):
            continue
        remainder = k_upper[len(binding_prefix):]

        matched_raw_alias = None
        param_key = None

        for a in sorted_aliases:
            marker = f"{a}_"
            if remainder.startswith(marker):
                matched_raw_alias = a
                param_key = remainder[len(marker):].lower()
                break

        if not matched_raw_alias or not param_key:
            continue

        alias = known_aliases[matched_raw_alias]

        if param_key == "binding_name":
            bindings[alias]["binding_name"] = v
        elif param_key == "verify_ssl_certificate":
            bool_ssl = _convert_to_bool(v)
            bindings[alias]["verify_ssl_certificate"] = bool_ssl
            bindings[alias].setdefault("binding_config", {})["verify_ssl_certificate"] = bool_ssl
        elif param_key == "certificate_file_path":
            cert_val = str(v).strip()
            bindings[alias]["certificate_file_path"] = cert_val
            bindings[alias].setdefault("binding_config", {})["certificate_file_path"] = cert_val
        else:
            bindings[alias].setdefault("binding_config", {})[param_key] = v

    return {a: b for a, b in bindings.items() if b.get("binding_name")}

_PROFILE_KNOWN_KEYS = (
    "BINDING_ALIAS", "BINDING_NAME", "MODEL_NAME", "IS_DEFAULT",
    "VISION_ENABLED", "VIDEO_ENABLED", "IMAGE_TOKEN_COST",
    "THINKING_ENABLED", "TOOL_CALLING_ENABLED",
    "SUPPORTED_REASONING_EFFORTS", "DEFAULT_REASONING_EFFORT",
    "FORCED_CONTEXT_SIZE", "VERIFY_SSL_CERTIFICATE",
    "INSTANCE_NAME",
)

# Keys whose values must be coerced when edited via the wizard.
_PROFILE_BOOL_KEYS = frozenset({
    "VISION_ENABLED", "VIDEO_ENABLED", "IS_DEFAULT",
    "VERIFY_SSL_CERTIFICATE", "THINKING_ENABLED", "TOOL_CALLING_ENABLED",
})
_PROFILE_INT_KEYS = frozenset({"IMAGE_TOKEN_COST", "FORCED_CONTEXT_SIZE"})

# Defaults backfilled into existing LLM profiles so the edit menu always
# exposes the full set of profile-level attributes.
_LLM_PROFILE_DEFAULTS = {
    "VISION_ENABLED": "false",
    "VIDEO_ENABLED": "false",
    "IMAGE_TOKEN_COST": "256",
    "THINKING_ENABLED": "false",
    "TOOL_CALLING_ENABLED": "false",
    "SUPPORTED_REASONING_EFFORTS": "low, medium, high",
    "DEFAULT_REASONING_EFFORT": "",
}


def _auto_promote_default(profiles: Dict[str, Dict[str, Any]], registry_order: List[str]) -> None:
    """
    READ-TIME AUTO-PROMOTION (SINGLE-DEFAULT INVARIANT, read side).
    If a modality has profiles but none flagged as default, the FIRST profile
    (insertion order from the env map) is promoted in-memory. If several are
    flagged, the first flagged one wins and the others are demoted. This never
    writes to disk — persistence only happens when the user saves.
    """
    if not profiles:
        return
    flagged = [
        alias for alias, p_data in profiles.items()
        if isinstance(p_data, dict) and p_data.get("is_default")
    ]
    if len(flagged) == 1:
        return
    owner = flagged[0] if flagged else registry_order[0]
    for alias, p_data in profiles.items():
        if isinstance(p_data, dict):
            p_data["is_default"] = (alias == owner)


def _extract_profiles_from_env(prefix: str, bindings: Dict[str, Dict[str, Any]], env_data: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    clean_prefix = prefix.rstrip("_").upper()
    profiles = {}
    profile_prefix = f"{clean_prefix}_PROFILES_"
    default_flag_counter = 0

    for k, v in env_data.items():
        k_upper = k.upper()
        if not k_upper.startswith(profile_prefix):
            continue
        remainder = k_upper[len(profile_prefix):]

        alias = None
        key = None
        for known_key in _PROFILE_KNOWN_KEYS:
            marker = f"_{known_key}"
            if remainder.endswith(marker):
                alias = remainder[:-len(marker)]
                key = known_key
                break
        if key is None and "_ROUTING_" in remainder:
            idx = remainder.find("_ROUTING_")
            alias = remainder[:idx]
            key = "ROUTING"
        if alias is None or not alias:
            continue

        p_alias = _sanitize_alias(alias)
        if not p_alias:
            continue
        if p_alias not in profiles:
            profiles[p_alias] = {}

        if key == "BINDING_ALIAS":
            profiles[p_alias]["binding_alias"] = _sanitize_alias(v)
        elif key == "BINDING_NAME":
            profiles[p_alias]["binding_name"] = v
        elif key == "MODEL_NAME":
            profiles[p_alias]["model_name"] = v
        elif key == "INSTANCE_NAME":
            profiles[p_alias]["instance_name"] = v
        elif key == "IS_DEFAULT":
            profiles[p_alias]["is_default"] = _convert_to_bool(v)
            profiles[p_alias]["_default_flag_order"] = default_flag_counter
            default_flag_counter += 1
        elif key == "VISION_ENABLED":
            profiles[p_alias]["vision_enabled"] = _convert_to_bool(v)
        elif key == "VIDEO_ENABLED":
            profiles[p_alias]["video_enabled"] = _convert_to_bool(v)
        elif key == "IMAGE_TOKEN_COST":
            try:
                profiles[p_alias]["image_token_cost"] = int(v)
            except (TypeError, ValueError):
                pass
        elif key == "THINKING_ENABLED":
            profiles[p_alias]["thinking_enabled"] = _convert_to_bool(v)
        elif key == "TOOL_CALLING_ENABLED":
            profiles[p_alias]["tool_calling_enabled"] = _convert_to_bool(v)
        elif key == "SUPPORTED_REASONING_EFFORTS":
            raw = str(v or "").strip()
            if raw:
                efforts = [s.strip().lower() for s in raw.split(",") if s.strip()]
                profiles[p_alias]["supported_reasoning_efforts"] = efforts or ["low", "medium", "high"]
            else:
                profiles[p_alias]["supported_reasoning_efforts"] = ["low", "medium", "high"]
        elif key == "DEFAULT_REASONING_EFFORT":
            sv = str(v or "").strip().lower()
            profiles[p_alias]["default_reasoning_effort"] = sv or None
        elif key == "FORCED_CONTEXT_SIZE":
            try:
                profiles[p_alias]["forced_context_size"] = int(v)
            except (TypeError, ValueError):
                pass
        elif key == "ROUTING":
            routing_key = remainder[idx + len("_ROUTING_"):].lower()
            profiles[p_alias].setdefault("routing_config", {})[routing_key] = v
        elif key == "VERIFY_SSL_CERTIFICATE":
            profiles[p_alias].setdefault("binding_config", {})["verify_ssl_certificate"] = _convert_to_bool(v)

    resolved_profiles = {}
    if profiles:
        _auto_promote_default(profiles, registry_order=list(profiles.keys()))
    for p_alias, p_data in profiles.items():
        b_alias = _sanitize_alias(p_data.get("binding_alias") or "") or None
        b_info = bindings.get(b_alias, {}) if b_alias else {}
        binding_name = p_data.get("binding_name") or b_info.get("binding_name")
        base_b_config = b_info.get("binding_config", {})
        profile_b_config = {k: v for k, v in p_data.items() if k not in {
            "binding_alias", "is_default",
            "vision_enabled", "video_enabled", "image_token_cost",
            "thinking_enabled", "supported_reasoning_efforts", "default_reasoning_effort",
            "forced_context_size", "model_name", "instance_name",
            "binding_name", "routing_config",
            "tool_calling_enabled",
        }}
        merged_b_config = {**base_b_config, **profile_b_config}
        if "model_name" in p_data:
            merged_b_config["model_name"] = p_data["model_name"]
        # For connection bindings, instance_name maps to model_name conceptually
        elif "instance_name" in p_data:
            merged_b_config["model_name"] = p_data["instance_name"]
        if not binding_name:
            continue

        # Use instance_name as model_name for connections when model_name not set
        effective_model_name = p_data.get("model_name") or p_data.get("instance_name")

        resolved_profiles[p_alias] = {
            "binding_name": binding_name,
            "binding_alias": b_alias,
            "binding_profile_name": b_alias,
            "binding_config": merged_b_config,
            "model_name": effective_model_name,
            "is_default": p_data.get("is_default", False),
            "vision_enabled": p_data.get("vision_enabled", False),
            "video_enabled": p_data.get("video_enabled", False),
            "image_token_cost": p_data.get("image_token_cost", 256),
            "thinking_enabled": p_data.get("thinking_enabled", False),
            "tool_calling_enabled": p_data.get("tool_calling_enabled", False),
            "supported_reasoning_efforts": p_data.get("supported_reasoning_efforts") or ["low", "medium", "high"],
            "default_reasoning_effort": p_data.get("default_reasoning_effort") or None,
            "forced_context_size": p_data.get("forced_context_size"),
            "routing_config": p_data.get("routing_config", {})
        }
    return resolved_profiles

def _get_configured_aliases(binding_type: str, config_map: Dict[str, str], category: str = "BINDINGS") -> List[str]:
    cat = category.upper()
    prefix = f"{binding_type.rstrip('_').upper()}_{cat}_"
    aliases = set()

    if cat == "BINDINGS":
        # Discovered strictly from _BINDING_NAME declaration
        for k, v in config_map.items():
            k_upper = k.upper()
            if k_upper.startswith(prefix) and k_upper.endswith("_BINDING_NAME"):
                alias = k_upper[len(prefix):-len("_BINDING_NAME")]
                if alias:
                    aliases.add(alias)
        return sorted(aliases)

    elif cat == "PROFILES":
        # Discovered strictly from _BINDING_ALIAS or known profile keys
        for k, v in config_map.items():
            k_upper = k.upper()
            if not k_upper.startswith(prefix):
                continue
            remainder = k_upper[len(prefix):]
            if remainder.endswith("_BINDING_ALIAS"):
                alias = remainder[:-len("_BINDING_ALIAS")]
                if alias:
                    aliases.add(alias)
            else:
                for known_key in _PROFILE_KNOWN_KEYS:
                    marker = f"_{known_key}"
                    if remainder.endswith(marker):
                        alias = remainder[:-len(marker)]
                        if alias:
                            aliases.add(alias)
                        break
                if "_ROUTING_" in remainder:
                    idx = remainder.find("_ROUTING_")
                    if idx > 0:
                        alias = remainder[:idx]
                        if alias:
                            aliases.add(alias)
        return sorted(aliases)

    return sorted(aliases)

def _get_binding_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_BINDINGS_{alias.upper()}_"
    return {k[len(prefix):].upper(): v for k, v in config_map.items() if k.upper().startswith(prefix)}

def _get_profile_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_PROFILES_{alias.upper()}_"
    return {k[len(prefix):].upper(): v for k, v in config_map.items() if k.upper().startswith(prefix)}

def _get_configured_command_aliases(binding_type: str, config_map: Dict[str, str]) -> List[str]:
    """Returns the ``<binding_alias>_<command_alias>`` identifiers currently mapped.

    Command env keys live at ``<MOD>_CMDS_<binding_alias>_<command_alias>_<PARAM>``.
    We discover them by scanning for key suffixes that mark a well-formed
    command entry: ``_NAME`` (required) — this is what ``_extract_commands_from_env``
    uses too. The returned identifiers are the *full* aliases (including the
    binding alias prefix), because the wizard routes them back through
    ``_edit_keys_menu`` which builds ``<MOD>_CMDS_<alias>_`` prefixes.
    """
    mod = binding_type.rstrip("_").upper()
    common_prefix = f"{mod}_CMDS_"
    aliases: set = set()
    for k in config_map:
        ku = k.upper()
        if not ku.startswith(common_prefix) or not ku.endswith("_NAME"):
            continue
        remainder = ku[len(common_prefix):-len("_NAME")]
        if remainder:
            aliases.add(remainder.lower())
    return sorted(aliases)

def _get_binding_commands(b_name: str, b_type: str) -> List[Dict[str, Any]]:
    """Returns the ``commands:`` declarations from a binding's description.yaml.

    Command entries use the shape documented in DOC_USE/DOC_DEV:
        commands:
          - name: pull_model
            title: Pull Model from Hugging Face
            description: "…"
            parameters: [...]
            output: [...]
          - name: bind_multimodal_model
            ...
    Returns ``[]`` when the binding declares no commands or the descriptor is
    unavailable.
    """
    if not b_name:
        return []
    try:
        desc = _get_binding_description(b_name, b_type) or {}
    except Exception:
        return []
    raw = desc.get("commands")
    if not isinstance(raw, list):
        return []
    commands: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        key = name.strip().upper()
        if key in seen:
            continue
        seen.add(key)
        commands.append({
            "name": name.strip(),
            "title": str(entry.get("title") or name).strip(),
            "description": str(entry.get("description") or "").strip(),
            "parameters": entry.get("parameters") or [],
            "output": entry.get("output") or [],
        })
    return commands

def _command_env_prefix(b_type: str, binding_alias: str, command_alias: str) -> str:
    """Env-prefix for a single command: <MOD>_CMDS_<binding_alias>_<command_alias>_."""
    return (
        f"{b_type.upper()}_CMDS_{_sanitize_alias(binding_alias).upper()}_"
        f"{_sanitize_alias(command_alias).upper()}_"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified Client Resolver
# ─────────────────────────────────────────────────────────────────────────────

_LAZY_MODALITIES = frozenset({"tti", "tts", "stt", "ttm", "ttv", "connection", "rag"})


def _demote_non_llm_defaults(
    b_type: str,
    profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Prevents eager instantiation of non-LLM modalities at client startup.

    ``LollmsClient.__init__`` calls ``_eagerly_instantiate_default`` for every
    modality whose registry contains a profile flagged ``is_default=True``.
    For local spawner bindings (e.g. diffusers) this spawns a subprocess
    daemon even when the user only intended to use a remote LLM/TTI profile.

    This function strips ``is_default`` from all profiles of non-LLM
    modalities so the client constructor skips them.  Actual instantiation
    is deferred to the first explicit ``switch_*()`` call or modality use.

    LLM is intentionally untouched — it is the core engine and must be
    eagerly available.
    """
    if b_type in _LAZY_MODALITIES:
        for p_data in profiles.values():
            if isinstance(p_data, dict) and p_data.get("is_default"):
                p_data["is_default"] = False
    return profiles


def _is_modality_configured(b_type: str, env_data: Dict[str, str]) -> bool:
    prefix = b_type.upper()
    if env_data.get(f"{prefix}_BINDING_NAME"):
        return True
    for k, v in env_data.items():
        ku = k.upper()
        if not v:
            continue
        if ku.startswith(f"{prefix}_BINDINGS_") and ku.endswith("_BINDING_NAME"):
            return True
        if ku.startswith(f"{prefix}_PROFILES_") and (ku.endswith("_BINDING_ALIAS") or ku.endswith("_BINDING_NAME") or ku.endswith("_MODEL_NAME")):
            return True
    return False

def get_client_from_env(
    cli_env_path: Optional[str] = None,
    conf_dict: Optional[Dict[str, Any]] = None,
    conf_file: Optional[Union[str, Path]] = None,
    entry: Optional[str] = None,
    create_llm: bool = True,
    create_tti: Optional[bool] = None,
    create_stt: Optional[bool] = None,
    create_tts: Optional[bool] = None,
    create_ttm: Optional[bool] = None,
    create_ttv: Optional[bool] = None,
    create_connection: Optional[bool] = None,
    create_rag: Optional[bool] = None,
    run_wizard_if_fail: bool = True
) -> "LollmsClient":
    from lollms_client import LollmsClient

    resolved_env: Dict[str, str] = {}

    # Prioritize the shared, machine-wide config file (respects $LOLLMS_CONFIG_FILE).
    home_yaml = _get_default_config_file()
    if home_yaml.exists():
        try:
            yaml_data = load_yaml_file(home_yaml)
            resolved_env.update(_flatten_dict_to_env(yaml_data))
        except Exception as e:
            ASCIIColors.warning(f"Failed to parse global config at {home_yaml}: {e}")

    # Then local .lollms_code/config.yaml
    local_yaml = Path.cwd() / ".lollms_code" / "config.yaml"
    if local_yaml.exists():
        try:
            yaml_data = load_yaml_file(local_yaml)
            resolved_env.update(_flatten_dict_to_env(yaml_data))
        except Exception as e:
            ASCIIColors.warning(f"Failed to parse local config at {local_yaml}: {e}")

    # Finally, the specified conf_file (if any)
    if conf_file:
        p = Path(conf_file).expanduser()
        if p.exists() and p.suffix in (".yaml", ".yml"):
            try:
                yaml_data = load_yaml_file(p)
                data = _descend_into_entry(yaml_data, entry)
                resolved_env.update(_flatten_dict_to_env(data))
            except Exception as e:
                ASCIIColors.warning(f"Failed to parse configuration at {conf_file}: {e}")

    # Apply conf_dict overrides
    if conf_dict:
        resolved_env.update(_flatten_dict_to_env(conf_dict))

    # 📂 Pop the client-wide data folder out of the flat map so it does not
    # leak into modality resolution.
    lollms_data_folder = resolved_env.pop("LOLLMS_DATA_FOLDER", None)

    if create_llm and not _is_modality_configured("llm", resolved_env):
        if run_wizard_if_fail:
            ASCIIColors.yellow("⚠️ LLM Configuration incomplete. Starting wizard...")
            run_wizard_and_save()
            home_yaml = _get_default_config_file()
            if home_yaml.exists():
                try:
                    yaml_data = load_yaml_file(home_yaml)
                    resolved_env.update(_flatten_dict_to_env(yaml_data))
                except Exception:
                    pass
            if not _is_modality_configured("llm", resolved_env):
                raise ValueError("Wizard completed but LLM configuration is still missing.")
        else:
            raise ValueError("LLM configuration is missing.")

    auto_detect_map = {
        "tti": create_tti, "tts": create_tts, "stt": create_stt,
        "ttm": create_ttm, "ttv": create_ttv, "connection": create_connection,
        "rag": create_rag,
    }
    # Opt-in semantics: None means "do not create". Only True triggers loading.
    for mod, flag in auto_detect_map.items():
        if flag is None:
            auto_detect_map[mod] = False

    kwargs = {}
    binding_types = {
        "llm": create_llm,
        **auto_detect_map,
    }

    for b_type, should_create in binding_types.items():
        if not should_create:
            continue
        prefix = b_type.upper()

        bindings = _extract_bindings_from_env(prefix, resolved_env)
        profiles = _extract_profiles_from_env(prefix, bindings, resolved_env)

        if bindings:
            kwargs[f"{b_type}_binding_profiles"] = bindings
        if profiles:
            kwargs[f"{b_type}_model_profiles"] = profiles

        if bindings or profiles:
            ASCIIColors.info(f"[get_client_from_env] Hydrated {b_type.upper()} profiles into client.")

        default_profile = next((p for p in profiles.values() if p.get("is_default")), None)
        if not default_profile and profiles:
            default_profile = next(iter(profiles.values()))

        binding_name = None
        binding_config = {}

        if default_profile:
            binding_name = default_profile.get("binding_name")
            binding_config = dict(default_profile.get("binding_config", {}))
        elif bindings:
            first_alias = next(iter(bindings))
            binding_name = bindings[first_alias].get("binding_name")
            binding_config = dict(bindings[first_alias].get("binding_config", {}))

        if binding_name:
            if default_profile and default_profile.get("model_name") and "model_name" not in binding_config:
                binding_config["model_name"] = default_profile["model_name"]

            kwargs[f"{b_type}_binding_name"] = binding_name
            kwargs[f"{b_type}_binding_config"] = binding_config

    if lollms_data_folder:
        kwargs["lollms_data_folder"] = lollms_data_folder

    return LollmsClient(**kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive Wizard
# ─────────────────────────────────────────────────────────────────────────────

def _list_llm_bindings() -> List[str]:
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        return [b if isinstance(b, str) else b.get("name") for b in list_bindings("llm") if b]
    except: return ["ollama", "openai", "lollms", "vllm", "llama_cpp_server"]

def _list_bindings_by_type(b_type: str) -> List[str]:
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        return [b if isinstance(b, str) else b.get("name") for b in list_bindings(b_type) if b]
    except: return []

def _list_connection_bindings() -> List[str]:
    """List connection bindings by scanning the connection_bindings directory."""
    try:
        from lollms_client.lollms_connection_binding import get_available_bindings
        return [b.get("binding_name") or b.get("title", "") for b in get_available_bindings() if b]
    except Exception:
        return []

def _get_binding_description(b_name: str, b_type: str) -> Optional[Dict[str, Any]]:
    try:
        from lollms_client.lollms_bindings_utils import get_binding_desc
        d = get_binding_desc(b_name, b_type)
        return d if isinstance(d, dict) and "error" not in d else None
    except: return None

def _convert_value(raw: str, p_type: str) -> Any:
    from lollms_client.lollms_utilities import to_int, to_float

    if p_type == "bool": return raw.lower() in ("true", "1", "yes", "y")
    elif p_type == "int":
        coerced = to_int(raw)
        return coerced if coerced is not None else raw
    elif p_type == "float":
        coerced = to_float(raw)
        return coerced if coerced is not None else raw
    return raw

def _format_env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

def _safe_input(prompt: str, default: str = "") -> str:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    except EOFError: return default

def _safe_input_or_back(prompt: str, default: str = "") -> Optional[str]:
    """Text input where entering 'b' or 'back' (case-insensitive) returns None to go back."""
    try:
        val = input(f"{prompt} [{default}] (b=back): ").strip()
        if val.lower() in ("b", "back"):
            return None
        return val if val else default
    except EOFError: return default

def _safe_select(prompt: str, choices: List[str], allow_back: bool = True) -> Optional[str]:
    """Interactive selection with an explicit Back/Cancel escape entry.

    Returns the selected choice string, or None when the user goes back.
    """
    escape_label = "↩ Back" if allow_back else "🚫 Cancel"
    choices_with_escape = list(choices) + [escape_label]
    try:
        menu = Menu(prompt, mode=Menu.MODE_RETURN)
        menu.set_intro("Use arrow keys to navigate. Select an option and press Enter.")
        for c in choices_with_escape: menu.add_choice(c, value=c)
        selection = menu.run()
        if selection == escape_label:
            return None
        return selection
    except:
        ASCIIColors.yellow(f"\n(Fallback) {prompt}")
        for i, c in enumerate(choices_with_escape): ASCIIColors.cyan(f"  {i+1}. {c}")
        raw = _safe_input("Enter number", str(len(choices_with_escape)))
        try:
            val = int(raw)
            if 1 <= val <= len(choices):
                return choices[val-1]
            return None
        except: return None

def _safe_confirm(prompt: str, default: bool = False) -> bool:
    """Plain-text confirmation. Deliberately avoids the interactive Menu so it
    can be safely invoked from inside another menu's callback (nested menus
    corrupt the terminal state and silently return defaults)."""
    raw = _safe_input(f"{prompt} (y/n)", "y" if default else "n")
    return raw.lower().startswith("y")

_BACK_SENTINEL = "__WIZARD_BACK__"
_BACK_VALUE = "__BACK__"


def _is_back_choice(selection: Any) -> bool:
    """Checks whether the user's menu selection represents a Back or Exit action."""
    if selection is None or selection is False or selection == _BACK_VALUE:
        return True
    if isinstance(selection, str):
        clean = selection.strip()
        if clean in (
            _BACK_VALUE, "↩ Back", "Back", "back", "b", "↩",
            "🚪 Exit without Saving", "↩ Back to Chat", "↩ Back to Main Menu"
        ):
            return True
        if clean.endswith("Back") or clean.startswith("↩"):
            return True
    if isinstance(selection, dict):
        val = selection.get("value") or selection.get("name") or selection.get("title")
        return _is_back_choice(val)
    return False


def _prompt_param(name: str, desc: str, ptype: str, mandatory: bool, default: Any) -> Any:
    from lollms_client.lollms_utilities import to_int, to_float

    ASCIIColors.rich_print(f"\n[bold cyan]── {name} ──[/bold cyan]")
    if desc: ASCIIColors.rich_print(f"[dim]{desc[:120]}{'...' if len(desc)>120 else ''}[/dim]")
    ASCIIColors.rich_print(f"Type: [yellow]{ptype}[/yellow] {'[red](required)[/red]' if mandatory else '[dim](optional)[/dim]'}")
    if ptype == "bool":
        return _safe_confirm("Enter yes/no:", default if isinstance(default, bool) else False)
    elif ptype in ("int", "float"):
        ans = _safe_input_or_back("Enter value", str(default) if default is not None else "")
        if ans is None:
            return _BACK_SENTINEL
        if not ans.strip() and mandatory:
            ASCIIColors.red("  ⚠ Required. Please enter a numeric value.")
            return _prompt_param(name, desc, ptype, mandatory, default)
        coerced = to_int(ans) if ptype == "int" else to_float(ans)
        if coerced is None:
            if mandatory:
                ASCIIColors.red("  ⚠ Not a valid number. Please enter a numeric value.")
                return _prompt_param(name, desc, ptype, mandatory, default)
            return _convert_value(ans, ptype)
        return coerced
    else:
        ans = _safe_input_or_back("Enter value", str(default) if default is not None else "")
        if ans is None:
            return _BACK_SENTINEL
        if not ans.strip() and mandatory:
            ASCIIColors.red("  ⚠ Required. Please enter a value.")
            return _prompt_param(name, desc, ptype, mandatory, default)
        return _convert_value(ans, ptype)

def _normalize_legacy_input_parameters(entries: Any) -> List[Dict[str, Any]]:
    if not isinstance(entries, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            item = dict(entry)
            if "default" in item and "default_value" not in item:
                item["default_value"] = item.pop("default")
            normalized.append(item)
    return normalized


def _normalize_legacy_parameters_dict(desc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Rebuilds a modern `parameters` dict from a legacy `input_parameters` list.

    Legacy schema: list of {'name', 'type', 'description', 'default', ...}.
    Modern schema: dict keyed by parameter name.
    """
    params_dict: Dict[str, Dict[str, Any]] = {}
    for item in _normalize_legacy_input_parameters(desc.get("input_parameters", [])):
        params_dict[item["name"]] = item
    return params_dict


def _configure_binding_instance(b_type: str, b_name: str, alias: str, config_map: Dict[str, str]):
    prefix = f"{b_type.upper()}_BINDINGS_{alias.upper()}_"
    temp_params: Dict[str, str] = {"BINDING_NAME": b_name}
    ASCIIColors.green(f"\n  ✓ Selected {b_type.upper()} binding: {b_name} (Alias: {alias})")

    desc = _get_binding_description(b_name, b_type)
    if desc:
        raw_params = desc.get("global_input_parameters", []) + desc.get("model_input_parameters", [])
        if not raw_params:
            raw_params = _normalize_legacy_input_parameters(desc.get("input_parameters", []))
        seen_names = set()
        params = []
        for p in raw_params:
            pname = p.get("name", "")
            if not pname or pname == "model_name":
                continue
            name_key = pname.lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            params.append(p)
        for p in params:
            pname = p.get("name", "")
            val = _prompt_param(pname, p.get("description", ""), p.get("type", "str"), p.get("mandatory", False), p.get("default"))
            if val is _BACK_SENTINEL:
                ASCIIColors.yellow("\n  ⚠️ Binding configuration cancelled.")
                return
            temp_params[pname.upper()] = _format_env_value(val)
    else:
        ASCIIColors.yellow("\n  No description.yaml found. Using standard server configuration.\n")
        default_host = "http://localhost:9642" if b_type in ("tti", "tts", "stt") else "http://localhost:8000"
        host_val = _prompt_param("host_address", f"The host address of the {b_type.upper()} server", "str", False, default_host)
        if host_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Binding configuration cancelled.")
            return
        temp_params["HOST_ADDRESS"] = _format_env_value(host_val)

        key_val = _prompt_param("service_key", f"API / Service Key for the {b_type.upper()} server (leave blank if none)", "str", False, "")
        if key_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Binding configuration cancelled.")
            return
        if key_val:
            temp_params["SERVICE_KEY"] = _format_env_value(key_val)

        ssl_val = _prompt_param("verify_ssl_certificate", "Verify SSL certificate", "bool", False, False)
        if ssl_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Binding configuration cancelled.")
            return
        temp_params["VERIFY_SSL_CERTIFICATE"] = _format_env_value(ssl_val)

    for k, v in temp_params.items():
        config_map[prefix + k] = v
    ASCIIColors.green(f"\n  ✓ Successfully configured {b_type.upper()} binding '{alias}'.")

def _generate_unique_alias(base: str, existing_aliases: List[str]) -> str:
    """Generates a unique alias by auto-incrementing if the base name already exists.

    If 'master' exists, returns 'master_2', 'master_3', etc.
    The sanitized base is used for collision detection.
    """
    sanitized_base = _sanitize_alias(base)
    if sanitized_base not in existing_aliases:
        return sanitized_base

    counter = 2
    while f"{sanitized_base}_{counter}" in existing_aliases:
        counter += 1
    return f"{sanitized_base}_{counter}"


def _add_binding_flow(b_type: str, config_map: Dict[str, str]):
    bindings = _list_bindings_by_type(b_type)
    if not bindings: return
    selected = _safe_select(f"Select a {b_type.upper()} binding:", bindings)
    if not selected:
        ASCIIColors.yellow("\n  ⚠️ Binding selection cancelled.")
        return
    existing = _get_configured_aliases(b_type, config_map, "BINDINGS")
    raw_alias = _safe_input_or_back("Enter an alias for this binding", "master")
    if raw_alias is None:
        ASCIIColors.yellow("\n  ⚠️ Binding creation cancelled.")
        return
    if not raw_alias.strip():
        alias = _generate_unique_alias("master", existing)
        ASCIIColors.info(f"  ℹ️ Auto-named binding: {alias}")
    else:
        alias = _generate_unique_alias(raw_alias, existing)
        if alias != _sanitize_alias(raw_alias):
            ASCIIColors.info(f"  ℹ️ Name collision resolved: {alias}")
    if alias: _configure_binding_instance(b_type, selected, alias, config_map)

def _rename_entry(b_type: str, category: str, old_alias: str, config_map: Dict[str, str]) -> Optional[str]:
    """
    Renames a binding or profile alias by rewriting all keys in config_map.
    Example: LLM_BINDINGS_MASTER_HOST -> LLM_BINDINGS_NEW_ALIAS_HOST.
    Ensures referential integrity for profiles pointing to renamed bindings.
    """
    cat_upper = category.upper()
    old_alias_clean = _sanitize_alias(old_alias)
    raw_alias = _safe_input_or_back(f"Enter new instance name for '{old_alias}'", old_alias_clean)
    if raw_alias is None:
        ASCIIColors.yellow("\n  ⚠️ Rename cancelled.")
        return None

    new_alias = _sanitize_alias(raw_alias)
    if new_alias != raw_alias.strip().lower():
        ASCIIColors.info(f"  ℹ️ Sanitized alias to: '{new_alias}'")

    if new_alias == old_alias_clean:
        ASCIIColors.yellow("\n  ⚠️ Alias unchanged.")
        return None

    existing = _get_configured_aliases(b_type, config_map, cat_upper)
    if any(_sanitize_alias(a) == new_alias for a in existing):
        ASCIIColors.red(f"\n  ❌ Instance name '{new_alias}' already exists for {b_type.upper()} {cat_upper.lower()}.")
        return None

    old_prefix = f"{b_type.upper()}_{cat_upper}_{old_alias_clean}_"
    new_prefix = f"{b_type.upper()}_{cat_upper}_{new_alias}_"

    # We must do exact uppercase matching on keys to prevent substring collisions
    keys_to_rename = [k for k in config_map.keys() if k.upper().startswith(old_prefix.upper())]
    if not keys_to_rename:
        ASCIIColors.yellow(f"\n  ⚠️ No keys found for instance '{old_alias}'.")
        return None

    for k in keys_to_rename:
        suffix = k[len(old_prefix):]
        config_map[new_prefix + suffix] = config_map.pop(k)

    # Ensure referential integrity across profiles for the given modality
    profile_prefix = f"{b_type.upper()}_PROFILES_"
    for k in list(config_map.keys()):
        if k.upper().startswith(profile_prefix) and k.upper().endswith("_BINDING_ALIAS"):
            current_ref = _sanitize_alias(config_map[k])
            if current_ref == old_alias_clean:
                config_map[k] = new_alias
                ASCIIColors.info(f"  ℹ️ Updated profile reference '{k}' to point to '{new_alias}'")

    if cat_upper != "BINDINGS":
        _enforce_single_default_profile(b_type, config_map)

    label = "binding" if cat_upper == "BINDINGS" else "profile"
    ASCIIColors.green(f"\n  🏷️ Renamed {label} '{old_alias}' → '{new_alias}'.")
    return new_alias

def _bindings_menu(b_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{b_type.upper()} Bindings", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Add a new binding, edit, or delete an existing one.")
        menu.add_choice("➕ Add new binding", value=lambda: _add_binding_flow(b_type, config_map))

        existing_aliases = _get_configured_aliases(b_type, config_map, "BINDINGS")
        for alias in existing_aliases:
            menu.add_choice(f"✏️ Edit binding: {alias}", value=lambda a=alias: _edit_keys_menu(b_type, "BINDINGS", a, config_map))

        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()

def _commands_menu(b_type: str, binding_alias: str, config_map: Dict[str, str]) -> None:
    """Interactive menu to enable / edit / disable a binding's commands.

    Commands are declared in the binding's description.yaml (``commands:``) and
    materialized as env-keys under ``<MOD>_CMDS_<binding_alias>_<cmd>_*``. This
    menu presents:
      * the commands the binding author declared, and
      * the commands already enabled into the config map (which may include
        user-added commands, or ones from an older description).

    Actions are dispatched as plain prompts (``_safe_select`` /
    ``_safe_confirm`` / ``_safe_input_or_back``), avoiding nested ``Menu``
    objects (which corrupted terminal state on some platforms).
    """
    binding_prefix = f"{b_type.upper()}_BINDINGS_{_sanitize_alias(binding_alias).upper()}_"
    binding_name: Optional[str] = None
    for k, v in config_map.items():
        if k.upper() == f"{binding_prefix}BINDING_NAME".upper():
            binding_name = v
            break
    if not binding_name:
        ASCIIColors.red(f"\n  ❌ Binding '{binding_alias}' has no BINDING_NAME.")
        return

    declared = _get_binding_commands(binding_name, b_type)
    declared_by_name = {c["name"].lower(): c for c in declared}

    def _mapped_cmd_aliases() -> List[str]:
        """Returns the FULL `<binding_alias>_<cmd>` id for mapped commands only."""
        mod = b_type.upper()
        prefix = f"{mod}_CMDS_{_sanitize_alias(binding_alias).upper()}_"
        alias_ids: set = set()
        for k in config_map:
            ku = k.upper()
            if not ku.startswith(prefix) or not ku.endswith("_NAME"):
                continue
            cmd_alias = ku[len(prefix):-len("_NAME")]
            if cmd_alias:
                alias_ids.add(cmd_alias.lower())
        return sorted(alias_ids)

    def _mapped_keys_for(full_alias: str) -> Dict[str, str]:
        prefix = f"{b_type.upper()}_CMDS_{full_alias.upper()}_"
        out: Dict[str, str] = {}
        for k, v in config_map.items():
            ku = k.upper()
            if ku.startswith(prefix):
                out[ku[len(prefix):]] = v
        return out

    while True:
        mapped = _mapped_cmd_aliases()

        ASCIIColors.rich_print("\n[bold cyan]── Commands ──[/bold cyan]")
        ASCIIColors.rich_print(
            f"[dim]Binding: {binding_name} (alias '{binding_alias}')\n"
            f"Declared: {len(declared)}   |   Enabled: {len(mapped)}[/dim]"
        )

        # Build a flat, navigable choice list.
        choices: List[str] = []
        # 1. Enable from description: every declared command not yet mapped.
        enableable: List[Dict[str, Any]] = []
        for c in declared:
            already_enabled = any(
                full.endswith(f"_{_sanitize_alias(c['name']).lower()}")
                for full in mapped
            )
            if not already_enabled:
                enableable.append(c)
        if enableable:
            choices.append(f"➕ Enable a declared command ({len(enableable)} available)")
        # 2. Edit existing
        if mapped:
            choices.append(f"✏️ Edit an enabled command ({len(mapped)})")
        # 3. Disable existing
        if mapped:
            choices.append(f"🗑️ Disable an enabled command ({len(mapped)})")
        # 4. Add custom command (freeform name)
        choices.append("🆕 Add a custom command (not declared in description.yaml)")
        choices.append("↩ Back")

        choice = _safe_select(f"Manage commands for '{binding_alias}'", choices)
        if not choice or choice.startswith("↩"):
            return

        # ── Enable from description ──────────────────────────────────────
        if choice.startswith("➕ Enable"):
            names = [f"{c['name']} — {c['title']}" for c in enableable]
            picked = _safe_select("Select a command to enable:", names)
            if not picked:
                continue
            target = enableable[names.index(picked)]
            cmd_alias = _sanitize_alias(target["name"])
            full_alias = f"{binding_alias}_{cmd_alias}"
            prefix = _command_env_prefix(b_type, binding_alias, cmd_alias)
            if not _safe_confirm(
                f"Enable command '{target['name']}' for binding '{binding_alias}'?",
                default=True,
            ):
                continue
            config_map[prefix + "NAME"] = target["name"]
            config_map[prefix + "TITLE"] = target["title"]
            if target["description"]:
                config_map[prefix + "DESCRIPTION"] = target["description"]

            # Prompt for declared parameters with defaults.
            for p in target.get("parameters", []) or []:
                if not isinstance(p, dict):
                    continue
                pname = str(p.get("name") or "").strip()
                if not pname:
                    continue
                mandatory = bool(p.get("mandatory", False))
                default = p.get("default")
                ptype = str(p.get("type") or "str").lower()
                if ptype == "bool":
                    val = _prompt_param(
                        pname,
                        p.get("description", ""),
                        "bool",
                        mandatory,
                        bool(default) if isinstance(default, bool) else False,
                    )
                elif ptype in ("int", "float"):
                    val = _prompt_param(
                        pname,
                        p.get("description", ""),
                        ptype,
                        mandatory,
                        "" if default is None else str(default),
                    )
                else:
                    val = _prompt_param(
                        pname,
                        p.get("description", ""),
                        "str",
                        mandatory,
                        "" if default is None else str(default),
                    )
                if val is _BACK_SENTINEL:
                    continue
                config_map[prefix + pname.upper()] = _format_env_value(val)

            ASCIIColors.green(f"\n  ✓ Enabled command: {target['name']} → {full_alias}")
            continue

        # ── Edit an enabled command ───────────────────────────────────────
        if choice.startswith("✏️ Edit"):
            picked = _safe_select("Select a command to edit:", mapped)
            if not picked:
                continue
            _edit_keys_menu(b_type, "CMDS", picked, config_map)
            continue

        # ── Disable an enabled command ─────────────────────────────────────
        if choice.startswith("🗑️ Disable"):
            picked = _safe_select("Select a command to disable:", mapped)
            if not picked:
                continue
            keys = [k for k in list(config_map.keys())
                    if k.upper().startswith(f"{b_type.upper()}_CMDS_{picked.upper()}_")]
            if not keys:
                ASCIIColors.yellow(f"\n  ⚠️ No keys found for command '{picked}'.")
                continue
            if not _safe_confirm(
                f"Disable command '{picked}' ({len(keys)} keys will be removed)?",
                default=False,
            ):
                ASCIIColors.info("\n  ℹ️ Cancelled. No changes were made.")
                continue
            for k in keys:
                del config_map[k]
            ASCIIColors.green(f"\n  🗑️ Disabled command: {picked}")
            continue

        # ── Add a custom command ───────────────────────────────────────────
        if choice.startswith("🆕 Add"):
            raw = _safe_input_or_back("Command name (e.g. my_tool)", "")
            if not raw or not raw.strip():
                continue
            cmd_alias = _sanitize_alias(raw.strip())
            full_alias = f"{binding_alias}_{cmd_alias}"
            prefix = _command_env_prefix(b_type, binding_alias, cmd_alias)
            if any(k.upper().startswith(prefix.upper()) for k in config_map):
                ASCIIColors.red(
                    f"\n  ❌ Command '{raw.strip()}' already exists for binding '{binding_alias}'."
                )
                continue
            title = _safe_input_or_back("Command title", raw.strip()) or raw.strip()
            description = _safe_input_or_back("Command description", "") or ""
            config_map[prefix + "NAME"] = raw.strip()
            config_map[prefix + "TITLE"] = title
            if description:
                config_map[prefix + "DESCRIPTION"] = description
            ASCIIColors.green(f"\n  ✓ Added custom command: {full_alias}")
            continue

def _edit_keys_menu(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    # Backfill profile-level defaults so pre-existing profiles expose the
    # full editable attribute set (vision/video/image_token_cost/...).
    if category.upper() == "PROFILES" and b_type == "llm":
        prefix_upper = f"{b_type.upper()}_{category.upper()}_{alias.upper()}_"
        existing_upper = {k.upper() for k in config_map if k.upper().startswith(prefix_upper)}
        for default_key, default_val in _LLM_PROFILE_DEFAULTS.items():
            full_key = f"{prefix_upper}{default_key}"
            if full_key not in existing_upper:
                config_map[full_key] = default_val

    # 🧩 Backfill declared per-model parameters for existing profiles so the
    # edit menu surfaces them even if the profile was created before the
    # binding started declaring them. Existing user values are preserved.
    if category.upper() == "PROFILES":
        profile_binding_alias = config_map.get(
            f"{b_type.upper()}_PROFILES_{alias.upper()}_BINDING_ALIAS"
        )
        if profile_binding_alias:
            bound_binding_name = config_map.get(
                f"{b_type.upper()}_BINDINGS_{_sanitize_alias(profile_binding_alias).upper()}_BINDING_NAME"
            )
            if bound_binding_name:
                for declared in _get_model_profile_schema(b_type, bound_binding_name):
                    pkey = f"{b_type.upper()}_PROFILES_{alias.upper()}_{declared['name'].upper()}"
                    if pkey not in config_map and declared.get("default") is not None:
                        config_map[pkey] = _format_env_value(declared["default"])

    while True:
        prefix_upper = f"{b_type.upper()}_{category.upper()}_{alias.upper()}_"

        # Snapshot the editable keys once per loop iteration (used by both the
        # BINDINGS branch and the generic edit branch below). Computing this
        # unconditionally avoids the UnboundLocalError caused by assigning it
        # only inside the non-BINDINGS branch.
        keys: Dict[str, str] = {}
        for k, v in config_map.items():
            if k.upper().startswith(prefix_upper):
                raw_k = k[len(prefix_upper):]
                keys[raw_k] = v
        self_keys_for_edit = keys

        if not keys:
            ASCIIColors.yellow(f"\n  ⚠️ No keys found for {alias}.")
            return

        if category.upper() == "BINDINGS":
            menu = Menu(
                f"Edit {b_type.upper()} {category.capitalize()}: {alias}",
                mode=Menu.MODE_RETURN,
                exit_text="↩ Back",
            )
            menu.set_intro(
                "Select a key to edit, add a custom key, rename this instance, "
                "or auto-generate model profiles from this binding."
            )
            label_str = "binding"
            menu.add_choice(f"🏷️ Rename this {label_str} instance : {alias}", value="__RENAME_ENTRY__")
            menu.add_choice(
                f"🚀 Scan binding & create missing model profiles ('{alias}')",
                value="__AUTOGEN_PROFILES__",
            )
            menu.add_choice(
                f"🔁 Re-scan & apply inferred defaults to existing profiles ('{alias}')",
                value="__REAPPLY_INFERENCE__",
            )
            menu.add_choice(
                f"🛠️ Configure Commands ('{alias}')",
                value="__COMMANDS__",
            )
            for k, v in self_keys_for_edit.items():
                if k != "BINDING_NAME":  # binding name is not editable
                    menu.add_choice(
                        f"✏️ Edit {k}: {str(v)[:40]}",
                        value=lambda k=k: _edit_single_key(b_type, category, alias, k, config_map),
                    )
            menu.add_choice("➕ Add custom key", value=lambda: _add_custom_key(b_type, category, alias, config_map))
            menu.add_choice(f"🗑️ Delete this {label_str}", value="__DELETE_ENTRY__")
            menu.add_choice("↩ Back", value=_BACK_VALUE)

            selection = menu.run()
            if _is_back_choice(selection):
                break
            if selection == "__RENAME_ENTRY__":
                new_alias = _rename_entry(b_type, category, alias, config_map)
                if new_alias:
                    alias = new_alias
            elif selection == "__AUTOGEN_PROFILES__":
                _autogenerate_profiles_for_binding(b_type, alias, config_map)
            elif selection == "__REAPPLY_INFERENCE__":
                _reapply_inference_to_existing_profiles(b_type, alias, config_map)
            elif selection == "__COMMANDS__":
                _commands_menu(b_type, alias, config_map)
            elif selection == "__DELETE_ENTRY__":
                if _delete_entry(b_type, category, alias, config_map):
                    return
            elif callable(selection):
                selection()
            continue

        menu = Menu(f"Edit {b_type.upper()} {category.capitalize()}: {alias}", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Select a key to edit, add a custom key, or rename this instance.")
        label_str = "binding" if category.upper() == "BINDINGS" else "profile"
        menu.add_choice(f"🏷️ Rename this {label_str} instance : {alias}", value="__RENAME_ENTRY__")
        for k, v in keys.items():
            if k != "BINDING_NAME": # we can't update the binding name
                menu.add_choice(f"✏️ Edit {k}: {str(v)[:40]}", value=lambda k=k: _edit_single_key(b_type, category, alias, k, config_map))
        menu.add_choice("➕ Add custom key", value=lambda: _add_custom_key(b_type, category, alias, config_map))
        if category.upper() == "PROFILES":
            _is_def_key = f"{b_type.upper()}_PROFILES_{alias.upper()}_IS_DEFAULT"
            _is_default = str(config_map.get(_is_def_key, "")).lower() in ("true", "1", "yes", "y", "on")
            if not _is_default:
                menu.add_choice(f"⭐ Set as default: {alias}", value="__SET_DEFAULT__")
        menu.add_choice(f"🗑️ Delete this {label_str}", value="__DELETE_ENTRY__")
        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if selection == "__RENAME_ENTRY__":
            new_alias = _rename_entry(b_type, category, alias, config_map)
            if new_alias:
                # Update the local alias loop to ensure the next UI iteration
                # shows the new name in the title and selections
                alias = new_alias
        elif selection == "__SET_DEFAULT__":
            _set_default_profile_action(b_type, alias, config_map)
            # Re-render the edit menu so the set-default option disappears
            # (the profile is now default) and the badge state refreshes.
            continue
        elif selection == "__DELETE_ENTRY__":
            if _delete_entry(b_type, category, alias, config_map):
                # Entry was deleted — exit the edit loop so the caller
                # re-renders the list without the removed entry.
                return
        elif callable(selection):
            selection()

def _edit_single_key(b_type: str, category: str, alias: str, key: str, config_map: Dict[str, str]):
    prefix_upper = f"{b_type.upper()}_{category.upper()}_{alias.upper()}_"
    target_key = f"{prefix_upper}{key.upper()}"
    found_key = None
    for k in config_map:
        if k.upper() == target_key:
            found_key = k
            break
    curr_val = config_map.get(found_key or target_key, "")
    new_val = _safe_input_or_back(f"Enter new value for {key}", curr_val)
    if new_val is None:
        return

    key_upper = key.upper()
    if key_upper in _PROFILE_BOOL_KEYS:
        new_val = _format_env_value(_convert_to_bool(new_val))
    elif key_upper in _PROFILE_INT_KEYS:
        try:
            new_val = str(int(new_val))
        except (TypeError, ValueError):
            ASCIIColors.red(f"  ⚠ '{new_val}' is not a valid integer for {key}. Value unchanged.")
            return

    if found_key:
        config_map[found_key] = new_val
    else:
        config_map[target_key] = new_val
    ASCIIColors.green(f"  ✓ Updated {key}")

def _add_custom_key(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    new_key = _safe_input_or_back("Enter the name of the new key (e.g., SERVICE_KEY)", "")
    if new_key is not None and new_key.strip():
        new_key_clean = new_key.strip().upper()
        new_val = _safe_input_or_back(f"Enter value for {new_key_clean}", "")
        if new_val is not None:
            config_map[f"{b_type.upper()}_{category.upper()}_{alias.upper()}_{new_key_clean}"] = new_val
            ASCIIColors.green(f"  ✓ Added {new_key_clean}")

def _delete_entry(b_type: str, category: str, alias: str, config_map: Dict[str, str]) -> bool:
    """Deletes a binding or profile and all its keys after explicit confirmation.

    Returns True when the entry was deleted, False when cancelled or when no
    matching keys were found. Callers (e.g. the edit submenu) use the return
    value to decide whether to exit the edit loop.
    """
    cat_upper = category.upper()
    prefix = f"{b_type.upper()}_{cat_upper}_{alias}_".upper()
    keys_to_delete = [k for k in list(config_map.keys()) if k.upper().startswith(prefix)]

    if not keys_to_delete:
        ASCIIColors.yellow(f"\n  ⚠️ No {category.lower()[:-1]} found with alias '{alias}'.")
        return False

    label = "binding" if cat_upper == "BINDINGS" else "profile"
    if not _safe_confirm(
        f"Are you sure you want to delete {label} '{alias}' and all its {len(keys_to_delete)} keys?",
        default=False,
    ):
        ASCIIColors.info(f"\n  ℹ️ Deletion of {label} '{alias}' cancelled. No changes were made.")
        return False

    was_default = False
    for k in keys_to_delete:
        if k.upper().endswith("_IS_DEFAULT") and str(config_map[k]).lower() in ("true", "1", "yes", "y", "on"):
            was_default = True
        del config_map[k]
    ASCIIColors.green(f"\n  🗑️ Deleted {label}: {alias}")

    if cat_upper == "BINDINGS":
        # Find and delete profiles referencing this binding
        profile_prefix = f"{b_type.upper()}_PROFILES_"
        orphaned_profiles = []
        for k, v in list(config_map.items()):
            if k.upper().startswith(profile_prefix) and k.upper().endswith("_BINDING_ALIAS") and v.upper() == alias.upper():
                profile_alias = k[len(profile_prefix):-len("_BINDING_ALIAS")].rstrip("_")
                orphaned_profiles.append(profile_alias)

        for profile_alias in orphaned_profiles:
            profile_keys = [k for k in list(config_map.keys()) if k.startswith(f"{profile_prefix}{profile_alias.upper()}_")]
            for pk in profile_keys:
                del config_map[pk]
            ASCIIColors.info(f"  🗑️ Deleted associated profile: {profile_alias}")

        if was_default:
            ASCIIColors.info("  ℹ️ Default binding deleted; first remaining binding promoted.")
    elif cat_upper == "PROFILES":
        _enforce_single_default_profile(b_type, config_map)
        if was_default:
            ASCIIColors.info("  ℹ️ Default profile deleted; first remaining profile promoted.")

    return True

def _extract_model_name(m: Any) -> Optional[str]:
    """Robustly extracts model name string from raw items (string or dict)."""
    if not m:
        return None
    if isinstance(m, str):
        name = m.strip()
        return name if name else None
    if isinstance(m, dict):
        for key in ("model_name", "name", "id", "model", "voice_name", "title"):
            val = m.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    return None

def _fetch_available_models(b_type: str, b_name: str, config_map: Dict[str, str], b_alias: Optional[str] = None) -> List[str]:
    """Fetches available model names for a given binding instance."""
    try:
        from lollms_client import LollmsClient

        b_config = {}
        desc = _get_binding_description(b_name, b_type) or {}
        params_meta = {p.get("name", "").lower(): p for p in (desc.get("global_input_parameters", []) + desc.get("model_input_parameters", []))}

        prefix = f"{b_type.upper()}_BINDINGS_{b_alias.upper()}_" if b_alias else None

        for k, v in config_map.items():
            if not v:
                continue
            key_clean = k
            if prefix and key_clean.startswith(prefix):
                key_clean = key_clean[len(prefix):]
            elif key_clean.startswith(f"{b_type.upper()}_"):
                key_clean = key_clean[len(f"{b_type.upper()}_"):]

            key_lower = key_clean.lower()
            if key_lower == "binding_name":
                continue

            p_meta = params_meta.get(key_lower, {})
            p_type = p_meta.get("type", "str").lower()

            if p_type == "bool":
                b_config[key_lower] = _convert_to_bool(v)
            elif p_type == "int":
                try:
                    b_config[key_lower] = int(v)
                except ValueError:
                    pass
            elif p_type == "float":
                try:
                    b_config[key_lower] = float(v)
                except ValueError:
                    pass
            elif v.lower() in ("null", "none"):
                b_config[key_lower] = None
            else:
                b_config[key_lower] = v

        kwargs = {f"{b_type}_binding_name": b_name, f"{b_type}_binding_config": b_config}
        temp_client = LollmsClient(**kwargs)

        raw_models = []
        if b_type == "llm" and temp_client.llm:
            raw_models = temp_client.llm.list_models()
        elif b_type == "tti" and temp_client.tti:
            raw_models = temp_client.tti.list_models()
        elif b_type == "tts" and temp_client.tts:
            if hasattr(temp_client.tts, "list_voices"):
                raw_models = temp_client.tts.list_voices()
            elif hasattr(temp_client.tts, "list_models"):
                raw_models = temp_client.tts.list_models()
        elif b_type == "stt" and temp_client.stt:
            raw_models = temp_client.stt.list_models()
        elif b_type == "ttm" and temp_client.ttm:
            raw_models = temp_client.ttm.list_models()
        elif b_type == "ttv" and temp_client.ttv:
            raw_models = temp_client.ttv.list_models()

        model_names = []
        for m in raw_models:
            name = _extract_model_name(m)
            if name:
                model_names.append(name)

        return sorted(list(set(model_names)))
    except Exception as e:
        ASCIIColors.warning(f"Could not automatically fetch models for {b_name}: {e}")
        return []

def _get_model_profile_schema(b_type: str, binding_name: Optional[str]) -> List[Dict[str, Any]]:
    """Returns the per-model (per-profile) parameter declarations for a binding.

    A binding opts into per-model configuration by declaring a
    ``model_profiles_input_parameters`` list in its ``description.yaml``.
    Each entry has the same shape as ``global_input_parameters``
    (``name``/``type``/``mandatory``/``default``/``description``).

    These keys belong to *each profile* of the binding — not to the binding
    instance itself — so spawning bindings (llama_cpp_server, llama_cpp, vLLM,
    etc.) can vary ``n_gpu_layers``, ``context_size``, ``flash_attn``, ... per
    model without duplicating the whole binding configuration.

    Returns an empty list when the binding does not declare any per-model
    parameters (backward compatible with existing descriptions).
    """
    if not binding_name:
        return []
    try:
        desc = _get_binding_description(binding_name, b_type) or {}
    except Exception:
        return []
    raw = desc.get("model_profiles_input_parameters")
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not name or not isinstance(name, str):
            continue
        clean = name.strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        # Skip keys that are framework-managed; they must not be overridden
        # by a raw model-profile parameter (would conflict with routing/alias).
        if clean.upper() in _PROFILE_KNOWN_KEYS:
            continue
        normalized.append({
            "name": clean,
            "type": entry.get("type", "str"),
            "mandatory": bool(entry.get("mandatory", False)),
            "default": entry.get("default"),
            "description": entry.get("description", "") or "",
        })
    return normalized


def _configure_model_profile_parameters(
    b_type: str,
    b_name: Optional[str],
    b_alias: Optional[str],
    temp_profile: Dict[str, str],
) -> None:
    """Prompts for a binding's declared per-model parameters.

    Bindings declare ``model_profiles_input_parameters`` in ``description.yaml``.
    This routine walks that list and asks the user for each value, writing
    ``<KEY_UPPER>`` directly into the in-progress profile dict. Blank / back
    inputs fall through to the declared ``default`` so every profile ends up
    with all required keys populated — critically important for local spawner
    bindings, where a partial profile yields an unbootable server.
    """
    schema = _get_model_profile_schema(b_type, b_name)
    if not schema:
        return

    ASCIIColors.rich_print("\n[bold magenta]── Per-Model Parameters ──[/bold magenta]")
    ASCIIColors.rich_print(
        "[dim]These values apply to this specific model profile only and are\n"
        "passed to the binding as part of its per-profile `binding_config`.[/dim]"
    )

    for param in schema:
        pname = param["name"]
        ptype = str(param.get("type") or "str").lower()
        default = param.get("default")
        mandatory = bool(param.get("mandatory", False))
        desc = param.get("description") or ""

        if ptype == "bool":
            default_bool = bool(default) if isinstance(default, bool) else str(default).lower() in ("true", "1", "yes", "on")
            val = _prompt_param(pname, desc, "bool", mandatory, default_bool)
            if val is _BACK_SENTINEL:
                temp_profile[pname.upper()] = _format_env_value(default_bool)
            else:
                temp_profile[pname.upper()] = _format_env_value(bool(val))
        elif ptype in ("int", "float"):
            default_str = "" if default is None else str(default)
            val = _prompt_param(pname, desc, ptype, mandatory, default_str)
            if val is _BACK_SENTINEL:
                if default is not None:
                    temp_profile[pname.upper()] = _format_env_value(default)
            else:
                temp_profile[pname.upper()] = _format_env_value(val)
        else:
            default_str = "" if default is None else str(default)
            val = _prompt_param(pname, desc, "str", mandatory, default_str)
            if val is _BACK_SENTINEL:
                if default is not None and str(default) != "":
                    temp_profile[pname.upper()] = _format_env_value(default)
            elif val != "" or isinstance(val, str):
                temp_profile[pname.upper()] = _format_env_value(val)

    # Bindings may declare optional overrides for vision/thinking/etc. via
    # their defaults here, but those are normally inferred. Nothing further
    # to do — the caller merges into config_map and serialization just works.


def _configure_profile_instance(b_type: str, alias: str, config_map: Dict[str, str]):
    profile_prefix = f"{b_type.upper()}_PROFILES_{alias.upper()}_"

    configured_bindings = _get_configured_aliases(b_type, config_map, "BINDINGS")
    if not configured_bindings:
        ASCIIColors.yellow(f"\n  ⚠️ No {b_type.upper()} bindings configured. Please add a binding first.")
        return

    selected_b_alias = _safe_select(f"Select binding for profile '{alias}':", configured_bindings)
    if not selected_b_alias:
        ASCIIColors.yellow(f"\n  ⚠️ Binding selection cancelled for profile '{alias}'.")
        return

    temp_profile: Dict[str, str] = {"BINDING_ALIAS": selected_b_alias}

    b_name = config_map.get(f"{b_type.upper()}_BINDINGS_{selected_b_alias.upper()}_BINDING_NAME")
    if b_name:
        available_models = _fetch_available_models(b_type, b_name, config_map, selected_b_alias)
        if available_models:
            choices = list(available_models) + ["✍️ Enter model name manually"]
            selected_model = _safe_select(f"Select {b_type.upper()} Model for '{alias}':", choices)
            if not selected_model:
                ASCIIColors.yellow(f"\n  ⚠️ Model selection cancelled for profile '{alias}'.")
                return
            if selected_model == "✍️ Enter model name manually":
                m_name = _safe_input_or_back("Enter model name manually", "")
                if m_name is None:
                    return
                if m_name:
                    temp_profile["MODEL_NAME"] = m_name
            else:
                temp_profile["MODEL_NAME"] = selected_model
        else:
            m_name = _safe_input_or_back("Enter model name manually", "")
            if m_name is None:
                return
            if m_name:
                temp_profile["MODEL_NAME"] = m_name

    existing_profiles = _get_configured_aliases(b_type, config_map, "PROFILES")
    is_first = len(existing_profiles) == 0 or (len(existing_profiles) == 1 and existing_profiles[0].upper() == alias.upper())
    if _safe_confirm(f"Make '{alias}' the default profile?", default=(alias.lower() == "master" or is_first)):
        temp_profile["IS_DEFAULT"] = "true"
    else:
        temp_profile["IS_DEFAULT"] = "false"

    if b_type == "llm":
        if _safe_confirm(f"Does profile '{alias}' support vision?", default=False):
            temp_profile["VISION_ENABLED"] = "true"
        if _safe_confirm(f"Does profile '{alias}' support video input?", default=False):
            temp_profile["VIDEO_ENABLED"] = "true"
        img_cost = _safe_input_or_back("Tokens per image (leave blank for default 256)", "")
        if img_cost is not None and img_cost.strip():
            temp_profile["IMAGE_TOKEN_COST"] = img_cost.strip()
        if _safe_confirm(f"Does profile '{alias}' support thinking / reasoning mode?", default=False):
            temp_profile["THINKING_ENABLED"] = "true"
        efforts = _safe_input_or_back(
            "Supported reasoning efforts (comma-separated, blank = low, medium, high)", ""
        )
        if efforts is not None and efforts.strip():
            temp_profile["SUPPORTED_REASONING_EFFORTS"] = efforts.strip()
        default_effort = _safe_input_or_back(
            "Default reasoning effort (blank = middle of list)", ""
        )
        if default_effort is not None and default_effort.strip():
            temp_profile["DEFAULT_REASONING_EFFORT"] = default_effort.strip().lower()
        ctx = _safe_input_or_back("Force context size? (leave blank for auto)", "")
        if ctx is not None and ctx.strip():
            temp_profile["FORCED_CONTEXT_SIZE"] = ctx.strip()

        if _safe_confirm("Configure Smart Router metadata (optional)?", default=False):
            ASCIIColors.rich_print("\n[bold magenta]── Smart Router Metadata ──[/bold magenta]")
            r_desc = _safe_input_or_back("Routing description (keywords)", "")
            if r_desc: temp_profile["ROUTING_DESCRIPTION"] = r_desc
            r_cost = _safe_input_or_back("Cost per 1k tokens (0.0 for local)", "0.0")
            if r_cost: temp_profile["ROUTING_COST"] = r_cost
            r_lat = _safe_input_or_back("Average latency (ms)", "100")
            if r_lat: temp_profile["ROUTING_LATENCY"] = r_lat
            r_comp = _safe_select("Complexity tier (1=simple, 3=complex)", ["1", "2", "3"])
            if r_comp: temp_profile["ROUTING_COMPLEXITY"] = r_comp

        # 🧩 Per-model parameters declared by the binding in description.yaml.
        # For local spawner bindings (llama_cpp_server, ...) these distinguish
        # one model profile from another: n_gpu_layers, context_size, n_batch,
        # flash_attn, chat_format, etc. Skip if the binding does not declare any.
        _configure_model_profile_parameters(b_type, b_name, selected_b_alias, temp_profile)

    for k, v in temp_profile.items():
        config_map[profile_prefix + k] = v

    _enforce_single_default_profile(b_type, config_map)
    ASCIIColors.green(f"\n  ✓ Saved profile: {alias}")

def _enforce_single_default_profile(b_type: str, config_map: Dict[str, str]) -> None:
    """
    Enforces the SINGLE-DEFAULT INVARIANT for profiles of a modality.
    Ensures exactly one profile carries IS_DEFAULT=true; clears others,
    and promotes the first if none/multiple exist.
    Mutates config_map in place and keeps YAML consistent.
    """
    profile_prefix = f"{b_type.upper()}_PROFILES_"
    default_suffix = "_IS_DEFAULT"

    default_keys = [k for k in config_map if k.startswith(profile_prefix) and k.endswith(default_suffix)]
    if not default_keys:
        return

    sorted_keys = sorted(default_keys)
    chosen_profile_key = None

    # First, honor any explicitly defaulted profiles
    for k in sorted_keys:
        if config_map.get(k) and config_map[k].strip().lower() == "true":
            chosen_profile_key = k
            break

    # Fallback to first if none are default
    if not chosen_profile_key:
        chosen_profile_key = sorted_keys[0]

    chosen_alias = chosen_profile_key.replace(profile_prefix, "").removesuffix(default_suffix)

    # Rewrite the map cleanly: only keep the chosen default as true
    new_map = dict(config_map)
    for k in default_keys:
        new_map[k] = "false"
    new_map[chosen_profile_key] = "true"

    # Promote alias if needed for consistency (alias keys like PROFILES_ALIAS)
    alias_key = f"{b_type.upper()}_PROFILES_{chosen_alias}_BINDING_ALIAS"
    # Ensure BINDING_NAME is normalized before alias assignment
    if alias_key not in new_map:
        binding_alias_fallback = f"{b_type.upper()}_BINDINGS_{chosen_alias}"
        new_map[alias_key] = new_map.get(binding_alias_fallback, "")

    config_map.clear()
    config_map.update(new_map)


def _set_default_profile_action(b_type: str, alias: str, config_map: Dict[str, str]):
    profile_prefix = f"{b_type.upper()}_PROFILES_"
    for a in _get_configured_aliases(b_type, config_map, "PROFILES"):
        target_key = f"{profile_prefix}{a.upper()}_IS_DEFAULT"
        found_key = None
        for k in list(config_map.keys()):
            if k.upper() == target_key:
                found_key = k
                break
        val = "true" if a.upper() == alias.upper() else "false"
        if found_key:
            config_map[found_key] = val
        else:
            config_map[target_key] = val
    ASCIIColors.green(f"\n  ✓ Profile '{alias}' set as default for {b_type.upper()}.")


def _add_profile_flow(b_type: str, config_map: Dict[str, str]):
    existing = _get_configured_aliases(b_type, config_map, "PROFILES")
    raw_alias = _safe_input_or_back("Enter alias for the profile", "master")
    if raw_alias is None:
        ASCIIColors.yellow("\n  ⚠️ Profile creation cancelled.")
        return
    if not raw_alias.strip():
        alias = _generate_unique_alias("master", existing)
        ASCIIColors.info(f"  ℹ️ Auto-named profile: {alias}")
    else:
        alias = _generate_unique_alias(raw_alias, existing)
        if alias != _sanitize_alias(raw_alias):
            ASCIIColors.info(f"  ℹ️ Name collision resolved: {alias}")
    if alias: _configure_profile_instance(b_type, alias, config_map)


def _profiles_menu(b_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{b_type.upper()} Profiles", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Add a new profile, edit, set as default, or delete an existing one.")
        menu.add_choice("➕ Add new profile", value=lambda: _add_profile_flow(b_type, config_map))

        existing_aliases = _get_configured_aliases(b_type, config_map, "PROFILES")
        for alias in existing_aliases:
            is_def = config_map.get(f"{b_type.upper()}_PROFILES_{alias.upper()}_IS_DEFAULT", "").lower() in ("true", "1", "yes", "y", "on")
            def_badge = " ⭐ [DEFAULT]" if is_def else ""
            menu.add_choice(f"{alias}{def_badge}", value=lambda a=alias: _edit_keys_menu(b_type, "PROFILES", a, config_map))

        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()


# ─────────────────────────────────────────────────────────────────────────────
# Automatic model-profile generation from a binding
# ─────────────────────────────────────────────────────────────────────────────

# Name-based capability inference table.
# Ordering matters: the first match wins. Keys are lowercase substrings of
# the model name (the canonical identifier returned by list_models()).
_MODEL_FAMILY_HINTS: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = [
    # Vision-capable multimodal families (LLM-agnostic identifiers)
    (("qwen2.5-vl", "qwen2-vl", "qwen-vl", "qwen3-vl"),
     {"vision_enabled": True, "video_enabled": True, "image_token_cost": 512}),
    (("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5", "o4"),
     {"vision_enabled": True, "image_token_cost": 512}),
    (("claude-3", "claude-4", "claude-sonnet", "claude-opus", "claude-haiku"),
     {"vision_enabled": True, "image_token_cost": 1024}),
    (("gemini-1.5", "gemini-2", "gemini-pro-vision", "gemini-flash", "gemini-exp"),
     {"vision_enabled": True, "video_enabled": True, "image_token_cost": 258}),
    (("llava", "llava-next", "bakllava", "llama3.2-vision", "llama-3.2-11b-vision"),
     {"vision_enabled": True, "image_token_cost": 576}),
    (("pixtral", "idefics", "moondream", "llama-3-vision", "llama3-vision"),
     {"vision_enabled": True, "image_token_cost": 1024}),
    (("phi-3-vision", "phi-3.5-vision", "phi-4-vision", "phi4-multimodal"),
     {"vision_enabled": True, "image_token_cost": 144}),
    (("internvl", "minicpm-v", "minicpm-o", "cogvlm", "glm-4v", "glm-4.1v"),
     {"vision_enabled": True, "image_token_cost": 256}),
    (("gemma-3", "gemma3"), {"vision_enabled": True, "image_token_cost": 256}),
    (("smolvlm",), {"vision_enabled": True, "image_token_cost": 64}),

    # Reasoning / hybrid-reasoning families
    (("deepseek-r1", "deepseek-reasoner", "qwq", "qwen3-thinking"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium", "high"],
      "default_reasoning_effort": "medium"}),
    (("o1", "o3", "o4-mini", "gpt-5-thinking"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium", "high"],
      "default_reasoning_effort": "medium"}),
    (("magistral", "phi-4-reasoning", "phi-4-reason"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium", "high"],
      "default_reasoning_effort": "medium"}),
    (("qwen3", "qwen-3"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium", "high"]}),
    (("gemma-3n", "gemma3n"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium"]}),
    (("claude-3.7", "claude-sonnet-4-thinking"),
     {"thinking_enabled": True, "supported_reasoning_efforts": ["low", "medium", "high"]}),
]

# Best-effort context-size guesser (in tokens). Ordered — first match wins.
# Values chosen to reflect the *base* advertised context of each family;
# users can override with forced_context_size later.
_MODEL_CTX_HINTS: List[Tuple[str, int]] = [
    ("gpt-4o", 128_000), ("gpt-4.1", 1_000_000), ("gpt-5", 400_000),
    ("gpt-4-turbo", 128_000), ("gpt-4", 8_192), ("gpt-3.5", 16_385),
    ("o1", 200_000), ("o3", 200_000), ("o4", 200_000),
    ("claude-3", 200_000), ("claude-4", 200_000), ("claude-sonnet", 200_000),
    ("claude-opus", 200_000), ("claude-haiku", 200_000),
    ("gemini-1.5", 1_000_000), ("gemini-2", 1_000_000),
    ("qwen2.5-vl", 128_000), ("qwen2-vl", 32_768), ("qwen3", 131_072),
    ("qwen2.5", 32_768), ("qwen2", 32_768),
    ("llama-3.3", 128_000), ("llama-3.2", 128_000), ("llama-3.1", 128_000),
    ("llama-3", 8_192), ("llama-2", 4_096),
    ("mistral-nemo", 128_000), ("mistral-large", 128_000),
    ("mistral-small", 32_768), ("mixtral", 32_768), ("mistral", 8_192),
    ("deepseek-r1", 128_000), ("deepseek-v2", 128_000), ("deepseek", 64_000),
    ("gemma-3", 128_000), ("gemma3", 128_000), ("gemma-2", 8_192), ("gemma", 8_192),
    ("phi-4", 16_384), ("phi-3", 128_000), ("phi3", 128_000), ("phi-2", 2_048),
    ("command-r", 128_000), ("command", 4_096),
    ("dbrx", 32_768), ("solar", 32_768),
    ("tinyllama", 2_048),
]


def _load_ctx_size_map() -> Dict[str, int]:
    """Loads the optional assets/models_ctx_sizes.json registry (best-effort)."""
    try:
        assets_path = Path(__file__).parent / "assets" / "models_ctx_sizes.json"
        if not assets_path.is_file() or not json:
            return {}
        with open(assets_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        result: Dict[str, int] = {}
        for k, v in raw.items():
            try:
                result[str(k).lower()] = int(v)
            except (TypeError, ValueError):
                continue
        return result
    except Exception:
        return {}


def _binding_capability_hints(b_type: str, binding_name: str) -> List[Dict[str, Any]]:
    """Loads optional ``model_capability_hints`` from description.yaml.

    Shape honored:
        model_capability_hints:
          - match: ["qwen2.5", "qwen3"]
            vision_enabled: true
            image_token_cost: 512
    """
    desc = _get_binding_description(binding_name, b_type) or {}
    hints = desc.get("model_capability_hints")
    return hints if isinstance(hints, list) else []


# ── Tool-calling inference table (regex patterns over lowercased model names) ─
#
# Many names encode tool-calling capability implicitly through their family.
# The table is deliberately explicit (not just "any instruct") so we don't
# promise tool use where it isn't supported.
_TOOL_CALLING_HINTS: List[Tuple[Tuple[str, ...], bool]] = [
    # Excluded explicitly: reasoning-only families that historically lack tool calls.
    (("deepseek-r1", "deepseek-reasoner", "qwq", "o1-preview", "o1-mini"),  False),
    # Full-function-calling flagship families.
    (("qwen2.5", "qwen2_5", "qwen2-vl", "qwen3", "qwen-3", "qwq-32b-preview"), True),
    (("llama-3.1", "llama-3.2", "llama3.1", "llama3.2", "llama-3.3", "llama3.3"), True),
    (("llama-4", "llama4"), True),
    (("mistral-small", "mistral-large", "ministral", "devstral",
      "magistral", "mistral-nemo", "mistral-medium"), True),
    (("mixtral",), True),
    (("deepseek-v2", "deepseek-v3", "deepseek-chat"), True),
    (("command-r", "command-r-plus"), True),
    (("phi-4",), True),
    (("glm-4", "glm4"), True),
    (("granite-3", "granite3"), True),
    (("hermes-2", "hermes-3", "nous-hermes"), True),
    (("toolace", "toolbench", "-tools-", "-tool-"), True),
    (("gpt-4o", "gpt-4.1", "gpt-5", "claude-3", "claude-4",
      "claude-sonnet", "claude-opus", "claude-haiku",
      "gemini-1.5", "gemini-2"), True),
    (("smollm2", "smollm-2"), True),
    (("internlm2", "internlm-2", "yi-1.5", "yi-34b", "amber"), True),
]


def _infer_tool_calling(model_name: str) -> bool:
    """Returns True when the model name strongly suggests native tool calling."""
    name_lc = str(model_name or "").lower()
    for patterns, supported in _TOOL_CALLING_HINTS:
        if any(p in name_lc for p in patterns):
            return supported
    return False


def _infer_model_capabilities(
    model_name: str,
    b_type: str = "llm",
    binding_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Infers profile fields for a model name (regex-driven, overridable).

    Inference priority (highest first):
      1. ``model_capability_hints`` declared in the binding's description.yaml.
      2. Built-in family tables (``_MODEL_FAMILY_HINTS``, ``_TOOL_CALLING_HINTS``).
      3. Conservative defaults (all capability flags False, cost=256).

    Always returns a dict with keys:
      vision_enabled, video_enabled, thinking_enabled, tool_calling_enabled,
      image_token_cost, supported_reasoning_efforts,
      default_reasoning_effort, forced_context_size
      (``forced_context_size`` is ``None`` when unknown).
    """
    name_lc = str(model_name or "").lower()
    caps: Dict[str, Any] = {
        "vision_enabled": False,
        "video_enabled": False,
        "thinking_enabled": False,
        "tool_calling_enabled": False,
        "image_token_cost": 256,
        "supported_reasoning_efforts": ["low", "medium", "high"],
        "default_reasoning_effort": None,
        "forced_context_size": None,
    }

    # 1. Description.yaml hints (highest priority — authors know their binding).
    if binding_name:
        for hint in _binding_capability_hints(b_type, binding_name):
            if not isinstance(hint, dict):
                continue
            patterns = hint.get("match") or hint.get("matches") or hint.get("contains") or []
            if isinstance(patterns, str):
                patterns = [patterns]
            if not any(str(p).lower() in name_lc for p in patterns if p):
                continue
            for key in (
                "vision_enabled", "video_enabled", "thinking_enabled",
                "tool_calling_enabled", "image_token_cost",
                "supported_reasoning_efforts",
                "default_reasoning_effort", "forced_context_size",
            ):
                if key in hint and hint[key] is not None:
                    caps[key] = hint[key]
            break

    # 2. Built-in family table.
    for patterns, inferred in _MODEL_FAMILY_HINTS:
        if any(p in name_lc for p in patterns):
            for key, value in inferred.items():
                # Do not overwrite a description.yaml-provided value.
                if caps.get(key) in (None, False, 256, ["low", "medium", "high"]):
                    caps[key] = value
            break

    # 2b. Tool-calling inference (independent of the vision/thinking tables).
    if not caps.get("tool_calling_enabled"):
        caps["tool_calling_enabled"] = _infer_tool_calling(model_name)

    # 3. Context size (only when not already provided).
    if caps.get("forced_context_size") is None:
        ctx_map = _load_ctx_size_map()
        for known_name, size in ctx_map.items():
            if known_name in name_lc:
                caps["forced_context_size"] = size
                break
        if caps["forced_context_size"] is None:
            for marker, size in _MODEL_CTX_HINTS:
                if marker in name_lc:
                    caps["forced_context_size"] = size
                    break

    return caps


def _profile_alias_taken(b_type: str, alias: str, config_map: Dict[str, str]) -> bool:
    """True if a profile with this alias already exists for the modality."""
    prefix = f"{b_type.upper()}_PROFILES_{alias.upper()}_"
    return any(k.upper().startswith(prefix) for k in config_map)


def _unique_profile_alias(b_type: str, base: str, config_map: Dict[str, str]) -> str:
    """Returns a unique, sanitized profile alias (adds _2, _3, … on collision)."""
    sanitized = _sanitize_alias(base) or "profile"
    if not _profile_alias_taken(b_type, sanitized, config_map):
        return sanitized
    counter = 2
    while _profile_alias_taken(b_type, f"{sanitized}_{counter}", config_map):
        counter += 1
    return f"{sanitized}_{counter}"


def _autogenerate_profiles_for_binding(
    b_type: str,
    binding_alias: str,
    config_map: Dict[str, str],
) -> None:
    """Auto-creates one model profile per model exposed by a binding instance.

    Behaviour:
      * Skips profiles that already reference the same binding + model.
      * Populates each new profile with capability hints inferred from the
        model name (vision / video / thinking / image token cost / context).
      * Preserves an existing default profile; otherwise promotes the first
        generated profile so the modality keeps a default.
      * Idempotent: re-running after adding new models only adds the new ones.
    """
    MAX_PROFILES = 50

    binding_prefix = f"{b_type.upper()}_BINDINGS_{binding_alias.upper()}_"
    binding_name: Optional[str] = None
    for k, v in config_map.items():
        if k.upper() == f"{binding_prefix}BINDING_NAME":
            binding_name = v
            break
    if not binding_name:
        ASCIIColors.red(f"\n  ❌ Binding '{binding_alias}' has no BINDING_NAME.")
        return

    ASCIIColors.info(f"\n  🔎 Scanning models exposed by '{binding_name}' ({binding_alias})...")
    try:
        models = _fetch_available_models(b_type, binding_name, config_map, binding_alias) or []
    except Exception as exc:
        ASCIIColors.red(f"\n  ❌ Failed to list models: {exc}")
        return

    if not models:
        ASCIIColors.yellow("\n  ⚠️ The binding reported no models. Nothing to generate.")
        return

    models = sorted({m for m in models if isinstance(m, str) and m.strip()})
    if len(models) > MAX_PROFILES:
        ASCIIColors.warning(
            f"\n  ⚠️ {len(models)} models found; only the first {MAX_PROFILES} will be processed."
        )
        models = models[:MAX_PROFILES]

    # Index existing profiles by (binding_alias, model_name).
    existing_pairs: set = set()
    existing_default_alias: Optional[str] = None
    profile_prefix = f"{b_type.upper()}_PROFILES_"
    for k, v in config_map.items():
        ku = k.upper()
        if not ku.startswith(profile_prefix):
            continue
        if ku.endswith("_IS_DEFAULT") and str(v).strip().lower() in ("true", "1", "yes", "y", "on"):
            existing_default_alias = ku[len(profile_prefix):-len("_IS_DEFAULT")].lower()
        # We consolidate below using a single pass over per-alias keys.
    for palias in _get_configured_aliases(b_type, config_map, "PROFILES"):
        p_prefix = f"{profile_prefix}{palias.upper()}_"
        p_binding_alias = None
        p_model = None
        for k, v in config_map.items():
            ku = k.upper()
            if not ku.startswith(p_prefix):
                continue
            if ku.endswith("_BINDING_ALIAS"):
                p_binding_alias = _sanitize_alias(v)
            elif ku.endswith("_MODEL_NAME"):
                p_model = v
        if p_binding_alias and p_model:
            existing_pairs.add((p_binding_alias, p_model))

    to_create = [m for m in models if (_sanitize_alias(binding_alias), m) not in existing_pairs]
    if not to_create:
        ASCIIColors.info(
            f"\n  ℹ️ All {len(models)} model(s) already have profiles for binding '{binding_alias}'. Nothing to add."
        )
        return

    ASCIIColors.rich_print("\n[bold cyan]── Auto-Generate Model Profiles ──[/bold cyan]")
    ASCIIColors.rich_print(
        f"[dim]Binding: {binding_name} (alias '{binding_alias}')\n"
        f"Models found: {len(models)}  |  New profiles to create: {len(to_create)}[/dim]"
    )
    preview = ", ".join(to_create[:8]) + (" ..." if len(to_create) > 8 else "")
    ASCIIColors.rich_print(f"[dim]Will create profiles for: {preview}[/dim]")
    if not _safe_confirm(
        f"Create {len(to_create)} profile(s) with auto-inferred capabilities?",
        default=True,
    ):
        ASCIIColors.info("\n  ℹ️ Auto-generation cancelled. No changes were made.")
        return

    created_aliases: List[str] = []
    promote_first = existing_default_alias is None

    for model_name in to_create:
        caps = _infer_model_capabilities(model_name, b_type, binding_name)
        base = f"{binding_alias}_{model_name}"
        alias = _unique_profile_alias(b_type, base, config_map)
        prefix = f"{b_type.upper()}_PROFILES_{alias.upper()}_"

        config_map[prefix + "BINDING_ALIAS"] = _sanitize_alias(binding_alias)
        config_map[prefix + "MODEL_NAME"] = model_name
        config_map[prefix + "VISION_ENABLED"] = "true" if caps["vision_enabled"] else "false"
        config_map[prefix + "VIDEO_ENABLED"] = "true" if caps["video_enabled"] else "false"
        config_map[prefix + "THINKING_ENABLED"] = "true" if caps["thinking_enabled"] else "false"
        config_map[prefix + "TOOL_CALLING_ENABLED"] = "true" if caps["tool_calling_enabled"] else "false"
        config_map[prefix + "IMAGE_TOKEN_COST"] = str(int(caps["image_token_cost"]))
        efforts = caps.get("supported_reasoning_efforts") or ["low", "medium", "high"]
        config_map[prefix + "SUPPORTED_REASONING_EFFORTS"] = ", ".join(efforts)
        if caps.get("default_reasoning_effort"):
            config_map[prefix + "DEFAULT_REASONING_EFFORT"] = str(caps["default_reasoning_effort"]).lower()
        if caps.get("forced_context_size"):
            config_map[prefix + "FORCED_CONTEXT_SIZE"] = str(int(caps["forced_context_size"]))
        config_map[prefix + "IS_DEFAULT"] = "false"

        created_aliases.append(alias)

    if promote_first and created_aliases:
        # Promote only the first newly created profile; otherwise leave the
        # existing default untouched.
        config_map[f"{b_type.upper()}_PROFILES_{created_aliases[0].upper()}_IS_DEFAULT"] = "true"

    _enforce_single_default_profile(b_type, config_map)

    # 🧩 Backfill per-model parameters declared by the binding into every
    # newly created profile using the schema defaults. `ctx_size` in
    # particular is overridden by the *inferred* context of each model, so
    # spawner bindings (llama_cpp_server, ...) boot with a sensible per-model
    # context already in place. Everything else falls back to the declared
    # default so generated profiles are immediately runnable.
    schema = _get_model_profile_schema(b_type, binding_name)
    if schema:
        declared_by_name = {d["name"].lower(): d for d in schema}
        for model_name, alias in zip(to_create, created_aliases):
            p_prefix = f"{b_type.upper()}_PROFILES_{alias.upper()}_"
            inferred = _infer_model_capabilities(model_name, b_type, binding_name)
            for declared in schema:
                key_upper = declared["name"].upper()
                if (p_prefix + key_upper) in config_map:
                    continue
                value: Optional[Any] = None
                # Per-profile ctx_size: prefer the inferred context size when known.
                if declared["name"].lower() == "ctx_size" and inferred.get("forced_context_size"):
                    value = int(inferred["forced_context_size"])
                else:
                    value = declared.get("default")
                if value is None:
                    continue
                config_map[p_prefix + key_upper] = _format_env_value(value)
    # Avoid unused-variable lint when no schema is declared.
    _ = declared_by_name if schema else None

    ASCIIColors.green(
        f"\n  ✓ Created {len(created_aliases)} profile(s) for binding '{binding_alias}'."
        + (f" Default: {created_aliases[0]}" if promote_first and created_aliases else "")
    )
    for a in created_aliases[:5]:
        ASCIIColors.info(f"     • {a}")
    if len(created_aliases) > 5:
        ASCIIColors.info(f"     • … and {len(created_aliases) - 5} more")


def _reapply_inference_to_existing_profiles(
    b_type: str,
    binding_alias: str,
    config_map: Dict[str, str],
) -> None:
    """Re-runs capability inference on every profile of a binding instance and
    updates only the fields a user hasn't diverged from.

    "Diverged" means the current value no longer matches the previously
    inferred default for the model. In that case the value is preserved so a
    user who manually toggled tool-calling or bumped context size keeps their
    choice.

    Only fields formalised by inference are touched: ``VISION_ENABLED``,
    ``VIDEO_ENABLED``, ``THINKING_ENABLED``, ``TOOL_CALLING_ENABLED``,
    ``IMAGE_TOKEN_COST``, ``FORCED_CONTEXT_SIZE``, plus the binding-declared
    ``ctx_size`` (when the binding exposes it as a per-model parameter).
    """
    binding_prefix = f"{b_type.upper()}_BINDINGS_{binding_alias.upper()}_"
    binding_name: Optional[str] = None
    for k, v in config_map.items():
        if k.upper() == f"{binding_prefix}BINDING_NAME":
            binding_name = v
            break
    if not binding_name:
        ASCIIColors.red(f"\n  ❌ Binding '{binding_alias}' has no BINDING_NAME.")
        return

    schema = _get_model_profile_schema(b_type, binding_name)
    schema_names = {d["name"].lower() for d in schema}

    # Index profiles that belong to this binding instance, by (alias, model).
    targets: List[Tuple[str, str]] = []
    profile_prefix = f"{b_type.upper()}_PROFILES_"
    for p_alias in _get_configured_aliases(b_type, config_map, "PROFILES"):
        p_prefix = f"{profile_prefix}{p_alias.upper()}_"
        p_binding_alias = None
        p_model = None
        for k, v in config_map.items():
            ku = k.upper()
            if not ku.startswith(p_prefix):
                continue
            if ku.endswith("_BINDING_ALIAS"):
                p_binding_alias = _sanitize_alias(v)
            elif ku.endswith("_MODEL_NAME"):
                p_model = v
        if p_binding_alias == _sanitize_alias(binding_alias) and p_model:
            targets.append((p_alias, p_model))

    if not targets:
        ASCIIColors.info(
            f"\n  ℹ️ No profiles reference binding '{binding_alias}'. Nothing to re-infer."
        )
        return

    ASCIIColors.rich_print("\n[bold cyan]── Re-Scan & Apply Inference ──[/bold cyan]")
    ASCIIColors.rich_print(
        f"[dim]Binding: {binding_name} (alias '{binding_alias}')\n"
        f"Profiles to re-evaluate: {len(targets)}[/dim]"
    )
    if not _safe_confirm(
        f"Re-apply inference rules to {len(targets)} profile(s)? "
        f"Manual overrides will be preserved.",
        default=True,
    ):
        ASCIIColors.info("\n  ℹ️ Re-inference cancelled. No changes were made.")
        return

    updated_count = 0
    changed_fields = 0
    for p_alias, model_name in targets:
        caps = _infer_model_capabilities(model_name, b_type, binding_name)
        p_prefix = f"{profile_prefix}{p_alias.upper()}_"
        was_updated = False

        def _maybe_update(key: str, new_value: str, prior_default: Optional[str]) -> None:
            """Set only if empty OR current value still matches prior default."""
            nonlocal was_updated, changed_fields
            full = p_prefix + key
            current = config_map.get(full)
            if current is None or str(current).strip() == "":
                if new_value == "":
                    return
            elif prior_default is not None and str(current).strip().lower() != str(prior_default).strip().lower():
                return  # user diverged → preserve
            if str(current) != str(new_value):
                config_map[full] = str(new_value)
                was_updated = True
                changed_fields += 1

        # Vision / video / thinking / tool_calling
        _maybe_update("VISION_ENABLED", "true" if caps["vision_enabled"] else "false", "false")
        _maybe_update("VIDEO_ENABLED", "true" if caps["video_enabled"] else "false", "false")
        _maybe_update("THINKING_ENABLED", "true" if caps["thinking_enabled"] else "false", "false")
        _maybe_update("TOOL_CALLING_ENABLED", "true" if caps["tool_calling_enabled"] else "false", "false")
        _maybe_update("IMAGE_TOKEN_COST", str(int(caps["image_token_cost"])), "256")
        if caps.get("forced_context_size"):
            _maybe_update(
                "FORCED_CONTEXT_SIZE",
                str(int(caps["forced_context_size"])),
                None,
            )
        if "ctx_size" in schema_names and caps.get("forced_context_size"):
            _maybe_update(
                "CTX_SIZE",
                str(int(caps["forced_context_size"])),
                None,
            )

        if was_updated:
            updated_count += 1

    if updated_count:
        ASCIIColors.green(
            f"\n  ✓ Updated {updated_count} profile(s) ({changed_fields} field change(s))."
        )
    else:
        ASCIIColors.info(
            "\n  ℹ️ No changes needed — all profiles already match the inferred defaults."
        )

def _edit_data_folder(config_map: Dict[str, str]) -> None:
    """Prompt for the client-wide data folder and persist it in the map.

    Stored as the top-level env key ``LOLLMS_DATA_FOLDER`` which the serializer
    writes back as ``lollms_data_folder`` in the YAML root. Empty input clears
    the override so the shared default (``~/.lollms/data``, or
    ``$LOLLMS_DATA_FOLDER`` when set) is used.
    """
    try:
        from lollms_client.lollms_paths import get_data_folder
        default_folder = str(get_data_folder())
    except Exception:
        default_folder = str((Path.home() / ".lollms" / "data").resolve())

    current = config_map.get("LOLLMS_DATA_FOLDER", "")
    ASCIIColors.rich_print("\n[bold cyan]── Client Data Folder ──[/bold cyan]")
    ASCIIColors.rich_print(
        "[dim]Root folder forwarded to every binding for models, caches, venvs, DBs.\n"
        "Leave blank to use the shared default (~/.lollms/data, or $LOLLMS_DATA_FOLDER).[/dim]"
    )
    new_val = _safe_input_or_back("Enter data folder path", current or default_folder)
    if new_val is None:
        return

    cleaned = new_val.strip()
    if not cleaned:
        config_map.pop("LOLLMS_DATA_FOLDER", None)
        ASCIIColors.green("\n  ✓ Cleared override. Shared default data folder will be used.")
        return

    try:
        resolved = Path(cleaned).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        ASCIIColors.red(f"\n  ❌ Cannot use folder '{cleaned}': {exc}")
        return

    config_map["LOLLMS_DATA_FOLDER"] = str(resolved)
    ASCIIColors.green(f"\n  ✓ Data folder set to: {resolved}")


def _import_config_from_file(config_map: Dict[str, str]) -> Optional[str]:
    """Imports an existing YAML configuration and merges it into the live map.

    Use case: users migrating from another machine, a colleague's shared
    config, an exported wizard file, or an app-specific config. The imported
    flat env-keys are merged into the in-memory ``config_map`` — nothing is
    written to disk here; the standard Save/Save & Exit flow handles that.

    Returns:
        ``_BACK_VALUE`` when the merge succeeded, so the caller closes the
        current menu and re-renders it against the updated ``config_map``
        (the menu's entries are computed at build time — an in-place mutation
        without a rebuild would leave the UI stale). ``None`` on cancel/error
        so the menu keeps showing and the user can retry.
    """
    if not yaml:
        ASCIIColors.red("\n  ❌ PyYAML is required to import configuration files.")
        return None

    ASCIIColors.rich_print("\n[bold cyan]── Import Configuration File ──[/bold cyan]")
    ASCIIColors.rich_print(
        "[dim]Load a YAML configuration and merge it into the current session.\n"
        "Nothing is written to disk until you Save.[/dim]"
    )
    raw_path = _safe_input_or_back("Path to YAML file to import", "")
    if raw_path is None:
        return None
    cleaned = raw_path.strip()
    if not cleaned:
        ASCIIColors.yellow("\n  ⚠️ No path provided. Import cancelled.")
        return None

    try:
        src = Path(cleaned).expanduser().resolve()
    except (OSError, ValueError) as exc:
        ASCIIColors.red(f"\n  ❌ Invalid path '{cleaned}': {exc}")
        return None

    if not src.is_file():
        ASCIIColors.red(f"\n  ❌ File not found: {src}")
        return None
    if src.suffix.lower() not in (".yaml", ".yml"):
        ASCIIColors.red(f"\n  ❌ Only .yaml/.yml files can be imported (got '{src.suffix}').")
        return None

    try:
        data = load_yaml_file(src)
    except Exception as exc:
        ASCIIColors.red(f"\n  ❌ Failed to parse '{src}': {exc}")
        return None
    if not isinstance(data, dict):
        ASCIIColors.red(f"\n  ❌ Expected a YAML mapping at the top level, got {type(data).__name__}.")
        return None

    try:
        flat = _flatten_dict_to_env(data)
    except Exception as exc:
        ASCIIColors.red(f"\n  ❌ Failed to flatten configuration: {exc}")
        return None

    merged = 0
    for key, value in flat.items():
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue
        config_map[key] = value
        merged += 1

    ASCIIColors.green(
        f"\n  ✓ Imported {merged} entries from {src.name}. "
        f"Active config now has {len(config_map)} keys total."
    )
    # Signal to the outer wizard loop that the live map changed and the menu
    # must be rebuilt so imported bindings/profiles/defaults become visible.
    return _BACK_VALUE


def _change_config_file_path() -> Optional[str]:
    """Changes the active configuration file path globally via env var.

    Persists ``LOLLMS_CONFIG_FILE`` at user scope using
    ``lollms_paths.set_env_var_permanently`` so every future Python session
    inherits the new path without editing the app. The current session also
    picks it up immediately via ``os.environ``. Offers a reset-to-default
    option that removes the env var.

    Returns:
        ``_BACK_VALUE`` when the env var was successfully changed (or cleared),
        so the caller closes the current menu and re-renders it — the header
        banner and any subsequent "save to" labels pick up the new path.
        ``None`` on cancel/error.
    """
    try:
        from lollms_client.lollms_paths import (
            ENV_CONFIG_FILE,
            get_config_file,
            set_env_var_permanently,
            unset_env_var_permanently,
        )
    except Exception as exc:
        ASCIIColors.red(f"\n  ❌ Cannot access path policy: {exc}")
        return None

    current = str(get_config_file())
    ASCIIColors.rich_print("\n[bold cyan]── Change Config File Path ──[/bold cyan]")
    ASCIIColors.rich_print(
        f"[dim]Current active config: {current}[/dim]\n"
        "[dim]Persists the location as a user-scope env var (LOLLMS_CONFIG_FILE) so all\n"
        "future sessions use it. Leave blank for default (~/.lollms/config.yaml).[/dim]"
    )
    new_val = _safe_input_or_back("Enter new config file path (blank = reset to default)", current)
    if new_val is None:
        return None

    cleaned = new_val.strip()
    if not cleaned:
        result = unset_env_var_permanently(ENV_CONFIG_FILE)
        if result.get("ok"):
            ASCIIColors.green(f"\n  ✓ Cleared LOLLMS_CONFIG_FILE override.")
            ASCIIColors.info(f"  {result.get('message', '')}")
            return _BACK_VALUE
        ASCIIColors.red(f"\n  ❌ {result.get('message', 'Failed to unset env var.')}")
        return None

    try:
        resolved = Path(cleaned).expanduser().resolve()
    except (OSError, ValueError) as exc:
        ASCIIColors.red(f"\n  ❌ Invalid path '{cleaned}': {exc}")
        return None

    if resolved.suffix.lower() not in (".yaml", ".yml"):
        ASCIIColors.red(
            f"\n  ❌ Configuration file must use .yaml or .yml (got '{resolved.suffix}')."
        )
        return None

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        ASCIIColors.red(f"\n  ❌ Cannot create parent folder for '{resolved}': {exc}")
        return None

    result = set_env_var_permanently(ENV_CONFIG_FILE, str(resolved))
    if result.get("ok"):
        ASCIIColors.green(f"\n  ✓ Config file path set to: {resolved}")
        persisted_to = result.get("persisted_to") or []
        if persisted_to:
            ASCIIColors.info(f"  Persisted at user scope in: {', '.join(persisted_to)}")
        ASCIIColors.info(
            "  Note: already-open terminals/editors may need a restart to see the new value."
        )
        return _BACK_VALUE
    ASCIIColors.yellow(f"\n  ⚠️ {result.get('message', 'Persistence failed.')}")
    ASCIIColors.info("  The current session still uses the new path via os.environ.")
    # Even on persistence failure the session-level change took effect via
    # os.environ; rebuild so the banner reflects the new active path.
    return _BACK_VALUE


def _modality_menu(b_type: str, config_map: Dict[str, str]):
    if b_type == "connection":
        _connection_modality_menu(config_map)
        return

    while True:
        menu = Menu(f"{b_type.upper()} Configuration", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro(f"Configure {b_type.upper()} Bindings and Profiles.")
        menu.add_choice(f"🔌 Configure {b_type.upper()} Bindings", value=lambda: _bindings_menu(b_type, config_map))
        menu.add_choice(f"📋 Configure {b_type.upper()} Profiles", value=lambda: _profiles_menu(b_type, config_map))
        menu.add_choice("↩ Back", value=_BACK_VALUE)
        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()


def _connection_modality_menu(config_map: Dict[str, str]):
    """Specialized modality menu for CONNECTION bindings (uses instance_name, not model_name)."""
    while True:
        menu = Menu("CONNECTION Configuration", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro(
            "Configure Connection Bindings (Discord, Telegram, Slack, Webhook, etc.) "
            "and Instance Profiles (which channel to send to)."
        )
        menu.add_choice("🔗 Add Connection Binding", value=lambda: _add_connection_binding_flow(config_map))
        menu.add_choice("📱 Add Instance Profile (Channel)", value=lambda: _add_connection_profile_flow(config_map))
        menu.add_choice("✏️ Edit Existing", value=lambda: _connection_edit_menu(config_map))
        menu.add_choice("↩ Back", value=_BACK_VALUE)

        existing = _get_configured_aliases("connection", config_map, "BINDINGS")
        existing_profiles = _get_configured_aliases("connection", config_map, "PROFILES")
        if existing:
            ASCIIColors.info(f"\n  Configured bindings: {', '.join(existing)}")
        if existing_profiles:
            ASCIIColors.info(f"  Configured profiles: {', '.join(existing_profiles)}")

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()


def _add_connection_binding_flow(config_map: Dict[str, str]):
    """Add a new connection binding (e.g., generic_webhook) with an alias."""
    conn_bindings = _list_connection_bindings()
    if not conn_bindings:
        ASCIIColors.warning("No connection bindings available. Check connection_bindings/ directory.")
        return

    selected = _safe_select("Select a connection binding:", conn_bindings)
    if not selected:
        return

    existing = _get_configured_aliases("connection", config_map, "BINDINGS")
    raw_alias = _safe_input_or_back("Enter an alias for this connection (e.g., 'slack-main', 'discord-ops')", "main")
    if raw_alias is None:
        ASCIIColors.yellow("\n  ⚠️ Connection binding creation cancelled.")
        return
    if not raw_alias.strip():
        alias = _generate_unique_alias("main", existing)
        ASCIIColors.info(f"  ℹ️ Auto-named binding: {alias}")
    else:
        alias = _generate_unique_alias(raw_alias, existing)
        if alias != _sanitize_alias(raw_alias):
            ASCIIColors.info(f"  ℹ️ Name collision resolved: {alias}")
    if not alias:
        return

    _configure_connection_binding_instance(selected, alias, config_map)


def _configure_connection_binding_instance(b_name: str, alias: str, config_map: Dict[str, str]):
    """Configure a specific connection binding's global parameters."""
    prefix = f"CONNECTION_BINDINGS_{alias}_"
    config_map[prefix + "BINDING_NAME"] = b_name
    ASCIIColors.green(f"\n  ✓ Selected CONNECTION binding: {b_name} (Alias: {alias})")

    # Try to load description.yaml for the binding
    desc = _get_connection_binding_description(b_name)
    if desc:
        params = desc.get("global_input_parameters", [])
        for p in params:
            pname = p.get("name", "")
            if not pname:
                continue
            val = _prompt_param(pname, p.get("description", ""), p.get("type", "str"), p.get("mandatory", False), p.get("default"))
            if val is _BACK_SENTINEL:
                ASCIIColors.yellow("\n  ⚠️ Connection binding configuration cancelled. Rolling back partial keys.")
                for k in [k for k in config_map if k.startswith(prefix)]:
                    del config_map[k]
                return
            config_map[prefix + pname.upper()] = _format_env_value(val)
    else:
        # Fallback: prompt for standard connection params
        ASCIIColors.yellow("\n  No description.yaml found. Using standard connection parameters.\n")
        host_val = _prompt_param("host_address", "Platform API base URL or webhook endpoint", "str", False, "")
        if host_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Connection binding configuration cancelled. Rolling back partial keys.")
            for k in [k for k in config_map if k.startswith(prefix)]:
                del config_map[k]
            return
        if host_val:
            config_map[prefix + "HOST_ADDRESS"] = _format_env_value(host_val)
        key_val = _prompt_param("service_key", "API key, bot token, or webhook URL (includes credentials)", "str", False, "")
        if key_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Connection binding configuration cancelled. Rolling back partial keys.")
            for k in [k for k in config_map if k.startswith(prefix)]:
                del config_map[k]
            return
        if key_val:
            config_map[prefix + "SERVICE_KEY"] = _format_env_value(key_val)
        timeout_val = _prompt_param("timeout", "HTTP timeout in seconds", "int", False, 30)
        if timeout_val is _BACK_SENTINEL:
            ASCIIColors.yellow("\n  ⚠️ Connection binding configuration cancelled. Rolling back partial keys.")
            for k in [k for k in config_map if k.startswith(prefix)]:
                del config_map[k]
            return
        config_map[prefix + "TIMEOUT"] = _format_env_value(timeout_val)


def _get_connection_binding_description(b_name: str) -> Optional[Dict[str, Any]]:
    """Load description.yaml for a specific connection binding."""
    try:
        from lollms_client.lollms_connection_binding import LollmsConnectionBindingManager
        bindings_dir = Path(LollmsConnectionBindingManager.__init__.__code__.co_filename).parent / "connection_bindings" / b_name
        desc_file = bindings_dir / "description.yaml"
        if desc_file.exists():
            import yaml
            with open(desc_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return None


def _add_connection_profile_flow(config_map: Dict[str, str]):
    """Add a new instance profile (channel/conversation) for a connection binding."""
    connection_bindings = _get_configured_aliases("connection", config_map, "BINDINGS")
    if not connection_bindings:
        ASCIIColors.warning("No connection bindings configured yet. Add a binding first.")
        return

    selected_binding = _safe_select("Select connection binding:", connection_bindings)
    if not selected_binding:
        return

    existing = _get_configured_aliases("connection", config_map, "PROFILES")
    raw_alias = _safe_input_or_back(
        "Enter alias for this instance/channel (e.g., 'general', 'alerts', 'support')",
        "general"
    )
    if raw_alias is None:
        ASCIIColors.yellow("\n  ⚠️ Connection profile creation cancelled.")
        return
    if not raw_alias.strip():
        alias = _generate_unique_alias("general", existing)
        ASCIIColors.info(f"  ℹ️ Auto-named profile: {alias}")
    else:
        alias = _generate_unique_alias(raw_alias, existing)
        if alias != _sanitize_alias(raw_alias):
            ASCIIColors.info(f"  ℹ️ Name collision resolved: {alias}")
    if not alias:
        return

    _configure_connection_profile_instance(selected_binding, alias, config_map)


def _configure_connection_profile_instance(binding_alias: str, instance_alias: str, config_map: Dict[str, str]):
    """Configure a connection instance profile (which channel/conversation to target)."""
    prefix = f"CONNECTION_PROFILES_{instance_alias}_"
    config_map[prefix + "BINDING_ALIAS"] = binding_alias

    # Prompt for instance_name (channel name/chat ID)
    instance_name = _safe_input(
        "Enter the channel name or conversation ID "
        "(e.g., '#general', 'my-group-chat', or a numeric chat_id)",
        "general"
    )
    if not instance_name.strip():
        ASCIIColors.warning("Instance name is required. Skipping profile creation.")
        return
    config_map[prefix + "INSTANCE_NAME"] = instance_name.strip()

    if _safe_confirm(f"Make '{instance_alias}' the default instance profile?", default=(instance_alias == "general")):
        config_map[prefix + "IS_DEFAULT"] = "true"
        _enforce_single_default_profile("connection", config_map)
    else:
        config_map[prefix + "IS_DEFAULT"] = "false"
        _enforce_single_default_profile("connection", config_map)

    ASCIIColors.green(f"\n  ✓ Saved connection instance profile: {instance_alias} (→ {binding_alias}:{instance_name})")


def _connection_edit_menu(config_map: Dict[str, str]):
    """Edit an existing connection binding or instance profile."""
    bindings = _get_configured_aliases("connection", config_map, "BINDINGS")
    profiles = _get_configured_aliases("connection", config_map, "PROFILES")

    all_choices = []
    for b in bindings:
        all_choices.append(f"Binding: {b}")
        all_choices.append(f"Rename Binding Instance: {b}")
    for p in profiles:
        all_choices.append(f"Profile: {p}")
        all_choices.append(f"Rename Profile Instance: {p}")

    if not all_choices:
        ASCIIColors.warning("No connection bindings or profiles to edit.")
        return

    selected = _safe_select("Select what to edit:", all_choices)
    if not selected:
        return

    if selected.startswith("Binding:"):
        alias = selected.split(": ", 1)[1].strip()
        _edit_keys_menu("connection", "BINDINGS", alias, config_map)
    elif selected.startswith("Profile:"):
        alias = selected.split(": ", 1)[1].strip()
        _edit_keys_menu("connection", "PROFILES", alias, config_map)
    elif selected.startswith("Rename Binding Instance:"):
        alias = selected.split(": ", 1)[1].strip()
        _rename_entry("connection", "BINDINGS", alias, config_map)
    elif selected.startswith("Rename Profile Instance:"):
        alias = selected.split(": ", 1)[1].strip()
        _rename_entry("connection", "PROFILES", alias, config_map)


def _connection_delete_menu(config_map: Dict[str, str]):
    """Delete an existing connection binding or instance profile."""
    bindings = _get_configured_aliases("connection", config_map, "BINDINGS")
    profiles = _get_configured_aliases("connection", config_map, "PROFILES")

    all_choices = []
    for b in bindings:
        all_choices.append(f"Binding: {b}")
    for p in profiles:
        all_choices.append(f"Profile: {p}")

    if not all_choices:
        ASCIIColors.warning("No connection bindings or profiles to delete.")
        return

    selected = _safe_select("Select what to delete:", all_choices)
    if not selected:
        return

    if selected.startswith("Binding:"):
        alias = selected.split(": ", 1)[1].strip()
        _delete_entry("connection", "BINDINGS", alias, config_map)
    elif selected.startswith("Profile:"):
        alias = selected.split(": ", 1)[1].strip()
        _delete_entry("connection", "PROFILES", alias, config_map)




def _show_tool_registry(persona) -> None:
    """Discovers and lists available tools on the personality."""
    from lollms_client.lollms_personality.lollms_personality import LollmsPersonality
    active = persona._discover_tools(None, None) if isinstance(persona, LollmsPersonality) else {}
    print("\n🔧 Available Tools:")
    for t_name, t_spec in active.items():
        params = t_spec.get("parameters", [])
        param_str = ", ".join(p["name"] for p in params)
        print(f"   🛠️  {t_name}({param_str})")
        print(f"      {t_spec.get('description', '')[:80]}")

def build_client_with_connection() -> "LollmsClient":
    """
    Builds a LollmsClient with both LLM and CONNECTION profiles.
    Resolution order:
      1. Try get_client_from_env with create_connection=True (uses saved wizard config).
      2. If no connection profiles found, fall back to interactive webhook prompt.
    """
    from lollms_client import LollmsClient
    from lollms_client.lollms_config_cli_env import get_client_from_env

    ASCIIColors.info("🔧 Resolving LLM + Connection configuration...")

    # First attempt: Load everything from saved config (including CONNECTION profiles)
    try:
        full_client = get_client_from_env(
            create_llm=True,
            create_connection=True,
            run_wizard_if_fail=True,
        )
        if full_client.connection:
            ASCIIColors.success("✅ Connection loaded from saved configuration.")
            return full_client
    except Exception as e:
        ASCIIColors.warning(f"Full-config load (with connection) failed: {e}")
        ASCIIColors.info("Falling back to LLM-only load + interactive webhook setup...")

    # Second attempt: LLM only, then prompt for webhook
    try:
        llm_client = get_client_from_env(
            create_llm=True,
            run_wizard_if_fail=True,
        )
    except Exception as e:
        ASCIIColors.error(f"LLM config failed: {e}")
        sys.exit(1)

    active_binding = getattr(llm_client.llm, 'binding_name', '?')
    active_model = getattr(llm_client.llm, 'model_name', '?')
    ASCIIColors.success(f"✅ LLM ready: {active_binding} / {active_model}")

    # ── Interactive Connection Setup ──────────────────────────────────
    webhook_url = os.getenv("LOLLMS_WEBHOOK_URL", "").strip()

    if not webhook_url:
        print()
        ASCIIColors.info("No webhook URL found in LOLLMS_WEBHOOK_URL env var.")
        ASCIIColors.info("You can also create one via the wizard: python -m lollms_client.lollms_config_cli_env")
        ASCIIColors.info("  - Slack:  https://hooks.slack.com/services/T.../B.../XXX")
        ASCIIColors.info("  - Discord: (Server Settings → Integrations → Webhooks)")
        ASCIIColors.info("  - Test:   https://webhook.site")
        webhook_url = input("\n  Enter webhook URL (or press Enter to skip): ").strip()

    # Re-create the client with connection profiles injected
    conn_binding_profiles: Dict[str, Any] = {}
    conn_model_profiles: Dict[str, Any] = {}

    if webhook_url:
        ASCIIColors.info(f"📡 Using webhook: {webhook_url[:70]}...")
        conn_binding_profiles = {
            "webhook-main": {
                "binding_name": "generic_webhook",
                "binding_config": {
                    "service_key": webhook_url,
                    "timeout": 30,
                },
                "is_default": True,
            }
        }

        conn_model_profiles = {
            "webhook-default": {
                "binding_profile_name": "webhook-main",
                "model_name": "default",
                "is_default": True,
            },
            "webhook-alerts": {
                "binding_profile_name": "webhook-main",
                "model_name": "alerts",
                "is_default": False,
            },
        }

    client = LollmsClient(
        llm_binding_name=active_binding or "ollama",
        llm_binding_config={
            "model_name": active_model,
        },
        user_name="You",
        ai_name="Assistant",
        connection_binding_profiles=conn_binding_profiles if conn_binding_profiles else None,
        connection_model_profiles=conn_model_profiles if conn_model_profiles else None,
        debug=True,
    )

    return client

def _deduplicate_binding_keys(config_map: Dict[str, str]) -> None:
    """
    Removes duplicate keys in config_map that differ only by case (e.g., VERIFY_SSL_CERTIFICATE vs verify_ssl_certificate).
    Prefers the UPPERCASE version as the canonical key.
    """
    upper_map = {}
    for k in config_map.keys():
        k_upper = k.upper()
        if k_upper not in upper_map:
            upper_map[k_upper] = k
        else:
            # If we already have a key with this upper name, check if the current one is UPPERCASE
            existing_key = upper_map[k_upper]
            if k == k_upper and existing_key != k_upper:
                # Current key is UPPERCASE, existing is not. Prefer current.
                upper_map[k_upper] = k

    # Delete keys that are not the preferred (UPPERCASE or first) one
    for k in list(config_map.keys()):
        k_upper = k.upper()
        preferred = upper_map.get(k_upper)
        if preferred != k:
            del config_map[k]

def _normalize_config_map_keys(config_map: Dict[str, str]) -> None:
    """Normalizes all keys in config_map to UPPERCASE, resolving casing conflicts."""
    new_map = {}
    for k, v in config_map.items():
        new_map[k.upper()] = v
    config_map.clear()
    config_map.update(new_map)

def _load_existing_env_to_map(cli_env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    config_map = {}

    # Prioritize the shared, machine-wide config file (respects $LOLLMS_CONFIG_FILE).
    home_yaml = _get_default_config_file()
    if home_yaml.exists():
        try:
            config_map.update(_flatten_dict_to_env(load_yaml_file(home_yaml)))
        except Exception as e:
            ASCIIColors.warning(f"Failed to load existing config.yaml from {home_yaml}: {e}")

    # Then local .lollms_code/config.yaml
    local_yaml = Path.cwd() / ".lollms_code" / "config.yaml"
    if local_yaml.exists():
        try:
            config_map.update(_flatten_dict_to_env(load_yaml_file(local_yaml)))
        except Exception as e:
            ASCIIColors.warning(f"Failed to load local configuration from {local_yaml}: {e}")

    # Finally, the specified cli_env_path (if any)
    if cli_env_path:
        p = Path(cli_env_path).expanduser()
        if p.exists() and p.suffix in (".yaml", ".yml"):
            try:
                config_map.update(_flatten_dict_to_env(load_yaml_file(p)))
            except Exception as e:
                ASCIIColors.warning(f"Failed to load configuration from {p}: {e}")

    _normalize_config_map_keys(config_map)
    return config_map

def _save_and_validate(
    config_map: Dict[str, str],
    test_connection: bool = False,
    cli_env_path: Optional[Union[str, Path]] = None,
) -> bool:
    _normalize_config_map_keys(config_map)
    target_yaml_file = _get_default_config_file()
    target_yaml_file.parent.mkdir(parents=True, exist_ok=True)

    yaml_data = _serialize_config_map_to_yaml(config_map)

    try:
        # 1. Write config.yaml to ~/.lollms_client/
        if yaml:
            with open(target_yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        # 2. If explicit custom path was specified outside default home directory, write there too
        explicit_target = Path(cli_env_path).expanduser().resolve() if cli_env_path else None
        if explicit_target and explicit_target != target_yaml_file:
            explicit_target.parent.mkdir(parents=True, exist_ok=True)
            if explicit_target.suffix in (".yaml", ".yml"):
                if yaml:
                    with open(explicit_target, "w", encoding="utf-8") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            else:
                ASCIIColors.warning("⚠️ Non-YAML output format not supported. Skipping.")

        primary_saved_path = explicit_target if explicit_target else target_yaml_file
        ASCIIColors.panel(
            f"Configuration synchronized and saved to: [bold green]{primary_saved_path}[/bold green]",
            title="[bold]✅ Success[/bold]",
            border_style="green"
        )
    except Exception as e:
        ASCIIColors.red(f"\n  ❌ Failed to save configuration: {e}")
        return False

    if test_connection:
        ASCIIColors.info("\nTesting connection with saved configuration...")
        try:
            client = get_client_from_env(cli_env_path=str(explicit_target) if explicit_target else None, run_wizard_if_fail=False)
            ASCIIColors.green(f"✅ Connection verified successfully! Active model: {getattr(client.llm, 'model_name', 'default')}")
        except Exception as conn_err:
            ASCIIColors.warning(f"⚠️ Warning: Connection test failed: {conn_err}")

    return True

def build_wizard_menu(
    config_map: Optional[Dict[str, str]] = None,
    title: str = "Lollms Client Configuration",
    exit_text: str = "↩ Back",
    exit_behavior: str = "discard",
    include_save_exit: bool = False,
    include_save: Optional[bool] = None,
    cli_env_path: Optional[Union[str, Path]] = None,
    standalone: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Builds a configuration wizard menu that can run standalone or be
    embedded as a submenu inside a calling application menu.

    In Standalone Mode (standalone=True):
      - Directly offers '💾 Save', '💾 Save & Exit', and '🔍 Save & Validate Connection'.
      - Owns the complete persistence and exit lifecycle.

    In Embedded / Non-Standalone Mode (standalone=False):
      - Designed for host applications (e.g. lollms_code CLI, WebUI, GUI).
      - The calling application owns the saving lifecycle.
      - Edits directly mutate `config_map` in memory without premature disk writes.
      - Save options are omitted from the submenu by default; selecting `exit_text`
        returns control to the host app with `state["config_map"]` ready for
        the host app to persist when appropriate.

    Args:
        config_map: Mutable configuration map to edit. If None, a fresh map is
            created and pre-loaded from existing config files when available.
        title: Title displayed at the top of the menu.
        exit_text: Label of the menu's exit entry.
        exit_behavior: Governs what happens when the user selects the exit entry:
            "save"     -> persist config_map before returning.
            "ask"      -> prompt Yes/No, then persist if confirmed.
            "discard"  -> return without persisting (calling app owns saving).
        include_save_exit: When True, append 'Save & Exit' action. Defaults to True in standalone mode.
        include_save: When True, append 'Save' action. Defaults to True in standalone mode.
        cli_env_path: Target path for explicit .env/.yaml saving.
        standalone: When True, enables standalone saving choices. When False, lets calling app save.

    Returns:
        tuple: (menu, state) where menu is a configured Menu instance and
        state is {"config_map": dict, "saved": bool, "exited": bool}.
    """
    if exit_behavior not in ("save", "ask", "discard"):
        raise ValueError(
            f"Invalid exit_behavior '{exit_behavior}'. Must be 'save', 'ask', or 'discard'."
        )

    if config_map is None:
        config_map = _load_existing_env_to_map(cli_env_path)
        if config_map:
            source = Path(cli_env_path).expanduser() if cli_env_path else _get_default_config_file()
            ASCIIColors.green(f"✅ Loaded existing configuration from: {source}")

    menu = Menu(title, mode=Menu.MODE_RETURN, exit_text=exit_text)
    intro_desc = (
        "Select a modality to configure, or save your changes."
        if standalone
        else "Select a modality to configure in-memory. Return to the application to save."
    )
    menu.set_intro(intro_desc)


    # Sentinel-string entries (like __RENAME_ENTRY__/__DELETE_ENTRY__ elsewhere
    # in this module) are handled *after* menu.run() by the caller. This keeps
    # the top-level menu navigable even when the entries themselves are new,
    # and lets the caller rebuild the menu in place after the action mutates
    # config_map or the active config path.
    menu.add_choice("📥 Import Configuration From File", value="__IMPORT_CONFIG_FILE__")
    menu.add_choice("⚙️ Change Config File Path", value="__CHANGE_CONFIG_FILE__")
    menu.add_choice("📂 Set Data Folder", value=lambda: _edit_data_folder(config_map))
    menu.add_choice("🧠 Configure LLM", value=lambda: _modality_menu("llm", config_map))
    menu.add_choice("🎨 Configure TTI", value=lambda: _modality_menu("tti", config_map))
    menu.add_choice("🗣️ Configure TTS", value=lambda: _modality_menu("tts", config_map))
    menu.add_choice("👂 Configure STT", value=lambda: _modality_menu("stt", config_map))
    menu.add_choice("🎵 Configure TTM", value=lambda: _modality_menu("ttm", config_map))
    menu.add_choice("🎬 Configure TTV", value=lambda: _modality_menu("ttv", config_map))
    menu.add_choice("🔗 Configure CONNECTION", value=lambda: _modality_menu("connection", config_map))
    menu.add_choice("📚 Configure RAG", value=lambda: _modality_menu("rag", config_map))

    # In standalone mode, include direct saving options. In non-standalone mode, the calling app owns saving.
    should_include_save = include_save if include_save is not None else standalone
    should_include_save_exit = include_save_exit if include_save_exit else standalone

    if should_include_save:
        menu.add_choice("💾 Save", value=lambda: _save_and_validate(config_map, cli_env_path=cli_env_path))

    # `state` is CALLER-OWNED when provided. This is critical: the wizard loop
    # rebuilds the Menu on every iteration; any flag set by callbacks
    # (Save & Exit, Exit without saving) must survive across rebuilds so the
    # loop can act on it. When no state is supplied we create one for
    # backward compatibility (embedded single-shot callers).
    if state is None:
        state = {"config_map": config_map, "saved": False, "exited": False}
    else:
        # Ensure the passed state always carries the correct config_map object.
        state["config_map"] = config_map
        state.setdefault("saved", False)
        state.setdefault("exited", False)

    if should_include_save_exit:
        def _save_and_exit_action():
            _save_and_validate(state["config_map"], test_connection=False, cli_env_path=cli_env_path)
            state["saved"] = True
            state["exited"] = True
            return _BACK_VALUE

        menu.add_choice("💾 Save & Exit", value=_save_and_exit_action)
        menu.add_choice(
            "🔍 Save & Validate Connection",
            value=lambda: _save_and_validate(state["config_map"], test_connection=True, cli_env_path=cli_env_path),
        )

    def _exit_choice_action():
        if exit_behavior == "save":
            _save_and_validate(state["config_map"], test_connection=False, cli_env_path=cli_env_path)
            state["saved"] = True
        elif exit_behavior == "ask":
            if _safe_confirm("Save configuration before exiting?", default=True):
                _save_and_validate(state["config_map"], test_connection=False, cli_env_path=cli_env_path)
                state["saved"] = True
        state["exited"] = True
        return _BACK_VALUE

    menu.add_choice(exit_text, value=_exit_choice_action)

    return menu, state


def run_wizard_and_save(cli_env_path: Optional[Union[str, Path]] = None):
    """Standalone wizard entry point (backward-compatible wrapper).

    Runs the wizard as its own top-level menu. On exit, the configuration is
    saved to `cli_env_path` when provided, otherwise to ~/.lollms_client/config.yaml.
    """
    target_desc = Path(cli_env_path).expanduser() if cli_env_path else _get_default_config_file()
    ASCIIColors.panel(
        f"[bold]Lollms Client Configuration Wizard[/bold]\n[dim]Configure your bindings and profiles.[/dim]\n[dim]Target file: {target_desc}[/dim]",
        title="[bold magenta]🧙 Wizard[/bold magenta]",
        border_style="magenta"
    )

    config_map = _load_existing_env_to_map(cli_env_path)

    # 🚪 Caller-owned state shared across menu rebuilds. The Menu is torn down
    # and rebuilt every loop iteration; only a persistent state object lets
    # callbacks that MUST terminate the wizard (Save & Exit, Exit without
    # Saving) leave a durable signal that the loop can act on.
    state: Dict[str, Any] = {"config_map": config_map, "saved": False, "exited": False}

    while True:
        menu, _ = build_wizard_menu(
            config_map=config_map,
            title="Lollms Client Main Menu (Standalone Wizard)",
            exit_text="🚪 Exit without Saving",
            exit_behavior="discard",
            cli_env_path=cli_env_path,
            standalone=True,
            state=state,
        )

        # Menu entries with string sentinel values (__IMPORT_CONFIG_FILE__,
        # __CHANGE_CONFIG_FILE__) are dispatched here, after menu.run() has
        # returned control. Callable entries (Save & Exit, Exit without Saving,
        # Save, Validate) MUST be invoked here too, whether the Menu ran them
        # inline or returned them — the outcome is recorded on the shared
        # `state` object and inspected immediately afterwards.
        selection = menu.run()

        # Invoke callable selections if the Menu has not already. This is
        # idempotent for behaviour (each action mutates state/config_map) and
        # ensures `state["saved"]` / `state["exited"]` are always observed.
        if callable(selection):
            try:
                selection()
            except Exception as exc:
                ASCIIColors.error(f"\n  ❌ Action failed: {exc}")

        # Save & Exit / Exit without Saving set these flags from inside their
        # callbacks. Now that `state` is caller-owned, the flags survive the
        # rebuild and are reliably readable here.
        if state.get("saved") or state.get("exited"):
            break

        if selection == "__IMPORT_CONFIG_FILE__":
            _import_config_from_file(config_map)
            continue

        if selection == "__CHANGE_CONFIG_FILE__":
            _change_config_file_path()
            continue

        if _is_back_choice(selection):
            break


if __name__ == "__main__":
    run_wizard_and_save()