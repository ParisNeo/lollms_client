"""
lollms_config_cli_env.py
Interactive configuration wizard and unified configuration resolver for Lollms Client.
Supports Multi-Source Ingestion (env, json, yaml, ini) and the Two-Tier Profile System.
"""
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors, Menu

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

def resolve_env_file(cli_env_path: Optional[Union[str, Path]] = None) -> Tuple[Optional[Path], bool]:
    """Resolves the active .env configuration file path."""
    if cli_env_path:
        p = Path(cli_env_path)
        if p.exists():
            return p, False

    home_dir = Path.home() / ".lollms_client"
    home_env = home_dir / ".env"
    home_yaml = home_dir / "config.yaml"

    if home_env.exists():
        return home_env, False
    if home_yaml.exists():
        return home_yaml, False

    return None, True

def load_env_file(env_path: Path) -> Dict[str, str]:
    data = {}
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip("'\"")
    except Exception:
        pass
    return data

_SERIALIZATION_PROFILE_SUFFIXES = (
    "_BINDING_ALIAS", "_BINDING_NAME", "_MODEL_NAME", "_IS_DEFAULT",
    "_VISION_ENABLED", "_FORCED_CONTEXT_SIZE", "_VERIFY_SSL_CERTIFICATE",
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
    for mod in ("llm", "tti", "tts", "stt", "ttm", "ttv", "connection"):
        known_aliases_by_modality[mod] = {
            "bindings": _get_configured_aliases(mod, config_map, "BINDINGS"),
            "profiles": _get_configured_aliases(mod, config_map, "PROFILES"),
        }

    for k, v in config_map.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue

        k_upper = k.upper().strip()
        parts = k_upper.split("_", 2)
        if len(parts) < 3:
            continue

        modality = parts[0].lower()
        category = parts[1].lower()
        if category not in ("bindings", "profiles"):
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
            else:
                if remainder.endswith("_BINDING_NAME"):
                    alias = remainder[:-len("_BINDING_NAME")]
                    param_name = "binding_name"
                else:
                    idx = _find_first_upper_param_boundary(remainder)
                    if idx > 0:
                        alias = remainder[:idx]
                        param_name = remainder[idx + 1:].lower()

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
    bindings = {}
    binding_prefix = f"{clean_prefix}_BINDINGS_"
    for k, v in env_data.items():
        k_upper = k.upper()
        if not k_upper.startswith(binding_prefix):
            continue
        remainder = k_upper[len(binding_prefix):]

        # The binding_prefix already ends with a trailing underscore after BINDINGS.
        # So remainder is e.g. "GENERAL_HOST_ADDRESS" where GENERAL is the alias.
        # Use the helper to find the boundary between alias and parameter key.
        idx = _find_first_upper_param_boundary(remainder)
        if idx <= 0:
            continue

        raw_alias = remainder[:idx]
        raw_key = remainder[idx + 1:]  # skip the underscore separator

        alias = _sanitize_alias(raw_alias)
        key = raw_key.lower()
        if not alias or not key:
            continue
        if alias not in bindings:
            bindings[alias] = {}
        if key == "binding_name":
            bindings[alias]["binding_name"] = v
        elif key == "verify_ssl_certificate":
            bool_ssl = _convert_to_bool(v)
            bindings[alias]["verify_ssl_certificate"] = bool_ssl
            bindings[alias].setdefault("binding_config", {})["verify_ssl_certificate"] = bool_ssl
        elif key == "certificate_file_path":
            cert_val = str(v).strip()
            bindings[alias]["certificate_file_path"] = cert_val
            bindings[alias].setdefault("binding_config", {})["certificate_file_path"] = cert_val
        else:
            bindings[alias].setdefault("binding_config", {})[key] = v
    return bindings

_PROFILE_KNOWN_KEYS = (
    "BINDING_ALIAS", "BINDING_NAME", "MODEL_NAME", "IS_DEFAULT",
    "VISION_ENABLED", "FORCED_CONTEXT_SIZE", "VERIFY_SSL_CERTIFICATE",
    "INSTANCE_NAME",
)


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
        profile_b_config = {k: v for k, v in p_data.items() if k not in {"binding_alias", "is_default", "vision_enabled", "forced_context_size", "model_name", "instance_name", "binding_name", "routing_config"}}
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
            "forced_context_size": p_data.get("forced_context_size"),
            "routing_config": p_data.get("routing_config", {})
        }
    return resolved_profiles

def _get_configured_aliases(binding_type: str, config_map: Dict[str, str], category: str = "BINDINGS") -> List[str]:
    cat = category.upper()
    prefix = f"{binding_type.rstrip('_').upper()}_{cat}_"
    aliases = set()
    for k in config_map:
        k_upper = k.upper()
        if not k_upper.startswith(prefix):
            continue
        remainder = k_upper[len(prefix):]
        if cat == "BINDINGS":
            if remainder.endswith("_BINDING_NAME"):
                alias = remainder[:-len("_BINDING_NAME")]
                if alias:
                    aliases.add(alias)
            else:
                idx = _find_first_upper_param_boundary(remainder)
                if idx > 0:
                    alias = remainder[:idx]
                    if alias:
                        aliases.add(alias)
        elif cat == "PROFILES":
            alias = None
            for known_key in _PROFILE_KNOWN_KEYS:
                marker = f"_{known_key}"
                if remainder.endswith(marker):
                    alias = remainder[:-len(marker)]
                    break
            if alias is None and "_ROUTING_" in remainder:
                idx = remainder.find("_ROUTING_")
                if idx > 0:
                    alias = remainder[:idx]
            if alias:
                aliases.add(alias)
        else:
            parts = remainder.split("_", 1)
            if len(parts) == 2 and parts[0]:
                aliases.add(parts[0])
    return sorted(aliases)

def _get_binding_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_BINDINGS_{alias.upper()}_"
    return {k[len(prefix):].upper(): v for k, v in config_map.items() if k.upper().startswith(prefix)}

def _get_profile_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_PROFILES_{alias.upper()}_"
    return {k[len(prefix):].upper(): v for k, v in config_map.items() if k.upper().startswith(prefix)}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified Client Resolver
# ─────────────────────────────────────────────────────────────────────────────

def _is_modality_configured(b_type: str, env_data: Dict[str, str]) -> bool:
    prefix = b_type.upper()
    if env_data.get(f"{prefix}_BINDING_NAME"):
        return True
    if any(k.startswith(f"{prefix}_BINDINGS_") and k.endswith("_BINDING_NAME") and v for k, v in env_data.items()):
        return True
    return False

def get_client_from_env(
    cli_env_path: Optional[str] = None,
    conf_dict: Optional[Dict[str, Any]] = None,
    conf_file: Optional[Union[str, Path]] = None,
    entry: Optional[str] = None,
    create_llm: bool = True,
    create_tti: bool = False,
    create_stt: bool = False,
    create_tts: bool = False,
    create_ttm: bool = False,
    create_ttv: bool = False,
    create_connection: bool = False,
    run_wizard_if_fail: bool = True
) -> "LollmsClient":
    from lollms_client import LollmsClient

    resolved_env = dict(os.environ)

    home_dir = Path.home() / ".lollms_client"
    home_env = home_dir / ".env"
    if home_env.exists():
        resolved_env.update(load_env_file(home_env))

    home_yaml = home_dir / "config.yaml"
    if home_yaml.exists():
        try:
            yaml_data = load_yaml_file(home_yaml)
            resolved_env.update(_flatten_dict_to_env(yaml_data))
        except Exception as e:
            ASCIIColors.warning(f"Failed to parse {home_yaml}: {e}")

    for local_env in [Path.cwd() / ".env", Path.cwd() / "examples" / ".env"]:
        if local_env.exists():
            resolved_env.update(load_env_file(local_env))

    if conf_dict:
        resolved_env.update(_flatten_dict_to_env(conf_dict))

    if conf_file:
        p = Path(conf_file)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")

        if p.suffix == ".env":
            data = load_env_file(p)
        elif p.suffix == ".json":
            data = _descend_into_entry(load_json_file(p), entry)
        elif p.suffix in (".yaml", ".yml"):
            data = _descend_into_entry(load_yaml_file(p), entry)
        elif p.suffix == ".ini":
            data = load_ini_file(p, entry)
        else:
            raise ValueError(f"Unsupported configuration format: {p.suffix}")

        resolved_env.update(_flatten_dict_to_env(data))

    if cli_env_path:
        p = Path(cli_env_path)
        if p.exists():
            resolved_env.update(load_env_file(p))

    if create_llm and not _is_modality_configured("llm", resolved_env):
        if run_wizard_if_fail:
            ASCIIColors.yellow("⚠️ LLM Configuration incomplete. Starting wizard...")
            run_wizard_and_save()

            if home_yaml.exists():
                try:
                    yaml_data = load_yaml_file(home_yaml)
                    resolved_env.update(_flatten_dict_to_env(yaml_data))
                except Exception:
                    pass

            if home_env.exists():
                resolved_env.update(load_env_file(home_env))

            if not _is_modality_configured("llm", resolved_env):
                raise ValueError("Wizard completed but LLM configuration is still missing.")
        else:
            raise ValueError("LLM configuration is missing.")

    kwargs = {}
    binding_types = {
        "llm": create_llm, "tti": create_tti, "tts": create_tts,
        "stt": create_stt, "ttm": create_ttm, "ttv": create_ttv,
        "connection": create_connection,
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

        binding_name = None
        if "master" in bindings and bindings["master"].get("binding_name"):
            binding_name = bindings["master"]["binding_name"]
        elif bindings:
            first_alias = next(iter(bindings))
            binding_name = bindings[first_alias].get("binding_name")
        else:
            binding_name = resolved_env.get(f"{prefix}_BINDINGS_MASTER_BINDING_NAME") or resolved_env.get(f"{prefix}_BINDING_NAME")

        if binding_name:
            model_name = None
            default_profile = next((p for p in profiles.values() if p.get("is_default")), None)
            if default_profile and default_profile.get("model_name"):
                model_name = default_profile["model_name"]
            elif "master" in profiles and profiles["master"].get("model_name"):
                model_name = profiles["master"]["model_name"]
            elif profiles:
                first_p = next(iter(profiles.values()))
                model_name = first_p.get("model_name")
            else:
                model_name = resolved_env.get("MODEL_NAME") or resolved_env.get(f"{prefix}_MODEL_NAME")

            binding_config = {}
            if "master" in bindings:
                binding_config = dict(bindings["master"].get("binding_config", {}))
            elif bindings:
                first_alias = next(iter(bindings))
                binding_config = dict(bindings[first_alias].get("binding_config", {}))

            for k, v in resolved_env.items():
                if k.startswith(f"{prefix}_") and not k.startswith(f"{prefix}_BINDINGS_") and not k.startswith(f"{prefix}_PROFILES_"):
                    key_lower = k[len(f"{prefix}_"):].lower()
                    if key_lower not in binding_config:
                        binding_config[key_lower] = _convert_to_bool(v) if key_lower == "verify_ssl_certificate" else v

            if model_name and "model_name" not in binding_config:
                binding_config["model_name"] = model_name

            if "host_address" not in binding_config:
                host = resolved_env.get("HOST_ADDRESS") or resolved_env.get(f"{prefix}_HOST_ADDRESS")
                if host:
                    binding_config["host_address"] = host

            if "service_key" not in binding_config:
                key = resolved_env.get("API_KEY") or resolved_env.get(f"{prefix}_API_KEY") or resolved_env.get("SERVICE_KEY") or resolved_env.get(f"{prefix}_SERVICE_KEY")
                if key:
                    binding_config["service_key"] = key

            kwargs[f"{b_type}_binding_name"] = binding_name
            kwargs[f"{b_type}_binding_config"] = binding_config

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
    if p_type == "bool": return raw.lower() in ("true", "1", "yes", "y")
    elif p_type == "int":
        try: return int(raw)
        except: return raw
    elif p_type == "float":
        try: return float(raw)
        except: return raw
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
    ASCIIColors.rich_print(f"\n[bold cyan]── {name} ──[/bold cyan]")
    if desc: ASCIIColors.rich_print(f"[dim]{desc[:120]}{'...' if len(desc)>120 else ''}[/dim]")
    ASCIIColors.rich_print(f"Type: [yellow]{ptype}[/yellow] {'[red](required)[/red]' if mandatory else '[dim](optional)[/dim]'}")
    if ptype == "bool":
        return _safe_confirm("Enter yes/no:", default if isinstance(default, bool) else False)
    else:
        ans = _safe_input_or_back("Enter value", str(default) if default is not None else "")
        if ans is None:
            return _BACK_SENTINEL
        if not ans.strip() and mandatory:
            ASCIIColors.red("  ⚠ Required. Please enter a value.")
            return _prompt_param(name, desc, ptype, mandatory, default)
        return _convert_value(ans, ptype)

def _configure_binding_instance(b_type: str, b_name: str, alias: str, config_map: Dict[str, str]):
    prefix = f"{b_type.upper()}_BINDINGS_{alias.upper()}_"
    temp_params: Dict[str, str] = {"BINDING_NAME": b_name}
    ASCIIColors.green(f"\n  ✓ Selected {b_type.upper()} binding: {b_name} (Alias: {alias})")

    desc = _get_binding_description(b_name, b_type)
    if desc:
        params = desc.get("global_input_parameters", []) + desc.get("model_input_parameters", [])
        for p in params:
            pname = p.get("name", "")
            if not pname or pname == "model_name": continue
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

def _bindings_menu(b_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{b_type.upper()} Bindings", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Add a new binding, edit, or delete an existing one.")
        menu.add_choice("➕ Add new binding", value=lambda: _add_binding_flow(b_type, config_map))

        existing_aliases = _get_configured_aliases(b_type, config_map, "BINDINGS")
        for alias in existing_aliases:
            menu.add_choice(f"✏️ Edit binding: {alias}", value=lambda a=alias: _edit_keys_menu(b_type, "BINDINGS", a, config_map))
            menu.add_choice(f"🗑️ Delete binding: {alias}", value=lambda a=alias: _delete_entry(b_type, "BINDINGS", a, config_map))

        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()

def _edit_keys_menu(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    while True:
        prefix_upper = f"{b_type.upper()}_{category.upper()}_{alias.upper()}_"
        keys = {}
        for k, v in config_map.items():
            if k.upper().startswith(prefix_upper):
                raw_k = k[len(prefix_upper):]
                keys[raw_k] = v

        if not keys:
            ASCIIColors.yellow(f"\n  ⚠️ No keys found for {alias}.")
            return

        menu = Menu(f"Edit {b_type.upper()} {category.capitalize()}: {alias}", mode=Menu.MODE_RETURN, exit_text="↩ Back")
        menu.set_intro("Select a key to edit, add a custom key, or go back.")
        for k, v in keys.items():
            menu.add_choice(f"✏️ Edit {k}: {str(v)[:40]}", value=lambda k=k: _edit_single_key(b_type, category, alias, k, config_map))
        menu.add_choice("➕ Add custom key", value=lambda: _add_custom_key(b_type, category, alias, config_map))
        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
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
    if new_val is not None:
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

def _delete_entry(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    cat_upper = category.upper()
    prefix = f"{b_type.upper()}_{cat_upper}_{alias}_".upper()
    keys_to_delete = [k for k in list(config_map.keys()) if k.upper().startswith(prefix)]

    if not keys_to_delete:
        ASCIIColors.yellow(f"\n  ⚠️ No {category.lower()[:-1]} found with alias '{alias}'.")
        return

    label = "binding" if cat_upper == "BINDINGS" else "profile"
    if _safe_confirm(f"Are you sure you want to delete {label} '{alias}' and all its {len(keys_to_delete)} keys?", default=False):
        was_default = False
        for k in keys_to_delete:
            if k.upper().endswith("_IS_DEFAULT") and str(config_map[k]).lower() in ("true", "1", "yes", "y", "on"):
                was_default = True
            del config_map[k]
        ASCIIColors.green(f"\n  🗑️ Deleted {label}: {alias}")

        if cat_upper == "PROFILES":
            _enforce_single_default_profile(b_type, config_map)
            if was_default:
                ASCIIColors.info("  ℹ️ Default profile deleted; first remaining profile promoted.")
        elif cat_upper == "BINDINGS":
            profile_prefix = f"{b_type.upper()}_PROFILES_"
            orphaned = []
            for k, v in config_map.items():
                if k.upper().startswith(profile_prefix) and k.upper().endswith("_BINDING_ALIAS"):
                    if v.upper() == alias.upper():
                        p_alias = k[len(profile_prefix):-len("_BINDING_ALIAS")].rstrip("_")
                        orphaned.append(p_alias)
            if orphaned:
                ASCIIColors.warning(f"  ⚠️ Note: The following profile(s) still point to deleted binding '{alias}': {', '.join(orphaned)}.")

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

    for k, v in temp_profile.items():
        config_map[profile_prefix + k] = v

    _enforce_single_default_profile(b_type, config_map)
    ASCIIColors.green(f"\n  ✓ Saved profile: {alias}")

def _enforce_single_default_profile(b_type: str, config_map: Dict[str, str]) -> None:
    """
    Enforces the SINGLE-DEFAULT INVARIANT across all profiles of a modality:
    exactly one profile may carry IS_DEFAULT=true; all others are cleared.
    If several (or none) are flagged, the FIRST profile is promoted.
    Mutates config_map in place.
    """
    configured_profiles = _get_configured_aliases(b_type, config_map, "PROFILES")
    if not configured_profiles:
        return

    profile_prefix = f"{b_type.upper()}_PROFILES_"

    flagged = []
    for alias in configured_profiles:
        target_key = f"{profile_prefix}{alias.upper()}_IS_DEFAULT"
        for k, v in config_map.items():
            if k.upper() == target_key:
                if str(v).lower() in ("true", "1", "yes", "y", "on"):
                    flagged.append(alias)
                break

    if len(flagged) == 1:
        owner_alias = flagged[0]
    elif len(flagged) > 1:
        owner_alias = flagged[0]
    else:
        owner_alias = configured_profiles[0]

    for alias in configured_profiles:
        target_key = f"{profile_prefix}{alias.upper()}_IS_DEFAULT"
        found_key = None
        for k in list(config_map.keys()):
            if k.upper() == target_key:
                found_key = k
                break

        val = "true" if alias == owner_alias else "false"
        if found_key:
            config_map[found_key] = val
        else:
            config_map[target_key] = val


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
            menu.add_choice(f"✏️ Edit profile: {alias}{def_badge}", value=lambda a=alias: _edit_keys_menu(b_type, "PROFILES", a, config_map))
            if not is_def:
                menu.add_choice(f"⭐ Set as default: {alias}", value=lambda a=alias: _set_default_profile_action(b_type, a, config_map))
            menu.add_choice(f"🗑️ Delete profile: {alias}", value=lambda a=alias: _delete_entry(b_type, "PROFILES", a, config_map))

        menu.add_choice("↩ Back", value=_BACK_VALUE)

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            selection()


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
        menu.add_choice("🗑️ Delete", value=lambda: _connection_delete_menu(config_map))
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
    for p in profiles:
        all_choices.append(f"Profile: {p}")

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
    conn_binding_profiles = {}
    conn_model_profiles = {}

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

def _load_existing_env_to_map(cli_env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    config_map = {}

    if cli_env_path:
        p = Path(cli_env_path).expanduser()
        if p.exists():
            try:
                if p.suffix in (".yaml", ".yml"):
                    config_map.update(_flatten_dict_to_env(load_yaml_file(p)))
                else:
                    config_map.update(load_env_file(p))
            except Exception as e:
                ASCIIColors.warning(f"Failed to load configuration file {p}: {e}")
        return config_map

    home_dir = Path.home() / ".lollms_client"
    home_env = home_dir / ".env"
    if home_env.exists():
        config_map.update(load_env_file(home_env))

    home_yaml = home_dir / "config.yaml"
    if home_yaml.exists():
        try:
            config_map.update(_flatten_dict_to_env(load_yaml_file(home_yaml)))
        except Exception as e:
            ASCIIColors.warning(f"Failed to load existing config.yaml: {e}")

    for local_env in (Path.cwd() / ".lollms_code" / ".env", Path.cwd() / ".lollms_code" / "config.yaml"):
        if local_env.exists():
            try:
                if local_env.suffix in (".yaml", ".yml"):
                    config_map.update(_flatten_dict_to_env(load_yaml_file(local_env)))
                else:
                    config_map.update(load_env_file(local_env))
            except Exception as e:
                ASCIIColors.warning(f"Failed to load local configuration {local_env}: {e}")

    return config_map

def _save_and_validate(
    config_map: Dict[str, str],
    test_connection: bool = False,
    cli_env_path: Optional[Union[str, Path]] = None,
) -> bool:
    target_dir = Path.home() / ".lollms_client"
    target_dir.mkdir(parents=True, exist_ok=True)

    yaml_data = _serialize_config_map_to_yaml(config_map)
    target_yaml_file = target_dir / "config.yaml"
    target_env_file = target_dir / ".env"

    try:
        # 1. Always write synchronized config.yaml to ~/.lollms_client/
        if yaml:
            with open(target_yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        # 2. Always write synchronized .env to ~/.lollms_client/
        with open(target_env_file, "w", encoding="utf-8") as f:
            f.write("# Lollms Client Configuration\n# Synchronized with config.yaml\n\n")
            for k, v in sorted(config_map.items()):
                if v is not None and str(v).strip() != "":
                    f.write(f"{k}={v}\n")

        # 3. If explicit custom path was specified outside default home directory, write there too
        explicit_target = Path(cli_env_path).expanduser().resolve() if cli_env_path else None
        if explicit_target and explicit_target != target_yaml_file and explicit_target != target_env_file:
            explicit_target.parent.mkdir(parents=True, exist_ok=True)
            if explicit_target.suffix in (".yaml", ".yml"):
                if yaml:
                    with open(explicit_target, "w", encoding="utf-8") as f:
                        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            else:
                with open(explicit_target, "w", encoding="utf-8") as f:
                    f.write("# Lollms Client Configuration\n# Generated by wizard\n\n")
                    for k, v in sorted(config_map.items()):
                        if v is not None and str(v).strip() != "":
                            f.write(f"{k}={v}\n")

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
            source = Path(cli_env_path).expanduser() if cli_env_path else Path.home() / ".lollms_client" / "config.yaml"
            ASCIIColors.green(f"✅ Loaded existing configuration from: {source}")

    menu = Menu(title, mode=Menu.MODE_RETURN, exit_text=exit_text)
    intro_desc = (
        "Select a modality to configure, or save your changes."
        if standalone
        else "Select a modality to configure in-memory. Return to the application to save."
    )
    menu.set_intro(intro_desc)

    menu.add_choice("🧠 Configure LLM", value=lambda: _modality_menu("llm", config_map))
    menu.add_choice("🎨 Configure TTI", value=lambda: _modality_menu("tti", config_map))
    menu.add_choice("🗣️ Configure TTS", value=lambda: _modality_menu("tts", config_map))
    menu.add_choice("👂 Configure STT", value=lambda: _modality_menu("stt", config_map))
    menu.add_choice("🎵 Configure TTM", value=lambda: _modality_menu("ttm", config_map))
    menu.add_choice("🎬 Configure TTV", value=lambda: _modality_menu("ttv", config_map))
    menu.add_choice("🔗 Configure CONNECTION", value=lambda: _modality_menu("connection", config_map))

    # In standalone mode, include direct saving options. In non-standalone mode, the calling app owns saving.
    should_include_save = include_save if include_save is not None else standalone
    should_include_save_exit = include_save_exit if include_save_exit else standalone

    if should_include_save:
        menu.add_choice("💾 Save", value=lambda: _save_and_validate(config_map, cli_env_path=cli_env_path))

    state = {"config_map": config_map, "saved": False, "exited": False}

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
    target_desc = Path(cli_env_path).expanduser() if cli_env_path else Path.home() / ".lollms_client" / "config.yaml"
    ASCIIColors.panel(
        f"[bold]Lollms Client Configuration Wizard[/bold]\n[dim]Configure your bindings and profiles.[/dim]\n[dim]Target file: {target_desc}[/dim]",
        title="[bold magenta]🧙 Wizard[/bold magenta]",
        border_style="magenta"
    )

    config_map = _load_existing_env_to_map(cli_env_path)
    while True:
        menu, state = build_wizard_menu(
            config_map=config_map,
            title="Lollms Client Main Menu (Standalone Wizard)",
            exit_text="🚪 Exit without Saving",
            exit_behavior="discard",
            cli_env_path=cli_env_path,
            standalone=True,
        )

        selection = menu.run()
        if _is_back_choice(selection):
            break
        if callable(selection):
            res = selection()
            if _is_back_choice(res):
                break
        if state.get("saved") or state.get("exited"):
            break


if __name__ == "__main__":
    run_wizard_and_save()