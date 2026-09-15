# lollms_config_api.py
# Headless configuration engine for building custom UI configuration systems.
#
# This module exposes the resolution/saving machinery used by the CLI wizard
# as pure functions with zero terminal dependencies. Any frontend (PyQt/PySide,
# NiceGUI, Vue/React via a REST backend, etc.) drives the full lifecycle:
#
#   discover_bindings(modality)              -> available binding names
#   get_binding_schema(binding, modality)    -> parameter schema for form building
#   load_config_map(...)                     -> flat env-style config map
#   set_binding_params / set_profile_params  -> mutate the map (headless "wizard")
#   set_default_profile(modality, alias)     -> exclusive default promotion
#   delete_binding / delete_profile          -> remove entries
#   save_config_map(map, path)               -> persist (yaml/json/env)
#   build_client_config(map)                 -> LollmsClient kwargs
#
# The config map is a flat Dict[str, str] (LLM_BINDINGS_LOCAL_HOST_ADDRESS=...)
# which serializes losslessly to/from the Two-Tier YAML schema, a database
# JSON column, or an HTTP payload.
#
# SINGLE-DEFAULT INVARIANT: exactly one profile per modality may be flagged
# IS_DEFAULT. set_default_profile()/set_profile_params(is_default=True)
# enforce exclusivity at write time; build_client_config() auto-promotes the
# first profile at read time when none is flagged.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ascii_colors import ASCIIColors

from lollms_client.lollms_config_cli_env import (
    _flatten_dict_to_env,
    _sanitize_alias,
    _serialize_config_map_to_yaml,
    load_env_file,
    load_json_file,
    load_yaml_file,
    _get_binding_description,
    _get_configured_aliases,
    _extract_bindings_from_env,
    _extract_profiles_from_env,
)

__all__ = [
    "SUPPORTED_MODALITIES",
    "discover_bindings",
    "get_binding_schema",
    "load_config_map",
    "config_map_to_dict",
    "dict_to_config_map",
    "get_configured_bindings",
    "get_configured_profiles",
    "get_binding_params",
    "get_profile_params",
    "set_binding_params",
    "set_profile_params",
    "set_default_profile",
    "delete_binding",
    "delete_profile",
    "save_config_map",
    "build_client_config",
    "build_client_from_map",
]


SUPPORTED_MODALITIES: Tuple[str, ...] = ("llm", "tti", "tts", "stt", "ttm", "ttv", "connection")

_GLOBAL_CONFIG_DIR = Path.home() / ".lollms_client"
_GLOBAL_CONFIG_FILE = _GLOBAL_CONFIG_DIR / "config.yaml"


def _validate_modality(modality: str) -> str:
    normalized = modality.lower().strip()
    if normalized not in SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unknown modality '{modality}'. Supported: {', '.join(SUPPORTED_MODALITIES)}"
        )
    return normalized


def discover_bindings(modality: str) -> List[Dict[str, Any]]:
    """
    Returns the available binding names for a modality.

    Returns:
        [{"name": "ollama"}, ...] — empty list when nothing is installed.
    """
    modality = _validate_modality(modality)

    if modality == "connection":
        try:
            from lollms_client.lollms_connection_binding import get_available_bindings
            names = [
                b.get("binding_name") or b.get("title", "")
                for b in get_available_bindings()
                if b
            ]
        except Exception:
            names = []
    else:
        try:
            from lollms_client.lollms_bindings_utils import list_bindings
            names = [
                b if isinstance(b, str) else b.get("name")
                for b in list_bindings(modality)
                if b
            ]
        except Exception:
            names = []

    seen: set = set()
    unique_names: List[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique_names.append(name)

    return [{"name": n} for n in unique_names]


def get_binding_schema(binding_name: str, modality: str) -> List[Dict[str, Any]]:
    """
    Returns the parameter schema for a binding, ready to drive auto-generated
    UI forms (name / type / mandatory / default / description).
    """
    modality = _validate_modality(modality)
    binding_name = binding_name.strip()

    desc = _get_binding_description(binding_name, modality)
    if desc is None:
        return [
            {
                "name": "host_address",
                "type": "str",
                "mandatory": False,
                "default": "http://localhost:9642",
                "description": "Server host address",
            },
            {
                "name": "service_key",
                "type": "str",
                "mandatory": False,
                "default": "",
                "description": "API / Service key",
            },
            {
                "name": "verify_ssl_certificate",
                "type": "bool",
                "mandatory": False,
                "default": False,
                "description": "Verify SSL certificate",
            },
        ]

    schema: List[Dict[str, Any]] = []
    seen: set = set()
    for p in desc.get("global_input_parameters", []) + desc.get("model_input_parameters", []):
        pname = p.get("name", "")
        if not pname or pname == "model_name" or pname in seen:
            continue
        seen.add(pname)
        schema.append(
            {
                "name": pname,
                "type": p.get("type", "str"),
                "mandatory": bool(p.get("mandatory", False)),
                "default": p.get("default", ""),
                "description": p.get("description", ""),
            }
        )
    return schema


def load_config_map(
    conf_file: Optional[Union[str, Path]] = None,
    conf_dict: Optional[Dict[str, Any]] = None,
    global_config: bool = True,
    local_env: bool = True,
) -> Dict[str, str]:
    """
    Loads a flat, env-style configuration map from layered sources.

    Resolution order (later sources override earlier ones):
        1. ~/.lollms_client/ global config (yaml + .env)
        2. Local project files (./.env and ./examples/.env)
        3. conf_dict (nested structured dict, e.g. from a database JSON column)
        4. conf_file (yaml / json / env file)

    Returns:
        Dict[str, str]: flat map like {"LLM_BINDINGS_MASTER_HOST_ADDRESS": "..."}
    """
    config_map: Dict[str, str] = {}

    if global_config:
        home_env = _GLOBAL_CONFIG_DIR / ".env"
        if home_env.exists():
            config_map.update(load_env_file(home_env))

        if _GLOBAL_CONFIG_FILE.exists():
            try:
                config_map.update(_flatten_dict_to_env(load_yaml_file(_GLOBAL_CONFIG_FILE)))
            except Exception as e:
                ASCIIColors.warning(f"[lollms_config_api] Failed to parse {_GLOBAL_CONFIG_FILE}: {e}")

    if local_env:
        for candidate in (Path.cwd() / ".env", Path.cwd() / "examples" / ".env"):
            if candidate.exists():
                config_map.update(load_env_file(candidate))

    if conf_dict:
        config_map.update(_flatten_dict_to_env(conf_dict))

    if conf_file:
        p = Path(conf_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")

        if p.suffix == ".env":
            file_data = load_env_file(p)
        elif p.suffix == ".json":
            file_data = _flatten_dict_to_env(load_json_file(p))
        elif p.suffix in (".yaml", ".yml"):
            file_data = _flatten_dict_to_env(load_yaml_file(p))
        else:
            raise ValueError(f"Unsupported configuration format: {p.suffix}. Use .yaml, .json, or .env")

        config_map.update(file_data)

    return config_map


def config_map_to_dict(config_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Converts a flat env-style config map back into the structured Two-Tier
    dict (modality -> bindings/profiles -> alias -> params). Canonical shape
    for storing the config inside a database JSON column.
    """
    return _serialize_config_map_to_yaml(config_map)


def dict_to_config_map(structured: Dict[str, Any]) -> Dict[str, str]:
    """
    Converts a structured Two-Tier dict back into a flat env-style config map.
    Inverse of config_map_to_dict.
    """
    return _flatten_dict_to_env(structured)


def get_configured_bindings(config_map: Dict[str, str], modality: str) -> List[str]:
    """Returns the sanitized aliases of all configured bindings for a modality."""
    modality = _validate_modality(modality)
    return _get_configured_aliases(modality, config_map, "BINDINGS")


def get_configured_profiles(config_map: Dict[str, str], modality: str) -> List[str]:
    """Returns the sanitized aliases of all configured profiles for a modality."""
    modality = _validate_modality(modality)
    return _get_configured_aliases(modality, config_map, "PROFILES")


def get_binding_params(config_map: Dict[str, str], modality: str, alias: str) -> Dict[str, str]:
    """Returns the raw parameters of a configured binding (without the env prefix)."""
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    prefix = f"{modality.upper()}_BINDINGS_{alias.upper()}_"
    return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}


def get_profile_params(config_map: Dict[str, str], modality: str, alias: str) -> Dict[str, str]:
    """
    Returns the raw parameters of a configured profile (without the env prefix).
    Includes the aggregated IS_DEFAULT flag so UIs can highlight the default.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    prefix = f"{modality.upper()}_PROFILES_{alias.upper()}_"
    return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}


def set_binding_params(
    config_map: Dict[str, str],
    modality: str,
    alias: str,
    binding_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Creates or updates a binding instance in the config map.

    Args:
        config_map: the flat map to mutate.
        modality: one of SUPPORTED_MODALITIES.
        alias: user-facing alias for this binding instance.
        binding_name: canonical binding implementation name (e.g. "ollama").
        params: binding parameters (host_address, service_key, ...) — values
            are coerced to their env string form.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    prefix = f"{modality.upper()}_BINDINGS_{alias.upper()}_"

    config_map[prefix + "BINDING_NAME"] = binding_name.strip()
    for pname, value in (params or {}).items():
        if value is None:
            continue
        pname_clean = str(pname).strip().upper()
        if not pname_clean or pname_clean == "BINDING_NAME":
            continue
        config_map[prefix + pname_clean] = "true" if isinstance(value, bool) else str(value)

    return config_map


def set_profile_params(
    config_map: Dict[str, str],
    modality: str,
    alias: str,
    binding_alias: str,
    model_name: Optional[str] = None,
    instance_name: Optional[str] = None,
    is_default: bool = False,
    vision_enabled: bool = False,
    forced_context_size: Optional[int] = None,
    routing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Creates or updates a model profile in the config map. When is_default is
    True, the SINGLE-DEFAULT INVARIANT is enforced: every other profile of the
    same modality loses its default flag.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    binding_alias = _sanitize_alias(binding_alias)
    prefix = f"{modality.upper()}_PROFILES_{alias.upper()}_"

    config_map[prefix + "BINDING_ALIAS"] = binding_alias
    if model_name is not None:
        config_map[prefix + "MODEL_NAME"] = str(model_name)
    if instance_name is not None:
        config_map[prefix + "INSTANCE_NAME"] = str(instance_name)
    config_map[prefix + "VISION_ENABLED"] = "true" if vision_enabled else "false"
    if forced_context_size is not None:
        config_map[prefix + "FORCED_CONTEXT_SIZE"] = str(int(forced_context_size))
    for rkey, rvalue in (routing_config or {}).items():
        rkey_clean = str(rkey).strip().upper()
        if rkey_clean:
            config_map[prefix + "ROUTING_" + rkey_clean] = str(rvalue)

    config_map[prefix + "IS_DEFAULT"] = "true" if is_default else "false"
    if is_default:
        set_default_profile(config_map, modality, alias)

    return config_map


def set_default_profile(config_map: Dict[str, str], modality: str, alias: str) -> Dict[str, str]:
    """
    Promotes a profile to be THE default for its modality (exclusive).
    All other profiles of the same modality lose their default flag.
    Raises KeyError if the profile does not exist in the map.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    profile_prefix = f"{modality.upper()}_PROFILES_"
    target_key = f"{profile_prefix}{alias.upper()}_IS_DEFAULT"

    if not any(k.startswith(f"{profile_prefix}{alias.upper()}_") for k in config_map):
        raise KeyError(
            f"Profile '{alias}' not found in {modality} config map. "
            f"Create it first with set_profile_params()."
        )

    for key in list(config_map.keys()):
        if key.startswith(profile_prefix) and key.endswith("_IS_DEFAULT"):
            config_map[key] = "true" if key == target_key else "false"
    config_map[target_key] = "true"

    return config_map


def delete_binding(config_map: Dict[str, str], modality: str, alias: str) -> Dict[str, str]:
    """
    Removes a binding instance and its parameters. Profiles referencing it are
    left untouched; the caller may want to delete or re-link them.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    prefix = f"{modality.upper()}_BINDINGS_{alias.upper()}_"
    for key in [k for k in config_map if k.startswith(prefix)]:
        del config_map[key]
    return config_map


def delete_profile(config_map: Dict[str, str], modality: str, alias: str) -> Dict[str, str]:
    """
    Removes a profile. If the deleted profile was the default, the first
    remaining profile is auto-promoted so the modality never loses its default.
    """
    modality = _validate_modality(modality)
    alias = _sanitize_alias(alias)
    prefix = f"{modality.upper()}_PROFILES_{alias.upper()}_"

    was_default = config_map.get(prefix + "IS_DEFAULT", "").lower() in ("true", "1", "yes", "y", "on")
    for key in [k for k in config_map if k.startswith(prefix)]:
        del config_map[key]

    if was_default:
        remaining_prefix = f"{modality.upper()}_PROFILES_"
        remaining = [
            k[len(remaining_prefix):].split("_", 1)[0]
            for k in config_map
            if k.startswith(remaining_prefix) and k.endswith("_BINDING_ALIAS")
        ]
        if remaining:
            set_default_profile(config_map, modality, remaining[0])

    return config_map


def save_config_map(
    config_map: Dict[str, str],
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persists a config map to disk. When `path` is None, saves to the global
    machine-wide config (~/.lollms_client/config.yaml).

    Returns:
        Path: the resolved path the configuration was written to.
    """
    import yaml

    target = Path(path).expanduser() if path else _GLOBAL_CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)

    suffix = target.suffix.lower()
    if suffix in (".yaml", ".yml"):
        yaml_data = _serialize_config_map_to_yaml(config_map)
        with open(target, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    elif suffix == ".json":
        with open(target, "w", encoding="utf-8") as f:
            json.dump(_serialize_config_map_to_yaml(config_map), f, indent=2)
    elif suffix == ".env":
        with open(target, "w", encoding="utf-8") as f:
            f.write("# Lollms Client Configuration\n")
            for key, value in config_map.items():
                if value:
                    f.write(f"{key}={value}\n")
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}. Use .yaml, .json, or .env")

    return target


def build_client_config(config_map: Dict[str, str], **modalities: bool) -> Dict[str, Any]:
    """
    Translates a flat config map into the exact kwargs LollmsClient expects.
    Modalities default to llm-only unless explicitly enabled via keyword
    (e.g. build_client_config(cfg, tts=True, stt=True)).
    """
    kwargs: Dict[str, Any] = {}
    for modality in SUPPORTED_MODALITIES:
        enabled = modalities.get(modality, modality == "llm")
        if not enabled:
            continue
        prefix = modality.upper()

        bindings = _extract_bindings_from_env(prefix, config_map)
        profiles = _extract_profiles_from_env(prefix, bindings, config_map)

        if bindings:
            kwargs[f"{modality}_binding_profiles"] = bindings
        if profiles:
            kwargs[f"{modality}_model_profiles"] = profiles

        binding_name = None
        if "master" in bindings and bindings["master"].get("binding_name"):
            binding_name = bindings["master"].get("binding_name")
        elif bindings:
            first_alias = next(iter(bindings))
            binding_name = bindings[first_alias].get("binding_name")
        else:
            binding_name = (
                config_map.get(f"{prefix}_BINDINGS_MASTER_BINDING_NAME")
                or config_map.get(f"{prefix}_BINDING_NAME")
            )

        if binding_name:
            default_profile = next((p for p in profiles.values() if p.get("is_default")), None)
            model_name = None
            if default_profile and default_profile.get("model_name"):
                model_name = default_profile["model_name"]
            elif profiles:
                first_p = next(iter(profiles.values()))
                model_name = first_p.get("model_name")

            binding_config: Dict[str, Any] = {}
            if "master" in bindings:
                binding_config = dict(bindings["master"].get("binding_config", {}))
            elif bindings:
                first_alias = next(iter(bindings))
                binding_config = dict(bindings[first_alias].get("binding_config", {}))

            if model_name and "model_name" not in binding_config:
                binding_config["model_name"] = model_name

            kwargs[f"{modality}_binding_name"] = binding_name
            kwargs[f"{modality}_binding_config"] = binding_config

    return kwargs


def build_client_from_map(config_map: Dict[str, str], **modalities: bool) -> "LollmsClient":
    """
    Builds a fully instantiated LollmsClient from a config map.
    """
    from lollms_client import LollmsClient

    client_kwargs = build_client_config(config_map, **modalities)
    return LollmsClient(**client_kwargs)