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
    Alias extraction anchors on known parameter suffixes (the mirror of
    _extract_profiles_from_env) so aliases containing underscores or spaces
    (e.g. glm_5_3, GLM 5.3) are never truncated.
    """
    yaml_data: Dict[str, Any] = {}

    def _convert_scalar(v: str) -> Any:
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    for k, v in config_map.items():
        if not v:
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

        if category == "profiles":
            alias = None
            param_name = None
            for suffix in _SERIALIZATION_PROFILE_SUFFIXES:
                if remainder.endswith(suffix):
                    alias = remainder[: -len(suffix)]
                    param_name = suffix.lstrip("_").lower()
                    break
            if param_name is None and "_ROUTING_" in remainder:
                idx = remainder.find("_ROUTING_")
                alias = remainder[:idx]
                param_name = remainder[idx + 1:].lower()
            if not alias:
                continue
            alias = _sanitize_alias(alias)
            yaml_data.setdefault(modality, {}).setdefault("profiles", {}).setdefault(alias, {})[param_name] = _convert_scalar(v)
        else:
            idx = remainder.find("_")
            if idx <= 0:
                continue
            alias = _sanitize_alias(remainder[:idx])
            param_name = remainder[idx + 1:].lower()
            yaml_data.setdefault(modality, {}).setdefault("bindings", {}).setdefault(alias, {})[param_name] = _convert_scalar(v)

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
            bindings[alias]["verify_ssl_certificate"] = _convert_to_bool(v)
        else:
            bindings[alias].setdefault("binding_config", {})[key] = v
    return bindings

_PROFILE_KNOWN_KEYS = (
    "BINDING_ALIAS", "BINDING_NAME", "MODEL_NAME", "IS_DEFAULT",
    "VISION_ENABLED", "FORCED_CONTEXT_SIZE", "VERIFY_SSL_CERTIFICATE",
    "INSTANCE_NAME",
)


def _extract_profiles_from_env(prefix: str, bindings: Dict[str, Dict[str, Any]], env_data: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    clean_prefix = prefix.rstrip("_").upper()
    profiles = {}
    profile_prefix = f"{clean_prefix}_PROFILES_"

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
    prefix = f"{binding_type.rstrip('_').upper()}_{category}_"
    aliases = set()
    for k in config_map:
        if k.startswith(prefix):
            parts = k[len(prefix):].split("_", 1)
            if len(parts) == 2:
                aliases.add(parts[0])
    return sorted(aliases)

def _get_binding_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_BINDINGS_{alias.upper()}_"
    return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}

def _get_profile_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.rstrip('_').upper()}_PROFILES_{alias.upper()}_"
    return {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}

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
    return "true" if isinstance(value, bool) else str(value)

def _safe_input(prompt: str, default: str = "") -> str:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        return val if val else default
    except EOFError: return default

def _safe_select(prompt: str, choices: List[str]) -> Optional[str]:
    choices_with_cancel = list(choices) + ["🚫 Cancel"]
    try:
        menu = Menu(prompt, mode=Menu.MODE_RETURN)
        menu.set_intro("Use arrow keys to navigate. Select an option and press Enter.")
        for c in choices_with_cancel: menu.add_choice(c, value=c)
        selection = menu.run()
        if selection == "🚫 Cancel":
            return None
        return selection
    except:
        ASCIIColors.yellow(f"\n(Fallback) {prompt}")
        for i, c in enumerate(choices_with_cancel): ASCIIColors.cyan(f"  {i+1}. {c}")
        raw = _safe_input("Enter number", str(len(choices_with_cancel)))
        try:
            val = int(raw)
            if 1 <= val <= len(choices):
                return choices[val-1]
            return None
        except: return None

def _safe_confirm(prompt: str, default: bool = False) -> bool:
    try:
        menu = Menu(prompt, mode=Menu.MODE_RETURN)
        menu.set_intro("Select Yes or No.")
        menu.add_choice("Yes", value=True)
        menu.add_choice("No", value=False)
        res = menu.run()
        return res if res is not None else default
    except:
        raw = _safe_input(f"{prompt} (y/n)", "y" if default else "n")
        return raw.lower().startswith("y")

def _prompt_param(name: str, desc: str, ptype: str, mandatory: bool, default: Any) -> Any:
    ASCIIColors.rich_print(f"\n[bold cyan]── {name} ──[/bold cyan]")
    if desc: ASCIIColors.rich_print(f"[dim]{desc[:120]}{'...' if len(desc)>120 else ''}[/dim]")
    ASCIIColors.rich_print(f"Type: [yellow]{ptype}[/yellow] {'[red](required)[/red]' if mandatory else '[dim](optional)[/dim]'}")
    if ptype == "bool":
        return _safe_confirm("Enter yes/no:", default if isinstance(default, bool) else False)
    else:
        ans = _safe_input("Enter value", str(default) if default is not None else "")
        if not ans.strip() and mandatory:
            ASCIIColors.red("  ⚠ Required. Please enter a value.")
            return _prompt_param(name, desc, ptype, mandatory, default)
        return _convert_value(ans, ptype)

def _configure_binding_instance(b_type: str, b_name: str, alias: str, config_map: Dict[str, str]):
    prefix = f"{b_type.upper()}_BINDINGS_{alias}_"
    config_map[prefix + "BINDING_NAME"] = b_name
    ASCIIColors.green(f"\n  ✓ Selected {b_type.upper()} binding: {b_name} (Alias: {alias})")

    desc = _get_binding_description(b_name, b_type)
    if desc:
        params = desc.get("global_input_parameters", []) + desc.get("model_input_parameters", [])
        for p in params:
            pname = p.get("name", "")
            if not pname or pname == "model_name": continue
            val = _prompt_param(pname, p.get("description", ""), p.get("type", "str"), p.get("mandatory", False), p.get("default"))
            config_map[prefix + pname.upper()] = _format_env_value(val)
    else:
        ASCIIColors.yellow("\n  No description.yaml found. Using standard server configuration.\n")
        default_host = "http://localhost:9642" if b_type in ("tti", "tts", "stt") else "http://localhost:8000"
        host_val = _prompt_param("host_address", f"The host address of the {b_type.upper()} server", "str", False, default_host)
        config_map[prefix + "HOST_ADDRESS"] = _format_env_value(host_val)

        key_val = _prompt_param("service_key", f"API / Service Key for the {b_type.upper()} server (leave blank if none)", "str", False, "")
        if key_val:
            config_map[prefix + "SERVICE_KEY"] = _format_env_value(key_val)

        ssl_val = _prompt_param("verify_ssl_certificate", "Verify SSL certificate", "bool", False, False)
        config_map[prefix + "VERIFY_SSL_CERTIFICATE"] = _format_env_value(ssl_val)

def _add_binding_flow(b_type: str, config_map: Dict[str, str]):
    bindings = _list_bindings_by_type(b_type)
    if not bindings: return
    selected = _safe_select(f"Select a {b_type.upper()} binding:", bindings)
    if not selected:
        ASCIIColors.yellow("\n  ⚠️ Binding selection cancelled.")
        return
    alias = _sanitize_alias(_safe_input("Enter an alias for this binding", "master"))
    if alias: _configure_binding_instance(b_type, selected, alias, config_map)

def _bindings_menu(b_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{b_type.upper()} Bindings", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro("Add a new binding, edit, or delete an existing one.")
        menu.add_action("Add new binding", lambda: _add_binding_flow(b_type, config_map))

        prefix = f"{b_type.upper()}_BINDINGS_"
        existing_aliases = []
        for k in list(config_map.keys()):
            if k.startswith(prefix) and k.endswith("_BINDING_NAME"):
                alias = k[len(prefix):-len("_BINDING_NAME")]
                existing_aliases.append(alias)
                menu.add_action(f"Edit binding: {alias}", lambda a=alias: _edit_keys_menu(b_type, "BINDINGS", a, config_map))
                menu.add_action(f"🗑️ Delete binding: {alias}", lambda a=alias: _delete_entry(b_type, "BINDINGS", a, config_map))

        if menu.run() is None: break

def _edit_keys_menu(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    while True:
        prefix = f"{b_type.upper()}_{category}_{alias}_"
        keys = {k[len(prefix):]: v for k, v in config_map.items() if k.startswith(prefix)}
        if not keys: return

        menu = Menu(f"Edit {b_type.upper()} {category}: {alias}", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro("Select a key to edit or go back.")
        for k, v in keys.items():
            menu.add_action(f"Edit {k}: {v[:40]}", lambda k=k: _edit_single_key(b_type, category, alias, k, config_map))
        menu.add_action("➕ Add custom key", lambda: _add_custom_key(b_type, category, alias, config_map))
        if menu.run() is None: break

def _edit_single_key(b_type: str, category: str, alias: str, key: str, config_map: Dict[str, str]):
    full_key = f"{b_type.upper()}_{category}_{alias}_{key}"
    new_val = _safe_input(f"Enter new value for {key}", config_map.get(full_key, ""))
    if new_val is not None:
        config_map[full_key] = new_val
        ASCIIColors.green(f"  ✓ Updated {key}")

def _add_custom_key(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    new_key = _safe_input("Enter the name of the new key (e.g., SERVICE_KEY)", "").strip().upper()
    if new_key:
        new_val = _safe_input(f"Enter value for {new_key}", "")
        if new_val is not None:
            config_map[f"{b_type.upper()}_{category}_{alias}_{new_key}"] = new_val
            ASCIIColors.green(f"  ✓ Added {new_key}")

def _delete_entry(b_type: str, category: str, alias: str, config_map: Dict[str, str]):
    prefix = f"{b_type.upper()}_{category}_{alias}_"
    keys_to_delete = [k for k in config_map.keys() if k.startswith(prefix)]

    if not keys_to_delete:
        ASCIIColors.yellow(f"\n  ⚠️ No {category.lower()[:-1]} found with alias '{alias}'.")
        return

    if _safe_confirm(f"Are you sure you want to delete {category.lower()[:-1]} '{alias}' and all its {len(keys_to_delete)} keys?", default=False):
        for k in keys_to_delete:
            del config_map[k]
        ASCIIColors.green(f"\n  🗑️ Deleted {category.lower()[:-1]}: {alias}")

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
    profile_prefix = f"{b_type.upper()}_PROFILES_{alias}_"

    configured_bindings = [k[len(f"{b_type.upper()}_BINDINGS_"):-len("_BINDING_NAME")] for k in config_map if k.startswith(f"{b_type.upper()}_BINDINGS_") and k.endswith("_BINDING_NAME")]
    if not configured_bindings:
        ASCIIColors.yellow(f"\n  ⚠️ No {b_type.upper()} bindings configured. Please add a binding first.")
        return

    selected_b_alias = _safe_select(f"Select binding for profile '{alias}':", configured_bindings)
    if not selected_b_alias:
        ASCIIColors.yellow(f"\n  ⚠️ Binding selection cancelled for profile '{alias}'.")
        return
    config_map[profile_prefix + "BINDING_ALIAS"] = selected_b_alias

    b_name = config_map.get(f"{b_type.upper()}_BINDINGS_{selected_b_alias}_BINDING_NAME")
    if b_name:
        available_models = _fetch_available_models(b_type, b_name, config_map, selected_b_alias)
        if available_models:
            choices = list(available_models) + ["✍️ Enter model name manually"]
            selected_model = _safe_select(f"Select {b_type.upper()} Model for '{alias}':", choices)
            if not selected_model:
                ASCIIColors.yellow(f"\n  ⚠️ Model selection cancelled for profile '{alias}'.")
                return
            if selected_model == "✍️ Enter model name manually":
                m_name = _safe_input("Enter model name manually", "")
                if m_name: config_map[profile_prefix + "MODEL_NAME"] = m_name
            else:
                config_map[profile_prefix + "MODEL_NAME"] = selected_model
        else:
            m_name = _safe_input("Enter model name manually", "")
            if m_name: config_map[profile_prefix + "MODEL_NAME"] = m_name

    if _safe_confirm(f"Make '{alias}' the default profile?", default=(alias == "master")):
        config_map[profile_prefix + "IS_DEFAULT"] = "true"

    if b_type == "llm":
        if _safe_confirm(f"Does profile '{alias}' support vision?", default=False):
            config_map[profile_prefix + "VISION_ENABLED"] = "true"
        ctx = _safe_input("Force context size? (leave blank for auto)", "")
        if ctx.strip(): config_map[profile_prefix + "FORCED_CONTEXT_SIZE"] = ctx.strip()

        ASCIIColors.rich_print("\n[bold magenta]── Smart Router Metadata ──[/bold magenta]")
        r_desc = _safe_input("Routing description (keywords)", "")
        if r_desc: config_map[profile_prefix + "ROUTING_DESCRIPTION"] = r_desc
        r_cost = _safe_input("Cost per 1k tokens (0.0 for local)", "0.0")
        if r_cost: config_map[profile_prefix + "ROUTING_COST"] = r_cost
        r_lat = _safe_input("Average latency (ms)", "100")
        if r_lat: config_map[profile_prefix + "ROUTING_LATENCY"] = r_lat
        r_comp = _safe_select("Complexity tier (1=simple, 3=complex)", ["1", "2", "3"])
        if r_comp: config_map[profile_prefix + "ROUTING_COMPLEXITY"] = r_comp

    ASCIIColors.green(f"\n  ✓ Saved profile: {alias}")

def _add_profile_flow(b_type: str, config_map: Dict[str, str]):
    alias = _sanitize_alias(_safe_input("Enter alias for the profile", "master"))
    if alias: _configure_profile_instance(b_type, alias, config_map)

def _profiles_menu(b_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{b_type.upper()} Profiles", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro("Add a new profile, edit, or delete an existing one.")
        menu.add_action("Add new profile", lambda: _add_profile_flow(b_type, config_map))

        prefix = f"{b_type.upper()}_PROFILES_"
        existing_aliases = []
        for k in list(config_map.keys()):
            if k.startswith(prefix) and k.endswith("_BINDING_ALIAS"):
                alias = k[len(prefix):-len("_BINDING_ALIAS")]
                existing_aliases.append(alias)
                menu.add_action(f"Edit profile: {alias}", lambda a=alias: _edit_keys_menu(b_type, "PROFILES", a, config_map))
                menu.add_action(f"🗑️ Delete profile: {alias}", lambda a=alias: _delete_entry(b_type, "PROFILES", a, config_map))

        if menu.run() is None: break

def _modality_menu(b_type: str, config_map: Dict[str, str]):
    if b_type == "connection":
        _connection_modality_menu(config_map)
        return

    while True:
        menu = Menu(f"{b_type.upper()} Configuration", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro(f"Configure {b_type.upper()} Bindings and Profiles.")
        menu.add_action(f"Configure {b_type.upper()} Bindings", lambda: _bindings_menu(b_type, config_map))
        menu.add_action(f"Configure {b_type.upper()} Profiles", lambda: _profiles_menu(b_type, config_map))
        if menu.run() is None: break


def _connection_modality_menu(config_map: Dict[str, str]):
    """Specialized modality menu for CONNECTION bindings (uses instance_name, not model_name)."""
    while True:
        menu = Menu("CONNECTION Configuration", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro(
            "Configure Connection Bindings (Discord, Telegram, Slack, Webhook, etc.) "
            "and Instance Profiles (which channel to send to)."
        )
        menu.add_action("🔗 Add Connection Binding", lambda: _add_connection_binding_flow(config_map))
        menu.add_action("📱 Add Instance Profile (Channel)", lambda: _add_connection_profile_flow(config_map))
        menu.add_action("✏️ Edit Existing", lambda: _connection_edit_menu(config_map))
        menu.add_action("🗑️ Delete", lambda: _connection_delete_menu(config_map))

        existing = _get_configured_aliases("connection", config_map, "BINDINGS")
        existing_profiles = _get_configured_aliases("connection", config_map, "PROFILES")
        if existing:
            ASCIIColors.info(f"\n  Configured bindings: {', '.join(existing)}")
        if existing_profiles:
            ASCIIColors.info(f"  Configured profiles: {', '.join(existing_profiles)}")

        if menu.run() is None: break


def _add_connection_binding_flow(config_map: Dict[str, str]):
    """Add a new connection binding (e.g., generic_webhook) with an alias."""
    conn_bindings = _list_connection_bindings()
    if not conn_bindings:
        ASCIIColors.warning("No connection bindings available. Check connection_bindings/ directory.")
        return

    selected = _safe_select("Select a connection binding:", conn_bindings)
    if not selected:
        return

    alias = _sanitize_alias(_safe_input("Enter an alias for this connection (e.g., 'slack-main', 'discord-ops')", "main"))
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
            config_map[prefix + pname.upper()] = _format_env_value(val)
    else:
        # Fallback: prompt for standard connection params
        ASCIIColors.yellow("\n  No description.yaml found. Using standard connection parameters.\n")
        host_val = _prompt_param("host_address", "Platform API base URL or webhook endpoint", "str", False, "")
        if host_val:
            config_map[prefix + "HOST_ADDRESS"] = _format_env_value(host_val)
        key_val = _prompt_param("service_key", "API key, bot token, or webhook URL (includes credentials)", "str", False, "")
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

    alias = _sanitize_alias(_safe_input(
        "Enter alias for this instance/channel (e.g., 'general', 'alerts', 'support')",
        "general"
    ))
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


def _list_bindings_by_type(b_type: str) -> List[str]:
    """List bindings by modality type, falling back to manual discovery for 'connection'."""
    if b_type == "connection":
        return _list_connection_bindings()
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        return [b if isinstance(b, str) else b.get("name") for b in list_bindings(b_type) if b]
    except: return []


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
    home_yaml = home_dir / "config.yaml"
    if home_yaml.exists():
        try:
            config_map.update(_flatten_dict_to_env(load_yaml_file(home_yaml)))
        except Exception as e:
            ASCIIColors.warning(f"Failed to load existing config.yaml: {e}")

    home_env = home_dir / ".env"
    if home_env.exists():
        config_map.update(load_env_file(home_env))

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
    explicit_target = Path(cli_env_path).expanduser() if cli_env_path else None
    if explicit_target:
        try:
            explicit_target.parent.mkdir(parents=True, exist_ok=True)
            if explicit_target.suffix in (".yaml", ".yml"):
                if not yaml:
                    raise ImportError("PyYAML is required to save .yaml configuration files.")
                yaml_data = _serialize_config_map_to_yaml(config_map)
                with open(explicit_target, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            else:
                with open(explicit_target, "w", encoding="utf-8") as f:
                    f.write("# Lollms Client Configuration\n# Generated by wizard\n\n")
                    for k, v in config_map.items():
                        if v:
                            f.write(f"{k}={v}\n")
            ASCIIColors.panel(
                f"Configuration saved to: [bold green]{explicit_target}[/bold green]",
                title="[bold]✅ Success[/bold]",
                border_style="green"
            )
        except Exception as e:
            ASCIIColors.red(f"\n  ❌ Failed to save configuration to {explicit_target}: {e}")
            return False
    else:
        target_dir = Path.home() / ".lollms_client"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_yaml_file = target_dir / "config.yaml"
            if not yaml:
                raise ImportError("PyYAML is required to save the configuration.")
            yaml_data = _serialize_config_map_to_yaml(config_map)
            with open(target_yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            ASCIIColors.panel(
                f"Configuration saved to: [bold green]{target_yaml_file}[/bold green]",
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
    cli_env_path: Optional[Union[str, Path]] = None,
) -> tuple:
    """Builds a configuration wizard menu that can run standalone or be

    embedded as a submenu inside a bigger menu.

    Args:
        config_map: Mutable configuration map to edit. If None, a fresh map is
            created and pre-loaded from existing config files when available.
        title: Title displayed at the top of the menu.
        exit_text: Label of the menu's exit entry.
        exit_behavior: Governs what happens when the user selects the exit
            entry:
                "save"     -> persist config_map before returning.
                "ask"      -> prompt Yes/No, then persist if confirmed.
                "discard"  -> return without persisting (embedder owns saving).
        include_save_exit: When True, append a "Save & Exit" action that
            persists config_map and marks state["saved"] = True. When False,
            the embedder owns persistence.

    Returns:
        tuple: (menu, state) where menu is a configured Menu instance and
        state is {"config_map": dict, "saved": bool}. The caller is
        responsible for invoking menu.run() (standalone) or wiring the menu
        as a submenu action inside a parent menu.
    """
    """Builds a configuration wizard menu that can run standalone or be
    embedded as a submenu inside a bigger menu.

    Args:
        config_map: Mutable configuration map to edit. If None, a fresh map is
            created and pre-loaded from existing config files when available.
        title: Title displayed at the top of the menu.
        exit_text: Label of the menu's exit entry.
        exit_behavior: Governs what happens when the user selects the exit
            entry:
                "save"     -> persist config_map before returning.
                "ask"      -> prompt Yes/No, then persist if confirmed.
                "discard"  -> return without persisting (embedder owns saving).
        include_save_exit: When True, append a "Save & Exit" action that
            persists config_map and marks state["saved"] = True. When False,
            the embedder owns persistence.

    Returns:
        tuple: (menu, state) where menu is a configured Menu instance and
        state is {"config_map": dict, "saved": bool}. The caller is
        responsible for invoking menu.run() (standalone) or wiring the menu
        as a submenu action inside a parent menu.
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

    menu = Menu(title, mode=Menu.MODE_EXECUTE, exit_text=exit_text)
    menu.set_intro("Select a modality to configure, or save your changes.")

    menu.add_action("🧠 Configure LLM", lambda: _modality_menu("llm", config_map))
    menu.add_action("🎨 Configure TTI", lambda: _modality_menu("tti", config_map))
    menu.add_action("🗣️ Configure TTS", lambda: _modality_menu("tts", config_map))
    menu.add_action("👂 Configure STT", lambda: _modality_menu("stt", config_map))
    menu.add_action("🎵 Configure TTM", lambda: _modality_menu("ttm", config_map))
    menu.add_action("🎬 Configure TTV", lambda: _modality_menu("ttv", config_map))
    menu.add_action("🔗 Configure CONNECTION", lambda: _modality_menu("connection", config_map))
    menu.add_action("💾 Save", lambda: _save_and_validate(config_map, cli_env_path=cli_env_path))

    # CONNECTION modality hint for couples counseling
    if "CONNECTION_BINDINGS_" in str(config_map.keys()):
        ASCIIColors.info(
            "\n⭐ Tip: For multi-user apps (e.g., couples counseling), use "
            "separate CONNECTION profiles per partner. Each partner's channel is "
            "their own private vault. See examples_perso/proactive_partner/channel_router.py"
        )

    state = {"config_map": config_map, "saved": False}

    if include_save_exit:
        if exit_behavior == "save":
            def _save_and_exit_action():
                _save_and_validate(state["config_map"], cli_env_path=cli_env_path)
                state["saved"] = True
            menu.add_action("💾 Save & Exit", _save_and_exit_action)
        elif exit_behavior == "ask":
            def _save_and_exit_action():
                if _safe_confirm("Save configuration before exiting?", default=True):
                    _save_and_validate(state["config_map"], cli_env_path=cli_env_path)
                    state["saved"] = True
            menu.add_action("💾 Save & Exit", _save_and_exit_action)
        else:
            menu.add_action(
                "🔍 Save & Validate Connection",
                lambda: _save_and_validate(state["config_map"], test_connection=True, cli_env_path=cli_env_path),
            )

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

    while True:
        config_map = _load_existing_env_to_map(cli_env_path)
        menu, state = build_wizard_menu(
            config_map=config_map,
            title="Lollms Client Main Menu",
            exit_text="🚪 Exit without Saving",
            include_save_exit=True,
            cli_env_path=cli_env_path,
        )

        def _validate_action():
            _save_and_validate(state["config_map"], test_connection=True, cli_env_path=cli_env_path)
        menu.add_action("🔍 Save & Validate Connection", _validate_action)

        if menu.run() is None or state["saved"]:
            break


if __name__ == "__main__":
    run_wizard_and_save()