"""
lollms_config_cli_env.py
Interactive configuration wizard and unified configuration resolver for Lollms Client.
Supports Multi-Source Ingestion (env, json, yaml, ini) and the Two-Tier Profile System.
"""
import os
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

def _serialize_config_map_to_yaml(config_map: Dict[str, str]) -> Dict[str, Any]:
    """Reconstructs a hierarchical dictionary from flattened env-style keys."""
    yaml_data = {}
    for k, v in config_map.items():
        parts = k.lower().split("_")
        current = yaml_data
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Leaf node
                if v.lower() in ("true", "false"):
                    current[part] = v.lower() == "true"
                else:
                    try:
                        current[part] = int(v)
                    except ValueError:
                        try:
                            current[part] = float(v)
                        except ValueError:
                            current[part] = v
            else:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
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
    """Flattens nested dicts into environment-style keys (e.g. A_B_C = val)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
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
    if isinstance(val, bool): return val
    if isinstance(val, str): return val.lower().strip() in ("true", "1", "yes", "y")
    return False

def _extract_bindings_from_env(prefix: str, env_data: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    bindings = {}
    binding_prefix = f"{prefix}_BINDINGS_"
    for k, v in env_data.items():
        if k.startswith(binding_prefix):
            remainder = k[len(binding_prefix):]
            parts = remainder.split("_", 1)
            if len(parts) == 2:
                alias, key = parts[0].lower(), parts[1].lower()
                if alias not in bindings: bindings[alias] = {}
                if key == "binding_name":
                    bindings[alias]["binding_name"] = v
                elif key == "verify_ssl_certificate":
                    bindings[alias]["verify_ssl_certificate"] = _convert_to_bool(v)
                else:
                    bindings[alias].setdefault("binding_config", {})[key] = v
    return bindings

def _extract_profiles_from_env(prefix: str, bindings: Dict[str, Dict[str, Any]], env_data: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    profiles = {}
    profile_prefix = f"{prefix}_PROFILES_"
    for k, v in env_data.items():
        if k.startswith(profile_prefix):
            remainder = k[len(profile_prefix):]
            parts = remainder.split("_", 1)
            if len(parts) == 2:
                alias, key = parts[0].lower(), parts[1].lower()
                if alias not in profiles: profiles[alias] = {}
                if key == "binding_alias":
                    profiles[alias]["binding_alias"] = v.lower()
                elif key == "model_name":
                    profiles[alias]["model_name"] = v
                elif key == "is_default":
                    profiles[alias]["is_default"] = _convert_to_bool(v)
                elif key == "vision_enabled":
                    profiles[alias]["vision_enabled"] = _convert_to_bool(v)
                elif key == "forced_context_size":
                    try: profiles[alias]["forced_context_size"] = int(v)
                    except: pass
                elif key.startswith("routing_"):
                    profiles[alias].setdefault("routing_config", {})[key[len("routing_"):]] = v
                else:
                    profiles[alias].setdefault("binding_config", {})[key] = v

    resolved_profiles = {}
    for p_alias, p_data in profiles.items():
        b_alias = p_data.get("binding_alias")
        b_info = bindings.get(b_alias, {}) if b_alias else {}
        binding_name = p_data.get("binding_name") or b_info.get("binding_name")
        base_b_config = b_info.get("binding_config", {})
        profile_b_config = {k: v for k, v in p_data.items() if k not in {"binding_alias", "is_default", "vision_enabled", "forced_context_size", "model_name", "binding_name", "routing_config"}}
        merged_b_config = {**base_b_config, **profile_b_config}
        if "model_name" in p_data:
            merged_b_config["model_name"] = p_data["model_name"]
        if not binding_name and not b_alias: continue

        resolved_profiles[p_alias] = {
            "binding_name": binding_name,
            "binding_alias": b_alias,
            "binding_config": merged_b_config,
            "model_name": p_data.get("model_name"),
            "is_default": p_data.get("is_default", False),
            "vision_enabled": p_data.get("vision_enabled", False),
            "forced_context_size": p_data.get("forced_context_size"),
            "routing_config": p_data.get("routing_config", {})
        }
    return resolved_profiles

# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified Client Resolver
# ─────────────────────────────────────────────────────────────────────────────

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
    run_wizard_if_fail: bool = True
) -> "LollmsClient":
    from lollms_client import LollmsClient

    # 1. Start with OS Environment baseline
    resolved_env = dict(os.environ)

    # 2. Merge explicit dict
    if conf_dict:
        resolved_env.update(_flatten_dict_to_env(conf_dict))

    # 3. Merge conf_file
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

    # 4. Merge CLI env path (highest file precedence)
    if cli_env_path:
        p = Path(cli_env_path)
        if p.exists():
            resolved_env.update(load_env_file(p))

    # 4.5 Merge Home YAML config (Highest structural precedence)
    home_yaml = Path.home() / ".lollms_client" / "config.yaml"
    if home_yaml.exists():
        try:
            yaml_data = load_yaml_file(home_yaml)
            resolved_env.update(_flatten_dict_to_env(yaml_data))
        except Exception as e:
            ASCIIColors.warning(f"Failed to parse {home_yaml}: {e}")

    # 5. Auto-resolve or run wizard if incomplete
    required_keys = []
    if create_llm: required_keys.append("LLM_BINDINGS_MASTER_BINDING_NAME")
    # Add checks for other modalities...

    missing = [k for k in required_keys if k not in resolved_env]
    if missing:
        if run_wizard_if_fail:
            ASCIIColors.yellow("⚠️ Configuration incomplete. Starting wizard...")
            run_wizard_and_save()
            home_dir = Path.home() / ".lollms_client"
            home_env = home_dir / ".env"
            home_yaml = home_dir / "config.yaml"

            if home_env.exists():
                resolved_env.update(load_env_file(home_env))
            if home_yaml.exists():
                try:
                    yaml_data = load_yaml_file(home_yaml)
                    resolved_env.update(_flatten_dict_to_env(yaml_data))
                except Exception:
                    pass

            if not resolved_env:
                raise ValueError("Wizard completed but configuration is still missing.")
        else:
            raise ValueError(f"Configuration incomplete. Missing keys: {missing}")

    # 6. Build kwargs
    kwargs = {}
    binding_types = {
        "llm": create_llm, "tti": create_tti, "tts": create_tts,
        "stt": create_stt, "ttm": create_ttm, "ttv": create_ttv
    }

    for b_type, should_create in binding_types.items():
        if not should_create: continue
        prefix = b_type.upper() + "_"

        bindings = _extract_bindings_from_env(prefix, resolved_env)
        profiles = _extract_profiles_from_env(prefix, bindings, resolved_env)

        if bindings: kwargs[f"{b_type}_binding_profiles"] = bindings
        if profiles: kwargs[f"{b_type}_model_profiles"] = profiles

        binding_name = resolved_env.get(prefix + "BINDINGS_MASTER_BINDING_NAME") or resolved_env.get(prefix + "BINDING_NAME")
        if binding_name:
            binding_config = {}
            # Extract legacy or master binding config
            for k, v in resolved_env.items():
                if k.startswith(f"{prefix}_BINDINGS_MASTER_") and k != f"{prefix}_BINDINGS_MASTER_BINDING_NAME":
                    key_lower = k[len(f"{prefix}_BINDINGS_MASTER_"):].lower()
                    binding_config[key_lower] = _convert_to_bool(v) if key_lower == "verify_ssl_certificate" else v
                elif k.startswith(prefix) and not k.startswith(f"{prefix}_BINDINGS_") and not k.startswith(f"{prefix}_PROFILES_"):
                    key_lower = k[len(prefix):].lower()
                    binding_config[key_lower] = _convert_to_bool(v) if key_lower == "verify_ssl_certificate" else v

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
        ASCIIColors.yellow("\n  No description.yaml found. Using basic configuration.\n")
        val = _prompt_param("host_address", "The host address of the server", "str", False, "http://localhost:8000")
        config_map[prefix + "HOST_ADDRESS"] = _format_env_value(val)

def _add_binding_flow(b_type: str, config_map: Dict[str, str]):
    bindings = _list_bindings_by_type(b_type)
    if not bindings: return
    selected = _safe_select(f"Select a {b_type.upper()} binding:", bindings)
    if not selected:
        ASCIIColors.yellow("\n  ⚠️ Binding selection cancelled.")
        return
    alias = _safe_input("Enter an alias for this binding", "master").strip().upper()
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
            model = _safe_select(f"Select {b_type.upper()} Model for '{alias}':", available_models)
            if not model:
                ASCIIColors.yellow(f"\n  ⚠️ Model selection cancelled for profile '{alias}'.")
                return
            config_map[profile_prefix + "MODEL_NAME"] = model
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

def _fetch_available_models(b_type: str, b_name: str, config_map: Dict[str, str], b_alias: str) -> List[str]:
    try:
        from lollms_client import LollmsClient
        from lollms_client.lollms_bindings_utils import get_binding_desc

        b_config = {}
        prefix = f"{b_type.upper()}_BINDINGS_{b_alias}_"

        # Fetch binding description to enforce strict type safety
        desc = get_binding_desc(b_name, b_type) or {}
        params_meta = {p.get("name"): p for p in (desc.get("global_input_parameters", []) + desc.get("model_input_parameters", []))}

        for k, v in config_map.items():
            if k.startswith(prefix) and k != f"{prefix}BINDING_NAME":
                key_lower = k[len(prefix):].lower()
                p_meta = params_meta.get(key_lower, {})
                p_type = p_meta.get("type", "str").lower()

                # Enforce type safety based on metadata
                if p_type == "bool":
                    b_config[key_lower] = _convert_to_bool(v)
                elif p_type == "int":
                    try: b_config[key_lower] = int(v)
                    except ValueError: pass
                elif p_type == "float":
                    try: b_config[key_lower] = float(v)
                    except ValueError: pass
                elif v.lower() in ("null", "none"):
                    b_config[key_lower] = None
                else:
                    b_config[key_lower] = v

        kwargs = {f"{b_type}_binding_name": b_name, f"{b_type}_binding_config": b_config}
        temp_client = LollmsClient(**kwargs)

        models = []
        if b_type == "llm": models = temp_client.list_models()
        elif b_type == "tti": models = temp_client.list_tti_models()
        elif b_type == "tts": models = temp_client.list_tts_voices()
        elif b_type == "stt": models = temp_client.list_stt_models()
        elif b_type == "ttm": models = temp_client.list_ttm_models()
        elif b_type == "ttv": models = temp_client.list_ttv_models()

        return sorted(list(set([m if isinstance(m, str) else m.get("name", m.get("id", "")) for m in models])))
    except: return []

def _add_profile_flow(b_type: str, config_map: Dict[str, str]):
    alias = _safe_input("Enter alias for the profile", "master").strip().upper()
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
    while True:
        menu = Menu(f"{b_type.upper()} Configuration", mode=Menu.MODE_EXECUTE, exit_text="↩ Back")
        menu.set_intro(f"Configure {b_type.upper()} Bindings and Profiles.")
        menu.add_action(f"Configure {b_type.upper()} Bindings", lambda: _bindings_menu(b_type, config_map))
        menu.add_action(f"Configure {b_type.upper()} Profiles", lambda: _profiles_menu(b_type, config_map))
        if menu.run() is None: break

def _load_existing_env_to_map() -> Dict[str, str]:
    config_map = {}
    home_dir = Path.home() / ".lollms_client"

    # 1. Load from .env (backward compatibility)
    home_env = home_dir / ".env"
    if home_env.exists():
        config_map.update(load_env_file(home_env))

    # 2. Load from config.yaml (New standard)
    home_yaml = home_dir / "config.yaml"
    if home_yaml.exists():
        try:
            yaml_data = load_yaml_file(home_yaml)
            config_map.update(_flatten_dict_to_env(yaml_data))
        except Exception as e:
            ASCIIColors.warning(f"Failed to load existing config.yaml: {e}")

    return config_map

def _save_and_validate(config_map: Dict[str, str]):
    target_dir = Path.home() / ".lollms_client"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_env_file = target_dir / ".env"
    target_yaml_file = target_dir / "config.yaml"

    try:
        # 1. Save Flat .env for backward compatibility & OS env injection
        with open(target_env_file, "w", encoding="utf-8") as f:
            f.write("# Lollms Client Configuration\n# Generated by wizard\n\n")
            for k, v in config_map.items():
                if v: f.write(f"{k}={v}\n")

        # 2. Save Hierarchical config.yaml for structured access
        if yaml:
            yaml_data = _serialize_config_map_to_yaml(config_map)
            with open(target_yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            ASCIIColors.panel(f"Configuration saved to: [bold green]{target_yaml_file}[/bold green] and [bold green]{target_env_file}[/bold green]", title="[bold]✅ Success[/bold]", border_style="green")
        else:
            ASCIIColors.panel(f"Configuration saved to: [bold green]{target_env_file}[/bold green] (PyYAML not installed, skipped config.yaml)", title="[bold]✅ Success[/bold]", border_style="green")

    except Exception as e:
        ASCIIColors.red(f"\n  ❌ Failed to save configuration: {e}")

def run_wizard_and_save():
    ASCIIColors.panel(
        "[bold]Lollms Client Configuration Wizard[/bold]\n[dim]Configure your bindings and profiles.[/dim]",
        title="[bold magenta]🧙 Wizard[/bold magenta]",
        border_style="magenta"
    )

    config_map = _load_existing_env_to_map()
    if config_map: ASCIIColors.green("✅ Loaded existing configuration from .env file.")

    while True:
        menu = Menu("Lollms Client Main Menu", mode=Menu.MODE_EXECUTE, exit_text="💾 Save & Exit")
        menu.set_intro("Select a modality to configure or save and exit.")

        menu.add_action("🧠 Configure LLM", lambda: _modality_menu("llm", config_map))
        menu.add_action("🎨 Configure TTI", lambda: _modality_menu("tti", config_map))
        menu.add_action("🗣️ Configure TTS", lambda: _modality_menu("tts", config_map))
        menu.add_action("👂 Configure STT", lambda: _modality_menu("stt", config_map))
        menu.add_action("🎵 Configure TTM", lambda: _modality_menu("ttm", config_map))
        menu.add_action("🎬 Configure TTV", lambda: _modality_menu("ttv", config_map))
        menu.add_action("💾 Save & Validate", lambda: _save_and_validate(config_map))

        if menu.run() is None: break

if __name__ == "__main__":
    run_wizard_and_save()