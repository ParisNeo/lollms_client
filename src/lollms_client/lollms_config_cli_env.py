"""
lollms_config_cli_env.py
========================
Interactive configuration wizard for Lollms Client.
Scans available LLM bindings and writes a standardized .env file.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ascii_colors import ASCIIColors, Menu
from ascii_colors import questionary

def resolve_env_file(cli_env_path: Optional[str] = None) -> Tuple[Optional[Path], bool]:
    if cli_env_path:
        p = Path(cli_env_path).resolve()
        if p.exists():
            return p, False

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env, False

    home_env = Path.home() / ".lollms-client" / ".env"
    if home_env.exists():
        return home_env, False

    if os.getenv("LLM_BINDING_NAME") and os.getenv("MODEL_NAME"):
        return None, False

    return None, True

def load_env_file(env_path: Path):
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    except ImportError:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip("'\"")

def _list_llm_bindings() -> List[str]:
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        result = list_bindings("llm")
        names = set()
        for item in result:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("binding_name") or ""
                if name:
                    names.add(str(name))
        return sorted(list(names))
    except Exception:
        return ["ollama", "openai", "lollms", "vllm", "llama_cpp_server"]

def _list_bindings_by_type(binding_type: str) -> List[str]:
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        result = list_bindings(binding_type)
        names = set()
        for item in result:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("binding_name") or ""
                if name:
                    names.add(str(name))
        return sorted(list(names))
    except Exception:
        return []

def _get_binding_description(binding_name: str, binding_type: str) -> Optional[Dict[str, Any]]:
    try:
        from lollms_client.lollms_bindings_utils import get_binding_desc
        desc = get_binding_desc(binding_name, binding_type)
        if isinstance(desc, dict) and "error" not in desc:
            return desc
    except Exception:
        pass
    return None

def _convert_value(raw: str, param_type: str) -> Any:
    if param_type == "bool":
        return raw.lower().strip() in ("true", "1", "yes", "y")
    elif param_type == "int":
        try:
            return int(raw)
        except ValueError:
            return raw
    elif param_type == "float":
        try:
            return float(raw)
        except ValueError:
            return raw
    else:
        return raw

def _prompt_param(name: str, desc: str, ptype: str, mandatory: bool, default: Any) -> Any:
    ASCIIColors.rich_print(f"\n[bold cyan]── {name} ──[/bold cyan]")
    if desc:
        short_desc = desc if len(desc) <= 120 else desc[:117] + "..."
        ASCIIColors.rich_print(f"[dim]{short_desc}[/dim]")
    
    mandatory_str = "[red](required)[/red]" if mandatory else "[dim](optional)[/dim]"
    ASCIIColors.rich_print(f"Type: [yellow]{ptype}[/yellow] {mandatory_str}")

    if ptype == "bool":
        answer = questionary.confirm(f"Enter yes/no:", default=default).ask()
        if answer is None:
            return default
        return answer
    else:
        if default is not None and default != "":
            answer = questionary.text(f"Enter value [{default}]:").ask()
            if answer is None:
                return default
            if not answer.strip():
                return default
            return _convert_value(answer, ptype)
        else:
            answer = questionary.text(f"Enter value:").ask()
            if answer is None:
                answer = ""
            if not answer.strip() and mandatory:
                ASCIIColors.red("  ⚠ This parameter is required. Please enter a value.")
                return _prompt_param(name, desc, ptype, mandatory, default)
            return _convert_value(answer, ptype)

def _format_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

def _fetch_available_models(binding_type: str, binding_name: str, config_map: Dict[str, str]) -> List[str]:
    try:
        from lollms_client import LollmsClient
        
        def _convert_to_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower().strip() in ("true", "1", "yes", "y")
            return False

        prefix = binding_type.upper() + "_"
        binding_config = {}
        for k, v in config_map.items():
            if k.startswith(prefix):
                key_lower = k[len(prefix):].lower()
                if key_lower == "verify_ssl_certificate":
                    binding_config[key_lower] = _convert_to_bool(v)
                else:
                    binding_config[key_lower] = v

        kwargs = {f"{binding_type}_binding_name": binding_name, f"{binding_type}_binding_config": binding_config}
        temp_client = LollmsClient(**kwargs)
        
        if binding_type == "llm":
            models = temp_client.list_models()
        elif binding_type == "tti":
            models = temp_client.list_tti_models()
        elif binding_type == "tts":
            models = temp_client.list_tts_voices()
        elif binding_type == "stt":
            models = temp_client.list_stt_models()
        elif binding_type == "ttm":
            models = temp_client.list_ttm_models()
        elif binding_type == "ttv":
            models = temp_client.list_ttv_models()
        else:
            models = []

        model_names = []
        if isinstance(models, list):
            for m in models:
                if isinstance(m, str):
                    model_names.append(m)
                elif isinstance(m, dict):
                    name = m.get("name") or m.get("id") or m.get("model") or m.get("voice")
                    if name:
                        model_names.append(str(name))
        return sorted(list(set(model_names)))
    except Exception:
        return []

def _configure_binding_instance(binding_type: str, binding_name: str, alias: str, config_map: Dict[str, str]):
    """Configures parameters for a single binding instance and stores them in config_map."""
    prefix = f"{binding_type.upper()}_BINDINGS_{alias}_"
    
    config_map[prefix + "BINDING_NAME"] = binding_name
    ASCIIColors.green(f"\n  ✓ Selected {binding_type.upper()} binding: {binding_name} (Alias: {alias})")

    desc = _get_binding_description(binding_name, binding_type)
    if desc:
        global_params = desc.get("global_input_parameters") or desc.get("input_parameters") or []
        model_params = desc.get("model_input_parameters") or []
        all_params = global_params + model_params

        for param in all_params:
            pname = param.get("name", "")
            if not pname or pname == "model_name":
                continue

            pdesc = param.get("description", "")
            ptype = param.get("type", "str")
            pmandatory = param.get("mandatory", False)
            pdefault = param.get("default")

            value = _prompt_param(pname, pdesc, ptype, pmandatory, pdefault)
            config_map[prefix + pname.upper()] = _format_env_value(value)
    else:
        ASCIIColors.yellow("\n  No description.yaml found. Using basic configuration.\n")
        value = _prompt_param("host_address", "The host address of the server", "str", False, "http://localhost:8000")
        config_map[prefix + "HOST_ADDRESS"] = _format_env_value(value)

def _get_configured_aliases(binding_type: str, config_map: Dict[str, str], category: str = "BINDINGS") -> List[str]:
    prefix = f"{binding_type.upper()}_{category}_"
    configured_aliases = set()
    for k in config_map.keys():
        if k.startswith(prefix):
            remainder = k[len(prefix):]
            parts = remainder.split("_", 1)
            if len(parts) == 2:
                configured_aliases.add(parts[0])
    return sorted(list(configured_aliases))

def _get_binding_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.upper()}_BINDINGS_{alias}_"
    keys = {}
    for k, v in config_map.items():
        if k.startswith(prefix):
            key_name = k[len(prefix):]
            keys[key_name] = v
    return keys

def _get_profile_keys(binding_type: str, alias: str, config_map: Dict[str, str]) -> Dict[str, str]:
    prefix = f"{binding_type.upper()}_PROFILES_{alias}_"
    keys = {}
    for k, v in config_map.items():
        if k.startswith(prefix):
            key_name = k[len(prefix):]
            keys[key_name] = v
    return keys

def _edit_binding_keys_menu(binding_type: str, alias: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"Edit {binding_type.upper()} Binding: {alias}", mode='execute')
        
        keys = _get_binding_keys(binding_type, alias, config_map)
        if not keys:
            ASCIIColors.yellow(f"\n  No configuration keys found for binding '{alias}'.")
            return

        for k, v in keys.items():
            display_val = v if len(v) <= 40 else v[:37] + "..."
            menu.add_action(f"Edit {k}: {display_val}", lambda k=k: _edit_single_key(binding_type, "BINDINGS", alias, k, config_map))
            
        menu.add_action("➕ Add custom key", lambda: _add_custom_key(binding_type, "BINDINGS", alias, config_map))
        menu.add_action("⏎ Back to Bindings List", lambda: None)
        menu.run()

def _edit_profile_keys_menu(binding_type: str, alias: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"Edit {binding_type.upper()} Profile: {alias}", mode='execute')
        
        keys = _get_profile_keys(binding_type, alias, config_map)
        if not keys:
            ASCIIColors.yellow(f"\n  No configuration keys found for profile '{alias}'.")
            return

        for k, v in keys.items():
            display_val = v if len(v) <= 40 else v[:37] + "..."
            menu.add_action(f"Edit {k}: {display_val}", lambda k=k: _edit_single_key(binding_type, "PROFILES", alias, k, config_map))
            
        menu.add_action("➕ Add custom key", lambda: _add_custom_key(binding_type, "PROFILES", alias, config_map))
        menu.add_action("⏎ Back to Profiles List", lambda: None)
        menu.run()

def _edit_single_key(binding_type: str, category: str, alias: str, key_name: str, config_map: Dict[str, str]):
    full_key = f"{binding_type.upper()}_{category}_{alias}_{key_name}"
    current_val = config_map.get(full_key, "")
    
    ASCIIColors.cyan(f"\nEditing key: {key_name}")
    new_val = questionary.text(f"Enter new value:", default=current_val).ask()
    
    if new_val is not None:
        config_map[full_key] = new_val
        ASCIIColors.green(f"  ✓ Updated {key_name}")

def _add_custom_key(binding_type: str, category: str, alias: str, config_map: Dict[str, str]):
    new_key = questionary.text("Enter the name of the new key (e.g., SERVICE_KEY):").ask()
    if not new_key:
        return
    
    new_key = new_key.strip().upper()
    new_val = questionary.text(f"Enter value for {new_key}:").ask()
    
    if new_val is not None:
        full_key = f"{binding_type.upper()}_{category}_{alias}_{new_key}"
        config_map[full_key] = new_val
        ASCIIColors.green(f"  ✓ Added {new_key}")

def _bindings_menu(binding_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{binding_type.upper()} Bindings Configuration", mode='execute')
        menu.add_action("Add new binding", lambda: _add_binding_flow(binding_type, config_map))
        
        configured = _get_configured_aliases(binding_type, config_map, "BINDINGS")
        for alias in configured:
            keys = _get_binding_keys(binding_type, alias, config_map)
            b_name = keys.get("BINDING_NAME", "unknown")
            menu.add_action(f"Edit binding: {alias} ({b_name})", lambda a=alias: _edit_binding_keys_menu(binding_type, a, config_map))
            
        menu.add_action("⏎ Back to Modality Menu", lambda: None)
        menu.run()

def _add_binding_flow(binding_type: str, config_map: Dict[str, str], edit_alias: Optional[str] = None):
    bindings = _list_bindings_by_type(binding_type)
    if not bindings:
        ASCIIColors.yellow(f"\n  ⚠️ No {binding_type.upper()} bindings found.")
        return

    selected = questionary.select(
        f"Select a {binding_type.upper()} binding to configure:",
        choices=bindings
    ).ask()

    if not selected:
        return

    default_alias = edit_alias if edit_alias else "master"
    alias = questionary.text(f"Enter an alias for this binding:", default=default_alias).ask()
    if not alias:
        return
    
    alias = alias.strip().upper()
    _configure_binding_instance(binding_type, selected, alias, config_map)

def _configure_profile_instance(binding_type: str, alias: str, config_map: Dict[str, str], edit: bool = False):
    profile_prefix = f"{binding_type.upper()}_PROFILES_{alias}_"
    
    configured = _get_configured_aliases(binding_type, config_map, "BINDINGS")
    if not configured:
        ASCIIColors.yellow(f"\n  ⚠️ No {binding_type.upper()} bindings configured. Please add a binding first.")
        return

    selected_binding_alias = questionary.select(
        f"Select binding for profile '{alias}':",
        choices=configured
    ).ask()

    if not selected_binding_alias:
        return

    config_map[profile_prefix + "BINDING_ALIAS"] = selected_binding_alias

    binding_config_map = {k: v for k, v in config_map.items() if k.startswith(f"{binding_type.upper()}_BINDINGS_{selected_binding_alias}_")}
    binding_name = binding_config_map.get(f"{binding_type.upper()}_BINDINGS_{selected_binding_alias}_BINDING_NAME")
    
    if binding_name:
        ASCIIColors.rich_print(f"\n[bold cyan]Select Model for profile '{alias}'[/bold cyan]")
        with ASCIIColors.status(f"[cyan]Fetching available models for {binding_name}...[/cyan]", spinner="dots"):
            available_models = _fetch_available_models(binding_type, binding_name, binding_config_map)

        if available_models:
            model_choice = questionary.select(
                f"Select {binding_type.upper()} Model for '{alias}':",
                choices=available_models
            ).ask()
            if model_choice:
                config_map[profile_prefix + "MODEL_NAME"] = model_choice
        else:
            ASCIIColors.yellow(f"  ⚠️ Could not fetch models automatically for {binding_name}.")
            p_model_name = questionary.text("Enter model name manually:").ask()
            if p_model_name:
                config_map[profile_prefix + "MODEL_NAME"] = p_model_name

    is_default = questionary.confirm(f"Make '{alias}' the default profile?", default=(alias == "master")).ask()
    if is_default:
        config_map[profile_prefix + "IS_DEFAULT"] = "true"

    if binding_type == "llm":
        vision_enabled = questionary.confirm(f"Does profile '{alias}' support vision?", default=False).ask()
        if vision_enabled:
            config_map[profile_prefix + "VISION_ENABLED"] = "true"

        forced_ctx = questionary.text("Force context size? (leave blank for auto)", default="").ask()
        if forced_ctx.strip():
            config_map[profile_prefix + "FORCED_CONTEXT_SIZE"] = forced_ctx.strip()

        ASCIIColors.rich_print("\n[bold magenta]── Smart Router Metadata ──[/bold magenta]")
        ASCIIColors.rich_print("[dim]Used by the 'smart_router' binding to route prompts intelligently.[/dim]")
        
        r_desc = questionary.text("Routing description (keywords for this model):", default="").ask()
        if r_desc:
            config_map[profile_prefix + "ROUTING_DESCRIPTION"] = r_desc
            
        r_cost = questionary.text("Cost per 1k tokens (0.0 for local):", default="0.0").ask()
        if r_cost:
            config_map[profile_prefix + "ROUTING_COST"] = r_cost
            
        r_latency = questionary.text("Average latency (ms):", default="100").ask()
        if r_latency:
            config_map[profile_prefix + "ROUTING_LATENCY"] = r_latency
            
        r_complexity = questionary.select("Complexity tier (1=simple, 3=complex):", choices=["1", "2", "3"]).ask()
        if r_complexity:
            config_map[profile_prefix + "ROUTING_COMPLEXITY"] = r_complexity

    ASCIIColors.green(f"\n  ✓ Saved profile: {alias}")

def _profiles_menu(binding_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{binding_type.upper()} Profiles Configuration", mode='execute')
        menu.add_action("Add new profile", lambda: _add_profile_flow(binding_type, config_map))
        
        configured = _get_configured_aliases(binding_type, config_map, "PROFILES")
        for alias in configured:
            keys = _get_profile_keys(binding_type, alias, config_map)
            b_alias = keys.get("BINDING_ALIAS", "unknown")
            m_name = keys.get("MODEL_NAME", "unknown")
            menu.add_action(f"Edit profile: {alias} ({b_alias}/{m_name})", lambda a=alias: _edit_profile_keys_menu(binding_type, a, config_map))
            
        menu.add_action("⏎ Back to Modality Menu", lambda: None)
        menu.run()

def _add_profile_flow(binding_type: str, config_map: Dict[str, str], edit_alias: Optional[str] = None):
    default_alias = edit_alias if edit_alias else "master"
    alias = questionary.text(f"Enter alias for the profile:", default=default_alias).ask()
    if not alias:
        return
    alias = alias.strip().upper()
    _configure_profile_instance(binding_type, alias, config_map, edit=bool(edit_alias))

def _modality_menu(binding_type: str, config_map: Dict[str, str]):
    while True:
        menu = Menu(f"{binding_type.upper()} Configuration", mode='execute')
        menu.add_action(f"Configure {binding_type.upper()} Bindings", lambda: _bindings_menu(binding_type, config_map))
        menu.add_action(f"Configure {binding_type.upper()} Profiles", lambda: _profiles_menu(binding_type, config_map))
        menu.add_action("⏎ Back to Main Menu", lambda: None)
        menu.run()

def _save_and_validate(config_map: Dict[str, str]):
    ASCIIColors.rule("[bold cyan]Validating Connections[/bold cyan]")
    
    try:
        from lollms_client import LollmsClient
        
        def _convert_to_bool(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower().strip() in ("true", "1", "yes", "y")
            return False

        kwargs = {}
        for b_type in ["llm", "tti", "tts", "stt", "ttm", "ttv"]:
            b_name = config_map.get(f"{b_type.upper()}_BINDINGS_MASTER_BINDING_NAME")
            if b_name:
                prefix = f"{b_type.upper()}_BINDINGS_MASTER_"
                b_config = {}
                for k, v in config_map.items():
                    if k.startswith(prefix) and k != prefix + "BINDING_NAME":
                        key_lower = k[len(prefix):].lower()
                        if key_lower == "verify_ssl_certificate":
                            b_config[key_lower] = _convert_to_bool(v)
                        else:
                            b_config[key_lower] = v
                kwargs[f"{b_type}_binding_name"] = b_name
                kwargs[f"{b_type}_binding_config"] = b_config

        with ASCIIColors.status("[cyan]Pinging servers...[/cyan]", spinner="dots"):
            temp_client = LollmsClient(**kwargs)
        ASCIIColors.green("  ✅ Connections validated successfully.")
    except Exception as e:
        ASCIIColors.red(f"\n  ❌ Connection validation failed: {e}")
        ASCIIColors.yellow("  The configuration has been discarded. Please re-run the wizard.")
        return

    target_dir = Path.home() / ".lollms-client"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / ".env"

    try:
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("# Lollms Client Configuration\n")
            f.write("# Generated by lollms_config_cli_env wizard\n\n")
            for k, v in config_map.items():
                if not v:
                    continue
                f.write(f"{k}={v}\n")
        
        ASCIIColors.panel(f"Configuration saved to: [bold green]{target_file}[/bold green]", title="[bold]✅ Success[/bold]", border_style="green")
    except Exception as e:
        ASCIIColors.red(f"\n  ❌ Failed to save configuration: {e}")

def run_wizard_and_save():
    ASCIIColors.panel(
        "[bold]Lollms Client Configuration Wizard[/bold]\n[dim]This wizard will help you configure your bindings and profiles.[/dim]",
        title="[bold magenta]🧙 Wizard[/bold magenta]",
        border_style="magenta"
    )

    config_map = {}

    while True:
        menu = Menu("Lollms Client Main Menu", mode='execute')
        
        def _llm(): _modality_menu("llm", config_map)
        def _tti(): _modality_menu("tti", config_map)
        def _tts(): _modality_menu("tts", config_map)
        def _stt(): _modality_menu("stt", config_map)
        def _ttm(): _modality_menu("ttm", config_map)
        def _ttv(): _modality_menu("ttv", config_map)
        def _save(): _save_and_validate(config_map)
        
        menu.add_action("🧠 Configure LLM", _llm)
        menu.add_action("🎨 Configure TTI", _tti)
        menu.add_action("🗣️ Configure TTS", _tts)
        menu.add_action("👂 Configure STT", _stt)
        menu.add_action("🎵 Configure TTM", _ttm)
        menu.add_action("🎬 Configure TTV", _ttv)
        menu.add_action("💾 Save & Validate", _save)
        menu.add_action("🚪 Exit", lambda: None)
        
        choice = menu.run()
        if choice is None:
            break

def auto_resolve_or_wizard(cli_env_path: Optional[str] = None) -> bool:
    env_path, needs_wizard = resolve_env_file(cli_env_path)
    
    if env_path:
        load_env_file(env_path)
        return True
        
    if not needs_wizard:
        return True
        
    ASCIIColors.yellow("⚠️ No configuration found in .env files or environment variables.")
    ASCIIColors.cyan("🧙 Starting configuration wizard...")
    try:
        run_wizard_and_save()
        home_env = Path.home() / ".lollms-client" / ".env"
        if home_env.exists():
            load_env_file(home_env)
            return True
        else:
            ASCIIColors.red("❌ Wizard did not generate a configuration file. Exiting.")
            return False
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration wizard failed: {e}")
        return False

def _extract_bindings_from_env(prefix: str) -> Dict[str, Dict[str, Any]]:
    """
    Scans environment variables for Binding configurations.
    Expected format: {PREFIX}_BINDINGS_{ALIAS}_{KEY}
    """
    bindings = {}
    binding_prefix = f"{prefix}_BINDINGS_"

    for k, v in os.environ.items():
        if k.startswith(binding_prefix):
            remainder = k[len(binding_prefix):]
            parts = remainder.split("_", 1)
            if len(parts) == 2:
                alias, key = parts
                alias = alias.lower()
                key = key.lower()

                if alias not in bindings:
                    bindings[alias] = {}

                if key == "binding_name":
                    bindings[alias]["binding_name"] = v
                elif key == "verify_ssl_certificate":
                    bindings[alias]["verify_ssl_certificate"] = v.lower() in ("true", "1", "yes")
                else:
                    if "binding_config" not in bindings[alias]:
                        bindings[alias]["binding_config"] = {}
                    bindings[alias]["binding_config"][key] = v

    return bindings

def _extract_profiles_from_env(prefix: str, bindings: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Scans environment variables for Profile configurations.
    Expected format: {PREFIX}_PROFILES_{ALIAS}_{KEY}
    Merges the binding_config from the referenced binding.
    """
    profiles = {}
    profile_prefix = f"{prefix}_PROFILES_"

    for k, v in os.environ.items():
        if k.startswith(profile_prefix):
            remainder = k[len(profile_prefix):]
            parts = remainder.split("_", 1)
            if len(parts) == 2:
                alias, key = parts
                alias = alias.lower()
                key = key.lower()

                if alias not in profiles:
                    profiles[alias] = {}

                if key == "binding_alias":
                    profiles[alias]["binding_alias"] = v.lower()
                elif key == "model_name":
                    profiles[alias]["model_name"] = v
                elif key == "is_default":
                    profiles[alias]["is_default"] = v.lower() in ("true", "1", "yes")
                elif key == "vision_enabled":
                    profiles[alias]["vision_enabled"] = v.lower() in ("true", "1", "yes")
                elif key == "forced_context_size":
                    try:
                        profiles[alias]["forced_context_size"] = int(v)
                    except ValueError:
                        pass
                elif key == "binding_name":
                    profiles[alias]["binding_name"] = v
                elif key.startswith("routing_"):
                    if "routing_profile" not in profiles[alias]:
                        profiles[alias]["routing_profile"] = {}
                    r_key = key[len("routing_"):]
                    if r_key == "cost":
                        try:
                            profiles[alias]["routing_profile"]["cost_per_1k_tokens"] = float(v)
                        except ValueError:
                            pass
                    elif r_key == "latency":
                        try:
                            profiles[alias]["routing_profile"]["avg_latency_ms"] = int(v)
                        except ValueError:
                            pass
                    elif r_key == "complexity":
                        try:
                            profiles[alias]["routing_profile"]["complexity_tier"] = int(v)
                        except ValueError:
                            pass
                    elif r_key == "description":
                        profiles[alias]["routing_profile"]["description"] = v
                else:
                    if "binding_config" not in profiles[alias]:
                        profiles[alias]["binding_config"] = {}
                    profiles[alias]["binding_config"][key] = v

    _RESERVED_PROFILE_KEYS = {
        "binding_alias", "is_default", "vision_enabled", 
        "forced_context_size", "model_name", "binding_name", "routing_profile"
    }

    resolved_profiles = {}
    for p_alias, p_data in profiles.items():
        b_alias = p_data.get("binding_alias")
        if b_alias and b_alias in bindings:
            b_info = bindings[b_alias]
            resolved_profiles[p_alias] = {
                "binding_name": b_info.get("binding_name"),
                "binding_config": {
                    **b_info.get("binding_config", {}),
                    "model_name": p_data.get("model_name", b_info.get("binding_config", {}).get("model_name", ""))
                },
                "is_default": p_data.get("is_default", False),
                "vision_enabled": p_data.get("vision_enabled", False),
                "forced_context_size": p_data.get("forced_context_size"),
                "routing_profile": p_data.get("routing_profile", {})
            }
        elif b_alias is None and "binding_name" in p_data:
            binding_config = {}
            for k, v in p_data.items():
                if k not in _RESERVED_PROFILE_KEYS:
                    binding_config[k] = v
            
            if "model_name" in p_data:
                binding_config["model_name"] = p_data["model_name"]

            resolved_profiles[p_alias] = {
                "binding_name": p_data.get("binding_name"),
                "binding_config": binding_config,
                "is_default": p_data.get("is_default", False),
                "vision_enabled": p_data.get("vision_enabled", False),
                "forced_context_size": p_data.get("forced_context_size"),
                "routing_profile": p_data.get("routing_profile", {})
            }

    return resolved_profiles

def get_client_from_env(
    cli_env_path: Optional[str] = None,
    create_llm: bool = True,
    create_tti: bool = False,
    create_stt: bool = False,
    create_tts: bool = False,
    create_ttm: bool = False,
    create_ttv: bool = False,
    run_wizard_if_fail: bool = True
) -> "LollmsClient":
    from lollms_client import LollmsClient

    def _convert_to_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower().strip() in ("true", "1", "yes", "y")
        return False

    if not auto_resolve_or_wizard(cli_env_path):
        if run_wizard_if_fail:
            ASCIIColors.yellow("⚠️ Configuration auto-resolution failed. Starting wizard...")
            try:
                run_wizard_and_save()
                home_env = Path.home() / ".lollms-client" / ".env"
                if home_env.exists():
                    load_env_file(home_env)
                else:
                    raise ValueError("Wizard did not generate a configuration file.")
            except Exception as e:
                raise ValueError(f"Configuration wizard failed: {e}")
        else:
            raise ValueError("Failed to load or create configuration.")

    kwargs = {}

    binding_types = {
        "llm": create_llm,
        "tti": create_tti,
        "stt": create_stt,
        "tts": create_tts,
        "ttm": create_ttm,
        "ttv": create_ttv
    }

    for b_type, should_create in binding_types.items():
        if not should_create:
            continue

        prefix = b_type.upper() + "_"
        binding_name = os.getenv(prefix + "BINDING_NAME")

        if not binding_name:
            if b_type == "llm":
                if run_wizard_if_fail:
                    ASCIIColors.yellow(f"⚠️ Missing {prefix}BINDING_NAME. Starting wizard...")
                    try:
                        run_wizard_and_save()
                        home_env = Path.home() / ".lollms-client" / ".env"
                        if home_env.exists():
                            load_env_file(home_env)
                            binding_name = os.getenv(prefix + "BINDING_NAME")
                            if not binding_name:
                                raise ValueError(f"Wizard completed but {prefix}BINDING_NAME is still missing.")
                        else:
                            raise ValueError("Wizard did not generate a configuration file.")
                    except Exception as e:
                        raise ValueError(f"Configuration wizard failed: {e}")
                else:
                    raise ValueError(f"Configuration is incomplete. Missing {prefix}BINDING_NAME.")
            else:
                continue

        bindings = _extract_bindings_from_env(prefix)
        profiles = _extract_profiles_from_env(prefix, bindings)
        if profiles:
            kwargs[f"{b_type}_profiles"] = profiles

        binding_config = {
            "model_name": os.getenv(prefix + "MODEL_NAME", "")
        }

        for k, v in os.environ.items():
            if k.startswith(f"{prefix}_PROFILE_") or k in [prefix+"BINDING_NAME", prefix+"MODEL_NAME"]:
                continue

            if k.startswith(prefix):
                key_lower = k[len(prefix):].lower()
                if key_lower == "verify_ssl_certificate":
                    binding_config[key_lower] = _convert_to_bool(v)
                else:
                    binding_config[key_lower] = v

        if b_type == "llm":
            legacy_map = {
                "HOST_ADDRESS": "host_address",
                "SERVICE_KEY": "service_key",
                "VERIFY_SSL_CERTIFICATE": "verify_ssl_certificate",
                "MODEL_NAME": "model_name"
            }
            for legacy_key, config_key in legacy_map.items():
                if config_key not in binding_config or not binding_config[config_key]:
                    val = os.getenv(legacy_key)
                    if val is not None:
                        if config_key == "verify_ssl_certificate":
                            binding_config[config_key] = _convert_to_bool(val)
                        else:
                            binding_config[config_key] = val

        kwargs[f"{b_type}_binding_name"] = binding_name
        kwargs[f"{b_type}_binding_config"] = binding_config

    return LollmsClient(**kwargs)

if __name__ == "__main__":
    run_wizard_and_save()