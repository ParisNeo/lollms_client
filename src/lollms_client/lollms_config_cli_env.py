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

from ascii_colors import ASCIIColors
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
        default_str = "yes" if default else "no"
        answer = questionary.confirm(f"Enter yes/no:", default=default).ask()
        if answer is None:
            return default
        return answer
    else:
        prompt_text = f"Enter value"
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

def _configure_binding(binding_type: str, config_map: Dict[str, str]):
    prefix = binding_type.upper() + "_"
    bindings = _list_bindings_by_type(binding_type)
    if not bindings:
        ASCIIColors.yellow(f"\n  ⚠️ No {binding_type.upper()} bindings found. Skipping.")
        return

    ASCIIColors.panel(f"Configure {binding_type.upper()} Binding", title=f"[bold]{binding_type.upper()} Setup[/bold]", border_style="cyan")
    
    skip_choice = "Skip this binding"
    choices = bindings + [skip_choice]
    selected = questionary.select(
        f"Select {binding_type.upper()} binding:",
        choices=choices
    ).ask()

    if not selected or selected == skip_choice:
        return

    config_map[prefix + "BINDING_NAME"] = selected
    ASCIIColors.green(f"\n  ✓ Selected {binding_type.upper()} binding: {selected}")

    desc = _get_binding_description(selected, binding_type)
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

    ASCIIColors.rich_print("\n[bold cyan]Select Model[/bold cyan]")
    with ASCIIColors.status("[cyan]Fetching available models...[/cyan]", spinner="dots"):
        available_models = _fetch_available_models(binding_type, selected, config_map)
    
    if available_models:
        model_choice = questionary.select(
            f"Select {binding_type.upper()} Model:",
            choices=available_models
        ).ask()
        if model_choice:
            config_map[prefix + "MODEL_NAME"] = model_choice
    else:
        ASCIIColors.yellow("  ⚠️ Could not fetch models automatically.")
        model_name = questionary.text("Enter model name manually:").ask()
        config_map[prefix + "MODEL_NAME"] = model_name or "default"

def run_wizard_and_save():
    ASCIIColors.panel(
        "[bold]Lollms Client Configuration Wizard[/bold]\n[dim]This wizard will help you configure your bindings.[/dim]",
        title="[bold magenta]🧙 Wizard[/bold magenta]",
        border_style="magenta"
    )

    config_map = {}

    for b_type in ["llm", "tti", "tts", "stt", "ttm", "ttv"]:
        _configure_binding(b_type, config_map)

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
            b_name = config_map.get(f"{b_type.upper()}_BINDING_NAME")
            if b_name:
                prefix = b_type.upper() + "_"
                b_config = {
                    "model_name": config_map.get(prefix + "MODEL_NAME", "")
                }
                for k, v in config_map.items():
                    if k.startswith(prefix) and k not in [prefix+"BINDING_NAME", prefix+"MODEL_NAME"]:
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

def _convert_to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower().strip() in ("true", "1", "yes", "y")
    return False

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

        binding_config = {
            "model_name": os.getenv(prefix + "MODEL_NAME", "")
        }
        
        for k, v in os.environ.items():
            if k.startswith(prefix) and k not in [prefix+"BINDING_NAME", prefix+"MODEL_NAME"]:
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