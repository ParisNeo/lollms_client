"""
config_wizard.py — Interactive configuration wizard for lollms-code.

Triggers at startup when no configuration exists, or when the user
runs `lollms-code --config`. Reads description.yaml from each LLM
binding to present parameters with pre-filled defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ascii_colors import ASCIIColors

_LLM_BINDINGS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "llm_bindings"
)


def _list_llm_bindings() -> List[str]:
    """Returns a sorted list of available LLM binding names."""
    try:
        from lollms_client.lollms_bindings_utils import list_bindings
        result = list_bindings()
        names = []
        for item in result:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("binding_name")
                    or item.get("title", "")
                )
                if name:
                    names.append(str(name))
        valid = [
            n for n in names
            if (_LLM_BINDINGS_DIR / n / "description.yaml").exists()
        ]
        if valid:
            return sorted(set(valid))
    except Exception:
        pass

    try:
        if _LLM_BINDINGS_DIR.exists():
            names = []
            for d in sorted(_LLM_BINDINGS_DIR.iterdir()):
                if d.is_dir() and not d.name.startswith("_") and not d.name.startswith("."):
                    if (d / "description.yaml").exists():
                        names.append(d.name)
            return names
    except Exception:
        pass

    return []


def _get_binding_description(binding_name: str) -> Optional[Dict[str, Any]]:
    """Returns the parsed description.yaml for a binding, or None."""
    try:
        from lollms_client.lollms_bindings_utils import get_binding_desc
        desc = get_binding_desc(binding_name)
        if isinstance(desc, dict):
            return desc
    except Exception:
        pass

    try:
        import yaml
        desc_path = _LLM_BINDINGS_DIR / binding_name / "description.yaml"
        if desc_path.exists():
            with open(desc_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass

    return None


def _convert_value(raw: str, param_type: str) -> Any:
    """Converts a string input to the target type."""
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
    elif param_type in ("list", "array"):
        return [v.strip() for v in raw.split(",") if v.strip()]
    elif param_type in ("dict", "object"):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    else:
        return raw


def _format_default(value: Any, param_type: str) -> str:
    """Formats a default value for display."""
    if value is None:
        return ""
    if param_type == "bool":
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _prompt_param(
    name: str,
    desc: str,
    param_type: str,
    mandatory: bool,
    default: Any,
    current: Any = None,
) -> Any:
    """
    Prompts the user for a single parameter value.
    Returns the chosen value (current, default, or user input).
    """
    display_value = current if current is not None else default

    ASCIIColors.cyan(f"\n  ── {name} ──")
    if desc:
        short_desc = desc if len(desc) <= 120 else desc[:117] + "..."
        ASCIIColors.gray(f"  {short_desc}")

    mandatory_str = " (required)" if mandatory else " (optional)"
    ASCIIColors.gray(f"  Type: {param_type}{mandatory_str}")

    if param_type == "bool":
        default_str = "yes" if display_value else "no"
        prompt_text = f"  Enter (yes/no) [{default_str}]: "
    elif display_value is not None and display_value != "":
        prompt_text = f"  Enter [{_format_default(display_value, param_type)}]: "
    else:
        prompt_text = f"  Enter: "

    try:
        user_input = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        return display_value

    if not user_input:
        if mandatory and display_value is None:
            ASCIIColors.red("  ⚠ This parameter is required. Please enter a value.")
            return _prompt_param(name, desc, param_type, mandatory, default, current)
        return display_value

    return _convert_value(user_input, param_type)


def run_config_wizard(config) -> None:
    """
    Interactive configuration wizard. Modifies the config object in-place.

    Parameters
    ----------
    config : CodeAgentConfig
        The configuration object to populate.
    """
    ASCIIColors.cyan("\n" + "=" * 60)
    ASCIIColors.cyan("🔧 lollms-code Configuration Wizard")
    ASCIIColors.cyan("=" * 60)
    ASCIIColors.white(
        "\n  This wizard will help you configure lollms-code.\n"
        "  Press Enter at any prompt to accept the default value.\n"
        "  Press Ctrl+C at any time to cancel.\n"
    )

    bindings = _list_llm_bindings()

    if not bindings:
        ASCIIColors.red("\n  ❌ No LLM bindings found!")
        ASCIIColors.red("  Please ensure lollms_client is properly installed.")
        return

    ASCIIColors.cyan("\n  Step 1: Select your LLM binding\n")

    current_binding = config.llm_binding if config.llm_binding else None

    for idx, name in enumerate(bindings, 1):
        marker = " ← current" if name == current_binding else ""
        ASCIIColors.white(f"    {idx}. {name}{marker}")

    try:
        choice_input = input(
            f"\n  Enter number or name [{current_binding or bindings[0]}]: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        ASCIIColors.yellow("\n\n  Wizard cancelled.")
        return

    if not choice_input:
        selected = current_binding or bindings[0]
    elif choice_input.isdigit() and 1 <= int(choice_input) <= len(bindings):
        selected = bindings[int(choice_input) - 1]
    elif choice_input in bindings:
        selected = choice_input
    else:
        ASCIIColors.yellow(f"  '{choice_input}' not recognized. Using default.")
        selected = current_binding or bindings[0]

    config.llm_binding = selected
    ASCIIColors.green(f"  ✓ Selected binding: {selected}")

    desc = _get_binding_description(selected)

    if desc:
        binding_title = desc.get("title", selected)
        binding_desc_text = desc.get("description", "")
        ASCIIColors.cyan(f"\n  Step 2: Configure {binding_title}\n")
        if binding_desc_text:
            short = binding_desc_text if len(binding_desc_text) <= 200 else binding_desc_text[:197] + "..."
            ASCIIColors.gray(f"  {short}\n")
    else:
        ASCIIColors.cyan(f"\n  Step 2: Configure {selected}\n")
        ASCIIColors.yellow("  ⚠ No description.yaml found. Using basic configuration.\n")

    binding_config: Dict[str, Any] = {}

    existing_config = config.llm_binding_config if config.llm_binding_config else {}

    global_params = desc.get("global_input_parameters") or desc.get("input_parameters") or [] if desc else []
    model_params = desc.get("model_input_parameters") or [] if desc else []

    if global_params:
        ASCIIColors.cyan("  ── Global Parameters ──")

    for param in global_params:
        pname = param.get("name", "")
        if not pname:
            continue
        pdesc = param.get("description", "")
        ptype = param.get("type", "str")
        pmandatory = param.get("mandatory", False)
        pdefault = param.get("default")

        current_val = existing_config.get(pname)

        value = _prompt_param(pname, pdesc, ptype, pmandatory, pdefault, current_val)
        binding_config[pname] = value

    if model_params:
        ASCIIColors.cyan("\n  ── Model Parameters ──")

    for param in model_params:
        pname = param.get("name", "")
        if not pname:
            continue
        pdesc = param.get("description", "")
        ptype = param.get("type", "str")
        pmandatory = param.get("mandatory", False)
        pdefault = param.get("default")

        current_val = existing_config.get(pname)

        value = _prompt_param(pname, pdesc, ptype, pmandatory, pdefault, current_val)
        binding_config[pname] = value

        if pname == "model_name":
            config.model_name = value

    if not desc:
        ASCIIColors.cyan("\n  ── Basic Configuration ──")

        current_model = config.model_name or existing_config.get("model_name", "")
        model_val = _prompt_param(
            "model_name", "The name of the model to use", "str", True, "", current_model
        )
        binding_config["model_name"] = model_val
        config.model_name = model_val

        current_host = config.host_address or existing_config.get("host_address", "http://localhost:11434")
        host_val = _prompt_param(
            "host_address", "The host address of the LLM server", "str", False,
            "http://localhost:11434", current_host
        )
        binding_config["host_address"] = host_val
        config.host_address = host_val

        current_key = config.api_key or existing_config.get("service_key", "")
        key_val = _prompt_param(
            "service_key", "API key (leave empty if not needed)", "str", False, "", current_key
        )
        if key_val:
            binding_config["service_key"] = key_val
            config.api_key = key_val

        current_ssl = config.verify_ssl if not config.llm_binding_config else existing_config.get("verify_ssl_certificate", False)
        ssl_val = _prompt_param(
            "verify_ssl_certificate", "Verify SSL certificates", "bool", False, False, current_ssl
        )
        binding_config["verify_ssl_certificate"] = ssl_val
        config.verify_ssl = ssl_val

    config.llm_binding_config = binding_config

    if "host_address" in binding_config:
        config.host_address = binding_config["host_address"]
    if "service_key" in binding_config:
        config.api_key = binding_config["service_key"]
    if "verify_ssl_certificate" in binding_config:
        config.verify_ssl = binding_config["verify_ssl_certificate"]

    ASCIIColors.cyan("\n  Step 3: Configure Agent Settings\n")

    config.max_reasoning_steps = _prompt_param(
        "max_reasoning_steps",
        "Maximum reasoning steps for autonomous loops (higher = more autonomy)",
        "int", False, 100, config.max_reasoning_steps
    )

    config.temperature = _prompt_param(
        "temperature",
        "Sampling temperature (0.1 = deterministic, 0.7 = creative)",
        "float", False, 0.3, config.temperature
    )

    config.max_tokens_per_turn = _prompt_param(
        "max_tokens_per_turn",
        "Maximum tokens per generation turn",
        "int", False, 8192, config.max_tokens_per_turn
    )

    config.enable_code_execution = _prompt_param(
        "enable_code_execution",
        "Allow the agent to execute Python code (needed for build->test->fix loops)",
        "bool", False, True, config.enable_code_execution
    )

    config.enable_sub_agents = _prompt_param(
        "enable_sub_agents",
        "Allow the agent to spawn focused sub-agents for complex tasks",
        "bool", False, True, config.enable_sub_agents
    )

    config.enable_model_switching = _prompt_param(
        "enable_model_switching",
        "Allow the agent to switch models mid-task",
        "bool", False, False, config.enable_model_switching
    )

    config.enable_memory = _prompt_param(
        "enable_memory",
        "Enable persistent memory (cross-session learning)",
        "bool", False, True, config.enable_memory
    )

    skills_choices = ["loadable", "always_visible", "mixed"]
    ASCIIColors.cyan("\n  ── skills_mode ──")
    ASCIIColors.gray("  How skills are displayed in the agent's context")
    ASCIIColors.gray("    loadable       = listed by title, loaded on demand")
    ASCIIColors.gray("    always_visible = full content always in context")
    ASCIIColors.gray("    mixed          = visible skills + loadable index")
    current_mode = config.skills_mode or "mixed"
    mode_val = _prompt_param(
        "skills_mode", "", "str", False, "mixed", current_mode
    )
    if mode_val not in skills_choices:
        ASCIIColors.yellow(f"  '{mode_val}' not recognized. Using 'mixed'.")
        mode_val = "mixed"
    config.skills_mode = mode_val

    config.wizard_completed = True

    ASCIIColors.cyan("\n" + "=" * 60)
    ASCIIColors.green("  ✅ Configuration complete!")
    ASCIIColors.cyan("=" * 60)

    ASCIIColors.white(f"\n  Summary:")
    ASCIIColors.white(f"    Binding:      {config.llm_binding}")
    if binding_config.get("model_name"):
        ASCIIColors.white(f"    Model:        {binding_config['model_name']}")
    if binding_config.get("host_address"):
        ASCIIColors.white(f"    Host:         {binding_config['host_address']}")
    ASCIIColors.white(f"    Max steps:    {config.max_reasoning_steps}")
    ASCIIColors.white(f"    Temperature:  {config.temperature}")
    ASCIIColors.white(f"    Memory:      {'enabled' if config.enable_memory else 'disabled'}")
    ASCIIColors.white(f"    Sub-agents:  {'enabled' if config.enable_sub_agents else 'disabled'}")
    ASCIIColors.white(f"    Skills mode: {config.skills_mode}")
    config_file = Path.home() / ".lollms_hub" / "lollms_code" / "config.json"
    ASCIIColors.gray(f"\n  Config will be saved to: {config_file}")


def needs_configuration(config) -> bool:
    """
    Checks if the wizard should run.
    Returns True if configuration is incomplete.
    """
    return not config.wizard_completed or not config.llm_binding_config
```