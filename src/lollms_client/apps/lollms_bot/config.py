import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from ascii_colors import ASCIIColors


def _active_client_config() -> Dict[str, Any]:
    """
    Resolves the active LollmsClient configuration using the SAME global
    resolution path as lollms-code (config.yaml + .env + two-tier profiles).
    Returns a small dict describing the active LLM, or an empty dict when the
    client is not configured yet.
    """
    try:
        from lollms_client.lollms_config_cli_env import get_client_from_env
        client = get_client_from_env(
            create_llm=True,
            run_wizard_if_fail=False,
        )
    except Exception as e:
        ASCIIColors.warning(f"Could not resolve the active LollmsClient config: {e}")
        return {}

    llm = getattr(client, "llm", None)
    if llm is None:
        return {}
    return {
        "binding": getattr(llm, "binding_name", None),
        "model": getattr(llm, "model_name", None),
    }


class BotConfig:
    """
    Manages the lollms-bot configuration file.

    The bot does NOT store its own LLM profile: it reuses the global
    LollmsClient configuration (the same one lollms-code uses). This file only
    holds the platform connection settings (tokens, enable flags).
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path.home() / ".lollms_client" / "bot_config.yaml"

        self.config: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return self._default_config()
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or self._default_config()
        except Exception as e:
            ASCIIColors.red(f"Failed to load config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "connections": {
                "telegram": {"enabled": False, "token": ""},
                "discord": {"enabled": False, "token": ""},
                "slack": {"enabled": False, "token": ""},
                "whatsapp": {"enabled": False, "config_path": ""},
                "fastapi": {"enabled": False, "host": "127.0.0.1", "port": 8000},
            }
        }

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
        ASCIIColors.green(f"Configuration saved to: {self.config_path}")

    def _prompt_platform(self, name: str, label: str, credential_label: str, credential_key: str) -> None:
        conn = self.config["connections"].setdefault(name, {})
        enabled = input(f"Enable {label}? [y/N]: ").strip().lower() == "y"
        conn["enabled"] = enabled
        if enabled:
            conn[credential_key] = input(f"{credential_label}: ").strip()

    def _prompt_fastapi(self) -> None:
        conn = self.config["connections"].setdefault("fastapi", {})
        enabled = input("Enable FastAPI HTTP API? [y/N]: ").strip().lower() == "y"
        conn["enabled"] = enabled
        if enabled:
            host = input("Host [127.0.0.1]: ").strip() or "127.0.0.1"
            port_raw = input("Port [8000]: ").strip() or "8000"
            try:
                port = int(port_raw)
            except ValueError:
                ASCIIColors.yellow(f"Invalid port '{port_raw}', defaulting to 8000.")
                port = 8000
            conn["host"] = host
            conn["port"] = port

    def run_wizard(self):
        ASCIIColors.info("🛠️  Lollms-Bot Configuration Wizard")

        ASCIIColors.info("\n--- LLM Configuration ---")
        ASCIIColors.info("The bot reuses your global LollmsClient configuration (the same one lollms-code uses).")
        active = _active_client_config()
        if active.get("binding") or active.get("model"):
            ASCIIColors.green(
                f"  Active LLM → binding: {active.get('binding')}, model: {active.get('model')}"
            )
            reconfig = input("  Reconfigure your LLM now? [y/N]: ").strip().lower() == "y"
            if reconfig:
                try:
                    from lollms_client.lollms_config_cli_env import run_wizard_and_save
                    run_wizard_and_save()
                except Exception as e:
                    ASCIIColors.red(f"LLM wizard failed: {e}")
        else:
            ASCIIColors.yellow("  No LLM configured yet. Run the global wizard first:")
            ASCIIColors.info("     python -m lollms_client.lollms_config_cli_env")

        ASCIIColors.info("\n--- Connections ---")
        self._prompt_platform("telegram", "Telegram", "Telegram Bot Token", "token")
        self._prompt_platform("discord", "Discord", "Discord Bot Token", "token")
        self._prompt_platform("slack", "Slack", "Slack Bot Token", "token")
        self._prompt_platform("whatsapp", "WhatsApp", "WhatsApp Config Path (optional)", "config_path")
        self._prompt_fastapi()

        self.save()
        ASCIIColors.green("✅ Configuration complete.")