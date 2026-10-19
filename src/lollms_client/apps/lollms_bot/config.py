import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from ascii_colors import ASCIIColors

DEFAULT_SOUL_CONTENT = """---
name: LollmsBot
author: lollms-client
version: '1.0'
category: assistant
description: The main lollms-bot personality loaded from a handbag.
---

You are LollmsBot, a helpful assistant that lives inside a handbag.
You answer questions, use your tools when appropriate, and deliver any files
you create as attachments on the chat channel where the request arrived.
Keep answers concise and well formatted for chat platforms.
"""

_SKILLS_MODES = ("visible", "loadable", "searchable", "mixed")

_CAPABILITY_KEYS = (
    "enable_code_execution",
    "enable_workspace_tools",
    "enable_skill_loading",
    "enable_skill_creation",
    "enable_sub_agents",
    "enable_model_switching",
    "enable_image_generation",
    "enable_image_editing",
    "enable_tts",
    "enable_stt",
    "enable_ttm",
    "enable_ttv",
    "enable_computer_use",
)

_CAPABILITY_PROMPTS = {
    "enable_code_execution": ("Enable code execution", False),
    "enable_workspace_tools": ("Enable workspace tools", True),
    "enable_skill_loading": ("Enable skill loading", True),
    "enable_skill_creation": ("Enable skill creation", True),
    "enable_sub_agents": ("Enable sub-agents", True),
    "enable_model_switching": ("Enable model switching", True),
    "enable_image_generation": ("Enable image generation", False),
    "enable_image_editing": ("Enable image editing", False),
    "enable_tts": ("Enable text-to-speech", False),
    "enable_stt": ("Enable speech-to-text", False),
    "enable_ttm": ("Enable text-to-music", False),
    "enable_ttv": ("Enable text-to-video", False),
    "enable_computer_use": ("Enable computer use", False),
}

_CHAT_INT_KEYS = {
    "max_reasoning_steps": ("Max reasoning steps per turn", 12),
    "max_tokens_per_turn": ("Max tokens per turn", 4096),
}

_CHAT_FLOAT_KEYS = {
    "temperature": ("Generation temperature", 0.4),
}

_CHAT_BOOL_KEYS = {
    "use_internal_history": ("Use internal conversation history", True),
}


def get_bot_home() -> Path:
    """
    Resolves the lollms-bot home directory.

    Precedence: LOLLMS_BOT_HOME environment variable, then ~/.lollms-bot.
    The directory is created if missing and is the single root for the bot's
    config, handbag, logs, and state.
    """
    env_home = os.getenv("LOLLMS_BOT_HOME")
    if env_home:
        home = Path(env_home).expanduser()
    else:
        home = Path.home() / ".lollms-bot"
    home.mkdir(parents=True, exist_ok=True)
    return home


def ensure_bot_home() -> Path:
    """Creates the lollms-bot home directory and its standard subfolders."""
    home = get_bot_home()
    for subfolder in ("logs", "state"):
        (home / subfolder).mkdir(parents=True, exist_ok=True)
    return home


def _write_default_soul(handbag_path: Path) -> None:
    """Writes the default bot SOUL.md if the handbag has none."""
    soul_path = handbag_path / "SOUL.md"
    if soul_path.exists():
        return
    soul_path.write_text(DEFAULT_SOUL_CONTENT, encoding="utf-8")
    ASCIIColors.green(f"Default bot personality written to: {soul_path}")


def get_or_create_handbag(handbag_path: Optional[Union[str, Path]] = None) -> "Handbag":
    """
    Returns the bot's main handbag, creating the full handbag skeleton
    (SOUL.md, coworkers/, tools/, skills/, rag/, docs/, memory/, assets/,
    workspace/) on first run.
    """
    from lollms_client.lollms_personality.handbag import Handbag

    if handbag_path is not None:
        hb_path = Path(handbag_path).expanduser()
    else:
        hb_path = get_bot_home() / "handbag"

    if not hb_path.exists():
        ASCIIColors.info(f"Creating handbag skeleton at: {hb_path}")
        Handbag.create_structure(hb_path, name="LollmsBot Handbag")
        _write_default_soul(hb_path)

    hb = Handbag(hb_path)
    if not hb.soul_path.exists():
        _write_default_soul(hb.path)
        hb = Handbag(hb.path)
    return hb


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


def _prompt_bool(label: str, default: bool) -> bool:
    default_str = "Y/n" if default else "y/N"
    raw = input(f"{label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        ASCIIColors.yellow(f"Invalid number '{raw}', keeping {default}.")
        return default


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        ASCIIColors.yellow(f"Invalid number '{raw}', keeping {default}.")
        return default


class BotConfig:
    """
    Manages the lollms-bot configuration file living in ~/.lollms-bot/config.yaml.

    The bot does NOT store its own LLM profile: it reuses the global
    LollmsClient configuration (the same one lollms-code uses). This file only
    holds the handbag path, personality capabilities, chat tuning options, and
    platform connection settings (tokens, enable flags).
    """

    def __init__(self, config_path: Optional[str] = None, home: Optional[str] = None):
        if home:
            os.environ["LOLLMS_BOT_HOME"] = str(Path(home).expanduser())

        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = get_bot_home() / "config.yaml"

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
        personality: Dict[str, Any] = {
            "role": "assistant",
            "skills_mode": "loadable",
            "max_tokens_per_turn": 4096,
        }
        for key in _CAPABILITY_KEYS:
            personality[key] = _CAPABILITY_PROMPTS[key][1]

        return {
            "handbag_path": str(get_bot_home() / "handbag"),
            "personality": personality,
            "personas": {
                "enabled": True,
                "dirs": [],
            },
            "chat": {
                "max_reasoning_steps": 12,
                "temperature": 0.4,
                "use_internal_history": True,
            },
            "connections": {
                "telegram": {"enabled": False, "token": ""},
                "discord": {"enabled": False, "token": ""},
                "slack": {"enabled": False, "token": ""},
                "whatsapp": {"enabled": False, "config_path": ""},
                "fastapi": {"enabled": False, "host": "127.0.0.1", "port": 8000},
            },
        }

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)
        ASCIIColors.green(f"Configuration saved to: {self.config_path}")

    @staticmethod
    def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        defaults = BotConfig(Path("/nonexistent"))._default_config()
        for section_key, section_value in defaults.items():
            if section_key == "handbag_path":
                config.setdefault(section_key, section_value)
                continue
            if isinstance(section_value, dict):
                merged_section = config.setdefault(section_key, {})
                if isinstance(merged_section, dict):
                    for k, v in section_value.items():
                        merged_section.setdefault(k, v)
        return config

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

    def _prompt_personality(self) -> None:
        personality = self.config.setdefault("personality", {})
        BotConfig._merge_defaults({"personality": personality})

        ASCIIColors.info("Personality options (loaded from the handbag):")
        personality["role"] = input("Role [assistant]: ").strip() or "assistant"

        skills_mode = input("Skills mode (visible/loadable/searchable/mixed) [loadable]: ").strip() or "loadable"
        if skills_mode not in _SKILLS_MODES:
            ASCIIColors.yellow(f"Invalid skills mode '{skills_mode}', using 'loadable'.")
            skills_mode = "loadable"
        personality["skills_mode"] = skills_mode

        for key in _CAPABILITY_KEYS:
            label, default = _CAPABILITY_PROMPTS[key]
            personality[key] = _prompt_bool(label, bool(personality.get(key, default)))

    def _prompt_chat(self) -> None:
        chat = self.config.setdefault("chat", {})
        BotConfig._merge_defaults({"chat": chat})

        for key, (label, default) in _CHAT_INT_KEYS.items():
            chat[key] = _prompt_int(label, int(chat.get(key, default)))

        for key, (label, default) in _CHAT_FLOAT_KEYS.items():
            chat[key] = _prompt_float(label, float(chat.get(key, default)))

        for key, (label, default) in _CHAT_BOOL_KEYS.items():
            chat[key] = _prompt_bool(label, bool(chat.get(key, default)))

    def run_wizard(self):
        ASCIIColors.info("🛠️  Lollms-Bot Configuration Wizard")

        home = ensure_bot_home()
        ASCIIColors.info(f"Bot home: {home}")

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

        ASCIIColors.info("\n--- Handbag ---")
        handbag_path = self.config.get("handbag_path") or str(home / "handbag")
        handbag_path = input(f"Handbag path [{handbag_path}]: ").strip() or handbag_path
        self.config["handbag_path"] = str(Path(handbag_path).expanduser())
        try:
            get_or_create_handbag(self.config["handbag_path"])
            ASCIIColors.green(f"  ✅ Handbag ready at: {self.config['handbag_path']}")
        except Exception as e:
            ASCIIColors.red(f"  Failed to prepare handbag: {e}")

        ASCIIColors.info("\n--- Personality Options ---")
        self._prompt_personality()

        ASCIIColors.info("\n--- Personas ---")
        personas = self.config.setdefault("personas", {})
        BotConfig._merge_defaults({"personas": personas})
        enabled = _prompt_bool("Enable multi-persona support (/personas, /persona)", bool(personas.get("enabled", True)))
        personas["enabled"] = enabled
        if enabled:
            extra_dirs_raw = input("Extra persona folders to scan (comma-separated, optional): ").strip()
            personas["dirs"] = [d.strip() for d in extra_dirs_raw.split(",") if d.strip()]

        ASCIIColors.info("\n--- Chat Options ---")
        self._prompt_chat()

        ASCIIColors.info("\n--- Connections ---")
        self._prompt_platform("telegram", "Telegram", "Telegram Bot Token", "token")
        self._prompt_platform("discord", "Discord", "Discord Bot Token", "token")
        self._prompt_platform("slack", "Slack", "Slack Bot Token", "token")
        self._prompt_platform("whatsapp", "WhatsApp", "WhatsApp Config Path (optional)", "config_path")
        self._prompt_fastapi()

        self.save()
        ASCIIColors.green("✅ Configuration complete.")