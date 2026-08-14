"""
gui_prefs.py — lollms_code-GUI-specific preferences (agent behavior, paths,
appearance). NOT the LLM/TTI/TTS/etc. connection config — that's owned by
env_config.py and lives in ~/.lollms-client/.env, matching the real wizard
(lollms_config_cli_env.py) so the CLI and GUI share one source of truth.
"""
from __future__ import annotations

import json
import dataclasses
from pathlib import Path

APP_CONFIG_DIR = Path.home() / ".lollms_client" / "lollms_code"
GUI_PREFS_FILE = APP_CONFIG_DIR / "gui_prefs.json"
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"
APP_DEFAULT_HANDBAG_DIR = APP_CONFIG_DIR / "handbags"

SHELL_AUTONOMY_LEVELS = ["safe", "full_access"]
SKILLS_MODES = ["mixed", "loadable", "always_on", "off"]
ACCENT_PRESETS = {
    "LoLLMS Blue": "#2563eb",
    "Terminal Green": "#16a34a",
    "Amber": "#d97706",
    "Violet": "#7c3aed",
    "Rose": "#e11d48",
}


@dataclasses.dataclass
class GuiPrefs:
    # --- Agent behavior (was CLI --max-steps/--temperature/etc.) ---
    temperature: float = 0.3
    max_tokens_per_turn: int = 8192
    max_reasoning_steps: int = 100
    enable_shell_execution: bool = True
    shell_autonomy_level: str = "safe"
    enable_sub_agents: bool = True
    max_sub_agent_depth: int = 2
    max_sub_agents_per_turn: int = 3
    enable_model_switching: bool = False
    enable_skill_creation: bool = True
    enable_skill_loading: bool = True
    enable_memory: bool = True
    skills_mode: str = "mixed"

    # --- Paths ---
    workspace_path: str = str(Path.cwd())
    skills_dir: str = str(APP_DEFAULT_SKILLS_DIR)
    memory_db: str = f"sqlite:///{APP_DEFAULT_MEMORY_DB}"
    handbag_path: str = str(APP_DEFAULT_HANDBAG_DIR / "default_coder")

    # --- Appearance ---
    dark_mode: bool = True
    accent_color: str = ACCENT_PRESETS["LoLLMS Blue"]
    font_family: str = "JetBrains Mono, monospace"
    window_width: int = 1440
    window_height: int = 900
    show_tool_calls: bool = True
    show_workspace_changes: bool = True
    show_skills_activity: bool = True

    @classmethod
    def load(cls) -> "GuiPrefs":
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if GUI_PREFS_FILE.exists():
            try:
                data = json.loads(GUI_PREFS_FILE.read_text(encoding="utf-8"))
                known = {f.name for f in dataclasses.fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in known})
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        GUI_PREFS_FILE.write_text(
            json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )