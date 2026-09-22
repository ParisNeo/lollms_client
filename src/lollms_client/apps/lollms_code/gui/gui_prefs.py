"""
gui_prefs.py — lollms_code-GUI-specific preferences (agent behavior, paths,
appearance, and project workspace deck).
"""
from __future__ import annotations

import json
import uuid
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_CONFIG_DIR = Path.home() / ".lollms_client" / "lollms_code"
GUI_PREFS_FILE = APP_CONFIG_DIR / "gui_prefs.json"
APP_DEFAULT_SKILLS_DIR = APP_CONFIG_DIR / "skills"
APP_DEFAULT_MEMORY_DB = APP_CONFIG_DIR / "memory.db"
APP_DEFAULT_HANDBAG_DIR = APP_CONFIG_DIR / "handbags"

SHELL_AUTONOMY_LEVELS = ["strict", "safe", "full_access"]
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
    # --- Agent behavior ---
    temperature: float = 0.3
    max_tokens_per_turn: int = 8192
    max_reasoning_steps: int = 100
    enable_shell_execution: bool = True
    shell_autonomy_level: str = "safe"
    auto_approve_python: bool = False
    enable_sub_agents: bool = True
    max_sub_agent_depth: int = 2
    max_sub_agents_per_turn: int = 3
    enable_model_switching: bool = False
    enable_skill_creation: bool = True
    enable_skill_loading: bool = True
    enable_memory: bool = True
    skills_mode: str = "mixed"
    debug: bool = False

    # --- Paths & Workspaces ---
    workspace_path: str = str(Path.cwd())
    skills_dir: str = str(APP_DEFAULT_SKILLS_DIR)
    memory_db: str = f"sqlite:///{APP_DEFAULT_MEMORY_DB}"
    handbag_path: str = str(APP_DEFAULT_HANDBAG_DIR / "default_coder")
    workspaces: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    # --- Appearance & Window ---
    theme_mode: str = "auto"
    theme_day_start_hour: int = 7
    theme_night_start_hour: int = 19
    dark_mode: bool = True
    start_fullscreen: bool = True
    accent_color: str = ACCENT_PRESETS["LoLLMS Blue"]
    font_family: str = "JetBrains Mono, monospace"
    window_width: int = 1440
    window_height: int = 900
    show_tool_calls: bool = True
    show_workspace_changes: bool = True
    show_skills_activity: bool = True
    show_live_sidebar: bool = True

    @classmethod
    def load(cls) -> "GuiPrefs":
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if GUI_PREFS_FILE.exists():
            try:
                data = json.loads(GUI_PREFS_FILE.read_text(encoding="utf-8"))
                known = {f.name for f in dataclasses.fields(cls)}
                inst = cls(**{k: v for k, v in data.items() if k in known})
                # Ephemeral session permissions always start False on fresh app start
                inst.auto_approve_python = False
                inst._ensure_workspaces_seeded()
                return inst
            except Exception:
                pass
        inst = cls()
        inst.auto_approve_python = False
        inst._ensure_workspaces_seeded()
        return inst

    def is_dark(self) -> bool:
        mode = getattr(self, "theme_mode", "auto")
        if mode == "dark":
            return True
        if mode == "light":
            return False
        from datetime import datetime
        hour = datetime.now().hour
        day_start = int(getattr(self, "theme_day_start_hour", 7))
        night_start = int(getattr(self, "theme_night_start_hour", 19))
        if day_start <= night_start:
            return hour < day_start or hour >= night_start
        return night_start <= hour < day_start

    def save(self) -> None:
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.dark_mode = self.is_dark()
        GUI_PREFS_FILE.write_text(
            json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Workspace Deck CRUD ─────────────────────────────────────────────────

    def _ensure_workspaces_seeded(self) -> None:
        """Ensures at least the current workspace path is registered in the deck."""
        if not self.workspaces:
            cur = Path(self.workspace_path).resolve()
            self.workspaces = [{
                "id": f"ws_{uuid.uuid4().hex[:8]}",
                "name": cur.name or "My Workspace",
                "path": str(cur),
                "description": "Default workspace directory",
                "last_opened": "",
            }]

    def get_workspaces(self) -> List[Dict[str, Any]]:
        self._ensure_workspaces_seeded()
        return self.workspaces

    def add_workspace(self, name: str, path: str, description: str = "") -> Dict[str, Any]:
        resolved = str(Path(path).resolve())
        # If already exists, update name and description
        for ws in self.workspaces:
            if Path(ws["path"]).resolve() == Path(resolved).resolve():
                ws["name"] = name.strip() or ws["name"]
                ws["description"] = description.strip() or ws.get("description", "")
                self.save()
                return ws

        entry = {
            "id": f"ws_{uuid.uuid4().hex[:8]}",
            "name": name.strip() or Path(resolved).name or "Workspace",
            "path": resolved,
            "description": description.strip(),
            "last_opened": "",
        }
        self.workspaces.insert(0, entry)
        self.save()
        return entry

    def update_workspace(self, ws_id: str, name: str, path: str, description: str = "") -> bool:
        for ws in self.workspaces:
            if ws.get("id") == ws_id:
                ws["name"] = name.strip() or ws["name"]
                ws["path"] = str(Path(path).resolve())
                ws["description"] = description.strip()
                self.save()
                return True
        return False

    def remove_workspace(self, ws_id: str) -> bool:
        initial = len(self.workspaces)
        self.workspaces = [w for w in self.workspaces if w.get("id") != ws_id]
        if len(self.workspaces) < initial:
            self.save()
            return True
        return False