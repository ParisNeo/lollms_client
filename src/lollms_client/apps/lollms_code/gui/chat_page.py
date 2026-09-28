"""app.chat_page — main agent session UI. Replaces run_interactive()/run_single_prompt()."""
from __future__ import annotations

import queue
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from nicegui import ui

from gui_prefs import GuiPrefs
from env_config import EnvStore
import agent_bridge
try:
    from folder_picker import pick_folder, pick_file
except ImportError:
    try:
        from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder, pick_file
    except ImportError:
        pick_folder = None
        pick_file = None

try:
    from memory_explorer import open_memory_explorer_dialog
except ImportError:
    try:
        from lollms_client.apps.lollms_code.gui.memory_explorer import open_memory_explorer_dialog
    except ImportError:
        open_memory_explorer_dialog = None
from pathlib import Path
import json
from ascii_colors import ASCIIColors


HELP_TEXT = """\
**Commands**

- `/help` — this list
- `/plan` (alias `/current`) — view and edit the active task plan (CURRENT.md)
- `/scratchpad` — view and edit the agent's persistent scratchpad
- `/history` — browse and resend prompt history (Ctrl+H)
- `/clear-history` (alias `/clear`) — clear the conversation shown here (and the agent's in-memory history)
- `/clear-files` (alias `/unload-all`) — unload every currently loaded file from context
- `/load <file1> [file2] ...` — load files into context (`/load all` loads everything indexed)
- `/unload <file1> ...` — remove specific files from context
- `/lock <file1> ...` — lock files (agent can't unlock them)
- `/hide <file1> ...` — hide files from the workspace tree entirely
- `/unhide <file1> ...` — restore hidden files to the tree
- `/skills` — list learned skills
- `/export` — download the full session (including tool calls) as a Markdown file
- `/files` — show which workspace files are currently loaded into context
- `/forget` — permanently wipe the agent's persistent memory (asks to confirm)
- `/workspace <path>` — switch the active workspace directory
- `/config` — open Settings
- `/models` — model switching info

Anything else is sent to the agent as a task.

**Keyboard**: Enter to send · Shift+Enter for newline · ↑/↓ on an empty input to browse
prompt history · Ctrl+K command palette · Ctrl+F search the conversation · Ctrl+/ shortcuts help ·
Ctrl+Shift+C copy the last agent message.
"""

SLASH_COMMANDS = [
("/help", "Show command list"),
("/explorer", "Open the workspace folder in system file explorer"),
("/open", "Open the workspace folder in system file explorer"),
("/dynamic", "Toggle Dynamic Mode on/off (autonomous effort, temperature, tokens)"),
("/resume", "Resume the current incomplete turn from its round checkpoint"),
("/sessions", "Open Sessions Manager (switch, resume, or start new sessions)"),
("/workflow", "Open Workflow Studio (deterministic graph state-machine with hard guards)"),
("/studio", "Open Workflow Studio (deterministic graph state-machine with hard guards)"),
("/plan", "View and edit active macro plan (CURRENT.md)"),
("/current", "View active macro plan (CURRENT.md)"),
("/scratchpad", "View and edit agent scratchpad notes"),
("/zoo", "Open Zoo Package Hub (Tools, Skills, Personas)"),
("/subws", "Open Sub-Workspace Manager (Documentation & Reference files)"),
("/reference", "Open Sub-Workspace Manager (Documentation & Reference files)"),
("/history", "Browse and resend prompt history"),
("/clear-history", "Clear the conversation"),
("/clear-files", "Unload all files from context"),
("/load", "Load file(s) into context (or 'all')"),
("/unload", "Remove file(s) from context"),
("/lock", "Lock file(s) so the agent can't unlock them"),
("/hide", "Hide file(s) from the workspace tree"),
("/unhide", "Restore hidden file(s) to the tree"),
("/skills", "List learned skills"),
("/export", "Download the session as Markdown"),
("/files", "Show loaded context files"),
("/inspect", "Inspect full context and generation parameters sent to LLM (Ctrl+I)"),
("/context", "Inspect full context and generation parameters sent to LLM (Ctrl+I)"),
("/effort", "Configure reasoning effort (none/low/medium/high/dynamic/default)"),
("/memories", "Open interactive Memory Explorer"),
("/memory", "Open Memory Explorer or toggle with /memory on|off|toggle"),
("/forget", "Wipe persistent memory"),
("/workspace", "Switch workspace directory"),
("/config", "Open Settings"),
("/models", "Model switching info"),
]

SHORTCUTS = [
    ("Enter", "Send message"),
    ("Shift+Enter", "New line"),
    ("↑ / ↓ (empty input)", "Browse prompt history"),
    ("Tab", "Accept slash-command suggestion"),
    ("Ctrl+K", "Command palette"),
    ("Ctrl+H", "Open prompt history (browse & resend)"),
    ("Ctrl+F", "Search the conversation"),
    ("Esc", "Close search"),
    ("Ctrl+/", "Show this shortcut list"),
    ("Ctrl+I", "Inspect full context payload & parameters sent to LLM"),
    ("Ctrl+Shift+C", "Copy the last agent message"),
    ("Theme button", "Cycle Auto → Light → Dark (Auto follows the clock)"),
]


class ChatSession:
    def __init__(self, env: EnvStore, prefs: GuiPrefs, session_id: Optional[str] = None):
        self.env = env
        self.prefs = prefs
        self.client = None
        self.personality = None
        self.event_queue: "queue.Queue[agent_bridge.AgentEvent]" = queue.Queue()
        self.busy = False
        self._chunk_buffer = ""
        self.current_round: int = 0
        self.live_skills_count: int = -1
        self.message_counter: int = 0
        self.prompt_history: List[str] = []
        self.history_index: int = -1
        self.turn_start_ts: Optional[float] = None

        # Multi-session persistence attributes
        self.session_id: str = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_title: str = "New Discussion"
        self.debug_log: List[Dict[str, Any]] = []

        self.load_prompt_history()
        if session_id:
            self.load_from_disk(session_id)

    @classmethod
    def get_sessions_dir(cls, workspace_path: str) -> Path:
        s_dir = Path(workspace_path).resolve() / ".lollms_code" / "sessions"
        s_dir.mkdir(parents=True, exist_ok=True)
        return s_dir

    @classmethod
    def list_saved_sessions(cls, workspace_path: str) -> List[Dict[str, Any]]:
        s_dir = cls.get_sessions_dir(workspace_path)
        sessions_meta = []
        for f in sorted(s_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions_meta.append({
                    "id": data.get("session_id", f.stem),
                    "title": data.get("title") or "Untitled Session",
                    "updated_at": data.get("updated_at") or datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "created_at": data.get("created_at") or "",
                    "message_count": len(data.get("debug_log", [])),
                    "path": str(f)
                })
            except Exception:
                pass
        return sessions_meta

    def reconstruct_conversation_from_debug_log(self) -> List[Dict[str, str]]:
        """Synthesizes valid user/assistant conversation history from debug_log entries."""
        reconstructed = []
        current_user = None
        current_assistant_parts = []

        for entry in self.debug_log:
            k = entry.get("type")
            if k == "user":
                if current_user is not None:
                    asst_txt = "\n\n".join(p for p in current_assistant_parts if p.strip()).strip()
                    if not asst_txt:
                        asst_txt = "[Turn interrupted before final answer]"
                    reconstructed.append({"role": "user", "content": current_user})
                    reconstructed.append({"role": "assistant", "content": asst_txt})
                    current_assistant_parts = []
                current_user = entry.get("text", "")
            elif k == "agent":
                t = entry.get("text", "").strip()
                if t:
                    current_assistant_parts.append(t)
            elif k == "event":
                title = entry.get("title", "")
                body = entry.get("body", "")
                if "Saved:" in title or "Patched:" in title or "Finished:" in title:
                    current_assistant_parts.append(f"[{title}]")

        if current_user is not None:
            asst_txt = "\n\n".join(p for p in current_assistant_parts if p.strip()).strip()
            reconstructed.append({"role": "user", "content": current_user})
            if asst_txt:
                reconstructed.append({"role": "assistant", "content": asst_txt})

        return reconstructed

    def get_last_user_prompt(self) -> Optional[str]:
        for entry in reversed(self.debug_log):
            if entry.get("type") == "user":
                t = entry.get("text", "").strip()
                if t and not t.startswith("/"):
                    return t
        return None

    def has_incomplete_turn(self) -> bool:
        if self.busy or not self.debug_log:
            return False
        # Check if the last entry is a user message without completed agent response
        last_user_idx = -1
        last_agent_idx = -1
        for idx, entry in enumerate(self.debug_log):
            if entry.get("type") == "user":
                last_user_idx = idx
            elif entry.get("type") == "agent" and entry.get("text", "").strip():
                last_agent_idx = idx

        if last_user_idx != -1 and last_agent_idx < last_user_idx:
            return True

        # Check turn checkpoint on disk
        p_chk = Path(self.prefs.workspace_path).resolve() / ".lollms_code" / "turn_checkpoint.json"
        if p_chk.exists():
            try:
                c_data = json.loads(p_chk.read_text(encoding="utf-8"))
                if c_data.get("status") in ("in_progress", "cancelled"):
                    return True
            except Exception:
                pass
        return False

    def save_to_disk(self) -> None:
        try:
            s_dir = self.get_sessions_dir(self.prefs.workspace_path)
            f_path = s_dir / f"{self.session_id}.json"
            conv = []
            if self.personality and hasattr(self.personality, "_conversation") and self.personality._conversation:
                conv = self.personality._conversation
            else:
                conv = self.reconstruct_conversation_from_debug_log()

            if self.session_title == "New Discussion":
                for entry in self.debug_log:
                    if entry.get("type") == "user":
                        t = entry.get("text", "").strip()
                        if t and not t.startswith("/"):
                            self.session_title = t[:60] + ("..." if len(t) > 60 else "")
                            break

            payload = {
                "session_id": self.session_id,
                "title": self.session_title,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "workspace_path": self.prefs.workspace_path,
                "debug_log": self.debug_log,
                "conversation": conv,
            }
            f_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def load_from_disk(self, target_session_id: str) -> bool:
        try:
            s_dir = self.get_sessions_dir(self.prefs.workspace_path)
            f_path = s_dir / f"{target_session_id}.json"
            if not f_path.exists():
                return False
            data = json.loads(f_path.read_text(encoding="utf-8"))
            self.session_id = data.get("session_id", target_session_id)
            self.session_title = data.get("title", "Saved Discussion")
            self.debug_log = data.get("debug_log", [])

            self.ensure_ready()
            conv = data.get("conversation", [])
            if not conv:
                conv = self.reconstruct_conversation_from_debug_log()

            if self.personality:
                self.personality._conversation = conv
            return True
        except Exception:
            return False

    def get_prompt_history_path(self) -> Path:
        return Path(self.prefs.workspace_path).resolve() / ".lollms_code" / "prompt_history.json"

    def load_prompt_history(self) -> None:
        p = self.get_prompt_history_path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.prompt_history = [str(x) for x in data if str(x).strip()]
            except Exception:
                self.prompt_history = []
        else:
            self.prompt_history = []
        self.history_index = -1

    def save_prompt_history(self) -> None:
        p = self.get_prompt_history_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.prompt_history, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def ensure_ready(self):
        if self.client is None:
            self.client = agent_bridge.create_client(self.env, self.prefs)
        if self.personality is None:
            self.personality = agent_bridge.create_personality(self.prefs, self.client)


def build_chat_page(env: EnvStore, prefs: GuiPrefs, tools_toggle=None, session: Optional[ChatSession] = None) -> None:
    if session is None:
        session = ChatSession(env, prefs)
    # Sync with persistent session log
    debug_log: List[Dict[str, Any]] = session.debug_log
    message_refs: Dict[str, Dict[str, Any]] = {}
    last_agent_state: Dict[str, Any] = {"entry": None}
    _msg_counter = 0

    def next_message_id() -> int:
        nonlocal _msg_counter
        _msg_counter += 1
        return _msg_counter

    show_tree_sidebar = True

    # ── Theme State (Directly driven by GuiPrefs.is_dark()) ─────────────────
    dark_mode = ui.dark_mode(value=prefs.is_dark())
    THEME_MODES = ("auto", "light", "dark")
    THEME_ICONS = {"auto": "brightness_auto", "light": "light_mode", "dark": "dark_mode"}

    # Dual-mode Tailwind tokens with verified high contrast in both themes.
    SURFACE = "bg-slate-100 dark:bg-slate-900"
    SURFACE_ALT = "bg-slate-100/80 dark:bg-slate-900/80"
    CANVAS = "bg-slate-50 dark:bg-slate-950"
    BORDER = "border-slate-300 dark:border-slate-800"
    MUTED = "text-slate-700 dark:text-slate-300"
    MUTED_DIM = "text-slate-600 dark:text-slate-400"
    STRONG = "text-slate-900 dark:text-slate-100"

    # UI Element Forward References
    prompt_input = None
    transcript = None
    scroll_area = None
    input_counter = None
    theme_btn = None
    thinking_indicator = None
    thinking_label = None

    # Smart Scroll State: auto-follows stream ONLY when at the bottom
    scroll_state = {"auto_follow": True}

    def open_workspace_in_explorer(target_path: Optional[str] = None):
        """Cross-platform launcher to open a directory in the OS default file explorer."""
        import os
        import subprocess
        import sys
        try:
            p = Path(target_path or prefs.workspace_path).resolve()
            if not p.exists():
                ui.notify(f"Path does not exist: {p}", type="warning")
                return
            if sys.platform.startswith("win"):
                os.startfile(str(p))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
            ui.notify(f"Opened in file explorer: {p.name}", type="info", timeout=1500)
        except Exception as ex:
            notify_error(f"Could not open file explorer: {ex}")

    # ── Core Action & Dialog Helpers (Defined First to Avoid UnboundLocalError) ─
    def set_prompt_input(text_to_set: str):
        if prompt_input is None:
            return
        clean = re.sub(r"^🔁 _Rerun:_\s*", "", text_to_set).strip()
        prompt_input.value = clean
        prompt_input.run_method("focus")
        _update_input_counter()
        ui.notify("Copied to prompt input.", type="info", timeout=1200)

    def _theme_tooltip() -> str:
        mode = getattr(prefs, "theme_mode", "auto")
        if mode == "auto":
            is_d = prefs.is_dark()
            return f"Theme: Auto ({'Dark' if is_d else 'Light'} at night/day) — Click for Light"
        elif mode == "light":
            return "Theme: Light — Click for Dark"
        else:
            return "Theme: Dark — Click for Auto"

    def _sync_theme_button():
        if theme_btn is None:
            return
        try:
            mode = getattr(prefs, "theme_mode", "auto")
            theme_btn._props["icon"] = THEME_ICONS.get(mode, "brightness_auto")
            theme_btn._props["title"] = _theme_tooltip()
            theme_btn.update()
        except Exception:
            pass

    def toggle_theme():
        modes = list(THEME_MODES)
        curr = getattr(prefs, "theme_mode", "auto")
        next_mode = modes[(modes.index(curr) + 1) % len(modes)] if curr in modes else "dark"
        prefs.theme_mode = next_mode
        want_dark = prefs.is_dark()
        prefs.dark_mode = want_dark
        dark_mode.set_value(want_dark)
        try:
            prefs.save()
        except Exception:
            pass
        _sync_theme_button()
        label = next_mode.capitalize()
        if next_mode == "auto":
            label += f" ({'Dark' if want_dark else 'Light'} right now)"
        ui.notify(f"Theme: {label}", type="info", timeout=1500)

    def show_thinking_indicator(message: str = "Thinking…"):
        nonlocal thinking_indicator, thinking_label
        if transcript is None:
            return
        if thinking_indicator is not None:
            if thinking_label is not None:
                thinking_label.set_text(message)
            return

        with transcript:
            thinking_indicator = ui.row().classes(
                f"w-fit items-center gap-2.5 px-4 py-2.5 rounded-xl border {BORDER} "
                f"{SURFACE} text-slate-800 dark:text-slate-200 shadow-sm transition-all select-none"
            )
            with thinking_indicator:
                ui.icon("psychology", size="18px").classes("text-primary shrink-0 animate-pulse")
                thinking_label = ui.label(message).classes(f"text-xs font-mono {MUTED}")
                with ui.row().classes("items-center gap-1 shrink-0 ml-1"):
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: -0.32s; animation-duration: 1.1s;"
                    )
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: -0.16s; animation-duration: 1.1s;"
                    )
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: 0s; animation-duration: 1.1s;"
                    )
        if scroll_area and scroll_state.get("auto_follow", True):
            scroll_area.scroll_to(percent=1.0)

    def hide_thinking_indicator():
        nonlocal thinking_indicator, thinking_label
        if thinking_indicator is not None:
            try:
                thinking_indicator.delete()
            except Exception:
                pass
            thinking_indicator = None
            thinking_label = None

    def open_history_dialog():
        dialog = ui.dialog()
        with dialog, ui.card().classes(
            f"w-[680px] max-w-[95vw] h-[580px] max-h-[90vh] flex flex-col p-4 gap-3 "
            f"{CANVAS} text-slate-900 dark:text-slate-100 rounded-xl shadow-2xl border {BORDER}"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("history", size="24px").classes("text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label("Prompt History").classes("text-base font-bold")
                        ui.label("Browse, reuse, or resend messages previously sent to the agent.").classes(
                            f"text-xs {MUTED_DIM}"
                        )
                ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

            search_bar = ui.input(placeholder="Search previous prompts…").classes(
                "w-full text-xs"
            ).props("dense outlined clearable input-debounce=100")

            scroll = ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2")
            with scroll:
                history_list = ui.column().classes("w-full gap-2")

            def refresh_history_items():
                history_list.clear()
                q = (search_bar.value or "").strip().lower()
                entries = list(reversed(session.prompt_history))
                if q:
                    entries = [e for e in entries if q in e.lower()]

                with history_list:
                    if not entries:
                        ui.label("No history matching your search." if q else "No prompt history recorded yet.").classes(
                            f"text-xs {MUTED_DIM} italic p-4 text-center w-full"
                        )
                        return

                    for prompt_text in entries:
                        with ui.card().classes(
                            f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} hover:border-primary/50 transition-colors gap-1.5 shadow-none"
                        ):
                            ui.label(prompt_text).classes(
                                "text-xs font-mono break-words whitespace-pre-wrap max-h-24 overflow-hidden select-all"
                            )
                            with ui.row().classes("w-full items-center justify-end gap-1.5 pt-1"):
                                def _use_in_input(p=prompt_text):
                                    set_prompt_input(p)
                                    dialog.close()

                                async def _resend_now(p=prompt_text):
                                    dialog.close()
                                    if session.busy:
                                        ui.notify("Agent is busy — wait for current turn to finish.", type="warning")
                                        return
                                    await send_prompt_with_text(p)

                                ui.button("Use in input", icon="edit_note", on_click=_use_in_input).props(
                                    "flat dense size=xs no-caps text-color=primary font-semibold"
                                ).tooltip("Place into textarea to edit before sending")

                                ui.button("Resend", icon="send", on_click=_resend_now).props(
                                    "unelevated dense size=xs color=primary no-caps font-semibold"
                                ).tooltip("Send this prompt immediately")

            search_bar.on_value_change(lambda _: refresh_history_items())
            refresh_history_items()

            with ui.row().classes(f"w-full items-center justify-between pt-2 border-t {BORDER}"):
                def _clear_all_history():
                    session.prompt_history.clear()
                    session.save_prompt_history()
                    refresh_history_items()
                    ui.notify("Prompt history cleared.", type="positive")

                ui.button("Clear History", icon="delete_sweep", on_click=_clear_all_history).props(
                    "flat dense size=xs color=red no-caps"
                ).tooltip("Delete all saved prompt history from this workspace")

                ui.button("Close", on_click=dialog.close).props("flat dense size=sm no-caps")

        dialog.open()

    with ui.column().classes("w-full h-full flex-1 min-h-0 flex-nowrap gap-0 overflow-hidden flex flex-col"):
        # ---- Slim status strip (replaces the old sidebar cards) ----
        with ui.row().classes(
            f"w-full items-center justify-between px-3 py-1.5 shrink-0 {SURFACE} border-b {BORDER}"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.button(
                    "Projects", icon="view_carousel",
                    on_click=lambda: (session.save_to_disk(), ui.navigate.to("/")),
                ).props("flat dense size=sm no-caps text-color=primary font-semibold").tooltip("Return to Workspace Deck")
                tree_toggle_btn = ui.button(
                    "Tree", icon="folder",
                    on_click=lambda: toggle_tree_visibility(),
                ).props("flat dense size=sm no-caps text-color=primary").tooltip("Toggle Workspace Tree")
                ui.button(
                    "Sessions", icon="history_edu",
                    on_click=lambda: open_sessions_dialog(),
                ).props("flat dense size=sm no-caps text-color=primary font-semibold").tooltip("Manage and resume sessions in this workspace")
                
                # ---- Dynamic Mode Quick Toggle ----
                def _toggle_dynamic_mode():
                    is_active = getattr(prefs, "dynamic_effort", False)
                    new_state = not is_active
                    prefs.dynamic_effort = new_state
                    prefs.auto_temperature = new_state
                    prefs.auto_max_tokens = new_state
                    try:
                        prefs.save()
                    except Exception:
                        pass

                    if session.personality:
                        session.personality.dynamic_effort = new_state

                    effort_select.value = "dynamic" if new_state else "default"
                    _sync_dynamic_button()
                    ui.notify(
                        f"⚡ Dynamic Mode {'ACTIVATED' if new_state else 'DEACTIVATED'} (Auto effort, temperature, tokens).",
                        type="positive" if new_state else "info"
                    )

                def _sync_dynamic_button():
                    is_active = getattr(prefs, "dynamic_effort", False)
                    dynamic_btn.text = "⚡ Dynamic: ON" if is_active else "⚡ Dynamic: OFF"
                    dynamic_btn._props["color"] = "amber" if is_active else "grey"
                    dynamic_btn._props["text-color"] = "black" if is_active else "white"
                    dynamic_btn.update()

                dynamic_btn = ui.button(
                    "⚡ Dynamic: ON" if getattr(prefs, "dynamic_effort", False) else "⚡ Dynamic: OFF",
                    icon="bolt",
                    on_click=_toggle_dynamic_mode,
                ).props(
                    f"unelevated dense size=sm no-caps font-bold "
                    + (f"color=amber text-color=black" if getattr(prefs, "dynamic_effort", False) else "color=grey text-color=white")
                ).tooltip("Toggle Dynamic Mode: Autonomous reasoning effort scaling, auto temperature, and auto tokens")

                resume_turn_btn = ui.button(
                    "Resume Turn", icon="play_arrow",
                    on_click=lambda: resume_active_turn(),
                ).props("unelevated dense size=sm no-caps color=emerald text-color=white font-bold shadow-sm")
                resume_turn_btn.tooltip("Resume an interrupted or paused turn from its round checkpoint")
                resume_turn_btn.set_visibility(False)
                status_label = ui.label("Idle").classes(f"text-xs {MUTED} font-mono font-medium")
                elapsed_label = ui.label("").classes(f"text-xs {MUTED_DIM} font-mono")

            
            with ui.row().classes("items-center gap-2"):
                rounds_label = ui.label("").classes(f"text-xs {MUTED} font-mono font-medium")
                ctx_label = ui.label("").classes(f"text-xs {MUTED} font-mono font-medium")

                # ---- Fast Effort Selector ----
                effort_options = {
                    "default": "⚡ Effort: Default",
                    "none": "⚡ Effort: Off",
                    "low": "🧠 Effort: Low",
                    "medium": "🧠 Effort: Med",
                    "high": "🧠 Effort: High",
                    "dynamic": "🔄 Effort: Dynamic",
                }
                curr_effort_val = "dynamic" if getattr(prefs, "dynamic_effort", False) else (getattr(prefs, "reasoning_effort", None) or "default")

                def _on_fast_effort_change(e):
                    val = e.value
                    if val == "dynamic":
                        prefs.dynamic_effort = True
                        prefs.reasoning_effort = None
                        ui.notify("Reasoning effort: Dynamic (Auto-adjusting via <effort> tags).", type="positive")
                    elif val == "default":
                        prefs.dynamic_effort = False
                        prefs.reasoning_effort = None
                        ui.notify("Reasoning effort: Model Default.", type="info")
                    else:
                        prefs.dynamic_effort = False
                        prefs.reasoning_effort = val
                        ui.notify(f"Reasoning effort: {val.capitalize()}.", type="positive")
                    try:
                        prefs.save()
                    except Exception:
                        pass

                effort_select = ui.select(
                    effort_options,
                    value=curr_effort_val,
                ).props("dense options-dense outlined size=sm").classes(
                    f"text-xs w-36 bg-slate-50 dark:bg-slate-900 {STRONG}"
                ).tooltip("Fast reasoning effort selector: switch between Off, Low, Med, High, and Dynamic Auto-escalation")
                effort_select.on_value_change(_on_fast_effort_change)

                if tools_toggle is None:
                    tools_toggle = ui.switch("Tool panels", value=prefs.show_tool_calls).props("dense")
                ui.button(
                    "New", icon="add_comment",
                    on_click=lambda: confirm_new_session(),
                ).props("flat dense size=sm no-caps").tooltip("Start a new session (clears the visible conversation)")
                ui.button(
                    icon="search",
                    on_click=lambda: open_search(),
                ).props("flat dense round size=sm").tooltip("Search this conversation (Ctrl+F)")
                theme_btn = ui.button(
                    icon=THEME_ICONS.get(getattr(prefs, "theme_mode", "auto"), "brightness_auto"),
                    on_click=lambda: toggle_theme(),
                ).props("flat dense round size=sm").tooltip(_theme_tooltip())
                ui.button(
                    icon="keyboard",
                    on_click=lambda: open_shortcuts_dialog(),
                ).props("flat dense round size=sm").tooltip("Keyboard shortcuts (Ctrl+/)")
                def _toggle_mem_quick():
                    prefs.enable_memory = not prefs.enable_memory
                    prefs.save()
                    session.personality = agent_bridge.create_personality(prefs, session.client)
                    mem_toggle_btn._props["text-color"] = "purple" if prefs.enable_memory else "grey"
                    mem_toggle_btn.text = "Memories: ON" if prefs.enable_memory else "Memories: OFF"
                    mem_toggle_btn.update()
                    ui.notify(f"Memory is now {'ENABLED' if prefs.enable_memory else 'DISABLED'}.", type="positive" if prefs.enable_memory else "info")

                mem_toggle_btn = ui.button(
                    "Memories: ON" if prefs.enable_memory else "Memories: OFF", icon="psychology",
                    on_click=lambda: open_memory_explorer_dialog(session, prefs) if open_memory_explorer_dialog else ui.notify("Memory Explorer not available", type="warning"),
                ).props(f"flat dense size=sm no-caps text-color={'purple' if prefs.enable_memory else 'grey'}").tooltip("Open Memory Explorer (or click to inspect; toggle via /memory on|off)")

                ui.button(
                    "Workflows", icon="account_tree",
                    on_click=lambda: open_workflow_studio_dialog(),
                ).props("flat dense size=sm no-caps text-color=indigo font-semibold").tooltip("Open Workflow Studio: Graph-based execution harness with hard constraints and per-subtask model selection")
                ui.button(
                    "Inspect Context", icon="manage_search",
                    on_click=lambda: open_context_inspector_dialog(),
                ).props("flat dense size=sm no-caps text-color=cyan font-semibold").tooltip("Inspect full context, prompt messages, and parameters sent to the agent (Ctrl+I)")
                ui.button(
                    "Zoo Hub", icon="pets",
                    on_click=lambda: open_zoo_dialog(),
                ).props("flat dense size=sm no-caps text-color=amber font-semibold").tooltip("Open Zoo Hub: install & manage tools, skills, and personas from GitHub")
                ui.button(
                    "Reference", icon="auto_stories",
                    on_click=lambda: open_sub_workspace_dialog(),
                ).props("flat dense size=sm no-caps text-color=emerald font-semibold").tooltip("Open Sub-Workspace (Documentation & Reference files in .lollms_code/sub_workspace)")
                ui.button(
                    "Plan", icon="checklist",
                    on_click=lambda: open_current_plan_dialog(),
                ).props("flat dense size=sm no-caps text-color=primary font-semibold").tooltip("View and edit the active task roadmap (.lollms_code/CURRENT.md)")
                ui.button(
                    "Scratchpad", icon="edit_note",
                    on_click=lambda: open_scratchpad_dialog(),
                ).props("flat dense size=sm no-caps").tooltip("View the agent's persistent scratchpad notes")
                ui.button(
                    "Copy as Markdown", icon="content_copy",
                    on_click=lambda: copy_debug_markdown(),
                ).props("flat dense size=sm no-caps").tooltip("Copy the full discussion, including tool calls, for debugging")
                ui.button(
                    "Export History", icon="download",
                    on_click=lambda: export_history(),
                ).props("flat dense size=sm no-caps").tooltip("Download the full session as a Markdown file")
                ui.button("Settings", icon="settings", on_click=lambda: (session.save_to_disk(), ui.navigate.to("/settings"))).props(
                    "flat dense size=sm no-caps"
                )

        # ---- Transcript search bar (hidden until Ctrl+F / search icon) ----
        search_row = ui.row().classes(
            f"w-full items-center gap-2 px-3 py-1.5 shrink-0 {SURFACE} border-b {BORDER}"
        )
        search_row.visible = False
        with search_row:
            ui.icon("search").classes(MUTED_DIM)
            search_input = ui.input(placeholder="Search conversation…").classes(
                "flex-1 bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 text-xs"
            ).props(':dark="Quasar.Dark.isActive" dense outlined input-debounce=100')
            search_count_label = ui.label("").classes(f"text-xs {MUTED_DIM} font-mono")
            ui.button(icon="close", on_click=lambda: close_search()).props("flat dense round size=xs")

        # ---- Center Workspace Area (Tree View + Chat Transcript + Telemetry) ----
        with ui.row().classes("w-full flex-1 min-h-0 items-stretch overflow-hidden flex-nowrap gap-0"):

            # ---- Left-hand Lazy Workspace Tree Panel ----
            tree_panel = ui.column().classes(
                f"w-72 h-full shrink-0 border-r {BORDER} {SURFACE_ALT} overflow-hidden flex flex-col p-0 gap-0"
            )
            with tree_panel:
                with ui.tabs().classes(f"w-full {SURFACE} border-b {BORDER} shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as sidebar_tabs:
                    tab_ws = ui.tab('workspace', label='Workspace', icon='folder').classes('text-xs py-1.5 flex-1')
                    with ui.tab('subws', label='Sub-WS', icon='auto_stories').classes('text-xs py-1.5 flex-1') as tab_subws:
                        subws_tab_badge = ui.badge("0", color="emerald").props("floating dense").classes("text-[9px]")
                        subws_tab_badge.visible = False

                with ui.tab_panels(sidebar_tabs, value='workspace').classes('w-full flex-1 min-h-0 p-0 bg-transparent flex flex-col overflow-hidden'):
                    # --- Tab 1: Project Workspace Tree ---
                    with ui.tab_panel('workspace').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-0'):
                        with ui.row().classes(f"w-full items-center justify-between px-3 py-2 border-b {BORDER} shrink-0"):
                            ui.label("📁 Project Root").classes(f"text-xs font-bold {STRONG}")
                            with ui.row().classes("gap-1 items-center"):
                                ui.button(icon="folder_open", on_click=lambda: open_workspace_in_explorer()).props(
                                    "flat round dense size=xs text-color=primary"
                                ).tooltip("Open workspace in system file explorer (Explorer / Finder)")
                                ui.button(icon="upload_file", on_click=lambda: upload_dialog.open()).props(
                                    "flat round dense size=xs"
                                ).tooltip("Upload a file into the workspace root")
                                ui.button(icon="refresh", on_click=lambda: refresh_workspace_tree()).props(
                                    "flat round dense size=xs"
                                ).tooltip("Refresh workspace tree")

                        with ui.row().classes("w-full px-2 pt-2 shrink-0"):
                            tree_search_input = ui.input(placeholder="Filter files…").props(
                                ':dark="Quasar.Dark.isActive" dense outlined clearable'
                            ).classes("w-full bg-slate-50 dark:bg-slate-900 text-xs text-slate-900 dark:text-slate-100")

                        tree_scroll = ui.scroll_area().classes("w-full flex-1 p-2")
                        with tree_scroll:
                            tree_container = ui.column().classes("w-full gap-0 p-0")

                    # --- Tab 2: Sub-Workspace Reference Tree (.lollms_code/sub_workspace/) ---
                    with ui.tab_panel('subws').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-0'):
                        with ui.row().classes(f"w-full items-center justify-between px-3 py-2 border-b {BORDER} shrink-0"):
                            with ui.row().classes("items-center gap-1.5"):
                                ui.label("📚 Sub-Workspace").classes(f"text-xs font-bold {STRONG}")
                                ui.button(icon="folder_open", on_click=lambda: open_workspace_in_explorer(str(Path(prefs.workspace_path).resolve() / ".lollms_code" / "sub_workspace"))).props(
                                    "flat round dense size=xs text-color=emerald"
                                ).tooltip("Open sub_workspace in system file explorer")
                            with ui.row().classes("gap-0.5"):
                                async def _import_subws_file_sidebar():
                                    _picker = pick_file
                                    if _picker:
                                        chosen = await _picker(
                                            title="Select Reference File to Import",
                                            file_types=[("All Files", "*.*")]
                                        )
                                        if chosen:
                                            try:
                                                p = Path(chosen)
                                                from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
                                                sub_ws = SubWorkspaceManager(prefs.workspace_path)
                                                if p.is_dir():
                                                    imported = sub_ws.import_folder(p)
                                                    ui.notify(f"Imported folder with {len(imported)} reference file(s).", type="positive")
                                                else:
                                                    dest = sub_ws.import_file(p)
                                                    ui.notify(f"Imported reference: {dest.name}", type="positive")
                                                refresh_subws_tree()
                                            except Exception as ex:
                                                notify_error(f"Import failed: {ex}")

                                async def _import_subws_folder_sidebar():
                                    _picker = pick_folder
                                    if _picker:
                                        chosen = await _picker(title="Select Folder to Import into Sub-Workspace")
                                        if chosen:
                                            try:
                                                from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
                                                sub_ws = SubWorkspaceManager(prefs.workspace_path)
                                                imported = sub_ws.import_folder(chosen)
                                                ui.notify(f"Imported {len(imported)} reference files.", type="positive")
                                                refresh_subws_tree()
                                            except Exception as ex:
                                                notify_error(f"Import folder failed: {ex}")

                                def _load_all_subws_sidebar():
                                    from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
                                    sub_ws = SubWorkspaceManager(prefs.workspace_path)
                                    cnt = sub_ws.load_all()
                                    ui.notify(f"Loaded all {cnt} reference file(s) [C]", type="positive")
                                    refresh_subws_tree()

                                def _unload_all_subws_sidebar():
                                    from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
                                    sub_ws = SubWorkspaceManager(prefs.workspace_path)
                                    sub_ws.unload_all()
                                    ui.notify("Unloaded all reference files [U]", type="info")
                                    refresh_subws_tree()

                                ui.button(icon="note_add", on_click=lambda: open_paste_reference_dialog()).props(
                                    "flat round dense size=xs color=primary"
                                ).tooltip("Paste text as a new reference file")
                                ui.button(icon="upload_file", on_click=_import_subws_file_sidebar).props(
                                    "flat round dense size=xs"
                                ).tooltip("Import reference file into .lollms_code/sub_workspace")
                                ui.button(icon="drive_folder_upload", on_click=_import_subws_folder_sidebar).props(
                                    "flat round dense size=xs"
                                ).tooltip("Import external folder into sub-workspace")
                                ui.button(icon="download", on_click=_load_all_subws_sidebar).props(
                                    "flat round dense size=xs color=emerald"
                                ).tooltip("Load all reference files into context [C]")
                                ui.button(icon="clear_all", on_click=_unload_all_subws_sidebar).props(
                                    "flat round dense size=xs color=amber"
                                ).tooltip("Unload all reference files from context [U]")
                                ui.button(icon="refresh", on_click=lambda: refresh_subws_tree()).props(
                                    "flat round dense size=xs"
                                ).tooltip("Refresh sub-workspace tree")

                        with ui.row().classes("w-full px-2 pt-2 shrink-0"):
                            subws_tree_search_input = ui.input(placeholder="Filter reference files…").props(
                                ':dark="Quasar.Dark.isActive" dense outlined clearable'
                            ).classes("w-full bg-slate-50 dark:bg-slate-900 text-xs text-slate-900 dark:text-slate-100")

                        subws_tree_scroll = ui.scroll_area().classes("w-full flex-1 p-2")
                        with subws_tree_scroll:
                            subws_tree_container = ui.column().classes("w-full gap-0 p-0")

            # ---- Transcript with Smart Auto-Scroll ----
            scroll_area = ui.scroll_area().classes(f"flex-1 h-full min-w-0 {CANVAS} relative")

            def handle_transcript_scroll(e):
                """Tracks whether user has manually scrolled away from bottom."""
                try:
                    v_pos = getattr(e, "vertical_position", None)
                    v_size = getattr(e, "vertical_size", None)
                    c_size = getattr(e, "vertical_container_size", None)

                    if v_pos is not None and v_size is not None and c_size is not None:
                        # User is at bottom if viewport position + container height is near total content height
                        at_bottom = (v_pos + c_size) >= (v_size - 60)
                        scroll_state["auto_follow"] = at_bottom
                    elif hasattr(e, "vertical_percentage") and e.vertical_percentage is not None:
                        scroll_state["auto_follow"] = e.vertical_percentage >= 0.95
                except Exception:
                    pass

            scroll_area.on_scroll(handle_transcript_scroll)

            with scroll_area:
                transcript = ui.column().classes("w-full gap-3 p-4 pb-16")

            # ---- Right-hand live panels ----
            live_sidebar = ui.column().classes(
                f"w-64 h-full shrink-0 border-l {BORDER} {SURFACE_ALT} overflow-y-auto p-2 gap-2"
            ).bind_visibility_from(prefs, "show_live_sidebar")
            with live_sidebar:
                with ui.card().classes(f"w-full no-shadow border {BORDER} {SURFACE}"):
                    ui.label("⏱️ Round Timeline").classes(f"text-xs font-bold {STRONG}")
                    timeline_container = ui.column().classes("w-full gap-0.5 mt-1")
                timeline_slots: Dict[int, Any] = {}

                with ui.card().classes(f"w-full no-shadow border {BORDER} {SURFACE}"):
                    ui.label("📊 Context Health").classes(f"text-xs font-bold {STRONG}")
                    health_label = ui.label("no data yet").classes(f"text-xs {MUTED} font-mono")
                    health_bar = ui.linear_progress(value=0.0, show_value=False).props("instant-feedback")

                with ui.card().classes(f"w-full no-shadow border {BORDER} {SURFACE}"):
                    ui.label("🎓 Skills & State (live)").classes(f"text-xs font-bold {STRONG}")
                    skills_label = ui.label("—").classes(f"text-xs {MUTED}")
                    ui.separator().classes(f"my-1 {BORDER}")
                    peek_plan_button = ui.button(
                        "📋 Peek Plan (CURRENT.md)", icon="checklist",
                        on_click=lambda: open_current_plan_dialog(),
                    ).props("flat dense size=xs no-caps").classes("w-full text-left justify-start")
                    peek_plan_button.tooltip("Inspect macro steps plan in .lollms_code/CURRENT.md")

                    peek_button = ui.button(
                        "📝 Peek Scratchpad", icon="visibility",
                        on_click=lambda: open_scratchpad_dialog(),
                    ).props("flat dense size=xs no-caps").classes("w-full text-left justify-start")
                    peek_button.tooltip("Read the agent's scratchpad right now — safe while generating")
                    scratchpad_badge = ui.badge("0", color="blue").props("floating").bind_visibility_from(
                        session, "current_round", backward=lambda r: session.busy and r > 0
                    )

        # ---- Incomplete Turn / Resume Task Banner ----
        resume_banner = ui.row().classes(
            f"w-full items-center justify-between px-4 py-2 shrink-0 bg-amber-500/10 dark:bg-amber-400/10 "
            f"border-t border-b border-amber-500/30 text-xs text-amber-800 dark:text-amber-200 transition-all"
        )
        resume_banner.visible = False
        with resume_banner:
            with ui.row().classes("items-center gap-2 flex-1 min-w-0"):
                ui.icon("pending_actions", size="18px").classes("text-amber-500 shrink-0 animate-pulse")
                resume_banner_label = ui.label("Incomplete turn detected.").classes("truncate font-semibold")

            with ui.row().classes("items-center gap-2 shrink-0"):
                ui.button(
                    "Resume Task", icon="play_arrow",
                    on_click=lambda: resume_active_turn(),
                ).props("unelevated dense size=sm color=primary no-caps font-bold").tooltip("Resume the interrupted task directly")
                ui.button(icon="close", on_click=lambda: resume_banner.set_visibility(False)).props("flat dense round size=xs color=grey")

        # ---- Slash-command suggestions (shown above the input, hidden by default) ----
        suggestions_row = ui.row().classes(
            f"w-full gap-1 px-3 py-1.5 shrink-0 flex-wrap {SURFACE} border-t {BORDER}"
        )
        suggestions_row.visible = False

        # ---- Input Box (Pinned to Bottom) ----
        with ui.row().classes(
            f"w-full items-end gap-2 px-3 py-2 shrink-0 {SURFACE} border-t {BORDER}"
        ):
            with ui.column().classes("flex-1 gap-0"):
                prompt_input = ui.textarea(
                    placeholder="Describe task, or type / for commands… (Enter to send, Shift+Enter for new line)"
                ).classes(
                    "w-full bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded"
                ).props(':dark="Quasar.Dark.isActive" outlined autogrow dense rows=1 input-debounce=0')
                input_counter = ui.label("").classes(f"text-[10px] {MUTED_DIM} self-end pr-1")
            history_button = ui.button(icon="history", on_click=lambda: open_history_dialog()).props(
                "round flat dense size=md"
            ).tooltip("Prompt history (Ctrl+H) — browse and resend past messages")
            regen_button = ui.button(icon="autorenew", on_click=lambda: regenerate_last()).props(
                "round flat dense size=md"
            ).tooltip("Regenerate last response")
            stop_button = ui.button(icon="stop", on_click=lambda: stop_generation()).props(
                "round dense size=md color=negative"
            ).tooltip("Stop generation")
            stop_button.bind_visibility_from(session, "busy")
            send_button = ui.button(icon="send").props("round dense size=md color=primary")

    # ---- Upload dialog (drop a file straight into the workspace root) ----
    upload_dialog = ui.dialog()
    with upload_dialog, ui.card().classes("w-[420px]"):
        ui.label("Upload to workspace root").classes("font-bold")

        def handle_upload(e):
            try:
                ws_root = Path(prefs.workspace_path).resolve()
                dest = ws_root / e.name
                dest.write_bytes(e.content.read())
                ui.notify(f"Uploaded {e.name}", type="positive")
                upload_dialog.close()
                refresh_workspace_tree()
            except Exception as ex:
                notify_error(f"Upload failed: {ex}")

        ui.upload(on_upload=handle_upload, auto_upload=True).props("flat dense accept='*'").classes("w-full")
        with ui.row().classes("w-full justify-end mt-2"):
            ui.button("Close", on_click=upload_dialog.close).props("flat")

    def add_user_bubble(text: str, persist: bool = True) -> str:
        msg_id = f"msg_{next_message_id()}"
        entry = {"type": "user", "text": text, "id": msg_id}
        debug_log.append(entry)
        if persist:
            session.save_to_disk()
        with transcript:
            outer_row = ui.row().classes("w-full justify-end items-start gap-1")
            with outer_row:
                actions = ui.row().classes("opacity-50 hover:opacity-100 transition-opacity gap-0.5 items-center")
                with actions:
                    edit_btn = ui.button(icon="edit", on_click=lambda m=msg_id: open_edit_dialog(m)).props(
                        "flat round dense size=xs text-color=primary"
                    ).tooltip("Edit and resend from this point (truncates following history)")

                    resend_btn = ui.button(icon="replay", on_click=lambda m=msg_id: resend_from_point(m)).props(
                        "flat round dense size=xs text-color=amber"
                    ).tooltip("Resend from this point (truncates following history)")

                    use_btn = ui.button(icon="edit_note", on_click=lambda t=text: set_prompt_input(t)).props(
                        "flat round dense size=xs color=grey"
                    ).tooltip("Copy to input box")

                    del_btn = ui.button(icon="delete", on_click=lambda m=msg_id: delete_message(m)).props(
                        "flat round dense size=xs color=red"
                    ).tooltip("Delete message")

                bubble = ui.markdown(text).classes(
                    "bg-primary text-white rounded-lg px-3 py-1.5 max-w-[75%]"
                )
        message_refs[msg_id] = {"entry": entry, "row": outer_row, "bubble": bubble, "text": text}
        scroll_state["auto_follow"] = True
        scroll_area.scroll_to(percent=1.0)
        return msg_id

    def add_agent_message_container() -> ui.markdown:
        msg_id = f"agent_{next_message_id()}"
        entry = {"type": "agent", "text": "", "id": msg_id, "pinned": False}
        debug_log.append(entry)
        with transcript:
            outer_row = ui.row().classes("w-full justify-start items-start gap-1")
            with outer_row:
                md = ui.markdown("").classes(
                    "bg-slate-100 dark:bg-slate-800/90 text-slate-900 dark:text-slate-100 rounded-lg px-4 py-2.5 max-w-[85%] text-sm break-words leading-relaxed border border-slate-300 dark:border-slate-700 shadow-sm"
                )

                def toggle_pin(entry=entry):
                    entry["pinned"] = not entry.get("pinned", False)
                    pin_btn.props(
                        f"flat round dense size=xs color={'amber' if entry['pinned'] else 'grey'}"
                    )

                pin_btn = ui.button(icon="star", on_click=toggle_pin).props("flat round dense size=xs color=grey")
                pin_btn.tooltip("Pin this message")
                copy_btn = ui.button(icon="content_copy", on_click=lambda: copy_agent_message(entry)).props(
                    "flat round dense size=xs"
                )
                copy_btn.tooltip("Copy this message")
        md._debug_entry = entry
        last_agent_state["entry"] = entry
        message_refs[msg_id] = {"entry": entry, "row": outer_row, "bubble": md}
        return md

    def copy_agent_message(entry: Dict[str, Any]):
        try:
            text = entry.get("text", "")
            if text:
                ui.clipboard.write(text)
                ui.notify("Message copied to clipboard.", type="positive")
            else:
                ui.notify("Nothing to copy yet.", type="warning")
        except Exception as e:
            notify_error(f"Copy failed: {e}")

    def copy_system_notice(entry: Dict[str, Any]):
        try:
            ui.clipboard.write(entry.get("text", "")
                if entry.get("text") is not None else "")
            ui.notify("Notice copied to clipboard.", type="positive")
        except Exception as e:
            notify_error(f"Copy failed: {e}")

    def notify_error(message: str):
        """Single funnel for every error surfaced to the user.
        Writes a persistent, copyable red card into the transcript and shows a
        short toast pointing at it. Long messages (tracebacks) live ONLY in the
        card, so nothing error-shaped is ever un-copyable or lost."""
        short = message.strip()[:120]
        toast = f"⚠️ {short}" if len(message.strip()) > 120 else f"⚠️ {message.strip()}"
        try:
            add_system_notice(f"**Error**\n\n```\n{message.strip()}\n```", is_error=True)
        except Exception:
            ui.notify(toast, type="negative", timeout=8000)
            return
        ui.notify(
            f"{toast} — full details in the transcript card.",
            type="negative",
            timeout=7000
        )

    def add_system_notice(text: str, is_error: bool = False):
        entry = {"type": "system", "text": text, "error": is_error}
        debug_log.append(entry)
        with transcript:
            with ui.row().classes("w-full justify-start items-start gap-1"):
                ui.markdown(text).classes(
                    ("bg-red-50 dark:bg-red-950 text-red-800 dark:text-red-200 border border-red-300 dark:border-red-800" if is_error
                     else "bg-amber-50 dark:bg-amber-950 text-amber-900 dark:text-amber-200 border border-amber-300 dark:border-amber-800")
                    + " rounded-lg px-3 py-1.5 max-w-[90%] text-sm shadow-sm font-medium"
                )
                copy_btn = ui.button(
                    icon="content_copy",
                    on_click=lambda entry_ref=entry: copy_system_notice(entry_ref),
                ).props("flat round dense size=xs")
                copy_btn.tooltip("Copy this notice" + (" (error)" if is_error else ""))
        if scroll_state.get("auto_follow", True):
            scroll_area.scroll_to(percent=1.0)

    def update_code_box(cb, text: str):
        if not cb:
            return
        clean_text = text if text else "(waiting for output...)"
        try:
            if hasattr(cb, "set_text"):
                cb.set_text(clean_text)
            else:
                cb.text = clean_text
            cb.update()
        except Exception:
            try:
                cb.text = clean_text
                cb.update()
            except Exception:
                pass

    def add_event_panel(
        title: str,
        subtitle: str,
        body: str,
        color: str,
        icon: str,
        with_spinner: bool = False,
        expanded: bool = False,
        record: bool = True
    ) -> ui.expansion:
        entry = {"type": "event", "title": title, "subtitle": subtitle, "body": body}
        if record:
            debug_log.append(entry)
        border_color_class = f"border-{color}" if color.startswith("red") or color.startswith("green") or color.startswith("blue") or color.startswith("amber") or color.startswith("purple") else "border-primary"

        with transcript:
            panel = ui.expansion("", value=expanded).classes(
                f"w-full max-w-[90%] border-l-4 {border_color_class} bg-slate-100/90 dark:bg-slate-900/80 "
                f"border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-100 rounded-r-md text-xs shadow-sm"
            ).props(':dark="Quasar.Dark.isActive" header-class="py-1 px-2.5 text-xs text-slate-900 dark:text-slate-100"')
            panel.bind_visibility_from(tools_toggle, "value")

            with panel.add_slot("header"):
                with ui.row().classes("w-full items-center justify-between gap-2 flex-nowrap"):
                    with ui.row().classes("items-center gap-2 min-w-0 flex-1"):
                        header_icon = ui.icon(icon, size="16px").classes(f"text-{color} shrink-0")
                        spinner = ui.spinner(size="14px", color="primary").classes("shrink-0")
                        spinner.set_visibility(with_spinner)
                        title_label = ui.label(title).classes("font-semibold text-xs truncate")
                        sub_label = ui.label(subtitle).classes(f"text-[11px] {MUTED_DIM} font-mono truncate flex-1")
                        if not subtitle:
                            sub_label.set_visibility(False)

            with panel:
                with ui.row().classes("w-full items-center justify-end mb-1"):
                    copy_btn = ui.button(
                        icon="content_copy",
                        on_click=lambda e_ref=entry: copy_event_body(e_ref),
                    ).props("flat round dense size=xs color=grey")
                    copy_btn.tooltip("Copy panel content")

                # High-contrast reactive code box adapting to light/dark themes
                code_box = ui.label(body or "(waiting for output...)").classes(
                    "w-full text-xs p-3 rounded-lg border font-mono whitespace-pre-wrap select-all leading-relaxed max-h-96 overflow-auto block "
                    "bg-slate-900 text-slate-100 border-slate-700 dark:bg-slate-950 dark:text-slate-200 dark:border-slate-800"
                )

        panel._title_label = title_label
        panel._sub_label = sub_label
        panel._header_icon = header_icon
        panel._spinner = spinner
        panel._code_box = code_box
        panel._debug_entry = entry
        panel._user_toggled = False

        panel.on("click", lambda: setattr(panel, "_user_toggled", True))
        return panel

    def _extract_structural_header_title(text_buffer: str) -> Optional[str]:
        """
        Extracts ONLY structural symbols (Markdown headers, function names, class definitions).
        Never returns raw content lines, table rows, or variable statements.
        """
        if not text_buffer:
            return None
        lines = text_buffer.splitlines()
        for line in reversed(lines):
            st = line.strip()
            if not st:
                continue

            # Markdown headings
            m_h = re.match(r'^(#{1,6})\s+(.+)$', st)
            if m_h:
                lvl = len(m_h.group(1))
                h_name = m_h.group(2).strip()
                prefix = "Section" if lvl <= 2 else "Subsection"
                return f"{prefix}: {h_name}"

            # Python function/class
            m_py = re.match(r'^(?:async\s+)?(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', st)
            if m_py:
                return f"{m_py.group(1)} {m_py.group(2)}()"

            # JS/TS/Rust functions & components
            m_js = re.match(r'^(?:export\s+)?(?:async\s+)?(function|class|interface|type)\s+([a-zA-Z_][a-zA-Z0-9_]*)', st)
            if m_js:
                return f"{m_js.group(1)} {m_js.group(2)}"

        return None

    def find_active_artefact_item(target_title: str) -> Optional[Dict[str, Any]]:
        if target_title in active_artefact_panels:
            return active_artefact_panels[target_title]
        clean_target = target_title.replace("\\", "/").strip().lstrip("./")
        for k, v in active_artefact_panels.items():
            clean_k = k.replace("\\", "/").strip().lstrip("./")
            if clean_k == clean_target or Path(k).name == Path(target_title).name:
                return v
        return None

    def copy_event_body(entry: Dict[str, Any]):
        try:
            text = entry.get("body") or "(no output)"
            ui.clipboard.write(text)
            ui.notify("Copied to clipboard.", type="positive")
        except Exception as e:
            notify_error(f"Copy failed: {e}")

    def build_debug_markdown() -> str:
        resolved = env.resolve_default_connection("llm")
        lines = [
            "# lollms_code — session transcript",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Workspace: `{prefs.workspace_path}`",
            f"- Binding / Model: `{resolved.get('binding_name') or '?'}` / `{resolved.get('model_name') or '?'}`",
            "",
            "---",
            "",
        ]
        for entry in debug_log:
            kind = entry["type"]
            if kind == "user":
                lines += ["**You:**", "", entry["text"], ""]
            elif kind == "agent":
                star = "⭐ " if entry.get("pinned") else ""
                lines += [f"**{star}Agent:**", "", (entry["text"] or "_(empty)_"), ""]
            elif kind == "system":
                prefix = "⚠️" if entry.get("error") else "ℹ️"
                lines += [f"> {prefix} {entry['text']}", ""]
            elif kind == "event":
                lines += [f"<details><summary>{entry['title']}</summary>", ""]
                if entry.get("subtitle"):
                    lines += [f"_{entry['subtitle']}_", ""]
                lines += ["```", entry.get("body") or "(no output)", "```", "</details>", ""]
        return "\n".join(lines)

    def copy_debug_markdown():
        md_text = build_debug_markdown()
        ui.clipboard.write(md_text)
        ui.notify("Discussion copied as Markdown.", type="positive")

    def export_history():
        """Exports the full session transcript (user, agent, system, events) as a downloadable Markdown file."""
        try:
            md_text = build_debug_markdown()
            ui.download(md_text.encode("utf-8"), filename="lollms_code_session.md")
            ui.notify("Session exported as Markdown.", type="positive")
        except Exception as e:
            notify_error(f"Export failed: {e}")

    def find_message_entry(msg_id: str) -> Optional[Dict[str, Any]]:
        for entry in debug_log:
            if entry.get("id") == msg_id:
                return entry
        return None

    def set_prompt_input(text_to_set: str):
        clean = re.sub(r"^🔁 _Rerun:_\s*", "", text_to_set).strip()
        prompt_input.value = clean
        prompt_input.run_method("focus")
        _update_input_counter()
        ui.notify("Copied to prompt input.", type="info", timeout=1200)

    def truncate_history_from_user_msg(target_msg_id: str) -> Optional[str]:
        """
        Enforces linear history by discarding all messages, events, and conversation turns
        occurring AFTER the specified user message. Returns the prompt text of the target message.
        """
        entry = find_message_entry(target_msg_id)
        if entry is None or entry.get("type") != "user":
            return None

        # 1. Truncate debug_log
        try:
            target_idx = session.debug_log.index(entry)
        except ValueError:
            target_idx = -1
            for idx, e in enumerate(session.debug_log):
                if e.get("id") == target_msg_id:
                    target_idx = idx
                    break

        if target_idx == -1:
            return None

        # Count how many user messages preceded this one
        user_turn_index = 0
        for i in range(target_idx):
            if session.debug_log[i].get("type") == "user":
                user_turn_index += 1

        # Keep debug_log up to target_idx
        session.debug_log[:] = session.debug_log[:target_idx + 1]

        # 2. Truncate personality._conversation to match linear history
        if session.personality and hasattr(session.personality, "_conversation"):
            conv = session.personality._conversation
            # Each turn has 1 user + 1 assistant message. Keep only prior user turns.
            keep_conv_len = user_turn_index * 2
            if keep_conv_len < len(conv):
                session.personality._conversation = conv[:keep_conv_len]

            # Synchronize truncated history with project conversation file on disk
            if hasattr(session.personality, "_project_history_file") and session.personality._project_history_file:
                try:
                    session.personality.save_history_to_disk(session.personality._project_history_file)
                except Exception:
                    pass

        # 3. Clean turn checkpoints on disk
        if session.personality and hasattr(session.personality, "_get_checkpoint_path"):
            chk_p = session.personality._get_checkpoint_path()
            if chk_p and chk_p.exists():
                try:
                    chk_p.unlink(missing_ok=True)
                except Exception:
                    pass

        session.save_to_disk()
        return entry.get("text", "")

    async def resend_from_point(msg_id: str):
        """Discards all subsequent turns and re-executes the agent from this user prompt."""
        if session.busy:
            ui.notify("Agent is busy — stop or wait for current turn to finish.", type="warning")
            return

        prompt_text = truncate_history_from_user_msg(msg_id)
        if not prompt_text:
            ui.notify("Message could not be found to resend.", type="warning")
            return

        # Repaint transcript to reflect truncated linear history
        replay_transcript_from_log()
        ui.notify("Discarded subsequent history. Continuing from this point...", type="info")

        # Launch fresh turn from this linear point
        session.busy = True
        session.turn_start_ts = time.time()
        send_button.props("loading")
        status_label.set_text("Thinking…")
        show_thinking_indicator("Thinking…")

        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
            session.turn_start_ts = None
            send_button.props(remove="loading")
            return

        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, prompt_text, prefs, session.event_queue, use_history=True
        )

    def open_edit_dialog(msg_id: str):
        ref = message_refs.get(msg_id)
        entry = find_message_entry(msg_id)
        if ref is None or entry is None:
            ui.notify("Message not found.", type="warning")
            return

        with ui.dialog() as dialog, ui.card().classes(
            f"w-[640px] max-w-[95vw] p-5 {CANVAS} text-slate-900 dark:text-slate-100 rounded-xl border {BORDER} shadow-2xl gap-3"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("edit", size="22px").classes("text-primary")
                    ui.label("Edit Prompt & Resend (Linear History)").classes("text-base font-bold")
                ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

            ui.label(
                "Editing this prompt will discard all subsequent discussion turns and continue execution linearly from this point."
            ).classes(f"text-xs {MUTED_DIM}")

            editor = ui.textarea(value=entry.get("text", "")).classes(
                f"w-full text-xs font-mono {SURFACE} rounded border {BORDER} p-2"
            ).props(':dark="Quasar.Dark.isActive" outlined autogrow rows=4')

            with ui.row().classes(f"w-full items-center justify-between pt-2 border-t {BORDER}"):
                ui.button("Cancel", on_click=dialog.close).props("flat dense no-caps")

                async def save_and_resend():
                    new_text = (editor.value or "").strip()
                    if not new_text:
                        ui.notify("Prompt cannot be empty.", type="warning")
                        return

                    dialog.close()
                    if session.busy:
                        ui.notify("Agent is busy — wait for current turn to finish.", type="warning")
                        return

                    # Update entry text, truncate history after this point
                    entry["text"] = new_text
                    truncate_history_from_user_msg(msg_id)

                    # Repaint transcript
                    replay_transcript_from_log()
                    ui.notify("History truncated. Executing from edited prompt...", type="info")

                    # Launch fresh turn
                    session.busy = True
                    session.turn_start_ts = time.time()
                    send_button.props("loading")
                    status_label.set_text("Thinking…")
                    show_thinking_indicator("Thinking…")

                    try:
                        session.ensure_ready()
                    except Exception as e:
                        notify_error(f"Could not start agent: {e}")
                        session.busy = False
                        session.turn_start_ts = None
                        send_button.props(remove="loading")
                        return

                    agent_bridge.run_agent_turn_in_thread(
                        session.personality, session.client, new_text, prefs, session.event_queue, use_history=True
                    )

                ui.button("Save & Resend from Here", icon="send", on_click=save_and_resend).props(
                    "unelevated dense color=primary no-caps font-semibold"
                )

        dialog.open()

    def regenerate_last():
        """Regenerates the most recent user prompt by truncating back to it."""
        last_user = None
        for entry in reversed(debug_log):
            if entry.get("type") == "user":
                last_user = entry
                break
        if last_user is None:
            ui.notify("No previous prompt to regenerate.", type="warning")
            return
        if session.busy:
            ui.notify("Agent is busy — wait for the current turn to finish.", type="warning")
            return
        msg_id = last_user.get("id")
        if msg_id:
            async def _go():
                await resend_from_point(msg_id)
            ui.timer(0.05, _go, once=True)

    def stop_generation():
        hide_thinking_indicator()
        if not session.busy:
            return

        # Clean up any active approval modal dialog if open
        if active_approval_dialog_holder.get("dialog") is not None:
            try:
                active_approval_dialog_holder["dialog"].close()
            except Exception:
                pass
            rq = active_approval_dialog_holder.get("resp_queue")
            if rq:
                try:
                    rq.put(("reject", "Generation cancelled by user."))
                except Exception:
                    pass
            active_approval_dialog_holder["dialog"] = None
            active_approval_dialog_holder["resp_queue"] = None

        try:
            session.ensure_ready()
            cancelled = False

            if hasattr(agent_bridge, "cancel_agent_turn"):
                cancelled = agent_bridge.cancel_agent_turn(session.personality, session.client)
            elif session.personality is not None:
                if hasattr(session.personality, "cancel_generation"):
                    session.personality.cancel_generation()
                    cancelled = True
                elif hasattr(session.personality, "cancel"):
                    session.personality.cancel()
                    cancelled = True

            if session.client is not None:
                if hasattr(session.client, "cancel"):
                    try:
                        session.client.cancel()
                        cancelled = True
                    except Exception:
                        pass
                elif hasattr(session.client, "llm") and hasattr(session.client.llm, "cancel"):
                    try:
                        session.client.llm.cancel()
                        cancelled = True
                    except Exception:
                        pass

            # Seal any active text block and immediately save live state to disk
            seal_current_text_block()
            session.save_to_disk()

            if cancelled:
                add_system_notice("⏹️ Stop requested — cancelling generation...")
                ui.notify("Stopping generation...", type="info")
            else:
                ui.notify("Cancellation isn't supported by this agent backend yet.", type="warning")
        except Exception as e:
            notify_error(f"Could not stop generation: {e}")

    async def send_prompt_with_text(prompt_text: str, rerun_marker: bool = False):
        """Shared send pipeline used by both the input box and rerun/edit actions."""
        raw_prompt = prompt_text.strip()
        if not raw_prompt or session.busy:
            return

        scroll_state["auto_follow"] = True
        display_text = f"🔁 _Rerun:_ {raw_prompt}" if rerun_marker else raw_prompt
        add_user_bubble(display_text)

        if raw_prompt.startswith("/"):
            await handle_slash_command(raw_prompt)
            return

        _remember_prompt(raw_prompt)
        session.busy = True
        session.turn_start_ts = time.time()
        resume_banner.set_visibility(False)
        resume_turn_btn.set_visibility(False)
        send_button.props("loading")
        status_label.set_text("Thinking…")
        show_thinking_indicator("Thinking…")

        # ── 🔄 MULTI-WORD CONTINUATION HYDRATION ("continue with the rest", "next batch", etc.) ──
        effective_prompt = raw_prompt
        is_continuation_request = bool(re.search(
            r'\b(?:continue|resume|proceed|go\s+on|keep\s+going|next\s+batch|the\s+rest|remaining|organize\s+(?:the\s+)?rest)\b',
            raw_prompt,
            re.IGNORECASE
        ))
        is_approval_request = raw_prompt.lower().strip() in ("yes", "y", "oui", "proceed", "approved", "ok", "do it", "sure", "go ahead")

        ws_dir = Path(prefs.workspace_path).resolve()
        has_pending_migration_plan = any((ws_dir / p).exists() for p in ("mapping.yaml", "mapping.json"))

        if is_approval_request and has_pending_migration_plan:
            plan_file = "mapping.yaml" if (ws_dir / "mapping.yaml").exists() else "mapping.json"
            effective_prompt = (
                f"{raw_prompt}\n\n"
                f"[SYSTEM DIRECTIVE: USER CONFIRMED PLAN APPROVAL]\n"
                f"The user approved the migration plan '{plan_file}'.\n"
                f"You MUST now execute Phase 4 immediately:\n"
                f"Call `<tool>{{\"name\": \"tool_organize_files_from_plan\", \"parameters\": {{\"plan_file\": \"{plan_file}\", \"move_files\": true}}}}</tool>` as your very first token now!\n"
                f"Do NOT output conversational preambles or step lists without the tool call tag."
            )
            ASCIIColors.info(f"[ChatPage] Hydrated approval prompt with migration execution directive for '{plan_file}'.")
        elif is_continuation_request:
            last_task = session.get_last_user_prompt()
            plan_content = agent_bridge.get_current_plan_content(workspace_path=prefs.workspace_path)
            task_hint = last_task or "file organization task"
            effective_prompt = (
                f"[SYSTEM DIRECTIVE: User requested to continue with the rest of the task]\n"
                f"Original Task: '{task_hint}'\n"
                f"User Instruction: '{raw_prompt}'\n\n"
                "1. If files were already moved in the previous batch: Call `tool_list_files(directory=\".\")` to scan the remaining items in the workspace root.\n"
                "2. Select the next batch of up to 50 items and create `<artifact name=\"mapping.yaml\">` (or call `tool_organize_files_from_plan` if a plan is ready).\n"
                "3. DO NOT claim in text that files were moved without executing the tool! Proceed with the real execution now."
            )
            if plan_content and "No active task plan" not in plan_content:
                effective_prompt += f"\nActive Roadmap in CURRENT.md:\n{plan_content[:600]}\n"
            ASCIIColors.info(f"[ChatPage] Hydrated multi-word continuation prompt: '{raw_prompt}'")
        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
            session.turn_start_ts = None
            send_button.props(remove="loading")
            return
        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, effective_prompt, prefs, session.event_queue, use_history=True
        )

    def open_edit_dialog(msg_id: str):
        ref = message_refs.get(msg_id)
        entry = find_message_entry(msg_id)
        if ref is None or entry is None:
            ui.notify("Message not found.", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-[560px]"):
            ui.label("Edit message").classes("text-lg font-bold mb-2")
            editor = ui.textarea(value=ref["text"]).classes("w-full").props("outlined autogrow")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def save_edit():
                    new_text = (editor.value or "").strip()
                    if not new_text:
                        ui.notify("Message cannot be empty.", type="warning")
                        return
                    entry["text"] = new_text
                    ref["text"] = new_text
                    ref["bubble"].set_content(new_text)
                    dialog.close()
                    ui.notify("Message updated.", type="positive")

                ui.button("Save", on_click=save_edit).props("flat color=primary")

                async def save_and_rerun():
                    new_text = (editor.value or "").strip()
                    if not new_text:
                        ui.notify("Message cannot be empty.", type="warning")
                        return
                    entry["text"] = new_text
                    ref["text"] = new_text
                    ref["bubble"].set_content(new_text)
                    dialog.close()
                    if session.busy:
                        ui.notify("Agent is busy — wait for the current turn to finish.", type="warning")
                        return
                    await send_prompt_with_text(new_text, rerun_marker=True)

                ui.button("Save & Rerun", on_click=save_and_rerun).props("color=primary")
        dialog.open()

    def delete_message(msg_id: str):
        entry = find_message_entry(msg_id)
        ref = message_refs.pop(msg_id, None)
        if entry is not None:
            try:
                session.debug_log.remove(entry)
            except ValueError:
                pass
            session.save_to_disk()
        try:
            ui.notify("Message deleted.", type="positive")
        except Exception:
            pass
        if ref is not None and ref.get("row") is not None:
            try:
                ref["row"].delete()
            except Exception:
                pass

    def confirm_new_session():
        confirm = ui.dialog()
        with confirm, ui.card():
            ui.label("Start a new session?").classes("font-bold")
            ui.label("This clears the visible conversation and the agent's in-memory history.").classes(
                "text-sm text-gray-500"
            )
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=confirm.close).props("flat")

                def go():
                    confirm.close()
                    new_session()

                ui.button("New session", on_click=go).props("color=primary")
        confirm.open()

    def new_session():
        hide_thinking_indicator()
        if active_approval_dialog_holder.get("dialog") is not None:
            try:
                active_approval_dialog_holder["dialog"].close()
            except Exception:
                pass
            active_approval_dialog_holder["dialog"] = None
            active_approval_dialog_holder["resp_queue"] = None

        transcript.clear()
        debug_log.clear()
        message_refs.clear()
        ws_root = Path(prefs.workspace_path).resolve()
        current_md = ws_root / ".lollms_code" / "CURRENT.md"
        if current_md.exists():
            try:
                current_md.write_text("# Current Task\n\nNo active task plan defined yet.\n", encoding="utf-8")
            except Exception:
                pass

        # Wipe ephemeral scratchpad on new session
        scratchpad_md = ws_root / ".lollms_code" / "scratchpad.md"
        if scratchpad_md.exists():
            try:
                scratchpad_md.write_text("# Scratchpad\n\n(Empty - session notes only)\n", encoding="utf-8")
            except Exception:
                pass

        if session.personality is not None:
            try:
                session.personality._conversation = []
                object.__setattr__(session.personality, '_scratchpad_content', '')
            except Exception:
                pass
        session.prompt_history.clear()
        session.history_index = -1
        rounds_label.set_text("")
        ctx_label.set_text("")
        elapsed_label.set_text("")
        health_label.set_text("no data yet")
        health_bar.set_value(0.0)
        timeline_container.clear()
        timeline_slots.clear()
        session.current_round = 0
        add_system_notice("🆕 New session started.")

    def open_current_plan_dialog():
        dialog = ui.dialog().props("maximized")
        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("checklist", size="26px").classes("text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label("Task Plan (CURRENT.md)").classes("text-base font-bold")
                        ui.label("Macro steps plan tracked by the agent in .lollms_code/CURRENT.md").classes(
                            f"text-xs {MUTED_DIM}"
                        )

                with ui.row().classes("gap-2 items-center"):
                    edit_mode = {"active": False}
                    current_plan_text = {"content": ""}

                    def _toggle_edit():
                        edit_mode["active"] = not edit_mode["active"]
                        plan_md_view.set_visibility(not edit_mode["active"])
                        plan_editor.set_visibility(edit_mode["active"])
                        save_btn.set_visibility(edit_mode["active"])
                        edit_btn.text = "Preview" if edit_mode["active"] else "Edit"
                        edit_btn._props["icon"] = "visibility" if edit_mode["active"] else "edit"
                        if edit_mode["active"]:
                            plan_editor.value = current_plan_text["content"]

                    def _save_plan():
                        try:
                            ws = Path(prefs.workspace_path).resolve()
                            plan_file = ws / ".lollms_code" / "CURRENT.md"
                            plan_file.parent.mkdir(parents=True, exist_ok=True)
                            new_text = plan_editor.value or ""
                            plan_file.write_text(new_text, encoding="utf-8")
                            current_plan_text["content"] = new_text
                            plan_md_view.set_content(new_text if new_text.strip() else "# Current Task Plan\n\n_(No active task plan defined yet)_")
                            _toggle_edit()
                            ui.notify("Plan saved to .lollms_code/CURRENT.md", type="positive")
                        except Exception as ex:
                            notify_error(f"Failed to save plan: {ex}")

                    def _refresh_plan():
                        try:
                            session.ensure_ready()
                            content = agent_bridge.get_current_plan_content(session.personality, prefs.workspace_path)
                            current_plan_text["content"] = content
                            plan_md_view.set_content(
                                content if content.strip() else "# Current Task Plan\n\n_(No active task plan in .lollms_code/CURRENT.md)_"
                            )
                            plan_editor.value = content
                            ui.notify("Plan refreshed from CURRENT.md.", type="positive", timeout=1200)
                        except Exception as e:
                            notify_error(f"Failed to read CURRENT.md: {e}")

                    edit_btn = ui.button("Edit", icon="edit", on_click=_toggle_edit).props("flat size=sm no-caps")
                    save_btn = ui.button("Save", icon="save", on_click=_save_plan).props("unelevated size=sm color=primary no-caps")
                    save_btn.visible = False
                    ui.button("Refresh", icon="refresh", on_click=_refresh_plan).props("flat size=sm no-caps")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat size=sm no-caps")

            plan_md_view = ui.markdown("").classes(
                f"flex-1 overflow-auto p-4 bg-slate-100 dark:bg-slate-900 rounded border {BORDER} text-sm leading-relaxed"
            )
            plan_editor = ui.textarea().classes(
                f"flex-1 w-full bg-slate-100 dark:bg-slate-900 text-xs font-mono rounded border {BORDER} p-2"
            ).props(':dark="Quasar.Dark.isActive" outlined autogrow')
            plan_editor.visible = False

            _refresh_plan()
        dialog.open()

    def open_scratchpad_dialog():
        dialog = ui.dialog().props("maximized")
        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("edit_note", size="26px").classes("text-cyan-500")
                    with ui.column().classes("gap-0"):
                        ui.label("Agent Scratchpad").classes("text-lg font-bold")
                        ui.label("Persistent intermediate notes and architectural state in .lollms_code/scratchpad.md").classes(
                            f"text-xs {MUTED_DIM}"
                        )

                with ui.row().classes("gap-2 items-center"):
                    edit_mode = {"active": False}
                    current_scratch_text = {"content": ""}

                    def _toggle_scratch_edit():
                        edit_mode["active"] = not edit_mode["active"]
                        scratchpad_md.set_visibility(not edit_mode["active"])
                        scratch_editor.set_visibility(edit_mode["active"])
                        save_scratch_btn.set_visibility(edit_mode["active"])
                        edit_scratch_btn.text = "Preview" if edit_mode["active"] else "Edit"
                        edit_scratch_btn._props["icon"] = "visibility" if edit_mode["active"] else "edit"
                        if edit_mode["active"]:
                            scratch_editor.value = current_scratch_text["content"]

                    def _save_scratchpad():
                        try:
                            ws = Path(prefs.workspace_path).resolve()
                            scratch_file = ws / ".lollms_code" / "scratchpad.md"
                            scratch_file.parent.mkdir(parents=True, exist_ok=True)
                            new_text = scratch_editor.value or ""
                            scratch_file.write_text(new_text, encoding="utf-8")
                            current_scratch_text["content"] = new_text
                            scratchpad_md.set_content(new_text if new_text.strip() else "_(Scratchpad is empty)_")
                            _toggle_scratch_edit()
                            ui.notify("Scratchpad saved.", type="positive")
                        except Exception as ex:
                            notify_error(f"Failed to save scratchpad: {ex}")

                    def _refresh_scratchpad():
                        try:
                            session.ensure_ready()
                            content = agent_bridge.get_scratchpad_content(session.personality, prefs.workspace_path)
                            current_scratch_text["content"] = content
                            scratchpad_md.set_content(content if content.strip() else "# Agent Scratchpad\n\n_(Scratchpad is empty)_")
                            scratch_editor.value = content
                            ui.notify("Scratchpad refreshed.", type="positive", timeout=1200)
                        except Exception as e:
                            notify_error(f"Failed to read scratchpad: {e}")

                    edit_scratch_btn = ui.button("Edit", icon="edit", on_click=_toggle_scratch_edit).props("flat size=sm no-caps")
                    save_scratch_btn = ui.button("Save", icon="save", on_click=_save_scratchpad).props("unelevated size=sm color=primary no-caps")
                    save_scratch_btn.visible = False
                    ui.button("Refresh", icon="refresh", on_click=_refresh_scratchpad).props("flat size=sm no-caps")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat size=sm no-caps")

            scratchpad_md = ui.markdown("").classes(
                f"flex-1 overflow-auto p-4 bg-slate-100 dark:bg-slate-900 rounded border {BORDER} text-sm leading-relaxed"
            )
            scratch_editor = ui.textarea().classes(
                f"flex-1 w-full bg-slate-100 dark:bg-slate-900 text-xs font-mono rounded border {BORDER} p-2"
            ).props(':dark="Quasar.Dark.isActive" outlined autogrow')
            scratch_editor.visible = False

            _refresh_scratchpad()
        dialog.open()

    def open_shortcuts_dialog():
        with ui.dialog() as dialog, ui.card().classes("w-[420px]"):
            ui.label("⌨️ Keyboard shortcuts").classes("text-lg font-bold mb-2")
            for key, desc in SHORTCUTS:
                with ui.row().classes("w-full justify-between items-center gap-4"):
                    ui.label(key).classes("font-mono text-xs text-primary")
                    ui.label(desc).classes("text-xs text-gray-500")
            with ui.row().classes("w-full justify-end mt-3"):
                ui.button("Close", on_click=dialog.close).props("flat")
        dialog.open()

    def _sync_theme_button():
        try:
            is_dark = bool(dark_mode.value)
            theme_btn._props["icon"] = "dark_mode" if is_dark else "light_mode"
            theme_btn._props["title"] = "Theme: Dark (click for Light)" if is_dark else "Theme: Light (click for Dark)"
            theme_btn.update()
        except Exception:
            pass

    def toggle_theme():
        new_dark = not bool(dark_mode.value)
        dark_mode.set_value(new_dark)
        prefs.dark_mode = new_dark
        try:
            prefs.save()
        except Exception:
            pass
        _sync_theme_button()
        ui.notify(f"Theme: {'Dark' if new_dark else 'Light'}", type="info", timeout=1200)

    _sync_theme_button()

    # ---------------- Transcript search ----------------

    def open_search():
        search_row.visible = True
        search_input.run_method("focus")

    def close_search():
        search_row.visible = False
        search_input.value = ""
        apply_search()

    def apply_search():
        q = (search_input.value or "").lower().strip()
        matches = 0
        for ref in message_refs.values():
            row = ref.get("row")
            entry = ref.get("entry", {})
            if row is None:
                continue
            text = (entry.get("text") or "").lower()
            visible = (not q) or (q in text)
            if visible:
                matches += 1
            row.set_visibility(visible)
        search_count_label.set_text(f"{matches} match(es)" if q else "")

    search_input.on_value_change(lambda e: apply_search())

    def _strip_processing_tags(text: str) -> str:
        if not text:
            return ""
        # 1. Strip complete processing blocks
        cleaned = re.sub(r"<processing.*?</processing>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # 2. Strip any trailing or unclosed processing tag block
        cleaned = re.sub(r"<processing[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        # 3. Strip any stray closing tags or status comments
        cleaned = re.sub(r"</?processing[^>]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<!--\s*status:[^>]*-->", "", cleaned, flags=re.IGNORECASE)

        # 4. Strip model tool call template tokens and leaked skill headers
        cleaned = re.sub(r'\[TOOL_CALLS\][^\n]*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'=== END (?:ACTIVE )?SKILLS ===\s*', '', cleaned, flags=re.IGNORECASE)

        # 5. Strip leaked blockquoted raw tool JSON fragments and pseudo-code tags
        cleaned = re.sub(r'(?m)^\s*>?(?:```(?:json)?\s*)?\{"name":\s*"tool_\w+.*$', '', cleaned)
        cleaned = re.sub(r'(?s)>?\s*\{"name":\s*"tool_\w+".*?\}\s*(?:</tool>)?', '', cleaned)
        cleaned = re.sub(r'</?tool\b[^>]*>', '', cleaned, flags=re.IGNORECASE)
        # Strip pseudo-tag syntax (e.g. ```{tool}{name=...} and ```{artifact}{name=...})
        cleaned = re.sub(r'(?s)(?:```)?\{tool\}[^\n]*.*?\}*(?:```)?', '', cleaned)
        cleaned = re.sub(r'(?s)(?:```)?\{artifact\}[^\n]*.*?\}*(?:```)?', '', cleaned)
        cleaned = re.sub(r'scratchpad_append(?:\[ARGS\])?[^\n]*\n?', '', cleaned)
        cleaned = re.sub(r'```(?:python)?\s*```', '', cleaned)
        cleaned = re.sub(r'`{4,}', '```', cleaned)
        # Strip unclosed broken backtick markers and stray delimiters (e.g. `}`, ` `}, `---`, `>, `)
        cleaned = re.sub(r'(?m)^\s*[`>]{1,4}\s*$', '', cleaned)
        cleaned = re.sub(r'(?m)^\s*[`>\s]*\}\s*$', '', cleaned)
        cleaned = re.sub(r'(?m)^\s*[-=]{3,}\s*$', '', cleaned)
        cleaned = re.sub(r'```(?:python|bash|sh|json|xml)?\s*$', '', cleaned).strip()
        cleaned = re.sub(r'[`>]{1,4}\s*$', '', cleaned).strip()

        # Strip unexecuted or raw functional action tags from displaying inside speech bubbles
        cleaned = re.sub(
            r"<(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update|mem_load|mem_delete|mem_search|mem_tag).*?(?:/>|</(?:unlock_file|lock_file|hide_file|pin_file|unpin_file|collapse_folder|uncollapse_folder|scratchpad_append|scratchpad_patch|scratchpad_clear|user_profile_update|user_profile_clear|mem_new|mem_update|mem_load|mem_delete|mem_search|mem_tag)>)",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        # 4. Comprehensive de-duplication (exact halving, duplicate paragraphs, repeated sentences)
        cleaned_text = cleaned.strip()
        if not cleaned_text:
            return ""

        # Exact-halving check (e.g. text literally echoed as A + A)
        h = len(cleaned_text) // 2
        for offset in (0, -1, 1, -2, 2):
            test_h = h + offset
            if 15 < test_h < len(cleaned_text):
                p1 = cleaned_text[:test_h].strip()
                p2 = cleaned_text[test_h:].strip()
                if p1 and p1 == p2:
                    cleaned_text = p1
                    break

        # Paragraph & line collapse
        lines = cleaned_text.splitlines()
        deduped_lines = []
        recent_lines = []
        for line in lines:
            st = line.strip()
            if not st:
                if deduped_lines and deduped_lines[-1] != "":
                    deduped_lines.append("")
                continue
            if recent_lines and st == recent_lines[-1]:
                continue
            recent_lines.append(st)
            if len(recent_lines) > 5:
                recent_lines.pop(0)
            deduped_lines.append(line)

        cleaned_text = "\n".join(deduped_lines).strip()

        # Sentence-level duplication collapse
        def _collapse_sentences(para: str) -> str:
            parts = re.split(r'(?<=[.!?])\s+', para)
            if len(parts) < 2:
                return para
            out = []
            for s in parts:
                st = s.strip()
                if st and out and out[-1].strip() == st:
                    continue
                out.append(s)
            return " ".join(out)

        paras = cleaned_text.split("\n\n")
        cleaned_paras = [_collapse_sentences(p) for p in paras]
        return "\n\n".join(cleaned_paras).strip()

    current_agent_md: Optional[ui.markdown] = None
    agent_text_buffer = ""
    active_tool_panels: Dict[str, Dict[str, Any]] = {}
    active_artefact_panels: Dict[str, Dict[str, Any]] = {}
    active_approval_dialog_holder: Dict[str, Any] = {"dialog": None, "resp_queue": None}
    thinking_indicator: Optional[ui.element] = None
    thinking_label: Optional[ui.label] = None

    def show_thinking_indicator(message: str = "Thinking…"):
        nonlocal thinking_indicator, thinking_label
        if thinking_indicator is not None:
            if thinking_label is not None:
                thinking_label.set_text(message)
            return

        with transcript:
            thinking_indicator = ui.row().classes(
                f"w-fit items-center gap-2.5 px-4 py-2.5 rounded-xl border {BORDER} "
                f"{SURFACE} text-slate-800 dark:text-slate-200 shadow-sm transition-all select-none"
            )
            with thinking_indicator:
                ui.icon("psychology", size="18px").classes("text-primary shrink-0 animate-pulse")
                thinking_label = ui.label(message).classes(f"text-xs font-mono {MUTED}")
                with ui.row().classes("items-center gap-1 shrink-0 ml-1"):
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: -0.32s; animation-duration: 1.1s;"
                    )
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: -0.16s; animation-duration: 1.1s;"
                    )
                    ui.element("span").classes("w-1.5 h-1.5 rounded-full bg-primary animate-bounce").style(
                        "animation-delay: 0s; animation-duration: 1.1s;"
                    )
        scroll_area.scroll_to(percent=1.0)

    def hide_thinking_indicator():
        nonlocal thinking_indicator, thinking_label
        if thinking_indicator is not None:
            try:
                thinking_indicator.delete()
            except Exception:
                pass
            thinking_indicator = None
            thinking_label = None

    def open_python_approval_dialog(source: str, script_label: str, argv: Optional[List[Any]], resp_queue: Any):
        """Displays an in-app modal authorization dialog for LCP tool execution in safe mode."""
        dialog = ui.dialog().props("persistent")
        active_approval_dialog_holder["dialog"] = dialog
        active_approval_dialog_holder["resp_queue"] = resp_queue

        source_lines = source.splitlines()
        line_count = len(source_lines)
        is_shell = script_label.startswith("shell:") or script_label.startswith("[RISKY:") or script_label.startswith("[STRICT]")

        with dialog, ui.card().classes(
            "w-[780px] max-w-[95vw] max-h-[90vh] flex flex-col p-4 gap-3 "
            "bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 "
            "rounded-xl shadow-2xl border-2 border-amber-500/80"
        ):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("security", size="28px").classes("text-amber-500")
                    with ui.column().classes("gap-0"):
                        ui.label(f"🛡️ Authorization Request ({prefs.shell_autonomy_level.upper()} Mode)").classes("text-base font-bold")
                        ui.label("The agent requested execution of the command/code below.").classes(
                            "text-xs text-slate-500 dark:text-slate-400"
                        )

            with ui.row().classes("items-center gap-3 text-xs font-mono"):
                ui.label(f"Target: {script_label}").classes("font-semibold text-primary")
                if argv and len(argv) > 1:
                    ui.label(f"Args: {argv[1:]}").classes("text-slate-500")
                ui.label(f"Lines: {line_count:,}").classes("text-slate-400")

            with ui.scroll_area().classes(
                "w-full max-h-[360px] border border-slate-300 dark:border-slate-800 rounded bg-slate-50 dark:bg-slate-950 p-2"
            ):
                lang = "bash" if is_shell else ("python" if script_label.endswith(".py") or "def " in source or "import " in source else "text")
                ui.code(source, language=lang).classes("w-full text-xs")

            feedback_input = ui.input(
                placeholder="Optional feedback / instruction if rejecting..."
            ).classes("w-full text-xs").props("outlined dense clearable")

            with ui.row().classes("w-full items-center justify-between pt-2 border-t border-slate-200 dark:border-slate-800"):
                def _do_reject():
                    dialog.close()
                    fb = (feedback_input.value or "").strip()
                    if resp_queue:
                        resp_queue.put(("reject", fb or "User declined execution."))
                    active_approval_dialog_holder["dialog"] = None
                    active_approval_dialog_holder["resp_queue"] = None
                    ui.notify("Execution rejected.", type="warning")

                def _do_allow():
                    dialog.close()
                    if resp_queue:
                        resp_queue.put(("allow", ""))
                    active_approval_dialog_holder["dialog"] = None
                    active_approval_dialog_holder["resp_queue"] = None
                    ui.notify("Authorized single execution.", type="positive")

                def _do_always():
                    dialog.close()
                    if resp_queue:
                        resp_queue.put(("always", ""))
                    prefs.auto_approve_python = True
                    active_approval_dialog_holder["dialog"] = None
                    active_approval_dialog_holder["resp_queue"] = None
                    ui.notify("Auto-approval enabled for this active session.", type="positive")

                ui.button("Reject", icon="cancel", on_click=_do_reject).props(
                    "unelevated color=negative size=sm no-caps"
                ).tooltip("Reject execution and pass optional feedback to the LLM")

                with ui.row().classes("items-center gap-2"):
                    ui.button("Always Allow (This Session)", icon="done_all", on_click=_do_always).props(
                        "outline color=emerald size=sm no-caps"
                    ).tooltip("Auto-approve all Python executions for this session (removes human bottleneck)")
                    ui.button("Run Once", icon="play_arrow", on_click=_do_allow).props(
                        "unelevated color=primary size=sm no-caps"
                    ).tooltip("Authorize this execution only")

        dialog.open()

    def seal_current_text_block():
        """Seals the current conversational agent bubble, deduplicating identical repeat bubbles."""
        nonlocal current_agent_md, agent_text_buffer
        if current_agent_md is not None:
            clean_text = _strip_processing_tags(agent_text_buffer).strip()
            entry = getattr(current_agent_md, "_debug_entry", None)

            # Check for duplicate bubble with identical content to previous agent entry
            is_duplicate = False
            if clean_text and len(debug_log) > 1:
                for prev in reversed(debug_log[:-1]):
                    if prev.get("type") == "agent":
                        prev_text = prev.get("text", "").strip()
                        if prev_text and prev_text == clean_text:
                            is_duplicate = True
                        break

            if clean_text and not is_duplicate:
                current_agent_md.set_content(clean_text)
                if entry is not None:
                    entry["text"] = clean_text
            else:
                try:
                    current_agent_md.delete()
                    if entry in debug_log:
                        debug_log.remove(entry)
                except Exception:
                    pass

            current_agent_md = None
            agent_text_buffer = ""
            session.save_to_disk()

    def drain_queue():
        nonlocal current_agent_md, agent_text_buffer

        drained_any = False
        while True:
            try:
                ev = session.event_queue.get_nowait()
            except queue.Empty:
                break
            drained_any = True

            if ev.kind == "chunk":
                if ev.data.get("was_processed"):
                    continue
                chunk_text = ev.data.get("text", "")
                if not chunk_text or "<processing" in chunk_text or "</processing>" in chunk_text or "<!-- status:" in chunk_text:
                    continue

                # Strip out raw tool JSON that might have bypassed low-level filters
                if chunk_text.strip().startswith('{"') and chunk_text.strip().endswith('}'):
                    continue

                hide_thinking_indicator()

                if current_agent_md is None:
                    current_agent_md = add_agent_message_container()
                    agent_text_buffer = ""

                agent_text_buffer += chunk_text
                cleaned = _strip_processing_tags(agent_text_buffer)
                current_agent_md.set_content(cleaned)
                entry = getattr(current_agent_md, "_debug_entry", None)
                if entry is not None:
                    entry["text"] = cleaned

            elif ev.kind == "python_approval_request":
                source = ev.data.get("source", "")
                script_label = ev.data.get("script_label", "script.py")
                argv = ev.data.get("argv")
                resp_queue = ev.data.get("response_queue")
                open_python_approval_dialog(source, script_label, argv, resp_queue)

            elif ev.kind == "effort_change":
                lvl = ev.data.get("level", "")
                ui.notify(f"⚡ Reasoning effort scaled to: {lvl.upper()}", type="info", timeout=2500)
                add_event_panel(f"⚡ Dynamic Effort: {lvl.upper()}", "Reasoning tier updated for next round", f"The agent adjusted its reasoning effort to '{lvl}' using <effort level='{lvl}'/>.", "amber-500", "psychology")

            elif ev.kind == "thought":
                hide_thinking_indicator()
                seal_current_text_block()
                sub_agent_tag = f"[{ev.data.get('sub_agent')}] " if ev.data.get("sub_agent") else ""
                add_event_panel(f"💭 {sub_agent_tag}Thinking", "", ev.data.get("text", ""), "gray-400", "psychology")

            elif ev.kind == "worker_spawn_start":
                seal_current_text_block()
                ag_name = ev.data.get("agent_name", "Specialist")
                task_txt = ev.data.get("task", "")
                depth = ev.data.get("depth", 1)
                max_steps = ev.data.get("max_steps", 6)
                effort = ev.data.get("effort", "default")
                model_name = ev.data.get("model_name", "parent model")
                conditioning = ev.data.get("personality_conditioning", "")

                info_lines = [
                    f"**Task Directive:**\n```\n{task_txt}\n```",
                    f"- **Delegation Tier**: Depth {depth}",
                    f"- **Reasoning Budget**: Up to {max_steps} rounds",
                    f"- **Effort Tier**: `{effort}`",
                    f"- **Assigned Model**: `{model_name}`",
                ]
                if conditioning and conditioning != "Autonomous Worker Specialist":
                    info_lines.append(f"- **Specialization Conditioning**:\n  _{conditioning[:200]}..._")

                body = "\n".join(info_lines)
                add_event_panel(f"🤖 Sub-Agent Active: {ag_name}", f"depth {depth} · max {max_steps} rounds · effort {effort}", body, "purple-500", "smart_toy")
                status_label.set_text(f"Sub-agent '{ag_name}' executing autonomously…")

            elif ev.kind == "worker_spawn_end":
                seal_current_text_block()
                ag_name = ev.data.get("agent_name", "Specialist")
                rounds = ev.data.get("rounds", 0)
                tools_cnt = ev.data.get("tools_count", 0)
                elapsed = ev.data.get("elapsed_seconds", 0)
                success = ev.data.get("success", True)
                digest = ev.data.get("report_digest", "")

                status_color = "emerald-500" if success else "red-500"
                status_icon = "task_alt" if success else "error"
                status_badge = "COMPLETED" if success else "INTERRUPTED/FAILED"

                body_lines = [
                    f"**Execution Summary:**",
                    f"- **Status**: {status_badge}",
                    f"- **Duration**: {elapsed}s",
                    f"- **Rounds Taken**: {rounds}",
                    f"- **Tools Executed**: {tools_cnt}",
                    "",
                    f"**Specialist Report to Orchestrator:**\n```\n{digest or '(No text reported)'}\n```",
                ]

                add_event_panel(
                    f"{'✅' if success else '❌'} Sub-Agent Report: {ag_name}",
                    f"{rounds} round(s) · {tools_cnt} tool(s) · {elapsed}s",
                    "\n".join(body_lines),
                    status_color,
                    status_icon
                )
                status_label.set_text("Idle")
                session.save_to_disk()

            elif ev.kind == "info":
                inf_type = ev.data.get("type")
                if inf_type == "memory_consolidated":
                    seal_current_text_block()
                    mem_content = ev.data.get("content", "")
                    tags_str = ", ".join(ev.data.get("tags", []))
                    add_event_panel(f"💾 Memory Consolidated", f"tags: {tags_str}", mem_content, "purple-500", "psychology")
                    ui.notify(f"💾 Memory Saved: {mem_content[:60]}...", type="info")
                else:
                    status_label.set_text(ev.data.get("text", "")[:80])

            elif ev.kind == "tool_start":
                name = ev.data.get("tool_name", "tool")
                if name == "pending":
                    continue
                hide_thinking_indicator()
                seal_current_text_block()
                params = ev.data.get("parameters", {})
                params_str = json.dumps(params, indent=2, ensure_ascii=False) if isinstance(params, dict) else str(params)

                # Remove previous running panel and debug entry for the same tool if present
                if name in active_tool_panels:
                    old_item = active_tool_panels.pop(name)
                    try:
                        old_item["panel"].delete()
                        if hasattr(old_item["panel"], "_debug_entry") and old_item["panel"]._debug_entry in debug_log:
                            debug_log.remove(old_item["panel"]._debug_entry)
                    except Exception:
                        pass

                subtitle = "executing…"
                if name == "tool_execute_shell_command" and isinstance(params, dict) and "command" in params:
                    subtitle = f"$ {params['command']}"
                elif isinstance(params, dict) and "file_name" in params:
                    subtitle = f"{params['file_name']} · executing…"

                panel = add_event_panel(f"🛠️ Running: {name}", subtitle, params_str, "blue-500", "build")
                active_tool_panels[name] = {"panel": panel, "params": params}
                status_label.set_text(f"Running {name}…")
                _paint_round(timeline_slots, session.current_round, "bg-blue-500 animate-pulse")

            elif ev.kind == "tool_end":
                name = ev.data.get("tool_name", "tool")
                if name == "pending":
                    continue
                # Discard parser stream completion signals that lack an actual execution output/status
                has_result = "success" in ev.data or bool(ev.data.get("output")) or bool(ev.data.get("error"))
                if not has_result:
                    continue

                seal_current_text_block()
                success = bool(ev.data.get("success", False))
                output = ev.data.get("output") or ev.data.get("error") or ""
                color = "green-500" if success else "red-500"
                params = ev.data.get("parameters", {})

                # If params wasn't in ev.data, extract from active_tool_panels
                if not params and name in active_tool_panels:
                    params = active_tool_panels[name].get("params", {})

                # Remove the 'Running...' panel from both UI and debug_log so only the finished result is preserved
                if name in active_tool_panels:
                    active_item = active_tool_panels.pop(name)
                    if not params:
                        params = active_item.get("params", {})
                    try:
                        active_item["panel"].delete()
                        if hasattr(active_item["panel"], "_debug_entry") and active_item["panel"]._debug_entry in debug_log:
                            debug_log.remove(active_item["panel"]._debug_entry)
                    except Exception:
                        pass

                subtitle = "success" if success else "failed"
                body_content = output

                if name == "tool_execute_shell_command":
                    cmd_str = (params.get("command") if isinstance(params, dict) else None) or ev.data.get("command")
                    if cmd_str:
                        subtitle = f"$ {cmd_str}"
                        if not output.startswith("$ "):
                            body_content = f"$ {cmd_str}\n\n{output}"
                elif name in ("tool_execute_python_file", "tool_read_document_content", "tool_inspect_document") and isinstance(params, dict) and "file_name" in params:
                    subtitle = f"{params['file_name']} · {'success' if success else 'failed'}"

                add_event_panel(
                    f"{'✅' if success else '❌'} Finished: {name}",
                    subtitle,
                    body_content,
                    color,
                    "build_circle"
                )
                _paint_round(
                    timeline_slots, session.current_round,
                    "bg-green-500" if success else "bg-red-500"
                )
                session.save_to_disk()

            elif ev.kind == "context_update":
                files = ev.data.get("files", [])
                status = ev.data.get("status", "")
                error = ev.data.get("error")

                # Discard streaming/parser signals that carry no files and no error
                if not files and not error:
                    continue

                seal_current_text_block()
                _paint_round(timeline_slots, session.current_round, "bg-amber-500")

                action = ev.data.get("action", "context")
                color = "green-500" if status == "success" else ("red-500" if status == "failure" else "amber-500")
                body = "\n".join(files) if files else str(error or "(no files)")

                add_event_panel(
                    f"📂 Context: {action.replace('_', ' ').title()}",
                    status or "completed",
                    body,
                    color,
                    "folder_open",
                )
                session.save_to_disk()

            elif ev.kind == "scratchpad_update":
                seal_current_text_block()
                action = ev.data.get("action", "update")
                message = ev.data.get("message", "Scratchpad updated.")
                ui.notify(f"📝 {message}", type="info")
                add_event_panel("📝 Scratchpad", action, message, "yellow-600", "edit_note")

            elif ev.kind == "artefact_start":
                hide_thinking_indicator()
                title = ev.data.get("title", "artifact")
                lang = ev.data.get("language", "")
                op = ev.data.get("operation", "write")
                is_patch = ev.data.get("is_patch", False) or (op == "patch")
                sec = ev.data.get("current_section") or ""
                op_label = "patch" if is_patch else op
                subtitle = f"{op_label} · {lang}..." if lang else f"{op_label}..."
                if sec:
                    subtitle = f"{sec}..."
                seal_current_text_block()

                existing_item = find_active_artefact_item(title)
                if existing_item:
                    old_panel = existing_item["panel"]
                    try:
                        old_panel.delete()
                        if hasattr(old_panel, "_debug_entry") and old_panel._debug_entry in debug_log:
                            debug_log.remove(old_panel._debug_entry)
                    except Exception:
                        pass
                    for k in list(active_artefact_panels.keys()):
                        if active_artefact_panels[k] is existing_item:
                            del active_artefact_panels[k]

                header_icon_name = "build" if is_patch else "description"
                header_color = "amber-500" if is_patch else "purple-500"
                panel_title = f"🔧 Patching: {title}" if is_patch else f"📝 Writing: {title}"

                panel = add_event_panel(
                    title=panel_title,
                    subtitle=subtitle,
                    body="",
                    color=header_color,
                    icon=header_icon_name,
                    with_spinner=True,
                    expanded=True
                )
                active_artefact_panels[title] = {
                    "panel": panel,
                    "title": title,
                    "op": "patch" if is_patch else op,
                    "lang": lang,
                    "buffer": "",
                    "last_line": "",
                    "entry": getattr(panel, "_debug_entry", {})
                }
                status_label.set_text(f"{'Patching' if is_patch else 'Writing'} {title}…")
                _paint_round(timeline_slots, session.current_round, f"bg-{header_color} animate-pulse")

            elif ev.kind == "artefact_chunk":
                title = ev.data.get("title", "artifact")
                chunk_txt = ev.data.get("text", "")
                item = find_active_artefact_item(title)
                if item and chunk_txt:
                    item["buffer"] += chunk_txt
                    panel = item["panel"]

                    # Dynamic patch detection if chunk reveals SEARCH block
                    if item.get("op") != "patch" and (
                        "<<<<<<< SEARCH" in item["buffer"]
                        or "<<<<<<<" in chunk_txt
                    ):
                        item["op"] = "patch"
                        if hasattr(panel, "_title_label"):
                            panel._title_label.set_text(f"🔧 Patching: {title}")
                        if hasattr(panel, "_header_icon"):
                            panel._header_icon._props["name"] = "build"
                            panel._header_icon.classes(replace="text-amber-500 shrink-0")
                        status_label.set_text(f"Patching {title}…")

                    # Extract ONLY structural headings or function titles for header display
                    structural_title = _extract_structural_header_title(item["buffer"])
                    if structural_title:
                        item["last_symbol"] = structural_title
                        if hasattr(panel, "_sub_label"):
                            panel._sub_label.set_text(f"• {structural_title}")
                            panel._sub_label.set_visibility(True)
                        status_label.set_text(f"{'Patching' if item.get('op') == 'patch' else 'Writing'} {title}: {structural_title[:40]}")
                    elif not item.get("last_symbol") and hasattr(panel, "_sub_label"):
                        panel._sub_label.set_text("• patching content..." if item.get("op") == "patch" else "• writing content...")
                        panel._sub_label.set_visibility(True)

                    # Stream whole verbatim content directly into code box without clamping
                    buf = item["buffer"]
                    if hasattr(panel, "_code_box"):
                        update_code_box(panel._code_box, buf)
                    if item.get("entry"):
                        item["entry"]["body"] = buf

            elif ev.kind == "artefact_symbol":
                sym = ev.data.get("symbol", {})
                detail = sym.get("detail") or ev.data.get("detail", "")
                title = ev.data.get("title", "artifact")
                status_label.set_text(f"Writing {title}: {detail}")
                item = find_active_artefact_item(title)
                if item:
                    panel = item["panel"]
                    if hasattr(panel, "_sub_label"):
                        panel._sub_label.set_text(f"• {detail}")
                        panel._sub_label.set_visibility(True)

            elif ev.kind == "artefact_end":
                title = ev.data.get("title", "artifact")
                seal_current_text_block()
                success = bool(ev.data.get("success", False))
                version = ev.data.get("version", 1)
                lines = ev.data.get("line_count", 0)
                chars = ev.data.get("size_chars", 0)
                is_patch = ev.data.get("is_patch", False)

                meta_details = []
                if version: meta_details.append(f"v{version}")
                if lines: meta_details.append(f"{lines} lines")
                if chars: meta_details.append(f"{chars:,} chars")
                final_sub = " · ".join(meta_details) if success else str(ev.data.get("error", "failed"))

                item = find_active_artefact_item(title)
                if item:
                    for k in list(active_artefact_panels.keys()):
                        if active_artefact_panels[k] is item:
                            del active_artefact_panels[k]

                    panel = item["panel"]
                    if hasattr(panel, "_spinner"):
                        panel._spinner.set_visibility(False)
                    if hasattr(panel, "_header_icon"):
                        panel._header_icon._props["name"] = "task_alt" if success else "error"
                        panel._header_icon.classes(replace=f"text-{'green-500' if success else 'red-500'} shrink-0")
                    if hasattr(panel, "_title_label"):
                        panel._title_label.set_text(f"{'✅' if success else '❌'} {'Patched' if is_patch else 'Saved'}: {title}")
                    if hasattr(panel, "_sub_label"):
                        panel._sub_label.set_text(final_sub)
                        panel._sub_label.set_visibility(True)

                    full_c = ev.data.get("content") or item["buffer"]
                    if full_c and hasattr(panel, "_code_box"):
                        c_lines = full_c.strip().splitlines()
                        preview_slice = full_c if len(c_lines) <= 60 else "\n".join(c_lines[:30] + [f"\n... [{len(c_lines)-50:,} lines hidden] ...\n"] + c_lines[-20:])
                        update_code_box(panel._code_box, preview_slice)
                    if item.get("entry"):
                        item["entry"]["body"] = full_c
                else:
                    end_sig = (title, version, is_patch, success, session.current_round)
                    if getattr(session, "_last_rendered_artefact_end", None) == end_sig:
                        continue
                    session._last_rendered_artefact_end = end_sig

                    body_lines = []
                    sections = ev.data.get("sections", [])
                    if sections:
                        body_lines.append("Sections/Symbols:")
                        for s in sections[:10]:
                            body_lines.append(f"  • {s.get('type', 'item')}: {s.get('name', '')} (line {s.get('line', '?')})")
                    if not body_lines and ev.data.get("content"):
                        raw_c = str(ev.data["content"]).strip()
                        c_lines = raw_c.splitlines()
                        body_lines.append("\n".join(c_lines[:25] if len(c_lines) <= 30 else c_lines[:15] + [f"\n... (+{len(c_lines)-25} more lines)\n"] + c_lines[-10:]))

                    add_event_panel(
                        f"{'✅' if success else '❌'} {'Patched' if is_patch else 'Saved'}: {title}",
                        final_sub,
                        "\n".join(body_lines) if body_lines else "(Document saved)",
                        "green-500" if success else "red-500",
                        "task_alt",
                        with_spinner=False,
                        expanded=False
                    )
                session.save_to_disk()

            elif ev.kind == "round_start":
                seal_current_text_block()
                r = ev.data.get("round_id", 1)
                max_cfg = getattr(prefs, "max_reasoning_steps", 100)
                m = "∞ ⚠️" if max_cfg <= 0 else ev.data.get("max_rounds", max_cfg)
                status_label.set_text(f"Round {r}/{m}")
                try:
                    r_int = int(r)
                except (TypeError, ValueError):
                    r_int = session.current_round
                session.current_round = r_int
                with timeline_container:
                    with ui.row().classes("w-full items-center gap-1"):
                        ui.label(f"R{r_int}").classes("text-[10px] font-mono text-gray-400 w-6 shrink-0")
                        dot = ui.element("div").classes("h-2.5 w-2.5 rounded-full bg-gray-300")
                timeline_slots[r_int] = dot
                show_thinking_indicator(f"Round {r_int}: Thinking…")

            elif ev.kind == "round_info":
                r = ev.data.get("round", "?")
                m = ev.data.get("max_rounds", "?")
                status_label.set_text(f"Round {r}/{m}")

            elif ev.kind == "round_end":
                r_status = ev.data.get("status")
                r_id = ev.data.get("round_id", session.current_round)
                if r_id in timeline_slots and r_status == "done":
                    _paint_round(timeline_slots, r_id, "bg-green-500")
                refresh_workspace_tree()

            elif ev.kind == "done":
                hide_thinking_indicator()
                seal_current_text_block()
                result = ev.data.get("result", {}) or {}
                session.busy = False
                elapsed = None
                if session.turn_start_ts is not None:
                    elapsed = time.time() - session.turn_start_ts
                    elapsed_label.set_text(f"{elapsed:.1f}s")
                session.turn_start_ts = None
                send_button.props(remove="loading")
                status_label.set_text("Idle")

                # Auto-save session state to disk immediately
                session.save_to_disk()

                # Fallback display guarantee: if no agent text was rendered, display result response
                final_resp = (result.get("response") or "").strip()
                has_visible_agent_msg = False
                for entry in reversed(debug_log):
                    if entry.get("type") == "agent":
                        if (entry.get("text") or "").strip():
                            has_visible_agent_msg = True
                        break

                if not has_visible_agent_msg:
                    if final_resp:
                        fallback_md = add_agent_message_container()
                        fallback_md.set_content(final_resp)
                        if hasattr(fallback_md, "_debug_entry") and fallback_md._debug_entry:
                            fallback_md._debug_entry["text"] = final_resp
                    elif result.get("tool_calls") or result.get("workspace_changes"):
                        summary_lines = ["✅ **Task completed.**\n"]
                        ws_ch = result.get("workspace_changes", [])
                        if ws_ch:
                            summary_lines.append("**Files created/modified:**")
                            for ch in ws_ch:
                                summary_lines.append(f"- `{ch.get('path', '?')}` ({ch.get('action', 'saved')})")
                        tc = result.get("tool_calls", [])
                        if tc:
                            summary_lines.append(f"\nExecuted {len(tc)} tool call(s).")
                        fallback_msg = "\n".join(summary_lines)
                rounds_txt = f"Rounds: {result.get('rounds', 0)} · Tools: {len(result.get('tool_calls', []))}"
                ctx = result.get("context_health") or {}
                used_tokens = ctx.get("used_tokens")
                if elapsed and used_tokens and elapsed > 0:
                    rounds_txt += f" · {used_tokens / elapsed:.0f} tok/s"
                rounds_label.set_text(rounds_txt)
                if ctx.get("max_tokens"):
                    ctx_label.set_text(
                        f"Context: {ctx.get('used_tokens', 0):,}/{ctx.get('max_tokens', 0):,} "
                        f"({ctx.get('fill_percentage', 0):.1f}%)"
                    )
                _set_health_bar(health_bar, health_label, ctx)

                _refresh_live_skills(skills_label)

                with timeline_container:
                    ui.separator().classes("my-1")
                    ui.label(f"✅ done — {result.get('rounds', 0)} round(s)").classes(
                        "text-[10px] text-green-600 dark:text-green-400"
                    )
                timeline_slots.clear()
                session.current_round = 0

                skills_created = result.get("skills_created") or []
                skills_updated = result.get("skills_updated") or []
                if (skills_created or skills_updated) and prefs.show_skills_activity:
                    body = "\n".join([f"created: {s}" for s in skills_created] +
                                      [f"updated: {s}" for s in skills_updated])
                    add_event_panel("🎓 Skills activity", "", body, "yellow-600", "school")

                if result.get("was_cancelled"):
                    add_system_notice("⏹️ Generation was cancelled by user.")
                    ui.notify("Generation stopped.", type="info", timeout=2000)
                elif prefs.__dict__.get("notify_on_done", True):
                    ui.notify("✅ Agent turn finished.", type="positive", timeout=2000)

                refresh_workspace_tree()

            elif ev.kind == "error":
                hide_thinking_indicator()
                seal_current_text_block()
                session.busy = False
                session.turn_start_ts = None
                elapsed_label.set_text("")
                send_button.props(remove="loading")
                status_label.set_text("Error")
                notify_error(str(ev.data.get("message", "Unknown agent error.")))

        if drained_any:
            if apply_search.__closure__ is not None and (search_input.value or "").strip():
                apply_search()
            # Smart auto-scroll: ONLY scroll if the user is already at the bottom
            if scroll_state.get("auto_follow", True):
                scroll_area.scroll_to(percent=1.0)

    ui.timer(0.15, drain_queue)

    _last_periodic_save = 0.0

    def _tick_elapsed():
        nonlocal _last_periodic_save
        now = time.time()
        if session.busy and session.turn_start_ts is not None:
            elapsed_label.set_text(f"{now - session.turn_start_ts:.1f}s")
            # Continuous live auto-save every 2.5s while actively generating
            if now - _last_periodic_save > 2.5:
                _last_periodic_save = now
                session.save_to_disk()

    ui.timer(0.2, _tick_elapsed)

    # ---------------- Live telemetry helpers ----------------

    def _paint_round(slots: Dict[int, Any], round_no: int, css: str):
        dot = slots.get(round_no)
        if dot is not None:
            dot.classes(replace=f"h-2.5 w-2.5 rounded-full {css}")

    def _set_health_bar(bar, label_widget, ctx: Dict[str, Any]):
        try:
            fill = float(ctx.get("fill_percentage", 0.0)) / 100.0
            used = int(ctx.get("used_tokens", 0) or 0)
            max_t = int(ctx.get("max_tokens", 0) or 0)
        except (TypeError, ValueError):
            return
        if max_t <= 0:
            return
        bar.set_value(fill)
        bar.props(f"color={'green' if fill <= 0.65 else 'orange' if fill <= 0.85 else 'red'}")
        label_widget.set_text(f"{used:,} / {max_t:,} tokens ({fill * 100:.1f}%)")

    def _refresh_live_skills(label_widget):
        try:
            session.ensure_ready()
            skills = agent_bridge.get_live_skills(session.personality)
            count = len(skills)
            if count != session.live_skills_count:
                grew = count > session.live_skills_count and session.busy
                session.live_skills_count = count
                label_widget.set_text(f"{count} skill(s)")
                if grew:
                    ui.notify("🎓 A skill was created or updated during this turn.", type="info")
        except Exception:
            label_widget.set_text("—")

    # ---------------- Slash-command autocomplete + prompt history ----------------

    def _remember_prompt(text: str):
        if not text or text.startswith("/"):
            return
        clean_text = text.strip()
        if not clean_text:
            return
        if not session.prompt_history or session.prompt_history[-1] != clean_text:
            session.prompt_history.append(clean_text)
            if len(session.prompt_history) > 200:
                session.prompt_history = session.prompt_history[-200:]
            session.save_prompt_history()
        session.history_index = -1

    def open_history_dialog():
        dialog = ui.dialog()
        with dialog, ui.card().classes(
            f"w-[680px] max-w-[95vw] h-[580px] max-h-[90vh] flex flex-col p-4 gap-3 "
            f"{CANVAS} text-slate-900 dark:text-slate-100 rounded-xl shadow-2xl border {BORDER}"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("history", size="24px").classes("text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label("Prompt History").classes("text-base font-bold")
                        ui.label("Browse, reuse, or resend messages previously sent to the agent.").classes(
                            f"text-xs {MUTED_DIM}"
                        )
                ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

            search_bar = ui.input(placeholder="Search previous prompts…").classes(
                "w-full text-xs"
            ).props("dense outlined clearable input-debounce=100")

            scroll = ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2")
            with scroll:
                history_list = ui.column().classes("w-full gap-2")

            def refresh_history_items():
                history_list.clear()
                q = (search_bar.value or "").strip().lower()
                entries = list(reversed(session.prompt_history))
                if q:
                    entries = [e for e in entries if q in e.lower()]

                with history_list:
                    if not entries:
                        ui.label("No history matching your search." if q else "No prompt history recorded yet.").classes(
                            f"text-xs {MUTED_DIM} italic p-4 text-center w-full"
                        )
                        return

                    for prompt_text in entries:
                        with ui.card().classes(
                            f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} hover:border-primary/50 transition-colors gap-1.5 shadow-none"
                        ):
                            ui.label(prompt_text).classes(
                                "text-xs font-mono break-words whitespace-pre-wrap max-h-24 overflow-hidden select-all"
                            )
                            with ui.row().classes("w-full items-center justify-end gap-1.5 pt-1"):
                                def _use_in_input(p=prompt_text):
                                    set_prompt_input(p)
                                    dialog.close()

                                async def _resend_now(p=prompt_text):
                                    dialog.close()
                                    if session.busy:
                                        ui.notify("Agent is busy — wait for current turn to finish.", type="warning")
                                        return
                                    await send_prompt_with_text(p)

                                ui.button("Use in input", icon="edit_note", on_click=_use_in_input).props(
                                    "flat dense size=xs no-caps text-color=primary font-semibold"
                                ).tooltip("Place into textarea to edit before sending")

                                ui.button("Resend", icon="send", on_click=_resend_now).props(
                                    "unelevated dense size=xs color=primary no-caps font-semibold"
                                ).tooltip("Send this prompt immediately")

            search_bar.on_value_change(lambda _: refresh_history_items())
            refresh_history_items()

            with ui.row().classes(f"w-full items-center justify-between pt-2 border-t {BORDER}"):
                def _clear_all_history():
                    session.prompt_history.clear()
                    session.save_prompt_history()
                    refresh_history_items()
                    ui.notify("Prompt history cleared.", type="positive")

                ui.button("Clear History", icon="delete_sweep", on_click=_clear_all_history).props(
                    "flat dense size=xs color=red no-caps"
                ).tooltip("Delete all saved prompt history from this workspace")

                ui.button("Close", on_click=dialog.close).props("flat dense size=sm no-caps")

        dialog.open()

    def _update_input_counter():
        n = len(prompt_input.value or "")
        input_counter.set_text(f"{n} chars · ~{max(1, n // 4)} tok" if n else "")

    def _history_up():
        if (prompt_input.value or "").strip():
            return
        if not session.prompt_history:
            return
        session.history_index = min(session.history_index + 1, len(session.prompt_history) - 1)
        prompt_input.value = session.prompt_history[-(session.history_index + 1)]

    def _history_down():
        if session.history_index <= 0:
            session.history_index = -1
            prompt_input.value = ""
            return
        session.history_index -= 1
        prompt_input.value = session.prompt_history[-(session.history_index + 1)]

    def refresh_suggestions():
        text = prompt_input.value or ""
        suggestions_row.clear()
        if not text.startswith("/") or " " in text:
            suggestions_row.visible = False
            return
        matches = [c for c in SLASH_COMMANDS if c[0].startswith(text)]
        if not matches:
            suggestions_row.visible = False
            return
        suggestions_row.visible = True
        with suggestions_row:
            for cmd, desc in matches:
                def pick(cmd=cmd):
                    prompt_input.value = cmd + " "
                    suggestions_row.visible = False
                    prompt_input.run_method("focus")

                with ui.button(cmd, on_click=pick).props("dense outline size=sm no-caps"):
                    ui.tooltip(desc)

    def accept_first_suggestion():
        text = prompt_input.value or ""
        if not text.startswith("/") or " " in text:
            return
        matches = [c for c in SLASH_COMMANDS if c[0].startswith(text)]
        if matches:
            prompt_input.value = matches[0][0] + " "
            suggestions_row.visible = False

    def _on_input_change(e):
        refresh_suggestions()
        _update_input_counter()

    prompt_input.on_value_change(_on_input_change)
    prompt_input.on("keyup", lambda e: refresh_suggestions())
    prompt_input.on("keydown.tab.prevent", lambda e: accept_first_suggestion())
    prompt_input.on("keydown.up", lambda e: _history_up())
    prompt_input.on("keydown.down", lambda e: _history_down())

    # ---------------- Slash-command execution ----------------

    async def handle_slash_command(text: str) -> bool:
        try:
            return await _dispatch_slash_command(text)
        except Exception as e:
            notify_error(f"Command '{text.split(' ', 1)[0]}' failed: {e}")
            return True

    async def _dispatch_slash_command(text: str) -> bool:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd in ("/exit", "/quit"):
            add_system_notice("Nothing to exit to in the GUI — just close the window.")
            return True

        if cmd in ("/workflow", "/workflows", "/studio", "/graph"):
            open_workflow_studio_dialog()
            return True

        if cmd == "/export":
            export_history()
            return True

        if cmd == "/history":
            open_history_dialog()
            return True

        if cmd == "/help":
            add_system_notice(HELP_TEXT)
            return True

        if cmd in ("/plan", "/current", "/current-plan"):
            open_current_plan_dialog()
            return True

        if cmd in ("/inspect", "/context", "/prompt"):
            open_context_inspector_dialog()
            return True

        if cmd in ("/subws", "/reference", "/sub-workspace", "/ref"):
            if arg.lower() in ("paste", "new", "add-text"):
                open_paste_reference_dialog()
            else:
                open_sub_workspace_dialog()
            return True

        if cmd in ("/zoo", "/zoos", "/hub"):
            open_zoo_dialog()
            return True

        if cmd in ("/scratchpad", "/scratch"):
            open_scratchpad_dialog()
            return True

        if cmd in ("/dynamic", "/dynamic-mode"):
            arg_clean = arg.lower().strip()
            if arg_clean in ("on", "1", "enable", "true"):
                prefs.dynamic_effort = True
                prefs.auto_temperature = True
                prefs.auto_max_tokens = True
            elif arg_clean in ("off", "0", "disable", "false"):
                prefs.dynamic_effort = False
                prefs.auto_temperature = False
                prefs.auto_max_tokens = False
            else:
                prefs.dynamic_effort = not getattr(prefs, "dynamic_effort", False)
                prefs.auto_temperature = prefs.dynamic_effort
                prefs.auto_max_tokens = prefs.dynamic_effort

            prefs.save()
            if session.personality:
                session.personality.dynamic_effort = prefs.dynamic_effort

            _sync_dynamic_button()
            effort_select.value = "dynamic" if prefs.dynamic_effort else "default"
            add_system_notice(
                f"⚡ Dynamic Mode is now **{'ACTIVATED' if prefs.dynamic_effort else 'DEACTIVATED'}** "
                f"(Auto effort scaling, task-adapted temperature, auto tokens)."
            )
            return True

        if cmd in ("/resume", "/continue"):
            await resume_active_turn()
            return True

        if cmd in ("/explorer", "/open", "/reveal"):
            open_workspace_in_explorer()
            return True

        if cmd in ("/sessions", "/session"):
            open_sessions_dialog()
            return True

        if cmd == "/config":
            session.save_to_disk()
            ui.navigate.to("/settings")
            return True

        if cmd in ("/models", "/model"):
            try:
                session.ensure_ready()
                if session.client and hasattr(session.client, "llm_model_profiles_registry"):
                    registry = session.client.llm_model_profiles_registry
                    if arg:
                        if session.client.switch_model(arg):
                            active_alias = getattr(session.client, "_active_llm_alias", arg)
                            active_model = getattr(session.client.llm, "model_name", "unknown")
                            active_binding = getattr(session.client.llm, "binding_name", "unknown")
                            add_system_notice(f"🔄 Switched active LLM profile to **{active_alias}** (`{active_binding}` / `{active_model}`).")
                        else:
                            available = ", ".join(f"`{k}`" for k in registry.keys())
                            add_system_notice(f"Failed to switch to profile '{arg}'. Available profiles: {available}", is_error=True)
                        return True
                    else:
                        active_alias = getattr(session.client, "_active_llm_alias", None)
                        lines = ["**Available LLM Model Profiles:**\n"]
                        for alias, prof in registry.items():
                            marker = "⭐ **[ACTIVE]**" if alias == active_alias else "•"
                            b_name = prof.binding_profile_name
                            m_name = prof.model_name or "default"
                            v_flag = " [Vision]" if prof.vision_enabled else ""
                            lines.append(f"{marker} `{alias}` ({b_name} / {m_name}{v_flag})")
                        lines.append("\n_Use `/models <alias>` to switch to another profile._")
                        add_system_notice("\n".join(lines))
                        return True
            except Exception as e:
                add_system_notice(f"Error checking models: {e}", is_error=True)
                return True
            add_system_notice("Model switching is managed via LLM profiles — open `/config` (Settings).")
            return True

        if cmd in ("/clear-history", "/clear"):
            transcript.clear()
            debug_log.clear()
            message_refs.clear()
            ws_root = Path(prefs.workspace_path).resolve()
            current_md = ws_root / ".lollms_code" / "CURRENT.md"
            if current_md.exists():
                try:
                    current_md.write_text("# Current Task\n\nNo active task plan defined yet.\n", encoding="utf-8")
                except Exception:
                    pass

            # Wipe ephemeral scratchpad on clear
            scratchpad_md = ws_root / ".lollms_code" / "scratchpad.md"
            if scratchpad_md.exists():
                try:
                    scratchpad_md.write_text("# Scratchpad\n\n(Empty - session notes only)\n", encoding="utf-8")
                except Exception:
                    pass

            if session.personality is not None:
                session.personality._conversation = []
                object.__setattr__(session.personality, '_scratchpad_content', '')
            add_system_notice("Conversation cleared, task roadmap reset, and ephemeral scratchpad wiped.")
            return True

        if cmd in ("/clear-files", "/unload-all"):
            try:
                session.ensure_ready()
                result = agent_bridge.clear_all_loaded_files(session.personality)
                add_system_notice(result.get("status_str", "Files unloaded."))
            except Exception as e:
                add_system_notice(f"Could not unload files: {e}", is_error=True)
            return True

        if cmd in ("/load", "/unload", "/lock", "/hide", "/unhide"):
            if not arg:
                add_system_notice(f"Usage: `{cmd} <file1> [file2] ...` or `{cmd} all`", is_error=True)
                return True
            action = cmd[1:]  # strip leading "/"
            targets = [t.strip() for t in arg.replace(",", " ").split() if t.strip()]
            try:
                session.ensure_ready()
                result = agent_bridge.change_file_visibility(session.personality, targets, action)
                status = result.get("status_str", "Action completed.")
                add_system_notice(status, is_error=("❌" in status or "BLOCKED" in status))
            except Exception as e:
                add_system_notice(f"Could not change file visibility: {e}", is_error=True)
            return True

        if cmd in ("/memories", "/memory"):
            sub_arg = arg.lower().strip()
            if sub_arg in ("toggle", "switch"):
                prefs.enable_memory = not prefs.enable_memory
                prefs.save()
                state_str = "ENABLED" if prefs.enable_memory else "DISABLED"
                session.personality = agent_bridge.create_personality(prefs, session.client)
                add_system_notice(f"🧠 Cognitive Memory is now **{state_str}**.")
                return True
            elif sub_arg in ("on", "enable", "1", "true"):
                prefs.enable_memory = True
                prefs.save()
                session.personality = agent_bridge.create_personality(prefs, session.client)
                add_system_notice("🧠 Cognitive Memory is now **ENABLED**.")
                return True
            elif sub_arg in ("off", "disable", "0", "false"):
                prefs.enable_memory = False
                prefs.save()
                session.personality = agent_bridge.create_personality(prefs, session.client)
                add_system_notice("🧠 Cognitive Memory is now **DISABLED**.")
                return True
            else:
                if open_memory_explorer_dialog is not None:
                    open_memory_explorer_dialog(session, prefs)
                else:
                    add_system_notice("Memory Explorer component could not be loaded.", is_error=True)
                return True

        if cmd == "/forget":
            confirm_dialog = ui.dialog()
            with confirm_dialog, ui.card():
                ui.label("⚠️ Permanently delete ALL agent memories?").classes("font-bold text-red-500")
                ui.label("This includes learned facts and episodic history. This can't be undone.").classes(
                    "text-sm text-gray-500"
                )
                with ui.row().classes("w-full justify-end gap-2 mt-2"):
                    ui.button("Cancel", on_click=confirm_dialog.close).props("flat")

                    def do_wipe():
                        confirm_dialog.close()
                        try:
                            session.ensure_ready()
                            if hasattr(session.personality, "wipe_all_memories") and session.personality.wipe_all_memories():
                                add_system_notice("🧠 All memories wiped.")
                            else:
                                add_system_notice("Memory manager not initialized or wipe failed.", is_error=True)
                        except Exception as e:
                            add_system_notice(f"Could not wipe memory: {e}", is_error=True)

                    ui.button("Wipe memories", on_click=do_wipe).props("color=red")
            confirm_dialog.open()
            return True

        if cmd == "/skills":
            try:
                session.ensure_ready()
                skills = session.personality.skills_manager.list_skills() if session.personality.skills_manager else []
            except Exception as e:
                add_system_notice(f"Could not load skills: {e}", is_error=True)
                return True
            if not skills:
                add_system_notice("No skills learned yet.")
            else:
                lines = "\n".join(f"- **{s['title']}** ({s.get('category', '')}) — {s.get('description', '')}" for s in skills)
                add_system_notice(f"**Learned skills**\n\n{lines}")
            return True

        if cmd in ("/effort", "/reasoning-effort"):
            if not arg:
                curr = "dynamic" if getattr(prefs, "dynamic_effort", False) else (getattr(prefs, "reasoning_effort", None) or "default")
                add_system_notice(
                    f"**Current Reasoning Effort**: `{curr}`\n\n"
                    "Usage: `/effort <none|low|medium|high|dynamic|default>`\n"
                    "- `none`: Turn off thinking/reasoning (fastest)\n"
                    "- `low`: Light reasoning\n"
                    "- `medium`: Standard deep thinking\n"
                    "- `high`: Maximum reasoning depth\n"
                    "- `dynamic`: Autonomous effort scaling via `<effort>` tags\n"
                    "- `default`: Revert to model profile default"
                )
            else:
                arg_clean = arg.lower().strip()
                if arg_clean == "dynamic":
                    prefs.dynamic_effort = True
                    prefs.reasoning_effort = None
                    prefs.save()
                    effort_select.value = "dynamic"
                    add_system_notice("🔄 Reasoning effort updated to **Dynamic** (auto-escalates via `<effort>` tags).")
                elif arg_clean == "default":
                    prefs.dynamic_effort = False
                    prefs.reasoning_effort = None
                    prefs.save()
                    effort_select.value = "default"
                    add_system_notice("⚡ Reasoning effort reset to **Model Default**.")
                elif arg_clean in ("none", "low", "medium", "high", "max"):
                    prefs.dynamic_effort = False
                    prefs.reasoning_effort = arg_clean
                    prefs.save()
                    effort_select.value = arg_clean if arg_clean in effort_options else "high"
                    add_system_notice(f"🧠 Reasoning effort set to **{arg_clean.capitalize()}**.")
                else:
                    add_system_notice(f"Unknown effort level: '{arg}'. Choose from: `none`, `low`, `medium`, `high`, `dynamic`, `default`.", is_error=True)
            return True

        if cmd == "/files":
            try:
                session.ensure_ready()
                stats = agent_bridge.get_workspace_stats(session.personality)
            except Exception as e:
                add_system_notice(f"Could not read workspace stats: {e}", is_error=True)
                return True
            if not stats["loaded_files"]:
                add_system_notice("No files are currently loaded in context.")
            else:
                lines = "\n".join(f"- `{f['path']}` ({f['size']:,} bytes)" for f in stats["loaded_files"])
                add_system_notice(
                    f"**Loaded context files** ({stats['total_loaded']}/{stats['total_indexed']} indexed)\n\n{lines}"
                )
            return True

        if cmd == "/workspace":
            async def do_switch(path: str):
                try:
                    session.ensure_ready()
                    session.personality = agent_bridge.switch_workspace(prefs, session.client, path)
                    session.load_prompt_history()
                    add_system_notice(f"📂 Workspace switched to `{prefs.workspace_path}`")
                    refresh_workspace_tree()
                except Exception as e:
                    add_system_notice(f"Could not switch workspace: {e}", is_error=True)

            if arg:
                await do_switch(arg)
            else:
                dialog = ui.dialog()
                with dialog, ui.card().classes("w-[480px]"):
                    ui.label("Switch workspace").classes("font-bold")
                    path_input = ui.input("New workspace path", value=prefs.workspace_path).classes("w-full")

                    async def pick_folder_action():
                        _picker = pick_folder
                        if not _picker:
                            try:
                                from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder as _p
                                _picker = _p
                            except Exception:
                                pass
                        if _picker:
                            chosen = await _picker(title="Select New Workspace", initial_dir=path_input.value)
                            if chosen:
                                path_input.value = chosen
                        else:
                            ui.notify("Folder picker unavailable — enter path manually.", type="warning")

                    ui.button("Browse…", icon="folder_open", on_click=pick_folder_action).props("flat")
                    with ui.row().classes("w-full justify-end gap-2 mt-2"):
                        ui.button("Cancel", on_click=dialog.close).props("flat")

                        async def confirm():
                            dialog.close()
                            await do_switch(path_input.value)

                        ui.button("Switch", on_click=confirm).props("color=primary")
                dialog.open()
            return True

        return False

    async def send_prompt():
        text = prompt_input.value.strip()
        if not text or session.busy:
            return
        scroll_state["auto_follow"] = True
        prompt_input.value = ""
        suggestions_row.visible = False
        _update_input_counter()

        if text.startswith("/"):
            add_user_bubble(text)
            await handle_slash_command(text)
            return

        add_user_bubble(text)
        _remember_prompt(text)
        session.busy = True
        session.turn_start_ts = time.time()
        send_button.props("loading")
        status_label.set_text("Thinking…")
        show_thinking_indicator("Thinking…")

        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
            session.turn_start_ts = None
            send_button.props(remove="loading")
            return

        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, text, prefs, session.event_queue, use_history=True
        )

    send_button.on("click", send_prompt)
    prompt_input.on("keydown.enter.exact.prevent", send_prompt)

    # ---------------- Workspace Tree (icons · context menu · open) ----------------

    from nicegui import run as _nicegui_run  # local import: only this block needs it

    IGNORED_NAMES = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".lollms_code"}
    MAX_ENTRIES_PER_DIR = 1500   # hard cap so one huge/broken folder can't stall the page
    MAX_SEARCH_HITS = 300

    # extension -> (material icon, tailwind text colour)
    EXT_ICONS: Dict[str, tuple] = {
        ".py": ("code", "text-yellow-500"),
        ".pyw": ("code", "text-yellow-500"),
        ".ipynb": ("science", "text-orange-500"),
        ".js": ("javascript", "text-yellow-400"),
        ".mjs": ("javascript", "text-yellow-400"),
        ".cjs": ("javascript", "text-yellow-400"),
        ".ts": ("code", "text-blue-400"),
        ".tsx": ("code", "text-blue-400"),
        ".jsx": ("code", "text-cyan-400"),
        ".vue": ("code", "text-emerald-400"),
        ".html": ("html", "text-orange-500"),
        ".htm": ("html", "text-orange-500"),
        ".css": ("css", "text-blue-500"),
        ".scss": ("css", "text-pink-400"),
        ".sass": ("css", "text-pink-400"),
        ".json": ("data_object", "text-amber-500"),
        ".jsonl": ("data_object", "text-amber-500"),
        ".yaml": ("settings_ethernet", "text-purple-400"),
        ".yml": ("settings_ethernet", "text-purple-400"),
        ".toml": ("settings", "text-purple-400"),
        ".ini": ("settings", "text-slate-400"),
        ".cfg": ("settings", "text-slate-400"),
        ".env": ("key", "text-lime-500"),
        ".md": ("article", "text-sky-400"),
        ".rst": ("article", "text-sky-400"),
        ".txt": ("description", "text-slate-400"),
        ".log": ("receipt_long", "text-slate-500"),
        ".csv": ("table_chart", "text-green-500"),
        ".tsv": ("table_chart", "text-green-500"),
        ".xlsx": ("table_view", "text-green-600"),
        ".xls": ("table_view", "text-green-600"),
        ".pdf": ("picture_as_pdf", "text-red-500"),
        ".doc": ("description", "text-blue-600"),
        ".docx": ("description", "text-blue-600"),
        ".ppt": ("slideshow", "text-orange-600"),
        ".pptx": ("slideshow", "text-orange-600"),
        ".png": ("image", "text-fuchsia-400"),
        ".jpg": ("image", "text-fuchsia-400"),
        ".jpeg": ("image", "text-fuchsia-400"),
        ".gif": ("gif", "text-fuchsia-400"),
        ".webp": ("image", "text-fuchsia-400"),
        ".bmp": ("image", "text-fuchsia-400"),
        ".ico": ("image", "text-fuchsia-400"),
        ".svg": ("shape_line", "text-fuchsia-500"),
        ".mp3": ("audio_file", "text-indigo-400"),
        ".wav": ("audio_file", "text-indigo-400"),
        ".flac": ("audio_file", "text-indigo-400"),
        ".ogg": ("audio_file", "text-indigo-400"),
        ".mp4": ("movie", "text-indigo-500"),
        ".mkv": ("movie", "text-indigo-500"),
        ".avi": ("movie", "text-indigo-500"),
        ".mov": ("movie", "text-indigo-500"),
        ".webm": ("movie", "text-indigo-500"),
        ".zip": ("folder_zip", "text-amber-600"),
        ".tar": ("folder_zip", "text-amber-600"),
        ".gz": ("folder_zip", "text-amber-600"),
        ".bz2": ("folder_zip", "text-amber-600"),
        ".xz": ("folder_zip", "text-amber-600"),
        ".7z": ("folder_zip", "text-amber-600"),
        ".rar": ("folder_zip", "text-amber-600"),
        ".sh": ("terminal", "text-green-400"),
        ".bash": ("terminal", "text-green-400"),
        ".zsh": ("terminal", "text-green-400"),
        ".bat": ("terminal", "text-green-500"),
        ".cmd": ("terminal", "text-green-500"),
        ".ps1": ("terminal", "text-blue-400"),
        ".c": ("memory", "text-blue-300"),
        ".h": ("memory", "text-blue-300"),
        ".cpp": ("memory", "text-blue-400"),
        ".hpp": ("memory", "text-blue-400"),
        ".cs": ("memory", "text-violet-400"),
        ".java": ("coffee", "text-red-400"),
        ".go": ("bolt", "text-cyan-400"),
        ".rs": ("settings", "text-orange-400"),
        ".rb": ("diamond", "text-red-500"),
        ".php": ("php", "text-indigo-400"),
        ".sql": ("storage", "text-teal-400"),
        ".db": ("storage", "text-teal-500"),
        ".sqlite": ("storage", "text-teal-500"),
        ".sqlite3": ("storage", "text-teal-500"),
        ".lock": ("lock", "text-slate-500"),
    }

    # exact filename (lowercased) -> (icon, colour), checked before extension
    NAME_ICONS: Dict[str, tuple] = {
        "dockerfile": ("inventory_2", "text-blue-400"),
        "docker-compose.yml": ("inventory_2", "text-blue-400"),
        "makefile": ("build", "text-amber-500"),
        "readme.md": ("menu_book", "text-sky-300"),
        "license": ("gavel", "text-slate-400"),
        "requirements.txt": ("inventory", "text-yellow-500"),
        "pyproject.toml": ("inventory", "text-yellow-500"),
        "package.json": ("inventory", "text-red-400"),
        ".gitignore": ("visibility_off", "text-slate-500"),
    }

    def _icon_for(name: str, is_dir: bool, expanded: bool = False) -> tuple:
        if is_dir:
            return ("folder_open", "text-amber-400") if expanded else ("folder", "text-amber-400")
        lowered = name.lower()
        if lowered in NAME_ICONS:
            return NAME_ICONS[lowered]
        return EXT_ICONS.get(Path(lowered).suffix, ("description", "text-slate-400"))

    # rel_path -> list of child node dicts (populated on first expand)
    tree_children: Dict[str, list] = {}
    tree_expanded: set = set()
    tree_loaded_set: set = set()
    tree_loading: set = set()   # rel paths currently being scanned in the background
    _refreshing_tree: bool = False
    _pending_refresh: bool = False

    def toggle_tree_visibility():
        nonlocal show_tree_sidebar
        show_tree_sidebar = not show_tree_sidebar
        tree_panel.set_visibility(show_tree_sidebar)

    def _get_loaded_files_set() -> set:
        """Ask agent_bridge which files are currently FULL-visibility in context.
        Paths are normalised to forward slashes, relative to the workspace root."""
        try:
            session.ensure_ready()
            stats = agent_bridge.get_workspace_stats(session.personality)
            out = set()
            for f in stats.get("loaded_files", []):
                p = str(f.get("path", "")).replace("\\", "/").lstrip("./")
                if p:
                    out.add(p)
                    out.add(Path(p).name)
            return out
        except Exception:
            return set()

    def _should_skip(name: str) -> bool:
        if name in IGNORED_NAMES:
            return True
        if name.startswith(".") and name.lower() not in NAME_ICONS:
            return True
        return False

    def _scan_dir_sync(folder_str: str, root_str: str) -> list:
        """Runs in a worker thread (see _children_of_async). Plain os.scandir,
        never follows symlinked directories (the classic cause of a tree that
        hangs on a cyclic or dead symlink), capped at MAX_ENTRIES_PER_DIR, and
        never lets one bad entry (permission error, broken link) abort the scan."""
        import os
        root = Path(root_str)
        nodes = []
        try:
            with os.scandir(folder_str) as it:
                entries = sorted(it, key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except Exception as e:
            return [{"rel": "__err__", "name": f"(cannot read folder: {e})", "path": "", "is_dir": False, "error": True}]

        count = 0
        for entry in entries:
            if count >= MAX_ENTRIES_PER_DIR:
                nodes.append({"rel": "__more__", "name": f"… {len(entries) - count} more (truncated)",
                              "path": "", "is_dir": False, "error": True})
                break
            try:
                if entry.is_symlink():
                    continue  # never follow — avoids symlink-loop / dead-network-mount hangs
                if _should_skip(entry.name):
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                p = Path(entry.path)
                rel = str(p.relative_to(root)).replace("\\", "/")
                nodes.append({"rel": rel, "name": entry.name, "path": str(p), "is_dir": is_dir})
                count += 1
            except OSError:
                continue  # unreadable entry — skip rather than abort the whole listing
        return nodes

    async def _children_of_async(rel: str) -> list:
        """Lazily scans and caches a directory's children off the event loop,
        so a slow/huge folder shows a spinner instead of freezing the whole app
        for every connected user."""
        if rel in tree_children:
            return tree_children[rel]
        ws_root = Path(prefs.workspace_path).resolve()
        folder = ws_root if rel == "" else (ws_root / rel)
        nodes = await _nicegui_run.io_bound(_scan_dir_sync, str(folder), str(ws_root))
        tree_children[rel] = nodes
        return nodes

    def _children_of_cached(rel: str) -> list:
        return tree_children.get(rel, [])

    # ---- filesystem / OS actions ----

    def open_in_default_editor(abs_path: str):
        """Opens the file in its native software using the OS default application."""
        import os
        import subprocess
        import sys
        try:
            p = Path(abs_path).resolve()
            if not p.exists():
                ui.notify(f"File not found on disk: {p.name}", type="warning")
                return
            if sys.platform.startswith("win"):
                try:
                    os.startfile(str(p))  # type: ignore[attr-defined]
                except OSError:
                    try:
                        subprocess.Popen(["cmd.exe", "/c", "start", "", str(p)], shell=True)
                    except Exception:
                        subprocess.Popen(["notepad.exe", str(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
            ui.notify(f"Opening {p.name} in native software…", type="info", timeout=2000)
        except Exception as e:
            notify_error(f"Could not open {abs_path}: {e}")

    def reveal_in_file_manager(abs_path: str):
        import subprocess
        import sys
        try:
            p = Path(abs_path)
            target = p if p.is_dir() else p.parent
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as e:
            notify_error(f"Could not reveal {abs_path}: {e}")

    def _visibility_action(rel_paths: List[str], action: str):
        """load / unload / lock / hide / unhide via agent_bridge, then repaint."""
        if not rel_paths:
            return
        try:
            session.ensure_ready()
            result = agent_bridge.change_file_visibility(session.personality, rel_paths, action)
            status = result.get("status_str", f"{action} done.")
            add_system_notice(status, is_error=("❌" in status or "BLOCKED" in status))
        except Exception as e:
            notify_error(f"Could not {action} file(s): {e}")
        _repaint_after_visibility_change()

    def _repaint_after_visibility_change():
        nonlocal tree_loaded_set
        tree_loaded_set = _get_loaded_files_set()
        _paint_tree()

    async def _collect_files_under(rel: str) -> List[str]:
        """All file rel-paths beneath a directory, for 'Load whole folder'.
        Runs in a worker thread and never follows symlinks, so it can't loop
        on a cyclic link the way a naive rglob() can."""
        def _walk() -> List[str]:
            import os
            ws_root = Path(prefs.workspace_path).resolve()
            base = ws_root / rel if rel else ws_root
            out: List[str] = []
            stack = [str(base)]
            visited_dirs = set()
            while stack and len(out) < 200:
                current = stack.pop()
                if current in visited_dirs:
                    continue
                visited_dirs.add(current)
                try:
                    with os.scandir(current) as it:
                        for entry in it:
                            if entry.is_symlink():
                                continue
                            if _should_skip(entry.name):
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            else:
                                out.append(str(Path(entry.path).relative_to(ws_root)).replace("\\", "/"))
                                if len(out) >= 200:
                                    break
                except OSError:
                    continue
            return out

        return await _nicegui_run.io_bound(_walk)

    def _insert_into_prompt(rel: str):
        current_val = prompt_input.value or ""
        prompt_input.value = f"{current_val.rstrip()} {rel} " if current_val else f"{rel} "
        prompt_input.run_method("focus")

    def _copy_text(text: str, what: str = "Path"):
        try:
            ui.clipboard.write(text)
            ui.notify(f"{what} copied.", type="positive")
        except Exception as e:
            notify_error(f"Copy failed: {e}")

    def _file_meta(abs_path: str) -> str:
        try:
            st = Path(abs_path).stat()
            size = st.st_size
            unit = "B"
            for u in ("KB", "MB", "GB"):
                if size < 1024:
                    break
                size /= 1024.0
                unit = u
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            return f"{size:,.0f} {unit} · modified {mtime}"
        except Exception:
            return ""

    # ---- row rendering ----

    def _attach_context_menu(container, node: Dict[str, Any], is_loaded: bool):
        """Uses only ui.context_menu / ui.menu_item / ui.separator — the widely
        supported NiceGUI menu primitives — rather than lower-level QItem
        building blocks that vary across NiceGUI versions."""
        rel = node["rel"]
        abs_path = node["path"]
        with container:
            with ui.context_menu():
                if node["is_dir"]:
                    async def _load_folder(r=rel):
                        files = await _collect_files_under(r)
                        _visibility_action(files, "load")

                    async def _unload_folder(r=rel):
                        files = await _collect_files_under(r)
                        _visibility_action(files, "unload")

                    ui.menu_item("📥 Load all files into context [C]", _load_folder)
                    ui.menu_item("📤 Unload all files from context [U]", _unload_folder)
                    ui.separator()
                    ui.menu_item("🙈 Hide from tree", lambda r=rel: _visibility_action([r], "hide"))
                    ui.menu_item("👁️ Unhide", lambda r=rel: _visibility_action([r], "unhide"))
                    ui.separator()
                    ui.menu_item("📂 Open folder in native software", lambda p=abs_path: open_in_default_editor(p))
                    ui.menu_item("🗂️ Switch workspace here",
                                 lambda p=abs_path: _switch_workspace_from_tree(p))
                else:
                    if is_loaded:
                        ui.menu_item("📤 Unload from context [U]",
                                     lambda r=rel: _visibility_action([r], "unload"))
                    else:
                        ui.menu_item("📥 Load into context [C]",
                                     lambda r=rel: _visibility_action([r], "load"))
                    ui.menu_item("🔒 Lock in tree [L] (agent can't unload)",
                                 lambda r=rel: _visibility_action([r], "lock"))
                    ui.separator()
                    ui.menu_item("✏️ Open in native software",
                                 lambda p=abs_path: open_in_default_editor(p))
                    ui.menu_item("📁 Reveal in file manager",
                                 lambda p=abs_path: reveal_in_file_manager(p))
                    ui.separator()
                    ui.menu_item("💬 Mention in prompt", lambda r=rel: _insert_into_prompt(r))
                    ui.menu_item("🙈 Hide from tree", lambda r=rel: _visibility_action([r], "hide"))
                ui.separator()
                ui.menu_item("📋 Copy relative path", lambda r=rel: _copy_text(r, "Relative path"))
                ui.menu_item("📋 Copy absolute path", lambda p=abs_path: _copy_text(p, "Absolute path"))

    def _switch_workspace_from_tree(abs_path: str):
        try:
            session.ensure_ready()
            session.personality = agent_bridge.switch_workspace(prefs, session.client, abs_path)
            add_system_notice(f"📂 Workspace switched to `{prefs.workspace_path}`")
            tree_children.clear()
            tree_expanded.clear()
            refresh_workspace_tree()
        except Exception as e:
            notify_error(f"Could not switch workspace: {e}")

    def _row_classes(is_loaded: bool) -> str:
        base = (
            "w-full items-center gap-1 flex-nowrap rounded cursor-pointer select-none "
            "py-0.5 pr-1 hover:bg-slate-200/70 dark:hover:bg-slate-800/70"
        )
        if is_loaded:
            base += (
                " bg-emerald-500/10 border-l-2 border-emerald-500 "
                "dark:bg-emerald-400/10 dark:border-emerald-400"
            )
        return base

    def _render_nodes(nodes: list, depth: int):
        for node in nodes:
            try:
                _render_one_node(node, depth)
            except Exception as e:
                # One malformed entry must never blank out the rest of the tree.
                ui.label(f"⚠ {node.get('name', '?')} ({e})").classes("text-xs text-red-400 pl-2")

    def _render_one_node(node: Dict[str, Any], depth: int):
        if node.get("error"):
            ui.label(node["name"]).classes(f"text-xs {MUTED_DIM} pl-2 italic")
            return

        rel = node["rel"]
        is_dir = node["is_dir"]
        expanded = rel in tree_expanded
        is_loaded = (not is_dir) and (rel in tree_loaded_set or node["name"] in tree_loaded_set)
        icon, colour = _icon_for(node["name"], is_dir, expanded)

        row = ui.row().classes(_row_classes(is_loaded)).style(f"padding-left: {6 + depth * 12}px")
        with row:
            if is_dir:
                if rel in tree_loading:
                    ui.spinner(size="14px").classes("shrink-0")
                else:
                    ui.icon("chevron_right" if not expanded else "expand_more").classes(
                        "text-slate-500 shrink-0"
                    ).props("size=14px")
            else:
                ui.element("div").classes("shrink-0").style("width: 14px")
            ui.icon(icon).classes(f"{colour} shrink-0").props("size=16px")
            label = ui.label(node["name"]).classes(
                "text-xs truncate "
                + ("font-semibold " if is_loaded else "")
                + ("text-emerald-700 dark:text-emerald-300" if is_loaded else STRONG)
            )
            ui.element("div").classes("flex-1")
            if is_loaded:
                ui.icon("task_alt").classes("text-emerald-500 shrink-0").props("size=13px").tooltip(
                    "Loaded into the agent's context"
                )

        if not is_dir:
            label.tooltip(f"{rel}\n{_file_meta(node['path'])}")

        # Double click: open in native software. Single click on folder: expand/collapse.
        if is_dir:
            row.on("click", lambda r=rel: _toggle_dir(r))
        else:
            row.on("dblclick", lambda p=node["path"]: open_in_default_editor(p))

        _attach_context_menu(row, node, is_loaded)

        if is_dir and expanded:
            _render_nodes(_children_of_cached(rel), depth + 1)

    async def _toggle_dir(rel: str):
        if rel in tree_expanded:
            tree_expanded.discard(rel)
            _paint_tree()
            return
        tree_expanded.add(rel)
        if rel not in tree_children:
            tree_loading.add(rel)
            _paint_tree()  # show the spinner immediately
            try:
                await _children_of_async(rel)
            finally:
                tree_loading.discard(rel)
        _paint_tree()

    def _paint_tree():
        """Rebuilds the visible rows from cached state — no disk access unless a
        directory is being expanded for the first time (handled in _toggle_dir)."""
        tree_container.clear()
        with tree_container:
            q = (tree_search_input.value or "").lower().strip()
            if q:
                _render_search_results(q)
                return
            roots = _children_of_cached("")
            if not roots:
                ui.label("Loading…" if "" in tree_loading else "(Workspace empty)").classes(
                    f"text-xs {MUTED_DIM} p-2"
                )
                return
            _render_nodes(roots, 0)

    def _render_search_results(q: str):
        """Flat, capped result list built off the event loop — a filtered tree
        would otherwise hide where a match actually lives."""

        def _search() -> list:
            import os
            ws_root = Path(prefs.workspace_path).resolve()
            hits = []
            stack = [str(ws_root)]
            visited = set()
            while stack and len(hits) < MAX_SEARCH_HITS:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                try:
                    with os.scandir(current) as it:
                        for entry in it:
                            if entry.is_symlink() or _should_skip(entry.name):
                                continue
                            is_dir = entry.is_dir(follow_symlinks=False)
                            if is_dir:
                                stack.append(entry.path)
                            if q in entry.name.lower():
                                p = Path(entry.path)
                                hits.append({
                                    "rel": str(p.relative_to(ws_root)).replace("\\", "/"),
                                    "name": entry.name, "path": str(p), "is_dir": is_dir,
                                })
                                if len(hits) >= MAX_SEARCH_HITS:
                                    break
                except OSError:
                    continue
            return hits

        ui.label("Searching…").classes(f"text-xs {MUTED_DIM} p-2")

        async def _run_search():
            hits = await _nicegui_run.io_bound(_search)
            if (tree_search_input.value or "").lower().strip() != q:
                return  # user kept typing — a newer search will replace this
            tree_container.clear()
            with tree_container:
                if not hits:
                    ui.label("(no matches)").classes(f"text-xs {MUTED_DIM} p-2")
                    return
                ui.label(f"{len(hits)} match(es)").classes(f"text-[10px] {MUTED_DIM} px-2 pb-1")
                for node in hits:
                    _render_search_row(node)

        ui.timer(0.01, _run_search, once=True)

    def _render_search_row(node: Dict[str, Any]):
        is_loaded = node["rel"] in tree_loaded_set or node["name"] in tree_loaded_set
        icon, colour = _icon_for(node["name"], node["is_dir"])
        row = ui.row().classes(_row_classes(is_loaded) + " px-1.5")
        with row:
            ui.icon(icon).classes(f"{colour} shrink-0").props("size=16px")
            with ui.column().classes("gap-0 min-w-0 flex-1"):
                ui.label(node["name"]).classes(
                    "text-xs truncate "
                    + ("text-emerald-700 dark:text-emerald-300 font-semibold" if is_loaded else STRONG)
                )
                ui.label(node["rel"]).classes(f"text-[10px] truncate {MUTED_DIM}")
            if is_loaded:
                ui.icon("task_alt").classes("text-emerald-500 shrink-0").props("size=13px")
        if node["is_dir"]:
            row.on("click", lambda r=node["rel"]: _reveal_path_in_tree(r))
        else:
            row.on("dblclick", lambda p=node["path"]: open_in_default_editor(p))
        _attach_context_menu(row, node, is_loaded)

    def _reveal_path_in_tree(rel: str):
        """Clear the filter and expand every ancestor down to `rel`."""
        tree_search_input.value = ""
        parts = rel.split("/")

        async def _expand_ancestors():
            path_so_far = ""
            for part in parts[:-1]:
                path_so_far = f"{path_so_far}/{part}" if path_so_far else part
                tree_expanded.add(path_so_far)
                if path_so_far not in tree_children:
                    await _children_of_async(path_so_far)
            _paint_tree()

        ui.timer(0.01, _expand_ancestors, once=True)

    def refresh_workspace_tree():
        """Re-reads the loaded-file set from agent_bridge, drops the directory
        cache so on-disk changes show up, and repaints. Only root and
        currently expanded folders are re-scanned (off the event loop),
        preserving lazy loading and avoiding full-tree traversal bloat."""
        nonlocal _refreshing_tree, _pending_refresh, tree_loaded_set
        if _refreshing_tree:
            _pending_refresh = True
            return
        _refreshing_tree = True

        async def _do_refresh():
            nonlocal tree_loaded_set, _refreshing_tree, _pending_refresh
            try:
                tree_loaded_set = _get_loaded_files_set()
                tree_children.clear()

                ws_root = Path(prefs.workspace_path).resolve()
                still_expanded = {
                    rel for rel in tree_expanded
                    if (ws_root / rel).is_dir()
                }
                tree_expanded.clear()
                tree_expanded.update(still_expanded)

                tree_loading.add("")
                try:
                    await _children_of_async("")
                finally:
                    tree_loading.discard("")

                sorted_expanded = sorted(list(tree_expanded), key=lambda x: x.count('/'))
                for rel in sorted_expanded:
                    tree_loading.add(rel)
                    try:
                        await _children_of_async(rel)
                    finally:
                        tree_loading.discard(rel)

                _paint_tree()

                try:
                    refresh_subws_tree()
                except Exception:
                    pass
            finally:
                _refreshing_tree = False
                if _pending_refresh:
                    _pending_refresh = False
                    refresh_workspace_tree()

        ui.timer(0.01, _do_refresh, once=True)

    def apply_tree_filter():
        _paint_tree()

    tree_search_input.on_value_change(lambda e: apply_tree_filter())

    # ---- Sub-Workspace Interactive Dialog ----------------
    def open_sub_workspace_dialog():
        from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
        sub_ws = SubWorkspaceManager(prefs.workspace_path)

        dialog = ui.dialog()
        with dialog, ui.card().classes(
            f"w-[880px] max-w-[95vw] h-[640px] max-h-[92vh] flex flex-col p-4 gap-3 "
            f"{CANVAS} text-slate-900 dark:text-slate-100 rounded-xl shadow-2xl border {BORDER}"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("auto_stories", size="26px").classes("text-emerald-500")
                    with ui.column().classes("gap-0"):
                        ui.label("Sub-Workspace Manager").classes("text-base font-bold")
                        ui.label("Reference documentation and external files in .lollms_code/sub_workspace/").classes(f"text-xs {MUTED_DIM}")

                with ui.row().classes("items-center gap-1.5"):
                    async def do_import_file():
                        _picker = pick_file
                        if _picker:
                            chosen = await _picker(
                                title="Select Reference File to Import",
                                file_types=[("All Files", "*.*")]
                            )
                            if chosen:
                                try:
                                    p = Path(chosen)
                                    if p.is_dir():
                                        imported = sub_ws.import_folder(p)
                                        ui.notify(f"Imported folder with {len(imported)} reference file(s).", type="positive")
                                    else:
                                        dest = sub_ws.import_file(p)
                                        ui.notify(f"Imported reference: {dest.name}", type="positive")
                                    refresh_sub_ws_items()
                                    refresh_workspace_tree()
                                except Exception as ex:
                                    notify_error(f"Import file failed: {ex}")

                    async def do_import_folder():
                        _picker = pick_folder
                        if _picker:
                            chosen = await _picker(title="Select Folder to Import as Reference")
                            if chosen:
                                try:
                                    imported = sub_ws.import_folder(chosen)
                                    ui.notify(f"Imported {len(imported)} files into sub-workspace.", type="positive")
                                    refresh_sub_ws_items()
                                    refresh_workspace_tree()
                                except Exception as ex:
                                    notify_error(f"Import folder failed: {ex}")

                    def do_load_all():
                        cnt = sub_ws.load_all()
                        ui.notify(f"Loaded all {cnt} reference file(s) into context.", type="positive")
                        refresh_sub_ws_items()
                        refresh_workspace_tree()

                    def do_unload_all():
                        sub_ws.unload_all()
                        ui.notify("Unloaded all reference files from context.", type="info")
                        refresh_sub_ws_items()
                        refresh_workspace_tree()

                    ui.button("Paste Text", icon="note_add", on_click=lambda: open_paste_reference_dialog()).props(
                        "unelevated dense size=xs color=primary no-caps"
                    ).tooltip("Paste text directly as a new reference file")
                    ui.button("Import File", icon="upload_file", on_click=do_import_file).props("outline dense size=xs color=primary no-caps").tooltip("Import a reference file")
                    ui.button("Import Folder", icon="drive_folder_upload", on_click=do_import_folder).props("outline dense size=xs color=primary no-caps").tooltip("Import a folder into sub-workspace")
                    ui.button("Load All [C]", icon="download", on_click=do_load_all).props("flat dense size=xs color=emerald no-caps")
                    ui.button("Unload All", icon="clear_all", on_click=do_unload_all).props("flat dense size=xs color=amber no-caps")
                    ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

            sub_search_input = ui.input(placeholder="Search reference files…").props("dense outlined clearable").classes("w-full text-xs")

            sub_scroll = ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2")
            with sub_scroll:
                sub_list_container = ui.column().classes("w-full gap-2")

            def refresh_sub_ws_items():
                sub_list_container.clear()
                q = (sub_search_input.value or "").strip().lower()
                files = sub_ws.list_files(q)
                with sub_list_container:
                    if not files:
                        ui.label("No files in sub-workspace yet. Use 'Import File' or 'Import Folder' above.").classes(
                            f"text-xs {MUTED_DIM} italic p-4 text-center w-full"
                        )
                        return

                    for item in files:
                        rel = item["rel_path"]
                        is_loaded = item["is_loaded"]
                        size_str = f"{item['size'] / 1024:.1f} KB"
                        with ui.card().classes(
                            f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} hover:border-emerald-500/50 transition-colors gap-1.5 shadow-none"
                        ):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.row().classes("items-center gap-2 flex-1 min-w-0"):
                                    ui.icon("description", size="18px").classes("text-emerald-500 shrink-0")
                                    ui.label(rel).classes("text-xs font-mono font-semibold truncate text-slate-900 dark:text-slate-100")
                                    ui.label(size_str).classes(f"text-[10px] {MUTED_DIM} shrink-0")
                                    badge_color = "emerald" if is_loaded else "grey"
                                    ui.badge("[C] LOADED" if is_loaded else "[U] UNLOADED", color=badge_color).props("dense rounded").classes("text-[10px]")

                                with ui.row().classes("items-center gap-1 shrink-0"):
                                    def _toggle_load(r=rel, loaded=is_loaded):
                                        if loaded:
                                            sub_ws.unload_file(r)
                                            ui.notify(f"Unloaded {r}", type="info")
                                        else:
                                            sub_ws.load_file(r)
                                            ui.notify(f"Loaded {r} into context [C]", type="positive")
                                        refresh_sub_ws_items()
                                        refresh_workspace_tree()

                                    def _peek(r=rel):
                                        content = sub_ws.peek_file(r)
                                        peek_dlg = ui.dialog()
                                        with peek_dlg, ui.card().classes(f"w-[760px] max-w-[95vw] h-[550px] flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100 rounded-xl border {BORDER}"):
                                            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                                                ui.label(f"👁️ Peek: sub_workspace/{r}").classes("text-sm font-bold font-mono")
                                                ui.button(icon="close", on_click=peek_dlg.close).props("flat round dense size=xs")
                                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2 bg-slate-900 dark:bg-slate-950"):
                                                ui.code(content, language="markdown" if r.endswith(".md") else "text").classes("w-full text-xs")
                                        peek_dlg.open()

                                    def _delete_ref(r=rel):
                                        sub_ws.remove_path(r)
                                        ui.notify(f"Removed {r}", type="info")
                                        refresh_sub_ws_items()
                                        refresh_workspace_tree()

                                    ui.button("Unload" if is_loaded else "Load", icon="remove_circle_outline" if is_loaded else "check_circle_outline", on_click=_toggle_load).props(
                                        f"flat dense size=xs no-caps color={'amber' if is_loaded else 'emerald'}"
                                    )
                                    ui.button("Peek", icon="visibility", on_click=_peek).props("flat dense size=xs no-caps color=primary")
                                    ui.button(icon="delete", on_click=_delete_ref).props("flat dense round size=xs color=red")

            sub_search_input.on_value_change(lambda _: refresh_sub_ws_items())
            refresh_sub_ws_items()

        dialog.open()

    # ── Zoo Hub Modal Dialog (Tools, Skills, Personas) ─────────────────
    def open_zoo_dialog():
        from lollms_client.apps.lollms_code.zoo import ZooManager
        zm = ZooManager(prefs.workspace_path)

        dialog = ui.dialog().props("maximized")
        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100 gap-3"
        ):
            # Header
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("pets", size="26px").classes("text-amber-500")
                    with ui.column().classes("gap-0"):
                        ui.label("LoLLMS Zoo Package Hub").classes("text-base font-bold")
                        ui.label("Browse categorized Tools, Skills, and Personas across categories and subcategories.").classes(
                            f"text-xs {MUTED_DIM}"
                        )

                with ui.row().classes("items-center gap-2"):
                    def _sync_all_dialog():
                        ui.notify("Synchronizing all zoos from GitHub...", type="info")
                        results = zm.sync_all()
                        for z, (ok, msg) in results.items():
                            if ok:
                                ui.notify(f"✓ {z.capitalize()}: {msg}", type="positive")
                            else:
                                ui.notify(f"⚠ {z.capitalize()}: {msg}", type="warning")
                        _refresh_zoo_view()

                    ui.button("Sync Zoos", icon="sync", on_click=_sync_all_dialog).props(
                        "unelevated dense size=sm color=primary no-caps"
                    ).tooltip("Clone or update all three zoo repositories from GitHub")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

            # Main Body: Tab switcher & Category + Items layout
            with ui.tabs().classes(f"w-full {SURFACE} border-b {BORDER} shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as zoo_tabs:
                tab_tools = ui.tab('tools', label='🛠️ Tools Zoo', icon='build').classes('text-xs py-1.5 flex-1')
                tab_skills = ui.tab('skills', label='🧠 Skills Zoo', icon='psychology').classes('text-xs py-1.5 flex-1')
                tab_personas = ui.tab('personalities', label='🎭 Personalities Zoo', icon='face').classes('text-xs py-1.5 flex-1')

            active_zoo_type = {"type": "tools"}
            active_cat = {"name": "", "label": ""}

            with ui.row().classes("w-full flex-1 min-h-0 items-stretch overflow-hidden flex-nowrap gap-3 pt-2"):
                # Left Pane: Categories with Subcategories
                cat_sidebar = ui.column().classes(f"w-72 h-full shrink-0 border-r {BORDER} {SURFACE_ALT} p-2 overflow-y-auto gap-0.5")

                # Center/Right Pane: Search, Items & Documentation Preview
                with ui.column().classes("flex-1 h-full min-w-0 flex flex-col gap-2 overflow-hidden"):
                    with ui.row().classes("w-full items-center justify-between gap-2 shrink-0"):
                        zoo_search_input = ui.input(placeholder="Search packages by name, description, tags…").props(
                            ':dark="Quasar.Dark.isActive" dense outlined clearable'
                        ).classes("flex-1 text-xs bg-slate-50 dark:bg-slate-900")

                        cat_readme_btn = ui.button("Category Documentation", icon="menu_book", on_click=lambda: _show_cat_readme()).props(
                            "flat dense size=sm no-caps color=primary"
                        )
                        cat_readme_btn.visible = False

                    items_scroll = ui.scroll_area().classes("w-full flex-1 min-h-0")
                    with items_scroll:
                        items_container = ui.column().classes("w-full gap-3 p-1")

            def _show_cat_readme():
                c_name = active_cat["name"]
                z_type = active_zoo_type["type"]
                cats = zm.list_categories(z_type)
                match = next((c for c in cats if c.full_path == c_name or c.name == c_name), None)
                if match and match.readme_content:
                    r_dlg = ui.dialog()
                    with r_dlg, ui.card().classes(f"w-[780px] max-w-[95vw] h-[580px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER}"):
                        with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                            ui.label(f"📖 Category Documentation: {match.full_path}").classes("text-sm font-bold")
                            ui.button(icon="close", on_click=r_dlg.close).props("flat dense round size=xs")
                        with ui.scroll_area().classes("w-full flex-1 p-2"):
                            ui.markdown(match.readme_content).classes("text-xs leading-relaxed")
                    r_dlg.open()
                else:
                    ui.notify(f"No README.md found for category '{c_name}'.", type="info")

            def _refresh_zoo_view():
                z_type = active_zoo_type["type"]
                cat_sidebar.clear()
                categories = zm.list_categories(z_type)

                if not zm.is_repo_cloned(z_type):
                    with cat_sidebar:
                        ui.label(f"{z_type.capitalize()} Zoo not downloaded.").classes(f"text-xs {MUTED_DIM} p-2")
                        def _do_clone_now():
                            ui.notify(f"Cloning {z_type} zoo...", type="info")
                            ok, msg = zm.sync_repo(z_type)
                            if ok:
                                ui.notify(msg, type="positive")
                                _refresh_zoo_view()
                            else:
                                ui.notify(msg, type="negative")
                        ui.button("Clone Repo", icon="download", on_click=_do_clone_now).props("unelevated dense size=xs color=primary no-caps")
                    items_container.clear()
                    return

                with cat_sidebar:
                    ui.label("CATEGORIES & SUBCATEGORIES").classes(f"text-[10px] font-bold text-slate-500 px-1 pt-1 mb-1")

                    def _select_cat(cat_path="", cat_label=""):
                        active_cat["name"] = cat_path
                        active_cat["label"] = cat_label
                        # Check if selected category has documentation
                        cats = zm.list_categories(active_zoo_type["type"])
                        match = next((c for c in cats if c.full_path == cat_path), None)
                        cat_readme_btn.visible = bool(match and match.readme_content)
                        if cat_readme_btn.visible:
                            cat_readme_btn.text = f"Docs: {match.name}"
                        _refresh_items_view()
                        _refresh_cat_buttons()

                    def _refresh_cat_buttons():
                        for btn_obj, c_id in cat_btn_refs:
                            is_sel = (c_id == active_cat["name"])
                            btn_obj.classes(
                                replace="w-full justify-start text-xs rounded transition-colors " +
                                        ("bg-primary/20 text-primary font-bold border-l-2 border-primary" if is_sel else "hover:bg-slate-200 dark:hover:bg-slate-800")
                            )

                    cat_btn_refs = []
                    all_count = sum(c.items_count for c in categories if not c.is_subcategory)
                    all_btn = ui.button(f"All Categories ({all_count})", on_click=lambda: _select_cat("", "All")).props("flat dense no-caps")
                    cat_btn_refs.append((all_btn, ""))

                    for c in categories:
                        indent = "pl-4 text-[11px] " if c.is_subcategory else "font-semibold "
                        icon_prefix = "↳ " if c.is_subcategory else "📁 "
                        btn = ui.button(
                            f"{icon_prefix}{c.name} ({c.items_count})",
                            on_click=lambda cp=c.full_path, cl=c.name: _select_cat(cp, cl)
                        ).props("flat dense no-caps").classes(f"{indent}")
                        cat_btn_refs.append((btn, c.full_path))

                    _refresh_cat_buttons()

                _refresh_items_view()

            def _refresh_items_view():
                items_container.clear()
                z_type = active_zoo_type["type"]
                q = (zoo_search_input.value or "").strip().lower()
                c_name = active_cat["name"]

                if q:
                    items = zm.search(q, zoo_type=z_type)
                else:
                    items = zm.list_items(z_type, category=c_name or None)

                with items_container:
                    if not items:
                        ui.label("No packages found matching the selected filters.").classes(
                            f"text-xs {MUTED_DIM} p-4 italic text-center w-full"
                        )
                        return

                    for it in items:
                        with ui.card().classes(
                            f"w-full p-3 rounded-lg border {BORDER} {SURFACE} hover:border-primary/50 transition-colors gap-2 shadow-none"
                        ):
                            with ui.row().classes("w-full items-start justify-between flex-nowrap"):
                                with ui.column().classes("gap-0.5 flex-1 min-w-0"):
                                    with ui.row().classes("items-center gap-2"):
                                        item_icon = "build" if it.zoo_type == "tools" else ("psychology" if it.zoo_type == "skills" else "face")
                                        ui.icon(item_icon, size="18px").classes("text-primary shrink-0")
                                        ui.label(it.name).classes("text-sm font-bold truncate text-slate-900 dark:text-slate-100")
                                        ui.label(f"in {it.category}").classes(f"text-[10px] {MUTED_DIM} font-mono")

                                    ui.label(it.description).classes(f"text-xs {MUTED} line-clamp-2")

                                with ui.row().classes("items-center gap-1 shrink-0"):
                                    if it.is_installed_project:
                                        ui.badge("Project", color="emerald").props("dense rounded text-[10px]").tooltip("Installed in active project (.lollms_code/)")
                                    if it.is_installed_global:
                                        ui.badge("Global", color="indigo").props("dense rounded text-[10px]").tooltip("Installed globally (~/.lollms_client/)")

                            # Actions Row
                            with ui.row().classes("w-full items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800"):
                                with ui.row().classes("items-center gap-1.5"):
                                    def _show_doc(target=it):
                                        d_dlg = ui.dialog()
                                        with d_dlg, ui.card().classes(f"w-[780px] max-w-[95vw] h-[580px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER}"):
                                            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                                                ui.label(f"📖 {target.name} ({target.category})").classes("text-sm font-bold")
                                                ui.button(icon="close", on_click=d_dlg.close).props("flat dense round size=xs")
                                            with ui.scroll_area().classes("w-full flex-1 p-2"):
                                                content = target.readme_content or "No documentation provided."
                                                ui.markdown(content).classes("text-xs leading-relaxed")
                                        d_dlg.open()

                                    ui.button("Read Docs", icon="menu_book", on_click=_show_doc).props("flat dense size=xs no-caps text-color=primary")

                                    if it.zoo_type == "personalities" and it.is_installed:
                                        def _activate_p(target=it):
                                            prefs.handbag_path = str((zm.get_project_target_dir("personalities") / target.name).resolve() if target.is_installed_project else (zm.get_global_target_dir("personalities") / target.name).resolve())
                                            prefs.save()
                                            try:
                                                session.personality = agent_bridge.create_personality(prefs, session.client)
                                                ui.notify(f"Active persona switched to '{session.personality.name}'!", type="positive")
                                            except Exception as ex:
                                                ui.notify(f"Failed to switch persona: {ex}", type="negative")

                                        ui.button("Use Persona", icon="play_arrow", on_click=_activate_p).props("unelevated dense size=xs color=purple no-caps font-semibold")

                                with ui.row().classes("items-center gap-1"):
                                    def _install(target=it, scope="project"):
                                        ok, msg = zm.install_item(target, scope=scope)
                                        if ok:
                                            ui.notify(msg, type="positive")
                                            # Reload in-memory components if active
                                            if target.zoo_type == "skills" and session.personality and session.personality.skills_manager:
                                                session.personality.skills_manager.reload()
                                            elif target.zoo_type == "tools" and session.client and session.client.tools:
                                                if hasattr(session.client.tools, "_discover_local_tools"):
                                                    session.client.tools._discover_local_tools()
                                            _refresh_items_view()
                                            refresh_subws_panel()
                                        else:
                                            ui.notify(msg, type="negative")

                                    def _uninstall(target=it, scope="project"):
                                        ok, msg = zm.uninstall_item(target, scope=scope)
                                        if ok:
                                            ui.notify(msg, type="info")
                                            if target.zoo_type == "skills" and session.personality and session.personality.skills_manager:
                                                session.personality.skills_manager.reload()
                                            elif target.zoo_type == "tools" and session.client and session.client.tools:
                                                if hasattr(session.client.tools, "_discover_local_tools"):
                                                    session.client.tools._discover_local_tools()
                                            _refresh_items_view()
                                            refresh_subws_panel()
                                        else:
                                            ui.notify(msg, type="negative")

                                    if not it.is_installed_project:
                                        ui.button("+ Project", on_click=lambda t=it: _install(t, "project")).props("unelevated dense size=xs color=primary no-caps").tooltip("Install into current project (.lollms_code/)")
                                    else:
                                        ui.button("Remove (Proj)", on_click=lambda t=it: _uninstall(t, "project")).props("flat dense size=xs color=red no-caps")

                                    if not it.is_installed_global:
                                        ui.button("+ Global", on_click=lambda t=it: _install(t, "global")).props("outline dense size=xs color=primary no-caps").tooltip("Install globally (~/.lollms_client/)")
                                    else:
                                        ui.button("Remove (Glob)", on_click=lambda t=it: _uninstall(t, "global")).props("flat dense size=xs color=red no-caps")

            zoo_tabs.on('update:model-value', lambda e: (active_zoo_type.update({"type": e.args}), active_cat.update({"name": "", "label": ""}), _refresh_zoo_view()))
            zoo_search_input.on_value_change(lambda _: _refresh_items_view())

            _refresh_zoo_view()

        dialog.open()

    # ── Paste Text as Reference Dialog ────────────────────────────────
    def open_paste_reference_dialog(initial_filename: str = "reference.md", initial_text: str = ""):
        from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
        sub_ws = SubWorkspaceManager(prefs.workspace_path)

        d = ui.dialog().props("persistent")
        with d, ui.card().classes(f"w-[760px] max-w-[95vw] h-[580px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-3"):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("post_add", size="24px").classes("text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label("New Reference Document").classes("text-sm font-bold text-slate-900 dark:text-slate-100")
                        ui.label("Paste text or Markdown into .lollms_code/sub_workspace/").classes(f"text-[10px] {MUTED_DIM}")
                ui.button(icon="close", on_click=d.close).props("flat dense round size=xs")

            with ui.row().classes("w-full items-center gap-3 shrink-0"):
                filename_input = ui.input(
                    "Reference Filename",
                    value=initial_filename,
                    placeholder="e.g. api_spec.md, notes.txt, design.md"
                ).classes("flex-1 text-xs").props("outlined dense")
                load_now_check = ui.checkbox("Load into context [C] immediately", value=True).props("dense").tooltip("Automatically mark as [C] so the agent sees this content")

            with ui.column().classes("w-full flex-1 min-h-0 gap-1"):
                ui.label("Content:").classes(f"text-xs font-semibold {STRONG}")
                content_input = ui.textarea(
                    placeholder="Paste your documentation, requirements, code snippets, or notes here…"
                ).classes(f"w-full flex-1 text-xs font-mono {SURFACE} rounded border {BORDER}").props(
                    ':dark="Quasar.Dark.isActive" outlined autogrow rows=12'
                )
                if initial_text:
                    content_input.value = initial_text

            with ui.row().classes(f"w-full items-center justify-between pt-2 border-t {BORDER} shrink-0"):
                ui.button("Cancel", on_click=d.close).props("flat dense no-caps")

                def _do_save():
                    raw_fn = (filename_input.value or "").strip()
                    if not raw_fn:
                        raw_fn = "reference.md"
                    if "." not in raw_fn:
                        raw_fn = f"{raw_fn}.md"

                    raw_text = content_input.value or ""
                    if not raw_text.strip():
                        ui.notify("Reference content cannot be empty.", type="warning")
                        return

                    try:
                        dest = sub_ws.save_text_file(raw_fn, raw_text)
                        if load_now_check.value:
                            sub_ws.load_file(raw_fn)
                            ui.notify(f"✓ Saved & loaded: sub_workspace/{raw_fn} [C]", type="positive")
                        else:
                            ui.notify(f"✓ Saved: sub_workspace/{raw_fn} [U]", type="positive")
                        d.close()
                        refresh_subws_panel()
                    except Exception as ex:
                        notify_error(f"Failed to save reference: {ex}")

                ui.button("Save Reference", icon="check", on_click=_do_save).props(
                    "unelevated dense color=primary no-caps"
                )

        d.open()

    # ── Sub-Workspace Hub: Subscribed Persona, Tools, Skills & Reference Files ──

    def refresh_subws_panel():
        """Refreshes the entire Sub-WS sidebar panel, separating Handbag from Project Extra assets."""
        try:
            session.ensure_ready()
        except Exception:
            pass

        data = agent_bridge.get_subws_tools_and_skills(session.personality, prefs, session.client)
        from lollms_client.apps.lollms_code.sub_workspace import SubWorkspaceManager
        sub_ws = SubWorkspaceManager(prefs.workspace_path)
        ref_files = sub_ws.list_files()

        # Update badge count
        p_tools_cnt = len(data["tools"]["project"])
        p_skills_cnt = len(data["skills"]["project"])
        ref_loaded_cnt = sum(1 for f in ref_files if f["is_loaded"])
        total_subscribed = p_tools_cnt + p_skills_cnt + ref_loaded_cnt

        if total_subscribed > 0:
            subws_tab_badge.set_text(str(total_subscribed))
            subws_tab_badge.visible = True
        else:
            subws_tab_badge.visible = False

        subws_tree_container.clear()

        with subws_tree_container:
            # ── 1. ACTIVE PERSONA / HANDBAG SECTION ──
            persona_info = data["persona"]
            src_label_map = {
                "handbag": ("Handbag", "purple"),
                "project": ("Project Extra", "emerald"),
                "global": ("Global Machine", "indigo"),
                "default": ("Default Coder", "blue"),
            }
            src_txt, src_col = src_label_map.get(persona_info["source"], ("Handbag", "purple"))

            with ui.card().classes(f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} gap-1 shadow-none mb-1"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-1.5"):
                        ui.icon("face", size="18px").classes("text-purple-500")
                        ui.label(persona_info["name"]).classes("text-xs font-bold truncate text-slate-900 dark:text-slate-100")
                    ui.badge(src_txt, color=src_col).props("dense rounded text-[9px]")

                if persona_info["description"]:
                    ui.label(persona_info["description"]).classes(f"text-[10px] {MUTED} line-clamp-1 italic")

                with ui.row().classes("w-full items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800"):
                    def _view_soul():
                        soul_text = persona_info["soul_content"] or f"You are {persona_info['name']}."

                        from lollms_client.lollms_personality import PersonalityBundle
                        meta, prompt_body = PersonalityBundle.parse_soul_md(soul_text)

                        d = ui.dialog()
                        with d, ui.card().classes(f"w-[760px] max-w-[95vw] h-[580px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-2"):
                            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER}"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("theater_masks", size="22px").classes("text-purple-500")
                                    ui.label(f"Persona SOUL: {meta.get('name') or persona_info['name']}").classes("text-sm font-bold")
                                ui.button(icon="close", on_click=d.close).props("flat dense round size=xs")

                            # Metadata summary row (cleanly styled without Markdown Setext heading artifacts)
                            with ui.row().classes("w-full items-center gap-2 px-2 py-1.5 bg-slate-100 dark:bg-slate-900 rounded border border-slate-200 dark:border-slate-800 flex-wrap text-xs"):
                                ui.label(f"Author: {meta.get('author', 'ParisNeo')}").classes("font-semibold text-slate-700 dark:text-slate-300")
                                ui.label("·").classes("text-slate-400")
                                ui.label(f"Category: {meta.get('category', persona_info['category'])}").classes("text-purple-600 dark:text-purple-400 font-mono")
                                if meta.get("description"):
                                    ui.label("·").classes("text-slate-400")
                                    ui.label(meta.get("description")).classes("text-[11px] text-slate-500 dark:text-slate-400 italic truncate max-w-sm")

                            with ui.scroll_area().classes("w-full flex-1 p-3 bg-white dark:bg-slate-950 rounded border border-slate-200 dark:border-slate-800"):
                                ui.markdown(prompt_body.strip()).classes("text-xs leading-relaxed text-slate-900 dark:text-slate-100")

                        d.open()

                    ui.button("View SOUL", icon="visibility", on_click=_view_soul).props("flat dense size=xs no-caps text-color=purple")

                    # Switch Persona Dropdown
                    handbags_list = data["available_handbags"]
                    if len(handbags_list) > 1:
                        def _switch_p(e):
                            val = e.value
                            if val:
                                try:
                                    session.personality = agent_bridge.switch_persona_handbag(prefs, session.client, val)
                                    ui.notify(f"Switched persona to: {session.personality.name}", type="positive")
                                    refresh_subws_panel()
                                except Exception as err:
                                    notify_error(f"Switch persona failed: {err}")

                        hb_options = {h["path"]: f"{h['title']} ({h['scope']})" for h in handbags_list}
                        curr_val = persona_info["handbag_path"]
                        hb_sel = ui.select(hb_options, value=curr_val if curr_val in hb_options else None).classes("w-32 text-[10px]").props("dense options-dense")
                        hb_sel.on_value_change(_switch_p)

            # ── 2. TOOLS SECTION (Handbag vs Project Extra) ──
            h_tools = data["tools"]["handbag"]
            p_tools = data["tools"]["project"]
            b_tools = data["tools"]["builtin"]

            def _view_tool_content(t_item: Dict[str, Any]):
                title = t_item.get("name", "Tool")
                desc = t_item.get("description", "(No description)")
                src_path = t_item.get("source_file") or ""
                params = t_item.get("parameters", [])

                # ── Resilient Source Path Resolution ──
                if not src_path or not Path(src_path).exists():
                    # 1. Check LCP discovered tools
                    if session.client and session.client.tools and hasattr(session.client.tools, "discovered_tools"):
                        match = next((t for t in session.client.tools.discovered_tools if t.get("name") == title), None)
                        if match and match.get("_python_file_path") and Path(match["_python_file_path"]).exists():
                            src_path = match["_python_file_path"]

                    # 2. Check default tools path in lollms_client
                    if not src_path or not Path(src_path).exists():
                        import lollms_client
                        default_tools_base = Path(lollms_client.__file__).resolve().parent / "tools_bindings" / "lcp" / "default_tools"
                        clean_stem = title[5:] if title.startswith("tool_") else title
                        candidates = [
                            default_tools_base / clean_stem / f"{clean_stem}.py",
                            default_tools_base / "document_editor" / "document_editor.py",
                            default_tools_base / "as_is_document_tools" / "as_is_document_tools.py",
                            default_tools_base / "execute_python" / "execute_python.py",
                            default_tools_base / "system_shell" / "system_shell.py",
                            default_tools_base / "workspace_tools" / "workspace_tools.py",
                            default_tools_base / "git_manager" / "git_manager.py",
                            Path(lollms_client.__file__).resolve().parent / "lollms_agentic" / "spinoff_tools.py",
                        ]
                        for cand in candidates:
                            if cand.exists():
                                try:
                                    content = cand.read_text(encoding="utf-8", errors="ignore")
                                    if title in content:
                                        src_path = str(cand.resolve())
                                        break
                                except Exception:
                                    pass

                # Extract Documentation (README.md) and Python code
                doc_text = ""
                code_text = ""

                if src_path and Path(src_path).exists():
                    p_file = Path(src_path)
                    try:
                        if p_file.is_file():
                            code_text = p_file.read_text(encoding="utf-8", errors="ignore")
                            # Check sibling README.md
                            sibling_readme = p_file.parent / "README.md"
                            if sibling_readme.exists():
                                doc_text = sibling_readme.read_text(encoding="utf-8", errors="ignore")
                        elif p_file.is_dir():
                            readme = p_file / "README.md"
                            if readme.exists():
                                doc_text = readme.read_text(encoding="utf-8", errors="ignore")
                            for py_f in p_file.glob("*.py"):
                                if py_f.name != "__init__.py":
                                    code_text = py_f.read_text(encoding="utf-8", errors="ignore")
                                    break
                    except Exception as err:
                        code_text = f"Error reading file: {err}"

                d = ui.dialog()
                with d, ui.card().classes(f"w-[800px] max-w-[95vw] h-[600px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-2"):
                    with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("construction", size="22px").classes("text-primary")
                            ui.label(f"Tool Specification: {title}").classes("text-sm font-bold text-slate-900 dark:text-slate-100")
                        ui.button(icon="close", on_click=d.close).props("flat dense round size=xs")

                    ui.label(desc).classes(f"text-xs {MUTED} italic shrink-0 px-1")
                    if src_path:
                        ui.label(f"Source: {src_path}").classes(f"text-[10px] {MUTED_DIM} font-mono truncate px-1 shrink-0")

                    # Tab switcher between Documentation, Python Code, and Parameters
                    has_both = bool(doc_text and code_text)
                    default_tab = 'doc' if doc_text else 'code'

                    with ui.tabs().classes(f"w-full {SURFACE} border-b {BORDER} shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as tool_tabs:
                        if doc_text:
                            tab_doc = ui.tab('doc', label='📖 Documentation', icon='menu_book').classes('text-xs py-1 flex-1')
                        tab_code = ui.tab('code', label='🐍 Python Source Code', icon='code').classes('text-xs py-1 flex-1')
                        if params:
                            tab_params = ui.tab('params', label=f'⚙️ Parameters ({len(params)})', icon='tune').classes('text-xs py-1 flex-1')

                    with ui.tab_panels(tool_tabs, value=default_tab).classes('w-full flex-1 min-h-0 p-0 bg-transparent flex flex-col overflow-hidden'):
                        if doc_text:
                            with ui.tab_panel('doc').classes('w-full h-full p-2 flex flex-col overflow-hidden'):
                                with ui.scroll_area().classes(f"w-full flex-1 p-3 {SURFACE} rounded border {BORDER}"):
                                    ui.markdown(doc_text).classes("text-xs leading-relaxed text-slate-900 dark:text-slate-100")

                        with ui.tab_panel('code').classes('w-full h-full p-2 flex flex-col overflow-hidden'):
                            with ui.scroll_area().classes(f"w-full flex-1 p-2 {SURFACE} rounded border {BORDER}"):
                                ui.code(code_text or "(No Python source code directly attached)", language="python").classes("w-full text-xs")

                        if params:
                            with ui.tab_panel('params').classes('w-full h-full p-2 flex flex-col overflow-hidden'):
                                with ui.scroll_area().classes(f"w-full flex-1 p-3 {SURFACE} rounded border {BORDER}"):
                                    ui.label("PARAMETER SCHEMA:").classes(f"text-[10px] font-bold text-slate-500 mb-2")
                                    for p in params:
                                        with ui.column().classes(f"w-full p-2 rounded border {BORDER} bg-white dark:bg-slate-900 mb-1.5 gap-0.5"):
                                            with ui.row().classes("items-center gap-2"):
                                                ui.label(p.get("name", "param")).classes("text-xs font-mono font-bold text-primary")
                                                ui.badge(p.get("type", "string"), color="slate").props("dense rounded text-[9px]")
                                                if p.get("optional"):
                                                    ui.badge("optional", color="grey").props("dense rounded text-[9px]")
                                            if p.get("description"):
                                                ui.label(p["description"]).classes(f"text-[11px] {MUTED} pl-1")
                d.open()

            with ui.expansion(f"🧰 Tools (H:{len(h_tools)} | P:{len(p_tools)})", icon="build").classes(
                f"w-full border {BORDER} rounded-lg {SURFACE} mb-1"
            ).props('header-class="py-1 px-2 text-xs font-bold text-slate-900 dark:text-slate-100 flex-nowrap"'):
                with ui.column().classes("w-full gap-2 p-1.5"):
                    # 1. Project Extra Tools (.lollms_code/tools/) — HAS REMOVE / UNINSTALL BUTTON
                    with ui.column().classes("w-full gap-1"):
                        ui.label(f"📁 PROJECT EXTRA TOOLS ({len(p_tools)})").classes("text-[10px] font-bold text-emerald-600 dark:text-emerald-400")
                        if not p_tools:
                            ui.label("(No project tools in .lollms_code/tools)").classes(f"text-[10px] {MUTED_DIM} italic pl-1")
                        for t in p_tools:
                            with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0 flex-nowrap"):
                                    ui.icon("build_circle", size="14px").classes("text-emerald-500 shrink-0")
                                    ui.label(t["name"]).classes("text-xs font-mono font-semibold truncate text-slate-900 dark:text-slate-100 max-w-[120px]").tooltip(f"{t['name']}\n{t.get('description', '')}")
                                with ui.row().classes("items-center gap-1 shrink-0 flex-nowrap"):
                                    ui.badge("Project", color="emerald").props("dense rounded text-[9px]")
                                    ui.button(icon="visibility", on_click=lambda item=t: _view_tool_content(item)).props("flat dense round size=xs color=primary").tooltip("View Tool Code / Documentation")

                                    def _confirm_delete_tool(t_item=t):
                                        dlg = ui.dialog()
                                        with dlg, ui.card().classes(f"w-[420px] p-4 gap-3 bg-white dark:bg-slate-900 rounded-xl border {BORDER}"):
                                            ui.label("Remove Tool from Project?").classes("text-sm font-bold text-red-500")
                                            ui.label(f"Are you sure you want to remove '{t_item['name']}' from this project's .lollms_code/tools folder?").classes("text-xs text-slate-600 dark:text-slate-300")
                                            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                                                ui.button("Cancel", on_click=dlg.close).props("flat dense")
                                                def _do_remove():
                                                    dlg.close()
                                                    p_to_del = Path(t_item.get("source_file", ""))
                                                    if p_to_del.exists():
                                                        try:
                                                            if p_to_del.is_dir():
                                                                import shutil
                                                                shutil.rmtree(str(p_to_del))
                                                            else:
                                                                p_to_del.unlink()
                                                            if session.client and session.client.tools and hasattr(session.client.tools, "_discover_local_tools"):
                                                                session.client.tools._discover_local_tools()
                                                            ui.notify(f"Removed tool '{t_item['name']}' from project", type="info")
                                                            refresh_subws_panel()
                                                        except Exception as err:
                                                            notify_error(f"Failed to delete tool: {err}")
                                                ui.button("Remove from Project", on_click=_do_remove).props("unelevated dense color=red no-caps")
                                        dlg.open()

                                    ui.button(icon="delete", on_click=_confirm_delete_tool).props("flat dense round size=xs color=red").tooltip("Remove tool from project (.lollms_code/tools/)")

                    # 2. Handbag Tools (PROTECTED - VIEW ONLY, NO DELETE BUTTON)
                    with ui.column().classes("w-full gap-1 pt-1 border-t border-slate-200 dark:border-slate-800"):
                        ui.label(f"👜 HANDBAG TOOLS ({len(h_tools)})").classes("text-[10px] font-bold text-purple-600 dark:text-purple-400")
                        if not h_tools:
                            ui.label("(None bundled in active handbag)").classes(f"text-[10px] {MUTED_DIM} italic pl-1")
                        for t in h_tools:
                            with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0 flex-nowrap"):
                                    ui.icon("construction", size="14px").classes("text-purple-500 shrink-0")
                                    ui.label(t["name"]).classes("text-xs font-mono font-semibold truncate text-slate-900 dark:text-slate-100 max-w-[130px]").tooltip(f"{t['name']}\n{t.get('description', '')}")
                                with ui.row().classes("items-center gap-1 shrink-0 flex-nowrap"):
                                    ui.badge("Handbag", color="purple").props("dense rounded text-[9px]").tooltip("Handbag native tool (read-only, cannot be removed from project)")
                                    ui.button(icon="visibility", on_click=lambda item=t: _view_tool_content(item)).props("flat dense round size=xs color=purple").tooltip("View Tool Code / Info")

                    # 3. Built-in & System Tools (PROTECTED - VIEW ONLY)
                    if b_tools:
                        with ui.expansion(f"⚙️ System & LCP Built-ins ({len(b_tools)})", icon="settings").classes(
                            f"w-full border border-slate-200 dark:border-slate-800 rounded bg-slate-100/50 dark:bg-slate-900/50"
                        ).props('header-class="py-0.5 px-1.5 text-[10px] font-bold text-slate-500 flex-nowrap"'):
                            with ui.column().classes("w-full gap-0.5 p-1"):
                                for t in b_tools:
                                    with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                        with ui.row().classes("items-center gap-1 flex-1 min-w-0 flex-nowrap"):
                                            ui.icon("memory", size="13px").classes("text-slate-400 shrink-0")
                                            ui.label(t["name"]).classes("text-[11px] font-mono truncate text-slate-700 dark:text-slate-300 max-w-[130px]").tooltip(f"{t['name']}\n{t.get('description', '')}")
                                        ui.button(icon="visibility", on_click=lambda item=t: _view_tool_content(item)).props("flat dense round size=xs color=grey").tooltip("View Built-in Tool Specification")

                    # Add tools from zoo shortcut
                    with ui.row().classes("w-full justify-end pt-1"):
                        ui.button("+ Add Tool from Zoo", icon="add", on_click=lambda: open_zoo_dialog()).props("flat dense size=xs color=primary no-caps")

            # ── 3. SKILLS SECTION (Handbag vs Project Extra vs Global) ──
            h_skills = data["skills"]["handbag"]
            p_skills = data["skills"]["project"]
            o_skills = data["skills"].get("other", [])
            total_skills_count = len(h_skills) + len(p_skills) + len(o_skills)

            def _parse_skill_markdown(raw_text: str) -> tuple[Dict[str, str], str]:
                """Extracts YAML frontmatter cleanly and returns (metadata_dict, stripped_body_text)."""
                meta: Dict[str, str] = {}
                body = (raw_text or "").strip()

                if body.startswith("---"):
                    parts = body.split("---", 2)
                    if len(parts) >= 3:
                        yaml_text = parts[1].strip()
                        body = parts[2].strip()

                        try:
                            import yaml
                            parsed = yaml.safe_load(yaml_text)
                            if isinstance(parsed, dict):
                                for k, v in parsed.items():
                                    meta[str(k).lower().strip()] = str(v).strip()
                        except Exception:
                            for line in yaml_text.splitlines():
                                if ":" in line:
                                    k, _, v = line.partition(":")
                                    meta[k.strip().lower()] = v.strip().strip("'\"")

                return meta, body

            def _view_skill_content(s_item: Dict[str, Any]):
                title = s_item.get("title", "Skill")
                desc = s_item.get("description", "")
                fp = s_item.get("file_path", "")
                content = s_item.get("content_preview", "")
                if fp and Path(fp).exists():
                    try:
                        content = Path(fp).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        pass

                meta, clean_body = _parse_skill_markdown(content)

                # Consolidate metadata
                display_title = meta.get("title") or meta.get("name") or title
                display_desc = meta.get("description") or desc
                author = meta.get("author")
                category = meta.get("category") or s_item.get("category")
                version = meta.get("version")
                created = meta.get("created")
                tags_str = meta.get("tags") or ""
                tags = [t.strip().strip("'\"[]") for t in re.split(r"[,;]+", tags_str) if t.strip().strip("'\"[]")]

                d = ui.dialog()
                with d, ui.card().classes(f"w-[780px] max-w-[95vw] h-[600px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-2.5"):
                    with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("school", size="22px").classes("text-primary")
                            ui.label(f"Skill: {display_title}").classes("text-sm font-bold text-slate-900 dark:text-slate-100")
                        ui.button(icon="close", on_click=d.close).props("flat dense round size=xs")

                    # Structured Metadata Header Card
                    with ui.card().classes(f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} gap-1.5 shadow-none shrink-0"):
                        if display_desc:
                            ui.label(display_desc).classes(f"text-xs {MUTED} italic leading-relaxed")

                        with ui.row().classes("w-full items-center gap-2 pt-1 border-t border-slate-200 dark:border-slate-800 flex-wrap text-xs"):
                            if category:
                                ui.badge(category, color="indigo").props("dense rounded text-[10px]")
                            if version:
                                ui.badge(f"v{version}", color="slate").props("dense rounded text-[10px]")
                            if author:
                                ui.label(f"👤 {author}").classes(f"text-[11px] {MUTED_DIM} font-mono")
                            if created:
                                ui.label(f"📅 {created}").classes(f"text-[11px] {MUTED_DIM} font-mono")
                            if tags:
                                with ui.row().classes("items-center gap-1"):
                                    for t in tags[:6]:
                                        ui.badge(f"#{t}", color="grey").props("dense rounded text-[9px]")

                    # Clean documentation body without frontmatter artifacts
                    with ui.scroll_area().classes(f"w-full flex-1 p-3.5 {SURFACE} rounded-lg border {BORDER}"):
                        ui.markdown(clean_body or "No documentation content.").classes("text-xs leading-relaxed text-slate-900 dark:text-slate-100")
                d.open()

            with ui.expansion(f"🧠 Skills ({total_skills_count})", icon="psychology").classes(
                f"w-full border {BORDER} rounded-lg {SURFACE} mb-1"
            ).props('header-class="py-1 px-2 text-xs font-bold text-slate-900 dark:text-slate-100 flex-nowrap"'):
                with ui.column().classes("w-full gap-2 p-1.5"):
                    # 1. Project Extra Skills (.lollms_code/skills/) — WITH REMOVE / DELETE BUTTON
                    with ui.column().classes("w-full gap-1"):
                        ui.label(f"📁 PROJECT EXTRA SKILLS ({len(p_skills)})").classes("text-[10px] font-bold text-emerald-600 dark:text-emerald-400")
                        if not p_skills:
                            ui.label("(No project skills in .lollms_code/skills)").classes(f"text-[10px] {MUTED_DIM} italic pl-1")
                        for s in p_skills:
                            with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0 flex-nowrap"):
                                    ui.icon("psychology", size="14px").classes("text-emerald-500 shrink-0")
                                    ui.label(s["title"]).classes("text-xs font-semibold truncate text-slate-900 dark:text-slate-100 max-w-[110px]").tooltip(f"{s['title']}\n{s.get('description', '')}")

                                with ui.row().classes("items-center gap-1 shrink-0 flex-nowrap"):
                                    vis = s.get("visibility", "loadable")
                                    is_vis = (vis == "visible")

                                    def _toggle_skill_vis(s_item=s, curr_vis=is_vis):
                                        if session.personality and session.personality.skills_manager:
                                            new_v = "loadable" if curr_vis else "visible"
                                            try:
                                                session.personality.skills_manager.set_skill_visibility(s_item["title"], new_v)
                                                ui.notify(f"Skill '{s_item['title']}' set to {new_v.upper()}", type="positive")
                                                refresh_subws_panel()
                                            except Exception as ex:
                                                notify_error(f"Failed to change visibility: {ex}")

                                    ui.button(
                                        "[C]" if is_vis else "[U]",
                                        on_click=_toggle_skill_vis,
                                    ).props(f"flat dense size=xs color={'emerald' if is_vis else 'grey'} no-caps").tooltip("Toggle in-context: [C]=Loaded, [U]=Loadable")

                                    ui.button(icon="visibility", on_click=lambda item=s: _view_skill_content(item)).props("flat dense round size=xs color=primary").tooltip("View Skill Content")

                                    def _confirm_delete_skill(s_item=s):
                                        dlg = ui.dialog()
                                        with dlg, ui.card().classes(f"w-[420px] p-4 gap-3 bg-white dark:bg-slate-900 rounded-xl border {BORDER}"):
                                            ui.label("Remove Skill from Project?").classes("text-sm font-bold text-red-500")
                                            ui.label(f"Are you sure you want to remove '{s_item['title']}' from this project's .lollms_code/skills folder?").classes("text-xs text-slate-600 dark:text-slate-300")
                                            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                                                ui.button("Cancel", on_click=dlg.close).props("flat dense")
                                                def _do_remove():
                                                    dlg.close()
                                                    fp = Path(s_item.get("file_path", ""))
                                                    if fp.exists():
                                                        try:
                                                            if fp.is_dir():
                                                                import shutil
                                                                shutil.rmtree(str(fp))
                                                            else:
                                                                parent_dir = fp.parent
                                                                fp.unlink()
                                                                if parent_dir.name != "skills" and not any(parent_dir.iterdir()):
                                                                    parent_dir.rmdir()

                                                            if session.personality and session.personality.skills_manager:
                                                                session.personality.skills_manager.reload()
                                                            ui.notify(f"Removed skill '{s_item['title']}' from project", type="info")
                                                            refresh_subws_panel()
                                                        except Exception as err:
                                                            notify_error(f"Failed to delete skill: {err}")
                                                ui.button("Remove from Project", on_click=_do_remove).props("unelevated dense color=red no-caps")
                                        dlg.open()

                                    ui.button(icon="delete", on_click=_confirm_delete_skill).props("flat dense round size=xs color=red").tooltip("Remove skill from project (.lollms_code/skills/)")

                    # 2. Handbag Skills
                    with ui.column().classes("w-full gap-1 pt-1 border-t border-slate-200 dark:border-slate-800"):
                        ui.label(f"👜 HANDBAG SKILLS ({len(h_skills)})").classes("text-[10px] font-bold text-purple-600 dark:text-purple-400")
                        if not h_skills:
                            ui.label("(None bundled in active handbag)").classes(f"text-[10px] {MUTED_DIM} italic pl-1")
                        for s in h_skills:
                            with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0 flex-nowrap"):
                                    ui.icon("school", size="14px").classes("text-purple-500 shrink-0")
                                    ui.label(s["title"]).classes("text-xs font-semibold truncate text-slate-900 dark:text-slate-100 max-w-[130px]").tooltip(f"{s['title']}\n{s.get('description', '')}")
                                with ui.row().classes("items-center gap-1 shrink-0 flex-nowrap"):
                                    vis = s.get("visibility", "loadable")
                                    is_vis = (vis == "visible")

                                    def _toggle_h_skill_vis(s_item=s, curr_vis=is_vis):
                                        if session.personality and session.personality.skills_manager:
                                            new_v = "loadable" if curr_vis else "visible"
                                            try:
                                                session.personality.skills_manager.set_skill_visibility(s_item["title"], new_v)
                                                ui.notify(f"Skill '{s_item['title']}' set to {new_v.upper()}", type="positive")
                                                refresh_subws_panel()
                                            except Exception as ex:
                                                notify_error(f"Failed to change visibility: {ex}")

                                    ui.button(
                                        "[C]" if is_vis else "[U]",
                                        on_click=_toggle_h_skill_vis,
                                    ).props(f"flat dense size=xs color={'emerald' if is_vis else 'grey'} no-caps").tooltip("Toggle in-context: [C]=Loaded, [U]=Loadable")

                                    ui.badge("Handbag", color="purple").props("dense rounded text-[9px]").tooltip("Handbag native skill")
                                    ui.button(icon="visibility", on_click=lambda item=s: _view_skill_content(item)).props("flat dense round size=xs color=purple").tooltip("View Skill Content")

                    # 3. Global & Bundled Skills
                    if o_skills:
                        with ui.column().classes("w-full gap-1 pt-1 border-t border-slate-200 dark:border-slate-800"):
                            ui.label(f"🌐 GLOBAL & BUNDLED SKILLS ({len(o_skills)})").classes("text-[10px] font-bold text-indigo-600 dark:text-indigo-400")
                            for s in o_skills:
                                with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50 flex-nowrap"):
                                    with ui.row().classes("items-center gap-1.5 flex-1 min-w-0 flex-nowrap"):
                                        ui.icon("psychology", size="14px").classes("text-indigo-500 shrink-0")
                                        ui.label(s["title"]).classes("text-xs font-semibold truncate text-slate-900 dark:text-slate-100 max-w-[130px]").tooltip(f"{s['title']}\n{s.get('description', '')}")
                                    with ui.row().classes("items-center gap-1 shrink-0 flex-nowrap"):
                                        vis = s.get("visibility", "loadable")
                                        is_vis = (vis == "visible")

                                        def _toggle_o_skill_vis(s_item=s, curr_vis=is_vis):
                                            if session.personality and session.personality.skills_manager:
                                                new_v = "loadable" if curr_vis else "visible"
                                                try:
                                                    session.personality.skills_manager.set_skill_visibility(s_item["title"], new_v)
                                                    ui.notify(f"Skill '{s_item['title']}' set to {new_v.upper()}", type="positive")
                                                    refresh_subws_panel()
                                                except Exception as ex:
                                                    notify_error(f"Failed to change visibility: {ex}")

                                        ui.button(
                                            "[C]" if is_vis else "[U]",
                                            on_click=_toggle_o_skill_vis,
                                        ).props(f"flat dense size=xs color={'emerald' if is_vis else 'grey'} no-caps").tooltip("Toggle in-context: [C]=Loaded, [U]=Loadable")

                                        ui.badge(s.get("source", "global").capitalize(), color="indigo").props("dense rounded text-[9px]")
                                        ui.button(icon="visibility", on_click=lambda item=s: _view_skill_content(item)).props("flat dense round size=xs color=primary").tooltip("View Skill Content")

                    # Add skills from zoo shortcut
                    with ui.row().classes("w-full justify-end pt-1"):
                        ui.button("+ Add Skill from Zoo", icon="add", on_click=lambda: open_zoo_dialog()).props("flat dense size=xs color=primary no-caps")

            # ── 4. REFERENCE FILES SECTION (.lollms_code/sub_workspace/) ──
            loaded_ref_count = sum(1 for f in ref_files if f["is_loaded"])
            with ui.expansion(f"📚 Reference Docs ({len(ref_files)} | [C]:{loaded_ref_count})", icon="auto_stories").classes(
                f"w-full border {BORDER} rounded-lg {SURFACE}"
            ).props('header-class="py-1 px-2 text-xs font-bold text-slate-900 dark:text-slate-100 flex-nowrap"'):
                with ui.column().classes("w-full gap-1.5 p-1.5"):
                    # Toolbar
                    with ui.row().classes("w-full items-center justify-between pb-1 border-b border-slate-200 dark:border-slate-800"):
                        async def _import_f():
                            _picker = pick_file
                            if _picker:
                                chosen = await _picker(
                                    title="Select Reference File to Import",
                                    file_types=[("All Files", "*.*")]
                                )
                                if chosen:
                                    try:
                                        p = Path(chosen)
                                        if p.is_dir():
                                            imported = sub_ws.import_folder(p)
                                            ui.notify(f"Imported folder with {len(imported)} reference file(s).", type="positive")
                                        else:
                                            dest = sub_ws.import_file(p)
                                            ui.notify(f"Imported reference file: {dest.name}", type="positive")
                                        refresh_subws_panel()
                                    except Exception as ex:
                                        notify_error(f"Import failed: {ex}")

                        async def _import_d():
                            _picker = pick_folder
                            if _picker:
                                chosen = await _picker(title="Select Folder to Import into Sub-Workspace")
                                if chosen:
                                    sub_ws.import_folder(chosen)
                                    ui.notify("Imported reference folder.", type="positive")
                                    refresh_subws_panel()

                        def _load_all_ref():
                            cnt = sub_ws.load_all()
                            ui.notify(f"Loaded all {cnt} reference files [C]", type="positive")
                            refresh_subws_panel()

                        def _unload_all_ref():
                            sub_ws.unload_all()
                            ui.notify("Unloaded all reference files [U]", type="info")
                            refresh_subws_panel()

                        with ui.row().classes("gap-0.5"):
                            ui.button(icon="note_add", on_click=lambda: open_paste_reference_dialog()).props(
                                "flat dense round size=xs color=primary"
                            ).tooltip("Paste text as reference document")
                            ui.button(icon="upload_file", on_click=_import_f).props("flat dense round size=xs").tooltip("Import reference file")
                            ui.button(icon="drive_folder_upload", on_click=_import_d).props("flat dense round size=xs").tooltip("Import reference folder")
                        with ui.row().classes("gap-0.5"):
                            ui.button(icon="download", on_click=_load_all_ref).props("flat dense round size=xs color=emerald").tooltip("Load all [C]")
                            ui.button(icon="clear_all", on_click=_unload_all_ref).props("flat dense round size=xs color=amber").tooltip("Unload all [U]")

                    if not ref_files:
                        ui.label("(No reference files in .lollms_code/sub_workspace/)").classes(f"text-[10px] {MUTED_DIM} italic p-1")
                    else:
                        for rf in ref_files:
                            with ui.row().classes("w-full items-center justify-between p-1 rounded hover:bg-slate-200/50 dark:hover:bg-slate-800/50"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0"):
                                    ui.icon("description", size="14px").classes("text-slate-400 shrink-0")
                                    ui.label(rf["rel_path"]).classes("text-xs font-mono truncate text-slate-900 dark:text-slate-100")

                                with ui.row().classes("items-center gap-1"):
                                    is_l = rf["is_loaded"]

                                    def _toggle_ref_load(rel=rf["rel_path"], loaded=is_l):
                                        if loaded:
                                            sub_ws.unload_file(rel)
                                            ui.notify(f"Unloaded {rel} [U]", type="info")
                                        else:
                                            sub_ws.load_file(rel)
                                            ui.notify(f"Loaded {rel} [C]", type="positive")
                                        refresh_subws_panel()

                                    def _peek_ref(rel=rf["rel_path"]):
                                        content = sub_ws.peek_file(rel)
                                        dlg = ui.dialog()
                                        with dlg, ui.card().classes(f"w-[760px] max-w-[95vw] h-[550px] flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-2"):
                                            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                                                ui.label(f"👁️ Reference: sub_workspace/{rel}").classes("text-sm font-bold font-mono text-slate-900 dark:text-slate-100")
                                                ui.button(icon="close", on_click=dlg.close).props("flat round dense size=xs")
                                            with ui.scroll_area().classes(f"w-full flex-1 p-3 {SURFACE} rounded border {BORDER}"):
                                                if rel.endswith((".md", ".markdown", ".txt")):
                                                    ui.markdown(content).classes("text-xs leading-relaxed text-slate-900 dark:text-slate-100")
                                                else:
                                                    ui.code(content, language="json" if rel.endswith(".json") else ("python" if rel.endswith(".py") else "text")).classes("w-full text-xs")
                                        dlg.open()

                                    def _del_ref(rel=rf["rel_path"]):
                                        sub_ws.remove_path(rel)
                                        ui.notify(f"Removed {rel}", type="info")
                                        refresh_subws_panel()

                                    ui.button("[C]" if is_l else "[U]", on_click=_toggle_ref_load).props(
                                        f"flat dense size=xs color={'emerald' if is_l else 'grey'} no-caps"
                                    ).tooltip("Toggle context load [C]/[U]")
                                    ui.button(icon="visibility", on_click=_peek_ref).props("flat dense round size=xs color=primary").tooltip("Peek file content")
                                    ui.button(icon="delete", on_click=_del_ref).props("flat dense round size=xs color=red").tooltip("Delete reference file")

    def refresh_subws_tree():
        """Refreshes the Sub-Workspace panel and all subscribed assets."""
        refresh_subws_panel()

    # ── Sessions Dialog (Browse, Resume, Switch, Delete) ─────────────────
    def open_sessions_dialog():
        dialog = ui.dialog().props("maximized")
        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100 gap-3"
        ):
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("history_edu", size="28px").classes("text-primary")
                    with ui.column().classes("gap-0"):
                        ui.label("Saved Sessions & Discussions").classes("text-base font-bold")
                        ui.label(f"Resume previous sessions in workspace: {prefs.workspace_path}").classes(f"text-xs {MUTED_DIM}")

                with ui.row().classes("items-center gap-2"):
                    def _start_fresh_session():
                        dialog.close()
                        session.save_to_disk()
                        new_session()
                        ui.notify("Started fresh session.", type="positive")

                    ui.button("New Session", icon="add", on_click=_start_fresh_session).props(
                        "unelevated dense size=sm color=primary no-caps font-semibold"
                    )
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

            sessions_scroll = ui.scroll_area().classes("w-full flex-1 p-2")
            with sessions_scroll:
                sessions_container = ui.column().classes("w-full gap-2.5")

            def refresh_sessions_list():
                sessions_container.clear()
                saved_list = session.list_saved_sessions(prefs.workspace_path)
                with sessions_container:
                    if not saved_list:
                        ui.label("No saved sessions found in this workspace.").classes(
                            f"text-xs {MUTED_DIM} p-4 italic text-center w-full"
                        )
                        return

                    for item in saved_list:
                        s_id = item["id"]
                        is_active = (s_id == session.session_id)
                        with ui.card().classes(
                            f"w-full p-3 rounded-lg border {BORDER} {SURFACE} "
                            + ("border-l-4 border-l-primary shadow-sm" if is_active else "hover:border-primary/50")
                        ):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.column().classes("gap-0.5 flex-1 min-w-0"):
                                    with ui.row().classes("items-center gap-2"):
                                        ui.icon("chat", size="18px").classes("text-primary shrink-0")
                                        ui.label(item["title"]).classes("text-sm font-bold truncate text-slate-900 dark:text-slate-100")
                                        if is_active:
                                            ui.badge("ACTIVE", color="primary").props("dense rounded text-[9px]")
                                    ui.label(f"ID: {s_id} · Updated: {item['updated_at']} · {item['message_count']} messages").classes(
                                        f"text-[10px] {MUTED_DIM} font-mono"
                                    )

                                with ui.row().classes("items-center gap-2 shrink-0"):
                                    if not is_active:
                                        def _do_resume(target_id=s_id):
                                            dialog.close()
                                            resume_session(target_id)

                                        ui.button("Resume", icon="play_arrow", on_click=_do_resume).props(
                                            "unelevated dense size=xs color=primary no-caps font-semibold"
                                        )

                                    def _delete_sess(target_id=s_id, p_str=item["path"]):
                                        try:
                                            Path(p_str).unlink(missing_ok=True)
                                            ui.notify(f"Deleted session '{target_id}'", type="info")
                                            refresh_sessions_list()
                                        except Exception as ex:
                                            notify_error(f"Failed to delete session: {ex}")

                                    ui.button(icon="delete", on_click=_delete_sess).props("flat dense round size=xs color=red")

            refresh_sessions_list()

        dialog.open()

    def resume_session(target_session_id: str):
        """Loads a session from disk and replays all messages onto the transcript."""
        nonlocal current_agent_md, agent_text_buffer
        if session.busy:
            ui.notify("Cannot switch sessions while the agent is busy.", type="warning")
            return

        session.save_to_disk()
        ok = session.load_from_disk(target_session_id)
        if not ok:
            notify_error(f"Failed to load session '{target_session_id}'.")
            return

        # Clear active UI elements and replay
        transcript.clear()
        message_refs.clear()
        active_tool_panels.clear()
        active_artefact_panels.clear()
        current_agent_md = None
        agent_text_buffer = ""

        replay_transcript_from_log()
        ui.notify(f"Resumed session: {session.session_title}", type="positive")

    def replay_transcript_from_log():
        """Reconstructs the conversation view from a safe snapshot of the session's debug_log."""
        nonlocal current_agent_md, agent_text_buffer
        transcript.clear()
        message_refs.clear()
        active_tool_panels.clear()
        active_artefact_panels.clear()
        current_agent_md = None
        agent_text_buffer = ""

        entries_to_replay = list(session.debug_log)
        for entry in entries_to_replay:
            kind = entry.get("type")
            if kind == "user":
                msg_id = entry.get("id") or f"msg_{next_message_id()}"
                text = entry.get("text", "")
                with transcript:
                    outer_row = ui.row().classes("w-full justify-end items-start gap-1")
                    with outer_row:
                        actions = ui.row().classes("opacity-50 hover:opacity-100 transition-opacity gap-0.5 items-center")
                        with actions:
                            ui.button(icon="edit", on_click=lambda m=msg_id: open_edit_dialog(m)).props(
                                "flat round dense size=xs text-color=primary"
                            ).tooltip("Edit and resend from this point (truncates following history)")

                            ui.button(icon="replay", on_click=lambda m=msg_id: resend_from_point(m)).props(
                                "flat round dense size=xs text-color=amber"
                            ).tooltip("Resend from this point (truncates following history)")

                            ui.button(icon="edit_note", on_click=lambda t=text: set_prompt_input(t)).props(
                                "flat round dense size=xs color=grey"
                            ).tooltip("Copy to input box")

                            ui.button(icon="delete", on_click=lambda m=msg_id: delete_message(m)).props(
                                "flat round dense size=xs color=red"
                            ).tooltip("Delete message")

                        bubble = ui.markdown(text).classes(
                            "bg-primary text-white rounded-lg px-3 py-1.5 max-w-[75%]"
                        )
                message_refs[msg_id] = {"entry": entry, "row": outer_row, "bubble": bubble, "text": text}

            elif kind == "agent":
                msg_id = entry.get("id") or f"agent_{next_message_id()}"
                text = entry.get("text", "")
                with transcript:
                    outer_row = ui.row().classes("w-full justify-start items-start gap-1")
                    with outer_row:
                        md = ui.markdown(text).classes(
                            "bg-slate-100 dark:bg-slate-800/90 text-slate-900 dark:text-slate-100 rounded-lg px-4 py-2.5 max-w-[85%] text-sm break-words leading-relaxed border border-slate-300 dark:border-slate-700 shadow-sm"
                        )
                        ui.button(icon="content_copy", on_click=lambda e=entry: copy_agent_message(e)).props(
                            "flat round dense size=xs"
                        )
                md._debug_entry = entry
                message_refs[msg_id] = {"entry": entry, "row": outer_row, "bubble": md}

            elif kind == "system":
                text = entry.get("text", "")
                is_error = entry.get("error", False)
                with transcript:
                    with ui.row().classes("w-full justify-start items-start gap-1"):
                        ui.markdown(text).classes(
                            ("bg-red-50 dark:bg-red-950 text-red-800 dark:text-red-200 border border-red-300 dark:border-red-800" if is_error
                             else "bg-amber-50 dark:bg-amber-950 text-amber-900 dark:text-amber-200 border border-amber-300 dark:border-amber-800")
                            + " rounded-lg px-3 py-1.5 max-w-[90%] text-sm shadow-sm font-medium"
                        )

            elif kind == "event":
                title = entry.get("title", "")
                subtitle = entry.get("subtitle", "")
                body = entry.get("body", "")
                panel = add_event_panel(title, subtitle, body, "blue-500", "task_alt", record=False)
                panel._debug_entry = entry

        scroll_area.scroll_to(percent=1.0)

    async def resume_active_turn():
        """Resumes an incomplete or interrupted turn from its checkpoint or last prompt."""
        if session.busy:
            ui.notify("Agent is already running.", type="warning")
            return

        session.ensure_ready()
        resume_banner.set_visibility(False)
        resume_turn_btn.set_visibility(False)

        chk = None
        if hasattr(session.personality, "load_turn_checkpoint"):
            chk = session.personality.load_turn_checkpoint()

        last_prompt = session.get_last_user_prompt()
        prompt_to_resume = (chk.get("prompt") if chk else None) or last_prompt

        if not prompt_to_resume:
            ui.notify("No incomplete turn found to resume.", type="info")
            return

        # Hydrate resume prompt so the model continues execution
        plan_content = agent_bridge.get_current_plan_content(workspace_path=prefs.workspace_path)
        effective_resume_prompt = (
            f"[SYSTEM DIRECTIVE: Turn Resumed from Checkpoint]\n"
            f"Resume the task: '{prompt_to_resume}'.\n"
        )
        if plan_content and "No active task plan" not in plan_content:
            effective_resume_prompt += f"\nActive Roadmap:\n{plan_content[:600]}\n"
        effective_resume_prompt += (
            "Continue directly with the remaining actions. "
            "Do NOT ask what to do and do NOT output conversational pleasantries. Execute the next step now."
        )

        session.busy = True
        session.turn_start_ts = time.time()
        send_button.props("loading")
        status_label.set_text("Resuming task…")
        show_thinking_indicator("Resuming task execution…")

        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, effective_resume_prompt, prefs, session.event_queue,
            use_history=True, resume_turn=True
        )

    def _sync_resume_button_visibility():
        """Synchronizes visibility of both the top-bar button and the inline prompt banner."""
        if session.busy:
            resume_turn_btn.set_visibility(False)
            resume_banner.set_visibility(False)
            return

        is_incomplete = session.has_incomplete_turn()
        resume_turn_btn.set_visibility(is_incomplete)
        resume_banner.set_visibility(is_incomplete)

        if is_incomplete:
            last_p = session.get_last_user_prompt() or "Previous task"
            display_hint = last_p[:60] + ("..." if len(last_p) > 60 else "")
            resume_banner_label.set_text(f'Incomplete task: "{display_hint}" — click to continue from checkpoint')

    # Replay existing transcript if resuming or returning from Settings!
    if session.debug_log:
        replay_transcript_from_log()

    _sync_resume_button_visibility()

    # Initial tree population
    refresh_workspace_tree()
    refresh_subws_panel()
    # ---------------- Command Palette (Ctrl+K) ----------------

    def open_command_palette():
        with ui.dialog() as palette_dialog, ui.card().classes("w-[480px]"):
            search = ui.input(placeholder="Search commands…").classes("w-full").props("outlined dense")
            results_container = ui.column().classes("w-full gap-1 mt-2")

            def _filter():
                results_container.clear()
                q = (search.value or "").lower().strip()
                matches = [
                    (cmd, desc) for cmd, desc in SLASH_COMMANDS
                    if q in cmd.lower() or q in desc.lower()
                ][:8]
                with results_container:
                    if not matches:
                        ui.label("No matching commands.").classes("text-xs text-gray-500")
                        return
                    for cmd, desc in matches:
                        def _run(c=cmd):
                            palette_dialog.close()
                            prompt_input.value = c + " "
                            prompt_input.run_method("focus")

                        with ui.row().classes(
                            "w-full items-center gap-2 p-1 rounded cursor-pointer "
                            "hover:bg-gray-100 dark:hover:bg-gray-800"
                        ).on("click", _run):
                            ui.label(c).classes("font-mono text-xs text-primary")
                            ui.label(desc).classes("text-xs text-gray-500 truncate")

            search.on_value_change(lambda _: _filter())
            _filter()

            with ui.row().classes("w-full justify-end mt-2"):
                ui.button("Close", on_click=palette_dialog.close).props("flat dense no-caps")

        palette_dialog.open()

    ui.keyboard(
        on_key=lambda e: open_command_palette()
        if e.action.keydown and e.modifiers.ctrl and e.key == "k" else None,
        ignore=[],
    )

    ui.keyboard(
        on_key=lambda e: open_history_dialog()
        if e.action.keydown and e.modifiers.ctrl and e.key == "h" else None,
        ignore=[],
    )

    ui.keyboard(
        on_key=lambda e: (
            copy_agent_message(last_agent_state["entry"])
            if e.action.keydown and e.modifiers.ctrl and e.modifiers.shift and e.key == "c"
            and last_agent_state["entry"] is not None
            else None
        ),
        ignore=[],
    )

    ui.keyboard(
        on_key=lambda e: (
            open_search()
            if e.action.keydown and e.modifiers.ctrl and e.key == "f"
            else close_search()
            if e.action.keydown and e.key == "Escape" and search_row.visible
            else None
        ),
        ignore=[],
    )

    ui.keyboard(
        on_key=lambda e: open_shortcuts_dialog()
        if e.action.keydown and e.modifiers.ctrl and e.key == "/" else None,
        ignore=[],
    )

    ui.keyboard(
        on_key=lambda e: open_context_inspector_dialog()
        if e.action.keydown and e.modifiers.ctrl and e.key == "i" else None,
        ignore=[],
    )

    # ── 🕸️ WORKFLOW STUDIO IDE (GRAPH-BASED HARNESS WITH HARD CONSTRAINTS) ──
    def open_workflow_studio_dialog():
        dialog = ui.dialog().props("maximized")

        try:
            from lollms_client.lollms_workflow.workflow_types import (
                Workflow,
                WorkflowNode,
                WorkflowEdge,
                NodeType,
                WorkflowStatus,
                GuardType,
                WorkflowContext,
            )
            from lollms_client.lollms_workflow.workflow_engine import (
                WorkflowEngine,
                GuardViolationError,
                create_file_organizer_workflow,
                create_dual_model_review_workflow,
                list_project_workflows,
                save_project_workflow,
                delete_project_workflow,
                load_project_workflow,
            )
        except ImportError:
            from lollms_client.lollms_workflow import (
                Workflow,
                WorkflowNode,
                WorkflowEdge,
                NodeType,
                WorkflowStatus,
                GuardType,
                WorkflowContext,
                WorkflowEngine,
                GuardViolationError,
                create_file_organizer_workflow,
                create_dual_model_review_workflow,
                list_project_workflows,
                save_project_workflow,
                delete_project_workflow,
                load_project_workflow,
            )

        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100 gap-3 overflow-hidden"
        ):
            # Top Header Bar
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("account_tree", size="28px").classes("text-indigo-500")
                    with ui.column().classes("gap-0"):
                        ui.label("Workflow Studio — Project Graph Automation").classes("text-base font-bold")
                        ui.label("Build, configure, save, and execute deterministic workflow graphs in your project.").classes(
                            f"text-xs {MUTED_DIM}"
                        )

                with ui.row().classes("items-center gap-1.5"):
                    def _do_new_wf():
                        nw_dlg = ui.dialog()
                        with nw_dlg, ui.card().classes(f"w-[460px] p-4 gap-3 bg-white dark:bg-slate-900 border {BORDER} rounded-xl shadow-xl"):
                            ui.label("Create New Workflow").classes("font-bold text-sm text-slate-900 dark:text-slate-100")
                            nw_name = ui.input("Workflow Name", value="New Custom Workflow").classes("w-full text-xs").props("outlined dense")
                            nw_id = ui.input("Workflow ID (Slug)", value="new_custom_workflow").classes("w-full text-xs font-mono").props("outlined dense")
                            nw_desc = ui.textarea("Description", value="Custom automated pipeline.").classes("w-full text-xs").props("outlined dense rows=2")

                            with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
                                ui.button("Cancel", on_click=nw_dlg.close).props("flat dense no-caps")
                                def _create_and_load():
                                    name_val = nw_name.value.strip() or "Untitled Workflow"
                                    id_val = re.sub(r"[^a-zA-Z0-9_-]", "_", nw_id.value.strip().lower()) or "custom_wf"
                                    new_wf = Workflow(id=id_val, name=name_val, description=nw_desc.value.strip())
                                    new_wf.add_node(WorkflowNode(
                                        id="step_start",
                                        name="Start Task",
                                        node_type=NodeType.LLM,
                                        config={"prompt": "Begin task: {{task_prompt}}", "output_variable": "step_output"}
                                    ))
                                    new_wf.add_node(WorkflowNode(
                                        id="step_finish",
                                        name="Finish Workflow",
                                        node_type=NodeType.TERMINAL,
                                        config={"summary": "Workflow completed.\n{{step_output}}"}
                                    ))
                                    new_wf.add_edge("step_start", "step_finish")
                                    save_project_workflow(prefs.workspace_path, new_wf)
                                    nw_dlg.close()
                                    _reload_workflows_list(select_id=new_wf.id)
                                    ui.notify(f"Created project workflow: {new_wf.name}", type="positive")

                                ui.button("Create", icon="add", on_click=_create_and_load).props("unelevated dense color=primary no-caps")
                        nw_dlg.open()

                    def _save_current_wf():
                        wf = active_state["wf"]
                        wf.name = wf_name_input.value.strip() or wf.name
                        wf.description = wf_desc_input.value.strip() or wf.description
                        saved_path = save_project_workflow(prefs.workspace_path, wf)
                        _reload_workflows_list(select_id=wf.id)
                        ui.notify(f"Saved workflow '{wf.name}' to project ({saved_path.name})", type="positive")

                    def _delete_current_wf():
                        wf = active_state["wf"]
                        del_ok = delete_project_workflow(prefs.workspace_path, wf.id)
                        if del_ok:
                            ui.notify(f"Deleted workflow '{wf.name}'", type="info")
                            _reload_workflows_list()
                        else:
                            ui.notify(f"Cannot delete built-in template workflow.", type="warning")

                    ui.button("New Workflow", icon="add", on_click=_do_new_wf).props("unelevated dense size=xs color=primary no-caps")
                    ui.button("Save to Project", icon="save", on_click=_save_current_wf).props("outline dense size=xs color=emerald no-caps")
                    ui.button("Delete", icon="delete", on_click=_delete_current_wf).props("flat dense size=xs color=red no-caps")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

            # Workflow Switcher Bar
            all_wfs_init = list_project_workflows(prefs.workspace_path)
            wf_options_init = {}
            for w in all_wfs_init:
                badge = "[Project]" if w["source"] == "project" else "[Template]"
                wf_options_init[w["id"]] = f"{badge} {w['name']}"

            init_wf = create_file_organizer_workflow()
            if all_wfs_init:
                init_wf = all_wfs_init[0]["workflow"]

            active_state = {
                "wf": init_wf,
                "engine": None,
                "context": None,
                "running": False,
                "workspace_path": str(Path(prefs.workspace_path).resolve()),
                "initial_state": {"task_prompt": "Organize and structure all files in the workspace."},
            }

            initial_wf_id = active_state["wf"].id if active_state["wf"].id in wf_options_init else (next(iter(wf_options_init)) if wf_options_init else None)

            # Workspace and Workflow Switcher Bar
            with ui.row().classes(f"w-full items-center justify-between gap-3 p-2 {SURFACE} rounded border {BORDER} shrink-0 flex-wrap"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("folder", size="18px").classes("text-amber-500")
                    ui.label("Execution Workspace:").classes(f"text-xs font-bold {STRONG}")

                    ws_options = {p["path"]: f"{p['name']} ({p['path']})" for p in prefs.get_workspaces()}
                    cur_ws = str(Path(prefs.workspace_path).resolve())
                    if cur_ws not in ws_options:
                        ws_options[cur_ws] = f"Active ({prefs.workspace_path})"

                    def _on_wf_workspace_change(e):
                        if e.value:
                            _switch_workspace_context(e.value)

                    ws_select = ui.select(
                        ws_options,
                        value=active_state["workspace_path"]
                    ).props("dense outlined size=sm options-dense").classes("text-xs min-w-[280px]")
                    ws_select.on_value_change(_on_wf_workspace_change)

                    async def _browse_wf_ws():
                        _picker = pick_folder
                        if _picker:
                            chosen = await _picker(title="Select Workflow Execution Workspace", initial_dir=active_state["workspace_path"])
                            if chosen:
                                p_res = str(Path(chosen).resolve())
                                if p_res not in ws_options:
                                    ws_options[p_res] = f"{Path(chosen).name} ({p_res})"
                                    ws_select.options = ws_options
                                ws_select.value = p_res
                                _switch_workspace_context(p_res)

                    ui.button(icon="folder_open", on_click=_browse_wf_ws).props(
                        "flat dense round size=xs color=amber"
                    ).tooltip("Browse and select execution directory")

                with ui.row().classes("items-center gap-2 flex-1 min-w-[320px]"):
                    ui.label("Active Workflow:").classes(f"text-xs font-bold {STRONG}")
                    wf_select = ui.select(
                        wf_options_init,
                        value=initial_wf_id
                    ).props("dense outlined size=sm options-dense").classes("text-xs flex-1")

                with ui.row().classes("items-center gap-1.5 shrink-0"):
                    ui.badge("Project Workflows (.lollms_code/workflows/)", color="indigo").props("dense rounded text-[10px]")

            # Workflow Metadata Header Card
            initial_node_options = {nid: f"{n.name} ({nid})" for nid, n in active_state["wf"].nodes.items()}
            initial_start_id = active_state["wf"].start_node_id
            if initial_start_id not in initial_node_options and initial_node_options:
                initial_start_id = next(iter(initial_node_options))

            with ui.card().classes(f"w-full p-2.5 rounded-lg border {BORDER} {SURFACE} gap-2 shadow-none shrink-0"):
                with ui.row().classes("w-full items-center gap-3"):
                    wf_name_input = ui.input("Workflow Name", value=active_state["wf"].name).classes("flex-1 text-xs").props("outlined dense")
                    start_node_select = ui.select(
                        initial_node_options,
                        value=initial_start_id,
                        label="Start Step"
                    ).classes("w-56 text-xs").props("outlined dense options-dense")
                    wf_desc_input = ui.input("Description", value=active_state["wf"].description).classes("flex-1 text-xs").props("outlined dense")

            # Studio Canvas Controls & Mode Switcher Bar
            studio_mode = {"view": "graph"}  # "graph" or "pipeline"

            with ui.row().classes(f"w-full items-center justify-between px-2 py-1 {SURFACE} rounded border {BORDER} shrink-0 text-xs"):
                with ui.row().classes("items-center gap-1.5"):
                    view_toggle = ui.toggle(
                        {"graph": "🕸️ 2D Graph Canvas (Elements & Links)", "pipeline": "📋 Step List"},
                        value=studio_mode["view"]
                    ).props("dense unelevated size=xs").classes("text-xs font-semibold")

                    ui.button("Auto-Layout Graph", icon="auto_awesome", on_click=lambda: _do_auto_layout()).props(
                        "outline dense size=xs color=indigo no-caps"
                    ).tooltip("Arrange elements and bezier links into neat topological DAG columns")

                    ui.button("Reset View", icon="center_focus_strong", on_click=lambda: _reset_graph_view()).props(
                        "flat dense size=xs no-caps color=grey"
                    ).tooltip("Center canvas and reset zoom")

                with ui.row().classes("items-center gap-1.5"):
                    # Add Step Menu
                    with ui.button("➕ Add Element", icon="add").props("unelevated dense size=xs color=primary no-caps font-bold"):
                        with ui.menu():
                            def _add_step(ntype: NodeType):
                                wf = active_state["wf"]
                                idx = len(wf.nodes) + 1
                                nid = f"step_{idx}_{ntype.value}"
                                default_cfg = {
                                    NodeType.LLM: {"prompt": "Analyze: {{task_prompt}}", "output_variable": f"output_{idx}"},
                                    NodeType.AGENT: {"instruction": "Execute: {{task_prompt}}", "max_steps": 6, "output_variable": f"agent_out_{idx}"},
                                    NodeType.TOOL: {"tool_name": "tool_list_files", "tool_params": {"directory": "."}, "output_variable": f"tool_res_{idx}"},
                                    NodeType.GUARD: {"guard_type": "file_exists", "guard_target": "mapping.yaml"},
                                    NodeType.GATE: {"prompt": "Confirm execution of step?"},
                                    NodeType.CONDITION: {"condition_expr": "bool(state.get('task_prompt'))"},
                                    NodeType.TERMINAL: {"summary": "Done: {{task_prompt}}"},
                                }.get(ntype, {})

                                pos_x = 80 + (len(wf.nodes) * 320)
                                pos_y = 160 + (len(wf.nodes) % 3 * 80)
                                default_cfg["ui_pos"] = {"x": pos_x, "y": pos_y}

                                new_node = WorkflowNode(id=nid, name=f"Step {idx}: {ntype.name.title()}", node_type=ntype, config=default_cfg)
                                wf.add_node(new_node)
                                refresh_studio_canvas()
                                _open_node_editor(new_node)

                            ui.menu_item("🧠 LLM Step (Prompt + Model Override)", lambda: _add_step(NodeType.LLM))
                            ui.menu_item("🤖 Sub-Agent Step (Specialist Sandbox)", lambda: _add_step(NodeType.AGENT))
                            ui.menu_item("🛠️ Tool Execution (Deterministic Tool)", lambda: _add_step(NodeType.TOOL))
                            ui.menu_item("🛡️ Hard Guard Wall (Invariant Wall)", lambda: _add_step(NodeType.GUARD))
                            ui.menu_item("✋ Operator Approval Gate (Human Pause)", lambda: _add_step(NodeType.GATE))
                            ui.menu_item("🔀 Branch Condition (If / Else)", lambda: _add_step(NodeType.CONDITION))
                            ui.menu_item("🏁 Terminal Conclusion", lambda: _add_step(NodeType.TERMINAL))

            # Main Two-Pane IDE: Left: Interactive 2D Graph Canvas; Right: Execution Console & Live State
            with ui.row().classes("w-full flex-1 min-h-0 items-stretch overflow-hidden flex-nowrap gap-3 pt-1"):
                # ── Left Column: Interactive 2D Graph Canvas & Links ──
                with ui.column().classes("w-3/5 h-full shrink-0 border-r " + BORDER + " p-0 flex flex-col gap-0 overflow-hidden relative"):
                    # 2D Graph Canvas Viewport
                    graph_canvas_slot = ui.element("div").classes(
                        "w-full h-full relative overflow-hidden select-none bg-slate-950"
                    ).style(
                        "background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px); "
                        "background-size: 20px 20px;"
                    )

                    # Linear Fallback Pipeline View (when toggled)
                    pipeline_scroll = ui.scroll_area().classes("w-full h-full p-2 " + SURFACE + " rounded border " + BORDER)
                    pipeline_scroll.visible = False
                    with pipeline_scroll:
                        pipeline_container = ui.column().classes("w-full gap-2")

                # ── Right Column: Execution Console & Live State ──
                with ui.column().classes("flex-1 h-full min-w-0 flex flex-col gap-2 overflow-hidden p-2"):
                    with ui.tabs().classes(f"w-full {SURFACE} border-b {BORDER} shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as run_tabs:
                        tab_console = ui.tab('console', label='📺 Execution Console', icon='terminal').classes('text-xs py-1 flex-1')
                        tab_state = ui.tab('state', label='📊 Live Variables (context.state)', icon='data_object').classes('text-xs py-1 flex-1')

                    # Dynamic Configurable Variables Inputs Panel
                    with ui.card().classes(f"w-full p-2.5 rounded border {BORDER} {SURFACE} gap-2 shrink-0 shadow-none"):
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-1.5"):
                                ui.icon("tune", size="18px").classes("text-primary")
                                ui.label("Workflow Parameters & Inputs").classes(f"text-xs font-bold {STRONG}")
                            with ui.row().classes("items-center gap-1"):
                                def _add_custom_input():
                                    dlg_var = ui.dialog()
                                    with dlg_var, ui.card().classes(f"w-[380px] p-4 gap-2 bg-white dark:bg-slate-900 rounded-xl border {BORDER}"):
                                        ui.label("Add Input Parameter").classes("font-bold text-xs")
                                        v_key_in = ui.input("Variable Key", placeholder="e.g. folder_name").classes("w-full text-xs").props("outlined dense")
                                        v_val_in = ui.input("Initial Value", placeholder="Value").classes("w-full text-xs").props("outlined dense")
                                        with ui.row().classes("w-full justify-end gap-2 pt-2"):
                                            ui.button("Cancel", on_click=dlg_var.close).props("flat dense")
                                            def _set_var():
                                                k = v_key_in.value.strip()
                                                if k:
                                                    active_state["initial_state"][k] = v_val_in.value
                                                    _refresh_variable_inputs()
                                                    dlg_var.close()
                                            ui.button("Add", on_click=_set_var).props("unelevated dense color=primary no-caps")
                                    dlg_var.open()

                                ui.button("+ Add Variable", icon="add", on_click=_add_custom_input).props("flat dense size=xs color=primary no-caps")

                        variables_container = ui.column().classes("w-full gap-2")

                        with ui.row().classes("w-full items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800"):
                            with ui.row().classes("items-center gap-1 text-[11px] text-slate-500 font-mono"):
                                ui.label(f"Target: {Path(active_state['workspace_path']).name}").classes("truncate max-w-[200px]")
                            with ui.row().classes("items-center gap-1.5"):
                                run_btn = ui.button("▶ Run Workflow", icon="play_arrow").props("unelevated dense size=sm color=primary no-caps font-bold")
                                cancel_btn = ui.button(icon="stop").props("unelevated dense round size=sm color=red")
                                cancel_btn.visible = False

                    # Interactive Gate Approval Banner (Appears when WAITING_APPROVAL)
                    gate_card = ui.card().classes("w-full p-3 rounded-lg border-2 border-amber-500 bg-amber-50 dark:bg-amber-950/60 gap-2 shrink-0")
                    gate_card.visible = False
                    with gate_card:
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("pan_tool", size="20px").classes("text-amber-500 animate-pulse")
                                ui.label("Operator Approval Gate Reached").classes("text-xs font-bold text-amber-800 dark:text-amber-200")
                            ui.badge("HUMAN IN THE LOOP", color="amber").props("dense rounded text-[9px]")

                        gate_prompt_label = ui.label("").classes("text-xs text-slate-800 dark:text-slate-200 font-semibold px-1")

                        with ui.row().classes("w-full items-center justify-end gap-2 pt-1 border-t border-amber-200 dark:border-amber-800/80"):
                            def _approve_gate():
                                if active_state["engine"] and active_state["context"]:
                                    gate_card.visible = False
                                    run_btn.props(add="loading")
                                    status_bar.visible = True
                                    _log("✅ Operator approved gate. Resuming workflow execution...", "text-green-400 font-bold")

                                    async def _do_resume():
                                        try:
                                            from nicegui import run as _ng_run
                                            ctx = await _ng_run.io_bound(
                                                active_state["engine"].resume,
                                                active_state["wf"],
                                                active_state["context"],
                                                True,
                                                _wf_event_listener
                                            )
                                            _finish_run(ctx)
                                        except Exception as ex:
                                            _log(f"Resume crash: {ex}", "text-red-500 font-bold")
                                        finally:
                                            run_btn.props(remove="loading")

                                    ui.timer(0.05, _do_resume, once=True)

                            def _reject_gate():
                                if active_state["engine"] and active_state["context"]:
                                    gate_card.visible = False
                                    _log("❌ Operator rejected approval gate. Halting workflow.", "text-red-400 font-bold")
                                    active_state["engine"].resume(active_state["wf"], active_state["context"], False, _wf_event_listener)
                                    status_lbl.set_text("Halted: REJECTED")
                                    run_btn.props(remove="loading")
                                    status_bar.visible = False

                            ui.button("Reject & Halt", icon="close", on_click=_reject_gate).props("flat dense size=sm color=red no-caps")
                            ui.button("Approve & Resume", icon="check", on_click=_approve_gate).props("unelevated dense size=sm color=primary no-caps font-bold")

                    # Live Status & Progress
                    status_bar = ui.linear_progress(value=0.0).props("instant-feedback color=indigo")
                    status_bar.visible = False
                    status_lbl = ui.label("Ready to run.").classes("text-xs font-mono font-semibold text-slate-500 px-1")

                    with ui.tab_panels(run_tabs, value='console').classes('w-full flex-1 min-h-0 p-0 bg-transparent flex flex-col overflow-hidden'):
                        with ui.tab_panel('console').classes('w-full h-full p-2 flex flex-col overflow-hidden'):
                            with ui.scroll_area().classes(f"w-full flex-1 p-3 bg-slate-950 text-slate-100 rounded border {BORDER} font-mono text-xs"):
                                log_container = ui.column().classes("w-full gap-1")

                        with ui.tab_panel('state').classes('w-full h-full p-2 flex flex-col overflow-hidden'):
                            with ui.scroll_area().classes(f"w-full flex-1 p-3 bg-slate-950 text-slate-100 rounded border {BORDER} font-mono text-xs"):
                                state_display = ui.label("{}").classes("whitespace-pre-wrap select-all text-emerald-400 leading-relaxed")

            def _do_auto_layout():
                wf = active_state["wf"]
                wf.compute_auto_layout()
                refresh_studio_canvas()
                ui.notify("Arranged elements into topological DAG layout.", type="positive")

            def _reset_graph_view():
                ui.run_javascript("""
                    if (window._wfGraph) {
                        window._wfGraph.panX = 0;
                        window._wfGraph.panY = 0;
                        window._wfGraph.zoom = 1.0;
                        window._wfGraph.applyTransform();
                    }
                """)

            # Hidden input to bridge node configure/edit events from client-side JS into Python
            node_edit_trigger = ui.input().classes("hidden").props("id=wf-node-edit-trigger")

            def _on_node_edit_signal(val: str):
                if not val:
                    return
                node_edit_trigger.value = ""
                target_node = active_state["wf"].nodes.get(val)
                if target_node:
                    _open_node_editor(target_node)

            node_edit_trigger.on_value_change(lambda e: _on_node_edit_signal(e.value))

            def _render_interactive_graph_parts(wf: Workflow, active_node_id: Optional[str] = None) -> Tuple[str, str]:
                """Generates clean SVG canvas markup and separate executable JS to avoid NiceGUI <script> validation errors."""
                if not getattr(wf, "nodes", None):
                    return (
                        "<div style='color:#64748b;padding:24px;text-align:center;'>No elements in workflow. Click 'Add Element' to begin.</div>",
                        ""
                    )

                nodes_dict = wf.nodes
                auto_pos = wf.compute_auto_layout()

                type_colors = {
                    NodeType.LLM: ("#3b82f6", "🧠", "LLM GENERATION"),
                    NodeType.AGENT: ("#a855f7", "🤖", "SPECIALIST AGENT"),
                    NodeType.TOOL: ("#10b981", "🛠️", "DETERMINISTIC TOOL"),
                    NodeType.GUARD: ("#ef4444", "🛡️", "HARD GUARD WALL"),
                    NodeType.GATE: ("#f59e0b", "✋", "OPERATOR GATE"),
                    NodeType.CONDITION: ("#06b6d4", "🔀", "BRANCH CONDITION"),
                    NodeType.TERMINAL: ("#64748b", "🏁", "TERMINAL FINISH"),
                }

                nodes_data = []
                for nid, n in nodes_dict.items():
                    pos = n.config.get("ui_pos") or auto_pos.get(nid, (100, 100))
                    color, icon, badge = type_colors.get(n.node_type, ("#64748b", "📦", "STEP"))
                    assigned_m = n.config.get("model_name") or n.config.get("model") or ""
                    model_tag = f"🤖 {assigned_m}" if assigned_m else ""
                    desc = n.description or n.config.get("guard_target") or n.config.get("tool_name") or n.config.get("prompt", "")[:36] or ""

                    # Badges for node footer
                    p_name = n.config.get("persona_name") or (Path(n.config["handbag_path"]).name if n.config.get("handbag_path") else "")
                    persona_tag = f"👜 {p_name}" if p_name else ""
                    tools_cnt = len(n.config.get("tools", []))
                    tools_tag = f"🛠️ {tools_cnt}" if tools_cnt > 0 else ("🛠️ 1 tool" if n.node_type == NodeType.TOOL and n.config.get("tool_name") else "")
                    skills_cnt = len(n.config.get("skills", []))
                    skills_tag = f"🧠 {skills_cnt} skills" if skills_cnt > 0 else ""

                    nodes_data.append({
                        "id": nid,
                        "name": n.name,
                        "type": n.node_type.value,
                        "color": color,
                        "icon": icon,
                        "badge": badge,
                        "model": model_tag,
                        "persona": persona_tag,
                        "tools": tools_tag,
                        "skills": skills_tag,
                        "desc": desc,
                        "x": pos[0] if isinstance(pos, (tuple, list)) else pos.get("x", 100),
                        "y": pos[1] if isinstance(pos, (tuple, list)) else pos.get("y", 100),
                        "is_start": nid == wf.start_node_id,
                        "is_active": nid == active_node_id,
                        "on_success": n.on_success or "",
                        "on_failure": n.on_failure or "",
                    })

                links_data = []
                for edge in wf.edges:
                    links_data.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "label": edge.label or str(edge.condition_value or ""),
                        "type": "default",
                        "color": "#3b82f6"
                    })

                for n in nodes_dict.values():
                    if n.on_success and not any(e.source_id == n.id and e.target_id == n.on_success for e in wf.edges):
                        links_data.append({
                            "source": n.id,
                            "target": n.on_success,
                            "label": "next",
                            "type": "success",
                            "color": "#3b82f6"
                        })
                    if n.on_failure and not any(e.source_id == n.id and e.target_id == n.on_failure for e in wf.edges):
                        links_data.append({
                            "source": n.id,
                            "target": n.on_failure,
                            "label": "loopback",
                            "type": "failure",
                            "color": "#ef4444"
                        })

                payload_json = json.dumps({"nodes": nodes_data, "links": links_data, "active_id": active_node_id})

                html_markup = """
                <div id="wf-graph-root" style="width:100%; height:100%; position:relative; overflow:hidden; user-select:none;">
                    <svg id="wf-graph-svg" style="width:100%; height:100%; position:absolute; inset:0; cursor:grab;">
                        <defs>
                            <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 1 L 10 5 L 0 9 z" fill="#3b82f6" />
                            </marker>
                            <marker id="arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 1 L 10 5 L 0 9 z" fill="#ef4444" />
                            </marker>
                            <marker id="arrow-cyan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                                <path d="M 0 1 L 10 5 L 0 9 z" fill="#06b6d4" />
                            </marker>
                            <filter id="glow-active" x="-20%" y="-20%" width="140%" height="140%">
                                <feGaussianBlur stdDeviation="6" result="blur" />
                                <feComposite in="SourceGraphic" in2="blur" operator="over" />
                            </filter>
                        </defs>
                        <g id="wf-graph-viewport">
                            <g id="wf-links-group"></g>
                            <g id="wf-nodes-group"></g>
                        </g>
                    </svg>
                </div>
                """

                js_code = f"""
                (function() {{
                    let attempts = 0;
                    function initWfGraph() {{
                        const data = {payload_json};
                        const root = document.getElementById('wf-graph-root');
                        const svg = document.getElementById('wf-graph-svg');
                        const viewport = document.getElementById('wf-graph-viewport');
                        const linksGroup = document.getElementById('wf-links-group');
                        const nodesGroup = document.getElementById('wf-nodes-group');

                        if (!svg || !viewport || !linksGroup || !nodesGroup) {{
                            if (attempts++ < 30) {{
                                setTimeout(initWfGraph, 35);
                            }}
                            return;
                        }}

                        window._wfGraph = window._wfGraph || {{
                            panX: 40,
                            panY: 40,
                            zoom: 1.0,
                            isPanning: false,
                            startPanX: 0,
                            startPanY: 0,
                            dragNode: null,
                            dragOffsetX: 0,
                            dragOffsetY: 0,
                        }};
                        const state = window._wfGraph;

                        function applyTransform() {{
                            viewport.setAttribute('transform', `translate(${{state.panX}}, ${{state.panY}}) scale(${{state.zoom}})`);
                        }}
                        window._wfGraph.applyTransform = applyTransform;
                        applyTransform();

                        svg.onmousedown = (e) => {{
                            if (e.target === svg || e.target.id === 'wf-graph-viewport' || e.target.tagName === 'path') {{
                                state.isPanning = true;
                                state.startPanX = e.clientX - state.panX;
                                state.startPanY = e.clientY - state.panY;
                                svg.style.cursor = 'grabbing';
                            }}
                        }};

                        window.onmousemove = (e) => {{
                            if (state.isPanning) {{
                                state.panX = e.clientX - state.startPanX;
                                state.panY = e.clientY - state.startPanY;
                                applyTransform();
                            }} else if (state.dragNode) {{
                                const mouseX = (e.clientX - state.panX) / state.zoom;
                                const mouseY = (e.clientY - state.panY) / state.zoom;
                                state.dragNode.x = mouseX - state.dragOffsetX;
                                state.dragNode.y = mouseY - state.dragOffsetY;
                                renderNodes();
                                renderLinks();
                            }}
                        }};

                        window.onmouseup = () => {{
                            if (state.isPanning) {{
                                state.isPanning = false;
                                if (svg) svg.style.cursor = 'grab';
                            }}
                            if (state.dragNode) {{
                                state.dragNode = null;
                            }}
                        }};

                        svg.onwheel = (e) => {{
                            e.preventDefault();
                            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
                            state.zoom = Math.max(0.3, Math.min(2.5, state.zoom * zoomFactor));
                            applyTransform();
                        }};

                        function renderLinks() {{
                            linksGroup.innerHTML = '';
                            const NODE_W = 260;
                            const NODE_H = 110;

                            data.links.forEach(link => {{
                                const src = data.nodes.find(n => n.id === link.source);
                                const tgt = data.nodes.find(n => n.id === link.target);
                                if (!src || !tgt) return;

                                const x1 = src.x + NODE_W;
                                const y1 = src.y + (NODE_H / 2);
                                const x2 = tgt.x;
                                const y2 = tgt.y + (NODE_H / 2);

                                let d = '';
                                if (x2 >= x1 - 20) {{
                                    const dx = Math.max(60, (x2 - x1) * 0.5);
                                    d = `M ${{x1}} ${{y1}} C ${{x1 + dx}} ${{y1}}, ${{x2 - dx}} ${{y2}}, ${{x2}} ${{y2}}`;
                                }} else {{
                                    const loopOffsetY = 90;
                                    d = `M ${{x1}} ${{y1}} C ${{x1 + 100}} ${{y1 + loopOffsetY}}, ${{x2 - 100}} ${{y2 + loopOffsetY}}, ${{x2}} ${{y2}}`;
                                }}

                                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                                path.setAttribute('d', d);
                                path.setAttribute('fill', 'none');
                                path.setAttribute('stroke', link.color || '#3b82f6');
                                path.setAttribute('stroke-width', link.type === 'failure' ? '2.5' : '3');
                                if (link.type === 'failure') {{
                                    path.setAttribute('stroke-dasharray', '6,4');
                                    path.setAttribute('marker-end', 'url(#arrow-red)');
                                }} else {{
                                    path.setAttribute('marker-end', 'url(#arrow-blue)');
                                }}
                                path.style.transition = 'stroke-width 0.2s';
                                path.addEventListener('mouseenter', () => path.setAttribute('stroke-width', '5'));
                                path.addEventListener('mouseleave', () => path.setAttribute('stroke-width', link.type === 'failure' ? '2.5' : '3'));
                                linksGroup.appendChild(path);
                            }});
                        }}

                        function renderNodes() {{
                            nodesGroup.innerHTML = '';
                            const NODE_W = 260;
                            const NODE_H = 110;

                            data.nodes.forEach(node => {{
                                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                                g.setAttribute('transform', `translate(${{node.x}}, ${{node.y}})`);
                                g.style.cursor = 'move';

                                const fo = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
                                fo.setAttribute('width', NODE_W);
                                fo.setAttribute('height', NODE_H);

                                const isActive = node.id === data.active_id;
                                const glowClass = isActive ? 'box-shadow: 0 0 16px #06b6d4; border-color: #06b6d4 !important;' : '';

                                fo.innerHTML = `
                                    <div xmlns="http://www.w3.org/1999/xhtml" style="width:100%; height:100%; background:#0f172a; border-radius:10px; border:2px solid ${{node.color}}; border-left: 6px solid ${{node.color}}; padding:8px 10px; display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 6px 16px rgba(0,0,0,0.5); font-family:ui-monospace, monospace; ${{glowClass}}">
                                        <div style="display:flex; align-items:center; justify-content:space-between;">
                                            <div style="display:flex; align-items:center; gap:6px;">
                                                <span style="font-size:14px;">${{node.icon}}</span>
                                                <span style="font-size:11px; font-weight:bold; color:#f1f5f9; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:130px;">${{node.name}}</span>
                                            </div>
                                            <div style="display:flex; align-items:center; gap:4px;">
                                                <span style="font-size:8px; font-weight:bold; background:${{node.color}}33; color:${{node.color}}; padding:2px 5px; border-radius:4px; text-transform:uppercase;">${{node.badge}}</span>
                                                <button onclick="event.stopPropagation(); const tr = document.getElementById('wf-node-edit-trigger'); if (tr) {{ tr.value = '${{node.id}}'; tr.dispatchEvent(new Event('input')); }}" style="background:none; border:none; color:#94a3b8; cursor:pointer; font-size:12px; padding:0 2px;" title="Configure Node">⚙️</button>
                                            </div>
                                        </div>
                                        <div style="font-size:10px; color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                                            ${{node.desc || '(no description)'}}
                                        </div>
                                        <div style="display:flex; align-items:center; gap:6px; font-size:9px; overflow:hidden;">
                                            ${{node.persona ? `<span style="color:#c084fc;">${{node.persona}}</span>` : ''}}
                                            ${{node.tools ? `<span style="color:#34d399;">${{node.tools}}</span>` : ''}}
                                            ${{node.skills ? `<span style="color:#fcd34d;">${{node.skills}}</span>` : ''}}
                                        </div>
                                        <div style="display:flex; align-items:center; justify-content:space-between; border-top:1px solid #1e293b; padding-top:3px; font-size:9px;">
                                            <span style="color:#38bdf8;">${{node.model || ''}}</span>
                                            <span style="color:#64748b;">${{node.is_start ? '🚩 START' : ''}}</span>
                                        </div>
                                        <div style="position:absolute; left:-7px; top:46px; width:12px; height:12px; border-radius:50%; background:#0284c7; border:2px solid #fff;" title="Inflow"></div>
                                        <div style="position:absolute; right:-7px; top:46px; width:12px; height:12px; border-radius:50%; background:${{node.color}}; border:2px solid #fff;" title="Outflow"></div>
                                    </div>
                                `;

                                fo.addEventListener('mousedown', (e) => {{
                                    e.stopPropagation();
                                    state.dragNode = node;
                                    const mouseX = (e.clientX - state.panX) / state.zoom;
                                    const mouseY = (e.clientY - state.panY) / state.zoom;
                                    state.dragOffsetX = mouseX - node.x;
                                    state.dragOffsetY = mouseY - node.y;
                                }});

                                fo.addEventListener('dblclick', (e) => {{
                                    e.stopPropagation();
                                    const trigger = document.getElementById('wf-node-edit-trigger');
                                    if (trigger) {{
                                        trigger.value = node.id;
                                        trigger.dispatchEvent(new Event('input'));
                                    }}
                                }});

                                g.appendChild(fo);
                                nodesGroup.appendChild(g);
                            }});
                        }}

                        renderNodes();
                        renderLinks();
                    }}

                    initWfGraph();
                }})();
                """

                return html_markup, js_code

            def refresh_studio_canvas():
                wf = active_state["wf"]
                if studio_mode["view"] == "graph":
                    pipeline_scroll.visible = False
                    graph_canvas_slot.visible = True
                    graph_canvas_slot.clear()
                    with graph_canvas_slot:
                        curr_id = active_state["context"].current_node_id if active_state["context"] else None
                        html_markup, js_code = _render_interactive_graph_parts(wf, active_node_id=curr_id)
                        ui.html(html_markup).classes("w-full h-full")
                        if js_code:
                            ui.run_javascript(js_code)
                else:
                    graph_canvas_slot.visible = False
                    pipeline_scroll.visible = True
                    refresh_pipeline_ui()

            def _on_view_toggle(e):
                studio_mode["view"] = e.value
                refresh_studio_canvas()

            view_toggle.on_value_change(_on_view_toggle)

            def _log(msg: str, color="text-slate-300"):
                with log_container:
                    ui.label(msg).classes(f"leading-relaxed {color}")

            def _wf_event_listener(kind: str, node_id: str, data: Dict[str, Any]):
                if kind == "node_start":
                    status_lbl.set_text(f"Executing: {data.get('name')} (Model: {data.get('model')})")
                    _log(f"▶ [{data.get('type')}] Entering {data.get('name')} (Model: {data.get('model')})...", "text-cyan-400 font-bold")
                elif kind == "guard_passed":
                    _log(f"  ✓ Wall verified: {data.get('explanation')}", "text-green-400")
                elif kind == "guard_failed":
                    _log(f"  🛑 WALL VIOLATION: {data.get('explanation')}", "text-red-400 font-bold")
                elif kind == "gate_reached":
                    status_lbl.set_text("Paused at Operator Gate")
                    gate_prompt_label.set_text(data.get("prompt", "Please confirm step execution."))
                    gate_card.visible = True
                    _log(f"✋ [gate] Human Approval Required: {data.get('prompt')}", "text-amber-400 font-bold")
                elif kind == "workflow_end":
                    status_lbl.set_text(f"Finished: {data.get('status').upper()}")
                    st_col = "text-green-400" if data.get("status") == "completed" else "text-red-400"
                    _log(f"🏁 Workflow completed: {data.get('status').upper()} in {data.get('steps')} steps.", f"{st_col} font-bold")

            def _finish_run(ctx: WorkflowContext):
                status_bar.value = 1.0
                state_display.set_text(json.dumps(ctx.state, indent=2, default=str))
                if ctx.status == WorkflowStatus.WAITING_APPROVAL:
                    gate_prompt_label.set_text(ctx.pending_gate_prompt or "Operator approval needed.")
                    gate_card.visible = True
                    status_lbl.set_text("Waiting Operator Approval")
                elif ctx.status == WorkflowStatus.COMPLETED:
                    status_lbl.set_text("Completed Successfully")
                    ui.notify(f"✓ Workflow finished successfully!", type="positive")
                elif ctx.status == WorkflowStatus.GUARD_BLOCKED:
                    status_lbl.set_text("Halted: Guard Wall Blocked")
                    ui.notify(f"🛑 Workflow Wall Hit: {ctx.last_error}", type="negative")
                else:
                    status_lbl.set_text(f"Halted: {ctx.last_error or 'Error'}")
                    ui.notify(f"Workflow halted: {ctx.last_error}", type="warning")

            def _refresh_variable_inputs():
                variables_container.clear()
                wf = active_state["wf"]
                extracted_vars = wf.get_template_variables() if hasattr(wf, "get_template_variables") else []
                if "task_prompt" not in extracted_vars and not extracted_vars:
                    extracted_vars = ["task_prompt"]

                # Merge discovered variables with initial_state
                for v in extracted_vars:
                    if v not in active_state["initial_state"]:
                        active_state["initial_state"][v] = ""

                with variables_container:
                    for v_name in sorted(list(active_state["initial_state"].keys())):
                        curr_v = active_state["initial_state"].get(v_name, "")
                        with ui.row().classes("w-full items-center gap-2"):
                            def _make_change(key_name=v_name):
                                return lambda e: active_state["initial_state"].update({key_name: e.value})
                            inp = ui.input(f"{{{{{v_name}}}}}", value=str(curr_v)).classes("flex-1 text-xs").props("outlined dense")
                            inp.on_value_change(_make_change(v_name))

            _refresh_variable_inputs()

            async def _run_current_workflow():
                session.ensure_ready()
                wf = active_state["wf"]
                wf.name = wf_name_input.value.strip() or wf.name
                wf.description = wf_desc_input.value.strip() or wf.description
                wf.start_node_id = start_node_select.value or wf.start_node_id

                ws_target = active_state["workspace_path"]
                engine = WorkflowEngine(
                    client=session.client,
                    personality=session.personality,
                    workspace_path=ws_target,
                    debug=True,
                )
                active_state["engine"] = engine

                run_btn.props(add="loading")
                status_bar.visible = True
                status_bar.value = 0.1
                log_container.clear()
                gate_card.visible = False

                init_state = dict(active_state["initial_state"])

                try:
                    from nicegui import run as _ng_run
                    ctx = await _ng_run.io_bound(engine.run, wf, init_state, None, None, _wf_event_listener)
                    active_state["context"] = ctx
                    _finish_run(ctx)
                except Exception as ex:
                    _log(f"Crash: {ex}", "text-red-500 font-bold")
                    ui.notify(f"Execution failed: {ex}", type="negative")
                finally:
                    if not (active_state["context"] and active_state["context"].status == WorkflowStatus.WAITING_APPROVAL):
                        run_btn.props(remove="loading")

            run_btn.on("click", _run_current_workflow)

            def _switch_workspace_context(new_ws: str):
                active_state["workspace_path"] = str(Path(new_ws).resolve())
                ui.notify(f"Workflow execution workspace set to: {Path(new_ws).name}", type="info")
                _reload_workflows_list()
                refresh_studio_canvas()

            # Node Editor Modal Dialog
            def _open_node_editor(node: WorkflowNode):
                e_dlg = ui.dialog().props("maximized")
                wf = active_state["wf"]

                # Extract model choices
                llm_profiles = env.get_model_profiles("llm") if hasattr(env, "get_model_profiles") else {}
                model_choices = {"": "Default (Parent Model)"}
                for m_alias in llm_profiles:
                    model_choices[m_alias] = f"🤖 {m_alias}"

                # Extract available tools
                tool_choices = [
                    "tool_organize_files_from_plan",
                    "tool_read_file",
                    "tool_write_file",
                    "tool_list_files",
                    "tool_find_files",
                    "tool_grep_files",
                    "tool_execute_shell_command",
                    "tool_execute_python_code",
                    "tool_execute_python_file",
                    "tool_inspect_image",
                    "tool_vlm_query",
                    "tool_inspect_document",
                    "tool_read_document_content",
                    "tool_annotate_document",
                    "tool_edit_document_text",
                    "tool_execute_python_data_query",
                    "tool_get_table_schema",
                    "tool_query_database_sql",
                    "tool_git_status",
                    "tool_git_commit",
                    "tool_git_diff",
                    "tool_generate_image",
                    "tool_load_skill",
                ]
                if session.personality and hasattr(session.personality, "list_tools_structured"):
                    try:
                        for t in session.personality.list_tools_structured():
                            if t["name"] not in tool_choices:
                                tool_choices.append(t["name"])
                    except Exception:
                        pass

                # Extract available personas / handbags
                subws_data = agent_bridge.get_subws_tools_and_skills(session.personality, prefs, session.client)
                available_handbags = subws_data.get("available_handbags", [])
                persona_choices = {"": "Default Specialist (SubAgent)"}
                for hb in available_handbags:
                    persona_choices[hb["path"]] = f"👜 {hb['title']} ({hb['scope']})"

                # Extract available skills (from subws_data and personality)
                all_skills = []
                h_skills = subws_data.get("skills", {}).get("handbag", [])
                p_skills = subws_data.get("skills", {}).get("project", [])
                o_skills = subws_data.get("skills", {}).get("other", [])
                for s in h_skills + p_skills + o_skills:
                    if s.get("title") and s["title"] not in all_skills:
                        all_skills.append(s["title"])
                if not all_skills:
                    all_skills = ["file_organization", "git_workflow_mastery", "document_analysis_and_extraction", "desktop_automation", "fullstack_development", "bibliography_and_research", "deep_websearch_and_extraction"]

                with e_dlg, ui.card().classes(f"w-full h-full flex flex-col p-4 {CANVAS} rounded-xl border {BORDER} gap-3"):
                    with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("settings", size="22px").classes("text-primary")
                            ui.label(f"Configure Step: {node.name} ({node.node_type.name})").classes("text-sm font-bold")
                        ui.button(icon="close", on_click=e_dlg.close).props("flat dense round size=xs")

                    with ui.scroll_area().classes("w-full flex-1 p-2"):
                        with ui.column().classes("w-full max-w-3xl mx-auto gap-3"):
                            # General Settings
                            with ui.card().classes(f"w-full p-3 rounded border {BORDER} {SURFACE} gap-2"):
                                ui.label("GENERAL SETTINGS").classes("text-[10px] font-bold text-slate-500")
                                with ui.row().classes("w-full gap-2"):
                                    n_name_in = ui.input("Step Name", value=node.name).classes("flex-1 text-xs").props("outlined dense")
                                    n_id_in = ui.input("Step ID", value=node.id).classes("w-48 text-xs font-mono").props("outlined dense")
                                n_desc_in = ui.input("Description", value=node.description).classes("w-full text-xs").props("outlined dense")

                            # Subtask Model Override
                            with ui.card().classes(f"w-full p-3 rounded border {BORDER} {SURFACE} gap-2"):
                                ui.label("SUBTASK MODEL ROUTING").classes("text-[10px] font-bold text-slate-500")
                                ui.label("You can route this specific subtask to a different model (e.g. fast model for planning, coder model for implementation).").classes(f"text-[10px] {MUTED_DIM}")
                                curr_model = node.config.get("model_name") or node.config.get("model") or ""
                                if curr_model not in model_choices:
                                    model_choices[curr_model] = f"🤖 {curr_model}"
                                n_model_sel = ui.select(model_choices, value=curr_model, label="Assigned Model Profile").classes("w-full text-xs").props("outlined dense")

                            # Type-Specific Configuration
                            cfg_inputs = {}
                            with ui.card().classes(f"w-full p-3 rounded border {BORDER} {SURFACE} gap-2"):
                                ui.label(f"{node.node_type.name} PARAMETERS").classes("text-[10px] font-bold text-slate-500")

                                if node.node_type == NodeType.LLM:
                                    ui.label("Tip: Use {{variable_name}} to substitute state variables from prior steps.").classes(f"text-[10px] text-primary italic")
                                    cfg_inputs["prompt"] = ui.textarea("Prompt Template", value=node.config.get("prompt", "")).classes("w-full text-xs font-mono").props("outlined dense rows=4")
                                    cfg_inputs["system_prompt"] = ui.textarea("System Prompt Override", value=node.config.get("system_prompt", "")).classes("w-full text-xs").props("outlined dense rows=2")
                                    with ui.row().classes("w-full gap-2"):
                                        cfg_inputs["temperature"] = ui.number("Temperature", value=float(node.config.get("temperature", 0.3)), step=0.05).classes("flex-1 text-xs").props("outlined dense")
                                        cfg_inputs["output_variable"] = ui.input("Output Variable Key", value=node.config.get("output_variable", "llm_output")).classes("flex-1 text-xs font-mono").props("outlined dense")

                                elif node.node_type == NodeType.AGENT:
                                    # 1. Handbag Persona Selection
                                    curr_hb = node.config.get("handbag_path", "")
                                    if curr_hb and curr_hb not in persona_choices:
                                        persona_choices[curr_hb] = f"👜 {Path(curr_hb).name}"

                                    with ui.row().classes("w-full items-center gap-2"):
                                        ui.icon("face", size="20px").classes("text-purple-500")
                                        cfg_inputs["handbag_path"] = ui.select(
                                            persona_choices,
                                            value=curr_hb,
                                            label="Select Agent Persona (from Handbag)"
                                        ).classes("flex-1 text-xs").props("outlined dense options-dense")

                                    cfg_inputs["instruction"] = ui.textarea(
                                        "Task Directive (Headless Worker)",
                                        value=node.config.get("instruction", node.config.get("task", ""))
                                    ).classes("w-full text-xs font-mono").props("outlined dense rows=3").tooltip("Specific task with {{variable}} interpolation support")

                                    cfg_inputs["personality_conditioning"] = ui.textarea(
                                        "Persona Conditioning / System Directives",
                                        value=node.config.get("personality_conditioning", node.config.get("persona", ""))
                                    ).classes("w-full text-xs").props("outlined dense rows=2")

                                    # 2. Skills Multiselect Chips
                                    curr_skills = node.config.get("skills", [])
                                    if not isinstance(curr_skills, list):
                                        curr_skills = [curr_skills] if curr_skills else []

                                    with ui.column().classes("w-full gap-1 pt-1"):
                                        with ui.row().classes("items-center justify-between w-full"):
                                            ui.label("Assigned Skills (Methodologies injected into this Agent):").classes("text-xs font-semibold text-amber-500")
                                            ui.badge(f"{len(curr_skills)} active", color="amber").props("dense rounded text-[9px]")

                                        skills_select = ui.select(
                                            all_skills,
                                            value=curr_skills,
                                            multiple=True,
                                            label="Pick Skills to assign to this Agent node"
                                        ).classes("w-full text-xs").props("outlined dense use-chips clearable")
                                        cfg_inputs["skills"] = skills_select

                                    # 3. Tools Multiselect Chips
                                    curr_agent_tools = node.config.get("tools", [])
                                    if not isinstance(curr_agent_tools, list):
                                        curr_agent_tools = [curr_agent_tools] if curr_agent_tools else []

                                    with ui.column().classes("w-full gap-1 pt-1"):
                                        with ui.row().classes("items-center justify-between w-full"):
                                            ui.label("Allowed Tools (Capabilities granted to this Agent):").classes("text-xs font-semibold text-emerald-500")
                                            ui.badge(f"{len(curr_agent_tools)} active", color="emerald").props("dense rounded text-[9px]")

                                        tools_select = ui.select(
                                            tool_choices,
                                            value=curr_agent_tools,
                                            multiple=True,
                                            label="Pick Tools granted to this Agent"
                                        ).classes("w-full text-xs font-mono").props("outlined dense use-chips clearable")
                                        cfg_inputs["tools"] = tools_select

                                    with ui.row().classes("w-full gap-2 pt-1"):
                                        cfg_inputs["max_steps"] = ui.number("Max Reasoning Budget", value=int(node.config.get("max_steps", 6)), step=1).classes("flex-1 text-xs").props("outlined dense")
                                        cfg_inputs["output_variable"] = ui.input("Output Variable Key", value=node.config.get("output_variable", "agent_output")).classes("flex-1 text-xs font-mono").props("outlined dense")

                                elif node.node_type == NodeType.TOOL:
                                    curr_tool = node.config.get("tool_name", "tool_organize_files_from_plan")
                                    if curr_tool not in tool_choices:
                                        tool_choices.insert(0, curr_tool)
                                    cfg_inputs["tool_name"] = ui.select(tool_choices, value=curr_tool, label="Tool to Execute", new_value_mode="add-unique").classes("w-full text-xs font-mono").props("outlined dense")
                                    raw_p = node.config.get("tool_params", {})
                                    params_str = json.dumps(raw_p, indent=2) if isinstance(raw_p, dict) else str(raw_p)
                                    cfg_inputs["tool_params_str"] = ui.textarea("Tool Parameters (JSON with {{var}})", value=params_str).classes("w-full text-xs font-mono").props("outlined dense rows=4")
                                    cfg_inputs["output_variable"] = ui.input("Output Variable Key", value=node.config.get("output_variable", "tool_output")).classes("w-full text-xs font-mono").props("outlined dense")

                                elif node.node_type == NodeType.GUARD:
                                    g_types = [g.value for g in GuardType]
                                    curr_gt = node.config.get("guard_type", "file_exists")
                                    if curr_gt not in g_types:
                                        g_types.insert(0, curr_gt)
                                    cfg_inputs["guard_type"] = ui.select(g_types, value=curr_gt, label="Guard Invariant Type").classes("w-full text-xs").props("outlined dense")
                                    cfg_inputs["guard_target"] = ui.input("Guard Target (File or State Key)", value=node.config.get("guard_target", "")).classes("w-full text-xs font-mono").props("outlined dense")
                                    cfg_inputs["guard_pattern"] = ui.input("Regex Pattern / Expression (Optional)", value=node.config.get("guard_pattern", "")).classes("w-full text-xs font-mono").props("outlined dense")

                                elif node.node_type == NodeType.GATE:
                                    cfg_inputs["prompt"] = ui.input("Confirmation Prompt to Operator", value=node.config.get("prompt", "Do you approve proceeding with the next step?")).classes("w-full text-xs").props("outlined dense")

                                elif node.node_type == NodeType.CONDITION:
                                    cfg_inputs["condition_expr"] = ui.input("Python Expression (e.g. bool(state.get('task_prompt')))", value=node.config.get("condition_expr", "")).classes("w-full text-xs font-mono").props("outlined dense")

                                elif node.node_type == NodeType.TERMINAL:
                                    cfg_inputs["summary"] = ui.textarea("Summary Output Template", value=node.config.get("summary", "Workflow completed.\n{{output}}")).classes("w-full text-xs font-mono").props("outlined dense rows=4")

                            # Transitions & Edge Connections
                            with ui.card().classes(f"w-full p-3 rounded border {BORDER} {SURFACE} gap-2"):
                                ui.label("TRANSITIONS & NEXT STEPS").classes("text-[10px] font-bold text-slate-500")
                                other_node_ids = {n_id: f"{n.name} ({n_id})" for n_id, n in wf.nodes.items() if n_id != node.id}
                                other_node_ids[""] = "(None / End)"
                                succ_val = node.on_success if node.on_success in other_node_ids else ""
                                fail_val = node.on_failure if node.on_failure in other_node_ids else ""

                                with ui.row().classes("w-full gap-2"):
                                    n_succ_in = ui.select(other_node_ids, value=succ_val, label="Next Step on Success / Default").classes("flex-1 text-xs").props("outlined dense")
                                    n_fail_in = ui.select(other_node_ids, value=fail_val, label="Target on Failure / Rejection").classes("flex-1 text-xs").props("outlined dense")
                                    
                    # Modal Footer
                    with ui.row().classes(f"w-full items-center justify-end gap-2 pt-2 border-t {BORDER} shrink-0"):
                        ui.button("Cancel", on_click=e_dlg.close).props("flat dense no-caps")
                        def _save_node_changes():
                            node.name = n_name_in.value.strip() or node.name
                            node.description = n_desc_in.value.strip()
                            node.on_success = n_succ_in.value or None
                            node.on_failure = n_fail_in.value or None

                            if n_model_sel.value:
                                node.config["model_name"] = n_model_sel.value
                            else:
                                node.config.pop("model_name", None)

                            for k, widget in cfg_inputs.items():
                                if k == "tool_params_str":
                                    try:
                                        node.config["tool_params"] = json.loads(widget.value)
                                    except Exception:
                                        node.config["tool_params"] = {}
                                elif k == "handbag_path":
                                    val = widget.value
                                    if val:
                                        node.config["handbag_path"] = val
                                        node.config["persona_name"] = Path(val).name
                                    else:
                                        node.config.pop("handbag_path", None)
                                        node.config.pop("persona_name", None)
                                else:
                                    node.config[k] = widget.value

                            e_dlg.close()
                            refresh_studio_canvas()
                            refresh_pipeline_ui()
                            _refresh_variable_inputs()
                            ui.notify(f"Updated step '{node.name}'", type="positive")

                        ui.button("Save Step Config", icon="check", on_click=_save_node_changes).props("unelevated dense color=primary no-caps font-bold")

                e_dlg.open()

            def refresh_pipeline_ui():
                pipeline_container.clear()
                wf = active_state["wf"]

                # Update Start Node dropdown options
                node_keys = {nid: f"{n.name} ({nid})" for nid, n in wf.nodes.items()}
                start_node_select.options = node_keys
                if wf.start_node_id not in node_keys and node_keys:
                    wf.start_node_id = next(iter(node_keys))
                start_node_select.value = wf.start_node_id

                node_items = list(wf.nodes.items())

                with pipeline_container:
                    if not node_items:
                        ui.label("No steps in pipeline. Click 'Add Step' above.").classes(f"text-xs {MUTED_DIM} p-4 italic text-center w-full")
                        return

                    for idx, (nid, node) in enumerate(node_items, 1):
                        type_styles = {
                            NodeType.LLM: ("border-l-blue-500", "psychology", "LLM GENERATION", "blue"),
                            NodeType.AGENT: ("border-l-purple-500", "smart_toy", "SPECIALIST AGENT", "purple"),
                            NodeType.TOOL: ("border-l-emerald-500", "build", "DETERMINISTIC TOOL", "emerald"),
                            NodeType.GUARD: ("border-l-red-500", "shield", "HARD GUARD WALL", "red"),
                            NodeType.GATE: ("border-l-amber-500", "pan_tool", "OPERATOR GATE", "amber"),
                            NodeType.CONDITION: ("border-l-cyan-500", "call_split", "BRANCH CONDITION", "cyan"),
                            NodeType.TERMINAL: ("border-l-slate-500", "flag", "TERMINAL FINISH", "slate"),
                        }
                        border_cls, icon_name, badge_label, badge_col = type_styles.get(node.node_type, ("border-l-slate-700", "radio_button_checked", "STEP", "slate"))
                        is_start = (nid == wf.start_node_id)

                        with ui.card().classes(f"w-full p-2.5 rounded-lg border {BORDER} border-l-4 {border_cls} {SURFACE} gap-1 shadow-sm"):
                            with ui.row().classes("w-full items-center justify-between flex-nowrap"):
                                with ui.row().classes("items-center gap-1.5 flex-1 min-w-0"):
                                    ui.icon(icon_name, size="16px").classes(f"text-{badge_col}-500 shrink-0")
                                    ui.label(f"{idx}. {node.name}").classes("text-xs font-bold truncate text-slate-900 dark:text-slate-100")
                                    if is_start:
                                        ui.badge("START", color="emerald").props("dense rounded text-[9px]")

                                with ui.row().classes("items-center gap-0.5 shrink-0"):
                                    ui.badge(badge_label, color=badge_col).props("dense rounded text-[9px]")

                                    # Move Step Up
                                    if idx > 1:
                                        def _move_up(i=idx-1):
                                            items = list(wf.nodes.items())
                                            items[i], items[i-1] = items[i-1], items[i]
                                            wf.nodes = dict(items)
                                            refresh_pipeline_ui()
                                        ui.button(icon="arrow_upward", on_click=_move_up).props("flat dense round size=xs color=grey").tooltip("Move step up")

                                    # Move Step Down
                                    if idx < len(node_items):
                                        def _move_down(i=idx-1):
                                            items = list(wf.nodes.items())
                                            items[i], items[i+1] = items[i+1], items[i]
                                            wf.nodes = dict(items)
                                            refresh_pipeline_ui()
                                        ui.button(icon="arrow_downward", on_click=_move_down).props("flat dense round size=xs color=grey").tooltip("Move step down")

                                    ui.button(icon="settings", on_click=lambda n=node: _open_node_editor(n)).props("flat dense round size=xs color=primary").tooltip("Configure Step")

                                    def _delete_node(target_id=nid):
                                        wf.nodes.pop(target_id, None)
                                        wf.edges = [e for e in wf.edges if e.source_id != target_id and e.target_id != target_id]
                                        refresh_pipeline_ui()
                                    ui.button(icon="delete", on_click=_delete_node).props("flat dense round size=xs color=red").tooltip("Delete step")

                            # Config summary line
                            assigned_m = node.config.get("model_name") or node.config.get("model")
                            summary_parts = []
                            if assigned_m:
                                summary_parts.append(f"Model: {assigned_m}")
                            if node.node_type == NodeType.GUARD:
                                summary_parts.append(f"Wall: [{node.config.get('guard_type', '')}] on '{node.config.get('guard_target', '')}'")
                            elif node.node_type == NodeType.TOOL:
                                summary_parts.append(f"Tool: {node.config.get('tool_name', '')}")
                            elif node.node_type == NodeType.GATE:
                                summary_parts.append(f"Prompt: {node.config.get('prompt', '')[:40]}...")
                            elif node.node_type == NodeType.LLM:
                                summary_parts.append(f"Prompt: {node.config.get('prompt', '')[:40]}...")

                            if summary_parts:
                                ui.label(" · ".join(summary_parts)).classes(f"text-[10px] {MUTED_DIM} font-mono truncate px-1")

                            # Edge / Transition indicators
                            with ui.row().classes("w-full items-center justify-between text-[10px] font-mono pt-1 border-t border-slate-200 dark:border-slate-800/80"):
                                succ_tgt = node.on_success or (node_items[idx][0] if idx < len(node_items) else "Finish")
                                ui.label(f"──> Next: {succ_tgt}").classes("text-slate-500 truncate")
                                if node.on_failure:
                                    ui.label(f"On Fail: ↺ {node.on_failure}").classes("text-red-400 font-bold truncate")

            def _reload_workflows_list(select_id: Optional[str] = None):
                all_wfs = list_project_workflows(prefs.workspace_path)
                options = {}
                for w in all_wfs:
                    badge = "[Project]" if w["source"] == "project" else "[Template]"
                    options[w["id"]] = f"{badge} {w['name']}"

                wf_select.options = options
                target_id = select_id or active_state["wf"].id
                if target_id not in options and options:
                    target_id = next(iter(options))

                wf_select.value = target_id
                _switch_active_workflow(target_id)

            def _switch_active_workflow(w_id: str):
                wf = load_project_workflow(prefs.workspace_path, w_id)
                if wf:
                    active_state["wf"] = wf
                    wf_name_input.value = wf.name
                    wf_desc_input.value = wf.description
                    node_opts = {nid: f"{n.name} ({nid})" for nid, n in wf.nodes.items()}
                    start_node_select.options = node_opts
                    if wf.start_node_id in node_opts:
                        start_node_select.value = wf.start_node_id
                    elif node_opts:
                        start_node_select.value = next(iter(node_opts))
                    else:
                        start_node_select.value = None
                    refresh_studio_canvas()

            wf_select.on_value_change(lambda e: _switch_active_workflow(e.value) if e.value else None)
            _reload_workflows_list()

        dialog.open()

    # ── Context & Generation Parameters Inspector Modal Dialog ──
    def open_context_inspector_dialog():
        current_input_text = (prompt_input.value or "").strip()
        dialog = ui.dialog().props("maximized")

        with dialog, ui.card().classes(
            f"w-full h-full flex flex-col p-4 {CANVAS} text-slate-900 dark:text-slate-100 gap-3 overflow-hidden"
        ):
            # Header
            with ui.row().classes(f"w-full items-center justify-between pb-2 border-b {BORDER} shrink-0"):
                with ui.row().classes("items-center gap-2.5"):
                    ui.icon("manage_search", size="28px").classes("text-cyan-500")
                    with ui.column().classes("gap-0"):
                        ui.label("Agent Context & Configuration Inspector").classes("text-base font-bold")
                        ui.label("Examine the verbatim messages payload, system prompt, and parameters sent to the LLM.").classes(
                            f"text-xs {MUTED_DIM}"
                        )

                with ui.row().classes("items-center gap-2"):
                    def _copy_full_report():
                        try:
                            diag = agent_bridge.get_context_preview(session, prefs, current_input_text)
                            lines = [
                                "# Agent Context & Parameter Inspection Report",
                                f"- Generated: {datetime.now().isoformat()}",
                                f"- Model: {diag['configuration']['model_name']} ({diag['configuration']['binding_name']})",
                                f"- Context Tokens: {diag['configuration']['total_tokens']:,} / {diag['configuration']['max_ctx']:,} ({diag['configuration']['fill_pct']}%)",
                                f"- Memory Active: {diag['configuration']['memory_enabled']} (DB: {diag['configuration']['memory_db_path']})",
                                "",
                                "## Active Configuration",
                                "```json",
                                json.dumps(diag['configuration'], indent=2),
                                "```",
                                "",
                                "## Verbatim Messages Payload",
                                "```json",
                                json.dumps(diag['messages'], indent=2),
                                "```",
                                "",
                                "## Complete System Prompt",
                                "```markdown",
                                diag['system_prompt'],
                                "```"
                            ]
                            ui.clipboard.write("\n".join(lines))
                            ui.notify("Complete context & config report copied to clipboard.", type="positive")
                        except Exception as ex:
                            notify_error(f"Failed to copy report: {ex}")

                    ui.button("Copy All (Markdown)", icon="content_copy", on_click=_copy_full_report).props(
                        "unelevated dense size=sm color=primary no-caps font-semibold"
                    )
                    ui.button("Refresh", icon="refresh", on_click=lambda: _refresh_inspector()).props("flat dense size=sm no-caps")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

            # Main content area with tabs
            with ui.tabs().classes(f"w-full {SURFACE} border-b {BORDER} shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as insp_tabs:
                tab_full = ui.tab('full_context', label='📜 Assembled Context (LLM View)', icon='terminal').classes('text-xs py-1.5 flex-1 font-bold')
                tab_msgs = ui.tab('messages', label='📨 Messages Payload', icon='chat').classes('text-xs py-1.5 flex-1')
                tab_sys = ui.tab('system', label='🧠 Complete System Prompt', icon='psychology').classes('text-xs py-1.5 flex-1')
                tab_mem = ui.tab('memory', label='💾 Active Memories & Handles', icon='memory').classes('text-xs py-1.5 flex-1')
                tab_tools = ui.tab('tools', label='🛠️ Active Tools Schema', icon='build').classes('text-xs py-1.5 flex-1')
                tab_config = ui.tab('config', label='⚙️ Generation Parameters', icon='tune').classes('text-xs py-1.5 flex-1')

            inspector_slot = ui.column().classes("w-full flex-1 min-h-0 overflow-hidden p-0 m-0")

            def _refresh_inspector():
                inspector_slot.clear()
                active_text = (prompt_input.value or "").strip() if prompt_input else ""
                try:
                    diag = agent_bridge.get_context_preview(session, prefs, active_text)
                except Exception as ex:
                    with inspector_slot:
                        ui.label(f"Failed to compile context preview: {ex}").classes("text-sm text-red-500 p-4")
                    return

                cfg = diag["configuration"]

                with inspector_slot:
                    # Metrics Banner
                    with ui.row().classes(f"w-full items-center justify-between px-3 py-2 {SURFACE} border-b {BORDER} shrink-0 text-xs flex-wrap gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.badge(f"Model: {cfg['model_name']}", color="indigo").props("rounded dense")
                            ui.badge(f"Binding: {cfg['binding_name']}", color="blue").props("rounded dense")
                            mem_badge_col = "purple" if cfg["memory_enabled"] and cfg["memory_manager_attached"] else "grey"
                            mem_badge_txt = f"Memory: {'ACTIVE' if cfg['memory_enabled'] and cfg['memory_manager_attached'] else 'OFF'}"
                            ui.badge(mem_badge_txt, color=mem_badge_col).props("rounded dense")
                            ui.badge(f"Effort: {cfg['reasoning_effort']}", color="amber").props("rounded dense")

                        with ui.row().classes("items-center gap-2 font-mono"):
                            pct_col = "text-green-500" if cfg["fill_pct"] < 65 else ("text-yellow-500" if cfg["fill_pct"] < 85 else "text-red-500 font-bold")
                            ui.label(f"Context: {cfg['total_tokens']:,} / {cfg['max_ctx']:,} tokens").classes(f"font-semibold {pct_col}")
                            ui.label(f"({cfg['fill_pct']}%)")
                            ui.label(f"| {len(diag['messages'])} Messages")

                    with ui.tab_panels(insp_tabs, value='full_context').classes('w-full flex-1 min-h-0 p-2 bg-transparent flex flex-col overflow-hidden'):
                        # --- Tab 0: Full Assembled Context (Exact LLM View) ---
                        with ui.tab_panel('full_context').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-2'):
                            with ui.row().classes("w-full items-center justify-between pb-1"):
                                full_ctx_str = diag.get("full_assembled_context", "")
                                est_tok = session.client.count_tokens(full_ctx_str) if (session.client and hasattr(session.client, "count_tokens")) else len(full_ctx_str) // 4
                                ui.label(f"Verbatim Assembled Context for Next Turn ({len(full_ctx_str):,} chars · ~{est_tok:,} tokens):").classes(f"text-xs font-semibold {STRONG}")

                                def _copy_full_ctx():
                                    ui.clipboard.write(full_ctx_str)
                                    ui.notify("Verbatim LLM context copied to clipboard.", type="positive")

                                ui.button("Copy Verbatim Context", icon="content_copy", on_click=_copy_full_ctx).props(
                                    "unelevated dense size=xs color=primary no-caps font-semibold"
                                )

                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-3 bg-slate-950 text-slate-100 shadow-inner"):
                                ui.label(full_ctx_str).classes("w-full text-xs font-mono whitespace-pre-wrap select-all m-0 leading-relaxed text-slate-200 block")

                        # --- Tab 1: Messages Payload ---
                        with ui.tab_panel('messages').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-2'):
                            with ui.row().classes("w-full items-center justify-between pb-1"):
                                ui.label(f"Exact messages list passed to generate_from_messages ({len(diag['messages'])} message(s)):").classes(f"text-xs {MUTED_DIM}")
                                def _copy_msgs():
                                    ui.clipboard.write(json.dumps(diag["messages"], indent=2))
                                    ui.notify("Messages JSON copied to clipboard.", type="positive")
                                ui.button("Copy JSON", icon="content_copy", on_click=_copy_msgs).props("flat dense size=xs no-caps text-color=primary")

                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2 {SURFACE}"):
                                with ui.column().classes("w-full gap-3"):
                                    for idx, msg in enumerate(diag["messages"]):
                                        role = msg.get("role", "unknown")
                                        content = msg.get("content", "")
                                        toks = msg.get("tokens", 0)

                                        role_colors = {
                                            "system": ("bg-purple-900/30 text-purple-300 border-purple-800", "purple"),
                                            "user": ("bg-blue-900/30 text-blue-300 border-blue-800", "blue"),
                                            "assistant": ("bg-emerald-900/30 text-emerald-300 border-emerald-800", "emerald"),
                                        }
                                        box_css, badge_col = role_colors.get(role, ("bg-slate-800 text-slate-300 border-slate-700", "slate"))

                                        with ui.card().classes(f"w-full p-2.5 rounded-lg border {box_css} gap-1 shadow-none"):
                                            with ui.row().classes("w-full items-center justify-between"):
                                                with ui.row().classes("items-center gap-2"):
                                                    ui.badge(f"[{idx}] {role.upper()}", color=badge_col).props("dense rounded text-[10px]")
                                                    ui.label(f"~{toks:,} tokens").classes(f"text-[10px] {MUTED_DIM} font-mono")
                                                def _copy_single_msg(text_to_copy=content):
                                                    txt = text_to_copy if isinstance(text_to_copy, str) else json.dumps(text_to_copy, indent=2)
                                                    ui.clipboard.write(txt)
                                                    ui.notify(f"Message {idx} copied.", type="positive")
                                                ui.button(icon="content_copy", on_click=lambda c=content: _copy_single_msg(c)).props("flat dense round size=xs color=grey")

                                            with ui.scroll_area().classes("w-full max-h-64 p-2 bg-slate-950/80 rounded border border-slate-800/60"):
                                                if isinstance(content, str):
                                                    ui.label(content).classes("w-full text-xs font-mono whitespace-pre-wrap select-all m-0 leading-relaxed text-slate-200 block")
                                                else:
                                                    ui.code(json.dumps(content, indent=2), language="json").classes("w-full text-xs")

                        # --- Tab 2: Complete System Prompt ---
                        with ui.tab_panel('system').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-2'):
                            with ui.row().classes("w-full items-center justify-between pb-1"):
                                sys_prompt_str = diag.get("system_prompt", "")
                                sys_toks = session.client.count_tokens(sys_prompt_str) if hasattr(session.client, "count_tokens") else len(sys_prompt_str) // 4
                                ui.label(f"Full System Prompt ({len(sys_prompt_str):,} chars · ~{sys_toks:,} tokens):").classes(f"text-xs {MUTED_DIM}")
                                def _copy_sys():
                                    ui.clipboard.write(sys_prompt_str)
                                    ui.notify("System prompt copied.", type="positive")
                                ui.button("Copy Prompt", icon="content_copy", on_click=_copy_sys).props("flat dense size=xs no-caps text-color=primary")

                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-3 bg-slate-900 text-slate-100 shadow-inner"):
                                ui.label(sys_prompt_str).classes("w-full text-xs font-mono whitespace-pre-wrap select-all m-0 leading-relaxed text-slate-200 block")

                        # --- Tab 3: Active Memories & Deep Handles ---
                        with ui.tab_panel('memory').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-3'):
                            with ui.card().classes(f"w-full p-3 rounded-lg border {BORDER} {SURFACE} gap-2"):
                                ui.label("🧠 Persistent Memory Telemetry").classes("text-sm font-bold text-primary")
                                with ui.row().classes("w-full items-center gap-3 text-xs font-mono"):
                                    ui.label(f"Active in Prefs: {'YES' if cfg['memory_enabled'] else 'NO'}").classes("font-semibold")
                                    ui.label(f"Manager Attached: {'YES' if cfg['memory_manager_attached'] else 'NO'}")
                                    ui.label(f"Total Database Records: {cfg['memory_counts']['total']}")
                                    ui.label(f"(L1 Working: {cfg['memory_counts']['working']} | L2 Deep: {cfg['memory_counts']['deep']} | L3 Archived: {cfg['memory_counts']['archived']})")
                                ui.label(f"SQLite DB Path: {cfg['memory_db_path']}").classes(f"text-[10px] {MUTED_DIM} font-mono select-all")

                            with ui.row().classes("w-full flex-1 min-h-0 gap-3 flex-nowrap"):
                                with ui.column().classes("flex-1 h-full min-w-0 flex flex-col gap-1"):
                                    ui.label("Level 1: Working Memory Zone (Injected Verbatim into Context)").classes("text-xs font-bold text-emerald-500")
                                    with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2 {SURFACE}"):
                                        ui.code(diag["memory_working_zone"], language="markdown").classes("w-full text-xs")

                                with ui.column().classes("flex-1 h-full min-w-0 flex flex-col gap-1"):
                                    ui.label("Level 2: Deep Memory Handles Zone (Injected as Compact Handles)").classes("text-xs font-bold text-amber-500")
                                    with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2 {SURFACE}"):
                                        ui.code(diag["memory_handles_zone"], language="markdown").classes("w-full text-xs")

                        # --- Tab 4: Active Tools Schema ---
                        with ui.tab_panel('tools').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-2'):
                            ui.label(f"Active Tool Registry ({len(diag['active_tools'])} tool(s) registered for this turn):").classes(f"text-xs {MUTED_DIM}")
                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-2 {SURFACE}"):
                                with ui.column().classes("w-full gap-2"):
                                    for t_name, t_spec in sorted(diag["active_tools"].items()):
                                        with ui.card().classes(f"w-full p-2.5 rounded border {BORDER} bg-slate-900/60 gap-1 shadow-none"):
                                            with ui.row().classes("w-full items-center justify-between"):
                                                ui.label(t_name).classes("text-xs font-mono font-bold text-primary")
                                                ui.label(f"{len(t_spec.get('parameters', []))} parameter(s)").classes(f"text-[10px] {MUTED_DIM}")
                                            ui.label(t_spec.get("description", "(No description)")).classes(f"text-[11px] {MUTED}")
                                            if t_spec.get("parameters"):
                                                with ui.row().classes("gap-1 pt-1 flex-wrap"):
                                                    for p in t_spec["parameters"]:
                                                        ui.badge(f"{p.get('name')}: {p.get('type')}", color="slate").props("dense rounded text-[9px]")

                        # --- Tab 5: Generation Parameters ---
                        with ui.tab_panel('config').classes('w-full h-full p-0 flex flex-col overflow-hidden gap-2'):
                            ui.label("Runtime Execution Parameters:").classes(f"text-xs {MUTED_DIM}")
                            with ui.scroll_area().classes(f"w-full flex-1 border {BORDER} rounded p-3 {SURFACE}"):
                                ui.code(json.dumps(cfg, indent=2), language="json").classes("w-full text-xs")

            _refresh_inspector()

        dialog.open()