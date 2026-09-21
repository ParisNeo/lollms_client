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
    from memory_explorer import open_memory_explorer_dialog
except ImportError:
    try:
        from lollms_client.apps.lollms_code.gui.memory_explorer import open_memory_explorer_dialog
    except ImportError:
        open_memory_explorer_dialog = None
from pathlib import Path
import json


HELP_TEXT = """\
**Commands**

- `/help` — this list
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
    ("/memories", "Open interactive Memory Explorer"),
    ("/memory", "Open interactive Memory Explorer"),
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
    ("Ctrl+F", "Search the conversation"),
    ("Esc", "Close search"),
    ("Ctrl+/", "Show this shortcut list"),
    ("Ctrl+Shift+C", "Copy the last agent message"),
    ("Theme button", "Cycle Auto → Light → Dark (Auto follows the clock)"),
]


class ChatSession:
    def __init__(self, env: EnvStore, prefs: GuiPrefs):
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

    def ensure_ready(self):
        if self.client is None:
            self.client = agent_bridge.create_client(self.env, self.prefs)
        if self.personality is None:
            self.personality = agent_bridge.create_personality(self.prefs, self.client)


def build_chat_page(env: EnvStore, prefs: GuiPrefs, tools_toggle=None) -> None:
    session = ChatSession(env, prefs)
    debug_log: List[Dict[str, Any]] = []
    message_refs: Dict[str, Dict[str, Any]] = {}
    last_agent_state: Dict[str, Any] = {"entry": None}
    _msg_counter = 0

    def next_message_id() -> int:
        nonlocal _msg_counter
        _msg_counter += 1
        return _msg_counter

    show_tree_sidebar = True

    # ---- Theme ----------------------------------------------------------
    # Created FIRST so every widget below can reference it. NiceGUI's
    # ui.dark_mode drives Quasar's global dark plugin, so any component that
    # does NOT carry a hardcoded `dark` prop follows it automatically.
    #
    # Three modes, cycled by the toolbar button: auto → light → dark → auto.
    # "auto" follows the wall clock: light during the day, dark at night.
    THEME_MODES = ("auto", "light", "dark")
    DAY_START_HOUR = int(getattr(prefs, "theme_day_start_hour", 7))    # ≥ this hour → light
    NIGHT_START_HOUR = int(getattr(prefs, "theme_night_start_hour", 19))  # ≥ this hour → dark

    theme_state = {"mode": getattr(prefs, "theme_mode", "auto")}
    if theme_state["mode"] not in THEME_MODES:
        theme_state["mode"] = "auto"

    def _is_night(now: Optional[datetime] = None) -> bool:
        hour = (now or datetime.now()).hour
        if DAY_START_HOUR <= NIGHT_START_HOUR:
            # Normal case, e.g. day 07:00–19:00.
            return hour < DAY_START_HOUR or hour >= NIGHT_START_HOUR
        # Inverted window (day spans midnight) — rare, but don't break on it.
        return NIGHT_START_HOUR <= hour < DAY_START_HOUR

    def _resolve_dark(mode: str) -> bool:
        if mode == "light":
            return False
        if mode == "dark":
            return True
        return _is_night()

    dark_mode = ui.dark_mode(value=_resolve_dark(theme_state["mode"]))

    # Dual-mode Tailwind tokens with verified high contrast in both themes.
    SURFACE = "bg-slate-100 dark:bg-slate-900"
    SURFACE_ALT = "bg-slate-100/80 dark:bg-slate-900/80"
    CANVAS = "bg-white dark:bg-slate-950"
    BORDER = "border-slate-300 dark:border-slate-800"
    MUTED = "text-slate-700 dark:text-slate-300"
    MUTED_DIM = "text-slate-600 dark:text-slate-400"
    STRONG = "text-slate-900 dark:text-slate-100"

    with ui.column().classes("w-full h-full flex-1 min-h-0 flex-nowrap gap-0 overflow-hidden flex flex-col"):
        # ---- Slim status strip (replaces the old sidebar cards) ----
        with ui.row().classes(
            f"w-full items-center justify-between px-3 py-1.5 shrink-0 {SURFACE} border-b {BORDER}"
        ):
            with ui.row().classes("items-center gap-2"):
                tree_toggle_btn = ui.button(
                    "Tree", icon="folder",
                    on_click=lambda: toggle_tree_visibility(),
                ).props("flat dense size=sm no-caps text-color=primary").tooltip("Toggle Workspace Tree")
                status_label = ui.label("Idle").classes(f"text-xs {MUTED} font-mono font-medium")
                elapsed_label = ui.label("").classes(f"text-xs {MUTED_DIM} font-mono")

            with ui.row().classes("items-center gap-2"):
                rounds_label = ui.label("").classes(f"text-xs {MUTED} font-mono font-medium")
                ctx_label = ui.label("").classes(f"text-xs {MUTED} font-mono font-medium")
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
                    icon="brightness_auto",
                    on_click=lambda: toggle_theme(),
                ).props("flat dense round size=sm").tooltip("Theme: Auto / Light / Dark")
                ui.button(
                    icon="keyboard",
                    on_click=lambda: open_shortcuts_dialog(),
                ).props("flat dense round size=sm").tooltip("Keyboard shortcuts (Ctrl+/)")
                ui.button(
                    "Memories", icon="psychology",
                    on_click=lambda: open_memory_explorer_dialog(session, prefs) if open_memory_explorer_dialog else ui.notify("Memory Explorer not available", type="warning"),
                ).props("flat dense size=sm no-caps text-color=purple").tooltip("Open Memory Explorer (inspect, edit, dream)")
                ui.button(
                    "Scratchpad", icon="edit_note",
                    on_click=lambda: open_scratchpad_dialog(),
                ).props("flat dense size=sm no-caps").tooltip("View the agent's persistent notes and thoughts")
                ui.button(
                    "Copy as Markdown", icon="content_copy",
                    on_click=lambda: copy_debug_markdown(),
                ).props("flat dense size=sm no-caps").tooltip("Copy the full discussion, including tool calls, for debugging")
                ui.button(
                    "Export History", icon="download",
                    on_click=lambda: export_history(),
                ).props("flat dense size=sm no-caps").tooltip("Download the full session as a Markdown file")
                ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
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
                "flex-1"
            ).props("dense outlined input-debounce=100")
            search_count_label = ui.label("").classes(f"text-xs {MUTED_DIM} font-mono")
            ui.button(icon="close", on_click=lambda: close_search()).props("flat dense round size=xs")

        # ---- Center Workspace Area (Tree View + Chat Transcript + Telemetry) ----
        with ui.row().classes("w-full flex-1 min-h-0 items-stretch overflow-hidden flex-nowrap gap-0"):

            # ---- Left-hand Lazy Workspace Tree Panel ----
            tree_panel = ui.column().classes(
                f"w-72 h-full shrink-0 border-r {BORDER} {SURFACE_ALT} overflow-hidden flex flex-col p-0 gap-0"
            )
            with tree_panel:
                with ui.row().classes(f"w-full items-center justify-between px-3 py-2 border-b {BORDER} shrink-0"):
                    ui.label("📁 Workspace").classes(f"text-xs font-bold {STRONG}")
                    with ui.row().classes("gap-1"):
                        ui.button(icon="upload_file", on_click=lambda: upload_dialog.open()).props(
                            "flat round dense size=xs"
                        ).tooltip("Upload a file into the workspace root")
                        ui.button(icon="refresh", on_click=lambda: refresh_workspace_tree()).props(
                            "flat round dense size=xs"
                        ).tooltip("Refresh tree")

                with ui.row().classes("w-full px-2 pt-2 shrink-0"):
                    tree_search_input = ui.input(placeholder="Filter files…").props(
                        "dense outlined clearable"
                    ).classes("w-full")

                tree_scroll = ui.scroll_area().classes("w-full flex-1 p-2")
                with tree_scroll:
                    tree_container = ui.column().classes("w-full gap-0 p-0")

            # ---- Transcript ----
            scroll_area = ui.scroll_area().classes(f"flex-1 h-full min-w-0 {CANVAS}")
            with scroll_area:
                transcript = ui.column().classes("w-full gap-3 p-4")

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
                    ui.label("🎓 Skills (live)").classes(f"text-xs font-bold {STRONG}")
                    skills_label = ui.label("—").classes(f"text-xs {MUTED}")
                    ui.separator().classes(f"my-1 {BORDER}")
                    peek_button = ui.button(
                        "🔍 Peek Scratchpad", icon="visibility",
                        on_click=lambda: open_scratchpad_dialog(),
                    ).props("flat dense size=xs no-caps").classes("w-full")
                    peek_button.tooltip("Read the agent's scratchpad right now — safe while generating")
                    scratchpad_badge = ui.badge("0", color="blue").props("floating").bind_visibility_from(
                        session, "current_round", backward=lambda r: session.busy and r > 0
                    )

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
                    "w-full bg-white dark:bg-slate-800/80 text-slate-900 dark:text-slate-100 rounded"
                ).props("outlined autogrow dense rows=1 input-debounce=0")
                input_counter = ui.label("").classes(f"text-[10px] {MUTED_DIM} self-end pr-1")
            regen_button = ui.button(icon="autorenew", on_click=lambda: regenerate_last()).props(
                "round flat"
            ).tooltip("Regenerate last response")
            stop_button = ui.button(icon="stop", on_click=lambda: stop_generation()).props(
                "round color=negative"
            ).tooltip("Stop generation")
            stop_button.bind_visibility_from(session, "busy")
            send_button = ui.button(icon="send").props("round color=primary")

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

    def add_user_bubble(text: str) -> str:
        msg_id = f"msg_{next_message_id()}"
        entry = {"type": "user", "text": text, "id": msg_id}
        debug_log.append(entry)
        with transcript:
            outer_row = ui.row().classes("w-full justify-end items-start gap-1")
            with outer_row:
                actions = ui.row().classes("opacity-40 hover:opacity-100 transition-opacity gap-0")
                with actions:
                    edit_btn = ui.button(icon="edit", on_click=lambda: open_edit_dialog(msg_id)).props(
                        "flat round dense size=xs"
                    )
                    edit_btn.tooltip("Edit this prompt")
                    rerun_btn = ui.button(icon="replay", on_click=lambda: rerun_prompt(msg_id)).props(
                        "flat round dense size=xs"
                    )
                    rerun_btn.tooltip("Rerun this prompt")
                    del_btn = ui.button(icon="delete", on_click=lambda: delete_message(msg_id)).props(
                        "flat round dense size=xs color=red"
                    )
                    del_btn.tooltip("Delete message")
                bubble = ui.markdown(text).classes(
                    "bg-primary text-white rounded-lg px-3 py-1.5 max-w-[75%]"
                )
        message_refs[msg_id] = {"entry": entry, "row": outer_row, "bubble": bubble, "text": text}
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
        ui.notify(f"{toast} — full details in the transcript card (copyable).", type="negative", timeout=5000)

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
        scroll_area.scroll_to(percent=1.0)

    def add_event_panel(title: str, subtitle: str, body: str, color: str, icon: str) -> ui.expansion:
        entry = {"type": "event", "title": title, "subtitle": subtitle, "body": body}
        debug_log.append(entry)
        border_color_class = f"border-{color}" if color.startswith("red") or color.startswith("green") or color.startswith("blue") or color.startswith("amber") or color.startswith("purple") else "border-primary"
        with transcript:
            panel = ui.expansion(title, icon=icon).classes(
                f"w-full max-w-[90%] border-l-4 {border_color_class} bg-slate-100/90 dark:bg-slate-900/80 "
                f"border border-slate-300 dark:border-slate-800 text-slate-900 dark:text-slate-100 rounded-r-md text-xs shadow-sm"
            ).props('header-class="text-slate-900 dark:text-slate-100 font-semibold text-xs py-1.5"')
            panel.bind_visibility_from(tools_toggle, "value")
            with panel:
                with ui.row().classes("w-full items-center justify-between mb-1"):
                    if subtitle:
                        ui.label(subtitle).classes("text-xs text-slate-600 dark:text-slate-400 font-mono")
                    copy_btn = ui.button(
                        icon="content_copy",
                        on_click=lambda e_ref=entry: copy_event_body(e_ref),
                    ).props("flat round dense size=xs color=grey")
                    copy_btn.tooltip("Copy panel content")
                ui.code(body or "(no output)").classes(
                    "w-full text-xs bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 p-2.5 rounded border border-slate-300 dark:border-slate-800"
                )
        return panel

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

    def rerun_prompt(msg_id: str):
        """Re-sends a previously sent prompt as a fresh agent turn (history preserved)."""
        entry = find_message_entry(msg_id)
        if entry is None or entry.get("type") != "user":
            ui.notify("Cannot rerun this message.", type="warning")
            return
        prompt_text = entry.get("text", "").strip()
        if not prompt_text:
            ui.notify("Nothing to rerun.", type="warning")
            return
        if session.busy:
            ui.notify("Agent is busy — wait for the current turn to finish.", type="warning")
            return
        if prompt_text.startswith("/"):
            async def _rerun_slash():
                await send_prompt_with_text(prompt_text)
            ui.timer(0.1, _rerun_slash, once=True)
            return
        send_prompt_with_text(prompt_text, rerun_marker=True)

    def regenerate_last():
        """Re-sends the most recent user prompt (skips /slash commands)."""
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
        prompt_text = re.sub(r"^🔁 _Rerun:_\s*", "", last_user.get("text", "").strip())
        if not prompt_text or prompt_text.startswith("/"):
            ui.notify("Nothing to regenerate.", type="warning")
            return

        async def _go():
            await send_prompt_with_text(prompt_text, rerun_marker=True)

        ui.timer(0.05, _go, once=True)

    def stop_generation():
        if not session.busy:
            return
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

            if cancelled:
                add_system_notice("⏹️ Stop requested — cancelling generation...")
                ui.notify("Stopping generation...", type="info")
            else:
                ui.notify("Cancellation isn't supported by this agent backend yet.", type="warning")
        except Exception as e:
            notify_error(f"Could not stop generation: {e}")

    async def send_prompt_with_text(prompt_text: str, rerun_marker: bool = False):
        """Shared send pipeline used by both the input box and rerun/edit actions."""
        if not prompt_text.strip() or session.busy:
            return
        display_text = f"🔁 _Rerun:_ {prompt_text}" if rerun_marker else prompt_text
        add_user_bubble(display_text)
        if prompt_text.strip().startswith("/"):
            await handle_slash_command(prompt_text.strip())
            return
        _remember_prompt(prompt_text.strip())
        session.busy = True
        session.turn_start_ts = time.time()
        send_button.props("loading")
        status_label.set_text("Thinking…")
        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
            session.turn_start_ts = None
            send_button.props(remove="loading")
            return
        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, prompt_text.strip(), prefs, session.event_queue, use_history=True
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
                debug_log.remove(entry)
            except ValueError:
                pass
        if ref is not None and ref.get("row") is not None:
            try:
                ref["row"].delete()
            except Exception:
                pass
        ui.notify("Message deleted.", type="positive")

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
        transcript.clear()
        debug_log.clear()
        message_refs.clear()
        if session.personality is not None:
            try:
                session.personality._conversation = []
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

    def open_scratchpad_dialog():
        dialog = ui.dialog().props("maximized")
        with dialog, ui.card().classes("w-full h-full flex flex-col"):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("📝 Agent Scratchpad").classes("text-lg font-bold")
                with ui.row().classes("gap-2"):
                    def _refresh_scratchpad():
                        try:
                            session.ensure_ready()
                            content = agent_bridge.get_scratchpad_content(session.personality)
                            scratchpad_md.set_content(content if content.strip() else "_(scratchpad is empty)_")
                            ui.notify("Scratchpad refreshed.", type="positive")
                        except Exception as e:
                            ui.notify(f"Failed to read scratchpad: {e}", type="negative")
                    ui.button("Refresh", icon="refresh", on_click=_refresh_scratchpad).props("flat size=sm no-caps")
                    ui.button("Close", icon="close", on_click=dialog.close).props("flat size=sm no-caps")
            scratchpad_md = ui.markdown("").classes("flex-1 overflow-auto p-2 bg-gray-50 dark:bg-gray-900 rounded")
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

    THEME_ICONS = {"auto": "brightness_auto", "light": "light_mode", "dark": "dark_mode"}

    def _theme_tooltip() -> str:
        mode = theme_state["mode"]
        if mode == "auto":
            return (
                f"Theme: Auto — light from {DAY_START_HOUR:02d}:00, "
                f"dark from {NIGHT_START_HOUR:02d}:00 (click for Light)"
            )
        if mode == "light":
            return "Theme: Light (click for Dark)"
        return "Theme: Dark (click for Auto)"

    def _sync_theme_button():
        # Re-issuing `.props("icon=…")` appends rather than replaces, so Quasar
        # can keep the stale icon. Set the prop dict directly and force an update.
        theme_btn._props["icon"] = THEME_ICONS[theme_state["mode"]]
        theme_btn._props["title"] = _theme_tooltip()
        theme_btn.update()

    def _apply_theme(notify: bool = False):
        want_dark = _resolve_dark(theme_state["mode"])
        if dark_mode.value != want_dark:
            dark_mode.set_value(want_dark)
        _sync_theme_button()
        if notify:
            label = theme_state["mode"].capitalize()
            if theme_state["mode"] == "auto":
                label += f" ({'dark' if want_dark else 'light'} right now)"
            ui.notify(f"Theme: {label}", type="info", timeout=1500)

    def toggle_theme():
        idx = THEME_MODES.index(theme_state["mode"])
        theme_state["mode"] = THEME_MODES[(idx + 1) % len(THEME_MODES)]
        _apply_theme(notify=True)
        try:
            prefs.theme_mode = theme_state["mode"]
            # Kept for any older code still reading prefs.dark_mode.
            prefs.dark_mode = dark_mode.value
            if hasattr(prefs, "save"):
                prefs.save()
        except Exception:
            pass

    def _theme_autotick():
        """In auto mode, flip the theme when the clock crosses dawn/dusk."""
        if theme_state["mode"] != "auto":
            return
        want_dark = _resolve_dark("auto")
        if dark_mode.value != want_dark:
            dark_mode.set_value(want_dark)
            _sync_theme_button()

    _apply_theme()
    ui.timer(60.0, _theme_autotick)

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
        return cleaned.strip()

    current_agent_md: Optional[ui.markdown] = None
    agent_text_buffer = ""
    active_tool_panels: Dict[str, Dict[str, Any]] = {}
    active_artefact_panels: Dict[str, Dict[str, Any]] = {}

    def seal_current_text_block():
        """Seals the current conversational agent bubble so that the next action
        or response text is inserted chronologically below the preceding event."""
        nonlocal current_agent_md, agent_text_buffer
        if current_agent_md is not None:
            clean_text = _strip_processing_tags(agent_text_buffer)
            entry = getattr(current_agent_md, "_debug_entry", None)
            if clean_text:
                current_agent_md.set_content(clean_text)
                if entry is not None:
                    entry["text"] = clean_text
            else:
                try:
                    current_agent_md.delete()
                except Exception:
                    pass
            current_agent_md = None
            agent_text_buffer = ""

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

                if current_agent_md is None:
                    current_agent_md = add_agent_message_container()
                    agent_text_buffer = ""

                agent_text_buffer += chunk_text
                cleaned = _strip_processing_tags(agent_text_buffer)
                current_agent_md.set_content(cleaned)
                entry = getattr(current_agent_md, "_debug_entry", None)
                if entry is not None:
                    entry["text"] = cleaned

            elif ev.kind == "thought":
                seal_current_text_block()
                add_event_panel("💭 Thinking", "", ev.data.get("text", ""), "gray-400", "psychology")

            elif ev.kind == "info":
                status_label.set_text(ev.data.get("text", "")[:80])

            elif ev.kind == "tool_start":
                name = ev.data.get("tool_name", "tool")
                if name == "pending":
                    continue
                seal_current_text_block()
                params = ev.data.get("parameters", {})
                params_str = json.dumps(params, indent=2, ensure_ascii=False) if isinstance(params, dict) else str(params)

                panel = add_event_panel(f"🛠️ Running: {name}", "executing…", params_str, "blue-500", "build")
                active_tool_panels[name] = {"panel": panel, "params": params}
                status_label.set_text(f"Running {name}…")
                _paint_round(timeline_slots, session.current_round, "bg-blue-500 animate-pulse")

            elif ev.kind == "tool_end":
                name = ev.data.get("tool_name", "tool")
                if name == "pending" and not ev.data.get("output") and not ev.data.get("error"):
                    continue
                seal_current_text_block()
                success = ev.data.get("success", False)
                output = ev.data.get("output") or ev.data.get("error") or ""
                color = "green-500" if success else "red-500"

                # Update existing running panel if present to prevent duplicate disjoint panels
                if name in active_tool_panels:
                    active_item = active_tool_panels.pop(name)
                    try:
                        active_item["panel"].delete()
                    except Exception:
                        pass

                add_event_panel(
                    f"{'✅' if success else '❌'} Finished: {name}",
                    "success" if success else "failed",
                    output,
                    color,
                    "build_circle"
                )
                _paint_round(
                    timeline_slots, session.current_round,
                    "bg-green-500" if success else "bg-red-500"
                )

            elif ev.kind == "context_update":
                seal_current_text_block()
                _paint_round(timeline_slots, session.current_round, "bg-amber-500")
                add_event_panel(
                    "📂 Context update",
                    ev.data.get("action", ""),
                    "\n".join(ev.data.get("files", [])) or "(no files)",
                    "amber-500", "folder_open",
                )

            elif ev.kind == "scratchpad_update":
                seal_current_text_block()
                action = ev.data.get("action", "update")
                message = ev.data.get("message", "Scratchpad updated.")
                ui.notify(f"📝 {message}", type="info")
                add_event_panel("📝 Scratchpad", action, message, "yellow-600", "edit_note")

            elif ev.kind == "artefact_start":
                title = ev.data.get("title", "artifact")
                lang = ev.data.get("language", "")
                op = ev.data.get("operation", "write")
                sec = ev.data.get("current_section") or ""
                subtitle = f"{op} · {lang}" if lang else op
                if sec:
                    subtitle += f" · {sec}"
                seal_current_text_block()
                panel = add_event_panel(f"📝 Writing: {title}", subtitle, "", "purple-500", "description")
                active_artefact_panels[title] = {"panel": panel}
                _paint_round(timeline_slots, session.current_round, "bg-purple-500 animate-pulse")

            elif ev.kind == "artefact_symbol":
                sym = ev.data.get("symbol", {})
                detail = sym.get("detail") or ev.data.get("detail", "")
                title = ev.data.get("title", "artifact")
                status_label.set_text(f"Writing {title}: {detail}")

            elif ev.kind == "artefact_end":
                title = ev.data.get("title", "artifact")
                success = ev.data.get("success", False)
                version = ev.data.get("version", 1)
                lines = ev.data.get("line_count", 0)
                chars = ev.data.get("size_chars", 0)
                is_patch = ev.data.get("is_patch", False)

                seal_current_text_block()

                if title in active_artefact_panels:
                    active_art_item = active_artefact_panels.pop(title)
                    try:
                        active_art_item["panel"].delete()
                    except Exception:
                        pass

                meta_details = []
                if version: meta_details.append(f"v{version}")
                if lines: meta_details.append(f"{lines} lines")
                if chars: meta_details.append(f"{chars:,} chars")
                subtitle = " · ".join(meta_details) if success else str(ev.data.get("error", "failed"))

                body_lines = []
                sections = ev.data.get("sections", [])
                if sections:
                    body_lines.append("Sections/Symbols:")
                    for s in sections[:10]:
                        body_lines.append(f"  • {s.get('type', 'item')}: {s.get('name', '')} (line {s.get('line', '?')})")
                    if len(sections) > 10:
                        body_lines.append(f"  ... (+{len(sections) - 10} more)")

                patch_stats = ev.data.get("patch_stats")
                if patch_stats:
                    body_lines.append(f"\nPatch Hunks: {patch_stats.get('hunks_count', 1)}")

                add_event_panel(
                    f"{'✅' if success else '❌'} {'Patched' if is_patch else 'Saved'}: {title}",
                    subtitle,
                    "\n".join(body_lines),
                    "green-500" if success else "red-500",
                    "task_alt",
                )

            elif ev.kind == "round_start":
                seal_current_text_block()
                r = ev.data.get("round_id", 1)
                m = ev.data.get("max_rounds", prefs.max_reasoning_steps)
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

            elif ev.kind == "round_info":
                r = ev.data.get("round", "?")
                m = ev.data.get("max_rounds", "?")
                status_label.set_text(f"Round {r}/{m}")

            elif ev.kind == "done":
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

            elif ev.kind == "error":
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
            scroll_area.scroll_to(percent=1.0)

    ui.timer(0.15, drain_queue)

    def _tick_elapsed():
        if session.busy and session.turn_start_ts is not None:
            elapsed_label.set_text(f"{time.time() - session.turn_start_ts:.1f}s")

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
        if not session.prompt_history or session.prompt_history[-1] != text:
            session.prompt_history.append(text)
        session.history_index = -1

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

        if cmd == "/export":
            export_history()
            return True

        if cmd == "/help":
            add_system_notice(HELP_TEXT)
            return True

        if cmd == "/config":
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
            if session.personality is not None:
                session.personality._conversation = []
            add_system_notice("Conversation cleared.")
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

                    def pick_folder():
                        try:
                            import webview
                            result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                            if result:
                                path_input.value = result[0]
                        except Exception:
                            ui.notify("Native folder picker unavailable — type the path manually.", type="warning")

                    ui.button("Browse…", icon="folder_open", on_click=pick_folder).props("flat")
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
        """Hands the file to the OS default application (double-click action).
        Local-desktop only: this runs on the machine hosting the server."""
        import os
        import subprocess
        import sys
        try:
            p = Path(abs_path)
            if not p.exists():
                ui.notify(f"No longer on disk: {p.name}", type="warning")
                return
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
            ui.notify(f"Opening {p.name}…", type="info")
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
        prompt_input.value = f"{current_val.rstrip()} {rel} " if current_val else f"/load {rel}"
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

                    ui.menu_item("📥 Load all files in folder", _load_folder)
                    ui.menu_item("📤 Unload all files in folder", _unload_folder)
                    ui.separator()
                    ui.menu_item("🙈 Hide from tree", lambda r=rel: _visibility_action([r], "hide"))
                    ui.menu_item("👁️ Unhide", lambda r=rel: _visibility_action([r], "unhide"))
                    ui.separator()
                    ui.menu_item("📂 Open folder", lambda p=abs_path: open_in_default_editor(p))
                    ui.menu_item("🗂️ Switch workspace here",
                                 lambda p=abs_path: _switch_workspace_from_tree(p))
                else:
                    if is_loaded:
                        ui.menu_item("📤 Remove from context",
                                     lambda r=rel: _visibility_action([r], "unload"))
                    else:
                        ui.menu_item("📥 Add to context",
                                     lambda r=rel: _visibility_action([r], "load"))
                    ui.menu_item("🔒 Add & lock (agent can't unload)",
                                 lambda r=rel: _visibility_action([r], "lock"))
                    ui.separator()
                    ui.menu_item("✏️ Open in default editor",
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

        # Single click: folders expand/collapse, files get mentioned in the prompt.
        if is_dir:
            row.on("click", lambda r=rel: _toggle_dir(r))
        else:
            row.on("click", lambda r=rel: _insert_into_prompt(r))
            # Double click: hand off to the OS default application.
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
            row.on("click", lambda r=node["rel"]: _insert_into_prompt(r))
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
        cache so on-disk changes show up, and repaints. The root listing is
        fetched in the background so this never blocks the caller."""
        nonlocal tree_loaded_set
        tree_loaded_set = _get_loaded_files_set()
        tree_children.clear()
        tree_expanded.clear()

        async def _load_root():
            tree_loading.add("")
            _paint_tree()
            try:
                await _children_of_async("")
            finally:
                tree_loading.discard("")
            _paint_tree()

        ui.timer(0.01, _load_root, once=True)

    def apply_tree_filter():
        _paint_tree()

    tree_search_input.on_value_change(lambda e: apply_tree_filter())

    # Initial tree population
    refresh_workspace_tree()
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