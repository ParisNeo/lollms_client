"""app.chat_page — main agent session UI. Replaces run_interactive()/run_single_prompt()."""
from __future__ import annotations

import queue
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from nicegui import ui

from gui_prefs import GuiPrefs
from env_config import EnvStore
import agent_bridge


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
    ("/forget", "Wipe persistent memory"),
    ("/workspace", "Switch workspace directory"),
    ("/config", "Open Settings"),
    ("/models", "Model switching info"),
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

    with ui.column().classes("w-full h-full flex-nowrap gap-0"):
        # ---- Slim status strip (replaces the old sidebar cards) ----
        with ui.row().classes(
            "w-full items-center justify-between px-3 py-1 shrink-0 "
            "bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800"
        ):
            status_label = ui.label("Idle").classes("text-xs text-gray-500")
            with ui.row().classes("items-center gap-3"):
                rounds_label = ui.label("").classes("text-xs text-gray-500")
                ctx_label = ui.label("").classes("text-xs text-gray-500")
                if tools_toggle is None:
                    tools_toggle = ui.switch("Tool panels", value=prefs.show_tool_calls).props("dense")
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

        # ---- Live telemetry sidebar (toggleable via Settings → Appearance) ----
        with ui.row().classes("w-full flex-1 min-h-0"):
            # ---- Transcript ----
            scroll_area = ui.scroll_area().classes("flex-1 min-w-0")
            with scroll_area:
                transcript = ui.column().classes("w-full gap-2 p-3")

            # ---- Right-hand live panels ----
            live_sidebar = ui.column().classes(
                "w-64 shrink-0 border-l border-gray-200 dark:border-gray-800 "
                "bg-gray-50 dark:bg-gray-900 overflow-y-auto p-2 gap-2"
            ).bind_visibility_from(prefs, "show_live_sidebar")
            with live_sidebar:
                with ui.card().classes("w-full no-shadow border"):
                    ui.label("⏱️ Round Timeline").classes("text-xs font-bold text-gray-600 dark:text-gray-300")
                    timeline_container = ui.column().classes("w-full gap-0.5 mt-1")
                timeline_slots: Dict[int, Any] = {}

                with ui.card().classes("w-full no-shadow border"):
                    ui.label("📊 Context Health").classes("text-xs font-bold text-gray-600 dark:text-gray-300")
                    health_label = ui.label("no data yet").classes("text-xs text-gray-500")
                    health_bar = ui.linear_progress(value=0.0, show_value=False).props("instant-feedback")

                with ui.card().classes("w-full no-shadow border"):
                    ui.label("🎓 Skills (live)").classes("text-xs font-bold text-gray-600 dark:text-gray-300")
                    skills_label = ui.label("—").classes("text-xs text-gray-500")
                    ui.separator().classes("my-1")
                    peek_button = ui.button(
                        "🔍 Peek Scratchpad", icon="visibility",
                        on_click=lambda: open_scratchpad_dialog(),
                    ).props("flat dense size=xs no-caps").classes("w-full")
                    peek_button.tooltip("Read the agent's scratchpad right now — safe while generating")
                    scratchpad_badge = ui.badge("0", color="blue").props("floating").bind_visibility_from(
                        session, "current_round", backward=lambda r: session.busy and r > 0
                    )

        # ---- Slash-command suggestions (shown above the input, hidden by default) ----
        suggestions_row = ui.row().classes("w-full gap-1 px-3 flex-wrap")
        suggestions_row.visible = False

        # ---- Input ----
        with ui.row().classes("w-full items-end gap-2 px-3 py-2 shrink-0"):
            prompt_input = ui.textarea(placeholder="Describe the task, or type / for commands…").classes(
                "flex-1"
            ).props("outlined autogrow dense rows=1 input-debounce=0")
            send_button = ui.button(icon="send").props("round color=primary")

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
        entry = {"type": "agent", "text": ""}
        debug_log.append(entry)
        with transcript:
            with ui.row().classes("w-full justify-start items-start gap-1"):
                md = ui.markdown("").classes(
                    "bg-gray-100 dark:bg-gray-800 rounded-lg px-3 py-1.5 max-w-[85%] whitespace-pre-wrap"
                )
                copy_btn = ui.button(icon="content_copy", on_click=lambda: copy_agent_message(entry)).props(
                    "flat round dense size=xs"
                )
                copy_btn.tooltip("Copy this message")
        md._debug_entry = entry
        last_agent_state["entry"] = entry
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
                    ("bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300" if is_error
                     else "bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300")
                    + " rounded-lg px-3 py-1.5 max-w-[90%] text-sm"
                )
                copy_btn = ui.button(
                    icon="content_copy",
                    on_click=lambda entry_ref=entry: copy_system_notice(entry_ref),
                ).props("flat round dense size=xs")
                copy_btn.tooltip("Copy this notice" + (" (error)" if is_error else ""))
        scroll_area.scroll_to(percent=1.0)

    def add_event_panel(title: str, subtitle: str, body: str, color: str, icon: str):
        entry = {"type": "event", "title": title, "subtitle": subtitle, "body": body}
        debug_log.append(entry)
        with transcript:
            panel = ui.expansion(title, icon=icon).classes(f"w-full border-l-4 border-{color}")
            panel.bind_visibility_from(tools_toggle, "value")
            with panel:
                with ui.row().classes("w-full items-center justify-between"):
                    if subtitle:
                        ui.label(subtitle).classes("text-xs text-gray-500")
                    copy_btn = ui.button(
                        icon="content_copy",
                        on_click=lambda e_ref=entry: copy_event_body(e_ref),
                    ).props("flat round dense size=xs")
                    copy_btn.tooltip("Copy panel content")
                ui.code(body or "(no output)").classes("w-full text-xs")

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
                lines += ["**Agent:**", "", (entry["text"] or "_(empty)_"), ""]
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

    async def send_prompt_with_text(prompt_text: str, rerun_marker: bool = False):
        """Shared send pipeline used by both the input box and rerun/edit actions."""
        if not prompt_text.strip() or session.busy:
            return
        display_text = f"🔁 _Rerun:_ {prompt_text}" if rerun_marker else prompt_text
        add_user_bubble(display_text)
        if prompt_text.strip().startswith("/"):
            await handle_slash_command(prompt_text.strip())
            return
        session.busy = True
        send_button.props("loading")
        status_label.set_text("Thinking…")
        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
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

    def _strip_processing_tags(text: str) -> str:
        return re.sub(r"<processing.*?</processing>", "", text, flags=re.DOTALL)

    current_agent_md: Optional[ui.markdown] = None
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
                if current_agent_md is None:
                    current_agent_md = add_agent_message_container()
                    agent_text_buffer = ""
                agent_text_buffer += ev.data.get("text", "")
                current_agent_md.set_content(_strip_processing_tags(agent_text_buffer))

            elif ev.kind == "thought":
                add_event_panel("💭 Thinking", "", ev.data.get("text", ""), "gray-400", "psychology")

            elif ev.kind == "info":
                status_label.set_text(ev.data.get("text", "")[:80])

            elif ev.kind == "tool_start":
                name = ev.data.get("tool_name", "tool")
                params = ev.data.get("parameters", {})
                add_event_panel(f"🛠️ Running: {name}", "executing…", str(params), "blue-500", "build")
                status_label.set_text(f"Running {name}…")
                _paint_round(timeline_slots, session.current_round, "bg-blue-500 animate-pulse")

            elif ev.kind == "tool_end":
                name = ev.data.get("tool_name", "tool")
                success = ev.data.get("success", False)
                output = ev.data.get("output") or ev.data.get("error") or ""
                color = "green-500" if success else "red-500"
                add_event_panel(
                    f"{'✅' if success else '❌'} Finished: {name}", "", output, color, "build_circle"
                )
                _paint_round(
                    timeline_slots, session.current_round,
                    "bg-green-500" if success else "bg-red-500"
                )

            elif ev.kind == "context_update":
                _paint_round(timeline_slots, session.current_round, "bg-amber-500")
                add_event_panel(
                    "📂 Context update",
                    ev.data.get("action", ""),
                    "\n".join(ev.data.get("files", [])) or "(no files)",
                    "amber-500", "folder_open",
                )

            elif ev.kind == "scratchpad_update":
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
                add_event_panel(f"📝 Writing: {title}", subtitle, "", "purple-500", "description")
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

                meta_details = []
                if version: meta_details.append(f"v{version}")
                if lines: meta_details.append(f"{lines} lines")
                if chars: meta_details.append(f"{chars:,} chars")
                subtitle = " · ".join(meta_details) if success else str(ev.data.get("error", "failed"))

                # Build summary of symbols/sections without printing full content
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

            elif ev.kind == "round_info":
                r = ev.data.get("round", "?")
                m = ev.data.get("max_rounds", "?")
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

            elif ev.kind == "done":
                result = ev.data.get("result", {}) or {}
                current_agent_md = None
                session.busy = False
                send_button.props(remove="loading")
                status_label.set_text("Idle")
                rounds_label.set_text(f"Rounds: {result.get('rounds', 0)} · Tools: {len(result.get('tool_calls', []))}")
                ctx = result.get("context_health") or {}
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
                if skills_created or skills_updated and prefs.show_skills_activity:
                    body = "\n".join([f"created: {s}" for s in skills_created] +
                                      [f"updated: {s}" for s in skills_updated])
                    add_event_panel("🎓 Skills activity", "", body, "yellow-600", "school")

            elif ev.kind == "error":
                current_agent_md = None
                session.busy = False
                send_button.props(remove="loading")
                status_label.set_text("Error")
                notify_error(str(ev.data.get("message", "Unknown agent error.")))

        if drained_any:
            scroll_area.scroll_to(percent=1.0)

    ui.timer(0.15, drain_queue)

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

    # ---------------- Slash-command autocomplete ----------------

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

    prompt_input.on_value_change(lambda e: refresh_suggestions())
    prompt_input.on("keyup", lambda e: refresh_suggestions())
    prompt_input.on("keydown.tab.prevent", lambda e: accept_first_suggestion())

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

        if text.startswith("/"):
            add_user_bubble(text)
            await handle_slash_command(text)
            return

        add_user_bubble(text)
        session.busy = True
        send_button.props("loading")
        status_label.set_text("Thinking…")

        try:
            session.ensure_ready()
        except Exception as e:
            notify_error(f"Could not start agent: {e}")
            session.busy = False
            send_button.props(remove="loading")
            return

        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, text, prefs, session.event_queue, use_history=True
        )

    send_button.on("click", send_prompt)
    prompt_input.on("keydown.enter.prevent", send_prompt)

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