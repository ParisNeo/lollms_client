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

    def ensure_ready(self):
        if self.client is None:
            self.client = agent_bridge.create_client(self.env, self.prefs)
        if self.personality is None:
            self.personality = agent_bridge.create_personality(self.prefs, self.client)


def build_chat_page(env: EnvStore, prefs: GuiPrefs, tools_toggle=None) -> None:
    session = ChatSession(env, prefs)
    debug_log: List[Dict[str, Any]] = []

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
                ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
                    "flat dense size=sm no-caps"
                )

        # ---- Transcript ----
        scroll_area = ui.scroll_area().classes("w-full flex-1")
        with scroll_area:
            transcript = ui.column().classes("w-full gap-2 p-3")

        # ---- Slash-command suggestions (shown above the input, hidden by default) ----
        suggestions_row = ui.row().classes("w-full gap-1 px-3 flex-wrap")
        suggestions_row.visible = False

        # ---- Input ----
        with ui.row().classes("w-full items-end gap-2 px-3 py-2 shrink-0"):
            prompt_input = ui.textarea(placeholder="Describe the task, or type / for commands…").classes(
                "flex-1"
            ).props("outlined autogrow dense rows=1 input-debounce=0")
            send_button = ui.button(icon="send").props("round color=primary")

    def add_user_bubble(text: str):
        debug_log.append({"type": "user", "text": text})
        with transcript:
            with ui.row().classes("w-full justify-end"):
                ui.markdown(text).classes(
                    "bg-primary text-white rounded-lg px-3 py-1.5 max-w-[75%]"
                )
        scroll_area.scroll_to(percent=1.0)

    def add_agent_message_container() -> ui.markdown:
        entry = {"type": "agent", "text": ""}
        debug_log.append(entry)
        with transcript:
            with ui.row().classes("w-full justify-start"):
                md = ui.markdown("").classes(
                    "bg-gray-100 dark:bg-gray-800 rounded-lg px-3 py-1.5 max-w-[85%] whitespace-pre-wrap"
                )
        md._debug_entry = entry  # tag so drain_queue can update the same log entry as it streams
        return md

    def add_system_notice(text: str, is_error: bool = False):
        debug_log.append({"type": "system", "text": text, "error": is_error})
        with transcript:
            with ui.row().classes("w-full justify-start"):
                ui.markdown(text).classes(
                    ("bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-300" if is_error
                     else "bg-amber-50 dark:bg-amber-950 text-amber-700 dark:text-amber-300")
                    + " rounded-lg px-3 py-1.5 max-w-[90%] text-sm"
                )
        scroll_area.scroll_to(percent=1.0)

    def add_event_panel(title: str, subtitle: str, body: str, color: str, icon: str):
        debug_log.append({"type": "event", "title": title, "subtitle": subtitle, "body": body})
        # Always create the panel; visibility is bound live to the toggle so
        # flipping it retroactively shows/hides everything already logged,
        # not just future events.
        with transcript:
            panel = ui.expansion(title, icon=icon).classes(f"w-full border-l-4 border-{color}")
            panel.bind_visibility_from(tools_toggle, "value")
            with panel:
                if subtitle:
                    ui.label(subtitle).classes("text-xs text-gray-500")
                ui.code(body or "(no output)").classes("w-full text-xs")

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

            elif ev.kind == "tool_end":
                name = ev.data.get("tool_name", "tool")
                success = ev.data.get("success", False)
                output = ev.data.get("output") or ev.data.get("error") or ""
                color = "green-500" if success else "red-500"
                add_event_panel(
                    f"{'✅' if success else '❌'} Finished: {name}", "", output, color, "build_circle"
                )

            elif ev.kind == "artefact_start":
                title = ev.data.get("title", "artifact")
                lang = ev.data.get("language", "")
                op = ev.data.get("operation", "write")
                sec = ev.data.get("current_section") or ""
                subtitle = f"{op} · {lang}" if lang else op
                if sec:
                    subtitle += f" · {sec}"
                add_event_panel(f"📝 Writing: {title}", subtitle, "", "purple-500", "description")

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
                ui.notify(f"Agent error: {ev.data.get('message')}", type="negative", timeout=8000)

        if drained_any:
            scroll_area.scroll_to(percent=1.0)

    ui.timer(0.15, drain_queue)

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
        cmd, _, arg = text.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd in ("/exit", "/quit"):
            add_system_notice("Nothing to exit to in the GUI — just close the window.")
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
            ui.notify(f"Could not start agent: {e}", type="negative", timeout=8000)
            session.busy = False
            send_button.props(remove="loading")
            return

        agent_bridge.run_agent_turn_in_thread(
            session.personality, session.client, text, prefs, session.event_queue, use_history=True
        )

    send_button.on("click", send_prompt)
    prompt_input.on("keydown.enter.prevent", send_prompt)