"""app.chat_page — main agent session UI. Replaces run_interactive()/run_single_prompt()."""
from __future__ import annotations

import queue
import re
from typing import Optional

from nicegui import ui

from gui_prefs import GuiPrefs
from env_config import EnvStore
import agent_bridge


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


def build_chat_page(env: EnvStore, prefs: GuiPrefs) -> None:
    session = ChatSession(env, prefs)
    resolved = env.resolve_default_connection("llm")

    with ui.row().classes("w-full h-full no-wrap"):
        # ---- Sidebar: workspace + session info ----
        with ui.column().classes("w-64 shrink-0 gap-2"):
            with ui.card().classes("w-full"):
                ui.label("Workspace").classes("text-xs text-gray-500 uppercase")
                ui.label(prefs.workspace_path).classes("text-sm break-all")
                ui.label(f"Model: {resolved.get('model_name') or '(none)'}").classes("text-xs text-gray-500 mt-2")
                ui.label(f"Binding: {resolved.get('binding_name') or '(none)'}").classes("text-xs text-gray-500")

            status_card = ui.card().classes("w-full")
            with status_card:
                ui.label("Session").classes("text-xs text-gray-500 uppercase")
                status_label = ui.label("Idle").classes("text-sm")
                rounds_label = ui.label("").classes("text-xs text-gray-500")
                ctx_label = ui.label("").classes("text-xs text-gray-500")

            tools_toggle = ui.switch("Show tool panels", value=prefs.show_tool_calls)

        # ---- Main column: transcript + input ----
        with ui.column().classes("flex-1 h-full"):
            transcript = ui.column().classes("w-full flex-1 overflow-y-auto gap-2 p-2")

            with ui.row().classes("w-full items-end gap-2"):
                prompt_input = ui.textarea(placeholder="Describe the task…").classes("flex-1").props(
                    "outlined autogrow"
                )
                send_button = ui.button(icon="send").props("round color=primary")

    def add_user_bubble(text: str):
        with transcript:
            with ui.row().classes("w-full justify-end"):
                ui.markdown(text).classes(
                    "bg-primary text-white rounded-lg px-4 py-2 max-w-[75%]"
                )

    def add_agent_message_container() -> ui.markdown:
        with transcript:
            with ui.row().classes("w-full justify-start"):
                md = ui.markdown("").classes(
                    "bg-gray-100 dark:bg-gray-800 rounded-lg px-4 py-2 max-w-[85%] whitespace-pre-wrap"
                )
        return md

    def add_event_panel(title: str, subtitle: str, body: str, color: str, icon: str):
        if not tools_toggle.value:
            return
        with transcript:
            with ui.expansion(title, icon=icon).classes(f"w-full border-l-4 border-{color}"):
                if subtitle:
                    ui.label(subtitle).classes("text-xs text-gray-500")
                ui.code(body or "(no output)").classes("w-full text-xs")

    def _strip_processing_tags(text: str) -> str:
        # The raw stream may still contain <processing ...>...</processing>
        # blocks meant for the terminal renderer; hide them from the chat bubble.
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
                add_event_panel(f"📝 Writing: {title}", ev.data.get("language", ""), "", "purple-500", "description")

            elif ev.kind == "artefact_end":
                title = ev.data.get("title", "artifact")
                success = ev.data.get("success", False)
                version = ev.data.get("version", 1)
                add_event_panel(
                    f"{'✅' if success else '❌'} Saved: {title}",
                    f"v{version}" if success else str(ev.data.get("error", "")),
                    "", "green-500" if success else "red-500", "task_alt",
                )

            elif ev.kind == "context_update":
                files = ev.data.get("files", [])
                if files:
                    add_event_panel("📂 Context updated", "", "\n".join(files), "amber-500", "folder_open")

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
            transcript.scroll_to(percent=1.0)

    ui.timer(0.15, drain_queue)

    def send_prompt():
        text = prompt_input.value.strip()
        if not text or session.busy:
            return
        prompt_input.value = ""
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