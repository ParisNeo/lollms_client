"""
settings_page.py — Revamped modern settings interface for lollms_code.
Features a master-detail sidebar layout, high-contrast cards, two-tier
binding/profile managers, and complete preference curation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from nicegui import ui, run

from env_config import EnvStore, MODALITIES, MODALITY_LABELS
from gui_prefs import GuiPrefs, SHELL_AUTONOMY_LEVELS, SKILLS_MODES, ACCENT_PRESETS

try:
    from folder_picker import pick_folder
except ImportError:
    try:
        from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder
    except ImportError:
        pick_folder = None

MODALITY_ICONS = {
    "llm": "psychology",
    "tti": "palette",
    "tts": "record_voice_over",
    "stt": "hearing",
    "ttm": "music_note",
    "ttv": "videocam",
    "connection": "hub",
}


def build_settings_page(
    env: EnvStore,
    prefs: GuiPrefs,
    on_saved: Optional[Callable[[EnvStore, GuiPrefs], None]] = None,
    on_back: Optional[Callable[[], None]] = None,
) -> None:
    # ---- Theme Tokens (Matching Main Chat UI) ----
    SURFACE = "bg-slate-100/90 dark:bg-slate-900/90"
    SURFACE_ALT = "bg-slate-200/50 dark:bg-slate-800/50"
    CANVAS = "bg-slate-50 dark:bg-slate-950"
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"
    TEXT_MAIN = "text-slate-900 dark:text-slate-100"
    TEXT_MUTED = "text-slate-600 dark:text-slate-400"

    HEADER_H = 42

    # Active Navigation State
    nav_state = {"section": "llm"}

    # ---- Top Navigation Header (Identical to Main Chat Bar) ----
    resolved = env.resolve_default_connection("llm")
    with ui.row().classes(
        f"w-full items-center justify-between px-3 flex-nowrap bg-primary text-white shrink-0 shadow-sm"
    ).style(f"min-height: {HEADER_H}px; height: {HEADER_H}px;"):
        with ui.row().classes("items-center gap-2 flex-nowrap overflow-hidden"):
            if on_back:
                ui.button(icon="arrow_back", on_click=on_back).props("flat round dense size=sm").tooltip("Back to Chat")
            ui.icon("settings", size="18px")
            ui.label("Settings & Configuration").classes("text-sm font-bold shrink-0")
            ui.label("·").classes("text-xs opacity-40 shrink-0")
            ui.label(prefs.workspace_path).classes("text-xs opacity-80 truncate max-w-xs")
            ui.label("·").classes("text-xs opacity-40 shrink-0")
            model_info = f"{resolved.get('binding_name') or '?'} / {resolved.get('model_name') or '?'}"
            ui.label(model_info).classes("text-xs opacity-80 truncate max-w-xs")

        with ui.row().classes("items-center gap-2 shrink-0"):
            def toggle_theme():
                prefs.dark_mode = not prefs.dark_mode
                ui.dark_mode(prefs.dark_mode)
                prefs.save()

            def _toggle_fs():
                try:
                    import webview
                    if webview.windows and len(webview.windows) > 0:
                        webview.windows[0].toggle_fullscreen()
                        return
                except Exception:
                    pass
                ui.run_javascript(
                    "if (!document.fullscreenElement) { document.documentElement.requestFullscreen(); } "
                    "else { if (document.exitFullscreen) { document.exitFullscreen(); } }"
                )

            ui.button(
                icon="fullscreen",
                on_click=_toggle_fs,
            ).props("flat round dense size=sm").tooltip("Toggle Fullscreen (F11)")

            ui.button(
                icon="dark_mode" if not prefs.dark_mode else "light_mode",
                on_click=toggle_theme,
            ).props("flat round dense size=sm").tooltip("Toggle Theme")

            if on_back:
                ui.button("Back to Chat", icon="chat", on_click=on_back).props("flat dense size=sm no-caps")

            ui.button(
                icon="close",
                on_click=lambda: _exit_from_settings(),
            ).props("flat round dense size=sm color=red text-color=white").tooltip("Exit Application")

    def _exit_from_settings():
        from main import confirm_exit_dialog
        confirm_exit_dialog()

    # ---- Main Two-Pane Body ----
    with ui.row().classes(f"w-full flex-1 min-h-0 items-stretch overflow-hidden flex-nowrap gap-0 {CANVAS}"):

        # ── Left Navigation Sidebar ──
        sidebar = ui.column().classes(
            f"w-72 h-full shrink-0 border-r {BORDER} {SURFACE} p-3 gap-1 overflow-y-auto flex flex-col"
        )

        # ── Right Content Area ──
        content_scroll = ui.scroll_area().classes("flex-1 h-full min-w-0 p-6")
        with content_scroll:
            content_container = ui.column().classes("w-full max-w-4xl mx-auto gap-5 pb-20")

        # ---- Section Switcher Function ----
        def render_navigation():
            sidebar.clear()
            with sidebar:
                ui.label("MODELS & ENGINES").classes(f"text-[10px] font-bold tracking-wider px-2 pt-1 text-slate-500")

                for m in MODALITIES:
                    bindings_cnt = len(env.configured_binding_aliases(m))
                    profiles_cnt = len(env.configured_profile_aliases(m))
                    badge_str = f"{profiles_cnt}P" if profiles_cnt else f"{bindings_cnt}B" if bindings_cnt else ""

                    is_active = (nav_state["section"] == m)
                    bg_class = "bg-primary/15 text-primary font-bold shadow-sm" if is_active else f"hover:bg-slate-200/60 dark:hover:bg-slate-800/60 {TEXT_MAIN}"

                    with ui.row().classes(
                        f"w-full items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all {bg_class}"
                    ).on("click", lambda sec=m: switch_section(sec)):
                        with ui.row().classes("items-center gap-2.5"):
                            ui.icon(MODALITY_ICONS.get(m, "cable"), size="18px").classes(
                                "text-primary" if is_active else "text-slate-500 dark:text-slate-400"
                            )
                            ui.label(MODALITY_LABELS[m].split(" ")[0]).classes("text-xs")
                        if badge_str:
                            ui.badge(badge_str, color="primary" if is_active else "grey").props("dense rounded")

                ui.separator().classes(f"my-2 {BORDER}")
                ui.label("AGENT & SYSTEM").classes(f"text-[10px] font-bold tracking-wider px-2 text-slate-500")

                agent_sections = [
                    ("agent", "Agent Behavior", "smart_toy"),
                    ("paths", "Workspace & Paths", "folder"),
                    ("appearance", "Appearance & Theme", "palette"),
                ]

                for sec_id, sec_label, sec_icon in agent_sections:
                    is_active = (nav_state["section"] == sec_id)
                    bg_class = "bg-primary/15 text-primary font-bold shadow-sm" if is_active else f"hover:bg-slate-200/60 dark:hover:bg-slate-800/60 {TEXT_MAIN}"

                    with ui.row().classes(
                        f"w-full items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition-all {bg_class}"
                    ).on("click", lambda s=sec_id: switch_section(s)):
                        with ui.row().classes("items-center gap-2.5"):
                            ui.icon(sec_icon, size="18px").classes(
                                "text-primary" if is_active else "text-slate-500 dark:text-slate-400"
                            )
                            ui.label(sec_label).classes("text-xs")

                ui.element("div").classes("flex-1")

                # Test Connection button in sidebar bottom
                ui.button("Test Connection", icon="network_check", on_click=lambda: test_connection()).props(
                    "outline dense size=xs no-caps"
                ).classes("w-full mb-1")

        def switch_section(sec: str):
            nav_state["section"] = sec
            render_navigation()
            render_content()

        def test_connection():
            ui.notify("Testing default connection…", type="info")
            success, msg = env.validate()
            if success:
                ui.notify(f"✅ {msg}", type="positive", timeout=4000)
            else:
                ui.notify(f"❌ Connection test failed: {msg}", type="negative", timeout=6000)

        # ---- Dynamic Content Renderer ----
        def render_content():
            content_container.clear()
            sec = nav_state["section"]

            with content_container:
                if env.import_error:
                    with ui.card().classes(f"w-full p-4 border border-red-300 dark:border-red-900 bg-red-50 dark:bg-red-950/50 rounded-xl gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("warning", color="red-500")
                            ui.label("Package Import Warning").classes("font-bold text-sm text-red-700 dark:text-red-300")
                        ui.label(
                            f"lollms_client import failed ({env.import_error}). Configuration can be saved, but active introspection is disabled."
                        ).classes("text-xs text-red-600 dark:text-red-400")

                if sec in MODALITIES:
                    _render_modality_section(env, sec, render_content)
                elif sec == "agent":
                    _render_agent_section(prefs)
                elif sec == "paths":
                    _render_paths_section(prefs)
                elif sec == "appearance":
                    _render_appearance_section(prefs)

        render_navigation()
        render_content()

    # ---- Sticky Save Footer ----
    with ui.row().classes(
        f"w-full items-center justify-between px-6 py-3 border-t {BORDER} {SURFACE} shadow-lg z-10 shrink-0"
    ):
        with ui.row().classes("items-center gap-2 text-xs text-slate-500 font-mono"):
            ui.icon("info", size="16px")
            ui.label("Settings apply across CLI and GUI sessions.")

        with ui.row().classes("items-center gap-3"):
            if on_back:
                ui.button("Cancel", on_click=on_back).props("flat dense size=sm no-caps")

            def save_all():
                try:
                    prefs.save()
                    env.save()
                    ui.notify("All settings and preferences saved successfully.", type="positive")
                    if on_saved:
                        on_saved(env, prefs)
                except Exception as ex:
                    ui.notify(f"Failed to save settings: {ex}", type="negative")

            ui.button("Save Preferences", icon="save", on_click=save_all).props(
                "unelevated color=primary size=sm no-caps"
            ).classes("px-4")


# ==============================================================================
# Modality View (Bindings + Profiles with Two-Tier Cards)
# ==============================================================================

def _render_modality_section(env: EnvStore, modality: str, refresh_parent: Callable[[], None]) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    title_label = MODALITY_LABELS[modality]
    icon_name = MODALITY_ICONS.get(modality, "cable")

    # Section Header Card
    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.icon(icon_name, size="32px").classes("text-primary")
                with ui.column().classes("gap-0"):
                    ui.label(f"{title_label} Configuration").classes("text-lg font-bold text-slate-900 dark:text-slate-100")
                    ui.label(
                        f"Configure server connections (Bindings) and model targets (Profiles) for {title_label}."
                    ).classes("text-xs text-slate-500")

    # Modality Sub-Tabs (Pill Toggle)
    view_state = {"tab": "profiles"}
    with ui.row().classes("w-full items-center justify-between"):
        sub_toggle = ui.toggle(
            {"profiles": "📋 Model Profiles", "bindings": "🔌 Server Bindings"},
            value=view_state["tab"],
        ).props("dense unelevated size=sm").classes("text-xs")

        add_btn = ui.button(
            "Add Model Profile", icon="add",
            on_click=lambda: _open_add_profile_dialog(env, modality, refresh_parent),
        ).props("outline dense size=sm no-caps color=primary")

    panels_slot = ui.column().classes("w-full gap-3")

    def sync_views():
        panels_slot.clear()
        tab = view_state["tab"]
        if tab == "profiles":
            add_btn.text = "Add Model Profile"
            add_btn._props["icon"] = "add"
            add_btn.on("click", lambda: _open_add_profile_dialog(env, modality, refresh_parent))
            _render_profiles_cards(env, modality, panels_slot, refresh_parent)
        else:
            add_btn.text = "Add Server Binding"
            add_btn._props["icon"] = "add_link"
            add_btn.on("click", lambda: _open_add_binding_dialog(env, modality, refresh_parent))
            _render_bindings_cards(env, modality, panels_slot, refresh_parent)

    def on_toggle_change(e):
        view_state["tab"] = e.value
        sync_views()

    sub_toggle.on_value_change(on_toggle_change)
    sync_views()


def _render_bindings_cards(env: EnvStore, modality: str, container: ui.column, refresh) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    aliases = env.configured_binding_aliases(modality)
    with container:
        if not aliases:
            with ui.card().classes(f"w-full p-8 border {BORDER} {CARD_BG} rounded-xl items-center justify-center gap-2 text-center"):
                ui.icon("cable", size="36px").classes("text-slate-400")
                ui.label("No Server Bindings Configured").classes("font-bold text-sm text-slate-700 dark:text-slate-300")
                ui.label(
                    "Add a binding to connect to an LLM provider (Ollama, OpenAI, vLLM, Llama.cpp server, etc.)."
                ).classes("text-xs text-slate-500 max-w-md")
                ui.button("Add Binding Now", icon="add", on_click=lambda: _open_add_binding_dialog(env, modality, refresh)).props(
                    "unelevated size=sm color=primary no-caps mt-2"
                )
            return

        for alias in aliases:
            keys = env.binding_keys(modality, alias)
            b_name = keys.get("BINDING_NAME", "unknown")
            host = keys.get("HOST_ADDRESS", "default host")

            with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-3"):
                        ui.badge(b_name.upper(), color="indigo").props("rounded dense")
                        with ui.column().classes("gap-0"):
                            ui.label(alias).classes("font-mono font-bold text-sm text-slate-900 dark:text-slate-100")
                            ui.label(f"Endpoint: {host}").classes("text-xs text-slate-500 font-mono")

                    with ui.row().classes("items-center gap-1"):
                        ui.button(
                            icon="edit",
                            on_click=lambda a=alias: _open_edit_binding_dialog(env, modality, a, refresh),
                        ).props("flat round dense size=sm color=primary").tooltip("Edit Binding")

                        def do_delete(a=alias):
                            env.delete_binding(modality, a)
                            ui.notify(f"Binding '{a}' deleted.", type="info")
                            refresh()

                        ui.button(icon="delete", on_click=do_delete).props("flat round dense size=sm color=red").tooltip("Delete Binding")


def _render_profiles_cards(env: EnvStore, modality: str, container: ui.column, refresh) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    aliases = env.configured_profile_aliases(modality)
    with container:
        if not aliases:
            with ui.card().classes(f"w-full p-8 border {BORDER} {CARD_BG} rounded-xl items-center justify-center gap-2 text-center"):
                ui.icon("badge", size="36px").classes("text-slate-400")
                ui.label("No Model Profiles Configured").classes("font-bold text-sm text-slate-700 dark:text-slate-300")
                ui.label(
                    "Create a model profile that links a specific model name to a configured server binding."
                ).classes("text-xs text-slate-500 max-w-md")
                ui.button("Add Profile Now", icon="add", on_click=lambda: _open_add_profile_dialog(env, modality, refresh)).props(
                    "unelevated size=sm color=primary no-caps mt-2"
                )
            return

        for alias in aliases:
            keys = env.profile_keys(modality, alias)
            b_alias = keys.get("BINDING_ALIAS", "?")
            model = keys.get("MODEL_NAME", "default")
            is_default = keys.get("IS_DEFAULT", "").lower() == "true"
            vision_enabled = keys.get("VISION_ENABLED", "").lower() == "true"
            forced_ctx = keys.get("FORCED_CONTEXT_SIZE")

            with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-3 flex-wrap"):
                        ui.label(alias).classes("font-mono font-bold text-sm text-slate-900 dark:text-slate-100")
                        if is_default:
                            ui.badge("⭐ Default", color="emerald").props("rounded dense")
                        if vision_enabled:
                            ui.badge("👁️ Vision", color="purple").props("rounded dense")
                        if forced_ctx:
                            ui.badge(f"{forced_ctx} ctx", color="slate").props("rounded dense")

                    with ui.row().classes("items-center gap-1"):
                        if not is_default:
                            def make_def(a=alias):
                                env.save_profile(
                                    modality, a,
                                    binding_alias=keys.get("BINDING_ALIAS", ""),
                                    model_name=keys.get("MODEL_NAME", ""),
                                    is_default=True,
                                    vision_enabled=keys.get("VISION_ENABLED", "").lower() == "true",
                                    forced_context_size=keys.get("FORCED_CONTEXT_SIZE", ""),
                                )
                                ui.notify(f"Profile '{a}' set as default.", type="positive")
                                refresh()

                            ui.button(icon="star", on_click=make_def).props("flat round dense size=sm color=amber").tooltip("Set as Default Profile")

                        ui.button(
                            icon="edit",
                            on_click=lambda a=alias: _open_edit_profile_dialog(env, modality, a, refresh),
                        ).props("flat round dense size=sm color=primary").tooltip("Edit Profile")

                        def do_del(a=alias):
                            env.delete_profile(modality, a)
                            ui.notify(f"Profile '{a}' deleted.", type="info")
                            refresh()

                        ui.button(icon="delete", on_click=do_del).props("flat round dense size=sm color=red").tooltip("Delete Profile")

                with ui.row().classes("items-center gap-2 text-xs font-mono text-slate-600 dark:text-slate-400"):
                    ui.label(f"Model: {model}")
                    ui.label("·")
                    ui.label(f"Via: {b_alias}")


# ==============================================================================
# Agent Behavior Section
# ==============================================================================

def _render_agent_section(prefs: GuiPrefs) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    # Header Card
    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("smart_toy", size="24px").classes("text-primary")
            ui.label("Agent Behavior & Reasoning Controls").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Tune cognitive sampling, token budgets, execution autonomy, and sub-agent delegation.").classes("text-xs text-slate-500")

    # Card 1: Reasoning & Turn Budgets
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4"):
        ui.label("Reasoning & Token Budgets").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        with ui.column().classes("w-full gap-1"):
            ui.label("Sampling Temperature").classes("text-xs font-semibold text-slate-700 dark:text-slate-300")
            temp_slider = ui.slider(min=0.0, max=1.2, step=0.05, value=prefs.temperature).props("label-always dense")
            ui.label().bind_text_from(temp_slider, "value", lambda v: f"Value: {v:.2f} (lower = more deterministic, higher = more creative)").classes("text-xs text-slate-500")

        with ui.row().classes("w-full gap-4 items-center"):
            tokens_in = ui.number("Max Tokens / Turn", value=prefs.max_tokens_per_turn, min=512, step=512).classes("flex-1").props("outlined dense")
            steps_in = ui.number("Max Reasoning Steps (Rounds)", value=prefs.max_reasoning_steps, min=1, max=200, step=1).classes("flex-1").props("outlined dense")
            compact_in = ui.number("Auto-Compaction Threshold (%)", value=int(getattr(prefs, "context_compaction_threshold", 0.85) * 100), min=50, max=95, step=5).classes("flex-1").props("outlined dense").tooltip("Context fill % at which non-essential files are locked and history compacted to prevent server disconnects")

        tokens_in.on_value_change(lambda e: setattr(prefs, "max_tokens_per_turn", int(e.value)))
        steps_in.on_value_change(lambda e: setattr(prefs, "max_reasoning_steps", int(e.value)))
        compact_in.on_value_change(lambda e: setattr(prefs, "context_compaction_threshold", float(e.value) / 100.0))
        temp_slider.on_value_change(lambda e: setattr(prefs, "temperature", float(e.value)))

    # Card 2: Shell Autonomy & Python Execution Security Policies
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3"):
        with ui.row().classes("items-center justify-between"):
            ui.label("Shell & Python Execution Security Policies").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
            badge_map = {"strict": ("STRICT", "amber"), "safe": ("SAFE", "emerald"), "full_access": ("FULL ACCESS", "red")}
            badge_label, badge_color = badge_map.get(prefs.shell_autonomy_level, ("SAFE", "emerald"))
            ui.badge(badge_label, color=badge_color).props("rounded dense")

        shell_switch = ui.switch("Enable System Shell Tool", value=prefs.enable_shell_execution)
        shell_switch.on_value_change(lambda e: setattr(prefs, "enable_shell_execution", e.value))

        with ui.column().classes("w-full gap-2.5 pt-1").bind_visibility_from(shell_switch, "value"):
            autonomy_select = ui.select(
                {
                    "strict": "Strict Mode (Prompt operator on EVERY Python execution and shell command)",
                    "safe": "Safe Mode (Auto-run benign algorithms, plots, docx/pdf/pptx; prompt on process spawning & shell escapes)",
                    "full_access": "Full Access (Unrestricted shell & Python execution without confirmation prompts)"
                },
                value=prefs.shell_autonomy_level,
                label="Execution Autonomy & Security Boundary"
            ).classes("w-full").props("outlined dense")
            autonomy_select.on_value_change(lambda e: setattr(prefs, "shell_autonomy_level", e.value))

            auto_py_switch = ui.switch(
                "Auto-Approve All Python Executions (Skip confirmation dialogs completely)",
                value=getattr(prefs, "auto_approve_python", False)
            ).props("dense")
            auto_py_switch.on_value_change(lambda e: setattr(prefs, "auto_approve_python", e.value))

            ui.label(
                "In Safe Mode, standard scripts, calculations, data science (pandas/numpy), document creation (pptx/pdf/docx), "
                "and plotting (matplotlib/seaborn) run autonomously. The system prompts you ONLY when risky operations "
                "(spawning processes via subprocess, os.system, shell scripting escapes) are detected."
            ).classes("text-[11px] text-slate-500 dark:text-slate-400 pl-1")

    # Card 3: Sub-Agents & Model Switching
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3"):
        ui.label("Sub-Agent Delegation").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        sub_switch = ui.switch("Enable Sub-Agent Spawning", value=prefs.enable_sub_agents)
        sub_switch.on_value_change(lambda e: setattr(prefs, "enable_sub_agents", e.value))

        with ui.row().classes("w-full gap-4 items-center").bind_visibility_from(sub_switch, "value"):
            depth_in = ui.number("Max Sub-Agent Recursion Depth", value=prefs.max_sub_agent_depth, min=1, max=5).classes("flex-1").props("outlined dense")
            count_in = ui.number("Max Sub-Agents / Turn", value=prefs.max_sub_agents_per_turn, min=1, max=10).classes("flex-1").props("outlined dense")

        depth_in.on_value_change(lambda e: setattr(prefs, "max_sub_agent_depth", int(e.value)))
        count_in.on_value_change(lambda e: setattr(prefs, "max_sub_agents_per_turn", int(e.value)))

        model_switch = ui.switch("Allow Model Switching Mid-Task", value=prefs.enable_model_switching)
        model_switch.on_value_change(lambda e: setattr(prefs, "enable_model_switching", e.value))

    # Card 4: Memory & Skills
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3"):
        ui.label("Memory & Skills Engine").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        mem_switch = ui.switch("Enable Persistent Memory", value=prefs.enable_memory)
        mem_switch.on_value_change(lambda e: setattr(prefs, "enable_memory", e.value))

        with ui.row().classes("w-full gap-4 items-center"):
            create_skill_switch = ui.switch("Enable Skill Creation", value=prefs.enable_skill_creation)
            load_skill_switch = ui.switch("Enable Skill Loading", value=prefs.enable_skill_loading)

        create_skill_switch.on_value_change(lambda e: setattr(prefs, "enable_skill_creation", e.value))
        load_skill_switch.on_value_change(lambda e: setattr(prefs, "enable_skill_loading", e.value))

        skills_mode = ui.select(
            SKILLS_MODES,
            value=prefs.skills_mode,
            label="Skills Visibility Mode"
        ).classes("w-full").props("outlined dense")
        skills_mode.on_value_change(lambda e: setattr(prefs, "skills_mode", e.value))

    # Card 5: Debug Mode
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2"):
        ui.label("Diagnostic & Debug").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
        debug_switch = ui.switch("Enable Debug Mode (Context & Prompt Dumps in .lollms_code/_debug_dumps)", value=prefs.debug)
        debug_switch.on_value_change(lambda e: setattr(prefs, "debug", e.value))


# ==============================================================================
# Workspace & Paths Section
# ==============================================================================

def _render_paths_section(prefs: GuiPrefs) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("folder", size="24px").classes("text-primary")
            ui.label("Workspace & System Paths").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Manage project sandbox root, global skills library, handbag folders, and memory databases.").classes("text-xs text-slate-500")

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4"):
        ui.label("Project Workspace Directory").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
        with ui.row().classes("w-full items-center gap-2"):
            ws_input = ui.input("Active Workspace Path", value=prefs.workspace_path).classes("flex-1").props("outlined dense")

            async def pick_ws():
                _picker = pick_folder
                if not _picker:
                    try:
                        from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder as _p
                        _picker = _p
                    except Exception:
                        pass
                if _picker:
                    chosen = await _picker(title="Select Workspace Directory", initial_dir=ws_input.value)
                    if chosen:
                        ws_input.value = chosen
                        prefs.workspace_path = chosen
                else:
                    ui.notify("Folder picker unavailable — enter path manually.", type="warning")

            ui.button("Browse…", icon="folder_open", on_click=pick_ws).props("outline dense no-caps")

        ws_input.on_value_change(lambda e: setattr(prefs, "workspace_path", e.value))

        ui.separator().classes(f"my-1 {BORDER}")
        ui.label("Persistent Repositories").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        skills_in = ui.input("Skills Directory Path", value=prefs.skills_dir).classes("w-full").props("outlined dense")
        handbag_in = ui.input("Handbag Directory Path", value=prefs.handbag_path).classes("w-full").props("outlined dense")
        db_in = ui.input("Global Memory DB URL", value=prefs.memory_db).classes("w-full").props("outlined dense")

        skills_in.on_value_change(lambda e: setattr(prefs, "skills_dir", e.value))
        handbag_in.on_value_change(lambda e: setattr(prefs, "handbag_path", e.value))
        db_in.on_value_change(lambda e: setattr(prefs, "memory_db", e.value))


# ==============================================================================
# Appearance Section
# ==============================================================================

def _render_appearance_section(prefs: GuiPrefs) -> None:
    CARD_BG = "bg-white dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"

    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("palette", size="24px").classes("text-primary")
            ui.label("Theme & Visual Customization").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Customize color palettes, typography, UI panel visibility, and window dimensions.").classes("text-xs text-slate-500")

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4"):
        ui.label("Theme & Accents").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        dark_sw = ui.switch("Dark Mode Enabled", value=prefs.dark_mode)
        def on_dark_toggle(e):
            prefs.dark_mode = e.value
            ui.dark_mode(e.value)
        dark_sw.on_value_change(on_dark_toggle)

        fullscreen_sw = ui.switch("Start Application in Fullscreen", value=getattr(prefs, "start_fullscreen", True))
        fullscreen_sw.on_value_change(lambda e: setattr(prefs, "start_fullscreen", e.value))

        with ui.row().classes("w-full gap-4 items-center"):
            preset_select = ui.select(
                list(ACCENT_PRESETS.keys()),
                value=next((k for k, v in ACCENT_PRESETS.items() if v == prefs.accent_color), "LoLLMS Blue"),
                label="Preset Accent Color"
            ).classes("flex-1").props("outlined dense")

            color_picker = ui.color_input("Custom Accent Hex", value=prefs.accent_color).classes("flex-1").props("outlined dense")

        def on_preset(e):
            hex_val = ACCENT_PRESETS.get(e.value, "#2563eb")
            color_picker.value = hex_val
            prefs.accent_color = hex_val
            ui.colors(primary=hex_val)

        def on_custom_color(e):
            prefs.accent_color = e.value
            ui.colors(primary=e.value)

        preset_select.on_value_change(on_preset)
        color_picker.on_value_change(on_custom_color)

        font_select = ui.select(
            ["JetBrains Mono, monospace", "Fira Code, monospace", "Cascadia Code, monospace", "system-ui, sans-serif"],
            value=prefs.font_family,
            label="Application Font"
        ).classes("w-full").props("outlined dense")

        def on_font(e):
            prefs.font_family = e.value
            ui.query("body").style(f"font-family: {e.value}")

        font_select.on_value_change(on_font)

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3"):
        ui.label("Chat Panel Visibility").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        sw_tools = ui.switch("Show Tool Execution Panels in Transcript", value=prefs.show_tool_calls)
        sw_changes = ui.switch("Show Workspace File Changes Table", value=prefs.show_workspace_changes)
        sw_skills = ui.switch("Show Skills Activity Badges", value=prefs.show_skills_activity)
        sw_sidebar = ui.switch("Show Live Telemetry Sidebar in Chat", value=prefs.show_live_sidebar)

        sw_tools.on_value_change(lambda e: setattr(prefs, "show_tool_calls", e.value))
        sw_changes.on_value_change(lambda e: setattr(prefs, "show_workspace_changes", e.value))
        sw_skills.on_value_change(lambda e: setattr(prefs, "show_skills_activity", e.value))
        sw_sidebar.on_value_change(lambda e: setattr(prefs, "show_live_sidebar", e.value))

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3"):
        ui.label("Window Dimensions (Native Mode)").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
        with ui.row().classes("w-full gap-4 items-center"):
            w_in = ui.number("Initial Width (px)", value=prefs.window_width, min=800, step=20).classes("flex-1").props("outlined dense")
            h_in = ui.number("Initial Height (px)", value=prefs.window_height, min=600, step=20).classes("flex-1").props("outlined dense")

        w_in.on_value_change(lambda e: setattr(prefs, "window_width", int(e.value)))
        h_in.on_value_change(lambda e: setattr(prefs, "window_height", int(e.value)))


# ==============================================================================
# Modal Dialogs for Adding / Editing Bindings and Profiles
# ==============================================================================

def _open_add_binding_dialog(env: EnvStore, modality: str, refresh) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg gap-3"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            ui.label(f"Add {MODALITY_LABELS[modality]} Server Binding").classes("text-base font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        available = env.available_bindings(modality)
        if not available:
            ui.label("No bindings discovered for this modality.").classes("text-xs text-slate-500")
            ui.button("Close", on_click=dialog.close).props("flat")
            dialog.open()
            return

        with ui.row().classes("w-full gap-3 items-center"):
            binding_select = ui.select(available, value=available[0], label="Engine Type").classes("flex-1").props("outlined dense")
            alias_input = ui.input("Alias (Unique Name)", value="MASTER").classes("flex-1").props("outlined dense")

        form_area = ui.column().classes("w-full gap-2")
        reader_holder = {"read": lambda: {}}

        def rebuild_form():
            form_area.clear()
            with form_area:
                reader_holder["read"] = _render_param_form(env, modality, binding_select.value)

        binding_select.on("update:model-value", lambda _: rebuild_form())
        rebuild_form()

        def do_save():
            alias_val = alias_input.value.strip()
            if not alias_val:
                ui.notify("Alias is required.", type="warning")
                return
            params = reader_holder["read"]()
            env.save_binding(modality, binding_select.value, alias_val, params)
            env.save()
            ui.notify(f"Binding '{alias_val.upper()}' registered.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            ui.button("Save Binding", on_click=do_save).props("unelevated dense color=primary")

    dialog.open()


def _open_edit_binding_dialog(env: EnvStore, modality: str, alias: str, refresh) -> None:
    dialog = ui.dialog()
    keys = env.binding_keys(modality, alias)
    binding_name = keys.get("BINDING_NAME", "")
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg gap-3"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            with ui.column().classes("gap-0"):
                ui.label(f"Edit Binding: {alias}").classes("text-base font-bold")
                ui.label(f"Provider: {binding_name}").classes("text-xs text-slate-500")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        form_area = ui.column().classes("w-full gap-2")
        with form_area:
            reader = _render_param_form(env, modality, binding_name, existing=keys)

        def do_save():
            params = reader()
            env.save_binding(modality, binding_name, alias, params)
            env.save()
            ui.notify(f"Binding '{alias}' updated and saved.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            ui.button("Update Binding", on_click=do_save).props("unelevated dense color=primary")

    dialog.open()


def _render_param_form(env: EnvStore, modality: str, binding_name: str, existing: Optional[Dict[str, str]] = None):
    schema = env.binding_param_schema(modality, binding_name)
    existing = existing or {}
    widgets: Dict[str, Any] = {}

    if not schema:
        ui.label("No formal schema file found — configure endpoint directly.").classes("text-xs text-slate-500")
        default_host = existing.get("HOST_ADDRESS", "http://localhost:8000")
        widgets["host_address"] = ui.input("Host Address", value=default_host).classes("w-full").props("outlined dense")

        def read_no_schema():
            return {"host_address": widgets["host_address"].value}
        return read_no_schema

    for p in schema:
        pname = p.get("name", "")
        ptype = p.get("type", "str")
        pdesc = p.get("description", "")
        pdefault = p.get("default")
        existing_val = existing.get(pname.upper())
        if existing_val is None:
            existing_val = existing.get(pname.lower())

        with ui.column().classes("w-full gap-0.5 mb-1"):
            if ptype == "bool":
                if existing_val is not None:
                    if isinstance(existing_val, bool):
                        init = existing_val
                    else:
                        init = str(existing_val).lower().strip() in ("true", "1", "yes", "y", "on")
                else:
                    init = bool(pdefault)
                widgets[pname] = ui.switch(pname.replace("_", " ").title(), value=init)
            elif ptype in ("int", "float"):
                init = existing_val if existing_val is not None else pdefault
                try:
                    init = float(init) if ptype == "float" else int(init)
                except (TypeError, ValueError):
                    init = 0
                widgets[pname] = ui.number(pname.replace("_", " ").title(), value=init, step=1 if ptype == "int" else 0.1).classes("w-full").props("outlined dense")
            elif "certificate" in pname.lower() or pname.lower().endswith("cert_path"):
                init = existing_val if existing_val is not None else (pdefault or "")
                with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                    cert_input = ui.input(
                        pname.replace("_", " ").title(),
                        value=str(init),
                    ).classes("flex-1").props("outlined dense clearable")
                    widgets[pname] = cert_input

                    async def _pick_cert_file(inp=cert_input):
                        from folder_picker import pick_file
                        chosen = await pick_file(
                            title="Select SSL Certificate File (.pem, .crt, .cer)",
                            initial_dir=inp.value or None,
                            file_types=[
                                ("Certificate Files", "*.pem;*.crt;*.cer;*.key"),
                                ("All Files", "*.*")
                            ]
                        )
                        if chosen:
                            inp.value = chosen
                            inp.update()

                    ui.button("Browse...", icon="file_open", on_click=_pick_cert_file).props(
                        "outline dense no-caps text-xs"
                    ).tooltip("Browse for certificate file (.pem, .crt, .cer)")
            else:
                init = existing_val if existing_val is not None else (pdefault or "")
                is_secret = any(s in pname.lower() for s in ("key", "token", "password", "secret"))
                widgets[pname] = ui.input(
                    pname.replace("_", " ").title(),
                    value=str(init),
                    password=is_secret,
                    password_toggle_button=is_secret
                ).classes("w-full").props("outlined dense")
            if pdesc:
                ui.label(pdesc[:140]).classes("text-[11px] text-slate-500 pl-1")

    def read_values():
        return {name: w.value for name, w in widgets.items()}

    return read_values


def _open_add_profile_dialog(env: EnvStore, modality: str, refresh) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg gap-3"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            ui.label(f"Add {MODALITY_LABELS[modality]} Model Profile").classes("text-base font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        alias_input = ui.input("Profile Alias (e.g. fast_local, gpt4o)", value="MASTER").classes("w-full").props("outlined dense")
        reader = _profile_form_body(env, modality)

        def do_save():
            if not reader:
                dialog.close()
                return
            alias_val = alias_input.value.strip()
            if not alias_val:
                ui.notify("Alias is required.", type="warning")
                return
            values = reader()
            env.save_profile(modality, alias_val, **values)
            env.save()
            ui.notify(f"Profile '{alias_val.upper()}' registered and saved.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            if reader:
                ui.button("Save Profile", on_click=do_save).props("unelevated dense color=primary")

    dialog.open()


def _open_edit_profile_dialog(env: EnvStore, modality: str, alias: str, refresh) -> None:
    dialog = ui.dialog()
    existing = env.profile_keys(modality, alias)
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg gap-3"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            ui.label(f"Edit Profile: {alias}").classes("text-base font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        reader = _profile_form_body(env, modality, existing=existing)

        def do_save():
            if not reader:
                dialog.close()
                return
            values = reader()
            env.save_profile(modality, alias, **values)
            env.save()
            ui.notify(f"Profile '{alias}' updated and saved.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            if reader:
                ui.button("Update Profile", on_click=do_save).props("unelevated dense color=primary")

    dialog.open()


def _profile_form_body(env: EnvStore, modality: str, existing: Optional[Dict[str, str]] = None):
    existing = existing or {}
    binding_aliases = env.configured_binding_aliases(modality)

    if not binding_aliases:
        ui.label("Add a server binding first — a profile must point to one.").classes("text-xs text-slate-500")
        return None

    default_binding_alias = existing.get("BINDING_ALIAS", binding_aliases[0])
    binding_alias_select = ui.select(binding_aliases, value=default_binding_alias, label="Linked Server Binding").classes("w-full").props("outlined dense")

    # ── Interactive Combobox (Type manual model OR select from fetched dropdown) ──
    initial_model = existing.get("MODEL_NAME", "").strip()
    initial_options = [initial_model] if initial_model else []
    typed_model = {"val": initial_model}

    with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
        model_select = ui.select(
            options=initial_options,
            value=initial_model or None,
            label="Model Name / Identifier",
            new_value_mode="add-unique",
        ).classes("flex-1 text-xs").props(
            ':dark="Quasar.Dark.isActive" outlined dense options-dense use-input fill-input hide-selected input-debounce=0 clearable'
        )

        def _on_input_val(e):
            if isinstance(e.args, str):
                typed_model["val"] = e.args.strip()

        def _on_select_change(e):
            if e.value:
                typed_model["val"] = str(e.value).strip()

        def _on_blur(_):
            val = typed_model["val"]
            if val and val not in model_select.options:
                model_select.options = list(model_select.options) + [val]
                model_select.value = val

        model_select.on("input-value", _on_input_val)
        model_select.on_value_change(_on_select_change)
        model_select.on("blur", _on_blur)

        async def fetch_models():
            selected_binding = binding_alias_select.value
            if not selected_binding:
                ui.notify("Please select a linked server binding first.", type="warning")
                return

            # Trigger loading spinner animation on the Fetch button
            fetch_btn.props(add="loading")
            ui.notify(f"Querying models from '{selected_binding}'...", type="info", timeout=2000)

            try:
                # Run the network query in a background thread so UI spinner animates fluidly
                models = await run.io_bound(env.fetch_models, modality, selected_binding)
                if not models:
                    ui.notify(f"No models discovered from '{selected_binding}'. Enter name manually.", type="warning", timeout=3000)
                    return

                # Preserve current model if it exists
                current_val = (model_select.value or typed_model["val"] or "").strip()
                merged_options = list(models)
                if current_val and current_val not in merged_options:
                    merged_options.insert(0, current_val)

                # Populate the combobox options directly in place
                model_select.options = merged_options
                if current_val:
                    model_select.value = current_val
                elif models:
                    model_select.value = models[0]
                    typed_model["val"] = models[0]

                model_select.update()

                # Automatically expand the dropdown menu in place
                model_select.run_method("showPopup")
                ui.notify(f"Discovered {len(models)} model(s)!", type="positive", timeout=2500)

            except Exception as ex:
                ui.notify(f"Failed to fetch models: {ex}", type="negative", timeout=5000)
            finally:
                fetch_btn.props(remove="loading")

        fetch_btn = ui.button("Fetch", icon="sync", on_click=fetch_models).props("outline dense no-caps").tooltip("Query endpoint to discover available models into dropdown")

    is_default_switch = ui.switch(
        "Make this the Default Profile for this Modality",
        value=existing.get("IS_DEFAULT", "").lower() == "true"
    )

    vision_switch = None
    ctx_input = None
    routing_widgets: Dict[str, Any] = {}

    if modality == "llm":
        vision_switch = ui.switch("Multimodal Vision Support Enabled", value=existing.get("VISION_ENABLED", "").lower() == "true")
        ctx_input = ui.input(
            "Forced Context Size (tokens, blank = auto-detect)",
            value=existing.get("FORCED_CONTEXT_SIZE", "")
        ).classes("w-full").props("outlined dense")

        with ui.expansion("Smart Router Routing Metadata (Optional)", icon="route").classes("w-full text-slate-900 dark:text-slate-100"):
            routing_widgets["description"] = ui.input(
                "Subject / Capability Keywords", value=existing.get("ROUTING_DESCRIPTION", "")
            ).classes("w-full").props("outlined dense")
            with ui.row().classes("w-full gap-2"):
                routing_widgets["cost"] = ui.number(
                    "Cost / 1k Tokens", value=float(existing.get("ROUTING_COST", "0.0") or 0.0), step=0.001
                ).classes("flex-1").props("outlined dense")
                routing_widgets["latency"] = ui.number(
                    "Avg Latency (ms)", value=int(existing.get("ROUTING_LATENCY", "100") or 100)
                ).classes("flex-1").props("outlined dense")
                routing_widgets["complexity"] = ui.select(
                    ["1", "2", "3"], value=existing.get("ROUTING_COMPLEXITY", "1") or "1", label="Complexity Tier"
                ).classes("flex-1").props("outlined dense")

    def read():
        routing = {}
        if routing_widgets:
            routing = {
                "description": routing_widgets["description"].value,
                "cost": routing_widgets["cost"].value,
                "latency": routing_widgets["latency"].value,
                "complexity": routing_widgets["complexity"].value,
            }
        resolved_model_name = (model_select.value or typed_model["val"] or "").strip()
        return dict(
            binding_alias=binding_alias_select.value,
            model_name=resolved_model_name,
            is_default=is_default_switch.value,
            vision_enabled=vision_switch.value if vision_switch else False,
            forced_context_size=ctx_input.value if ctx_input else "",
            routing=routing,
        )

    return read