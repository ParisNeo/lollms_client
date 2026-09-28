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
    "rag": "auto_stories",
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
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-200 dark:border-slate-800"
    TEXT_MAIN = "text-slate-900 dark:text-slate-100"
    TEXT_MUTED = "text-slate-600 dark:text-slate-400"

    HEADER_H = 42

    dark_mode = ui.dark_mode(value=prefs.is_dark())

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
                new_dark = not prefs.is_dark()
                prefs.theme_mode = "dark" if new_dark else "light"
                prefs.dark_mode = new_dark
                dark_mode.set_value(new_dark)
                try:
                    prefs.save()
                except Exception:
                    pass
                ui.notify(f"Theme: {'Dark' if new_dark else 'Light'}", type="info", timeout=1200)

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
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    title_label = MODALITY_LABELS[modality]
    icon_name = MODALITY_ICONS.get(modality, "cable")

    # Section Header Card
    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.icon(icon_name, size="32px").classes("text-primary")
                with ui.column().classes("gap-0"):
                    ui.label(f"{title_label} Configuration").classes("text-lg font-bold text-slate-900 dark:text-slate-100")
                    ui.label(
                        f"Configure server connections (Bindings) and model targets (Profiles) for {title_label}."
                    ).classes("text-xs text-slate-500 dark:text-slate-400")

    # Modality Sub-Tabs (Pill Toggle)
    view_state = {"tab": "profiles"}

    def on_add_click():
        if view_state["tab"] == "profiles":
            _open_add_profile_dialog(env, modality, refresh_parent)
        else:
            _open_add_binding_dialog(env, modality, refresh_parent)

    def on_top_commands_click():
        aliases = env.configured_binding_aliases(modality)
        if not aliases:
            ui.notify("No bindings configured yet. Add a binding first.", type="warning")
            return
        # Find bindings that have commands
        candidates = []
        for a in aliases:
            b_keys = env.binding_keys(modality, a)
            b_name = b_keys.get("BINDING_NAME")
            if b_name:
                cmds = _get_binding_commands_schema(env, modality, b_name)
                if cmds:
                    candidates.append((a, b_name, len(cmds)))
        if not candidates:
            ui.notify(f"None of the configured {modality.upper()} bindings define extra commands.", type="info")
            return
        if len(candidates) == 1:
            _open_binding_commands_dialog(env, modality, candidates[0][0], candidates[0][1])
        else:
            # Menu to pick which binding's commands to open
            d_pick = ui.dialog()
            with d_pick, ui.card().classes(f"w-[420px] p-4 gap-2 {CARD_BG} border {BORDER} rounded-xl shadow-xl"):
                ui.label(f"Select {modality.upper()} Binding Commands").classes("font-bold text-sm")
                for a_name, b_n, c_cnt in candidates:
                    def _open(al=a_name, bn=b_n):
                        d_pick.close()
                        _open_binding_commands_dialog(env, modality, al, bn)
                    ui.button(f"{a_name} ({b_n}) — {c_cnt} command(s)", icon="bolt", on_click=_open).props(
                        "outline dense no-caps text-xs w-full text-left"
                    )
            d_pick.open()

    with ui.row().classes("w-full items-center justify-between"):
        sub_toggle = ui.toggle(
            {"profiles": "📋 Model Profiles", "bindings": "🔌 Server Bindings"},
            value=view_state["tab"],
        ).props("dense unelevated size=sm").classes("text-xs")

        with ui.row().classes("items-center gap-2"):
            commands_top_btn = ui.button(
                "Commands", icon="bolt",
                on_click=on_top_commands_click,
            ).props("flat dense size=sm no-caps color=amber")
            commands_top_btn.tooltip(f"Execute special binding commands (e.g. pull model, bind mmproj, update binaries)")

            add_btn = ui.button(
                "Add Model Profile", icon="add",
                on_click=on_add_click,
            ).props("outline dense size=sm no-caps color=primary")

    panels_slot = ui.column().classes("w-full gap-3")

    def sync_views():
        panels_slot.clear()
        tab = view_state["tab"]
        if tab == "profiles":
            add_btn.text = "Add Model Profile"
            add_btn._props["icon"] = "add"
            add_btn.update()
            _render_profiles_cards(env, modality, panels_slot, refresh_parent)
        else:
            add_btn.text = "Add Server Binding"
            add_btn._props["icon"] = "add_link"
            add_btn.update()
            _render_bindings_cards(env, modality, panels_slot, refresh_parent)

    def on_toggle_change(e):
        view_state["tab"] = e.value
        sync_views()

    sub_toggle.on_value_change(on_toggle_change)
    sync_views()


def _render_bindings_cards(env: EnvStore, modality: str, container: ui.column, refresh) -> None:
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    aliases = env.configured_binding_aliases(modality)
    with container:
        if not aliases:
            with ui.card().classes(f"w-full p-8 border {BORDER} {CARD_BG} rounded-xl items-center justify-center gap-2 text-center").props(':dark="Quasar.Dark.isActive"'):
                ui.icon("cable", size="36px").classes("text-slate-400")
                ui.label("No Server Bindings Configured").classes("font-bold text-sm text-slate-700 dark:text-slate-300")
                ui.label(
                    "Add a binding to connect to an LLM provider (Ollama, OpenAI, vLLM, Llama.cpp server, etc.)."
                ).classes("text-xs text-slate-500 dark:text-slate-400 max-w-md")
                ui.button("Add Binding Now", icon="add", on_click=lambda: _open_add_binding_dialog(env, modality, refresh)).props(
                    "unelevated size=sm color=primary no-caps mt-2"
                )
            return

        for alias in aliases:
            keys = env.binding_keys(modality, alias)
            b_name = keys.get("BINDING_NAME", "unknown")
            host = keys.get("HOST_ADDRESS", "default host")
            cmds = _get_binding_commands_schema(env, modality, b_name)

            with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2").props(':dark="Quasar.Dark.isActive"'):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-3"):
                        ui.badge(b_name.upper(), color="indigo").props("rounded dense")
                        with ui.column().classes("gap-0"):
                            ui.label(alias).classes("font-mono font-bold text-sm text-slate-900 dark:text-slate-100")
                            ui.label(f"Endpoint: {host}").classes("text-xs text-slate-500 dark:text-slate-400 font-mono")

                    with ui.row().classes("items-center gap-1.5"):
                        if cmds:
                            ui.button(
                                f"Commands ({len(cmds)})", icon="bolt",
                                on_click=lambda a=alias, bn=b_name: _open_binding_commands_dialog(env, modality, a, bn),
                            ).props("outline dense size=sm no-caps color=amber font-semibold").tooltip(f"Execute special commands for {b_name}")

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
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    aliases = env.configured_profile_aliases(modality)
    with container:
        if not aliases:
            with ui.card().classes(f"w-full p-8 border {BORDER} {CARD_BG} rounded-xl items-center justify-center gap-2 text-center").props(':dark="Quasar.Dark.isActive"'):
                ui.icon("badge", size="36px").classes("text-slate-400")
                ui.label("No Model Profiles Configured").classes("font-bold text-sm text-slate-700 dark:text-slate-300")
                ui.label(
                    "Create a model profile that links a specific model name to a configured server binding."
                ).classes("text-xs text-slate-500 dark:text-slate-400 max-w-md")
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
            video_enabled = keys.get("VIDEO_ENABLED", "").lower() == "true"
            glm_embedding = keys.get("GLM_IMAGE_EMBEDDING", "").lower() == "true"
            efforts = keys.get("SUPPORTED_REASONING_EFFORTS", "")
            forced_ctx = keys.get("FORCED_CONTEXT_SIZE")

            with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2").props(':dark="Quasar.Dark.isActive"'):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        ui.label(alias).classes("font-mono font-bold text-sm text-slate-900 dark:text-slate-100")
                        if is_default:
                            ui.badge("⭐ Default", color="emerald").props("rounded dense")
                        if vision_enabled:
                            ui.badge("👁️ Vision", color="purple").props("rounded dense")
                        if video_enabled:
                            ui.badge("🎬 Video", color="teal").props("rounded dense")
                        if glm_embedding:
                            ui.badge("📐 GLM-5", color="blue").props("rounded dense")
                        if efforts:
                            ui.badge(f"🧠 {efforts}", color="indigo").props("rounded dense")
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
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    # Header Card
    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("smart_toy", size="24px").classes("text-primary")
            ui.label("Agent Behavior & Reasoning Controls").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Tune cognitive sampling, token budgets, execution autonomy, and sub-agent delegation.").classes("text-xs text-slate-500 dark:text-slate-400")

    # Card 1: Reasoning & Turn Budgets
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4").props(':dark="Quasar.Dark.isActive"'):
        ui.label("Reasoning & Token Budgets").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Sampling Temperature").classes("text-xs font-semibold text-slate-700 dark:text-slate-300")
                auto_temp_sw = ui.switch(
                    "Auto Temperature (Task-adapted)",
                    value=getattr(prefs, "auto_temperature", False)
                ).props("dense").tooltip("When Auto is enabled, temperature is chosen automatically: 0.15 for code & Aider patches, 0.7 for creative & conversational turns.")
                auto_temp_sw.on_value_change(lambda e: setattr(prefs, "auto_temperature", e.value))

            temp_slider = ui.slider(min=0.0, max=1.2, step=0.05, value=prefs.temperature).props("label-always dense")
            temp_desc = ui.label().bind_text_from(temp_slider, "value", lambda v: f"Value: {v:.2f} (lower = more deterministic, higher = more creative)").classes("text-xs text-slate-500")

            def _sync_temp_ui():
                is_auto = getattr(prefs, "auto_temperature", False)
                temp_slider.set_visibility(not is_auto)
                if is_auto:
                    temp_desc.set_text("Auto (Deterministic for code / creative for dialogue)")
                else:
                    temp_desc.set_text(f"Value: {prefs.temperature:.2f} (manual)")

            auto_temp_sw.on_value_change(lambda _: _sync_temp_ui())
            _sync_temp_ui()

        with ui.row().classes("w-full gap-4 items-center"):
            with ui.column().classes("flex-1 gap-1"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("Max Generation Tokens").classes("text-xs font-semibold")
                    auto_tokens_sw = ui.switch(
                        "Auto (Fill Remaining Ctx)",
                        value=getattr(prefs, "auto_max_tokens", False) or prefs.max_tokens_per_turn <= 0
                    ).props("dense").tooltip("Automatically use the model's full remaining context window per turn without artificial token cuts.")

                tokens_in = ui.number(
                    "Max Tokens / Turn",
                    value=prefs.max_tokens_per_turn if prefs.max_tokens_per_turn > 0 else 8192,
                    min=512, step=512
                ).classes("w-full").props("outlined dense")

                def _sync_tokens_ui(e=None):
                    is_auto = auto_tokens_sw.value
                    prefs.auto_max_tokens = is_auto
                    tokens_in.set_visibility(not is_auto)
                    if is_auto:
                        prefs.max_tokens_per_turn = 0
                    else:
                        prefs.max_tokens_per_turn = int(tokens_in.value or 8192)

                auto_tokens_sw.on_value_change(_sync_tokens_ui)
                tokens_in.on_value_change(lambda e: setattr(prefs, "max_tokens_per_turn", int(e.value or 8192)))
                tokens_in.set_visibility(not (getattr(prefs, "auto_max_tokens", False) or prefs.max_tokens_per_turn <= 0))

            steps_in = ui.number("Max Reasoning Steps (0 = Infinite ⚠️)", value=prefs.max_reasoning_steps, min=0, max=500, step=1).classes("flex-1").props("outlined dense").tooltip("Number of reasoning turns allowed. Set to 0 or -1 for infinite unbounded turns.")
            compact_in = ui.number("Auto-Compaction Threshold (%)", value=int(getattr(prefs, "context_compaction_threshold", 0.85) * 100), min=50, max=95, step=5).classes("flex-1").props("outlined dense").tooltip("Context fill % at which non-essential files are locked and history compacted to prevent server disconnects")

        with ui.row().classes("w-full gap-4 items-center"):
            effort_select = ui.select(
                {
                    "": "Model Default",
                    "none": "None / Deactivated",
                    "low": "Low Effort",
                    "medium": "Medium Effort",
                    "high": "High Effort",
                    "max": "Max Effort",
                },
                value=getattr(prefs, "reasoning_effort", "") or "",
                label="Base Reasoning Effort"
            ).classes("flex-1").props("outlined dense")
            effort_select.on_value_change(lambda e: setattr(prefs, "reasoning_effort", e.value if e.value else None))

            dynamic_effort_sw = ui.switch(
                "Dynamic Effort Scaling (<effort level='...'/>)",
                value=getattr(prefs, "dynamic_effort", False)
            ).props("dense").tooltip("Allow the agent to dynamically scale its reasoning effort up or down based on task complexity across rounds.")
            dynamic_effort_sw.on_value_change(lambda e: setattr(prefs, "dynamic_effort", e.value))

        tokens_in.on_value_change(lambda e: setattr(prefs, "max_tokens_per_turn", int(e.value)))

        def _on_steps_change(e):
            val = int(e.value or 0)
            setattr(prefs, "max_reasoning_steps", val)
            if val <= 0:
                ui.notify(
                    "⚠️ WARNING: Infinite reasoning rounds enabled (0). The agent will run indefinitely until <done/> is emitted or stopped manually. Monitor token budget!",
                    type="warning",
                    timeout=7000
                )

        steps_in.on_value_change(_on_steps_change)
        compact_in.on_value_change(lambda e: setattr(prefs, "context_compaction_threshold", float(e.value) / 100.0))
        temp_slider.on_value_change(lambda e: setattr(prefs, "temperature", float(e.value)))

    # Card 2: Shell Autonomy & Python Execution Security Policies
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("items-center justify-between"):
            ui.label("Shell & Python Execution Security Policies").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
            badge_map = {"strict": ("STRICT", "amber"), "safe": ("SAFE", "emerald"), "full_access": ("FULL ACCESS", "red")}
            badge_label, badge_color = badge_map.get(prefs.shell_autonomy_level, ("SAFE", "emerald"))
            ui.badge(badge_label, color=badge_color).props("rounded dense")

        shell_switch = ui.switch("Enable System Shell Tool", value=prefs.enable_shell_execution)
        shell_switch.on_value_change(lambda e: setattr(prefs, "enable_shell_execution", e.value))

        computer_use_sw = ui.switch(
            "Allow Desktop Automation / Computer Use (Requires Vision Model)",
            value=getattr(prefs, "allow_computer_use", False)
        ).props("dense").tooltip("Enables screenshotting, mouse clicking, moving, typing, and dragging when the active model supports vision.")
        computer_use_sw.on_value_change(lambda e: setattr(prefs, "allow_computer_use", e.value))

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
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3").props(':dark="Quasar.Dark.isActive"'):
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
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-3").props(':dark="Quasar.Dark.isActive"'):
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
    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-2").props(':dark="Quasar.Dark.isActive"'):
        ui.label("Diagnostic & Debug").classes("text-sm font-bold text-slate-800 dark:text-slate-200")
        debug_switch = ui.switch("Enable Debug Mode (Context & Prompt Dumps in .lollms_code/_debug_dumps)", value=prefs.debug)
        debug_switch.on_value_change(lambda e: setattr(prefs, "debug", e.value))


# ==============================================================================
# Workspace & Paths Section
# ==============================================================================

def _render_paths_section(prefs: GuiPrefs) -> None:
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("folder", size="24px").classes("text-primary")
            ui.label("Workspace & System Paths").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Manage project sandbox root, global skills library, handbag folders, and memory databases.").classes("text-xs text-slate-500 dark:text-slate-400")

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4").props(':dark="Quasar.Dark.isActive"'):
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
    CARD_BG = "bg-slate-100 dark:bg-slate-900"
    BORDER = "border-slate-300 dark:border-slate-800"

    with ui.card().classes(f"w-full p-4 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-1").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("items-center gap-2"):
            ui.icon("palette", size="24px").classes("text-primary")
            ui.label("Theme & Visual Customization").classes("text-base font-bold text-slate-900 dark:text-slate-100")
        ui.label("Customize color palettes, typography, UI panel visibility, and window dimensions.").classes("text-xs text-slate-500 dark:text-slate-400")

    with ui.card().classes(f"w-full p-5 border {BORDER} {CARD_BG} rounded-xl shadow-sm gap-4").props(':dark="Quasar.Dark.isActive"'):
        ui.label("Theme & Accents").classes("text-sm font-bold text-slate-800 dark:text-slate-200")

        dark_sw = ui.switch("Dark Mode Enabled", value=prefs.dark_mode)
        def on_dark_toggle(e):
            prefs.dark_mode = e.value
            ui.dark_mode(e.value)
        dark_sw.on_value_change(on_dark_toggle)

        maximized_sw = ui.switch("Start Application Maximized", value=getattr(prefs, "start_maximized", True))
        maximized_sw.on_value_change(lambda e: setattr(prefs, "start_maximized", e.value))

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

def _get_next_available_alias(base_alias: str, existing_list: List[str]) -> str:
    """Computes a unique incremental alias by testing _2, _3, etc."""
    upper_existing = {a.upper() for a in existing_list}
    if base_alias.upper() not in upper_existing:
        return base_alias
    counter = 2
    candidate = f"{base_alias}_{counter}"
    while candidate.upper() in upper_existing:
        counter += 1
        candidate = f"{base_alias}_{counter}"
    return candidate


def _show_alias_collision_dialog(
    item_type: str,
    chosen_alias: str,
    suggested_alias: str,
    on_use_suffix: Callable[[str], None],
    on_change_name: Optional[Callable[[], None]] = None,
):
    """Presents a confirmation dialog when an alias collision occurs."""
    warn_dialog = ui.dialog().props("persistent")
    with warn_dialog, ui.card().classes(
        "w-[480px] max-w-[95vw] p-5 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 "
        "border-2 border-amber-500 rounded-xl shadow-2xl gap-3"
    ):
        with ui.row().classes("items-center gap-2.5"):
            ui.icon("warning", size="26px").classes("text-amber-500")
            ui.label(f"{item_type.capitalize()} Name Already Exists").classes("text-base font-bold")

        ui.label(
            f'A {item_type} named "{chosen_alias}" is already configured.\n\n'
            f'Would you like to return to edit the name, or automatically save it as "{suggested_alias}"?'
        ).classes("text-xs text-slate-600 dark:text-slate-300 leading-relaxed whitespace-pre-line")

        with ui.row().classes("w-full items-center justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            def _back():
                warn_dialog.close()
                if on_change_name:
                    on_change_name()

            def _accept_suffix():
                warn_dialog.close()
                on_use_suffix(suggested_alias)

            ui.button("Change Name", icon="edit", on_click=_back).props("flat dense no-caps")
            ui.button(
                f'Use Suffix ("{suggested_alias}")',
                icon="auto_fix_high",
                on_click=_accept_suffix,
            ).props("unelevated dense color=primary no-caps font-semibold")

    warn_dialog.open()


def _open_add_binding_dialog(env: EnvStore, modality: str, refresh) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-300 dark:border-slate-800 rounded-xl shadow-lg gap-3").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            ui.label(f"Add {MODALITY_LABELS[modality]} Server Binding").classes("text-base font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        available = env.available_bindings(modality)
        if not available:
            ui.label("No bindings discovered for this modality.").classes("text-xs text-slate-500")
            ui.button("Close", on_click=dialog.close).props("flat")
            dialog.open()
            return

        existing_bindings = env.configured_binding_aliases(modality)
        default_alias = _get_next_available_alias("MASTER", existing_bindings)

        with ui.row().classes("w-full gap-3 items-center"):
            binding_select = ui.select(available, value=available[0], label="Engine Type").classes("flex-1").props("outlined dense")
            alias_input = ui.input("Alias (Unique Name)", value=default_alias).classes("flex-1").props("outlined dense")

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
            current_bindings = env.configured_binding_aliases(modality)

            # Check for collision
            if any(b.upper() == alias_val.upper() for b in current_bindings):
                suggested = _get_next_available_alias(alias_val, current_bindings)

                def _use_suffixed(suffixed_name: str):
                    alias_input.value = suffixed_name
                    env.save_binding(modality, binding_select.value, suffixed_name, params)
                    env.save()
                    ui.notify(f"Binding registered as '{suffixed_name.upper()}'.", type="positive")
                    dialog.close()
                    refresh()

                _show_alias_collision_dialog(
                    item_type="binding",
                    chosen_alias=alias_val,
                    suggested_alias=suggested,
                    on_use_suffix=_use_suffixed,
                    on_change_name=lambda: alias_input.run_method("focus")
                )
                return

            env.save_binding(modality, binding_select.value, alias_val, params)
            env.save()
            ui.notify(f"Binding '{alias_val.upper()}' registered.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")
            ui.button("Save Binding", on_click=do_save).props("unelevated dense color=primary")

    dialog.open()


def _get_binding_commands_schema(env: EnvStore, modality: str, binding_name: str) -> List[Dict[str, Any]]:
    """Loads description.yaml for binding_name and returns command specs."""
    try:
        from lollms_client.lollms_config_cli_env import _get_binding_description
        desc = _get_binding_description(binding_name, modality)
        if desc and isinstance(desc, dict) and "commands" in desc:
            cmds = desc["commands"]
            return cmds if isinstance(cmds, list) else []
    except Exception:
        pass
    return []


def _instantiate_temporary_binding(env: EnvStore, modality: str, alias: str, binding_name: str) -> Any:
    """Instantiates a live binding object configured with the alias's parameters."""
    keys = env.binding_keys(modality, alias)
    b_config = {}
    for k, v in keys.items():
        if k == "BINDING_NAME":
            continue
        key_lower = k.lower()
        if key_lower == "verify_ssl_certificate":
            b_config[key_lower] = str(v).lower() in ("true", "1", "yes", "on")
        else:
            try:
                b_config[key_lower] = int(v)
            except (ValueError, TypeError):
                b_config[key_lower] = v

    try:
        if modality == "llm":
            from lollms_client.lollms_llm_binding import LollmsLLMBindingManager
            mgr = LollmsLLMBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "tti":
            from lollms_client.lollms_tti_binding import LollmsTTIBindingManager
            mgr = LollmsTTIBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "ttm":
            from lollms_client.lollms_ttm_binding import LollmsTTMBindingManager
            mgr = LollmsTTMBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "ttv":
            from lollms_client.lollms_ttv_binding import LollmsTTVBindingManager
            mgr = LollmsTTVBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "tts":
            from lollms_client.lollms_tts_binding import LollmsTTSBindingManager
            mgr = LollmsTTSBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "stt":
            from lollms_client.lollms_stt_binding import LollmsSTTBindingManager
            mgr = LollmsSTTBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
        elif modality == "rag":
            from lollms_client.lollms_rag_binding import LollmsRAGBindingManager
            mgr = LollmsRAGBindingManager()
            return mgr.create_binding(binding_name=binding_name, **b_config)
    except Exception as e:
        ASCIIColors.warning(f"Could not instantiate temporary binding {binding_name}: {e}")
    return None


def _open_binding_commands_dialog(env: EnvStore, modality: str, alias: str, binding_name: str) -> None:
    """Renders the interactive command runner dialog driven by description.yaml."""
    commands_spec = _get_binding_commands_schema(env, modality, binding_name)
    if not commands_spec:
        ui.notify(f"No special commands defined in description.yaml for '{binding_name}'.", type="info")
        return

    # Instantiate binding to verify methods
    binding_inst = _instantiate_temporary_binding(env, modality, alias, binding_name)

    # Filter to commands actually implemented on the binding class
    implemented_commands = []
    for cmd in commands_spec:
        c_name = cmd.get("name", "")
        if binding_inst and hasattr(binding_inst, c_name) and callable(getattr(binding_inst, c_name)):
            implemented_commands.append(cmd)
        elif not binding_inst:
            # If temporary instance couldn't be spun up, assume declared commands exist
            implemented_commands.append(cmd)

    if not implemented_commands:
        ui.notify(f"Declared commands for '{binding_name}' are not implemented in the binding code.", type="warning")
        return

    dialog = ui.dialog().props("maximized")
    with dialog, ui.card().classes(
        "w-full h-full flex flex-col p-5 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 gap-3 overflow-hidden"
    ):
        # Header
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800 shrink-0"):
            with ui.row().classes("items-center gap-2.5"):
                ui.icon("bolt", size="28px").classes("text-amber-500")
                with ui.column().classes("gap-0"):
                    ui.label(f"Binding Commands: {alias} ({binding_name})").classes("text-base font-bold")
                    ui.label(f"Execute special operations provided by {binding_name}.").classes("text-xs text-slate-500")
            ui.button("Close", icon="close", on_click=dialog.close).props("flat dense round size=sm")

        # Tabs for commands
        with ui.tabs().classes("w-full bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 shrink-0").props('dense no-caps active-color="primary" indicator-color="primary"') as cmd_tabs:
            tab_widgets = {}
            for idx, cmd in enumerate(implemented_commands):
                c_title = cmd.get("title") or cmd.get("name", "Command")
                t_w = ui.tab(f"cmd_{idx}", label=c_title, icon="play_arrow").classes("text-xs py-1.5 flex-1")
                tab_widgets[f"cmd_{idx}"] = t_w

        # Tab panels
        first_tab_name = f"cmd_0" if implemented_commands else None
        with ui.tab_panels(cmd_tabs, value=first_tab_name).classes("w-full flex-1 min-h-0 p-2 bg-transparent flex flex-col overflow-hidden"):
            for idx, cmd in enumerate(implemented_commands):
                c_name = cmd.get("name", "")
                c_title = cmd.get("title") or c_name
                c_desc = cmd.get("description", "")
                c_params = cmd.get("parameters", [])

                with ui.tab_panel(f"cmd_{idx}").classes("w-full h-full p-2 flex flex-col overflow-hidden gap-3"):
                    with ui.card().classes("w-full p-3 bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-xl gap-1 shrink-0").props(':dark="Quasar.Dark.isActive"'):
                        ui.label(c_title).classes("font-bold text-sm text-primary")
                        ui.label(c_desc).classes("text-xs text-slate-500 dark:text-slate-400 leading-relaxed")

                    # Form inputs area
                    input_widgets = {}
                    with ui.scroll_area().classes("w-full flex-1 p-3 bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 rounded-xl"):
                        with ui.column().classes("w-full gap-3"):
                            if not c_params:
                                ui.label("This command requires no additional parameters. Click 'Execute Command' below to run.").classes("text-xs text-slate-500 italic p-2")

                            for p in c_params:
                                p_name = p.get("name", "")
                                p_type = p.get("type", "str")
                                p_desc = p.get("description", "")
                                p_req = p.get("mandatory", False)
                                p_default = p.get("default", "")

                                with ui.column().classes("w-full gap-0.5"):
                                    lbl = f"{p_name.replace('_', ' ').title()} {'*' if p_req else ''}"
                                    if p_type == "bool":
                                        sw = ui.switch(lbl, value=bool(p_default)).props("dense")
                                        input_widgets[p_name] = sw
                                    elif p_name in ("mmproj_name", "projector_name", "mmproj_filename"):
                                        proj_names = []
                                        if binding_inst and hasattr(binding_inst, "list_mmproj_models"):
                                            try:
                                                raw_projs = binding_inst.list_mmproj_models()
                                                proj_names = [m.get("model_name", str(m)) if isinstance(m, dict) else str(m) for m in raw_projs]
                                            except Exception:
                                                pass
                                        if not proj_names and binding_inst and hasattr(binding_inst, "models_dir") and getattr(binding_inst, "models_dir", None):
                                            try:
                                                md = Path(binding_inst.models_dir)
                                                if md.exists():
                                                    for f in md.glob("*"):
                                                        if f.is_file() and "mmproj" in f.name.lower() and f.name.lower().endswith((".gguf", ".mmproj", ".bin")):
                                                            proj_names.append(f.name)
                                            except Exception:
                                                pass
                                        sel = ui.select(
                                            options=proj_names,
                                            value=proj_names[0] if proj_names else None,
                                            label=lbl,
                                            new_value_mode="add-unique",
                                        ).classes("w-full text-xs").props("outlined dense use-input fill-input clearable")
                                        input_widgets[p_name] = sel
                                    elif p_name in ("model_name", "filename") and binding_inst and hasattr(binding_inst, "list_models"):
                                        try:
                                            raw_models = binding_inst.list_models()
                                            model_names = [m.get("model_name", str(m)) if isinstance(m, dict) else str(m) for m in raw_models]
                                        except Exception:
                                            model_names = []
                                        sel = ui.select(
                                            options=model_names,
                                            value=model_names[0] if model_names else None,
                                            label=lbl,
                                            new_value_mode="add-unique",
                                        ).classes("w-full text-xs").props("outlined dense use-input fill-input clearable")
                                        input_widgets[p_name] = sel
                                    elif p_type in ("int", "float"):
                                        try:
                                            val_num = float(p_default) if p_type == "float" else int(p_default)
                                        except Exception:
                                            val_num = 0
                                        num_in = ui.number(lbl, value=val_num, step=1 if p_type == "int" else 0.1).classes("w-full").props("outlined dense")
                                        input_widgets[p_name] = num_in
                                    else:
                                        txt_in = ui.input(lbl, value=str(p_default or "")).classes("w-full text-xs").props("outlined dense clearable")
                                        input_widgets[p_name] = txt_in

                                    if p_desc:
                                        ui.label(p_desc).classes("text-[10px] text-slate-400 pl-1")

                    # Live execution and progress console
                    with ui.card().classes("w-full p-3 bg-slate-900 border border-slate-800 rounded-xl gap-2 shrink-0"):
                        progress_bar = ui.linear_progress(value=0.0).props("instant-feedback color=amber")
                        progress_bar.visible = False
                        status_label = ui.label("Ready").classes("text-xs font-mono text-slate-300")

                        with ui.row().classes("w-full items-center justify-between pt-1"):
                            exec_btn = ui.button(f"Execute {c_name}", icon="play_arrow").props("unelevated dense size=sm color=amber no-caps font-semibold")

                            def _make_runner(target_cmd=c_name, widgets=input_widgets, p_bar=progress_bar, s_lbl=status_label, btn=exec_btn):
                                async def _run_command():
                                    if not binding_inst:
                                        ui.notify("Binding instance could not be initialized.", type="negative")
                                        return
                                    fn = getattr(binding_inst, target_cmd, None)
                                    if not callable(fn):
                                        ui.notify(f"Method '{target_cmd}' is not implemented on binding.", type="negative")
                                        return

                                    # Gather parameters
                                    call_kwargs = {}
                                    for k, w in widgets.items():
                                        val = w.value
                                        if isinstance(val, str):
                                            val = val.strip()
                                        call_kwargs[k] = val

                                    # Validate mandatory parameters before execution
                                    missing_mandatory = []
                                    for p_spec in c_params:
                                        p_key = p_spec.get("name")
                                        if p_spec.get("mandatory") and not call_kwargs.get(p_key):
                                            missing_mandatory.append(p_spec.get("title") or p_key)
                                    if missing_mandatory:
                                        ui.notify(f"Please select or enter: {', '.join(missing_mandatory)}", type="warning")
                                        return

                                    # Wire progress callback if method accepts it
                                    import inspect
                                    sig = inspect.signature(fn)
                                    if "progress_callback" in sig.parameters:
                                        def _prog_cb(data):
                                            if isinstance(data, dict):
                                                msg = data.get("message") or data.get("status") or ""
                                                completed = data.get("completed", 0)
                                                total = data.get("total", 100)
                                                if total > 0:
                                                    p_bar.value = float(completed) / float(total)
                                                s_lbl.set_text(f"⏳ {msg}")
                                            elif isinstance(data, str):
                                                s_lbl.set_text(f"⏳ {data}")
                                        call_kwargs["progress_callback"] = _prog_cb

                                    btn.props(add="loading")
                                    p_bar.visible = True
                                    s_lbl.set_text(f"⏳ Executing {target_cmd}...")

                                    try:
                                        from nicegui import run as _ng_run
                                        res = await _ng_run.io_bound(fn, **call_kwargs)
                                        p_bar.value = 1.0
                                        res_str = json.dumps(res, indent=2) if isinstance(res, (dict, list)) else str(res)
                                        if isinstance(res, dict) and (res.get("status") is False or res.get("success") is False or res.get("error")):
                                            err_txt = res.get("error") or res.get("message") or "Command execution failed."
                                            s_lbl.set_text(f"❌ Failed: {err_txt}")
                                            ui.notify(f"{target_cmd} error: {err_txt}", type="negative")
                                        else:
                                            s_lbl.set_text(f"✅ Finished: {res_str[:120]}")
                                            ui.notify(f"✓ {target_cmd} completed successfully.", type="positive")
                                    except Exception as ex:
                                        s_lbl.set_text(f"❌ Error: {ex}")
                                        ui.notify(f"Command '{target_cmd}' failed: {ex}", type="negative")
                                    finally:
                                        btn.props(remove="loading")

                                return _run_command

                            exec_btn.on("click", _make_runner())

    dialog.open()


def _open_edit_binding_dialog(env: EnvStore, modality: str, alias: str, refresh) -> None:
    dialog = ui.dialog()
    keys = env.binding_keys(modality, alias)
    binding_name = keys.get("BINDING_NAME", "")
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-300 dark:border-slate-800 rounded-xl shadow-lg gap-3").props(':dark="Quasar.Dark.isActive"'):
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
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-300 dark:border-slate-800 rounded-xl shadow-lg gap-3").props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            ui.label(f"Add {MODALITY_LABELS[modality]} Model Profile").classes("text-base font-bold")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense size=xs")

        existing_profiles = env.configured_profile_aliases(modality)
        default_alias = _get_next_available_alias("MASTER", existing_profiles)

        alias_input = ui.input("Profile Alias (e.g. fast_local, gpt4o)", value=default_alias).classes("w-full").props("outlined dense")
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
            current_profiles = env.configured_profile_aliases(modality)

            if any(p.upper() == alias_val.upper() for p in current_profiles):
                suggested = _get_next_available_alias(alias_val, current_profiles)

                def _use_suffixed(suffixed_name: str):
                    alias_input.value = suffixed_name
                    env.save_profile(modality, suffixed_name, **values)
                    env.save()
                    ui.notify(f"Profile registered as '{suffixed_name.upper()}'.", type="positive")
                    dialog.close()
                    refresh()

                _show_alias_collision_dialog(
                    item_type="profile",
                    chosen_alias=alias_val,
                    suggested_alias=suggested,
                    on_use_suffix=_use_suffixed,
                    on_change_name=lambda: alias_input.run_method("focus")
                )
                return

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
    with dialog, ui.card().classes("w-[560px] max-w-[95vw] p-5 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 border border-slate-300 dark:border-slate-800 rounded-xl shadow-lg gap-3").props(':dark="Quasar.Dark.isActive"'):
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
    video_switch = None
    glm_switch = None
    efforts_input = None
    ctx_input = None
    routing_widgets: Dict[str, Any] = {}

    if modality == "llm":
        with ui.row().classes("w-full gap-4 items-center flex-wrap"):
            vision_switch = ui.switch("Multimodal Vision Support", value=existing.get("VISION_ENABLED", "").lower() == "true").props("dense")
            video_switch = ui.switch("Video Input Comprehension", value=existing.get("VIDEO_ENABLED", "").lower() == "true").props("dense")
            glm_switch = ui.switch("GLM-5.3-Flash / GLM-4V Image Embedding", value=existing.get("GLM_IMAGE_EMBEDDING", "").lower() == "true").props("dense")

        efforts_input = ui.input(
            "Supported Reasoning Efforts (e.g. low, high, max)",
            value=existing.get("SUPPORTED_REASONING_EFFORTS", "")
        ).classes("w-full").props("outlined dense").tooltip("Comma-separated list of reasoning levels supported by this model")

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
            video_enabled=video_switch.value if video_switch else False,
            glm_image_embedding=glm_switch.value if glm_switch else False,
            supported_reasoning_efforts=efforts_input.value if efforts_input else "",
            forced_context_size=ctx_input.value if ctx_input else "",
            routing=routing,
        )

    return read