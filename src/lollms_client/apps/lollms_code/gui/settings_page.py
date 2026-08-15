"""
settings_page.py — in-app replacement for run_wizard_and_save().

Structure mirrors the CLI wizard's menu tree:
    Modality (llm/tti/tts/stt/ttm/ttv)
      -> Bindings  (an aliased, configured instance of a binding, e.g. MASTER -> ollama)
      -> Profiles  (an alias that points at a binding + model + flags/routing)
Plus two GUI-only tabs (Agent Behavior, Paths, Appearance) that aren't part of
the lollms_client .env schema at all — those persist to gui_prefs.json instead.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from nicegui import ui

from env_config import EnvStore, MODALITIES, MODALITY_LABELS
from gui_prefs import GuiPrefs, SHELL_AUTONOMY_LEVELS, SKILLS_MODES, ACCENT_PRESETS


def build_settings_page(env: EnvStore, prefs: GuiPrefs, on_saved) -> None:
    ui.label("Settings").classes("text-2xl font-bold mb-2")

    if env.import_error:
        with ui.row().classes("w-full max-w-3xl items-center gap-2 bg-red-50 dark:bg-red-950 rounded p-3 mb-3"):
            ui.icon("warning", color="red-500")
            ui.label(
                f"lollms_client isn't importable ({env.import_error}). "
                "Bindings/models can't be listed until it's on the path."
            ).classes("text-sm text-red-700 dark:text-red-300")

    with ui.tabs().classes("w-full") as tabs:
        modality_tabs = {m: ui.tab(MODALITY_LABELS[m], icon="cable") for m in MODALITIES}
        t_agent = ui.tab("Agent Behavior", icon="smart_toy")
        t_paths = ui.tab("Paths", icon="folder")
        t_appearance = ui.tab("Appearance", icon="palette")

    with ui.tab_panels(tabs, value=modality_tabs["llm"]).classes("w-full"):
        for modality in MODALITIES:
            with ui.tab_panel(modality_tabs[modality]):
                _build_modality_panel(env, modality)

        with ui.tab_panel(t_agent):
            agent_setters = _build_agent_panel(prefs)

        with ui.tab_panel(t_paths):
            path_setters = _build_paths_panel(prefs)

        with ui.tab_panel(t_appearance):
            appearance_setters = _build_appearance_panel(prefs)

    def save_prefs_and_continue():
        agent_setters(prefs)
        path_setters(prefs)
        appearance_setters(prefs)
        prefs.save()
        ui.notify("Preferences saved.", type="positive")
        if on_saved:
            on_saved(env, prefs)

    with ui.row().classes("w-full max-w-3xl justify-end mt-4 gap-2"):
        ui.button("Save preferences", icon="save", on_click=save_prefs_and_continue).props("color=primary")


# ============================== Modality panel ==============================

def _build_modality_panel(env: EnvStore, modality: str) -> None:
    with ui.tabs().classes("w-full") as sub_tabs:
        t_bindings = ui.tab("Bindings", icon="cable")
        t_profiles = ui.tab("Profiles", icon="badge")

    with ui.tab_panels(sub_tabs, value=t_bindings).classes("w-full"):
        with ui.tab_panel(t_bindings):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(f"{MODALITY_LABELS[modality]} bindings").classes("text-sm text-gray-500")
                ui.button(
                    "Add binding", icon="add",
                    on_click=lambda: _open_add_binding_dialog(env, modality, lambda: refresh_bindings()),
                ).props("flat color=primary")
            bindings_container = ui.column().classes("w-full gap-2")

        with ui.tab_panel(t_profiles):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(f"{MODALITY_LABELS[modality]} profiles").classes("text-sm text-gray-500")
                ui.button(
                    "Add profile", icon="add",
                    on_click=lambda: _open_add_profile_dialog(env, modality, lambda: refresh_profiles()),
                ).props("flat color=primary")
            profiles_container = ui.column().classes("w-full gap-2")

    def refresh_bindings():
        bindings_container.clear()
        _render_bindings_list(env, modality, bindings_container, refresh_bindings)

    def refresh_profiles():
        profiles_container.clear()
        _render_profiles_list(env, modality, profiles_container, refresh_profiles)

    refresh_bindings()
    refresh_profiles()



def _render_bindings_list(env: EnvStore, modality: str, container: ui.column, refresh) -> None:
    aliases = env.configured_binding_aliases(modality)
    with container:
        if not aliases:
            ui.label("No bindings configured yet.").classes("text-sm text-gray-500")
        for alias in aliases:
            keys = env.binding_keys(modality, alias)
            b_name = keys.get("BINDING_NAME", "unknown")
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-0"):
                        ui.label(alias).classes("font-mono font-bold")
                        ui.label(b_name).classes("text-xs text-gray-500")
                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="edit",
                            on_click=lambda a=alias: _open_edit_binding_dialog(env, modality, a, refresh),
                        ).props("flat round dense")
                        ui.button(
                            icon="delete",
                            on_click=lambda a=alias: (env.delete_binding(modality, a), refresh()),
                        ).props("flat round dense color=red")


def _render_profiles_list(env: EnvStore, modality: str, container: ui.column, refresh) -> None:
    aliases = env.configured_profile_aliases(modality)
    with container:
        if not aliases:
            ui.label("No profiles configured yet.").classes("text-sm text-gray-500")
        for alias in aliases:
            keys = env.profile_keys(modality, alias)
            b_alias = keys.get("BINDING_ALIAS", "?")
            model = keys.get("MODEL_NAME", "?")
            is_default = keys.get("IS_DEFAULT", "").lower() == "true"
            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(alias).classes("font-mono font-bold")
                            if is_default:
                                ui.badge("default", color="primary")
                        ui.label(f"{b_alias} · {model}").classes("text-xs text-gray-500")
                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="edit",
                            on_click=lambda a=alias: _open_edit_profile_dialog(env, modality, a, refresh),
                        ).props("flat round dense")
                        ui.button(
                            icon="delete",
                            on_click=lambda a=alias: (env.delete_profile(modality, a), refresh()),
                        ).props("flat round dense color=red")


# ============================== Binding dialogs ==============================

def _render_param_form(env: EnvStore, modality: str, binding_name: str, existing: Optional[Dict[str, str]] = None):
    """Renders the dynamic parameter form for a binding (from its
    description.yaml schema) and returns a callable that reads back the
    entered values as {param_name: value}."""
    schema = env.binding_param_schema(modality, binding_name)
    existing = existing or {}
    widgets: Dict[str, Any] = {}

    if not schema:
        ui.label("No parameter schema found — enter host address manually.").classes("text-sm text-gray-500")
        default_host = existing.get("HOST_ADDRESS", "http://localhost:8000")
        widgets["host_address"] = ui.input("Host address", value=default_host).classes("w-full")

        def read_no_schema():
            return {"host_address": widgets["host_address"].value}
        return read_no_schema

    for p in schema:
        pname = p.get("name", "")
        ptype = p.get("type", "str")
        pdesc = p.get("description", "")
        pdefault = p.get("default")
        existing_val = existing.get(pname.upper())

        with ui.column().classes("w-full gap-0 mb-1"):
            if ptype == "bool":
                init = existing_val.lower() == "true" if existing_val is not None else bool(pdefault)
                widgets[pname] = ui.switch(pname, value=init)
            elif ptype in ("int", "float"):
                init = existing_val if existing_val is not None else pdefault
                try:
                    init = float(init) if ptype == "float" else int(init)
                except (TypeError, ValueError):
                    init = 0
                widgets[pname] = ui.number(pname, value=init, step=1 if ptype == "int" else 0.1).classes("w-full")
            else:
                init = existing_val if existing_val is not None else (pdefault or "")
                is_secret = "key" in pname.lower() or "token" in pname.lower() or "password" in pname.lower()
                widgets[pname] = ui.input(
                    pname, value=str(init), password=is_secret, password_toggle_button=is_secret
                ).classes("w-full")
            if pdesc:
                ui.label(pdesc[:140]).classes("text-xs text-gray-500")

    def read_values():
        return {name: w.value for name, w in widgets.items()}

    return read_values


def _open_add_binding_dialog(env: EnvStore, modality: str, refresh) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[480px]"):
        ui.label(f"Add {MODALITY_LABELS[modality]} binding").classes("text-lg font-bold")
        available = env.available_bindings(modality)
        if not available:
            ui.label("No bindings discovered for this modality.").classes("text-sm text-gray-500")
            ui.button("Close", on_click=dialog.close)
            dialog.open()
            return

        binding_select = ui.select(available, value=available[0], label="Binding").classes("w-full")
        alias_input = ui.input("Alias", value="MASTER").classes("w-full")

        form_area = ui.column().classes("w-full")
        reader_holder = {"read": lambda: {}}

        def rebuild_form():
            form_area.clear()
            with form_area:
                reader_holder["read"] = _render_param_form(env, modality, binding_select.value)

        binding_select.on("update:model-value", lambda _: rebuild_form())
        rebuild_form()

        def do_save():
            if not alias_input.value.strip():
                ui.notify("Alias is required.", type="warning")
                return
            params = reader_holder["read"]()
            env.save_binding(modality, binding_select.value, alias_input.value, params)
            ui.notify(f"Binding '{alias_input.value.upper()}' saved.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", on_click=do_save).props("color=primary")

    dialog.open()


def _open_edit_binding_dialog(env: EnvStore, modality: str, alias: str, refresh) -> None:
    dialog = ui.dialog()
    keys = env.binding_keys(modality, alias)
    binding_name = keys.get("BINDING_NAME", "")
    with dialog, ui.card().classes("w-[480px]"):
        ui.label(f"Edit binding: {alias}").classes("text-lg font-bold")
        ui.label(binding_name).classes("text-xs text-gray-500 mb-2")

        form_area = ui.column().classes("w-full")
        with form_area:
            reader = _render_param_form(env, modality, binding_name, existing=keys)

        def do_save():
            params = reader()
            env.save_binding(modality, binding_name, alias, params)
            ui.notify("Binding updated.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", on_click=do_save).props("color=primary")

    dialog.open()


# ============================== Profile dialogs ==============================

def _profile_form_body(env: EnvStore, modality: str, existing: Optional[Dict[str, str]] = None):
    existing = existing or {}
    binding_aliases = env.configured_binding_aliases(modality)

    if not binding_aliases:
        ui.label("Add a binding first — a profile needs one to point at.").classes("text-sm text-gray-500")
        return None

    default_binding_alias = existing.get("BINDING_ALIAS", binding_aliases[0])
    binding_alias_select = ui.select(binding_aliases, value=default_binding_alias, label="Binding").classes("w-full")

    model_state = {"value": existing.get("MODEL_NAME", "")}
    with ui.row().classes("w-full items-end gap-2"):
        model_input = ui.input("Model name", value=model_state["value"]).classes("flex-1")

        async def fetch_models():
            models = env.fetch_models(modality, binding_alias_select.value)
            if not models:
                ui.notify("No models found automatically — enter one manually.", type="warning")
                return
            model_pick_dialog = ui.dialog()
            with model_pick_dialog, ui.card():
                ui.label("Select a model").classes("font-bold mb-2")
                for m in models:
                    def pick(m=m):
                        model_input.value = m
                        model_pick_dialog.close()
                    ui.button(m, on_click=pick).props("flat align=left").classes("w-full justify-start")
            model_pick_dialog.open()

        ui.button("Fetch models", icon="search", on_click=fetch_models).props("outline")

    is_default_switch = ui.switch(
        "Default profile for this modality", value=existing.get("IS_DEFAULT", "").lower() == "true"
    )

    vision_switch = None
    ctx_input = None
    routing_widgets: Dict[str, Any] = {}

    if modality == "llm":
        vision_switch = ui.switch("Vision support", value=existing.get("VISION_ENABLED", "").lower() == "true")
        ctx_input = ui.input(
            "Forced context size (blank = auto)", value=existing.get("FORCED_CONTEXT_SIZE", "")
        ).classes("w-full")

        with ui.expansion("Smart router metadata (optional)", icon="route").classes("w-full"):
            routing_widgets["description"] = ui.input(
                "Routing description (keywords)", value=existing.get("ROUTING_DESCRIPTION", "")
            ).classes("w-full")
            with ui.row().classes("w-full gap-2"):
                routing_widgets["cost"] = ui.number(
                    "Cost / 1k tokens", value=float(existing.get("ROUTING_COST", "0.0") or 0.0), step=0.001
                ).classes("flex-1")
                routing_widgets["latency"] = ui.number(
                    "Avg latency (ms)", value=int(existing.get("ROUTING_LATENCY", "100") or 100)
                ).classes("flex-1")
                routing_widgets["complexity"] = ui.select(
                    ["1", "2", "3"], value=existing.get("ROUTING_COMPLEXITY", "1") or "1", label="Complexity"
                ).classes("flex-1")

    def read():
        routing = {}
        if routing_widgets:
            routing = {
                "description": routing_widgets["description"].value,
                "cost": routing_widgets["cost"].value,
                "latency": routing_widgets["latency"].value,
                "complexity": routing_widgets["complexity"].value,
            }
        return dict(
            binding_alias=binding_alias_select.value,
            model_name=model_input.value,
            is_default=is_default_switch.value,
            vision_enabled=vision_switch.value if vision_switch else False,
            forced_context_size=ctx_input.value if ctx_input else "",
            routing=routing,
        )

    return read


def _open_add_profile_dialog(env: EnvStore, modality: str, refresh) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[520px]"):
        ui.label(f"Add {MODALITY_LABELS[modality]} profile").classes("text-lg font-bold")
        alias_input = ui.input("Alias", value="MASTER").classes("w-full")
        reader = _profile_form_body(env, modality)

        def do_save():
            if not reader:
                dialog.close()
                return
            if not alias_input.value.strip():
                ui.notify("Alias is required.", type="warning")
                return
            values = reader()
            env.save_profile(modality, alias_input.value, **values)
            ui.notify(f"Profile '{alias_input.value.upper()}' saved.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            if reader:
                ui.button("Save", on_click=do_save).props("color=primary")

    dialog.open()


def _open_edit_profile_dialog(env: EnvStore, modality: str, alias: str, refresh) -> None:
    dialog = ui.dialog()
    existing = env.profile_keys(modality, alias)
    with dialog, ui.card().classes("w-[520px]"):
        ui.label(f"Edit profile: {alias}").classes("text-lg font-bold mb-2")
        reader = _profile_form_body(env, modality, existing=existing)

        def do_save():
            if not reader:
                dialog.close()
                return
            values = reader()
            env.save_profile(modality, alias, **values)
            ui.notify("Profile updated.", type="positive")
            dialog.close()
            refresh()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            if reader:
                ui.button("Save", on_click=do_save).props("color=primary")

    dialog.open()


# ============================== Agent / Paths / Appearance (GUI-only prefs) ==============================

def _build_agent_panel(prefs: GuiPrefs):
    with ui.card().classes("w-full max-w-2xl"):
        temperature_slider = ui.slider(min=0.0, max=1.5, step=0.05, value=prefs.temperature).props("label-always")
        ui.label().bind_text_from(temperature_slider, "value", lambda v: f"Temperature: {v:.2f}")

        with ui.row().classes("w-full gap-4"):
            max_tokens_input = ui.number(
                "Max tokens / turn", value=prefs.max_tokens_per_turn, min=256, step=256
            ).classes("flex-1")
            max_steps_input = ui.number(
                "Max reasoning steps", value=prefs.max_reasoning_steps, min=1, step=1
            ).classes("flex-1")

        ui.separator()
        shell_switch = ui.switch("Enable shell execution", value=prefs.enable_shell_execution)
        autonomy_select = ui.select(
            SHELL_AUTONOMY_LEVELS, value=prefs.shell_autonomy_level, label="Shell autonomy level"
        ).classes("w-full").bind_visibility_from(shell_switch, "value")

        ui.separator()
        subagents_switch = ui.switch("Enable sub-agent delegation", value=prefs.enable_sub_agents)
        with ui.row().classes("w-full gap-4").bind_visibility_from(subagents_switch, "value"):
            max_depth_input = ui.number(
                "Max sub-agent depth", value=prefs.max_sub_agent_depth, min=1, step=1
            ).classes("flex-1")
            max_per_turn_input = ui.number(
                "Max sub-agents / turn", value=prefs.max_sub_agents_per_turn, min=1, step=1
            ).classes("flex-1")

        model_switch_switch = ui.switch("Allow model switching mid-task", value=prefs.enable_model_switching)

        ui.separator()
        memory_switch = ui.switch("Enable persistent memory", value=prefs.enable_memory)
        skill_create_switch = ui.switch("Enable skill creation", value=prefs.enable_skill_creation)
        skill_load_switch = ui.switch("Enable skill loading", value=prefs.enable_skill_loading)
        skills_mode_select = ui.select(SKILLS_MODES, value=prefs.skills_mode, label="Skills mode").classes("w-full")

    def apply(p: GuiPrefs):
        p.temperature = float(temperature_slider.value)
        p.max_tokens_per_turn = int(max_tokens_input.value)
        p.max_reasoning_steps = int(max_steps_input.value)
        p.enable_shell_execution = shell_switch.value
        p.shell_autonomy_level = autonomy_select.value
        p.enable_sub_agents = subagents_switch.value
        p.max_sub_agent_depth = int(max_depth_input.value)
        p.max_sub_agents_per_turn = int(max_per_turn_input.value)
        p.enable_model_switching = model_switch_switch.value
        p.enable_memory = memory_switch.value
        p.enable_skill_creation = skill_create_switch.value
        p.enable_skill_loading = skill_load_switch.value
        p.skills_mode = skills_mode_select.value

    return apply


def _build_paths_panel(prefs: GuiPrefs):
    with ui.card().classes("w-full max-w-2xl"):
        def pick_folder(target_input: ui.input):
            try:
                import webview
                result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                if result:
                    target_input.value = result[0]
            except Exception:
                ui.notify("Native folder picker unavailable — type the path manually.", type="warning")

        workspace_input = ui.input("Workspace path", value=prefs.workspace_path).classes("w-full")
        ui.button("Browse…", icon="folder_open", on_click=lambda: pick_folder(workspace_input)).props("flat")

        skills_dir_input = ui.input("Skills directory", value=prefs.skills_dir).classes("w-full")
        handbag_input = ui.input("Handbag path", value=prefs.handbag_path).classes("w-full")
        memory_db_input = ui.input("Memory DB URL", value=prefs.memory_db).classes("w-full")

    def apply(p: GuiPrefs):
        p.workspace_path = workspace_input.value
        p.skills_dir = skills_dir_input.value
        p.handbag_path = handbag_input.value
        p.memory_db = memory_db_input.value

    return apply


def _build_appearance_panel(prefs: GuiPrefs):
    with ui.card().classes("w-full max-w-2xl"):
        dark_switch = ui.switch("Dark mode", value=prefs.dark_mode)

        ui.label("Accent color").classes("text-sm text-gray-500 mt-2")
        accent_select = ui.select(
            list(ACCENT_PRESETS.keys()),
            value=next((k for k, v in ACCENT_PRESETS.items() if v == prefs.accent_color), "LoLLMS Blue"),
        ).classes("w-full")
        accent_custom = ui.color_input("Custom accent (overrides preset)", value=prefs.accent_color)

        font_select = ui.select(
            ["JetBrains Mono, monospace", "Fira Code, monospace", "Cascadia Code, monospace", "system-ui, sans-serif"],
            value=prefs.font_family, label="Font",
        ).classes("w-full")

        with ui.row().classes("w-full gap-4"):
            win_w_input = ui.number("Window width", value=prefs.window_width, min=800, step=20).classes("flex-1")
            win_h_input = ui.number("Window height", value=prefs.window_height, min=600, step=20).classes("flex-1")
        ui.label("Window size takes effect on next launch.").classes("text-xs text-gray-500")

        ui.separator()
        show_tools_switch = ui.switch("Show tool call panels", value=prefs.show_tool_calls)
        show_ws_switch = ui.switch("Show workspace changes", value=prefs.show_workspace_changes)
        show_skills_switch = ui.switch("Show skills activity", value=prefs.show_skills_activity)

    def apply(p: GuiPrefs):
        p.dark_mode = dark_switch.value
        p.accent_color = accent_custom.value or ACCENT_PRESETS.get(accent_select.value, p.accent_color)
        p.font_family = font_select.value
        p.window_width = int(win_w_input.value)
        p.window_height = int(win_h_input.value)
        p.show_tool_calls = show_tools_switch.value
        p.show_workspace_changes = show_ws_switch.value
        p.show_skills_activity = show_skills_switch.value

    return apply
