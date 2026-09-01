#!/usr/bin/env python3
"""
lollms_code GUI — NiceGUI native-window front end for the lollms_code agent.

Run:
    pip install nicegui pywebview
    python main.py

Packaging into a standalone executable, once this works against your real
lollms_client install:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name lollms-code-gui main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Guarantee this file's own folder is importable regardless of the working
# directory main.py was launched from (double-click, IDE run config, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nicegui import ui, app as nicegui_app

from gui_prefs import GuiPrefs
from env_config import EnvStore
from settings_page import build_settings_page
from chat_page import build_chat_page

state = {"env": EnvStore(), "prefs": GuiPrefs.load()}


def apply_theme(prefs: GuiPrefs):
    ui.dark_mode(prefs.dark_mode)
    ui.colors(primary=prefs.accent_color)
    ui.query("body").style(f"font-family: {prefs.font_family}")


def main():
    p = state["prefs"]
    ui.run(
        title="lollms_code",
        native=True,
        window_size=(p.window_width, p.window_height),
        reload=False,
        dark=p.dark_mode,
    )


@ui.page("/")
def main_page():
    env = state["env"]
    prefs = state["prefs"]
    apply_theme(prefs)

    if not env.is_configured():
        with ui.column().classes("w-full h-full items-center justify-center p-8"):
            ui.icon("rocket_launch", size="48px").classes("text-primary")
            ui.label("Welcome to lollms_code").classes("text-2xl font-bold mt-2")
            ui.label("Let's set up your model connection before your first session.").classes(
                "text-sm text-gray-500 mb-4"
            )
            with ui.card().classes("w-full max-w-3xl"):
                def on_first_save(e: EnvStore, p: GuiPrefs):
                    e.save()
                    apply_theme(p)
                    ui.navigate.to("/")
                build_settings_page(env, prefs, on_saved=on_first_save)
        return

    # Default .nicegui-content has ~1rem padding + gap and no explicit height,
    # so a child "calc(100vh - HEADER)" div ends up sized by its content instead
    # of the viewport, pushing the input row below the fold. Strip that here
    # so the header + chat area exactly fill the native window.
    ui.query(".nicegui-content").classes("h-screen p-0 gap-0")

    # Fast access to Settings: Ctrl+, (also has header + in-chat "Settings"
    # buttons — three ways in, since it's a click users reach for often).
    def _global_shortcut(e):
        if e.action.keydown and e.modifiers.ctrl and e.key == ",":
            ui.navigate.to("/settings")

    ui.keyboard(on_key=_global_shortcut, ignore=[])

    HEADER_H = 40
    resolved = env.resolve_default_connection("llm")

    with ui.header().classes(
        "items-center justify-between px-3 flex-nowrap"
    ).style(f"min-height: {HEADER_H}px; height: {HEADER_H}px;"):
        with ui.row().classes("items-center gap-2 flex-nowrap overflow-hidden"):
            ui.icon("terminal", size="18px")
            ui.label("lollms_code").classes("text-sm font-bold shrink-0")
            ui.label("·").classes("text-xs opacity-40 shrink-0")
            ui.label(prefs.workspace_path).classes("text-xs opacity-70 truncate").style("max-width: 320px;")
            ui.label("·").classes("text-xs opacity-40 shrink-0")
            ui.label(f"{resolved.get('binding_name') or '?'} / {resolved.get('model_name') or '?'}").classes(
                "text-xs opacity-70 shrink-0"
            )
        with ui.row().classes("items-center gap-1 shrink-0"):
            ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
                "flat dense size=sm no-caps"
            )
            ui.button(
                icon="dark_mode" if not prefs.dark_mode else "light_mode",
                on_click=lambda: toggle_dark(prefs),
            ).props("flat round dense size=sm")

    with ui.element("div").classes("w-full").style(f"height: calc(100vh - {HEADER_H}px); overflow: hidden;"):
        build_chat_page(env, prefs)


def toggle_dark(prefs: GuiPrefs):
    prefs.dark_mode = not prefs.dark_mode
    prefs.save()
    ui.navigate.to("/")


@ui.page("/settings")
def settings_page_route():
    env = state["env"]
    prefs = state["prefs"]
    apply_theme(prefs)

    with ui.row().classes("items-center gap-2 mb-4"):
        ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props("flat round")
        ui.label("Back to chat").classes("text-sm text-gray-500")

    def on_saved(e: EnvStore, p: GuiPrefs):
        # Persist bindings/profiles to ~/.lollms-client/.env, same as the wizard.
        e.save()
        apply_theme(p)
        ui.notify("Applied. Returning to chat…", type="positive")
        ui.navigate.to("/")

    build_settings_page(env, prefs, on_saved=on_saved)


if __name__ in {"__main__", "__mp_main__"}:
    main()
