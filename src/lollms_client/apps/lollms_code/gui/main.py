"""
lollms_code GUI — NiceGUI native-window front end for the lollms_code agent.
Starts full-screen at the Projects Deck (Workspace Hub) for project selection and CRUD.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

# Guarantee this file's own folder is importable regardless of launch working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure persistent GUI diagnostic logger
LOG_DIR = Path.home() / ".lollms_client" / "lollms_code"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GUI_LOG_FILE = LOG_DIR / "gui.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(GUI_LOG_FILE), encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("lollms_code.gui")
logger.info("Starting lollms_code GUI session...")

import pipmaster as pm
pm.ensure_packages(["nicegui", "pywebview"])
from nicegui import ui, app as nicegui_app
from ascii_colors import ASCIIColors, trace_exception

from gui_prefs import GuiPrefs
from env_config import EnvStore
try:
    from deck_page import build_deck_page
    from chat_page import build_chat_page
    from settings_page import build_settings_page
except ImportError:
    from lollms_client.apps.lollms_code.gui.deck_page import build_deck_page
    from lollms_client.apps.lollms_code.gui.chat_page import build_chat_page
    from lollms_client.apps.lollms_code.gui.settings_page import build_settings_page

state = {"env": EnvStore(), "prefs": GuiPrefs.load()}

# Register global application exception listener to prevent silent drops
@nicegui_app.on_exception
def handle_app_exception(exception: Exception):
    err_msg = f"Unhandled GUI Exception: {exception}\n{traceback.format_exc()}"
    logger.error(err_msg)
    ASCIIColors.error(f"[GUI Error] {exception}")
    try:
        ui.notify(f"Application Error: {exception}", type="negative", timeout=8000)
    except Exception:
        pass

# ── CLIENT-SIDE HEAD INJECTIONS (ROBUST THEME & NOTIFICATION INTERCEPTOR) ──
CLIENT_HEAD_HTML = """
<style>
/* True Dark Theme Palette and High-Contrast Overrides */
html.dark, body.dark, body.body--dark {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

body.body--dark .q-card,
html.dark .q-card,
body.dark .q-card {
    background-color: #0f172a !important;
    border-color: #1e293b !important;
    color: #f1f5f9 !important;
}

body.body--dark .q-expansion-item,
html.dark .q-expansion-item,
body.body--dark .q-expansion-item__container,
html.dark .q-expansion-item__container {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}

body.body--dark .q-expansion-item__content,
html.dark .q-expansion-item__content {
    background-color: #0b0f19 !important;
    color: #f1f5f9 !important;
}

body.body--dark pre,
html.dark pre,
body.body--dark code,
html.dark code {
    background-color: #020617 !important;
    color: #f1f5f9 !important;
    border-color: #1e293b !important;
}

body.body--dark .q-scrollarea__content {
    color: #f1f5f9;
}

body.body--dark .q-field__control,
html.dark .q-field__control {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}

body.body--dark .q-field__native,
body.body--dark .q-field__input,
html.dark .q-field__native,
html.dark .q-field__input {
    color: #f1f5f9 !important;
}

body.body--dark .q-field--outlined .q-field__control:before,
html.dark .q-field--outlined .q-field__control:before {
    border-color: #334155 !important;
}

body.body--dark ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
body.body--dark ::-webkit-scrollbar-track {
    background: #0f172a;
}
body.body--dark ::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
body.body--dark ::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
</style>
<script>
// Configure Tailwind CDN runtime for class-based dark mode
window.tailwind = window.tailwind || {};
window.tailwind.config = window.tailwind.config || {};
window.tailwind.config.darkMode = 'class';
if (typeof tailwind !== 'undefined' && tailwind.config) {
    tailwind.config.darkMode = 'class';
}

(() => {
    // Global client diagnostic listener
    window.addEventListener('error', (e) => {
        console.error('[GUI Client Error]', e.message, e.filename, e.lineno);
    });

    // 1. Safe non-recursive theme synchronizer
    function syncTailwindDark() {
        try {
            const isDark = (window.Quasar && window.Quasar.Dark) ? window.Quasar.Dark.isActive : true;
            if (isDark) {
                if (!document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.add('dark');
                }
                if (document.body && !document.body.classList.contains('dark')) {
                    document.body.classList.add('dark');
                }
            } else {
                document.documentElement.classList.remove('dark');
                if (document.body) document.body.classList.remove('dark');
            }
        } catch (err) {
            console.warn('[Theme Sync]', err);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            syncTailwindDark();
            setInterval(syncTailwindDark, 500);
        });
    } else {
        syncTailwindDark();
        setInterval(syncTailwindDark, 500);
    }

    // 2. Notification copy-to-clipboard interceptor
    function setupNotifyCopy() {
        try {
            if (!window.Quasar || !window.Quasar.Notify || !window.Quasar.Notify.create) {
                setTimeout(setupNotifyCopy, 100);
                return;
            }
            if (window.Quasar.Notify._copy_intercepted) return;
            window.Quasar.Notify._copy_intercepted = true;

            const origCreate = window.Quasar.Notify.create;
            window.Quasar.Notify.create = function(opts) {
                let options = typeof opts === 'string' ? { message: opts } : Object.assign({}, opts);
                if (!options.actions) {
                    options.actions = [];
                }
                const hasCopy = options.actions.some(a => a.icon === 'content_copy');
                if (!hasCopy) {
                    options.actions.push({
                        icon: 'content_copy',
                        color: 'white',
                        dense: true,
                        round: true,
                        attrs: { title: 'Copy notification to clipboard' },
                        handler: () => {
                            const text = options.message || '';
                            if (navigator.clipboard && navigator.clipboard.writeText) {
                                navigator.clipboard.writeText(text);
                            }
                        }
                    });
                }
                return origCreate.call(this, options);
            };
        } catch (e) {
            console.warn('[Notify Interceptor]', e);
        }
    }
    setupNotifyCopy();
})();
</script>
"""

ui.add_head_html(CLIENT_HEAD_HTML, shared=True)


def apply_theme(prefs: GuiPrefs):
    is_dark = prefs.is_dark()
    prefs.dark_mode = is_dark
    ui.dark_mode(is_dark)
    ui.colors(primary=prefs.accent_color)
    ui.query("body").style(f"font-family: {prefs.font_family}")


def toggle_dark(prefs: GuiPrefs):
    modes = ["auto", "light", "dark"]
    curr = getattr(prefs, "theme_mode", "auto")
    next_mode = modes[(modes.index(curr) + 1) % len(modes)] if curr in modes else "dark"
    prefs.theme_mode = next_mode
    apply_theme(prefs)
    prefs.save()
    ui.notify(f"Theme: {next_mode.capitalize()} ({'Dark' if prefs.is_dark() else 'Light'})", type="info", timeout=1200)


def toggle_fullscreen():
    """Toggles native pywebview fullscreen or browser HTML5 fullscreen."""
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


def exit_app():
    """Cleanly terminates native pywebview window and NiceGUI server."""
    try:
        import webview
        if webview.windows and len(webview.windows) > 0:
            webview.windows[0].destroy()
    except Exception:
        pass
    nicegui_app.shutdown()


def confirm_exit_dialog():
    """Modal confirmation dialog to prevent accidental app closure."""
    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes(
        "w-[420px] max-w-[95vw] p-5 gap-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 "
        "rounded-xl shadow-2xl border border-red-300 dark:border-red-900/60"
    ):
        with ui.row().classes("items-center gap-2.5"):
            ui.icon("power_settings_new", size="26px").classes("text-red-500")
            ui.label("Exit lollms_code?").classes("text-base font-bold")
        ui.label("Are you sure you want to exit and close the application?").classes(
            "text-xs text-slate-500 dark:text-slate-400"
        )
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense no-caps")
            ui.button("Exit Application", icon="logout", on_click=exit_app).props("unelevated dense color=red no-caps")
    dialog.open()


def main():
    p = state["prefs"]
    # Configure native window for edge-to-edge full-screen startup
    if getattr(p, "start_fullscreen", True):
        try:
            nicegui_app.native.window_args["fullscreen"] = True
        except Exception:
            pass

    ui.run(
        title="lollms_code",
        native=True,
        window_size=(p.window_width, p.window_height),
        fullscreen=getattr(p, "start_fullscreen", False),
        reload=False,
        dark=p.is_dark(),
    )


# ── ROUTE 1: PROJECTS DECK (DEFAULT HOME) ────────────────────────────────────

@ui.page("/")
def deck_page_route():
    try:
        env = state["env"]
        env.load()
        prefs = state["prefs"]
        apply_theme(prefs)

        if not env.is_configured():
            def on_first_save(e: EnvStore, pr: GuiPrefs):
                e.save()
                state["env"].load()
                apply_theme(pr)
                ui.navigate.to("/")
            with ui.column().classes("w-full h-screen max-h-screen p-0 m-0 gap-0 flex flex-col overflow-hidden bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100"):
                build_settings_page(
                    env, prefs,
                    on_saved=on_first_save,
                )
            return

        ui.query(".nicegui-content").classes("h-full max-h-full p-0 gap-0 flex flex-col overflow-hidden")

        # Global keyboard shortcuts: Ctrl+, opens Settings; F11 toggles fullscreen
        ui.keyboard(
            on_key=lambda e: ui.navigate.to("/settings")
            if e.action.keydown and e.modifiers.ctrl and e.key == ","
            else toggle_fullscreen() if e.action.keydown and e.key == "F11" else None,
            ignore=[]
        )

        with ui.column().classes("w-full h-screen max-h-screen p-0 m-0 gap-0 flex flex-col overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"):
            build_deck_page(env, prefs)

    except Exception as e:
        err_msg = f"Crash rendering Projects Deck: {e}\n{traceback.format_exc()}"
        logger.error(err_msg)
        ASCIIColors.error(f"[Deck Render Crash] {e}")
        with ui.column().classes("w-full h-screen p-8 bg-slate-950 text-white items-center justify-center gap-4"):
            ui.icon("error", size="48px").classes("text-red-500")
            ui.label("Error Rendering Projects Deck").classes("text-xl font-bold text-red-400")
            ui.code(traceback.format_exc()).classes("w-full max-w-4xl p-4 bg-slate-900 rounded text-xs text-red-200 overflow-auto max-h-96")
            with ui.row().classes("gap-2"):
                ui.button("Retry", icon="refresh", on_click=lambda: ui.navigate.to("/")).props("unelevated color=primary")
                ui.button("Go to Chat", icon="chat", on_click=lambda: ui.navigate.to("/chat")).props("outline")
                ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props("outline")


# ── ROUTE 2: ACTIVE WORKSPACE CHAT & TREE ───────────────────────────────────

@ui.page("/chat")
def chat_page_route():
    try:
        env = state["env"]
        env.load()
        prefs = state["prefs"]
        apply_theme(prefs)

        if not env.is_configured():
            ui.navigate.to("/")
            return

        ui.query(".nicegui-content").classes("h-full max-h-full p-0 gap-0 flex flex-col overflow-hidden")
        ui.keyboard(
            on_key=lambda e: ui.navigate.to("/settings")
            if e.action.keydown and e.modifiers.ctrl and e.key == ","
            else toggle_fullscreen() if e.action.keydown and e.key == "F11" else None,
            ignore=[]
        )

        HEADER_H = 40
        resolved = env.resolve_default_connection("llm") or {}

        with ui.column().classes("w-full h-screen max-h-screen p-0 m-0 gap-0 flex flex-col overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100"):
            # Header strip
            with ui.row().classes(
                "w-full items-center justify-between px-3 flex-nowrap bg-primary text-white shrink-0 shadow-sm"
            ).style(f"min-height: {HEADER_H}px; height: {HEADER_H}px;"):
                with ui.row().classes("items-center gap-2 flex-nowrap overflow-hidden"):
                    ui.button(
                        "Workspaces", icon="view_carousel",
                        on_click=lambda: ui.navigate.to("/"),
                    ).props("flat dense size=sm no-caps").tooltip("Back to Projects Deck")
                    ui.label("·").classes("text-xs opacity-40 shrink-0")
                    ui.icon("terminal", size="18px")
                    ui.label("lollms_code").classes("text-sm font-bold shrink-0")
                    ui.label("·").classes("text-xs opacity-40 shrink-0")
                    ui.label(prefs.workspace_path).classes("text-xs opacity-80 truncate").style("max-width: 320px;").tooltip(prefs.workspace_path)
                    ui.label("·").classes("text-xs opacity-40 shrink-0")
                    profile_label = f"{resolved.get('binding_name') or '?'} / {resolved.get('model_name') or '?'}"
                    ui.label(profile_label).classes("text-xs opacity-80 shrink-0")

                with ui.row().classes("items-center gap-1 shrink-0"):
                    ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props(
                        "flat dense size=sm no-caps"
                    )
                    ui.button(
                        icon="fullscreen",
                        on_click=toggle_fullscreen,
                    ).props("flat round dense size=sm").tooltip("Toggle Fullscreen (F11)")
                    ui.button(
                        icon="dark_mode" if not prefs.dark_mode else "light_mode",
                        on_click=lambda: toggle_dark(prefs),
                    ).props("flat round dense size=sm").tooltip("Toggle Theme")
                    ui.button(
                        icon="close",
                        on_click=confirm_exit_dialog,
                    ).props("flat round dense size=sm color=red text-color=white").tooltip("Exit Application")

            with ui.element("div").classes("w-full flex-1 min-h-0 flex flex-col overflow-hidden"):
                build_chat_page(env, prefs)

    except Exception as e:
        err_msg = f"Crash rendering Chat Page: {e}\n{traceback.format_exc()}"
        logger.error(err_msg)
        ASCIIColors.error(f"[Chat Render Crash] {e}")
        with ui.column().classes("w-full h-screen p-8 bg-slate-950 text-white items-center justify-center gap-4"):
            ui.icon("error", size="48px").classes("text-red-500")
            ui.label("Error Rendering Chat Interface").classes("text-xl font-bold text-red-400")
            ui.code(traceback.format_exc()).classes("w-full max-w-4xl p-4 bg-slate-900 rounded text-xs text-red-200 overflow-auto max-h-96")
            with ui.row().classes("gap-2"):
                ui.button("Reload Chat", icon="refresh", on_click=lambda: ui.navigate.to("/chat")).props("unelevated color=primary")
                ui.button("Back to Workspaces", icon="view_carousel", on_click=lambda: ui.navigate.to("/")).props("outline")
                ui.button("Settings", icon="settings", on_click=lambda: ui.navigate.to("/settings")).props("outline")


# ── ROUTE 3: SETTINGS ────────────────────────────────────────────────────────

@ui.page("/settings")
def settings_page_route():
    env = state["env"]
    env.load()
    prefs = state["prefs"]
    apply_theme(prefs)

    def on_saved(e: EnvStore, p: GuiPrefs):
        e.save()
        state["env"].load()
        apply_theme(p)
        ui.notify("Settings saved.", type="positive")
        ui.navigate.to("/chat")

    with ui.column().classes("w-full h-screen max-h-screen p-0 m-0 gap-0 flex flex-col overflow-hidden bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100"):
        build_settings_page(
            env, prefs,
            on_saved=on_saved,
            on_back=lambda: ui.navigate.to("/chat"),
        )


if __name__ in {"__main__", "__mp_main__"}:
    main()