"""
deck_page.py — Projects Deck & Workspace Hub for lollms_code.
Displays registered project workspaces as an interactive card deck with
file statistics, Git branch detection, and full CRUD management.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from nicegui import ui

try:
    from gui_prefs import GuiPrefs
    from env_config import EnvStore
except ImportError:
    from lollms_client.apps.lollms_code.gui.gui_prefs import GuiPrefs
    from lollms_client.apps.lollms_code.gui.env_config import EnvStore

__all__ = ["build_deck_page"]


import sys
import importlib.util

try:
    from folder_picker import pick_folder
except ImportError:
    try:
        from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder
    except ImportError:
        pick_folder = None


def get_pick_folder():
    """Self-healing folder picker resolver with cache eviction."""
    for mod_name in ("folder_picker", "lollms_client.apps.lollms_code.gui.folder_picker"):
        if mod_name in sys.modules and not hasattr(sys.modules[mod_name], "pick_folder"):
            sys.modules.pop(mod_name, None)

    try:
        from folder_picker import pick_folder
        return pick_folder
    except Exception:
        pass

    try:
        from lollms_client.apps.lollms_code.gui.folder_picker import pick_folder
        return pick_folder
    except Exception:
        pass

    try:
        fp = Path(__file__).resolve().parent / "folder_picker.py"
        if fp.exists():
            spec = importlib.util.spec_from_file_location("folder_picker_direct", str(fp))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "pick_folder"):
                    return mod.pick_folder
    except Exception:
        pass

    return None

def _get_workspace_meta(path_str: str) -> Dict[str, Any]:
    """Inspects a workspace folder for Git branch, file count, and sandbox directory with safety caps."""
    meta = {
        "exists": False,
        "git_branch": None,
        "file_count": 0,
        "total_size": 0,
        "has_sandbox": False,
    }
    if not path_str:
        return meta

    try:
        p = Path(path_str).resolve()
        if not (p.exists() and p.is_dir()):
            return meta

        meta["exists"] = True
        sandbox_dir = p / ".lollms_code"
        meta["has_sandbox"] = sandbox_dir.exists() and sandbox_dir.is_dir()

        git_dir = p / ".git"
        if git_dir.exists():
            try:
                res = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=str(p), capture_output=True, text=True,
                    timeout=1.0, encoding="utf-8", errors="ignore"
                )
                if res.returncode == 0 and res.stdout.strip():
                    meta["git_branch"] = res.stdout.strip()
            except Exception:
                pass

        try:
            count = 0
            size = 0
            for item in p.iterdir():
                if item.name not in ("__pycache__", ".git", ".venv", "venv", "node_modules"):
                    count += 1
                    if item.is_file():
                        try:
                            size += item.stat().st_size
                        except Exception:
                            pass
                if count > 500:
                    break
            meta["file_count"] = count
            meta["total_size"] = size
        except Exception:
            pass

    except Exception:
        pass

    return meta


def _format_size(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def build_deck_page(env: EnvStore, prefs: GuiPrefs) -> None:
    SURFACE = "bg-slate-100/90 dark:bg-slate-900/90"
    CANVAS = "bg-slate-50 dark:bg-slate-950"
    BORDER = "border-slate-200 dark:border-slate-800"
    HEADER_H = 42

    resolved_llm = (env.resolve_default_connection("llm") if hasattr(env, "resolve_default_connection") else {}) or {}
    model_str = f"{resolved_llm.get('binding_name') or '?'}:{resolved_llm.get('model_name') or 'default'}"

    # ---- Top Navigation Bar ----
    with ui.row().classes(
        f"w-full items-center justify-between px-4 flex-nowrap bg-primary text-white shrink-0 shadow-sm"
    ).style(f"min-height: {HEADER_H}px; height: {HEADER_H}px;"):
        with ui.row().classes("items-center gap-2.5"):
            ui.icon("view_carousel", size="20px")
            ui.label("lollms_code").classes("text-sm font-bold tracking-wide")
            ui.label("·").classes("opacity-40 text-xs")
            ui.label("Projects Deck").classes("text-xs font-semibold opacity-90")
            ui.label("·").classes("opacity-40 text-xs")
            ui.badge(model_str, color="indigo").props("dense rounded text-color=white").classes("text-[10px] font-mono")

        with ui.row().classes("items-center gap-2"):
            ui.button("New Project", icon="add", on_click=lambda: open_add_workspace_dialog()).props(
                "unelevated dense size=sm color=white text-color=primary no-caps font-bold"
            ).classes("px-3 shadow-sm")
            ui.button(
                "Zoo Hub", icon="pets",
                on_click=lambda: _open_deck_zoo_dialog(),
            ).props("flat dense size=sm no-caps").tooltip("Open Zoo Package Hub: browse and install community tools, skills, and personas")
            ui.button(
                "Chat Active", icon="chat",
                on_click=lambda: ui.navigate.to("/chat"),
            ).props("flat dense size=sm no-caps").tooltip("Go to currently open workspace chat")
            ui.button(
                "Settings", icon="settings",
                on_click=lambda: ui.navigate.to("/settings"),
            ).props("flat dense size=sm no-caps")
            ui.button(
                icon="fullscreen",
                on_click=lambda: _toggle_fs(),
            ).props("flat round dense size=sm").tooltip("Toggle Fullscreen (F11)")
            ui.button(
                icon="dark_mode" if not prefs.dark_mode else "light_mode",
                on_click=lambda: toggle_theme(),
            ).props("flat round dense size=sm").tooltip("Toggle Theme")
            ui.button(
                icon="close",
                on_click=lambda: _confirm_exit(),
            ).props("flat round dense size=sm color=red text-color=white").tooltip("Exit Application")

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

    def _confirm_exit():
        from main import confirm_exit_dialog
        confirm_exit_dialog()

    def _open_deck_zoo_dialog():
        from lollms_client.apps.lollms_code.zoo import ZooManager
        zm = ZooManager(prefs.workspace_path)
        d = ui.dialog().props("maximized")
        with d, ui.card().classes("w-full h-full flex flex-col p-4 bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 gap-3"):
            with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("pets", size="26px").classes("text-amber-500")
                    ui.label("LoLLMS Zoo Package Hub").classes("text-base font-bold")
                ui.button("Close", icon="close", on_click=d.close).props("flat dense round size=sm")

            with ui.scroll_area().classes("w-full flex-1 p-2"):
                ui.label("Manage official LoLLMS community packages from GitHub.").classes("text-xs text-slate-500 mb-2")
                with ui.row().classes("w-full gap-2 mb-3"):
                    def _sync_all():
                        ui.notify("Syncing zoos from GitHub...", type="info")
                        zm.sync_all()
                        ui.notify("Zoos synced!", type="positive")
                    ui.button("Sync All Repositories", icon="sync", on_click=_sync_all).props("unelevated color=primary size=sm no-caps")

                for z in ("tools", "skills", "personalities"):
                    with ui.expansion(f"{z.upper()} ZOO", icon="inventory_2").classes("w-full bg-slate-100 dark:bg-slate-900 rounded mb-2"):
                        items = zm.list_items(z)
                        with ui.column().classes("w-full gap-1 p-2"):
                            if not items:
                                ui.label("No items or repo not synced yet. Click 'Sync All Repositories' above.").classes("text-xs text-slate-400 italic")
                            for it in items[:25]:
                                with ui.row().classes("w-full items-center justify-between text-xs py-1 border-b border-slate-200 dark:border-slate-800"):
                                    ui.label(f"{it.category}/{it.name}").classes("font-semibold truncate max-w-sm")
                                    with ui.row().classes("gap-1"):
                                        def _inst_g(target=it):
                                            ok, msg = zm.install_item(target, scope="global")
                                            ui.notify(msg, type="positive" if ok else "negative")
                                        def _inst_p(target=it):
                                            ok, msg = zm.install_item(target, scope="project")
                                            ui.notify(msg, type="positive" if ok else "negative")
                                        ui.button("+ Project", on_click=lambda t=it: _inst_p(t)).props("unelevated dense size=xs color=primary no-caps")
                                        ui.button("+ Global", on_click=lambda t=it: _inst_g(t)).props("outline dense size=xs color=primary no-caps")
        d.open()

    def toggle_theme():
        prefs.dark_mode = not prefs.dark_mode
        ui.dark_mode(prefs.dark_mode)
        prefs.save()

    # ---- Deck Body ----
    with ui.column().classes(f"w-full flex-1 min-h-0 p-6 overflow-y-auto {CANVAS} gap-5"):
        # Top banner & search
        with ui.row().classes("w-full max-w-6xl mx-auto items-center justify-between flex-wrap gap-3 pb-2 border-b " + BORDER):
            with ui.column().classes("gap-0"):
                ui.label("Select a Workspace").classes("text-xl font-bold tracking-tight text-slate-900 dark:text-slate-100")
                ui.label("Pick a project to start coding, or register a new workspace directory.").classes(
                    "text-xs text-slate-500 dark:text-slate-400"
                )

            with ui.row().classes("items-center gap-3"):
                search_input = ui.input(placeholder="Filter projects…").props(
                    ':dark="Quasar.Dark.isActive" outlined dense clearable input-debounce=100'
                ).classes("w-64 text-xs bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100")

        # Container for project cards
        cards_slot = ui.row().classes("w-full max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5")

        def refresh_cards():
            cards_slot.clear()
            filter_query = (search_input.value or "").lower().strip()
            workspaces = prefs.get_workspaces()

            filtered = [
                ws for ws in workspaces
                if not filter_query
                or filter_query in ws.get("name", "").lower()
                or filter_query in ws.get("path", "").lower()
                or filter_query in ws.get("description", "").lower()
            ]

            with cards_slot:
                if not filtered:
                    with ui.card().classes(
                        f"col-span-full w-full p-12 border {BORDER} bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 "
                        "rounded-xl items-center justify-center text-center gap-3"
                    ).props(':dark="Quasar.Dark.isActive"'):
                        ui.icon("folder_off", size="42px").classes("text-slate-400")
                        ui.label("No matching workspaces found").classes("font-bold text-sm")
                        ui.button("Register Workspace", icon="add", on_click=lambda: open_add_workspace_dialog()).props(
                            "unelevated size=sm color=primary no-caps"
                        )
                    return

                for ws in filtered:
                    _render_workspace_card(ws, prefs, refresh_cards)

        search_input.on_value_change(lambda _: refresh_cards())
        refresh_cards()

    # ---- Add Workspace Dialog ----
    add_dialog = ui.dialog().props("persistent")
    with add_dialog, ui.card().classes(
        f"w-[540px] max-w-[95vw] p-5 gap-3 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 "
        f"border {BORDER} rounded-xl shadow-2xl"
    ).props(':dark="Quasar.Dark.isActive"'):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b " + BORDER):
            with ui.row().classes("items-center gap-2"):
                ui.icon("create_new_folder", size="24px").classes("text-primary")
                ui.label("Register Workspace").classes("text-base font-bold")
            ui.button(icon="close", on_click=add_dialog.close).props("flat round dense size=xs")

        ws_name_input = ui.input("Project / Workspace Name", placeholder="e.g. My API Service").classes("w-full").props("outlined dense")
        with ui.row().classes("w-full items-center gap-2"):
            ws_path_input = ui.input("Directory Path", value=str(Path.cwd())).classes("flex-1").props("outlined dense")

            async def pick_dir():
                _picker = pick_folder or get_pick_folder()
                if _picker:
                    chosen = await _picker(title="Select Workspace Folder", initial_dir=ws_path_input.value)
                    if chosen:
                        ws_path_input.value = chosen
                        if not ws_name_input.value:
                            ws_name_input.value = Path(chosen).name
                else:
                    ui.notify("Folder picker unavailable — enter path manually.", type="warning")

            ui.button("Browse…", icon="folder_open", on_click=pick_dir).props("outline dense no-caps")

        ws_desc_input = ui.textarea("Description / Notes (optional)", placeholder="Purpose of this codebase…").classes("w-full").props("outlined autogrow dense rows=2")
        auto_open_check = ui.checkbox("Open workspace immediately after registration", value=True).props("dense")

        with ui.row().classes("w-full justify-end gap-2 pt-2 border-t " + BORDER):
            ui.button("Cancel", on_click=add_dialog.close).props("flat dense no-caps")

            def do_save():
                path_val = (ws_path_input.value or "").strip()
                if not path_val:
                    ui.notify("Directory path is required.", type="warning")
                    return
                p = Path(path_val).resolve()
                if not p.exists():
                    p.mkdir(parents=True, exist_ok=True)

                name_val = (ws_name_input.value or "").strip() or p.name or "Workspace"
                entry = prefs.add_workspace(name_val, str(p), (ws_desc_input.value or "").strip())
                ui.notify(f"Registered workspace '{entry['name']}'.", type="positive")
                add_dialog.close()

                if auto_open_check.value:
                    _open_workspace(entry, prefs)
                else:
                    refresh_cards()

            ui.button("Add Workspace", icon="check", on_click=do_save).props("unelevated color=primary dense no-caps")

    def open_add_workspace_dialog():
        ws_name_input.value = ""
        ws_path_input.value = str(Path.cwd().resolve())
        ws_desc_input.value = ""
        add_dialog.open()


def _render_workspace_card(ws: Dict[str, Any], prefs: GuiPrefs, on_change) -> None:
    BORDER = "border-slate-200 dark:border-slate-800"
    path_str = ws.get("path", "")
    meta = _get_workspace_meta(path_str)
    is_active = (Path(prefs.workspace_path).resolve() == Path(path_str).resolve())

    border_highlight = "border-primary shadow-md ring-2 ring-primary/20" if is_active else f"border {BORDER} shadow-sm"

    with ui.card().classes(
        f"w-full p-4 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl flex flex-col justify-between gap-3 "
        f"transition-all hover:shadow-md {border_highlight}"
    ).props(':dark="Quasar.Dark.isActive"'):
        # Top line: Icon, Name, Active Badge
        with ui.column().classes("w-full gap-1.5"):
            with ui.row().classes("w-full items-start justify-between flex-nowrap"):
                with ui.row().classes("items-center gap-2 min-w-0 flex-1"):
                    ui.icon("folder", size="22px").classes("text-amber-500 shrink-0")
                    ui.label(ws.get("name", "Workspace")).classes(
                        "text-sm font-bold truncate text-slate-900 dark:text-slate-100"
                    ).tooltip(ws.get("name"))

                with ui.row().classes("items-center gap-1 shrink-0"):
                    if is_active:
                        ui.badge("ACTIVE", color="primary").props("rounded dense").classes("text-[10px] font-bold")
                    if meta.get("git_branch"):
                        ui.badge(f"🌿 {meta['git_branch']}", color="teal").props("rounded dense").classes(
                            "text-[10px] font-mono"
                        )

            # Path line
            with ui.row().classes("w-full items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400 font-mono"):
                ui.label(path_str).classes("truncate flex-1 text-[11px]").tooltip(path_str)
                ui.button(
                    icon="content_copy",
                    on_click=lambda p=path_str: (ui.clipboard.write(p), ui.notify("Path copied.", type="positive")),
                ).props("flat round dense size=xs color=grey").tooltip("Copy path")

            if ws.get("description"):
                ui.label(ws["description"]).classes("text-xs text-slate-600 dark:text-slate-300 italic line-clamp-2 mt-0.5")

            # Metrics
            with ui.row().classes("w-full items-center gap-2 pt-1 text-[11px] text-slate-500 font-mono"):
                if meta["exists"]:
                    ui.label(f"{meta['file_count']} items")
                    if meta["total_size"] > 0:
                        ui.label("·")
                        ui.label(_format_size(meta["total_size"]))
                    if meta["has_sandbox"]:
                        ui.label("·")
                        ui.label("sandbox ready").classes("text-emerald-500 font-semibold")
                else:
                    ui.label("⚠️ Path not found on disk").classes("text-red-500 font-bold")

        # Bottom Actions
        with ui.row().classes("w-full items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-800"):
            with ui.row().classes("items-center gap-1"):
                ui.button(
                    icon="edit",
                    on_click=lambda: _open_edit_dialog(ws, prefs, on_change),
                ).props("flat round dense size=xs color=grey").tooltip("Edit project details")

                ui.button(
                    icon="delete",
                    on_click=lambda: _confirm_delete_dialog(ws, prefs, on_change),
                ).props("flat round dense size=xs color=red").tooltip("Remove from deck")

            ui.button(
                "Open Workspace", icon="play_arrow",
                on_click=lambda w=ws: _open_workspace(w, prefs),
            ).props("unelevated size=sm color=primary no-caps font-semibold").classes("px-3 shadow-sm")


def _open_workspace(ws: Dict[str, Any], prefs: GuiPrefs) -> None:
    p = str(Path(ws["path"]).resolve())
    if not Path(p).exists():
        ui.notify(f"Directory does not exist: {p}", type="negative")
        return

    prefs.workspace_path = p
    ws["last_opened"] = datetime.now().isoformat()
    prefs.save()
    ui.notify(f"Opening workspace '{ws.get('name')}'...", type="positive")
    ui.navigate.to("/chat")


def _open_edit_dialog(ws: Dict[str, Any], prefs: GuiPrefs, on_change) -> None:
    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-[500px] max-w-[95vw] p-5 gap-3 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl shadow-xl").props(':dark="Quasar.Dark.isActive"'):
        ui.label(f"Edit Workspace: {ws.get('name')}").classes("text-base font-bold")

        name_in = ui.input("Workspace Name", value=ws.get("name", "")).classes("w-full").props("outlined dense")
        with ui.row().classes("w-full items-center gap-2"):
            path_in = ui.input("Folder Path", value=ws.get("path", "")).classes("flex-1").props("outlined dense")

            async def edit_pick_dir():
                _picker = pick_folder or get_pick_folder()
                if _picker:
                    chosen = await _picker(title="Select Workspace Folder", initial_dir=path_in.value)
                    if chosen:
                        path_in.value = chosen
                        if not name_in.value:
                            name_in.value = Path(chosen).name
                else:
                    ui.notify("Folder picker unavailable — enter path manually.", type="warning")

            ui.button("Browse…", icon="folder_open", on_click=edit_pick_dir).props("outline dense no-caps")
        desc_in = ui.textarea("Description", value=ws.get("description", "")).classes("w-full").props("outlined autogrow dense rows=2")

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def do_update():
                prefs.update_workspace(
                    ws["id"],
                    name_in.value or ws["name"],
                    path_in.value or ws["path"],
                    desc_in.value or ""
                )
                dialog.close()
                ui.notify("Workspace updated.", type="positive")
                on_change()

            ui.button("Save", on_click=do_update).props("unelevated color=primary dense")

    dialog.open()


def _confirm_delete_dialog(ws: Dict[str, Any], prefs: GuiPrefs, on_change) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[420px] p-4 gap-3 bg-slate-100 dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl shadow-lg").props(':dark="Quasar.Dark.isActive"'):
        ui.label("Remove Workspace from Deck?").classes("text-base font-bold text-red-500")
        ui.label(f"Workspace: \"{ws.get('name')}\"").classes("text-xs font-mono")
        ui.label("This will remove the project from the deck selector. Files on your disk will NOT be touched.").classes(
            "text-xs text-slate-500 dark:text-slate-400"
        )

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat dense")

            def do_remove():
                prefs.remove_workspace(ws["id"])
                dialog.close()
                ui.notify("Workspace removed from deck.", type="info")
                on_change()

            ui.button("Remove", on_click=do_remove).props("unelevated color=red dense")

    dialog.open()