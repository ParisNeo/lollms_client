"""
folder_picker.py — Resilient multi-tier folder picker for NiceGUI.
Tiers:
  1. Native OS Explorer dialog via Tkinter (non-blocking run.io_bound, zero multiprocessing pickling)
  2. NiceGUI PyWebView Native Window with picklable webview.FileDialog.FOLDER enum
  3. Windows PowerShell FolderBrowserDialog
  4. In-App NiceGUI directory browser modal fallback
"""
from __future__ import annotations

import os
import sys
import subprocess
import asyncio
from pathlib import Path
from typing import Optional
from nicegui import app as nicegui_app, run, ui

__all__ = ["pick_folder", "pick_file"]


async def pick_file(
    title: str = "Select File",
    initial_dir: Optional[str] = None,
    file_types: Optional[List[Tuple[str, str]]] = None,
) -> Optional[str]:
    """
    Opens a native file selection dialog using a 4-tier fallback cascade.
    Always returns the chosen file path as a string, or None if cancelled.
    """
    init_path = str(Path(initial_dir).resolve().parent if initial_dir and Path(initial_dir).is_file() else Path(initial_dir).resolve()) if initial_dir and Path(initial_dir).exists() else None
    default_types = file_types or [
        ("Certificate Files", "*.pem;*.crt;*.cer;*.key"),
        ("All Files", "*.*")
    ]

    # ── Tier 1: Native OS Open File Dialog via Tkinter ──
    def _tk_pick_file() -> Optional[str]:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            tk_filetypes = [(label, ext.replace(";", " ")) for label, ext in default_types]
            selected = filedialog.askopenfilename(
                title=title,
                initialdir=init_path or None,
                filetypes=tk_filetypes,
            )
            root.destroy()
            return str(Path(selected).resolve()) if selected else None
        except Exception:
            return None

    try:
        f_path = await run.io_bound(_tk_pick_file)
        if f_path:
            return f_path
    except Exception:
        pass

    # ── Tier 2: NiceGUI PyWebView Native Window ──
    if getattr(nicegui_app, "native", None) and getattr(nicegui_app.native, "main_window", None):
        try:
            import webview
            open_enum = getattr(webview.FileDialog, "OPEN", 0) if hasattr(webview, "FileDialog") else 0
            filter_specs = tuple(f"{label} ({ext})" for label, ext in default_types)
            result = await nicegui_app.native.main_window.create_file_dialog(
                dialog_type=open_enum,
                directory=init_path or "",
                file_types=filter_specs,
            )
            if result:
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    return str(Path(result[0]).resolve())
                if isinstance(result, str):
                    return str(Path(result).resolve())
        except Exception:
            pass

    # ── Tier 3: Windows PowerShell OpenFileDialog ──
    if sys.platform == "win32":
        def _ps_pick_file() -> Optional[str]:
            try:
                filter_parts = []
                for label, ext in default_types:
                    filter_parts.append(f"{label} ({ext})|{ext}")
                full_filter = "|".join(filter_parts)
                init_script = f"$f.InitialDirectory = '{init_path}'; " if init_path else ""
                safe_title = title.replace("'", "''")
                ps_code = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
                    "$f = New-Object System.Windows.Forms.OpenFileDialog; "
                    f"$f.Title = '{safe_title}'; "
                    f"$f.Filter = '{full_filter}'; "
                    f"{init_script}"
                    "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }"
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_code],
                    capture_output=True, text=True, timeout=60, encoding="utf-8", errors="ignore"
                )
                if res.returncode == 0 and res.stdout.strip():
                    return str(Path(res.stdout.strip()).resolve())
            except Exception:
                pass
            return None

        try:
            f_path = await run.io_bound(_ps_pick_file)
            if f_path:
                return f_path
        except Exception:
            pass

    # ── Tier 4: In-App NiceGUI Directory / File Navigator Modal ──
    return await _open_in_app_file_picker(title, init_path)


async def pick_folder(
    title: str = "Select Folder",
    initial_dir: Optional[str] = None
) -> Optional[str]:
    """
    Opens a native folder selection dialog using a 4-tier fallback cascade.
    Always returns the chosen directory path as a string, or None if cancelled.
    """
    init_path = str(Path(initial_dir).resolve()) if initial_dir and Path(initial_dir).exists() else None

    # ── Tier 1: Native OS Dialog via Tkinter (Zero-Pickling, Native Windows Explorer) ──
    def _tk_pick() -> Optional[str]:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(
                title=title,
                initialdir=init_path or None
            )
            root.destroy()
            return str(Path(selected).resolve()) if selected else None
        except Exception:
            return None

    try:
        folder = await run.io_bound(_tk_pick)
        if folder:
            return folder
    except Exception:
        pass

    # ── Tier 2: NiceGUI PyWebView Native Window (Using Picklable FileDialog.FOLDER Enum) ──
    if getattr(nicegui_app, "native", None) and getattr(nicegui_app.native, "main_window", None):
        try:
            import webview
            folder_enum = None
            if hasattr(webview, "FileDialog") and hasattr(webview.FileDialog, "FOLDER"):
                folder_enum = webview.FileDialog.FOLDER

            if folder_enum is not None:
                kwargs = {"dialog_type": folder_enum}
                if init_path:
                    kwargs["directory"] = init_path

                result = await nicegui_app.native.main_window.create_file_dialog(**kwargs)
                if result:
                    if isinstance(result, (list, tuple)) and len(result) > 0:
                        return str(Path(result[0]).resolve())
                    if isinstance(result, str):
                        return str(Path(result).resolve())
                elif result is None:
                    return None
        except Exception:
            pass

    # ── Tier 3: Windows PowerShell Native Forms Dialog ────────────────────────
    if sys.platform == "win32":
        def _ps_pick() -> Optional[str]:
            try:
                desc = title.replace("'", "''")
                init_script = f"$f.SelectedPath = '{init_path}'; " if init_path else ""
                ps_code = (
                    "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
                    "$f = New-Object System.Windows.Forms.FolderBrowserDialog; "
                    f"$f.Description = '{desc}'; "
                    f"$f.ShowNewFolderButton = $true; "
                    f"{init_script}"
                    "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.SelectedPath }"
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_code],
                    capture_output=True, text=True, timeout=60, encoding="utf-8", errors="ignore"
                )
                if res.returncode == 0 and res.stdout.strip():
                    return str(Path(res.stdout.strip()).resolve())
            except Exception:
                pass
            return None

        try:
            folder = await run.io_bound(_ps_pick)
            if folder:
                return folder
        except Exception:
            pass

    # ── Tier 4: In-App NiceGUI Directory Explorer Modal (Zero-Fail Fallback) ──
    return await _open_in_app_directory_picker(title, init_path)


async def _open_in_app_file_picker(title: str, start_dir: Optional[str]) -> Optional[str]:
    """In-app modal file selector when all OS native dialogs are unavailable."""
    current = Path(start_dir or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    if not current.exists():
        current = Path.cwd().resolve()

    selected_file: Optional[str] = None
    done_event = asyncio.Event()

    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-[680px] max-w-[95vw] h-[540px] flex flex-col p-4 gap-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl shadow-2xl"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("file_open", size="24px").classes("text-primary")
                ui.label(title).classes("text-base font-bold")
            ui.button(icon="close", on_click=lambda: (dialog.close(), done_event.set())).props("flat round dense size=xs")

        path_label = ui.label(str(current)).classes("w-full text-xs font-mono text-slate-500 truncate p-1.5 bg-slate-100 dark:bg-slate-800 rounded")

        list_scroll = ui.scroll_area().classes("w-full flex-1 border border-slate-200 dark:border-slate-800 rounded p-1")
        with list_scroll:
            items_container = ui.column().classes("w-full gap-0.5")

        def refresh_listing(target: Path):
            nonlocal current
            current = target.resolve()
            path_label.set_text(str(current))
            items_container.clear()

            with items_container:
                if current.parent != current:
                    with ui.row().classes("w-full items-center gap-2 px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded cursor-pointer").on(
                        "click", lambda p=current.parent: refresh_listing(p)
                    ):
                        ui.icon("arrow_upward", size="18px").classes("text-slate-400")
                        ui.label(".. (Up one folder)").classes("text-xs font-semibold")

                try:
                    entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                    for entry in entries:
                        if entry.name.startswith(".") and entry.name not in (".lollms_code",):
                            continue
                        if entry.is_dir():
                            with ui.row().classes("w-full items-center gap-2 px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded cursor-pointer").on(
                                "click", lambda p=entry: refresh_listing(p)
                            ):
                                ui.icon("folder", size="18px").classes("text-amber-500")
                                ui.label(entry.name).classes("text-xs text-slate-900 dark:text-slate-100 font-medium truncate")
                        else:
                            def _pick(fp=entry):
                                nonlocal selected_file
                                selected_file = str(fp.resolve())
                                dialog.close()
                                done_event.set()

                            with ui.row().classes("w-full items-center justify-between px-2 py-1 hover:bg-primary/10 rounded cursor-pointer").on("click", _pick):
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("description", size="18px").classes("text-primary")
                                    ui.label(entry.name).classes("text-xs text-slate-900 dark:text-slate-100 truncate")
                                ui.label(f"{entry.stat().st_size / 1024:.1f} KB").classes("text-[10px] text-slate-400 font-mono")
                except Exception as ex:
                    ui.label(f"Cannot read directory: {ex}").classes("text-xs text-red-500 p-2")

        refresh_listing(current)

        with ui.row().classes("w-full items-center justify-end pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=lambda: (dialog.close(), done_event.set())).props("flat dense no-caps")

    dialog.open()
    await done_event.wait()
    return selected_file


async def _open_in_app_directory_picker(title: str, start_dir: Optional[str]) -> Optional[str]:
    """In-app modal directory navigator when all OS dialogs are unavailable."""
    current = Path(start_dir or Path.cwd()).resolve()
    if not current.exists():
        current = Path.cwd().resolve()

    selected_folder: Optional[str] = None
    done_event = asyncio.Event()

    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-[640px] max-w-[95vw] h-[520px] flex flex-col p-4 gap-3 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 rounded-xl shadow-2xl"):
        with ui.row().classes("w-full items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("folder_open", size="24px").classes("text-primary")
                ui.label(title).classes("text-base font-bold")
            ui.button(icon="close", on_click=lambda: (dialog.close(), done_event.set())).props("flat round dense size=xs")

        path_label = ui.label(str(current)).classes("w-full text-xs font-mono text-slate-500 truncate p-1.5 bg-slate-100 dark:bg-slate-800 rounded")

        list_scroll = ui.scroll_area().classes("w-full flex-1 border border-slate-200 dark:border-slate-800 rounded p-1")
        with list_scroll:
            items_container = ui.column().classes("w-full gap-0.5")

        def refresh_listing(target: Path):
            nonlocal current
            current = target.resolve()
            path_label.set_text(str(current))
            items_container.clear()

            with items_container:
                if current.parent != current:
                    with ui.row().classes("w-full items-center gap-2 px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded cursor-pointer").on(
                        "click", lambda p=current.parent: refresh_listing(p)
                    ):
                        ui.icon("arrow_upward", size="18px").classes("text-slate-400")
                        ui.label(".. (Up one folder)").classes("text-xs font-semibold")

                try:
                    subdirs = sorted([d for d in current.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
                    if not subdirs:
                        ui.label("(No subdirectories)").classes("text-xs text-slate-400 italic p-2")
                    for d in subdirs:
                        if d.name.startswith(".") and d.name not in (".lollms_code",):
                            continue
                        with ui.row().classes("w-full items-center gap-2 px-2 py-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded cursor-pointer").on(
                            "click", lambda p=d: refresh_listing(p)
                        ):
                            ui.icon("folder", size="18px").classes("text-amber-500")
                            ui.label(d.name).classes("text-xs text-slate-900 dark:text-slate-100 font-medium truncate")
                except Exception as ex:
                    ui.label(f"Cannot read directory: {ex}").classes("text-xs text-red-500 p-2")

        refresh_listing(current)

        with ui.row().classes("w-full items-center justify-between pt-2 border-t border-slate-200 dark:border-slate-800"):
            ui.button("Cancel", on_click=lambda: (dialog.close(), done_event.set())).props("flat dense no-caps")

            def on_select():
                nonlocal selected_folder
                selected_folder = str(current)
                dialog.close()
                done_event.set()

            ui.button("Select This Folder", icon="check", on_click=on_select).props("unelevated color=primary dense no-caps")

    dialog.open()
    await done_event.wait()
    return selected_folder