"""
Centralized path resolution for LOLLMS Client.

Single source of truth for:

* ``~/.lollms``             — the user's LOLLMS home (shared across apps on the machine)
* ``~/.lollms/data``        — default client-wide data folder (models, caches, venvs, DBs)
* ``~/.lollms/config.yaml`` — default shared configuration file

Resolution precedence for each value (highest wins):

1. Explicit argument passed by the running application (``kwarg``).
2. Environment variable (``LOLLMS_HOME`` / ``LOLLMS_DATA_FOLDER`` / ``LOLLMS_CONFIG_FILE``),
   which users set permanently to relocate storage.
3. Built-in default under ``~/.lollms``.

The module is intentionally dependency-free so it can be imported from any
layer (client, wizard, bindings, CLI tools) without pulling heavy imports.
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

DEFAULT_HOME_DIR_NAME = ".lollms"
DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_CONFIG_FILE_NAME = "config.yaml"

ENV_HOME = "LOLLMS_HOME"
ENV_DATA_FOLDER = "LOLLMS_DATA_FOLDER"
ENV_CONFIG_FILE = "LOLLMS_CONFIG_FILE"


def _materialize(path: Path, *, as_dir: bool) -> Path:
    """Best-effort creation; never raises so callers stay resilient."""
    try:
        if as_dir:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def get_home_dir(override: Optional[Union[str, Path]] = None) -> Path:
    """
    Returns the LOLLMS home directory (default ``~/.lollms``).

    Precedence: ``override`` > ``$LOLLMS_HOME`` > ``~/.lollms``.
    The directory is created on first access (best-effort).
    """
    if override:
        home = Path(override).expanduser().resolve()
    else:
        env_home = os.getenv(ENV_HOME)
        home = Path(env_home).expanduser().resolve() if env_home else (Path.home() / DEFAULT_HOME_DIR_NAME).resolve()
    return _materialize(home, as_dir=True)


def get_data_folder(override: Optional[Union[str, Path]] = None) -> Path:
    """
    Returns the client-wide data folder (default ``~/.lollms/data``).

    Precedence: ``override`` > ``$LOLLMS_DATA_FOLDER`` > ``<home>/data``.
    The directory is created on first access (best-effort) so bindings can
    persist into it immediately.
    """
    if override:
        folder = Path(override).expanduser().resolve()
    else:
        env_folder = os.getenv(ENV_DATA_FOLDER)
        if env_folder:
            folder = Path(env_folder).expanduser().resolve()
        else:
            folder = get_home_dir() / DEFAULT_DATA_DIR_NAME
    return _materialize(folder, as_dir=True)


def get_config_file(override: Optional[Union[str, Path]] = None) -> Path:
    """
    Returns the shared configuration file path (default ``~/.lollms/config.yaml``).

    Precedence: ``override`` > ``$LOLLMS_CONFIG_FILE`` > ``<home>/config.yaml``.
    The parent directory is created on first access (best-effort); the file
    itself is only created when a caller writes to it.
    """
    if override:
        cfg = Path(override).expanduser().resolve()
    else:
        env_file = os.getenv(ENV_CONFIG_FILE)
        if env_file:
            cfg = Path(env_file).expanduser().resolve()
        else:
            cfg = get_home_dir() / DEFAULT_CONFIG_FILE_NAME
    return _materialize(cfg, as_dir=False)


# ── Permanent environment variable persistence ────────────────────────────────

def _posix_rc_candidates() -> List[Path]:
    """Returns the user's shell rc files in priority order (best-match first).

    We prefer the rc file corresponding to the login shell, then fall back to
    the common defaults. Files are returned regardless of existence; callers
    choose which one to write.
    """
    home = Path.home()
    shell = os.getenv("SHELL", "")
    shell_name = Path(shell).name.lower() if shell else ""

    candidates: List[Path] = []
    if shell_name == "zsh":
        candidates.append(home / ".zshrc")
    elif shell_name == "bash":
        candidates.append(home / ".bashrc")
    elif shell_name == "fish":
        candidates.append(home / ".config" / "fish" / "config.fish")

    # Common fallbacks (do not duplicate already-listed entries)
    for fallback in (home / ".bashrc", home / ".zshrc", home / ".profile"):
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _persist_posix(name: str, value: str) -> Dict[str, object]:
    """Persists ``export NAME=value`` into the user's shell rc file.

    Idempotent: replaces an existing ``export NAME=`` line rather than
    appending a duplicate. Uses ``shlex.quote`` so paths with spaces or
    shell metacharacters are safe.
    """
    line = f'export {name}={shlex.quote(value)}'
    pattern = re.compile(rf"^\s*export\s+{re.escape(name)}\s*=")

    written_to: List[str] = []
    errors: List[str] = []
    for rc in _posix_rc_candidates():
        try:
            rc.parent.mkdir(parents=True, exist_ok=True)
            existing = rc.read_text(encoding="utf-8") if rc.exists() else ""
            lines = existing.splitlines()
            replaced = False
            for idx, current in enumerate(lines):
                if pattern.match(current):
                    lines[idx] = line
                    replaced = True
                    break
            if not replaced:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append(line)
            new_content = "\n".join(lines) + "\n"
            rc.write_text(new_content, encoding="utf-8")
            written_to.append(str(rc))
            # Succeed on the first rc whose shell matches the active SHELL,
            # but keep going on unrelated rc files only if none matched yet.
            shell_name = Path(os.getenv("SHELL", "")).name.lower()
            if (shell_name == "zsh" and rc.name == ".zshrc") or \
               (shell_name == "bash" and rc.name == ".bashrc") or \
               (shell_name == "fish" and rc.name == "config.fish"):
                break
            if shell_name == "" and rc.name == ".bashrc":
                break
        except OSError as exc:
            errors.append(f"{rc}: {exc}")

    if written_to:
        return {
            "ok": True,
            "message": f"Persisted {name} in shell rc: {', '.join(written_to)}.",
            "persisted_to": written_to,
        }
    return {
        "ok": False,
        "message": f"Could not persist {name} to any shell rc file. Errors: {'; '.join(errors) or 'no writable rc found'}.",
        "persisted_to": [],
    }


def _persist_windows(name: str, value: str) -> Dict[str, object]:
    """Persists the variable under HKCU\\Environment and broadcasts the change.

    Uses the registry directly (not ``setx``) to avoid the 1024-char truncation
    and sluggishness of ``setx``. No admin rights are required for user scope.
    """
    try:
        import winreg
    except ImportError:
        return {
            "ok": False,
            "message": "winreg unavailable; cannot persist environment variable on this platform.",
            "persisted_to": [],
        }

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
    except OSError as exc:
        return {
            "ok": False,
            "message": f"Failed to write HKCU\\Environment\\{name}: {exc}",
            "persisted_to": [],
        }

    # Best-effort broadcast so running shells/Explorer refresh their environment.
    try:
        import ctypes
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        SMTO_ABORTIFHUNG = 0x0002
        result = ctypes.c_ulong()
        ctypes.windll.user32.SendMessageTimeoutW(
            HWND_BROADCAST,
            WM_SETTINGCHANGE,
            0,
            ctypes.c_wchar_p("Environment"),
            SMTO_ABORTIFHUNG,
            5000,
            ctypes.byref(result),
        )
    except (AttributeError, OSError):
        # Non-fatal: value is persisted; only running processes won't refresh live.
        pass

    return {
        "ok": True,
        "message": f"Persisted {name} in HKCU\\Environment (user scope).",
        "persisted_to": [f"HKCU\\Environment\\{name}"],
    }


def set_env_var_permanently(name: str, value: str) -> Dict[str, object]:
    """
    Sets an environment variable **permanently** for the current user and makes
    it immediately effective in the running process.

    The result is best-effort persistence at *user scope* — no admin rights
    required, no global system-wide modification:

    * **Windows**: writes ``HKCU\\Environment\\<name>`` and broadcasts
      ``WM_SETTINGCHANGE`` so already-running shells pick up the change.
    * **POSIX**: appends (or replaces) ``export NAME=value`` in the user's
      shell rc (``~/.zshrc`` / ``~/.bashrc`` / ``~/.profile``; fish gets its
      own ``config.fish``).

    The running process always receives the change via ``os.environ`` before
    returning, regardless of the persistence outcome.

    Returns:
        dict: ``{"ok": bool, "message": str, "persisted_to": list[str]}``.
        Never raises. ``ok`` is True iff persistence at the OS level succeeded.
    """
    if not name or not name.strip():
        return {"ok": False, "message": "Empty environment variable name.", "persisted_to": []}
    if value is None:
        return {"ok": False, "message": "Value must be a string.", "persisted_to": []}

    value = str(value)
    os.environ[name] = value  # effective in this process right away

    if sys.platform == "win32":
        return _persist_windows(name, value)
    return _persist_posix(name, value)


def unset_env_var_permanently(name: str) -> Dict[str, object]:
    """
    Removes an environment variable **permanently** at user scope and from the
    current process. Best-effort; never raises.

    Returns:
        dict: ``{"ok": bool, "message": str, "persisted_to": list[str]}``.
    """
    if not name or not name.strip():
        return {"ok": False, "message": "Empty environment variable name.", "persisted_to": []}

    os.environ.pop(name, None)

    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.DeleteValue(key, name)
        except FileNotFoundError:
            return {"ok": True, "message": f"{name} was not set at user scope.", "persisted_to": []}
        except OSError as exc:
            return {"ok": False, "message": f"Failed to delete HKCU\\Environment\\{name}: {exc}", "persisted_to": []}
        return {"ok": True, "message": f"Removed {name} from HKCU\\Environment.", "persisted_to": [f"HKCU\\Environment\\{name}"]}

    pattern = re.compile(rf"^\s*export\s+{re.escape(name)}\s*=")
    removed_from: List[str] = []
    for rc in _posix_rc_candidates():
        if not rc.exists():
            continue
        try:
            lines = rc.read_text(encoding="utf-8").splitlines()
            kept = [ln for ln in lines if not pattern.match(ln)]
            if len(kept) != len(lines):
                rc.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                removed_from.append(str(rc))
        except OSError:
            continue

    if removed_from:
        return {
            "ok": True,
            "message": f"Removed {name} from: {', '.join(removed_from)}.",
            "persisted_to": removed_from,
        }
    return {"ok": True, "message": f"{name} was not set in any shell rc file.", "persisted_to": []}