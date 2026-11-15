#!/usr/bin/env python3
"""
lollms_shell — Autonomous Shell Companion
=========================================

A companionable CLI copilot that turns natural-language instructions into
audited shell commands. Designed to sit beside the user in their terminal and
behave like a knowledgeable friend who knows how their machine is set up:

  * "list the files here"                     → ls -la
  * "copy report.pdf to ~/Documents"          → cp report.pdf ~/Documents/
  * "why is my disk full?"                    → df -h + du analysis + diagnosis
  * "activate the project venv and run tests" → activate + pytest

It is built on the same LollmsPersonality + Handbag architecture as
lollms_code, reusing the LCP `system_shell` tool set for execution. Unlike
lollms_code, it is NOT a file-editing agent: its domain is the shell itself.

Key behaviours:
  * The prompt header shows the LIVE current working directory each turn, plus
    the active Python virtual environment if one is loaded.
  * Project-local cognitive memory (per working directory) remembers durable
    shell facts across sessions (paths, aliases, past diagnoses).
  * Destructive commands are refused unless the user explicitly approves them
    in prose — even in full_access autonomy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ascii_colors import ASCIIColors, trace_exception

from lollms_client import LollmsClient
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_personality.lollms_personality import CapabilityFlags
from lollms_client.lollms_types import EventMode, MSG_TYPE

# Persistent shell worker — makes `cd`, `set VAR`, `activate` etc. survive
# across tool calls. Imported lazily-safe (module has no side effects at
# import time besides stdlib imports).
from lollms_client.apps.lollms_shell.persistent_shell import (
    PersistentShellSession,
)

APP_NAME = "lollms_shell"
APP_VERSION = "1.0.0"
APP_CONFIG_DIR = Path.home() / ".lollms" / "lollms_shell"
APP_CONFIG_FILE = APP_CONFIG_DIR / "config.json"
APP_DEFAULT_HANDBAG_DIR = APP_CONFIG_DIR / "handbags"

# lollms_shell is *designed* to ship without a RAG database. The generic
# LollmsPersonality diagnostic ("RAG catalog is EMPTY — the agent WILL report
# 'no RAG database'") is meant to catch handbag misconfigurations — for this
# app there is nothing to configure. We install a narrow filter that drops ONLY
# that exact message (and its continuation line) so genuine warnings from
# [Handbag-SCAN], tool failures, config errors, etc. remain visible.
_RAG_EMPTY_WARNING_PREFIX = f"[{APP_NAME}] RAG catalog is EMPTY"
# Additional shapes of the same "no RAG" notice across the personality/handbag
# stack. lollms_shell intentionally has NO RAG — these are all noise to us.
_RAG_EMPTY_WARNING_FRAGMENTS = (
    _RAG_EMPTY_WARNING_PREFIX,
    # Handbag-SCAN: "⚠️ No 'rag/' directory under ..." (emoji may vary in
    # rendering — match the stable text after the emoji so we don't depend on
    # exact emoji byte sequences across terminals).
    "No 'rag/' directory under",
    "[Handbag] No 'rag/' folder",
    "is a RAG-free agent",
    # The LCP loader cannot see the handbag's tools/ folder (it is not in the
    # LCP search paths); the handbag loader mounts the tool instead. The mount
    # warning is noise for this app *because the tool IS loaded*, just via the
    # handbag path.
    "[LCP Mount] Library 'shell_session' not found",
)


class _DropShellRagEmptyWarning(logging.Filter):
    """Logging filter: drops lollms_shell's deliberately-empty-RAG notices only.

    Matches the family of "no RAG database" messages emitted by the personality
    and handbag loaders. lollms_shell ships without a `rag/` folder ON PURPOSE;
    the warning is intended for agents that lose their datastore, not for agents
    that never had one. All other diagnostics (tool errors, config failures,
    real tracebacks) pass through untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if not msg:
            return True
        return not any(frag in msg for frag in _RAG_EMPTY_WARNING_FRAGMENTS)


def _silence_empty_rag_warning() -> None:
    """Installs the narrow RAG-empty-warning filter on ASCIIColors' logger (idempotent)."""
    try:
        logger = logging.getLogger("ASCIIColors")
    except Exception:
        return
    for existing in logger.filters:
        if isinstance(existing, _DropShellRagEmptyWarning):
            return
    logger.addFilter(_DropShellRagEmptyWarning())


def _find_project_root() -> Path:
    p = Path(__file__).resolve().parent
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ── SOUL.md — SINGLE SOURCE OF TRUTH ───────────────────────────────────────
#
# Historically this module held an inline `SHELL_SYSTEM_PROMPT` constant that
# `ensure_handbag_structure()` wrote to the user's handbag SOUL.md on every
# launch, OVERWRITING any edit to the packaged file. That caused the actual
# doctrine the LLM sees to diverge from the packaged
# `default_handbag/SOUL.md` — e.g. an old doctrine that still told the model
# to call the stateless `tool_execute_shell_command`.
#
# Now the packaged `default_handbag/SOUL.md` is THE source of truth. The
# function below reads it from disk (in the installed package) so handbag
# refreshes always install the doctrine this release ships.
_SOUL_METADATA_HEADER = (
    "---\n"
    "name: lollms_shell\n"
    "author: ParisNeo\n"
    "category: system_administration\n"
    "description: A terse shell companion that runs native commands for the "
    "user's terminal, in one persistent shell session.\n"
    "---\n\n"
)


def _load_packaged_soul_body() -> str:
    """Reads the packaged SOUL.md body (without YAML front matter).

    Returns the doctrine text that must be installed into the handbag on
    every launch, or a minimal fallback if the file cannot be read.
    """
    soul_file = Path(__file__).resolve().parent / "default_handbag" / "SOUL.md"
    try:
        raw = soul_file.read_text(encoding="utf-8")
    except OSError as exc:
        ASCIIColors.warning(
            f"[CLI] Packaged SOUL.md not found at {soul_file}: {exc}. "
            f"Using a minimal fallback doctrine — the agent may be impaired."
        )
        return (
            "You are lollms_shell. Run every command via the tool "
            "`tool_shell_session_execute` in the SAME persistent shell session "
            "(state survives across calls). Conclude every reply with `<done/>`.\n"
        )

    # Strip leading YAML front matter (---\n...\n---\n) if present, keeping
    # only the doctinal body. The header we prepend is authoritative.
    if raw.lstrip().startswith("---"):
        lines = raw.splitlines(keepends=True)
        end_idx = None
        # Skip the opening '---' then scan for the closing '---'
        opening = None
        for i, line in enumerate(lines):
            if line.strip() == "---":
                if opening is None:
                    opening = i
                else:
                    end_idx = i
                    break
        if opening is not None and end_idx is not None:
            raw = "".join(lines[end_idx + 1:]).lstrip("\n")
    return raw


# ── SHELL / ENVIRONMENT DISCOVERY ──────────────────────────────────────────
#
# The parent-shell identification is AUTHORITATIVE via the process tree — env
# vars like PSModulePath are inherited and unreliable (a cmd.exe launched from
# a PowerShell session still carries PSModulePath, and so does a Python process
# started from cmd.exe if PowerShell was ever installed).

_SHELL_CACHE: dict[str, str] = {"shell": ""}

# Canonical internal shell identifiers used everywhere in this module.
#   "cmd"        → Windows cmd.exe
#   "powershell" → Windows PowerShell 5.x OR PowerShell 7+ (pwsh.exe)
#   "bash"       → GNU bash / git-bash / MSYS bash
#   "zsh"        → zsh
#   "fish"       → fish
#   "wsl"        → Windows Subsystem for Linux sh/bash launcher
#   "sh"         → bare POSIX sh (fallback)
_SHELL_PROCESS_NAMES = {
    "cmd.exe": "cmd",
    "powershell.exe": "powershell",
    "pwsh.exe": "powershell",
    "bash.exe": "bash",
    "zsh.exe": "zsh",
    "fish.exe": "fish",
    "wsl.exe": "wsl",
    "bash": "bash",
    "zsh": "zsh",
    "fish": "fish",
    "sh": "sh",
    "dash": "sh",
}


def _proc_name_from_pid_windows(pid: int) -> Optional[str]:
    """Returns the executable name (lowercase) for a Windows PID, or None."""
    try:
        # Get-CimInstance is fast, present on Win10+ (which the project supports).
        out = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').Name",
            ],
            capture_output=True, text=True, timeout=3, check=False,
        )
        name = (out.stdout or "").strip()
        return name.lower() if name else None
    except (OSError, subprocess.SubprocessError):
        return None


def _parent_pid_windows(pid: int) -> Optional[int]:
    """Returns the parent PID for a Windows PID, or None."""
    try:
        out = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').ParentProcessId",
            ],
            capture_output=True, text=True, timeout=3, check=False,
        )
        raw = (out.stdout or "").strip()
        return int(raw) if raw.isdigit() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _detect_shell_windows() -> str:
    """Walks up the parent-process chain until a known shell is found."""
    pid = os.getpid()
    for _ in range(8):  # bounded walk — never spin indefinitely
        parent = _parent_pid_windows(pid)
        if not parent or parent <= 0:
            break
        name = _proc_name_from_pid_windows(parent)
        if name and name in _SHELL_PROCESS_NAMES:
            detected = _SHELL_PROCESS_NAMES[name]
            # Distinguish WSL bash from native bash when possible.
            if detected == "bash" and name == "bash.exe":
                return "bash"
            return detected
        pid = parent
    # Fallback heuristics (best effort; the process-tree walk usually wins).
    if os.environ.get("WT_SESSION") and os.environ.get("PSModulePath"):
        return "powershell"
    if os.environ.get("PSModulePath"):
        return "powershell"
    return "cmd"


def _detect_shell_posix() -> str:
    """POSIX: SHELL env var is authoritative for interactive sessions."""
    shell_path = os.environ.get("SHELL", "")
    name = shell_path.rsplit("/", 1)[-1].lower() if shell_path else ""
    return _SHELL_PROCESS_NAMES.get(name, "sh")


def _active_shell() -> str:
    """Returns the canonical shell identifier (cached after first probe)."""
    if _SHELL_CACHE["shell"]:
        return _SHELL_CACHE["shell"]

    if platform.system() == "Windows":
        detected = _detect_shell_windows()
    else:
        detected = _detect_shell_posix()

    _SHELL_CACHE["shell"] = detected
    return detected


def _active_venv(session: Optional["PersistentShellSession"] = None) -> Optional[str]:
    """Returns the active Python virtual environment path, or None.

    Authoritative source is the WORKER shell's `VIRTUAL_ENV`/`CONDA_PREFIX`
    when a session is provided — `activate` / `deactivate` run *inside* the
    worker and never touch the Python REPL's `os.environ`. Reading
    `os.environ` directly would show a stale value and make the `(venv)`
    prompt prefix disagree with the actual shell state.

    Falls back to `os.environ` only when no session is available (very first
    banner render, or a broken session).
    """
    if session is not None:
        try:
            snap = session.snapshot_env(
                ("VIRTUAL_ENV", "CONDA_PREFIX", "VIRTUAL_ENV_PROMPT")
            )
        except Exception:
            snap = {}
        v = snap.get("VIRTUAL_ENV") or snap.get("CONDA_PREFIX")
        if v:
            return v
        # If the worker definitively has NO venv, do NOT fall back to
        # `os.environ`: the user may have DEACTIVATED a venv that Python
        # still remembers inheriting from its parent process. Announcing
        # `(venv)` there would be a lie.
        if snap or session is not None:
            # A successful snapshot (even empty) is authoritative.
            # On failure (`snap == {}`), we still prefer a clean "no venv".
            return None
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        return venv
    return None


def _disk_free_for(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Returns (free_bytes, total_bytes) for the filesystem containing `path`.

    On Windows, `shutil.disk_usage` may raise for network shares or exotic
    mount points; we swallow that and return (None, None).
    """
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free, usage.total
    except OSError:
        return None, None


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _probe_tool(name: str) -> Optional[str]:
    """Returns the resolved path for `name` if it exists on PATH, else None."""
    try:
        return shutil.which(name)
    except (OSError, ValueError):
        return None


def _shell_syntax_guide(shell: str) -> str:
    """Per-shell cheat sheet for the LLM's system prompt.

    The goal is not to teach a seasoned developer — it is to prevent the LLM
    from emitting commands that are syntactically invalid in the shell it is
    actually inside. Windows shells are especially divergent.
    """
    if shell == "powershell":
        return (
            "**PowerShell (native).** Cmdlets use Verb-Noun and PascalCase. "
            "List: `Get-ChildItem` (alias `ls`, `dir`, `gci`). Read: `Get-Content` (alias `cat`, `type`). "
            "Sizes: `Get-ChildItem -Recurse | Measure-Object -Property Length -Sum`. "
            "Disk: `Get-PSDrive -PSProvider FileSystem` or `Get-Volume`. "
            "Processes: `Get-Process`. Kill: `Stop-Process -Id <pid>`. "
            "Env: `$env:NAME`. Path join: `Join-Path`. "
            "Pipes: real object pipelines — use `Where-Object`, `Select-Object`, `ForEach-Object`. "
            "NEVER use `dir /b`, `where /r`, `findstr` unless explicitly requested. "
            "Chain with `;` (sequential) or `&&` (if-success, PS7+ only; use `if ($?) { ... }` on PS5). "
            "Quote paths with `\"...\"`; escape with backtick `` ` ``."
        )
    if shell == "cmd":
        return (
            "**cmd.exe (native).** List: `dir` (use `dir /b` for bare names, `dir /s` recursive). "
            "Read: `type`. Find in files: `findstr`. Find files: `where /r <path> <pattern>`. "
            "Disk: `wmic logicaldisk get size,freespace,caption` (or `fsutil volume diskfree C:`). "
            "Processes: `tasklist`. Kill: `taskkill /PID <pid> /F`. "
            "Env: `%NAME%`. Path join: `%~dp0` for script dir, plain `\\` otherwise. "
            "Chain with `&&` (if-success) or `&` (always). NOT `;` (cmd treats `;` as a token separator, NOT a command separator). "
            "NEVER emit PowerShell cmdlets or bash flags — they will error out. "
            "Quote paths with `\"...\"`. Escape `%` as `%%` inside scripts, `^` for literal `& | < > ^`."
        )
    if shell in ("bash", "zsh", "sh", "wsl"):
        return (
            f"**{'Bash/Zsh' if shell in ('bash', 'zsh') else shell} (native POSIX).** "
            "List: `ls -la`. Read: `cat`. Find files in tree: `find . -name '*.py'`. "
            "Find text: `grep -r`. Disk: `df -h` (mount points) and `du -sh * | sort -h` (sizes). "
            "Processes: `ps aux` or `ps -ef`. Kill: `kill <pid>` / `kill -9 <pid>`. "
            "Env: `$NAME`. Path join: `/`. "
            "Chain with `&&` (if-success), `;` (always), `||` (if-fail). "
            "Quote with `'...'` for literal, `\"...\"` for expansion. "
            "If this is WSL, remember Windows drives are under `/mnt/c/…`."
        )
    if shell == "fish":
        return (
            "**fish shell (native).** Similar to bash but: `set -x VAR value` instead of `export`, "
            "no `$(...)` (use `(cmd)`), `set PATH $PATH /dir` for PATH. "
            "List/read/disk/ps commands are the same as bash. Chain with `;` or `and`/`or`."
        )
    return f"**{shell} (POSIX-like).** Prefer portable POSIX commands; verify flags if unsure."


def build_environment_context(
    cwd: Path,
    session: Optional["PersistentShellSession"] = None,
) -> str:
    """Builds the per-turn environment context block.

    This block is INJECTED INTO THE LLM'S SYSTEM PROMPT at every turn. It is
    designed to make the agent useful out of the box: it tells the model which
    shell it is in (and how that shell differs), what machine it is on, what
    dev tools are available, and how much disk / RAM is left right now.

    When `session` is provided, `VIRTUAL_ENV`, `CONDA_PREFIX`, and `PATH` are
    read from the **worker shell** — so `activate`, `deactivate`, `set VAR`,
    `export VAR` all show up on the very next turn (Python's `os.environ` is
    unaffected by those worker-side changes and would be stale).
    """
    is_windows = platform.system() == "Windows"
    shell = _active_shell()
    path_sep = "\\" if is_windows else "/"
    venv = _active_venv(session)
    session_env: dict[str, str] = {}
    if session is not None:
        try:
            session_env = session.snapshot_env()
        except Exception:
            session_env = {}

    # Disk facts for the CURRENT working directory drive — typically the one
    # the user cares about when asking "why is my disk full?".
    free_bytes, total_bytes = _disk_free_for(cwd)
    used_bytes = None
    if free_bytes is not None and total_bytes is not None and total_bytes > 0:
        used_bytes = total_bytes - free_bytes
        used_pct = (used_bytes / total_bytes) * 100.0
        disk_line = (
            f"- Disk (drive holding CWD): {_human_bytes(used_bytes)} used of "
            f"{_human_bytes(total_bytes)} ({used_pct:.0f}%), {_human_bytes(free_bytes)} free"
        )
    else:
        disk_line = "- Disk (drive holding CWD): (unavailable)"

    # Machine facts.
    try:
        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0
    arch = platform.machine() or "?"
    try:
        sys_ver = platform.version() or "?"
    except Exception:
        sys_ver = "?"

    # Detect a handful of dev tools. Probe using the WORKER's PATH when
    # available — otherwise `activate`'s PATH mutation wouldn't be reflected
    # (e.g. a venv-local `python` would still resolve to the global one).
    tool_probes = ["git", "python", "pip", "node", "gcc", "make", "rg", "curl", "jq"]
    worker_path = session_env.get("PATH") or ""
    if worker_path:
        # Split on the platform-appropriate separator and probe each entry.
        sep = ";" if is_windows else ":"
        pathexts = (
            session_env.get("PATHEXT", "").split(";")
            if is_windows else [""]
        )
        exts = [e.lower() for e in pathexts] or [""]
        found_tools: list[str] = []
        for tool in tool_probes:
            hit = None
            for entry in worker_path.split(sep):
                entry = entry.strip()
                if not entry:
                    continue
                for ext in exts:
                    cand = Path(entry) / (tool + ext)
                    try:
                        if cand.is_file():
                            hit = str(cand)
                            break
                    except OSError:
                        continue
                if hit:
                    break
            if hit:
                found_tools.append(tool)
        missing_tools = [t for t in tool_probes if t not in found_tools]
    else:
        found_tools = [t for t in tool_probes if _probe_tool(t)]
        missing_tools = [t for t in tool_probes if t not in found_tools]

    lines = [
        "=== ENVIRONMENT CONTEXT ===",
        f"- OS: {platform.system()} {platform.release()} (build {sys_ver}) | Arch: {arch} | CPUs: {cpu_count}",
        f"- Shell: `{shell}` | Path separator: `{path_sep}`",
        _shell_syntax_guide(shell),
        f"- Python: {platform.python_version()} | Executable: `{sys.executable}`",
        f"- CWD (resolve ALL relative paths against this): `{cwd}`",
    ]
    if venv:
        venv_prompt = session_env.get("VIRTUAL_ENV_PROMPT") or ""
        lines.append(
            f"- Active virtual env (IN THE SHELL, from the worker process): `{venv}`"
            + (f" | prompt: `{venv_prompt}`" if venv_prompt else "")
            + " — `python`, `pip`, `pytest` resolve here. "
            "Bypass with the absolute interpreter path above if needed."
        )
    else:
        lines.append(
            "- Active virtual env: (none in the shell worker) — "
            "the global Python is used. `activate`/`deactivate` performed in the "
            "shell update this on the NEXT turn."
        )
    try:
        user = os.getenv("USERNAME") or os.getenv("USER") or "user"
        lines.append(f"- User: `{user}`")
    except Exception:
        pass
    lines.append(disk_line)
    if found_tools:
        lines.append(f"- Dev tools available on PATH: {', '.join(found_tools)}")
    if missing_tools:
        lines.append(
            f"- Dev tools NOT found on PATH: {', '.join(missing_tools)} "
            f"(do NOT assume they exist; probe with `{'where' if shell == 'cmd' else 'Get-Command' if shell == 'powershell' else 'command -v'}` before use)"
        )

    lines.append(
        "=== HOW TO USE THIS BLOCK ===\n"
        "  · Emit ONLY commands that are syntactically valid in the shell named above.\n"
        "  · Read-only commands (list / read / disk / processes / find) — run freely.\n"
        "  · Destructive commands (delete, overwrite, move-onto-existing, format, diskpart,\n"
        "    shutdown, reg add, setx, package install) — ask the user in prose first.\n"
        "  · When diagnosing, run concrete diagnostic commands and interpret the numbers."
    )
    lines.append("=== END ENVIRONMENT CONTEXT ===")
    return "\n".join(lines)


def _session_cwd(session: Optional["PersistentShellSession"]) -> Path:
    """Returns the WORKER shell's live cwd, or the REPL process cwd as fallback.

    The worker is the source of truth: a `cd ..` the agent runs via a tool
    changes the worker's cwd, and the REPL prompt must reflect that — otherwise
    the user sees the wrong directory and the agent sees a stale one.
    """
    if session is None:
        return Path.cwd()
    try:
        worker_cwd = session.cwd()
    except Exception:
        return Path.cwd()
    # Hardening: never let a stale `_last_cwd` / probe result leave us returning
    # a directory that no longer exists (e.g. after the shell deleted it).
    try:
        if not worker_cwd.is_dir():
            return Path.cwd()
    except OSError:
        return Path.cwd()
    return worker_cwd


_SESSION_ATTACHED = {"done": False}


def _wire_persistent_session_tool(
    session: "PersistentShellSession", config: "ShellAgentConfig"
) -> None:
    """Injects the session + confirm handler into every live shell_session module copy.

    The tool can be loaded by TWO loaders with different import mechanics:
      * the **LCP tools loader** — imports `<handbag>/tools/shell_session.py`
        as a standalone module (module name varies).
      * the **handbag loader** — imports the same file, usually as
        `lollms_shell_session_tool` or similar.
    Plus the **canonical** copy shipped with the package
    (`lollms_client.apps.lollms_shell.tools.shell_session`), which is the module
    that the handbag copy re-exports from when present.

    We attach to *all* of them so nothing depends on loader ordering. `attach_session`
    is idempotent: it just reassigns the module-level `_SESSION`/`_CONFIRM_HANDLER`
    globals, so calling it more than once is safe.
    """
    import importlib

    confirmed_handler = _make_confirm_handler(config)
    attached_any = False

    # 1. Canonical package module.
    try:
        from lollms_client.apps.lollms_shell.tools import shell_session as _canonical_mod
        attach = getattr(_canonical_mod, "attach_session", None)
        if callable(attach):
            attach(session, confirmed_handler)
            attached_any = True
    except ImportError:
        pass

    # 2. Every module in sys.modules that exposes BOTH `attach_session` AND
    #    `tool_shell_session_execute` — regardless of module name. The handbag
    #    loader imports `default_handbag/tools/shell_session.py` under a
    #    generated module name (e.g. `tool_shell_session_execute` or
    #    `lollms_shell_session_tool_<hash>`), so name-based matching alone
    #    misses the copy that the LLM actually dispatches to. Shape-based
    #    matching catches every copy.
    for mod_name, mod in list(sys.modules.items()):
        attach = getattr(mod, "attach_session", None)
        executor = getattr(mod, "tool_shell_session_execute", None)
        if not (callable(attach) and callable(executor)):
            continue
        try:
            attach(session, confirmed_handler)
            attached_any = True
        except Exception:
            pass

    # 3. If nothing was attached, try to import the handbag's copy directly.
    if not attached_any:
        try:
            tools_dir = Path(config.handbag_path).resolve() / "tools"
            shim = tools_dir / "shell_session.py"
            if shim.exists():
                spec = importlib.util.spec_from_file_location(
                    "lollms_shell_session_tool_hb", shim
                )
                if spec and spec.loader:
                    _mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_mod)
                    sys.modules["lollms_shell_session_tool_hb"] = _mod
                    attach = getattr(_mod, "attach_session", None)
                    if callable(attach):
                        attach(session, confirmed_handler)
                        attached_any = True
        except (OSError, ImportError, AttributeError):
            pass

    if not attached_any:
        # Not fatal: `persistent_shell.get_active_session()` is the fallback
        # path used by every tool copy's `_resolve_session()`. Warn only when
        # debug, and never abort startup.
        if config.debug:
            ASCIIColors.warning(
                "[CLI] No shell_session module explicitly attached — relying on "
                "the process-global session registry instead. This is safe."
            )
        return

    _SESSION_ATTACHED["done"] = True


def _venv_name_for_prompt(session: Optional["PersistentShellSession"] = None) -> str:
    """Returns the active venv/conda env *name* (not full path), or ''.

    When a session is provided, the worker's env is authoritative — so
    `activate` / `deactivate` performed inside the persistent shell are
    reflected on the very next prompt render.
    """
    venv = _active_venv(session)
    if not venv:
        return ""
    return Path(venv).name or ""


def _cwd_for_prompt(cwd: Optional[Path] = None) -> str:
    """CWD display used inside the shell prompt (real-shell style).

    Matches what the user's OWN terminal would show:
      * Windows cmd / PowerShell: the literal path ``C:\\Users\\…\\ai``.
        Windows shells NEVER collapse HOME to ``~`` — showing ``~`` there
        looks foreign and put users off in testing.
      * POSIX shells (bash/zsh/fish/sh/wsl): collapse ``$HOME`` to ``~``,
        matching what those prompts normally display.

    When `cwd` is provided (the worker's live cwd), it is used directly — this
    is how a `cd` performed by the agent surfaces in the prompt. When omitted,
    the REPL process's cwd is used (startup / fallbacks only).
    """
    cwd = cwd if cwd is not None else Path.cwd()
    shell = _active_shell()

    # Windows shells (cmd / powershell / native bash-in-windows without WSL):
    # show the real Windows path — no `~` collapsing.
    if platform.system() == "Windows" and shell in ("cmd", "powershell"):
        return str(cwd)

    try:
        rel = cwd.relative_to(Path.home())
        rel_str = str(rel).replace("\\", "/")
        return "~" if rel_str in ("", ".") else f"~/{rel_str}"
    except ValueError:
        return str(cwd)


def _build_shell_prompt(session: Optional["PersistentShellSession"] = None) -> str:
    """Builds a prompt string that MATCHES the detected shell.

    `session` is the persistent shell worker — its `cwd()` is authoritative,
    so a `cd` performed by the agent through the worker tool is reflected in
    the user's next prompt. When session is None we fall back to `Path.cwd()`
    (startup rendering).

    The `(venv)` prefix is ALSO resolved from the worker: `activate` /
    `deactivate` run inside the persistent session and MUST show up here on
    the next turn.
    """
    shell = _active_shell()
    venv_name = _venv_name_for_prompt(session)
    prefix = f"({venv_name}) " if venv_name else ""
    cwd_display = _cwd_for_prompt(_session_cwd(session))
    user = os.getenv("USER") or os.getenv("USERNAME") or "user"
    hostname_env = os.getenv("HOSTNAME") or platform.node().split(".")[0] or "host"

    if shell == "powershell":
        return f"{prefix}PS {cwd_display}> "
    if shell == "cmd":
        return f"{prefix}{cwd_display}> "
    if shell == "fish":
        return f"{prefix}{user}@{hostname_env} {cwd_display}> "
    # bash / zsh / sh / wsl
    suffix = "# " if user == "root" else "$ "
    return f"{prefix}{user}@{hostname_env}:{cwd_display}{suffix}"


def _render_startup_banner(
    config: "ShellAgentConfig",
    personality: LollmsPersonality,
    session: Optional["PersistentShellSession"] = None,
):
    """One-shot startup panel (line stays; not re-rendered per turn)."""
    venv = _active_venv(session)
    shell = _active_shell()

    lines = [
        f"[cyan]Shell:[/cyan]  {shell}",
    ]
    if venv:
        try:
            venv_display = Path(venv).relative_to(Path.home())
            venv_display = f"~{os.sep}{venv_display}"
        except ValueError:
            venv_display = venv
        lines.append(f"[cyan]Env:[/cyan]    [green]{venv_display}[/green]")
    else:
        lines.append(f"[cyan]Env:[/cyan]    [dim](no virtualenv active)[/dim]")
    lines.append(f"[cyan]Model:[/cyan]  {config.active_model_name} [dim]({config.active_binding_name})[/dim]")

    free_b, total_b = _disk_free_for(Path.cwd())
    if free_b is not None and total_b:
        used = total_b - free_b
        pct = (used / total_b) * 100.0
        lines.append(
            f"[cyan]Disk:[/cyan]   {_human_bytes(used)} used / {_human_bytes(total_b)} "
            f"({pct:.0f}%)"
        )
    lines.append(f"[cyan]CWD:[/cyan]    {_cwd_for_prompt()}")
    lines.append(f"[cyan]Mode:[/cyan]   shell={config.shell_autonomy_level.upper()} "
                 f"(doctrinal confirmation on destructive commands)")

    ASCIIColors.panel(
        "\n".join(lines),
        title=f"[bold green]🐚 lollms_shell v{APP_VERSION}[/bold green]",
        border_style="green",
    )


# ── CONFIG ─────────────────────────────────────────────────────────────────

class ShellAgentConfig:
    """Configuration for lollms_shell, reusing the Universal Two-Tier Profile system."""

    def __init__(self):
        self.llm_binding_profiles: dict[str, dict[str, Any]] = {}
        self.llm_model_profiles: dict[str, dict[str, Any]] = {}

        self.active_profile: str | None = None
        self.max_reasoning_steps: int = 20
        self.temperature: float = 0.2
        self.max_tokens_per_turn: int = 4096
        self.context_compaction_threshold: float = 0.85

        self.reasoning_effort: Optional[str] = None
        self.dynamic_effort: bool = False

        self.enable_shell_execution: bool = True
        # Dedicated shell companion: run read-only commands freely. The
        # DESTRUCTIVE-COMMAND doctrine (SOUL.md) is enforced *in addition*
        # to the tool gate — see `_make_confirm_handler`.
        self.shell_autonomy_level: str = "full_access"
        self.enable_memory: bool = True
        self.handbag_path: str = str(APP_DEFAULT_HANDBAG_DIR / "default_shell")

        self.show_tool_calls: bool = True
        self.debug: bool = False

    @property
    def active_profile_alias(self) -> str:
        if self.active_profile and self.active_profile in self.llm_model_profiles:
            return self.active_profile
        for alias, prof in self.llm_model_profiles.items():
            if prof.get("is_default"):
                return alias
        if self.llm_model_profiles:
            return next(iter(self.llm_model_profiles))
        return "default"

    @property
    def active_model_name(self) -> str:
        alias = self.active_profile_alias
        prof = self.llm_model_profiles.get(alias, {})
        return prof.get("model_name") or "default"

    @property
    def active_binding_name(self) -> str:
        alias = self.active_profile_alias
        prof = self.llm_model_profiles.get(alias, {})
        b_alias = prof.get("binding_profile_name") or prof.get("binding_alias")
        b_data = self.llm_binding_profiles.get(b_alias, {}) if b_alias else {}
        return b_data.get("binding_name") or prof.get("binding_name") or "ollama"

    @classmethod
    def load(cls, cli_args: argparse.Namespace) -> "ShellAgentConfig":
        config = cls()
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        PROFILE_DATA_KEYS = ("llm_binding_profiles", "llm_model_profiles")

        # 1. Preferences from JSON (never profile data)
        if APP_CONFIG_FILE.exists():
            try:
                file_config = json.loads(APP_CONFIG_FILE.read_text(encoding="utf-8"))
                for key, val in file_config.items():
                    if key in PROFILE_DATA_KEYS:
                        continue
                    if hasattr(config, key):
                        setattr(config, key, val)
            except (OSError, json.JSONDecodeError) as e:
                if config.debug:
                    ASCIIColors.warning(f"Failed to read config file: {e}")

        # 2. Resolve profiles from canonical config sources
        from lollms_client.lollms_config_cli_env import (
            load_env_file,
            load_yaml_file,
            load_json_file,
            _descend_into_entry,
            _flatten_dict_to_env,
            _extract_bindings_from_env,
            _extract_profiles_from_env,
            _load_existing_env_to_map,
        )

        resolved_env = dict(os.environ)
        explicit_config = getattr(cli_args, "config_path", None)

        if explicit_config:
            p = Path(explicit_config).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Configuration file not found: {p}")
            try:
                if p.suffix == ".env":
                    resolved_env.update(load_env_file(p))
                elif p.suffix == ".json":
                    data = _descend_into_entry(load_json_file(p), None)
                    resolved_env.update(_flatten_dict_to_env(data))
                elif p.suffix in (".yaml", ".yml"):
                    resolved_env.update(_flatten_dict_to_env(load_yaml_file(p)))
            except (OSError, ValueError) as e:
                if config.debug:
                    ASCIIColors.warning(f"Failed to parse {p}: {e}")
        else:
            resolved_env.update(_load_existing_env_to_map())

        for modality in ("llm",):
            prefix = modality.upper()
            bindings = _extract_bindings_from_env(prefix, resolved_env)
            profiles = _extract_profiles_from_env(prefix, bindings, resolved_env)
            setattr(config, f"{modality}_binding_profiles", bindings)
            setattr(config, f"{modality}_model_profiles", profiles)

        # 3. CLI profile override
        explicit_profile = (
            getattr(cli_args, "llm_profile", None)
            or getattr(cli_args, "profile", None)
        )
        if explicit_profile:
            selected_alias = str(explicit_profile).strip()
            if selected_alias and selected_alias not in config.llm_model_profiles:
                lowered = selected_alias.lower()
                for known in config.llm_model_profiles:
                    if known.lower() == lowered:
                        selected_alias = known
                        break
            if selected_alias in config.llm_model_profiles:
                config.active_profile = selected_alias
            else:
                ASCIIColors.warning(
                    f"[CLI] --llm-profile '{explicit_profile}' unknown. "
                    f"Available: {list(config.llm_model_profiles.keys())}"
                )

        # Fallback scaffolding if config is empty
        if not config.llm_binding_profiles:
            config.llm_binding_profiles["default"] = {
                "binding_name": "ollama",
                "binding_config": {"host_address": "http://localhost:11434"},
                "is_default": True,
            }
        if not config.llm_model_profiles:
            fallback_binding_alias = next(iter(config.llm_binding_profiles))
            config.llm_model_profiles["default"] = {
                "binding_profile_name": fallback_binding_alias,
                "binding_alias": fallback_binding_alias,
                "is_default": True,
                "forced_context_size": 8192,
            }

        if cli_args.max_steps:
            config.max_reasoning_steps = cli_args.max_steps
        if cli_args.temperature is not None:
            config.temperature = cli_args.temperature
        if cli_args.max_tokens:
            config.max_tokens_per_turn = cli_args.max_tokens
        if getattr(cli_args, "reasoning_effort", None):
            config.reasoning_effort = cli_args.reasoning_effort
        if getattr(cli_args, "dynamic_effort", False):
            config.dynamic_effort = True
        if cli_args.debug is not None:
            config.debug = cli_args.debug
        if cli_args.no_shell_execution:
            config.enable_shell_execution = False
        if cli_args.shell_autonomy:
            config.shell_autonomy_level = cli_args.shell_autonomy
        if cli_args.no_memory:
            config.enable_memory = False
        if cli_args.handbag_path:
            config.handbag_path = str(Path(cli_args.handbag_path).resolve())
        return config

    def save(self):
        APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PROFILE_KEYS = ("llm_binding_profiles", "llm_model_profiles")
        data = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in PROFILE_KEYS
        }
        try:
            APP_CONFIG_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            ASCIIColors.warning(f"Failed to save config: {e}")


# ── CLIENT + PERSONALITY SETUP ─────────────────────────────────────────────

# Destructive-command patterns. The SOUL.md doctrine tells the LLM to ask the
# user in prose before running one of these, but belt-and-suspenders: the
# confirm handler ALSO intercepts them and forces a prompt even when the
# autonomy level is `full_access`. This protects against LLM drift and against
# an over-eager ShoULD-not line in a custom handbag.
_DESTRUCTIVE_CMD_PATTERNS = re.compile(
    r"(^|[;&|`\n])\s*("
    r"rm|rmdir|del|erase|remove-item|rm\b|"       # file/dir deletion
    r"mv|move|rename|ren|move-item|"              # moving onto existing paths
    r"mkfs|format|diskpart|fdisk|parted|dd|"      # disk operations
    r"shutdown|reboot|halt|restart-computer|stop-computer|"
    r"reg\s+(add|delete|import|load)|setx|"        # registry / env changes
    r"sc\s+(create|delete|config|stop)|net\s+(user|localgroup)|"
    r"takeown|icacls|chmod|chown|"                # permissions
    r"pip\s+uninstall|npm\s+uninstall|apt\s+(remove|purge)|choco\s+uninstall|"
    r"winget\s+uninstall|scoop\s+uninstall"        # package removal
    r")\b",
    re.IGNORECASE,
)


def _looks_destructive(command: str) -> bool:
    """True if the command matches a destructive pattern (see pattern list)."""
    if not command:
        return False
    return bool(_DESTRUCTIVE_CMD_PATTERNS.search(command))


def _make_confirm_handler(config: ShellAgentConfig):
    """Confirmation callback for LCP tool execution.

    Two responsibilities:
      1. **Doctrinal safety net** — destructive commands ALWAYS prompt, even
         in `full_access` mode. This is the belt to SOUL.md's suspenders.
      2. **Autonomy bridge** — in `full_access`, non-destructive commands are
         auto-approved without prompting. In `safe`/`strict`, the tool itself
         gates them and calls us.
    """
    def _handler(*args, **kwargs) -> tuple[str, str]:
        command = ""
        if len(args) >= 1 and isinstance(args[0], dict):
            command = args[0].get("command", "") or args[0].get("code", "")
        elif len(args) >= 1:
            command = str(args[0])
        elif kwargs:
            command = kwargs.get("command", "") or kwargs.get("code", "")

        # In full_access, non-destructive commands flow through silently.
        if config.shell_autonomy_level == "full_access" and not _looks_destructive(command):
            return "allow", ""

        # Non-interactive: refuse anything that would have prompted.
        if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            return "reject", "Non-interactive environment: cannot prompt; refusing."

        title = (
            "[bold red]⚠️ DESTRUCTIVE COMMAND — confirm[/bold red]"
            if _looks_destructive(command)
            else "[bold yellow]🛡️ Execution Authorization[/bold yellow]"
        )
        ASCIIColors.panel(
            f"[bold cyan]Command:[/bold cyan] [yellow]{command or '(see request)'}[/yellow]\n"
            f"[dim]The agent wants to run this in your shell.[/dim]",
            title=title,
            border_style="red" if _looks_destructive(command) else "yellow",
        )
        while True:
            try:
                choice = input("  Authorize? [y]es / [n]o (default n): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "reject", "Execution cancelled by user."
            if choice in ("y", "yes"):
                return "allow", ""
            if choice in ("", "n", "no"):
                return "reject", "User declined execution."
            ASCIIColors.yellow("  Invalid choice. Enter y or n.")

    return _handler


_FORBIDDEN_SHELL_TOOL_NAMES = (
    "tool_execute_shell_command",
    "tool_execute_shell_command_prompt",
)


def _prune_personality_tools(personality: LollmsPersonality, config: "ShellAgentConfig") -> None:
    """Removes stateless shell tools from the personality's LLM-facing registries.

    The personality constructor (`LollmsPersonality.from_handbag`) re-discovers
    tools via its own scan, so pruning only `client.tools.*` at `create_client()`
    time is insufficient — the personality may re-register stateless shell tools.
    This runs AFTER construction and scrubs every plausible attribute:

      * ``_discovered_tools`` dict
      * ``_available_tools`` dict
      * ``_tool_registry`` dict
      * the per-call discovery result (via ``personality._discover_tools(None, [])``)
      * any other dict-valued attribute whose keys match forbidden names

    Idempotent, defensive.
    """
    pruned: list[str] = []

    def _scrub(mapping: Any, label: str) -> None:
        if not isinstance(mapping, dict):
            return
        for name in list(mapping.keys()):
            if name in _FORBIDDEN_SHELL_TOOL_NAMES:
                try:
                    del mapping[name]
                    pruned.append(f"{label}:{name}")
                except Exception:
                    pass

    # Direct attribute sweep (the concrete attribute names the personality
    # uses in this codebase, plus a broad sweep of dict-valued attributes for
    # future-proofing).
    for attr in list(vars(personality).keys()):
        val = getattr(personality, attr, None)
        if isinstance(val, dict):
            _scrub(val, attr)

    # `_discover_tools` returns the CURRENT map the chat loop will use — after
    # our direct attribute sweep it may still contain the forbidden names
    # because it caches internally; scrub the result once more.
    try:
        discovered = personality._discover_tools(None, [])  # type: ignore[attr-defined]
        _scrub(discovered, "_discover_tools")
    except (AttributeError, TypeError, RuntimeError):
        pass

    if pruned and config.debug:
        ASCIIColors.warning(
            f"[CLI] Pruned stateless shell tool(s) from personality registries: "
            f"{', '.join(pruned)}"
        )


def _verify_tool_wiring(expected_session: "PersistentShellSession") -> None:
    """Sanity-check that `tool_shell_session_execute` resolves the real session.

    Runs a no-op invocation of the tool as the LLM would call it, and asserts
    that:
      * the tool sees a session at all (not the "no session attached" error),
      * the tool returned cwd equal to the live worker cwd.

    Failure is LOUD and NON-debug-gated — a silent wiring failure leaves the
    user facing "no persistent shell session is attached" on every command.
    Every resolution path (`sys` attribute, persistent_shell registry,
    module-local `_SESSION`, per-copy attachments) is independently checked
    so the diagnostic pinpoints WHICH link broke.
    """
    # Gather a fingerprint of every resolution source we depend on.
    import sys as _sys

    sys_attr_present = hasattr(_sys, "_lollms_shell_active_session")
    try:
        from lollms_client.apps.lollms_shell import persistent_shell as _ps
        registry_present = _ps.get_active_session() is expected_session
    except Exception:
        registry_present = False

    # Attempt the same call the LLM will make.
    tool_result = None
    tool_error: Optional[str] = None
    try:
        from lollms_client.apps.lollms_shell.tools import shell_session as _canon
        tool_result = _canon.tool_shell_session_execute("", explanation="startup-wiring-check")
    except Exception as e:
        tool_error = str(e)

    # Success criteria: the tool must NOT return our wiring-bug marker.
    wiring_failed = (
        tool_result is None
        or not isinstance(tool_result, dict)
        or "No persistent shell session is attached" in str(tool_result.get("output", ""))
    )

    if not wiring_failed:
        if sys_attr_present and registry_present:
            return  # All good — silent success, no noise on the happy path.
        # Tool works but one registry source is missing — non-fatal but worth a
        # short note so future divergence is caught early.
        ASCIIColors.info(
            "[CLI] ℹ️  Persistent shell tool operational"
            + ("" if sys_attr_present else " (sys attribute missing — atypical)")
            + ("" if registry_present else " (persistent_shell registry mismatch)")
        )
        return

    # ── LOUD FAILURE ──
    ASCIIColors.panel(
        "[bold red]The persistent shell tool cannot see the session.[/bold red]\n\n"
        f"sys._lollms_shell_active_session present : {sys_attr_present}\n"
        f"persistent_shell.get_active_session() ok : {registry_present}\n"
        f"tool call raised                         : {tool_error or '(no)'}\n"
        f"tool returned                            : {tool_result}\n\n"
        "Shell commands will fail with 'No persistent shell session attached'\n"
        "until this is fixed. Likely causes:\n"
        "  · the installed handbag's tools/shell_session.py is a stale copy\n"
        "    from a previous release (delete it — it will be re-seeded);\n"
        "  · a loader imported shell_session under an unexpected module name\n"
        "    that bypassed attach_session; restart lollms-shell to retry.",
        title="[bold red]⚠️  SHELL TOOL WIRING FAILED[/bold red]",
        border_style="red",
    )


def _rewrite_forbidden_tool_references(text: str) -> str:
    """Rewrites any mention of the stateless tool name in text to our tool name.

    Two things happen inside a running agent that references the wrong tool:
      (a) The doctrine itself was written using the old name — rewriting the
          prompt makes the doctrine consistent with the mounted tool.
      (b) Any leftover context (pinned lessons, memory, previous-turn history
          continued across the session) may name the old tool. Rewriting the
          system prompt prevents it from resurfacing to the LLM.

    Args:
        text: the system prompt (or any string) to rewrite.

    Returns:
        the text with `tool_execute_shell_command` replaced by
        `tool_shell_session_execute`, and `tool_execute_shell_command_prompt`
        similarly rewritten. The longer name is rewritten first to avoid
        partial-match corruption.
    """
    if not text:
        return text or ""
    # Longer name FIRST — otherwise `tool_execute_shell_command_prompt`
    # becomes `tool_shell_session_execute_prompt` (a nonexistent tool).
    text = text.replace(
        "tool_execute_shell_command_prompt", "tool_shell_session_execute"
    )
    text = text.replace("tool_execute_shell_command", "tool_shell_session_execute")
    return text


def _prune_foreign_shell_tools(client: LollmsClient, config: "ShellAgentConfig") -> None:
    """Removes any *statically spawned* shell tool from the client's registry.

    The LCP binding's default-toolset scan can pre-register `system_shell`
    before our `mount_tool_library_if_absent("shell_session")` runs, and in
    some client paths the mount happens on the library object AFTER the
    LLM-facing tool list was already captured. This function prunes every
    plausible registry (`client.tools.available_tools`, `.tools`,
    `.mounted_libraries[...]`, tool-name mappings) so the LLM only ever sees
    `tool_shell_session_execute`.

    Idempotent and defensive — any missing/renamed attribute is skipped.
    """
    forbidden = {
        "tool_execute_shell_command",
        "tool_execute_shell_command_prompt",
    }
    pruned: list[str] = []

    def _scrub(mapping: Any, label: str) -> None:
        if not isinstance(mapping, dict):
            return
        for name in list(mapping.keys()):
            if name in forbidden:
                try:
                    del mapping[name]
                    pruned.append(f"{label}:{name}")
                except Exception:
                    pass

    tools_binding = getattr(client, "tools", None)
    if tools_binding is None:
        return

    for attr in (
        "available_tools",
        "tools",
        "tool_registry",
        "_tools",
        "_available_tools",
        "_tool_registry",
        "tools_dict",
    ):
        _scrub(getattr(tools_binding, attr, None), attr)

    # Each mounted library also has its own tool surface — scrub those too.
    libs = getattr(tools_binding, "mounted_libraries", None) or {}
    if isinstance(libs, dict):
        for lib_name, lib in list(libs.items()):
            for attr in ("tools", "available_tools", "_tools", "TOOLS"):
                _scrub(getattr(lib, attr, None), f"{lib_name}.{attr}")

    if pruned:
        msg = ", ".join(pruned)
        if config.debug:
            ASCIIColors.warning(
                f"[CLI] Removed static shell tool(s) from the LLM registry: {msg} "
                f"(persistent session tool is authoritative)."
            )
        else:
            ASCIIColors.info(
                f"[CLI] ⛔ Stripped stateless shell tool(s) ({msg}) — "
                f"only tool_shell_session_execute remains."
            )


def _handbag_tools_dir(config: "ShellAgentConfig") -> Optional[Path]:
    """Returns the handbag's `tools/` directory if it exists, else None.

    The handbag-resident `tools/shell_session.py` exposes the persistent
    shell tool. It must be added to the LCP `tools_folders` BEFORE the client
    is constructed so the loader discovers it.
    """
    try:
        tools_dir = Path(config.handbag_path).resolve() / "tools"
    except (OSError, ValueError):
        return None
    if tools_dir.is_dir() and any(tools_dir.glob("*.py")):
        return tools_dir
    return None


def create_client(config: ShellAgentConfig) -> LollmsClient:
    active_model_alias = config.active_profile_alias
    active_model_profile = config.llm_model_profiles.get(active_model_alias, {})
    binding_alias = (
        active_model_profile.get("binding_profile_name")
        or active_model_profile.get("binding_alias")
    )
    if not binding_alias or binding_alias not in config.llm_binding_profiles:
        for prof in config.llm_model_profiles.values():
            if prof.get("is_default"):
                binding_alias = prof.get("binding_profile_name") or prof.get("binding_alias")
                break
    if not binding_alias or binding_alias not in config.llm_binding_profiles:
        if config.llm_binding_profiles:
            binding_alias = next(iter(config.llm_binding_profiles))

    binding_profile = config.llm_binding_profiles.get(binding_alias, {}) if binding_alias else {}
    binding_config = binding_profile.get("binding_config", {}).copy()

    active_model_name = active_model_profile.get("model_name") or config.active_model_name
    active_binding_name = binding_profile.get("binding_name") or config.active_binding_name

    llm_config: dict[str, Any] = {
        "model_name": active_model_name,
        "host_address": binding_config.get("host_address", "http://localhost:11434"),
        "verify_ssl_certificate": binding_config.get("verify_ssl_certificate", False),
    }
    if binding_config.get("service_key"):
        llm_config["service_key"] = binding_config["service_key"]

    import lollms_client as _lc_pkg
    package_root = Path(_lc_pkg.__file__).resolve().parent
    default_tools_path = package_root / "tools_bindings" / "lcp" / "default_tools"
    tools_folders: list[str] = [str(default_tools_path)] if default_tools_path.exists() else []

    # Register the handbag's `tools/` folder — this is where our persistent
    # `shell_session.py` lives. Without this line, the LCP loader never sees
    # our stateful tool and the agent keeps using the stateless `system_shell`
    # one (which spawns a throwaway subprocess per call and cannot `cd`).
    handbag_tools = _handbag_tools_dir(config)
    if handbag_tools is not None:
        tools_folders.append(str(handbag_tools))

    lollms_system_dir = (Path.home() / ".lollms_client").resolve()
    lollms_system_dir.mkdir(parents=True, exist_ok=True)

    confirm_handler = _make_confirm_handler(config)
    # `system_shell` is DELIBERATELY DISABLED. lollms_shell ships its own
    # stateful `shell_session` tool; the stateless `system_shell` toolset
    # spawns a fresh subprocess per call, so `cd` / `set VAR` / `activate`
    # would not survive between calls — exactly the bug we removed.
    # `disabled: True` is the LCP tools binding's documented opt-out signal;
    # additionally `_prune_foreign_shell_tools` scrubs any residue that
    # slips into the LLM-facing registry after personality construction.
    host_tool_configs: dict[str, dict[str, Any]] = {
        "system_shell": {
            "disabled": True,
            "autonomy_level": config.shell_autonomy_level,
            "confirm_handler": confirm_handler,
        },
    }

    client_kwargs: dict[str, Any] = {
        "tools_binding_name": "lcp",
        "tools_binding_config": {
            "tools_folders": tools_folders,
            "host_tool_configs": host_tool_configs,
            "confirm_handler": confirm_handler,
            "system_dir": str(lollms_system_dir),
            "cwd": str(lollms_system_dir),
        },
        "system_dir": str(lollms_system_dir),
        "debug": config.debug,
    }

    if config.llm_binding_profiles and config.llm_model_profiles:
        client_kwargs["llm_binding_profiles"] = config.llm_binding_profiles
        client_kwargs["llm_model_profiles"] = config.llm_model_profiles
    else:
        client_kwargs["llm_binding_name"] = active_binding_name
        client_kwargs["llm_binding_config"] = llm_config

    client = LollmsClient(**client_kwargs)
    try:
        client.enable_fast_token_estimate()
    except Exception:
        pass

    if client.tools:
        # Mount our handbag-provided persistent-session library.
        # The stock `system_shell` toolset is intentionally NOT mounted: it
        # spawns a fresh subprocess per call, so `cd` / `set VAR` / `activate`
        # cannot survive between tool calls — the exact bug we're fixing.
        lib_name = "shell_session"
        try:
            mounted = False
            if hasattr(client.tools, "mount_tool_library_if_absent"):
                client.tools.mount_tool_library_if_absent(lib_name)
                mounted = True
            else:
                client.tools.mount_tool_library(lib_name)
                mounted = True
            # Defensive: explicitly unmount system_shell if it slipped in via
            # any default configuration. Two shell tools with different state
            # models is a recipe for LLM confusion.
            for stale in ("system_shell",):
                libs = getattr(client.tools, "mounted_libraries", None) or {}
                if stale in libs:
                    try:
                        unmount = getattr(client.tools, "unmount_tool_library", None)
                        if callable(unmount):
                            unmount(stale)
                        else:
                            libs.pop(stale, None)
                    except Exception:
                        pass
            if config.debug and mounted:
                ASCIIColors.success(f"[CLI] ✅ {lib_name} library mounted (persistent shell).")
        except (AttributeError, OSError, RuntimeError) as e:
            if config.debug:
                ASCIIColors.warning(f"Failed to mount {lib_name}: {e}")

        # ── FALLBACK-PROOF: prune any stale `tool_execute_shell_command` ──
        # The LCP binding may have pre-exposed `system_shell` tools during
        # client construction (they get registered before `mount_*` runs).
        # We remove them from every plausible tool registry on the client so
        # the LLM has exactly ONE shell tool: `tool_shell_session_execute`.
        _prune_foreign_shell_tools(client, config)

    return client


def ensure_handbag_structure(config: ShellAgentConfig):
    handbag_path = Path(config.handbag_path).resolve()
    default_handbag_path = (APP_DEFAULT_HANDBAG_DIR / "default_shell").resolve()
    is_default = handbag_path == default_handbag_path
    handbag_path.mkdir(parents=True, exist_ok=True)
    soul_path = handbag_path / "SOUL.md"

    if is_default:
        # The packaged `default_handbag/SOUL.md` is the SINGLE SOURCE OF TRUTH.
        # We do NOT keep an inline copy here anymore: an inline constant silently
        # shadowed every fix to the packaged file and shipped an outdated
        # doctrine (e.g. telling the LLM to use the stateless
        # `tool_execute_shell_command` instead of the persistent
        # `tool_shell_session_execute`).
        soul_content = _SOUL_METADATA_HEADER + _load_packaged_soul_body()
        try:
            installed_bytes = soul_path.read_bytes() if soul_path.exists() else b""
        except OSError:
            installed_bytes = b""
        if installed_bytes != soul_content.encode("utf-8"):
            backup = soul_path.with_suffix(".md.bak")
            try:
                if soul_path.exists():
                    shutil.copy2(soul_path, backup)
            except OSError:
                backup = None
            soul_path.write_text(soul_content, encoding="utf-8")
            ASCIIColors.info(
                "[CLI] Refreshed handbag SOUL.md doctrine"
                + (f" (previous copy backed up to {backup.name})" if backup else "")
                + "."
            )
    else:
        if not soul_path.exists():
            ASCIIColors.warning(f"[CLI] Custom handbag at {handbag_path} has no SOUL.md. Creating baseline.")
            name = handbag_path.name.replace("_", " ").title()
            soul_path.write_text(
                f'---\nname: "{name}"\nauthor: "User"\ncategory: "custom"\n'
                f'description: "Custom shell personality for {name}."\n---\n\n'
                f"You are {name}, a specialized shell copilot.\n",
                encoding="utf-8",
            )

    (handbag_path / "tools").mkdir(exist_ok=True)
    (handbag_path / "skills").mkdir(exist_ok=True)
    (handbag_path / "memory").mkdir(exist_ok=True)

    # Copy the packaged `tools/shell_session.py` into the handbag. This is the
    # persistent-session shell tool. We FORCE-OVERWRITE it whenever the
    # installed copy differs from the packaged one — otherwise a stale copy
    # from an earlier install would silently shadow every fix to the tool
    # (module-level `_SESSION` shipped BEFORE the process-global registry
    # fallback was added would keep claiming "no session attached" even
    # though `main()` publishes one). Only for the *default* handbag; a
    # custom handbag keeps whatever session tool the user installed.
    try:
        dest_tools = handbag_path / "tools"
        dest_tools.mkdir(exist_ok=True)
        packaged_tools = Path(__file__).resolve().parent / "default_handbag" / "tools"
        if packaged_tools.is_dir():
            for src_file in packaged_tools.glob("*.py"):
                dst_file = dest_tools / src_file.name
                try:
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
                    elif is_default:
                        # Content-hash compare; overwrite when the installed
                        # shim is stale (a prior release of the tool).
                        src_bytes = src_file.read_bytes()
                        try:
                            dst_bytes = dst_file.read_bytes()
                        except OSError:
                            dst_bytes = b""
                        if src_bytes != dst_bytes:
                            backup = dst_file.with_suffix(dst_file.suffix + ".bak")
                            try:
                                shutil.copy2(dst_file, backup)
                            except OSError:
                                backup = None
                            shutil.copy2(src_file, dst_file)
                            ASCIIColors.info(
                                f"[CLI] Refreshed handbag tool '{dst_file.name}'"
                                + (f" (previous copy backed up to {backup.name})" if backup else "")
                                + "."
                            )
                except OSError as inner:
                    if config.debug:
                        ASCIIColors.warning(f"[CLI] Could not seed {dst_file.name}: {inner}")
    except OSError as e:
        ASCIIColors.warning(f"[CLI] Could not seed handbag tools: {e}")


def create_shell_personality(
    config: ShellAgentConfig,
    client: LollmsClient,
    session: Optional["PersistentShellSession"] = None,
) -> LollmsPersonality:
    ASCIIColors.rich_print("\n[bold cyan]🔧 Initializing shell companion...[/bold cyan]")

    ASCIIColors.rich_print("  [dim]📂 Ensuring handbag structure...[/dim]", end="")
    ensure_handbag_structure(config)
    ASCIIColors.rich_print(" [green]✓[/green]")

    # Wire the persistent shell session into the handbag-provided tool. This
    # is what makes `cd`, `set VAR`, `activate` etc. survive between calls.
    if session is not None:
        _wire_persistent_session_tool(session, config)

    caps = CapabilityFlags(
        enable_sub_agents=False,
        enable_model_switching=False,
        enable_skill_creation=False,
        enable_skill_loading=False,
        enable_workspace_tools=False,
        allow_computer_use=False,
        skills_mode="off",
        enable_image_generation=False,
        enable_image_editing=False,
        enable_tts=False,
        enable_stt=False,
    )

    ASCIIColors.rich_print("  [dim]🧠 Loading handbag & building personality...[/dim]", end="")
    personality = LollmsPersonality.from_handbag(config.handbag_path, lollms_client=client)
    personality.lollms_client = client
    personality.capabilities = caps
    personality.max_tokens_per_turn = config.max_tokens_per_turn
    personality.debug_mode = config.debug

    # Inject per-turn environment block (CWD + shell + venv). Refreshed each turn
    # by run_interactive() so a `cd` performed by the agent (via the persistent
    # shell worker) is reflected in the next prompt AND in the agent's view.
    # We seed with the WORKER's cwd when a session is wired, falling back to
    # the REPL process cwd only if no session exists.
    seed_cwd = _session_cwd(session) if session is not None else Path.cwd()
    # Sanity: `_session_cwd` may return a mangled path if the shell misbehaved.
    # Require absolute + existing dir; otherwise default to the REPL's cwd.
    try:
        if not seed_cwd.is_absolute() or not seed_cwd.exists() or not seed_cwd.is_dir():
            seed_cwd = Path.cwd()
    except OSError:
        seed_cwd = Path.cwd()

    # HARD PRUNE: any stateless shell tool that survived the earlier registry
    # cleanup (the personality constructor re-discovers tools AFTER
    # `create_client()` prunes) MUST NOT reach the LLM. Otherwise the model
    # will use `tool_execute_shell_command` (fresh subprocess, `cd` lost) and
    # conclude — correctly! — that state does not persist, contradicting the
    # doctrine. We scrub every registry the personality exposes AND any
    # prompt fragment that still references the forbidden tool name.
    _prune_personality_tools(personality, config)

    env_context = build_environment_context(seed_cwd, session=session)
    if "=== ENVIRONMENT CONTEXT ===" not in personality.system_prompt:
        personality.system_prompt += "\n\n" + env_context

    # Also scrub the injected doctrine itself: if the system prompt currently
    # references the forbidden tool (e.g. leftover context from a previous
    # launch that used an old SOUL.md), rewrite it in place so the model has
    # no reason to choose the wrong tool.
    personality.system_prompt = _rewrite_forbidden_tool_references(
        personality.system_prompt or ""
    )

    # The workspace_path property calls .mkdir() with parents=True on whatever
    # we pass — a garbage seed (echo-polluted path) would raise WinError 123
    # and abort startup. lollms_shell does not need a real workspace tree, so
    # falling back to the REPL's cwd is always acceptable.
    try:
        personality.workspace_path = seed_cwd
    except (OSError, ValueError) as e:
        if config.debug:
            ASCIIColors.warning(f"[CLI] workspace_path seed rejected ({e}); using REPL cwd.")
        try:
            personality.workspace_path = Path.cwd()
        except (OSError, ValueError):
            pass
    ASCIIColors.rich_print(f" [green]✓[/green] [dim]({personality.name})[/dim]")

    # ── Project-local memory (per working directory) ──
    if config.enable_memory:
        ASCIIColors.rich_print("  [dim]💾 Initializing shell memory...[/dim]", end="")
        try:
            from lollms_client.lollms_memory import LollmsMemoryManager, MemoryConfig

            proj_tag = Path.cwd().name or "shell"
            memory_db = Path.home() / ".lollms" / "lollms_shell" / "memory" / f"{proj_tag}.db"
            memory_db.parent.mkdir(parents=True, exist_ok=True)

            personality.memory_manager = LollmsMemoryManager(
                db_path=f"sqlite:///{memory_db}",
                owner_id=f"shell_{proj_tag}",
                config=MemoryConfig(working_token_budget=1500),
            )
            try:
                personality.memory_manager.deduplicate_all()
            except Exception:
                pass
            ASCIIColors.rich_print(f" [green]✓[/green] [dim]({memory_db.name})[/dim]")
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            ASCIIColors.rich_print(" [red]✗[/red]")
            ASCIIColors.warning(f"[CLI] Memory init failed: {e}. Continuing without memory.")

    ASCIIColors.rich_print("[bold green]  ✅ Shell companion ready.[/bold green]\n")
    return personality


# ── RENDERING ──────────────────────────────────────────────────────────────

class ShellStreamRenderer:
    """Streaming renderer for the shell agent — clean panels, no markdown fences."""

    def __init__(self, config: ShellAgentConfig):
        self.config = config

    def _render_event(self, msg_type: Any, meta: dict | None):
        if not meta:
            return

        def _is(evt):
            if msg_type == evt:
                return True
            if hasattr(msg_type, "value") and hasattr(evt, "value"):
                return msg_type.value == evt.value
            return False

        try:
            if _is(MSG_TYPE.MSG_TYPE_TOOL_START):
                tool_name = meta.get("tool_name", "tool")
                params = meta.get("parameters", {}) or {}
                command = params.get("command") if isinstance(params, dict) else None
                body = f"[bold cyan]$ {command}[/bold cyan]" if command else "[dim]Preparing...[/dim]"
                ASCIIColors.panel(
                    body,
                    title=f"[bold blue]🛠️ {tool_name}[/bold blue]",
                    border_style="blue",
                )
                sys.stdout.flush()

            elif _is(MSG_TYPE.MSG_TYPE_TOOL_END):
                tool_name = meta.get("tool_name", "tool")
                success = meta.get("success", False)
                output = meta.get("output", "") or ""
                error = meta.get("error") or ""
                log = str(output if success else (error or output or "(no output)")).splitlines()
                max_lines = 40
                if len(log) > max_lines:
                    log = log[:20] + [f"... [{len(log) - max_lines} lines omitted] ..."] + log[-20:]
                color = "green" if success else "red"
                ASCIIColors.panel(
                    "\n".join(log) if log else "(no output)",
                    title=f"[bold {color}]{'✅' if success else '❌'} {tool_name}[/bold {color}]",
                    border_style=color,
                )
                sys.stdout.flush()

            elif _is(MSG_TYPE.MSG_TYPE_INFO):
                if meta.get("done_intercepted"):
                    print()
                    ASCIIColors.rule("[bold green]✅ Done[/bold green]")
                    sys.stdout.flush()
                    return

        except Exception as e:
            if self.config.debug:
                ASCIIColors.warning(f"[Renderer] {e}")

    def __call__(self, chunk: str, msg_type: Any = None, meta: dict | None = None) -> bool:
        if msg_type is None:
            msg_type = MSG_TYPE.MSG_TYPE_CHUNK

        if msg_type in (
            MSG_TYPE.MSG_TYPE_TOOL_START,
            MSG_TYPE.MSG_TYPE_TOOL_END,
            MSG_TYPE.MSG_TYPE_INFO,
        ):
            self._render_event(msg_type, meta)
            return True

        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if meta and meta.get("tool_progress"):
                ASCIIColors.rich_print(f"  [dim]│ {chunk}[/dim]")
                sys.stdout.flush()
                return True
            if meta and (meta.get("was_processed") or meta.get("live_tool_chunk")):
                return True
            if ("<done" in chunk and "/>" in chunk):
                return True
            ASCIIColors.rich_print(chunk, end="")
            sys.stdout.flush()
        return True


# ── MAIN LOOP ──────────────────────────────────────────────────────────────

def run_interactive(
    personality: LollmsPersonality,
    client: LollmsClient,
    config: ShellAgentConfig,
    session: "PersistentShellSession",
) -> int:
    _render_startup_banner(config, personality, session=session)
    ASCIIColors.rich_print(
        "[dim]Ask me to do something in your shell (e.g. 'list the files here', "
        "'why is my disk full?', 'copy report.pdf to ~/Documents'). "
        "Type 'exit' to quit. State (cwd, env vars, venv) persists across turns "
        "as in a real terminal.[/dim]\n"
    )

    renderer = ShellStreamRenderer(config)
    confirm_handler = _make_confirm_handler(config)

    while True:
        # Refresh CWD / env awareness before each user turn so a `cd` the agent
        # performed via the persistent worker is reflected in the agent's
        # prompt-block AND in the user's typed prompt.
        _refresh_environment_block(personality, session)

        # Live shell-style prompt: reflects the WORKER's current directory
        # (and active venv) on every turn, just like the user's own terminal.
        try:
            user_input = input("\n" + _build_shell_prompt(session)).strip()
        except (EOFError, KeyboardInterrupt):
            ASCIIColors.cyan("\n👋 Goodbye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            ASCIIColors.cyan("👋 Goodbye!")
            return 0
        if user_input.lower() in ("/cwd", "pwd"):
            live_cwd = _session_cwd(session)
            ASCIIColors.info(f"  CWD (worker): {live_cwd}")
            ASCIIColors.info(f"  CWD (REPL):   {Path.cwd()}")
            venv = _active_venv()
            if venv:
                ASCIIColors.info(f"  Venv: {venv}")
            continue
        if user_input.lower() == "/memory":
            _show_memory(personality)
            continue
        if user_input.lower() == "/reset":
            try:
                session.reset(Path.cwd())
                _refresh_environment_block(personality, session)
                ASCIIColors.green("  ✓ Shell session reset (cwd, env vars, venv).")
            except Exception as e:
                ASCIIColors.red(f"  ✗ Reset failed: {e}")
            continue
        if user_input.lower() in ("/help", "help", "?"):
            _show_help()
            continue

        ASCIIColors.rule("[bold green]🤖 Shell Agent[/bold green]")
        ASCIIColors.rich_print("")

        try:
            result = personality.chat(
                prompt=user_input,
                lollms_client=client,
                streaming_callback=renderer,
                max_reasoning_steps=config.max_reasoning_steps,
                temperature=config.temperature,
                n_predict=config.max_tokens_per_turn,
                enable_artefacts=False,
                use_internal_history=True,
                enable_shell=config.enable_shell_execution,
                enable_python_exec=False,
                enable_workspace_tools=False,
                allow_computer_use=False,
                event_mode=EventMode.FULL_CALLBACK_MODE,
                enforce_end_tag=True,
                shell_autonomy_level=config.shell_autonomy_level,
                auto_approve_python=False,
                confirm_handler=confirm_handler,
                debug=config.debug,
                debug_export=False,
                reasoning_effort=config.reasoning_effort,
                dynamic_effort=config.dynamic_effort,
            )
        except KeyboardInterrupt:
            if hasattr(client, "cancel"):
                client.cancel()
            ASCIIColors.yellow("\n\n⚠️  Cancelled.")
            continue
        except (RuntimeError, ValueError, OSError, ConnectionError) as e:
            if config.debug:
                trace_exception(e)
            ASCIIColors.red(f"\n💥 Error: {e}")
            continue

        print()
        elapsed = result.get("rounds", 0)
        ASCIIColors.rich_print(
            f"[dim]⏱️  {elapsed} round(s) | tools: {len(result.get('tool_calls', []))}[/dim]"
        )


def _refresh_environment_block(
    personality: LollmsPersonality,
    session: Optional["PersistentShellSession"] = None,
) -> None:
    """Replaces the ENVIRONMENT CONTEXT block with a fresh one.

    Reads **the worker shell's** cwd *and* its env (VIRTUAL_ENV, CONDA_PREFIX,
    PATH, …) on every call. That is what makes `activate` / `deactivate` /
    `set VAR=val` / `export VAR` performed by the agent show up in the live
    system prompt on the very next turn.

    Uses a CALLABLE replacement because ``re.sub`` parses backslash escapes in
    string replacements — Windows paths like ``C:\\Users`` contain ``\\U``,
    which raises ``re.error: bad escape \\U``. A callable is not escape-parsed.

    The CWD comes from the session (persistent worker) when available; a `cd`
    the agent performed via the worker tool therefore propagates into the
    agent's own view on the next turn, staying consistent with the prompt.
    """
    new_block = build_environment_context(_session_cwd(session), session=session)
    prompt = personality.system_prompt or ""
    pattern = re.compile(
        r"=== ENVIRONMENT CONTEXT ===.*?=== END ENVIRONMENT CONTEXT ===",
        re.DOTALL,
    )
    if pattern.search(prompt):
        personality.system_prompt = pattern.sub(lambda _m: new_block, prompt)
    else:
        personality.system_prompt = prompt.rstrip() + "\n\n" + new_block


def _show_memory(personality: LollmsPersonality) -> None:
    try:
        if not getattr(personality, "memory_manager", None):
            ASCIIColors.yellow("  Memory is disabled or not initialized.")
            return
        zone = personality.memory_manager.build_working_zone() or "(empty)"
        ASCIIColors.panel(zone, title="[bold cyan]🧠 Shell Memory[/bold cyan]", border_style="cyan")
    except Exception as e:
        ASCIIColors.warning(f"  Could not read memory: {e}")


def _show_help() -> None:
    ASCIIColors.rich_print(
        "[bold]Commands:[/bold]\n"
        "  [cyan]exit[/cyan] / [cyan]quit[/cyan]  — leave lollms_shell\n"
        "  [cyan]/cwd[/cyan] / [cyan]pwd[/cyan]  — print current directory (and venv if any)\n"
        "  [cyan]/memory[/cyan]           — show what I remember about this machine\n"
        "  [cyan]/help[/cyan]             — this message\n\n"
        "[bold]Shell-native conveniences:[/bold]\n"
        "  All of the agent's commands run in ONE persistent shell process —\n"
        "  `cd`, `set VAR=`, `activate` all survive between turns, just like\n"
        "  your own terminal. Use [cyan]/reset[/cyan] to discard that state.\n\n"
        "[bold]Examples:[/bold]\n"
        "  · list the files in this directory\n"
        "  · copy report.pdf to ~/Documents\n"
        "  · why is my C: drive full?\n"
        "  · find all .log files older than 7 days\n"
        "  · show my git status in this project\n\n"
        "[bold]Shell awareness:[/bold]\n"
        "  The agent is told which shell you are in (cmd, powershell, bash, zsh,\n"
        "  fish, wsl), so it writes commands that are native to YOUR terminal.\n"
        "  Read-only commands run freely; destructive ones always prompt."
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"lollms_shell v{APP_VERSION} — Autonomous shell companion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  lollms-shell
  lollms-shell "why is my disk full?"
  lollms-shell --shell-autonomy safe
""",
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Optional one-shot instruction; leave empty for REPL.")
    parser.add_argument("--model", type=str, default=None, help="Model name to use.")
    parser.add_argument("--llm-binding", type=str, default=None, help="LLM binding name.")
    parser.add_argument("--host", type=str, default=None, help="Host address for remote bindings.")
    parser.add_argument("--api-key", type=str, default=None, help="API key for gated services.")
    parser.add_argument("--context-size", type=int, default=None, help="Context window size.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum reasoning steps.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens per turn.")
    parser.add_argument("--effort", "--reasoning-effort", dest="reasoning_effort", type=str, default=None,
                        choices=["none", "low", "medium", "high", "max"], help="Reasoning effort level.")
    parser.add_argument("--dynamic-effort", action="store_true", help="Dynamic reasoning effort.")
    parser.add_argument("--shell-autonomy", type=str, default="full_access",
                        choices=["strict", "safe", "full_access"],
                        help=(
                            "Autonomy level for shell execution. Default is "
                            "'full_access' (agent runs read-only commands freely). "
                            "Destructive commands STILL prompt regardless of level."
                        ))
    parser.add_argument("--no-shell-execution", action="store_true", help="Disable shell execution.")
    parser.add_argument("--no-memory", action="store_true", help="Disable shell memory.")
    parser.add_argument("--handbag", "--handbag-path", dest="handbag_path", type=str, default=None,
                        help="Path to a custom Handbag folder.")
    parser.add_argument("--llm-profile", type=str, default=None, help="LLM model profile alias to activate.")
    parser.add_argument("--config-path", type=str, default=None, dest="config_path",
                        help="Path to a specific configuration file (.env, .json or .yaml).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"lollms_shell v{APP_VERSION}")
    return parser


def main() -> int:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    parser = build_arg_parser()
    args = parser.parse_args()

    # Install the RAG-empty noise filter IMMEDIATELY. lollms_shell is
    # intentionally RAG-free; everything from `config-resolver`, `[Handbag]`,
    # `[Handbag-SCAN]`, `[Personality-DBG]`, and the redundant LCP mount
    # warning is noise to the shell-companion user. All *other* records pass
    # through (see `_DropShellRagEmptyWarning`). Installing this BEFORE any
    # other code runs is the only way to catch warnings emitted during
    # construction of the client and personality.
    _silence_empty_rag_warning()

    config = ShellAgentConfig.load(args)

    # First-time setup gate — reuse the shared simple wizard.
    env_map: dict[str, str] = {}
    try:
        from lollms_client.lollms_config_cli_env import _load_existing_env_to_map
        env_map = _load_existing_env_to_map(args.config_path)
    except Exception:
        pass
    has_llm = any(
        k.upper().startswith("LLM_BINDINGS_") and k.upper().endswith("_BINDING_NAME") and v
        for k, v in env_map.items()
    )
    if not has_llm and not config.llm_model_profiles:
        ASCIIColors.yellow("⚠️ No LLM configured yet. Opening the setup wizard...")
        configured_ok = False
        try:
            from lollms_client.lollms_config_simple import ensure_configured
            configured_ok = bool(ensure_configured())
        except ImportError:
            from lollms_client.lollms_config_cli_env import run_wizard_and_save
            run_wizard_and_save(cli_env_path=args.config_path)
            configured_ok = True
        config = ShellAgentConfig.load(args)
        # Same abort contract as lollms-code: never proceed to create_client()
        # on an unconfigured LLM — the user gets a clear retry instruction
        # instead of a downstream binding error.
        if not configured_ok and not config.llm_model_profiles:
            ASCIIColors.red(
                "\n❌ Setup did not complete — no LLM is configured.\n"
                "   Run `lollms-config` (default: guided simple wizard) or\n"
                "   `lollms-config --expert` to finish configuration, then retry."
            )
            return 1

    # 1. Ensure the handbag directory (and its tools/) exists BEFORE the client
    #    is created — `create_client` registers the handbag tools folder with
    #    the LCP loader, which only works if the folder AND shell_session.py
    #    exist at that moment. On a fresh install this is the FIRST thing we do.
    try:
        ensure_handbag_structure(config)
    except OSError as e:
        ASCIIColors.warning(f"[CLI] Could not ensure handbag structure: {e}")

    # Second pass: run the ensure *again* AFTER the tool folder is populated
    # so the LCP loader (which scans `tools_folders` at construction time) sees
    # `shell_session.py` in the folder. `ensure_handbag_structure` is
    # idempotent and safe to re-run; this guards against ordering regressions
    # on fresh installs where the folder started empty.
    try:
        ensure_handbag_structure(config)
    except OSError as e:
        ASCIIColors.warning(f"[CLI] Could not re-ensure handbag structure: {e}")

    try:
        client = create_client(config)
    except (ImportError, RuntimeError, ValueError, OSError, ConnectionError) as e:
        trace_exception(e)
        ASCIIColors.red(f"Failed to create LollmsClient: {e}")
        return 1

    # 2. Start the persistent shell worker BEFORE the personality is created so
    #    the initial ENVIRONMENT CONTEXT reflects the worker's cwd.
    try:
        session = PersistentShellSession()
    except (OSError, RuntimeError) as e:
        trace_exception(e)
        ASCIIColors.red(f"Failed to start persistent shell: {e}")
        return 1

    # Publish the session IMMEDIATELY into the process-global registry. This is
    # the belt to `_wire_persistent_session_tool`'s suspenders: even if a
    # loader imported `shell_session.py` under a module name our scan misses,
    # `tool_shell_session_execute()`'s `_resolve_session()` still finds the
    # session via `persistent_shell.get_active_session()`.
    try:
        from lollms_client.apps.lollms_shell import persistent_shell as _ps
        _ps.set_active_session(session)
    except Exception:
        pass

    try:
        personality = create_shell_personality(config, client, session=session)
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        session.close()
        trace_exception(e)
        ASCIIColors.red(f"Failed to initialize shell companion: {e}")
        return 1

    # ── POST-SETUP PING: verify the tool actually resolves the session ──
    # A silent wiring failure here means EVERY shell command the agent runs
    # will error with "no persistent shell session is attached". That was a
    # real production bug. We verify with a no-op call to the tool and print
    # a LOUD, non-debug-gated diagnostic if it fails, so the user sees it
    # before typing anything. This test costs ONE cheap round-trip.
    _verify_tool_wiring(session)

    # (Filter already installed at the top of `main()` so startup noise is
    # suppressed too — see `_silence_empty_rag_warning()` call site above.)

    config.save()

    if args.prompt:
        renderer = ShellStreamRenderer(config)
        try:
            result = personality.chat(
                prompt=args.prompt,
                lollms_client=client,
                streaming_callback=renderer,
                max_reasoning_steps=config.max_reasoning_steps,
                temperature=config.temperature,
                n_predict=config.max_tokens_per_turn,
                enable_artefacts=False,
                use_internal_history=False,
                enable_shell=config.enable_shell_execution,
                enable_python_exec=False,
                enable_workspace_tools=False,
                event_mode=EventMode.FULL_CALLBACK_MODE,
                enforce_end_tag=True,
                shell_autonomy_level=config.shell_autonomy_level,
                confirm_handler=_make_confirm_handler(config),
                debug=config.debug,
            )
        except (RuntimeError, ValueError, OSError, ConnectionError) as e:
            trace_exception(e)
            ASCIIColors.red(f"Fatal error: {e}")
            session.close()
            return 1
        print()
        session.close()
        return 0 if not result.get("was_cancelled") else 130

    try:
        return run_interactive(personality, client, config, session)
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())