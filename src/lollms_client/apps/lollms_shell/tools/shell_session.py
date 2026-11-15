"""
shell_session — LCP tool that talks to the persistent shell worker.
====================================================================

Why a dedicated tool?
---------------------
The stock ``system_shell`` toolset spawns a fresh subprocess per call. That is
fine for one-shot read-only commands but fatal for a shell companion: ``cd``,
``export FOO=bar``, ``set VAR=val``, and ``source venv/bin/activate`` all die
the moment the child exits.

``lollms_shell`` therefore ships its OWN tool that operates on a single
long-lived shell worker (see ``persistent_shell.py``). ``cli.py`` unmounts
``system_shell`` for this app, so the LLM has exactly one path to run
commands: the stateful one.

Module identity
---------------
The LCP loader and the handbag loader may both ``import`` this file, under
different generated module names. To avoid "two copies, one wired" bugs, this
module defers its session resolution to the **canonical** copy
(``lollms_client.apps.lollms_shell.tools.shell_session``) whenever it detects
that it is a duplicate import. The canonical copy owns ``_SESSION`` and the
process-global registry in ``persistent_shell``.

Destructive-command safety net
------------------------------
Per the lollms_shell doctrine, destructive commands MUST be confirmed even in
``full_access`` autonomy. This tool enforces that guarantee itself (via the
injected ``confirm_handler``) instead of relying on the caller — so a custom
handbag or a future autonomy-level refactor can never bypass it.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional, Tuple

_log = logging.getLogger(__name__)

# ── CONSTANTS ──────────────────────────────────────────────────────────────

_CANONICAL_MODULE = "lollms_client.apps.lollms_shell.tools.shell_session"

# Hard cap on a single command's wall-clock time. The persistent worker's
# `run()` blocks on `readline` and has no built-in deadline — a hung command
# (e.g. a `pause`, a server that never returns) would freeze the REPL forever
# without this guard.
_MAX_COMMAND_TIMEOUT_SECONDS = 600.0
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 120.0

# Substrings that mark a command as destructive. Kept in sync with the
# doctrine-level regex in `cli._DESTRUCTIVE_CMD_PATTERNS`. This belt-and-
# suspenders layer is INTENTIONALLY redundant with the CLI's regex: if the
# CLI regex is ever edited in a way that opens a hole, this layer still gates.
# Using substrings (not a full regex) keeps the check fast and false-positive
# tolerant — we would rather over-prompt than miss a `rm -rf`.
_DESTRUCTIVE_MARKERS: Tuple[str, ...] = (
    "rm ", "rm\t", "rmdir", "del ", "del\t", "erase ",
    "remove-item", "ri ",
    "mv ", "mv\t", "move ", "move-item", "rename ", "ren ",
    "mkfs", "format ", "diskpart", "fdisk", "parted", "dd ",
    "shutdown", "reboot", "halt", "restart-computer", "stop-computer",
    "reg add", "reg delete", "reg import", "reg load", "setx ",
    "sc create", "sc delete", "sc config", "sc stop",
    "net user", "net localgroup",
    "takeown", "icacls", "chmod ", "chown ",
    "pip uninstall", "pipx uninstall",
    "npm uninstall", "apt remove", "apt purge",
    "choco uninstall", "winget uninstall", "scoop uninstall",
)

# ── WIRING STATE ───────────────────────────────────────────────────────────
#
# `_SESSION` is the local fast-path cache. The canonical source of truth is
# `persistent_shell._ACTIVE_SESSION`, which every copy of this module can read.
# `_CONFIRM_HANDLER` invokes the CLI's doctrinal gate for destructive commands.

_SESSION: Any = None  # PersistentShellSession | None
_CONFIRM_HANDLER: Optional[Callable[[dict], Tuple[str, str]]] = None
_WIRED_AT: Optional[float] = None


# ── SESSION RESOLUTION ─────────────────────────────────────────────────────

def _is_canonical() -> bool:
    """True when this module IS the canonical package copy (not a shim)."""
    return __name__ == _CANONICAL_MODULE


def _canonical_module() -> Optional[Any]:
    """Returns the canonical package module, importing it if necessary."""
    try:
        import importlib
        return importlib.import_module(_CANONICAL_MODULE)
    except ImportError:
        return None


def _resolve_session() -> Any:
    """Returns the wired session.

    Resolution order (each step is cheap; first hit wins):
      1. Local `_SESSION` cache — populated by `attach_session()`.
      2. `sys._lollms_shell_active_session` — published by `main()` and
         `persistent_shell.set_active_session()`. This is the ONLY reliable
         cross-module-identity source: `sys` is a singleton, so even if this
         module was imported under a different name than the canonical copy
         (handbag loader, LCP loader, src-vs-installed), this attribute is
         identical everywhere. It is checked BEFORE the canonical-defer
         because a broken import of the canonical copy is exactly the
         failure mode we are defending against.
      3. Canonical module's own `_resolve_session()` (recursion-safe — we
         skip it when our `__name__` IS the canonical).
      4. `persistent_shell.get_active_session()` via a direct import — the
         legacy fallback, retained in case `sys` attribute publishing was
         somehow bypassed.
    """
    # 1. Local fast path.
    if _SESSION is not None:
        return _SESSION

    # 2. sys-module singleton — works across ANY module identity.
    try:
        import sys as _sys
        global_val = getattr(_sys, "_lollms_shell_active_session", None)
        if global_val is not None:
            return global_val
    except Exception:  # pragma: no cover — sys import never fails in practice
        pass

    # 3. Defer to the canonical copy (skip recursion when we ARE canonical).
    if not _is_canonical():
        canonical = _canonical_module()
        if canonical is not None:
            resolver = getattr(canonical, "_resolve_session", None)
            if callable(resolver) and resolver is not _resolve_session:
                try:
                    val = resolver()
                    if val is not None:
                        return val
                except Exception:
                    pass

    # 4. Direct persistent_shell lookup (legacy fallback).
    try:
        from lollms_client.apps.lollms_shell import persistent_shell
        return persistent_shell.get_active_session()
    except Exception:  # pragma: no cover — defensive only
        return None


def _resolve_confirm_handler() -> Optional[Callable[[dict], Tuple[str, str]]]:
    """Returns the injected confirm handler (shim-aware)."""
    if not _is_canonical():
        canonical = _canonical_module()
        if canonical is not None:
            getter = getattr(canonical, "_resolve_confirm_handler", None)
            if callable(getter) and getter is not _resolve_confirm_handler:
                return getter()
    return _CONFIRM_HANDLER


# ── PUBLIC WIRING API ──────────────────────────────────────────────────────

def attach_session(session: Any, confirm_handler: Optional[Callable] = None) -> None:
    """Wire the persistent session and confirmation callback into this tool.

    Idempotent and safe to call from multiple module copies:

      * The session is also published to the process-global registry so every
        OTHER copy (the LCP-loader copy, the handbag-loader copy, the shim,
        the canonical) resolves the SAME instance.
      * The confirm handler is stored locally; shims read it via
        `_resolve_confirm_handler()` which defers to the canonical module.
    """
    global _SESSION, _CONFIRM_HANDLER, _WIRED_AT
    _SESSION = session
    _CONFIRM_HANDLER = confirm_handler
    _WIRED_AT = time.time()

    try:
        from lollms_client.apps.lollms_shell import persistent_shell
        persistent_shell.set_active_session(session)
    except Exception:  # pragma: no cover — registry publish is best-effort
        pass

    # Forward the wiring to the canonical copy when we ARE the canonical (no
    # forwarding needed) or when the canonical exists independently. This
    # guarantees that whichever copy the LCP loader dispatches to sees the
    # handler — but only ONE write to `_CONFIRM_HANDLER` occurs per module.
    if not _is_canonical():
        canonical = _canonical_module()
        if canonical is not None:
            attach = getattr(canonical, "attach_session", None)
            if callable(attach) and attach is not attach_session:
                try:
                    attach(session, confirm_handler)
                except Exception:
                    pass


# ── SAFETY NET ─────────────────────────────────────────────────────────────

def _looks_destructive(command: str) -> bool:
    """Fast substring check — paired with the authoritative regex in `cli.py`."""
    if not command:
        return False
    normalized = f" {command.lower()} "
    return any(marker in normalized for marker in _DESTRUCTIVE_MARKERS)


def _request_user_confirmation(command: str) -> Tuple[bool, str]:
    """Ask the CLI confirm handler. Returns (approved, feedback)."""
    handler = _resolve_confirm_handler()
    if handler is None:
        # No handler wired — refuse rather than assume. This can only happen
        # in unit tests that bypass `cli.main()`; the CLI always wires one.
        return False, (
            "No confirmation handler wired. Refusing destructive command as a "
            "safety precaution. To run it, use the interactive REPL (which "
            "wires the handler) or modify the command to be non-destructive."
        )
    try:
        decision, feedback = handler({"command": command})
    except Exception as exc:  # pragma: no cover — defensive only
        _log.warning("[shell_session] confirm handler raised: %s", exc)
        return False, f"Confirmation handler failed: {exc}"
    approved = str(decision).lower() in ("allow", "always")
    return approved, str(feedback or "")


# ── TOOL IMPLEMENTATION ────────────────────────────────────────────────────

def tool_shell_session_execute(
    command: str,
    explanation: Optional[str] = None,
    timeout_seconds: int = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> dict:
    """Execute a shell command in the SAME persistent shell session.

    State is preserved across calls: ``cd``, ``set VAR=val`` /
    ``$env:VAR='val'`` / ``export VAR=val``, activated virtual environments,
    and shell variables all survive to the next invocation — exactly like the
    user's own terminal.

    Because state persists, prefer incremental commands (``cd ..`` then
    ``pwd``) over chained one-liners (``cd .. && cd``) — each call is cheap.

    Args:
        command: The exact shell command to run. Must be valid syntax for the
            shell named in the ENVIRONMENT CONTEXT block (cmd / powershell /
            bash / zsh / fish / wsl).
        explanation: One-line description of what the command does and why.
            Logged for traceability; not sent to the shell.
        timeout_seconds: Safety timeout; defaults to 120, hard-capped at 600.

    Returns:
        dict with keys:
          * ``success`` (bool)
          * ``output`` (str) — stdout+stderr, sentinel/echo scrubbed
          * ``exit_code`` (int)
          * ``cwd`` (str) — the shell's working directory AFTER the command
    """
    if not command or not str(command).strip():
        return {
            "success": False,
            "output": "Empty command — nothing to execute.",
            "exit_code": 1,
            "cwd": "?",
        }

    session = _resolve_session()
    if session is None:
        return {
            "success": False,
            "output": (
                "No persistent shell session is attached. This is a wiring "
                "bug in lollms_shell — the agent cannot execute commands. "
                "Restart lollms-shell; if it persists, run `/reset` first."
            ),
            "exit_code": 1,
            "cwd": "?",
        }

    if explanation:
        _log.debug("[shell_session] %s  →  %s", explanation.strip(), command.strip())

    # Doctrinal safety net — destructive commands require explicit confirmation
    # even when the CLI autonomy is `full_access`. See module docstring.
    if _looks_destructive(command):
        approved, feedback = _request_user_confirmation(command)
        if not approved:
            return {
                "success": False,
                "output": (
                    "Command blocked (destructive and not confirmed).\n"
                    f"Feedback: {feedback or 'User declined.'}\n"
                    f"Command: {command}"
                ),
                "exit_code": 1,
                "cwd": _current_cwd(session),
            }

    # Enforce a hard timeout on top of whatever the session supports.
    try:
        bounded_timeout = max(1.0, min(float(timeout_seconds), _MAX_COMMAND_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        bounded_timeout = _DEFAULT_COMMAND_TIMEOUT_SECONDS

    try:
        output, code = session.run(command, timeout=bounded_timeout)
    except Exception as exc:  # pragma: no cover — session.run should not raise
        return {
            "success": False,
            "output": f"Shell session raised: {exc}",
            "exit_code": 1,
            "cwd": _current_cwd(session),
        }

    return {
        "success": int(code) == 0,
        "output": output,
        "exit_code": int(code),
        "cwd": _current_cwd(session),
    }


def _current_cwd(session: Any) -> str:
    """Safe `session.cwd()` — never raises into the tool return value."""
    try:
        return str(session.cwd())
    except Exception:
        return "?"


# ── LCP LOADER ENTRY POINT ─────────────────────────────────────────────────

def init_tools_library(config: Optional[dict] = None) -> dict:
    """LCP loader entry point — returns the tool registry for this library.

    Schema shape mirrors the sibling toolsets under
    ``tools_bindings/lcp/default_tools/*`` so the LCP loader and the LLM
    binding both consume it without shims.
    """
    return {
        "tool_shell_session_execute": {
            "description": (
                "Execute a shell command in the SAME persistent shell session. "
                "State (cwd, env vars, shell variables, activated venvs) is "
                "preserved across calls — like a real terminal. Use this for "
                "every shell command; prefer incremental commands over "
                "`A && B` chaining because each call keeps its state. "
                "Destructive commands (delete, format, shutdown, "
                "uninstall, etc.) are blocked until the user confirms."
            ),
            "function": tool_shell_session_execute,
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The exact shell command to execute. Must be valid "
                            "syntax for the shell named in the ENVIRONMENT "
                            "CONTEXT block."
                        ),
                    },
                    "explanation": {
                        "type": "string",
                        "description": (
                            "One-line description of what the command does and "
                            "why. Optional; improves audit trails."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": (
                            "Safety timeout in seconds. Default 120, hard-capped "
                            "at 600. Raise only for known long-running operations."
                        ),
                    },
                },
                "required": ["command"],
            },
        }
    }


__all__ = [
    "attach_session",
    "init_tools_library",
    "tool_shell_session_execute",
]