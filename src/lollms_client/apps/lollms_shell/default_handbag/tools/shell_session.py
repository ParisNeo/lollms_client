"""
shell_session — handbag-local shim for the persistent shell tool.
=================================================================

The real implementation lives at
``lollms_client/apps/lollms_shell/tools/shell_session.py`` (installed with the
package). This file exists so a **handbag-only** loader (one that scans the
handbag's ``tools/`` directory with ``importlib.spec_from_file_location``) can
discover ``tool_shell_session_execute`` and ``attach_session`` without needing
the app package on ``sys.path``.

Implementation is imported, NOT duplicated — there is exactly ONE module-level
``_SESSION`` / ``_resolve_session()`` pair (in the canonical package module).
That is critical: the LCP loader imports THIS file under a generated module
name, so any duplicate local state would be invisible to the wiring done from
``cli.py``. This shim also means ``persistent_shell.get_active_session()`` is
reachable from whichever copy the dispatcher ends up calling.

If the canonical import fails (broken install), we fall back to a stub that
reports the missing dependency — still importable, never crashes the loader.
"""

from __future__ import annotations

try:  # Normal path — the package is installed alongside the handbag.
    from lollms_client.apps.lollms_shell.tools.shell_session import (  # noqa: F401
        attach_session,
        init_tools_library,
        tool_shell_session_execute,
    )
except ImportError as _exc:  # pragma: no cover — defensive only
    _IMPORT_ERR = _exc

    def attach_session(session, confirm_handler=None):  # type: ignore[override]
        return None

    def tool_shell_session_execute(command: str, explanation=None, timeout_seconds: int = 120):  # type: ignore[override]
        return {
            "success": False,
            "output": (
                f"lollms_shell.tools.shell_session could not be imported "
                f"({_IMPORT_ERR}). Install or repair lollms_client."
            ),
            "exit_code": 1,
            "cwd": "?",
        }

    def init_tools_library(config=None):  # type: ignore[override]
        return {
            "tool_shell_session_execute": {
                "description": "Persistent shell tool (unavailable — import failure).",
                "function": tool_shell_session_execute,
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            }
        }