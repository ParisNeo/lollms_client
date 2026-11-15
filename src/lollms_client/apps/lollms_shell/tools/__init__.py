"""Package-internal location for lollms_shell companion tools.

The primary *loadable* copy lives under
``lollms_shell/default_handbag/tools/shell_session.py`` (so the LCP loader and
the handbag loader both discover it). This package re-exports that module so
``from lollms_client.apps.lollms_shell.tools import shell_session`` works
regardless of where the handbag copy was seeded — the CLI wires the session
into this module namespace, and the handbag copy imports from here when the
loader picks it up.
"""

from lollms_client.apps.lollms_shell.tools.shell_session import (  # re-export
    attach_session,
    init_tools_library,
    tool_shell_session_execute,
)

__all__ = [
    "attach_session",
    "init_tools_library",
    "tool_shell_session_execute",
]