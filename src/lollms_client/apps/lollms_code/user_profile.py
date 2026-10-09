"""
Shared persistence and system-prompt injection for the lollms_code agent's
USER.md file (.lollms_code/USER.md).

Used by BOTH the GUI (agent_bridge.py) and the CLI (cli.py) so the agent
remembers the user's name and identity across every session and start,
regardless of entry point.
"""
from __future__ import annotations

from pathlib import Path

USER_PROFILE_REL_PATH = ".lollms_code/USER.md"

USER_PROFILE_TEMPLATE = """\
# User Profile

This file is stored at `.lollms_code/USER.md`.
You can update this file using `<file>` tags to persist user identity and preferences.

## Identity
- Name: (unknown)

## Preferences
- (None)
"""


def get_user_profile_path(workspace_path: str | Path) -> Path:
    """Returns the absolute path to the workspace's .lollms_code/USER.md file."""
    return Path(workspace_path).resolve() / USER_PROFILE_REL_PATH


def seed_user_profile(workspace_path: str | Path) -> Path:
    """
    Ensures .lollms_code/USER.md exists for the workspace.
    Creates the directory and writes the template only if the file is absent.
    Returns the resolved USER.md path.
    """
    user_md_path = get_user_profile_path(workspace_path)
    user_md_path.parent.mkdir(parents=True, exist_ok=True)
    if not user_md_path.exists():
        user_md_path.write_text(USER_PROFILE_TEMPLATE, encoding="utf-8")
    return user_md_path


def build_user_profile_context(workspace_path: str | Path) -> str:
    """
    Reads .lollms_code/USER.md and wraps it in a bounded system-prompt block.
    Returns an empty string when the file is missing or empty so callers can
    append it unconditionally without polluting the prompt.
    """
    user_md_path = get_user_profile_path(workspace_path)
    if not user_md_path.exists():
        return ""
    try:
        content = user_md_path.read_text(encoding="utf-8", errors="ignore").strip()
    except (OSError, UnicodeDecodeError):
        return ""
    if not content:
        return ""
    return (
        "\n=== USER PROFILE (From .lollms_code/USER.md) ===\n"
        f"{content}\n"
        "=== END USER PROFILE ===\n"
    )
</file>