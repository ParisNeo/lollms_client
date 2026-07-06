"""
dummy_test_tool.py
==================
A dummy LCP tool library designed specifically for testing CWD management,
workspace isolation, and CWD restoration on crashes in the ChatMixin orchestrator.
"""

import os
from pathlib import Path
from typing import Dict, Any

TOOL_LIBRARY_NAME = "DUMMY_TEST_TOOL"
TOOL_LIBRARY_DESC = "A dummy tool for testing CWD management and workspace isolation in the LCP binding."
TOOL_LIBRARY_ICON = "🧪"


def init_tool_library() -> None:
    """No external dependencies required for this dummy tool."""
    pass


def tool_dummy_cwd_test(file_name: str = "dummy.txt") -> Dict[str, Any]:
    """
    Verifies that the CWD is correctly set to the isolated workspace directory.
    Reads a file from the current working directory and returns its content
    along with the resolved CWD path.

    Args:
        file_name (str): Name of the file to read from the workspace. Defaults to 'dummy.txt'.
    """
    cwd = Path(os.getcwd()).resolve()
    file_path = Path(file_name)

    if not file_path.exists():
        return {
            "success": False,
            "error": f"File '{file_name}' not found in CWD: {cwd}",
            "cwd": str(cwd),
        }

    content = file_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "success": True,
        "output": f"File '{file_name}' read successfully from CWD: {cwd}. Content: {content}",
        "cwd": str(cwd),
        "content": content,
    }


def tool_dummy_crash_test() -> Dict[str, Any]:
    """
    Intentionally raises a RuntimeError to test that the ChatMixin orchestrator
    correctly restores the CWD after an unhandled tool exception.

    This function never returns normally.
    """
    raise RuntimeError("Intentional crash for testing CWD restoration on exceptions.")
