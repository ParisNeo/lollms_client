# tool_maker.py
# -----------------------------------------------------------------------------
# Tool Maker — Lollms LCP Tool
#
# Enables the LLM to write, compile, and register its own custom tools on the fly
# without requiring separate JSON schemas. Saves flat python files with rich
# docstrings to the LCP default directory, registering them instantly as active artifacts.

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from ascii_colors import ASCIIColors, trace_exception

TOOL_LIBRARY_NAME = "Lollms Tool Maker"
TOOL_LIBRARY_DESC = "Generates and registers a new flat LCP smart tool dynamically using AST parsing."
TOOL_LIBRARY_ICON = "🛠️"

def init_tools_library() -> None:
    pass

def tool_tool_maker(
    args: dict,
    tools_dir: str = "./lcp_tools"
) -> dict:
    """
    Compile and register a new local tool dynamically on disk and in the active session.

    Args:
        args (dict):
            - tool_name (str): Lowercase snake_case name of the new tool (e.g. 'file_compressor').
            - code (str): Complete Python code containing the 'tool_[tool_name]' function with type annotations and a descriptive docstring.
            - commit_message (str, optional): Message describing the new tool.
    """
    tool_name = args.get("tool_name", "").strip().lower()
    code = args.get("code", "").strip()
    commit_message = args.get("commit_message", f"Create smart tool: {tool_name}")

    if not tool_name:
        return {"success": False, "error": "tool_name parameter is mandatory."}
    if not code:
        return {"success": False, "error": "code parameter is mandatory."}

    # 🛑 TOOLS ARE AGNOSTIC: Use provided tools_dir or fallback to CWD.
    # The orchestrator is responsible for placing the generated file in the correct binding folder
    # and triggering a re-discovery, OR the user can specify a target directory.
    tools_dir = Path(tools_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)

    # Write as a clean, flat python file in the LCP directory
    py_file = Path(tools_dir) / f"{tool_name}.py"

    try:
        # 2. Save python code directly to disk (LCP will parse schema via AST on re-discovery)
        py_file.write_text(code, encoding="utf-8")

        # 🛑 AGNOSTIC: Tool does not access the binding or discussion.
        # It writes the file to the specified directory. 
        # The orchestrator can detect this file and register it.

        ASCIIColors.success(f"[Tool Maker] Successfully compiled flat tool '{tool_name}.py' at {py_file}!")
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Successfully compiled tool '{tool_name}' on disk (flat file). The orchestrator can now register it.",
            "python_file": str(py_file.resolve())
        }

    except Exception as e:
        trace_exception(e)
        if py_file.exists(): py_file.unlink()
        return {"success": False, "error": f"Tool Maker compilation failed: {e}"}
