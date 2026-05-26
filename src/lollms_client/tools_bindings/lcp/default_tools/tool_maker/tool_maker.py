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

def init_tool_library() -> None:
    pass

def tool_tool_maker(
    args: dict,
    lollms_client_instance: Optional[Any] = None,
    discussion_instance: Optional[Any] = None
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

    # 1. Resolve Target Tools Folder
    lc = lollms_client_instance or (discussion_instance.lollmsClient if discussion_instance else None)
    if not lc:
        return {"success": False, "error": "LollmsClient instance is required to resolve tools folder."}

    lcp_binding = getattr(lc, "tools", None)
    if not lcp_binding:
        return {"success": False, "error": "LCP Tools Binding is not loaded on the client."}

    tools_dir = getattr(lcp_binding, "tools_folder_path", None)
    if not tools_dir:
        return {"success": False, "error": "LCP Tools folder path is unresolved."}

    # Write as a clean, flat python file in the LCP directory
    py_file = Path(tools_dir) / f"{tool_name}.py"

    try:
        # 2. Save python code directly to disk (LCP will parse schema via AST on re-discovery)
        py_file.write_text(code, encoding="utf-8")

        # 3. Reload tools in active LCP binding so it's instantly discoverable
        if hasattr(lcp_binding, "_discover_local_tools"):
            lcp_binding._discover_local_tools()

        # 4. Save as a Discussion Artifact
        if discussion_instance:
            art_content = (
                f"# Smart Tool: {tool_name}\n"
                f"Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                f"### Python Implementation (`{tool_name}.py`)\n"
                f"```python\n{code}\n```"
            )
            discussion_instance.artefacts.add(
                title=f"{tool_name}_tool",
                artefact_type="tool",
                content=art_content,
                active=True,
                commit_message=commit_message
            )
            discussion_instance.commit()

        ASCIIColors.success(f"[Tool Maker] Successfully compiled flat tool '{tool_name}.py'!")
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Successfully compiled and registered tool '{tool_name}' on disk (flat file) and as an active session artifact. Schema parsed via AST.",
            "python_file": str(py_file.resolve())
        }

    except Exception as e:
        trace_exception(e)
        if py_file.exists(): py_file.unlink()
        return {"success": False, "error": f"Tool Maker compilation failed: {e}"}
