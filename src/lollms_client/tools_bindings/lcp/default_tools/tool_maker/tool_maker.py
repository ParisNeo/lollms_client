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
    tool_name: str,
    description: str,
    commit_message: str = "",
    tools_dir: str = "./lcp_tools",
    discussion_instance: Optional[Any] = None,
    lollms_client_instance: Optional[Any] = None
) -> dict:
    """
    Spawns a specialized Tool Maker agent to compile and register a new local tool dynamically on disk.

    Args:
        tool_name (str): Lowercase snake_case name of the new tool (e.g. 'file_compressor').
        description (str): A precise, detailed description of what the tool should do, its parameters, and expected output.
        commit_message (str, optional): Message describing the new tool.
        tools_dir (str, optional): Directory path to save the tool. Defaults to './lcp_tools'.
        discussion_instance (Any, optional): Active discussion session instance. Injected by orchestrator.
        lollms_client_instance (Any, optional): Active client instance. Injected by orchestrator.
    """
    tool_name = tool_name.strip().lower()
    if not tool_name:
        return {"success": False, "error": "tool_name parameter is mandatory."}
    if not description:
        return {"success": False, "error": "description parameter is mandatory."}

    if not commit_message:
        commit_message = f"Create smart tool: {tool_name}"

    if not lollms_client_instance:
        return {"success": False, "error": "Lollms client instance is required for spinoff agent generation."}

    custom_system = (
        "You are an expert LCP Tool Forge.\n"
        "You operate in a hyper-focused sandbox isolated from the main conversation's noise.\n"
        "Your sole task is to implement a valid Python LCP tool file based on the user's description.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY a valid `<artifact type=\"tool\" name=\"{tool_name}.py\">` block containing your code.\n"
        "2. The code MUST contain a function starting with `tool_` that accepts typed arguments.\n"
        "3. The function MUST have a comprehensive docstring explaining its purpose, arguments, and return values.\n"
        "4. Do NOT write introductory/concluding prose outside the artifact tags.\n"
        "5. Ensure the code is robust, handles edge cases, and returns a dictionary with a 'success' key.\n"
        "6. Tools are strictly agnostic. Do NOT accept `discussion_instance` or `lollms_client_instance` in the tool signature.\n"
    )

    payload = (
        f"=== TOOL FORGE TASK ===\n"
        f"Tool Name: {tool_name}\n"
        f"Description: {description}\n"
        f"Commit Message: {commit_message}\n"
        f"========================\n"
        f"Generate the Python code for this tool now."
    )

    try:
        res = lollms_client_instance.generate_text(
            prompt=payload,
            system_prompt=custom_system,
            temperature=0.1,
        )
        raw_output = res.strip()

        # Extract code from the artifact tag if present
        import re
        code_match = re.search(r'<artifact[^>]*>(.*?)</artifact>', raw_output, re.DOTALL | re.IGNORECASE)
        if not code_match:
            return {"success": False, "error": "Spinoff agent failed to generate code inside <artifact> tags."}
        
        code = code_match.group(1).strip()

        # Determine save path
        if discussion_instance and hasattr(discussion_instance, 'workspace_data_path'):
            # Save to the discussion's workspace
            save_dir = Path(discussion_instance.workspace_data_path) / "lcp_tools"
        else:
            # Fallback to relative path
            save_dir = Path(tools_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        py_file = save_dir / f"{tool_name}.py"

        py_file.write_text(code, encoding="utf-8")

        ASCIIColors.success(f"[Tool Maker] Successfully compiled flat tool '{tool_name}.py' at {py_file}!")
        return {
            "success": True,
            "tool_name": tool_name,
            "message": f"Successfully compiled tool '{tool_name}' on disk (flat file). The orchestrator can now register it.",
            "python_file": str(py_file.resolve()),
            "prompt_injection": f"\n\n🛠️ **Tool Created:** `{tool_name}.py` has been forged and saved to the workspace. You can now use it by calling `<tool>{{\"name\": \"tool_{tool_name}\", \"parameters\": {{...}}}}</tool>`."
        }

    except Exception as e:
        trace_exception(e)
        return {"success": False, "error": f"Tool Maker spinoff generation failed: {e}"}
