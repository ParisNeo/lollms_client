import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


def init_tools_library() -> None:
    pass


def tool_dump_context(file_name: str = "context_dump.json", discussion_instance: Optional[Any] = None) -> Dict[str, Any]:
    """
    Dumps the raw, active conversation context (system prompt and message history) into a JSON file in the current working directory.
    Use this to debug context visibility issues, verify tool call history, or inspect the exact prompt sent to the LLM.

    Args:
        file_name (str, optional): The name of the JSON file to write. Defaults to 'context_dump.json'.
    """
    if discussion_instance is None:
        return {
            "success": False,
            "error": "Discussion instance is not available. This tool must be run from within a LollmsDiscussion or Agent chat loop."
        }

    try:
        safe_name = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in file_name)
        if not safe_name.endswith(".json"):
            safe_name += ".json"

        output_path = Path(safe_name)

        context_payload = {
            "discussion_id": getattr(discussion_instance, "id", "unknown"),
            "system_prompt": getattr(discussion_instance, "system_prompt", None) or "",
            "active_branch_id": getattr(discussion_instance, "active_branch_id", None),
        }

        exported_messages = []
        try:
            exported_messages = discussion_instance.export(format_type="openai_chat")
        except Exception as export_err:
            exported_messages = [{"error": f"Failed to export context: {export_err}"}]

        context_payload["messages"] = exported_messages

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context_payload, f, indent=2, default=str, ensure_ascii=False)

        return {
            "success": True,
            "output": f"Context successfully dumped to '{output_path}'.",
            "file_name": str(output_path),
            "message_count": len(exported_messages),
            "prompt_injection": f"\n\n✅ **Context Dumped Successfully!**\nThe raw context has been saved to `{output_path}`.\nUse `tool_read_file` to inspect it if needed. Do not output the contents directly in the chat."
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to dump context: {str(e)}",
            "traceback": traceback.format_exc()
        }