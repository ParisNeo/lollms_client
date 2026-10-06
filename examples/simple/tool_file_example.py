"""
Example: Using a Python file as a tool source for chat().

This demonstrates how to pass a .py file path in the tools list.
All tool_* functions defined in that file become available to the LLM.

Prerequisites:
- An Ollama server running with a model (e.g., llama3.2)
- lollms_client installed
"""
import tempfile
from pathlib import Path
from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client.lollms_config_cli_env import get_client_from_env

def create_example_tool_file() -> Path:
    """Creates a temporary Python file containing tool_* functions."""
    tool_code = '''
def tool_calculate_average(numbers: str) -> dict:
    """
    Calculates the average of a comma-separated list of numbers.

    Args:
        numbers: A comma-separated string of numbers (e.g., "1,2,3,4,5")

    Returns:
        A dict with the average and count.
    """
    try:
        nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
        if not nums:
            return {"success": False, "error": "No numbers provided"}
        avg = sum(nums) / len(nums)
        return {"success": True, "average": avg, "count": len(nums), "sum": sum(nums)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_reverse_string(text: str) -> dict:
    """
    Reverses a string.

    Args:
        text: The string to reverse

    Returns:
        A dict with the reversed string.
    """
    if not text:
        return {"success": False, "error": "Empty text"}
    return {"success": True, "reversed": text[::-1]}
'''
    tmp_dir = Path(tempfile.mkdtemp())
    tool_file = tmp_dir / "my_custom_tools.py"
    tool_file.write_text(tool_code, encoding="utf-8")
    return tool_file


def main():
    # ── Step 1: Create the tool file ─────────────────────────────────────
    tool_file = create_example_tool_file()
    print(f"✅ Tool file created: {tool_file}")
    print(f"   Tools defined: tool_calculate_average, tool_reverse_string\n")

    # ── Step 2: Initialize client ────────────────────────────────────────
    client = get_client_from_env()

    # ── Step 3: Create discussion ────────────────────────────────────────
    discussion = LollmsDiscussion(
        lollmsClient=client,
    )
    discussion.system_prompt = "You are a helpful assistant with access to custom tools."

    # ── Step 4: Call chat() with the tool file path ──────────────────────
    print("🤖 Asking the LLM to use the custom tools...\n")
    
    user_prompt = (
        "Please do two things:\n"
        "1. Use tool_calculate_average to find the average of 10, 20, 30, 40, 50\n"
        "2. Use tool_reverse_string to reverse the word 'hello'\n"
        "Report both results."
    )

    result = discussion.chat(
        user_message=user_prompt,
        tools=[str(tool_file)],  # ← The .py file path in the tools list
        max_nb_rounds=5,
    )

    ai_msg = result["ai_message"]
    print("=" * 60)
    print("AI Response:")
    print("=" * 60)
    print(ai_msg.content or "(empty)")
    print("=" * 60)

    # ── Cleanup ──────────────────────────────────────────────────────────
    tool_file.unlink(missing_ok=True)
    tool_file.parent.rmdir()


if __name__ == "__main__":
    main()