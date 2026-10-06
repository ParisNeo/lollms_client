#!/usr/bin/env python3
"""
handbag_extra_tools_example.py
==============================
Demonstrates the standard LollmsClient workflow:
  1. Create client from env config
  2. Create a LollmsPersonality with a handbag
  3. Pass the personality to a LollmsDiscussion
  4. Call discussion.chat() with a tools list containing the path to an extra tool file

Architecture:
  - Handbag tools → personality._tool_binding (auto-loaded)
  - Extra tool file → chat(tools=[path]) → _resolve_mixed_tools_list → LCPBinding._load_tool_file
  - Both merge into active_tools → system prompt
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from ascii_colors import ASCIIColors
from lollms_client import LollmsClient, LollmsDiscussion
from lollms_client.lollms_personality import LollmsPersonality, CapabilityFlags
from lollms_client.lollms_types import MSG_TYPE


# ── HANDBAG DIRECTORY ────────────────────────────────────────────────────────
HANDBAG_PATH = PROJECT_ROOT / "data_workspace" / "handbag_tools_test"

# ── TOOL CONTENTS ─────────────────────────────────────────────────────────────

HANDBAG_TOOL_CODE = """
def tool_handbag_greet(name: str = "World"):
    \"\"\"
    Returns a greeting from the handbag's built-in tool set.

    Args:
        name (str, optional): The name to greet. Defaults to 'World'.
    \"\"\"
    return {"success": True, "output": f"Hello {name}! This greeting comes from the HANDBAG tool set."}
"""

EXTRA_TOOL_CODE = """
def tool_extra_compute(a: int, b: int):
    \"\"\"
    Computes the sum of two integers.
    This tool is an EXTRA standalone file, not part of the handbag.

    Args:
        a (int): First integer.
        b (int): Second integer.
    \"\"\"
    return {"success": True, "output": f"The sum of {a} and {b} is {a + b} (from the EXTRA tool file)."}


def tool_extra_reverse(text: str):
    \"\"\"
    Reverses a string.
    This tool is an EXTRA standalone file, not part of the handbag.

    Args:
        text (str): The string to reverse.
    \"\"\"
    if not text:
        return {"success": False, "error": "Empty text"}
    return {"success": True, "output": f"Reversed: '{text[::-1]}' (from the EXTRA tool file)."}
"""

SOUL_CONTENT = """---
name: ToolTestAgent
author: test
version: '1.0'
category: testing
description: An agent that tests handbag tools plus extra tool files.
---

You are ToolTestAgent. You have access to tools from two sources:
1. Your handbag's built-in tools (e.g. tool_handbag_greet)
2. An extra standalone tool file passed at runtime (e.g. tool_extra_compute)

When asked to test your tools, call BOTH tool_handbag_greet AND tool_extra_compute.
Then summarize the results and emit <done/>.
"""


def setup_files() -> Path:
    """Creates the handbag folder and the extra tool file on disk."""
    if HANDBAG_PATH.exists():
        shutil.rmtree(HANDBAG_PATH, ignore_errors=True)

    HANDBAG_PATH.mkdir(parents=True, exist_ok=True)
    (HANDBAG_PATH / "SOUL.md").write_text(SOUL_CONTENT, encoding="utf-8")

    tools_dir = HANDBAG_PATH / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "handbag_tool.py").write_text(HANDBAG_TOOL_CODE, encoding="utf-8")

    (HANDBAG_PATH / "workspace").mkdir(parents=True, exist_ok=True)
    (HANDBAG_PATH / "memory").mkdir(parents=True, exist_ok=True)

    extra_tool_file = HANDBAG_PATH / "extra_standalone_tool.py"
    extra_tool_file.write_text(EXTRA_TOOL_CODE, encoding="utf-8")

    ASCIIColors.success(f"✅ Handbag created at: {HANDBAG_PATH}")
    ASCIIColors.info(f"   • Handbag tool: tools/handbag_tool.py → tool_handbag_greet")
    ASCIIColors.info(f"   • Extra tool:   extra_standalone_tool.py → tool_extra_compute, tool_extra_reverse")
    return extra_tool_file


def streaming_callback(chunk, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        print(chunk, end="", flush=True)
    return True


def main():
    ASCIIColors.panel(
        "[bold]Handbag + Extra Tools via LollmsDiscussion[/bold]\n"
        "1. Client from env\n"
        "2. LollmsPersonality from handbag\n"
        "3. LollmsDiscussion with that personality\n"
        "4. chat(tools=[extra_tool_path])\n",
        title="[bold green]🎒🔧 HANDBAG + EXTRA TOOLS TEST[/bold green]",
        border_style="green",
    )

    # ── 1. Create client from env ─────────────────────────────────────────────
    ASCIIColors.info("\n[1/4] Creating LollmsClient from environment...")
    from lollms_client.lollms_config_cli_env import get_client_from_env
    client = get_client_from_env(create_llm=True)
    ASCIIColors.green(f"   Connected: {getattr(client.llm, 'binding_name', '?')} / {getattr(client.llm, 'model_name', '?')}")

    # ── 2. Create LollmsPersonality with handbag ──────────────────────────────
    extra_tool_file = setup_files()

    ASCIIColors.info("\n[2/4] Creating LollmsPersonality from handbag...")
    personality = LollmsPersonality.from_handbag(
        lollms_client=client,
        path=HANDBAG_PATH,
    )
    ASCIIColors.green(f"   ✅ Personality '{personality.name}' loaded.")
    ASCIIColors.cyan(f"   • Handbag tool binding: {type(personality.tools).__name__}")

    handbag_tool_names = [
        t.get("name") for t in personality.tools.discovered_tools
    ] if hasattr(personality.tools, "discovered_tools") else []
    ASCIIColors.cyan(f"   • Handbag tools discovered: {handbag_tool_names}")

    # ── 3. Create LollmsDiscussion and pass the personality ───────────────────
    ASCIIColors.info("\n[3/4] Creating LollmsDiscussion...")
    discussion = LollmsDiscussion.create_new(
        lollms_client=client,
        workspace_path=str(HANDBAG_PATH / "workspace"),
    )
    ASCIIColors.green(f"   ✅ Discussion created: {discussion.id[:8]}...")

    # ── 4. Call chat() with tools list containing the extra tool file path ────
    ASCIIColors.info("\n[4/4] Running chat() with tools=[extra_tool_path]...")

    task_prompt = (
        "Test your tools by doing the following:\n"
        "1. Call tool_handbag_greet with name='Architect'\n"
        "2. Call tool_extra_compute with a=17 and b=25\n"
        "3. Call tool_extra_reverse with text='lollms'\n"
        "Then summarize all three results in a short paragraph and emit <done/>."
    )

    result = discussion.chat(
        user_message=task_prompt,
        personality=personality,
        tools=[extra_tool_file],
        streaming_callback=streaming_callback,
        max_nb_rounds=10,
        temperature=0.2,
        enable_code_execution=False,
    )

    print("\n")

    # ── Verification ──────────────────────────────────────────────────────────
    ASCIIColors.rule("[bold cyan]📊 VERIFICATION REPORT[/bold cyan]")

    ai_msg = result.get("ai_message")
    tool_calls = (ai_msg.metadata or {}).get("tool_calls", []) if ai_msg else []
    tool_call_names = [tc["name"] for tc in tool_calls]
    ASCIIColors.cyan(f"Tools called by LLM: {tool_call_names}")

    checks = {
        "tool_handbag_greet called": "tool_handbag_greet" in tool_call_names,
        "tool_extra_compute called": "tool_extra_compute" in tool_call_names,
        "tool_extra_reverse called": "tool_extra_reverse" in tool_call_names,
    }

    all_passed = True
    for check_name, passed in checks.items():
        if passed:
            ASCIIColors.green(f"   ✅ {check_name}")
        else:
            ASCIIColors.red(f"   ❌ {check_name}")
            all_passed = False

    if ai_msg and ai_msg.content:
        ASCIIColors.panel(
            ai_msg.content[:800],
            title="[bold]📝 Final Response[/bold]",
            border_style="cyan",
        )

    if all_passed:
        ASCIIColors.success("\n🎉 ALL CHECKS PASSED: LLM saw and used both handbag and extra tool sources!")
    else:
        ASCIIColors.error("\n⚠️ SOME CHECKS FAILED: Review the tool calls above.")

    # Cleanup
    discussion.close()
    shutil.rmtree(HANDBAG_PATH, ignore_errors=True)


if __name__ == "__main__":
    main()