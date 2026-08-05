#!/usr/bin/env python3
"""
minimal_agent_example.py
========================
Demonstrates how to create an autonomous agent with the built-in artefact system.
The agent can read, write, search, and execute Python code in an isolated workspace
using the integrated ArtefactManager for context-optimized file management.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ascii_colors import ASCIIColors
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_agent import Agent, AgentRole, CapabilityFlags
from lollms_client.lollms_personality import LollmsPersonality
from lollms_client.lollms_types import MSG_TYPE

_tool_buffer = []
_tool_name = "unknown"

def streaming_callback(chunk: str, msg_type: MSG_TYPE, meta: dict = None) -> bool:
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK and chunk:
        if meta and meta.get("is_processing_block"):
            if meta.get("processing_event") == "start":
                _tool_buffer.clear()
                _tool_name_holder = getattr(streaming_callback, "_tool_name", "unknown")
                import re
                m = re.search(r'title="Tool Execution:\s*([^"]+)"', chunk)
                if m:
                    _tool_name_holder = m.group(1)
                streaming_callback._tool_name = _tool_name_holder
            elif meta.get("processing_event") == "end":
                buffered_content = "".join(_tool_buffer).strip()
                tool_name = getattr(streaming_callback, "_tool_name", "unknown")

                panel_lines = []
                for line in buffered_content.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("<!-- status:"):
                        if "failure" in stripped:
                            panel_lines.append("[bold red]❌ FAILED[/bold red]")
                        elif "success" in stripped:
                            panel_lines.append("[bold green]✅ SUCCESS[/bold green]")
                    elif stripped and not stripped.startswith("<processing"):
                        panel_lines.append(stripped)

                panel_content = "\n".join(panel_lines) if panel_lines else "(no output)"
                ASCIIColors.panel(
                    panel_content,
                    title=f"[bold magenta]🛠️ {tool_name}[/bold magenta]",
                    border_style="magenta"
                )
                _tool_buffer.clear()
            else:
                _tool_buffer.append(chunk)
        else:
            ASCIIColors.rich_print(chunk, end="", flush=True)
    return True

def main():
    ASCIIColors.panel(
        "[bold]Minimal Agent — Built-in Artefact System Demo[/bold]",
        title="[bold green]📦 Minimal Agent[/bold green]",
        border_style="green"
    )

    try:
        with ASCIIColors.status("[cyan]Initializing client...[/cyan]", spinner="dots"):
            client = get_client_from_env()
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration failed: {e}")
        sys.exit(1)

    workspace = Path("./minimal_agent_workspace").resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    personality = LollmsPersonality(
        name="MinimalAgent",
        author="lollms-client",
        category="general",
        description="A minimalist autonomous agent that writes, searches, and executes Python scripts.",
        system_prompt=(
            "You are a Minimal Autonomous Agent.\n"
            "You have access to built-in workspace tools: tool_write_file, tool_read_file, tool_list_files, tool_search_files, and tool_execute_python_code.\n"
            "When asked to perform a task, write a Python script to accomplish it, "
            "execute the script, read the output, and report the results.\n"
            "If you need to find specific content across files, use tool_search_files with a regex pattern.\n"
            "Always emit <done/> when your task is complete."
        ),
    )

    caps = CapabilityFlags(
        enable_code_execution=True,
        enable_workspace_tools=True,
        enable_sub_agents=False,
        enable_model_switching=False,
        enable_skill_creation=False,
        enable_skill_loading=False,
    )

    agent = Agent(
        lc=client,
        personality=personality,
        name="MinimalBot",
        role=AgentRole.IMPLEMENTER,
        workspace_path=str(workspace),
        capabilities=caps,
        enable_artefact_system=True,
        disable_artefact_versioning=True,
        max_tokens_per_turn=4096,
    )

    prompt = (
        "Write a Python script named 'fibonacci.py' that calculates the first 10 numbers of the Fibonacci sequence, "
        "prints them to the console, and then executes the script. "
        "After execution, use tool_search_files to search for the word 'fibonacci' in the workspace to verify the file was created."
    )

    ASCIIColors.panel(f"[cyan]{prompt}[/cyan]", title="[bold]📝 Task[/bold]", border_style="cyan")
    ASCIIColors.rule("[bold green]🤖 Agent output[/bold green]")

    result = agent.chat(
        prompt=prompt,
        streaming_callback=streaming_callback,
        max_reasoning_steps=15,
        temperature=0.2,
    )
    print()
    ASCIIColors.rule("[bold cyan]📊 EXECUTION METRICS[/bold cyan]")
    
    metrics_rows = [
        ["Total rounds", str(result['rounds'])],
        ["Tool calls made", str(len(result['tool_calls']))]
    ]
    metrics_table = ASCIIColors.table(
        "Metric", "Value",
        rows=metrics_rows,
        title="[bold]Execution Summary[/bold]",
        box="round"
    )
    ASCIIColors.rich_print(metrics_table)
    
    if result['tool_calls']:
        tool_rows = []
        for tc in result['tool_calls']:
            params_str = json.dumps(tc['parameters'], indent=2)
            tool_rows.append([f"Round {tc['round']}", tc['name'], params_str])
            
        tools_table = ASCIIColors.table(
            "Round", "Tool", "Parameters",
            rows=tool_rows,
            title="[bold magenta]🛠️ Tool Calls[/bold magenta]",
            box="round"
        )
        ASCIIColors.rich_print(tools_table)

    ASCIIColors.rule("[bold cyan]📝 FINAL RESPONSE[/bold cyan]")
    ASCIIColors.rich_print("[bold cyan]Agent Response[/bold cyan]")
    ASCIIColors.rich_print(result["response"])

if __name__ == "__main__":
    main()