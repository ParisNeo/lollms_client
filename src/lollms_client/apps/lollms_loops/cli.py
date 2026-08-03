import argparse
import os
import sys
from pathlib import Path

from ascii_colors import ASCIIColors

def main():
    parser = argparse.ArgumentParser(description="Lollms Loops: Autonomous Task Execution")
    parser.add_argument("task_file", help="Path to JSON file defining the task profile.")
    parser.add_argument("--env", default=None, help="Path to a specific .env file to use.")
    parser.add_argument("--binding", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skills-dir", default=None)
    args = parser.parse_args()

    from lollms_client.lollms_config_cli_env import get_client_from_env

    try:
        with ASCIIColors.status("[cyan]Initializing client...[/cyan]", spinner="dots"):
            client = get_client_from_env(args.env, create_llm=True, create_tti=False, create_tts=False, create_stt=False, create_ttm=False, create_ttv=False)
    except Exception as e:
        ASCIIColors.red(f"❌ Configuration failed: {e}")
        sys.exit(1)

    from lollms_client.apps.lollms_loops.loop_builder import LollmsLoop, TaskProfile, LoopStatus
    import json

    if not Path(args.task_file).exists():
        ASCIIColors.red(f"❌ Task file not found: {args.task_file}")
        sys.exit(1)

    with open(args.task_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    profile = TaskProfile(
        goal=task_data["goal"],
        success_criteria=task_data["success_criteria"],
        allowed_tools=task_data.get("allowed_tools", []),
        max_reasoning_steps=task_data.get("max_reasoning_steps", 50),
        timeout_seconds=task_data.get("timeout_seconds", 300),
        temperature=task_data.get("temperature", 0.3),
        enable_code_execution=task_data.get("enable_code_execution", False),
        enable_sub_agents=task_data.get("enable_sub_agents", False),
        enable_internet=task_data.get("enable_internet", False),
        enable_file_ops=task_data.get("enable_file_ops", True),
        workspace_path=task_data.get("workspace_path"),
        system_prompt_override=task_data.get("system_prompt_override"),
    )

    loop = LollmsLoop(
        client=client,
        task_profile=profile,
        skills_dir=args.skills_dir,
    )

    result = loop.run()

    report_rows = [
        ["Status", result.status.value.upper()],
        ["Rounds", str(result.rounds)],
        ["Time Elapsed", f"{result.elapsed_time:.2f}s"],
        ["Tools Called", str(len(result.tool_calls))]
    ]
    if result.error:
        report_rows.append(["Error", result.error])
        
    ASCIIColors.table(
        "Metric", "Value",
        rows=report_rows,
        title="[bold cyan]📊 LOOP EXECUTION REPORT[/bold cyan]",
        box="round"
    )

    if result.status == LoopStatus.SUCCESS:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()