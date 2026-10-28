"""
Simple Web Search Agent Example

Demonstrates how to use the websearch tool in a discussion CLI app.
The LLM can autonomously search the web and fetch page content when
answering user questions.

The websearch tool file is loaded directly into the session via
ld.chat(tools=[...]), following the proven handbag_extra_tools_example.py
pattern.

Usage:
    python examples/simple/simple_websearch_agent.py

Requirements:
    - A configured LLM (via ~/.lollms_client/config.yaml or env vars)
    - Internet access for web search/fetch
"""

from pathlib import Path

from lollms_client import LollmsClient, MSG_TYPE
from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_discussion import LollmsDiscussion
from ascii_colors import ASCIIColors


# ── WEBSEARCH TOOL PATH ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEBSEARCH_TOOL_PATH = (
    PROJECT_ROOT
    / "src"
    / "lollms_client"
    / "tools_bindings"
    / "lcp"
    / "default_tools"
    / "websearch"
    / "websearch.py"
)

if not WEBSEARCH_TOOL_PATH.exists():
    raise RuntimeError(
        f"Websearch tool file not found at: {WEBSEARCH_TOOL_PATH}\n"
        "Ensure the lollms_client package is properly installed."
    )


def main():
    # Compute workspace path for the discussion
    workspace_path = Path(__file__).parent / "workspaces"
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client from environment/config
    lc: LollmsClient = get_client_from_env()

    # Create discussion with workspace
    ld = LollmsDiscussion(lc, workspace_path=workspace_path)

    def cb(chunk, msg_type, metadata):
        """Streaming callback to print LLM output in real-time."""
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            print(chunk, end="", flush=True)
        return True

    ASCIIColors.panel(
        "[bold]Web Search Agent[/bold]\n"
        "[dim]Ask questions and the LLM will search the web for answers.[/dim]\n"
        "[dim]Type 'exit' or 'quit' to stop.[/dim]",
        title="[bold cyan]🔍 Web Search Agent[/bold cyan]",
        border_style="cyan"
    )

    while True:
        try:
            user_input = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        try:
            out = ld.chat(
                user_input,
                tools=[str(WEBSEARCH_TOOL_PATH)],
                streaming_callback=cb,
                enable_websearch=True,
                enforce_end_tag=True,
            )
            print()
        except Exception as e:
            ASCIIColors.error(f"Error during chat: {e}")


if __name__ == "__main__":
    main()