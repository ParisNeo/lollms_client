"""
Diagnostic Example: Sovereign Tool Seeding Verification

This script verifies that a LollmsPersonality only seeds context-relevant tools
(data tools, document tools) based on the workspace contents and the `enable_data_tools`
flag, strictly excluding irrelevant tools (system_shell, execute_python_code, etc.).

It uses a real LollmsClient configured via environment variables.
"""
import sys
import shutil
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lollms_client.lollms_config_cli_env import get_client_from_env
from lollms_client.lollms_personality import LollmsPersonality
from ascii_colors import ASCIIColors


class _ToolAuditorPersonality(LollmsPersonality):
    """
    A specialized subclass that intercepts _build_system_prompt to extract
    the active tools without modifying the core library.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_active_tools: Dict[str, Any] = {}

    def _build_system_prompt(self, active_tools: Optional[Dict] = None) -> str:
        if active_tools is not None:
            self.last_active_tools = active_tools
        return super()._build_system_prompt(active_tools)


def run_diagnostics():
    ASCIIColors.info("=" * 60)
    ASCIIColors.info("🧪 Sovereign Tool Seeding Diagnostic (Live Client)")
    ASCIIColors.info("=" * 60)

    # 1. Initialize Real LollmsClient from Environment
    ASCIIColors.info("\n🔌 Initializing LollmsClient from environment...")
    try:
        lc = get_client_from_env()
        if not lc or not hasattr(lc, "generate_text"):
            ASCIIColors.error("❌ Failed to initialize LollmsClient. Check your .env configuration.")
            return
        ASCIIColors.success("✅ LollmsClient initialized successfully.")
    except Exception as e:
        ASCIIColors.error(f"❌ Exception during client initialization: {e}")
        return

    # Determine model name explicitly to satisfy OpenAI binding requirements
    model_name = None
    if hasattr(lc, "llm") and hasattr(lc.llm, "model_name"):
        model_name = lc.llm.model_name
    if not model_name:
        model_name = "default-model"

    # 2. Create a temporary workspace
    temp_dir = Path(tempfile.mkdtemp(prefix="lollms_tool_test_"))
    ASCIIColors.info(f"📁 Created temp workspace: {temp_dir}")

    try:
        # 3. Seed workspace with context files
        (temp_dir / "data.csv").write_text("id,value\n1,100\n2,200", encoding="utf-8")
        (temp_dir / "report.pdf").write_bytes(b"%PDF-1.4 mock pdf content")
        (temp_dir / "script.py").write_text("print('hello')", encoding="utf-8")

        # 4. Instantiate Auditor Personality
        personality = _ToolAuditorPersonality(
            name="ToolAuditor",
            system_prompt="You are a tool auditing assistant. Your goal is to verify your available tools.",
            workspace_path=temp_dir,
            lollms_client=lc,
        )

        # 5. Run chat with data tools ENABLED
        ASCIIColors.info("\n▶️ Running chat with enable_data_tools=True...")
        try:
            personality.chat(
                prompt="List all tools you have available. Do not use any tools, just list them.",
                use_internal_history=False,
                enable_data_tools=True,
                max_reasoning_steps=1,
                model_name=model_name,
                n_predict=512
            )
        except Exception as chat_ex:
            ASCIIColors.warning(f"Chat execution encountered an error (binding mismatch): {chat_ex}")

        discovered_tools_enabled = list(personality.last_active_tools.keys())

        # 6. Assertions & Reporting (Enabled)
        ASCIIColors.info("\n🔍 Discovered Tools (Data Tools Enabled):")
        for t in discovered_tools_enabled:
            ASCIIColors.info(f"  - {t}")

        expected_present = ["tool_read_file", "tool_write_file", "tool_execute_python_data_query"]
        for t in expected_present:
            if t in discovered_tools_enabled:
                ASCIIColors.success(f"✅ {t} correctly seeded.")
            else:
                ASCIIColors.error(f"❌ {t} MISSING!")

        expected_absent = ["tool_execute_shell_command", "tool_execute_python_code", "tool_search_files"]
        for t in expected_absent:
            if t not in discovered_tools_enabled:
                ASCIIColors.success(f"✅ {t} correctly excluded.")
            else:
                ASCIIColors.error(f"❌ {t} WAS SEEDED (Security Risk)!")

        # 7. Run chat with data tools DISABLED
        ASCIIColors.info("\n▶️ Running chat with enable_data_tools=False...")
        try:
            personality.chat(
                prompt="List all tools you have available. Do not use any tools, just list them.",
                use_internal_history=False,
                enable_data_tools=False,
                max_reasoning_steps=1,
                model_name=model_name,
                n_predict=512
            )
        except Exception as chat_ex:
            ASCIIColors.warning(f"Chat execution encountered an error (binding mismatch): {chat_ex}")

        discovered_tools_disabled = list(personality.last_active_tools.keys())

        # 8. Assertions & Reporting (Disabled)
        ASCIIColors.info("\n🔍 Discovered Tools (Data Tools Disabled):")
        for t in discovered_tools_disabled:
            ASCIIColors.info(f"  - {t}")

        if "tool_execute_python_data_query" not in discovered_tools_disabled:
            ASCIIColors.success("✅ Data query tool correctly disabled when enable_data_tools=False.")
        else:
            ASCIIColors.error("❌ Data query tool was seeded despite enable_data_tools=False!")

        if "tool_read_file" in discovered_tools_disabled:
            ASCIIColors.success("✅ Workspace file tools (tool_read_file) remain active (safe).")
        else:
            ASCIIColors.error("❌ Workspace file tools (tool_read_file) were incorrectly disabled!")

        ASCIIColors.info("\n" + "=" * 60)
        ASCIIColors.success("✅ Diagnostic Complete.")
        ASCIIColors.info("=" * 60)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        ASCIIColors.info(f"🧹 Cleaned up temp workspace.")


if __name__ == "__main__":
    run_diagnostics()