"""
test_code_execution.py
======================
A comprehensive example demonstrating how to use LollmsDiscussion to allow an LLM 
to build, execute, and interpret the output of a Python script.

This script relies on a .env file for configuration. 
Copy 'test_code_execution.env.example' to '.env' and modify the values to match 
your local LLM server configuration.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lollms_client import LollmsClient
from lollms_client.lollms_discussion import LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors

def run_test():
    # 1. Load Configuration from .env
    load_dotenv(Path(__file__).parent / ".env")
    
    binding_name = os.getenv("LLM_BINDING_NAME", "ollama")
    model_name = os.getenv("MODEL_NAME", "gemma3:4b")
    host_address = os.getenv("HOST_ADDRESS", "http://localhost:11434")
    verify_ssl = os.getenv("VERIFY_SSL", "false").lower() == "true"
    debug_mode = os.getenv("DEBUG", "true").lower() == "true"
    api_key = os.getenv("API_KEY")

    ASCIIColors.cyan("=" * 60)
    ASCIIColors.cyan("🚀 Starting LollmsDiscussion Code Execution Test")
    ASCIIColors.cyan("=" * 60)
    ASCIIColors.info(f"Binding: {binding_name}")
    ASCIIColors.info(f"Model:   {model_name}")
    ASCIIColors.info(f"Host:    {host_address}")
    ASCIIColors.info(f"SSL:     {'Verified' if verify_ssl else 'Unverified'}")
    if api_key:
        ASCIIColors.info("Auth:    API Key Detected (Gated Service Mode)")
    else:
        ASCIIColors.info("Auth:    None (Local/Open Mode)")
    ASCIIColors.cyan("-" * 60)

    # 2. Initialize LollmsClient
    # We explicitly configure the LCP tools binding to point to the default tools directory
    default_tools_path = PROJECT_ROOT / "src" / "lollms_client" / "tools_bindings" / "lcp" / "default_tools"

    # Build LLM Binding Configuration
    llm_config = {
        "model_name": model_name,
        "host_address": host_address,
        "verify_ssl_certificate": verify_ssl
    }
    
    # Conditionally inject API key for gated services (OpenAI, Mistral, Groq, etc.)
    # Note: The Lollms binding expects the API key under the 'service_key' parameter.
    if api_key:
        llm_config["service_key"] = api_key

    client = LollmsClient(
        llm_binding_name=binding_name,
        llm_binding_config=llm_config,
        tools_binding_name="lcp",
        tools_binding_config={
            "tools_folders": [str(default_tools_path)]
        }
    )

    # 3. Setup Isolated Discussion Workspace
    tmp_workspace = tempfile.mkdtemp(prefix="lollms_code_exec_")
    db_manager = LollmsDataManager("sqlite:///:memory:")

    discussion = LollmsDiscussion.create_new(
        lollms_client=client,
        db_manager=db_manager,
        id="code_exec_test_session",
        autosave=True,
        workspace_path=tmp_workspace
    )

    # 4. Define the Streaming Callback
    # This captures the live stream to print to the console
    # CRITICAL: The callback MUST return True to signal the binding to continue generation.
    # Returning None or False will immediately halt the stream.
    def stream_relay(chunk, msg_type, meta=None):
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
            if not meta or not meta.get("was_processed"):
                print(chunk, end="", flush=True)
            elif meta.get("is_heartbeat"):
                print(f"\n{chunk}", end="", flush=True)
        elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
            print(f"\n[INFO] {chunk}", end="", flush=True)
        return True

    # 5. Define the User Prompt
    user_prompt = (
        "I need to know the 10th Fibonacci number. "
        "Please write a Python script to calculate it, execute the script, "
        "and then tell me the final result."
    )

    ASCIIColors.magenta("\n👤 User:")
    print(user_prompt)
    ASCIIColors.magenta("\n🤖 Assistant:")

    # 6. Execute the Chat Turn
    # CRITICAL: enable_code_execution=True is required to unlock the 
    # tool_execute_python_code LCP tool for the LLM.
    # We use max_reasoning_steps=3 to allow: 1 (tool call) + 1 (execution) + 1 (final answer)
    response = discussion.chat(
        user_message=user_prompt,
        streaming_callback=stream_relay,
        enable_artefacts=True,
        enable_code_execution=True,
        max_reasoning_steps=3,
        debug=debug_mode
    )

    print("\n")
    ASCIIColors.cyan("-" * 60)
    ASCIIColors.cyan("📊 TEST RESULTS & VERIFICATION")
    ASCIIColors.cyan("-" * 60)

    # 7. Verify the Agentic Loop
    ai_msg = response.get("ai_message")
    tool_calls = ai_msg.metadata.get("tool_calls", []) if ai_msg else []
    artifacts_created = response.get("artefacts", [])

    # Check 1: Did the LLM create a Python artifact OR pass code directly to the tool?
    # Modern tool-calling LLMs often skip artifact creation and pass the code string 
    # directly as a parameter to the execution tool. Both paths are valid.
    py_artifacts = [a for a in artifacts_created if a.get("type") == "code" and a.get("language") == "python"]
    direct_code_calls = [tc for tc in tool_calls if tc["name"] == "tool_execute_python_code" and tc["params"].get("code")]
    
    if py_artifacts or direct_code_calls:
        ASCIIColors.green(f"✅ [PASS] LLM generated Python code ({len(py_artifacts)} artifact(s), {len(direct_code_calls)} direct call(s)).")
        for art in py_artifacts:
            ASCIIColors.info(f"   - File: {art['title']} (v{art['version']})")
    else:
        ASCIIColors.red("❌ [FAIL] LLM did not generate any Python code (neither as artifact nor direct tool call).")

    # Check 2: Did the LLM execute the code?
    code_executed = any(tc["name"] == "tool_execute_python_code" for tc in tool_calls)
    if code_executed:
        ASCIIColors.green("✅ [PASS] LLM executed the Python code via tool_execute_python_code.")
    else:
        ASCIIColors.red("❌ [FAIL] LLM did not execute any Python code.")

    # Check 3: Did the LLM interpret the result?
    # The 10th Fibonacci number is 55 (starting F0=0, F1=1: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55)
    # We check if "55" appears in the final message content.
    final_content = ai_msg.content if ai_msg else ""
    if "55" in final_content:
        ASCIIColors.green("✅ [PASS] LLM interpreted the output and included '55' in the final answer.")
    else:
        ASCIIColors.red("❌ [FAIL] LLM failed to interpret the output or '55' is missing from the final answer.")

    ASCIIColors.cyan("-" * 60)
    
    # 8. Cleanup
    discussion.close()
    shutil.rmtree(tmp_workspace, ignore_errors=True)
    ASCIIColors.yellow("🧹 Cleaned up temporary workspace.")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        ASCIIColors.red(f"\n💥 Test crashed with error: {e}")
        import traceback
        traceback.print_exc()
