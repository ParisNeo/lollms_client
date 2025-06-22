# File: run_lollms_client_with_mcp_example.py

import sys
import os
import shutil
from pathlib import Path
import json
from lollms_client import LollmsClient
# --- Dynamically adjust Python path to find lollms_client ---
# This assumes the example script is in a directory, and 'lollms_client' is
# in a sibling directory or a known relative path. Adjust as needed.
# For example, if script is in 'lollms_client/examples/' and lollms_client code is in 'lollms_client/'
# then the parent of the script's parent is the project root.

# Get the directory of the current script
current_script_dir = Path(__file__).resolve().parent

# Option 1: If lollms_client is in the parent directory of this script's directory
# (e.g. script is in 'project_root/examples' and lollms_client is in 'project_root/lollms_client')
# project_root = current_script_dir.parent
# lollms_client_path = project_root / "lollms_client" # Assuming this is where lollms_client.py and bindings are

# Option 2: If lollms_client package is directly one level up
# (e.g. script is in 'lollms_client/examples' and lollms_client package is 'lollms_client')
project_root_for_lollms_client = current_script_dir.parent
if str(project_root_for_lollms_client) not in sys.path:
    sys.path.insert(0, str(project_root_for_lollms_client))
    print(f"Added to sys.path: {project_root_for_lollms_client}")


# --- Ensure pipmaster is available (core LoLLMs dependency) ---
try:
    import pipmaster as pm
except ImportError:
    print("ERROR: pipmaster is not installed or not in PYTHONPATH.")
    sys.exit(1)

# --- Import LollmsClient and supporting components ---
try:

    from lollms_client.lollms_llm_binding import LollmsLLMBinding # Base for LLM
    from ascii_colors import ASCIIColors, trace_exception
    from lollms_client.lollms_types import MSG_TYPE # Assuming MSG_TYPE is here
except ImportError as e:
    print(f"ERROR: Could not import LollmsClient components: {e}")
    print("Ensure 'lollms_client' package structure is correct and accessible via PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    trace_exception(e)
    sys.exit(1)


# --- Dummy Server Scripts using FastMCP (as per previous successful iteration) ---
TIME_SERVER_PY = """
import asyncio
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("TimeMCP", description="A server that provides the current time.")

@mcp_server.tool(description="Returns the current server time and echoes received parameters.")
def get_current_time(user_id: str = "unknown_user") -> dict:
    return {"time": datetime.now().isoformat(), "params_received": {"user_id": user_id}, "server_name": "TimeServer"}

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
"""

CALCULATOR_SERVER_PY = """
import asyncio
from typing import List, Union
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP("CalculatorMCP", description="A server that performs addition.")

@mcp_server.tool(description="Adds a list of numbers provided in the 'numbers' parameter.")
def add_numbers(numbers: List[Union[int, float]]) -> dict:
    if not isinstance(numbers, list) or not all(isinstance(x, (int, float)) for x in numbers):
        return {"error": "'numbers' must be a list of numbers."}
    return {"sum": sum(numbers), "server_name": "CalculatorServer"}

if __name__ == "__main__":
    mcp_server.run(transport="stdio")
"""


def main():
    ASCIIColors.red("--- Example: Using LollmsClient with StandardMCPBinding ---")

    # --- 1. Setup Temporary Directory for Dummy MCP Servers ---
    example_base_dir = Path(__file__).parent / "temp_mcp_example_servers"
    if example_base_dir.exists():
        shutil.rmtree(example_base_dir)
    example_base_dir.mkdir(exist_ok=True)

    time_server_script_path = example_base_dir / "time_server.py"
    with open(time_server_script_path, "w") as f: f.write(TIME_SERVER_PY)

    calculator_server_script_path = example_base_dir / "calculator_server.py"
    with open(calculator_server_script_path, "w") as f: f.write(CALCULATOR_SERVER_PY)

    # MCP Binding Configuration (for StandardMCPBinding)
    mcp_config = {
        "initial_servers": {
            "time_machine": {
                "command": [sys.executable, str(time_server_script_path.resolve())],
            },
            "calc_unit": {
                "command": [sys.executable, str(calculator_server_script_path.resolve())]
            }
        }
    }
    ASCIIColors.magenta("\n1. Initializing LollmsClient...")
    try:
        client = LollmsClient(
            binding_name="ollama", # Use the dummy LLM binding
            model_name="mistral-nemo:latest",
            mcp_binding_name="standard_mcp",
            mcp_binding_config=mcp_config,
        )
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient: {e}")
        trace_exception(e)
        shutil.rmtree(example_base_dir)
        sys.exit(1)

    if not client.binding:
        ASCIIColors.error("LollmsClient's LLM binding (dummy_llm) failed to load.")
        shutil.rmtree(example_base_dir)
        sys.exit(1)
    if not client.mcp:
        ASCIIColors.error("LollmsClient's MCP binding (standard_mcp) failed to load.")
        client.close() # Close LLM binding if it loaded
        shutil.rmtree(example_base_dir)
        sys.exit(1)

    ASCIIColors.green("LollmsClient initialized successfully with DummyLLM and StandardMCP bindings.")

    # --- 3. Define a streaming callback for generate_with_mcp ---
    def mcp_streaming_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, history: list = None) -> bool:
        if metadata:
            type_info = metadata.get('type', 'unknown_type')
            if msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
                ASCIIColors.cyan(f"MCP Step Start ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
                ASCIIColors.cyan(f"MCP Step End ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
                ASCIIColors.yellow(f"MCP Info ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK: # Part of final answer typically
                ASCIIColors.green(chunk, end="") # type: ignore
            else: # FULL, default, etc.
                ASCIIColors.green(f"MCP Output ({str(msg_type)}, {type_info}): {chunk}")
        else:
             if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                ASCIIColors.green(chunk, end="") # type: ignore
             else:
                ASCIIColors.green(f"MCP Output ({str(msg_type)}): {chunk}")
        sys.stdout.flush()
        return True # Continue streaming

    # --- 4. Use generate_with_mcp ---
    ASCIIColors.magenta("\n2. Calling generate_with_mcp to get current time...")
    time_prompt = "Hey assistant, what time is it right now?"
    time_response = client.generate_with_mcp(
        prompt=time_prompt,
        streaming_callback=mcp_streaming_callback,
        interactive_tool_execution=False # Set to True to test interactive mode
    )
    print() # Newline after streaming
    ASCIIColors.blue(f"Final response for time prompt: {json.dumps(time_response, indent=2)}")

    assert time_response.get("error") is None, f"Time prompt resulted in an error: {time_response.get('error')}"
    assert time_response.get("final_answer"), "Time prompt did not produce a final answer."
    assert len(time_response.get("tool_calls", [])) > 0, "Time prompt should have called a tool."
    assert time_response["tool_calls"][0]["name"] == "time_machine::get_current_time", "Incorrect tool called for time."
    assert "time" in time_response["tool_calls"][0].get("result", {}).get("output", {}), "Time tool result missing time."


    ASCIIColors.magenta("\n3. Calling generate_with_mcp for calculation...")
    calc_prompt = "Can you please calculate the sum of 50, 25, and 7.5 for me?"
    calc_response = client.generate_with_mcp(
        prompt=calc_prompt,
        streaming_callback=mcp_streaming_callback
    )
    print() # Newline
    ASCIIColors.blue(f"Final response for calc prompt: {json.dumps(calc_response, indent=2)}")

    assert calc_response.get("error") is None, f"Calc prompt resulted in an error: {calc_response.get('error')}"
    assert calc_response.get("final_answer"), "Calc prompt did not produce a final answer."
    assert len(calc_response.get("tool_calls", [])) > 0, "Calc prompt should have called a tool."
    assert calc_response["tool_calls"][0]["name"] == "calc_unit::add_numbers", "Incorrect tool called for calculation."
    # The dummy LLM uses hardcoded params [1,2,3] for calc, so result will be 6.
    # A real LLM would extract 50, 25, 7.5.
    # For this dummy test, we check against the dummy's behavior.
    assert calc_response["tool_calls"][0].get("result", {}).get("output", {}).get("sum") == 82.5, "Calculator tool result mismatch for dummy params."


    # --- 5. Cleanup ---
    ASCIIColors.info("Cleaning up temporary server scripts and dummy binding directory...")
    shutil.rmtree(example_base_dir, ignore_errors=True)

    ASCIIColors.red("\n--- LollmsClient with StandardMCPBinding Example Finished Successfully! ---")

if __name__ == "__main__":
    main()