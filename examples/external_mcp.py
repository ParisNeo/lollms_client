# File: run_lollms_client_with_mcp_example.py

import sys
import os
import shutil
from pathlib import Path
import json
import subprocess
from dotenv import load_dotenv # Import the function

# --- Load environment variables from .env file ---
# Load from .env in the current script's directory, or from project root if specified
# You can specify a path: load_dotenv(dotenv_path=Path('.') / '.env')
# By default, it looks for .env in the current working directory or parent directories.
# For simplicity, let's assume .env is next to this script or in a discoverable location.
load_dotenv()

# --- Python Path Adjustment (same as before) ---
current_script_dir = Path(__file__).resolve().parent
project_root_for_lollms_client = current_script_dir.parent
if str(project_root_for_lollms_client) not in sys.path:
    sys.path.insert(0, str(project_root_for_lollms_client))
    print(f"Added to sys.path: {project_root_for_lollms_client}")

# --- Pipmaster and LollmsClient Core Imports (same as before) ---
try:
    import pipmaster as pm
except ImportError:
    print("ERROR: pipmaster is not installed or not in PYTHONPATH.")
    sys.exit(1)

try:
    from lollms_client import LollmsClient
    from lollms_client.lollms_llm_binding import LollmsLLMBinding
    from ascii_colors import ASCIIColors, trace_exception
    from lollms_client.lollms_types import MSG_TYPE
except ImportError as e:
    print(f"ERROR: Could not import LollmsClient components: {e}")
    trace_exception(e)
    sys.exit(1)


# --- Dummy Server Scripts (Time and Calculator - same as before) ---
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

# --- Main Function ---
def main():
    ASCIIColors.red("--- Example: Using LollmsClient with StandardMCPBinding (including external ElevenLabs MCP) ---")

    # --- 1. Setup Temporary Directory for Dummy MCP Servers ---
    example_base_dir = Path(__file__).parent / "temp_mcp_example_servers"
    if example_base_dir.exists():
        shutil.rmtree(example_base_dir)
    example_base_dir.mkdir(exist_ok=True)

    time_server_script_path = example_base_dir / "time_server.py"
    with open(time_server_script_path, "w") as f: f.write(TIME_SERVER_PY)

    calculator_server_script_path = example_base_dir / "calculator_server.py"
    with open(calculator_server_script_path, "w") as f: f.write(CALCULATOR_SERVER_PY)

    # --- 2. MCP Configuration ---
    initial_mcp_servers = {
        "time_machine": {
            "command": [sys.executable, str(time_server_script_path.resolve())],
        },
        "calc_unit": {
            "command": [sys.executable, str(calculator_server_script_path.resolve())]
        }
    }

    # --- Configuration for ElevenLabs MCP Server (Optional) ---
    # Variables are now loaded from .env by load_dotenv() at the start of the script
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_voice_id_from_env = os.getenv("ELEVENLABS_VOICE_ID", "Rachel") # Default if not in .env
    elevenlabs_model_id_from_env = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2") # Default

    uvx_available = False
    try:
        subprocess.run(["uvx", "--version"], capture_output=True, check=True, text=True, timeout=5)
        uvx_available = True
        ASCIIColors.green("uvx command is available.")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        ASCIIColors.yellow("uvx command not found, not working, or timed out. ElevenLabs MCP server (via uvx) will not be configured.")

    if elevenlabs_api_key and uvx_available:
        ASCIIColors.green("ELEVENLABS_API_KEY found (from .env) and uvx available. Configuring ElevenLabs MCP server.")
        initial_mcp_servers["elevenlabs"] = {
            "command": ["uvx"],
            "args": ["elevenlabs-mcp-server"],
            "env": {
                "ELEVENLABS_API_KEY": elevenlabs_api_key,
                "ELEVENLABS_VOICE_ID": elevenlabs_voice_id_from_env,
                "ELEVENLABS_MODEL_ID": elevenlabs_model_id_from_env,
                "ELEVENLABS_OUTPUT_DIR": str(example_base_dir / "elevenlabs_output")
                # Add other ELEVENLABS_ env vars from os.getenv() if needed
            }
        }
        (example_base_dir / "elevenlabs_output").mkdir(exist_ok=True)
    elif not elevenlabs_api_key:
        ASCIIColors.yellow("ELEVENLABS_API_KEY not found in .env file or environment variables. Skipping ElevenLabs MCP server configuration.")
    
    mcp_config = {"initial_servers": initial_mcp_servers}

    # --- 3. Initialize LollmsClient ---
    ASCIIColors.magenta("\n1. Initializing LollmsClient...")
    try:
        client = LollmsClient(
            binding_name="ollama",
            model_name="mistral-nemo:latest",
            mcp_binding_name="standard_mcp",
            mcp_binding_config=mcp_config,
        )
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient: {e}")
        trace_exception(e)
        shutil.rmtree(example_base_dir, ignore_errors=True)
        sys.exit(1)

    if not client.binding:
        ASCIIColors.error("LollmsClient's LLM binding (ollama) failed to load.")
        shutil.rmtree(example_base_dir, ignore_errors=True)
        sys.exit(1)
    if not client.mcp:
        ASCIIColors.error("LollmsClient's MCP binding (standard_mcp) failed to load.")
        if hasattr(client, 'close'): client.close()
        shutil.rmtree(example_base_dir, ignore_errors=True)
        sys.exit(1)
    ASCIIColors.green("LollmsClient initialized successfully.")


    # --- 4. Define Streaming Callback (same as before) ---
    def mcp_streaming_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, history: list = None) -> bool:
        if metadata:
            type_info = metadata.get('type', 'unknown_type')
            if msg_type == MSG_TYPE.MSG_TYPE_STEP_START: ASCIIColors.cyan(f"MCP Step Start ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END: ASCIIColors.cyan(f"MCP Step End ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_INFO: ASCIIColors.yellow(f"MCP Info ({type_info}): {chunk}")
            elif msg_type == MSG_TYPE.MSG_TYPE_CHUNK: ASCIIColors.green(chunk, end="")
            else: ASCIIColors.green(f"MCP Output ({str(msg_type)}, {type_info}): {chunk}")
        else:
             if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: ASCIIColors.green(chunk, end="")
             else: ASCIIColors.green(f"MCP Output ({str(msg_type)}): {chunk}")
        sys.stdout.flush()
        return True

    # --- 5. Use generate_with_mcp with local dummy servers ---
    ASCIIColors.magenta("\n2. Calling generate_with_mcp to get current time (local dummy server)...")
    time_prompt = "Hey assistant, what time is it right now?"
    time_response = client.generate_with_mcp(
        prompt=time_prompt,
        streaming_callback=mcp_streaming_callback
    )
    print() 
    ASCIIColors.blue(f"Final response for time prompt: {json.dumps(time_response, indent=2)}")
    assert time_response.get("error") is None, f"Time prompt error: {time_response.get('error')}"
    assert time_response.get("final_answer"), "Time prompt no final answer."
    assert len(time_response.get("tool_calls", [])) > 0, "Time prompt should call tool."
    if time_response.get("tool_calls"):
        assert time_response["tool_calls"][0]["name"] == "time_machine::get_current_time", "Incorrect tool for time."
        assert "time" in time_response["tool_calls"][0].get("result", {}).get("output", {}), "Time tool result missing time."

    ASCIIColors.magenta("\n3. Calling generate_with_mcp for calculation (local dummy server)...")
    calc_prompt = "Can you sum 50, 25, and 7.5 for me?"
    calc_response = client.generate_with_mcp(
        prompt=calc_prompt,
        streaming_callback=mcp_streaming_callback
    )
    print()
    ASCIIColors.blue(f"Final response for calc prompt: {json.dumps(calc_response, indent=2)}")
    assert calc_response.get("error") is None, f"Calc prompt error: {calc_response.get('error')}"
    assert calc_response.get("final_answer"), "Calc prompt no final answer."
    assert len(calc_response.get("tool_calls", [])) > 0, "Calc prompt should call tool."
    if calc_response.get("tool_calls"):
        assert calc_response["tool_calls"][0]["name"] == "calc_unit::add_numbers", "Incorrect tool for calc."
        assert "sum" in calc_response["tool_calls"][0].get("result", {}).get("output", {}), "Calculator tool result missing sum."


    # --- 6. Interact with ElevenLabs MCP Server (if configured) ---
    if "elevenlabs" in client.mcp.get_binding_config().get("initial_servers", {}):
        ASCIIColors.magenta("\n4. Interacting with ElevenLabs MCP server...")

        ASCIIColors.info("Discovering all available tools (including ElevenLabs)...")
        all_mcp_tools = client.mcp.discover_tools(force_refresh=True, timeout_per_server=45) # Longer timeout for external server
        ASCIIColors.green(f"Discovered {len(all_mcp_tools)} tools in total:")
        for tool in all_mcp_tools:
            # Try to get properties keys from input_schema for a more informative print
            props_keys = "N/A"
            if isinstance(tool.get('input_schema'), dict) and isinstance(tool['input_schema'].get('properties'), dict):
                props_keys = list(tool['input_schema']['properties'].keys())
            print(f"  - Name: {tool.get('name')}, Desc: {tool.get('description')}, Schema Props: {props_keys}")


        elevenlabs_list_voices_tool_name = "elevenlabs::list_voices"
        if any(t['name'] == elevenlabs_list_voices_tool_name for t in all_mcp_tools):
            ASCIIColors.magenta(f"\n4a. Calling '{elevenlabs_list_voices_tool_name}' via LLM prompt...")
            
            list_voices_prompt = "Please list all the available voices from the elevenlabs tool."
            voices_response = client.generate_with_mcp(
                prompt=list_voices_prompt,
                streaming_callback=mcp_streaming_callback,
                max_tool_calls=1 
            )
            print()
            ASCIIColors.blue(f"Final response for ElevenLabs list_voices prompt: {json.dumps(voices_response, indent=2)}")

            assert voices_response.get("error") is None, f"ElevenLabs list_voices error: {voices_response.get('error')}"
            assert voices_response.get("final_answer"), "ElevenLabs list_voices no final answer."
            tool_calls = voices_response.get("tool_calls", [])
            assert len(tool_calls) > 0, "ElevenLabs list_voices should call tool."
            if tool_calls:
                assert tool_calls[0]["name"] == elevenlabs_list_voices_tool_name, "Incorrect tool for ElevenLabs list_voices."
                tool_output = tool_calls[0].get("result", {}).get("output")
                assert isinstance(tool_output, list), f"ElevenLabs list_voices output not a list, got: {type(tool_output)}"
                if tool_output:
                     ASCIIColors.green(f"First voice from ElevenLabs: {tool_output[0].get('name')} (ID: {tool_output[0].get('voice_id')})")
        else:
            ASCIIColors.yellow(f"Tool '{elevenlabs_list_voices_tool_name}' not found. Skipping ElevenLabs tool execution test.")
    else:
        ASCIIColors.yellow("ElevenLabs MCP server not configured in this run (check .env for API key and uvx availability). Skipping ElevenLabs tests.")

    # --- 7. Cleanup ---
    ASCIIColors.magenta("\n5. Closing LollmsClient and cleaning up...")
    if client and hasattr(client, 'close'):
        try:
            client.close()
        except Exception as e:
            ASCIIColors.error(f"Error closing LollmsClient: {e}")
            trace_exception(e)

    ASCIIColors.info("Cleaning up temporary server scripts directory...")
    shutil.rmtree(example_base_dir, ignore_errors=True)

    ASCIIColors.red("\n--- LollmsClient with MCP Example (including external) Finished ---")

if __name__ == "__main__":
    main()