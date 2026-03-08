from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
from pathlib import Path
import json # For pretty printing results

# --- Configuration ---
# LLM Configuration
LLM_BINDING_NAME = "ollama"
OLLAMA_HOST_ADDRESS = "http://localhost:11434" # Default Ollama host
# Ensure you have a model capable of following instructions and generating JSON.
# Mistral, Llama 3, or Phi-3 variants often work well.
OLLAMA_MODEL_NAME = "mistral-nemo:latest" # Or "llama3:latest", "phi3:latest" - ensure it's pulled

# Local MCP Binding Configuration
# This path should point to the directory containing your tool subdirectories
# (e.g., 'get_weather/', 'sum_numbers/')
# For this example, we assume 'temp_mcp_tools_for_test' is in the parent directory
# of this examples folder.
TOOLS_FOLDER = Path(__file__).parent.parent / "temp_mcp_tools_for_test"

# Function Calling Parameters
MAX_LLM_ITERATIONS_FOR_TOOL_CALLS = 3 # How many times LLM can decide to call a tool in a sequence
MAX_TOOL_CALLS_PER_TURN = 2 # Max distinct tools executed per user prompt

# --- Helper to Create Dummy Tools (if they don't exist) ---
def ensure_dummy_tools_exist(base_tools_dir: Path):
    if not base_tools_dir.exists():
        ASCIIColors.info(f"Creating dummy tools directory: {base_tools_dir}")
        base_tools_dir.mkdir(parents=True, exist_ok=True)

    tool_defs = {
        "get_weather": {
            "mcp": {
                "name": "get_weather",
                "description": "Fetches the current weather for a given city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name."},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                    },
                    "required": ["city"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"}, "condition": {"type": "string"}, "unit": {"type": "string"}
                    }
                }
            },
            "py": """
import random
def execute(params: dict) -> dict:
    city = params.get("city")
    unit = params.get("unit", "celsius")
    if not city: return {"error": "City not provided"}
    conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temp = random.randint(-5 if unit == "celsius" else 23, 30 if unit == "celsius" else 86)
    return {"temperature": temp, "condition": random.choice(conditions), "unit": unit}
"""
        },
        "sum_numbers": {
            "mcp": {
                "name": "sum_numbers",
                "description": "Calculates the sum of a list of numbers.",
                "input_schema": {
                    "type": "object",
                    "properties": {"numbers": {"type": "array", "items": {"type": "number"}}},
                    "required": ["numbers"]
                },
                "output_schema": {"type": "object", "properties": {"sum": {"type": "number"}}}
            },
            "py": """
def execute(params: dict) -> dict:
    numbers = params.get("numbers", [])
    if not isinstance(numbers, list) or not all(isinstance(n, (int, float)) for n in numbers):
        return {"error": "Invalid input: 'numbers' must be a list of numbers."}
    return {"sum": sum(numbers)}
"""
        }
    }

    for tool_name, files_content in tool_defs.items():
        tool_dir = base_tools_dir / tool_name
        tool_dir.mkdir(exist_ok=True)
        
        mcp_file = tool_dir / f"{tool_name}.mcp.json"
        py_file = tool_dir / f"{tool_name}.py"

        if not mcp_file.exists():
            with open(mcp_file, "w") as f:
                json.dump(files_content["mcp"], f, indent=2)
            ASCIIColors.info(f"Created MCP definition for {tool_name}")
        
        if not py_file.exists():
            with open(py_file, "w") as f:
                f.write(files_content["py"])
            ASCIIColors.info(f"Created Python code for {tool_name}")

# --- Callback for streaming ---
def function_calling_stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, turn_history: list = None) -> bool:
    """
    Callback to handle streamed output during function calling.
    """
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:  # LLM generating text
        ASCIIColors.success(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        step_type = metadata.get("type", "step") if metadata else "step"
        step_info = metadata.get("tool_name", "") if metadata and "tool_name" in metadata else chunk
        ASCIIColors.info(f">> Starting {step_type}: {step_info}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        step_type = metadata.get("type", "step") if metadata else "step"
        step_info = metadata.get("tool_name", "") if metadata and "tool_name" in metadata else chunk
        if metadata and "result" in metadata:
            ASCIIColors.success(f"<< Finished {step_type}: {step_info} -> Result: {json.dumps(metadata['result'])}", flush=True)
        else:
            ASCIIColors.success(f"<< Finished {step_type}: {step_info}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
        if metadata and metadata.get("type") == "tool_call_request":
            ASCIIColors.info(f"AI requests to call tool: {metadata.get('name')} with params: {metadata.get('params')}", flush=True)
        else:
            ASCIIColors.info(f"INFO: {chunk}", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"ERROR in stream: {chunk}", flush=True)

    # Optional debug info:
    # ASCIIColors.info(f"DEBUG Turn History (so far): {turn_history}")
    return True



def run_function_calling_example():
    ASCIIColors.red("--- LoLLMs Client with Local MCP Function Calling Example ---")
    
    ensure_dummy_tools_exist(TOOLS_FOLDER) # Make sure our example tools are present

    try:
        ASCIIColors.magenta("\n1. Initializing LollmsClient...")
        # MCP binding config is passed directly to the binding constructor
        mcp_binding_configuration = {"tools_folder_path": str(TOOLS_FOLDER)}
        
        lc = LollmsClient(
            binding_name=LLM_BINDING_NAME,
            host_address=OLLAMA_HOST_ADDRESS,
            model_name=OLLAMA_MODEL_NAME,
            mcp_binding_name="local_mcp", # Activate the LocalMCP binding
            mcp_binding_config=mcp_binding_configuration, # Pass its specific config
            # Optional: Configure default LLM generation params if needed
            temperature=0.2, # Lower temp for more focused tool decisions / final answer
            n_predict=1024
        )
        ASCIIColors.green("LollmsClient initialized successfully.")
        if lc.mcp:
            ASCIIColors.info(f"MCP Binding '{lc.mcp.binding_name}' loaded.")
            discovered_tools_on_init = lc.mcp.discover_tools()
            ASCIIColors.info(f"Tools discovered by MCP binding on init: {[t['name'] for t in discovered_tools_on_init]}")
        else:
            ASCIIColors.error("MCP binding was not loaded correctly. Aborting.")
            return

        # --- Example Interaction 1: Weather Request ---
        ASCIIColors.magenta("\n2. Example 1: Asking for weather")
        user_prompt_weather = "What's the weather like in Paris today, and can you tell me in Fahrenheit?"
        ASCIIColors.blue(f"User: {user_prompt_weather}")
        ASCIIColors.yellow(f"AI thinking and interacting with tools (streaming output):")

        weather_result = lc.generate_with_mcp(
            prompt=user_prompt_weather,
            # tools=None, # Let it discover from the binding
            max_tool_calls=MAX_TOOL_CALLS_PER_TURN,
            max_llm_iterations=MAX_LLM_ITERATIONS_FOR_TOOL_CALLS,
            streaming_callback=function_calling_stream_callback,
            # interactive_tool_execution=True # Uncomment to confirm tool calls
        )
        print("\n--- End of AI Response for Weather ---")
        if weather_result["error"]:
            ASCIIColors.error(f"Error in weather example: {weather_result['error']}")
        else:
            ASCIIColors.cyan(f"\nFinal Answer (Weather): {weather_result['final_answer']}")
        ASCIIColors.info("\nTool Calls Made (Weather Example):")
        for tc in weather_result["tool_calls"]:
            print(f"  - Tool: {tc['name']}, Params: {tc['params']}, Result: {json.dumps(tc['result'])}")


        # --- Example Interaction 2: Summation Request ---
        ASCIIColors.magenta("\n3. Example 2: Asking to sum numbers")
        user_prompt_sum = "Hey, can you please calculate the sum of 15.5, 25, and -5.5 for me?"
        ASCIIColors.blue(f"User: {user_prompt_sum}")
        ASCIIColors.yellow(f"AI thinking and interacting with tools (streaming output):")

        sum_result_data = lc.generate_with_mcp(
            prompt=user_prompt_sum,
            max_tool_calls=MAX_TOOL_CALLS_PER_TURN,
            max_llm_iterations=MAX_LLM_ITERATIONS_FOR_TOOL_CALLS,
            streaming_callback=function_calling_stream_callback
        )
        print("\n--- End of AI Response for Sum ---")
        if sum_result_data["error"]:
            ASCIIColors.error(f"Error in sum example: {sum_result_data['error']}")
        else:
            ASCIIColors.cyan(f"\nFinal Answer (Sum): {sum_result_data['final_answer']}")
        ASCIIColors.info("\nTool Calls Made (Sum Example):")
        for tc in sum_result_data["tool_calls"]:
            print(f"  - Tool: {tc['name']}, Params: {tc['params']}, Result: {json.dumps(tc['result'])}")


        # --- Example Interaction 3: Multi-step (hypothetical, weather then sum) ---
        ASCIIColors.magenta("\n4. Example 3: Multi-step (Weather, then maybe sum if AI decides)")
        user_prompt_multi = "What's the weather in Berlin? And also, what's 100 + 200 + 300?"
        ASCIIColors.blue(f"User: {user_prompt_sum}")
        ASCIIColors.yellow(f"AI thinking and interacting with tools (streaming output):")

        multi_result_data = lc.generate_with_mcp(
            prompt=user_prompt_multi,
            max_tool_calls=MAX_TOOL_CALLS_PER_TURN, # Allow up to 2 different tools
            max_llm_iterations=MAX_LLM_ITERATIONS_FOR_TOOL_CALLS +1, # Allow a bit more LLM thinking
            streaming_callback=function_calling_stream_callback
        )
        print("\n--- End of AI Response for Multi-step ---")
        if multi_result_data["error"]:
            ASCIIColors.error(f"Error in multi-step example: {multi_result_data['error']}")
        else:
            ASCIIColors.cyan(f"\nFinal Answer (Multi-step): {multi_result_data['final_answer']}")
        ASCIIColors.info("\nTool Calls Made (Multi-step Example):")
        for tc in multi_result_data["tool_calls"]:
            print(f"  - Tool: {tc['name']}, Params: {tc['params']}, Result: {json.dumps(tc['result'])}")


    except ValueError as ve: # Catch init errors
        ASCIIColors.error(f"Initialization Error: {ve}")
        trace_exception(ve)
    except ConnectionRefusedError:
        ASCIIColors.error(f"Connection refused. Is the Ollama server running at {OLLAMA_HOST_ADDRESS}?")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)
    finally:
        ASCIIColors.info(f"If dummy tools were created, they are in: {TOOLS_FOLDER.resolve()}")
        # Consider cleaning up TOOLS_FOLDER if it was created purely for this test run
        # For this example, we'll leave them.
        # import shutil
        # if "ensure_dummy_tools_exist" in globals() and TOOLS_FOLDER.exists() and "temp_mcp_tools_for_test" in str(TOOLS_FOLDER):
        #     if input(f"Clean up dummy tools at {TOOLS_FOLDER}? (y/n): ").lower() == 'y':
        #         shutil.rmtree(TOOLS_FOLDER)
        #         ASCIIColors.info("Cleaned up dummy tools folder.")

    ASCIIColors.red("\n--- Example Finished ---")

if __name__ == "__main__":
    run_function_calling_example()