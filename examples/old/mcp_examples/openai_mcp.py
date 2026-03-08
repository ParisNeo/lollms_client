# File: run_openai_mcp_example.py
# (Keep imports, path adjustments, helper functions, and initial setup as before)

import sys
import os
import shutil
from pathlib import Path
import json
import base64
from dotenv import load_dotenv

load_dotenv() # For this script's own potential .env

try:
    from lollms_client import LollmsClient
    from ascii_colors import ASCIIColors, trace_exception
    from lollms_client.lollms_types import MSG_TYPE
except ImportError as e:
    print(f"ERROR: Could not import LollmsClient components: {e}")
    trace_exception(e)
    sys.exit(1)

PATH_TO_OPENAI_MCP_SERVER_PROJECT = Path(__file__).resolve().parent # Standard if script is in PArisNeoMCPServers root
if not PATH_TO_OPENAI_MCP_SERVER_PROJECT.is_dir():
    print(f"ERROR: openai-mcp-server project not found at {PATH_TO_OPENAI_MCP_SERVER_PROJECT}")
    sys.exit(1)

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "mcp_example_outputs"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

def save_base64_audio(base64_str: str, filename_stem: str, audio_format: str) -> Path:
    audio_bytes = base64.b64decode(base64_str)
    file_path = OUTPUT_DIRECTORY / f"{filename_stem}.{audio_format}"
    with open(file_path, "wb") as f: f.write(audio_bytes)
    ASCIIColors.green(f"Audio saved to: {file_path}")
    return file_path

def save_base64_image(base64_str: str, filename_stem: str) -> Path:
    image_bytes = base64.b64decode(base64_str)
    file_path = OUTPUT_DIRECTORY / f"{filename_stem}.png"
    with open(file_path, "wb") as f: f.write(image_bytes)
    ASCIIColors.green(f"Image saved to: {file_path}")
    return file_path

def main():
    ASCIIColors.red(f"--- Example: Using LollmsClient with OpenAI MCP Server (TTS & DALL-E) ---")
    ASCIIColors.red(f"--- Make sure OPENAI_API_KEY is set in: {PATH_TO_OPENAI_MCP_SERVER_PROJECT / '.env'} ---")
    ASCIIColors.red(f"--- And that 'uv pip install -e .' has been run in: {PATH_TO_OPENAI_MCP_SERVER_PROJECT} ---")

    # Determine the Python executable within the server's .venv IF IT EXISTS
    # This is the most robust way to ensure the server runs with its own isolated dependencies.
    path_to_openai_server_venv_python = PATH_TO_OPENAI_MCP_SERVER_PROJECT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    python_exe_to_use = None
    if path_to_openai_server_venv_python.exists():
        python_exe_to_use = str(path_to_openai_server_venv_python.resolve())
        ASCIIColors.cyan(f"Attempting to use Python from server's .venv: {python_exe_to_use}")
    else:
        python_exe_to_use = sys.executable # Fallback to current script's Python
        ASCIIColors.yellow(f"Server's .venv Python not found at {path_to_openai_server_venv_python}. Using current environment's Python: {python_exe_to_use}")
        ASCIIColors.yellow("Ensure openai-mcp-server dependencies are met in the current environment if its .venv is not used.")

    mcp_config = {
        "initial_servers": {
            "my_openai_server": {
                "command": [
                    "uv",       # Use uv to manage the environment for the python execution
                    "run",
                    "--quiet",  # Optional: reduce uv's own output unless there's an error
                    "--",       # Separator: arguments after this are for the command being run by `uv run`
                    python_exe_to_use, # Explicitly specify the Python interpreter
                    str((PATH_TO_OPENAI_MCP_SERVER_PROJECT / "openai_mcp_server" / "server.py").resolve()) # Full path to your server script
                ],
                "args": [], # No *additional* arguments for server.py itself here
                "cwd": str(PATH_TO_OPENAI_MCP_SERVER_PROJECT.resolve()), # CRUCIAL
            }
        }
    }


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
        sys.exit(1)

    if not client.binding or not client.mcp:
        ASCIIColors.error("LollmsClient LLM or MCP binding failed to load.")
        if hasattr(client, 'close'): client.close()
        sys.exit(1)
    ASCIIColors.green("LollmsClient initialized successfully.")

    def mcp_streaming_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, history: list = None) -> bool:
        prefix = ""
        color_func = ASCIIColors.green
        if metadata:
            type_info = metadata.get('type', 'unknown_type')
            tool_name_info = metadata.get('tool_name', '')
            prefix = f"MCP ({type_info}{f' - {tool_name_info}' if tool_name_info else ''})"
            if msg_type == MSG_TYPE.MSG_TYPE_STEP_START: color_func = ASCIIColors.cyan; prefix += " Step Start"
            elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END: color_func = ASCIIColors.cyan; prefix += " Step End"
            elif msg_type == MSG_TYPE.MSG_TYPE_INFO: color_func = ASCIIColors.yellow; prefix += " Info"
            elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION: color_func = ASCIIColors.red; prefix += " Exception"
        else:
            prefix = f"MCP (Type: {str(msg_type).split('.')[-1]})"
        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK: ASCIIColors.green(chunk, end="")
        else: color_func(f"{prefix}: {chunk}")
        sys.stdout.flush()
        return True

    # --- Test 1: General Text Query (handled by Ollama, no MCP tool expected) ---
    ASCIIColors.magenta("\n2. Test: General Text Query (should be handled by Ollama)")
    general_query = "What is the capital of France?"
    general_response = client.generate_with_mcp( # generate_with_mcp will discover no suitable text tool
        prompt=general_query,
        streaming_callback=mcp_streaming_callback,
        # tools=[] # Optionally explicitly pass an empty list of tools if you want to be sure
                  # generate_with_mcp will discover tools from the binding if not passed
    )
    print()
    ASCIIColors.blue(f"Final response for general query: {json.dumps(general_response, indent=2)}")
    assert general_response.get("error") is None, f"General query error: {general_response.get('error')}"
    assert general_response.get("final_answer"), "General query: no final answer."
    tool_calls_general = general_response.get("tool_calls", [])
    assert len(tool_calls_general) == 0, "General query should NOT have called an MCP tool from my_openai_server."
    ASCIIColors.green(f"General query handled by LLM directly, as expected. Answer: {general_response.get('final_answer')[:100]}...")


    # --- Test 2: Text-to-Speech (TTS) ---
    ASCIIColors.magenta("\n3. Test: OpenAI TTS via MCP")
    tts_text = "This audio was generated by the OpenAI MCP server through Lollms Client."
    tts_prompt_for_llm = f"Please use the OpenAI tool to say the following using tts: '{tts_text}'."
    
    tts_response = client.generate_with_mcp(
        prompt=tts_prompt_for_llm,
        streaming_callback=mcp_streaming_callback,
        max_tool_calls=1
    )
    print()
    ASCIIColors.blue(f"Final response for TTS prompt: {json.dumps(tts_response, indent=2)}")

    assert tts_response.get("error") is None, f"TTS error: {tts_response.get('error')}"
    assert tts_response.get("final_answer"), "TTS: no final answer (LLM should confirm action)."
    tool_calls_tts = tts_response.get("tool_calls", [])
    assert len(tool_calls_tts) > 0, "TTS should have called a tool."
    if tool_calls_tts:
        assert tool_calls_tts[0]["name"] == "my_openai_server::generate_tts", "Incorrect tool for TTS."
        tts_result_output = tool_calls_tts[0].get("result", {}).get("output", {})
        assert "audio_base64" in tts_result_output, "TTS tool result missing 'audio_base64'."
        assert "format" in tts_result_output, "TTS tool result missing 'format'."
        if tts_result_output.get("audio_base64"):
            save_base64_audio(tts_result_output["audio_base64"], "openai_tts_example_output", tts_result_output["format"])

    # --- Test 3: DALL-E Image Generation ---
    ASCIIColors.magenta("\n4. Test: OpenAI DALL-E Image Generation via MCP")
    dalle_image_prompt = "A vibrant illustration of a friendly AI robot helping a human plant a tree on a futuristic Earth."
    dalle_prompt_for_llm = f"I need an image for a presentation. Can you use DALL-E to create this: {dalle_image_prompt}. Please use URL format for the image."

    dalle_response = client.generate_with_mcp(
        prompt=dalle_prompt_for_llm,
        streaming_callback=mcp_streaming_callback,
        max_tool_calls=1,
        # You could also try to force params for the tool if LLM struggles:
        # Example: if LLM isn't picking response_format="url"
        # This requires knowing the exact tool name and schema, usually let LLM handle it.
    )
    print()
    ASCIIColors.blue(f"Final response for DALL-E prompt: {json.dumps(dalle_response, indent=2)}")
    
    assert dalle_response.get("error") is None, f"DALL-E error: {dalle_response.get('error')}"
    assert dalle_response.get("final_answer"), "DALL-E: no final answer (LLM should confirm action)."
    tool_calls_dalle = dalle_response.get("tool_calls", [])
    assert len(tool_calls_dalle) > 0, "DALL-E should have called a tool."
    if tool_calls_dalle:
        assert tool_calls_dalle[0]["name"] == "my_openai_server::generate_image_dalle", "Incorrect tool for DALL-E."
        dalle_result_output = tool_calls_dalle[0].get("result", {}).get("output", {})
        assert "images" in dalle_result_output and isinstance(dalle_result_output["images"], list), "DALL-E result missing 'images' list."
        if dalle_result_output.get("images"):
            image_data = dalle_result_output["images"][0]
            if image_data.get("url"):
                ASCIIColors.green(f"DALL-E image URL: {image_data['url']}")
                ASCIIColors.info(f"Revised prompt by DALL-E: {image_data.get('revised_prompt')}")
            elif image_data.get("b64_json"):
                save_base64_image(image_data["b64_json"], "openai_dalle_example_output")
                ASCIIColors.info(f"Revised prompt by DALL-E: {image_data.get('revised_prompt')}")

    ASCIIColors.magenta("\n5. Closing LollmsClient...")
    if client and hasattr(client, 'close'):
        try: client.close()
        except Exception as e: ASCIIColors.error(f"Error closing LollmsClient: {e}"); trace_exception(e)

    ASCIIColors.info(f"Example finished. Check {OUTPUT_DIRECTORY} for any generated files.")
    ASCIIColors.red("\n--- LollmsClient with OpenAI MCP Server (TTS & DALL-E) Example Finished ---")

if __name__ == "__main__":
    main()