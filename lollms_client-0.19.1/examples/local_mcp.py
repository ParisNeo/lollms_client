from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception
from pathlib import Path
import json # For pretty printing results
import os # For OPENAI_API_KEY

# --- Configuration ---
# LLM Configuration
LLM_BINDING_NAME = "ollama" # Or "openai", "lollms", etc.
OLLAMA_HOST_ADDRESS = "http://localhost:11434" 
OLLAMA_MODEL_NAME = "mistral:latest" # Ensure this model is capable of JSON and tool use decisions

# TTI Configuration (for the generate_image_from_prompt MCP tool)
TTI_BINDING_NAME = "dalle" # To use DALL-E via LollmsClient's TTI
# OPENAI_API_KEY should be set as an environment variable for DALL-E

# MCP Configuration
# We will NOT provide mcp_binding_config.tools_folder_path,
# so LocalMCPBinding should use its packaged default_tools.
MCP_BINDING_NAME = "local_mcp"

# Function Calling Parameters
MAX_LLM_ITERATIONS_FOR_TOOL_CALLS = 4
MAX_TOOL_CALLS_PER_TURN = 3

# --- Callback for streaming ---
def function_calling_stream_callback(chunk: str, msg_type: MSG_TYPE, metadata: dict = None, turn_history: list = None) -> bool:
    """
    Callback to handle streamed output during function calling.
    """
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        ASCIIColors.success(chunk, end="", flush=True)

    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_START:
        step_type = metadata.get("type", "step") if metadata else "step"
        step_info = metadata.get("tool_name", "") if metadata and "tool_name" in metadata else chunk
        ASCIIColors.info(f"\n>> Starting {step_type}: {step_info}", flush=True)

    elif msg_type == MSG_TYPE.MSG_TYPE_STEP_END:
        step_type = metadata.get("type", "step") if metadata else "step"
        step_info = metadata.get("tool_name", "") if metadata and "tool_name" in metadata else chunk
        if metadata and "result" in metadata:
            ASCIIColors.success(f"\n<< Finished {step_type}: {step_info} -> Result: {json.dumps(metadata['result'])}", flush=True)
        else:
            ASCIIColors.success(f"\n<< Finished {step_type}: {step_info}", flush=True)

    elif msg_type == MSG_TYPE.MSG_TYPE_INFO:
        if metadata and metadata.get("type") == "tool_call_request":
            ASCIIColors.info(f"\nAI requests to call tool: {metadata.get('name')} with params: {metadata.get('params')}", flush=True)
        else:
            ASCIIColors.info(f"\nINFO: {chunk}", flush=True)

    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nERROR in stream: {chunk}", flush=True)

    return True



def run_default_tools_example():
    ASCIIColors.red("--- LoLLMs Client with Default Local MCP Tools Example ---")
    
    # Check for OpenAI API Key if DALL-E is used
    if TTI_BINDING_NAME.lower() == "dalle" and not os.getenv("OPENAI_API_KEY"):
        ASCIIColors.error("OPENAI_API_KEY environment variable is not set. DALL-E TTI will fail.")
        ASCIIColors.error("Please set it or choose a different TTI_BINDING_NAME.")
        # return # Optionally exit if key is critical for the test

    try:
        ASCIIColors.magenta("\n1. Initializing LollmsClient...")
        
        lc = LollmsClient(
            binding_name=LLM_BINDING_NAME,
            host_address=OLLAMA_HOST_ADDRESS, # For Ollama LLM
            model_name=OLLAMA_MODEL_NAME,     # For Ollama LLM
            
            mcp_binding_name=MCP_BINDING_NAME,
            # No mcp_binding_config, so LocalMCPBinding should use its 'default_tools'
            
            tti_binding_name=TTI_BINDING_NAME, # For the 'generate_image_from_prompt' tool
            # tti_binding_config would be needed here if DALL-E or other TTI bindings
            # require specific init params beyond API key (which DALL-E binding gets from env).
            # e.g. tti_binding_config={"api_key": "your_key_here"} if not using env for DALL-E.
            
            temperature=0.1, 
            n_predict=1500 # Allow more tokens for complex reasoning and tool outputs
        )
        ASCIIColors.green("LollmsClient initialized successfully.")
        if lc.mcp:
            ASCIIColors.info(f"MCP Binding '{lc.mcp.binding_name}' loaded.")
            discovered_tools_on_init = lc.mcp.discover_tools() # Should pick up default_tools
            ASCIIColors.info(f"Tools initially discovered by MCP binding: {[t['name'] for t in discovered_tools_on_init]}")
            assert any(t['name'] == 'internet_search' for t in discovered_tools_on_init), "Default 'internet_search' tool not found."
            assert any(t['name'] == 'file_writer' for t in discovered_tools_on_init), "Default 'file_writer' tool not found."
            assert any(t['name'] == 'python_interpreter' for t in discovered_tools_on_init), "Default 'python_interpreter' tool not found."
            assert any(t['name'] == 'generate_image_from_prompt' for t in discovered_tools_on_init), "Default 'generate_image_from_prompt' tool not found."
        else:
            ASCIIColors.error("MCP binding was not loaded correctly. Aborting.")
            return
        
        if TTI_BINDING_NAME and not lc.tti:
            ASCIIColors.warning(f"TTI binding '{TTI_BINDING_NAME}' was specified but not loaded in LollmsClient. The 'generate_image_from_prompt' tool may fail.")


        # --- Example Interaction 1: Internet Search ---
        ASCIIColors.magenta("\n2. Example: Asking for information requiring internet search")
        user_prompt_search = "What were the main headlines on AI ethics in the last month?"
        ASCIIColors.blue(f"User: {user_prompt_search}")
        ASCIIColors.yellow(f"AI processing (streaming output):")

        search_result_data = lc.generate_with_mcp(
            prompt=user_prompt_search,
            max_tool_calls=1, # Limit to one search for this
            max_llm_iterations=2,
            streaming_callback=function_calling_stream_callback,
        )
        print("\n--- End of AI Response (Search) ---")
        if search_result_data["error"]:
            ASCIIColors.error(f"Error in search example: {search_result_data['error']}")
        else:
            ASCIIColors.cyan(f"\nFinal Answer (Search): {search_result_data['final_answer']}")
        ASCIIColors.info("\nTool Calls Made (Search Example):")
        for tc in search_result_data["tool_calls"]:
            # Truncate long snippets for display
            if tc['name'] == 'internet_search' and 'output' in tc['result'] and 'search_results' in tc['result']['output']:
                for res_item in tc['result']['output']['search_results']:
                    if 'snippet' in res_item and len(res_item['snippet']) > 100:
                        res_item['snippet'] = res_item['snippet'][:100] + "..."
            print(f"  - Tool: {tc['name']}, Params: {tc['params']}, Result: {json.dumps(tc['result'], indent=2)}")


        # --- Example Interaction 2: Image Generation ---
        ASCIIColors.magenta("\n3. Example: Requesting an image generation")
        user_prompt_image = "Please generate an image of a futuristic robot holding a glowing orb."
        ASCIIColors.blue(f"User: {user_prompt_image}")
        ASCIIColors.yellow(f"AI processing (streaming output):")

        image_gen_result_data = lc.generate_with_mcp(
            prompt=user_prompt_image,
            max_tool_calls=1,
            max_llm_iterations=2,
            streaming_callback=function_calling_stream_callback,
        )
        print("\n--- End of AI Response (Image Gen) ---")
        if image_gen_result_data["error"]:
            ASCIIColors.error(f"Error in image gen example: {image_gen_result_data['error']}")
        else:
            ASCIIColors.cyan(f"\nFinal Answer (Image Gen): {image_gen_result_data['final_answer']}")
        ASCIIColors.info("\nTool Calls Made (Image Gen Example):")
        for tc in image_gen_result_data["tool_calls"]:
            print(f"  - Tool: {tc['name']}, Params: {tc['params']}, Result: {json.dumps(tc['result'], indent=2)}")
            if tc['name'] == 'generate_image_from_prompt' and tc['result'].get('output', {}).get('status') == 'success':
                img_path = tc['result']['output'].get('image_path')
                img_url = tc['result']['output'].get('image_url')
                ASCIIColors.green(f"Image was reportedly saved. Path hint: {img_path}, URL: {img_url}")
                ASCIIColors.info("Check your LollmsClient outputs/mcp_generated_images/ directory (or similar based on tool's save logic).")


    except ValueError as ve: 
        ASCIIColors.error(f"Initialization Error: {ve}")
        trace_exception(ve)
    except ConnectionRefusedError:
        ASCIIColors.error(f"Connection refused. Is the Ollama server running at {OLLAMA_HOST_ADDRESS}?")
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)

    ASCIIColors.red("\n--- Default Tools Example Finished ---")

if __name__ == "__main__":
    run_default_tools_example()