from lollms_client import LollmsClient, ELF_COMPLETION_FORMAT
from lollms_client.lollms_types import MSG_TYPE # For callback signature
from ascii_colors import ASCIIColors, trace_exception

# --- Configuration ---
# Choose your LLM binding and parameters here
# Option 1: Default LOLLMS server binding
# BINDING_NAME = "lollms"
# HOST_ADDRESS = "http://localhost:9600"
# MODEL_NAME = None # Server will use its default or last loaded model

# Option 2: Ollama binding
# BINDING_NAME = "ollama"
# HOST_ADDRESS = "http://localhost:11434" # Default Ollama host
# MODEL_NAME = "llava:latest" # Or "llama3:latest", "phi3:latest", etc. - ensure it's pulled in Ollama

# Option 2: llamacpp binding
BINDING_NAME = "llamacpp"
MODELS_PATH = r"E:\drumber" # Change to your own models folder
MODEL_NAME = "llava-v1.6-mistral-7b.Q3_K_XS.gguf" # Change to your vision capable model (make sure you have a mmprj file with the gguf model with the same name but without the quantization name and with mmproj- prefix (mmproj-llava-v1.6-mistral-7b.gguf))
# You can also add a clip_model_path parameter to your lc_params
img = "E:\\drumber\\1711741182996.jpg"
# Option 3: OpenAI binding (requires OPENAI_API_KEY environment variable or service_key)
# BINDING_NAME = "openai"
# HOST_ADDRESS = None # Defaults to OpenAI API
# MODEL_NAME = "gpt-3.5-turbo"
# SERVICE_KEY = "sk-your_openai_api_key_here" # Optional, can use env var

# --- Callback for streaming ---
def simple_streaming_callback(chunk: str, msg_type: MSG_TYPE, params=None, metadata=None) -> bool:
    """
    Simple callback function to print streamed text chunks.
    """
    if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
        ASCIIColors.error(f"\nStreaming Error: {chunk}")
    # Return True to continue streaming, False to stop
    return True

def test_text_generation():
    ASCIIColors.cyan(f"\n--- Testing Text Generation with '{BINDING_NAME}' binding ---")

    if BINDING_NAME!="llamacpp":
        ASCIIColors.cyan(f"Host: {HOST_ADDRESS or 'Default'}, Model: {MODEL_NAME or 'Default'}")
    else:
        ASCIIColors.cyan(f"Host: {MODELS_PATH or 'Default'}, Model: {MODEL_NAME or 'Default'}")
    try:
        # Initialize LollmsClient
        lc_params = {
            "binding_name": BINDING_NAME,
            "model_name": MODEL_NAME,
            # "service_key": SERVICE_KEY, # Uncomment for OpenAI if needed
        }
        if BINDING_NAME!="llamacpp":
            lc_params["host_address"]= HOST_ADDRESS
            # Remove None host_address for bindings that have internal defaults (like OpenAI)
            if lc_params["host_address"] is None and BINDING_NAME in ["openai"]:
                del lc_params["host_address"]
        else:
            lc_params["models_path"]= MODELS_PATH


        lc = LollmsClient(**lc_params)

        # 1. Test basic non-streaming generation
        ASCIIColors.magenta("\n1. Basic Non-Streaming Generation:")
        prompt_non_stream = "describe this image"
        ASCIIColors.yellow(f"Prompt: {prompt_non_stream}")
        response_non_stream = lc.generate_text(
            prompt=prompt_non_stream,
            images=[img],
            stream=False,
            temperature=0.7,
            n_predict=100 # Max tokens for the joke
        )

        if isinstance(response_non_stream, str):
            ASCIIColors.green("Response:")
            print(response_non_stream)
        elif isinstance(response_non_stream, dict) and "error" in response_non_stream:
            ASCIIColors.error(f"Error in non-streaming generation: {response_non_stream['error']}")
        else:
            ASCIIColors.warning(f"Unexpected response format: {response_non_stream}")

        # 2. Test streaming generation
        ASCIIColors.magenta("\n\n2. Streaming Generation:")
        prompt_stream = "Explain the concept of recursion in one sentence."
        ASCIIColors.yellow(f"Prompt: {prompt_stream}")
        ASCIIColors.green("Response (streaming):")
        response_stream = lc.generate_text(
            prompt=prompt_stream,
            stream=True,
            streaming_callback=simple_streaming_callback,
            temperature=0.5,
            n_predict=150
        )
        print() # Newline after streaming

        # The 'response_stream' variable will contain the full concatenated text if streaming_callback returns True throughout
        # or an error dictionary if generation failed.
        if isinstance(response_stream, str):
            ASCIIColors.cyan(f"\n(Full streamed text was: {response_stream[:100]}...)") # Show a snippet of full text
        elif isinstance(response_stream, dict) and "error" in response_stream:
            ASCIIColors.error(f"Error in streaming generation: {response_stream['error']}")
        # else: if callback returns False early, response_stream might be partial.

        # 3. Test generation with a specific model (if applicable and different from default)
        #    This tests the switch_model or model loading mechanism of the binding.
        #    For 'lollms' binding, this would set the model on the server.
        #    For 'ollama' or 'openai', it means the next generate_text will use this model.
        ASCIIColors.magenta("\n\n3. List Available Models & Generate with Specific Model:")
        available_models = lc.listModels()
        if isinstance(available_models, list) and available_models:
            ASCIIColors.green("Available models:")
            for i, model_info in enumerate(available_models[:5]): # Print first 5
                model_id = model_info.get('model_name', model_info.get('id', str(model_info)))
                print(f"  - {model_id}")

            # Try to use the first available model (or a known one if list is too generic)
            target_model = None
            if BINDING_NAME == "ollama":
                # For Ollama, try using a different small model if available, or the same one
                if "phi3:latest" in [m.get('name') for m in available_models if isinstance(m, dict)]:
                    target_model = "phi3:latest"
                elif available_models: # Fallback to first model in list if phi3 not present
                     first_model_entry = available_models[0]
                     target_model = first_model_entry.get('name', first_model_entry.get('model_name'))


            elif BINDING_NAME == "lollms":
                # For lollms, this would typically be a path or server-recognized name
                # This part is harder to make generic without knowing server's models
                ASCIIColors.yellow("For 'lollms' binding, ensure the target model is known to the server.")
                if available_models and isinstance(available_models[0], str): # e.g. gptq model paths
                    target_model = available_models[0]


            if target_model and target_model != lc.binding.model_name: # Only if different and valid
                ASCIIColors.info(f"\nSwitching to model (or using for next gen): {target_model}")
                # For bindings like ollama/openai, setting model_name on binding directly works.
                # For 'lollms' server binding, LollmsClient doesn't have a direct 'switch_model_on_server'
                # but setting lc.binding.model_name will make the next generate_text request it.
                lc.binding.model_name = target_model # Update the binding's current model_name

                prompt_specific_model = f"What is the main capability of the {target_model.split(':')[0]} language model?"
                ASCIIColors.yellow(f"Prompt (for {target_model}): {prompt_specific_model}")
                ASCIIColors.green("Response:")
                response_specific = lc.generate_text(
                    prompt=prompt_specific_model,
                    stream=True, # Keep it streaming for responsiveness
                    streaming_callback=simple_streaming_callback,
                    n_predict=200
                )
                print()
            elif target_model == lc.binding.model_name:
                ASCIIColors.yellow(f"Target model '{target_model}' is already the current model. Skipping specific model test.")
            else:
                ASCIIColors.yellow("Could not determine a different target model from the list to test specific model generation.")

        elif isinstance(available_models, dict) and "error" in available_models:
            ASCIIColors.error(f"Error listing models: {available_models['error']}")
        else:
            ASCIIColors.yellow("No models listed by the binding or format not recognized.")


    except ValueError as ve:
        ASCIIColors.error(f"Initialization Error: {ve}")
        trace_exception(ve)
    except RuntimeError as re:
        ASCIIColors.error(f"Runtime Error (binding likely not initialized): {re}")
        trace_exception(re)
    except Exception as e:
        ASCIIColors.error(f"An unexpected error occurred: {e}")
        trace_exception(e)

if __name__ == "__main__":
    test_text_generation()