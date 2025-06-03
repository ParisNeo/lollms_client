from lollms_client import LollmsClient # Removed LollmsTTS import
from lollms_client.lollms_types import MSG_TYPE # Import MSG_TYPE if callback uses it
import random
from ascii_colors import ASCIIColors # Assuming this might be used for better output

# Initialize the LollmsClient instance, enabling the TTS binding
# We'll use the 'lollms' tts binding by default.
# The host_address in LollmsClient will be used by the lollms tts binding.
lc = LollmsClient(
    tts_binding_name="lollms" # Explicitly enable the lollms TTS binding
)

if not lc.tts:
    ASCIIColors.error("TTS binding could not be initialized. Please check your LollmsClient setup and server.")
    exit()

voices = lc.tts.list_voices() # Use the new method via lc.tts

# Pick a voice randomly
if voices:
    random_voice = random.choice(voices)
    ASCIIColors.info(f"Selected voice: {random_voice}")
else:
    ASCIIColors.warning("No voices found. Using server default.")
    random_voice = None # Or a known default like "main_voice"

# Generate Text
# response = lc.generate_text(prompt="Once upon a time", stream=False, temperature=0.5)
# print(response)

# # Generate Completion
# response = lc.generate_completion(prompt="What is the capital of France", stream=False, temperature=0.5)
# print(response)


def cb(chunk, msg_type: MSG_TYPE, params=None, metadata=None): # Added params and metadata for full signature
    print(chunk,end="",flush=True)
    return True # Callback should return True to continue streaming

response_text = lc.generate_text(prompt="One plus one equals ", stream=False, temperature=0.5, streaming_callback=cb)
print() # For newline after streaming
ASCIIColors.green(f"Generated text: {response_text}")
print()

if response_text and not isinstance(response_text, dict): # Check if generation was successful
    try:
        # Assuming generate_audio now might return status or file path rather than direct audio bytes for 'lollms' binding
        # based on its current server behavior.
        # If generate_audio for 'lollms' binding is expected to save a file and return status:
        audio_generation_status = lc.tts.generate_audio(response_text, voice=random_voice, fn="output_example_text_2_audio.wav") # Example filename
        ASCIIColors.info(f"Audio generation request status: {audio_generation_status}")
        ASCIIColors.yellow(f"Audio should be saved as 'output_example_text_2_audio.wav' by the server in its default output path.")

    except Exception as e:
        ASCIIColors.error(f"Error during text to audio conversion: {e}")
else:
    ASCIIColors.error(f"Text generation failed or returned an error: {response_text}")


# List Mounted Personalities (This is an LLM feature, specific to 'lollms' LLM binding)
if lc.binding and hasattr(lc.binding, 'lollms_listMountedPersonalities'):
    personalities_response = lc.listMountedPersonalities()
    ASCIIColors.blue("\nMounted Personalities:")
    print(personalities_response)
else:
    ASCIIColors.yellow("\nlistMountedPersonalities not available for the current LLM binding.")


# List Models (This is an LLM feature)
models_response = lc.listModels()
ASCIIColors.blue("\nAvailable LLM Models:")
print(models_response)

# List available TTS bindings (for demonstration)
if hasattr(lc, 'tts_binding_manager'):
    available_tts_bindings = lc.tts_binding_manager.get_available_bindings()
    ASCIIColors.cyan(f"\nAvailable TTS bindings in client: {available_tts_bindings}")