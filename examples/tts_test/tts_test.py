import numpy as np
import wave
from pathlib import Path

# Assuming your lollms_client is in the same directory or in the python path
from lollms_client.lollms_tts_binding import LollmsTTSBindingManager, get_available_bindings

# --- Test Functions for Each Binding ---

def test_xtts_binding(manager: LollmsTTSBindingManager):
    """Tests the XTTS v2 binding."""
    print("\n" + "="*50)
    print("  Testing XTTS Binding (Voice Cloning)")
    print("="*50)
    
    dummy_voice_path = Path(__file__).parent/"main_voice.wav"
    try:
        # 1. Create the necessary reference voice file

        # 2. Create the binding instance through the manager
        # The config dict could be loaded from a user's settings file
        print("Instantiating XTTS binding...")
        xtts_binding = manager.create_binding("xtts", config={"device": "auto"})
        
        if not xtts_binding:
            print("Failed to create XTTS binding. Test skipped.")
            return

        # 3. Define the text and generate audio
        file_path = Path(__file__).parent/'text_to_speak.md'
        text_to_synthesize = file_path.read_text(encoding='utf-8').strip()
        print(f"Synthesizing text: '{text_to_synthesize}'")
        
        audio_bytes = xtts_binding.generate_audio(
            text=text_to_synthesize,
            voice=str(dummy_voice_path),  # The 'voice' is the path to the reference file
            language="en"
        )

        # 4. Save the output
        output_path = Path(__file__).parent/"test_output_xtts.wav"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        print(f"\nSUCCESS: XTTS audio generated and saved to '{output_path.resolve()}'")

    except Exception as e:
        print(f"\nERROR during XTTS test: {e}")
    finally:
        print(f"Done")

def test_bark_binding(manager: LollmsTTSBindingManager):
    """Tests the Bark binding."""
    print("\n" + "="*50)
    print("  Testing Bark Binding (Generative Audio)")
    print("="*50)

    try:
        # 1. Create the binding instance with a specific config
        print("Instantiating Bark binding (using small model for speed)...")
        bark_binding = manager.create_binding("bark", config={
            "model_name": "suno/bark-small",
            "device": "auto"
        })
        
        if not bark_binding:
            print("Failed to create Bark binding. Test skipped.")
            return

        # 2. List available voices and choose one
        voices = bark_binding.list_voices()
        print("\nAvailable Bark presets:")
        print(", ".join(voices))
        
        chosen_voice = "v2/en_speaker_6"  # A reliable deep male voice
        print(f"\nSelected voice preset: {chosen_voice}")

        # 3. Define the text (including a non-speech sound) and generate audio
        text_to_synthesize = "This is a test of the Bark system. Not all demons are born of fire... [laughter] but of pure logic."
        print(f"Synthesizing text: '{text_to_synthesize}'")
        
        audio_bytes = bark_binding.generate_audio(
            text=text_to_synthesize,
            voice=chosen_voice  # The 'voice' is the preset ID
        )

        # 4. Save the output
        output_path =  Path(__file__).parent/"test_output_bark.wav"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        print(f"\nSUCCESS: Bark audio generated and saved to '{output_path.resolve()}'")

    except Exception as e:
        print(f"\nERROR during Bark test: {e}")


# --- Main Execution Block ---

if __name__ == "__main__":
    print("Starting LoLLMS TTS Binding Framework Test")
    
    # Use the static method to get detailed info from description.yaml files
    print("\n--- Listing Available TTS Bindings from YAMLs ---")
    try:
        bindings_info = get_available_bindings()
        if not bindings_info:
            print("No bindings found. Please check your 'tts_bindings' directory.")
        else:
            for info in bindings_info:
                print(f"\n  - Title: {info.get('title', 'N/A')}")
                print(f"    Binding Name: {info.get('binding_name', 'N/A')}")
                print(f"    Author: {info.get('author', 'N/A')}")
                # Print a snippet of the description
                desc_snippet = info.get('description', 'No description.').strip().split('\n')[0]
                print(f"    Description: {desc_snippet}...")
    except Exception as e:
        print(f"Could not list bindings: {e}")

    # Initialize the manager to create binding instances
    manager = LollmsTTSBindingManager()

    # Run the tests for each binding
    test_xtts_binding(manager)
    test_bark_binding(manager)
    
    print("\n" + "="*50)
    print("All tests completed.")
    print("Check 'test_output_xtts.wav' and 'test_output_bark.wav' to hear the results.")
    print("="*50)