# lollms_client/examples/text_and_speech_demo/generate_and_speak.py
from pathlib import Path
import time
import argparse

# Ensure pygame is installed for this example
try:
    import pipmaster as pm
    pm.ensure_packages(["pygame"])
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    print("Pygame not found or pipmaster failed. Please install it manually: pip install pygame")
    PYGAME_AVAILABLE = False
except Exception as e:
    print(f"Could not ensure pygame: {e}")
    PYGAME_AVAILABLE = False

from lollms_client import LollmsClient, MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

# --- Configuration ---
SPEECH_OUTPUT_DIR = Path(__file__).parent / "speech_output"
SPEECH_OUTPUT_DIR.mkdir(exist_ok=True)

# Default path for Piper voices relative to this example script for convenience
DEFAULT_PIPER_VOICES_SUBDIR = Path(__file__).parent / "piper_voices_for_demo"
DEFAULT_PIPER_VOICE_FILENAME = "en_US-lessac-medium.onnx" # A common, good quality English voice

def text_stream_callback(chunk: str, message_type: MSG_TYPE, params: dict = None, metadata: list = None) -> bool:
    if message_type == MSG_TYPE.MSG_TYPE_CHUNK:
        print(chunk, end="", flush=True)
    elif message_type == MSG_TYPE.MSG_TYPE_STEP_START:
        ASCIIColors.yellow(f"\n>> Starting step: {chunk}")
    elif message_type == MSG_TYPE.MSG_TYPE_STEP_END:
        ASCIIColors.green(f"\n<< Finished step: {chunk}")
    return True

def ensure_default_piper_voice_for_demo(voices_dir: Path, voice_filename: str):
    """Helper to download a default Piper voice if not present for the demo."""
    voices_dir.mkdir(exist_ok=True)
    onnx_path = voices_dir / voice_filename
    json_path = voices_dir / f"{voice_filename}.json"

    if not onnx_path.exists() or not json_path.exists():
        ASCIIColors.info(f"Default Piper test voice '{voice_filename}' not found in {voices_dir}. Attempting to download...")
        try:
            import requests
            # Construct URLs (assuming en_US/lessac/medium structure)
            voice_parts = voice_filename.split('-') # e.g., ['en_US', 'lessac', 'medium.onnx']
            lang_code = voice_parts[0].split('_')[0] # en
            voice_name_path = "/".join(voice_parts[0:2]) # en_US/lessac
            quality_path = voice_parts[2].split('.')[0] # medium

            # Base URL for Piper voices on Hugging Face
            PIPER_VOICES_HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

            onnx_url = f"{PIPER_VOICES_HF_BASE_URL}{lang_code}/{voice_name_path}/{quality_path}/{voice_filename}"
            json_url = f"{PIPER_VOICES_HF_BASE_URL}{lang_code}/{voice_name_path}/{quality_path}/{voice_filename}.json"


            if not onnx_path.exists():
                ASCIIColors.info(f"Downloading {onnx_url} to {onnx_path}")
                r_onnx = requests.get(onnx_url, stream=True)
                r_onnx.raise_for_status()
                with open(onnx_path, 'wb') as f:
                    for chunk in r_onnx.iter_content(chunk_size=8192): f.write(chunk)
            
            if not json_path.exists():
                ASCIIColors.info(f"Downloading {json_url} to {json_path}")
                r_json = requests.get(json_url)
                r_json.raise_for_status()
                with open(json_path, 'w', encoding='utf-8') as f: f.write(r_json.text)
            ASCIIColors.green(f"Default Piper test voice '{voice_filename}' downloaded successfully to {voices_dir}.")
            return True
        except Exception as e_download:
            ASCIIColors.error(f"Failed to download default Piper test voice '{voice_filename}': {e_download}")
            ASCIIColors.warning(f"Please manually download '{voice_filename}' and '{voice_filename}.json' "
                                f"from rhasspy.github.io/piper-voices/ or Hugging Face "
                                f"and place them in {voices_dir.resolve()}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate text with an LLM and synthesize it to speech using LOLLMS.")
    # LLM Arguments
    parser.add_argument(
        "--llm_binding", type=str, default="ollama", choices=["ollama", "openai", "lollms", "llamacpp", "pythonllamacpp", "transformers", "vllm"],
        help="The LLM binding to use for text generation."
    )
    parser.add_argument(
        "--llm_model", type=str, default="mistral",
        help="Model name or path for the LLM binding."
    )
    parser.add_argument("--llm_host", type=str, default=None, help="Host address for server-based LLM bindings.")
    parser.add_argument("--models_path", type=str, default=None, help="Path to models directory for local LLM bindings.")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API key.")

    # TTS Arguments
    parser.add_argument(
        "--tts_binding", type=str, default="bark", choices=["bark", "lollms", "xtts", "piper"],
        help="The TTS binding to use for speech synthesis."
    )
    # Bark specific
    parser.add_argument("--bark_model", type=str, default="suno/bark-small", help="Bark model ID for TTS.")
    parser.add_argument("--bark_voice_preset", type=str, default="v2/en_speaker_6", help="Bark voice preset.")
    # XTTS specific
    parser.add_argument("--xtts_model", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2", help="XTTS model identifier for Coqui TTS.")
    parser.add_argument("--xtts_speaker_wav", type=str, default=None, help="Path to speaker WAV for XTTS voice cloning.")
    parser.add_argument("--xtts_language", type=str, default="en", help="Language for XTTS.")
    # Piper specific
    parser.add_argument("--piper_default_voice_model_path", type=str, default=None, help="Path to the default .onnx Piper voice model.")
    parser.add_argument("--piper_voices_dir", type=str, default=str(DEFAULT_PIPER_VOICES_SUBDIR), help="Directory containing Piper voice models.")
    parser.add_argument("--piper_voice_file", type=str, default=DEFAULT_PIPER_VOICE_FILENAME, help="Filename of the Piper voice to use from piper_voices_dir (e.g., en_US-ryan-medium.onnx).")

    # Common TTS/LLM args
    parser.add_argument("--tts_host", type=str, default=None, help="Host address for server-based TTS bindings (e.g., lollms TTS).")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps", None], help="Device for local TTS/LLM models.")
    args = parser.parse_args()

    ASCIIColors.red("--- LOLLMS Text Generation and Speech Synthesis Demo ---")
    ASCIIColors.info(f"Using LLM Binding: {args.llm_binding} (Model: {args.llm_model})")
    ASCIIColors.info(f"Using TTS Binding: {args.tts_binding}")
    if args.tts_binding == "bark":
        ASCIIColors.info(f"  Bark Model: {args.bark_model}, Voice Preset: {args.bark_voice_preset}")
    elif args.tts_binding == "xtts":
        ASCIIColors.info(f"  XTTS Model: {args.xtts_model}, Speaker WAV: {args.xtts_speaker_wav or 'Default in binding'}, Lang: {args.xtts_language}")
    elif args.tts_binding == "piper":
        ASCIIColors.info(f"  Piper Voices Dir: {args.piper_voices_dir}, Voice File: {args.piper_voice_file}")
        # Ensure default Piper voice for demo if Piper is selected and no specific default path is given
        if not args.piper_default_voice_model_path:
            ensure_default_piper_voice_for_demo(Path(args.piper_voices_dir), args.piper_voice_file)
            args.piper_default_voice_model_path = str(Path(args.piper_voices_dir) / args.piper_voice_file)


    llm_binding_config = {}
    if args.llm_binding == "openai" and args.openai_key: llm_binding_config["service_key"] = args.openai_key
    elif args.llm_binding in ["llamacpp", "pythonllamacpp", "transformers", "vllm"]:
        if args.device: llm_binding_config["device"] = args.device
        if args.llm_binding == "pythonllamacpp": llm_binding_config["n_gpu_layers"] = -1 if args.device == "cuda" else 0

    tts_binding_config = {"device": args.device}
    if args.tts_binding == "bark":
        tts_binding_config["model_name"] = args.bark_model
        tts_binding_config["default_voice"] = args.bark_voice_preset
    elif args.tts_binding == "xtts":
        tts_binding_config["model_name"] = args.xtts_model
        tts_binding_config["default_speaker_wav"] = args.xtts_speaker_wav
        tts_binding_config["default_language"] = args.xtts_language
    elif args.tts_binding == "piper":
        tts_binding_config["default_voice_model_path"] = args.piper_default_voice_model_path
        tts_binding_config["piper_voices_dir"] = args.piper_voices_dir
    elif args.tts_binding == "lollms":
        tts_binding_config["model_name"] = "default_lollms_voice" # Placeholder, server handles actual voice

    lollms_client = None
    try:
        ASCIIColors.magenta("Initializing LollmsClient...")
        lollms_client = LollmsClient(
            binding_name=args.llm_binding, model_name=args.llm_model,
            host_address=args.llm_host, models_path=args.models_path,
            llm_binding_config=llm_binding_config,
            tts_binding_name=args.tts_binding, tts_host_address=args.tts_host,
            tts_binding_config=tts_binding_config,
            verify_ssl_certificate=False
        )
        ASCIIColors.green("LollmsClient initialized.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient: {e}"); trace_exception(e)
        return

    generated_text = ""
    text_prompt = "Craft a very short, cheerful message about the joy of discovery."
    ASCIIColors.cyan(f"\n--- Generating Text (Prompt: '{text_prompt[:50]}...') ---")
    if not lollms_client.binding:
        ASCIIColors.error("LLM binding not available."); return
    try:
        print(f"{ASCIIColors.YELLOW}AI is thinking: {ASCIIColors.RESET}", end="")
        generated_text = lollms_client.generate_text(
            prompt=text_prompt, n_predict=100, stream=True,
            streaming_callback=text_stream_callback, temperature=0.7
        )
        print("\n"); ASCIIColors.green("Text generation complete.")
        ASCIIColors.magenta("Generated Text:\n"); ASCIIColors.yellow(generated_text)
    except Exception as e:
        ASCIIColors.error(f"Text generation failed: {e}"); trace_exception(e); return
    if not generated_text:
        ASCIIColors.warning("LLM did not generate any text."); return

    speech_file_path = None
    ASCIIColors.cyan(f"\n--- Synthesizing Speech (using {args.tts_binding}) ---")
    if not lollms_client.tts:
        ASCIIColors.error("TTS binding not available."); return
    try:
        tts_call_kwargs = {}
        if args.tts_binding == "bark":
            # For Bark, 'voice' in generate_audio is the voice_preset.
            # If not using the default from init, pass it here.
            # tts_call_kwargs['voice'] = args.bark_voice_preset 
            pass # Uses default_voice from init if args.bark_voice_preset not specified to override
        elif args.tts_binding == "xtts":
            tts_call_kwargs['language'] = args.xtts_language
            # 'voice' for XTTS is the speaker_wav path. If not using default from init, pass here.
            # tts_call_kwargs['voice'] = args.xtts_speaker_wav 
        elif args.tts_binding == "piper":
            # 'voice' for Piper is the .onnx filename.
            tts_call_kwargs['voice'] = args.piper_voice_file
            # Example Piper specific param:
            # tts_call_kwargs['length_scale'] = 1.0 

        audio_bytes = lollms_client.tts.generate_audio(text=generated_text, **tts_call_kwargs)

        if audio_bytes:
            filename_stem = f"speech_output_{args.llm_binding}_{args.tts_binding}"
            speech_file_path = SPEECH_OUTPUT_DIR / f"{filename_stem.replace('/', '_')}.wav"
            with open(speech_file_path, "wb") as f: f.write(audio_bytes)
            ASCIIColors.green(f"Speech synthesized and saved to: {speech_file_path}")
        elif args.tts_binding == "lollms":
            ASCIIColors.warning("LOLLMS TTS binding returned empty bytes. Server might have saved file if 'fn' was used.")
            speech_file_path = None
        else:
            ASCIIColors.warning("Speech synthesis returned empty bytes."); speech_file_path = None
    except Exception as e:
        ASCIIColors.error(f"Speech synthesis failed: {e}"); trace_exception(e); return

    if speech_file_path and PYGAME_AVAILABLE:
        ASCIIColors.magenta("\n--- Playing Synthesized Speech ---")
        try:
            pygame.mixer.init()
            speech_sound = pygame.mixer.Sound(str(speech_file_path))
            ASCIIColors.cyan("Playing audio... Press Ctrl+C in console to stop playback early.")
            speech_sound.play()
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(10)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.mixer.stop(); break
            ASCIIColors.green("Playback finished.")
        except pygame.error as e: ASCIIColors.warning(f"Could not play audio with pygame: {e}")
        except KeyboardInterrupt: pygame.mixer.stop(); ASCIIColors.yellow("\nPlayback interrupted.")
        finally: pygame.quit()
    elif not PYGAME_AVAILABLE:
        ASCIIColors.warning("Pygame is not available for playback.")
        if speech_file_path: ASCIIColors.info(f"Generated speech: {speech_file_path.resolve()}")
    elif not speech_file_path:
         ASCIIColors.warning("No speech file generated/path unknown. Skipping playback.")

    ASCIIColors.red("\n--- Demo Finished ---")

if __name__ == "__main__":
    main()