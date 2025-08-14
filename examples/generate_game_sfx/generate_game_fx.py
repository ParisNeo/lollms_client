# lollms_client/examples/game_sfx_generation/generate_game_sfx.py
from pathlib import Path
import time
import argparse # For command-line arguments

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


from lollms_client import LollmsClient # Removed LollmsDiscussion, LollmsMessage as not used
from ascii_colors import ASCIIColors, trace_exception

# --- Configuration ---
# Output directory for generated sound effects
SFX_OUTPUT_DIR = Path(__file__).parent / "sfx_output"
SFX_OUTPUT_DIR.mkdir(exist_ok=True)

# Sound effect descriptions. Note: 'duration' is more relevant for audiocraft.
# Bark's duration is more implicit based on prompt content.
SOUND_EFFECTS_TO_GENERATE = [
    {
        "filename": "sfx_crunch",
        "prompt": "Sound effect of a single, sharp, dry crunch, like stepping on a crisp autumn leaf or a piece of dry wood breaking.",
        "duration": 1, # audiocraft specific
        "bark_params": {"fine_temperature": 0.4, "coarse_temperature": 0.6} # Example bark specific
    },
    {
        "filename": "sfx_death_electronic",
        "prompt": "Short, impactful electronic death sound effect for a video game character, like a quick digital zap or a brief power-down sound.",
        "duration": 1.5,
        "bark_params": {"voice_preset": None} # Try without preset for more raw SFX
    },
    {
        "filename": "sfx_powerup_positive",
        "prompt": "Bright, positive, short power-up collection sound effect, like a magical chime, a sparkling shimmer, or an uplifting notification. [SFX]",
        "duration": 1.5
    },
    {
        "filename": "sfx_laser_shot",
        "prompt": "Sound effect of a futuristic laser gun firing a single shot, a quick 'pew' sound. [SFX: laser pew]",
        "duration": 0.5
    },
    {
        "filename": "sfx_coin_collect",
        "prompt": "Classic video game coin collection sound effect, a short, metallic, cheerful 'ding' or 'jingle'. [SFX: coin]",
        "duration": 0.7
    }
]

def generate_sfx(lollms_client: LollmsClient, sfx_info: dict) -> Path | None:
    """Generates a single sound effect using the LollmsClient's TTM binding."""
    filename_stem = sfx_info["filename"]
    prompt = sfx_info["prompt"]

    # Default output format
    output_format = "wav" # WAV is generally best for SFX in pygame
    output_path = SFX_OUTPUT_DIR / f"{filename_stem}_{lollms_client.ttm.binding_name}.{output_format}" # Add binding name to filename

    ASCIIColors.cyan(f"\nGenerating SFX using '{lollms_client.ttm.binding_name}': '{filename_stem}'")
    ASCIIColors.info(f"Prompt: '{prompt[:60]}...'")


    if not lollms_client.ttm:
        ASCIIColors.error("TTM (Text-to-Music/Sound) binding is not available in LollmsClient.")
        return None

    ttm_params = {"progress": True} # Common param for both

    if lollms_client.ttm.binding_name == "audiocraft":
        ttm_params["duration"] = sfx_info.get("duration", 1.0)
        ttm_params["temperature"] = sfx_info.get("audiocraft_temperature", 1.0)
        ttm_params["cfg_coef"] = sfx_info.get("audiocraft_cfg_coef", 3.0)
        ASCIIColors.info(f"AudioCraft Params: duration={ttm_params['duration']}, temp={ttm_params['temperature']}, cfg={ttm_params['cfg_coef']}")
    elif lollms_client.ttm.binding_name == "bark":
        # Bark duration is implicit. Parameters are different.
        bark_specific_params = sfx_info.get("bark_params", {})
        ttm_params["voice_preset"] = bark_specific_params.get("voice_preset", None) # None might be good for SFX
        ttm_params["fine_temperature"] = bark_specific_params.get("fine_temperature", 0.5)
        ttm_params["coarse_temperature"] = bark_specific_params.get("coarse_temperature", 0.7)
        ASCIIColors.info(f"Bark Params: preset={ttm_params['voice_preset']}, fine_temp={ttm_params['fine_temperature']}, coarse_temp={ttm_params['coarse_temperature']}")
    else:
        ASCIIColors.warning(f"Unknown TTM binding '{lollms_client.ttm.binding_name}'. Using generic parameters.")


    try:
        music_bytes = lollms_client.ttm.generate_music(prompt=prompt, **ttm_params)

        if music_bytes:
            with open(output_path, "wb") as f:
                f.write(music_bytes)
            ASCIIColors.green(f"SFX '{filename_stem}' ({lollms_client.ttm.binding_name}) saved to: {output_path}")
            return output_path
        else:
            ASCIIColors.warning(f"SFX generation for '{filename_stem}' ({lollms_client.ttm.binding_name}) returned empty bytes.")
            return None
    except Exception as e:
        ASCIIColors.error(f"Error generating SFX '{filename_stem}' ({lollms_client.ttm.binding_name}): {e}")
        trace_exception(e)
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate game sound effects using LOLLMS TTM bindings.")
    parser.add_argument(
        "--ttm_binding",
        type=str,
        choices=["audiocraft", "bark"],
        default="bark", # Default to audiocraft
        help="The TTM binding to use for generation."
    )
    parser.add_argument(
        "--audiocraft_model",
        type=str,
        default="facebook/musicgen-small",
        help="Hugging Face model ID for AudioCraft (e.g., facebook/musicgen-small, facebook/musicgen-melody)."
    )
    parser.add_argument(
        "--bark_model",
        type=str,
        default="suno/bark-small",
        help="Hugging Face model ID for Bark (e.g., suno/bark-small, suno/bark)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Auto-detect
        choices=["cpu", "cuda", "mps", None],
        help="Device to run the TTM model on (cpu, cuda, mps, or auto-detect)."
    )
    args = parser.parse_args()

    ASCIIColors.red(f"--- LOLLMS Game SFX Generation Example (Using: {args.ttm_binding}) ---")

    ttm_binding_config = {"device": args.device} # Common device config
    if args.ttm_binding == "audiocraft":
        ttm_binding_config["model_name"] = args.audiocraft_model
        ttm_binding_config["output_format"] = "wav" # Audiocraft binding defaults to wav for bytes
    elif args.ttm_binding == "bark":
        ttm_binding_config["model_name"] = args.bark_model
        # Bark binding currently outputs WAV by default for bytes
    else:
        ASCIIColors.error(f"Unsupported TTM binding: {args.ttm_binding}")
        return

    try:
        ASCIIColors.magenta(f"Initializing LollmsClient with {args.ttm_binding} for TTM...")
        lollms_client = LollmsClient(
            llm_binding_name="lollms", # Can be a dummy if only using TTM
            ttm_binding_name=args.ttm_binding,
            ttm_binding_config=ttm_binding_config
        )
        ASCIIColors.green("LollmsClient initialized.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient: {e}")
        trace_exception(e)
        return

    if not lollms_client.ttm:
        ASCIIColors.error(f"{args.ttm_binding.capitalize()} TTM binding could not be loaded. Exiting.")
        return

    generated_sfx_paths = {}
    for sfx_info_item in SOUND_EFFECTS_TO_GENERATE:
        sfx_path = generate_sfx(lollms_client, sfx_info_item)
        if sfx_path:
            generated_sfx_paths[sfx_info_item["filename"]] = {
                "path": sfx_path,
                "binding": args.ttm_binding # Store which binding generated it
            }
        time.sleep(0.5) # Small delay

    ASCIIColors.red("\n--- SFX Generation Complete ---")
    if not generated_sfx_paths:
        ASCIIColors.warning("No sound effects were successfully generated.")
        return

    if not PYGAME_AVAILABLE:
        ASCIIColors.warning("Pygame is not available. Skipping sound playback demo.")
        ASCIIColors.info(f"Generated SFX can be found in: {SFX_OUTPUT_DIR.resolve()}")
        return

    ASCIIColors.magenta("\n--- Pygame SFX Playback Demo ---")
    pygame.mixer.init()
    game_sounds = {}
    sfx_playback_order = [] # To map number keys to sounds

    for filename_stem, sfx_data in generated_sfx_paths.items():
        path = sfx_data["path"]
        binding_used = sfx_data["binding"]
        playback_name = f"{filename_stem} ({binding_used})"
        try:
            sound = pygame.mixer.Sound(str(path))
            game_sounds[playback_name] = sound
            sfx_playback_order.append(playback_name)
            ASCIIColors.green(f"Loaded '{path.name}' into pygame as '{playback_name}'.")
        except pygame.error as e:
            ASCIIColors.warning(f"Could not load sound '{path.name}' into pygame: {e}")

    if not game_sounds:
        ASCIIColors.warning("No sounds loaded into pygame. Exiting demo.")
        return

    print("\nInstructions:")
    for i, sfx_name_to_play in enumerate(sfx_playback_order):
        print(f"  Press key '{i+1}' to play: {sfx_name_to_play}")
    print("  Press 'Q' to quit the demo.")

    pygame.display.set_mode((400, 200))
    pygame.display.set_caption(f"SFX Player ({args.ttm_binding.capitalize()})")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                for i in range(len(sfx_playback_order)):
                    if event.key == getattr(pygame, f"K_{i+1}", None): # Check if K_i+1 exists
                        sfx_name_to_play = sfx_playback_order[i]
                        if sfx_name_to_play in game_sounds:
                            ASCIIColors.cyan(f"Playing: {sfx_name_to_play}")
                            game_sounds[sfx_name_to_play].play()
                        break
        pygame.time.Clock().tick(30)

    pygame.quit()
    ASCIIColors.red("--- Demo Finished ---")
    ASCIIColors.info(f"Generated SFX are in: {SFX_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()