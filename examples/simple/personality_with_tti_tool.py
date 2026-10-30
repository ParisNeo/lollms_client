"""
Personality with injected TTI capabilities example.

Demonstrates how to attach a tool to an AIPersonality that uses the
system-injected LollmsClient instance (via lollms_client_instance parameter)
to access the active TTI binding, never constructing its own client.

Usage:
    python examples/simple/personality_with_tti_tool.py
"""
import argparse
from pathlib import Path
from lollms_client import AIPersonality, LollmsClient
from ascii_colors import ASCIIColors

# ---------------------------------------------------------------------------
# Tool definition (uses host-injected client, never creates its own)
# ---------------------------------------------------------------------------

def tool_generate_visual(lollms_client_instance=None, prompt: str = "",
                         negative_prompt: str = "blurry, lowres, deformed",
                         style: str = "photorealistic"):
    """Generate an image using the host's active TTI binding.

    Args:
        prompt (str): Detailed visual description.
        negative_prompt (str): Elements to avoid.
        style (str): Artistic style (photorealistic, digital_art, sketch, cyberpunk).
    """
    # Access the live client injected by the dispatcher (never construct one)
    lc = lollms_client_instance
    if lc is None or getattr(lc, "tti_binding", None) is None:
        return {
            "success": False,
            "error": "TTI service not available on active client"
        }

    try:
        # Use the active TTI binding (respects user settings)
        result = lc.tti_binding.generate_image(
            prompt=f"{style}, {prompt}",
            negative_prompt=negative_prompt,
            width=1024,
            height=1024
        )

        # Handle result (str = path, bytes = raw data)
        if isinstance(result, (str, Path)):
            return {
                "success": True,
                "image_path": str(result),
                "output": f"Generated image saved to: {result}"
            }
        elif isinstance(result, (bytes, bytearray)):
            # Save to workspace (CWD is set to workspace by dispatcher)
            out_path = f"generated_{hash(prompt) % 999999}.png"
            with open(out_path, "wb") as f:
                f.write(result)
            return {
                "success": True,
                "image_path": out_path,
                "output": f"Generated image saved to: {out_path}"
            }
        else:
            return {"success": False, "error": "TTI returned no image data"}

    except Exception as ex:
        return {"success": False, "error": f"Generation failed: {ex}"}


# ---------------------------------------------------------------------------
# Personality setup
# ---------------------------------------------------------------------------

def build_visual_artist_personality() -> AIPersonality:
    """Construct a personality with TTI tool capabilities attached."""
    persona = AIPersonality(
        name="VisualArtist",
        system_prompt=(
            "You are an artistic assistant. When the user describes a scene, "
            "use the generate_visual_asset tool to create the image. "
            "Consider composition, lighting, and style in your prompts."
        )
    )

    # Attach the tool function directly (no static binding imports)
    persona.attach_tool_functions([tool_generate_visual])

    return persona


# ---------------------------------------------------------------------------
# Bootstrap / Usage
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Personality with TTI Tool")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--tti-model", default="stable-diffusion",
                       help="TTI model profile to activate")
    args = parser.parse_args()

    # Build the client with explicit LLM and TTI profiles
    client = LollmsClient(
        binding_profiles=[
            {
                "profile_name": "ollama",
                "binding_name": "ollama",
                "config": {"host_address": args.host}
            },
            {
                "profile_name": "diffusers",
                "binding_name": "diffusers",
                "config": {"host_address": "http://localhost:7860"}
            }
        ],
        model_profiles=[
            {
                "profile_name": "chat_model",
                "binding_profile": "ollama",
                "model_name": args.model
            },
            {
                "profile_name": "tti_model",
                "binding_profile": "diffusers",
                "model_name": args.tti_model
            }
        ],
        default_binding_profile="ollama",
        default_model_profile="chat_model"
    )

    # Ensure TTI binding is active (loaded profile)
    client.switch_model("tti_model")
    if lc.tti_binding is None:
        raise RuntimeError("TTI binding not active; check diffusers profile")

    # Build personality and attach to discussion
    persona = build_visual_artist_personality()
    discussion = AIPersonality.create_discussion(
        persona,
        client,
        workspace_path=Path("./workspace_painter")
    )

    print(f"{ASCIIColors.yellow}Personality '{persona.name}' ready.")
    print(f"{ASCIIColors.cyan}Tool 'generate_visual_asset' attached.")
    print(f"{ASCIIColors.color_reset}Enter a visual description (or 'quit'):")

    # Interactive loop
    while True:
        try:
            user_input = input(f"\n{ASCIIColors.cyan}You > {ASCIIColors.color_reset}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ('quit', 'exit'):
            break
        if not user_input:
            continue

        try:
            # Chat injects lollms_client_instance into tool automatically
            response = discussion.chat(user_input)
            print(f"{ASCIIColors.green}AI: {response}{ASCIIColors.color_reset}")
        except Exception as ex:
            print(f"{ASCIIColors.red}Error: {ex}{ASCIIColors.color_reset}")


if __name__ == "__main__":
    main()