#!/usr/bin/env python3
"""
universal_profiles_example.py
Demonstrates the Two-Tier Lazy Profile System in LollmsClient.

Instead of declaring connection details repeatedly for each model, we decouple
the configuration into:
1. Connection Layer (`LollmsBindingProfile`): Defines the server/engine ONCE.
2. Execution Layer (`LollmsModelProfile`): Defines specific models that reference the binding.

Only the model marked as `is_default=True` is loaded at startup.
Other models are instantiated lazily *on-demand* when switched to, saving VRAM.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ascii_colors import ASCIIColors
from lollms_client import LollmsClient, LollmsBindingProfile, LollmsModelProfile

def main():
    ASCIIColors.panel(
        "[bold]Two-Tier Lazy Profiles Demo (One Binding, Multiple Models)[/bold]",
        title="[bold green]🧠 Multi-Model Architecture[/bold green]",
        border_style="green"
    )

    # 1. Define the Connection Layer ONCE
    # This holds the server address and binding library.
    local_ollama_binding = LollmsBindingProfile(
        name="local_ollama",
        binding_name="ollama",
        binding_config={"host_address": "http://localhost:11434"}
    )

    # 2. Define the Execution Layer (Models)
    # These reference the binding profile via `binding_profile_name`.
    fast_chat_model = LollmsModelProfile(
        name="fast_chat",
        binding_profile_name="local_ollama",
        model_name="llama3.2:3b",
        is_default=True # Loaded eagerly at startup
    )

    vision_model = LollmsModelProfile(
        name="vision_model",
        binding_profile_name="local_ollama",
        model_name="llava",
        vision_enabled=True
        # is_default=False (Implicit) -> Loaded lazily on switch
    )

    ASCIIColors.cyan("Initializing LollmsClient with Two-Tier Profiles...")

    # 3. Initialize the client
    client = LollmsClient(
        llm_binding_profiles={"local_ollama": local_ollama_binding},
        llm_model_profiles={
            "fast_chat": fast_chat_model,
            "vision_model": vision_model
        }
    )

    ASCIIColors.green("\n✅ Client initialized!")
    ASCIIColors.magenta(f"Active LLM Alias: {client._active_llm_alias}")
    ASCIIColors.magenta(f"Active LLM Model: {client.llm.model_name}")

    # Verify registries
    ASCIIColors.cyan(f"\nBinding Registry: {list(client.llm_binding_profiles_registry.keys())}")
    ASCIIColors.cyan(f"Model Registry:   {list(client.llm_model_profiles_registry.keys())}")

    # Show that the vision model is NOT loaded yet (Lazy Loading)
    ASCIIColors.yellow(f"\nInstantiated models cache: {list(client.llms.keys())}")
    ASCIIColors.info("Notice that 'vision_model' is not in the cache yet.")

    # 4. Generate with the default model
    ASCIIColors.cyan("\nGenerating text with default model...")
    response = client.generate_text("Write a Python one-liner to reverse a string.", temperature=0.1)
    ASCIIColors.green(f"Response from '{client._active_llm_alias}': {response.strip()[:100]}...")

    # 5. Switch to the vision model (Instantiated on-the-fly)
    ASCIIColors.cyan("\nSwitching to 'vision_model'...")
    success = client.switch_model("vision_model")

    if success:
        ASCIIColors.green(f"✅ Successfully switched!")
        ASCIIColors.magenta(f"New Active LLM Alias: {client._active_llm_alias}")
        ASCIIColors.magenta(f"New Active LLM Model: {client.llm.model_name}")

        # The cache now contains both models
        ASCIIColors.yellow(f"\nInstantiated models cache: {list(client.llms.keys())}")

        # Generate text using the newly activated profile
        response = client.generate_text("What do you see in this image?", temperature=0.2)
        ASCIIColors.green(f"\nResponse from '{client._active_llm_alias}': {response.strip()[:100]}...")

        # 6. Switch back to the fast chat model (Retrieved from cache, zero re-instantiation)
        ASCIIColors.cyan("\nSwitching back to 'fast_chat' profile...")
        client.switch_model("fast_chat")
        ASCIIColors.magenta(f"Active LLM Alias: {client._active_llm_alias}")
    else:
        ASCIIColors.red(f"❌ Failed to switch to profile 'vision_model'.")

if __name__ == "__main__":
    main()