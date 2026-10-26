"""
Simple Image Generation Agent Example
Explicitly opts into TTI bindings while keeping LLM creation enabled.
"""
from lollms_client.lollms_config_cli_env import get_client_from_env


def main() -> None:
    # get_client_from_env now requires explicit opt-in for non-LLM modalities.
    # We pass create_tti=True to ensure the TTI profile is hydrated.
    client = get_client_from_env(create_tti=True)

    if not client.tti:
        print("No TTI binding available. Check your configuration.")
        return

    # Generate an image using the TTI binding
    prompt = "A futuristic cityscape at sunset, digital art"
    try:
        image = client.tti.generate_image(prompt)
        print(f"Image generated successfully: {image}")
    except Exception as exc:
        print(f"Image generation failed: {exc}")


if __name__ == "__main__":
    main()