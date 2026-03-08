from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE # If using callbacks
from ascii_colors import ASCIIColors, trace_exception
from PIL import Image
from pathlib import Path
import io
import os

# --- Configuration ---
# This client_id should match one known by your LoLLMs WebUI if security is enabled for these endpoints.
# For a default local setup, it might not be strictly checked for /generate_image,
# but IS required for /list_tti_services, /get_active_tti_settings, /set_active_tti_settings.
LOLLMS_CLIENT_ID = "my_lollms_client_id" # Replace with your actual client ID or a test one

# Initialize LollmsClient, enabling the TTI 'lollms' binding
# The service_key here is used as client_id by the TTI binding for lollms
# lc = LollmsClient(
#     tti_binding_name="lollms"
# )

# make sure you have a OPENAI_API_KEY environment variable
lc = LollmsClient(
    tti_binding_name="dalle",
    tti_binding_config={
        "model_name": "dall-e-3",  # Example model, adjust as needed
    }
)

if not lc.tti:
    ASCIIColors.error("TTI binding could not be initialized. Please check LollmsClient setup.")
    exit()

def test_list_tti_services():
    ASCIIColors.cyan("\n--- Testing List TTI Services ---")
    try:
        # client_id is taken from lc.service_key by the binding
        services = lc.tti.list_services()
        if services:
            ASCIIColors.green("Available TTI Services:")
            for i, service in enumerate(services):
                print(f"  {i+1}. Name: {service.get('name')}, Caption: {service.get('caption')}")
        else:
            ASCIIColors.yellow("No TTI services listed or an empty list was returned.")
    except Exception as e:
        ASCIIColors.error(f"Error listing TTI services: {e}")
        trace_exception(e)

def test_get_tti_settings():
    ASCIIColors.cyan("\n--- Testing Get Active TTI Settings ---")
    try:
        # client_id is taken from lc.service_key by the binding
        settings = lc.tti.get_settings()
        if settings: # Server returns a list for settings template
            ASCIIColors.green("Current Active TTI Settings/Template:")
            # Assuming settings is a list of dicts (template format)
            for setting_item in settings:
                print(f"  - Name: {setting_item.get('name')}, Type: {setting_item.get('type')}, Value: {setting_item.get('value')}, Help: {setting_item.get('help')}")
        elif isinstance(settings, dict) and not settings: # Empty dict if no TTI active
             ASCIIColors.yellow("No active TTI service or settings configured on the server.")
        else:
            ASCIIColors.yellow("Could not retrieve TTI settings or format unexpected.")
            print(f"Received: {settings}")
    except Exception as e:
        ASCIIColors.error(f"Error getting TTI settings: {e}")
        trace_exception(e)

def test_set_tti_settings():
    ASCIIColors.cyan("\n--- Testing Set Active TTI Settings (Illustrative) ---")
    ASCIIColors.yellow("Note: This test requires knowing the exact settings structure of your active TTI service.")
    ASCIIColors.yellow("Skipping actual setting change to avoid misconfiguration.")
    # Example: If you knew your TTI service had a 'quality' setting:
    # example_settings_to_set = [
    #     {"name": "quality", "value": "high", "type": "str", "help": "Image quality"},
    #     # ... other settings from get_settings()
    # ]
    # try:
    #     # client_id is taken from lc.service_key
    #     success = lc.tti.set_settings(example_settings_to_set)
    #     if success:
    #         ASCIIColors.green("Successfully sent settings update request.")
    #     else:
    #         ASCIIColors.red("Failed to set TTI settings (server indicated failure or no change).")
    # except Exception as e:
    #     ASCIIColors.error(f"Error setting TTI settings: {e}")

def test_generate_image():
    ASCIIColors.cyan("\n--- Testing Generate Image ---")
    prompt = "A futuristic cityscape at sunset, neon lights, flying vehicles"
    negative_prompt = "blurry, low quality, ugly, text, watermark"
    width = 1024
    height = 1024
    home_dir = Path.home()
    documents_dir = home_dir / "Documents"
    output_filename = documents_dir/"generated_lollms_image.jpg"

    ASCIIColors.info(f"Prompt: {prompt}")
    ASCIIColors.info(f"Negative Prompt: {negative_prompt}")
    ASCIIColors.info(f"Dimensions: {width}x{height}")

    try:
        image_bytes = lc.tti.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height
            # You can add other kwargs here if your TTI service supports them, e.g., seed=12345
        )

        if image_bytes:
            ASCIIColors.green(f"Image generated successfully ({len(image_bytes)} bytes).")
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.save(output_filename)
                ASCIIColors.green(f"Image saved as {output_filename}")
                # Attempt to show image if possible (platform dependent)
                if os.name == 'nt': # Windows
                    os.startfile(output_filename)
                elif os.name == 'posix': # MacOS/Linux
                    try:
                        import subprocess
                        opener = "open" if platform.system() == "Darwin" else "xdg-open"
                        subprocess.call([opener, output_filename])
                    except:
                        ASCIIColors.yellow(f"Could not auto-open image. Please find it at {output_filename}")

            except Exception as e:
                ASCIIColors.error(f"Error processing or saving image: {e}")
                # Save raw bytes if PIL fails, for debugging
                with open("generated_lollms_image_raw.data", "wb") as f_raw:
                    f_raw.write(image_bytes)
                ASCIIColors.yellow("Raw image data saved as generated_lollms_image_raw.data for inspection.")

        else:
            ASCIIColors.red("Image generation returned empty bytes.")

    except Exception as e:
        ASCIIColors.error(f"Error during image generation: {e}")
        trace_exception(e)

if __name__ == "__main__":
    # Test management functions first
    test_list_tti_services()
    test_get_tti_settings()
    test_set_tti_settings() # Currently illustrative

    # Then test image generation
    test_generate_image()