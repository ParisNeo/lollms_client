from pathlib import Path
from typing import Dict, Any
import uuid
import io
# We expect LollmsClient instance to be passed if the tool needs it.
# from lollms_client import LollmsClient # Not imported directly, but type hint is useful

def execute(params: Dict[str, Any], lollms_client_instance: Any = None) -> Dict[str, Any]:
    """
    Generates an image using the LollmsClient's TTI binding.
    The lollms_client_instance is expected to be passed by the LocalMCPBinding if available.
    """
    prompt = params.get("prompt")
    negative_prompt = params.get("negative_prompt", "")
    width = params.get("width") # Will be None if not provided
    height = params.get("height") # Will be None if not provided
    output_filename_suggestion = params.get("output_filename_suggestion", "generated_image.png")

    if not prompt:
        return {"status": "error", "message": "'prompt' parameter is required."}

    if not lollms_client_instance or not hasattr(lollms_client_instance, 'tti') or not lollms_client_instance.tti:
        return {
            "status": "tti_not_available",
            "message": "LollmsClient instance with an active TTI binding was not provided or TTI is not configured.",
            "image_path": None,
            "image_url": None
        }

    try:
        tti_binding = lollms_client_instance.tti
        
        # Prepare arguments for the TTI binding's generate_image method
        tti_kwargs = {}
        if width is not None:
            tti_kwargs['width'] = width
        if height is not None:
            tti_kwargs['height'] = height
        # Add other common TTI params if desired, e.g. seed, steps, cfg_scale
        # These would need to be added to the MCP JSON input_schema as well.

        image_bytes = tti_binding.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **tti_kwargs
        )

        if not image_bytes:
            return {
                "status": "error",
                "message": "TTI binding returned no image data.",
                "image_path": None,
                "image_url": None
            }

        # Define where to save the image.
        # This should ideally be a secure, configurable workspace.
        # For this example, we save it into a 'mcp_generated_images' subdirectory
        # of the current working directory OR a path derived from lollms_paths if available.
        
        # Prefer using a path from lollms_client_instance if available (e.g., an output or data path)
        # This part needs careful consideration for a real application.
        save_dir_base = Path.cwd() 
        if hasattr(lollms_client_instance, 'lollms_paths_config') and lollms_client_instance.lollms_paths_config.get('personal_outputs_path'):
            save_dir_base = Path(lollms_client_instance.lollms_paths_config['personal_outputs_path'])
        elif hasattr(lollms_client_instance, 'lollms_paths_config') and lollms_client_instance.lollms_paths_config.get('shared_outputs_path'): # type: ignore
            save_dir_base = Path(lollms_client_instance.lollms_paths_config['shared_outputs_path']) # type: ignore

        save_dir = save_dir_base / "mcp_generated_images"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename and make it unique
        base, ext = Path(output_filename_suggestion).stem, Path(output_filename_suggestion).suffix
        if not ext: ext = ".png" # Default to png
        safe_base = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in base)
        unique_filename = f"{safe_base}_{uuid.uuid4().hex[:8]}{ext}"
        
        image_save_path = save_dir / unique_filename

        with open(image_save_path, "wb") as f:
            f.write(image_bytes)
        
        # TODO: How to best represent the URL? For local files, it might be a file:// URL
        # or a relative path that the client application understands.
        # For now, returning a relative path from a conceptual 'outputs' root.
        # A more robust solution would involve the LollmsClient serving these images
        # or providing data URLs.
        
        # Create a relative path for client display if possible
        # This assumes the client knows how to interpret "mcp_generated_images/..."
        relative_image_path = f"mcp_generated_images/{unique_filename}"
        image_url = f"file:///{image_save_path.resolve()}" # Example data URL or file path

        return {
            "status": "success",
            "message": f"Image generated and saved successfully.",
            "image_path": relative_image_path, # More of a hint
            "image_url": image_url # More concrete path
        }

    except Exception as e:
        # from ascii_colors import trace_exception # If you need full trace
        # trace_exception(e)
        return {
            "status": "error",
            "message": f"Failed to generate image via TTI binding: {str(e)}",
            "image_path": None,
            "image_url": None
        }

if __name__ == '__main__':
    import json
    from PIL import Image as PILImage # To avoid conflict with module-level Image if any
    print("--- Generate Image (via LollmsClient TTI) Tool Test ---")

    # This test requires a LollmsClient instance with a configured TTI binding.
    # We'll mock it for a standalone test of the execute function's logic.

    class MockTTIBinding:
        def __init__(self, works=True):
            self.works = works
            self.config = {"default_width": 512, "default_height": 512} # Mock config

        def generate_image(self, prompt, negative_prompt="", width=None, height=None, **kwargs):
            if not self.works:
                # return None # Simulate TTI returning no data
                raise ValueError("Simulated TTI error in generate_image")
            
            print(f"MockTTI: Generating image for prompt: '{prompt}', neg: '{negative_prompt}', W:{width}, H:{height}")
            # Create a dummy PIL image and return its bytes
            img = PILImage.new('RGB', (width or 512, height or 512), color = 'skyblue' if "sky" in prompt else "lightcoral")
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((10,10), prompt[:30], fill=(0,0,0))
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')
            return byte_arr.getvalue()

    class MockLollmsClient:
        def __init__(self, tti_works=True):
            self.tti = MockTTIBinding(works=tti_works)
            self.lollms_paths_config = {"personal_outputs_path": Path("./test_lollms_client_outputs")} # Mock path
            Path(self.lollms_paths_config["personal_outputs_path"]).mkdir(exist_ok=True)


    mock_lc_success = MockLollmsClient(tti_works=True)
    mock_lc_tti_fail = MockLollmsClient(tti_works=False)
    mock_lc_no_tti = MockLollmsClient()
    mock_lc_no_tti.tti = None # Simulate TTI not configured

    # Test 1: Successful generation
    params1 = {"prompt": "A beautiful sunset over mountains", "width": 256, "height": 256}
    result1 = execute(params1, lollms_client_instance=mock_lc_success)
    print(f"\nTest 1 Result (Success):\n{json.dumps(result1, indent=2)}")
    assert result1["status"] == "success"
    assert result1["image_path"] is not None
    if result1["image_path"]:
        # Verify file was created (adjust path based on where it's saved by tool)
        # For this test, it's saved in ./test_lollms_client_outputs/mcp_generated_images/
        full_saved_path = Path(mock_lc_success.lollms_paths_config["personal_outputs_path"]) / result1["image_path"]
        print(f"Checking for image at: {full_saved_path}")
        assert full_saved_path.exists()
        # Optional: Clean up created image after test
        # full_saved_path.unlink(missing_ok=True)


    # Test 2: TTI binding itself fails to generate
    params2 = {"prompt": "A futuristic city"}
    result2 = execute(params2, lollms_client_instance=mock_lc_tti_fail)
    print(f"\nTest 2 Result (TTI Fails):\n{json.dumps(result2, indent=2)}")
    assert result2["status"] == "error"
    assert "Simulated TTI error" in result2["message"]

    # Test 3: No TTI binding available in LollmsClient
    params3 = {"prompt": "A cat wearing a hat"}
    result3 = execute(params3, lollms_client_instance=mock_lc_no_tti)
    print(f"\nTest 3 Result (No TTI Binding):\n{json.dumps(result3, indent=2)}")
    assert result3["status"] == "tti_not_available"

    # Test 4: Missing prompt
    params4 = {}
    result4 = execute(params4, lollms_client_instance=mock_lc_success)
    print(f"\nTest 4 Result (Missing Prompt):\n{json.dumps(result4, indent=2)}")
    assert result4["status"] == "error"
    assert "'prompt' parameter is required" in result4["message"]
    
    # Test 5: No LollmsClient instance passed (should also result in tti_not_available)
    params5 = {"prompt": "A dog"}
    result5 = execute(params5, lollms_client_instance=None)
    print(f"\nTest 5 Result (No LollmsClient passed):\n{json.dumps(result5, indent=2)}")
    assert result5["status"] == "tti_not_available"


    print("\n--- Tests Finished ---")
    print(f"Generated test images (if any) are in subdirectories of: {mock_lc_success.lollms_paths_config['personal_outputs_path']}")