# lollms_client/examples/test_tti_bindings.py
from lollms_client import LollmsClient
from lollms_client.lollms_types import MSG_TYPE # If using callbacks
from ascii_colors import ASCIIColors, trace_exception
from PIL import Image
from pathlib import Path
import io
import os
import platform # For opening image
import subprocess # For opening image
import shutil # For cleanup of diffusers env
import json # For pretty printing dicts
from typing import Optional
try:
    from huggingface_hub import snapshot_download
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False
    snapshot_download = None
    ASCIIColors.warning("huggingface_hub library not found. Diffusers model download test will be skipped.")
    ASCIIColors.warning("Please install with: pip install huggingface-hub")

LOLLMS_CLIENT_ID = "my_lollms_test_client_id" 

# --- Diffusers Test Specific Configuration ---
# Using a standard Stable Diffusion model
DIFFUSERS_MODEL_ID = "runwayml/stable-diffusion-v1-5" # Standard SD 1.5 model
DIFFUSERS_LOCAL_MODEL_NAME = "sd-v1-5-test-model" # Folder name for the downloaded model
TEMP_DIFFUSERS_ENV_DIR_NAME = "temp_diffusers_lollms_env_for_test"
BASE_TEST_PATH = Path(__file__).parent 

def setup_diffusers_environment(base_path: Path):
    ASCIIColors.cyan(f"\n--- Setting up Diffusers Test Environment for {DIFFUSERS_MODEL_ID} at {base_path} ---")
    if not HUGGINGFACE_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is not available. Cannot set up Diffusers environment.")

    temp_env_root = base_path / TEMP_DIFFUSERS_ENV_DIR_NAME
    if temp_env_root.exists(): 
        shutil.rmtree(temp_env_root)
    temp_env_root.mkdir(parents=True, exist_ok=True)

    lollms_paths = {
        "personal_models_path": temp_env_root / "personal_models",
        "models_zoo_path": temp_env_root / "models_zoo", 
        "shared_cache_path": temp_env_root / "shared_cache", 
        "tti_bindings_path": base_path.parent / "tti_bindings" 
    }
    for p_key, p_val in lollms_paths.items():
        if p_key != "tti_bindings_path": 
            Path(p_val).mkdir(parents=True, exist_ok=True)
    
    diffusers_models_dir = lollms_paths["personal_models_path"] / "diffusers_models"
    diffusers_models_dir.mkdir(parents=True, exist_ok=True)
    
    model_target_dir = diffusers_models_dir / DIFFUSERS_LOCAL_MODEL_NAME
    
    ASCIIColors.info(f"Attempting to download {DIFFUSERS_MODEL_ID} to {model_target_dir}...")
    try:
        # SD 1.5 often has fp16 revision, which is smaller and faster if GPU is used
        # For CPU test, main revision is fine. Safetensors are preferred.
        snapshot_download(
            repo_id=DIFFUSERS_MODEL_ID,
            local_dir=str(model_target_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(lollms_paths["shared_cache_path"] / "huggingface_hub_cache"),
            # revision="fp16", # Optional: if you want the fp16 variant specifically
            allow_patterns=["*.json", "*.txt", "*.safetensors"], # Prefer safetensors
            # ignore_patterns=["*.bin", "*.ckpt"], # Ignore older formats if safetensors exist
        )
        ASCIIColors.green(f"Model {DIFFUSERS_MODEL_ID} downloaded successfully.")
        
    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Failed to download model {DIFFUSERS_MODEL_ID} or assertion failed: {e}")
        if model_target_dir.exists():
            ASCIIColors.info(f"Contents of {model_target_dir} ({model_target_dir.resolve()}):")
            for item_path in model_target_dir.rglob('*'):
                if item_path.is_file(): ASCIIColors.info(f"  - {item_path.relative_to(model_target_dir)}")
                elif item_path.is_dir(): ASCIIColors.info(f"  DIR: {item_path.relative_to(model_target_dir)}")
        raise

    binding_instance_config = {
        "model_id_or_path": DIFFUSERS_LOCAL_MODEL_NAME, 
        "pipeline_class_name": None, # Let AutoPipelineForText2Image handle SD 1.5
        "device": "cpu", # Keep CPU for stable CI/testing
        "torch_dtype_str": "float32", # float32 for CPU
        "num_inference_steps": 10, # Fewer steps for faster test
        "default_width": 512, # Standard for SD 1.5
        "default_height": 512,
        "safety_checker_on": False, # Commonly disabled for local testing
        "lollms_paths": lollms_paths,
        "hf_variant": None # If using main revision, no variant needed. Use "fp16" if you downloaded that revision.
    }
    return binding_instance_config, lollms_paths

def cleanup_diffusers_environment(base_path: Path):
    ASCIIColors.cyan("\n--- Cleaning up Diffusers Test Environment ---")
    temp_env_root = base_path / TEMP_DIFFUSERS_ENV_DIR_NAME
    if temp_env_root.exists():
        try:
            shutil.rmtree(temp_env_root)
            ASCIIColors.info(f"Cleaned up Diffusers temp environment: {temp_env_root}")
        except Exception as e:
            ASCIIColors.warning(f"Could not fully clean up {temp_env_root}: {e}")
            trace_exception(e)

def test_list_tti_services(lc: LollmsClient):
    ASCIIColors.cyan("\n--- Testing List TTI Services ---")
    try:
        services = lc.tti.list_services()
        if services:
            ASCIIColors.green(f"Available TTI Services for binding '{lc.tti.binding_name}':")
            for i, service in enumerate(services):
                print(f"  {i+1}. Name: {service.get('name')}, Caption: {service.get('caption')}, Help: {service.get('help')}")
            assert len(services) > 0, "Expected at least one service to be listed."
        else:
            ASCIIColors.yellow("No TTI services listed or an empty list was returned.")
    except Exception as e:
        ASCIIColors.error(f"Error listing TTI services: {e}")
        trace_exception(e)
        raise

def test_get_tti_settings(lc: LollmsClient):
    ASCIIColors.cyan("\n--- Testing Get Active TTI Settings ---")
    try:
        settings = lc.tti.get_settings() # This should be a list of dicts
        if settings: 
            ASCIIColors.green(f"Current Active TTI Settings/Template for binding '{lc.tti.binding_name}':")
            for setting_item in settings:
                # Ensure setting_item is a dictionary before trying to access .get()
                if isinstance(setting_item, dict):
                    print(f"  - Name: {setting_item.get('name')}, Type: {setting_item.get('type')}, Value: {setting_item.get('value')}") 
                else:
                    ASCIIColors.warning(f"Found non-dict item in settings list: {setting_item}")
            assert isinstance(settings, list) and len(settings) > 0, "Expected settings to be a non-empty list of dicts."
        elif isinstance(settings, dict) and not settings: 
             ASCIIColors.yellow("No active TTI service or settings configured on the server (empty dict).")
        else:
            ASCIIColors.yellow("Could not retrieve TTI settings or format unexpected.")
            print(f"Received: {settings}")
    except Exception as e:
        ASCIIColors.error(f"Error getting TTI settings: {e}")
        trace_exception(e)
        raise

def test_set_tti_settings(lc: LollmsClient):
    ASCIIColors.cyan("\n--- Testing Set Active TTI Settings ---")
    if lc.tti.binding_name == "diffusers":
        ASCIIColors.info("Attempting to change 'num_inference_steps' for Diffusers.")
        try:
            original_settings = lc.tti.get_settings()
            original_steps = None
            for s_item in original_settings:
                if isinstance(s_item, dict) and s_item.get('name') == 'num_inference_steps':
                    original_steps = s_item['value']
                    break
            assert original_steps is not None, "Could not find 'num_inference_steps' in Diffusers settings."
            
            new_steps = int(original_steps) + 1
            settings_to_set = [{"name": "num_inference_steps", "value": new_steps}] 
            
            success = lc.tti.set_settings(settings_to_set) 
            if success:
                ASCIIColors.green(f"Successfully sent request to set 'num_inference_steps' to {new_steps}.")
                current_settings_after_set = lc.tti.get_settings()
                current_config_steps = None
                for s_item_after in current_settings_after_set:
                     if isinstance(s_item_after, dict) and s_item_after.get('name') == 'num_inference_steps':
                        current_config_steps = s_item_after['value']
                        break
                assert current_config_steps == new_steps, f"Verification failed: settings show {current_config_steps}, expected {new_steps}"
                ASCIIColors.green("Setting change verified in binding's settings.")
            else:
                ASCIIColors.red("Failed to set TTI settings (binding indicated failure or no change).")
                assert False, "set_settings returned False"
        except Exception as e:
            ASCIIColors.error(f"Error setting Diffusers TTI settings: {e}")
            trace_exception(e)
            raise
    else: 
        ASCIIColors.yellow(f"Skipping actual setting change for '{lc.tti.binding_name}' in this detailed test.")

def test_generate_image(lc: LollmsClient, output_filename: Path, prompt: str, negative_prompt: Optional[str], width: int, height: int):
    ASCIIColors.cyan(f"\n--- Testing Generate Image ({lc.tti.binding_name}) ---")
    ASCIIColors.info(f"Output to: {output_filename}")
    ASCIIColors.info(f"Prompt: {prompt}")
    if negative_prompt: ASCIIColors.info(f"Negative Prompt: {negative_prompt}")
    ASCIIColors.info(f"Dimensions: {width}x{height}")
    
    try:
        image_bytes = lc.tti.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height
        )

        if image_bytes:
            ASCIIColors.green(f"Image generated successfully ({len(image_bytes)} bytes).")
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.save(output_filename)
                ASCIIColors.green(f"Image saved as {output_filename}")
                if os.name == 'nt': 
                    os.startfile(str(output_filename))
                elif os.name == 'posix': 
                    try:
                        opener = "open" if platform.system() == "Darwin" else "xdg-open"
                        subprocess.run([opener, str(output_filename)], check=False, timeout=5)
                    except Exception:
                        ASCIIColors.yellow(f"Could not auto-open image. Please find it at {output_filename}")
            except Exception as e:
                ASCIIColors.error(f"Error processing or saving image: {e}")
                trace_exception(e)
                raw_output_filename = output_filename.with_suffix(".raw_data")
                with open(raw_output_filename, "wb") as f_raw: f_raw.write(image_bytes)
                ASCIIColors.yellow(f"Raw image data saved as {raw_output_filename} for inspection.")
                raise
        else:
            ASCIIColors.red("Image generation returned empty bytes.")
            assert False, "Image generation returned empty bytes"
    except Exception as e:
        ASCIIColors.error(f"Error during image generation: {e}")
        trace_exception(e)
        raise

if __name__ == "__main__":
    # --- DALL-E Test ---
    ASCIIColors.magenta("\n\n========== DALL-E Binding Test ==========")
    if not os.environ.get("OPENAI_API_KEY"):
        ASCIIColors.warning("OPENAI_API_KEY environment variable not set. Skipping DALL-E tests.")
    else:
        try:
            lc_dalle = LollmsClient(tti_binding_name="dalle", service_key=LOLLMS_CLIENT_ID)
            if not lc_dalle.tti: ASCIIColors.error("DALL-E TTI binding could not be initialized.")
            else:
                test_list_tti_services(lc_dalle)
                test_get_tti_settings(lc_dalle)
                test_set_tti_settings(lc_dalle) 
                test_generate_image(lc_dalle, BASE_TEST_PATH / "generated_dalle_image.png",
                                    "A vibrant oil painting of a mythical creature in an enchanted forest",
                                    "photorealistic, modern, ugly, deformed", 1024, 1024)
        except Exception as e:
            ASCIIColors.error(f"DALL-E test block failed: {e}"); trace_exception(e)

    # --- Diffusers Test ---
    ASCIIColors.magenta("\n\n========== Diffusers Binding Test ==========")
    if not HUGGINGFACE_HUB_AVAILABLE:
        ASCIIColors.warning("Skipping Diffusers tests as huggingface_hub is not available.")
    else:
        diffusers_binding_config = None
        try:
            diffusers_binding_config, _ = setup_diffusers_environment(BASE_TEST_PATH)
            lc_diffusers = LollmsClient(tti_binding_name="diffusers", binding_config=diffusers_binding_config, service_key=LOLLMS_CLIENT_ID)
            if not lc_diffusers.tti: raise RuntimeError("Diffusers TTI binding failed to initialize")
            
            test_list_tti_services(lc_diffusers)
            test_get_tti_settings(lc_diffusers)
            test_set_tti_settings(lc_diffusers)
            
            gen_width = lc_diffusers.tti.config.get("default_width", 512) 
            gen_height = lc_diffusers.tti.config.get("default_height", 512)
            diffusers_prompt = "A majestic griffin soaring through a cloudy sky, detailed feathers, fantasy art" 
            diffusers_negative_prompt = "ugly, blurry, low quality, watermark, text, simple background"
            
            test_generate_image(lc_diffusers, BASE_TEST_PATH / "generated_diffusers_image.png",
                                diffusers_prompt, diffusers_negative_prompt, 
                                gen_width, gen_height)
        except Exception as e:
            ASCIIColors.error(f"Diffusers test block failed: {e}"); trace_exception(e)
        finally:
            cleanup_diffusers_environment(BASE_TEST_PATH)

    ASCIIColors.magenta("\n\n========== All Tests Finished ==========")