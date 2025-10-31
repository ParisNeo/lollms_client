import os
import requests
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

pm.ensure_packages(["requests", "Pillow"])

from PIL import Image

BindingName = "StabilityAITTIBinding"

# Sourced from https://platform.stability.ai/docs/getting-started/models
STABILITY_AI_MODELS = [
    # SD3
    {"model_name": "stable-diffusion-3-medium", "display_name": "Stable Diffusion 3 Medium", "description": "Most advanced text-to-image model.", "owned_by": "Stability AI"},
    {"model_name": "stable-diffusion-3-large", "display_name": "Stable Diffusion 3 Large", "description": "Most advanced model with higher quality.", "owned_by": "Stability AI"},
    {"model_name": "stable-diffusion-3-large-turbo", "display_name": "Stable Diffusion 3 Large Turbo", "description": "Fast, high-quality generation.", "owned_by": "Stability AI"},
    # SDXL
    {"model_name": "stable-diffusion-xl-1024-v1-0", "display_name": "Stable Diffusion XL 1.0", "description": "High-quality 1024x1024 generation.", "owned_by": "Stability AI"},
    {"model_name": "stable-diffusion-xl-beta-v2-2-2", "display_name": "SDXL Beta", "description": "Legacy anime-focused SDXL model.", "owned_by": "Stability AI"},
    # SD 1.x & 2.x
    {"model_name": "stable-diffusion-v1-6", "display_name": "Stable Diffusion 1.6", "description": "Improved version of SD 1.5.", "owned_by": "Stability AI"},
    {"model_name": "stable-diffusion-2-1", "display_name": "Stable Diffusion 2.1", "description": "768x768 native resolution model.", "owned_by": "Stability AI"},
]

class StabilityAITTIBinding(LollmsTTIBinding):
    """Stability AI TTI binding for LoLLMS"""

    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)
        self.api_key = self.config.get("api_key") or os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("Stability AI API key is required. Please set it in the configuration or as STABILITY_API_KEY environment variable.")
        self.model_name = self.config.get("model_name", "stable-diffusion-3-medium")

    def list_models(self) -> list:
        return STABILITY_AI_MODELS

    def _get_api_url(self, task: str) -> str:
        base_url = "https://api.stability.ai/v2beta/stable-image"
        # SD3 models use a different endpoint structure
        if "stable-diffusion-3" in self.model_name:
            return f"{base_url}/generate/sd3"
        
        task_map = {
            "text2image": "generate/core",
            "image2image": "edit/image-to-image",
            "inpainting": "edit/in-painting",
            "upscale": "edit/upscale"
        }
        if task not in task_map:
            raise ValueError(f"Unsupported task for this model family: {task}")
        return f"{base_url}/{task_map[task]}"

    def _decode_image_input(self, item: Union[str, Path, bytes]) -> Image.Image:
        if isinstance(item, bytes):
            return Image.open(BytesIO(item))
        s = str(item).strip()
        if s.startswith("data:image/") and ";base64," in s:
            b64 = s.split(";base64,")[-1]
            return Image.open(BytesIO(base64.b64decode(b64)))
        try:
            p = Path(s)
            if p.exists():
                return Image.open(p)
        except:
            pass
        if s.startswith("http"):
            response = requests.get(s, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
        # Fallback for raw base64
        return Image.open(BytesIO(base64.b64decode(s)))

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, **kwargs) -> bytes:
        url = self._get_api_url("text2image")
        
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_format": "png",
            "seed": kwargs.get("seed", 0)
        }

        # SD3 uses aspect_ratio, older models use width/height
        if "stable-diffusion-3" in self.model_name:
            data["aspect_ratio"] = f"{width}:{height}"
            data["model"] = self.model_name
        else:
            data["width"] = width
            data["height"] = height
            data["style_preset"] = kwargs.get("style_preset", "photographic")

        headers = {"authorization": f"Bearer {self.api_key}", "accept": "image/*"}
        
        try:
            ASCIIColors.info(f"Requesting image from Stability AI ({self.model_name})...")
            response = requests.post(url, headers=headers, files={"none": ''}, data=data)
            response.raise_for_status()
            return response.content
        except Exception as e:
            trace_exception(e)
            try:
                error_msg = response.json()
                raise Exception(f"Stability AI API error: {error_msg}")
            except:
                raise Exception(f"Stability AI API request failed: {e}")

    def edit_image(self, images: Union[str, List[str]], prompt: str, negative_prompt: Optional[str] = "", mask: Optional[str] = None, **kwargs) -> bytes:
        init_image_bytes = BytesIO()
        init_image = self._decode_image_input(images[0] if isinstance(images, list) else images)
        init_image.save(init_image_bytes, format="PNG")
        
        task = "inpainting" if mask else "image2image"
        url = self._get_api_url(task)

        files = {"image": init_image_bytes.getvalue()}
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "output_format": "png",
            "seed": kwargs.get("seed", 0)
        }

        if task == "inpainting":
            mask_image_bytes = BytesIO()
            mask_image = self._decode_image_input(mask)
            mask_image.save(mask_image_bytes, format="PNG")
            files["mask"] = mask_image_bytes.getvalue()
        else: # image2image
            data["strength"] = kwargs.get("strength", 0.6) # mode IMAGE_STRENGTH
        
        headers = {"authorization": f"Bearer {self.api_key}", "accept": "image/*"}

        try:
            ASCIIColors.info(f"Requesting image edit from Stability AI ({self.model_name})...")
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.content
        except Exception as e:
            trace_exception(e)
            try:
                error_msg = response.json()
                raise Exception(f"Stability AI API error: {error_msg}")
            except:
                raise Exception(f"Stability AI API request failed: {e}")

if __name__ == '__main__':
    ASCIIColors.magenta("--- Stability AI TTI Binding Test ---")
    if "STABILITY_API_KEY" not in os.environ:
        ASCIIColors.error("STABILITY_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = StabilityAITTIBinding(model_name="stable-diffusion-3-medium")
        
        ASCIIColors.cyan("\n--- Test: Text-to-Image ---")
        prompt = "a cinematic photo of a robot drinking coffee in a Parisian cafe"
        img_bytes = binding.generate_image(prompt, width=1024, height=1024)
        
        assert len(img_bytes) > 1000, "Generated image bytes are too small."
        output_path = Path(__file__).parent / "tmp_stability_t2i.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        ASCIIColors.green(f"Text-to-Image generation OK. Image saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Stability AI binding test failed: {e}")