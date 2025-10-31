import os
import requests
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

pm.ensure_packages(["requests"])

BindingName = "NovitaAITTIBinding"

# Sourced from https://docs.novita.ai/image-generation/models
NOVITA_AI_MODELS = [
    {"model_name": "sd_xl_base_1.0.safetensors", "display_name": "Stable Diffusion XL 1.0", "description": "Official SDXL 1.0 Base model."},
    {"model_name": "dreamshaper_xl_1_0.safetensors", "display_name": "DreamShaper XL 1.0", "description": "Versatile artistic SDXL model."},
    {"model_name": "juggernaut_xl_v9_rundiffusion.safetensors", "display_name": "Juggernaut XL v9", "description": "High-quality realistic and cinematic model."},
    {"model_name": "realistic_vision_v5.1.safetensors", "display_name": "Realistic Vision v5.1", "description": "Popular photorealistic SD1.5 model."},
    {"model_name": "absolutereality_v1.8.1.safetensors", "display_name": "Absolute Reality v1.8.1", "description": "General-purpose realistic SD1.5 model."},
    {"model_name": "meinamix_meina_v11.safetensors", "display_name": "MeinaMix v11", "description": "High-quality anime illustration model."},
]

class NovitaAITTIBinding(LollmsTTIBinding):
    """Novita.ai TTI binding for LoLLMS"""

    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)
        self.config = kwargs
        self.api_key = self.config.get("api_key") or os.environ.get("NOVITA_API_KEY")
        if not self.api_key:
            raise ValueError("Novita.ai API key is required.")
        self.model_name = self.config.get("model_name", "juggernaut_xl_v9_rundiffusion.safetensors")
        self.base_url = "https://api.novita.ai/v3"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list:
        return NOVITA_AI_MODELS
    
    def generate_image(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, **kwargs) -> bytes:
        url = f"{self.base_url}/text2img"
        payload = {
            "model_name": self.model_name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "sampler_name": "DPM++ 2M Karras",
            "cfg_scale": kwargs.get("guidance_scale", 7.0),
            "steps": kwargs.get("num_inference_steps", 25),
            "seed": kwargs.get("seed", -1),
            "n_iter": 1,
            "batch_size": 1
        }
        
        try:
            ASCIIColors.info(f"Requesting image from Novita.ai ({self.model_name})...")
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if "images" not in data or not data["images"]:
                raise Exception(f"API returned no images. Response: {data}")
            
            b64_image = data["images"][0]["image_base64"]
            return base64.b64decode(b64_image)
            
        except Exception as e:
            trace_exception(e)
            try:
                error_msg = response.json()
                raise Exception(f"Novita.ai API error: {error_msg}")
            except:
                raise Exception(f"Novita.ai API request failed: {e}")

    def edit_image(self, **kwargs) -> bytes:
        ASCIIColors.warning("Novita.ai edit_image (inpainting/img2img) is not yet implemented in this binding.")
        raise NotImplementedError("This binding does not yet support image editing.")

if __name__ == '__main__':
    ASCIIColors.magenta("--- Novita.ai TTI Binding Test ---")
    if "NOVITA_API_KEY" not in os.environ:
        ASCIIColors.error("NOVITA_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = NovitaAITTIBinding()
        
        ASCIIColors.cyan("\n--- Test: Text-to-Image ---")
        prompt = "A cute capybara wearing a top hat, sitting in a field of flowers, painterly style"
        img_bytes = binding.generate_image(prompt, width=1024, height=1024, num_inference_steps=30)
        
        assert len(img_bytes) > 1000
        output_path = Path(__file__).parent / "tmp_novita_t2i.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        ASCIIColors.green(f"Text-to-Image generation OK. Image saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Novita.ai binding test failed: {e}")