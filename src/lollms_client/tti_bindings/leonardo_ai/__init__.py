import os
import requests
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

pm.ensure_packages(["requests", "Pillow"])
from PIL import Image

BindingName = "LeonardoAITTIBinding"

# Sourced from https://docs.leonardo.ai/docs/models
LEONARDO_AI_MODELS = [
    {"model_name": "ac4f3991-8a40-42cd-b174-14a8e33738e4", "display_name": "Leonardo Phoenix", "description": "Fast, high-quality photorealism."},
    {"model_name": "1e65d070-22c9-4aed-a5be-ce58a1b65b38", "display_name": "Leonardo Diffusion XL", "description": "The flagship general-purpose SDXL model."},
    {"model_name": "b24e16ff-06e3-43eb-a255-db4322b0f345", "display_name": "AlbedoBase XL", "description": "Versatile model for photorealism and artistic styles."},
    {"model_name": "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3", "display_name": "Absolute Reality v1.6", "description": "Classic photorealistic model."},
    {"model_name": "f3296a34-a868-4665-8b2f-f4313f8c8533", "display_name": "RPG v5", "description": "Specialized in RPG characters and assets."},
    {"model_name": "2067ae58-a02e-4318-9742-2b55b2a4c813", "display_name": "DreamShaper v7", "description": "Popular versatile artistic model."},
]

class LeonardoAITTIBinding(LollmsTTIBinding):
    """Leonardo.ai TTI binding for LoLLMS"""

    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)

        self.api_key = self.config.get("api_key") or os.environ.get("LEONARDO_API_KEY")
        if not self.api_key:
            raise ValueError("Leonardo.ai API key is required.")
        self.model_name = self.config.get("model_name", "ac4f3991-8a40-42cd-b174-14a8e33738e4")
        self.base_url = "https://cloud.leonardo.ai/api/rest/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self) -> list:
        # You could also fetch this dynamically from /models endpoint
        return LEONARDO_AI_MODELS
    
    def _wait_for_generation(self, generation_id: str) -> List[bytes]:
        while True:
            url = f"{self.base_url}/generations/{generation_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json().get("generations_by_pk", {})
            status = data.get("status")

            if status == "COMPLETE":
                ASCIIColors.green("Generation complete.")
                images_data = []
                for img in data.get("generated_images", []):
                    img_url = img.get("url")
                    if img_url:
                        img_response = requests.get(img_url)
                        img_response.raise_for_status()
                        images_data.append(img_response.content)
                return images_data
            elif status == "FAILED":
                raise Exception("Leonardo.ai generation failed.")
            else:
                ASCIIColors.info(f"Generation status: {status}. Waiting...")
                time.sleep(3)

    def generate_image(self, prompt: str, negative_prompt: str = "", width: int = 1024, height: int = 1024, **kwargs) -> bytes:
        url = f"{self.base_url}/generations"
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "modelId": self.model_name,
            "width": width,
            "height": height,
            "num_images": 1,
            "guidance_scale": kwargs.get("guidance_scale", 7),
            "seed": kwargs.get("seed"),
            "sd_version": "SDXL" # Most models are SDXL based
        }
        
        try:
            ASCIIColors.info(f"Submitting generation job to Leonardo.ai ({self.model_name})...")
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            generation_id = response.json()["sdGenerationJob"]["generationId"]
            ASCIIColors.info(f"Job submitted with ID: {generation_id}")
            images = self._wait_for_generation(generation_id)
            return images[0]
        except Exception as e:
            trace_exception(e)
            try:
                error_msg = response.json()
                raise Exception(f"Leonardo.ai API error: {error_msg}")
            except:
                raise Exception(f"Leonardo.ai API request failed: {e}")

    def edit_image(self, **kwargs) -> bytes:
        ASCIIColors.warning("Leonardo.ai edit_image (inpainting/img2img) is not yet implemented in this binding.")
        raise NotImplementedError("This binding does not yet support image editing.")

if __name__ == '__main__':
    ASCIIColors.magenta("--- Leonardo.ai TTI Binding Test ---")
    if "LEONARDO_API_KEY" not in os.environ:
        ASCIIColors.error("LEONARDO_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = LeonardoAITTIBinding()
        
        ASCIIColors.cyan("\n--- Test: Text-to-Image ---")
        prompt = "A majestic lion wearing a crown, hyperrealistic, 8k"
        img_bytes = binding.generate_image(prompt, width=1024, height=1024)
        
        assert len(img_bytes) > 1000
        output_path = Path(__file__).parent / "tmp_leonardo_t2i.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        ASCIIColors.green(f"Text-to-Image generation OK. Image saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Leonardo.ai binding test failed: {e}")