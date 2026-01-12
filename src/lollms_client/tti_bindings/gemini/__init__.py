# lollms_client/tti_bindings/gemini/__init__.py
import sys
from typing import Optional, List, Dict, Any, Union
import os
import io
import base64
import requests
import binascii
import time

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors

# --- SDK & Dependency Management ---
try:
    import pipmaster as pm
    pm.ensure_packages(['google-cloud-aiplatform', 'google-genai', 'Pillow', 'requests'])
except ImportError:
    pass

# Attempt to import Vertex AI
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel as VertexImageGenerationModel, Image as VertexImage
    from google.api_core import exceptions as google_exceptions
    from PIL import Image as PILImage
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Attempt to import Gemini API (NEW SDK)
try:
    import google.genai as genai
    from google.genai import types
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False

# Defines the binding name for the manager
BindingName = "GeminiTTIBinding_Impl"

# Static list for Vertex AI, as it's project-based and more predictable
IMAGEN_VERTEX_MODELS = ["imagegeneration@006", "imagen-3.0-generate-002", "gemini-2.5-flash-image"]
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"


def _is_base64(s):
    try:
        base64.b64decode(s.split(',')[-1], validate=True)
        return True
    except (TypeError, ValueError, binascii.Error):
        return False

def _load_image_from_str(image_str: str) -> bytes:
    if image_str.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_str)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to download image from URL: {image_str}") from e
    elif _is_base64(image_str):
        header, encoded = image_str.split(',', 1)
        return base64.b64decode(encoded)
    else:
        raise ValueError("Image string is not a valid URL or base64 string.")


class GeminiTTIBinding_Impl(LollmsTTIBinding):
    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)
        self.auth_method = kwargs.get("auth_method", "vertex_ai")
        self.client: Optional[Any] = None
        self.available_models = []

        if self.auth_method == "vertex_ai":
            if not VERTEX_AI_AVAILABLE:
                raise ImportError("Vertex AI selected, but 'google-cloud-aiplatform' is not installed.")
            self.project_id = kwargs.get("project_id")
            self.location = kwargs.get("location", "us-central1")
            if not self.project_id:
                raise ValueError("For 'vertex_ai' auth, 'project_id' is required.")
            self.model_name = kwargs.get("model_name") # Can be None initially
            self.available_models = IMAGEN_VERTEX_MODELS

        elif self.auth_method == "api_key":
            if not GEMINI_API_AVAILABLE:
                raise ImportError("API Key selected, but 'google-genai' is not installed.")
            self.gemini_api_key = kwargs.get("service_key") or os.environ.get(GEMINI_API_KEY_ENV_VAR)
            if not self.gemini_api_key:
                raise ValueError(f"For 'api_key' auth, 'service_key' or env var '{GEMINI_API_KEY_ENV_VAR}' is required.")
            self.model_name = kwargs.get("model_name") # Can be None initially
        else:
            raise ValueError(f"Invalid auth_method: '{self.auth_method}'. Must be 'vertex_ai' or 'api_key'.")

        self.default_seed = int(kwargs.get("default_seed", -1))
        self.default_guidance_scale = float(kwargs.get("default_guidance_scale", 7.5))
        self._initialize_client()

    def _initialize_client(self):
        ASCIIColors.info(f"Initializing Google client with auth method: '{self.auth_method}'...")
        try:
            if self.auth_method == "vertex_ai":
                vertexai.init(project=self.project_id, location=self.location)
                if not self.model_name:
                    self.model_name = self.available_models[0]
                self.client = VertexImageGenerationModel.from_pretrained(self.model_name)
                ASCIIColors.green(f"Vertex AI initialized successfully. Project: '{self.project_id}', Model: '{self.model_name}'")
            
            elif self.auth_method == "api_key":
                # NEW: Use Client-based initialization
                self.client = genai.Client(api_key=self.gemini_api_key)
                
                # --- DYNAMIC MODEL DISCOVERY ---
                ASCIIColors.info("Discovering available image models for your API key...")
                
                # NEW: Use client.models.list()
                all_models = list(self.client.models.list())
                self.available_models = [
                    m.name for m in all_models
                    if 'imagen' in m.name.lower()
                ]
                
                if not self.available_models:
                    raise Exception("Your API key does not have access to any compatible image generation models. Please check your Google AI Studio project settings.")
                
                ASCIIColors.green(f"Found available models: {self.available_models}")

                # Validate or set the model_name
                if self.model_name and self.model_name not in self.available_models:
                    ASCIIColors.warning(f"Model '{self.model_name}' is not available for your key. Falling back to default.")
                    self.model_name = None
                
                if not self.model_name:
                    self.model_name = self.available_models[0]

                ASCIIColors.green(f"Gemini API configured successfully. Using Model: '{self.model_name}'")

        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to initialize Google client: {e}") from e

    def generate_image(self, prompt: str, negative_prompt: Optional[str] = "", width: int = 1024, height: int = 1024, **kwargs) -> bytes:
        if not self.client:
            raise RuntimeError("Google client is not initialized.")
        
        ASCIIColors.info(f"Generating image with prompt: '{prompt[:100]}...'")
        
        try:
            if self.auth_method == "vertex_ai":
                return self._generate_with_vertex_ai(prompt, negative_prompt, width, height, **kwargs)
            elif self.auth_method == "api_key":
                return self._generate_with_api_key(prompt, negative_prompt, width, height, **kwargs)
        except Exception as e:
            if "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                 raise Exception(f"Image generation failed due to a quota error. This means you have exceeded the free tier limit for your API key. To fix this, please enable billing on your Google Cloud project. Original error: {e}")
            raise Exception(f"Image generation failed: {e}")

    def _generate_with_api_key(self, prompt, negative_prompt, width, height, **kwargs):
        full_prompt = f"Generate an image of: {prompt}"
        if negative_prompt:
            full_prompt += f". Do not include: {negative_prompt}."

        max_retries = 3
        initial_delay = 5

        for attempt in range(max_retries):
            try:
                # NEW: Use client.models.generate_content
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt
                )
                
                # Check if response contains image data
                if not response.candidates:
                    raise Exception(f"API response did not contain any candidates. Response: {response}")
                
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    raise Exception(f"API response did not contain image data. Check safety filters in your Google AI Studio.")
                
                # Extract image data from response
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        return part.inline_data.data
                    elif hasattr(part, 'file_data') and part.file_data:
                        # If it's a file reference, we need to download it
                        # For now, just return the data if available
                        if hasattr(part.file_data, 'data'):
                            return part.file_data.data
                        else:
                            raise Exception("File data returned but no direct data available")
                
                raise Exception("No image data found in response parts")
            
            except Exception as e:
                # Check for rate limiting errors
                error_str = str(e).lower()
                if "resource_exhausted" in error_str or "quota" in error_str or "rate" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2 ** attempt)
                        ASCIIColors.warning(f"Rate limit exceeded. Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        ASCIIColors.error(f"Failed to generate image after {max_retries} attempts due to rate limiting.")
                        raise e
                else:
                    raise e

    def _generate_with_vertex_ai(self, prompt, negative_prompt, width, height, **kwargs):
        self._validate_dimensions_vertex(width, height)
        gen_params = {
            "prompt": prompt, "number_of_images": 1, "width": width, "height": height,
            "guidance_scale": kwargs.get("guidance_scale", self.default_guidance_scale),
        }
        if negative_prompt: gen_params["negative_prompt"] = negative_prompt
        seed = kwargs.get("seed", self.default_seed)
        if seed != -1: gen_params["seed"] = seed
        
        response = self.client.generate_images(**gen_params)
        if not response.images: raise Exception("Generation resulted in no images (Vertex AI).")
        return response.images[0]._image_bytes

    def edit_image(self, images: Union[str, List[str]], prompt: str, negative_prompt: Optional[str] = "", mask: Optional[str] = None, **kwargs) -> bytes:
        if self.auth_method != "vertex_ai":
            raise NotImplementedError("Image editing is only supported via the 'vertex_ai' method.")
        
        image_str = images[0] if isinstance(images, list) else images
        ASCIIColors.info(f"Editing image with prompt: '{prompt[:100]}...'")
        
        try:
            base_image = VertexImage(image_bytes=_load_image_from_str(image_str))
            mask_image = VertexImage(image_bytes=_load_image_from_str(mask)) if mask else None
            edit_params = {"prompt": prompt, "base_image": base_image, "mask": mask_image, "negative_prompt": negative_prompt or None}
            response = self.client.edit_image(**edit_params)
            if not response.images: raise Exception("Image editing resulted in no images.")
            return response.images[0]._image_bytes
        except Exception as e:
            raise Exception(f"Imagen image editing failed: {e}") from e

    def list_models(self) -> list:
        return [{'model_name': name, 'display_name': f"Google ({name})"} for name in self.available_models]

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        service_name = "Vertex AI" if self.auth_method == "vertex_ai" else "Gemini API"
        return [{"name": name, "caption": f"Google ({name}) via {service_name}"} for name in self.available_models]

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        # Simplified for clarity, full logic is complex and stateful
        needs_reinit = False
        if "auth_method" in settings and self.auth_method != settings["auth_method"]:
            self.auth_method = settings["auth_method"]
            needs_reinit = True
        if "project_id" in settings and self.project_id != settings.get("project_id"):
            self.project_id = settings["project_id"]
            needs_reinit = True
        if "service_key" in settings and self.gemini_api_key != settings.get("service_key"):
             self.gemini_api_key = settings["service_key"]
             needs_reinit = True
        if "model_name" in settings and self.model_name != settings.get("model_name"):
            self.model_name = settings["model_name"]
            needs_reinit = True
        if needs_reinit:
            try:
                self._initialize_client()
            except Exception as e:
                ASCIIColors.error(f"Failed to re-initialize client: {e}")
                return False
        return True
        
    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return {
            "auth_method": self.auth_method,
            "project_id": self.project_id if self.auth_method == "vertex_ai" else None,
            "location": self.location if self.auth_method == "vertex_ai" else None,
            "model_name": self.model_name,
            "default_seed": self.default_seed,
            "default_guidance_scale": self.default_guidance_scale
        }

    def _validate_dimensions_vertex(self, width: int, height: int) -> None:
        if not (256 <= width <= 1536 and width % 8 == 0):
            raise ValueError(f"Invalid width for Vertex AI: {width}. Must be 256-1536 and a multiple of 8.")
        if not (256 <= height <= 1536 and height % 8 == 0):
            raise ValueError(f"Invalid height for Vertex AI: {height}. Must be 256-1536 and a multiple of 8.")