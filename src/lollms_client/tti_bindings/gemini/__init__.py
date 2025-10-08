# lollms_client/tti_bindings/gemini/__init__.py
import sys
from typing import Optional, List, Dict, Any, Union
import os
import io
import base64
import requests

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import math

# --- SDK & Dependency Management ---
try:
    import pipmaster as pm
    # Ensure necessary packages are available
    pm.ensure_packages(['google-cloud-aiplatform', 'google-generativeai', 'Pillow', 'requests'])
except ImportError:
    pass # pipmaster is optional

# Attempt to import Vertex AI (google-cloud-aiplatform)
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel, Image
    from google.api_core import exceptions as google_exceptions
    from PIL import Image as PILImage
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Attempt to import Gemini API (google-generativeai)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False

# Defines the binding name for the manager
BindingName = "GeminiTTIBinding_Impl"

# Known Imagen models for each service
IMAGEN_VERTEX_MODELS = ["imagegeneration@006", "imagen-3.0-generate-002", "gemini-2.5-flash-image"]
IMAGEN_GEMINI_API_MODELS = ["imagen-3", "gemini-1.5-flash-preview-0514", "gemini-2.5-flash-image"]
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"


def _is_base64(s):
    """Check if a string is a valid base64 encoded string."""
    try:
        # We don't need the result, just to see if it decodes
        base64.b64decode(s.split(',')[-1], validate=True)
        return True
    except (TypeError, ValueError, binascii.Error):
        return False

def _load_image_from_str(image_str: str) -> bytes:
    """Loads image data from a URL or a base64 encoded string."""
    if image_str.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_str)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to download image from URL: {image_str}") from e
    elif _is_base64(image_str):
        # Handle data URLs (e.g., "data:image/png;base64,iVBOR...")
        header, encoded = image_str.split(',', 1)
        return base64.b64decode(encoded)
    else:
        raise ValueError("Image string is not a valid URL or base64 string.")


class GeminiTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for Google's Imagen and Gemini models.
    Supports both Vertex AI (project_id) and Gemini API (api_key) authentication.
    Includes support for image generation, editing, and inpainting.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Gemini (Vertex AI / API) TTI binding.
        """
        super().__init__(binding_name="gemini")

        # Core settings
        self.auth_method = kwargs.get("auth_method", "vertex_ai")

        # Vertex AI specific settings
        self.project_id = kwargs.get("project_id")
        self.location = kwargs.get("location", "us-central1")

        # Gemini API specific settings
        self.gemini_api_key = kwargs.get("service_key")

        # Common settings
        self.model_name = kwargs.get("model_name")
        self.default_seed = int(kwargs.get("default_seed", -1))
        self.default_guidance_scale = float(kwargs.get("default_guidance_scale", 7.5))
        self.client_id = kwargs.get("client_id", "gemini_client_user")
        
        self.client: Optional[Any] = None

        if self.auth_method == "vertex_ai":
            if not VERTEX_AI_AVAILABLE:
                raise ImportError("Vertex AI authentication selected, but 'google-cloud-aiplatform' or 'Pillow' are not installed.")
            if not self.project_id:
                raise ValueError("For 'vertex_ai' auth, a Google Cloud 'project_id' is required.")
            if not self.model_name:
                self.model_name = IMAGEN_VERTEX_MODELS[0]
        elif self.auth_method == "api_key":
            if not GEMINI_API_AVAILABLE:
                raise ImportError("API Key authentication selected, but 'google-generativeai' is not installed.")
            
            if not self.gemini_api_key:
                ASCIIColors.info(f"API key not provided directly, checking environment variable '{GEMINI_API_KEY_ENV_VAR}'...")
                self.gemini_api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR)

            if not self.gemini_api_key:
                raise ValueError(f"For 'api_key' auth, a Gemini API Key is required. Provide it as 'service_key' or set the '{GEMINI_API_KEY_ENV_VAR}' environment variable.")

            if not self.model_name:
                self.model_name = IMAGEN_GEMINI_API_MODELS[0]
        else:
            raise ValueError(f"Invalid auth_method: '{self.auth_method}'. Must be 'vertex_ai' or 'api_key'.")

        self._initialize_client()

    def _initialize_client(self):
        """Initializes the appropriate client based on the selected auth_method."""
        ASCIIColors.info(f"Initializing Google client with auth method: '{self.auth_method}'...")
        try:
            if self.auth_method == "vertex_ai":
                vertexai.init(project=self.project_id, location=self.location)
                self.client = ImageGenerationModel.from_pretrained(self.model_name)
                ASCIIColors.green(f"Vertex AI initialized successfully. Project: '{self.project_id}', Model: '{self.model_name}'")
            elif self.auth_method == "api_key":
                genai.configure(api_key=self.gemini_api_key)
                self.client = genai
                ASCIIColors.green(f"Gemini API configured successfully. Model to be used: '{self.model_name}'")
        except google_exceptions.PermissionDenied as e:
            trace_exception(e)
            raise Exception(
                "Authentication failed. For Vertex AI, run 'gcloud auth application-default login'. For API Key, check if the key is valid and has permissions."
            ) from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to initialize Google client: {e}") from e

    def _validate_dimensions_vertex(self, width: int, height: int) -> None:
        """Validates image dimensions against Vertex AI Imagen constraints."""
        if not (256 <= width <= 1536 and width % 8 == 0):
            raise ValueError(f"Invalid width for Vertex AI: {width}. Must be 256-1536 and a multiple of 8.")
        if not (256 <= height <= 1536 and height % 8 == 0):
            raise ValueError(f"Invalid height for Vertex AI: {height}. Must be 256-1536 and a multiple of 8.")

    def _get_aspect_ratio_for_api(self, width: int, height: int) -> str:
        """Finds the closest supported aspect ratio string for the Gemini API."""
        ratios = {"1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4, "21:9":21/9, "3:2":3/2, "2:3":2/3}
        target_ratio = width / height
        closest_ratio_name = min(ratios, key=lambda r: abs(ratios[r] - target_ratio))
        ASCIIColors.info(f"Converted {width}x{height} to closest aspect ratio: '{closest_ratio_name}' for Gemini API.")
        return closest_ratio_name

    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 1024,
                       height: int = 1024,
                       **kwargs) -> bytes:
        if not self.client:
            raise RuntimeError("Google client is not initialized. Cannot generate image.")

        seed = kwargs.get("seed", self.default_seed)
        guidance_scale = kwargs.get("guidance_scale", self.default_guidance_scale)
        gen_seed = seed if seed != -1 else None

        ASCIIColors.info(f"Generating image with prompt: '{prompt[:100]}...'")

        try:
            if self.auth_method == "vertex_ai":
                self._validate_dimensions_vertex(width, height)
                gen_params = {
                    "prompt": prompt,
                    "number_of_images": 1,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale,
                }
                if negative_prompt:
                    gen_params["negative_prompt"] = negative_prompt
                if gen_seed is not None:
                    gen_params["seed"] = gen_seed
                
                ASCIIColors.debug(f"Vertex AI generation parameters: {gen_params}")
                response = self.client.generate_images(**gen_params)
                
                if not response.images:
                    raise Exception("Image generation resulted in no images (Vertex AI). Check safety filters.")
                
                return response.images[0]._image_bytes

            elif self.auth_method == "api_key":
                final_prompt = f"{prompt}. Do not include: {negative_prompt}." if negative_prompt else prompt
                aspect_ratio = self._get_aspect_ratio_for_api(width, height)
                gen_params = {
                    "model": self.model_name,
                    "prompt": final_prompt,
                    "number_of_images": 1,
                    "aspect_ratio": aspect_ratio
                }
                ASCIIColors.debug(f"Gemini API generation parameters: {gen_params}")
                response = self.client.generate_image(**gen_params)

                if not response.images:
                    raise Exception("Image generation resulted in no images (Gemini API). Check safety filters.")

                return response.images[0].image_bytes

        except (google_exceptions.InvalidArgument, AttributeError) as e:
            trace_exception(e)
            raise ValueError(f"Invalid argument sent to Google API: {e}") from e
        except google_exceptions.GoogleAPICallError as e:
            trace_exception(e)
            raise Exception(f"A Google API call error occurred: {e}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Imagen image generation failed: {e}") from e

    def edit_image(self,
                   images: Union[str, List[str]],
                   prompt: str,
                   negative_prompt: Optional[str] = "",
                   mask: Optional[str] = None,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   **kwargs) -> bytes:
        if self.auth_method != "vertex_ai":
            raise NotImplementedError("Image editing is only supported via the 'vertex_ai' authentication method.")
        if not self.client:
            raise RuntimeError("Vertex AI client is not initialized.")

        if isinstance(images, list):
            if len(images) > 1:
                raise ValueError("Vertex AI edit_image only supports a single base image.")
            image_str = images[0]
        else:
            image_str = images

        ASCIIColors.info(f"Editing image with prompt: '{prompt[:100]}...'")
        
        try:
            image_bytes = _load_image_from_str(image_str)
            base_image = Image(image_bytes=image_bytes)
            
            mask_image = None
            if mask:
                mask_bytes = _load_image_from_str(mask)
                mask_image = Image(image_bytes=mask_bytes)

            seed = kwargs.get("seed", self.default_seed)
            gen_seed = seed if seed != -1 else None
            
            edit_params = {
                "prompt": prompt,
                "base_image": base_image,
                "mask": mask_image,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "number_of_images": 1
            }
            # Add any extra valid parameters from kwargs (like edit_mode)
            if "edit_mode" in kwargs:
                edit_params["edit_mode"] = kwargs["edit_mode"]

            if gen_seed is not None:
                edit_params["seed"] = gen_seed
            
            ASCIIColors.debug(f"Vertex AI edit parameters: {edit_params}")
            response = self.client.edit_image(**edit_params)

            if not response.images:
                raise Exception("Image editing resulted in no images. Check safety filters.")
            
            return response.images[0]._image_bytes

        except Exception as e:
            trace_exception(e)
            raise Exception(f"Imagen image editing failed: {e}") from e

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """Lists available models for the current auth method."""
        models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        service_name = "Vertex AI" if self.auth_method == "vertex_ai" else "Gemini API"
        return [
            {
                "name": name,
                "caption": f"Google ({name}) via {service_name}",
                "help": "High-quality text-to-image model from Google."
            } for name in models
        ]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieves the current configurable settings for the binding."""
        # Adheres to the Dict[str, Any] return type
        return {
            "auth_method": self.auth_method,
            "project_id": self.project_id if self.auth_method == "vertex_ai" else None,
            "location": self.location if self.auth_method == "vertex_ai" else None,
            "model_name": self.model_name,
            "default_seed": self.default_seed,
            "default_guidance_scale": self.default_guidance_scale
        }

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        """Applies new settings. Re-initializes the client if core settings change."""
        # Adheres to the Dict[str, Any] input type
        applied_some_settings = False
        needs_reinit = False
        
        if "auth_method" in settings and self.auth_method != settings["auth_method"]:
            self.auth_method = settings["auth_method"]
            ASCIIColors.info(f"Authentication method changed to: {self.auth_method}")
            if self.auth_method == "vertex_ai":
                self.model_name = IMAGEN_VERTEX_MODELS[0]
            else:
                self.model_name = IMAGEN_GEMINI_API_MODELS[0]
            ASCIIColors.info(f"Model name reset to default for new auth method: {self.model_name}")
            needs_reinit = True
            applied_some_settings = True
        
        if self.auth_method == "vertex_ai":
            if "project_id" in settings and self.project_id != settings["project_id"]:
                self.project_id = settings["project_id"]
                needs_reinit = True; applied_some_settings = True
            if "location" in settings and self.location != settings["location"]:
                self.location = settings["location"]
                needs_reinit = True; applied_some_settings = True

        current_models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        if "model_name" in settings:
            new_model = settings["model_name"]
            if new_model not in current_models:
                ASCIIColors.warning(f"Invalid model '{new_model}' for auth method '{self.auth_method}'. Keeping '{self.model_name}'.")
            elif self.model_name != new_model:
                self.model_name = new_model
                needs_reinit = True; applied_some_settings = True

        if "default_seed" in settings and self.default_seed != int(settings["default_seed"]):
            self.default_seed = int(settings["default_seed"])
            applied_some_settings = True
        if "default_guidance_scale" in settings and self.default_guidance_scale != float(settings["default_guidance_scale"]):
            self.default_guidance_scale = float(settings["default_guidance_scale"])
            applied_some_settings = True

        if needs_reinit:
            try:
                self._initialize_client()
            except Exception as e:
                ASCIIColors.error(f"Failed to re-initialize client with new settings: {e}")
                return False
        
        return applied_some_settings

    def list_models(self) -> list:
        """Lists available Imagen/Gemini models in a standardized format."""
        models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        return [
            {
                'model_name': name,
                'display_name': f"Google ({name})",
                'description': f"Google's Imagen/Gemini model, version {name}",
                'owned_by': 'Google'
            } for name in models
        ]