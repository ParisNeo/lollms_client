# lollms_client/tti_bindings/gemini/__init__.py
import sys
from typing import Optional, List, Dict, Any, Union
import os

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import math

# --- SDK & Dependency Management ---
try:
    import pipmaster as pm
    # Ensure both potential SDKs and Pillow are available
    pm.ensure_packages(['google-cloud-aiplatform', 'google-generativeai', 'Pillow'])
except ImportError:
    pass # pipmaster is optional

# Attempt to import Vertex AI (google-cloud-aiplatform)
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from google.api_core import exceptions as google_exceptions
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
IMAGEN_VERTEX_MODELS = ["imagegeneration@006", "imagegeneration@005", "imagegeneration@002"]
IMAGEN_GEMINI_API_MODELS = ["imagen-3", "gemini-1.5-flash-preview-0514"] # Short names are often aliases
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"

class GeminiTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for Google's Imagen models.
    Supports both Vertex AI (project_id) and Gemini API (api_key) authentication.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Gemini (Vertex AI / API) TTI binding.

        Args:
            **kwargs: Configuration parameters.
                      - auth_method (str): "vertex_ai" or "api_key". (Required)
                      - project_id (str): Google Cloud project ID (for vertex_ai).
                      - location (str): Google Cloud region (for vertex_ai).
                      - service_key (str): Gemini API Key (for api_key).
                      - model_name (str): The Imagen model to use.
                      - default_seed (int): Default seed for generation (-1 for random).
                      - default_guidance_scale (float): Default guidance scale (CFG).
        """
        super().__init__(binding_name="gemini")

        # Core settings
        self.auth_method = kwargs.get("auth_method", "vertex_ai") # Default to vertex_ai for backward compatibility

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
        
        # The actual client/model instance
        self.client: Optional[Any] = None

        # --- Validation and Initialization ---
        if self.auth_method == "vertex_ai":
            if not VERTEX_AI_AVAILABLE:
                raise ImportError("Vertex AI authentication selected, but 'google-cloud-aiplatform' is not installed.")
            if not self.project_id:
                raise ValueError("For 'vertex_ai' auth, a Google Cloud 'project_id' is required.")
            if not self.model_name:
                self.model_name = IMAGEN_VERTEX_MODELS[0]
        elif self.auth_method == "api_key":
            if not GEMINI_API_AVAILABLE:
                raise ImportError("API Key authentication selected, but 'google-generativeai' is not installed.")
            
            # Resolve API key from kwargs or environment variable
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
                # For the genai SDK, the "client" is the configured module itself,
                # and we specify the model per-call. Let's store the genai module.
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
        ratios = {"1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4}
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
        """
        Generates image data using the configured Google Imagen model.
        """
        if not self.client:
            raise RuntimeError("Google client is not initialized. Cannot generate image.")

        # Use overrides from kwargs, otherwise instance defaults
        seed = kwargs.get("seed", self.default_seed)
        guidance_scale = kwargs.get("guidance_scale", self.default_guidance_scale)
        gen_seed = seed if seed != -1 else None

        final_prompt = prompt
        if negative_prompt:
            final_prompt = f"{prompt}. Do not include: {negative_prompt}."

        ASCIIColors.info(f"Generating image with prompt: '{final_prompt[:100]}...'")

        try:
            if self.auth_method == "vertex_ai":
                self._validate_dimensions_vertex(width, height)
                gen_params = {
                    "prompt": final_prompt,
                    "number_of_images": 1,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale,
                }
                if gen_seed is not None:
                    gen_params["seed"] = gen_seed
                
                ASCIIColors.debug(f"Vertex AI generation parameters: {gen_params}")
                response = self.client.generate_images(**gen_params)
                
                if not response.images:
                    raise Exception("Image generation resulted in no images (Vertex AI). Check safety filters.")
                
                return response.images[0]._image_bytes

            elif self.auth_method == "api_key":
                aspect_ratio = self._get_aspect_ratio_for_api(width, height)
                gen_params = {
                    "model": self.model_name,
                    "prompt": final_prompt,
                    "number_of_images": 1,
                    "aspect_ratio": aspect_ratio
                    # Note: seed and guidance_scale are not standard in this simpler API call
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

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """Lists available Imagen models for the current auth method."""
        models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        service_name = "Vertex AI" if self.auth_method == "vertex_ai" else "Gemini API"
        return [
            {
                "name": name,
                "caption": f"Google Imagen ({name}) via {service_name}",
                "help": "High-quality text-to-image model from Google."
            } for name in models
        ]

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieves the current configurable settings for the binding."""
        settings = [
            {"name": "auth_method", "type": "str", "value": self.auth_method, "description": "Authentication method to use.", "options": ["vertex_ai", "api_key"], "category": "Authentication"},
        ]
        if self.auth_method == "vertex_ai":
            settings.extend([
                {"name": "project_id", "type": "str", "value": self.project_id, "description": "Your Google Cloud project ID.", "category": "Authentication"},
                {"name": "location", "type": "str", "value": self.location, "description": "Google Cloud region (e.g., 'us-central1').", "category": "Authentication"},
                {"name": "model_name", "type": "str", "value": self.model_name, "description": "Default Imagen model for generation.", "options": IMAGEN_VERTEX_MODELS, "category": "Model Configuration"},
            ])
        elif self.auth_method == "api_key":
            settings.extend([
                {"name": "api_key_status", "type": "str", "value": "Set" if self.gemini_api_key else "Not Set", "description": f"Gemini API Key status (set at initialization via service_key or '{GEMINI_API_KEY_ENV_VAR}').", "category": "Authentication", "read_only": True},
                {"name": "model_name", "type": "str", "value": self.model_name, "description": "Default Imagen model for generation.", "options": IMAGEN_GEMINI_API_MODELS, "category": "Model Configuration"},
            ])
        
        settings.extend([
            {"name": "default_seed", "type": "int", "value": self.default_seed, "description": "Default seed (-1 for random).", "category": "Image Generation Defaults"},
            {"name": "default_guidance_scale", "type": "float", "value": self.default_guidance_scale, "description": "Default guidance scale (CFG). (Vertex AI only)", "category": "Image Generation Defaults"},
        ])
        return settings

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        """Applies new settings. Re-initializes the client if core settings change."""
        applied_some_settings = False
        settings_dict = {item["name"]: item["value"] for item in settings} if isinstance(settings, list) else settings

        needs_reinit = False
        
        # Phase 1: Check for auth_method or core credential changes
        if "auth_method" in settings_dict and self.auth_method != settings_dict["auth_method"]:
            self.auth_method = settings_dict["auth_method"]
            ASCIIColors.info(f"Authentication method changed to: {self.auth_method}")
            # Reset model to a valid default for the new method
            if self.auth_method == "vertex_ai":
                self.model_name = IMAGEN_VERTEX_MODELS[0]
            else:
                self.model_name = IMAGEN_GEMINI_API_MODELS[0]
            ASCIIColors.info(f"Model name reset to default for new auth method: {self.model_name}")
            needs_reinit = True
            applied_some_settings = True
        
        if self.auth_method == "vertex_ai":
            if "project_id" in settings_dict and self.project_id != settings_dict["project_id"]:
                self.project_id = settings_dict["project_id"]
                needs_reinit = True; applied_some_settings = True
            if "location" in settings_dict and self.location != settings_dict["location"]:
                self.location = settings_dict["location"]
                needs_reinit = True; applied_some_settings = True
        # API key is not settable after init, so we don't check for it here.

        # Phase 2: Apply other settings
        current_models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        if "model_name" in settings_dict:
            new_model = settings_dict["model_name"]
            if new_model not in current_models:
                ASCIIColors.warning(f"Invalid model '{new_model}' for auth method '{self.auth_method}'. Keeping '{self.model_name}'.")
            elif self.model_name != new_model:
                self.model_name = new_model
                needs_reinit = True; applied_some_settings = True

        if "default_seed" in settings_dict and self.default_seed != int(settings_dict["default_seed"]):
            self.default_seed = int(settings_dict["default_seed"])
            applied_some_settings = True
        if "default_guidance_scale" in settings_dict and self.default_guidance_scale != float(settings_dict["default_guidance_scale"]):
            self.default_guidance_scale = float(settings_dict["default_guidance_scale"])
            applied_some_settings = True

        # Phase 3: Re-initialize if needed
        if needs_reinit:
            try:
                self._initialize_client()
            except Exception as e:
                ASCIIColors.error(f"Failed to re-initialize client with new settings: {e}")
                return False
        
        return applied_some_settings

    def listModels(self) -> list:
        """Lists available Imagen models in a standardized format."""
        models = IMAGEN_VERTEX_MODELS if self.auth_method == "vertex_ai" else IMAGEN_GEMINI_API_MODELS
        return [
            {
                'model_name': name,
                'display_name': f"Imagen ({name})",
                'description': f"Google's Imagen model, version {name}",
                'owned_by': 'Google'
            } for name in models
        ]