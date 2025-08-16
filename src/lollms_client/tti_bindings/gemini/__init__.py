# lollms_client/tti_bindings/gemini/__init__.py
import sys
from typing import Optional, List, Dict, Any, Union

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import json

try:
    import pipmaster as pm
    # google-cloud-aiplatform is the main dependency for Vertex AI
    pm.ensure_packages(['google-cloud-aiplatform', 'Pillow'])
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from google.api_core import exceptions as google_exceptions
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    _gemini_installation_error = e

# Defines the binding name for the manager
BindingName = "GeminiTTIBinding_Impl"

# Known Imagen models on Vertex AI
IMAGEN_MODELS = ["imagegeneration@006", "imagegeneration@005", "imagegeneration@002"]

class GeminiTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for Google's Imagen models via Vertex AI.
    """
    DEFAULT_CONFIG = {
        "project_id": None,
        "location": "us-central1",
        "model_name": IMAGEN_MODELS[0],
        "seed": -1, # -1 for random
        "guidance_scale": 7.5,
        "number_of_images": 1
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the Gemini (Vertex AI Imagen) TTI binding.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Overrides DEFAULT_CONFIG.
            **kwargs: Catches other potential parameters.
        """
        super().__init__(binding_name="gemini")

        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini (Vertex AI) binding dependencies are not met. "
                "Please ensure 'google-cloud-aiplatform' is installed. "
                f"Error: {_gemini_installation_error}"
            )
        
        self.config = {**self.DEFAULT_CONFIG, **(config or {}), **kwargs}
        self.model: Optional[ImageGenerationModel] = None

        self._initialize_client()

    def _initialize_client(self):
        """Initializes the Vertex AI client and loads the model."""
        project_id = self.config.get("project_id")
        location = self.config.get("location")
        model_name = self.config.get("model_name")

        if not project_id:
            raise ValueError("Google Cloud 'project_id' is required for the Gemini (Vertex AI) binding.")
        
        ASCIIColors.info("Initializing Vertex AI client...")
        try:
            vertexai.init(project=project_id, location=location)
            self.model = ImageGenerationModel.from_pretrained(model_name)
            ASCIIColors.green(f"Vertex AI initialized successfully. Loaded model: {model_name}")
        except google_exceptions.PermissionDenied as e:
            trace_exception(e)
            raise Exception(
                "Authentication failed. Ensure you have run 'gcloud auth application-default login' "
                "and that the Vertex AI API is enabled for your project."
            ) from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Failed to initialize Vertex AI client: {e}") from e

    def _validate_dimensions(self, width: int, height: int) -> None:
        """Validates image dimensions against Imagen 2 constraints."""
        if not (256 <= width <= 1536 and width % 64 == 0):
            raise ValueError(f"Invalid width: {width}. Must be between 256 and 1536 and a multiple of 64.")
        if not (256 <= height <= 1536 and height % 64 == 0):
            raise ValueError(f"Invalid height: {height}. Must be between 256 and 1536 and a multiple of 64.")
        if width * height > 1536 * 1536: # Max pixels might be more constrained, 1536*1536 is a safe upper bound.
             raise ValueError(f"Invalid dimensions: {width}x{height}. The total number of pixels cannot exceed 1536*1536.")

    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 1024,
                       height: int = 1024,
                       **kwargs) -> bytes:
        """
        Generates image data using the Vertex AI Imagen model.

        Args:
            prompt (str): The positive text prompt.
            negative_prompt (Optional[str]): The negative prompt.
            width (int): Image width. Must be 256-1536 and a multiple of 64.
            height (int): Image height. Must be 256-1536 and a multiple of 64.
            **kwargs: Additional parameters:
                      - seed (int)
                      - guidance_scale (float)
        Returns:
            bytes: The generated image data (PNG format).
        Raises:
            Exception: If the request fails or image generation fails.
        """
        if not self.model:
            raise RuntimeError("Vertex AI model is not loaded. Cannot generate image.")

        self._validate_dimensions(width, height)

        seed = kwargs.get("seed", self.config["seed"])
        guidance_scale = kwargs.get("guidance_scale", self.config["guidance_scale"])
        
        # Use -1 for random seed, otherwise pass the integer value.
        gen_seed = seed if seed != -1 else None

        gen_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "number_of_images": 1, # This binding returns one image
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
        }
        if gen_seed is not None:
            gen_params["seed"] = gen_seed

        ASCIIColors.info(f"Generating image with prompt: '{prompt[:100]}...'")
        ASCIIColors.debug(f"Imagen generation parameters: {gen_params}")

        try:
            response = self.model.generate_images(**gen_params)
            
            if not response.images:
                raise Exception("Image generation resulted in no images. This may be due to safety filters.")

            img_bytes = response.images[0]._image_bytes
            return img_bytes

        except google_exceptions.InvalidArgument as e:
            trace_exception(e)
            raise ValueError(f"Invalid argument sent to Vertex AI API: {e.message}") from e
        except google_exceptions.GoogleAPICallError as e:
            trace_exception(e)
            raise Exception(f"A Google API call error occurred: {e.message}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"Imagen image generation failed: {e}") from e

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """
        Lists available Imagen models supported by this binding.
        """
        services = []
        for model_name in IMAGEN_MODELS:
            services.append({
                "name": model_name,
                "caption": f"Google Imagen 2 ({model_name})",
                "help": "High-quality text-to-image model from Google, available on Vertex AI."
            })
        return services

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieves the current configurable settings for the binding.
        """
        return [
            {"name": "project_id", "type": "str", "value": self.config["project_id"], "description": "Your Google Cloud project ID."},
            {"name": "location", "type": "str", "value": self.config["location"], "description": "Google Cloud region for the project (e.g., 'us-central1')."},
            {"name": "model_name", "type": "str", "value": self.config["model_name"], "description": "The Imagen model version to use.", "options": IMAGEN_MODELS},
            {"name": "seed", "type": "int", "value": self.config["seed"], "description": "Default seed for generation (-1 for random)."},
            {"name": "guidance_scale", "type": "float", "value": self.config["guidance_scale"], "description": "Default guidance scale (CFG). Higher values follow the prompt more strictly."},
        ]

    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        """
        Applies new settings to the binding. Re-initializes the client if needed.
        """
        if isinstance(settings, list):
            parsed_settings = {item["name"]: item["value"] for item in settings if "name" in item and "value" in item}
        elif isinstance(settings, dict):
            parsed_settings = settings
        else:
            ASCIIColors.error("Invalid settings format. Expected a dictionary or list of dictionaries.")
            return False

        needs_reinit = False
        for key, value in parsed_settings.items():
            if key in self.config and self.config[key] != value:
                self.config[key] = value
                ASCIIColors.info(f"Setting '{key}' changed to: {value}")
                if key in ["project_id", "location", "model_name"]:
                    needs_reinit = True
        
        if needs_reinit:
            try:
                self._initialize_client()
                ASCIIColors.green("Vertex AI client re-initialized successfully with new settings.")
            except Exception as e:
                ASCIIColors.error(f"Failed to re-initialize client with new settings: {e}")
                # Optionally, revert to old config here to maintain a working state
                return False
        
        return True