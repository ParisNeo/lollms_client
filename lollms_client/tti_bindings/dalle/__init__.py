# lollms_client/tti_bindings/dalle/__init__.py
import requests
import base64
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from typing import Optional, List, Dict, Any, Union
from ascii_colors import trace_exception, ASCIIColors
import json # For json.JSONDecodeError in error handling, and general JSON operations
import os # Added for environment variable access

# Defines the binding name for the manager
BindingName = "DalleTTIBinding_Impl"

# DALL-E specific constants
DALLE_API_HOST = "https://api.openai.com/v1"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY" # Environment variable name

# Supported models and their properties
DALLE_MODELS = {
    "dall-e-2": {
        "sizes": ["256x256", "512x512", "1024x1024"],
        "default_size": "1024x1024",
        "supports_quality": False,
        "supports_style": False,
        "max_prompt_length": 1000 # Characters
    },
    "dall-e-3": {
        "sizes": ["1024x1024", "1792x1024", "1024x1792"],
        "default_size": "1024x1024",
        "qualities": ["standard", "hd"],
        "default_quality": "standard",
        "styles": ["vivid", "natural"],
        "default_style": "vivid",
        "supports_quality": True,
        "supports_style": True,
        "max_prompt_length": 4000 # Characters
    }
}

class DalleTTIBinding_Impl(LollmsTTIBinding):
    """
    Concrete implementation of LollmsTTIBinding for OpenAI's DALL-E API.
    """

    def __init__(self,
                 api_key: Optional[str] = None, # Can be None to check env var
                 model_name: str = "dall-e-3", # Default to DALL-E 3
                 default_size: Optional[str] = None, # e.g. "1024x1024"
                 default_quality: Optional[str] = None, # "standard" or "hd" (DALL-E 3)
                 default_style: Optional[str] = None, # "vivid" or "natural" (DALL-E 3)
                 host_address: str = DALLE_API_HOST, # OpenAI API host
                 verify_ssl_certificate: bool = True,
                 **kwargs # To catch any other lollms_client specific params like service_key/client_id
                 ):
        """
        Initialize the DALL-E TTI binding.

        Args:
            api_key (Optional[str]): OpenAI API key. If None or empty, attempts to read
                                     from the OPENAI_API_KEY environment variable.
            model_name (str): Name of the DALL-E model to use (e.g., "dall-e-3", "dall-e-2").
            default_size (Optional[str]): Default image size (e.g., "1024x1024").
                                          If None, uses model's default.
            default_quality (Optional[str]): Default image quality for DALL-E 3 ("standard", "hd").
                                             If None, uses model's default if applicable.
            default_style (Optional[str]): Default image style for DALL-E 3 ("vivid", "natural").
                                           If None, uses model's default if applicable.
            host_address (str): The API host address. Defaults to OpenAI's public API.
            verify_ssl_certificate (bool): Whether to verify SSL certificates.
            **kwargs: Catches other potential parameters like 'service_key' or 'client_id'.
        """
        super().__init__(binding_name="dalle")

        resolved_api_key = api_key
        if not resolved_api_key:
            ASCIIColors.info(f"API key not provided directly, checking environment variable '{OPENAI_API_KEY_ENV_VAR}'...")
            resolved_api_key = os.environ.get(OPENAI_API_KEY_ENV_VAR)

        if not resolved_api_key:
            raise ValueError(f"OpenAI API key is required. Provide it directly or set the '{OPENAI_API_KEY_ENV_VAR}' environment variable.")
        
        self.api_key = resolved_api_key
        self.host_address = host_address
        self.verify_ssl_certificate = verify_ssl_certificate

        if model_name not in DALLE_MODELS:
            raise ValueError(f"Unsupported DALL-E model: {model_name}. Supported models: {list(DALLE_MODELS.keys())}")
        self.model_name = model_name
        
        model_props = DALLE_MODELS[self.model_name]

        # Set defaults from model_props, overridden by user-provided defaults
        self.current_size = default_size or model_props["default_size"]
        if self.current_size not in model_props["sizes"]:
            raise ValueError(f"Unsupported size '{self.current_size}' for model '{self.model_name}'. Supported sizes: {model_props['sizes']}")

        if model_props["supports_quality"]:
            self.current_quality = default_quality or model_props["default_quality"]
            if self.current_quality not in model_props["qualities"]:
                raise ValueError(f"Unsupported quality '{self.current_quality}' for model '{self.model_name}'. Supported qualities: {model_props['qualities']}")
        else:
            self.current_quality = None # Explicitly None if not supported

        if model_props["supports_style"]:
            self.current_style = default_style or model_props["default_style"]
            if self.current_style not in model_props["styles"]:
                raise ValueError(f"Unsupported style '{self.current_style}' for model '{self.model_name}'. Supported styles: {model_props['styles']}")
        else:
            self.current_style = None # Explicitly None if not supported
        
        # For potential lollms client specific features, if `service_key` is passed as `client_id`
        self.client_id = kwargs.get("service_key", kwargs.get("client_id", "dalle_client_user"))


    def _get_model_properties(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Helper to get properties for a given model name, or the instance's current model."""
        return DALLE_MODELS.get(model_name or self.model_name, {})

    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 1024, # Default width
                       height: int = 1024, # Default height
                       **kwargs) -> bytes:
        """
        Generates image data using the DALL-E API.

        Args:
            prompt (str): The positive text prompt.
            negative_prompt (Optional[str]): The negative prompt. For DALL-E 3, this is
                                             appended to the main prompt. For DALL-E 2, it's ignored.
            width (int): Image width.
            height (int): Image height.
            **kwargs: Additional parameters:
                      - model (str): Override the instance's default model for this call.
                      - quality (str): Override quality ("standard", "hd" for DALL-E 3).
                      - style (str): Override style ("vivid", "natural" for DALL-E 3).
                      - n (int): Number of images to generate (OpenAI supports >1, but this binding returns one). Default 1.
                      - user (str): A unique identifier for your end-user (OpenAI abuse monitoring).
        Returns:
            bytes: The generated image data (PNG format from DALL-E).

        Raises:
            Exception: If the request fails or image generation fails on the server.
        """
        model_override = kwargs.get("model")
        active_model_name = model_override if model_override else self.model_name
        
        model_props = self._get_model_properties(active_model_name)
        if not model_props:
            raise ValueError(f"Model {active_model_name} properties not found. Supported: {list(DALLE_MODELS.keys())}")

        # Format size string and validate against the active model for this generation
        size_str = f"{width}x{height}"
        if size_str not in model_props["sizes"]:
            raise ValueError(f"Unsupported size '{size_str}' for model '{active_model_name}'. Supported sizes: {model_props['sizes']}. Adjust width/height for this model.")

        # Handle prompt and negative prompt based on the active model
        final_prompt = prompt
        if active_model_name == "dall-e-3" and negative_prompt:
            final_prompt = f"{prompt}. Avoid: {negative_prompt}."
            ASCIIColors.info(f"DALL-E 3: Appended negative prompt. Final prompt: '{final_prompt[:100]}...'")
        elif active_model_name == "dall-e-2" and negative_prompt:
            ASCIIColors.warning("DALL-E 2 does not support negative_prompt. It will be ignored.")
        
        # Truncate prompt if too long for the active model
        max_len = model_props.get("max_prompt_length", 4000)
        if len(final_prompt) > max_len:
            ASCIIColors.warning(f"Prompt for {active_model_name} is too long ({len(final_prompt)} chars). Truncating to {max_len} characters.")
            final_prompt = final_prompt[:max_len]

        endpoint = f"{self.host_address}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": active_model_name,
            "prompt": final_prompt,
            "n": kwargs.get("n", 1), # This binding expects to return one image
            "size": size_str,
            "response_format": "b64_json" # Request base64 encoded image
        }
        
        # Add model-specific parameters (quality, style)
        # Use kwargs if provided, otherwise instance defaults, but only if the active model supports them.
        if model_props["supports_quality"]:
            payload["quality"] = kwargs.get("quality", self.current_quality)
        if model_props["supports_style"]:
            payload["style"] = kwargs.get("style", self.current_style)
        
        if "user" in kwargs: # Pass user param if provided for moderation
            payload["user"] = kwargs["user"]

        try:
            response = requests.post(endpoint, json=payload, headers=headers, verify=self.verify_ssl_certificate)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_json = response.json()
            
            if not response_json.get("data") or not response_json["data"][0].get("b64_json"):
                raise Exception("Server did not return image data in expected b64_json format.")

            img_base64 = response_json["data"][0]["b64_json"]
            img_bytes = base64.b64decode(img_base64)
            return img_bytes

        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown server error"
            if e.response is not None:
                try:
                    err_json = e.response.json()
                    if "error" in err_json and isinstance(err_json["error"], dict) and "message" in err_json["error"]:
                        error_detail = err_json["error"]["message"]
                    elif "detail" in err_json: # Fallback for other error structures
                         error_detail = err_json["detail"]
                    else: # If no specific error message, use raw text (limited)
                        error_detail = e.response.text[:500] 
                except (json.JSONDecodeError, requests.exceptions.JSONDecodeError): # If response is not JSON
                    error_detail = e.response.text[:500] 
                trace_exception(e)
                raise Exception(f"HTTP request failed: {e.response.status_code} {e.response.reason} - Detail: {error_detail}") from e
            else: # HTTPError without a response object (less common)
                trace_exception(e)
                raise Exception(f"HTTP request failed without a response body: {e}") from e
        except requests.exceptions.RequestException as e: # Catches other network errors (DNS, ConnectionError, etc.)
            trace_exception(e)
            raise Exception(f"Request failed due to network issue: {e}") from e
        except Exception as e: # Catches other errors (e.g., base64 decoding, unexpected issues)
            trace_exception(e)
            raise Exception(f"Image generation process failed: {e}") from e


    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """
        Lists available DALL-E models supported by this binding.
        `client_id` from kwargs is ignored as DALL-E auth is via API key.
        """
        services = []
        for model_name, props in DALLE_MODELS.items():
            caption = f"OpenAI {model_name.upper()}"
            help_text = f"Size options: {', '.join(props['sizes'])}. "
            if props["supports_quality"]:
                help_text += f"Qualities: {', '.join(props['qualities'])}. "
            if props["supports_style"]:
                help_text += f"Styles: {', '.join(props['styles'])}. "
            services.append({
                "name": model_name,
                "caption": caption,
                "help": help_text.strip()
            })
        return services

    def get_settings(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieves the current configurable default settings for the DALL-E binding.
        `client_id` from kwargs is ignored. Returns settings in a ConfigTemplate-like format.
        """
        model_props = self._get_model_properties(self.model_name) # Settings relative to current default model
        
        settings = [
            {
                "name": "model_name",
                "type": "str",
                "value": self.model_name,
                "description": "Default DALL-E model for generation.",
                "options": list(DALLE_MODELS.keys()),
                "category": "Model Configuration"
            },
            {
                "name": "current_size",
                "type": "str",
                "value": self.current_size,
                "description": "Default image size (e.g., 1024x1024). Format: widthxheight.",
                "options": model_props.get("sizes", []), # Options relevant to the current default model
                "category": "Image Generation Defaults"
            }
        ]

        if model_props.get("supports_quality", False):
            settings.append({
                "name": "current_quality",
                "type": "str",
                "value": self.current_quality,
                "description": "Default image quality (e.g., 'standard', 'hd' for DALL-E 3).",
                "options": model_props.get("qualities", []),
                "category": "Image Generation Defaults"
            })
        
        if model_props.get("supports_style", False):
            settings.append({
                "name": "current_style",
                "type": "str",
                "value": self.current_style,
                "description": "Default image style (e.g., 'vivid', 'natural' for DALL-E 3).",
                "options": model_props.get("styles", []),
                "category": "Image Generation Defaults"
            })
        
        settings.append({
                "name": "api_key_status", 
                "type": "str",
                "value": "Set (loaded)" if self.api_key else "Not Set", # Indicate if API key is present
                "description": f"OpenAI API Key status (set at initialization or via '{OPENAI_API_KEY_ENV_VAR}', not changeable here).",
                "category": "Authentication",
                "read_only": True # Custom attribute indicating it's informational
            })

        return settings


    def set_settings(self, settings: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> bool:
        """
        Applies new default settings to the DALL-E binding instance.
        `client_id` from kwargs is ignored.

        Args:
            settings (Union[Dict[str, Any], List[Dict[str, Any]]]): 
                New settings to apply.
                Can be a flat dict: `{"model_name": "dall-e-2", "current_size": "512x512"}`
                Or a list of dicts (ConfigTemplate format):
                `[{"name": "model_name", "value": "dall-e-2"}, ...]`

        Returns:
            bool: True if at least one setting was successfully applied, False otherwise.
        """
        applied_some_settings = False
        original_model_name = self.model_name # To detect if model changes
        
        # Normalize settings input to a flat dictionary
        if isinstance(settings, list):
            parsed_settings = {}
            for item in settings:
                if isinstance(item, dict) and "name" in item and "value" in item:
                    if item["name"] == "api_key_status": # This is read-only
                        continue
                    parsed_settings[item["name"]] = item["value"]
            settings_dict = parsed_settings
        elif isinstance(settings, dict):
            settings_dict = settings
        else:
            ASCIIColors.error("Invalid settings format. Expected a dictionary or list of dictionaries.")
            return False

        try:
            # Phase 1: Apply model_name change if present, as it affects other settings' validity
            if "model_name" in settings_dict:
                new_model_name = settings_dict["model_name"]
                if new_model_name not in DALLE_MODELS:
                    ASCIIColors.warning(f"Invalid model_name '{new_model_name}' provided in settings. Keeping current model '{self.model_name}'.")
                elif self.model_name != new_model_name:
                    self.model_name = new_model_name
                    ASCIIColors.info(f"Default model changed to: {self.model_name}")
                    applied_some_settings = True # Mark that model name was processed

            # Phase 2: If model changed, or for initial setup, adjust dependent settings to be consistent
            # Run this phase if model_name was specifically in settings_dict and changed, OR if this is the first time settings are processed.
            # The 'applied_some_settings' flag after model_name processing indicates a change.
            if "model_name" in settings_dict and applied_some_settings : 
                new_model_props = self._get_model_properties(self.model_name)

                # Update current_size if invalid for new model or if model changed
                if self.current_size not in new_model_props["sizes"]:
                    old_val = self.current_size
                    self.current_size = new_model_props["default_size"]
                    if old_val != self.current_size: ASCIIColors.info(f"Default size reset to '{self.current_size}' for model '{self.model_name}'.")
                
                # Update current_quality
                if new_model_props["supports_quality"]:
                    if self.current_quality not in new_model_props.get("qualities", []):
                        old_val = self.current_quality
                        self.current_quality = new_model_props["default_quality"]
                        if old_val != self.current_quality: ASCIIColors.info(f"Default quality reset to '{self.current_quality}' for model '{self.model_name}'.")
                elif self.current_quality is not None: # New model doesn't support quality
                     self.current_quality = None
                     ASCIIColors.info(f"Quality setting removed as model '{self.model_name}' does not support it.")
                
                # Update current_style
                if new_model_props["supports_style"]:
                    if self.current_style not in new_model_props.get("styles", []):
                        old_val = self.current_style
                        self.current_style = new_model_props["default_style"]
                        if old_val != self.current_style: ASCIIColors.info(f"Default style reset to '{self.current_style}' for model '{self.model_name}'.")
                elif self.current_style is not None: # New model doesn't support style
                    self.current_style = None
                    ASCIIColors.info(f"Style setting removed as model '{self.model_name}' does not support it.")

            # Phase 3: Apply other specific settings from input, validating against the (potentially new) model
            current_model_props = self._get_model_properties(self.model_name) # Re-fetch props if model changed

            if "current_size" in settings_dict:
                new_size = settings_dict["current_size"]
                if new_size not in current_model_props["sizes"]:
                    ASCIIColors.warning(f"Invalid size '{new_size}' for model '{self.model_name}'. Keeping '{self.current_size}'. Supported: {current_model_props['sizes']}")
                elif self.current_size != new_size:
                    self.current_size = new_size
                    ASCIIColors.info(f"Default size set to: {self.current_size}")
                    applied_some_settings = True
            
            if "current_quality" in settings_dict:
                if current_model_props["supports_quality"]:
                    new_quality = settings_dict["current_quality"]
                    if new_quality not in current_model_props["qualities"]:
                        ASCIIColors.warning(f"Invalid quality '{new_quality}' for model '{self.model_name}'. Keeping '{self.current_quality}'. Supported: {current_model_props['qualities']}")
                    elif self.current_quality != new_quality:
                        self.current_quality = new_quality
                        ASCIIColors.info(f"Default quality set to: {self.current_quality}")
                        applied_some_settings = True
                elif "current_quality" in settings_dict: # Only warn if user explicitly tried to set it
                     ASCIIColors.warning(f"Model '{self.model_name}' does not support quality. Ignoring 'current_quality' setting.")

            if "current_style" in settings_dict:
                if current_model_props["supports_style"]:
                    new_style = settings_dict["current_style"]
                    if new_style not in current_model_props["styles"]:
                        ASCIIColors.warning(f"Invalid style '{new_style}' for model '{self.model_name}'. Keeping '{self.current_style}'. Supported: {current_model_props['styles']}")
                    elif self.current_style != new_style:
                        self.current_style = new_style
                        ASCIIColors.info(f"Default style set to: {self.current_style}")
                        applied_some_settings = True
                elif "current_style" in settings_dict: # Only warn if user explicitly tried to set it
                     ASCIIColors.warning(f"Model '{self.model_name}' does not support style. Ignoring 'current_style' setting.")

            if "api_key" in settings_dict: # Should not be settable here
                ASCIIColors.warning("API key cannot be changed after initialization via set_settings. This setting was ignored.")

            return applied_some_settings
        
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"Failed to apply settings due to an unexpected error: {e}")
            return False