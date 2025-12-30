import requests
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from lollms_client.lollms_types import MSG_TYPE
from ascii_colors import ASCIIColors, trace_exception

BindingName = "OpenRouterTTIBinding"

class OpenRouterTTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
            
        # The manager passes binding_name in kwargs. To avoid "multiple values for keyword argument",
        # we pop it and use the local BindingName constant or the passed value.
        binding_name = kwargs.pop('binding_name', BindingName)
        super().__init__(binding_name=binding_name, **kwargs)
        
        self.service_key = kwargs.get("service_key", "")
        # Default to a known stable image-capable model
        self.model_name = kwargs.get("model_name", "google/gemini-2.0-flash-exp:free")
        self.host_address = "https://openrouter.ai/api/v1"
        self.config = kwargs

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Helper to map width/height to OpenRouter supported aspect ratios."""
        ratio = width / height
        if abs(ratio - 1.0) < 0.1: return "1:1"
        if abs(ratio - 0.66) < 0.1: return "2:3"
        if abs(ratio - 1.5) < 0.1: return "3:2"
        if abs(ratio - 0.75) < 0.1: return "3:4"
        if abs(ratio - 1.33) < 0.1: return "4:3"
        if abs(ratio - 0.8) < 0.1: return "4:5"
        if abs(ratio - 1.25) < 0.1: return "5:4"
        if abs(ratio - 0.56) < 0.1: return "9:16"
        if abs(ratio - 1.77) < 0.1: return "16:9"
        if abs(ratio - 2.33) < 0.1: return "21:9"
        return "1:1"

    def generate_image(self, 
                       prompt: str, 
                       negative_prompt: str = "", 
                       width: int = 1024, 
                       height: int = 1024, 
                       **kwargs) -> bytes:
        """
        Generates an image using Open Router's /chat/completions endpoint with modalities.
        """
        model = kwargs.get("model_name", self.model_name)
        
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ParisNeo/lollms_client",
            "X-Title": "LoLLMS Client"
        }
        
        # Open Router specific payload using Chat Completions
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt if not negative_prompt else f"{prompt}\n\nNegative prompt: {negative_prompt}"
                }
            ],
            "modalities": ["image", "text"],
            "stream": False
        }

        # Add image configuration if relevant
        image_config = {}
        aspect_ratio = kwargs.get("aspect_ratio")
        if not aspect_ratio:
            aspect_ratio = self._get_aspect_ratio(width, height)
        
        image_config["aspect_ratio"] = aspect_ratio
        
        if "image_size" in kwargs:
            image_config["image_size"] = kwargs["image_size"]
        
        payload["image_config"] = image_config

        try:
            ASCIIColors.info(f"Open Router TTI Request: Model={model}, Aspect Ratio={aspect_ratio}")
            response = requests.post(
                f"{self.host_address}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("choices"):
                message = result["choices"][0].get("message", {})
                images = message.get("images", [])
                
                if images:
                    image_url = images[0]["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",")[1]
                        return base64.b64decode(base64_str)
                    else:
                        img_res = requests.get(image_url)
                        img_res.raise_for_status()
                        return img_res.content
                    
            raise ValueError(f"No image found in Open Router response: {result}")

        except Exception as e:
            ASCIIColors.error(f"Open Router TTI Error: {e}")
            trace_exception(e)
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists available models from Open Router and filters for TTI capabilities.
        """
        try:
            response = requests.get(f"{self.host_address}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            
            filtered_models = []
            # Curated list of keywords that identify Image Generation models on OpenRouter
            tti_keywords = ["flux", "dall-e", "imagen", "stable-diffusion", "midjourney", "riverflow"]
            
            for m in models:
                m_id = m.get("id", "").lower()
                description = m.get("description", "").lower()
                # OpenRouter sometimes provides output_modalities in the architecture block
                modality = m.get("architecture", {}).get("modality", "text")
                
                # Check if it's explicitly an image model or matches TTI keywords
                is_tti = "image" in modality or \
                         any(kw in m_id for kw in tti_keywords) or \
                         "generate images" in description or \
                         "text-to-image" in description

                if is_tti:
                    filtered_models.append({
                        "model_name": m.get("id"),
                        "display_name": m.get("name"),
                        "description": m.get("description", "Image generation model")
                    })
            
            # Sort by name for easier navigation
            filtered_models = sorted(filtered_models, key=lambda x: x["display_name"])

            # Hardcoded fallbacks if API discovery fails or to ensure core models are present
            if not filtered_models:
                return [
                    {"model_name": "google/gemini-3-pro-image-preview", "display_name": "Gemini 3.0 Pro image"},
                    {"model_name": "black-forest-labs/flux.1-pro", "display_name": "FLUX.1 Pro"},
                    {"model_name": "openai/gpt-5-image", "display_name": "gpt-5-image"}
                ]
                
            return filtered_models
        except Exception as e:
            ASCIIColors.error(f"Failed to list Open Router models: {e}")
            return []

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": "Open Router TTI", "id": "open_router"}]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.config.update(settings)
        if "service_key" in settings:
            self.service_key = settings["service_key"]
        if "model_name" in settings:
            self.model_name = settings["model_name"]
        return True

    def get_credits(self) -> Optional[Dict[str, float]]:
        """
        Get total credits purchased and used for the authenticated user.
        """
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "HTTP-Referer": "https://github.com/ParisNeo/lollms_client",
            "X-Title": "LoLLMS Client"
        }
        
        try:
            response = requests.get(
                f"{self.host_address}/credits",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            return {
                "total_credits": float(data.get("total_credits", 0.0)),
                "total_usage": float(data.get("total_usage", 0.0))
            }
        except Exception as e:
            ASCIIColors.error(f"Open Router Credits Error: {e}")
            trace_exception(e)
            return None

    def edit_image(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], prompt: str, **kwargs) -> bytes:

        model = kwargs.get("model_name", self.model_name)
        negative_prompt = kwargs.get("negative_prompt", "")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        
        headers = {
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ParisNeo/lollms_client",
            "X-Title": "LoLLMS Client"
        }
        
        # Convert images to base64 data URLs
        if not isinstance(images, list):
            images = [images]
        
        image_contents = []
        for img in images:
            if isinstance(img, str):
                # Check if it's already a data URL
                if img.startswith('data:image'):
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                # Check if it's a regular URL
                elif img.startswith(('http://', 'https://')):
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                # Check if it's base64 data without the data URL prefix
                elif len(img) > 100 and not '/' in img[:50] and not '\\' in img[:50]:
                    # Assume it's raw base64 data
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}
                    })
                else:
                    # Assume it's a local file path
                    try:
                        with open(img, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                            image_contents.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                            })
                    except FileNotFoundError:
                        ASCIIColors.warning(f"Could not find file: {img[:50]}... treating as base64 data")
                        # If file not found, treat as base64
                        image_contents.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"}
                        })
            else:
                # PIL Image object
                from io import BytesIO
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"}
                })
        
        # Build message content with images and text
        content = image_contents + [{
            "type": "text",
            "text": prompt if not negative_prompt else f"{prompt}\n\nNegative prompt: {negative_prompt}"
        }]
        
        # Open Router specific payload using Chat Completions
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "modalities": ["image", "text"],
            "stream": False
        }

        # Add image configuration if relevant
        image_config = {}
        aspect_ratio = kwargs.get("aspect_ratio")
        if not aspect_ratio:
            aspect_ratio = self._get_aspect_ratio(width, height)
        
        image_config["aspect_ratio"] = aspect_ratio
        
        if "image_size" in kwargs:
            image_config["image_size"] = kwargs["image_size"]
        
        payload["image_config"] = image_config

        try:
            ASCIIColors.info(f"Open Router Image Edit Request: Model={model}, Aspect Ratio={aspect_ratio}, Images={len(images)}")
            response = requests.post(
                f"{self.host_address}/chat/completions", 
                headers=headers, 
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("choices"):
                choice = result["choices"][0]
                
                # Check if there's an error in the choice
                if "error" in choice:
                    error_info = choice["error"]
                    error_msg = error_info.get("message", "Unknown error")
                    error_code = error_info.get("code", "unknown")
                    raise ValueError(f"API Error ({error_code}): {error_msg}")
                
                message = choice.get("message", {})
                
                # Check if there are images in the response
                response_images = message.get("images", [])
                if response_images:
                    image_url = response_images[0]["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        base64_str = image_url.split(",")[1]
                        return base64.b64decode(base64_str)
                    else:
                        img_res = requests.get(image_url)
                        img_res.raise_for_status()
                        return img_res.content
                
                # If no images, check if there's a text response (refusal or error)
                content = message.get("content", "")
                if content:
                    raise ValueError(f"Model returned text instead of image: {content}")
                    
            raise ValueError(f"No image found in Open Router response: {result}")

        except ValueError as ve:
            # Re-raise ValueError with the message for user feedback
            ASCIIColors.error(f"Open Router Image Edit Error: {ve}")
            raise
        except Exception as e:
            ASCIIColors.error(f"Open Router Image Edit Error: {e}")
            trace_exception(e)
            return None
        
if __name__ == "__main__":
    import os
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print("Please set OPENROUTER_API_KEY env var.")
    else:
        binding = OpenRouterTTIBinding(service_key=key)
        
        # Test generation
        # img = binding.generate_image("A cute robot painting a picture")
        # if img:
        #     with open("output.png", "wb") as f: f.write(img)
            
        # Test credits
        credits = binding.get_credits()
        if credits:
            print(f"Credits: {credits}")
