from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
import os
import requests
import pipmaster as pm
pm.ensure_packages({"openai":">=2.32.0"})
from openai import OpenAI
import openai
print(openai.__version__)
from lollms_client.lollms_tti_binding import LollmsTTIBinding

BindingName = "OpenAITTIBinding"


class OpenAITTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        binding_name = kwargs.pop("binding_name", BindingName)
        super().__init__(binding_name=binding_name, config=kwargs)

        self.api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        self.config = kwargs

        self.global_params = {
            "model": kwargs.get("model_name", "gpt-image-1"),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "auto"),
        }

    def _resolve_param(self, name: str, kwargs: Dict[str, Any], default: Any) -> Any:
        return kwargs.get(name, self.global_params.get(name, default))

    def _load_image(self, image: Union[str, Path]) -> Any:
        if isinstance(image, Path) or Path(str(image)).exists():
            return Path(image).read_bytes()

        if isinstance(image, str) and image.startswith("http"):
            return image

        if isinstance(image, str) and image.strip().startswith(("iVBOR", "/9j/")):
            return base64.b64decode(image)

        return image

    def generate_image(self, prompt: str, **kwargs) -> bytes:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        quality = self._resolve_param("quality", kwargs, "auto")

        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )

        image_bytes = base64.b64decode(response.data[0].b64_json)
        return self.process_image(image_bytes, **kwargs)

    def edit_image(
        self,
        images: Union[str, List[str]],
        prompt: str,
        **kwargs,
    ) -> bytes:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        quality = self._resolve_param("quality", kwargs, "standard")

        if not isinstance(images, list):
            images = [images]

        input_images = [self._load_image(img) for img in images]

        response = self.client.images.generate(
            model=model,
            input=prompt,
            images=input_images,
            size=size,
            quality=quality,
        )

        image_bytes = base64.b64decode(response.data[0].b64_json)
        return self.process_image(image_bytes, **kwargs)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists available OpenAI models and filters for image generation capability.
        """
        try:
            models = self.client.models.list()
            tti_models = []

            for m in models.data:
                model_id = m.id.lower()

                # Heuristic filter (OpenAI doesn't always tag modalities cleanly)
                if any(k in model_id for k in ["image", "dall", "vision"]):
                    tti_models.append({
                        "model_name": m.id,
                        "display_name": m.id,
                        "description": "OpenAI image generation model"
                    })

            # Ensure core models exist
            fallback = [
                {"model_name": "gpt-image-1", "display_name": "GPT Image 1"},
            ]

            return sorted(tti_models, key=lambda x: x["display_name"]) or fallback

        except Exception:
            return [
                {"model_name": "gpt-image-1", "display_name": "GPT Image 1"}
            ]

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": "OpenAI TTI", "id": "openai"}]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.config.update(settings)

        if "api_key" in settings:
            self.api_key = settings["api_key"]
            self.client = OpenAI(api_key=self.api_key)

        if "model_name" in settings:
            self.global_params["model"] = settings["model_name"]

        return True

    def get_credits(self) -> Optional[Dict[str, float]]:
        """
        OpenAI doesn't expose credits cleanly via SDK.
        This uses the billing endpoint (may require proper org permissions).
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.get(
                "https://api.openai.com/v1/dashboard/billing/credit_grants",
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                return None

            data = response.json()
            total = data.get("total_granted", 0.0)
            used = data.get("total_used", 0.0)

            return {
                "total_credits": float(total),
                "total_usage": float(used)
            }

        except Exception:
            return None
