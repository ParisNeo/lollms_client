from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
import requests
from io import BytesIO
from ascii_colors import trace_exception
from openai import OpenAI
from lollms_client.lollms_tti_binding import LollmsTTIBinding
import os
BindingName = "OpenAITTIBinding"


class OpenAITTIBinding(LollmsTTIBinding):
    """
    OpenAI Text-to-Image (TTI) binding for LoLLMS.
    """

    def __init__(
        self,
        **kwargs,
    ):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)
        self.client = OpenAI(api_key=kwargs.get("api_key" or os.environ.get("OPENAI_API_KEY")))
        self.global_params = {
            "model": kwargs.get("model_name") or "gpt-image-1",
            "size": kwargs.get("size", "1024x1024"),
            "n": kwargs.get("n", 1),
            "quality": kwargs.get("quality", "standard"),
        }

    def _resolve_param(self, name: str, kwargs: Dict[str, Any], default: Any) -> Any:
        """Resolve a parameter from runtime kwargs, global config, or default."""
        return kwargs.get(name, self.global_params.get(name, default))

    def _load_image(self, image: Union[str, Path]) -> Any:
        """Helper to load an image from path, URL, or base64 string."""
        if isinstance(image, Path) or Path(str(image)).exists():
            with open(image, "rb") as f:
                return f.read()
        if isinstance(image, str) and image.startswith("http"):
            return image
        if isinstance(image, str) and image.strip().startswith(("iVBOR", "/9j/")):  # base64
            return base64.b64decode(image)
        return image

    def generate_image(self, prompt: str, **kwargs) -> bytes:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        n = 1 # We return single bytes for compliance with the base class logic
        quality = self._resolve_param("quality", kwargs, "standard")

        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            response_format="b64_json",
            **({"quality": quality} if model == "dall-e-3" else {}),
        )

        image_bytes = base64.b64decode(response.data[0].b64_json)
        return self.process_image(image_bytes, **kwargs)

    def edit_image(
        self,
        images: Union[str, List[str]],
        prompt: str,
        mask: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> bytes:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        n = 1
        quality = self._resolve_param("quality", kwargs, "standard")

        image_data = self._load_image(images[0] if isinstance(images, list) else images)
        mask_data = self._load_image(mask) if mask else None

        response = self.client.images.edit(
            model=model,
            prompt=prompt,
            image=image_data,
            mask=mask_data,
            size=size,
            n=n,
            response_format="b64_json",
            **({"quality": quality} if model == "dall-e-3" else {}),
        )

        image_bytes = base64.b64decode(response.data[0].b64_json)
        return self.process_image(image_bytes, **kwargs)
