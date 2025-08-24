from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
import requests
from io import BytesIO
from ascii_colors import trace_exception
from openai import OpenAI
from lollms_client.lollms_tti_binding import LollmsTTIBinding

BindingName = "OpenAITTIBinding"


class OpenAITTIBinding(LollmsTTIBinding):
    """
    OpenAI Text-to-Image (TTI) binding for LoLLMS.

    This binding provides access to OpenAI's image generation models
    (`gpt-image-1`, `dall-e-2`, `dall-e-3`).

    Parameters can be set globally at initialization or per-request during
    generation. Runtime parameters override initialization ones.

    ----------------------------
    Global parameters (init):
    ----------------------------
    - model (str): Model name. ["gpt-image-1", "dall-e-2", "dall-e-3"]. Default: "gpt-image-1".
    - api_key (str): OpenAI API key. If empty, uses OPENAI_API_KEY from environment.
    - size (str): Default image size. ["256x256", "512x512", "1024x1024", "2048x2048"]. Default: "1024x1024".
    - n (int): Default number of images per request (max 10). Default: 1.
    - quality (str): Image quality. ["standard", "hd"]. Only supported by "dall-e-3". Default: "standard".

    ----------------------------
    Runtime parameters (kwargs in generate or edit):
    ----------------------------
    - prompt (str): Required. The text prompt for image generation or editing.
    - size (str): Output size. Overrides global. Default: global "size".
    - n (int): Number of images to generate. Overrides global. Default: global "n".
    - quality (str): Image quality. Only for dall-e-3. Overrides global. Default: global "quality".
    - image (Union[str, Path]): For edit. Input image (base64, URL, or file path).
    - mask (Union[str, Path]): For edit. Optional mask image.

    Methods:
    --------
    - generate_image(prompt: str, **kwargs) -> List[bytes]:
        Generates images from a prompt.

    - edit_image(prompt: str, image: Union[str, Path], mask: Optional[Union[str, Path]] = None, **kwargs) -> List[bytes]:
        Edits an existing image using a prompt and optional mask.
    """

    def __init__(
        self,
        model: str = "gpt-image-1",
        api_key: Optional[str] = None,
        size: str = "1024x1024",
        n: int = 1,
        quality: str = "standard",
        **kwargs,
    ):
        self.client = OpenAI(api_key=api_key)
        self.global_params = {
            "model": model,
            "size": size,
            "n": n,
            "quality": quality,
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

    def generate_image(self, prompt: str, **kwargs) -> List[bytes]:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        n = self._resolve_param("n", kwargs, 1)
        quality = self._resolve_param("quality", kwargs, "standard")

        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            n=n,
            **({"quality": quality} if model == "dall-e-3" else {}),
        )

        return [base64.b64decode(img.b64_json) for img in response.data]

    def edit_image(
        self,
        prompt: str,
        image: Union[str, Path],
        mask: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> List[bytes]:
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        size = self._resolve_param("size", kwargs, "1024x1024")
        n = self._resolve_param("n", kwargs, 1)
        quality = self._resolve_param("quality", kwargs, "standard")

        image_data = self._load_image(image)
        mask_data = self._load_image(mask) if mask else None

        response = self.client.images.edit(
            model=model,
            prompt=prompt,
            image=image_data,
            mask=mask_data,
            size=size,
            n=n,
            **({"quality": quality} if model == "dall-e-3" else {}),
        )

        return [base64.b64decode(img.b64_json) for img in response.data]