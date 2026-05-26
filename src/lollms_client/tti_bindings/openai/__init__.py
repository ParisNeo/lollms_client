from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from io import BytesIO
import base64
import os
import requests
import pipmaster as pm
pm.ensure_packages({"openai": ">=2.32.0"})
from openai import OpenAI
import openai
print(openai.__version__)
from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import ASCIIColors, trace_exception

BindingName = "OpenAITTIBinding"

# ---------------------------------------------------------------------------
# Model capability registry
# ---------------------------------------------------------------------------
# Each entry describes what a model supports so we can build API calls
# correctly without hard-coding per-model branches all over the place.
#
# Keys:
#   "edit"        – supports images.edit (mask-based inpainting via DALL-E 2,
#                   or the newer multi-image edit endpoint)
#   "response_fmt"– how the model returns the image:
#                     "b64_json"  – response.data[0].b64_json  (DALL-E 2/3, gpt-image-1)
#                     "url"       – response.data[0].url        (some older models)
#   "sizes"       – valid size strings accepted by this model
#   "qualities"   – valid quality values (empty list = not supported / ignored)
#   "n_max"       – maximum number of images per request
#   "style"       – whether the "style" parameter is accepted (DALL-E 3 only)
# ---------------------------------------------------------------------------
_MODEL_CAPS: Dict[str, Dict[str, Any]] = {
    # --- GPT-image family (latest, native multimodal) ---
    "gpt-image-1": {
        "edit": True,
        "response_fmt": "b64_json",
        "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "qualities": ["auto", "low", "medium", "high"],
        "n_max": 1,
        "style": False,
    },
    # gpt-5 ships with the same image-generation surface as gpt-image-1
    "gpt-5": {
        "edit": True,
        "response_fmt": "b64_json",
        "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "qualities": ["auto", "low", "medium", "high"],
        "n_max": 1,
        "style": False,
    },
    # Explicit alias that some callers may use
    "gpt-5-image": {
        "edit": True,
        "response_fmt": "b64_json",
        "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
        "qualities": ["auto", "low", "medium", "high"],
        "n_max": 1,
        "style": False,
    },
    # --- DALL-E 3 ---
    "dall-e-3": {
        "edit": False,
        "response_fmt": "b64_json",
        "sizes": ["1024x1024", "1792x1024", "1024x1792"],
        "qualities": ["standard", "hd"],
        "n_max": 1,
        "style": True,
    },
    # --- DALL-E 2 ---
    "dall-e-2": {
        "edit": True,
        "response_fmt": "b64_json",
        "sizes": ["256x256", "512x512", "1024x1024"],
        "qualities": [],   # no quality param
        "n_max": 10,
        "style": False,
    },
}

# Default caps for any model not listed above (forward-compatible fallback).
_DEFAULT_CAPS: Dict[str, Any] = {
    "edit": True,
    "response_fmt": "b64_json",
    "sizes": ["1024x1024", "1536x1024", "1024x1536", "auto"],
    "qualities": ["auto", "low", "medium", "high"],
    "n_max": 1,
    "style": False,
}

# Static fallback list shown when the live API cannot be reached.
_FALLBACK_MODELS: List[Dict[str, Any]] = [
    {"model_name": "gpt-image-1",  "display_name": "GPT Image 1",   "description": "Latest OpenAI image generation model"},
    {"model_name": "gpt-5",        "display_name": "GPT-5",          "description": "GPT-5 with native image generation"},
    {"model_name": "gpt-5-image",  "display_name": "GPT-5 Image",    "description": "GPT-5 image generation alias"},
    {"model_name": "dall-e-3",     "display_name": "DALL-E 3",       "description": "High-quality DALL-E 3"},
    {"model_name": "dall-e-2",     "display_name": "DALL-E 2",       "description": "DALL-E 2 with inpainting support"},
]


class OpenAITTIBinding(LollmsTTIBinding):
    """
    OpenAI Text-to-Image binding for lollms_client.

    Supports the full gpt-image / DALL-E family including gpt-5 image
    generation.  Interface mirrors OpenRouterTTIBinding so both can be
    used interchangeably by callers.
    """

    def __init__(self, **kwargs):
        # Accept both 'model' and 'model_name' for config-file compatibility.
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        binding_name = kwargs.pop("binding_name", BindingName)
        super().__init__(binding_name=binding_name, config=kwargs)

        # Mirror OpenRouter's "service_key" naming while keeping "api_key" as alias.
        self.service_key = (
            kwargs.get("service_key")
            or kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )

        # Optional custom base URL for OpenAI-compatible servers
        # (e.g. Ollama, LM Studio, LocalAI, vLLM, Mistral, Together …).
        # Accepted as "host_address" (OpenRouter convention) or "base_url".
        # When omitted the official OpenAI endpoint is used.
        self.host_address: Optional[str] = (
            kwargs.get("host_address")
            or kwargs.get("base_url")
            or None
        )

        # SSL verification:
        #   verify_ssl_certificate=False         -> skip cert check (self-signed)
        #   certificate_file_path='/path/to.pem' -> custom CA bundle / client cert
        #   default                              -> system CA store
        cert_path   = kwargs.get("certificate_file_path", "") or ""
        verify_flag = kwargs.get("verify_ssl_certificate", True)
        if cert_path:
            self.verify_ssl: Union[bool, str] = cert_path
        elif verify_flag is False or str(verify_flag).lower() in ("false", "0", "no"):
            self.verify_ssl = False
        else:
            self.verify_ssl = True

        self.client = self._build_client()
        self.config = kwargs

        self.global_params = {
            "model":   kwargs.get("model_name", "gpt-image-1"),
            "size":    kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "auto"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> OpenAI:
        """
        Construct an OpenAI client, optionally pointed at a custom base URL.

        Any OpenAI-compatible server works here — Ollama, LM Studio, LocalAI,
        vLLM, Together AI, Mistral, Groq, etc. — as long as it exposes the
        ``/v1/images/generations`` (and optionally ``/v1/images/edits``) paths.

        Examples
        --------
        >>> # Ollama running locally
        >>> OpenAITTIBinding(host_address="http://localhost:11434/v1", service_key="ollama")
        >>> # LM Studio
        >>> OpenAITTIBinding(host_address="http://localhost:1234/v1")
        >>> # Remote vLLM instance
        >>> OpenAITTIBinding(host_address="https://my-vllm.example.com/v1", service_key="token")
        """
        import httpx
        client_kwargs: Dict[str, Any] = {"api_key": self.service_key or "not-needed"}
        if self.host_address:
            client_kwargs["base_url"] = self.host_address.rstrip("/")
        # Pass a custom httpx client so we can control SSL verification.
        # verify_ssl may be True (default CA), False (skip), or a str path to a PEM bundle.
        if self.verify_ssl is not True:
            client_kwargs["http_client"] = httpx.Client(verify=self.verify_ssl)
        return OpenAI(**client_kwargs)

    def _caps(self, model: str) -> Dict[str, Any]:
        """Return capability dict for *model*, falling back gracefully."""
        # Exact match first, then prefix match for versioned aliases.
        if model in _MODEL_CAPS:
            return _MODEL_CAPS[model]
        for key in _MODEL_CAPS:
            if model.startswith(key):
                return _MODEL_CAPS[key]
        return _DEFAULT_CAPS

    def _resolve_param(self, name: str, kwargs: Dict[str, Any], default: Any) -> Any:
        return kwargs.get(name, self.global_params.get(name, default))

    def _size_from_dimensions(self, width: int, height: int, caps: Dict[str, Any]) -> str:
        """Pick the closest supported size string for the given pixel dimensions."""
        target = f"{width}x{height}"
        if target in caps["sizes"]:
            return target
        # Prefer auto when available (model decides best fit).
        if "auto" in caps["sizes"]:
            return "auto"
        # Fall back to first listed size.
        return caps["sizes"][0]

    def _to_bytes(self, image: Union[str, Path, "Image.Image"]) -> bytes:  # noqa: F821
        """Normalise any image representation to raw bytes."""
        # PIL Image
        if hasattr(image, "save"):
            buf = BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

        path = Path(str(image))
        if path.exists():
            return path.read_bytes()

        if isinstance(image, str):
            if image.startswith("http"):
                resp = requests.get(image, timeout=30, verify=self.verify_ssl)
                resp.raise_for_status()
                return resp.content
            # data-URL
            if "," in image and image.startswith("data:"):
                return base64.b64decode(image.split(",", 1)[1])
            # Raw base64
            try:
                return base64.b64decode(image)
            except Exception:
                pass

        raise ValueError(f"Cannot convert image of type {type(image)} to bytes.")

    def _wrap_image_file(self, image: Union[str, Path, "Image.Image"]) -> tuple:  # noqa: F821
        """Return (filename, bytes, mime) ready for the OpenAI files API."""
        data = self._to_bytes(image)
        # Detect PNG vs JPEG by magic bytes.
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        ext  = "png" if mime == "image/png" else "jpg"
        return (f"image.{ext}", data, mime)

    def _extract_image_bytes(self, response) -> bytes:
        """Pull image bytes out of an images.generate / images.edit response."""
        item = response.data[0]
        if getattr(item, "b64_json", None):
            return base64.b64decode(item.b64_json)
        if getattr(item, "url", None):
            resp = requests.get(item.url, timeout=60, verify=self.verify_ssl)
            resp.raise_for_status()
            return resp.content
        raise ValueError("Response contained neither b64_json nor url.")

    # ------------------------------------------------------------------
    # Public API  (mirrors OpenRouterTTIBinding)
    # ------------------------------------------------------------------

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> bytes:
        """
        Generate an image from *prompt*.

        Parameters
        ----------
        prompt          : Positive text prompt.
        negative_prompt : Appended to the prompt for models that handle it via
                          text (OpenAI has no native negative-prompt field).
        width / height  : Desired output dimensions; mapped to the nearest
                          supported size string for the chosen model.
        **kwargs        : Override model, quality, size, style, n, etc.
        """
        model   = self._resolve_param("model",   kwargs, "gpt-image-1")
        caps    = self._caps(model)
        size    = kwargs.get("size") or self._size_from_dimensions(width, height, caps)
        quality = self._resolve_param("quality", kwargs, caps["qualities"][0] if caps["qualities"] else None)

        # Merge negative prompt into positive when provided.
        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        call_kwargs: Dict[str, Any] = dict(
            model=model,
            prompt=full_prompt,
            size=size,
            n=min(kwargs.get("n", 1), caps["n_max"]),
        )
        if quality and caps["qualities"]:
            call_kwargs["quality"] = quality
        if caps["style"] and "style" in kwargs:
            call_kwargs["style"] = kwargs["style"]

        # All supported models return b64_json when asked.
        call_kwargs["response_format"] = "b64_json"

        ASCIIColors.panel(
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Size:[/bold]  {size}\n"
            f"[bold]Quality:[/bold] {quality}",
            "[bold]OpenAI TTI generate_image[/bold]",
        )

        response = self.client.images.generate(**call_kwargs)
        image_bytes = self._extract_image_bytes(response)
        return self.process_image(image_bytes, **kwargs)

    def edit_image(
        self,
        images: Union[str, Path, List, "Image.Image"],  # noqa: F821
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        mask: Optional[Union[str, Path, "Image.Image"]] = None,  # noqa: F821
        **kwargs,
    ) -> bytes:
        """
        Edit / vary an existing image.

        For models that support the images.edit endpoint (DALL-E 2, gpt-image-1,
        gpt-5) the native edit API is used.  For DALL-E 3 (no edit support) we
        fall back to a generate call with the prompt.

        Parameters
        ----------
        images          : One or more source images (path, URL, base64, PIL).
        prompt          : Edit instruction.
        negative_prompt : Appended to the prompt text.
        width / height  : Desired output size.
        mask            : Optional RGBA mask (transparent = area to regenerate).
        """
        model = self._resolve_param("model", kwargs, "gpt-image-1")
        caps  = self._caps(model)
        size  = kwargs.get("size") or self._size_from_dimensions(width, height, caps)

        full_prompt = prompt
        if negative_prompt:
            full_prompt = f"{prompt}\n\nAvoid: {negative_prompt}"

        # Models without edit support → fall back to generate.
        if not caps["edit"]:
            ASCIIColors.warning(
                f"Model '{model}' does not support image editing. "
                "Falling back to generate_image."
            )
            return self.generate_image(
                full_prompt, width=width, height=height, **kwargs
            )

        if not isinstance(images, list):
            images = [images]

        quality = self._resolve_param("quality", kwargs, caps["qualities"][0] if caps["qualities"] else None)

        ASCIIColors.panel(
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Size:[/bold]  {size}\n"
            f"[bold]Images:[/bold] {len(images)}",
            "[bold]OpenAI TTI edit_image[/bold]",
        )

        # DALL-E 2 uses the legacy single-image + mask endpoint.
        if model == "dall-e-2":
            img_file  = self._wrap_image_file(images[0])
            call_kwargs: Dict[str, Any] = dict(
                model=model,
                image=img_file,
                prompt=full_prompt,
                size=size,
                n=1,
                response_format="b64_json",
            )
            if mask is not None:
                call_kwargs["mask"] = self._wrap_image_file(mask)
            response = self.client.images.edit(**call_kwargs)

        else:
            # gpt-image-1 / gpt-5 multi-image edit endpoint.
            wrapped = [self._wrap_image_file(img) for img in images]
            call_kwargs = dict(
                model=model,
                image=wrapped if len(wrapped) > 1 else wrapped[0],
                prompt=full_prompt,
                size=size,
                n=1,
                response_format="b64_json",
            )
            if quality and caps["qualities"]:
                call_kwargs["quality"] = quality
            if mask is not None:
                call_kwargs["mask"] = self._wrap_image_file(mask)
            response = self.client.images.edit(**call_kwargs)

        image_bytes = self._extract_image_bytes(response)
        return self.process_image(image_bytes, **kwargs)

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Return available OpenAI image-generation models.

        Queries the live API and augments with our capability registry so
        that models released after the last registry update are still surfaced.
        """
        try:
            api_models = self.client.models.list()
        except Exception as exc:
            ASCIIColors.warning(f"Could not reach OpenAI models endpoint: {exc}")
            return _FALLBACK_MODELS

        tti_keywords = {"image", "dall", "vision"}
        # Also include any model in our own registry.
        registry_ids = set(_MODEL_CAPS.keys())

        seen: set = set()
        result: List[Dict[str, Any]] = []

        for m in api_models.data:
            mid = m.id
            mid_lower = mid.lower()

            is_tti = (
                any(kw in mid_lower for kw in tti_keywords)
                or mid in registry_ids
            )
            if is_tti and mid not in seen:
                seen.add(mid)
                caps = self._caps(mid)
                result.append({
                    "model_name":   mid,
                    "display_name": mid,
                    "description":  (
                        f"OpenAI image model — "
                        f"edit={'yes' if caps['edit'] else 'no'}, "
                        f"sizes={caps['sizes']}"
                    ),
                })

        # Ensure every model in our registry appears even if not yet in the API list.
        for entry in _FALLBACK_MODELS:
            if entry["model_name"] not in seen:
                seen.add(entry["model_name"])
                result.append(entry)

        return sorted(result, key=lambda x: x["display_name"]) or _FALLBACK_MODELS

    # ------------------------------------------------------------------
    # Settings / credits  (mirrors OpenRouterTTIBinding)
    # ------------------------------------------------------------------

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": "OpenAI TTI", "id": "openai"}]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.config.update(settings)

        # Support both naming conventions for the API key.
        new_key = settings.get("service_key") or settings.get("api_key")
        if new_key:
            self.service_key = new_key

        # Support both naming conventions for the base URL.
        new_host = settings.get("host_address") or settings.get("base_url")
        if new_host is not None:
            self.host_address = new_host or None

        # SSL certificate settings.
        cert_path   = settings.get("certificate_file_path", "") or ""
        verify_flag = settings.get("verify_ssl_certificate")
        if cert_path:
            self.verify_ssl = cert_path
        elif verify_flag is not None:
            if verify_flag is False or str(verify_flag).lower() in ("false", "0", "no"):
                self.verify_ssl = False
            else:
                self.verify_ssl = True

        # Rebuild the client whenever any connection param changes.
        if new_key or new_host is not None or cert_path or verify_flag is not None:
            self.client = self._build_client()

        if "model_name" in settings:
            self.global_params["model"] = settings["model_name"]

        return True

    def get_credits(self) -> Optional[Dict[str, float]]:
        """
        Attempt to fetch remaining credit balance.

        For the official OpenAI endpoint the billing API is used.
        For custom hosts the endpoint is skipped (most don't expose billing)
        and ``None`` is returned silently.
        """
        # Custom hosts almost never expose the OpenAI billing endpoint.
        if self.host_address:
            return None
        try:
            resp = requests.get(
                "https://api.openai.com/v1/dashboard/billing/credit_grants",
                headers={"Authorization": f"Bearer {self.service_key}"},
                timeout=10,
                verify=self.verify_ssl,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return {
                "total_credits": float(data.get("total_granted", 0.0)),
                "total_usage":   float(data.get("total_used",    0.0)),
            }
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    key  = os.getenv("OPENAI_API_KEY", "")
    host = os.getenv("OPENAI_HOST", "")   # e.g. http://localhost:11434/v1

    if not key and not host:
        print("Set OPENAI_API_KEY (official API) or OPENAI_HOST (custom server).")
        sys.exit(1)

    binding = OpenAITTIBinding(
        service_key=key or "not-needed",
        host_address=host or None,
    )

    print(f"Host : {binding.host_address or 'api.openai.com (default)'}")
    print("Available models:")
    for m in binding.list_models():
        print(f"  {m['model_name']}")

    print("\nGenerating test image …")
    img = binding.generate_image(
        "A futuristic Paris skyline at dusk, digital art",
        width=1024, height=1024,
    )
    if img:
        Path("test_output.png").write_bytes(img)
        print("Saved test_output.png")

    credits = binding.get_credits()
    if credits:
        print(f"Credits: {credits}")
