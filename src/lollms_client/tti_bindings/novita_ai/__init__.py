from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import pipmaster as pm
import requests
from ascii_colors import ASCIIColors

from lollms_client.lollms_tti_binding import LollmsTTIBinding, TTIGenerationResult

pm.ensure_packages(["requests"])

BindingName = "NovitaAITTIBinding"

NOVITA_DEFAULT_HOST = "https://api.novita.ai"
DEFAULT_MODEL = "ming-image-0.1-design"
DEFAULT_EDIT_MODEL = "ming-image-0.1-design-layer"

FAMILY_QWEN = "qwen"
FAMILY_MING = "ming"

NOVITA_AI_MODELS = [
    {
        "model_name": "ming-image-0.1-design",
        "display_name": "Ming Image 0.1 Design (Text to Image)",
        "description": "Ming Image 0.1 text-to-image foundation model using the OpenAI image generations protocol.",
        "family": FAMILY_MING,
        "capabilities": ["generation"],
    },
    {
        "model_name": "ming-image-0.1-design-layer",
        "display_name": "Ming Image 0.1 Design (Layer Decoupling & Editing)",
        "description": "Decomposes input images into decoupled transparent RGBA layers (foreground subject, background) using the OpenAI image edits protocol.",
        "family": FAMILY_MING,
        "capabilities": ["editing"],
    },
    {
        "model_name": "qwen-image",
        "display_name": "Qwen-Image (Retired by Novita)",
        "description": "Legacy Qwen-Image model. Notice: As of Novita's Sep 30, 2026 update, /v3/async/qwen-image* endpoints are retired and return 404.",
        "family": FAMILY_QWEN,
        "capabilities": ["deprecated"],
    },
]

SUPPORTED_MODEL_NAMES = [model["model_name"] for model in NOVITA_AI_MODELS]


class NovitaAITTIBinding(LollmsTTIBinding):
    """Novita.ai TTI binding supporting Ming-Image generation and layer-decoupling editing."""

    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")
        binding_name = kwargs.pop("binding_name", BindingName)
        super().__init__(binding_name=binding_name, **kwargs)
        self.config = kwargs
        self.api_key = (
            self.config.get("api_key")
            or self.config.get("service_key")
            or os.environ.get("NOVITA_API_KEY")
            or ""
        )
        self.model_name = self._clean_model_name(self.config.get("model_name"))
        raw_host = (
            self.config.get("base_url")
            or self.config.get("host_address")
            or NOVITA_DEFAULT_HOST
        ).rstrip("/")
        self.base_url = self._normalize_v3_url(raw_host)
        self.v1_base_url = f"{self.base_url[: -len('/v3')]}/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LoLLMS-Client/3.2",
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Novita.ai API key is required. Please set it via 'api_key', "
                "'service_key', or the NOVITA_API_KEY environment variable."
            )

    # ------------------------------------------------------------------
    # Settings & Service Discovery
    # ------------------------------------------------------------------

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": "Novita.ai TTI", "id": "novita_ai"}]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        if not isinstance(settings, dict):
            return False
        self.config.update(settings)
        new_key = settings.get("api_key") or settings.get("service_key")
        if new_key:
            self.api_key = new_key
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if settings.get("model_name"):
            self.model_name = self._clean_model_name(settings["model_name"])
        return True

    @staticmethod
    def _clean_model_name(name: Optional[str]) -> str:
        if not name:
            return DEFAULT_MODEL
        cleaned = str(name).strip()
        for separator in ("/", "\\"):
            if separator in cleaned:
                cleaned = cleaned.split(separator)[-1].strip()
        return cleaned or DEFAULT_MODEL

    @staticmethod
    def _normalize_v3_url(raw_host: str) -> str:
        if raw_host.endswith("/v3"):
            return raw_host
        if raw_host.endswith("/v1") or raw_host.endswith("/v2") or raw_host.endswith("/openai"):
            return f"{raw_host[: raw_host.rfind('/')]}/v3"
        return f"{raw_host}/v3"

    @staticmethod
    def _clamp_dimension(value: Any, low: int, high: int, default: int) -> int:
        try:
            dimension = int(value)
        except (TypeError, ValueError):
            dimension = default
        return max(low, min(high, dimension))

    def _resolve_family(self) -> str:
        lowered = (self.model_name or "").lower()
        if "ming" in lowered:
            return FAMILY_MING
        if "qwen" in lowered:
            return FAMILY_QWEN
        return FAMILY_MING

    def _is_layer_model(self) -> bool:
        return "layer" in (self.model_name or "").lower()

    def get_zoo(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": model["display_name"],
                "description": model.get("description", ""),
                "size": "N/A (Cloud Hosted)",
                "type": "model",
                "link": model["model_name"],
            }
            for model in NOVITA_AI_MODELS
        ]

    def list_models(self) -> list:
        return [dict(model) for model in NOVITA_AI_MODELS]

    # ------------------------------------------------------------------
    # Unified Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        width: int = 512,
        height: int = 512,
        images: Optional[Union[str, List[str]]] = None,
        mask: Optional[str] = None,
        n: int = 1,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> TTIGenerationResult:
        # If input images are provided, execute image editing / layer decoupling
        if images:
            self._require_api_key()
            image_bytes = self._read_image_bytes(images)
            layers, layer_structure = self._edit_ming_layers(image_bytes, prompt, width, height, **kwargs)
            return TTIGenerationResult(
                images=layers,
                raw=layer_structure,
                metadata={"model": DEFAULT_EDIT_MODEL, "layer_structure": layer_structure},
            )

        return super().generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            images=images,
            mask=mask,
            n=n,
            modalities=modalities,
            **kwargs
        )

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> bytes:
        self._require_api_key()
        family = self._resolve_family()

        if family == FAMILY_QWEN:
            ASCIIColors.warning(
                f"[{self.binding_name}] Qwen-Image was retired by Novita AI on Sep 30, 2026. "
                f"Routing call to '{DEFAULT_MODEL}'."
            )
            family = FAMILY_MING

        if family == FAMILY_MING:
            if self._is_layer_model():
                ASCIIColors.info(
                    f"[{self.binding_name}] Switching model from '{self.model_name}' to "
                    f"'{DEFAULT_MODEL}' for text-to-image generation."
                )
            if negative_prompt:
                ASCIIColors.info(f"[{self.binding_name}] Ming-Image does not support negative prompts; ignoring it.")
            return self._generate_ming_image(prompt, width, height, **kwargs)

        raise RuntimeError(f"Unsupported model family for generation: {self.model_name}")

    def edit_image(
        self,
        images: Union[str, bytes, List[Any]],
        prompt: str,
        negative_prompt: str = "",
        mask: Optional[Union[str, bytes]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs
    ) -> bytes:
        self._require_api_key()
        if mask:
            ASCIIColors.info(
                f"[{self.binding_name}] Novita.ai layer decoupling does not require an inpainting mask; ignoring mask."
            )

        image_bytes = self._read_image_bytes(images)
        family = self._resolve_family()

        if family == FAMILY_QWEN:
            ASCIIColors.warning(
                f"[{self.binding_name}] Qwen-Image Edit was retired by Novita AI on Sep 30, 2026. "
                f"Automatically using '{DEFAULT_EDIT_MODEL}' (Ming Layer Decoupling)."
            )

        # Ming-Image-0.1-Design-Layer handles layer decoupling / alpha background extraction
        layers, _ = self._edit_ming_layers(image_bytes, prompt, width, height, **kwargs)

        # Identify which layer to return:
        # Layer decoupling usually yields: layer 0 = background, layer 1 = foreground / transparent subject.
        # If the user specifically wants the subject/transparent cutout, layer -1 is the extracted foreground.
        lower_prompt = prompt.lower()
        if any(w in lower_prompt for w in ["background only", "isolate background", "only background"]):
            return layers[0]
        if len(layers) > 1:
            return layers[-1]
        return layers[0]

    # ------------------------------------------------------------------
    # Ming-Image (OpenAI-compatible image protocol)
    # ------------------------------------------------------------------

    def _ming_image_endpoints(self, action: str) -> List[str]:
        endpoints: List[str] = [
            f"{NOVITA_DEFAULT_HOST}/openai/v1/images/{action}",
            f"{NOVITA_DEFAULT_HOST}/v1/images/{action}",
            f"{NOVITA_DEFAULT_HOST}/v3/openai/images/{action}",
            f"{NOVITA_DEFAULT_HOST}/openapi/v1/images/{action}",
        ]
        api_root = self.v1_base_url[: -len("/v1")]
        if api_root != NOVITA_DEFAULT_HOST:
            endpoints.insert(0, f"{api_root}/openai/v1/images/{action}")
            endpoints.insert(1, f"{api_root}/v1/images/{action}")
        return endpoints

    def _generate_ming_image(self, prompt: str, width: int, height: int, **kwargs) -> bytes:
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": prompt,
            "output_format": kwargs.get("output_format", "png"),
            "response_format": "b64_json",
            "size": self._format_ming_size(width, height, is_edit=False, **kwargs),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        data = self._post_with_fallback(
            self._ming_image_endpoints("generations"),
            payload,
            request_timeout=float(kwargs.get("timeout", 300)),
        )
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            b64_image = item.get("b64_json")
            if b64_image:
                return base64.b64decode(b64_image)
            image_url = item.get("url")
            if image_url:
                download = requests.get(image_url, timeout=60)
                download.raise_for_status()
                return download.content
        raise RuntimeError(f"Novita.ai Ming-Image returned no image data. Response: {str(data)[:300]}")

    def _edit_ming_layers(
        self,
        image_bytes: bytes,
        prompt: str,
        width: Optional[int],
        height: Optional[int],
        **kwargs
    ) -> Tuple[List[bytes], Optional[str]]:
        form_fields: Dict[str, Any] = {
            "model": DEFAULT_EDIT_MODEL,
            "prompt": prompt,
            "output_format": kwargs.get("output_format", "png"),
            "response_format": "b64_json",
            "watermark": "true" if kwargs.get("watermark") else "false",
        }
        # Novita documentation: ming-image-0.1-design-layer preserves input dimensions.
        # Explicit sizes are rejected; pass size only if explicitly provided in kwargs.
        size_str = self._format_ming_size(width, height, is_edit=True, **kwargs)
        if size_str:
            form_fields["size"] = size_str

        # Endpoint uses multipart/form-data with field name 'image[]'
        data = self._post_with_fallback(
            self._ming_image_endpoints("edits"),
            form_fields,
            request_timeout=float(kwargs.get("timeout", 300)),
            files={"image[]": ("input.png", image_bytes, "image/png")},
        )
        layers: List[bytes] = []
        layer_structure: Optional[str] = None
        for item in data.get("data") or []:
            if not isinstance(item, dict):
                continue
            if layer_structure is None and item.get("revised_prompt"):
                layer_structure = item["revised_prompt"]
            b64_image = item.get("b64_json")
            if b64_image:
                layers.append(base64.b64decode(b64_image))
            elif item.get("url"):
                dl = requests.get(item["url"], timeout=60)
                dl.raise_for_status()
                layers.append(dl.content)

        if not layers:
            raise RuntimeError(
                f"Novita.ai Ming layer decoupling returned no layers. Response: {str(data)[:300]}"
            )
        return layers, layer_structure

    def _format_ming_size(
        self,
        width: Optional[int],
        height: Optional[int],
        is_edit: bool = False,
        **kwargs
    ) -> Optional[str]:
        explicit_size = kwargs.get("size")
        if explicit_size:
            return str(explicit_size)
        if is_edit:
            # Ming Layer decoupling rejects explicit resize dims; default to auto
            return "auto"
        if width is None and height is None:
            return "1024x1024"
        return (
            f"{self._clamp_dimension(width, 1024, 8192, 1024)}"
            f"x{self._clamp_dimension(height, 1024, 8192, 1024)}"
        )

    # ------------------------------------------------------------------
    # Shared HTTP Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image_bytes(source: Union[str, bytes, bytearray, Path, List[Any]]) -> bytes:
        if isinstance(source, list):
            if not source:
                raise ValueError("Empty image list provided.")
            source = source[0]
        if isinstance(source, (bytes, bytearray)):
            return bytes(source)
        if isinstance(source, Path):
            return source.read_bytes()
        if isinstance(source, str):
            if source.startswith("http://") or source.startswith("https://"):
                download = requests.get(source, timeout=60)
                download.raise_for_status()
                return download.content
            if os.path.isfile(source):
                return Path(source).read_bytes()
            stripped = re.sub(r"^data:image/[^;]+;base64,", "", source.strip())
            try:
                return base64.b64decode(stripped)
            except Exception as decode_error:
                raise ValueError(
                    "Unsupported image input. Provide raw bytes, a base64 string, a data URI, a file path, or an image URL."
                ) from decode_error
        raise ValueError("Unsupported image input type. Provide bytes, base64, a data URI, a file path, or an image URL.")

    def _post_with_fallback(
        self,
        path_variants: List[str],
        payload: Dict[str, Any],
        request_timeout: float = 60.0,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        unique_urls: List[str] = []
        for variant in path_variants:
            if variant.startswith("http://") or variant.startswith("https://"):
                url = variant
            else:
                url = f"{self.base_url}/{variant.lstrip('/')}"
            if url not in unique_urls:
                unique_urls.append(url)

        last_error: Optional[Exception] = None
        for url in unique_urls:
            request_headers = dict(self.headers)
            if files is not None:
                # requests automatically sets multipart/form-data boundary when Content-Type is omitted
                request_headers.pop("Content-Type", None)

            try:
                ASCIIColors.info(f"[{self.binding_name}] Requesting {url}...")
                if files is None:
                    response = requests.post(url, json=payload, headers=request_headers, timeout=request_timeout)
                else:
                    response = requests.post(url, data=payload, files=files, headers=request_headers, timeout=request_timeout)

                if response.status_code == 404:
                    detail = response.text[:200] if response.text else "No response body"
                    ASCIIColors.warning(f"[{self.binding_name}] 404 from {url}: {detail}")
                    last_error = requests.exceptions.HTTPError(f"404 Client Error for url: {url}", response=response)
                    continue

                if response.status_code != 200:
                    ASCIIColors.error(f"[{self.binding_name}] HTTP {response.status_code} from {url}: {response.text[:500]}")

                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as http_error:
                last_error = http_error
                if http_error.response is not None and http_error.response.status_code == 404:
                    continue
                body_detail = http_error.response.text[:300] if http_error.response is not None else ""
                raise RuntimeError(
                    f"Novita.ai request failed: {http_error}. Body: {body_detail}"
                ) from http_error
            except Exception as ex:
                last_error = ex
                continue

        raise RuntimeError(
            f"Novita.ai endpoint not found. Tried routes: {', '.join(unique_urls)}. Last error: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Management Commands
    # ------------------------------------------------------------------

    def get_user_balance(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"status": False, "message": "No API key configured."}
        urls = [
            "https://api.novita.ai/openapi/v1/billing/balance/detail",
            f"{NOVITA_DEFAULT_HOST}/v3/user/balance",
        ]
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    available = (
                        data.get("availableBalance")
                        or data.get("balance")
                        or (data.get("data") or {}).get("balance")
                    )
                    if available is not None:
                        available_float = float(available)
                        usd = available_float / 10000.0 if available_float > 1000 else available_float
                        return {"status": True, "raw_balance": available, "balance_usd": f"${usd:.2f}", "data": data}
                    return {"status": True, "data": data}
            except Exception:
                continue
        return {"status": False, "message": "Failed to query Novita AI account balance."}

    def validate_key(self) -> Dict[str, Any]:
        balance = self.get_user_balance()
        if balance.get("status"):
            return {"status": True, "message": f"API key is valid. Balance: {balance.get('balance_usd', 'OK')}"}
        return {"status": False, "message": "Validation failed: Could not authenticate with Novita.ai."}

    def get_credits(self) -> Optional[Dict[str, float]]:
        balance = self.get_user_balance()
        if balance.get("status"):
            raw = balance.get("raw_balance")
            if raw is not None:
                try:
                    raw_float = float(raw)
                    usd = raw_float / 10000.0 if raw_float > 1000 else raw_float
                    return {"total_credits": usd, "total_usage": 0.0}
                except (TypeError, ValueError):
                    return None
        return None