from __future__ import annotations

import base64
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
DEFAULT_MODEL = "qwen-image"

FAMILY_QWEN = "qwen"
FAMILY_MING = "ming"

NOVITA_AI_MODELS = [
    {
        "model_name": "qwen-image",
        "display_name": "Qwen-Image",
        "description": "20B MMDiT next-gen text-to-image model. Excellent at graphic posters with native text rendering.",
        "family": FAMILY_QWEN,
    },
    {
        "model_name": "ming-image-0.1-design",
        "display_name": "Ming Image 0.1 Design",
        "description": "Ming Image text-to-image generation using the OpenAI image generations protocol.",
        "family": FAMILY_MING,
    },
    {
        "model_name": "ming-image-0.1-design-layer",
        "display_name": "Ming Image 0.1 Design (Layer Decoupling)",
        "description": "Splits an input image into decoupled layers (foreground/background) using the OpenAI image edits protocol.",
        "family": FAMILY_MING,
    },
]

SUPPORTED_MODEL_NAMES = [model["model_name"] for model in NOVITA_AI_MODELS]


class NovitaAITTIBinding(LollmsTTIBinding):
    """Novita.ai TTI binding for the Qwen-Image and Ming-Image model APIs."""

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
            or f"{NOVITA_DEFAULT_HOST}/v3"
        ).rstrip("/")
        self.base_url = self._normalize_v3_url(raw_host)
        self.v1_base_url = f"{self.base_url[: -len('/v3')]}/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "LoLLMS-Client/3.1",
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
        if raw_host.endswith("/v1") or raw_host.endswith("/v2"):
            return f"{raw_host[: raw_host.rfind('/')]}/v3"
        return f"{raw_host}/v3"

    @staticmethod
    def _clamp_dimension(value: Any, low: int, high: int, default: int) -> int:
        try:
            dimension = int(value)
        except (TypeError, ValueError):
            dimension = default
        return max(low, min(high, dimension))

    def _resolve_family(self) -> Optional[str]:
        lowered = (self.model_name or "").lower()
        if "qwen" in lowered:
            return FAMILY_QWEN
        if "ming" in lowered:
            return FAMILY_MING
        return None

    def _is_layer_model(self) -> bool:
        return "layer" in (self.model_name or "").lower()

    def _raise_unsupported_model(self) -> NoReturn:
        raise RuntimeError(
            f"Novita.ai no longer supports generic Stable Diffusion checkpoint generation. "
            f"The legacy /v3/async/txt2img and /v3/async/img2img routes have been retired, "
            f"so model '{self.model_name}' cannot be used. "
            f"Set model_name to one of the supported foundation models: {', '.join(SUPPORTED_MODEL_NAMES)}."
        )

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

    # ------------------------------------------------------------------
    # Model Listing
    # ------------------------------------------------------------------

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
        if images and self._resolve_family() == FAMILY_MING and self._is_layer_model():
            self._require_api_key()
            image_bytes = self._read_image_bytes(images)
            layers, layer_structure = self._edit_ming_layers(image_bytes, prompt, width, height, **kwargs)
            return TTIGenerationResult(
                images=layers,
                raw=layer_structure,
                metadata={"model": self.model_name, "layer_structure": layer_structure},
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
            if negative_prompt:
                ASCIIColors.info(f"[{self.binding_name}] Qwen-Image does not support negative prompts; ignoring it.")
            return self._generate_qwen_image(prompt, width, height, **kwargs)
        if family == FAMILY_MING:
            if self._is_layer_model():
                raise ValueError(
                    "ming-image-0.1-design-layer only supports layer decoupling on an input image. "
                    "Provide images via edit_image or the generate() images argument."
                )
            if negative_prompt:
                ASCIIColors.info(f"[{self.binding_name}] Ming-Image does not support negative prompts; ignoring it.")
            return self._generate_ming_image(prompt, width, height, **kwargs)
        self._raise_unsupported_model()

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
                f"[{self.binding_name}] Novita.ai image edit APIs do not support inpainting masks; ignoring the provided mask."
            )
        image_bytes = self._read_image_bytes(images)
        family = self._resolve_family()
        if family == FAMILY_QWEN:
            return self._edit_qwen_image(image_bytes, prompt, **kwargs)
        if family == FAMILY_MING:
            if not self._is_layer_model():
                raise ValueError(
                    "ming-image-0.1-design only supports text-to-image generation. "
                    "Use ming-image-0.1-design-layer for layer decoupling edits."
                )
            layers, _ = self._edit_ming_layers(image_bytes, prompt, width, height, **kwargs)
            return layers[0]
        self._raise_unsupported_model()

    # ------------------------------------------------------------------
    # Qwen-Image (v3 asynchronous task pipeline)
    # ------------------------------------------------------------------

    def _generate_qwen_image(self, prompt: str, width: int, height: int, **kwargs) -> bytes:
        payload = {
            "prompt": prompt,
            "size": (
                f"{self._clamp_dimension(width, 256, 1536, 1024)}"
                f"*{self._clamp_dimension(height, 256, 1536, 1024)}"
            ),
        }
        endpoints = [
            f"{self.base_url}/async/qwen-image-txt2img",
            f"{NOVITA_DEFAULT_HOST}/v3/async/qwen-image-txt2img",
        ]
        data = self._post_with_fallback(endpoints, payload)
        task_id = self._extract_task_id(data)
        if not task_id:
            raise RuntimeError(f"Novita.ai did not return a task_id for Qwen-Image. Response: {data}")
        return self._poll_task_result(task_id, timeout=kwargs.get("timeout", 300))

    def _edit_qwen_image(self, image_bytes: bytes, prompt: str, **kwargs) -> bytes:
        payload = {
            "prompt": prompt,
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "seed": int(kwargs.get("seed", -1)),
            "output_format": kwargs.get("output_format", "jpeg"),
        }
        endpoints = [
            f"{self.base_url}/async/qwen-image-img2img",
            f"{NOVITA_DEFAULT_HOST}/v3/async/qwen-image-img2img",
            f"{self.base_url}/async/qwen-image-image-edit",
        ]
        data = self._post_with_fallback(endpoints, payload)
        task_id = self._extract_task_id(data)
        if not task_id:
            raise RuntimeError(f"Novita.ai did not return a task_id for the Qwen-Image edit. Response: {data}")
        return self._poll_task_result(task_id, timeout=kwargs.get("timeout", 300))

    # ------------------------------------------------------------------
    # Ming-Image (v1 OpenAI-compatible protocol)
    # ------------------------------------------------------------------

    def _generate_ming_image(self, prompt: str, width: int, height: int, **kwargs) -> bytes:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "output_format": kwargs.get("output_format", "png"),
            "response_format": "b64_json",
            "size": self._format_ming_size(width, height, **kwargs),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        response = requests.post(f"{self.v1_base_url}/images/generations", json=payload, headers=self.headers, timeout=180)
        if response.status_code != 200:
            raise RuntimeError(
                f"Novita.ai Ming-Image generation failed: HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
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
        form_fields = {
            "model": self.model_name,
            "prompt": prompt,
            "output_format": kwargs.get("output_format", "png"),
            "response_format": "b64_json",
            "size": self._format_ming_size(width, height, **kwargs),
            "watermark": "true" if kwargs.get("watermark") else "false",
        }
        headers = {"Accept": "application/json"}
        authorization = self.headers.get("Authorization")
        if authorization:
            headers["Authorization"] = authorization
        response = requests.post(
            f"{self.v1_base_url}/images/edits",
            data=form_fields,
            files={"image[]": ("input.png", image_bytes, "image/png")},
            headers=headers,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Novita.ai Ming layer decoupling failed: HTTP {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
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
        if not layers:
            raise RuntimeError(
                f"Novita.ai Ming layer decoupling returned no layer images. Response: {str(data)[:300]}"
            )
        return layers, layer_structure

    def _format_ming_size(self, width: Optional[int], height: Optional[int], **kwargs) -> str:
        explicit_size = kwargs.get("size")
        if explicit_size:
            return str(explicit_size)
        if width is None and height is None:
            return "auto"
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

    @staticmethod
    def _extract_task_id(data: Any) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        task_id = data.get("task_id")
        if not task_id and isinstance(data.get("data"), dict):
            task_id = data["data"].get("task_id")
        return task_id

    @staticmethod
    def _extract_task_image(result_json: Dict[str, Any], task_info: Dict[str, Any]) -> Optional[bytes]:
        images = result_json.get("images") or task_info.get("images") or []
        if not images:
            return None
        first_image = images[0]
        if isinstance(first_image, dict):
            if first_image.get("image_base64"):
                return base64.b64decode(first_image["image_base64"])
            if first_image.get("image_url"):
                download = requests.get(first_image["image_url"], timeout=60)
                download.raise_for_status()
                return download.content
            return None
        if isinstance(first_image, str):
            if first_image.startswith("http://") or first_image.startswith("https://"):
                download = requests.get(first_image, timeout=60)
                download.raise_for_status()
                return download.content
            return base64.b64decode(first_image)
        return None

    def _post_with_fallback(self, path_variants: List[str], payload: Dict[str, Any]) -> Dict[str, Any]:
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
            try:
                ASCIIColors.info(f"[{self.binding_name}] Requesting {url}...")
                response = requests.post(url, json=payload, headers=self.headers, timeout=60)
                if response.status_code == 404:
                    detail = response.text[:300] if response.text else "No response body"
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
                raise
            except Exception as ex:
                last_error = ex
                continue
        raise last_error or RuntimeError("All Novita.ai endpoint variants failed.")

    def _poll_task_result(self, task_id: str, timeout: float = 180.0) -> bytes:
        poll_urls = [
            f"{self.base_url}/async/task-result",
            f"{NOVITA_DEFAULT_HOST}/v3/async/task-result",
        ]
        start_time = time.time()
        working_url = poll_urls[0]
        while time.time() - start_time < timeout:
            time.sleep(1.5)
            result_json: Optional[Dict[str, Any]] = None
            for url in [working_url] + [candidate for candidate in poll_urls if candidate != working_url]:
                try:
                    response = requests.get(url, params={"task_id": task_id}, headers=self.headers, timeout=30)
                    if response.status_code == 200:
                        working_url = url
                        result_json = response.json()
                        break
                    if response.status_code != 404:
                        response.raise_for_status()
                except Exception as poll_error:
                    ASCIIColors.warning(f"[{self.binding_name}] Task poll error on {url}: {poll_error}")
                    continue
            if not isinstance(result_json, dict):
                continue
            task_info = result_json.get("task") if isinstance(result_json.get("task"), dict) else {}
            status = task_info.get("status") or result_json.get("status") or ""
            if status in ("TASK_STATUS_SUCCEED", "SUCCEED", "SUCCESS"):
                image_bytes = self._extract_task_image(result_json, task_info)
                if image_bytes is None:
                    raise RuntimeError(f"Novita.ai task succeeded but returned no images. Response: {result_json}")
                return image_bytes
            if status in ("TASK_STATUS_FAILED", "FAILED"):
                reason = task_info.get("reason") or result_json.get("reason") or "Unknown task failure."
                raise RuntimeError(f"Novita.ai task failed: {reason}")
        raise TimeoutError(f"Novita.ai task timed out after {timeout} seconds.")

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
        if not self.api_key:
            return {"status": False, "message": "API key is missing."}
        try:
            response = requests.get(
                f"{self.base_url}/model",
                params={
                    "filter.visibility": "public",
                    "filter.types": "checkpoint",
                    "pagination.limit": 1,
                },
                headers=self.headers,
                timeout=15,
            )
            if response.status_code == 200:
                model_count = len(response.json().get("models") or [])
                return {"status": True, "message": f"API key is valid. Novita AI responded with {model_count} model(s)."}
            return {
                "status": False,
                "message": f"Validation failed: HTTP {response.status_code}: {response.text[:200]}",
            }
        except Exception as e:
            return {"status": False, "message": f"Validation failed: {e}"}

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