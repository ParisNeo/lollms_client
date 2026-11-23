"""OpenAI-compatible Text-to-Image binding for OpenImageGen services."""
import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_tti_binding import LollmsTTIBinding, TTIGenerationResult

BindingName = "OpenImageGenTTIBinding"

_DEFAULT_HOST = "http://localhost:8000"
_ENV_KEY_VAR = "OPENIMAGEGEN_API_KEY"

# Provider-side kwargs forwarded verbatim in the JSON body.
_FORWARDABLE_GENERATION_KWARGS = frozenset({
    "n", "seed", "num_inference_steps", "guidance_scale",
    "steps", "cfg_scale", "scheduler",
    "quality", "style",
    "negative_prompt",
})

# Suffixes that users may include in host_address; we strip them to normalize.
_HOST_PATH_SUFFIXES = ("/v1", "/openai/v1", "/openimagegen/v1")


class OpenImageGenTTIBinding(LollmsTTIBinding):
    """OpenImageGen TTI client (OpenAI Images API contract)."""

    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")
        super().__init__(BindingName, **kwargs)

        global requests
        import pipmaster as pm
        pm.ensure_packages(["requests"])
        import requests as _requests
        requests = _requests

        host = str(kwargs.get("host_address") or _DEFAULT_HOST).rstrip("/")
        for suffix in _HOST_PATH_SUFFIXES:
            if host.endswith(suffix):
                host = host[: -len(suffix)].rstrip("/")
                break

        self.base_address = host
        self.open_ai_host_address = f"{self.base_address}/v1"

        self.model_name = kwargs.get("model_name") or None
        self.service_key = kwargs.get("service_key") or os.getenv(_ENV_KEY_VAR)
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path")

        if not self.verify_ssl_certificate:
            self.verify: Union[bool, str] = False
        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)
            if not cert_path.exists():
                raise FileNotFoundError(f"Certificate file not found: {cert_path}")
            self.verify = str(cert_path)
        else:
            self.verify = self.verify_ssl_certificate

    # ── HTTP helpers ─────────────────────────────────────────────────────
    def _auth_headers(self, *, include_json_content_type: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if include_json_content_type:
            headers["Content-Type"] = "application/json"
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
        return headers

    def _extract_first_image(self, payload: Any) -> bytes:
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected OpenImageGen response shape: {type(payload).__name__}"
            )
        data = payload.get("data") or payload.get("images") or []
        if not data:
            raise ValueError(f"OpenImageGen returned an empty data array: {payload!r}")

        first = data[0]
        if isinstance(first, str):
            return base64.b64decode(first)
        if isinstance(first, dict):
            if first.get("b64_json"):
                return base64.b64decode(first["b64_json"])
            if first.get("url"):
                img_resp = requests.get(first["url"], timeout=120, verify=self.verify)
                img_resp.raise_for_status()
                return img_resp.content
        raise ValueError(f"Unexpected item in OpenImageGen response: {first!r}")

    def _decode_input_image(self, img: Union[str, bytes]) -> bytes:
        if isinstance(img, bytes):
            return img
        if ";base64," in img:
            return base64.b64decode(img.split(";base64,", 1)[1])
        if img.startswith(("http://", "https://")):
            resp = requests.get(img, timeout=120, verify=self.verify)
            resp.raise_for_status()
            return resp.content
        return base64.b64decode(img)

    # ── Generation ───────────────────────────────────────────────────────
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs: Any,
    ) -> bytes:
        size_override = kwargs.pop("size", None)
        if size_override and "x" in str(size_override):
            try:
                w_str, h_str = str(size_override).split("x", 1)
                width, height = int(w_str), int(h_str)
            except ValueError:
                pass
        size_str = f"{width}x{height}"

        response_format = kwargs.pop("response_format", "b64_json")

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "size": size_str,
            "response_format": response_format,
        }
        if self.model_name:
            payload["model"] = self.model_name
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        for k, v in kwargs.items():
            if k in _FORWARDABLE_GENERATION_KWARGS and v is not None and k not in payload:
                payload[k] = v

        response = requests.post(
            f"{self.open_ai_host_address}/images/generations",
            json=payload,
            headers=self._auth_headers(include_json_content_type=True),
            timeout=300,
            verify=self.verify,
        )
        response.raise_for_status()
        return self.process_image(self._extract_first_image(response.json()), **kwargs)

    # ── Editing (multipart) ──────────────────────────────────────────────
    def edit_image(
        self,
        images: Union[str, List[str]],
        prompt: str,
        negative_prompt: Optional[str] = "",
        mask: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any,
    ) -> bytes:
        img_item: Union[str, bytes, list] = (
            images[0] if isinstance(images, list) and images else images
        )
        if not img_item:
            raise ValueError("No valid image provided for edit_image.")
        img_bytes = self._decode_input_image(img_item)

        files: Dict[str, Any] = {"image": ("image.png", img_bytes, "image/png")}
        if mask:
            files["mask"] = ("mask.png", self._decode_input_image(mask), "image/png")

        data: Dict[str, Any] = {
            "prompt": prompt,
            "size": f"{width or 1024}x{height or 1024}",
            "response_format": "b64_json",
        }
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if self.model_name:
            data["model"] = self.model_name

        response = requests.post(
            f"{self.open_ai_host_address}/images/edits",
            files=files,
            data=data,
            headers=self._auth_headers(),
            timeout=300,
            verify=self.verify,
        )
        response.raise_for_status()
        return self.process_image(self._extract_first_image(response.json()), **kwargs)

    # ── Unified entrypoint ───────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        width: int = 1024,
        height: int = 1024,
        images: Optional[Union[str, List[str]]] = None,
        mask: Optional[str] = None,
        n: int = 1,
        modalities: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> TTIGenerationResult:
        all_images: List[bytes] = []
        for _ in range(max(1, n)):
            if images:
                raw = self.edit_image(
                    images=images, prompt=prompt, negative_prompt=negative_prompt,
                    mask=mask, width=width, height=height, **kwargs,
                )
            else:
                raw = self.generate_image(
                    prompt=prompt, negative_prompt=negative_prompt,
                    width=width, height=height, **kwargs,
                )
            all_images.append(raw)
        return TTIGenerationResult(
            images=all_images, text=None, raw=all_images[0] if all_images else None
        )

    # ── Discovery / settings ─────────────────────────────────────────────
    def list_models(self) -> list:
        try:
            response = requests.get(
                f"{self.open_ai_host_address}/models",
                headers=self._auth_headers(),
                timeout=10,
                verify=self.verify,
            )
            if response.status_code != 200:
                return []
            payload = response.json()
        except Exception as ex:
            trace_exception(ex)
            return []

        if isinstance(payload, dict):
            raw_models = payload.get("data", [])
        elif isinstance(payload, list):
            raw_models = payload
        else:
            raw_models = []

        normalized: List[Dict[str, str]] = []
        for m in raw_models:
            if isinstance(m, dict):
                name = m.get("id") or m.get("model_name") or m.get("name")
                if name:
                    normalized.append({"model_name": name})
            elif isinstance(m, str):
                normalized.append({"model_name": m})
        return normalized

    def list_services(self, **kwargs: Any) -> List[Dict[str, str]]:
        return self.list_models()

    def get_settings(self, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs: Any) -> bool:
        self.config.update(settings)
        if "model_name" in settings:
            self.model_name = settings["model_name"]
        if "service_key" in settings:
            self.service_key = settings["service_key"]
        return True

    # ── Resource management (remote service) ────────────────────────────
    def is_local(self) -> bool:
        return False

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        return True

    def has_active_resources(self) -> bool:
        return False

    def get_loaded_models(self) -> List[str]:
        return [self.model_name] if self.model_name else []

    def unload_model(self, model_name: Optional[str] = None) -> bool:
        return True

    def install_model(self, model_name: str, **kwargs: Any) -> dict:
        return {
            "status": False,
            "message": "OpenImageGen is a remote service; models are managed server-side.",
        }