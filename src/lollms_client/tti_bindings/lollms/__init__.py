import base64
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from lollms_client.lollms_tti_binding import LollmsTTIBinding, TTIGenerationResult
import os
import ssl
from ascii_colors import trace_exception, ASCIIColors

BindingName = "LollmsTTIBinding"


class LollmsTTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")
        super().__init__(BindingName, **kwargs)

        host = kwargs.get("host_address", "http://localhost:9642").rstrip("/")

        if host.endswith("/lollms/v1"):
            host = host[: -len("/lollms/v1")].rstrip("/")
        elif host.endswith("/v1"):
            host = host[: -len("/v1")].rstrip("/")

        self.base_address = host
        self.open_ai_host_address = f"{self.base_address}/v1"
        self.lollms_host_address = f"{self.base_address}/lollms/v1"

        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path")

        if not self.service_key:
            self.service_key = os.getenv("LOLLMS_API_KEY")

        if not self.verify_ssl_certificate:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            self.verify = ssl_context
        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)
            if not cert_path.exists():
                raise FileNotFoundError(f"Certificate file not found: {cert_path}")
            ssl_context = ssl.create_default_context(cafile=str(cert_path))
            self.verify = ssl_context
        else:
            self.verify = self.verify_ssl_certificate

    # ── Helpers ───────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {}
        if self.service_key:
            h["Authorization"] = f"Bearer {self.service_key}"
        return h

    def _lollms_get(self, path: str, timeout: int = 10) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.get(url, headers=self._headers(), timeout=timeout, verify=self.verify)
        response.raise_for_status()
        return response.json()

    def _lollms_post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        url = f"{self.lollms_host_address}{path}"
        response = requests.post(url, json=payload, headers=self._headers(), timeout=timeout, verify=self.verify)
        response.raise_for_status()
        return response.json()

    # ── Capabilities ──────────────────────────────────────────────────────

    def get_capabilities(self) -> Dict:
        try:
            return self._lollms_get("/capabilities", timeout=10)
        except Exception as e:
            ASCIIColors.warning(f"Failed to fetch capabilities: {e}")
            return {"capabilities": [], "active_bindings": {}}

    # ── Per-Binding Model Listing ─────────────────────────────────────────

    def list_models_by_type(self, binding_type: str = "tti", binding_alias: Optional[str] = None) -> List[Dict]:
        params = {}
        if binding_alias:
            params["binding_alias"] = binding_alias
        try:
            url = f"{self.lollms_host_address}/{binding_type}/models"
            response = requests.get(
                url, headers=self._headers(), params=params, timeout=15, verify=self.verify
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            ASCIIColors.warning(f"Failed to list {binding_type} models: {e}")
            return []

    # ── Image Generation (OpenAI-compatible) ─────────────────────────────

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs,
    ) -> bytes:
        url = f"{self.open_ai_host_address}/images/generations"
        headers = self._headers()

        size = f"{width}x{height}"

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": self.model_name,
            "size": size,
            "response_format": "b64_json",
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        for k, v in kwargs.items():
            if k not in payload and v is not None and k not in (
                "watermark_path", "watermark_size_x", "watermark_size_y",
                "watermark_pos_x", "watermark_pos_y", "author", "system", "metadata",
            ):
                payload[k] = v

        response = requests.post(url, json=payload, headers=headers, timeout=300, verify=self.verify)
        response.raise_for_status()
        res_data = response.json()

        b64_data = res_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)

        return self.process_image(image_bytes, **kwargs)

    # ── Image Edit (OpenAI-compatible multipart) ──────────────────────────

    def edit_image(
        self,
        images: Union[str, List[str]],
        prompt: str,
        negative_prompt: Optional[str] = "",
        mask: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        url = f"{self.open_ai_host_address}/images/edits"
        headers = self._headers()

        target_img_b64 = None
        if isinstance(images, list) and images:
            img_item = images[0]
        else:
            img_item = images

        if isinstance(img_item, str):
            if ";base64," in img_item:
                target_img_b64 = img_item.split(";base64,")[1]
            else:
                target_img_b64 = img_item

        if not target_img_b64:
            raise ValueError("No valid image provided for edit_image.")

        img_bytes = base64.b64decode(target_img_b64)

        files: Dict[str, Any] = {
            "image": ("image.png", img_bytes, "image/png"),
        }
        if mask:
            mask_b64 = mask.split(";base64,")[1] if ";base64," in mask else mask
            files["mask"] = ("mask.png", base64.b64decode(mask_b64), "image/png")

        w = width or 1024
        h = height or 1024
        size_str = f"{w}x{h}"

        data: Dict[str, Any] = {
            "prompt": prompt,
            "size": size_str,
            "response_format": "b64_json",
        }
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if self.model_name:
            data["model"] = self.model_name

        response = requests.post(url, files=files, data=data, headers=headers, timeout=300, verify=self.verify)
        response.raise_for_status()
        res_data = response.json()
        b64_data = res_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)

        return self.process_image(image_bytes, **kwargs)

    # ── Image Edit (LoLLMS-native JSON endpoint) ──────────────────────────

    def edit_image_lollms(
        self,
        prompt: str,
        image_b64: str,
        mask_b64: Optional[str] = None,
        model: Optional[str] = None,
    ) -> bytes:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "image": image_b64,
        }
        if mask_b64:
            payload["mask"] = mask_b64
        if model or self.model_name:
            payload["model"] = model or self.model_name

        data = self._lollms_post("/images/edit", payload, timeout=300)
        b64_result = data.get("data", [{}])[0].get("b64_json", "")
        if not b64_result:
            raise ValueError("LoLLMS image edit returned no image data.")
        return base64.b64decode(b64_result)

    # ── Unified generate() override ───────────────────────────────────────

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
        **kwargs,
    ) -> TTIGenerationResult:
        all_images: List[bytes] = []
        for _ in range(max(1, n)):
            if images:
                raw = self.edit_image(
                    images=images,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    mask=mask,
                    width=width,
                    height=height,
                    **kwargs,
                )
            else:
                raw = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    **kwargs,
                )
            all_images.append(raw)

        return TTIGenerationResult(images=all_images, text=None, raw=all_images[0] if all_images else None)

    # ── Models / Services / Settings ──────────────────────────────────────

    def list_models(self) -> list:
        url = f"{self.open_ai_host_address}/models"
        headers = self._headers()
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=self.verify)
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                return [{"model_name": m["id"]} for m in models_data]
        except Exception as ex:
            trace_exception(ex)
        return []

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return self.list_models()

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        self.config.update(settings)
        if "model_name" in settings:
            self.model_name = settings["model_name"]
        return True