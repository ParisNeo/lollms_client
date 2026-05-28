import base64
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from lollms_client.lollms_tti_binding import LollmsTTIBinding
import os
import ssl
from ascii_colors import trace_exception
BindingName = "LollmsTTIBinding"

class LollmsTTIBinding(LollmsTTIBinding):
    def __init__(self, **kwargs):
        # Allow 'model' as an alias for 'model_name'
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(BindingName, **kwargs)

        host = kwargs.get("host_address", "http://localhost:9642").rstrip("/")

        if host.endswith("/lollms/v1"):
            host = host[:-10].rstrip("/")
        elif host.endswith("/v1"):
            host = host[:-3].rstrip("/")

        self.base_address = host
        self.open_ai_host_address = f"{self.base_address}/v1"
        self.lollms_host_address = f"{self.base_address}/lollms/v1"

        self.model_name = kwargs.get("model_name")
        self.service_key = kwargs.get("service_key")
        self.verify_ssl_certificate = kwargs.get("verify_ssl_certificate", True)
        self.certificate_file_path = kwargs.get("certificate_file_path")

        if not self.service_key:
            self.service_key = os.getenv("LOLLMS_API_KEY")

        self.verify = True

        if not self.verify_ssl_certificate:
            self.verify = False

        elif self.certificate_file_path:
            cert_path = Path(self.certificate_file_path)

            if not cert_path.exists():
                raise FileNotFoundError(
                    f"Certificate file not found: {cert_path}"
                )

            ssl_context = ssl.create_default_context(
                cafile=str(cert_path)
            )

            self.verify = ssl_context


    def generate_image(self, prompt: str, negative_prompt: Optional[str] = "", width: int = 1024, height: int = 1024, **kwargs) -> bytes:
        url = f"{self.open_ai_host_address}/images/generations"
        headers = {}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"

        # Convert width/height to size string format expected by server
        size = f"{width}x{height}"

        payload = {
            "prompt": prompt,
            "model": None,# self.model_name or None,
            "size": size,
            "response_format": "b64_json"
        }
        for k, v in kwargs.items():
            if k not in payload and v is not None:
                payload[k] = v

        import sys
        print(f"[DEBUG] Sending payload to {url}: {payload}", file=sys.stderr)

        response = requests.post(url, json=payload, headers=headers, timeout=300, verify=self.verify)
        print(f"[DEBUG] Response status: {response.status_code}", file=sys.stderr)
        if response.status_code != 200:
            print(f"[DEBUG] Response body: {response.text}", file=sys.stderr)
        response.raise_for_status()
        res_data = response.json()

        # Handle URL response format - download the image
        b64_data = res_data["data"][0]["b64_json"]
        image_bytes = base64.b64decode(b64_data)
        
        return self.process_image(image_bytes, **kwargs)

    def edit_image(self, images: Union[str, List[str]], prompt: str, negative_prompt: Optional[str] = "", mask: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None, **kwargs) -> bytes:
        url = f"{self.open_ai_host_address}/images/edits"
        headers = {}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"

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

        files = {
            "image": ("image.png", img_bytes, "image/png")
        }
        if mask:
            mask_b64 = mask.split(";base64,")[1] if ";base64," in mask else mask
            files["mask"] = ("mask.png", base64.b64decode(mask_b64), "image/png")

        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width or 1024,
            "height": height or 1024,
            "model": self.model_name or None
        }

        response = requests.post(url, files=files, data=data, headers=headers, timeout=300, verify=self.verify)
        response.raise_for_status()
        res_data = response.json()
        b64_data = res_data["data"][0]["b64_json"]
        return base64.b64decode(b64_data)

    def list_models(self) -> list:
        url = f"{self.open_ai_host_address}/models"
        headers = {}
        if self.service_key:
            headers["Authorization"] = f"Bearer {self.service_key}"
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
