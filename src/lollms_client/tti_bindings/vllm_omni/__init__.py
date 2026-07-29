# lollms_client/tti_bindings/vllm_omni/__init__.py
import base64
import io
import re
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import requests
from ascii_colors import ASCIIColors, trace_exception

from lollms_client.lollms_tti_binding import LollmsTTIBinding, TTIGenerationResult

BindingName = "VllmOmniTTIBinding"


def _to_data_url(path_or_b64_or_bytes: Union[str, bytes], mime: str = "image/png") -> str:
    """Converts bytes, file paths, or raw base64 strings into a strict data URL."""
    if isinstance(path_or_b64_or_bytes, bytes):
        b64 = base64.b64encode(path_or_b64_or_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    
    s = str(path_or_b64_or_bytes).strip()
    
    if s.startswith("http") or s.startswith("data:"):
        return s
        
    # Guard against Errno 36: only check the filesystem if the string is short enough 
    # to be a valid path and lacks typical base64 characters (like '+', '/', '=' at the end).
    # OS max path is ~260 on Windows, 4096 on Linux. We use 1024 as a safe threshold.
    if len(s) < 1024:
        p = Path(s)
        if p.exists():
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:{mime};base64,{b64}"
        
    # Assume it's raw base64. Strip any accidental whitespace/newlines
    s = s.replace("\n", "").replace("\r", "").replace(" ", "")
    return f"data:{mime};base64,{s}"

EDIT_ONLY_MODEL_PATTERNS = ["qwen-image-edit", "-edit-plus", "-edit-2511", "-edit-2509"]

def _is_edit_only_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(p in name for p in EDIT_ONLY_MODEL_PATTERNS)

def _generate_noise_image_data_url(width: int = 1024, height: int = 1024) -> str:
    """Generates a random noise image and returns it as a base64 data URL."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class VllmOmniTTIBinding(LollmsTTIBinding):
    """
    LoLLMS TTI binding for models served via vLLM-Omni's OpenAI-compatible
    server (https://docs.vllm.ai/projects/vllm-omni/).

    Works with:
      - Pure diffusion / image models (Z-Image, FLUX, Ovis-Image, GLM-Image)
      - True omni chat models (Qwen3-Omni, BAGEL) that can return text
        alongside generated images in one /v1/chat/completions call.
    """

    def __init__(self,
                 host_address: str = "http://localhost:8091",
                 model_name: str = "",
                 service_key: Optional[str] = None,
                 verify_ssl_certificate: bool = True,
                 **kwargs):
        super().__init__(
            binding_name="vllm_omni",
            supports_omni=True,
            supports_text_output=True,
            **kwargs
        )
        self.host_address = host_address.rstrip("/")
        self.model_name = model_name
        self.service_key = service_key
        self.verify_ssl_certificate = verify_ssl_certificate

        # Model-level defaults, overridable per-call via kwargs
        self.default_num_inference_steps = kwargs.get("num_inference_steps", 30)
        self.default_guidance_scale = kwargs.get("guidance_scale", 7.5)
        self.default_seed = kwargs.get("seed", -1)
        self.default_modalities = kwargs.get("modalities", ["image"])

        self.config = kwargs

    # ------------------------------------------------------------------
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.service_key:
            h["Authorization"] = f"Bearer {self.service_key}"
        return h

    def _extract_images_and_text(self, resp_json: Dict[str, Any]):
        images, text_parts = [], []
        for choice in resp_json.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content:
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype in ("image_url", "image"):
                        url = part.get("image_url", {}).get("url") or part.get("url")
                        if url and url.startswith("data:"):
                            b64 = url.split(",", 1)[1]
                            images.append(base64.b64decode(b64))
                        elif url:
                            r = requests.get(url, verify=self.verify_ssl_certificate)
                            images.append(r.content)
            # some diffusion-only responses put base64 image directly at message level
            if "images" in msg:
                for img_entry in msg["images"]:
                    b64 = img_entry.get("b64_json") or img_entry.get("data")
                    if b64:
                        images.append(base64.b64decode(b64))
        return images, ("\n".join(t for t in text_parts if t) or None)

    def _build_content(self, prompt: str, images: Optional[Union[str, List[str]]] = None):
        content = [{"type": "text", "text": prompt}]
        if images:
            img_list = images if isinstance(images, list) else [images]
            for img in img_list:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": _to_data_url(img)}
                })
        return content

    # ------------------------------------------------------------------
    # Unified omni entrypoint (preferred)
    # ------------------------------------------------------------------
    def generate(self,
                 prompt: str,
                 negative_prompt: Optional[str] = "",
                 width: int = 1024,
                 height: int = 1024,
                 images: Optional[Union[str, List[str]]] = None,
                 mask: Optional[str] = None,
                 n: int = 1,
                 modalities: Optional[List[str]] = None,
                 **kwargs) -> TTIGenerationResult:
        url = f"{self.host_address}/v1/chat/completions"
        
        active_model = kwargs.get("model_name", self.model_name)

        # Edit-only models (like QwenImageEditPlus) crash with NoneType if no image is provided.
        # We synthesize a random noise image to satisfy the pipeline's internal `.size` constraint.
        if not images and _is_edit_only_model(active_model):
            ASCIIColors.warning(f"[VllmOmni] Model '{active_model}' is edit-only. Synthesizing noise reference image.")
            images = _generate_noise_image_data_url(width=width, height=height)

        extra_body = {
            "sampling_params_list": [{
                "num_inference_steps": kwargs.get("num_inference_steps", self.default_num_inference_steps),
                "guidance_scale": kwargs.get("guidance_scale", self.default_guidance_scale),
                "negative_prompt": negative_prompt or "",
                "seed": kwargs.get("seed", self.default_seed),
                "width": width,
                "height": height,
                "n": n,
            }]
        }

        payload = {
            "model": active_model,
            "messages": [{"role": "user", "content": self._build_content(prompt, images)}],
            "modalities": modalities or self.default_modalities,
            "extra_body": extra_body,
        }

        try:
            resp = requests.post(url, json=payload, headers=self._headers(),
                                  verify=self.verify_ssl_certificate, timeout=kwargs.get("timeout", 600))
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            trace_exception(e)
            ASCIIColors.error(f"vLLM-Omni generation request failed: {e}")
            return TTIGenerationResult(images=[], text=None, raw=str(e))

        raw_images, text = self._extract_images_and_text(data)
        processed = [self.process_image(img, **kwargs) for img in raw_images]
        return TTIGenerationResult(images=processed, text=text, raw=data, metadata={"model": payload["model"]})

    # ------------------------------------------------------------------
    # Legacy retrocompatible wrappers
    # ------------------------------------------------------------------
    def generate_image(self,
                       prompt: str,
                       negative_prompt: Optional[str] = "",
                       width: int = 1024,
                       height: int = 1024,
                       **kwargs) -> bytes:
        
        result = self.generate(prompt=prompt, negative_prompt=negative_prompt,
                                width=width, height=height, modalities=["image"], **kwargs)
        return result.first_image_bytes()

    def edit_image(self,
                   images: Union[str, List[str]],
                   prompt: str,
                   negative_prompt: Optional[str] = "",
                   mask: Optional[str] = None,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   **kwargs) -> bytes:
        result = self.generate(prompt=prompt, negative_prompt=negative_prompt,
                                width=width or 1024, height=height or 1024,
                                images=images, mask=mask, modalities=["image"], **kwargs)
        return result.first_image_bytes()

    # ------------------------------------------------------------------
    # Service / model management
    # ------------------------------------------------------------------
    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        return [{"name": self.model_name, "host": self.host_address}]

    def list_models(self) -> list:
        try:
            resp = requests.get(f"{self.host_address}/v1/models", headers=self._headers(),
                                 verify=self.verify_ssl_certificate, timeout=15)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception as e:
            trace_exception(e)
            return []

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        return {
            "host_address": self.host_address,
            "model_name": self.model_name,
            "num_inference_steps": self.default_num_inference_steps,
            "guidance_scale": self.default_guidance_scale,
            "seed": self.default_seed,
            "modalities": self.default_modalities,
        }

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        try:
            self.host_address = settings.get("host_address", self.host_address).rstrip("/")
            self.model_name = settings.get("model_name", self.model_name)
            self.default_num_inference_steps = settings.get("num_inference_steps", self.default_num_inference_steps)
            self.default_guidance_scale = settings.get("guidance_scale", self.default_guidance_scale)
            self.default_seed = settings.get("seed", self.default_seed)
            self.default_modalities = settings.get("modalities", self.default_modalities)
            return True
        except Exception as e:
            trace_exception(e)
            return False
