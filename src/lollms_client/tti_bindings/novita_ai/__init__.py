from __future__ import annotations

import os
import time
import requests
import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

from lollms_client.lollms_tti_binding import LollmsTTIBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

pm.ensure_packages(["requests", "pillow"])
from PIL import Image

BindingName = "NovitaAITTIBinding"

NOVITA_AI_MODELS = [
    {"model_name": "sd_xl_base_1.0.safetensors", "display_name": "Stable Diffusion XL 1.0", "description": "Official SDXL 1.0 Base model."},
    {"model_name": "dreamshaper_xl_1_0.safetensors", "display_name": "DreamShaper XL 1.0", "description": "Versatile artistic SDXL model."},
    {"model_name": "juggernaut_xl_v9_rundiffusion.safetensors", "display_name": "Juggernaut XL v9", "description": "High-quality realistic and cinematic model."},
    {"model_name": "realistic_vision_v5.1.safetensors", "display_name": "Realistic Vision v5.1", "description": "Popular photorealistic SD1.5 model."},
    {"model_name": "absolutereality_v1.8.1.safetensors", "display_name": "Absolute Reality v1.8.1", "description": "General-purpose realistic SD1.5 model."},
    {"model_name": "meinamix_meina_v11.safetensors", "display_name": "MeinaMix v11", "description": "High-quality anime illustration model."},
]


class NovitaAITTIBinding(LollmsTTIBinding):
    """Novita.ai TTI binding for LoLLMS with V3 async task polling and full settings lifecycle."""

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
        self.model_name = self.config.get("model_name", "sd_xl_base_1.0.safetensors")
        self.base_url = "https://api.novita.ai/v3"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
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
    # Settings & Service Discovery (Contract with LollmsTTIBinding)
    # ------------------------------------------------------------------

    def list_services(self, **kwargs) -> List[Dict[str, str]]:
        """List provider services exposed by this binding."""
        return [{"name": "Novita.ai TTI", "id": "novita_ai"}]

    def get_settings(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Returns the active configuration dictionary."""
        return self.config

    def set_settings(self, settings: Dict[str, Any], **kwargs) -> bool:
        """Applies updated configuration settings at runtime."""
        if not isinstance(settings, dict):
            return False
        self.config.update(settings)
        new_key = settings.get("api_key") or settings.get("service_key")
        if new_key:
            self.api_key = new_key
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if "model_name" in settings and settings["model_name"]:
            self.model_name = settings["model_name"]
        return True

    def get_zoo(self) -> List[Dict[str, Any]]:
        """Provides available cloud-hosted model checkpoints for the UI zoo."""
        return [
            {
                "name": m.get("display_name", m["model_name"]),
                "description": m.get("description", ""),
                "size": "N/A (Cloud Hosted)",
                "type": "checkpoint",
                "link": m["model_name"],
            }
            for m in NOVITA_AI_MODELS
        ]

    def list_models(self) -> list:
        """
        Lists available models dynamically from Novita AI model query API,
        falling back to the curated list if unavailable or unauthenticated.
        """
        if self.api_key:
            url = f"{self.base_url}/model?filter.visibility=public&pagination.limit=100&filter.types=checkpoint"
            try:
                resp = requests.get(url, headers=self.headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    models_list = data.get("models", [])
                    if isinstance(models_list, list) and models_list:
                        discovered = []
                        for m in models_list:
                            if not isinstance(m, dict):
                                continue
                            sd_name = m.get("sd_name") or m.get("sd_name_in_api") or m.get("name")
                            if sd_name:
                                discovered.append({
                                    "model_name": sd_name,
                                    "display_name": m.get("name") or sd_name,
                                    "description": m.get("description", "") or "Novita AI community checkpoint.",
                                })
                        if discovered:
                            return discovered
            except Exception as ex:
                ASCIIColors.warning(f"[{self.binding_name}] Dynamic model fetch failed: {ex}. Using fallback models.")

        return NOVITA_AI_MODELS

    # ------------------------------------------------------------------
    # Asynchronous Task Execution Pipeline
    # ------------------------------------------------------------------

    def _poll_task_result(self, task_id: str, timeout: float = 120.0) -> bytes:
        """
        Polls the Novita AI V3 async task result endpoint until the image is generated.
        """
        poll_url = f"{self.base_url}/async/task-result?task_id={task_id}"
        start_time = time.time()

        while time.time() - start_time < timeout:
            time.sleep(1.0)
            try:
                poll_resp = requests.get(poll_url, headers=self.headers, timeout=20)
                poll_resp.raise_for_status()
                res_json = poll_resp.json()
            except Exception as poll_ex:
                ASCIIColors.warning(f"[{self.binding_name}] Task poll error: {poll_ex}")
                continue

            task_info = res_json.get("task", {}) if isinstance(res_json, dict) else {}
            status = (
                task_info.get("status")
                or res_json.get("task_status")
                or res_json.get("status")
                or ""
            )

            if status in ("TASK_STATUS_SUCCEED", "SUCCEED", "SUCCESS"):
                images = res_json.get("images", []) or task_info.get("images", [])
                if not images:
                    raise Exception(f"Task succeeded but returned no images. Response: {res_json}")
                first_img = images[0]

                if isinstance(first_img, dict):
                    if first_img.get("image_url"):
                        img_dl = requests.get(first_img["image_url"], timeout=30)
                        img_dl.raise_for_status()
                        return img_dl.content
                    elif first_img.get("image_base64"):
                        return base64.b64decode(first_img["image_base64"])
                elif isinstance(first_img, str):
                    if first_img.startswith("http://") or first_img.startswith("https://"):
                        img_dl = requests.get(first_img, timeout=30)
                        img_dl.raise_for_status()
                        return img_dl.content
                    return base64.b64decode(first_img)

            elif status in ("TASK_STATUS_FAILED", "FAILED"):
                reason = task_info.get("reason") or res_json.get("reason", "Unknown task failure.")
                raise Exception(f"Novita.ai image generation task failed: {reason}")

        raise TimeoutError(f"Novita.ai image generation task timed out after {timeout} seconds.")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> bytes:
        self._require_api_key()

        url = f"{self.base_url}/async/txt2img"
        steps = int(kwargs.get("num_inference_steps", kwargs.get("steps", 25)))
        cfg_scale = float(kwargs.get("guidance_scale", kwargs.get("cfg_scale", 7.0)))
        sampler_name = kwargs.get("sampler_name", "Euler a")
        seed = int(kwargs.get("seed", -1))

        payload = {
            "extra": {
                "response_image_type": kwargs.get("response_image_type", "jpeg"),
                "enable_nsfw_detection": kwargs.get("enable_nsfw_detection", False),
            },
            "request": {
                "model_name": self.model_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": int(width),
                "height": int(height),
                "image_num": 1,
                "steps": steps,
                "guidance_scale": cfg_scale,
                "sampler_name": sampler_name,
                "seed": seed,
            },
        }

        try:
            ASCIIColors.info(f"[{self.binding_name}] Submitting txt2img task to Novita.ai ({self.model_name})...")
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "images" in data and data["images"]:
                b64_image = data["images"][0].get("image_base64")
                if b64_image:
                    return base64.b64decode(b64_image)

            task_id = None
            if isinstance(data, dict):
                task_id = data.get("task_id")
                if not task_id and isinstance(data.get("data"), dict):
                    task_id = data["data"].get("task_id")

            if not task_id:
                raise Exception(f"Novita.ai did not return a valid task_id. Response: {data}")

            return self._poll_task_result(task_id, timeout=kwargs.get("timeout", 120))

        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"Novita.ai generate_image failed: {e}") from e

    def edit_image(
        self,
        images: Union[str, bytes, List[Any]],
        prompt: str,
        negative_prompt: str = "",
        mask: Optional[Union[str, bytes]] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        **kwargs
    ) -> bytes:
        self._require_api_key()

        url = f"{self.base_url}/async/img2img"

        source_img = images[0] if isinstance(images, list) and images else images
        if isinstance(source_img, bytes):
            image_b64 = base64.b64encode(source_img).decode("utf-8")
        elif isinstance(source_img, str):
            image_b64 = re.sub(r"^data:image/[^;]+;base64,", "", source_img)
            if os.path.isfile(source_img):
                image_b64 = base64.b64encode(Path(source_img).read_bytes()).decode("utf-8")
        else:
            raise ValueError("Unsupported image input format for edit_image.")

        mask_b64 = None
        if mask:
            if isinstance(mask, bytes):
                mask_b64 = base64.b64encode(mask).decode("utf-8")
            elif isinstance(mask, str):
                mask_b64 = re.sub(r"^data:image/[^;]+;base64,", "", mask)
                if os.path.isfile(mask):
                    mask_b64 = base64.b64encode(Path(mask).read_bytes()).decode("utf-8")

        steps = int(kwargs.get("num_inference_steps", kwargs.get("steps", 25)))
        cfg_scale = float(kwargs.get("guidance_scale", kwargs.get("cfg_scale", 7.0)))
        sampler_name = kwargs.get("sampler_name", "Euler a")
        seed = int(kwargs.get("seed", -1))

        request_body: Dict[str, Any] = {
            "model_name": self.model_name,
            "image_base64": image_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width or 1024),
            "height": int(height or 1024),
            "image_num": 1,
            "steps": steps,
            "guidance_scale": cfg_scale,
            "sampler_name": sampler_name,
            "seed": seed,
        }
        if mask_b64:
            request_body["mask_image_base64"] = mask_b64

        payload = {
            "extra": {
                "response_image_type": kwargs.get("response_image_type", "jpeg"),
                "enable_nsfw_detection": kwargs.get("enable_nsfw_detection", False),
            },
            "request": request_body,
        }

        try:
            ASCIIColors.info(f"[{self.binding_name}] Submitting img2img edit task to Novita.ai...")
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            task_id = data.get("task_id") or (data.get("data", {}).get("task_id") if isinstance(data.get("data"), dict) else None)
            if not task_id:
                raise Exception(f"Novita.ai did not return a valid task_id for edit. Response: {data}")

            return self._poll_task_result(task_id, timeout=kwargs.get("timeout", 120))

        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"Novita.ai edit_image failed: {e}") from e

    # ------------------------------------------------------------------
    # Management Commands
    # ------------------------------------------------------------------

    def get_user_balance(self) -> Dict[str, Any]:
        """Queries the user account balance from Novita AI."""
        if not self.api_key:
            return {"status": False, "message": "No API key configured."}

        urls = [
            "https://api.novita.ai/openapi/v1/billing/balance/detail",
            "https://api.novita.ai/v3/user/balance",
        ]
        for u in urls:
            try:
                resp = requests.get(u, headers=self.headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    avail = data.get("availableBalance") or data.get("balance") or data.get("data", {}).get("balance")
                    if avail is not None:
                        avail_f = float(avail)
                        usd = avail_f / 10000.0 if avail_f > 1000 else avail_f
                        return {"status": True, "raw_balance": avail, "balance_usd": f"${usd:.2f}", "data": data}
                    return {"status": True, "data": data}
            except Exception:
                continue
        return {"status": False, "message": "Failed to query Novita AI account balance."}

    def validate_key(self) -> Dict[str, Any]:
        """Tests the Novita AI API key against the model endpoint."""
        if not self.api_key:
            return {"status": False, "message": "API key is missing."}
        try:
            models = self.list_models()
            if models:
                return {"status": True, "message": f"API key is valid. Model query returned {len(models)} models."}
        except Exception as e:
            return {"status": False, "message": f"Validation failed: {e}"}
        return {"status": False, "message": "Could not validate key with Novita AI."}

    def get_credits(self) -> Optional[Dict[str, float]]:
        """Fetches remaining credit balance formatted as total_credits and total_usage."""
        bal = self.get_user_balance()
        if bal.get("status"):
            raw = bal.get("raw_balance")
            if raw is not None:
                try:
                    raw_f = float(raw)
                    usd = raw_f / 10000.0 if raw_f > 1000 else raw_f
                    return {"total_credits": usd, "total_usage": 0.0}
                except Exception:
                    pass
        return None