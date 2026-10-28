"""
MiniMax-H3 TTV (Text-to-Video) binding for lollms_client.

Supports video generation models served via vllm-omni:
  - MiniMax-H3 (FL2VA / T2VA partition)

Server startup example:
    vllm serve ParisNeo/MiniMax-H3-FL2VA-FP8 \
        --omni --task-type fl2va --trust-remote-code \
        --enable-layerwise-offload --cache-backend cache_dit \
        --port 8002
"""

from __future__ import annotations
import io
import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from ascii_colors import trace_exception

from lollms_client.lollms_ttv_binding import LollmsTTVBinding

# Required by LollmsTTVBindingManager – must match the class name below.
BindingName = "MiniMaxH3TTVBinding"


# ---------------------------------------------------------------------------
# Model zoo: curated list of MiniMax-H3 video models
# ---------------------------------------------------------------------------
_TTV_ZOO: List[Dict[str, Any]] = [
    {
        "name": "MiniMax-H3-FL2VA-FP8",
        "hf_id": "ParisNeo/MiniMax-H3-FL2VA-FP8",
        "description": "Optimized FP8 quantized MiniMax-H3 Text-to-Video model for vLLM-Omni",
        "size": "~35-40 GB",
        "type": "t2va",
        "link": "https://huggingface.co/ParisNeo/MiniMax-H3-FL2VA-FP8",
    },
    {
        "name": "MiniMax-H3 (Original Base)",
        "hf_id": "MiniMaxAI/MiniMax-H3",
        "description": "Original unquantized MiniMax-H3 multimodal model base",
        "size": "~100+ GB",
        "type": "t2va",
        "link": "https://huggingface.co/MiniMaxAI/MiniMax-H3",
    },
]


# ---------------------------------------------------------------------------
# Binding implementation
# ---------------------------------------------------------------------------

class MiniMaxH3TTVBinding(LollmsTTVBinding):
    """
    lollms TTV binding for MiniMax-H3 video generation endpoints via vLLM-Omni.

    Parameters
    ----------
    base_url : str
        Base URL of the running vllm-omni server,
        e.g. "http://localhost:8002".
    model : str, optional
        Model identifier forwarded in the request body.
    default_aspect_ratio : str
        Default video aspect ratio (e.g. "16:9", "9:16", "1:1").
    default_duration : float
        Default video duration in seconds (must be between 4.0 and 15.0).
    default_fps : int
        Fixed frames per second (MiniMax-H3 enforces 24 fps).
    timeout : int
        HTTP request timeout in seconds (video generation can be slow).
    api_key : str, optional
        Bearer token, if the server requires auth.
    debug : bool
        Enable verbose error traces.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002",
        model: Optional[str] = "ParisNeo/MiniMax-H3-FL2VA-FP8",
        default_aspect_ratio: str = "16:9",
        default_duration: float = 4.0,
        default_fps: int = 24,
        timeout: int = 600,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(binding_name="minimax_h3", debug=debug, **kwargs)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_aspect_ratio = default_aspect_ratio
        self.default_duration = default_duration
        self.default_fps = default_fps
        self.timeout = timeout

        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        output_format: str = "mp4",
        task: str = "t2va",
        **kwargs,
    ) -> bytes:
        """
        Generate a video from a text prompt using MiniMax-H3 via vLLM-Omni.

        Parameters
        ----------
        prompt : str
            Text description of the desired video.
        negative_prompt : str, optional
            Negative guidance text.
        aspect_ratio : str, optional
            Video aspect ratio string ("16:9", "9:16", "1:1", "4:3", "21:9").
        duration : float, optional
            Video duration in seconds (4.0 to 15.0).
        fps : int, optional
            Frames per second (MiniMax-H3 strictly requires 24).
        num_inference_steps : int
            Denoising steps.
        seed : int, optional
            RNG seed for reproducible generation.
        output_format : str
            Container format hint: "mp4" or "webm".
        task : str
            Generation task type: "t2va" (text-to-video) or "fl2va" (first-last frame).

        Returns
        -------
        bytes
            Raw video bytes (MP4 container).
        """
        resolved_aspect_ratio = aspect_ratio or self.default_aspect_ratio
        resolved_duration = duration if duration is not None else self.default_duration
        resolved_fps = fps or self.default_fps

        # Build multipart/form-data payload as required by vLLM-Omni video endpoints
        data: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": resolved_aspect_ratio,
            "fps": resolved_fps,
            "num_inference_steps": num_inference_steps,
        }

        if self.model:
            data["model"] = self.model

        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        if seed is not None and seed != -1:
            data["seed"] = seed

        # MiniMax-H3 requires task and duration inside extra_params JSON payload
        extra_params = {
            "task": task,
            "duration": resolved_duration,
        }
        extra_params.update(kwargs.get("extra_params", {}))
        data["extra_params"] = json.dumps(extra_params)

        try:
            resp = self._session.post(
                self._generate_url("/v1/videos/sync"),
                data=data,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"MiniMax-H3 generation failed [{resp.status_code}]: {detail}"
            ) from e
        except requests.RequestException as e:
            trace_exception(e)
            raise RuntimeError(f"MiniMax-H3 request error: {e}") from e

        content_type = resp.headers.get("Content-Type", "")
        if "video" in content_type or "octet-stream" in content_type:
            return resp.content

        # Fallback for JSON envelope
        try:
            res_json = resp.json()
            if "video" in res_json:
                return base64.b64decode(res_json["video"])
            raise RuntimeError(f"Unexpected JSON response from server: {list(res_json.keys())}")
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Could not parse response. Content-Type: {content_type}, body[:200]: {resp.text[:200]}"
            ) from e

    def generate_video_from_image(
        self,
        prompt: str,
        image: Union[bytes, str, Path],
        negative_prompt: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """
        First/Last Frame to Video generation (FL2VA mode) for MiniMax-H3.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                image_bytes = path.read_bytes()
            else:
                image_bytes = base64.b64decode(image)
        else:
            image_bytes = image

        # Pass image file bytes via files parameter in multipart form data
        resolved_aspect_ratio = aspect_ratio or self.default_aspect_ratio
        resolved_duration = duration if duration is not None else self.default_duration
        resolved_fps = fps or self.default_fps

        data: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": resolved_aspect_ratio,
            "fps": resolved_fps,
            "num_inference_steps": num_inference_steps,
        }

        if self.model:
            data["model"] = self.model

        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        if seed is not None and seed != -1:
            data["seed"] = seed

        extra_params = {
            "task": "fl2va",
            "duration": resolved_duration,
        }
        data["extra_params"] = json.dumps(extra_params)

        files = {
            "image": ("input_image.png", io.BytesIO(image_bytes), "image/png")
        }

        try:
            resp = self._session.post(
                self._generate_url("/v1/videos/sync"),
                data=data,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"MiniMax-H3 FL2VA generation failed: {detail}") from e
        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"MiniMax-H3 request error: {e}") from e

    def list_models(self, **kwargs) -> List[str]:
        """Query running vLLM server for models via OpenAI-compatible endpoint."""
        try:
            resp = self._session.get(self._generate_url("/v1/models"), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            if self.debug:
                trace_exception(e)
            return [entry["name"] for entry in _TTV_ZOO]

    def get_zoo(self) -> List[Dict[str, Any]]:
        return _TTV_ZOO

    def download_from_zoo(
        self,
        index: int,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        if index < 0 or index >= len(_TTV_ZOO):
            return {"status": False, "message": f"Index {index} out of range"}

        entry = _TTV_ZOO[index]
        hf_id = entry["hf_id"]

        try:
            from huggingface_hub import snapshot_download

            if progress_callback:
                progress_callback({"status": "starting", "progress": 0.0, "model": hf_id})

            local_dir = snapshot_download(
                repo_id=hf_id,
                ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            )

            if progress_callback:
                progress_callback({"status": "done", "progress": 1.0, "model": hf_id})

            return {"status": True, "path": local_dir, "model": hf_id}

        except ImportError:
            return {"status": False, "message": "huggingface_hub is required: pip install huggingface_hub"}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": str(e)}

    def health_check(self) -> bool:
        try:
            resp = self._session.get(self._generate_url("/health"), timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def server_info(self) -> Dict[str, Any]:
        for path in ("/info", "/v1/models"):
            try:
                resp = self._session.get(self._generate_url(path), timeout=5)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return {}