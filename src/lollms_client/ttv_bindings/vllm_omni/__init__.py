"""
vLLM-Omni TTV (Text-to-Video) binding for lollms_client.

Supports video generation models served via vllm-omni:
  - Wan2.1/2.2 T2V, I2V, TI2V
  - HunyuanVideo 1.5
  - LTX-2 / LTX-2.3
  - Cosmos3 T2V/I2V/V2V

Server startup example:
    vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
        --deploy-config vllm_omni/deploy/wan_t2v.yaml \
        --omni --port 8000 --trust-remote-code
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from ascii_colors import trace_exception

from lollms_client.lollms_ttv_binding import LollmsTTVBinding

# Required by LollmsTTVBindingManager – must match the class name below.
BindingName = "VllmOmniTTVBinding"


# ---------------------------------------------------------------------------
# Model zoo: curated list of vLLM-Omni text-to-video models
# ---------------------------------------------------------------------------
_TTV_ZOO: List[Dict[str, Any]] = [
    {
        "name": "Wan2.1-T2V-1.3B",
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "description": "Wan 2.1 text-to-video, 1.3B lightweight model",
        "size": "~3 GB",
        "type": "t2v",
        "link": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    },
    {
        "name": "Wan2.1-T2V-14B",
        "hf_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "description": "Wan 2.1 text-to-video, 14B flagship model",
        "size": "~28 GB",
        "type": "t2v",
        "link": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers",
    },
    {
        "name": "Wan2.2-T2V-A14B",
        "hf_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "description": "Wan 2.2 text-to-video MoE, ~14B active params",
        "size": "~28 GB",
        "type": "t2v",
        "link": "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    },
    {
        "name": "Wan2.2-TI2V-5B",
        "hf_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "description": "Wan 2.2 text-image-to-video, 5B",
        "size": "~10 GB",
        "type": "ti2v",
        "link": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    },
    {
        "name": "Wan2.2-I2V-A14B",
        "hf_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "description": "Wan 2.2 image-to-video MoE, ~14B active params",
        "size": "~28 GB",
        "type": "i2v",
        "link": "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    },
    {
        "name": "HunyuanVideo-1.5-T2V-480p",
        "hf_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "description": "HunyuanVideo 1.5, text-to-video 480p",
        "size": "~30 GB",
        "type": "t2v",
        "link": "https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    },
    {
        "name": "HunyuanVideo-1.5-T2V-720p",
        "hf_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "description": "HunyuanVideo 1.5, text-to-video 720p",
        "size": "~30 GB",
        "type": "t2v",
        "link": "https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    },
    {
        "name": "LTX-2-T2V",
        "hf_id": "Lightricks/LTX-2",
        "description": "LTX-2 text-to-video by Lightricks",
        "size": "~20 GB",
        "type": "t2v",
        "link": "https://huggingface.co/Lightricks/LTX-2",
    },
    {
        "name": "Cosmos3-Nano-T2V",
        "hf_id": "nvidia/Cosmos3-Nano",
        "description": "NVIDIA Cosmos3 Nano: T2I / T2V / I2V / V2V",
        "size": "~8 GB",
        "type": "t2v",
        "link": "https://huggingface.co/nvidia/Cosmos3-Nano",
    },
]


# ---------------------------------------------------------------------------
# Binding implementation
# ---------------------------------------------------------------------------

class VllmOmniTTVBinding(LollmsTTVBinding):
    """
    lollms TTV binding for vLLM-Omni video generation endpoints.

    vLLM-Omni exposes an OpenAI-compatible REST API for diffusion-based
    video generation (Wan, HunyuanVideo, LTX-2, Cosmos3, …).

    The generation endpoint follows the convention:
        POST /v1/video/generate
    with a JSON body mirroring the diffusers pipeline kwargs.

    Parameters
    ----------
    base_url : str
        Base URL of the running vllm-omni server,
        e.g. "http://localhost:8000".
    model : str, optional
        Model identifier forwarded in the request body.
        If None the server's loaded model is used.
    default_num_frames : int
        Default number of frames to request (fps × duration).
    default_height : int
        Default video height in pixels.
    default_width : int
        Default video width in pixels.
    default_fps : int
        Default frames-per-second hint forwarded to the pipeline.
    timeout : int
        HTTP request timeout in seconds (video gen can be slow).
    api_key : str, optional
        Bearer token, if the server requires auth.
    debug : bool
        Enable verbose error traces.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: Optional[str] = None,
        default_num_frames: int = 49,
        default_height: int = 480,
        default_width: int = 832,
        default_fps: int = 16,
        timeout: int = 300,
        api_key: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(binding_name="vllm_omni", debug=debug, **kwargs)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_num_frames = default_num_frames
        self.default_height = default_height
        self.default_width = default_width
        self.default_fps = default_fps
        self.timeout = timeout

        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _build_payload(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        num_frames: int,
        height: int,
        width: int,
        fps: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if self.model:
            payload["model"] = self.model
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None:
            payload["seed"] = seed
        payload.update(extra)
        return payload

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        output_format: str = "mp4",
        **kwargs,
    ) -> bytes:
        """
        Generate a video from a text prompt using vLLM-Omni.

        Parameters
        ----------
        prompt : str
            Text description of the desired video.
        negative_prompt : str, optional
            Negative guidance text.
        num_frames : int, optional
            Number of frames. Defaults to ``self.default_num_frames``.
        height : int, optional
            Video height in pixels.
        width : int, optional
            Video width in pixels.
        fps : int, optional
            Frames per second hint.
        num_inference_steps : int
            Denoising steps (higher = better quality, slower).
        guidance_scale : float
            Classifier-free guidance strength.
        seed : int, optional
            RNG seed for reproducible generation.
        output_format : str
            Container format hint: "mp4" or "webm".
        **kwargs
            Extra pipeline kwargs forwarded verbatim to the server
            (e.g. ``image`` as base64 for I2V models).

        Returns
        -------
        bytes
            Raw video bytes (MP4 / WebM container).

        Raises
        ------
        RuntimeError
            On HTTP error or unexpected server response.
        """
        payload = self._build_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames or self.default_num_frames,
            height=height or self.default_height,
            width=width or self.default_width,
            fps=fps or self.default_fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            extra={"output_format": output_format, **kwargs},
        )

        try:
            resp = self._session.post(
                self._generate_url("/v1/video/generate"),
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"vLLM-Omni generation failed [{resp.status_code}]: {detail}"
            ) from e
        except requests.RequestException as e:
            trace_exception(e)
            raise RuntimeError(f"vLLM-Omni request error: {e}") from e

        # The server may return raw bytes or a JSON envelope with b64 video.
        content_type = resp.headers.get("Content-Type", "")
        if "video" in content_type or "octet-stream" in content_type:
            return resp.content

        # JSON envelope: {"video": "<base64>", "format": "mp4"}
        try:
            data = resp.json()
            if "video" in data:
                return base64.b64decode(data["video"])
            raise RuntimeError(
                f"Unexpected JSON response from vLLM-Omni: {list(data.keys())}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Could not parse vLLM-Omni response. "
                f"Content-Type: {content_type}, body[:200]: {resp.text[:200]}"
            ) from e

    def generate_video_from_image(
        self,
        prompt: str,
        image: Union[bytes, str, Path],
        negative_prompt: Optional[str] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        fps: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """
        Image-to-video generation (I2V / TI2V models).

        Parameters
        ----------
        image : bytes | str | Path
            Input image as raw bytes, a file path, or a base64 string.
        All other params identical to ``generate_video``.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                image_b64 = base64.b64encode(path.read_bytes()).decode()
            else:
                # Assume already base64
                image_b64 = str(image)
        else:
            image_b64 = base64.b64encode(image).decode()

        return self.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image=image_b64,
            **kwargs,
        )

    def list_models(self, **kwargs) -> List[str]:
        """
        Query the running vLLM-Omni server for its loaded model(s).

        Falls back to querying the standard OpenAI /v1/models endpoint.
        Returns only the model ID strings.
        """
        try:
            resp = self._session.get(
                self._generate_url("/v1/models"),
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            if self.debug:
                trace_exception(e)
            # If the server is not reachable, return zoo names as fallback
            return [entry["name"] for entry in _TTV_ZOO]

    # ------------------------------------------------------------------
    # Zoo
    # ------------------------------------------------------------------

    def get_zoo(self) -> List[Dict[str, Any]]:
        """Return the curated list of supported vLLM-Omni video models."""
        return _TTV_ZOO

    def download_from_zoo(
        self,
        index: int,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        """
        Pull a model from HuggingFace Hub using huggingface_hub.

        Parameters
        ----------
        index : int
            Index into the list returned by ``get_zoo()``.
        progress_callback : callable, optional
            Called with ``{"status": str, "progress": float, "model": str}``.

        Returns
        -------
        dict
            ``{"status": True, "path": str}`` on success,
            ``{"status": False, "message": str}`` on failure.
        """
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
            msg = "huggingface_hub is required: pip install huggingface_hub"
            return {"status": False, "message": msg}
        except Exception as e:
            trace_exception(e)
            return {"status": False, "message": str(e)}

    # ------------------------------------------------------------------
    # Health / convenience
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the vLLM-Omni server is reachable and healthy."""
        try:
            resp = self._session.get(
                self._generate_url("/health"), timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def server_info(self) -> Dict[str, Any]:
        """
        Fetch server metadata (model, version, GPU info) from /info or
        /v1/models and return as a dict. Returns empty dict on failure.
        """
        for path in ("/info", "/v1/models"):
            try:
                resp = self._session.get(self._generate_url(path), timeout=5)
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                pass
        return {}
