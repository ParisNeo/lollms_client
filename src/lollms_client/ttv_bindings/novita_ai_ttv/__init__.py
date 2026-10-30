import os
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

try:
    import requests
except ImportError:
    requests = None

from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, MediaInput

SUPPORTED_MODELS: List[Dict[str, Any]] = [
    {"name": "wan2.2-t2v", "description": "Wan 2.2 Text-to-Video (latest open-weights SOTA)", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["480p", "720p", "1080p"], "supports_duration": True},
    {"name": "wan2.1-t2v", "description": "Wan 2.1 Text-to-Video (legacy, kept for back-compat)", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["480p", "720p"], "supports_duration": True},
    {"name": "kling-v3.0-std-t2v", "description": "Kling 3.0 Standard Text-to-Video (accepts camera_control extras)", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["720p", "1080p"], "supports_duration": True},
    {"name": "kling-v2.1-std-t2v", "description": "Kling 2.1 Standard Text-to-Video", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["720p", "1080p"], "supports_duration": True},
    {"name": "pixverse-v4.5-t2v", "description": "PixVerse v4.5 Text-to-Video", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["480p", "720p", "1080p"], "supports_duration": True},
    {"name": "hunyuan-video-fast", "description": "Tencent Hunyuan Video Fast (fixed duration)", "aspect_ratios": ["16:9", "9:16"], "resolutions": ["480p", "720p"], "supports_duration": False},
    {"name": "vidu-q1-t2v", "description": "Vidu Q1 Text-to-Video (fixed duration)", "aspect_ratios": ["16:9", "9:16"], "resolutions": ["720p", "1080p"], "supports_duration": False},
    {"name": "seedance-1-0-lite-t2v", "description": "ByteDance Seedance 1.0 Lite T2V (fixed duration)", "aspect_ratios": ["16:9", "9:16", "1:1"], "resolutions": ["480p", "720p", "1080p"], "supports_duration": False},
    {"name": "minimax-hailuo-02-t2v", "description": "MiniMax Hailuo 02 Text-to-Video", "aspect_ratios": ["16:9", "9:16"], "resolutions": ["720p", "1080p"], "supports_duration": True},
]

DEFAULT_MODEL = "wan2.2-t2v"

_MODEL_ALIASES: Dict[str, str] = {
    "wan-t2v": "wan2.2-t2v",
    "wan2-t2v": "wan2.2-t2v",
    "kling-t2v": "kling-v3.0-std-t2v",
    "hailuo-t2v": "minimax-hailuo-02-t2v",
}

BindingName = "NovitaAITTV"


class NovitaAITTV(LollmsTTVBinding):
    def __init__(self, **kwargs: Any):
        super().__init__(
            binding_name="novita_ai_ttv",
            debug=bool(kwargs.get("debug", False))
        )
        if requests is None:
            raise ImportError("novita_ai_ttv requires 'requests': pip install requests")
        self.config = kwargs or {}
        self.api_key = (
            self.config.get("novita_api_key")
            or self.config.get("api_key")
            or os.environ.get("NOVITA_API_KEY", "")
        )
        self.base_url = self.config.get("base_url", "https://api.novita.ai/v3/async")
        self.output_folder = Path(self.config.get("output_folder", "./data/ttv_outputs/novita_ai"))
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _headers(self, key_override: Optional[str] = None) -> Dict[str, str]:
        key = key_override or self.api_key
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

    def _shape_payload(
        self,
        model: str,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: Optional[str],
        duration: Optional[int],
        resolution: Optional[str],
        seed: Optional[int],
    ) -> Dict[str, Any]:
        """Builds the per-model request payload, omitting unsupported fields."""
        payload: Dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect_ratio}
        entry = next((m for m in SUPPORTED_MODELS if m["name"] == model), None)
        if duration is not None and (entry is None or entry.get("supports_duration", True)):
            payload["duration"] = duration
        if resolution:
            payload["resolution"] = resolution
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if seed is not None and seed != -1:
            payload["seed"] = seed
        extras = (self.config.get("model_extras") or {}).get(model, {})
        payload.update(extras)
        return {k: v for k, v in payload.items() if v is not None}

    # --- Declared Commands ---

    def test_connection(self, api_key: str = "") -> Dict[str, Any]:
        """Command: test_connection"""
        target_key = api_key or self.api_key
        if not target_key:
            return {"status": False, "balance": 0.0, "message": "No API key configured."}

        try:
            url = "https://api.novita.ai/v3/user/info"
            res = requests.get(url, headers=self._headers(target_key), timeout=10)
            if res.status_code == 200:
                data = res.json()
                balance = float(data.get("credit_balance", data.get("credit", 0))) / 100.0  # Converted to USD
                return {"status": True, "balance": balance, "message": "Connection OK"}
            return {"status": False, "balance": 0.0, "message": f"API error: {res.status_code} {res.text[:300]}"}
        except Exception as e:
            return {"status": False, "balance": 0.0, "message": str(e)}

    def list_models(self, **kwargs: Any) -> Dict[str, Any]:
        """Command: list_models (an instance method — the manager calls binding.list_models())."""
        return {"models": SUPPORTED_MODELS}

    def check_task_status(self, task_id: str) -> Dict[str, Any]:
        """Command: check_task_status"""
        try:
            poll_url = f"{self.base_url}/task-result?task_id={task_id}"
            res = requests.get(poll_url, headers=self._headers(), timeout=10)
            res.raise_for_status()
            data = res.json()

            task_info = data.get("task", {})
            status = task_info.get("status", "UNKNOWN")
            real_progress = task_info.get("progress")
            if isinstance(real_progress, (int, float)):
                progress = max(0, min(99, int(real_progress)))
            else:
                progress = None
            video_url = ""

            if status in ["TASK_STATUS_SUCCEED", "SUCCEED"]:
                videos = data.get("videos") or data.get("data", {}).get("videos", [])
                if videos:
                    video_url = videos[0].get("video_url", "")
                return {"status": status, "progress": 100, "video_url": video_url, "message": "Ready"}

            if status in ["TASK_STATUS_FAILED", "FAILED"]:
                return {"status": status, "progress": 0, "video_url": "", "message": task_info.get("reason", "Failed")}

            return {
                "status": status,
                "progress": progress if progress is not None else 50,
                "video_url": "",
                "message": "In progress",
            }
        except Exception as e:
            return {"status": "ERROR", "progress": 0, "video_url": "", "message": str(e)}

    # --- Core Generation Method ---

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: Optional[str] = None,
        duration: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        timeout: Optional[int] = None,
        poll_interval: Optional[int] = None,
        cancel_event: Optional[Any] = None,
    ) -> Optional[str]:
        """Core TTV pipeline.

        Note: diverges from the abstract `-> bytes` annotation by returning a local
        .mp4 path string (None on failure), matching the sd.cpp binding convention.
        """
        model_name = model or self.config.get("model", DEFAULT_MODEL)
        model_name = _MODEL_ALIASES.get(model_name, model_name)
        ar = aspect_ratio or self.config.get("aspect_ratio", "16:9")
        dur = duration if duration is not None else self.config.get("duration", 5)
        res_tier = resolution or self.config.get("resolution", "")
        sd = seed if seed is not None else self.config.get("seed", -1)
        neg_prompt = negative_prompt or self.config.get("negative_prompt", "")
        timeout = timeout or self.config.get("timeout_s", 900)
        interval = poll_interval or self.config.get("poll_interval_s", 3)

        payload = self._shape_payload(model_name, prompt, neg_prompt, ar, dur, res_tier, sd)

        if progress_callback:
            progress_callback(5, f"Submitting task to {model_name}...")

        try:
            req_url = f"{self.base_url}/{model_name}"
            resp = requests.post(req_url, headers=self._headers(), json=payload, timeout=30)
            if resp.status_code >= 400:
                ASCIIColors.error(
                    f"Novita submit failed: HTTP {resp.status_code} — {resp.text[:500]}"
                )
                resp.raise_for_status()
            resp_json = resp.json()
            task_id = resp_json.get("task_id") or resp_json.get("data", {}).get("task_id")
            if not task_id:
                ASCIIColors.error("Novita submit returned no task_id.")
                return None
        except Exception as e:
            trace_exception(e)
            return None

        start = time.time()
        percent = 10
        attempts = 0
        while (time.time() - start) < timeout:
            if cancel_event is not None and cancel_event.is_set():
                if progress_callback:
                    progress_callback(percent, "Cancelled by user")
                return None
            time.sleep(min(interval * (1 + attempts // 20), 10))
            attempts += 1
            status_info = self.check_task_status(task_id)
            real = status_info.get("progress", 0)
            if real > 0:
                percent = max(percent, min(real, 90))
            elif percent < 90:
                percent += 4
            if progress_callback:
                progress_callback(percent, f"Novita Status: {status_info['status']}")

            if status_info["status"] in ["TASK_STATUS_SUCCEED", "SUCCEED"]:
                video_url = status_info["video_url"]
                return self._download(video_url, task_id, progress_callback)
            if status_info["status"] in ["TASK_STATUS_FAILED", "FAILED", "ERROR"]:
                ASCIIColors.error(f"Novita task failed: {status_info.get('message', '')}")
                return None

        ASCIIColors.error(f"Novita task {task_id} timed out after {timeout}s.")
        return None

    def _download(self, url: str, task_id: str, callback: Optional[Callable] = None) -> Optional[str]:
        if callback:
            callback(95, "Downloading video file...")
        try:
            out_file = self.output_folder / f"{task_id}.mp4"
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_file, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            if callback:
                callback(100, "Done")
            return str(out_file.resolve())
        except Exception as e:
            trace_exception(e)
            return None