import os
import time
import requests
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from ascii_colors import ASCIIColors, trace_exception
from lollms_client.lollms_ttv_binding import LollmsTTVBinding, MediaInput

SUPPORTED_MODELS = [
    {"name": "wan2.1-t2v", "description": "Wan 2.1 Text-to-Video (Open-Weights SOTA)", "aspect_ratios": ["16:9", "9:16", "1:1"]},
    {"name": "kling-v3.0-std-t2v", "description": "Kling 3.0 Standard Text-to-Video", "aspect_ratios": ["16:9", "9:16", "1:1"]},
    {"name": "hunyuan-video-fast", "description": "Tencent Hunyuan Video Fast", "aspect_ratios": ["16:9", "9:16"]},
    {"name": "pixverse-v4.5-t2v", "description": "PixVerse v4.5 Text-to-Video", "aspect_ratios": ["16:9", "9:16"]}
]

BindingName = "NovitaAITTV"

class NovitaAITTV(LollmsTTVBinding):
    def __init__(
        self,
        **kwargs: Any
    ):
        super().__init__(
            binding_name="novita_ai_ttv",
            debug=bool(kwargs.get("debug", False))
        )
        self.config = kwargs or {}
        self.api_key = self.config.get("novita_api_key") or os.environ.get("NOVITA_API_KEY", "")
        self.base_url = self.config.get("base_url", "https://api.novita.ai/v3/async")
        self.output_folder = Path(self.config.get("output_folder", "./data/ttv_outputs/novita_ai"))
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _headers(self, key_override: Optional[str] = None) -> Dict[str, str]:
        key = key_override or self.api_key
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

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
                balance = float(data.get("credit", 0)) / 100.0  # Converted to USD
                return {"status": True, "balance": balance, "message": "Connection OK"}
            return {"status": False, "balance": 0.0, "message": f"API error: {res.status_code} {res.text}"}
        except Exception as e:
            return {"status": False, "balance": 0.0, "message": str(e)}

    @property
    def list_models(self) -> Dict[str, Any]:
        """Command: list_models"""
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
            video_url = ""

            if status in ["TASK_STATUS_SUCCEED", "SUCCEED"]:
                videos = data.get("videos") or data.get("data", {}).get("videos", [])
                if videos:
                    video_url = videos[0].get("video_url", "")
                return {"status": status, "progress": 100, "video_url": video_url, "message": "Ready"}
            
            elif status in ["TASK_STATUS_FAILED", "FAILED"]:
                return {"status": status, "progress": 0, "video_url": "", "message": task_info.get("reason", "Failed")}
            
            return {"status": status, "progress": 50, "video_url": "", "message": "In progress"}
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
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Optional[str]:
        """Core TTV pipeline matching model_input_parameters."""
        model_name = model or self.config.get("model", "wan2.1-t2v")
        ar = aspect_ratio or self.config.get("aspect_ratio", "16:9")
        dur = duration or self.config.get("duration", 5)
        sd = seed if seed is not None else self.config.get("seed", -1)
        neg_prompt = negative_prompt or self.config.get("negative_prompt", "")
        timeout = self.config.get("timeout_s", 600)
        interval = self.config.get("poll_interval_s", 3)

        payload = {
            "prompt": prompt,
            "aspect_ratio": ar,
            "duration": dur
        }
        if neg_prompt:
            payload["negative_prompt"] = neg_prompt
        if sd != -1:
            payload["seed"] = sd

        if progress_callback:
            progress_callback(5, f"Submitting task to {model_name}...")

        try:
            req_url = f"{self.base_url}/{model_name}"
            resp = requests.post(req_url, headers=self._headers(), json=payload, timeout=30)
            resp.raise_for_status()
            task_id = resp.json().get("task_id") or resp.json().get("data", {}).get("task_id")
            if not task_id:
                return None
        except Exception as e:
            trace_exception(e)
            return None

        # Polling
        start = time.time()
        percent = 10
        while (time.time() - start) < timeout:
            time.sleep(interval)
            status_info = self.check_task_status(task_id)
            if percent < 90:
                percent += 4
            if progress_callback:
                progress_callback(percent, f"Novita Status: {status_info['status']}")

            if status_info["status"] in ["TASK_STATUS_SUCCEED", "SUCCEED"]:
                video_url = status_info["video_url"]
                return self._download(video_url, task_id, progress_callback)
            elif status_info["status"] in ["TASK_STATUS_FAILED", "FAILED", "ERROR"]:
                return None

        return None

    def _download(self, url: str, task_id: str, callback: Optional[Callable] = None) -> Optional[str]:
        if callback:
            callback(95, "Downloading video file...")
        out_file = self.output_folder / f"{task_id}.mp4"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        if callback:
            callback(100, "Done")
        return str(out_file.resolve())