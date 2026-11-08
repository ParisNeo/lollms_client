import os
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

import pipmaster as pm

from lollms_client.lollms_ttm_binding import LollmsTTMBinding
from ascii_colors import trace_exception, ASCIIColors

pm.ensure_packages(["requests"])

import requests

BindingName = "StabilityAITTMBinding"

# Stability AI audio endpoints (Stable Audio 2.x family).
# Sourced from: https://platform.stability.ai/docs/api-reference#tag/Generate
STABILITY_AI_MODELS = [
    {
        "model_name": "stable-audio-2",
        "display_name": "Stable Audio 2",
        "description": "Stability AI text-to-audio (music, sound effects, up to 3 minutes).",
    },
    {
        "model_name": "stable-audio-2-mini",
        "display_name": "Stable Audio 2 Mini",
        "description": "Faster, smaller Stable Audio model for short clips.",
    },
]

_STABILITY_GEN_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"

class StabilityAITTMBinding(LollmsTTMBinding):
    """A Text-to-Music binding for the Stability AI audio API."""

    def __init__(self, **kwargs):
        if "model" in kwargs and "model_name" not in kwargs:
            kwargs["model_name"] = kwargs.pop("model")

        super().__init__(binding_name=BindingName, config=kwargs)

        self.api_key = self.config.get("api_key") or os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Stability AI API key is required. Please set it in config or as STABILITY_API_KEY env var."
            )

        self.model_name = self.config.get("model_name", STABILITY_AI_MODELS[0]["model_name"])
        self.request_timeout = int(self.config.get("request_timeout", 300))
        self.default_output_format = self.config.get("output_format", "mp3")

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Accept": "audio/*"}

    def list_models(self, **kwargs) -> List[Dict[str, str]]:
        return STABILITY_AI_MODELS

    def _build_data(
        self,
        prompt: str,
        duration: Optional[float],
        output_format: str,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "prompt": prompt,
            "output_format": output_format,
            "model": self.model_name,
        }
        if duration is not None:
            data["duration"] = duration
        for key, value in extra.items():
            if value is not None and key not in data:
                data[key] = value
        return data

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates a music/audio clip via Stability AI.

        Accepted kwargs:
            duration (float): clip length in seconds (0.4–190 as per Stable Audio 2)
            output_format (str): "mp3" | "wav"
            seed (int): deterministic generation
        """
        duration = kwargs.pop("duration", None)
        output_format = kwargs.pop("output_format", self.default_output_format)

        data = self._build_data(
            prompt=prompt,
            duration=duration,
            output_format=output_format,
            extra=kwargs,
        )

        try:
            ASCIIColors.info(
                f"Submitting music generation job to Stability AI ({self.model_name})..."
            )
            response = requests.post(
                _STABILITY_GEN_URL,
                headers=self.headers,
                files={"none": ""},
                data=data,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.content

        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = getattr(e.response, "text", str(e))
            raise RuntimeError(f"Stability AI API HTTP Error: {detail}") from e
        except Exception as e:
            trace_exception(e)
            raise RuntimeError(f"Stability AI TTM request failed: {e}") from e

    # ── Resource management contract ────────────────────────────────────────
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

    def install_model(self, model_name: str, **kwargs) -> dict:
        return {"status": False, "message": "Remote API; models are managed by Stability AI."}

    def get_zoo(self) -> List[Dict[str, Any]]:
        return list(STABILITY_AI_MODELS)

    def download_from_zoo(self, index: int, progress_callback: Optional[Callable[[dict], None]] = None) -> dict:
        if index < 0 or index >= len(STABILITY_AI_MODELS):
            return {"status": False, "message": "Index out of bounds."}
        return {"status": False, "message": "Remote API; models are managed by Stability AI."}


if __name__ == "__main__":
    ASCIIColors.magenta("--- Stability AI TTM Binding Test ---")
    if "STABILITY_API_KEY" not in os.environ:
        ASCIIColors.error("STABILITY_API_KEY environment variable not set. Cannot run test.")
        exit(1)

    try:
        binding = StabilityAITTMBinding()

        ASCIIColors.cyan("\n--- Test: Music Generation ---")
        prompt = "A cinematic orchestral piece with sweeping strings and timpani"
        audio_bytes = binding.generate_music(prompt, duration=20)

        assert len(audio_bytes) > 1000, "Generated audio bytes are too small."
        output_path = Path(__file__).parent / "tmp_stability_music.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        ASCIIColors.green(f"Music generation OK. Audio saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Stability AI TTM binding test failed: {e}")