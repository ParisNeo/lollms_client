import os
import requests
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any

from lollms_client.lollms_ttm_binding import LollmsTTMBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

# Ensure required packages are installed
pm.ensure_packages(["requests"])

BindingName = "StabilityAITTMBinding"

# Models available via the Stability AI Audio API
# Sourced from: https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v1~1generation~1stable-audio-2.0~1text-to-audio/post
STABILITY_AI_MODELS = [
    {"model_name": "stable-audio-2.0", "display_name": "Stable Audio 2.0", "description": "High-quality, full-track music generation up to 3 minutes."},
    {"model_name": "stable-audio-1.0", "display_name": "Stable Audio 1.0", "description": "Original model, best for short clips and sound effects."},
]

class StabilityAITTMBinding(LollmsTTMBinding):
    """A Text-to-Music binding for Stability AI's Stable Audio API."""

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.api_key = self.settings.get("api_key") or os.environ.get("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("Stability AI API key is required. Please set it in the configuration or as STABILITY_API_KEY environment variable.")
        self.model_name = self.settings.get("model_name", "stable-audio-2.0")

    def list_models(self, **kwargs) -> List[Dict[str, str]]:
        return STABILITY_AI_MODELS

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music using the Stable Audio API.

        Args:
            prompt (str): The text prompt describing the desired music.
            duration (int): The duration of the audio in seconds. Defaults to 29.
            **kwargs: Additional parameters for the API.
        
        Returns:
            bytes: The generated audio data in WAV format.
        """
        url = f"https://api.stability.ai/v1/generation/{self.model_name}/text-to-audio"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "audio/wav",
        }
        
        # Get duration, with a default of 29 seconds as it's a common value
        duration = kwargs.get("duration", 29)
        
        payload = {
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": 1.0,
            "seed": kwargs.get("seed", 0), # 0 for random in API
            "steps": kwargs.get("steps", 100),
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
        }

        # Handle different parameter names for duration
        if self.model_name == "stable-audio-2.0":
            payload["duration"] = duration
        else: # stable-audio-1.0
            payload["sample_length"] = duration * 44100 # v1 uses sample length

        try:
            ASCIIColors.info(f"Requesting music from Stability AI ({self.model_name})...")
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()
            
            ASCIIColors.green("Successfully generated music from Stability AI.")
            return response.content
        except requests.exceptions.HTTPError as e:
            try:
                error_details = e.response.json()
                error_message = error_details.get("message", e.response.text)
            except:
                error_message = e.response.text
            ASCIIColors.error(f"HTTP Error from Stability AI: {e.response.status_code} - {error_message}")
            raise Exception(f"Stability AI API Error: {error_message}") from e
        except Exception as e:
            trace_exception(e)
            raise Exception(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    ASCIIColors.magenta("--- Stability AI TTM Binding Test ---")
    if "STABILITY_API_KEY" not in os.environ:
        ASCIIColors.error("STABILITY_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        # Test with default settings
        binding = StabilityAITTMBinding()
        
        ASCIIColors.cyan("\n--- Test: Music Generation ---")
        prompt = "80s synthwave, retro futuristic, driving beat, cinematic"
        music_bytes = binding.generate_music(prompt, duration=10)
        
        assert len(music_bytes) > 1000, "Generated music bytes are too small."
        output_path = Path(__file__).parent / "tmp_stability_music.wav"
        with open(output_path, "wb") as f:
            f.write(music_bytes)
        ASCIIColors.green(f"Music generation OK. Audio saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Stability AI TTM binding test failed: {e}")