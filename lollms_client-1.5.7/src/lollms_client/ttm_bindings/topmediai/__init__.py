import os
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

from lollms_client.lollms_ttm_binding import LollmsTTMBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

# Ensure required packages are installed
pm.ensure_packages(["requests"])

BindingName = "TopMediaiTTMBinding"

class TopMediaiTTMBinding(LollmsTTMBinding):
    """A Text-to-Music binding for the TopMediai API."""

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.api_key = self.settings.get("api_key") or os.environ.get("TOPMEDIAI_API_KEY")
        if not self.api_key:
            raise ValueError("TopMediai API key is required. Please set it in config or as TOPMEDIAI_API_KEY env var.")
        self.base_url = "https://api.topmediai.com/v1"
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}

    def list_models(self, **kwargs) -> List[str]:
        # The API does not provide a list of selectable models.
        # It's a single, prompt-based system.
        return ["default"]

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music using the TopMediai synchronous API.
        """
        url = f"{self.base_url}/music"
        duration = kwargs.get("duration", 30)

        payload = {
            "text": prompt,
            "duration": f"{duration}", # API expects duration as a string
        }

        try:
            ASCIIColors.info("Requesting music from TopMediai...")
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise Exception(f"TopMediai API returned an error: {data.get('message', 'Unknown error')}")

            audio_url = data.get("data", {}).get("music_url")
            if not audio_url:
                raise Exception("API response did not contain a music URL.")

            ASCIIColors.info(f"Downloading generated audio from {audio_url}")
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            
            return audio_response.content

        except requests.exceptions.HTTPError as e:
            try:
                error_details = e.response.json()
                raise Exception(f"TopMediai API HTTP Error: {error_details}") from e
            except:
                raise Exception(f"TopMediai API HTTP Error: {e.response.text}") from e
        except Exception as e:
            trace_exception(e)
            raise

if __name__ == '__main__':
    ASCIIColors.magenta("--- TopMediai TTM Binding Test ---")
    if "TOPMEDIAI_API_KEY" not in os.environ:
        ASCIIColors.error("TOPMEDIAI_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = TopMediaiTTMBinding()
        
        ASCIIColors.cyan("\n--- Test: Music Generation ---")
        prompt = "lo-fi hip hop beat, chill, relaxing, perfect for studying"
        music_bytes = binding.generate_music(prompt, duration=30)
        
        assert len(music_bytes) > 1000, "Generated music bytes are too small."
        output_path = Path(__file__).parent / "tmp_topmediai_music.mp3"
        with open(output_path, "wb") as f:
            f.write(music_bytes)
        ASCIIColors.green(f"Music generation OK. Audio saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"TopMediai TTM binding test failed: {e}")