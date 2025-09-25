import os
import requests
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from lollms_client.lollms_ttm_binding import LollmsTTMBinding
from ascii_colors import trace_exception, ASCIIColors
import pipmaster as pm

# Ensure required packages are installed
pm.ensure_packages(["requests"])

BindingName = "ReplicateTTMBinding"

# Popular music models available on Replicate
# Sourced from: https://replicate.com/collections/text-to-music
REPLICATE_MODELS = [
    {"model_name": "meta/musicgen:b05b1dff1d8c6ac63d42422dd565e23b63869bf2d51acda751e04b5dd304535d", "display_name": "Meta - MusicGen", "description": "State-of-the-art controllable text-to-music model from Meta."},
    {"model_name": "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787", "display_name": "Suno - Bark", "description": "Text-to-audio model capable of music, voice, and sound effects."},
    {"model_name": "joehoover/musicgen-melody:7a76a8258b23fae65c5a24debbe88414f9bed22c2422a63465731103f6990803", "display_name": "MusicGen Melody", "description": "MusicGen fine-tuned for generating melodies."},
]

class ReplicateTTMBinding(LollmsTTMBinding):
    """A Text-to-Music binding for models hosted on Replicate."""

    def __init__(self, **kwargs):
        super().__init__(binding_name=BindingName, **kwargs)
        self.api_key = self.settings.get("api_key") or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise ValueError("Replicate API token is required. Please set it in config or as REPLICATE_API_TOKEN env var.")
        self.model_version = self.settings.get("model_name", "meta/musicgen:b05b1dff1d8c6ac63d42422dd565e23b63869bf2d51acda751e04b5dd304535d")
        self.base_url = "https://api.replicate.com/v1"
        self.headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}

    def list_models(self, **kwargs) -> List[Dict[str, str]]:
        return REPLICATE_MODELS

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music via Replicate by starting a prediction and polling for the result.
        """
        model_id, version_id = self.model_version.split(":")
        
        payload = {
            "version": version_id,
            "input": {
                "prompt": prompt,
                "duration": kwargs.get("duration", 8),
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 0.9),
                # Add other model-specific parameters here
            }
        }
        
        try:
            # 1. Start the prediction
            ASCIIColors.info(f"Submitting music generation job to Replicate ({model_id})...")
            start_response = requests.post(f"{self.base_url}/predictions", json=payload, headers=self.headers)
            start_response.raise_for_status()
            job_data = start_response.json()
            get_url = job_data["urls"]["get"]
            ASCIIColors.info(f"Job submitted. Polling for results at: {get_url}")

            # 2. Poll for the result
            while True:
                poll_response = requests.get(get_url, headers=self.headers)
                poll_response.raise_for_status()
                poll_data = poll_response.json()
                status = poll_data["status"]

                if status == "succeeded":
                    ASCIIColors.green("Generation successful!")
                    output_url = poll_data["output"]
                    # Download the resulting audio file
                    audio_response = requests.get(output_url)
                    audio_response.raise_for_status()
                    return audio_response.content
                elif status in ["starting", "processing"]:
                    ASCIIColors.info(f"Job status: {status}. Waiting...")
                    time.sleep(3)
                else: # failed, canceled
                    error_log = poll_data.get("logs", "No logs available.")
                    raise Exception(f"Replicate job failed with status '{status}'. Log: {error_log}")
                    
        except requests.exceptions.HTTPError as e:
            error_details = e.response.json().get("detail", e.response.text)
            raise Exception(f"Replicate API HTTP Error: {error_details}") from e
        except Exception as e:
            trace_exception(e)
            raise

if __name__ == '__main__':
    ASCIIColors.magenta("--- Replicate TTM Binding Test ---")
    if "REPLICATE_API_TOKEN" not in os.environ:
        ASCIIColors.error("REPLICATE_API_TOKEN environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = ReplicateTTMBinding()
        
        ASCIIColors.cyan("\n--- Test: Music Generation with MusicGen ---")
        prompt = "An epic cinematic orchestral piece, with soaring strings and dramatic percussion, fit for a movie trailer"
        music_bytes = binding.generate_music(prompt, duration=10)
        
        assert len(music_bytes) > 1000, "Generated music bytes are too small."
        output_path = Path(__file__).parent / "tmp_replicate_music.wav"
        with open(output_path, "wb") as f:
            f.write(music_bytes)
        ASCIIColors.green(f"Music generation OK. Audio saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Replicate TTM binding test failed: {e}")