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

BindingName = "BeatovenAITTMBinding"

class BeatovenAITTMBinding(LollmsTTMBinding):
    """A Text-to-Music binding for the Beatoven.ai API."""

    def __init__(self,
                 **kwargs):
        # Prioritize 'model_name' but accept 'model' as an alias from config files.
        if 'model' in kwargs and 'model_name' not in kwargs:
            kwargs['model_name'] = kwargs.pop('model')
        super().__init__(binding_name=BindingName, config=kwargs)    
        self.api_key = self.config.get("api_key") or os.environ.get("BEATOVEN_API_KEY")
        if not self.api_key:
            raise ValueError("Beatoven.ai API key is required. Please set it in config or as BEATOVEN_API_KEY env var.")
        self.base_url = "https://api.beatoven.ai/api/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def list_models(self, **kwargs) -> List[str]:
        # Beatoven.ai does not expose different models via the API.
        # Customization is done via genre, mood, and tempo.
        return ["default"]

    def _poll_for_completion(self, task_id: str) -> Dict[str, Any]:
        """Polls the tasks endpoint until the composition is complete."""
        poll_url = f"{self.base_url}/tasks/{task_id}"
        while True:
            try:
                response = requests.get(poll_url, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                status = data.get("status")

                if status == "success":
                    ASCIIColors.green("Composition task successful.")
                    return data
                elif status == "failed":
                    error_info = data.get("error", "Unknown error.")
                    raise Exception(f"Beatoven.ai task failed: {error_info}")
                else:
                    ASCIIColors.info(f"Task status is '{status}'. Waiting...")
                    time.sleep(5)
            except requests.exceptions.HTTPError as e:
                raise Exception(f"Failed to poll task status: {e.response.text}")

    def generate_music(self, prompt: str, **kwargs) -> bytes:
        """
        Generates music by creating a track, waiting for composition, and downloading the result.
        """
        # Step 1: Create a track
        create_track_url = f"{self.base_url}/tracks"
        payload = {
            "title": prompt[:100], # Use prompt as title, truncated
            "duration_in_seconds": kwargs.get("duration", 30),
            "genre": kwargs.get("genre", "Cinematic"),
            "tempo": kwargs.get("tempo", "medium"),
            "prompt": prompt
        }

        try:
            ASCIIColors.info("Submitting music track request to Beatoven.ai...")
            create_response = requests.post(create_track_url, json=payload, headers=self.headers)
            create_response.raise_for_status()
            task_id = create_response.json().get("task_id")
            ASCIIColors.info(f"Track creation submitted. Task ID: {task_id}")
            
            # Step 2: Poll for task completion
            task_result = self._poll_for_completion(task_id)
            track_id = task_result.get("track_id")
            if not track_id:
                raise Exception("Task completed but did not return a track_id.")

            # Step 3: Get track details to find the audio URL
            track_url = f"{self.base_url}/tracks/{track_id}"
            track_response = requests.get(track_url, headers=self.headers)
            track_response.raise_for_status()
            
            audio_url = track_response.json().get("renders", {}).get("wav")
            if not audio_url:
                raise Exception("Could not find WAV render URL in the completed track details.")

            # Step 4: Download the audio file
            ASCIIColors.info(f"Downloading generated audio from {audio_url}")
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            
            return audio_response.content

        except requests.exceptions.HTTPError as e:
            error_details = e.response.json()
            raise Exception(f"Beatoven.ai API HTTP Error: {error_details}") from e
        except Exception as e:
            trace_exception(e)
            raise

if __name__ == '__main__':
    ASCIIColors.magenta("--- Beatoven.ai TTM Binding Test ---")
    if "BEATOVEN_API_KEY" not in os.environ:
        ASCIIColors.error("BEATOVEN_API_KEY environment variable not set. Cannot run test.")
        exit(1)
        
    try:
        binding = BeatovenAITTMBinding()
        
        ASCIIColors.cyan("\n--- Test: Music Generation ---")
        prompt = "A mysterious and suspenseful cinematic track with soft piano and eerie strings, building tension."
        music_bytes = binding.generate_music(prompt, duration=45, genre="Cinematic", tempo="slow")
        
        assert len(music_bytes) > 1000, "Generated music bytes are too small."
        output_path = Path(__file__).parent / "tmp_beatoven_music.wav"
        with open(output_path, "wb") as f:
            f.write(music_bytes)
        ASCIIColors.green(f"Music generation OK. Audio saved to {output_path}")

    except Exception as e:
        trace_exception(e)
        ASCIIColors.error(f"Beatoven.ai TTM binding test failed: {e}")