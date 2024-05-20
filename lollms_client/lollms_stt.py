import requests
from pydantic import BaseModel
from lollms_client.lollms_core import LollmsClient
from typing import Optional

class LollmSTTRequest(BaseModel):
    wave_file_path: str
    model: str = None
    fn:str = None

class LollmsSTT:
    def __init__(self, lollmsClient:LollmsClient):
        self.base_url = lollmsClient.host_address

    def audio2text(self, text, model=None, fn=None):
        endpoint = f"{self.base_url}/text2Audio"
        request_data = LollmSTTRequest(text=text, model=model if model else "base", fn=fn if fn else "fn.wav")
        response = requests.post(endpoint, json=request_data.model_dump())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def get_voices(self):
        endpoint = f"{self.base_url}/list_stt_models"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # Raise an error for bad status codes
            voices = response.json()  # Assuming the response is in JSON format
            return voices["voices"]
        except requests.exceptions.RequestException as e:
            print(f"Couldn't list voices: {e}")
            return ["main_voice"]

