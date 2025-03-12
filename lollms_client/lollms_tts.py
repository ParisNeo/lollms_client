import requests
from pydantic import BaseModel
from lollms_client.lollms_core import LollmsClient
from typing import Optional
from ascii_colors import ASCIIColors
class LollmsTTSRequest(BaseModel):
    text: str
    voice: str|None = None
    fn: str|None = None

class LollmsTTS:
    def __init__(self, lollmsClient:LollmsClient):
        self.base_url = lollmsClient.binding.host_address

    def text2Audio(self, text, voice=None, fn=None):
        endpoint = f"{self.base_url}/text2Audio"
        request_data = LollmsTTSRequest(text=text, voice=voice if voice else "main_voice", fn=fn if fn else "fn.wav")
        response = requests.post(endpoint, json=request_data.model_dump())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def get_voices(self):
        endpoint = f"{self.base_url}/list_voices"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # Raise an error for bad status codes
            voices = response.json()  # Assuming the response is in JSON format
            if "error" in voices:
                ASCIIColors.error(voices["error"])
                return []
            else:
                return voices["voices"]
        except requests.exceptions.RequestException as e:
            print(f"Couldn't list voices: {e}")
            return ["main_voice"]

