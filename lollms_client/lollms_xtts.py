import requests
from pydantic import BaseModel
from lollms_client.lollms_core import LollmsClient
from typing import Optional

class LollmsXTTSRequest(BaseModel):
    text: str
    voice: str|None = None
    fn: str|None = None

class LollmsXTTS:
    def __init__(self, lollmsClient:LollmsClient):
        self.base_url = lollmsClient.host_address

    def text2Audio(self, text, voice=None, fn=None):
        endpoint = f"{self.base_url}/text2Audio"
        request_data = LollmsXTTSRequest(text=text, voice=voice if voice else "main_voice", fn=fn if fn else "fn.wav")
        response = requests.post(endpoint, json=request_data.dict())
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

# Example usage
if __name__ == "__main__":
    base_url = "http://your_flask_server_url"
    lollms_xtts = LollmsXTTS(LollmsClient("http://localhost:9600"))
    
    response = lollms_xtts.text2Audio(text="Hello, world!", voice="default", fn="output_audio.wav")
    print(response)
