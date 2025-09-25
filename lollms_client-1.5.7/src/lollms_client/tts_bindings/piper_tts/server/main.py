# File: lollms_client/tts_bindings/piper/server/main.py

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import argparse
import sys
from pathlib import Path
import asyncio
import traceback
import os
import json
from typing import Optional, List, Dict
import io
import aiohttp
import aiofiles

# --- Piper TTS Implementation ---
try:
    print("Server: Loading Piper dependencies...")
    import piper
    import numpy as np
    import soundfile as sf
    print("Server: Piper dependencies loaded successfully")
    piper_available = True
    
except Exception as e:
    print(f"Server: Failed to load Piper dependencies: {e}")
    print(f"Server: Traceback:\n{traceback.format_exc()}")
    piper_available = False

# --- API Models ---
class GenerationRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speaker_id: Optional[int] = None
    length_scale: Optional[float] = 1.0
    noise_scale: Optional[float] = 0.667
    noise_w: Optional[float] = 0.8

class VoiceRequest(BaseModel):
    voice: str

class DownloadRequest(BaseModel):
    voice: str

class PiperServer:
    def __init__(self):
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.current_voice = None
        self.current_model = None
        self.loaded_models = {}  # Cache for loaded models
        
        # Available voice models (subset of popular ones)
        self.available_voices = self._get_available_voice_list()
        self.installed_voices = self._scan_installed_models()
        
        # Auto-download a default voice if none installed
        if not self.installed_voices and piper_available:
            asyncio.create_task(self._download_default_voice())
    
    def _get_available_voice_list(self) -> Dict[str, Dict]:
        """Get list of available voice models from Piper repository"""
        # Popular high-quality voices across different languages
        return {
            # English voices
            "en_US-lessac-medium": {
                "language": "en_US", 
                "quality": "medium",
                "description": "US English, female, clear",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
            },
            "en_US-lessac-low": {
                "language": "en_US",
                "quality": "low", 
                "description": "US English, female, fast",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx"
            },
            "en_US-ryan-high": {
                "language": "en_US",
                "quality": "high",
                "description": "US English, male, high quality",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
            },
            "en_US-ryan-medium": {
                "language": "en_US",
                "quality": "medium", 
                "description": "US English, male, balanced",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
            },
            "en_GB-alan-medium": {
                "language": "en_GB",
                "quality": "medium",
                "description": "British English, male",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
            },
            
            # French voices  
            "fr_FR-siwis-medium": {
                "language": "fr_FR",
                "quality": "medium",
                "description": "French, female",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
            },
            
            # German voices
            "de_DE-thorsten-medium": {
                "language": "de_DE", 
                "quality": "medium",
                "description": "German, male",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
            },
            
            # Spanish voices
            "es_ES-mls_9972-low": {
                "language": "es_ES",
                "quality": "low", 
                "description": "Spanish, female",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/mls_9972/low/es_ES-mls_9972-low.onnx"
            },
            
            # Italian voices
            "it_IT-riccardo-x_low": {
                "language": "it_IT",
                "quality": "x_low",
                "description": "Italian, male, fast", 
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx"
            },
            
            # Dutch voices
            "nl_NL-mls_5809-low": {
                "language": "nl_NL",
                "quality": "low",
                "description": "Dutch, female",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/nl/nl_NL/mls_5809/low/nl_NL-mls_5809-low.onnx"
            },
            
            # Portuguese voices
            "pt_BR-faber-medium": {
                "language": "pt_BR",
                "quality": "medium", 
                "description": "Brazilian Portuguese, male",
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx"
            }
        }
    
    def _scan_installed_models(self) -> List[str]:
        """Scan for already downloaded models"""
        installed = []
        for onnx_file in self.models_dir.glob("*.onnx"):
            voice_name = onnx_file.stem
            # Check if corresponding JSON config exists
            json_file = onnx_file.with_suffix('.onnx.json')
            if json_file.exists():
                installed.append(voice_name)
        return installed
    
    async def _download_default_voice(self):
        """Download a default voice if none are installed"""
        try:
            print("Server: No voices installed, downloading default voice...")
            await self.download_voice("en_US-lessac-medium")
        except Exception as e:
            print(f"Server: Failed to download default voice: {e}")
    
    async def download_voice(self, voice_name: str) -> bool:
        """Download a voice model and its config"""
        if voice_name not in self.available_voices:
            raise ValueError(f"Voice '{voice_name}' not available")
        
        voice_info = self.available_voices[voice_name]
        model_url = voice_info["url"]
        config_url = model_url + ".json"
        
        model_path = self.models_dir / f"{voice_name}.onnx"
        config_path = self.models_dir / f"{voice_name}.onnx.json"
        
        # Check if already downloaded
        if model_path.exists() and config_path.exists():
            print(f"Server: Voice '{voice_name}' already downloaded")
            if voice_name not in self.installed_voices:
                self.installed_voices.append(voice_name)
            return True
        
        try:
            print(f"Server: Downloading voice '{voice_name}'...")
            
            async with aiohttp.ClientSession() as session:
                # Download model file
                print(f"Server: Downloading model from {model_url}")
                async with session.get(model_url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(model_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                
                # Download config file
                print(f"Server: Downloading config from {config_url}")
                async with session.get(config_url) as response:
                    response.raise_for_status()
                    async with aiofiles.open(config_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
            
            # Update installed voices list
            if voice_name not in self.installed_voices:
                self.installed_voices.append(voice_name)
                
            print(f"Server: Successfully downloaded voice '{voice_name}'")
            return True
            
        except Exception as e:
            print(f"Server: Failed to download voice '{voice_name}': {e}")
            # Clean up partial downloads
            for path in [model_path, config_path]:
                if path.exists():
                    path.unlink()
            raise
    
    def _load_model(self, voice_name: str):
        """Load a Piper model"""
        if voice_name in self.loaded_models:
            return self.loaded_models[voice_name]
        
        model_path = self.models_dir / f"{voice_name}.onnx"
        config_path = self.models_dir / f"{voice_name}.onnx.json"
        
        if not (model_path.exists() and config_path.exists()):
            raise FileNotFoundError(f"Voice '{voice_name}' not found. Please download it first.")
        
        print(f"Server: Loading model for voice '{voice_name}'...")
        
        # Load the model using piper
        voice = piper.PiperVoice.load(str(model_path), config_path=str(config_path))
        
        self.loaded_models[voice_name] = voice
        print(f"Server: Model '{voice_name}' loaded successfully")
        
        return voice
    
    def generate_audio(self, text: str, voice: Optional[str] = None, 
                      speaker_id: Optional[int] = None, length_scale: float = 1.0,
                      noise_scale: float = 0.667, noise_w: float = 0.8) -> bytes:
        """Generate audio from text using Piper"""
        if not piper_available:
            raise RuntimeError("Piper library not available")
        
        # Use provided voice or current default
        target_voice = voice or self.current_voice
        
        # If no voice specified and no default, use first available
        if not target_voice and self.installed_voices:
            target_voice = self.installed_voices[0]
            self.current_voice = target_voice
        
        if not target_voice:
            raise RuntimeError("No voice available. Please download a voice first.")
        
        if target_voice not in self.installed_voices:
            raise RuntimeError(f"Voice '{target_voice}' not installed. Please download it first.")
        
        try:
            print(f"Server: Generating audio for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"Server: Using voice: {target_voice}")
            
            # Load the model
            voice_model = self._load_model(target_voice)
            
            # Generate audio
            audio_stream = io.BytesIO()
            
            # Synthesize to the stream
            voice_model.synthesize(
                text, 
                audio_stream,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w
            )
            
            # Get the raw audio data
            audio_stream.seek(0)
            audio_data = audio_stream.getvalue()
            
            print(f"Server: Generated {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            print(f"Server: Error generating audio: {e}")
            print(f"Server: Traceback:\n{traceback.format_exc()}")
            raise
    
    def set_voice(self, voice: str) -> bool:
        """Set the current default voice"""
        if voice in self.installed_voices:
            self.current_voice = voice
            print(f"Server: Voice changed to: {voice}")
            return True
        else:
            print(f"Server: Voice '{voice}' not installed")
            return False
    
    def list_voices(self) -> List[str]:
        """Return list of installed voices"""
        return self.installed_voices.copy()
    
    def list_available_voices(self) -> Dict[str, Dict]:
        """Return list of all available voices for download"""
        return self.available_voices.copy()
    
    def list_models(self) -> List[str]:
        """Return list of available models"""
        return ["piper"]

# --- Globals ---
app = FastAPI(title="Piper TTS Server")
router = APIRouter()
piper_server = PiperServer()
model_lock = asyncio.Lock()  # Ensure thread-safe access

# --- API Endpoints ---
@router.post("/generate_audio")
async def generate_audio(request: GenerationRequest):
    async with model_lock:
        try:
            audio_bytes = piper_server.generate_audio(
                text=request.text,
                voice=request.voice,
                speaker_id=request.speaker_id,
                length_scale=request.length_scale,
                noise_scale=request.noise_scale,
                noise_w=request.noise_w
            )
            from fastapi.responses import Response
            return Response(content=audio_bytes, media_type="audio/wav")
        except Exception as e:
            print(f"Server: ERROR in generate_audio endpoint: {e}")
            print(f"Server: ERROR traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

@router.post("/download_voice")
async def download_voice(request: DownloadRequest):
    try:
        success = await piper_server.download_voice(request.voice)
        return {"success": success, "message": f"Voice '{request.voice}' downloaded successfully"}
    except Exception as e:
        print(f"Server: ERROR in download_voice endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set_voice")
async def set_voice(request: VoiceRequest):
    try:
        success = piper_server.set_voice(request.voice)
        if success:
            return {"success": True, "message": f"Voice set to {request.voice}"}
        else:
            return {"success": False, "message": f"Voice {request.voice} not installed"}
    except Exception as e:
        print(f"Server: ERROR in set_voice endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_voices")
async def list_voices():
    try:
        voices = piper_server.list_voices()
        print(f"Server: Returning {len(voices)} installed voices")
        return {"voices": voices}
    except Exception as e:
        print(f"Server: ERROR in list_voices endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_available_voices")
async def list_available_voices():
    try:
        voices = piper_server.list_available_voices()
        return {"voices": voices}
    except Exception as e:
        print(f"Server: ERROR in list_available_voices endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_models")
async def list_models():
    try:
        models = piper_server.list_models()
        return {"models": models}
    except Exception as e:
        print(f"Server: ERROR in list_models endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def status():
    return {
        "status": "running",
        "piper_available": piper_available,
        "current_voice": piper_server.current_voice,
        "installed_voices_count": len(piper_server.installed_voices),
        "available_voices_count": len(piper_server.available_voices),
        "installed_voices": piper_server.installed_voices
    }

app.include_router(router)

# --- Server Startup ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Piper TTS Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8083, help="Port to bind the server to.")
    
    args = parser.parse_args()

    print(f"Server: Starting Piper TTS server on {args.host}:{args.port}")
    print(f"Server: Piper available: {piper_available}")
    print(f"Server: Models directory: {piper_server.models_dir}")
    print(f"Server: Installed voices: {len(piper_server.installed_voices)}")
    print(f"Server: Available voices for download: {len(piper_server.available_voices)}")
    
    if piper_server.installed_voices:
        print(f"Server: Current voice: {piper_server.current_voice or piper_server.installed_voices[0]}")
    else:
        print("Server: No voices installed - will download default voice on startup")
    
    uvicorn.run(app, host=args.host, port=args.port)